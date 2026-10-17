"""Driver: per-agent attribution (spec §6.7).

For a given saved baseline cell, runs:
- **Leave-one-out** (LOO): replaces each specialist agent with a NullAgent in
  turn, re-runs the backtest, reports `sharpe(full) - sharpe(without_agent)`.
  6 backtest re-runs per cell.
- **DAG-Shapley approximation** (HiveMind-style truncated MC): random
  permutation of agents, walk the prefix, average marginal Sharpe contribution.
  `n_permutations × n_agents` re-runs per cell — opt-in via `--shapley`.

Outputs:
  results/tables/attribution_loo.csv
  results/tables/attribution_shapley.csv (only when --shapley)

Usage:
  python scripts/run_attribution.py --cell <cell_id>            # LOO only
  python scripts/run_attribution.py --cell <cell_id> --shapley  # +Shapley
  python scripts/run_attribution.py --all-protocols PROTO       # LOO across all cells of one protocol

Note: Re-runs the backtest from scratch for each subset, so the cell's
universe and date range are recovered from the saved metadata. Heavy on long
windows; use a short-window cell first (`--quick` from run_ablation_matrix).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from agents.null_agent import NullAgent
from backtest.engine import BacktestConfig, run_backtest
from evaluation.metrics import sharpe
from pact_logging import get_logger
from portfolio.construction import PortfolioConfig
from scripts._common import (
    RAW_DIR,
    TABLES_DIR,
    build_agents,
    build_llm_client,
    build_protocol,
)

log = get_logger(__name__)

# Specialist agents (PM aggregates these — not itself a candidate for LOO).
SPECIALISTS = (
    "macro_regime",
    "narrative_event",
    "cross_asset_transmission",
    "technical_trend",
    "fundamentals_carry",
    "risk_correlation",
)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_meta(cell_id: str) -> dict:
    path = RAW_DIR / cell_id / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"cell not found: {cell_id} ({path})")
    return json.loads(path.read_text())


def _backtest_with_subset(
    *,
    universe: tuple[str, ...],
    bt_cfg: BacktestConfig,
    regime: str,
    protocol_key: str,
    keep_agents: tuple[str, ...],
    offline_llm: bool,
    cell_window: tuple[date, date] | None = None,
    enable_altdata: bool = False,
    enable_secondary: tuple[str, ...] = (),
) -> pd.Series:
    """Run a backtest where specialists not in `keep_agents` are replaced by NullAgent.

    Portfolio manager and risk_correlation are kept by default (PM is the
    aggregator; risk doesn't take directional views, only scaling).

    `cell_window`, `enable_altdata`, `enable_secondary` are threaded through
    so the LOO's "full" run reproduces the original cell exactly — without
    them the Sharpe(full) drifts from the matrix run's published value.
    """
    llm = build_llm_client(regime, offline=offline_llm)
    full = build_agents(
        llm, universe,
        cell_window=cell_window,
        enable_altdata=enable_altdata,
        enable_secondary=enable_secondary,
    )
    keep = set(keep_agents) | {"portfolio_manager"}
    nulled: dict[str, object] = {}
    for name, agent in full.items():
        if name in keep:
            nulled[name] = agent
        else:
            nulled[name] = NullAgent(replaces_name=name, universe=universe)
    proto = build_protocol(protocol_key, nulled, llm, universe)
    result = run_backtest(proto, universe, bt_cfg)
    return result.excess_returns


def _make_runner(cell_meta: dict, offline_llm: bool):
    universe = tuple(cell_meta["universe"])
    start = _parse_date(cell_meta["start"])
    end = _parse_date(cell_meta["end"])
    bt_cfg = BacktestConfig(
        start=start,
        end=end,
        starting_capital=float(cell_meta["starting_capital"]),
        transaction_cost_bps_roundtrip=float(cell_meta["transaction_cost_bps_roundtrip"]),
        portfolio=PortfolioConfig(),
    )
    regime = cell_meta["regime"]
    protocol_key = cell_meta["protocol"]
    # Reproduce the original cell's agent context exactly so the LOO's "full"
    # baseline matches the matrix-run Sharpe.
    cell_window = (start, end)
    # Reproduce the original cell's flags. For matrix-run cells (commit
    # b6e... onwards) `enable_altdata` is stored explicitly. For older
    # cells we fall back to True since every Window B sweep ran with
    # --altdata, which is the only way the equity-fundamentals factors
    # actually flow through the LLM.
    if "enable_altdata" in cell_meta:
        enable_altdata = bool(cell_meta["enable_altdata"])
    else:
        enable_altdata = bool(cell_meta.get("enable_altdata_in_fundamentals", True))
    enable_secondary = tuple(cell_meta.get("enable_secondary", ()))

    def run_with_agents(subset: tuple[str, ...]) -> pd.Series:
        return _backtest_with_subset(
            universe=universe,
            bt_cfg=bt_cfg,
            regime=regime,
            protocol_key=protocol_key,
            keep_agents=subset,
            offline_llm=offline_llm,
            cell_window=cell_window,
            enable_altdata=enable_altdata,
            enable_secondary=enable_secondary,
        )

    return run_with_agents


def loo_attribution(cell_id: str, offline_llm: bool = False) -> list[dict]:
    """6 + 1 = 7 backtest runs per cell (full + each LOO)."""
    meta = _load_meta(cell_id)
    log.info("loo cell=%s", cell_id)
    runner = _make_runner(meta, offline_llm)

    full_returns = runner(SPECIALISTS)
    full_sharpe = sharpe(full_returns)
    log.info("loo full set sharpe=%.4f", full_sharpe)

    rows = []
    for dropped in SPECIALISTS:
        subset = tuple(a for a in SPECIALISTS if a != dropped)
        sub_returns = runner(subset)
        sub_sharpe = sharpe(sub_returns)
        delta = full_sharpe - sub_sharpe
        log.info("loo dropped=%s sharpe_without=%.4f delta=%.4f", dropped, sub_sharpe, delta)
        rows.append({
            "cell_id": cell_id,
            "window": meta.get("window"),
            "regime": meta.get("regime"),
            "protocol": meta.get("protocol"),
            "dropped_agent": dropped,
            "sharpe_full": full_sharpe,
            "sharpe_without": sub_sharpe,
            "sharpe_delta": delta,
            "schema_version": 2,
        })
    return rows


def shapley_attribution(
    cell_id: str,
    n_permutations: int = 12,
    seed: int = 42,
    offline_llm: bool = False,
) -> list[dict]:
    """`n_permutations × |SPECIALISTS|` backtest runs per cell. Default 12 × 6 = 72."""
    meta = _load_meta(cell_id)
    log.info("shapley cell=%s n_permutations=%d (~%d backtests)", cell_id, n_permutations, n_permutations * len(SPECIALISTS))
    runner = _make_runner(meta, offline_llm)

    rng = random.Random(seed)
    contrib = {a: 0.0 for a in SPECIALISTS}
    counts = {a: 0 for a in SPECIALISTS}

    for perm_idx in range(n_permutations):
        order = list(SPECIALISTS)
        rng.shuffle(order)
        prev_sr = 0.0
        for k in range(1, len(order) + 1):
            subset = tuple(order[:k])
            sr = sharpe(runner(subset))
            agent_added = order[k - 1]
            contrib[agent_added] += sr - prev_sr
            counts[agent_added] += 1
            prev_sr = sr
        log.info("shapley perm=%d/%d done", perm_idx + 1, n_permutations)

    rows = []
    for a in SPECIALISTS:
        rows.append({
            "cell_id": cell_id,
            "window": meta.get("window"),
            "regime": meta.get("regime"),
            "protocol": meta.get("protocol"),
            "agent": a,
            "shapley_value": contrib[a] / max(counts[a], 1),
            "n_permutations": n_permutations,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Run per-agent attribution on saved baseline cells.")
    p.add_argument("--cell", default=None, help="single cell id")
    p.add_argument("--all-protocols", default=None, help="run LOO for every cell whose protocol matches this key")
    p.add_argument("--shapley", action="store_true", help="also run DAG-Shapley approximation (slow)")
    p.add_argument("--n-permutations", type=int, default=12, help="Shapley permutation count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline-llm", action="store_true")
    args = p.parse_args()

    if not args.cell and not args.all_protocols:
        print("specify --cell <cell_id> or --all-protocols <protocol_key>")
        return

    if args.cell:
        cell_ids = [args.cell]
    else:
        cell_ids = []
        for sub in sorted(RAW_DIR.iterdir()):
            mp = sub / "metadata.json"
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            if meta.get("protocol") == args.all_protocols:
                cell_ids.append(sub.name)
        if not cell_ids:
            print(f"no saved cells with protocol={args.all_protocols}")
            return

    loo_all: list[dict] = []
    shap_all: list[dict] = []
    for cid in cell_ids:
        loo_all.extend(loo_attribution(cid, offline_llm=args.offline_llm))
        if args.shapley:
            shap_all.extend(shapley_attribution(
                cid,
                n_permutations=args.n_permutations,
                seed=args.seed,
                offline_llm=args.offline_llm,
            ))

    loo_path = TABLES_DIR / "attribution_loo.csv"
    pd.DataFrame(loo_all).to_csv(loo_path, index=False)
    log.info("wrote %s rows=%d", loo_path, len(loo_all))
    print("=== LOO attribution ===")
    print(pd.DataFrame(loo_all).to_string(index=False))

    if shap_all:
        shap_path = TABLES_DIR / "attribution_shapley.csv"
        pd.DataFrame(shap_all).to_csv(shap_path, index=False)
        log.info("wrote %s rows=%d", shap_path, len(shap_all))
        print()
        print("=== DAG-Shapley attribution ===")
        print(pd.DataFrame(shap_all).to_string(index=False))


if __name__ == "__main__":
    main()
