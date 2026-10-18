"""Driver: 7 protocols × 2 LLM regimes ablation matrix (spec §6.2).

Reads `configs/ablation_matrix.yaml`, runs one backtest per cell, writes raw
artifacts to `results/raw/<window>__<regime>__<protocol>/`. The companion
`build_paper_tables.py` consumes those artifacts to produce paper tables.

Usage:
    # Full headline matrix on window A (2010-2024, no Polymarket):
    python scripts/run_ablation_matrix.py --window a

    # Quick sanity loop: deterministic_only on a 1y window, no LLM:
    python scripts/run_ablation_matrix.py --quick

    # Subset (one regime, one protocol):
    python scripts/run_ablation_matrix.py --window a --regime open_source --protocol deterministic_only

Notes:
- `--regime none` skips LLM entirely (deterministic_only is the only protocol
  whose output is meaningful here; the others fall back to zero views).
- Runs are idempotent: re-running a cell overwrites its artifacts.
- The data fetcher cache makes second runs fast; first run on window A will
  hit yfinance hundreds of times.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from data.universe import TRADEABLE_SYMBOLS
from pact_logging import get_logger
from portfolio.construction import PortfolioConfig
from scripts._common import (
    build_agents,
    build_llm_client,
    build_protocol,
    cell_dir,
    cell_id,
    load_config,
    save_backtest_result,
)

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _resolve_window(cfg: dict, window_key: str) -> tuple[date, date]:
    block = cfg[f"window_{window_key}"]
    return _parse_date(block["start"]), _parse_date(block["end"])


def run_cell(
    window: str,
    start: date,
    end: date,
    regime: str,
    protocol_key: str,
    universe: tuple[str, ...],
    bt_cfg: BacktestConfig,
    offline: bool,
    enable_altdata: bool = False,
) -> Path:
    """Run one ablation cell and persist its artifacts. Returns its output dir."""
    cid = cell_id(window, regime, protocol_key)
    out = cell_dir(window, regime, protocol_key)
    log.info("cell start id=%s start=%s end=%s n=%d altdata=%s", cid, start, end, len(universe), enable_altdata)

    llm = build_llm_client(regime, offline=offline)
    agents = build_agents(
        llm, universe,
        cell_window=(start, end),
        enable_altdata=enable_altdata,
    )
    proto = build_protocol(protocol_key, agents, llm, universe)

    result = run_backtest(proto, universe, bt_cfg)
    save_backtest_result(
        out,
        result,
        meta_extra={
            "cell_id": cid,
            "window": window,
            "regime": regime,
            "protocol": protocol_key,
            "universe": list(universe),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "starting_capital": bt_cfg.starting_capital,
            "transaction_cost_bps_roundtrip": bt_cfg.transaction_cost_bps_roundtrip,
            "enable_altdata": bool(enable_altdata),
        },
    )
    final_eq = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
    log.info("cell done id=%s final_equity=%.2f rebalances=%d", cid, final_eq, len(result.weights))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run the 7×2 coordination ablation matrix.")
    p.add_argument("--window", choices=["a", "b"], default="a", help="window_a (full) or window_b (modern)")
    p.add_argument("--regime", choices=["open_source", "none", "all", "frontier", "frontier_r2", "frontier_r3", "frontier_anon", "frontier_haiku"], default="all")
    p.add_argument("--protocol", default="all", help="protocol key from coordination.REGISTRY, or 'all'")
    p.add_argument("--quick", action="store_true", help="1y window, 5 ETFs, regime=none — sanity check")
    p.add_argument("--offline-llm", action="store_true", help="LLM clients run in cache-only mode")
    p.add_argument("--universe", nargs="*", default=None, help="override universe (e.g. SPY QQQ IWM)")
    p.add_argument("--altdata", action="store_true", help="enable alt-data factors (GDELT, EIA, etc.)")
    args = p.parse_args()

    cfg = load_config("ablation_matrix.yaml")

    if args.quick:
        start, end = date(2023, 1, 1), date(2024, 1, 1)
        universe = ("SPY", "QQQ", "IWM", "IEF", "GLD")
        regimes = ["none"]
        protocol_keys = ["deterministic_only"]
        window_label = "quick"
    else:
        start, end = _resolve_window(cfg, args.window)
        universe = tuple(args.universe) if args.universe else TRADEABLE_SYMBOLS
        regimes = (
            [r["name"] for r in cfg["llm_regimes"]] if args.regime == "all" else [args.regime]
        )
        protocol_keys = (
            [pr["key"] for pr in cfg["protocols"]] if args.protocol == "all" else [args.protocol]
        )
        window_label = args.window

    bt_cfg = BacktestConfig(
        start=start,
        end=end,
        starting_capital=float(cfg["backtest"]["starting_capital"]),
        rebalance_freq=cfg["backtest"]["rebalance_freq"],
        transaction_cost_bps_roundtrip=float(cfg["backtest"]["transaction_cost_bps_roundtrip"]),
        portfolio=PortfolioConfig(),
    )

    log.info(
        "matrix run window=%s regimes=%s protocols=%s universe=%d",
        window_label, regimes, protocol_keys, len(universe),
    )

    summary_rows = []
    for regime in regimes:
        for proto_key in protocol_keys:
            try:
                out = run_cell(
                    window=window_label,
                    start=start,
                    end=end,
                    regime=regime,
                    protocol_key=proto_key,
                    universe=universe,
                    bt_cfg=bt_cfg,
                    offline=args.offline_llm,
                    enable_altdata=args.altdata,
                )
                meta = json.loads((out / "metadata.json").read_text())
                summary_rows.append(
                    {
                        "cell_id": meta["cell_id"],
                        "window": window_label,
                        "regime": regime,
                        "protocol": proto_key,
                        "rebalances": meta.get("rebalances"),
                        "out_dir": str(out.relative_to(out.parent.parent.parent)),
                    }
                )
            except Exception as e:
                log.exception("cell failed regime=%s protocol=%s: %s", regime, proto_key, e)
                summary_rows.append(
                    {
                        "cell_id": cell_id(window_label, regime, proto_key),
                        "window": window_label,
                        "regime": regime,
                        "protocol": proto_key,
                        "error": str(e),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("results") / f"matrix_run_{window_label}.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("matrix run summary written to %s rows=%d", summary_path, len(summary_df))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
