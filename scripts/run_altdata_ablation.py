"""Driver: 5-variant alt-data ablation (spec §6.6).

Runs the 5 variants from `configs/altdata_ablation.yaml` against one or
more coordination protocols, recording artifacts under
`results/raw/altdata_<variant>__<window>__<regime>__<protocol>/`.

Usage:
  # Run all 5 variants on the best 2 protocols (per spec recommendation):
  python scripts/run_altdata_ablation.py --protocols sequential_pipeline debate \\
      --regime frontier --window b

  # Faster variant: regime=none on sequential_pipeline only:
  python scripts/run_altdata_ablation.py --protocols sequential_pipeline \\
      --regime none --window b
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from data.universe import TRADEABLE_SYMBOLS
from pact_logging import get_logger
from portfolio.construction import PortfolioConfig
from scripts._common import (
    RAW_DIR,
    build_agents,
    build_llm_client,
    build_protocol,
    load_config,
    save_backtest_result,
)

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _resolve_window(matrix_cfg: dict, window_key: str) -> tuple[date, date]:
    block = matrix_cfg[f"window_{window_key}"]
    return _parse_date(block["start"]), _parse_date(block["end"])


def _cell_dir(variant: str, window: str, regime: str, protocol: str) -> Path:
    d = RAW_DIR / f"altdata_{variant}__{window}__{regime}__{protocol}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    p = argparse.ArgumentParser(description="5-variant alt-data ablation (spec §6.6)")
    p.add_argument("--protocols", nargs="+", default=["sequential_pipeline"])
    p.add_argument("--regime", choices=["none", "open_source", "frontier", "frontier_r2", "frontier_r3", "frontier_anon", "frontier_haiku"], default="none")
    p.add_argument("--window", choices=["a", "b"], default="b")
    p.add_argument("--universe", nargs="*", default=None)
    p.add_argument("--variants", nargs="*", default=None,
                   help="subset variant ids (default: all 5)")
    p.add_argument("--offline-llm", action="store_true")
    args = p.parse_args()

    altdata_cfg = load_config("altdata_ablation.yaml")
    matrix_cfg = load_config("ablation_matrix.yaml")
    start, end = _resolve_window(matrix_cfg, args.window)
    universe = tuple(args.universe) if args.universe else TRADEABLE_SYMBOLS

    variants = altdata_cfg["variants"]
    if args.variants:
        keep = set(args.variants)
        variants = [v for v in variants if v["id"] in keep]
    log.info("altdata ablation variants=%s protocols=%s window=%s",
             [v["id"] for v in variants], args.protocols, args.window)

    bt_cfg = BacktestConfig(
        start=start, end=end,
        starting_capital=float(matrix_cfg["backtest"]["starting_capital"]),
        rebalance_freq=matrix_cfg["backtest"]["rebalance_freq"],
        transaction_cost_bps_roundtrip=float(matrix_cfg["backtest"]["transaction_cost_bps_roundtrip"]),
        portfolio=PortfolioConfig(),
    )

    rows: list[dict] = []
    for variant in variants:
        # Per-variant policy: skip Window A for variants that include Polymarket
        # (only meaningful 2022+, per spec §6.4).
        if variant.get("window") == "window_b" and args.window != "b":
            log.info("variant=%s requires window_b; skipping --window=%s", variant["id"], args.window)
            continue
        enable_secondary = tuple(variant.get("enable_secondary", []))
        enable_altdata = bool(variant.get("enable_altdata_in_fundamentals", False))

        for proto_key in args.protocols:
            cell_id = f"altdata_{variant['id']}__{args.window}__{args.regime}__{proto_key}"
            log.info("cell start id=%s secondary=%s altdata=%s", cell_id, enable_secondary, enable_altdata)
            llm = build_llm_client(args.regime, offline=args.offline_llm)
            agents = build_agents(
                llm, universe,
                cell_window=(start, end),
                enable_altdata=enable_altdata,
                enable_secondary=enable_secondary,
            )
            proto = build_protocol(proto_key, agents, llm, universe)

            try:
                result = run_backtest(proto, universe, bt_cfg)
            except Exception as e:
                log.exception("cell failed id=%s: %s", cell_id, e)
                rows.append({"cell_id": cell_id, "error": str(e)})
                continue

            out = _cell_dir(variant["id"], args.window, args.regime, proto_key)
            save_backtest_result(out, result, meta_extra={
                "cell_id": cell_id,
                "altdata_variant": variant["id"],
                "window": args.window,
                "regime": args.regime,
                "protocol": proto_key,
                "universe": list(universe),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "starting_capital": bt_cfg.starting_capital,
                "transaction_cost_bps_roundtrip": bt_cfg.transaction_cost_bps_roundtrip,
                "enable_secondary": list(enable_secondary),
                "enable_altdata_in_fundamentals": enable_altdata,
            })
            final_eq = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")
            log.info("cell done id=%s final_equity=%.2f", cell_id, final_eq)
            rows.append({
                "cell_id": cell_id,
                "variant": variant["id"],
                "protocol": proto_key,
                "rebalances": len(result.weights),
                "final_equity": final_eq,
            })

    summary = pd.DataFrame(rows)
    out_path = Path("results") / f"altdata_run_{args.window}.csv"
    summary.to_csv(out_path, index=False)
    log.info("altdata summary written to %s rows=%d", out_path, len(summary))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
