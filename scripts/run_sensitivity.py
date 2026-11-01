"""Full-rerun sensitivity analyses on cached LLM decisions (no API calls).

1. Cost sweep: the whole backtest (including the NAV-based drawdown rule, the
   turnover cap and position drift) is rerun at each round-trip cost, rather
   than repricing a fixed trade list.
2. BTC exclusion: bitcoin is held at zero weight (its 00:00 UTC pricing before
   2024 is offset from the ETFs' 16:00 ET close).

Writes results/tables/sensitivity_<regime>.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from backtest.engine import BacktestConfig, run_backtest
from data.universe import TRADEABLE_SYMBOLS
from evaluation.metrics import sharpe
from portfolio.construction import PortfolioConfig
from scripts._common import TABLES_DIR, build_agents, build_llm_client, build_protocol, load_config

COSTS = (0, 10, 30, 50, 100, 200)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="frontier")
    p.add_argument("--protocols", nargs="+", default=["independent_ensemble", "sequential_pipeline",
                                                       "pipeline_placebo", "hierarchical", "debate"])
    a = p.parse_args()
    cfg = load_config("ablation_matrix.yaml")
    blk = cfg["window_b"]
    start = datetime.strptime(blk["start"], "%Y-%m-%d").date()
    end = datetime.strptime(blk["end"], "%Y-%m-%d").date()
    out = {"costs": list(COSTS), "cost_sweep": {}, "no_btc": {}}
    for proto_key in a.protocols:
        llm = build_llm_client(a.regime, offline=True) if a.regime != "none" else None
        agents = build_agents(llm, TRADEABLE_SYMBOLS, cell_window=(start, end), enable_altdata=True)
        proto = build_protocol(proto_key, agents, llm, TRADEABLE_SYMBOLS)
        row = {}
        for c in COSTS:
            bt = BacktestConfig(start=start, end=end, transaction_cost_bps_roundtrip=float(c),
                                rebalance_freq=cfg["backtest"]["rebalance_freq"], portfolio=PortfolioConfig())
            r = run_backtest(proto, TRADEABLE_SYMBOLS, bt)
            fails = sum(v for k, v in r.metadata.get("llm_stats", {}).items()
                        if k.endswith((":call_failed", ":unparseable")))
            if fails:
                raise RuntimeError(f"{proto_key} at {c}bp: {fails} LLM replies missing from the cache")
            row[str(c)] = sharpe(r.excess_returns)
        out["cost_sweep"][proto_key] = row
        bt = BacktestConfig(start=start, end=end, rebalance_freq=cfg["backtest"]["rebalance_freq"],
                            portfolio=PortfolioConfig(), exclude_symbols=("BTC",))
        out["no_btc"][proto_key] = sharpe(run_backtest(proto, TRADEABLE_SYMBOLS, bt).excess_returns)
        print(proto_key, row, out["no_btc"][proto_key], flush=True)
    (TABLES_DIR / f"sensitivity_{a.regime}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
