"""Where does the narrative specialist's (negative) contribution come from?

Reruns the sequential pipeline with and without the narrative specialist
(offline, from the LLM cache) and decomposes the difference in daily excess
returns by instrument, by turnover, and by week, for each regime given.

Writes results/tables/narrative_<regime>.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from agents.null_agent import NullAgent
from backtest.engine import WARMUP_CALENDAR_DAYS, BacktestConfig, run_backtest
from data.fetchers.prices import fetch_panel
from data.universe import TRADEABLE_SYMBOLS
from evaluation.metrics import annual_turnover, sharpe
from portfolio.construction import PortfolioConfig
from scripts._common import TABLES_DIR, build_agents, build_llm_client, build_protocol, load_config


def run(regime: str, drop: str | None, start, end, cfg):
    llm = build_llm_client(regime, offline=True)
    agents = build_agents(llm, TRADEABLE_SYMBOLS, cell_window=(start, end), enable_altdata=True)
    if drop:
        agents[drop] = NullAgent(replaces_name=drop, universe=TRADEABLE_SYMBOLS)
    proto = build_protocol("sequential_pipeline", agents, llm, TRADEABLE_SYMBOLS)
    bt = BacktestConfig(start=start, end=end, rebalance_freq=cfg["backtest"]["rebalance_freq"],
                        portfolio=PortfolioConfig())
    r = run_backtest(proto, TRADEABLE_SYMBOLS, bt)
    fails = sum(v for k, v in r.metadata.get("llm_stats", {}).items() if k.endswith((":call_failed", ":unparseable")))
    if fails:
        raise RuntimeError(f"{regime}: {fails} LLM replies missing from the cache")
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="frontier")
    a = p.parse_args()
    cfg = load_config("ablation_matrix.yaml")
    start = datetime.strptime(cfg["window_b"]["start"], "%Y-%m-%d").date()
    end = datetime.strptime(cfg["window_b"]["end"], "%Y-%m-%d").date()
    full = run(a.regime, None, start, end, cfg)
    without = run(a.regime, "narrative_event", start, end, cfg)

    px = fetch_panel(list(TRADEABLE_SYMBOLS), start - timedelta(days=WARMUP_CALENDAR_DAYS), end + timedelta(days=1),
                     field="Adj Close")
    rets = px.pct_change().reindex(full.weights.index)
    # P&L contribution of each instrument: previous session's weight times today's return
    contrib_full = (full.weights.shift(1) * rets).fillna(0.0)
    contrib_wo = (without.weights.shift(1) * rets).fillna(0.0)
    diff_by_asset = ((contrib_wo - contrib_full).sum() * 252 / len(rets)).sort_values()

    d = (without.excess_returns - full.excess_returns).dropna()
    weekly = d.groupby(d.index.to_period("W-FRI")).sum().sort_values()
    top = weekly.abs().sort_values(ascending=False)
    share_top10 = float(weekly.loc[top.index[:10]].sum() / weekly.sum()) if weekly.sum() != 0 else float("nan")
    sign_diff = (np.sign(full.weights.round(6)) != np.sign(without.weights.round(6))).to_numpy().mean()
    res = {
        "sharpe_full": sharpe(full.excess_returns), "sharpe_without": sharpe(without.excess_returns),
        "turnover_full": annual_turnover(full.turnover), "turnover_without": annual_turnover(without.turnover),
        "gross_exposure_full": float(full.weights.abs().sum(axis=1).mean()),
        "gross_exposure_without": float(without.weights.abs().sum(axis=1).mean()),
        "sign_disagreement": float(sign_diff),
        "ann_return_gain_by_asset": {k: float(v) for k, v in diff_by_asset.items()},
        "ann_return_gain_total": float(d.mean() * 252),
        "share_of_gain_from_top10_weeks": share_top10,
        "top_weeks": {str(k): float(weekly.loc[k]) for k in top.index[:5]},
        "gain_2022": float(d[d.index < "2023-01-01"].mean() * 252),
        "gain_2023_24": float(d[d.index >= "2023-01-01"].mean() * 252),
    }
    (TABLES_DIR / f"narrative_{a.regime}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
