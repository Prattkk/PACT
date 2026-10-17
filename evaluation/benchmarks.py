"""Passive benchmarks, computed with the same simulator, calendar, risk-free
rate and cost model as the strategies (single source for every table,
figure and test in the paper).

- spy_bh: buy SPY on the first session, hold.
- equal_weight: 1/N across the tradeable universe, rebalanced monthly.
- sixty_forty: 60% SPY / 40% IEF, rebalanced monthly.
- inverse_vol: weights proportional to 1 / 63-session realised vol,
  rebalanced monthly (vol estimated from data up to the rebalance date).

Rebalances execute at the close of the first session of each month; the
initial purchase is charged like any other trade.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.simulator import SimulationResult, simulate_target_schedule
from data.fetchers.prices import fetch_panel, fetch_risk_free_daily

BENCHMARK_LABELS = {
    "spy_bh": "SPY buy-and-hold",
    "equal_weight": "Equal weight (monthly)",
    "sixty_forty": "60/40 SPY--IEF (monthly)",
    "inverse_vol": "Inverse volatility (monthly)",
}


def _month_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("M")).min())


def benchmark_targets(
    returns: pd.DataFrame, sim_index: pd.DatetimeIndex, universe: list[str], vol_window: int = 63,
) -> dict[str, dict[pd.Timestamp, dict[str, float]]]:
    months = _month_starts(sim_index)
    first = sim_index[0]
    n = len(universe)
    out: dict[str, dict[pd.Timestamp, dict[str, float]]] = {
        "spy_bh": {first: {"SPY": 1.0}},
        "equal_weight": {m: {s: 1.0 / n for s in universe} for m in months},
        "sixty_forty": {m: {"SPY": 0.6, "IEF": 0.4} for m in months},
        "inverse_vol": {},
    }
    for m in months:
        vol = returns.loc[:m, universe].tail(vol_window).std()
        inv = (1.0 / vol.replace(0.0, np.nan)).dropna()
        out["inverse_vol"][m] = (inv / inv.sum()).to_dict()
    return out


def compute_benchmarks(
    universe: list[str],
    start: date,
    end: date,
    cost_bps_roundtrip: float = 30.0,
    starting_capital: float = 1_000_000.0,
    prices: pd.DataFrame | None = None,
    rf: pd.Series | None = None,
) -> dict[str, SimulationResult]:
    if prices is None:
        prices = fetch_panel(universe, start - timedelta(days=200), end + timedelta(days=1))
    rets = prices[universe].pct_change()
    sim_index = prices.index[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    if rf is None:
        rf = fetch_risk_free_daily(sim_index)
    targets = benchmark_targets(rets, sim_index, universe)
    return {
        key: simulate_target_schedule(rets.loc[sim_index], rf, sched, starting_capital, cost_bps_roundtrip)
        for key, sched in targets.items()
    }
