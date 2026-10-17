"""Performance metrics.

Sharpe and Sortino are computed on daily *excess* returns (over 13-week
T-bills) and annualised with sqrt(252); the series must be on the exchange
calendar. Annualised return is geometric on total (not excess) returns.

Coordination Breakeven Spread (CBS): the round-trip cost at which a
coordinated protocol's net Sharpe falls to that of the no-communication
ensemble, computed from gross-of-cost excess returns and one-way turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_DAILY = 252


def sharpe(excess_returns: pd.Series) -> float:
    r = excess_returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ANNUALIZATION_DAILY))


def sortino(excess_returns: pd.Series) -> float:
    r = excess_returns.dropna()
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    if len(r) < 2 or downside == 0:
        return float("nan")
    return float(r.mean() / downside * np.sqrt(ANNUALIZATION_DAILY))


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna()
    if e.empty:
        return float("nan")
    return float((e / e.cummax() - 1).min())


def annualized_return(total_returns: pd.Series) -> float:
    r = total_returns.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    return growth ** (ANNUALIZATION_DAILY / len(r)) - 1.0


def turnover_metric(turnover: pd.Series) -> float:
    """Average one-way turnover (fraction of NAV) per trading session with a trade."""
    s = turnover.dropna()
    s = s[s > 0]
    return float(s.mean()) if len(s) else 0.0


def annual_turnover(turnover: pd.Series) -> float:
    """One-way turnover per year, as a multiple of NAV."""
    s = turnover.dropna()
    return float(s.sum() / len(s) * ANNUALIZATION_DAILY) if len(s) else 0.0


def net_excess_at_cost(gross_excess: pd.Series, turnover: pd.Series, cost_bps_roundtrip: float) -> pd.Series:
    """Re-price a strategy at a different round-trip cost (half charged per one-way unit)."""
    t = turnover.reindex(gross_excess.index).fillna(0.0)
    return gross_excess - t * (cost_bps_roundtrip / 2.0 / 10_000.0)


def summary(equity: pd.Series, excess_returns: pd.Series, turnover: pd.Series,
            total_returns: pd.Series | None = None) -> dict[str, float]:
    tot = total_returns if total_returns is not None else excess_returns
    return {
        "sharpe": sharpe(excess_returns),
        "sortino": sortino(excess_returns),
        "max_drawdown": max_drawdown(equity),
        "annualized_return": annualized_return(tot),
        "annualized_vol": float(excess_returns.std() * np.sqrt(ANNUALIZATION_DAILY)),
        "annual_turnover": annual_turnover(turnover),
    }


def coordination_breakeven_spread(
    gross_coord: pd.Series,
    gross_ensemble: pd.Series,
    turnover_coord: pd.Series,
    turnover_ensemble: pd.Series,
    cost_grid_bps: np.ndarray | None = None,
) -> float:
    """Smallest round-trip cost (bps) at which SR(coord) <= SR(ensemble).

    Returns 0.0 if the coordinated protocol is already no better at zero cost,
    and NaN if it stays ahead over the whole grid.
    """
    grid = cost_grid_bps if cost_grid_bps is not None else np.linspace(0, 400, 801)
    for c in grid:
        gap = sharpe(net_excess_at_cost(gross_coord, turnover_coord, c)) - sharpe(
            net_excess_at_cost(gross_ensemble, turnover_ensemble, c)
        )
        if not np.isnan(gap) and gap <= 0:
            return float(c)
    return float("nan")
