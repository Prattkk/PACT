"""Holdings-based portfolio simulator.

Shared by the strategy engine and the passive benchmarks so both are
accounted for identically:

- Positions are held in currency units and drift with asset returns between
  trades (no implicit daily rebalancing).
- Cash and short-sale collateral accrue the daily risk-free rate; negative
  cash (leverage) is financed at the same rate. Excess returns are therefore
  sum_i w_i (r_i - rf), the standard definition for a self-financing
  long/short portfolio.
- Trading costs are charged on one-way traded notional at half the quoted
  round-trip cost (a 30 bps round trip = 15 bps per unit traded one way).
- Trades execute at the close of the day they are submitted for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    equity: pd.Series                 # NAV at each close
    total_returns: pd.Series          # net of costs, includes rf on cash
    excess_returns: pd.Series         # total_returns - rf
    gross_excess_returns: pd.Series   # excess returns before trading costs
    turnover: pd.Series               # one-way traded notional / NAV, per day
    weights: pd.DataFrame             # end-of-day weights after trading
    rf: pd.Series
    metadata: dict[str, object] = field(default_factory=dict)


class PortfolioSimulator:
    """Step-by-step simulator. Call `step(ts, asset_ret, rf, target)` once per day."""

    def __init__(self, columns: list[str], starting_capital: float, cost_bps_roundtrip: float):
        self.columns = list(columns)
        self.positions = np.zeros(len(self.columns))
        self.cash = float(starting_capital)
        self.one_way_cost = cost_bps_roundtrip / 2.0 / 10_000.0
        self._prev_nav = float(starting_capital)
        self._rows: list[tuple] = []
        self._weights: list[np.ndarray] = []
        self._index: list[pd.Timestamp] = []

    @property
    def nav(self) -> float:
        return float(self.cash + self.positions.sum())

    def current_weights(self) -> dict[str, float]:
        nav = self.nav
        if nav <= 0:
            return {c: 0.0 for c in self.columns}
        return {c: float(p / nav) for c, p in zip(self.columns, self.positions)}

    def equity_history(self) -> pd.Series:
        return pd.Series([r[0] for r in self._rows], index=self._index, dtype=float)

    def step(
        self,
        ts: pd.Timestamp,
        asset_ret: np.ndarray | None,
        rf: float,
        target: dict[str, float] | None = None,
    ) -> None:
        # 1) accrue the day's returns on yesterday's holdings
        if asset_ret is not None:
            r = np.nan_to_num(np.asarray(asset_ret, dtype=float), nan=0.0)
            self.positions = self.positions * (1.0 + r)
        self.cash *= 1.0 + rf
        nav_pre = self.nav

        # 2) trade to target weights at today's close
        traded = 0.0
        cost = 0.0
        if target is not None and nav_pre > 0:
            desired = np.array([float(target.get(c, 0.0)) for c in self.columns]) * nav_pre
            delta = desired - self.positions
            traded = float(np.abs(delta).sum())
            cost = traded * self.one_way_cost
            self.cash -= float(delta.sum()) + cost
            self.positions = desired

        nav = self.nav
        prev = self._prev_nav
        total = nav / prev - 1.0 if prev > 0 else 0.0
        gross_total = (nav + cost) / prev - 1.0 if prev > 0 else 0.0
        turnover = traded / nav_pre if nav_pre > 0 else 0.0
        self._rows.append((nav, total, total - rf, gross_total - rf, turnover, rf))
        self._weights.append(self.positions / nav if nav > 0 else np.zeros_like(self.positions))
        self._index.append(pd.Timestamp(ts))
        self._prev_nav = nav

    def result(self, metadata: dict[str, object] | None = None) -> SimulationResult:
        idx = pd.DatetimeIndex(self._index)
        arr = np.array(self._rows, dtype=float).reshape(-1, 6)
        return SimulationResult(
            equity=pd.Series(arr[:, 0], index=idx),
            total_returns=pd.Series(arr[:, 1], index=idx),
            excess_returns=pd.Series(arr[:, 2], index=idx),
            gross_excess_returns=pd.Series(arr[:, 3], index=idx),
            turnover=pd.Series(arr[:, 4], index=idx),
            rf=pd.Series(arr[:, 5], index=idx),
            weights=pd.DataFrame(np.array(self._weights).reshape(len(idx), len(self.columns)),
                                 index=idx, columns=self.columns),
            metadata=metadata or {},
        )


def simulate_target_schedule(
    asset_returns: pd.DataFrame,
    rf: pd.Series,
    targets: dict[pd.Timestamp, dict[str, float]],
    starting_capital: float = 1_000_000.0,
    cost_bps_roundtrip: float = 30.0,
) -> SimulationResult:
    """Simulate a fixed schedule of target weights (used for passive benchmarks).

    `targets[ts]` is executed at the close of `ts`. The first row's return is
    not accrued (the portfolio starts in cash at that close).
    """
    sim = PortfolioSimulator(list(asset_returns.columns), starting_capital, cost_bps_roundtrip)
    rf = rf.reindex(asset_returns.index).fillna(0.0)
    for i, ts in enumerate(asset_returns.index):
        ret = None if i == 0 else asset_returns.loc[ts].values
        sim.step(ts, ret, 0.0 if i == 0 else float(rf.loc[ts]), targets.get(ts))
    return sim.result()
