"""Backtest engine.

Weekly decisions, holdings-based accounting (see `backtest.simulator`).

Timing: the protocol is run with `as_of` = the last exchange session of each
week, using only information available at that session's close. The
resulting target weights are executed at the close of the *next* session
(`execution_lag_days=1`), so no decision trades at a price it has already
observed.

Costs: a flat round-trip cost (default 30 bps) charged as half per unit of
one-way traded notional. No market-impact model is applied: at the
$1M scale used here, trades are a negligible fraction of daily volume in
these ETFs.

Risk overlays (applied at each decision, in this order):
- per-name / gross caps and ex-ante portfolio-vol scaling (portfolio.construction)
- drawdown breaker on the strategy's own NAV: if NAV is >= 15% below its
  trailing 60-session peak, positions are halved for the next 4 decisions
- correlation throttle: 30-session mean pairwise correlation > 0.60 -> x0.70
- weekly one-way turnover cap of 50% of NAV
"""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from coordination.base import CoordinationProtocol
from data.fetchers.prices import fetch_panel, fetch_risk_free_daily
from pact_logging import get_logger
from portfolio.construction import PortfolioConfig, views_to_target_weights
from portfolio.risk_overlays import (
    DRAWDOWN_HALVING_WEEKS,
    DRAWDOWN_THRESHOLD,
    apply_correlation_throttle,
    apply_drawdown_breaker,
)
from agents._llm_helpers import LLM_STATS
from backtest.simulator import PortfolioSimulator

log = get_logger(__name__)

WARMUP_CALENDAR_DAYS = 400   # history loaded before `start` for vol/cov/corr estimates
DRAWDOWN_LOOKBACK = 60       # sessions


@dataclass
class BacktestConfig:
    start: date
    end: date
    starting_capital: float = 1_000_000.0
    rebalance_freq: str = "W-FRI"             # decision = last session of each week
    transaction_cost_bps_roundtrip: float = 30.0
    execution_lag_days: int = 1               # execute at close of the next session
    use_risk_free: bool = True                # excess returns over 13-week T-bills
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    turnover_cap_per_period: float = 0.50     # one-way, fraction of NAV per decision


@dataclass
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame
    returns: pd.Series               # total net return (includes rf on cash)
    turnover: pd.Series              # one-way traded / NAV, per session
    excess_returns: pd.Series | None = None
    gross_excess_returns: pd.Series | None = None
    rf: pd.Series | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def decision_dates(index: pd.DatetimeIndex, start: date, end: date, freq: str = "W-FRI") -> list[pd.Timestamp]:
    """Last exchange session of each period, restricted to [start, end]."""
    sel = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if len(sel) == 0:
        return []
    s = pd.Series(sel, index=sel)
    return list(s.groupby(s.index.to_period(freq)).max())


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_backtest(
    protocol: CoordinationProtocol,
    universe: tuple[str, ...],
    cfg: BacktestConfig,
) -> BacktestResult:
    log.info(
        "backtest start protocol=%s universe=%d start=%s end=%s capital=%s",
        protocol.name, len(universe), cfg.start, cfg.end, cfg.starting_capital,
    )
    prices = fetch_panel(list(universe), cfg.start - timedelta(days=WARMUP_CALENDAR_DAYS), cfg.end + timedelta(days=1),
                         field="Adj Close")
    if prices.empty:
        raise RuntimeError("price panel is empty; check universe / date range")
    cols = [s for s in universe if s in prices.columns]
    missing = sorted(set(universe) - set(cols))
    if missing:
        log.warning("backtest: no prices for %s; they will be held at zero weight", missing)
    rets_all = prices[cols].pct_change()
    sim_index = prices.index[(prices.index >= pd.Timestamp(cfg.start)) & (prices.index <= pd.Timestamp(cfg.end))]
    rf = fetch_risk_free_daily(sim_index) if cfg.use_risk_free else pd.Series(0.0, index=sim_index)

    llm_stats_before = Counter(LLM_STATS)
    decisions = set(decision_dates(prices.index, cfg.start, cfg.end, cfg.rebalance_freq))
    sim = PortfolioSimulator(cols, cfg.starting_capital, cfg.transaction_cost_bps_roundtrip)
    pending: list[tuple[int, dict[str, float]]] = []   # (sessions remaining, target)
    weights_log: dict[date, dict[str, float]] = {}
    weeks_since_breach = 999
    breaker_events = 0

    for i, ts in enumerate(sim_index):
        # execute any order whose lag has elapsed
        target = None
        still: list[tuple[int, dict[str, float]]] = []
        for left, tgt in pending:
            if left <= 1:
                target = tgt
            else:
                still.append((left - 1, tgt))
        pending = still
        asset_ret = None if i == 0 else rets_all.loc[ts].values
        sim.step(ts, asset_ret, 0.0 if i == 0 else float(rf.loc[ts]), target)

        if ts not in decisions:
            continue

        result = protocol.run(ts.date())
        hist = rets_all.loc[:ts]
        rv = (hist.tail(63).std() * np.sqrt(252)).dropna().to_dict()

        eq = sim.equity_history().tail(DRAWDOWN_LOOKBACK)
        dd_60d = float(eq.iloc[-1] / eq.max() - 1.0) if len(eq) else 0.0
        if dd_60d <= DRAWDOWN_THRESHOLD:
            if weeks_since_breach >= DRAWDOWN_HALVING_WEEKS:
                breaker_events += 1
                log.warning("drawdown breaker triggered as_of=%s dd_60d=%.4f", ts.date(), dd_60d)
            weeks_since_breach = 0
        else:
            weeks_since_breach += 1

        tail30 = hist.tail(30)
        avg_corr = (
            float(tail30.corr().where(~np.eye(len(cols), dtype=bool)).stack().mean())
            if len(tail30) >= 30 else 0.0
        )
        tgt = views_to_target_weights(
            result.final_views,
            realized_vol_by_symbol=rv,
            cov_matrix=hist.tail(252).cov() if len(hist) >= 252 else None,
            cfg=cfg.portfolio,
        )
        tgt = {s: tgt.get(s, 0.0) for s in cols}
        tgt = apply_drawdown_breaker(tgt, dd_60d, weeks_since_breach)
        tgt = apply_correlation_throttle(tgt, avg_corr)

        current = sim.current_weights()
        delta = {s: tgt[s] - current.get(s, 0.0) for s in cols}
        gross_delta = sum(abs(d) for d in delta.values())
        if gross_delta > cfg.turnover_cap_per_period:
            scale = cfg.turnover_cap_per_period / gross_delta
            tgt = {s: current.get(s, 0.0) + delta[s] * scale for s in cols}

        weights_log[ts.date()] = tgt
        if cfg.execution_lag_days <= 0:
            raise ValueError("execution_lag_days must be >= 1 (same-close execution uses observed prices)")
        pending.append((cfg.execution_lag_days, tgt))

    res = sim.result()
    failures = dict(getattr(protocol, "failures", Counter()))
    llm_stats = dict(Counter(LLM_STATS) - llm_stats_before)
    meta = {
        "protocol": protocol.name,
        "rebalances": len(weights_log),
        "breaker_events": breaker_events,
        "agent_failures": failures,
        "agent_failure_total": int(sum(failures.values())),
        "llm_stats": llm_stats,
        "execution_lag_days": cfg.execution_lag_days,
        "use_risk_free": cfg.use_risk_free,
        "cost_convention": "half of round-trip bps per unit one-way notional",
        "git_commit": _git_commit(),
    }
    log.info(
        "backtest done protocol=%s rebalances=%d final_equity=%.2f failures=%d",
        protocol.name, len(weights_log), float(res.equity.iloc[-1]), meta["agent_failure_total"],
    )
    return BacktestResult(
        equity=res.equity,
        weights=res.weights,
        returns=res.total_returns,
        turnover=res.turnover,
        excess_returns=res.excess_returns,
        gross_excess_returns=res.gross_excess_returns,
        rf=res.rf,
        metadata=meta,
    )
