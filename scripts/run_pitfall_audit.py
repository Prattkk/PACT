"""Implementation audit: how much do common backtest pitfalls move the Sharpe ratio?

Runs the deterministic baseline (prices only; no API keys needed) once with
every fix in place, then re-introduces one pitfall at a time:

  level_splice    BTC spot and IBIT joined on price levels (-99.9% splice-day return)
  union_calendar  crypto weekend rows kept in the panel (extra zero-return rows)
  double_cost     full round-trip cost charged on each one-way unit traded
  rf_zero         Sharpe on raw returns, cash earns nothing
  no_warmup       vol/covariance estimated only from data inside the test window

Writes results/tables/pitfall_audit.json (consumed by export_paper.py).
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

import backtest.engine as E
import data.fetchers.prices as P
from backtest.engine import BacktestConfig, run_backtest
from coordination.deterministic_only import DeterministicOnlyProtocol
from data.universe import TRADEABLE_SYMBOLS
from evaluation.metrics import annualized_return, max_drawdown, sharpe
from scripts._common import TABLES_DIR

START, END = date(2022, 1, 1), date(2024, 12, 31)


def _level_splice(start: date, end: date) -> pd.DataFrame:
    parts = []
    if start < P.BTC_SPLICE_DATE:
        parts.append(P._cached_yf_download("BTC-USD", start, min(end, P.BTC_SPLICE_DATE)))
    if end >= P.BTC_SPLICE_DATE:
        parts.append(P._cached_yf_download("IBIT", max(start, P.BTC_SPLICE_DATE), end))
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def run_variant(label: str, **kw) -> dict:
    orig = (P._fetch_btc_spliced, P.align_to_exchange_calendar, E.WARMUP_CALENDAR_DAYS)
    try:
        if kw.get("level_splice"):
            P._fetch_btc_spliced = _level_splice
        if kw.get("union_calendar"):
            P.align_to_exchange_calendar = lambda panel: panel
        if kw.get("no_warmup"):
            E.WARMUP_CALENDAR_DAYS = 0
        cfg = BacktestConfig(start=START, end=END,
                             transaction_cost_bps_roundtrip=60.0 if kw.get("double_cost") else 30.0,
                             use_risk_free=not kw.get("rf_zero", False))
        res = run_backtest(DeterministicOnlyProtocol({}, TRADEABLE_SYMBOLS), TRADEABLE_SYMBOLS, cfg)
    finally:
        P._fetch_btc_spliced, P.align_to_exchange_calendar, E.WARMUP_CALENDAR_DAYS = orig
    row = {"label": label, "sharpe": sharpe(res.excess_returns), "mdd": max_drawdown(res.equity),
           "ann_return": annualized_return(res.returns), "rows": len(res.excess_returns)}
    print(row, flush=True)
    return row


def main() -> None:
    rows = [run_variant("fixed")]
    for flag in ("level_splice", "union_calendar", "double_cost", "rf_zero", "no_warmup"):
        rows.append(run_variant(flag, **{flag: True}))
    out = TABLES_DIR / "pitfall_audit.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
