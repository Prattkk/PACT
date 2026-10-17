"""Risk/Correlation Agent (spec §4.1).

Outputs a per-instrument risk score and a portfolio-level overlay
(propagated via metadata; the Portfolio Manager applies it).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, InstrumentView
from data.fetchers.prices import fetch_panel
from factors.risk import cornish_fisher_var, ewma_volatility


class RiskCorrelationAgent(BaseAgent):
    name = "risk_correlation"

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        start = as_of - timedelta(days=400)
        prices = fetch_panel(list(self.universe), start, as_of + timedelta(days=1), field="Adj Close")  # end is exclusive
        if prices.empty:
            return {s: {} for s in self.universe}
        rets = prices.pct_change().dropna(how="all")

        # Universe-wide risk state
        avg_corr_30d = (
            rets.tail(30).corr().where(~np.eye(rets.shape[1], dtype=bool)).stack().mean()
            if rets.shape[1] > 1
            else 0.0
        )
        cs_vol = float(rets.tail(30).std(axis=1).mean()) if rets.shape[1] > 1 else 0.0

        out: dict[str, dict[str, float]] = {}
        for sym in self.universe:
            r = rets[sym].dropna() if sym in rets.columns else pd.Series(dtype=float)
            f = {
                "ewma_vol_ann": float(ewma_volatility(r).iloc[-1]) if len(r) else 0.0,
                "cf_var_5pct": cornish_fisher_var(r),
                "avg_pairwise_corr_30d": float(avg_corr_30d),
                "cross_sectional_vol_30d": cs_vol,
            }
            out[sym] = f
        return out

    def decide(
        self,
        as_of: date,
        factors_by_instrument: dict[str, dict[str, float]],
    ) -> list[InstrumentView]:
        # Risk agent doesn't take long/short views — it returns 0 direction
        # with conviction encoding "this is risky to size up". Portfolio
        # Manager reads `factors` to scale.
        views: list[InstrumentView] = []
        for sym, f in factors_by_instrument.items():
            vol = f.get("ewma_vol_ann", 0.0)
            risk_score = min(vol / 0.4, 1.0)  # 40% vol = max risk score
            views.append(
                InstrumentView(
                    instrument=sym,
                    direction=0,
                    conviction=risk_score,
                    horizon="1m",
                    factors=f,
                    rationale=f"risk overlay: vol={vol:.2%}",
                )
            )
        return views
