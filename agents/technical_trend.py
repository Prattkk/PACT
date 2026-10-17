"""Technical/Trend Agent (spec §4.1).

Pure price-derived inputs — no text by design. Keeping it text-free is a
control variable in the modality ablation, so do not add news here.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, InstrumentView
from data.fetchers.prices import fetch_panel
from factors.momentum import multi_horizon_momentum, trend_persistence
from factors.risk import cornish_fisher_var, ewma_volatility, garch_forecast


class TechnicalTrendAgent(BaseAgent):
    name = "technical_trend"

    def __init__(self, llm_client, universe: tuple[str, ...], lookback_days: int = 400):
        super().__init__(llm_client, universe)
        self.lookback_days = lookback_days

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        start = as_of - timedelta(days=self.lookback_days)
        prices = fetch_panel(list(self.universe), start, as_of + timedelta(days=1), field="Adj Close")  # end is exclusive
        if prices.empty:
            return {s: {} for s in self.universe}
        rets = prices.pct_change().dropna(how="all")
        moms = multi_horizon_momentum(prices)
        trend = trend_persistence(prices)

        out: dict[str, dict[str, float]] = {}
        for sym in self.universe:
            if sym not in prices.columns:
                out[sym] = {}
                continue
            f = {}
            for h in ["1m", "3m", "6m", "12m"]:
                v = moms[h][sym].iloc[-1] if (h in moms.columns.get_level_values(0)) else np.nan
                f[f"mom_{h}"] = float(v) if pd.notna(v) else 0.0
            r_sym = rets[sym].dropna() if sym in rets.columns else pd.Series(dtype=float)
            f["ewma_vol_ann"] = (
                float(ewma_volatility(r_sym).iloc[-1]) if not r_sym.empty else 0.0
            )
            f["garch_vol_ann_1d"] = garch_forecast(r_sym)
            f["cf_var_5pct"] = cornish_fisher_var(r_sym)
            tp = trend[sym].iloc[-1] if sym in trend.columns else np.nan
            f["trend_persistence"] = float(tp) if pd.notna(tp) else 0.0
            out[sym] = f
        # cross-sectional momentum rank (12m)
        if "12m" in moms.columns.get_level_values(0):
            row = moms["12m"].iloc[-1]
            ranks = row.rank(pct=True)
            for sym in self.universe:
                if sym in out and sym in ranks.index and pd.notna(ranks[sym]):
                    out[sym]["cs_mom_rank_12m"] = float(2 * ranks[sym] - 1)
        return out

    def decide(
        self,
        as_of: date,
        factors_by_instrument: dict[str, dict[str, float]],
    ) -> list[InstrumentView]:
        # Technical agent produces a deterministic anchor view from factor signs,
        # then asks the LLM only for a short rationale. This keeps the decision
        # auditable even if the LLM is swapped out.
        views: list[InstrumentView] = []
        for sym, f in factors_by_instrument.items():
            if not f:
                views.append(
                    InstrumentView(instrument=sym, direction=0, conviction=0.0, horizon="1m")
                )
                continue
            score = (
                np.sign(f.get("mom_1m", 0)) * 0.15
                + np.sign(f.get("mom_3m", 0)) * 0.25
                + np.sign(f.get("mom_6m", 0)) * 0.30
                + np.sign(f.get("mom_12m", 0)) * 0.20
                + f.get("trend_persistence", 0) * 0.10
            )
            direction = int(np.sign(score)) if abs(score) > 0.05 else 0
            conviction = float(min(abs(score), 1.0))
            rationale = self._rationale(sym, f, direction, conviction)
            views.append(
                InstrumentView(
                    instrument=sym,
                    direction=direction,
                    conviction=conviction,
                    horizon="1m",
                    factors=f,
                    rationale=rationale,
                )
            )
        return views

    def _rationale(self, sym: str, f: dict[str, float], direction: int, conv: float) -> str:
        if self.llm is None:
            return f"deterministic anchor: dir={direction}, conv={conv:.2f}"
        prompt = (
            f"Instrument {sym}. Technical factors:\n{json.dumps(f, indent=2)}\n"
            f"Anchor direction={direction}, conviction={conv:.2f}.\n"
            "In ≤2 sentences, explain the strongest signal and any conflicts. No new claims."
        )
        try:
            resp = self.llm.complete(
                system="You are a terse technical-analysis explainer. Output 1–2 sentences.",
                user=prompt,
                max_tokens=120,
            )
            return resp.text.strip()
        except Exception as e:
            return f"deterministic anchor (LLM unavailable: {type(e).__name__})"
