"""Macro Regime Agent (spec §4.1).

Inputs: ALFRED first-release macro series + FOMC text.
Output: regime quadrant (growth × inflation) + per-instrument view.

Two-stage decide():
1. Classify regime (growth × inflation) from the 3-month change in the
   year-over-year growth rate of industrial production (growth) and of CPI
   (inflation), computed from first-release, already-published values.
   (An earlier version used the 3-month change in the *levels* of nominal GDP
   and CPI, which are positive in almost every period, so the regime was
   effectively constant at growth-up x inflation-up.)
2. Apply a coarse `_macro_priors` table → anchor views.
3. If an LLM is available, refine those anchor views; else return the anchor.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from agents._llm_helpers import views_from_llm_or_anchor
from agents.base_agent import BaseAgent, InstrumentView
from data.fetchers.alfred import HEADLINE_SERIES, first_release, prefetch_window

SYSTEM_PROMPT = (
    "You are the Macro Regime Agent. Inputs are first-release ALFRED macro "
    "series (level + 3-month change) and a per-instrument anchor view from a "
    "growth × inflation regime prior. Refine direction and conviction with the "
    "macro context. Never invent factors. Output strict JSON: a list of "
    '{"instrument": SYMBOL, "direction": -1|0|1, "conviction": 0..1, '
    '"horizon": "1q", "rationale": "<=240 chars cite which series drove it"}.'
)


class MacroRegimeAgent(BaseAgent):
    name = "macro_regime"

    def __init__(
        self,
        llm_client,
        universe: tuple[str, ...],
        cell_window: tuple[date, date] | None = None,
    ):
        super().__init__(llm_client, universe)
        self.cell_window = cell_window
        self._prefetched = False

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        if self.cell_window and not self._prefetched:
            # Prime ALFRED cache with the full cell window — eliminates per-rebalance
            # network calls for every HEADLINE series.
            cell_start = self.cell_window[0] - timedelta(days=560)
            cell_end = self.cell_window[1]
            prefetch_window(HEADLINE_SERIES.values(), cell_start, cell_end)
            self._prefetched = True
        start = as_of - timedelta(days=560)   # >= 15 published months for YoY changes
        macro: dict[str, float] = {}
        for label, sid in HEADLINE_SERIES.items():
            try:
                s = first_release(sid, start, as_of)
                macro[label] = float(s.iloc[-1]) if len(s) else float("nan")
                macro[f"{label}_chg_3m"] = _change_over_days(s, 90)
            except Exception:
                macro[label] = float("nan")
                macro[f"{label}_chg_3m"] = float("nan")
                continue
            if label in ("cpi_yoy", "industrial_production"):
                yoy = _yoy_rate_monthly(s)
                macro[f"{label}_yoy_rate"] = float(yoy.iloc[-1]) if len(yoy) else float("nan")
                macro[f"{label}_yoy_rate_chg_3m"] = _change_over_days(yoy, 90) if len(yoy) else float("nan")
        return {sym: dict(macro) for sym in self.universe}

    def decide(
        self,
        as_of: date,
        factors_by_instrument: dict[str, dict[str, float]],
    ) -> list[InstrumentView]:
        macro = next(iter(factors_by_instrument.values()), {})
        regime = self._classify_regime(macro)
        prior = _macro_priors(regime)

        anchor: list[InstrumentView] = []
        for sym, f in factors_by_instrument.items():
            d = prior.get(sym, 0)
            anchor.append(
                InstrumentView(
                    instrument=sym,
                    direction=d,
                    conviction=0.4 if d != 0 else 0.0,
                    horizon="1q",
                    factors={**f, "regime": _regime_to_int(regime)},
                    rationale=f"regime={regime}; prior table",
                )
            )

        return views_from_llm_or_anchor(
            self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_payload={
                "as_of": as_of.isoformat(),
                "regime": regime,
                "macro": macro,
                "anchor": [
                    {"instrument": v.instrument, "direction": v.direction, "conviction": v.conviction}
                    for v in anchor
                ],
            },
            universe=self.universe,
            default_horizon="1q",
            anchor_views=anchor,
            agent_name=self.name,
        )

    @staticmethod
    def _classify_regime(m: dict[str, float]) -> str:
        def _ok(x):
            return x is not None and x == x  # not NaN
        g = m.get("industrial_production_yoy_rate_chg_3m")
        i = m.get("cpi_yoy_yoy_rate_chg_3m")
        growth_up = (g > 0) if _ok(g) else ((m.get("gdp_chg_3m") or 0.0) > 0)
        infl_up = (i > 0) if _ok(i) else False
        return f"{'growth_up' if growth_up else 'growth_down'}_x_{'infl_up' if infl_up else 'infl_down'}"


def _macro_priors(regime: str) -> dict[str, int]:
    table = {
        "growth_up_x_infl_up": {"SPY": 1, "QQQ": 1, "IWM": 1, "GLD": 1, "USO": 1, "IEF": -1, "SHY": -1, "UUP": 0, "EEM": 1, "BTC": 1},
        "growth_up_x_infl_down": {"SPY": 1, "QQQ": 1, "IEF": 1, "GLD": 0, "USO": 0, "UUP": 0, "EEM": 1, "BTC": 1, "SHY": 1, "IWM": 1},
        "growth_down_x_infl_up": {"SPY": -1, "QQQ": -1, "GLD": 1, "USO": 0, "IEF": -1, "UUP": 1, "EEM": -1, "BTC": -1, "SHY": 0, "IWM": -1},
        "growth_down_x_infl_down": {"SPY": -1, "QQQ": -1, "GLD": 0, "USO": -1, "IEF": 1, "UUP": 1, "EEM": -1, "BTC": -1, "SHY": 1, "IWM": -1},
    }
    return table.get(regime, {})


def _regime_to_int(r: str) -> float:
    return {
        "growth_up_x_infl_up": 0.0,
        "growth_up_x_infl_down": 1.0,
        "growth_down_x_infl_up": 2.0,
        "growth_down_x_infl_down": 3.0,
    }.get(r, -1.0)


def _change_over_days(s: pd.Series, days: int) -> float:
    """Difference between the latest value and the value ~`days` ago.

    Works for daily/monthly/quarterly series alike — falls back to the
    earliest available observation when the requested lookback predates the
    series. Returns NaN only if there are fewer than 2 observations.
    """
    if len(s) < 2:
        return float("nan")
    idx = pd.to_datetime(s.index)
    target = idx[-1] - pd.Timedelta(days=days)
    prior_mask = idx <= target
    if not prior_mask.any():
        return float("nan")
    prior_val = float(s.values[prior_mask][-1])
    return float(s.iloc[-1] - prior_val)


def _yoy_rate_monthly(s: pd.Series) -> pd.Series:
    """Year-over-year percent change of a monthly level series (12 observations back)."""
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    return (s / s.shift(12) - 1.0).dropna() * 100.0
