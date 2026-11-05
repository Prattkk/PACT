"""Cross-asset specialist with its own market evidence (pre-registration v3, controlled 2x2).

The frozen cross-asset agent reads only the macro and narrative agents' views, so it is neutral
whenever communication is switched off. In the controlled experiment the specialist must be
active in every cell, so this variant reasons from its own point-in-time market evidence: per
instrument the 20-session return and the 63-session return correlations with the main
transmission drivers (SPY equities, UUP dollar, IEF Treasuries, USO oil), plus the drivers'
own 20-session moves. It never sees other specialists' views in its first pass.

Deterministic anchor: the correlation-weighted sum of the drivers' standardised 20-session
moves (excluding the instrument itself), i.e. the move the instrument's recent co-movement
implies; direction = sign if |score| > 0.25, conviction = min(|score| / 2, 1).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from agents._llm_helpers import views_from_llm_or_anchor
from agents.base_agent import BaseAgent, InstrumentView
from data.fetchers.prices import fetch_panel

DRIVERS = ("SPY", "UUP", "IEF", "USO")
SYSTEM_PROMPT = (
    "You are the Cross-Asset Market Agent. Inputs per instrument: its 20-session return, its "
    "63-session return correlations with SPY (equities), UUP (US dollar), IEF (Treasuries) and USO "
    "(oil), the drivers' own standardised 20-session moves, and a deterministic anchor (the move "
    "implied by correlation-weighted driver moves). Reason about cross-asset transmission chains "
    "(e.g. dollar up -> EEM down; yields up -> GLD down; oil up -> inflation -> IEF down). Refine "
    "the anchor only when transmission logic supports it. Never invent factors. Output strict "
    'JSON: a list of {"instrument": SYMBOL, "direction": -1|0|1, "conviction": 0..1, '
    '"horizon": "1m", "rationale": "<=240 chars cite the transmission chain"}.'
)


class CrossAssetMarketAgent(BaseAgent):
    name = "cross_asset_transmission"   # same specialist role in the vote and in the decision trace

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        px = fetch_panel(list(self.universe), as_of - timedelta(days=150), as_of + timedelta(days=1),
                         field="Adj Close")  # end is exclusive: closes up to the decision close
        if px.empty or len(px) < 64:
            return {s: {} for s in self.universe}
        r = px.pct_change().iloc[-63:]
        ret20 = px.iloc[-1] / px.iloc[-21] - 1.0
        vol20 = r.iloc[-20:].std() * np.sqrt(20)
        drv = {d: float(ret20[d] / vol20[d]) if vol20.get(d, 0) > 0 else 0.0 for d in DRIVERS if d in px}
        corr = r.corr()
        out: dict[str, dict[str, float]] = {}
        for s in self.universe:
            if s not in px:
                out[s] = {}
                continue
            f = {"ret_20d": round(float(ret20[s]), 4)}
            for d in DRIVERS:
                if d in corr and s != d:
                    f[f"corr_{d.lower()}_63d"] = round(float(corr.loc[s, d]), 3)
            for d, z in drv.items():
                f[f"{d.lower()}_move_z"] = round(z, 3)
            out[s] = f
        return out

    @staticmethod
    def _anchor(sym: str, f: dict[str, float]) -> InstrumentView:
        score = sum(f.get(f"corr_{d.lower()}_63d", 0.0) * f.get(f"{d.lower()}_move_z", 0.0)
                    for d in DRIVERS if d != sym)
        direction = (1 if score > 0 else -1) if abs(score) > 0.25 else 0
        return InstrumentView(instrument=sym, direction=direction, conviction=min(abs(score) / 2, 1.0),
                              horizon="1m", factors=f, rationale="correlation-weighted driver moves")

    def decide(self, as_of: date, factors_by_instrument: dict[str, dict[str, float]]) -> list[InstrumentView]:
        anchor = [self._anchor(s, f) for s, f in factors_by_instrument.items()]
        if all(not f for f in factors_by_instrument.values()):
            return anchor
        return views_from_llm_or_anchor(
            self.llm, system_prompt=SYSTEM_PROMPT,
            user_payload={"as_of": as_of.isoformat(), "factors": factors_by_instrument,
                          "anchor": [{"instrument": v.instrument, "direction": v.direction,
                                      "conviction": round(v.conviction, 3)} for v in anchor]},
            universe=self.universe, default_horizon="1m", anchor_views=anchor, agent_name="cross_asset_market")
