"""Cell #6 — Deterministic anchor only (spec §6.2).

Piotroski (where applicable) + GARCH vol regime + multi-horizon momentum.
NO LLM. This is the "ARIMA / rule-based" baseline FINSABER calls out as
the surprising winner. Critical control variable.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from agents.base_agent import AgentDecision, InstrumentView
from coordination.base import CoordinationProtocol, CoordinationResult
from data.fetchers.prices import fetch_panel
from factors.momentum import multi_horizon_momentum
from factors.risk import ewma_volatility, garch_forecast


class DeterministicOnlyProtocol(CoordinationProtocol):
    name = "deterministic_only"

    def __init__(self, agents, universe: tuple[str, ...]):
        super().__init__(agents)
        self.universe = universe

    def run(self, as_of: date) -> CoordinationResult:
        start = as_of - timedelta(days=400)
        prices = fetch_panel(list(self.universe), start, as_of + timedelta(days=1), field="Adj Close")  # end is exclusive
        moms = multi_horizon_momentum(prices) if not prices.empty else None

        final: list[InstrumentView] = []
        for sym in self.universe:
            if moms is None or sym not in prices.columns:
                final.append(
                    InstrumentView(instrument=sym, direction=0, conviction=0.0, horizon="1m")
                )
                continue
            m1 = moms["1m"][sym].iloc[-1]
            m3 = moms["3m"][sym].iloc[-1]
            m6 = moms["6m"][sym].iloc[-1]
            m12 = moms["12m"][sym].iloc[-1]
            score = np.nansum([np.sign(m1) * 0.15, np.sign(m3) * 0.25, np.sign(m6) * 0.30, np.sign(m12) * 0.30])
            r = prices[sym].pct_change().dropna()
            vol_ann = float(ewma_volatility(r).iloc[-1]) if len(r) else 0.0
            # vol regime tilt: scale conviction down in high-vol regimes
            vol_scaler = 1.0 / (1.0 + max(vol_ann - 0.20, 0.0) * 2)
            conv = min(abs(score) * vol_scaler, 1.0)
            d = int(np.sign(score)) if abs(score) > 0.05 else 0
            final.append(
                InstrumentView(
                    instrument=sym,
                    direction=d,
                    conviction=float(conv),
                    horizon="1m",
                    factors={
                        "mom_1m": float(m1) if pd.notna(m1) else 0.0,
                        "mom_3m": float(m3) if pd.notna(m3) else 0.0,
                        "mom_6m": float(m6) if pd.notna(m6) else 0.0,
                        "mom_12m": float(m12) if pd.notna(m12) else 0.0,
                        "ewma_vol_ann": vol_ann,
                        "vol_scaler": vol_scaler,
                    },
                    rationale="deterministic anchor: momentum × inverse-vol scaler",
                )
            )

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=final,
        )
