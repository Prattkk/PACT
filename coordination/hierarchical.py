"""Cell #4 — Hierarchical / manager-analyst (spec §6.2, FinCon-style).

Manager (PM) issues per-instrument briefs to analysts; analysts (specialists)
return scoped views; manager re-aggregates with veto power on low-confidence
specialist outputs.
"""

from __future__ import annotations

from datetime import date

from agents.base_agent import AgentDecision, InstrumentView
from coordination.base import CoordinationProtocol, CoordinationResult

CONFIDENCE_VETO = 0.15


class HierarchicalProtocol(CoordinationProtocol):
    name = "hierarchical"

    def __init__(self, agents, universe: tuple[str, ...]):
        super().__init__(agents)
        self.universe = universe

    def run(self, as_of: date) -> CoordinationResult:
        # Round 1: specialists report independently
        specialists = [
            self.agents[name]
            for name in (
                "macro_regime",
                "narrative_event",
                "cross_asset_transmission",
                "technical_trend",
                "fundamentals_carry",
                "risk_correlation",
            )
            if name in self.agents
        ]
        round1: list[AgentDecision] = []
        for a in specialists:
            try:
                round1.append(a.run(as_of))
            except Exception as e:
                self._record_failure(getattr(a, "name", "specialist"), e)

        # Manager veto: drop low-confidence views before aggregation
        kept = [
            AgentDecision(
                agent_name=d.agent_name,
                as_of=d.as_of,
                views=[v for v in d.views if v.conviction >= CONFIDENCE_VETO],
            )
            for d in round1
        ]

        pm = self.agents["portfolio_manager"]
        if hasattr(pm, "set_inputs"):
            pm.set_inputs(kept)
        pm_decision = pm.run(as_of)

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=pm_decision.views,
            per_agent_decisions=round1 + [pm_decision],
            metadata={"veto_threshold": CONFIDENCE_VETO},
        )
