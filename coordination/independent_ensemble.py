"""Cell #2 — Independent ensemble (spec §6.2).

Same N specialists as the coordinated cells, but **no communication**.
Decisions averaged or majority-voted. This is the real test of whether
coordination > parallelism — the gap that "Stop Overvaluing MAD" calls out.
"""

from __future__ import annotations

from datetime import date
from statistics import median

from agents.base_agent import AgentDecision, InstrumentView
from coordination.base import CoordinationProtocol, CoordinationResult


class IndependentEnsembleProtocol(CoordinationProtocol):
    name = "independent_ensemble"

    def __init__(self, agents, universe: tuple[str, ...], aggregator: str = "majority"):
        super().__init__(agents)
        self.universe = universe
        if aggregator not in ("majority", "mean"):
            raise ValueError(f"unknown aggregator {aggregator}")
        self.aggregator = aggregator

    def run(self, as_of: date) -> CoordinationResult:
        decisions: list[AgentDecision] = []
        for name, agent in self.agents.items():
            if name == "portfolio_manager":
                continue
            try:
                decisions.append(agent.run(as_of))
            except Exception as e:
                self._record_failure(name, e)

        final: list[InstrumentView] = []
        for sym in self.universe:
            dirs, convs = [], []
            for d in decisions:
                v = d.view_for(sym)
                if v is None or d.agent_name == "risk_correlation":
                    continue
                dirs.append(v.direction)
                convs.append(v.conviction)
            if not dirs:
                final.append(
                    InstrumentView(instrument=sym, direction=0, conviction=0.0, horizon="1w")
                )
                continue
            if self.aggregator == "majority":
                d = int(round(median(dirs)))
            else:
                d_score = sum(d * c for d, c in zip(dirs, convs)) / max(sum(convs), 1e-9)
                d = int(1 if d_score > 0.1 else (-1 if d_score < -0.1 else 0))
            final.append(
                InstrumentView(
                    instrument=sym,
                    direction=d,
                    conviction=float(sum(convs) / len(convs)),
                    horizon="1w",
                    rationale=f"ensemble {self.aggregator}, n={len(dirs)}",
                )
            )

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=final,
            per_agent_decisions=decisions,
        )
