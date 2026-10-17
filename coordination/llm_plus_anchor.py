"""Cell #7 — LLM + deterministic anchor (spec §6.2).

LLM receives the deterministic anchor's view as one of its inputs alongside
the specialists' factors. Tests whether anchoring the LLM in a known-good
deterministic baseline improves coordination decisions.
"""

from __future__ import annotations

import json
from datetime import date

from agents.base_agent import AgentDecision, InstrumentView
from coordination.base import CoordinationProtocol, CoordinationResult
from coordination.deterministic_only import DeterministicOnlyProtocol


class LLMPlusAnchorProtocol(CoordinationProtocol):
    name = "llm_plus_anchor"

    def __init__(self, agents, llm_client, universe: tuple[str, ...]):
        super().__init__(agents)
        self.llm = llm_client
        self.universe = universe
        self._anchor = DeterministicOnlyProtocol(agents, universe)

    def run(self, as_of: date) -> CoordinationResult:
        anchor_result = self._anchor.run(as_of)
        anchor_by_sym = {v.instrument: v for v in anchor_result.final_views}

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
            anchor_v = anchor_by_sym.get(sym)
            specialist_views = [d.view_for(sym) for d in decisions]
            specialist_views = [v for v in specialist_views if v is not None]

            if self.llm is None or anchor_v is None:
                final.append(
                    anchor_v
                    or InstrumentView(instrument=sym, direction=0, conviction=0.0, horizon="1w")
                )
                continue

            try:
                resp = self.llm.complete(
                    system=(
                        "You are a portfolio manager. The deterministic anchor is your "
                        "default — only deviate when the specialists give consistent and "
                        "compelling reasons to. Emit one JSON object: "
                        '{"direction": -1|0|1, "conviction": 0..1, "rationale": str}.'
                    ),
                    user=json.dumps(
                        {
                            "instrument": sym,
                            "anchor": {
                                "direction": anchor_v.direction,
                                "conviction": anchor_v.conviction,
                                "factors": anchor_v.factors,
                            },
                            "specialists": [
                                {"factors": v.factors, "direction": v.direction, "conv": v.conviction}
                                for v in specialist_views
                            ],
                        },
                        default=float,
                    ),
                    max_tokens=400,
                )
                obj = json.loads(resp.text)
                final.append(
                    InstrumentView(
                        instrument=sym,
                        direction=int(obj.get("direction", anchor_v.direction)),
                        conviction=float(obj.get("conviction", anchor_v.conviction)),
                        horizon="1w",
                        rationale=str(obj.get("rationale", "")),
                    )
                )
            except Exception as e:
                self._record_failure("llm_refiner", e)
                final.append(anchor_v)

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=final,
            per_agent_decisions=decisions,
            metadata={"anchor": "deterministic_only"},
        )
