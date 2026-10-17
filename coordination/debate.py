"""Cell #5 — Debate / bull-bear (spec §6.2, TradingAgents-style).

Each instrument gets a short structured debate: bull case (built from
positive-direction specialists' factors) vs bear case (negative-direction).
Adjudicator (LLM or deterministic) emits the final view.
"""

from __future__ import annotations

import json
from datetime import date

from agents.base_agent import AgentDecision, InstrumentView
from agents._llm_helpers import parse_json_object, tally
from coordination.base import CoordinationProtocol, CoordinationResult


class DebateProtocol(CoordinationProtocol):
    name = "debate"

    def __init__(self, agents, llm_client, universe: tuple[str, ...], rounds: int = 1):
        super().__init__(agents)
        self.llm = llm_client
        self.universe = universe
        self.rounds = rounds

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
            bull, bear = [], []
            for d in decisions:
                v = d.view_for(sym)
                if v is None:
                    continue
                if v.direction > 0:
                    bull.append((d.agent_name, v))
                elif v.direction < 0:
                    bear.append((d.agent_name, v))

            if self.llm is None or (not bull and not bear):
                d_int = 1 if len(bull) > len(bear) else (-1 if len(bear) > len(bull) else 0)
                conv = abs(len(bull) - len(bear)) / max(len(bull) + len(bear), 1)
                final.append(
                    InstrumentView(
                        instrument=sym,
                        direction=d_int,
                        conviction=float(conv),
                        horizon="1w",
                        rationale=f"deterministic adjudication: bulls={len(bull)}, bears={len(bear)}",
                    )
                )
                continue

            tally("debate_adjudicator", "calls")
            try:
                resp = self.llm.complete(
                    system=(
                        "You are an adjudicator between bull and bear analysts on one "
                        "instrument. Read both sides' factors, then emit one JSON object: "
                        '{"direction": -1|0|1, "conviction": 0..1, "rationale": str}.'
                    ),
                    user=json.dumps(
                        {
                            "instrument": sym,
                            "as_of": as_of.isoformat(),
                            "bull": [{"agent": n, "direction": v.direction, "conviction": v.conviction,
                                      "factors": v.factors} for n, v in bull],
                            "bear": [{"agent": n, "direction": v.direction, "conviction": v.conviction,
                                      "factors": v.factors} for n, v in bear],
                        },
                        default=float,
                    ),
                    max_tokens=400,
                )
                obj = parse_json_object(resp.text)
                if obj is None:
                    raise ValueError("unparseable adjudicator output")
                final.append(
                    InstrumentView(
                        instrument=sym,
                        direction=int(obj.get("direction", 0)),
                        conviction=float(obj.get("conviction", 0.0)),
                        horizon="1w",
                        rationale=str(obj.get("rationale", "")),
                    )
                )
            except Exception as e:
                self._record_failure("llm_adjudicator", e)
                tally("debate_adjudicator", "call_failed")
                final.append(
                    InstrumentView(instrument=sym, direction=0, conviction=0.0, horizon="1w")
                )

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=final,
            per_agent_decisions=decisions,
        )
