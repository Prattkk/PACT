"""Cell #1 — Single-LLM monolith (spec §6.2).

One prompt, all data, one decision. Baseline that TradingAgents/FinCon-style
papers rarely publish honestly. We fold every specialist's *factors* into
one prompt and ask the LLM to emit per-instrument views directly.
"""

from __future__ import annotations

import json
from datetime import date

from agents.base_agent import AgentDecision, InstrumentView
from coordination.base import CoordinationProtocol, CoordinationResult


class SingleAgentProtocol(CoordinationProtocol):
    name = "single_agent"

    def __init__(self, agents, llm_client, universe: tuple[str, ...]):
        super().__init__(agents)
        self.llm = llm_client
        self.universe = universe

    def run(self, as_of: date) -> CoordinationResult:
        all_factors: dict[str, dict[str, dict[str, float]]] = {}
        for name, agent in self.agents.items():
            try:
                all_factors[name] = agent.extract_factors(as_of)
            except Exception as e:
                self._record_failure(name, e)
                all_factors[name] = {}

        # Single LLM call combining everything. In offline mode (no API key),
        # we fall back to a uniform-zero view so the harness still runs.
        views: list[InstrumentView] = []
        if self.llm is None:
            for sym in self.universe:
                views.append(
                    InstrumentView(instrument=sym, direction=0, conviction=0.0, horizon="1w")
                )
        else:
            try:
                resp = self.llm.complete(
                    system=(
                        "You are a single multi-asset PM. Read all factor blobs and emit a "
                        "JSON list of {instrument, direction (-1|0|1), conviction (0..1), "
                        "horizon (1w|1m|1q), rationale}. Do not invent factors."
                    ),
                    user=json.dumps(
                        {"as_of": as_of.isoformat(), "factors": all_factors},
                        default=float,
                    ),
                    max_tokens=2048,
                )
                parsed = _safe_parse_views(resp.text, self.universe)
                views = parsed
            except Exception as e:
                self._record_failure("llm_monolith", e)
                views = [
                    InstrumentView(instrument=s, direction=0, conviction=0.0, horizon="1w")
                    for s in self.universe
                ]

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=views,
            metadata={"factor_blob_keys": list(all_factors.keys())},
        )


def _safe_parse_views(text: str, universe: tuple[str, ...]) -> list[InstrumentView]:
    """Parse the LLM's JSON view list. Tolerates markdown code fences
    (```json...```) and prose around the JSON via the shared helper.
    Falls back to zero-views on parse failure."""
    from agents._llm_helpers import parse_view_list

    out = parse_view_list(text, universe, default_horizon="1w")
    seen = {v.instrument for v in out}
    for s in universe:
        if s not in seen:
            out.append(InstrumentView(instrument=s, direction=0, conviction=0.0, horizon="1w"))
    return out
