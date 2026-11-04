"""Cell #3 — Sequential 3-phase pipeline (spec §6.2).

Phase 1: Macro / screener (macro_regime + narrative_event in parallel)
Phase 2: Deep analysis (cross_asset_transmission, technical_trend, fundamentals_carry)
Phase 3: Risk + portfolio aggregation (risk_correlation → portfolio_manager)

Information flows downstream only — no debate, no voting.
"""

from __future__ import annotations

from datetime import date

from agents.base_agent import AgentDecision
from coordination.base import CoordinationProtocol, CoordinationResult


class SequentialPipelineProtocol(CoordinationProtocol):
    name = "sequential_pipeline"

    def __init__(self, agents, universe: tuple[str, ...]):
        super().__init__(agents)
        self.universe = universe

    def run(self, as_of: date) -> CoordinationResult:
        decisions: list[AgentDecision] = []

        # Phase 1
        macro = self.agents["macro_regime"].run(as_of)
        narrative = self.agents["narrative_event"].run(as_of)
        decisions.extend([macro, narrative])

        # Phase 2
        xat = self.agents["cross_asset_transmission"]
        if hasattr(xat, "set_upstream"):
            xat.set_upstream(macro, narrative)
        decisions.append(xat.run(as_of))
        decisions.append(self.agents["technical_trend"].run(as_of))
        decisions.append(self.agents["fundamentals_carry"].run(as_of))

        # Phase 3
        decisions.append(self.agents["risk_correlation"].run(as_of))
        pm = self.agents["portfolio_manager"]
        if hasattr(pm, "set_inputs"):
            pm.set_inputs(decisions)
        pm_decision = pm.run(as_of)

        return CoordinationResult(
            protocol=self.name,
            as_of=as_of,
            final_views=pm_decision.views,
            per_agent_decisions=decisions + [pm_decision],
        )


class PlaceboPipelineProtocol(SequentialPipelineProtocol):
    """Control for the pipeline: the cross-asset agent is active (it votes)
    but receives the macro and narrative views shifted to the wrong
    instruments (a fixed cyclic derangement), so its vote carries no relevant
    upstream information. Pipeline minus placebo isolates the value of
    communicating upstream views; placebo minus ensemble isolates the effect of
    adding a cross-asset voter."""

    name = "pipeline_placebo"
    seed: int | None = None

    def run(self, as_of: date) -> CoordinationResult:
        decisions: list[AgentDecision] = []
        macro = self.agents["macro_regime"].run(as_of)
        narrative = self.agents["narrative_event"].run(as_of)
        decisions.extend([macro, narrative])
        xat = self.agents["cross_asset_transmission"]
        if hasattr(xat, "set_upstream"):
            xat.set_upstream(_derange(macro, self.universe, self.seed), _derange(narrative, self.universe, self.seed))
        decisions.append(xat.run(as_of))
        decisions.append(self.agents["technical_trend"].run(as_of))
        decisions.append(self.agents["fundamentals_carry"].run(as_of))
        decisions.append(self.agents["risk_correlation"].run(as_of))
        pm = self.agents["portfolio_manager"]
        if hasattr(pm, "set_inputs"):
            pm.set_inputs(decisions)
        pm_decision = pm.run(as_of)
        return CoordinationResult(protocol=self.name, as_of=as_of, final_views=pm_decision.views,
                                  per_agent_decisions=decisions + [pm_decision])


def _derange(decision: AgentDecision, universe: tuple[str, ...], seed: int | None = None) -> AgentDecision:
    """Relabel each view with another instrument: the next one in the universe (cyclic shift,
    the frozen placebo) or, with `seed`, a fixed random derangement (diagnostic placebos)."""
    syms = list(universe)
    if seed is None:
        nxt = {s: syms[(i + 1) % len(syms)] for i, s in enumerate(syms)}
    else:
        import random
        rng = random.Random(seed)
        while True:
            perm = syms[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(syms, perm)):
                break
        nxt = dict(zip(syms, perm))
    views = [v.model_copy(update={"instrument": nxt[v.instrument]}) if v.instrument in nxt else v
             for v in decision.views]
    return decision.model_copy(update={"views": views})


def _seeded_placebo(k: int):
    return type(f"PlaceboPipelineSeed{k}", (PlaceboPipelineProtocol,), {"name": f"pipeline_placebo_s{k}", "seed": k})


SEEDED_PLACEBOS = {f"pipeline_placebo_s{k}": _seeded_placebo(k) for k in range(1, 6)}
