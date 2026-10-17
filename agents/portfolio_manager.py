"""Portfolio Manager Agent (spec §4.1, §5).

Aggregates per-instrument views from the 6 specialists according to the
active coordination protocol, then emits final target weights via the
portfolio construction layer (§5).
"""

from __future__ import annotations

from datetime import date

from agents.base_agent import AgentDecision, BaseAgent, InstrumentView

VOTE_THRESHOLD = 0.1


def conviction_weighted_vote(pairs: list[tuple[str, int, float]]) -> tuple[int, float, float]:
    """Shared aggregation rule: (agent, direction, conviction) -> (direction, conviction, score).

    score = sum(d*c)/sum(c) over directional agents (the risk agent is skipped);
    direction = sign(score) if |score| > 0.1; conviction = min(|score|, 1).
    """
    num = den = 0.0
    for agent, d, c in pairs:
        if agent.startswith("risk_correlation"):
            continue
        num += d * c
        den += c
    score = (num / den) if den > 0 else 0.0
    direction = int(1 if score > VOTE_THRESHOLD else (-1 if score < -VOTE_THRESHOLD else 0))
    return direction, min(abs(score), 1.0), score


class PortfolioManagerAgent(BaseAgent):
    name = "portfolio_manager"

    def set_inputs(self, specialist_decisions: list[AgentDecision]) -> None:
        self._specialists = specialist_decisions

    def extract_factors(self, as_of: date) -> dict[str, dict[str, float]]:
        specs: list[AgentDecision] = getattr(self, "_specialists", [])
        out: dict[str, dict[str, float]] = {}
        for sym in self.universe:
            f: dict[str, float] = {}
            for d in specs:
                v = d.view_for(sym)
                if v is None:
                    continue
                f[f"{d.agent_name}_dir"] = float(v.direction)
                f[f"{d.agent_name}_conv"] = v.conviction
            out[sym] = f
        return out

    def decide(
        self,
        as_of: date,
        factors_by_instrument: dict[str, dict[str, float]],
    ) -> list[InstrumentView]:
        # Default conviction-weighted average. Coordination protocols may
        # override this by injecting a different aggregator function.
        views: list[InstrumentView] = []
        for sym, f in factors_by_instrument.items():
            pairs = [(k[: -len("_dir")], int(v), float(f.get(k.replace("_dir", "_conv"), 0.0)))
                     for k, v in f.items() if k.endswith("_dir")]
            direction, _, score = conviction_weighted_vote(pairs)
            views.append(
                InstrumentView(
                    instrument=sym,
                    direction=direction,
                    conviction=min(abs(score), 1.0),
                    horizon="1w",
                    factors={**f, "aggregated_score": score},
                    rationale=f"PM aggregate score={score:.3f}",
                )
            )
        return views
