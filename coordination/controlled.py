"""Controlled communication x adjudication experiment (pre-registration v3, section 1).

    A ctrl_isolated_vote   isolated revision      + deterministic vote
    B ctrl_isolated_judge  isolated revision      + LLM judge
    C ctrl_shared_vote     communicating revision + deterministic vote
    D ctrl_shared_judge    communicating revision + the same LLM judge

First pass (identical in all cells): the frozen macro, narrative, fundamentals and technical
specialists, the risk agent, and the cross-asset specialist with its own market evidence
(agents/cross_asset_market.py), which never sees other views in the first pass.

Second pass: one revision call per LLM specialist (macro, narrative, fundamentals, cross-asset)
with the same system prompt and token budget in both conditions. The isolated payload holds the
specialist's own per-instrument evidence and first view; the communicating payload adds the other
four specialists' first-pass direction and conviction. The rule-based technical specialist keeps
its first view. A reply that cannot be parsed keeps the specialist's first view (counted).

Aggregation: the frozen conviction-weighted vote, or one judge call per date over every
specialist's revised view and evidence. A judge reply that cannot be parsed falls back to the
vote for that date (counted). Cells A/B (and C/D) send identical revision prompts, so the response
cache returns identical revised views; portfolio construction is the frozen engine's.
"""

from __future__ import annotations

import json
from datetime import date

from agents._llm_helpers import tally, views_from_llm_or_anchor
from agents.base_agent import AgentDecision, InstrumentView
from agents.cross_asset_market import CrossAssetMarketAgent
from agents.portfolio_manager import conviction_weighted_vote
from coordination.base import CoordinationProtocol, CoordinationResult

FIRST_PASS = ("macro_regime", "narrative_event", "fundamentals_carry", "technical_trend")
REVISED = ("macro_regime", "narrative_event", "fundamentals_carry", "cross_asset_transmission")
ROLE = {"macro_regime": "Macro Regime", "narrative_event": "Narrative/Event",
        "fundamentals_carry": "Fundamentals/Carry", "cross_asset_transmission": "Cross-Asset Market",
        "technical_trend": "Technical Trend"}
HORIZON = {"macro_regime": "1q", "narrative_event": "1w", "fundamentals_carry": "1q",
           "cross_asset_transmission": "1m"}

REVISION_PROMPT = (
    "You are the {role} specialist in a team of analysts, making a second, final assessment. The "
    "payload holds your evidence per instrument (`evidence`), your first view (`first_view`) and, "
    "if present, the other specialists' first views (`peer_views`). Re-examine the evidence and give "
    "your final view for every instrument; change a view only when the evidence supports it. Never "
    "invent data. Output strict JSON: a list of {{\"instrument\": SYMBOL, \"direction\": -1|0|1, "
    "\"conviction\": 0..1, \"horizon\": \"{h}\", \"rationale\": \"<=200 chars\"}}."
)
JUDGE_PROMPT = (
    "You are the portfolio adjudicator for a team of specialists. For each instrument the payload "
    "lists every specialist's final direction (-1/0/1), conviction (0..1) and evidence. Weigh the "
    "specialists' cases and decide the team's position for the next week. Never invent data. Output "
    "strict JSON: a list with one entry per instrument, {\"instrument\": SYMBOL, \"direction\": "
    "-1|0|1, \"conviction\": 0..1, \"horizon\": \"1w\", \"rationale\": \"<=200 chars\"}."
)


def _views(d: AgentDecision) -> dict[str, dict[str, float]]:
    return {v.instrument: {"direction": v.direction, "conviction": round(v.conviction, 3)} for v in d.views}


class ControlledProtocol(CoordinationProtocol):
    name = "ctrl"
    shared: bool = False
    judge: bool = False

    def __init__(self, agents, llm_client, universe: tuple[str, ...]):
        super().__init__(agents)
        self.llm = llm_client
        self.universe = universe
        self.cross = CrossAssetMarketAgent(llm_client, universe)

    # ----------------------------------------------------------------- passes
    def _first_pass(self, as_of: date) -> dict[str, AgentDecision]:
        out = {}
        for name in FIRST_PASS:
            out[name] = self.agents[name].run(as_of)
        out["cross_asset_transmission"] = self.cross.run(as_of)
        return out

    def _revise(self, as_of: date, first: dict[str, AgentDecision]) -> dict[str, AgentDecision]:
        revised = dict(first)
        if self.llm is None:
            return revised
        for name in REVISED:
            d = first[name]
            payload = {"as_of": as_of.isoformat(),
                       "evidence": {v.instrument: v.factors for v in d.views},
                       "first_view": _views(d)}
            if self.shared:
                payload["peer_views"] = {
                    sym: {p: first[p].view_for(sym) and {"direction": first[p].view_for(sym).direction,
                                                         "conviction": round(first[p].view_for(sym).conviction, 3)}
                          for p in first if p != name}
                    for sym in self.universe}
            views = views_from_llm_or_anchor(
                self.llm, system_prompt=REVISION_PROMPT.format(role=ROLE[name], h=HORIZON[name]),
                user_payload=payload, universe=self.universe, default_horizon=HORIZON[name],
                anchor_views=d.views, agent_name=f"{name}_revision")
            revised[name] = AgentDecision(agent_name=name, as_of=as_of, views=views)
        return revised

    def _vote(self, revised: dict[str, AgentDecision]) -> list[InstrumentView]:
        final = []
        for sym in self.universe:
            pairs = [(n, d.view_for(sym).direction, d.view_for(sym).conviction)
                     for n, d in revised.items() if d.view_for(sym) is not None]
            direction, conv, _ = conviction_weighted_vote(pairs)
            final.append(InstrumentView(instrument=sym, direction=direction, conviction=float(conv),
                                        horizon="1w", rationale=f"vote, n={len(pairs)}"))
        return final

    def _judge(self, as_of: date, revised: dict[str, AgentDecision], vote: list[InstrumentView]):
        payload = {"as_of": as_of.isoformat(), "instruments": {
            sym: [{"agent": n, "direction": d.view_for(sym).direction,
                   "conviction": round(d.view_for(sym).conviction, 3), "evidence": d.view_for(sym).factors}
                  for n, d in revised.items() if d.view_for(sym) is not None]
            for sym in self.universe}}
        tally("ctrl_judge", "calls")
        return views_from_llm_or_anchor(
            self.llm, system_prompt=JUDGE_PROMPT, user_payload=payload, universe=self.universe,
            default_horizon="1w", anchor_views=vote, agent_name="ctrl_judge_parse", max_tokens=2000)

    def run(self, as_of: date) -> CoordinationResult:
        first = self._first_pass(as_of)
        revised = self._revise(as_of, first)
        vote = self._vote(revised)
        final = self._judge(as_of, revised, vote) if (self.judge and self.llm is not None) else vote
        risk = self.agents["risk_correlation"].run(as_of)
        decisions = [revised[n] for n in (*FIRST_PASS, "cross_asset_transmission")] + [risk]
        return CoordinationResult(protocol=self.name, as_of=as_of, final_views=final,
                                  per_agent_decisions=decisions)


def _cell(key: str, shared: bool, judge: bool):
    return type(key, (ControlledProtocol,), {"name": key, "shared": shared, "judge": judge})


CONTROLLED = {
    "ctrl_isolated_vote": _cell("ctrl_isolated_vote", False, False),
    "ctrl_isolated_judge": _cell("ctrl_isolated_judge", False, True),
    "ctrl_shared_vote": _cell("ctrl_shared_vote", True, False),
    "ctrl_shared_judge": _cell("ctrl_shared_judge", True, True),
}
