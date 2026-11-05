"""Controlled 2x2 (pre-registration v3): the design's invariants, with stub agents and a fake LLM."""

import json
from datetime import date

from agents.base_agent import AgentDecision, InstrumentView
from coordination.controlled import CONTROLLED
from llm.datedgpt_scorer import DatedGPTScorer

U = ("SPY", "IEF", "GLD")


class Stub:
    def __init__(self, name, d):
        self.name, self.d = name, d

    def run(self, as_of):
        return AgentDecision(agent_name=self.name, as_of=as_of, views=[
            InstrumentView(instrument=s, direction=self.d, conviction=0.5, horizon="1w", factors={"x": 1.0})
            for s in U])


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, *, system, user, max_tokens=0, **_):
        self.calls.append((system, user))
        r = type("R", (), {})()
        r.text = json.dumps([{"instrument": s, "direction": -1, "conviction": 0.9, "horizon": "1w"} for s in U])
        return r


def _agents():
    return {"macro_regime": Stub("macro_regime", 1), "narrative_event": Stub("narrative_event", 1),
            "fundamentals_carry": Stub("fundamentals_carry", -1), "technical_trend": Stub("technical_trend", 1),
            "risk_correlation": Stub("risk_correlation", 0)}


def _run(key):
    llm = FakeLLM()
    p = CONTROLLED[key](_agents(), llm, U)
    p.cross = Stub("cross_asset_transmission", 1)   # own-evidence specialist, active without communication
    res = p.run(date(2023, 3, 3))
    return llm.calls, res


def test_revision_prompts_identical_within_condition_and_judge_only_in_judge_cells():
    a, _ = _run("ctrl_isolated_vote")
    b, _ = _run("ctrl_isolated_judge")
    c, _ = _run("ctrl_shared_vote")
    d, _ = _run("ctrl_shared_judge")
    assert len(a) == len(c) == 4 and len(b) == len(d) == 5          # 4 revisions (+1 judge call)
    assert a == b[:4] and c == d[:4]                                  # A/B and C/D share revision prompts
    assert b[4] [0] == d[4][0]                                         # same judge system prompt
    for (sa, ua), (sc, uc) in zip(a, c):
        assert sa == sc                                               # same system prompt in both conditions
        pa, pc = json.loads(ua), json.loads(uc)
        everyone = {"macro_regime", "narrative_event", "fundamentals_carry", "technical_trend",
                    "cross_asset_transmission"}
        assert "peer_views" not in pa
        assert set(pc["peer_views"]["SPY"]) == everyone - {_who(sa)}   # the other four specialists
        assert {k: v for k, v in pc.items() if k != "peer_views"} == pa   # same evidence and first view


def _who(system):
    return {"Macro Regime": "macro_regime", "Narrative/Event": "narrative_event",
            "Fundamentals/Carry": "fundamentals_carry", "Cross-Asset Market": "cross_asset_transmission"}[
        system.split("You are the ")[1].split(" specialist")[0]]


def test_cross_asset_active_and_vote_uses_revised_views():
    _, res = _run("ctrl_isolated_vote")
    xa = next(d for d in res.per_agent_decisions if d.agent_name == "cross_asset_transmission")
    assert all(v.direction == -1 for v in xa.views)     # revised (fake LLM says short)
    # vote over revised views: 4 revised shorts (0.9) vs the technical long (0.5) -> short
    assert all(v.direction == -1 for v in res.final_views)
    tech = next(d for d in res.per_agent_decisions if d.agent_name == "technical_trend")
    assert all(v.direction == 1 for v in tech.views)    # rule-based technical view is not revised


def test_datedgpt_questions_for_new_prompts():
    calls, _ = _run("ctrl_shared_judge")
    rev_sys, rev_user = calls[0]
    assert DatedGPTScorer.kind(rev_sys) == "v2"
    q = DatedGPTScorer.questions(rev_sys, json.loads(rev_user))
    assert set(q) == set(U) and "Other analysts' first assessments" in q["SPY"] and "quarter" in q["SPY"]
    jq = DatedGPTScorer.questions(calls[-1][0], json.loads(calls[-1][1]))
    assert "Analysts' final assessments" in jq["GLD"]
    from agents.cross_asset_market import SYSTEM_PROMPT
    assert DatedGPTScorer.kind(SYSTEM_PROMPT) == "v2"
    assert DatedGPTScorer.kind("You are the Macro Regime Agent. ...") == "v1"
