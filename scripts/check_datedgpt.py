"""Capability checks of the DatedGPT log-probability scorer (pre-registration v2, section 2).

C1 name-only prior, C2a within-call shuffle of instrument evidence, C2b evidence from another
week, C3 controlled directional pairs, C4 transmission of upstream views. Uses the frozen
vintages and the scorer's own question template. Writes results/tables/datedgpt_checks.json.
"""

from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from statistics import mean, median, pstdev

import numpy as np
from scipy.stats import spearmanr

from llm.datedgpt_scorer import AGENTS, DESCRIPTIONS, DatedGPTScorer
from scripts._common import REPO_ROOT, TABLES_DIR

SC = DatedGPTScorer()
MEMO: dict[tuple[int, str], float] = {}
CROSS = "You are the Cross-Asset"


def lo(v: int, q: str) -> float:
    if (v, q) not in MEMO:
        MEMO[(v, q)] = SC._log_odds(v, q)
    return MEMO[(v, q)]


def dirs(scores: dict[str, float]) -> dict[str, int]:
    """The scorer's own rule: z across instruments of the call, direction if |z| > 0.5."""
    mu, sd = mean(scores.values()), pstdev(scores.values())
    return {s: (0 if sd == 0 or abs((v - mu) / sd) <= 0.5 else (1 if v > mu else -1)) for s, v in scores.items()}


def spec(system):
    return next(v for k, v in AGENTS.items() if system.startswith(k))


def load_entries():
    out = []
    for f in glob.glob(str(REPO_ROOT / "data/cache/llm/datedgpt-instruct-scorer-v1/*.json")):
        d = json.load(open(f))
        c = json.loads(d["canonical_prompt"])
        if c["system"].startswith(CROSS):
            continue
        p = json.loads(c["user"])
        if not ("2022" <= p["as_of"][:4] <= "2024"):
            continue
        out.append({"system": c["system"], "payload": p, "vintage": int(d["response"]["vintage"]),
                    "log_odds": d["response"]["log_odds"]})
    return sorted(out, key=lambda e: (e["system"][:20], e["payload"]["as_of"]))


def evidence(e):
    """Per-instrument evidence blocks (data and anchor) of one call."""
    key = spec(e["system"])[0]
    p = e["payload"]
    data = {s: (p[key] if key == "macro" else p.get(key, {}).get(s)) for s in e["log_odds"]}
    anch = {a["instrument"]: a for a in p.get("anchor", [])}
    return key, data, anch


def rebuild(e, data, anch):
    key = spec(e["system"])[0]
    p = dict(e["payload"])
    if key != "macro":
        p[key] = {s: data[s] for s in data}
    p["anchor"] = [dict(anch[s], instrument=s) for s in anch]
    return p


def score(e, payload) -> dict[str, float]:
    qs = SC.questions(e["system"], payload)
    return {s: lo(e["vintage"], q) for s, q in qs.items() if s in e["log_odds"]}


def main() -> None:
    entries = load_entries()
    by_agent = defaultdict(list)
    for e in entries:
        by_agent[e["system"][:20]].append(e)
    res = {"n_calls": len(entries)}

    # C1: name-only prior
    agree, n = defaultdict(int), defaultdict(int)
    for e in sorted(entries, key=lambda e: e["vintage"]):
        _, label, horizon = spec(e["system"])
        q = {s: f"Asset: {s} ({DESCRIPTIONS[s]}).\nOver the next {horizon}, should an investor be long or short this asset?"
             for s in e["log_odds"]}
        prior = dirs({s: lo(e["vintage"], qq) for s, qq in q.items()})
        full = dirs(e["log_odds"])
        a = e["system"][:20]
        agree[a] += sum(prior[s] == full[s] for s in full)
        n[a] += len(full)
    res["C1"] = {"agreement_by_agent": {a: agree[a] / n[a] for a in n},
                 "agreement": sum(agree.values()) / sum(n.values())}
    res["C1"]["pass"] = res["C1"]["agreement"] < 0.80
    print("C1", res["C1"], flush=True)

    # C2: every fourth decision week
    rng_a, rng_b = random.Random(11), random.Random(7)
    rhos, changed, total = [], 0, 0
    for a, lst in by_agent.items():
        sub = lst[::4]
        for e in sorted(sub, key=lambda e: e["vintage"]):
            key, data, anch = evidence(e)
            syms = list(e["log_odds"])
            perm = syms[:]
            while True:
                rng_a.shuffle(perm)
                if all(x != y for x, y in zip(syms, perm)) or len(syms) < 2:
                    break
            m = dict(zip(syms, perm))
            pa = rebuild(e, {s: data[m[s]] for s in syms},
                         {s: anch[m[s]] for s in syms if m[s] in anch})
            sa = score(e, pa)
            r = spearmanr([e["log_odds"][s] for s in syms], [sa[s] for s in syms]).correlation
            rhos.append(0.0 if r != r else float(r))
            other = rng_b.choice([x for x in lst if x is not e])
            _, d2, a2 = evidence(other)
            pb = rebuild(e, {s: d2.get(s) or data[s] for s in syms}, {s: a2.get(s, anch.get(s)) for s in syms
                                                                         if a2.get(s, anch.get(s))})
            sb = dirs(score(e, pb))
            full = dirs(e["log_odds"])
            changed += sum(sb[s] != full[s] for s in syms)
            total += len(syms)
        print("C2", a, len(sub), flush=True)
    res["C2"] = {"n_calls": len(rhos), "median_rank_corr_shuffle": float(median(rhos)),
                 "share_directions_changed_other_week": changed / total}
    res["C2"]["pass_a"] = res["C2"]["median_rank_corr_shuffle"] < 0.5
    res["C2"]["pass_b"] = res["C2"]["share_directions_changed_other_week"] >= 0.20
    print("C2", res["C2"], flush=True)

    vintages = sorted({e["vintage"] for e in entries})
    # C3: controlled directional pairs
    pairs = json.load(open(REPO_ROOT / "docs/datedgpt_controlled_pairs.json"))
    res["C3"] = {}
    for v in vintages:
        ok = 0
        for p in pairs:
            q = lambda ev: (f"Asset: {p['asset']} ({p['desc']}).\n{p['label']}: {ev}."
                            f"\nOver the next {p['horizon']}, should an investor be long or short this asset?")
            ok += lo(v, q(p["bullish"])) > lo(v, q(p["bearish"]))
        res["C3"][str(v)] = {"correct": ok, "n": len(pairs), "share": ok / len(pairs), "pass": ok / len(pairs) >= 0.75}
    print("C3", res["C3"], flush=True)

    # C4: transmission of upstream views through the cross-asset prompt
    sysx = next(json.loads(json.load(open(f))["canonical_prompt"])["system"]
                for f in glob.glob(str(REPO_ROOT / "data/cache/llm/datedgpt-instruct-scorer-v1/*.json"))
                if json.loads(json.load(open(f))["canonical_prompt"])["system"].startswith(CROSS))
    ok = tot = 0
    for v in vintages:
        for s in DESCRIPTIONS:
            def pay(d):
                return {"as_of": f"{v + 1}-06-30", "anchor": [{"instrument": s, "direction": 0, "conviction": 0.0}],
                        "upstream_views": {s: {"macro_dir": float(d), "macro_conv": 0.8, "narr_dir": float(d), "narr_conv": 0.8}}}
            ql = SC.questions(sysx, pay(1))[s]
            qs = SC.questions(sysx, pay(-1))[s]
            ok += lo(v, ql) > lo(v, qs)
            tot += 1
    res["C4"] = {"correct": ok, "n": tot, "share": ok / tot, "pass": ok / tot >= 0.75}
    print("C4", res["C4"], flush=True)
    (TABLES_DIR / "datedgpt_checks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
