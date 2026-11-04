"""Where does coordination's effect disappear? (pre-registration v2, section 3)

Replays each matched protocol offline (cached LLM responses only) with the backtest engine
instrumented, and compares it with the ensemble of the same run at each stage:
  S1 final-view direction, S2 final-view conviction, S3 vol-scaled capped target,
  S4 after drawdown and correlation overlays, S5 submitted order (after the turnover cap),
  S6 held weights, S7 daily returns.
Initial disagreement per instrument-week = 1 - |mean anchor direction| of the macro, narrative
and fundamentals specialists (deterministic anchors, decision-time information), split at
the pooled median. Writes results/tables/decision_flow_<window>.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import backtest.engine as E
from backtest.engine import BacktestConfig, run_backtest
from data.universe import TRADEABLE_SYMBOLS
from portfolio.construction import PortfolioConfig
from scripts._common import TABLES_DIR, build_agents, build_llm_client, build_protocol, load_config

MATCHED = ("sequential_pipeline", "pipeline_placebo", "hierarchical")
SPECIALISTS = ("macro_regime", "narrative_event", "fundamentals_carry")
GROUPS = {
    "sonnet": [f"frontier_r{k}" for k in range(2, 7)],
    "haiku": [f"frontier_haiku_r{k}" for k in range(2, 12)],
    "clean": ["clean"],
}


def window(cfg, w):
    b = cfg[f"window_{w}"]
    return (datetime.strptime(b["start"], "%Y-%m-%d").date(), datetime.strptime(b["end"], "%Y-%m-%d").date())


def capture(regime: str, protocol_key: str, w: str) -> dict:
    """Run one cell offline and record every stage at every decision date."""
    cfg = load_config("ablation_matrix.yaml")
    start, end = window(cfg, w)
    llm = build_llm_client(regime, offline=True) if regime != "none" else None
    proto = build_protocol(protocol_key, build_agents(llm, TRADEABLE_SYMBOLS, cell_window=(start, end),
                                                      enable_altdata=True), llm, TRADEABLE_SYMBOLS)
    rec = {"views": {}, "s3": {}, "s4": {}, "s5": {}, "agents": {}}
    cur = {}
    run0 = proto.run

    def run(as_of):
        r = run0(as_of)
        cur["d"] = as_of
        rec["views"][as_of] = {v.instrument: (v.direction, v.conviction) for v in r.final_views}
        rec["agents"][as_of] = {a.agent_name: {v.instrument: v.direction for v in a.views}
                                for a in r.per_agent_decisions}
        return r
    proto.run = run
    v2t, brk, thr, Sim = E.views_to_target_weights, E.apply_drawdown_breaker, E.apply_correlation_throttle, E.PortfolioSimulator

    def v2t_(*a, **k):
        out = v2t(*a, **k)
        rec["s3"][cur["d"]] = dict(out)
        return out

    def thr_(tgt, *a, **k):
        out = thr(tgt, *a, **k)
        rec["s4"][cur["d"]] = dict(out)
        return out

    class Sim_(Sim):
        def step(self, ts, asset_ret, rf, target):
            if target is not None:
                rec["s5"][ts.date()] = dict(target)      # order executed at this session's close
            return super().step(ts, asset_ret, rf, target)
    E.views_to_target_weights, E.apply_correlation_throttle, E.PortfolioSimulator = v2t_, thr_, Sim_
    try:
        bt = BacktestConfig(start=start, end=end, rebalance_freq=cfg["backtest"]["rebalance_freq"],
                            portfolio=PortfolioConfig())
        res = run_backtest(proto, TRADEABLE_SYMBOLS, bt)
    finally:
        E.views_to_target_weights, E.apply_correlation_throttle, E.PortfolioSimulator = v2t, thr, Sim
    fails = sum(v for k, v in res.metadata.get("llm_stats", {}).items() if k.endswith(":call_failed"))
    if fails:
        raise RuntimeError(f"{regime}/{protocol_key}: {fails} LLM calls missing from the cache")
    rec["held"] = res.weights
    rec["ret"] = res.excess_returns
    return rec


def frame(d: dict, kind: str) -> pd.DataFrame:
    if kind in ("dir", "conv"):
        j = 0 if kind == "dir" else 1
        return pd.DataFrame({t: {s: v[j] for s, v in row.items()} for t, row in d.items()}).T.sort_index()
    return pd.DataFrame(d).T.sort_index()


def compare(e: dict, p: dict, high: pd.DataFrame) -> dict:
    out = {}
    pairs = {"S1_direction": (frame(e["views"], "dir"), frame(p["views"], "dir")),
             "S2_conviction": (frame(e["views"], "conv"), frame(p["views"], "conv")),
             "S3_target": (frame(e["s3"], "w"), frame(p["s3"], "w")),
             "S4_overlays": (frame(e["s4"], "w"), frame(p["s4"], "w")),
             "S5_order": (frame(e["s5"], "w"), frame(p["s5"], "w"))}
    for k, (a, b) in pairs.items():
        a, b = a.align(b, join="inner")
        diff = (b - a).abs()
        out[k] = {"share_changed": float((diff > 1e-6).to_numpy().mean()), "mean_abs_diff": float(diff.to_numpy().mean())}
        if k in ("S3_target", "S4_overlays", "S5_order"):
            out[k]["sign_changed"] = float((np.sign(a.round(6)) != np.sign(b.round(6))).to_numpy().mean())
    ha, hb = e["held"].align(p["held"], join="inner")
    out["S6_held"] = {"share_changed": float(((hb - ha).abs() > 1e-6).to_numpy().mean()),
                      "mean_abs_diff": float((hb - ha).abs().to_numpy().mean()),
                      "sign_changed": float((np.sign(ha.round(6)) != np.sign(hb.round(6))).to_numpy().mean())}
    ra, rb = e["ret"].align(p["ret"], join="inner")
    out["S7_returns"] = {"corr": float(ra.corr(rb)), "ann_mean_diff": float((rb - ra).mean() * 252)}
    # disagreement split on submitted orders (S5): order dates are one session after decisions
    a, b = pairs["S5_order"]
    a, b = a.align(b, join="inner")
    dec = sorted(high.index)
    lab = pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
    for t in a.index:
        prev = [d for d in dec if d < t]
        if prev:
            lab.loc[t] = high.loc[prev[-1]].reindex(a.columns).to_numpy(dtype=float)
    diff = (b - a).abs()
    for name, mask in (("high", lab == 1.0), ("low", lab == 0.0)):
        m = mask.to_numpy()
        out[f"disagreement_{name}"] = {"n": int(m.sum()), "mean_abs_order_diff": float(diff.to_numpy()[m].mean()),
                                       "share_changed": float((diff.to_numpy()[m] > 1e-6).mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="b")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--protocols", nargs="+", default=list(MATCHED))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    # decision-time disagreement from the deterministic anchors
    base = capture("none", "independent_ensemble", a.window)
    dis = {}
    for t, ag in base["agents"].items():
        dirs = pd.DataFrame({s: ag[s] for s in SPECIALISTS if s in ag})
        dis[t] = 1 - dirs.mean(axis=1).abs()
    dis = pd.DataFrame(dis).T.sort_index()
    med = float(np.nanmedian(dis.to_numpy()))
    high = (dis > med).astype(float)
    res = {"disagreement_median": med, "share_high": float(high.to_numpy().mean()), "groups": {}}
    for g in a.groups:
        regs = GROUPS[g] if a.window == "b" else (["clean"] if g == "clean" else [])
        per = {p: [] for p in a.protocols}
        for r in regs:
            e = capture(r, "independent_ensemble", a.window)
            for p in a.protocols:
                per[p].append(compare(e, capture(r, p, a.window), high))
            print(g, r, "done", flush=True)
        res["groups"][g] = {p: {"runs": v, "mean": _mean(v)} for p, v in per.items() if v}
    (TABLES_DIR / f"decision_flow_{a.window}{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({g: {p: x["mean"] for p, x in d.items()} for g, d in res["groups"].items()}, indent=1, default=float)[:6000])


def _mean(runs: list[dict]) -> dict:
    out = {}
    for k in runs[0]:
        out[k] = {m: float(np.mean([r[k][m] for r in runs])) for m in runs[0][k]}
    return out


if __name__ == "__main__":
    main()
