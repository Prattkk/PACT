"""Why do protocols differ? Disagreement, per-agent accuracy and statistical power.

Writes results/tables/mechanisms.json with three parts:

1. `pairs`: for each Claude protocol vs the Claude ensemble — daily excess-return
   correlation, mean active share, share of instrument-days with a different
   position sign, and the annualised mean / Sharpe of the difference return
   (r_protocol - r_ensemble), i.e. what the disagreement earned.
2. `agents`: per specialist, the directional hit rate and rank IC of its weekly
   views against the next holding-period return, for the rule-based anchor and
   for the Claude-refined view, plus how often Claude changed the anchor's
   direction and the hit rate of those changed calls. Views are replayed from
   the LLM cache (offline), so no API calls are made.
3. `power`: minimum detectable annualised Sharpe difference (80% power, 5%
   two-sided) as a function of return correlation and window length, and the
   observed correlations of the H1 pairs.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from agents._llm_helpers import LLM_STATS
from backtest.engine import WARMUP_CALENDAR_DAYS, decision_dates
from data.fetchers.prices import fetch_panel
from data.universe import TRADEABLE_SYMBOLS
from scripts._common import RAW_DIR, TABLES_DIR, build_agents, build_llm_client, load_config

SPECIALISTS = ("macro_regime", "narrative_event", "cross_asset_transmission", "technical_trend",
               "fundamentals_carry")
PROTOCOLS = ("single_agent", "single_agent_sc", "sequential_pipeline", "pipeline_placebo", "hierarchical",
             "debate", "debate_multiround", "llm_plus_anchor")
PRICES = {"claude-sonnet-5-5": (2e-6, 10e-6), "claude-haiku-5-5": (0.1e-6, 0.5e-6)}
Z = 1.959964 + 0.841621  # 5% two-sided, 80% power


def mde(rho: float, years: float) -> float:
    """Minimum detectable annualised Sharpe difference for two strategies with
    daily return correlation rho (large-sample approximation for small daily
    Sharpe ratios: Var(SR_a - SR_b) ~ 2(1 - rho)/T per day, annualised)."""
    return Z * np.sqrt(2.0 * (1.0 - rho) / years)


def years_needed(rho: float, delta: float) -> float:
    return 2.0 * (1.0 - rho) * (Z / delta) ** 2


def load(cell: str):
    d = RAW_DIR / cell
    meta = json.loads((d / "metadata.json").read_text())
    fails = sum(v for k, v in meta.get("llm_stats", {}).items() if k.endswith((":call_failed", ":unparseable")))
    if fails:
        raise FileNotFoundError(f"{cell}: {fails} LLM calls failed or missing; cell excluded")
    ex = pd.read_parquet(d / "excess_returns.parquet").iloc[:, 0]
    w = pd.read_parquet(d / "weights.parquet")
    return ex, w


def pair_stats(regime: str) -> dict:
    out = {}
    e_ret, e_w = load(f"b__{regime}__independent_ensemble")
    for p in PROTOCOLS:
        try:
            r, w = load(f"b__{regime}__{p}")
        except FileNotFoundError:
            continue
        idx = r.index.intersection(e_ret.index)
        diff = (r - e_ret).loc[idx]
        we, wp = e_w.loc[idx], w.loc[idx].reindex(columns=e_w.columns, fill_value=0.0)
        sign_diff = (np.sign(wp.round(6)) != np.sign(we.round(6))).to_numpy().mean()
        out[p] = {
            "rho": float(r.loc[idx].corr(e_ret.loc[idx])),
            "active_share": float((0.5 * (wp - we).abs().sum(axis=1)).mean()),
            "sign_disagreement": float(sign_diff),
            "diff_ann_mean": float(diff.mean() * 252),
            "diff_sharpe": float(diff.mean() / diff.std() * np.sqrt(252)) if diff.std() > 0 else 0.0,
            "mde_3y": float(mde(float(r.loc[idx].corr(e_ret.loc[idx])), len(idx) / 252)),
        }
    return out


def agent_accuracy(regime: str) -> dict:
    cfg = load_config("ablation_matrix.yaml")
    blk = cfg["window_b"]
    start = pd.Timestamp(blk["start"]).date()
    end = pd.Timestamp(blk["end"]).date()
    universe = TRADEABLE_SYMBOLS
    prices = fetch_panel(list(universe), start - timedelta(days=WARMUP_CALENDAR_DAYS), end + timedelta(days=40),
                         field="Adj Close")
    dates = decision_dates(prices.index, start, end, cfg["backtest"]["rebalance_freq"])
    sess = prices.index
    # holding period: from the close after the decision (execution) to the close
    # after the next decision, matching the engine's one-session execution lag
    fills = [sess[sess.get_loc(d) + 1] for d in dates if sess.get_loc(d) + 1 < len(sess)]
    fwd = {}
    for i, d in enumerate(dates[: len(fills) - 1]):
        fwd[d.date()] = prices.loc[fills[i + 1]] / prices.loc[fills[i]] - 1.0

    llm = build_llm_client(regime, offline=True)
    with_llm = build_agents(llm, universe, cell_window=(start, end), enable_altdata=True)
    anchors = build_agents(None, universe, cell_window=(start, end), enable_altdata=True)
    rows = []
    for d, ret in fwd.items():
        views = {}
        for kind, ags in (("anchor", anchors), ("llm", with_llm)):
            for name in SPECIALISTS:
                a = ags[name]
                if name == "cross_asset_transmission":
                    a.set_upstream(ags["macro_regime"].run(d), ags["narrative_event"].run(d))
                try:
                    dec = a.run(d)
                except KeyError:   # prompt not in the cache for this agent/date
                    continue
                for v in dec.views:
                    if v.instrument in ret.index and pd.notna(ret[v.instrument]):
                        views[(kind, name, v.instrument)] = (v.direction, v.conviction)
        for (kind, name, sym), (dr, cv) in views.items():
            rows.append({"date": d, "kind": kind, "agent": name, "sym": sym, "dir": dr, "conv": cv,
                         "ret": float(ret[sym])})
    misses = {k: v for k, v in LLM_STATS.items() if k.endswith((":call_failed", ":unparseable"))}
    if misses:
        # agents fall back to their anchor on a missing reply, which would be
        # mislabelled as an LLM view here; refuse rather than mix them
        raise RuntimeError(f"replay incomplete, LLM views missing from cache: {misses}")
    df = pd.DataFrame(rows)
    out = {}
    for name in SPECIALISTS:
        res = {}
        for kind in ("anchor", "llm"):
            s = df[(df.agent == name) & (df.kind == kind)]
            nz = s[s.dir != 0]
            res[kind] = {
                "n_calls": int(len(nz)),
                "coverage": float(len(nz) / max(len(s), 1)),
                "hit_rate": float((np.sign(nz.dir) == np.sign(nz.ret)).mean()) if len(nz) else float("nan"),
                "ic": float(stats.spearmanr(s.dir * s.conv, s.ret).correlation) if s.ret.std() > 0 and (s.dir * s.conv).std() > 0 else float("nan"),
            }
        a = df[(df.agent == name) & (df.kind == "anchor")].set_index(["date", "sym"])
        l = df[(df.agent == name) & (df.kind == "llm")].set_index(["date", "sym"])
        j = a.join(l, lsuffix="_a", rsuffix="_l", how="inner")
        changed = j[j.dir_a != j.dir_l]
        ch_nz = changed[changed.dir_l != 0]
        res["changed_share"] = float(len(changed) / max(len(j), 1))
        res["changed_hit_rate"] = float((np.sign(ch_nz.dir_l) == np.sign(ch_nz.ret_l)).mean()) if len(ch_nz) else float("nan")
        res["changed_n"] = int(len(ch_nz))
        out[name] = res
    return out


def compute_usage(regime: str) -> dict:
    """Calls, tokens, API cost and summed latency a standalone run of each
    protocol makes, replayed offline from the cache (shared specialist calls
    are counted for every protocol that makes them)."""
    cfg = load_config("ablation_matrix.yaml")
    start = pd.Timestamp(cfg["window_b"]["start"]).date()
    end = pd.Timestamp(cfg["window_b"]["end"]).date()
    prices = fetch_panel(list(TRADEABLE_SYMBOLS), start - timedelta(days=WARMUP_CALENDAR_DAYS), end + timedelta(days=1),
                         field="Adj Close")
    dates = [d.date() for d in decision_dates(prices.index, start, end, cfg["backtest"]["rebalance_freq"])]
    base = build_llm_client(regime, offline=True)

    class Counting:
        # counters live in a dict so shallow copies (sample-tagged clients) share them
        def __init__(self, inner):
            self.inner, self.model = inner, inner.model
            self.c = {"n": 0, "i": 0, "o": 0, "lat": 0.0, "miss": 0}
        def complete(self, **kw):
            try:
                r = self.inner.complete(**kw)
            except Exception:
                self.c["miss"] += 1
                raise
            u = (r.raw or {}).get("usage") or {}
            self.c["n"] += 1
            self.c["i"] += u.get("input_tokens", 0)
            self.c["o"] += u.get("output_tokens", 0)
            self.c["lat"] += (r.raw or {}).get("latency_s", 0.0) or 0.0
            return r

    out = {}
    for p in ("independent_ensemble",) + PROTOCOLS:
        llm = Counting(base)
        from scripts._common import build_protocol
        proto = build_protocol(p, build_agents(llm, TRADEABLE_SYMBOLS, cell_window=(start, end), enable_altdata=True),
                               llm, TRADEABLE_SYMBOLS)
        try:
            for d in dates:
                proto.run(d)
        except Exception as e:      # protocol not run in this regime
            out[p] = {"error": type(e).__name__}
            continue
        pi, po = PRICES.get(getattr(base, "model", ""), (0.0, 0.0))
        c = llm.c
        if c["miss"]:               # protocol not (fully) run in this regime
            out[p] = {"error": f"{c['miss']} calls missing from cache"}
            continue
        out[p] = {"calls": c["n"], "input_tokens": c["i"], "output_tokens": c["o"],
                  "cost_usd": c["i"] * pi + c["o"] * po, "latency_hours": c["lat"] / 3600}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="frontier")
    p.add_argument("--skip-agents", action="store_true")
    a = p.parse_args()
    res = {"pairs": pair_stats(a.regime)}
    res["power"] = {
        "grid": [{"rho": r, **{f"mde_{y}y": float(mde(r, y)) for y in (3, 10, 25)}, "mde_3m": float(mde(r, 0.25)),
                  "years_for_0.25": float(years_needed(r, 0.25))} for r in (0.0, 0.5, 0.8, 0.9, 0.95)],
    }
    if not a.skip_agents:
        res["agents"] = agent_accuracy(a.regime)
        res["compute"] = compute_usage(a.regime)
    out = TABLES_DIR / f"mechanisms_{a.regime}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
