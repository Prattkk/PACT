"""Numbers, tables and figure for the revised paper (pre-registration v2 analyses).

Reads results/tables/{robustness,bootstrap_validation,factorial_b,datedgpt_checks,decision_flow_b,
decision_flow_b_debate,disagreement_b,disagreement_c,unused_c,mechanisms_frontier_r2,
mechanisms_frontier_haiku_r2}.json and writes into --out (default results/paper_tex):
numbers_v2.tex (\\pactdef entries), tab_main2.tex, tab_unused.tex, tab_secondary2.tex, fig_flow.pdf.
Missing inputs leave their entries undefined, which the paper shows as pending.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._common import RAW_DIR, TABLES_DIR

CLS = {"practically equivalent": "E", "meaningful gain excluded": "X", "improvement": "I",
       "inconclusive": "?", "harm": "H"}
PENDING = r"\pending"


def J(name):
    p = TABLES_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def num(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return PENDING
    return "$" + (f"{x:+.{d}f}" if sign else f"{x:.{d}f}") + "$"


def pct(x, d=0):
    return PENDING if x is None else f"{100 * x:.{d}f}\\%"


def key(s):
    return s.replace("_", "").replace(" ", "")


def mark(cls, sim_lo=None):
    """Class superscript. Improvement is I_u (unadjusted) unless the simultaneous interval also
    excludes zero; the paper never reports an I that survives multiplicity correction as plain I
    without that check."""
    c = CLS.get(cls, cls)
    if c == "I" and not (sim_lo is not None and sim_lo > 0):
        return r"$^{\mathrm{I_u}}$"
    return r"$^{\mathrm{" + c + "}}$"


def cell(diff, lo, hi, cls=None, npos=None, runs=None, sim_lo=None):
    if diff is None:
        return "--"
    head = num(diff)
    if runs and runs > 1:
        head += r" {\tiny(" + f"{round(npos * runs) if npos <= 1 else npos}/{runs}" + ")}"
    if cls:
        head += mark(cls, sim_lo)
    return r"\makecell{" + head + r"\\ {\tiny$[" + f"{lo:+.2f},\\,{hi:+.2f}".replace("+", "") + "]$}}"


def sr(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper_tex")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    N: dict[str, str] = {}

    # ---- main table: matched comparisons and adjudication contrasts --------------------------
    rob, fac = J("robustness"), J("factorial_b")
    cols = ("sonnet", "haiku", "clean", "haikuanon")
    rows = [("sequential_pipeline", "independent_ensemble", "Pipeline vs ensemble"),
            ("pipeline_placebo", "independent_ensemble", "Placebo vs ensemble (added voter)"),
            ("sequential_pipeline", "pipeline_placebo", "Pipeline vs placebo (communication)"),
            ("hierarchical", "independent_ensemble", "Conviction gate vs ensemble")]
    lines, counts = [], {}
    if rob:
        b = rob["b"]
        for x, y, lab in rows:
            cells = []
            for g in cols:
                r = b.get(f"{g}:{x}:{y}")
                if not r:
                    cells.append("--")
                    continue
                cells.append(cell(r["b21"]["diff"], r["b21"]["ci_lo"], r["b21"]["ci_hi"], r["class"],
                                  sim_lo=r["ci_simultaneous"][0]))
                k = f"rob.{g}.{key(x)}.{key(y)}"
                N[k + ".cls"] = r["class"]
                N[k + ".lo90"], N[k + ".hi90"] = num(r["ci90"][0]), num(r["ci90"][1])
                N[k + ".simlo"], N[k + ".simhi"] = num(r["ci_simultaneous"][0]), num(r["ci_simultaneous"][1])
                N[k + ".stable"] = "yes" if len(set(r["class_by_block"].values())) == 1 else "no"
                counts[r["class"]] = counts.get(r["class"], 0) + 1
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
        for c, n in counts.items():
            N[f"rob.count.{key(c)}"] = str(n)
        N["rob.ntests"] = str(sum(counts.values()))
        N["rob.nboot"] = f"{rob['n_boot']:,}"
        stable = sum(1 for k, r in b.items() if len(set(r["class_by_block"].values())) == 1)
        N["rob.nstable"], N["rob.nall"] = str(stable), str(len(b))
        for kk, r in b.items():
            g, x, y = kk.split(":")
            k = f"rob.{g}.{key(x)}.{key(y)}"
            N[k + ".cls"] = r["class"]
            N[k + ".lo90"], N[k + ".hi90"] = num(r["ci90"][0]), num(r["ci90"][1])
            N[k + ".simlo"], N[k + ".simhi"] = num(r["ci_simultaneous"][0]), num(r["ci_simultaneous"][1])
    if fac:
        lines.append(r"\midrule")
        name = "B-A (judge, no communication)"
        cells = []
        for g in cols:
            r = fac["groups"].get(g, {}).get(name)
            rr = rob["b"].get(f"{g}:debate:independent_ensemble") if rob else None
            cells.append(cell(r["diff"], *r["ci"], r["class"], r["share_positive"], r["n_runs"],
                              sim_lo=rr["ci_simultaneous"][0] if rr else None) if r else "--")
        lines.append("Debate judge vs ensemble" + " & " + " & ".join(cells) + r" \\")
        for g in cols:   # keys still used in the text
            for nm in ("B-A (judge, no communication)", "D-B (communication, judge)"):
                r = fac["groups"].get(g, {}).get(nm)
                if r:
                    k = f"fac.{g}.{key(nm.split(' ')[0])}"
                    N[k + ".diff"], N[k + ".lo"], N[k + ".hi"] = num(r["diff"]), num(r["ci"][0]), num(r["ci"][1])
                    N[k + ".cls"], N[k + ".runs"] = r["class"], str(r["n_runs"])
    if lines:
        (out / "tab_main2.tex").write_text("\n".join(lines) + "\n")

    # ---- bootstrap validation -----------------------------------------------------------------
    bv = J("bootstrap_validation")
    if bv:
        N["bv.reps"], N["bv.T"], N["bv.rho"] = str(bv["reps"]), str(bv["T"]), f"{bv['rho']:.2f}"
        N["bv.cov0"], N["bv.rej0"] = pct(bv["delta_0.00"]["coverage_95"], 1), pct(bv["delta_0.00"]["reject_5pct"], 1)
        N["bv.cov2"], N["bv.pow2"] = pct(bv["delta_0.20"]["coverage_95"], 1), pct(bv["delta_0.20"]["reject_5pct"], 1)
        N["bv.nboot"] = f"{bv['n_boot']:,}"
        N["bv.est2"] = f"{bv['delta_0.20']['mean_estimate']:.3f}"
        N["bv.width"] = f"{bv['delta_0.00']['mean_ci_width']:.2f}"

    # ---- DatedGPT capability checks -----------------------------------------------------------
    ch = J("datedgpt_checks")
    if ch:
        N["chk.c1"] = pct(ch["C1"]["agreement"])
        N["chk.c1min"] = pct(min(ch["C1"]["agreement_by_agent"].values()))
        N["chk.c1max"] = pct(max(ch["C1"]["agreement_by_agent"].values()))
        N["chk.c2a"] = f"{ch['C2']['median_rank_corr_shuffle']:.2f}"
        N["chk.c2b"] = pct(ch["C2"]["share_directions_changed_other_week"])
        for y, v in ch["C3"].items():
            N[f"chk.c3.{y}"] = pct(v["share"])
        N["chk.c4"] = pct(ch["C4"]["share"])
        N["chk.c4n"] = str(ch["C4"]["n"])
        # pre-registered checks: C1, C2 (passes only if both (a) and (b) pass), C3 (every vintage), C4
        N["chk.npass"] = str(sum([ch["C1"]["pass"], ch["C2"]["pass_a"] and ch["C2"]["pass_b"],
                                  all(v["pass"] for v in ch["C3"].values()), ch["C4"]["pass"]]))
        N["chk.n"] = "4"
        N["chk.c2weeks"] = str(ch["C2"]["n_calls"] // 3)
    for c, lab in (("clean_nameonly", "nameonly"), ("clean", "clean"), ("none", "none")):
        p = RAW_DIR / f"b__{c}__independent_ensemble" / "excess_returns.parquet"
        if p.exists():
            N[f"lvlb.{lab}.ens"] = num(sr(pd.read_parquet(p).iloc[:, 0]))

    # ---- decision flow ------------------------------------------------------------------------
    df = J("decision_flow_b")
    dfd = J("decision_flow_b_debate")
    flow = {}
    for src in (df, dfd):
        if not src:
            continue
        for g, v in src["groups"].items():
            for p, x in v.items():
                runs = x["runs"]
                m = lambda s, f: float(np.mean([r[s][f] for r in runs]))  # noqa: E731
                f = {"dir": m("S1_direction", "share_changed"), "conv": m("S2_conviction", "mean_abs_diff"),
                     "target": m("S3_target", "sign_changed"), "order": m("S5_order", "sign_changed"),
                     "held": m("S6_held", "sign_changed"), "corr": float(np.mean([r["S7_returns"]["corr"] for r in runs])),
                     "w3": m("S3_target", "mean_abs_diff"), "w5": m("S5_order", "mean_abs_diff"),
                     "w6": m("S6_held", "mean_abs_diff")}
                flow[(g, p)] = f
                k = f"flow.{g}.{key(p)}"
                for a in ("dir", "target", "order", "held"):
                    N[f"{k}.{a}"] = pct(f[a], 1)
                N[k + ".conv"], N[k + ".corr"] = f"{f['conv']:.2f}", f"{f['corr']:.3f}"
                N[k + ".absorbed"] = pct(1 - f["order"] / f["target"]) if f["target"] > 0 else PENDING
    if df:
        N["flow.median"], N["flow.sharehigh"] = f"{df['disagreement_median']:.2f}", pct(df["share_high"])
    if flow:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5})
        fig, ax = plt.subplots(1, 2, figsize=(4.8, 1.75), constrained_layout=True)
        stages = ["dir", "target", "order", "held"]
        xl = ["vote\ndirection", "target\nweight", "order\n(turnover cap)", "held\nposition"]
        sty = {"sequential_pipeline": ("C0", "o", "pipeline"), "pipeline_placebo": ("C1", "s", "placebo"),
               "hierarchical": ("C2", "^", "conv. gate"), "debate": ("C3", "D", "debate")}
        ls = {"sonnet": "-", "haiku": "--", "clean": ":"}
        for (g, p), f in flow.items():
            if p not in sty:
                continue
            c, mk, lab = sty[p]
            ax[0].plot(range(4), [100 * f[s] for s in stages], ls[g], color=c, marker=mk, ms=2.5, lw=0.8,
                       label=lab if g == "sonnet" else None)
            ax[1].plot(range(3), [100 * f[s] for s in ("w3", "w5", "w6")], ls[g], color=c, marker=mk, ms=2.5, lw=0.8)
        ax[0].set_xticks(range(4), xl, fontsize=5.5)
        ax[0].set_ylabel("sign differs from\nensemble (%)", fontsize=6)
        ax[1].set_xticks(range(3), ["target", "order", "held"], fontsize=5.5)
        ax[1].set_ylabel(r"mean $|\Delta w|$ (% NAV)", fontsize=6)
        h = ax[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D

        h0 = h[0] + [Line2D([], [], color="k", ls=ls[g], lw=0.8) for g in ls]
        h1 = h[1] + ["Sonnet", "Haiku", "DatedGPT"]
        fig.legend(h0, h1, loc="outside right center", fontsize=5.5, frameon=False)
        for a in ax:
            a.tick_params(width=0.4, length=2, labelsize=5.5)
            a.set_ylim(bottom=0)
            a.grid(axis="y", lw=0.3, alpha=0.5)
        fig.savefig(out / "fig_flow.pdf")

    # ---- disagreement -------------------------------------------------------------------------
    for w in ("b", "c"):
        d = J(f"disagreement_{w}")
        if not d:
            continue
        N[f"dis.{w}.sharehigh"] = pct(d["share_high_instrument_weeks"])
        for g, v in d["groups"].items():
            for p, x in v.items():
                k = f"dis.{w}.{g}.{key(p)}"
                N[k + ".ratio"] = f"{x['high']['mean_abs_weight_diff'] / x['low']['mean_abs_weight_diff']:.2f}"
                hi, lo = x["high"]["ann_contribution"], x["low"]["ann_contribution"]
                N[k + ".hi"], N[k + ".lo"] = num(100 * hi, 1), num(100 * lo, 1)
                # per instrument-day rate in high relative to low (both annualised contributions per day)
                rh = hi / x["high"]["instrument_days"]
                rl = lo / x["low"]["instrument_days"]
                N[k + ".perday"] = num(1e6 * rh, 1) + "/" + num(1e6 * rl, 1)
                N[k + ".absratio"] = f"{x['high']['mean_abs_daily_contribution_bp'] / x['low']['mean_abs_daily_contribution_bp']:.2f}"

    # ---- unused period ------------------------------------------------------------------------
    un = J("unused_c")
    if un:
        comps = [("sequential_pipeline:independent_ensemble", "Pipeline $-$ ens."),
                 ("pipeline_placebo:independent_ensemble", "Placebo $-$ ens."),
                 ("sequential_pipeline:pipeline_placebo", "Pipeline $-$ placebo"),
                 ("hierarchical:independent_ensemble", "Conv.\\ gate $-$ ens."),
                 ("pipeline_placebo_s1:independent_ensemble", "Rand.\\ placebo 1 $-$ ens."),
                 ("pipeline_placebo_s2:independent_ensemble", "Rand.\\ placebo 2 $-$ ens.")]
        pers = ("2016-2021", "2016-2018", "2019-2021")
        years = [str(y) for y in range(2016, 2022)]
        regs = [r for r in ("none", "clean") if r in un["regimes"]]
        L = []
        lev = ["Ensemble SR"]
        for r in regs:
            for per in pers:
                lev.append(num(un["regimes"][r]["sharpe"][per].get("independent_ensemble")))
        L.append(" & ".join(lev) + r" \\ \midrule")
        for c, lab in comps:
            row = [lab]
            for r in regs:
                for per in pers:
                    x = un["regimes"][r]["comparisons"][per].get(c)
                    if not x:
                        row.append("--")
                    elif per == "2016-2021":
                        row.append(cell(x["diff"], x["ci_lo"], x["ci_hi"], x["class"]))
                    else:
                        row.append(num(x["diff"]) + r"$^{\mathrm{" + CLS[x["class"]] + "}}$")
            npos = []
            for r in regs:
                ys = [un["regimes"][r]["comparisons"][y].get(c) for y in years]
                ys = [y for y in ys if y]
                npos.append(f"{sum(y['diff'] > 0 for y in ys)}/{len(ys)}" if ys else "--")
            L.append(" & ".join(row) + " & " + " & ".join(npos) + r" \\")
        (out / "tab_unused.tex").write_text("\n".join(L) + "\n")
        for r in regs:
            for c, _ in comps:
                x = un["regimes"][r]["comparisons"]["2016-2021"].get(c)
                if x:
                    k = f"un.{r}.{key(c.replace(':', '.'))}"
                    N[k + ".diff"], N[k + ".lo"], N[k + ".hi"], N[k + ".cls"] = (
                        num(x["diff"]), num(x["ci_lo"]), num(x["ci_hi"]), x["class"])
            for p, v in un["regimes"][r]["sharpe"]["2016-2021"].items():
                N[f"un.{r}.lvl.{key(p)}"] = num(v)

    # ---- secondary protocols: Sharpe, gain over ensemble and cost excluding rationale ----------
    sec = [("independent_ensemble", "A: ensemble (vote)"), ("sequential_pipeline", "C: pipeline (vote)"),
           ("hierarchical", "Conviction gate"), ("debate", "B: debate, 1 round (judge)"),
           ("debate_multiround", "D: debate, 2 rounds"), ("llm_plus_anchor", "LLM + anchor"),
           ("single_agent", "Single-LLM monolith"), ("single_agent_sc", "Monolith, 5 samples")]
    ms = {g: J(f"mechanisms_{r}") for g, r in (("sonnet", "frontier_r2"), ("haiku", "frontier_haiku_r2"))}
    gen = J("results") or {}
    res_path = Path(out) / "results.json"
    if res_path.exists():
        gen = json.loads(res_path.read_text())
    nums = {}
    npath = Path(out) / "numbers.tex"
    if npath.exists():
        import re

        for k, v in re.findall(r"\\pactdef\{([^}]*)\}\{(.*)\}\n", npath.read_text()):
            nums[k] = v
    L = []
    for p, lab in sec:
        row = [lab]
        kp = key(p)
        for g in ("sonnet", "haiku"):
            row.append(nums.get(f"lvl.{g}.{kp}", "--"))
        for g in ("sonnet", "haiku"):
            d = nums.get(f"m.{g}.{kp}.independentensemble.diff")
            if d and p != "independent_ensemble":
                nr = nums.get(f"m.{g}.{kp}.independentensemble.runs", "1")
                row.append(d + r" {\tiny$[" + nums[f"m.{g}.{kp}.independentensemble.lo"].strip("$").replace("+", "")
                           + r",\," + nums[f"m.{g}.{kp}.independentensemble.hi"].strip("$").replace("+", "") + "]$}")
            else:
                row.append("--")
        for g in ("sonnet", "haiku"):
            u = (ms[g] or {}).get("compute", {}).get(p)
            row += [f"\\${u['cost_usd_excl_rationale']:.2f}"] if u and u.get(
                "calls_excl_rationale") is not None else ["--"]
            if u and u.get("calls_excl_rationale") is not None:
                base = ms[g]["compute"]["independent_ensemble"]["cost_usd_excl_rationale"]
                N[f"cost.{g}.{key(p)}.ratio"] = f"{u['cost_usd_excl_rationale'] / base:.1f}"
                N[f"cost.{g}.{key(p)}.usd"] = f"\\${u['cost_usd_excl_rationale']:.2f}"
                N[f"cost.{g}.{key(p)}.calls"] = f"{u['calls_excl_rationale']:,}"
        L.append(" & ".join(row) + r" \\")
    if any(ms.values()):
        (out / "tab_cost2.tex").write_text("\n".join(L) + "\n")
        u = ms["sonnet"]["compute"]["independent_ensemble"]
        N["cost.ratshare"] = pct(u["rationale_calls"] / u["calls"])

    # ---- placebo mappings, DatedGPT 2022-2024 (cyclic shift + five random derangements) ---------
    ps = J("placebo_seeds_b")
    if ps:
        pl = [v["diff"] for k, v in ps.items() if k.startswith("pipeline_placebo")]
        N["seeds.b.n"], N["seeds.b.min"], N["seeds.b.max"] = str(len(pl)), num(min(pl)), num(max(pl))
        N["seeds.b.median"] = num(float(np.median(pl)))
        pipe = ps["sequential_pipeline"]["diff"]
        N["seeds.b.pipe"] = num(pipe)
        N["seeds.b.below"] = str(sum(x < pipe for x in pl))

    # ---- debate judge: resizing vs sign changes ---------------------------------------------
    jd = J("judge_decomposition_b")
    if jd:
        for g, v in jd.items():
            m = v["mean"]
            N[f"judge.{g}.flipshare"] = pct(m["share_instrument_days_sign_differs"])
            N[f"judge.{g}.fromflip"] = pct(1 - m["share_of_total_from_same_sign"])
            N[f"judge.{g}.fromresize"] = pct(m["share_of_total_from_same_sign"])
            N[f"judge.{g}.allpos"] = "yes" if m["runs_same_sign_positive"] == m["runs_sign_differs_positive"] == m["n_runs"] else "no"

    # ---- responses in the reported runs (manifest) ---------------------------------------------
    man = Path(__file__).resolve().parent.parent / "docs" / "run_manifest.json"
    if man.exists():
        mk = json.loads(man.read_text())["cached_responses_by_model_key"]
        for fam in ("sonnet", "haiku"):
            N[f"calls.{fam}"] = f"{sum(n for k, n in mk.items() if k.startswith(f'claude-{fam}') and '|run=' in k):,}"

    # ---- controlled 2x2 (pre-registration v3) ----------------------------------------------------
    ct = J("controlled_clean")
    if ct:
        c21 = ct["contrasts"]["block21"]
        labels = {"D-B": r"D$-$B: communication, judge fixed", "B-A": r"B$-$A: judge vs vote, isolated",
                  "C-A": r"C$-$A: communication, vote fixed", "interaction": r"(D$-$B)$-$(C$-$A)"}
        rows = []
        for name, lab in labels.items():
            r = c21[name]
            stable = len({ct["contrasts"][f"block{b}"][name]["cls"] for b in (5, 21, 63)}) == 1
            sim = (r"$[" + f"{r['simlo']:+.2f},\\,{r['simhi']:+.2f}".replace("+", "") + "]$") if "simlo" in r else "--"
            rows.append(f"{lab} & {num(r['diff'])}{mark(r['cls'], r.get('simlo'))} & "
                        + r"$[" + f"{r['lo']:+.2f},\\,{r['hi']:+.2f}".replace("+", "") + "]$"
                        + f" & {sim} & {'yes' if stable else 'no'}" + r" \\")
            k = f"ctrl.{key(name).replace('-', '')}"
            N[k + ".diff"], N[k + ".lo"], N[k + ".hi"], N[k + ".cls"] = num(r["diff"]), num(r["lo"]), num(r["hi"]), r["cls"]
            N[k + ".lo90"], N[k + ".hi90"] = num(r["lo90"]), num(r["hi90"])
        (out / "tab_ctrl.tex").write_text("\n".join(rows) + "\n")
        for c, v in ct["contrasts"]["sharpe"].items():
            N[f"ctrl.sr.{c}"] = num(v)
        m = ct.get("mechanisms")
        if m:
            N["ctrl.revchg.iso"] = pct(m["revision_changes_direction_isolated"], 1)
            N["ctrl.revchg.shared"] = pct(m["revision_changes_direction_shared"], 1)
            N["ctrl.revdiff"] = pct(m["revised_direction_differs_shared_vs_isolated"], 1)
            for kk, v in m["final_direction_differs"].items():
                N[f"ctrl.finaldiff.{kk.replace('-', '')}"] = pct(v, 1)
            N["ctrl.identical"] = "yes" if (m["identical_revisions_A_B"] and m["identical_revisions_C_D"]
                                            and m["identical_first_pass"]) else "no"
            N["ctrl.replay"] = f"{max(m['replay_max_abs_diff'].values()):.1e}"

    # ---- bootstrap calibration (pre-registration v3) ---------------------------------------------
    cb = J("bootstrap_calibration")
    if cb:
        rows = []
        for R in (5, 10):
            if R == 10 and rows:
                rows.append(r"\midrule")
            for d in (0.0, 0.10, 0.20, -0.20):
                g = cb.get(f"R{R}_delta{d:+.2f}")
                if not g:
                    continue
                a, c = g["prereg_b21"], g["corrected_b21"]
                rows.append(f"{R} & ${d:+.2f}$ & {pct(a['coverage95'], 1)} & {pct(a['rate_I'], 1)} & {pct(a['rate_E'], 1)} & "
                            f"{pct(a['rate_X'], 1)} & {a['se_ratio']:.2f} & {pct(c['coverage95'], 1)} & {c['se_ratio']:.2f}" + r" \\")
                for est, e in (("pre", a), ("cor", c)):
                    kk = f"cal.R{R}.d{int(round(d * 100)):+d}.{est}"
                    N[kk + ".cov"], N[kk + ".I"] = pct(e["coverage95"], 1), pct(e["rate_I"], 1)
                    N[kk + ".E"], N[kk + ".X"], N[kk + ".ser"] = pct(e["rate_E"], 1), pct(e["rate_X"], 1), f"{e['se_ratio']:.2f}"
                for b in (5, 63):
                    N[f"cal.R{R}.d{int(round(d * 100)):+d}.pre.cov.b{b}"] = pct(g[f"prereg_b{b}"]["coverage95"], 1)
        (out / "tab_calib.tex").write_text("\n".join(rows) + "\n")
        for R in (5, 10):
            for est, v in cb.get(f"family4_R{R}", {}).items():
                N[f"cal.fam.R{R}.{est[:3]}.ind"] = pct(v["familywise_coverage_individual95"], 1)
                N[f"cal.fam.R{R}.{est[:3]}.bon"] = pct(v["familywise_coverage_bonferroni9875"], 1)
        N["cal.reps"], N["cal.nboot"] = f"{cb['design']['reps']:,}", f"{cb['design']['n_boot']:,}"
        for kk, e in cb.get("exploratory_single_run", {}).items():
            d = int(round(e["true"] * 100))
            for f, v in (("cov", e["coverage95"]), ("I", e["rate_I"]), ("E", e["rate_E"]), ("X", e["rate_X"])):
                N[f"cal.R1.d{d:+d}.{f}"] = pct(v, 1)

    (out / "numbers_v2.tex").write_text("".join(f"\\pactdef{{{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
    print(f"{len(N)} numbers; files in {out}")


if __name__ == "__main__":
    main()
