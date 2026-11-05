"""Calibration of the multi-run block bootstrap (pre-registration v3, section 3).

Extends scripts/validate_bootstrap.py (same data-generating process) to the decisions the paper
relies on: superiority (I), practical equivalence (E, 90% interval inside +/-0.20) and excluded
gains (X, 95% upper bound below 0.20), for 5- and 10-run designs, true differences -0.20, 0,
+0.10 and +0.20, and estimator block lengths 5, 21 and 63. Also a family of four contrasts
sharing one market path (individual vs Bonferroni intervals).

`multirun_draws` reproduces `evaluation.statistical_tests.multirun_sharpe_diff` draw for draw
(same random-number sequence; checked in `_selfcheck`) but computes the Sharpe ratios in one
vectorised step, and returns the per-run day-block differences needed by the diagnostic
moment-corrected variance (see the pre-registration). Writes results/tables/bootstrap_calibration.json.

  python -m scripts.calibrate_bootstrap [reps] [n_boot] [workers]
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm

from evaluation.statistical_tests import _circular_block_indices, multirun_sharpe_diff
from scripts._common import TABLES_DIR
from scripts.validate_bootstrap import ANN, MU_B, RHO, RUN_SD, T, VOL, garch_factor

DELTAS = (-0.20, 0.0, 0.10, 0.20)
BLOCKS = (5, 21, 63)
MARGIN = 0.20


def simulate(rng, delta: float, R: int, f=None, b=None):
    """R paired runs (protocol a_r, comparator b) with true Sharpe difference `delta`."""
    if f is None:
        f = garch_factor(rng, T)
        b = MU_B + f
    sd_eta = VOL * np.sqrt(1 / RHO ** 2 - 1)
    sd_a = np.sqrt(VOL ** 2 + sd_eta ** 2)
    A = np.empty((R, T))
    for r in range(R):
        mu_a = (MU_B / VOL + (delta + rng.normal(0.0, RUN_SD)) / ANN) * sd_a
        A[r] = mu_a + f + sd_eta * rng.standard_normal(T)
    return A, b, f


def _sr_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(-1) / x.std(-1, ddof=1) * ANN


def multirun_draws(A: np.ndarray, b: np.ndarray, block: int, n_boot: int, seed: int):
    """Pre-registered draws (identical to multirun_sharpe_diff) and the per-run day-block matrix D."""
    rng = np.random.default_rng(seed)
    R, n = A.shape
    runs = np.empty((n_boot, R), dtype=int)
    idx = np.empty((n_boot, n), dtype=int)
    for k in range(n_boot):                         # same call order as the production function
        runs[k] = rng.integers(0, R, R)
        idx[k] = _circular_block_indices(n, block, rng)
    sb = _sr_rows(b[idx])                                           # (K,)
    D = np.stack([_sr_rows(A[r][idx]) - sb for r in range(R)], 1)   # (K, R)
    draws = np.take_along_axis(D, runs, 1).mean(1)
    return draws, D


def decide(obs, lo95, hi95, lo90, hi90):
    if lo95 > 0:
        return "I"
    if lo90 > -MARGIN and hi90 < MARGIN:
        return "E"
    if hi95 < MARGIN:
        return "X"
    return "?"


def estimate(A, b, block, n_boot, seed):
    diffs = _sr_rows(A) - _sr_rows(b[None, :])
    obs = float(diffs.mean())
    draws, D = multirun_draws(A, b, block, n_boot, seed)
    q = np.nanquantile(draws, [0.025, 0.975, 0.05, 0.95, 0.00625, 0.99375])
    p = float(np.mean(np.abs(draws - np.nanmean(draws)) >= abs(obs)))
    R = A.shape[0]
    v_runs = diffs.var(ddof=1) / R if R > 1 else 0.0
    v_avg, v_each = D.mean(1).var(ddof=1), D.var(0, ddof=1).mean()
    v_corr = v_avg if R == 1 else v_runs + max(0.0, (R * v_avg - v_each) / (R - 1))
    sd_c = np.sqrt(v_corr)
    z95, z90 = norm.ppf(0.975), norm.ppf(0.95)
    pre = {"obs": obs, "lo95": q[0], "hi95": q[1], "lo90": q[2], "hi90": q[3], "lo9875": q[4], "hi9875": q[5],
           "p": p, "sd": float(np.nanstd(draws, ddof=1))}
    cor = {"obs": obs, "lo95": obs - z95 * sd_c, "hi95": obs + z95 * sd_c, "lo90": obs - z90 * sd_c,
           "hi90": obs + z90 * sd_c, "lo9875": obs - norm.ppf(0.99375) * sd_c, "hi9875": obs + norm.ppf(0.99375) * sd_c,
           "p": float(2 * norm.sf(abs(obs) / sd_c)), "sd": float(sd_c)}
    return pre, cor


def _job(args):
    R, delta, reps, n_boot, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        A, b, _ = simulate(rng, delta, R)
        s = int(rng.integers(1 << 30))
        for block in BLOCKS:
            pre, cor = estimate(A, b, block, n_boot, s)
            for name, e in (("prereg", pre), ("corrected", cor)):
                rows.append({"block": block, "est": name, "obs": e["obs"], "sd": e["sd"],
                             "cover95": e["lo95"] <= delta <= e["hi95"], "reject": e["lo95"] > 0 or e["hi95"] < 0,
                             "cls": decide(e["obs"], e["lo95"], e["hi95"], e["lo90"], e["hi90"])})
    return R, delta, rows


def _family_job(args):
    R, reps, n_boot, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        f = garch_factor(rng, T)
        b = MU_B + f
        res = []
        for _j in range(4):
            A, _, _ = simulate(rng, 0.0, R, f=f, b=b)
            pre, cor = estimate(A, b, 21, n_boot, int(rng.integers(1 << 30)))
            res.append((pre, cor))
        for k, name in ((0, "prereg"), (1, "corrected")):
            out.append({"est": name,
                        "all95": all(r[k]["lo95"] <= 0 <= r[k]["hi95"] for r in res),
                        "all9875": all(r[k]["lo9875"] <= 0 <= r[k]["hi9875"] for r in res),
                        "any_I_or_neg": any(r[k]["lo95"] > 0 or r[k]["hi95"] < 0 for r in res),
                        "pairwise_corr_obs": None})
    return R, out


def summarise(rows, delta):
    df = pd.DataFrame(rows)
    out = {}
    for (block, est), g in df.groupby(["block", "est"]):
        cls = g["cls"].value_counts(normalize=True)
        out[f"{est}_b{block}"] = {
            "coverage95": float(g["cover95"].mean()), "reject_5pct": float(g["reject"].mean()),
            "rate_I": float(cls.get("I", 0)), "rate_E": float(cls.get("E", 0)), "rate_X": float(cls.get("X", 0)),
            "rate_inconclusive": float(cls.get("?", 0)), "mean_estimate": float(g["obs"].mean()),
            "sd_estimate": float(g["obs"].std(ddof=1)), "mean_se": float(g["sd"].mean()),
            "se_ratio": float(g["sd"].mean() / g["obs"].std(ddof=1)), "true": delta}
    return out


def _selfcheck():
    rng = np.random.default_rng(5)
    A, b, _ = simulate(rng, 0.1, 5)
    idx = pd.RangeIndex(T)
    ref = multirun_sharpe_diff([(pd.Series(a, index=idx), pd.Series(b, index=idx)) for a in A],
                               block=21, n_boot=200, seed=123)
    pre, _ = estimate(A, b, 21, 200, 123)
    assert abs(ref["ci_lo"] - pre["lo95"]) < 1e-10 and abs(ref["ci_hi"] - pre["hi95"]) < 1e-10, (ref, pre)
    assert abs(ref["p_value"] - pre["p"]) < 1e-12


def single_run(reps: int = 1000, n_boot: int = 1000) -> dict:
    """Exploratory (not in pre-registration v3): one deterministic run (no sampling variance), as for
    the DatedGPT cells; the multi-run bootstrap then reduces to a plain day-block bootstrap."""
    import scripts.calibrate_bootstrap as m
    keep, m.RUN_SD = m.RUN_SD, 0.0
    try:
        out = {}
        for j, d in enumerate((0.0, 0.20)):
            rng = np.random.default_rng(4242 + j)
            rows = []
            for _ in range(reps):
                A, b, _ = simulate(rng, d, 1)
                pre, _ = estimate(A, b, 21, n_boot, int(rng.integers(1 << 30)))
                rows.append({"block": 21, "est": "prereg", "obs": pre["obs"], "sd": pre["sd"],
                             "cover95": pre["lo95"] <= d <= pre["hi95"], "reject": pre["lo95"] > 0 or pre["hi95"] < 0,
                             "cls": decide(pre["obs"], pre["lo95"], pre["hi95"], pre["lo90"], pre["hi90"])})
            out[f"R1_delta{d:+.2f}"] = summarise(rows, d)["prereg_b21"]
        return out
    finally:
        m.RUN_SD = keep


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    _selfcheck()
    chunks = 10
    jobs = [(R, d, reps // chunks, n_boot, 1000 * i + 17 * j + R)
            for R in (5, 10) for j, d in enumerate(DELTAS) for i in range(chunks)]
    fam = [(R, reps // chunks, n_boot, 777 + 31 * i + R) for R in (5, 10) for i in range(chunks)]
    res = {"design": {"T": T, "rho": RHO, "run_sd": RUN_SD, "reps": reps, "n_boot": n_boot,
                      "deltas": DELTAS, "blocks": BLOCKS, "margin": MARGIN}}
    acc: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        for R, d, rows in ex.map(_job, jobs):
            acc.setdefault((R, d), []).extend(rows)
        famacc: dict = {}
        for R, rows in ex.map(_family_job, fam):
            famacc.setdefault(R, []).extend(rows)
    for (R, d), rows in sorted(acc.items()):
        res[f"R{R}_delta{d:+.2f}"] = summarise(rows, d)
        print(R, d, {k: (v["coverage95"], v["rate_I"], v["rate_E"], v["rate_X"]) for k, v in res[f"R{R}_delta{d:+.2f}"].items()
                     if k.endswith("b21")}, flush=True)
    for R, rows in sorted(famacc.items()):
        df = pd.DataFrame(rows)
        res[f"family4_R{R}"] = {est: {"familywise_coverage_individual95": float(g["all95"].mean()),
                                      "familywise_coverage_bonferroni9875": float(g["all9875"].mean()),
                                      "any_false_rejection": float(g["any_I_or_neg"].mean())}
                                for est, g in df.groupby("est")}
        print("family", R, res[f"family4_R{R}"], flush=True)
    res["exploratory_single_run"] = single_run(reps, n_boot)
    (TABLES_DIR / "bootstrap_calibration.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
