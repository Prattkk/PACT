"""Simulation check of the multi-run block bootstrap (pre-registration v2, section 4).

Paired daily returns with a known population Sharpe difference (0 or 0.20): a common market
factor with GARCH(1,1) volatility and AR(1) mean (b), and a protocol (a) equal to the factor
plus run-specific noise chosen so corr(a, b) = 0.98, plus a run-level mean shift (model
sampling). T = 753 sessions, R = 5 runs. Calls the production `multirun_sharpe_diff` and
reports interval coverage and the rejection rate at the 5% level.
Writes results/tables/bootstrap_validation.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from evaluation.statistical_tests import multirun_sharpe_diff
from scripts._common import TABLES_DIR

T, R, RHO, ANN = 753, 5, 0.98, np.sqrt(252)
VOL = 0.12 / ANN          # daily vol of the factor
MU_B = 0.3 * VOL / ANN    # factor Sharpe 0.3
RUN_SD = 0.05             # across-run SD of the Sharpe difference (as observed, 0.02-0.07)


def garch_factor(rng, n, phi=0.05, alpha=0.08, beta=0.90):
    omega = VOL ** 2 * (1 - phi ** 2) * (1 - alpha - beta)
    h, e_prev, x_prev = VOL ** 2 * (1 - phi ** 2), 0.0, 0.0
    out = np.empty(n)
    for t in range(n):
        h = omega + alpha * e_prev ** 2 + beta * h
        e = np.sqrt(h) * rng.standard_normal()
        x = phi * x_prev + e
        out[t], e_prev, x_prev = x, e, x
    return out


def one_rep(rng, delta, n_boot):
    f = garch_factor(rng, T)
    sd_eta = VOL * np.sqrt(1 / RHO ** 2 - 1)
    sd_a = np.sqrt(VOL ** 2 + sd_eta ** 2)
    idx = pd.RangeIndex(T)
    b = pd.Series(MU_B + f, index=idx)
    pairs = []
    for _ in range(R):
        sr_shift = rng.normal(0.0, RUN_SD)
        mu_a = (MU_B / VOL + (delta + sr_shift) / ANN) * sd_a
        a = pd.Series(mu_a + f + sd_eta * rng.standard_normal(T), index=idx)
        pairs.append((a, b))
    return multirun_sharpe_diff(pairs, block=21, n_boot=n_boot, seed=int(rng.integers(1 << 30)))


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng = np.random.default_rng(2026)
    out = {"T": T, "R": R, "rho": RHO, "run_sd": RUN_SD, "reps": reps, "n_boot": n_boot}
    for delta in (0.0, 0.20):
        cover = reject = 0
        width, est = [], []
        for _ in range(reps):
            r = one_rep(rng, delta, n_boot)
            cover += r["ci_lo"] <= delta <= r["ci_hi"]
            reject += r["p_value"] < 0.05
            width.append(r["ci_hi"] - r["ci_lo"])
            est.append(r["diff"])
        out[f"delta_{delta:.2f}"] = {"coverage_95": cover / reps, "reject_5pct": reject / reps,
                                     "mean_estimate": float(np.mean(est)), "mean_ci_width": float(np.mean(width))}
        print(delta, out[f"delta_{delta:.2f}"], flush=True)
    (TABLES_DIR / "bootstrap_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
