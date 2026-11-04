"""Approximate communication x adjudication factorial from existing protocols (window b).

A = independent_ensemble (independent specialists, deterministic vote)
B = debate               (independent specialists, LLM judge)
C = sequential_pipeline  (upstream views passed to the cross-asset specialist, deterministic vote)
D = debate_multiround    (bull and bear researchers exchange arguments before the same judge)

This is not the matched 2x2 design (equal revision opportunities and token budgets); C and D
implement "communication" differently. Contrasts are linear combinations of run-level Sharpe
ratios; each bootstrap draw resamples runs and applies one common 21-day circular block resample
of days to every run, as in evaluation.statistical_tests.multirun_sharpe_diff. Writes
results/tables/factorial_b.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.statistical_tests import _circular_block_indices, _sr
from scripts._common import RAW_DIR, TABLES_DIR
from scripts.analyze_robustness import MEANINGFUL
from scripts.export_paper import GROUPS as _G

GROUPS = {k: regs for k, _, regs in _G}

CELLS = {"A": "independent_ensemble", "B": "debate", "C": "sequential_pipeline", "D": "debate_multiround"}
CONTRASTS = {
    "B-A (judge, no communication)": {"B": 1, "A": -1},
    "C-A (communication, vote)": {"C": 1, "A": -1},
    "D-B (communication, judge)": {"D": 1, "B": -1},
    "D-A (both)": {"D": 1, "A": -1},
    "interaction (D-B)-(C-A)": {"D": 1, "B": -1, "C": -1, "A": 1},
}


def load(regime, p):
    d = RAW_DIR / f"b__{regime}__{p}"
    if not (d / "excess_returns.parquet").exists():
        return None
    meta = json.loads((d / "metadata.json").read_text())
    if any(v for k, v in meta.get("llm_stats", {}).items() if k.endswith(":call_failed")):
        return None
    return pd.read_parquet(d / "excess_returns.parquet").iloc[:, 0]


def contrast(runs, w, n_boot=2000, block=21, seed=11):
    idx0 = runs[0][next(iter(w))].index
    for r in runs:
        for k in w:
            idx0 = idx0.intersection(r[k].index)
    arr = [{k: r[k].reindex(idx0).to_numpy() for k in w} for r in runs]
    n, R = len(idx0), len(arr)
    val = lambda a, ix: sum(c * _sr(a[k][ix]) for k, c in w.items())  # noqa: E731
    full = np.arange(n)
    per = np.array([val(a, full) for a in arr])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        rr = rng.integers(0, R, R)
        ix = _circular_block_indices(n, block, rng)
        draws[b] = np.mean([val(arr[r], ix) for r in rr])
    obs = float(per.mean())
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    lo90, hi90 = np.nanquantile(draws, [0.05, 0.95])
    p = float(np.mean(np.abs(draws - np.nanmean(draws)) >= abs(obs)))
    if lo > 0:
        cls = "improvement" if obs > 0 else "harm"
    elif hi < 0:
        cls = "harm"
    elif -MEANINGFUL < lo90 and hi90 < MEANINGFUL:
        cls = "practically equivalent"
    elif hi < MEANINGFUL:
        cls = "meaningful gain excluded"
    else:
        cls = "inconclusive"
    return {"diff": obs, "ci": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)], "p_value": p,
            "n_runs": R, "share_positive": float((per > 0).mean()), "per_run": per.tolist(), "class": cls}


def main() -> None:
    out = {"meaningful": MEANINGFUL, "note": __doc__.split("\n\n")[1].replace("\n", " "), "groups": {}}
    for g in ("sonnet", "haiku"):
        runs = []
        for reg in GROUPS[g]:
            s = {k: load(reg, p) for k, p in CELLS.items()}
            runs.append({k: v for k, v in s.items() if v is not None})
        res = {}
        for name, w in CONTRASTS.items():
            use = [r for r in runs if all(k in r for k in w)]
            if use:
                res[name] = contrast(use, w)
        out["groups"][g] = res
    (TABLES_DIR / "factorial_b.json").write_text(json.dumps(out, indent=1))
    for g, res in out["groups"].items():
        for k, v in res.items():
            print(g, k, v["n_runs"], round(v["diff"], 3), [round(x, 2) for x in v["ci"]], v["class"])


if __name__ == "__main__":
    main()
