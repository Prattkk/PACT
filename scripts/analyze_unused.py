"""Unused period 2016-2021 (pre-registration v2, section 1): pooled, three-year halves and
calendar years, for the no-LLM and point-in-time regimes. Same comparisons, threshold and
block bootstrap as the main experiment (single runs). Writes results/tables/unused_c.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.statistical_tests import multirun_sharpe_diff
from scripts._common import RAW_DIR, TABLES_DIR
from scripts.analyze_robustness import MEANINGFUL, PRIMARY, classify

PROTOCOLS = ("independent_ensemble", "sequential_pipeline", "pipeline_placebo", "hierarchical",
             "pipeline_placebo_s1", "pipeline_placebo_s2", "deterministic_only")
PERIODS = {"2016-2021": ("2016-01-01", "2021-12-31"), "2016-2018": ("2016-01-01", "2018-12-31"),
           "2019-2021": ("2019-01-01", "2021-12-31"),
           **{str(y): (f"{y}-01-01", f"{y}-12-31") for y in range(2016, 2022)}}


def load(regime, p):
    d = RAW_DIR / f"c__{regime}__{p}"
    if not (d / "excess_returns.parquet").exists():
        return None
    meta = json.loads((d / "metadata.json").read_text())
    if any(v for k, v in meta.get("llm_stats", {}).items() if k.endswith(":call_failed")):
        raise RuntimeError(f"{d.name}: failed LLM calls")
    return pd.read_parquet(d / "excess_returns.parquet").iloc[:, 0]


def sr(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def main() -> None:
    out = {"meaningful": MEANINGFUL, "regimes": {}}
    for regime in ("none", "clean"):
        ser = {p: load(regime, p) for p in PROTOCOLS}
        ser = {p: s for p, s in ser.items() if s is not None}
        if "independent_ensemble" not in ser:
            continue
        reg = {"sharpe": {}, "comparisons": {}}
        for name, (a, b) in PERIODS.items():
            reg["sharpe"][name] = {p: sr(s.loc[a:b]) for p, s in ser.items()}
            comps = {}
            for x, y in PRIMARY + [("pipeline_placebo_s1", "independent_ensemble"),
                                   ("pipeline_placebo_s2", "independent_ensemble")]:
                if x in ser and y in ser:
                    pr = [(ser[x].loc[a:b], ser[y].loc[a:b])]
                    r95 = multirun_sharpe_diff(pr)
                    r90 = multirun_sharpe_diff(pr, alpha=0.10)
                    comps[f"{x}:{y}"] = {"diff": r95["diff"], "ci_lo": r95["ci_lo"], "ci_hi": r95["ci_hi"],
                                         "p_value": r95["p_value"], "ci90": [r90["ci_lo"], r90["ci_hi"]],
                                         "class": classify(r95, r90)}
            reg["comparisons"][name] = comps
        out["regimes"][regime] = reg
    (TABLES_DIR / "unused_c.json").write_text(json.dumps(out, indent=1))
    for r, d in out["regimes"].items():
        for per in ("2016-2021", "2016-2018", "2019-2021"):
            print(r, per, {k.split(":")[0][:12]: (round(v["diff"], 2), v["class"]) for k, v in d["comparisons"][per].items()})


if __name__ == "__main__":
    main()
