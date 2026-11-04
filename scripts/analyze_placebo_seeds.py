"""Placebo mappings (pre-registration v2, amendment 1): DatedGPT 2022-2024, cyclic shift and five
random derangements, each against the ensemble (single deterministic runs, 21-day block bootstrap).
Writes results/tables/placebo_seeds_b.json."""

from __future__ import annotations

import json

import pandas as pd

from evaluation.statistical_tests import multirun_sharpe_diff
from scripts._common import RAW_DIR, TABLES_DIR
from scripts.analyze_robustness import classify


def load(p):
    return pd.read_parquet(RAW_DIR / f"b__clean__{p}" / "excess_returns.parquet").iloc[:, 0]


def main() -> None:
    e, out = load("independent_ensemble"), {}
    for p in ["pipeline_placebo", "sequential_pipeline"] + [f"pipeline_placebo_s{k}" for k in range(1, 6)]:
        m = json.loads((RAW_DIR / f"b__clean__{p}" / "metadata.json").read_text())
        fails = sum(v for k, v in m.get("llm_stats", {}).items() if k.endswith("call_failed"))
        r95, r90 = multirun_sharpe_diff([(load(p), e)]), multirun_sharpe_diff([(load(p), e)], alpha=0.10)
        out[p] = {"diff": r95["diff"], "ci": [r95["ci_lo"], r95["ci_hi"]], "class": classify(r95, r90), "fails": fails}
        print(p, round(r95["diff"], 2), out[p]["class"])
    (TABLES_DIR / "placebo_seeds_b.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
