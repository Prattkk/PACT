"""Statistical robustness of the primary comparisons (pre-registration v2, section 4).

For every model group and primary comparison: the multi-run block-bootstrap difference with
blocks of 5, 21 and 63 sessions; the 90% interval (two one-sided tests at +/-0.20); a
Bonferroni-simultaneous interval over the four primary comparisons (98.75%); and the
pre-registered classification. Writes results/tables/robustness.json.
"""

from __future__ import annotations

import json

import pandas as pd

from evaluation.statistical_tests import multirun_sharpe_diff
from scripts._common import RAW_DIR, TABLES_DIR

MEANINGFUL = 0.20
GROUPS = {
    "sonnet": [f"frontier_r{k}" for k in range(2, 7)],
    "haiku": [f"frontier_haiku_r{k}" for k in range(2, 12)],
    "clean": ["clean"],
    "haikuanon": ["frontier_haiku_anon"],
}
PRIMARY = [("sequential_pipeline", "independent_ensemble"), ("pipeline_placebo", "independent_ensemble"),
           ("sequential_pipeline", "pipeline_placebo"), ("hierarchical", "independent_ensemble")]
EXTRA = [("debate", "independent_ensemble")]
BLOCKS = (5, 21, 63)


def ex(window: str, regime: str, protocol: str) -> pd.Series | None:
    d = RAW_DIR / f"{window}__{regime}__{protocol}"
    if not (d / "excess_returns.parquet").exists():
        return None
    meta = json.loads((d / "metadata.json").read_text())
    if meta.get("agent_failure_total", 0) or any(
            v for k, v in meta.get("llm_stats", {}).items() if k.endswith(":call_failed")):
        return None
    return pd.read_parquet(d / "excess_returns.parquet").iloc[:, 0]


def classify(r95: dict, r90: dict) -> str:
    if r95["ci_lo"] > 0:
        return "improvement"
    if -MEANINGFUL < r90["ci_lo"] and r90["ci_hi"] < MEANINGFUL:
        return "practically equivalent"
    if r95["ci_hi"] < MEANINGFUL:
        return "meaningful gain excluded"
    return "inconclusive"


def analyse(window: str, groups: dict[str, list[str]], n_boot: int = 2000) -> dict:
    out = {}
    for g, regs in groups.items():
        for a, b in PRIMARY + EXTRA:
            pairs = [(ex(window, r, a), ex(window, r, b)) for r in regs]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            row = {"n_runs": len(pairs)}
            for blk in BLOCKS:
                r95 = multirun_sharpe_diff(pairs, block=blk, n_boot=n_boot)
                row[f"b{blk}"] = {k: r95[k] for k in ("diff", "ci_lo", "ci_hi", "p_value")}
            r90 = multirun_sharpe_diff(pairs, block=21, n_boot=n_boot, alpha=0.10)
            rsim = multirun_sharpe_diff(pairs, block=21, n_boot=n_boot, alpha=0.05 / len(PRIMARY))
            row["ci90"] = [r90["ci_lo"], r90["ci_hi"]]
            row["ci_simultaneous"] = [rsim["ci_lo"], rsim["ci_hi"]]
            r95 = multirun_sharpe_diff(pairs, block=21, n_boot=n_boot)
            row["class"] = classify(r95, r90)
            row["class_by_block"] = {str(blk): classify(
                multirun_sharpe_diff(pairs, block=blk, n_boot=n_boot),
                multirun_sharpe_diff(pairs, block=blk, n_boot=n_boot, alpha=0.10)) for blk in BLOCKS}
            out[f"{g}:{a}:{b}"] = row
    return out


def main() -> None:
    res = {"meaningful": MEANINGFUL, "n_boot": 2000, "blocks": BLOCKS,
           "p_value": "two-sided; share of centred bootstrap draws with |draw - mean| >= |observed|",
           "b": analyse("b", GROUPS)}
    (TABLES_DIR / "robustness.json").write_text(json.dumps(res, indent=1))
    for k, v in res["b"].items():
        print(k, v["class"], {b: round(v[f"b{b}"]["ci_hi"], 2) for b in BLOCKS}, [round(x, 2) for x in v["ci90"]])


if __name__ == "__main__":
    main()
