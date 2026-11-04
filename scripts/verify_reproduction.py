"""Compare a rebuilt run manifest against the committed one (docs/run_manifest.json).

Fails (exit 1) if any reported cell is missing, if its excess-return series differs (SHA-256), if its
failed-call or unparseable-reply counts differ, or if the number of released responses per
confirmatory model key differs. run_ablation_matrix catches cell exceptions and exits 0, so this is
the step of reproduce_paper.sh that detects a failed or divergent cell.

  python -m scripts.verify_reproduction results/run_manifest_rebuilt.json
"""

from __future__ import annotations

import json
import sys

from scripts._common import REPO_ROOT


def _cells(m: dict) -> dict:
    return {(c["window"], reg, c["protocol"]): c for reg, cs in m["cells"].items() for c in cs}


def _released(models: dict) -> dict:
    return {k: v for k, v in models.items() if "|run=" in k or k.startswith("datedgpt-")}


def main() -> None:
    ref = json.loads((REPO_ROOT / "docs" / "run_manifest.json").read_text())
    new = json.loads(open(sys.argv[1]).read())
    a, b = _cells(ref), _cells(new)
    problems = [f"missing cell {k}" for k in sorted(set(a) - set(b))]
    for k in sorted(set(a) & set(b)):
        for f in ("excess_returns_sha256", "failed_calls", "unparseable", "rebalances"):
            if a[k].get(f) != b[k].get(f):
                problems.append(f"{k}: {f} {b[k].get(f)} != reference {a[k].get(f)}")
    ra, rb = _released(ref["cached_responses_by_model_key"]), _released(new["cached_responses_by_model_key"])
    for k in sorted(set(ra) | set(rb)):
        if ra.get(k) != rb.get(k):
            problems.append(f"responses {k}: {rb.get(k)} != reference {ra.get(k)}")
    new_px = [v for k, v in new["input_data"].items() if k.startswith("prices")]
    for k, v in ref["input_data"].items():
        if k.startswith("prices"):  # key carries the snapshot folder name, so match on content
            if v not in new_px:
                problems.append(f"input data {k}: no supplied price snapshot matches the reference hashes")
        elif new["input_data"].get(k) != v:
            problems.append(f"input data {k} differs from the reference hashes")
    print(f"{len(set(a) & set(b))}/{len(a)} reference cells present; {len(problems)} problems")
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
