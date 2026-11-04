"""Build release/confirmatory_llm_cache.tar.gz: every cached response the paper uses.

Claude entries: only model keys with a confirmatory run tag ("|run=..."; Sonnet r2-r6, Haiku r2-r11,
masked anon1, and the monolith samples sc2-sc5 of run r2). Untagged entries are pre-freeze
exploratory samples and are excluded. DatedGPT entries: all (deterministic scorers, both periods,
and the name-only check).
"""

from __future__ import annotations

import glob
import json
import tarfile
from collections import Counter

from scripts._common import REPO_ROOT

OUT = REPO_ROOT / "release" / "confirmatory_llm_cache.tar.gz"


def main() -> None:
    keep, counts = [], Counter()
    for f in sorted(glob.glob(str(REPO_ROOT / "data/cache/llm/*/*.json"))):
        model = json.loads(json.load(open(f))["canonical_prompt"])["model"]
        if model.startswith("claude-") and "|run=" not in model:
            continue
        if not (model.startswith("claude-") or model.startswith("datedgpt-")):
            continue
        keep.append(f)
        counts[model] += 1
    with tarfile.open(OUT, "w:gz") as t:
        for f in keep:
            t.add(f, arcname=str(__import__("pathlib").Path(f).relative_to(REPO_ROOT)))
    print(f"{len(keep)} responses -> {OUT}")
    for m, n in sorted(counts.items()):
        print(f"  {n:6d}  {m}")


if __name__ == "__main__":
    main()
