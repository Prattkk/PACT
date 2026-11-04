"""Run manifest for the paper: which runs feed which result, with model configuration,
prompt hashes, response counts and input-data hashes. Writes docs/run_manifest.json.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import Counter, defaultdict

from scripts._common import REPO_ROOT, RAW_DIR

PAPER_GROUPS = {
    "Sonnet 5.5 (Tables 4-7)": [f"frontier_r{k}" for k in range(2, 7)],
    "Haiku 5.5 (Tables 4-7)": [f"frontier_haiku_r{k}" for k in range(2, 12)],
    "Haiku 5.5 masked (Table 4)": ["frontier_haiku_anon"],
    "DatedGPT point-in-time (Tables 4-5, unused period)": ["clean"],
    "No LLM (Table 5, audit)": ["none"],
}


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cells = defaultdict(list)
    for d in sorted(os.listdir(RAW_DIR)):
        w, reg, proto = d.split("__")
        meta = json.loads((RAW_DIR / d / "metadata.json").read_text())
        cells[reg].append({"window": w, "protocol": proto, "rebalances": meta.get("rebalances"),
                           "git_commit": meta.get("git_commit"),
                           "failed_calls": sum(v for k, v in meta.get("llm_stats", {}).items() if k.endswith(":call_failed")),
                           "unparseable": sum(v for k, v in meta.get("llm_stats", {}).items() if k.endswith(":unparseable")),
                           "excess_returns_sha256": sha(str(RAW_DIR / d / "excess_returns.parquet"))})
    models, prompts = Counter(), defaultdict(set)
    for f in glob.glob(str(REPO_ROOT / "data/cache/llm/*/*.json")):
        d = json.load(open(f))
        c = json.loads(d["canonical_prompt"])
        models[c.get("model", "")] += 1
        prompts[c.get("model", "").split("|")[0]].add(hashlib.sha256(c["system"].encode()).hexdigest()[:16])
    data = {}
    for pat in ("data/cache/gdelt_days/*.json",):
        files = sorted(glob.glob(str(REPO_ROOT / pat)))
        h = hashlib.sha256()
        for f in files:
            h.update(open(f, "rb").read())
        data[pat] = {"files": len(files), "sha256_concat": h.hexdigest()}
    px = os.environ.get("PACT_PX_DIR")
    if px:
        data["prices (Yahoo, not redistributed)"] = {os.path.basename(f): sha(f) for f in sorted(glob.glob(px + "/*.csv"))}
    out = {"protocol_freeze_commit": "3cf1f67",
           "paper_groups": PAPER_GROUPS,
           "cells": cells,
           "cached_responses_by_model_key": dict(sorted(models.items())),
           "system_prompt_sha256_prefixes": {k: sorted(v) for k, v in prompts.items()},
           "input_data": data}
    (REPO_ROOT / "docs" / "run_manifest.json").write_text(json.dumps(out, indent=1))
    print("cells:", sum(len(v) for v in cells.values()), "model keys:", len(models))


if __name__ == "__main__":
    main()
