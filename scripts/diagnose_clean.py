"""How dynamic are the DatedGPT scorer's views? Reads the clean-regime cache only.

Writes results/tables/clean_diag.json: per specialist, the share of weeks in which an
instrument's direction differs from the previous week, and the mean |average direction|
per instrument (1 = a constant long or short tilt).
"""
import glob
import json

import pandas as pd

from scripts._common import REPO_ROOT, TABLES_DIR

AG = {"Macro": "macro_regime", "Narrative": "narrative_event", "Fundamentals": "fundamentals_carry",
      "Cross-Asset": "cross_asset_transmission"}
rows = []
for f in glob.glob(str(REPO_ROOT / "data/cache/llm/datedgpt-instruct-scorer-v1/*.json")):
    d = json.load(open(f))
    c = json.loads(d["canonical_prompt"])
    ag = next(v for k, v in AG.items() if c["system"].startswith("You are the " + k))
    asof = json.loads(c["user"])["as_of"]
    for v in json.loads(d["response"]["text"]):
        rows.append((ag, v["instrument"], v["direction"], asof, d["prompt_hash"]))
df = pd.DataFrame(rows, columns=["agent", "sym", "dir", "asof", "h"])
df = df[df.agent != "cross_asset_transmission"].sort_values("asof")  # cross-asset depends on protocol
out = {}
for ag, g in df.groupby("agent"):
    change = g.groupby("sym")["dir"].apply(lambda s: (s.diff() != 0).iloc[1:].mean()).mean()
    tilt = g.groupby("sym")["dir"].mean().abs().mean()
    out[ag] = {"weekly_change_rate": round(float(change), 3), "mean_abs_tilt": round(float(tilt), 3)}
(TABLES_DIR / "clean_diag.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
