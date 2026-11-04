#!/bin/bash
# Regenerate every number and table in the paper from the released LLM responses (no API calls).
# Requires FRED_API_KEY and EIA_API_KEY in .env for the point-in-time data fetchers.
set -e
cd "$(dirname "$0")/.."
tar xzf release/confirmatory_llm_cache.tar.gz          # -> data/cache/llm/{claude-sonnet-5-5,claude-haiku-5-5,datedgpt-instruct-scorer-v1}
M="independent_ensemble sequential_pipeline pipeline_placebo hierarchical"
cell() { python -m scripts.run_ablation_matrix --window b --regime "$1" --protocol "$2" --altdata --offline-llm; }
for p in $M deterministic_only debate llm_plus_anchor single_agent single_agent_sc debate_multiround; do cell none $p; done
for p in $M; do cell clean $p; done
for R in frontier_r2 frontier_r3 frontier_r4 frontier_r5 frontier_r6 $(seq -f "frontier_haiku_r%g" 2 11) frontier_haiku_anon; do
  for p in $M; do cell $R $p; done
done
for R in frontier_r2 frontier_r3 frontier_r4 frontier_r5 frontier_r6 $(seq -f "frontier_haiku_r%g" 2 11); do cell $R debate; done
for p in llm_plus_anchor single_agent single_agent_sc; do cell frontier_r2 $p; cell frontier_haiku_r2 $p; done
cell frontier_haiku_r2 debate_multiround
for R in frontier_r2 frontier_haiku_r2 clean; do
  python -m scripts.analyze_mechanisms --regime $R
  python -m scripts.analyze_narrative --regime $R
done
for R in frontier_r2 frontier_haiku_r2; do python -m scripts.run_sensitivity --regime $R; done
python -m scripts.run_sensitivity --regime clean --protocols $M
python -m scripts.run_sensitivity --regime none --protocols $M debate
python -m scripts.diagnose_clean
python -m scripts.export_paper            # -> results/paper_tex/ (numbers.tex, tab_*.tex)
