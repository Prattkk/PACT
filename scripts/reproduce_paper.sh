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

# ---- pre-registration v2 analyses (docs/preregistration_v2.md) ----------------------------------
cell clean_nameonly independent_ensemble                       # capability check C1, ensemble Sharpe
for s in 1 2 3 4 5; do cell clean pipeline_placebo_s$s; done   # random-derangement placebos
cellc() { python -m scripts.run_ablation_matrix --window c --regime "$1" --protocol "$2" --altdata $3; }
for p in deterministic_only $M pipeline_placebo_s1 pipeline_placebo_s2; do cellc none $p; done
for p in $M pipeline_placebo_s1 pipeline_placebo_s2; do cellc clean $p --offline-llm; done
python -m scripts.analyze_robustness       # block lengths, equivalence classes, simultaneous intervals
python -m scripts.validate_bootstrap       # simulation check of the multi-run bootstrap (CPU, ~1 h)
python -m scripts.analyze_factorial        # communication x adjudication contrasts
python -m scripts.analyze_decision_flow --window b
python -m scripts.analyze_decision_flow --window b --protocols debate --tag _debate --groups sonnet haiku
python -m scripts.analyze_disagreement --window b
python -m scripts.analyze_unused
python -m scripts.analyze_decision_flow --window c --groups clean
python -m scripts.analyze_disagreement --window c --groups clean
# C1-C4 (scripts/check_datedgpt.py) need the DatedGPT weights (2021-2023 vintages) and several CPU
# hours; their outputs are archived in results/tables/datedgpt_checks.json.
python -m scripts.build_manifest           # -> docs/run_manifest.json (cells, prompt and data hashes)
python -m scripts.export_paper             # -> results/paper_tex/ (numbers.tex, tab_*.tex)
python -m scripts.export_v2                # -> results/paper_tex/ (numbers_v2.tex, tab_main2/unused/cost2, fig_flow.pdf)
