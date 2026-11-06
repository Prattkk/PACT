# PACT — Protocols for Agent Coordination in Trading

A coordination-and-attribution harness for multi-agent LLM trading systems.
It holds the specialist agents, data and portfolio construction fixed and
varies only the coordination protocol, under Claude (frontier) and no-LLM regimes,
against a deterministic baseline and passive benchmarks.

The contribution name comes from the headline ablation: 7 coordination
**protocols** × 2 LLM regimes, tested against single-agent and
no-communication ensemble baselines. PACT = the protocol-level study.

This README only documents how the code is wired. Findings live in the paper.

## Results schema v2 (read before reusing old results)

The `paper-fixes` changes alter results materially. Cells written by the
earlier engine are rejected by `load_backtest_result` and must be rerun:

- BTC/IBIT splice is now return-based (the level splice produced a -99.9%
  return on 2024-01-11).
- Mixed crypto/ETF panels are aligned to the exchange calendar.
- Holdings-based simulator: positions drift, cash and collateral earn the
  13-week T-bill rate, costs are half of the round-trip quote per one-way unit,
  orders execute at the next session's close, 400-day estimation warm-up,
  drawdown breaker on the strategy's own NAV.
- Sharpe ratios use excess returns over T-bills.
- ALFRED observations are filtered by first-release date (no pre-publication
  CPI/GDP); the macro regime uses changes in YoY IP growth and CPI inflation.
- Wikipedia uses previous-day data; Google Trends uses completed weeks only.
- The Ledoit-Wolf Sharpe test uses the correct HAC covariance matrix.
- Every swallowed agent/LLM failure is counted in cell metadata.
- The independent ensemble uses the manager's conviction-weighted vote, so the
  pipeline and hierarchical protocols each differ from it by one mechanism.
- EIA/NOAA values are used only after publication; the Polymarket variant is
  removed from the alt-data ablation and equity-index XBRL aggregation is
  disabled until they are fixed and validated.

`scripts/run_pitfall_audit.py` quantifies the effect of each fix on the
deterministic baseline (prices only, no API keys).

## Layout

```
agents/                7 trading agents (per spec §4) + NullAgent for LOO
coordination/          7 coordination protocols (spec §6.2)
llm/                   Claude (default frontier), GPT-4o + ChronoGPT clients (cached, prompt-hash keyed)
                       chronogpt_revisions.py pins HF SHAs for 1999-2024
data/fetchers/         prices, ALFRED, EDGAR, EDGAR XBRL aggregator,
                       GDELT (BigQuery), EIA, NOAA, Wikipedia, Polymarket,
                       Google Trends, GitHub Archive (BigQuery), BTC on-chain,
                       FOMC text
data/constituents/     Static SPY / QQQ / EEM / IWM index member lists
data/cache/            On-disk raw-response + LLM-prompt cache
factors/               Deterministic factor library (momentum, GARCH, VaR)
portfolio/             Vol-targeted sizing + risk overlays (spec §5)
backtest/              Engine + walk-forward (spec §6.5)
evaluation/            Metrics, Ledoit-Wolf, BH, DAG-Shapley + LOO,
                       regime stratification
scripts/               Driver scripts — see "Drivers" below
configs/               Ablation matrices + per-agent prompts
tests/                 Smoke tests for wiring
results/               Backtest cells, paper tables, paper figures (gitignored)
logs/                  Rotating pact.log (gitignored)
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ".[altdata]"  # google-cloud-bigquery, pytrends
```

Required environment (`.env` at repo root, gitignored):

| Variable | Required for | How to get |
|---|---|---|
| `ANTHROPIC_API_KEY` | frontier LLM regime (Claude, default) | https://console.anthropic.com/settings/keys |
| `OPENAI_API_KEY` | only if `PACT_FRONTIER=gpt4o` | https://platform.openai.com/api-keys |
| `FRED_API_KEY` | ALFRED macro series | https://fred.stlouisfed.org/docs/api/api_key.html |
| `SEC_USER_AGENT` | EDGAR filings + XBRL | string like `"Name email@x.com"` |
| `GCP_PROJECT_ID` + `GOOGLE_APPLICATION_CREDENTIALS` | GDELT + GitHub Archive (BigQuery) | GCP service account JSON |
| `EIA_API_KEY` | crude-oil inventories | https://www.eia.gov/opendata/register.php |
| `NOAA_TOKEN` (optional) | heating-degree days | https://www.ncdc.noaa.gov/cdo-web/token |

ChronoGPT (open-source LLM regime) is public on HuggingFace — no key.

## Drivers

```bash
# Headline coordination matrix (spec §6.2)
python scripts/run_ablation_matrix.py --window b --regime frontier --protocol all --altdata

# Pre-warm constituent prices before a frontier sweep (avoids yfinance hangs)
python scripts/prefetch_constituents.py --window b

# Alt-data 5-variant ablation (spec §6.6)
python scripts/run_altdata_ablation.py --protocols sequential_pipeline debate --regime frontier --window b

# Implementation audit on the deterministic baseline (prices only)
python scripts/run_pitfall_audit.py

# Per-agent attribution (LOO + optional Monte Carlo Shapley)
python scripts/run_attribution.py --cell b__frontier__sequential_pipeline [--shapley]

# Every number, table and figure in the paper (benchmarks, tests, CBS,
# cost sensitivity, sub-periods, LOO, alt-data) -- single source of truth
python -m scripts.export_paper            # -> results/paper_tex/ (default; --out to change)

# TradingAgents reproduction harness check (spec §12 step 4)
python scripts/reproduce_tradingagents.py
```

## Run smoke tests

```bash
pytest -q
```

## Reproducing the PAKDD 2027 paper

`PX_B=<2022-2024 snapshot> PX_C=<2016-2021 snapshot> bash scripts/reproduce_paper.sh` regenerates
every number and table in the paper offline (no API keys, no network) from two release files:

| File | Contents | SHA-256 |
|---|---|---|
| `release/confirmatory_llm_cache.tar.gz` | every Claude response of the reported runs (Sonnet 5.5 r2-r6: 22,133; Haiku 5.5 r2-r11 and masked anon1: 48,170; all after the protocol freeze at 3cf1f67) and the DatedGPT scores (3,918 frozen-protocol + 471 name-only + 1,727 controlled-experiment `scorer-v2-fp32`) | `7a5dd7000af70bdcd3e9c34c10ebdfdac2a3a42e27e506995cd4da31708c3d8f` |
| `release/input_data_snapshot.tar.gz` | point-in-time inputs other than prices: GDELT daily counts, first-release FRED/ALFRED vintages, EIA stocks, NOAA degree days, Wikimedia pageviews, blockchain.com charts | per-folder hashes in `docs/run_manifest.json` |

Run it from the head of branch `paper-fixes`; the paper's numbers come from that commit's code
(for example, capability check C2 passes only if both of its parts pass). Tested with CPython 3.13 (pandas 2.2.3,
numpy 2.2, scipy 1.18, pyarrow 26, statsmodels 0.15, arch 8, yfinance 1.7); `pip install -e .`
installs everything, and scikit-learn, google-cloud-bigquery, vectorbt, openai, seaborn and
tabulate are not imported on the offline path. Runtime on 2 cores is about 2.5 core-hours for the 123 cells and 4.5 for the
analyses; `SKIP_SLOW=1` skips the 1-hour bootstrap simulation and keeps its archived output.

The script rebuilds the run manifest into `results/run_manifest_rebuilt.json` and
`scripts/verify_reproduction.py` compares it with `docs/run_manifest.json`: every cell's
excess-return values (SHA-256 of dates and values rounded to 1e-10, so a different pandas or
pyarrow version that writes different parquet bytes still passes), failed-call and unparseable-reply counts, the released response counts and
the input hashes. It exits non-zero on any difference (the cell runner itself exits 0 when a cell
fails, so this is the check to rely on). An automated rerun (a separate Claude agent) from a fresh clone reproduced the
119 frozen-protocol cells exactly and every regenerated number the paper used at that point; a
second rerun reproduced the four controlled cells and `controlled_clean.json` exactly
(`docs/reproduction_report.md`). The calibration simulation has not been independently rerun.

Exploratory responses generated before the protocol freeze (Opus, untagged Sonnet and Haiku
samples) are not released and no number in the paper uses them. `export_paper` still has slots
for two exploratory pre-freeze analyses that are not in the paper (leave-one-agent-out
attribution, `tab_loo.tex` and the `loo.*`/`hfour.*` keys; alt-data variants, `tab_altdata.tex`);
without their inputs they print `\pending`. Two archived inputs are not regenerated by the
script: `datedgpt_checks.json` (needs the DatedGPT weights) and `pitfall_audit_prefix.json`
(the original code at commit cb8e79f; the file records how to rerun it).

Follow-up analyses are specified in `docs/preregistration_v2.md` (committed before they
were run) and regenerated by the second half of the same script:

| Paper item | Script | Output (`results/tables/`) |
|---|---|---|
| Equivalence classes, block lengths 5/21/63, simultaneous intervals | `analyze_robustness` | `robustness.json` |
| Simulation check of the multi-run bootstrap | `validate_bootstrap` | `bootstrap_validation.json` |
| Debate four-cell reading (frozen protocols, post hoc) | `analyze_factorial` | `factorial_b.json` |
| Decision trace, stages S1-S7 | `analyze_decision_flow` | `decision_flow_<window><tag>.json` |
| Disagreement condition | `analyze_disagreement` | `disagreement_<window>.json` |
| Unused period 2016-2021 | `analyze_unused` | `unused_c.json` |
| DatedGPT capability checks C1-C4 (needs model weights) | `check_datedgpt` | `datedgpt_checks.json` |
| Placebo mappings, DatedGPT 2022-2024 | `analyze_placebo_seeds` | `placebo_seeds_b.json` |
| Debate judge: sign changes vs resizing | `analyze_judge_decomposition` | `judge_decomposition_b.json` |
| Implementation audit (Table 7, left) | `run_pitfall_audit` | `pitfall_audit.json` |
| Controlled communication x adjudication 2x2 (pre-registration v3) | `analyze_controlled` | `controlled_clean.json` |
| Calibration of I/E/X decisions, 5 and 10 runs (pre-registration v3) | `calibrate_bootstrap` | `bootstrap_calibration.json` |
| Run-to-result manifest (prompt, config and data hashes) | `build_manifest` | `docs/run_manifest.json` |

`scripts/export_paper.py` and `scripts/export_v2.py` turn these into the paper's
`numbers*.tex`, tables and figure. Commit and file references for the public-code
audit are in `docs/public_code_audit.md`. Prices come from Yahoo Finance and are not
redistributed; `docs/run_manifest.json` records a SHA-256 hash of every price series
used, so a fresh download can be verified before rerunning.

## Reproducibility (spec §8)

- All LLM responses cached as JSON keyed by SHA-256 of the canonicalized
  prompt (`llm/base.py::canonicalize_prompt`).
- Frontier model is pinned to `claude-sonnet-5-5` (override with `PACT_CLAUDE_MODEL`)
  with extended thinking off; the API exposes no temperature or seed for this model.
  The API's returned model ID, message ID and token usage are recorded in every
  cache entry. `python -m scripts.prewarm_llm_cache` fills the cache in parallel. The old GPT-4o path
  (`gpt-4o-2024-11-20`, seed 42) is still available with `PACT_FRONTIER=gpt4o`.
- ChronoGPT yearly checkpoints (`manelalab/chrono-gpt-instruct-v1-YYYY1231`)
  are pinned by HF commit SHA in `llm/chronogpt_revisions.py` for every year
  1999-2024. Use `ChronoGPTClient.for_decision_year(Y)` to load the
  contamination-clean checkpoint with cutoff `Y-1`.
- All raw fetcher responses cached in `data/cache/store/`.
- ALFRED fetcher returns first-release values only — uses the REST
  `output_type=4` endpoint for revised series (CPI / GDP / UNRATE / INDPRO);
  daily non-revised series fetched via `fred.get_series`. Wider-window cache
  scan eliminates per-rebalance cache misses on rolling windows.
- EDGAR fetcher exposes `effective_session_date` enforcing the 17:30-ET
  cutoff rule (filings after 17:30 → next session's open). The XBRL
  aggregator (`data/fetchers/edgar_aggregates.py`) honors the same cutoff
  via fact-level `filed_date <= as_of` filtering.
- GDELT BigQuery queries use `maximum_bytes_billed` and dry-run pre-checks
  to stay inside the 1 TB/month free tier.
- "Fully reproducible" applies to the open-source path. Frontier path is
  "conditionally reproducible (snapshot dated [...])".

## Caveats documented in code

- **MVRV proxy** (BTC): blockchain.info doesn't expose realized cap, so the
  fetcher uses price / 200d-MA as a stand-in. See
  `data/fetchers/blockchain_info.py::mvrv_proxy`.
- **Equity constituent lists**: static-as-of-2025 in
  `data/constituents/*.json`, applied historically. Standard simplification
  in literature; survivorship bias is small for Window B (2022-2024).
- **IWM equity-fundamentals**: deliberately zeroed —
  see `agents/fundamentals_carry.py::EQUITY_INDICES_AGGREGATABLE`.
- **EEM aggregate**: skews toward US-listed ADRs (foreign filers without
  SEC `companyfacts` are skipped).
