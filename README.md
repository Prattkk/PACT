# PACT — Protocols for Agent Coordination in Trading

A coordination-and-attribution harness for multi-agent LLM trading systems.
It holds the specialist agents, data and portfolio construction fixed and
varies only the coordination protocol, under GPT-4o and no-LLM regimes,
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

`scripts/run_pitfall_audit.py` quantifies the effect of each fix on the
deterministic baseline (prices only, no API keys).

## Layout

```
agents/                7 trading agents (per spec §4) + NullAgent for LOO
coordination/          7 coordination protocols (spec §6.2)
llm/                   GPT-4o + ChronoGPT clients (cached, prompt-hash keyed)
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
| `OPENAI_API_KEY` | frontier LLM regime | https://platform.openai.com/api-keys |
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
python scripts/export_paper.py --out ../pact-paper/generated

# TradingAgents reproduction harness check (spec §12 step 4)
python scripts/reproduce_tradingagents.py
```

## Run smoke tests

```bash
pytest -q
```

## Reproducibility (spec §8)

- All LLM responses cached as JSON keyed by SHA-256 of the canonicalized
  prompt (`llm/base.py::canonicalize_prompt`).
- Frontier model ID is pinned (`gpt-4o-2024-11-20`); the OpenAI client uses
  `temperature=0`, `seed=42`, and the response's `system_fingerprint` is
  recorded in every cache entry.
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
