# Pre-registration v3: controlled communication x adjudication experiment, calibration study

Written and committed before any of the runs below. Frozen protocol `3cf1f67` is unchanged; the
new cells add specialist revision and a judge as separate, deliberately varied components.

## 1. Controlled 2x2 experiment

### Design

| Cell | Specialist revision (second pass) | Final aggregation |
|---|---|---|
| A `ctrl_isolated_vote` | each specialist revises using only its own evidence and first view | deterministic conviction-weighted vote |
| B `ctrl_isolated_judge` | the same isolated revision (identical outputs to A) | LLM judge |
| C `ctrl_shared_vote` | each specialist revises seeing the other specialists' first views | deterministic vote |
| D `ctrl_shared_judge` | the same communicating revision (identical outputs to C) | the same LLM judge |

**First pass (identical in all four cells).** Macro, narrative and fundamentals specialists
exactly as frozen (same prompts, so the same responses); technical specialist rule-based as
frozen; risk agent unchanged. The cross-asset specialist gets its own market evidence so that it
is active in every cell: per instrument the 20-session return, and the 63-session correlations
with SPY, UUP (dollar), IEF (Treasuries) and USO (oil), all from adjusted closes up to the
decision close. It never receives other specialists' views in the first pass.

**Second pass (one call per LLM specialist: macro, narrative, fundamentals, cross-asset).**
Both conditions have the same system prompt, call count and token budget. The isolated
condition gets the specialist's own per-instrument evidence and its first view. The
communicating condition gets the same plus the other four specialists' first-pass direction
and conviction per instrument (views only; no other specialist's evidence). The technical
specialist is rule-based and keeps its first view in all cells.

**Aggregation.** Vote: the frozen `conviction_weighted_vote` over the five revised specialist
views. Judge: one adjudicator call per decision date that receives, per instrument, every
specialist's revised direction, conviction and evidence, and returns per-instrument direction
and conviction. The judge prompt is the same in B and D.

**Identical inputs.** All cells see the same factual inputs (first-pass evidence). A and B (and
C and D) use the same revision prompts, so the response cache returns identical revised outputs.
Portfolio construction, risk overlays and execution are those of the frozen engine.

### Contrasts (Sharpe-ratio differences, 2022-2024)

Principal: **D - B** (communication with the judge held fixed) and **B - A** (judge vs vote
with isolated revision). Supporting: **C - A** (communication under the vote) and the
interaction **(D - B) - (C - A)**. Classification as in the paper: I if the 95% interval is
above 0, E if the 90% interval is inside +/-0.20, X if the 95% upper bound is below 0.20,
otherwise ?. Simultaneous inference over the two principal contrasts: Bonferroni 97.5%
intervals. Block bootstrap with 21-day circular blocks (2,000 draws); sensitivity 5 and 63.

### Execution in this revision: point-in-time scorer only

No paid model calls are made in this revision. The design is executed with the DatedGPT
point-in-time scorer (vintage Y-1), which is deterministic (one run, no sampling variance).
Revision and judging are log-probability questions with the same contents as the prompts
above (own evidence, first view, peers' views; per-instrument judge summaries), standardised
across instruments as in the frozen scorer. New questions are scored in float32 (bfloat16 is
about ten times slower on the available CPU); first-pass responses come from the frozen
bfloat16 cache and are identical in all four cells, so precision cannot differ between cells.
The scorer failed three of four pre-registered capability checks (v2), so this run tests the
design and the mechanisms, not whether communication helps a capable model.

### Claude execution (specified, not run in this revision)

Sonnet 5.5 runs r2-r6 and Haiku 5.5 runs r2-r11, reusing each run's cached first-pass
responses. New calls per run and week: 1 cross-asset first pass, 8 revisions, 2 judge calls
(157 weeks: 1,727 per run). Estimated cost at list prices: Sonnet about $60, Haiku about $6.

## 2. Independent market test with a checkpoint that predates it (specified, not run)

Claude Haiku 4.5 (`claude-haiku-4-5-20251001`, released 15 Oct 2025) and Sonnet 4.5
(`claude-sonnet-4-5-20250929`, released 29 Sep 2025; API retirement 30 Nov 2026), first
decision on the first Friday after 15 Oct 2025, last on 30 Oct 2026; the same four cells. A
capability check on 2025 inputs (name-only prior and shuffled inputs, thresholds as C1-C2)
must pass before any trading result is inspected. Precondition: confirmation that no strategy
result after 31 Dec 2024 informed development; this has not been confirmed.

## 3. Calibration of the multi-run bootstrap

Data-generating process as in v2 (GARCH(1,1) factor with AR(1) mean, protocol correlation
0.98, run-level Sharpe shift SD 0.05, T = 753). Grid: R in {5, 10} runs; true difference in
{-0.20, 0, +0.10, +0.20}; estimator block length in {5, 21, 63}; 1,000 replications of 1,000
draws each. Reported per cell: 95% coverage; superiority rejection rate (size at 0, power at
+0.10/+0.20); E rate (correct at 0, false at +/-0.20, nominal at most 5%); X rate (false at
+0.20, nominal at most 2.5%); mean interval width. A family of four contrasts sharing the
market path (true difference 0): family-wise coverage of individual 95% and Bonferroni 98.75%
intervals. Estimators: the pre-registered estimator (used for every result in the paper) and,
as a diagnostic, a moment-corrected variant that removes the double-counted within-run noise:
V = V_runs + (R V_avg - V_each) / (R - 1), where V_runs is the run-resampling variance, V_avg
the day-block variance of the run-averaged difference and V_each the mean per-run day-block
variance; normal interval. The paper's classifications use the pre-registered estimator.
