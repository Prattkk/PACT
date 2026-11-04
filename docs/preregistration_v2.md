# Pre-registration v2: follow-up analyses (written before any of them was run)

Frozen protocol: commit `3cf1f67` (prompts, models, aggregation, risk settings), unchanged.
Confirmatory Claude runs (2022-2024): Sonnet 5.5 `r2`-`r6`, Haiku 5.5 `r2`-`r11`, masked `anon1`.
Primary comparisons, threshold and inference are those of the paper: the four matched
comparisons (pipeline, placebo, pipeline vs placebo, hierarchical, each against the
no-communication ensemble), economically meaningful gain = 0.20 Sharpe, multi-run block
bootstrap (21-day circular blocks, 2,000 draws), Benjamini-Hochberg within model group.
Failures: a reply that cannot be parsed falls back to the specialist's rule-based view and is
reported as a count; a cell with any call failure is excluded.

No new paid model calls are made. Nothing below was run before this file was committed.

## 1. Evaluation on unused market data (point-in-time model)

**Period rule.** All calendar years before the 2022-2024 window in which every one of the ten
instruments has at least 400 calendar days of price history (the engine's warm-up) at the
first decision. Bitcoin spot prices start on 17 Sep 2014, so the eligible period is
**1 Jan 2016 - 31 Dec 2021** (six years). Bitcoin is held as spot BTC-USD throughout (IBIT did
not exist). Instruments, data sources, point-in-time rules, prompts and portfolio rules are
unchanged. The authors have not run any strategy on this period; the rule-based anchors were
inherited from the original codebase and we cannot certify that its authors never looked at it.

**Models.** DatedGPT instruct, vintage Y-1 for decisions in year Y (2015-2020 vintages, commit
SHAs recorded on first download), scored exactly as in the paper; and the no-LLM regime.

**Protocols.** Ensemble, pipeline, placebo pipeline (cyclic shift, as frozen) and hierarchical.

**Reporting.** Pooled 2016-2021 result; each non-overlapping calendar year; and the two
three-year halves (2016-2018, 2019-2021). Same comparisons, threshold and bootstrap.

## 2. Capability checks of the point-in-time scorer (2022-2024 inputs, frozen vintages)

Each check is reported whether it passes or fails.

- **C1 Name-only prior.** Ask each question with the asset description only (no factors, no
  anchor). Report the share of instrument-weeks on which the full scorer's direction equals the
  name-only direction, and the ensemble's Sharpe ratio with name-only views.
  *Pass:* agreement below 80%.
- **C2 Shuffled inputs.** (a) Permute the factor blocks across instruments within each call;
  (b) replace each instrument's factors with those of a randomly chosen other decision week
  (seed 7). Report the rank correlation between original and perturbed log-odds and the share
  of directions that change. *Pass:* median within-call rank correlation below 0.5 under (a),
  and at least 20% of directions changing under (b).
- **C3 Controlled directional prompts.** 40 hand-written pairs, each with a bullish and a
  bearish version of the same question that differ only in the evidence (trend, inflation and
  rates, inventories, news tone). *Pass:* log-odds(bullish) > log-odds(bearish) in at least
  75% of pairs, for each vintage (chance = 50%).
- **C4 Information transmission.** Cross-asset prompts with synthetic upstream views: both
  upstream specialists long with conviction 0.8 versus both short, and the anchor neutral.
  *Pass:* log-odds(long upstream) > log-odds(short upstream) in at least 75% of instruments ×
  vintages.

If C1-C2 fail, the paper presents the point-in-time regime as a limited diagnostic.

**Additional placebo mappings.** Five random derangements of the instruments (seeds 1-5) replace
the cyclic shift, as diagnostics for the placebo row (DatedGPT only, both periods).

## 3. Where coordination's effect disappears (cached runs, no new calls)

For each matched protocol against the ensemble of the same run, measure the share of
instrument-weeks that differ, and the mean absolute difference, at each stage:
(S1) final views: direction; (S2) final views: conviction; (S3) vol-scaled, capped target;
(S4) after drawdown and correlation overlays; (S5) after the turnover cap (submitted order);
(S6) held weights; (S7) daily return (correlation and difference).

**Single pre-specified condition.** Initial disagreement for an instrument-week =
1 - |mean of the active LLM specialists' anchor directions| (decision-time information only),
split at the median of the pooled distribution. Hypothesis: the matched protocols' absolute
weight differences and difference-return contributions are larger in high-disagreement
instrument-weeks. Tested once on the Claude runs; validated on DatedGPT 2016-2021.

## 4. Statistical robustness

- Block length 5, 21 and 63 sessions for all primary comparisons.
- Each comparison classified as: *improvement* (95% interval above 0); *meaningful gain
  excluded* (95% upper bound below 0.20); *practically equivalent* (90% interval inside
  ±0.20, i.e. two one-sided tests); *inconclusive* otherwise.
- Simultaneous intervals over the four primary comparisons per model (Bonferroni, 98.75%).
- Simulation check of the multi-run bootstrap: 1,000 replications of paired daily returns
  (T = 753, correlation 0.98, GARCH(1,1)-type volatility and AR(1) mean, 5 runs with run-level
  noise), true Sharpe difference 0 and 0.20. Report interval coverage and test size.
- Draw count and p-value construction stated in the paper (two-sided; share of centred
  bootstrap draws at least as extreme as the observed difference).

## 5. Efficiency

Calls, tokens and cost per protocol recomputed excluding the technical specialist's rationale
calls (which never change a position).
