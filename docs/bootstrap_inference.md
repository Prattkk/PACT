# What the multi-run bootstrap estimates, and why its intervals are too wide

## Target

For protocol `a` and comparator `b`, run `r` (an independent LLM sample with its own cache key)
gives daily excess returns `a_r(t)`, `b_r(t)` over the same `T` sessions. The reported
difference is

    D = (1/R) * sum_r [ SR(a_r) - SR(b_r) ],   SR = annualised Sharpe ratio.

The inferential target is `E[D]`, the expected Sharpe difference over two sources of
variation: (i) the LLM's sampling (which run we happened to draw) and (ii) market paths
like the observed 2022-2024 path (which days we happened to observe). It is not the difference
for one particular run, and not a claim about other periods or markets.

## The procedure (pre-registered; `evaluation.statistical_tests.multirun_sharpe_diff`)

Each bootstrap draw (i) resamples `R` runs with replacement and (ii) draws one set of 21-day
circular blocks of days, common to all runs and both protocols, and recomputes `D` on the
resampled runs and days. Intervals are percentiles of the draws; the p-value is the share of
centred draws at least as far from their mean as the observed `D`.

## Double counting

Write each run's difference as `d_r = mu + u_r + e_r`, with `u_r` the run (LLM-sampling) effect
and `e_r` the day-level estimation noise of that run's Sharpe difference. `Var(D) =
Var(u)/R + Var(mean_r e_r)`. Resampling runs (step i) reproduces the spread of the `d_r`,
which already contains `Var(u) + Var(e)` — that is, day noise is already in the between-run
spread. Step (ii) then adds day-level noise again through the common day blocks. Because the
`e_r` are highly correlated across runs (same market path; protocol-return correlation about
0.98 in our runs), `Var(mean_r e_r)` is close to `Var(e)`, and adding it on top of a run spread
that already contains it inflates the variance. In the calibration (`scripts/calibrate_bootstrap.py`,
`results/tables/bootstrap_calibration.json`) the bootstrap standard error is 1.30-1.55 times
the true sampling SD of `D`, and 95% intervals cover in 98.4-99.7% of 1,000 simulated panels
per design.

The diagnostic moment-corrected variance in the same script, `Var(runs)/R + max(0, (R*v_avg -
v_each)/(R-1))`, removes the double counting; its SE ratio is 0.85-1.00 but its normal
intervals cover in only 85-93%, so it is reported, not used.

## Status

These properties are established only for the simulated data-generating process (GARCH(1,1)
volatility, AR(1) means, protocol correlation 0.98, run-level spread 0.05, T = 753). They are
not a general guarantee. Pre-registration fixed the procedure before the confirmatory runs; it
records that the procedure was chosen in advance, it does not validate it. We keep it because
switching after seeing results would be an undisclosed analytic choice, and in the simulated
process its errors fall on the conservative side (fewer false improvements and false
equivalences than nominal, at the cost of power).

For a single deterministic run (the DatedGPT cells and the controlled experiment) step (i) is
empty and the procedure is an ordinary day-block bootstrap; an exploratory calibration gives
94.2% coverage at a true difference of zero.
