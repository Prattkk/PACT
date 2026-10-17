"""Statistical inference for Sharpe-ratio comparisons.

- `ledoit_wolf_sharpe_test`: HAC-based test of H0: SR_a = SR_b for two return
  series observed on the same dates (Ledoit & Wolf 2008, Sec. 3.1). The
  variance of the SR difference is obtained by the delta method from the
  Newey-West (Bartlett) long-run covariance matrix of the four moment
  conditions (mean and variance of each series).

  An earlier version estimated each entry of that matrix as the long-run
  variance of the *product* of two moment series, i.e. a fourth/eighth-moment
  quantity rather than a covariance. For daily returns this understated the
  standard error by orders of magnitude and produced |z| values above 100.

- `paired_bootstrap_sharpe_diff`: circular block bootstrap of the SR
  difference that resamples both series with the same blocks (preserving
  their cross-correlation and autocorrelation). Used as a robustness check.
- `sharpe_bootstrap_ci`: circular block bootstrap CI for one series' SR.
- `benjamini_hochberg`: FDR control across a family of tests.

All Sharpe ratios returned by these functions are annualised (x sqrt(252)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = np.sqrt(252.0)


def newey_west_bandwidth(n: int) -> int:
    """Standard rule-of-thumb lag truncation floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_covariance(psi: np.ndarray, q: int) -> np.ndarray:
    """Newey-West long-run covariance of the rows of `psi` (n x k, mean-zero)."""
    psi = np.asarray(psi, dtype=float)
    n = psi.shape[0]
    s = psi.T @ psi / n
    for j in range(1, q + 1):
        w = 1.0 - j / (q + 1.0)
        g = psi[j:].T @ psi[:-j] / n
        s += w * (g + g.T)
    return s


def _align(r1: pd.Series, r2: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    a, b = r1.align(r2, join="inner")
    m = a.notna() & b.notna()
    return a[m].to_numpy(dtype=float), b[m].to_numpy(dtype=float)


def ledoit_wolf_sharpe_test(
    r1: pd.Series,
    r2: pd.Series,
    lags: int | None = None,
) -> dict[str, float]:
    """Two-sided HAC test of equal Sharpe ratios. Inputs are (excess) returns."""
    a, b = _align(r1, r2)
    n = len(a)
    if n < 30:
        return {"z": float("nan"), "p_value": float("nan"), "sr_a": float("nan"),
                "sr_b": float("nan"), "sr_diff": float("nan"), "se_diff": float("nan"), "n": n}
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(ddof=0), b.var(ddof=0)
    sa, sb = np.sqrt(va), np.sqrt(vb)
    diff = mu_a / sa - mu_b / sb
    psi = np.column_stack([a - mu_a, (a - mu_a) ** 2 - va, b - mu_b, (b - mu_b) ** 2 - vb])
    grad = np.array([1.0 / sa, -mu_a / (2.0 * va ** 1.5), -1.0 / sb, mu_b / (2.0 * vb ** 1.5)])
    q = newey_west_bandwidth(n) if lags is None else lags
    omega = hac_covariance(psi, q)
    var_diff = float(grad @ omega @ grad) / n
    if not np.isfinite(var_diff) or var_diff <= 0:
        z, p, se = float("nan"), float("nan"), float("nan")
    else:
        se = np.sqrt(var_diff)
        z = diff / se
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        "z": float(z), "p_value": float(p),
        "sr_a": float(mu_a / sa * ANN), "sr_b": float(mu_b / sb * ANN),
        "sr_diff": float(diff * ANN), "se_diff": float(se * ANN), "n": n,
    }


def _circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def _sr(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * ANN) if s > 0 else float("nan")


def sharpe_bootstrap_ci(
    r: pd.Series, block: int = 21, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7,
) -> tuple[float, float]:
    x = r.dropna().to_numpy(dtype=float)
    if len(x) < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.array([_sr(x[_circular_block_indices(len(x), block, rng)]) for _ in range(n_boot)])
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_sharpe_diff(
    r1: pd.Series, r2: pd.Series, block: int = 21, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7,
) -> dict[str, float]:
    """CI and two-sided p-value for SR_a - SR_b via a paired circular block bootstrap."""
    a, b = _align(r1, r2)
    n = len(a)
    if n < 2 * block:
        return {"diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    obs = _sr(a) - _sr(b)
    draws = np.empty(n_boot)
    for k in range(n_boot):
        idx = _circular_block_indices(n, block, rng)
        draws[k] = _sr(a[idx]) - _sr(b[idx])
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    centred = draws - np.nanmean(draws)
    p = float(np.mean(np.abs(centred) >= abs(obs)))
    return {"diff": float(obs), "ci_lo": float(lo), "ci_hi": float(hi), "p_value": p}


def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[bool]:
    """Return per-test reject decisions controlling FDR at `fdr`."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= (rank / m) * fdr:
            for k in order[:rank]:
                reject[k] = True
    return reject


def bh_qvalues(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()
