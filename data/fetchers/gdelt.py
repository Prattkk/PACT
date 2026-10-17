"""GDELT GKG fetcher via BigQuery (spec §3.2, §4.1).

Real implementation backed by `gdelt-bq.gdeltv2.gkg_partitioned` — the
day-partitioned GKG table on the GDELT public BigQuery dataset.

For each instrument we map a set of GDELT theme codes + URL keywords to
news that's likely to drive that asset (SPY → ECON_STOCKMARKET; USO →
ECON_PETROLEUM, OPEC; etc.). One batched query covers a multi-month window
across all instruments → per-day per-instrument event-count and avg-tone.

Free-tier safety: every query passes `maximum_bytes_billed` to cap cost
(default 500 GB → ~half the monthly free-tier budget per call).

Coverage caveat: GDELT 2.0 (V2 schema with V2Themes/V2Tone) starts
2015-02-18. Pre-2015 needs GDELT 1.0 (different schema, separate table).
This fetcher refuses requests starting before 2015-02-18.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Iterable

import pandas as pd

from data.cache.cache import read as cache_read
from data.cache.cache import write as cache_write
from pact_logging import get_logger

log = get_logger(__name__)

NAMESPACE = "gdelt"

# 1 TB monthly free tier; cap each query at 500 GB by default.
DEFAULT_MAX_BYTES = 500 * 10**9

# GDELT 2.0 effective start (V2 schema rolled out 2015-02-18).
GDELT_V2_START = date(2015, 2, 18)


# Theme/keyword filters per tradeable instrument.
# Themes: GDELT GKG taxonomy (https://blog.gdeltproject.org/the-gdelt-vision-thematic-categories/).
# Keywords: case-insensitive substring matched against the article URL, so they
# are written as URL slugs (hyphens): the original space-separated phrases
# never occur in URLs and left EEM with no matches at all.
INSTRUMENT_FILTERS: dict[str, dict[str, list[str]]] = {
    "SPY": {
        "themes": ["ECON_STOCKMARKET", "USPOL", "ECON_INFLATION", "ECON_RECESSION"],
        "keywords": ["s-p-500", "sp-500", "stock-market"],
    },
    "QQQ": {
        "themes": ["ECON_STOCKMARKET", "TECH"],
        "keywords": ["nasdaq", "tech-stocks"],
    },
    "IWM": {
        "themes": ["ECON_STOCKMARKET", "SMALLBUSINESS"],
        "keywords": ["russell-2000", "small-cap"],
    },
    "IEF": {
        "themes": ["ECON_INTEREST_RATES", "ECON_BONDS"],
        "keywords": ["10-year-treasury", "treasury-yield", "treasury-bond"],
    },
    "SHY": {
        "themes": ["ECON_INTEREST_RATES"],
        "keywords": ["2-year-treasury", "fed-funds"],
    },
    "GLD": {
        "themes": ["ECON_PRICES", "ECON_INFLATION"],
        "keywords": ["gold-price", "gold-market"],
    },
    "USO": {
        "themes": ["ECON_PETROLEUM", "ENV_OIL"],
        "keywords": ["crude-oil", "oil-price", "opec"],
    },
    "UUP": {
        "themes": ["ECON_CURRENCY"],
        "keywords": ["dollar-index", "dxy", "us-dollar"],
    },
    "EEM": {
        "themes": ["ECON_EMERGINGMARKET"],
        "keywords": ["emerging-market", "emerging-econom"],
    },
    "BTC": {
        "themes": ["TECH_CRYPTOCURRENCY", "ECON_BITCOIN"],
        "keywords": ["bitcoin", "cryptocurrency"],
    },
    # VIX is regime-only, not tradeable; included for sentiment regime detection.
    "VIX": {
        "themes": ["CRISISLEX_CRISISLEXREC", "ECON_STOCKMARKET"],
        "keywords": ["volatility", "vix"],
    },
}


def _client():
    """Lazy import so callers without google-cloud-bigquery still pass type-checks."""
    project = os.environ.get("GCP_PROJECT_ID")
    if not project:
        raise RuntimeError(
            "GCP_PROJECT_ID not set; GDELT BigQuery fetcher requires a GCP project."
        )
    from google.cloud import bigquery

    return bigquery.Client(project=project)


def _build_query(start: date, end: date, instruments: Iterable[str]) -> str:
    """Single batched query → per-day per-instrument aggregates.

    Each instrument contributes two columns to the SELECT:
        n_<sym>     — count of GKG records matching the instrument's filters
        tone_<sym>  — mean V2Tone first component (avgTone), conditional on match
    """
    select_parts: list[str] = []
    for sym in instruments:
        filters = INSTRUMENT_FILTERS.get(sym)
        if not filters:
            continue
        clauses: list[str] = []
        for theme in filters["themes"]:
            clauses.append(f"V2Themes LIKE '%{theme}%'")
        for kw in filters.get("keywords", []):
            clauses.append(f"LOWER(DocumentIdentifier) LIKE '%{kw.lower()}%'")
        if not clauses:
            continue
        match = "(" + " OR ".join(clauses) + ")"
        select_parts.append(f"COUNTIF({match}) AS n_{sym}")
        select_parts.append(
            f"AVG(IF({match}, SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64), NULL)) "
            f"AS tone_{sym}"
        )
    if not select_parts:
        raise ValueError("no instruments match GDELT filter table")

    select_sql = ",\n  ".join(select_parts)
    return f"""
SELECT
  DATE(_PARTITIONTIME) AS day,
  {select_sql}
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{start.isoformat()}')
  AND _PARTITIONTIME <= TIMESTAMP('{end.isoformat()}')
GROUP BY day
ORDER BY day
"""


def fetch_panel(
    start: date,
    end: date,
    instruments: Iterable[str],
    max_bytes: int = DEFAULT_MAX_BYTES,
    dry_run_first: bool = True,
) -> pd.DataFrame:
    """Fetch a per-day per-instrument GDELT aggregate panel.

    Returns DataFrame with columns:
        day, n_<sym>, tone_<sym>  for each requested instrument.

    Cached locally by (start, end, instruments); first call hits BigQuery,
    subsequent calls hit the local JSON cache. Set `dry_run_first=True` to
    log the byte estimate before the real query runs.
    """
    if start < GDELT_V2_START:
        raise ValueError(
            f"GDELT 2.0 starts {GDELT_V2_START}; requested start={start}. "
            "Pre-2015 needs the V1 schema (different table)."
        )
    instruments_tuple = tuple(sorted(instruments))
    if not instruments_tuple:
        raise ValueError("instruments must be non-empty")

    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "instruments": list(instruments_tuple),
    }
    cached = cache_read(NAMESPACE, params)
    if cached is not None:
        log.info(
            "gdelt cache hit start=%s end=%s n_instruments=%d",
            start, end, len(instruments_tuple),
        )
        return pd.DataFrame(cached["payload"]["rows"])

    log.info(
        "gdelt fetch start=%s end=%s instruments=%s",
        start, end, list(instruments_tuple),
    )
    backend = os.environ.get("PACT_GDELT_BACKEND") or ("bigquery" if os.environ.get("GCP_PROJECT_ID") else "files")
    if backend == "files":
        from data.fetchers.gdelt_files import fetch_panel_files

        return fetch_panel_files(start, end, instruments_tuple)
    client = _client()
    sql = _build_query(start, end, instruments_tuple)

    from google.cloud import bigquery

    if dry_run_first:
        dry = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
        )
        gb_estimate = dry.total_bytes_processed / 1e9
        log.info("gdelt dry-run estimate: %.2f GB scan", gb_estimate)
        if dry.total_bytes_processed > max_bytes:
            raise RuntimeError(
                f"gdelt query estimated {gb_estimate:.1f} GB > max_bytes "
                f"({max_bytes / 1e9:.1f} GB). Reduce window or raise max_bytes."
            )

    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes),
    )
    df = job.to_dataframe()
    df["day"] = pd.to_datetime(df["day"])
    log.info(
        "gdelt fetched rows=%d bytes_billed=%.2f GB",
        len(df), job.total_bytes_billed / 1e9,
    )

    cache_write(
        NAMESPACE,
        params,
        {"rows": df.to_dict(orient="records")},
    )
    return df
