"""NOAA Climate Data Online (CDO) v2 — heating-degree days + hurricane state.

Used by `fundamentals_carry` agent for the USO branch (oil demand from
heating; supply disruption from active Atlantic hurricanes).

NOAA CDO API requires a free API key (`NOAA_TOKEN` in `.env`). Get one at
https://www.ncdc.noaa.gov/cdo-web/token. The fetcher is small-by-design —
we only pull a national-aggregate HDD series and a count of named storms,
both at monthly granularity. Cached locally.

Fallback: when no token is present, returns zero-valued series so the agent
schema stays valid (factor values just stay at 0).
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from data.cache.cache import read as cache_read
from data.cache.cache import write as cache_write
from pact_logging import get_logger

log = get_logger(__name__)

NAMESPACE = "noaa"
CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"


def _token() -> str | None:
    return os.environ.get("NOAA_TOKEN")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _get(url: str, params: dict, headers: dict) -> dict:
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def heating_degree_days_us(start: date, end: date) -> pd.Series:
    """National-aggregate monthly heating-degree days (HDD) for the U.S.

    Higher HDD → more demand for natural-gas heating (oil less so, but
    distillate inventory tightens via correlated heating-fuel demand).
    Returns Series indexed by date with HDD values, or empty if NOAA_TOKEN
    is missing.
    """
    if os.environ.get("PACT_NOAA_SOURCE", "cpc").lower() == "cpc":
        return cpc_conus_hdd_monthly(start, end)
    params = {"start": start.isoformat(), "end": end.isoformat(), "metric": "hdd"}
    cached = cache_read(NAMESPACE, params)
    if cached is not None:
        return _series_from_payload(cached["payload"])

    token = _token()
    if not token:
        log.warning("noaa: NOAA_TOKEN not set; returning empty HDD series")
        return pd.Series(dtype=float)

    log.info("noaa fetch hdd start=%s end=%s", start, end)
    rows: list[tuple[str, float]] = []
    offset = 1
    while True:
        payload = _get(
            f"{CDO_BASE}/data",
            params={
                "datasetid": "GSOM",  # Global Summary of the Month
                "datatypeid": "HTDD",  # heating-degree days
                "locationid": "FIPS:US",
                "startdate": start.isoformat(),
                "enddate": end.isoformat(),
                "limit": 1000,
                "offset": offset,
                "units": "standard",
            },
            headers={"token": token},
        )
        results = payload.get("results", [])
        if not results:
            break
        rows.extend((r["date"][:10], float(r["value"])) for r in results)
        if len(results) < 1000:
            break
        offset += 1000

    cache_write(NAMESPACE, params, {"rows": rows})
    log.info("noaa fetched hdd rows=%d", len(rows))
    return _series_from_payload({"rows": rows})


CPC_URL = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{y}/Population.Heating.txt"


def cpc_conus_hdd_monthly(start: date, end: date) -> pd.Series:
    """Monthly population-weighted CONUS heating degree days from NOAA CPC.

    Source: CPC daily degree-day files (no token). Daily CONUS values are
    summed within each calendar month; the series is dated at month start,
    matching the GSOM convention the agent's publication-lag rule expects.
    """
    params = {"start": start.isoformat(), "end": end.isoformat(), "metric": "cpc_conus_hdd_monthly"}
    cached = cache_read(NAMESPACE, params)
    if cached is not None:
        return _series_from_payload(cached["payload"])
    daily: dict[pd.Timestamp, float] = {}
    for y in range(start.year, end.year + 1):
        r = requests.get(CPC_URL.format(y=y), timeout=60)
        r.raise_for_status()
        lines = [ln for ln in r.text.splitlines() if "|" in ln]
        header = next(ln for ln in lines if ln.startswith("Region|")).split("|")[1:]
        conus = next(ln for ln in lines if ln.startswith("CONUS|")).split("|")[1:]
        for d, v in zip(header, conus):
            if d.strip() and v.strip():
                daily[pd.Timestamp(d.strip())] = float(v)
    s = pd.Series(daily).sort_index()
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    monthly = s.groupby(s.index.to_period("M")).agg(["sum", "count"])
    monthly = monthly[monthly["count"] >= 28]  # complete months only
    rows = [(p.to_timestamp().date().isoformat(), float(v)) for p, v in monthly["sum"].items()]
    cache_write(NAMESPACE, params, {"rows": rows})
    log.info("noaa cpc hdd months=%d", len(rows))
    return _series_from_payload({"rows": rows})


def named_storms_active(start: date, end: date) -> pd.Series:
    """Monthly count of active Atlantic named storms (hurricane + tropical storm).

    Implementation is a thin proxy: NOAA HURDAT2 best-track is the canonical
    source but is a flat file, not REST. For now we publish a placeholder
    that flags the Atlantic hurricane season months (Jun-Nov) — replace with
    HURDAT2 parsing when available. Returns 1.0 in season-months, 0.0 else.
    """
    params = {"start": start.isoformat(), "end": end.isoformat(), "metric": "storms_proxy"}
    cached = cache_read(NAMESPACE, params)
    if cached is not None:
        return _series_from_payload(cached["payload"])

    months = pd.date_range(start=start, end=end, freq="MS")
    rows = [(m.date().isoformat(), float(6 <= m.month <= 11)) for m in months]
    cache_write(NAMESPACE, params, {"rows": rows})
    return _series_from_payload({"rows": rows})


def _series_from_payload(payload: dict) -> pd.Series:
    rows = payload.get("rows", [])
    if not rows:
        return pd.Series(dtype=float)
    idx = [pd.Timestamp(d) for d, _ in rows]
    vals = [v for _, v in rows]
    return pd.Series(vals, index=idx).sort_index()
