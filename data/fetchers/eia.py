"""EIA Open Data v2 fetcher (spec §4.5 oil fundamentals).

Free public API, key required (`EIA_API_KEY`). Used by `fundamentals_carry`
agent for the USO branch — primarily weekly U.S. crude oil ending stocks
(series `WCESTUS1`), reported every Wednesday for the prior week.

The contamination-clean pattern: EIA publishes Wednesday around 14:30 ET.
For a backtest decision at date D, only use observations where the EIA
release date <= D.
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

NAMESPACE = "eia"
BASE = "https://api.eia.gov/v2"


def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY not set; EIA fetcher requires a free API key.")
    return key


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=10))
def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=120)  # EIA v2 responses can take ~60s
    r.raise_for_status()
    return r.json()


def fetch_series(
    route: str,
    series_id: str,
    frequency: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch a single EIA series; returns DataFrame with columns [period, value].

    Args:
        route: API route, e.g. 'petroleum/stoc/wstk' for weekly stocks.
        series_id: facet value, e.g. 'WCESTUS1' for U.S. crude ex-SPR.
        frequency: 'weekly' / 'monthly' / 'daily'.
        start, end: inclusive date bounds.
    """
    params = {
        "route": route,
        "series": series_id,
        "frequency": frequency,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    cached = cache_read(NAMESPACE, params)
    if cached is not None:
        log.debug("eia cache hit series=%s start=%s end=%s", series_id, start, end)
        return pd.DataFrame(cached["payload"]["rows"])

    log.info("eia fetch series=%s start=%s end=%s", series_id, start, end)
    url = f"{BASE}/{route}/data/"
    query = {
        "api_key": _api_key(),
        "frequency": frequency,
        "data[0]": "value",
        "facets[series][]": series_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    payload = _get(url, query)
    rows = [
        {"period": r["period"], "value": float(r["value"]) if r["value"] is not None else None}
        for r in payload.get("response", {}).get("data", [])
    ]
    log.info("eia fetched series=%s rows=%d", series_id, len(rows))
    cache_write(NAMESPACE, params, {"rows": rows})
    return pd.DataFrame(rows)


# Convenience accessors used by agents.

def crude_oil_inventory_us(start: date, end: date) -> pd.DataFrame:
    """Weekly U.S. ending stocks excluding SPR of crude oil (thousand barrels).

    Series: WCESTUS1 (route petroleum/stoc/wstk). Reported every Wednesday
    for the week ending the previous Friday.
    """
    return fetch_series("petroleum/stoc/wstk", "WCESTUS1", "weekly", start, end)
