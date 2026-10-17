"""Google Trends fetcher (spec §3.3 secondary; §4.1 narrative_event).

Uses pytrends (unofficial library) to scrape Google Trends. No API key, but
rate-limited and occasionally throttled — wrapped with tenacity retry.

Returns per-keyword weekly search-interest series (0-100). Used by
`narrative_event` for `gtrends_abnormal_level` and `gtrends_momentum`.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from data.cache.cache import read as cache_read
from data.cache.cache import write as cache_write
from pact_logging import get_logger

log = get_logger(__name__)

NAMESPACE = "gtrends"

# Per-instrument keyword set (kept short — pytrends throttles aggressively
# on >5 keywords per call).
INSTRUMENT_TO_KEYWORDS: dict[str, list[str]] = {
    "SPY": ["stock market"],
    "QQQ": ["nasdaq"],
    "IWM": ["russell 2000"],
    "IEF": ["treasury yield"],
    "SHY": ["fed rate"],
    "GLD": ["gold price"],
    "USO": ["oil price"],
    "UUP": ["dollar index"],
    "EEM": ["emerging markets"],
    "BTC": ["bitcoin price"],
    "VIX": ["VIX"],
}


def _client():
    from pytrends.request import TrendReq

    return TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=1.0)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def _fetch_one(keyword: str, start: date, end: date) -> pd.Series:
    pt = _client()
    timeframe = f"{start.isoformat()} {end.isoformat()}"
    pt.build_payload([keyword], cat=0, timeframe=timeframe, geo="", gprop="")
    df = pt.interest_over_time()
    if df.empty:
        return pd.Series(dtype=float)
    return df[keyword].astype(float)


def search_interest(keyword: str, start: date, end: date) -> pd.Series:
    """Weekly Google Trends interest (0-100) for a keyword."""
    params = {"keyword": keyword, "start": start.isoformat(), "end": end.isoformat()}
    cached = cache_read(NAMESPACE, params)
    if cached is not None:
        rows = cached["payload"].get("rows", [])
        if not rows:
            return pd.Series(dtype=float)
        return pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()

    log.info("gtrends fetch keyword=%s start=%s end=%s", keyword, start, end)
    try:
        s = _fetch_one(keyword, start, end)
    except Exception as e:
        # do not cache failures: a throttled call must be retried on the next run
        log.warning("gtrends fetch failed keyword=%s: %s", keyword, type(e).__name__)
        return pd.Series(dtype=float)

    rows = [(d.isoformat(), float(v)) for d, v in s.items()]
    cache_write(NAMESPACE, params, {"rows": rows})
    log.info("gtrends fetched keyword=%s rows=%d", keyword, len(rows))
    return pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()


def search_interest_for_instrument(symbol: str, start: date, end: date) -> pd.Series:
    """Mean search-interest across all keywords mapped to an instrument."""
    keywords = INSTRUMENT_TO_KEYWORDS.get(symbol.upper(), [])
    if not keywords:
        return pd.Series(dtype=float)
    series = [search_interest(k, start, end) for k in keywords]
    series = [s for s in series if not s.empty]
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).mean(axis=1).dropna()
