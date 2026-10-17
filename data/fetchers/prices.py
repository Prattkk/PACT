"""Daily OHLCV prices via yfinance.

BTC splice (spot pre-2024-01-11, IBIT post) is handled here so downstream
code sees a single 'BTC' series. The splice is done on *returns*, not price
levels: IBIT is rescaled so that its close on the splice date equals the
BTC-USD close on the same date. (An earlier version concatenated raw levels,
which produced a spurious -99.9% BTC return on 2024-01-11 and corrupted every
BTC momentum/volatility factor computed across that date.)

Panels that mix crypto with exchange-traded instruments are aligned to the
exchange trading calendar: crypto weekend/holiday moves accrue into the next
exchange session instead of creating extra zero-return rows for the ETFs.
(An earlier version kept the union of calendars, which inserted ~104 weekend
rows per year, distorted every rolling window measured in rows, and biased
annualised Sharpe ratios toward zero.)
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from data.cache.cache import read as cache_read
from data.cache.cache import write as cache_write
from data.universe import BTC_SPLICE_DATE
from pact_logging import get_logger

log = get_logger(__name__)

NAMESPACE = "prices"
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")
CRYPTO_SYMBOLS = frozenset({"BTC"})
RISK_FREE_TICKER = "^IRX"  # 13-week US T-bill, annualised discount yield in percent


def _cached_yf_download(symbol: str, start: date, end: date) -> pd.DataFrame:
    params = {"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()}
    hit = cache_read(NAMESPACE, params)
    if hit is not None:
        log.debug("prices cache hit symbol=%s start=%s end=%s", symbol, start, end)
        payload = hit["payload"]
        df = pd.DataFrame(payload["data"])
        df.index = pd.to_datetime(payload["index"])
        return df
    log.debug("prices cache miss symbol=%s start=%s end=%s", symbol, start, end)
    df = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cache_write(
        NAMESPACE,
        params,
        {
            "index": [d.isoformat() for d in df.index],
            "data": {c: [float(v) if pd.notna(v) else None for v in df[c].values] for c in df.columns},
        },
    )
    return df


def fetch_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Single-symbol daily OHLCV. Returns a DataFrame indexed by date."""
    if symbol.upper() == "BTC":
        return _fetch_btc_spliced(start, end)
    df = _cached_yf_download(symbol, start, end)
    if df.empty:
        log.warning("prices empty symbol=%s start=%s end=%s", symbol, start, end)
    return df


def _fetch_btc_spliced(start: date, end: date) -> pd.DataFrame:
    """BTC-USD spot before the splice date, rescaled IBIT from the splice date on.

    The rescaling factor is BTC-USD close / IBIT close on the splice date, so
    the spliced series has a continuous level and the splice-day return is the
    BTC-USD spot return. Caveat: BTC-USD closes at 00:00 UTC while IBIT closes
    at 16:00 ET; the timing mismatch affects only the splice day.
    """
    splice = BTC_SPLICE_DATE
    parts: list[pd.DataFrame] = []
    if start < splice:
        parts.append(_cached_yf_download("BTC-USD", start, min(end, splice)))
    if end > splice:
        ibit = _cached_yf_download("IBIT", max(start, splice), end)
        if not ibit.empty:
            anchor = _cached_yf_download("BTC-USD", splice, splice + timedelta(days=1))
            splice_ts = pd.Timestamp(splice)
            if anchor.empty or splice_ts not in ibit.index:
                raise RuntimeError(
                    "BTC splice: missing BTC-USD or IBIT close on the splice date; "
                    "cannot rescale IBIT safely"
                )
            ibit = ibit.copy()
            for col in PRICE_COLUMNS:
                if col in ibit.columns and col in anchor.columns:
                    scale = float(anchor[col].iloc[0]) / float(ibit.loc[splice_ts, col])
                    ibit[col] = ibit[col] * scale
            parts.append(ibit)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def align_to_exchange_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where at least one exchange-traded column has a price.

    Crypto columns are forward-filled first, so a Monday row carries the
    Monday crypto close and the Monday crypto return includes the weekend.
    Panels with only crypto columns are returned unchanged.
    """
    exchange_cols = [c for c in panel.columns if c.upper() not in CRYPTO_SYMBOLS]
    if not exchange_cols or len(exchange_cols) == len(panel.columns):
        return panel
    trading_days = panel[exchange_cols].dropna(how="all").index
    return panel.ffill().loc[trading_days]


def fetch_panel(symbols: list[str], start: date, end: date, field: str = "Adj Close") -> pd.DataFrame:
    """Wide panel of one OHLCV field across symbols, on the exchange calendar."""
    log.info("prices panel fetch symbols=%d field=%s start=%s end=%s", len(symbols), field, start, end)
    series = {}
    for s in symbols:
        df = fetch_ohlcv(s, start, end)
        if df.empty or field not in df.columns:
            log.warning("prices skip symbol=%s (empty or missing field=%s)", s, field)
            continue
        series[s.upper()] = df[field]
    panel = align_to_exchange_calendar(pd.DataFrame(series).sort_index())
    log.info("prices panel built rows=%d cols=%d", len(panel), len(panel.columns))
    return panel


def fetch_risk_free_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free rate on `index` from the 13-week T-bill yield (^IRX).

    ^IRX is quoted as an annualised percentage; the daily rate is
    (1 + y/100) ** (1/252) - 1, forward-filled onto `index`. Falls back to
    zero (with a warning) when the series cannot be fetched.
    """
    if len(index) == 0:
        return pd.Series(dtype=float)
    start = index[0].date() - timedelta(days=10)
    end = index[-1].date() + timedelta(days=1)
    try:
        df = _cached_yf_download(RISK_FREE_TICKER, start, end)
        y = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    except Exception as e:  # pragma: no cover - network path
        log.warning("risk-free fetch failed (%s); using rf=0", e)
        return pd.Series(0.0, index=index)
    if y.empty:
        log.warning("risk-free series empty; using rf=0")
        return pd.Series(0.0, index=index)
    daily = (1.0 + y.astype(float) / 100.0) ** (1.0 / 252.0) - 1.0
    daily.index = pd.to_datetime(daily.index)
    return daily.reindex(daily.index.union(index)).ffill().reindex(index).fillna(0.0)
