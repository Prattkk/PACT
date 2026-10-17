"""GDELT daily panel from the free GKG daily files (no BigQuery needed).

Source: https://data.gdeltproject.org/gkg/YYYYMMDD.gkg.csv.zip — one file per
UTC day, published the following morning. Columns: DATE, NUMARTS, COUNTS,
THEMES, LOCATIONS, PERSONS, ORGANIZATIONS, TONE, CAMEOEVENTIDS, SOURCES,
SOURCEURLS. Each row groups the articles that share identical extracted
metadata; NUMARTS is how many articles that is.

The filters are the same as the BigQuery fetcher (`INSTRUMENT_FILTERS`):
a row matches an instrument if THEMES contains any of its theme codes or
SOURCEURLS contains any of its URL keywords. Per day and instrument:
    n_<sym>    = number of matching articles (sum of NUMARTS)
    tone_<sym> = article-weighted mean of the average-tone field
which is the article-level equivalent of the BigQuery COUNTIF / AVG.

Per-day aggregates are cached under data/cache/gdelt_days/ so a long
download can resume; the raw zip is discarded after parsing.
"""

from __future__ import annotations

import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from pact_logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://data.gdeltproject.org/gkg/{d}.gkg.csv.zip"
DAY_CACHE = Path(__file__).resolve().parent.parent / "cache" / "gdelt_days"
DAY_CACHE.mkdir(parents=True, exist_ok=True)
FILTER_VERSION = 2  # bump if INSTRUMENT_FILTERS or the aggregation changes


def aggregate_rows(lines: Iterable[str], filters: dict[str, dict[str, list[str]]]) -> dict[str, float]:
    """Aggregate tab-separated GKG 1.0 rows into n_<sym> / tone_<sym>."""
    themes = {s: tuple(f["themes"]) for s, f in filters.items()}
    kws = {s: tuple(k.lower() for k in f.get("keywords", [])) for s, f in filters.items()}
    n = {s: 0 for s in filters}
    tone_sum = {s: 0.0 for s in filters}
    for line in lines:
        if line.startswith("DATE\t"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 11:
            continue
        try:
            arts = int(parts[1] or 0)
            tone = float(parts[7].split(",")[0]) if parts[7] else None
        except ValueError:
            continue
        if arts <= 0:
            continue
        th, urls = parts[3], parts[10].lower()
        for s in filters:
            if any(t in th for t in themes[s]) or any(k in urls for k in kws[s]):
                n[s] += arts
                if tone is not None:
                    tone_sum[s] += arts * tone
    out: dict[str, float] = {}
    for s in filters:
        out[f"n_{s}"] = float(n[s])
        out[f"tone_{s}"] = tone_sum[s] / n[s] if n[s] else float("nan")
    return out


def _day_path(d: date) -> Path:
    return DAY_CACHE / f"{d:%Y%m%d}.v{FILTER_VERSION}.json"


def _fetch_day(d: date, filters, session: requests.Session) -> dict | None:
    p = _day_path(d)
    if p.exists():
        return json.loads(p.read_text())
    url = BASE_URL.format(d=f"{d:%Y%m%d}")
    for attempt in range(4):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 404:
                log.warning("gdelt file missing for %s", d)
                rec = {"day": d.isoformat(), "missing": True}
                p.write_text(json.dumps(rec))
                return rec
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 3:
                log.error("gdelt download failed %s: %s", d, e)
                return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as fh:
            agg = aggregate_rows(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), filters)
    rec = {"day": d.isoformat(), **agg}
    p.write_text(json.dumps(rec))
    return rec


def fetch_panel_files(start: date, end: date, instruments: Iterable[str], workers: int = 4) -> pd.DataFrame:
    from data.fetchers.gdelt import INSTRUMENT_FILTERS

    filters = {s: INSTRUMENT_FILTERS[s] for s in sorted(instruments) if s in INSTRUMENT_FILTERS}
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if not _day_path(d).exists()]
    if todo:
        log.info("gdelt files: downloading %d of %d days", len(todo), len(days))
    # every day is aggregated for the full filter table so the cache is reusable
    with requests.Session() as sess, ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda d: _fetch_day(d, INSTRUMENT_FILTERS, sess), todo))
    rows = []
    for d in days:
        p = _day_path(d)
        if not p.exists():
            raise RuntimeError(f"gdelt day {d} could not be fetched; rerun to resume")
        rec = json.loads(p.read_text())
        if rec.get("missing"):
            continue
        rows.append({"day": rec["day"], **{k: rec.get(k) for c in filters for k in (f"n_{c}", f"tone_{c}")}})
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["day"])
    return df
