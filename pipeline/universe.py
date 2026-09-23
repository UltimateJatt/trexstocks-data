"""Index member lists. Refreshed weekly from public sources.

If a source fails or returns a suspiciously short list, the previous saved list is
kept, so a bad scrape can never wipe out the site.
"""
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

from .util import load_state, save_state, log, now_et

UA = {"User-Agent": "Mozilla/5.0 (TrexStocks data bot; https://trexstocks.com)"}
STATE_FILE = "constituents.json"
REFRESH_DAYS = 6

SOURCES = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 480, 520),
    "ndx": ("https://en.wikipedia.org/wiki/Nasdaq-100", 95, 110),
    "tsx": ("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", 180, 260),
}
SYMBOL_COLS = ("Symbol", "Ticker", "Ticker symbol")
NAME_COLS = ("Security", "Company", "Name")


def _wiki_table(url, lo, hi):
    html = requests.get(url, headers=UA, timeout=30).text
    for t in pd.read_html(StringIO(html)):
        cols = [str(c[-1] if isinstance(c, tuple) else c).strip() for c in t.columns]
        t.columns = cols
        sym = next((c for c in SYMBOL_COLS if c in cols), None)
        if sym and lo <= len(t) <= hi:
            name = next((c for c in NAME_COLS if c in cols), None)
            return [(str(r[sym]).strip(), str(r[name]).strip() if name else "")
                    for _, r in t.iterrows()]
    raise ValueError(f"no suitable table at {url}")


def _to_yahoo_us(s):
    return s.replace(".", "-").upper()


def _to_yahoo_tsx(s):
    s = s.upper().replace("TSX:", "").strip()
    if s.endswith(".TO"):
        s = s[:-3]
    return s.replace(".", "-") + ".TO"


def _nasdaq_midcaps(exclude):
    """All NASDAQ-listed stocks between $2B and $20B, minus index members."""
    url = ("https://api.nasdaq.com/api/screener/stocks"
           "?tableonly=true&limit=10000&exchange=nasdaq&download=true")
    headers = {**UA, "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
               "Accept": "application/json"}
    rows = requests.get(url, headers=headers, timeout=60).json()["data"]["rows"]
    out = []
    for r in rows:
        sym = (r.get("symbol") or "").strip()
        if not sym or any(ch in sym for ch in "^/ ") or sym in exclude:
            continue
        try:
            cap = float(r.get("marketCap") or 0)
        except ValueError:
            continue
        if 2e9 <= cap <= 20e9:
            out.append((sym.replace(".", "-"), (r.get("name") or "").strip()))
    return out


def refresh(force=False):
    old = load_state(STATE_FILE, {})
    updated = old.get("updated")
    if updated and not force:
        age = now_et() - datetime.fromisoformat(updated)
        if age < timedelta(days=REFRESH_DAYS):
            log(f"constituents fresh ({age.days}d old), skipping")
            return old

    new = dict(old)
    ok = 0
    for key, (url, lo, hi) in SOURCES.items():
        try:
            rows = _wiki_table(url, lo, hi)
            conv = _to_yahoo_tsx if key == "tsx" else _to_yahoo_us
            new[key] = {conv(s): n for s, n in rows}
            log(f"{key}: {len(new[key])} members")
            ok += 1
        except Exception as e:  # keep the old list
            log(f"WARNING {key} refresh failed, keeping old list: {e}")

    try:
        exclude = set(new.get("sp500", {})) | set(new.get("ndx", {}))
        mids = _nasdaq_midcaps(exclude)
        if len(mids) >= 100:
            new["nasdaqMid"] = dict(mids)
            log(f"nasdaqMid: {len(mids)} stocks")
        else:
            log(f"WARNING nasdaqMid only {len(mids)} rows, keeping old list")
    except Exception as e:
        log(f"WARNING nasdaqMid refresh failed, keeping old list: {e}")

    if ok == len(SOURCES):
        new["updated"] = now_et().isoformat()
        new.pop("seed", None)
    else:
        log("WARNING some index lists failed; will try again next run")
    save_state(STATE_FILE, new)
    return new


def load():
    c = load_state(STATE_FILE)
    if not c:
        raise RuntimeError("constituents.json missing; run the Daily prep workflow first")
    return c


def groups(c):
    """Which stocks belong to each tab.

    movers: what the Top Movers tab shows.
    pool:   what the daily picks can choose from.
    """
    sp, ndx, tsx, mid = (set(c.get(k, {})) for k in ("sp500", "ndx", "tsx", "nasdaqMid"))
    return {
        "sp500": {"movers": sp, "pool": sp},
        "nasdaq": {"movers": ndx, "pool": ndx | mid},
        "tsx": {"movers": tsx, "pool": tsx},
    }


def names(c):
    out = {}
    for k in ("nasdaqMid", "tsx", "ndx", "sp500"):
        out.update(c.get(k, {}))
    return out


def all_symbols(c, include_mid=True):
    keys = ["sp500", "ndx", "tsx"] + (["nasdaqMid"] if include_mid else [])
    s = set()
    for k in keys:
        s |= set(c.get(k, {}))
    return sorted(s)
