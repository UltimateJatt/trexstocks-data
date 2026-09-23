"""Company fundamentals (size, sector, P/E, growth, analysts, next earnings date).

Yahoo only allows one company per request here, so each daily prep run refreshes
about a third of the list (oldest first). Every stock gets refreshed every few days.
"""
import time
from datetime import datetime, timezone

import yfinance as yf

from .util import load_state, save_state, log, now_et

STATE_FILE = "fundamentals.json"
PER_RUN = 1500
FIELDS = {
    "marketCap": "marketCap", "sharesOutstanding": "sharesOutstanding",
    "sector": "sector", "industry": "industry", "shortName": "name",
    "forwardPE": "forwardPE", "trailingPE": "trailingPE",
    "revenueGrowth": "revenueGrowth", "earningsGrowth": "earningsGrowth",
    "profitMargins": "profitMargins", "returnOnEquity": "returnOnEquity",
    "debtToEquity": "debtToEquity", "numberOfAnalystOpinions": "analysts",
    "currency": "currency", "quoteType": "quoteType",
    # Dividends (for the Steady / Balanced risk levels). Yahoo changed the format of
    # "dividendYield" in 2025, so we store the dollar rate and work out the yield ourselves.
    "dividendRate": "dividendRate",
    "trailingAnnualDividendYield": "trailingDividendYield", "payoutRatio": "payoutRatio",
}


def _next_earnings(info):
    now = datetime.now(timezone.utc).timestamp()
    ts = [info.get(k) for k in ("earningsTimestampStart", "earningsTimestamp",
                                "earningsTimestampEnd")]
    future = [t for t in ts if isinstance(t, (int, float)) and t >= now - 86400]
    if not future:
        return None
    return datetime.fromtimestamp(min(future), timezone.utc).date().isoformat()


def _fetch(sym):
    info = yf.Ticker(sym).info or {}
    rec = {out: info.get(k) for k, out in FIELDS.items()}
    rec["nextEarnings"] = _next_earnings(info)
    rec["fetched"] = now_et().isoformat()
    return rec


def refresh(symbols, per_run=PER_RUN, priority=None):
    data = load_state(STATE_FILE, {})
    missing = [s for s in symbols if s not in data]
    stale = sorted((s for s in symbols if s in data),
                   key=lambda s: data[s].get("fetched") or "")
    # New companies first (index members before the wide US list), capped per run so
    # the first big fill-in is spread over a few mornings instead of timing out.
    missing.sort(key=lambda s: (s not in (priority or set()), s))
    todo = (missing + stale)[:per_run]
    ok = fail = 0
    for i, s in enumerate(todo):
        try:
            rec = _fetch(s)
            if rec.get("marketCap") or rec.get("sector"):
                data[s] = rec
                ok += 1
            else:
                # remember the attempt so it waits its turn instead of blocking new stocks
                data[s] = {"fetched": now_et().isoformat(), "empty": True}
                fail += 1
        except Exception as e:
            fail += 1
            if "Rate" in type(e).__name__ or "429" in str(e):
                log("rate limited by Yahoo; stopping fundamentals early")
                break
        if i % 50 == 49:
            log(f"fundamentals {i + 1}/{len(todo)}")
            save_state(STATE_FILE, data)  # save progress in case the run dies
        time.sleep(0.4)
    # drop companies no longer in any index
    keep = set(symbols)
    data = {k: v for k, v in data.items() if k in keep}
    save_state(STATE_FILE, data)
    log(f"fundamentals: refreshed {ok}, failed {fail}, total {len(data)}")
    return data


def load():
    return load_state(STATE_FILE, {})
