"""Price data from Yahoo Finance's chart API, one request per stock, several at a time.

Why not yfinance's bulk download: Yahoo sometimes leaves the most recent day's
candle empty (or missing) for hours after the close. The bulk download then
silently drops that day, so prices come out one day old. Each chart response also
carries a "meta" block with the true latest price and its timestamp, so we use it
to fill in or correct the latest day. This is the same data Yahoo's own quote
page shows.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from curl_cffi import requests as creq

from .util import log

WORKERS = 8
HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = creq.Session(impersonate="chrome")
    return _local.s


def _fetch_chart(symbol, rng):
    """Raw JSON from Yahoo's chart API, or None."""
    for attempt in range(3):
        host = HOSTS[attempt % 2]
        url = f"https://{host}/v8/finance/chart/{symbol}"
        try:
            r = _session().get(url, params={"range": rng, "interval": "1d",
                                            "includePrePost": "false"}, timeout=20)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1)
    return None


def _parse(j):
    """Turn a chart response into a daily DataFrame, patching the latest day from meta."""
    try:
        res = j["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        return None
    meta = res.get("meta") or {}
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    ts = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    df = pd.DataFrame({
        "Open": q.get("open") or [None] * len(ts),
        "High": q.get("high") or [None] * len(ts),
        "Low": q.get("low") or [None] * len(ts),
        "Close": q.get("close") or [None] * len(ts),
        "Volume": q.get("volume") or [None] * len(ts),
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz)
        .normalize().tz_localize(None), dtype="float64")
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])

    # Fill in / correct the latest day from the live quote block
    price, t = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if price and t:
        day = pd.Timestamp(t, unit="s", tz="UTC").tz_convert(tz).normalize().tz_localize(None)
        if len(df) == 0 or day > df.index[-1]:
            row = pd.DataFrame({
                "Open": [price],
                "High": [meta.get("regularMarketDayHigh") or price],
                "Low": [meta.get("regularMarketDayLow") or price],
                "Close": [price],
                "Volume": [meta.get("regularMarketVolume")],
            }, index=pd.DatetimeIndex([day]), dtype="float64")
            df = pd.concat([df, row])
        elif day == df.index[-1]:
            df.loc[day, "Close"] = price
            if meta.get("regularMarketVolume"):
                df.loc[day, "Volume"] = meta["regularMarketVolume"]
    return df if len(df) else None


def _get(symbol, rng):
    j = _fetch_chart(symbol, rng)
    return _parse(j) if j else None


def history(symbols, period="1y", auto_adjust=False):
    """Return {symbol: DataFrame[Open, High, Low, Close, Volume]} for every symbol found.

    Close prices are split-adjusted but not dividend-adjusted (what Yahoo displays).
    """
    symbols = list(dict.fromkeys(symbols))
    result = {}
    for attempt in (1, 2):
        todo = [s for s in symbols if s not in result]
        if not todo:
            break
        with ThreadPoolExecutor(WORKERS) as ex:
            for s, df in zip(todo, ex.map(lambda s: _get(s, period), todo)):
                if df is not None:
                    result[s] = df
        if attempt == 1 and len(result) < len(symbols):
            log(f"retrying {len(symbols) - len(result)} missing symbols")
            time.sleep(10)
    missing = [s for s in symbols if s not in result]
    latest = pd.Series([d.index[-1].date().isoformat() for d in result.values()]).value_counts()
    log(f"history {period}: got {len(result)}/{len(symbols)}; latest dates "
        f"{ {k: int(v) for k, v in latest.head(3).items()} }" + (f" (missing e.g. {missing[:8]})" if missing else ""))
    return result


def quotes(symbols):
    """Latest price, change vs previous close, and today's volume for each symbol."""
    data = history(symbols, period="5d")
    out = {}
    for s, d in data.items():
        if len(d) < 2:
            continue
        last, prev = d.iloc[-1], d.iloc[-2]
        price, pc = float(last["Close"]), float(prev["Close"])
        if not pc:
            continue
        out[s] = {
            "price": price,
            "prevClose": pc,
            "change": price - pc,
            "changePercent": (price - pc) / pc * 100,
            "volume": float(last["Volume"]) if pd.notna(last["Volume"]) else 0.0,
            "high": float(last["High"]) if pd.notna(last["High"]) else price,
            "low": float(last["Low"]) if pd.notna(last["Low"]) else price,
            "date": d.index[-1].date().isoformat(),
        }
    return out
