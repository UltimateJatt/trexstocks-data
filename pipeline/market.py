"""Price data from Yahoo Finance (via the yfinance library), fetched in bulk."""
import time

import pandas as pd
import yfinance as yf

from .util import log

CHUNK = 150


def _yf_download(symbols, period, auto_adjust):
    return yf.download(symbols, period=period, interval="1d", group_by="ticker",
                       auto_adjust=auto_adjust, threads=True, progress=False,
                       timeout=30)


def _split(df, symbols):
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for s in symbols:
            if s in df.columns.get_level_values(0):
                d = df[s].dropna(subset=["Close"])
                if len(d):
                    out[s] = d
    elif len(symbols) == 1:
        d = df.dropna(subset=["Close"])
        if len(d):
            out[symbols[0]] = d
    return out


def history(symbols, period="1y", auto_adjust=True):
    """Return {symbol: DataFrame[Open, High, Low, Close, Volume]} for every symbol found."""
    symbols = list(dict.fromkeys(symbols))
    result = {}
    for attempt in (1, 2):
        todo = [s for s in symbols if s not in result]
        if not todo:
            break
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            try:
                result.update(_split(_yf_download(chunk, period, auto_adjust), chunk))
            except Exception as e:
                log(f"download chunk failed ({e}); will retry")
            time.sleep(1.5)
        if attempt == 1 and len(result) < len(symbols):
            log(f"retrying {len(symbols) - len(result)} missing symbols")
            time.sleep(5)
    missing = [s for s in symbols if s not in result]
    log(f"history {period}: got {len(result)}/{len(symbols)}"
        + (f" (missing e.g. {missing[:8]})" if missing else ""))
    return result


def quotes(symbols):
    """Latest price, change vs previous close, and today's volume for each symbol."""
    data = history(symbols, period="5d", auto_adjust=False)
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
            "volume": float(last.get("Volume") or 0),
            "high": float(last.get("High") or price),
            "low": float(last.get("Low") or price),
            "date": d.index[-1].date().isoformat(),
        }
    return out
