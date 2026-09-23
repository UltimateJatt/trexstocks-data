"""Price-based measurements computed from one year of daily prices."""
import numpy as np


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _ret(c, a, b=None):
    """Return from `a` trading days ago to `b` days ago (b=None means today)."""
    if len(c) <= a:
        return None
    end = c.iloc[-1] if b is None else c.iloc[-1 - b]
    return float(end / c.iloc[-1 - a] - 1)


def compute(df):
    c, v = df["Close"].astype(float), df["Volume"].astype(float)
    n = len(c)
    if n < 60:
        return None
    close = float(c.iloc[-1])
    sma50 = float(c.tail(50).mean())
    sma200 = float(c.tail(200).mean()) if n >= 200 else None
    sma50_prev = float(c.iloc[-70:-20].mean()) if n >= 70 else None
    daily = c.pct_change().dropna()
    avg_vol50 = float(v.tail(50).mean())
    return {
        "bars": n,
        "close": close,
        "sma50": sma50,
        "sma200": sma200,
        "dist50": close / sma50 - 1,
        "dist200": (close / sma200 - 1) if sma200 else None,
        "slope50": (sma50 / sma50_prev - 1) if sma50_prev else None,
        "ret5": _ret(c, 5),
        "ret21": _ret(c, 21),
        "ret63": _ret(c, 63),
        "ret126_21": _ret(c, 126, 21),
        "rsi14": rsi(c),
        "volTrend": float(v.tail(10).mean() / avg_vol50) if avg_vol50 else None,
        "avgVol50": avg_vol50,
        "avgDollarVol50": float((c.tail(50) * v.tail(50)).mean()),
        "volatility60": float(daily.tail(60).std() * np.sqrt(252)),
        "high52": float(df["High"].astype(float).tail(252).max()),
        "low52": float(df["Low"].astype(float).tail(252).min()),
        "fromHigh": close / float(df["High"].astype(float).tail(252).max()) - 1,
        "asOf": df.index[-1].date().isoformat(),
    }


def compute_all(hist):
    out = {}
    for s, df in hist.items():
        try:
            t = compute(df)
            if t:
                out[s] = t
        except Exception:
            pass
    return out
