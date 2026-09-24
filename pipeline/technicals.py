"""Price-based measurements computed from one year of daily prices.

Returns, averages, RSI, volatility, beta and drawdown use dividend-adjusted closes
(AdjClose). Prices shown on the site (close, 52-week high/low) use the plain close.
"""
import numpy as np
import pandas as pd

RISK_MIN_OBS = 200      # daily returns needed for beta / drawdown
RISK_WINDOW = 252       # about one year of trading days


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


def _beta(daily, bench_daily):
    """(beta, correlation) vs the stock's home index over the last year of daily returns.

    Correlation matters for reading beta: when a stock's moves have little to do
    with the market's (correlation near 0), its beta is near 0 too, which says
    nothing about how much the stock swings on its own.
    """
    if bench_daily is None:
        return None, None
    both = pd.concat([daily, bench_daily], axis=1, join="inner").dropna().tail(RISK_WINDOW)
    if len(both) < RISK_MIN_OBS:
        return None, None
    var = both.iloc[:, 1].var()
    if not var:
        return None, None
    return (float(both.iloc[:, 0].cov(both.iloc[:, 1]) / var),
            float(both.iloc[:, 0].corr(both.iloc[:, 1])))


def _max_drawdown(adj):
    """Largest peak-to-trough fall over the last year, e.g. -0.18 for -18%."""
    a = adj.tail(RISK_WINDOW + 1)
    if len(a) < RISK_MIN_OBS:
        return None
    return float((a / a.cummax() - 1).min())


def compute(df, bench_daily=None):
    c = df["Close"].astype(float)
    a = (df["AdjClose"] if "AdjClose" in df else df["Close"]).astype(float)
    v = df["Volume"].astype(float)
    n = len(c)
    if n < 60:
        return None
    close = float(c.iloc[-1])
    adj = float(a.iloc[-1])
    sma50 = float(a.tail(50).mean())
    sma200 = float(a.tail(200).mean()) if n >= 200 else None
    sma50_prev = float(a.iloc[-70:-20].mean()) if n >= 70 else None
    daily = a.pct_change().dropna()
    down = daily[daily < 0].tail(60)
    avg_vol50 = float(v.tail(50).mean())
    high52 = float(df["High"].astype(float).tail(252).max())
    beta, corr = _beta(daily, bench_daily)
    return {
        "bars": n,
        "close": close,
        "sma50": sma50,
        "sma200": sma200,
        "dist50": adj / sma50 - 1,
        "dist200": (adj / sma200 - 1) if sma200 else None,
        "slope50": (sma50 / sma50_prev - 1) if sma50_prev else None,
        "ret5": _ret(a, 5),
        "ret21": _ret(a, 21),
        "ret63": _ret(a, 63),
        "ret126_21": _ret(a, 126, 21),
        "rsi14": rsi(a),
        "volTrend": float(v.tail(10).mean() / avg_vol50) if avg_vol50 else None,
        "avgVol50": avg_vol50,
        "avgDollarVol50": float((c.tail(50) * v.tail(50)).mean()),
        "volatility60": float(daily.tail(60).std() * np.sqrt(252)),
        "downsideVol60": float(np.sqrt((down ** 2).sum() / 60) * np.sqrt(252)) if len(down) else 0.0,
        "beta": beta,
        "marketCorr": corr,
        "maxDrawdown": _max_drawdown(a),
        "high52": high52,
        "low52": float(df["Low"].astype(float).tail(252).min()),
        "fromHigh": close / high52 - 1,
        "asOf": df.index[-1].date().isoformat(),
    }


def _daily(df):
    if df is None:
        return None
    a = (df["AdjClose"] if "AdjClose" in df else df["Close"]).astype(float)
    return a.pct_change().dropna()


def compute_all(hist, us_bench="^GSPC", ca_bench="^GSPTSE"):
    """Measurements for every symbol. Beta is vs the S&P 500 for US stocks and the
    TSX Composite for Canadian ones."""
    bench = {"US": _daily(hist.get(us_bench)), "CA": _daily(hist.get(ca_bench))}
    out = {}
    for s, df in hist.items():
        try:
            t = compute(df, bench["CA" if s.endswith(".TO") else "US"])
            if t:
                out[s] = t
        except Exception:
            pass
    return out
