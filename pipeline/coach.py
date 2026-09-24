"""Trex Chart Coach: a plain-English reading of a stock's daily chart.

Everything here is a fixed, published rule applied to daily prices (the close of
the last completed trading day). It describes what the chart has done; it does
not forecast what it will do. Prices are the split-adjusted closes and highs/lows
people see on a chart.

The rules match the TrexStocks Product Gap Analysis, "Gap 1: Chart Coach".
"""
import numpy as np
import pandas as pd

from .technicals import rsi as _rsi

CHART_DAYS = 126          # about 6 months on the stock page
LEVEL_LOOKBACK = 120      # days searched for swing highs/lows
SWING_K = 3               # a swing high is the highest of 3 days either side


def _f(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(x) or np.isinf(x):
        return None
    return round(x, nd)


def _price(x):
    """Round a level to sensible steps: cents above $1, 3 decimals below."""
    return None if x is None else round(float(x), 2 if x >= 1 else 3)


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr, tr.ewm(alpha=1 / n, adjust=False).mean()


def _swings(df):
    """Swing highs and lows in the lookback window (the newest 3 days can't qualify)."""
    h, l = df["High"].values, df["Low"].values
    n = len(df)
    start = max(SWING_K, n - LEVEL_LOOKBACK)
    highs, lows = [], []
    for i in range(start, n - SWING_K):
        win_h = h[i - SWING_K:i + SWING_K + 1]
        win_l = l[i - SWING_K:i + SWING_K + 1]
        if h[i] == win_h.max():
            highs.append((i, float(h[i])))
        if l[i] == win_l.min():
            lows.append((i, float(l[i])))
    return highs, lows


def _zones(points, atr):
    """Group swing points within 0.5 ATR of each other into zones."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[1])
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] - cur[-1][1] <= 0.5 * atr:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    out = []
    for g in groups:
        prices = [p[1] for p in g]
        out.append({"low": min(prices) - 0.25 * atr, "high": max(prices) + 0.25 * atr,
                    "touches": len(g), "last": max(p[0] for p in g)})
    return out


def _ceiling_floor(df, close, atr):
    highs, lows = _swings(df)
    zones = _zones(highs + lows, atr)
    above = [z for z in zones if z["low"] > close]
    below = [z for z in zones if z["high"] < close]
    strong_above = [z for z in above if z["touches"] >= 2]
    strong_below = [z for z in below if z["touches"] >= 2]
    ceiling = min(strong_above, key=lambda z: z["low"]) if strong_above else None
    floor = max(strong_below, key=lambda z: z["high"]) if strong_below else None
    if ceiling is None:
        hs = [p for p in highs if p[1] > close]
        if hs:
            i, p = max(hs, key=lambda x: x[0])
            ceiling = {"low": p, "high": p, "touches": 1, "last": i}
    if floor is None:
        ls = [p for p in lows if p[1] < close]
        if ls:
            i, p = max(ls, key=lambda x: x[0])
            floor = {"low": p, "high": p, "touches": 1, "last": i}
    last_swing_low = max(lows, key=lambda x: x[0]) if lows else None
    recent_lows_below = [p for p in lows if p[1] < close]
    last_low_below = max(recent_lows_below, key=lambda x: x[0]) if recent_lows_below else None
    return ceiling, floor, last_low_below or last_swing_low


def _rsi_word(r):
    if r is None:
        return "--"
    if r < 30:
        return "Weak"
    if r < 45:
        return "Soft"
    if r < 55:
        return "Neutral"
    if r < 70:
        return "Positive"
    return "Strong, stretched" if r >= 75 else "Strong"


def _money(x, ccy):
    if x is None:
        return "--"
    return ("C$" if ccy == "CAD" else "$") + (f"{x:,.2f}" if x >= 1 else f"{x:.3f}")


def _pct(a, b):
    return None if a is None or b in (None, 0) else (a / b - 1) * 100


def analyse(df, days_to_earnings=None, ccy="USD", bench_dates=None):
    """Chart Coach reading for one stock, or None if there isn't enough data."""
    if df is None or len(df) < 60:
        return None
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    n = len(df)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"].fillna(0), df["Open"]
    close = float(c.iloc[-1])
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    tr, atr_s = _atr(df)
    atr = float(atr_s.iloc[-1])
    s20, s50 = float(sma20.iloc[-1]), float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1]) if n >= 200 else None
    slope20 = float(sma20.iloc[-1] / sma20.iloc[-6] - 1) if n >= 26 else None
    slope50 = float(sma50.iloc[-1] / sma50.iloc[-11] - 1) if n >= 61 else None
    rsi14 = _rsi(c)
    ext20 = (close - s20) / atr if atr else None
    ext50 = (close - s50) / atr if atr else None
    avg_v50 = float(v.iloc[-51:-1].mean()) if n > 51 else float(v.mean())
    rvol = float(v.iloc[-1] / avg_v50) if avg_v50 else None
    chg = c.diff()
    v20, c20 = v.iloc[-20:], chg.iloc[-20:]
    up_v, dn_v = float(v20[c20 > 0].sum()), float(v20[c20 < 0].sum())
    updown = up_v / dn_v if dn_v else None
    atr_pct = atr / close if close else None
    gaps = (o / c.shift(1) - 1).abs()
    gap5 = float(gaps.iloc[-5:].max()) if n > 6 else 0.0
    range10 = float(h.iloc[-10:].max() - l.iloc[-10:].min())
    tight = range10 / atr if atr else None
    contraction = float(tr.iloc[-5:].mean() / tr.iloc[-50:].mean()) if n >= 50 else None

    trend = "Mixed"
    if s200 is not None and slope50 is not None:
        if close > s50 > s200 and slope50 > 0:
            trend = "Uptrend"
        elif close < s50 < s200 and slope50 < 0:
            trend = "Downtrend"

    ceiling, floor, last_low = _ceiling_floor(df, close, atr)
    high52_prior = float(h.iloc[-253:-1].max()) if n > 2 else None

    # ---- Data-quality: days the market traded but Yahoo has no row for this stock
    missing = []
    if bench_dates is not None and len(bench_dates):
        recent = [d for d in bench_dates[-20:] if d >= df.index[-20]]
        have = set(df.index[-25:])
        missing = [d.date().isoformat() for d in recent if d not in have]

    setup, reason, checks, area, brk, brk_why = None, None, [], None, None, None

    # 1. Exclusions
    if n < 200:
        setup, reason = "No clear setup", "Less than 200 days of price history."
    elif atr_pct is not None and atr_pct > 0.08:
        setup, reason = "No clear setup", "Daily moves are too erratic for a pattern reading."
    elif gap5 > 0.12:
        setup, reason = "No clear setup", ("A large price gap in the last 5 days, likely news-driven. "
                                           "Price patterns are unreliable right after news.")
    elif days_to_earnings is not None and days_to_earnings <= 3:
        setup, reason = "No clear setup", "Earnings are due within 3 days; the report will likely matter more than the chart."

    # Breakout and pullback checklists are computed for every stock (they teach what to look for)
    bo, bo_today = None, []
    for k in (0, 1, 2):
        t = n - 1 - k
        if t < 21:
            break
        level = float(h.iloc[t - 20:t].max())
        vk = float(v.iloc[t] / v.iloc[max(0, t - 50):t].mean()) if v.iloc[max(0, t - 50):t].mean() else 0
        rng = float(h.iloc[t] - l.iloc[t])
        loc = float((c.iloc[t] - l.iloc[t]) / rng) if rng else 1.0
        held = all(float(c.iloc[j]) > level for j in range(t, n))
        conds = [
            (f"Closed above the prior 20-day high ({_money(level, ccy)})", float(c.iloc[t]) > level),
            (f"Volume at least 1.5x normal ({vk:.1f}x)", vk >= 1.5),
            ("Closed in the upper half of the day's range", loc >= 0.5),
            ("Not in a downtrend", trend != "Downtrend"),
        ]
        if k > 0:
            conds.append(("Has held above the breakout level since", held))
        if all(ok for _, ok in conds):
            bo = {"k": k, "level": level, "low": float(l.iloc[t]), "rvol": vk,
                  "is52": high52_prior is not None and float(c.iloc[t]) > float(h.iloc[max(0, t - 252):t].max()),
                  "conds": conds}
            break
        if k == 0:
            bo_today = conds
    bo_checks = bo["conds"] if bo else bo_today

    # Pullback conditions
    prior20 = h.shift(1).rolling(20).max()
    recent_high_days = [i for i in range(max(21, n - 15), n) if h.iloc[i] >= prior20.iloc[i]]
    hi_day = recent_high_days[-1] if recent_high_days else None
    recent_high = float(h.iloc[hi_day]) if hi_day is not None else float(h.iloc[-15:].max())
    depth = (recent_high - close) / atr if atr else 0
    tested, tested_name = (s20, "20-day") if abs(close - s20) <= abs(close - s50) else (s50, "50-day")
    near = abs(close - tested) / atr if atr else 9
    pb_vol = float(v.iloc[hi_day + 1:].mean() / avg_v50) if hi_day is not None and hi_day < n - 1 and avg_v50 else None
    pb_checks = [
        ("In an uptrend (price above rising 50-day, 50-day above 200-day)", trend == "Uptrend"),
        ("Set a 20-day high within the last 15 days", hi_day is not None),
        (f"Pulled back at least 1.5 ATR from that high ({depth:.1f} ATR)", depth >= 1.5),
        (f"Within 1 ATR of its {tested_name} average and not far below the 50-day",
         near <= 1 and close >= s50 - 0.5 * atr),
        (f"RSI between 38 and 55 ({rsi14:.0f})" if rsi14 is not None else "RSI between 38 and 55",
         rsi14 is not None and 38 <= rsi14 <= 55),
        ("Lighter volume during the pullback" + (f" ({pb_vol:.1f}x normal)" if pb_vol is not None else ""),
         pb_vol is not None and pb_vol < 0.9),
    ]

    if setup is None:
        if trend == "Uptrend" and ((ext20 or 0) >= 3 or (ext50 or 0) >= 6 or (rsi14 or 0) >= 78):
            setup = "Extended"
        elif bo:
            setup = "Breakout"
            area = (bo["level"], bo["level"] + atr)
            if bo["low"] <= bo["level"]:
                brk, brk_why = bo["low"], "the breakout day's low"
            else:  # gapped above the level: the level itself is the line that matters
                brk, brk_why = bo["level"], "the breakout level"
            checks = bo_checks
        elif all(ok for _, ok in pb_checks):
            setup = "Pullback in uptrend"
            area = (tested - 0.5 * atr, tested + 0.5 * atr)
            if last_low and last_low[1] < close:
                brk, brk_why = last_low[1] - 0.25 * atr, "the most recent swing low"
            else:
                brk, brk_why = s50 - 0.5 * atr, "the 50-day average"
            checks = pb_checks
        elif close > s50 and tight is not None and tight <= 3.5 and contraction is not None and contraction <= 0.8:
            setup = "Consolidation"
            area = (float(l.iloc[-10:].min()), float(h.iloc[-10:].max()))
            brk, brk_why = float(l.iloc[-10:].min()), "the bottom of the 10-day range"
        elif (trend == "Uptrend" and (slope20 or 0) > 0 and close > s20
              and rsi14 is not None and 50 <= rsi14 <= 70):
            setup = "Steady uptrend"
            brk, brk_why = s50 - 0.5 * atr, "the 50-day average"
        elif (n > 12 and float(c.iloc[-11]) > float(sma50.iloc[-11])
              and float(c.iloc[-1]) < s50 and float(c.iloc[-2]) < float(sma50.iloc[-2])
              and (slope20 or 0) < 0):
            setup = "Trend weakening"
        elif trend == "Downtrend":
            setup = "Downtrend"
        else:
            setup = "No clear setup"

    # ---- Plain-English reading, built only from the numbers above
    M = lambda x: _money(x, ccy)
    s = []
    if setup == "Pullback in uptrend":
        s.append(f"In an uptrend and has pulled back to its {tested_name} average ({M(tested)}).")
        if pb_vol is not None:
            s.append(f"Volume has been lighter during the dip ({pb_vol:.1f}x normal).")
        s.append(f"Setup area {M(area[0])} to {M(area[1])}. The setup breaks on a daily close below "
                 f"{M(brk)}, {brk_why}.")
    elif setup == "Breakout":
        when = "Closed" if bo["k"] == 0 else f"{bo['k']} day{'s' if bo['k'] > 1 else ''} ago, closed"
        where = " at a new 52-week high," if bo["is52"] else ""
        s.append(f"{when}{where} above its 20-day high of {M(bo['level'])} on {bo['rvol']:.1f}x normal volume, "
                 "finishing in the upper half of the day's range.")
        if brk_why == "the breakout level":
            s.append(f"The breakout breaks on a daily close back below {M(brk)}, the breakout level.")
        else:
            s.append(f"The breakout holds while it stays above {M(bo['level'])}; it breaks on a daily close "
                     f"below {M(brk)}, {brk_why}.")
    elif setup == "Extended":
        bits = []
        if ext20 is not None:
            bits.append(f"{ext20:.1f} ATRs above its 20-day average")
        s.append(f"Up strongly, now {' and '.join(bits) or 'well above its averages'} (RSI {rsi14:.0f}).")
        s.append("Stretched moves often pause or pull back, though they can keep going. "
                 "There is no defined setup at this price.")
    elif setup == "Consolidation":
        s.append(f"Above its 50-day average and trading in a tight range ({M(area[0])} to {M(area[1])}) "
                 "as daily moves shrink.")
        s.append(f"The pattern breaks on a daily close below {M(brk)}, {brk_why}.")
    elif setup == "Steady uptrend":
        s.append(f"In a steady uptrend: above a rising 20-day average ({M(s20)}), RSI {rsi14:.0f}.")
        s.append(f"The trend picture changes on a daily close below {M(brk)}, just under {brk_why}.")
    elif setup == "Trend weakening":
        s.append(f"Has closed below its 50-day average ({M(s50)}) and the 20-day average is turning down. "
                 "The earlier uptrend is weakening.")
    elif setup == "Downtrend":
        s.append("In a downtrend: below falling 50-day and 200-day averages. "
                 "Trex does not describe setups against the trend.")
    else:
        if reason:
            s.append(reason)
        elif trend == "Uptrend":
            s.append("In an uptrend, but no defined setup right now: not a pullback to its averages, "
                     "a breakout or a tight range.")
        else:
            s.append("No clear technical setup: the trend is mixed and no defined pattern is present.")

    if ceiling is None:
        s.append("No recent price level above; the stock is at or near its 52-week high.")

    ceil_px = _price(ceiling["low"]) if ceiling else None
    floor_px = _price(floor["high"]) if floor else None
    return {
        "asOf": df.index[-1].date().isoformat(),
        "setup": setup,
        "trend": trend,
        "reading": " ".join(s),
        "checks": [[t, bool(ok)] for t, ok in checks],
        "breakoutChecks": [[t, bool(ok)] for t, ok in bo_checks],
        "pullbackChecks": [[t, bool(ok)] for t, ok in pb_checks],
        "setupArea": [_price(area[0]), _price(area[1])] if area else None,
        "breakLevel": _price(brk) if brk else None,
        "breakWhy": brk_why,
        "ceiling": ceil_px, "ceilingTouches": ceiling["touches"] if ceiling else None,
        "ceilingPct": _f(_pct(ceil_px, close), 1),
        "floor": floor_px, "floorTouches": floor["touches"] if floor else None,
        "floorPct": _f(_pct(floor_px, close), 1),
        "breakPct": _f(_pct(_price(brk) if brk else None, close), 1),
        "labels": {
            "trend": trend,
            "momentum": _rsi_word(rsi14),
            "rsi": _f(rsi14, 0),
            "relVolume": _f(rvol, 2),
            "upDownVolume": _f(updown, 2),
            "atrPct": _f(atr_pct * 100 if atr_pct else None, 1),
            "ext20": _f(ext20, 1), "ext50": _f(ext50, 1),
        },
        "sma": {"s20": _price(s20), "s50": _price(s50), "s200": _price(s200) if s200 else None},
        "missingDays": missing,
    }


def chart_series(df):
    """Compact 6-month daily series for the stock page chart."""
    df = df.dropna(subset=["Close"])
    c = df["Close"].astype(float)
    s20, s50, s200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    tail = df.index[-CHART_DAYS:]
    nd = 2 if float(c.iloc[-1]) >= 1 else 4
    r = lambda ser: [None if pd.isna(x) else round(float(x), nd) for x in ser.loc[tail]]
    return {
        "d": [int(d.strftime("%Y%m%d")) for d in tail],
        "c": r(c), "s20": r(s20), "s50": r(s50), "s200": r(s200),
        "v": [int(x // 1000) if pd.notna(x) else 0 for x in df["Volume"].loc[tail]],
    }


def backdrop(tech, groups):
    """Market backdrop per market: Supportive / Mixed / Weak, with the numbers behind it."""
    out = {}
    for market, bench, keys in (("us", "^GSPC", ("sp500", "nasdaq", "us")), ("ca", "^GSPTSE", ("tsx",))):
        syms = set().union(*(groups[k]["pool"] for k in keys if k in groups))
        rows = [tech[s] for s in syms if s in tech]
        b = tech.get(bench) or {}
        if not rows or b.get("dist200") is None:
            continue
        above50 = sum(1 for t in rows if (t.get("dist50") or -1) > 0) / len(rows) * 100
        near_hi = sum(1 for t in rows if t.get("fromHigh") is not None and t["fromHigh"] >= -0.02)
        near_lo = sum(1 for t in rows if t.get("low52") and t.get("close")
                      and t["close"] <= t["low52"] * 1.02)
        above200 = b["dist200"] > 0
        state = ("Supportive" if above200 and above50 >= 60 else
                 "Weak" if not above200 and above50 <= 40 else "Mixed")
        out[market] = {
            "state": state, "indexAbove200": above200, "indexDist200": round(b["dist200"] * 100, 1),
            "pctAbove50": round(above50), "nearHighs": near_hi, "nearLows": near_lo,
            "stocks": len(rows), "asOf": b.get("asOf"),
        }
    return out
