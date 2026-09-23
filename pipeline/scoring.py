"""TrexPicks v4 scoring.

Every stock gets ten factor scores from 0 to 100. Each score is a percentile rank
against the other stocks in the same index (100 = best in the index). Each category
(Blue Chip, Speculative, Hidden Gem) then blends the factors with its own weights
from config.WEIGHTS.
"""
from datetime import date, timedelta

import pandas as pd

from . import config
from .util import display_symbol, currency_of

FACTORS = ["trend", "relStrength", "entry", "volume", "risk", "quality", "value",
           "growth", "breakout", "radar"]
CATEGORIES = ["blueChip", "speculative", "hiddenGem"]


def _rank(s, higher_is_better=True):
    """Percentile rank 0-100; missing values get a neutral 50."""
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True, ascending=higher_is_better) * 100
    return r.fillna(50.0)


def _avg(*series):
    return pd.concat(series, axis=1).mean(axis=1)


def market_cap(sym, fund, price):
    f = fund.get(sym) or {}
    if f.get("sharesOutstanding") and price:
        return f["sharesOutstanding"] * price
    return f.get("marketCap")


def score_index(index_key, pool, tech, fund, bench, live, today=None):
    today = today or date.today()
    syms = [s for s in pool if s in tech]
    if not syms:
        return {}
    rows = []
    for s in syms:
        t, f = tech[s], fund.get(s) or {}
        q = live.get(s) or {}
        price = q.get("price") or t["close"]
        rows.append({
            "sym": s, **t, "price": price,
            "changePercent": q.get("changePercent", 0.0),
            "marketCap": market_cap(s, fund, price),
            "sector": f.get("sector") or "Unknown",
            "forwardPE": f.get("forwardPE"), "trailingPE": f.get("trailingPE"),
            "revenueGrowth": f.get("revenueGrowth"), "earningsGrowth": f.get("earningsGrowth"),
            "profitMargins": f.get("profitMargins"), "returnOnEquity": f.get("returnOnEquity"),
            "debtToEquity": f.get("debtToEquity"), "analysts": f.get("analysts"),
            "nextEarnings": f.get("nextEarnings"),
        })
    df = pd.DataFrame(rows).set_index("sym")

    # relative strength vs the index
    df["rs3m"] = df["ret63"] - (bench.get("ret63") or 0)
    df["rs6m"] = df["ret126_21"] - (bench.get("ret126_21") or 0)

    # forward P/E compared with the median of its sector in this index
    pe = pd.to_numeric(df["forwardPE"], errors="coerce")
    pe = pe.where(pe > 0)
    df["sectorPE"] = pe.groupby(df["sector"]).transform("median")
    df["peRatio"] = pe / df["sectorPE"]

    fs = pd.DataFrame(index=df.index)
    fs["trend"] = _avg(_rank(df["dist50"]), _rank(df["dist200"]), _rank(df["slope50"]))
    fs["relStrength"] = _avg(_rank(df["rs3m"]), _rank(df["rs6m"]))
    fs["entry"] = (100 - (df["rsi14"] - 50).abs() * 4).clip(0, 100).fillna(50)
    fs["volume"] = _rank(df["volTrend"])
    fs["risk"] = _rank(df["volatility60"], higher_is_better=False)
    fs["quality"] = _avg(_rank(df["profitMargins"]), _rank(df["returnOnEquity"]),
                         _rank(df["debtToEquity"], higher_is_better=False))
    value = _rank(df["peRatio"], higher_is_better=False)
    fs["value"] = value.where(df["peRatio"].notna(), 30.0)  # no/negative P/E: below average
    fs["growth"] = _avg(_rank(df["revenueGrowth"]), _rank(df["earningsGrowth"]))
    fs["breakout"] = _rank(df["fromHigh"])
    fs["radar"] = _rank(df["analysts"], higher_is_better=False)
    fs = fs.round(1)

    vol_pct = _rank(df["volatility60"])
    cur = "CAD" if index_key == "tsx" else "USD"
    caps = pd.to_numeric(df["marketCap"], errors="coerce")
    tsx_top = set(caps.sort_values(ascending=False).head(config.BLUE_CHIP_TSX_TOP_N).index)
    soon = today + timedelta(days=config.EARNINGS_WINDOW_DAYS)

    out = {}
    for s in df.index:
        r, f = df.loc[s], fs.loc[s]
        ccy = currency_of(s)
        cap = caps.get(s)
        cap = None if pd.isna(cap) else float(cap)

        failed = []
        if r["price"] < config.MIN_PRICE[ccy]:
            failed.append("price too low")
        if r["avgDollarVol50"] < config.MIN_DOLLAR_VOLUME[ccy]:
            failed.append("not enough trading volume")
        if r["bars"] < config.MIN_HISTORY_DAYS:
            failed.append("less than a year of history")
        if config.REQUIRE_ABOVE_200DMA and (pd.isna(r["dist200"]) or r["dist200"] <= 0):
            failed.append("below its 200-day average")
        if abs(r["changePercent"] or 0) > config.MAX_TODAY_MOVE:
            failed.append("moved more than 8% today")

        rev = r["revenueGrowth"]
        analysts = r["analysts"]
        lo, hi = config.GEM_CAP_RANGE[cur]
        eligible = {
            "blueChip": (s in tsx_top) if cur == "CAD"
            else bool(cap and cap >= config.BLUE_CHIP_MIN_CAP_USD),
            "speculative": bool(cap and cap >= config.SPEC_MIN_CAP[cur]) and (
                (not pd.isna(rev) and rev >= config.SPEC_MIN_REVENUE_GROWTH)
                or vol_pct[s] >= config.SPEC_VOL_PERCENTILE),
            "hiddenGem": bool(cap and lo <= cap <= hi) and (
                pd.isna(analysts) or analysts <= config.GEM_MAX_ANALYSTS),
        }

        ne = r["nextEarnings"] if isinstance(r["nextEarnings"], str) else None
        earnings_soon = bool(ne) and today <= date.fromisoformat(ne) <= soon

        scores = {}
        for cat in CATEGORIES:
            w = config.WEIGHTS[cat]
            sc = sum(f[k] * wt for k, wt in w.items())
            if earnings_soon:
                sc -= config.EARNINGS_PENALTY
            scores[cat] = round(float(sc), 1)

        out[s] = {
            "yahoo": s, "symbol": display_symbol(s), "index": index_key,
            "sector": r["sector"], "currency": ccy, "marketCap": cap,
            "price": float(r["price"]), "changePercent": float(r["changePercent"] or 0),
            "factors": {k: float(f[k]) for k in FACTORS},
            "metrics": {
                "dist200": r["dist200"], "rs3m": r["rs3m"], "ret63": r["ret63"],
                "rsi14": r["rsi14"], "volTrend": r["volTrend"],
                "volatility60": r["volatility60"], "fromHigh": r["fromHigh"],
                "high52": r["high52"], "low52": r["low52"],
                "forwardPE": None if pd.isna(pe.get(s)) else float(pe.get(s)),
                "sectorPE": None if pd.isna(r["sectorPE"]) else float(r["sectorPE"]),
                "revenueGrowth": None if pd.isna(rev) else float(rev),
                "profitMargin": None if pd.isna(r["profitMargins"]) else float(r["profitMargins"]),
                "roe": None if pd.isna(r["returnOnEquity"]) else float(r["returnOnEquity"]),
                "analysts": None if pd.isna(analysts) else int(analysts),
            },
            "eligible": eligible, "filtersFailed": failed,
            "earningsSoon": earnings_soon, "nextEarnings": ne,
            "scores": scores,
        }
    return out


# ---------- Plain-language reasons ----------
def _pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def _phrase(k, m):
    try:
        if k == "trend" and m["dist200"] is not None and m["dist200"] > 0:
            return f"Uptrend: {_pct(m['dist200'])} above its 200-day average"
        if k == "relStrength" and m["rs3m"] is not None and m["rs3m"] > 0:
            return f"Beating its index by {_pct(m['rs3m'], 1)} over 3 months"
        if k == "entry" and m["rsi14"] is not None:
            return f"Calm entry point (RSI {m['rsi14']:.0f})"
        if k == "volume" and m["volTrend"] is not None:
            return f"Volume running {m['volTrend']:.1f}x its normal level"
        if k == "risk" and m["volatility60"] is not None:
            return f"Low volatility ({_pct(m['volatility60'])} a year)"
        if k == "quality":
            parts = []
            if m["profitMargin"] is not None and m["profitMargin"] >= 0.10:
                parts.append(f"profit margin {_pct(m['profitMargin'])}")
            if m["roe"] is not None and m["roe"] >= 0.15:
                parts.append(f"return on equity {_pct(m['roe'])}")
            return ("Strong business: " + ", ".join(parts)) if parts else None
        if k == "value" and m["forwardPE"] and m["sectorPE"]:
            return f"Forward P/E {m['forwardPE']:.1f} vs sector {m['sectorPE']:.1f}"
        if k == "growth" and m["revenueGrowth"] is not None and m["revenueGrowth"] >= 0.08:
            return f"Revenue growing {_pct(m['revenueGrowth'])} a year"
        if k == "breakout" and m["fromHigh"] is not None:
            return f"{_pct(-m['fromHigh'], 1)} below its 52-week high"
        if k == "radar" and m["analysts"] is not None:
            return f"Only {m['analysts']} analysts cover it"
    except (TypeError, KeyError):
        pass
    return None


def reasons(rec, cat, n=3):
    w = config.WEIGHTS[cat]
    ranked = sorted(w, key=lambda k: rec["factors"][k] * w[k], reverse=True)
    out = []
    for k in ranked:
        if rec["factors"][k] < 60:
            continue
        p = _phrase(k, rec["metrics"])
        if p:
            out.append(p)
        if len(out) == n:
            break
    return out
