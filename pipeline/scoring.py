"""Trex Score v2 and Daily Picks v2 scoring.

Every stock gets eight factor scores from 0 to 100:
  - Trend, Relative Strength and Volume are percentile ranks against every stock
    TrexStocks covers (US and Canada together).
  - Quality, Value and Growth are percentile ranks within the stock's sector, so a
    bank is compared with banks, not with software companies. Sectors with fewer
    than config.MIN_SECTOR_RANK_COUNT comparable stocks rank against everyone.
  - Entry (RSI) and Risk (volatility, beta, drawdown) use fixed scales.
Missing data is never filled with a made-up value. A factor with no data is left
out and the other weights are scaled up to fill its place. If less than
config.MIN_SCORE_COVERAGE of the weight has data, the score is "insufficient data".

Trex Score = one fixed blend (config.TREX_WEIGHTS) for every stock.
Category scores (Blue Chip, Growth, Hidden Gem) = config.WEIGHTS, used by Daily Picks.
"""
from datetime import date

import numpy as np
import pandas as pd

from . import config
from .util import display_symbol, currency_of

FACTORS = ["trend", "relStrength", "entry", "volume", "risk", "quality", "value", "growth"]
CATEGORIES = ["blueChip", "speculative", "hiddenGem"]


# ---------------------------------------------------------------- helpers
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _rank(s, higher_is_better=True):
    """Percentile rank 0-100. Missing values stay missing."""
    return _num(s).rank(pct=True, ascending=higher_is_better) * 100


def _sector_rank(values, sectors, higher_is_better=True):
    """Rank within sector when the sector has enough comparable stocks, else vs everyone."""
    values = _num(values)
    overall = _rank(values, higher_is_better)
    within = values.groupby(sectors).rank(pct=True, ascending=higher_is_better) * 100
    counts = values.notna().groupby(sectors).transform("sum")
    use = (counts >= config.MIN_SECTOR_RANK_COUNT) & (sectors != "Unknown")
    return within.where(use, overall)


def _blend(parts):
    """Weighted average of the parts that have data; missing if none do."""
    num = sum(s.fillna(0) * w for s, w in parts)
    den = sum(s.notna() * w for s, w in parts)
    return (num / den.replace(0, np.nan))


def _clip100(x):
    return x.clip(0, 100)


def entry_score(rsi):
    rsi = _num(rsi)
    if config.ENTRY_MODEL == "v2":
        # Healthy momentum (RSI 55-65) scores best; very weak or very stretched scores low
        out = pd.Series(np.nan, index=rsi.index)
        out = out.mask(rsi < 45, (60 - 3 * (45 - rsi)).clip(lower=0))
        out = out.mask((rsi >= 45) & (rsi < 55), 60 + 4 * (rsi - 45))
        out = out.mask((rsi >= 55) & (rsi <= 65), 100.0)
        out = out.mask((rsi > 65) & (rsi <= 75), 100 - 4 * (rsi - 65))
        out = out.mask(rsi > 75, (60 - 5 * (rsi - 75)).clip(lower=0))
        return out
    return _clip100(100 - (rsi - 50).abs() * 4)  # v1: best at RSI 50


def risk_parts(df):
    """Fixed-scale risk scores (100 = calmest). The same numbers apply on any exchange."""
    vol = _clip100(100 * (0.80 - _num(df["volatility60"])) / (0.80 - 0.15))
    beta = _clip100(100 * (2.0 - _num(df["beta"]).clip(lower=0)) / (2.0 - 0.60))
    dd = _clip100(100 * (0.70 - _num(df["maxDrawdown"]).abs()) / (0.70 - 0.10))
    return vol, beta, dd


def weighted(fs, weights):
    """Blend factor columns with weights, skipping missing factors.

    Returns (score, coverage). Score is missing when coverage < MIN_SCORE_COVERAGE.
    """
    num = sum(fs[k].fillna(0) * w for k, w in weights.items())
    cov = sum(fs[k].notna() * w for k, w in weights.items())
    score = (num / cov.replace(0, np.nan)).where(cov >= config.MIN_SCORE_COVERAGE - 1e-9)
    return score, cov


def tier_of(pct):
    if pct is None:
        return None
    for floor, label in config.TIERS:
        if pct >= floor:
            return label
    return config.TIERS[-1][1]


def market_cap(sym, fund, price):
    f = fund.get(sym) or {}
    if f.get("sharesOutstanding") and price:
        return f["sharesOutstanding"] * price
    return f.get("marketCap")


def _f(x, as_int=False):
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    return int(x) if as_int else float(x)


def _pe_used(fpe, tpe):
    fpe, tpe = _num(fpe), _num(tpe)
    ok_f = (fpe > 0) & (fpe <= 100)
    ok_t = (tpe > 0) & (tpe <= 100)
    return fpe.where(ok_f, tpe.where(ok_t))


def _div_yield(f, price):
    """Dividend yield as a fraction (0.03 = 3%). 0 for non-payers, None if unknown."""
    rate = f.get("dividendRate")
    if isinstance(rate, (int, float)) and price:
        return rate / price if rate > 0 else 0.0
    ty = f.get("trailingDividendYield")
    if isinstance(ty, (int, float)) and 0 <= ty < 0.5:
        return float(ty)
    return 0.0 if f.get("fetched") and "dividendRate" in f else None


def days_to_earnings(ne, today):
    if not isinstance(ne, str):
        return None
    try:
        d = (date.fromisoformat(ne) - today).days
    except ValueError:
        return None
    return d if d >= 0 else None


def earnings_penalty(days):
    """Points taken off a Daily Pick score. None = excluded from picks."""
    if days is None:
        return 0
    if days <= config.EARNINGS_EXCLUDE_DAYS:
        return None
    for max_days, pts in config.EARNINGS_PENALTIES:
        if days <= max_days:
            return pts
    return 0


# ---------------------------------------------------------------- main
def score_all(groups, tech, fund, live, today=None):
    """Score every stock in every group once, against the whole covered universe.

    groups: {idx: {"pool": set_of_symbols, ...}}
    Returns {idx: {symbol: record}}; a stock in several groups has the same scores.
    """
    today = today or date.today()
    universe = sorted({s for g in groups.values() for s in g["pool"] if s in tech})
    if not universe:
        return {idx: {} for idx in groups}
    bench = {"USD": tech.get("^GSPC", {}), "CAD": tech.get("^GSPTSE", {})}

    rows = []
    for s in universe:
        t, f = tech[s], fund.get(s) or {}
        q = live.get(s) or {}
        price = q.get("price") or t["close"]
        b = bench[currency_of(s)]
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
            "dividendYield": _div_yield(f, price), "payoutRatio": f.get("payoutRatio"),
            "rs3m": (t["ret63"] - (b.get("ret63") or 0)) if t.get("ret63") is not None else None,
            "rs6m": (t["ret126_21"] - (b.get("ret126_21") or 0))
            if t.get("ret126_21") is not None else None,
        })
    df = pd.DataFrame(rows).set_index("sym")
    sec = df["sector"]
    fin = sec == "Financial Services"

    df["peUsed"] = _pe_used(df["forwardPE"], df["trailingPE"])
    df["sectorPE"] = df["peUsed"].groupby(sec).transform("median")

    fs = pd.DataFrame(index=df.index)
    fs["trend"] = _blend([(_rank(df["dist50"]), .35), (_rank(df["dist200"]), .35),
                          (_rank(df["slope50"]), .30)])
    fs["relStrength"] = _blend([(_rank(df["rs3m"]), .60), (_rank(df["rs6m"]), .40)])
    fs["entry"] = entry_score(df["rsi14"])
    fs["volume"] = _rank(_num(df["volTrend"]).clip(0.5, 3.0))
    vol_s, beta_s, dd_s = risk_parts(df)
    fs["risk"] = _blend([(vol_s, config.RISK_MIX["vol"]), (beta_s, config.RISK_MIX["beta"]),
                         (dd_s, config.RISK_MIX["drawdown"])])
    margin = _sector_rank(df["profitMargins"], sec)
    roe = _sector_rank(df["returnOnEquity"], sec)
    de = _sector_rank(df["debtToEquity"], sec, higher_is_better=False)
    q_fin = _blend([(margin, .55), (roe, .45)])          # banks: debt means something else
    q_other = _blend([(margin, .40), (roe, .40), (de, .20)])
    fs["quality"] = q_fin.where(fin, q_other)
    fs["value"] = _sector_rank(df["peUsed"], sec, higher_is_better=False)
    fs["growth"] = _blend([(_sector_rank(df["revenueGrowth"], sec), .50),
                           (_sector_rank(df["earningsGrowth"], sec), .50)])
    fs = fs.round(1)

    trex, cov = weighted(fs, config.TREX_WEIGHTS)
    trex_pct = trex.rank(pct=True) * 100
    cat_scores = {c: weighted(fs, config.WEIGHTS[c])[0] for c in CATEGORIES}

    caps = _num(df["marketCap"])
    tsx_pool = [s for s in groups.get("tsx", {}).get("pool", ()) if s in caps.index]
    tsx_top = set(caps.loc[tsx_pool].sort_values(ascending=False)
                  .head(config.BLUE_CHIP_TSX_TOP_N).index)

    base = {}
    for s in df.index:
        r, f = df.loc[s], fs.loc[s]
        ccy = currency_of(s)
        cap = _f(caps.get(s))
        dte = days_to_earnings(r["nextEarnings"], today)
        dist200, dist50, slope50 = _f(r["dist200"]), _f(r["dist50"]), _f(r["slope50"])

        failed = []
        if r["price"] < config.MIN_PRICE[ccy]:
            failed.append("price too low")
        if r["avgDollarVol50"] < config.MIN_DOLLAR_VOLUME[ccy]:
            failed.append("not enough trading volume")
        if r["bars"] < config.MIN_HISTORY_DAYS:
            failed.append("less than a year of history")
        if config.REQUIRE_ABOVE_200DMA:
            above = dist200 is not None and dist200 > 0
            recovering = (config.TREND_RECOVERY_RULE and dist200 is not None
                          and dist200 >= -0.05 and (dist50 or 0) >= 0.02
                          and (slope50 or 0) >= 0.02)
            if not (above or recovering):
                failed.append("below its 200-day average")
        if abs(r["changePercent"] or 0) > config.MAX_TODAY_MOVE:
            failed.append("moved more than 8% today")
        if dte is not None and dte <= config.EARNINGS_EXCLUDE_DAYS:
            failed.append(f"reports earnings within {config.EARNINGS_EXCLUDE_DAYS} days")

        rev, eg, analysts = _f(r["revenueGrowth"]), _f(r["earningsGrowth"]), _f(r["analysts"], True)
        lo, hi = config.GEM_CAP_RANGE[ccy]
        eligible = {
            "blueChip": (s in tsx_top) if ccy == "CAD"
            else bool(cap and cap >= config.BLUE_CHIP_MIN_CAP_USD),
            "speculative": bool(cap and cap >= config.GROWTH_MIN_CAP[ccy]) and (
                (rev is not None and rev >= config.GROWTH_MIN_REVENUE_GROWTH)
                or (eg is not None and eg >= config.GROWTH_MIN_EARNINGS_GROWTH)),
            "hiddenGem": bool(cap and lo <= cap <= hi)
            and analysts is not None and analysts <= config.GEM_MAX_ANALYSTS,
        }
        scores = {c: (None if pd.isna(cat_scores[c][s]) else round(float(cat_scores[c][s]), 1))
                  for c in CATEGORIES}
        fits = [c for c in CATEGORIES if eligible[c] and scores[c] is not None]
        coverage = float(cov[s])
        ts = None if pd.isna(trex[s]) else int(round(float(trex[s])))
        pct = None if pd.isna(trex_pct[s]) else int(round(float(trex_pct[s])))
        ne = r["nextEarnings"] if isinstance(r["nextEarnings"], str) else None

        base[s] = {
            "yahoo": s, "symbol": display_symbol(s),
            "sector": r["sector"], "currency": ccy, "marketCap": cap,
            "price": float(r["price"]), "changePercent": float(r["changePercent"] or 0),
            "factors": {k: _f(f[k]) for k in FACTORS},
            "metrics": {
                "dist200": dist200, "rs3m": _f(r["rs3m"]), "ret63": _f(r["ret63"]),
                "rsi14": _f(r["rsi14"]), "volTrend": _f(r["volTrend"]),
                "volatility60": _f(r["volatility60"]), "beta": _f(r["beta"]),
                "maxDrawdown": _f(r["maxDrawdown"]), "fromHigh": _f(r["fromHigh"]),
                "high52": _f(r["high52"]), "low52": _f(r["low52"]),
                "pe": _f(r["peUsed"]), "sectorPE": _f(r["sectorPE"]),
                "revenueGrowth": rev, "earningsGrowth": eg,
                "profitMargin": _f(r["profitMargins"]), "roe": _f(r["returnOnEquity"]),
                "debtToEquity": _f(r["debtToEquity"]), "analysts": analysts,
                "avgDollarVol50": _f(r["avgDollarVol50"]), "bars": _f(r["bars"], True),
                "dividendYield": _f(r["dividendYield"]), "payoutRatio": _f(r["payoutRatio"]),
            },
            "eligible": eligible, "filtersFailed": failed,
            "nextEarnings": ne, "daysToEarnings": dte,
            "earningsSoon": dte is not None and dte <= config.EARNINGS_WINDOW_DAYS,
            "earningsUnknown": ne is None,
            "scores": scores,
            "trexScore": ts, "trexPct": pct, "tier": tier_of(pct),
            "coverage": round(coverage, 2),
            "limitedData": ts is not None and coverage < config.LIMITED_DATA_WARNING - 1e-9,
            "bestFit": max(fits, key=lambda c: scores[c]) if fits else None,
            "model": config.SCORE_MODEL,
        }

    return {idx: {s: {**base[s], "index": idx} for s in g["pool"] if s in base}
            for idx, g in groups.items()}


# ---------------------------------------------------------------- reasons
def _pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def _phrase(k, m):
    try:
        if k == "trend" and m["dist200"] is not None and m["dist200"] > 0:
            return f"Uptrend: {_pct(m['dist200'])} above its 200-day average"
        if k == "relStrength" and m["rs3m"] is not None and m["rs3m"] > 0:
            return f"Beating its index by {_pct(m['rs3m'], 1)} over 3 months"
        if k == "entry" and m["rsi14"] is not None:
            return f"Steady entry point (RSI {m['rsi14']:.0f})"
        if k == "volume" and m["volTrend"] is not None and m["volTrend"] >= 1.1:
            return f"Volume running {m['volTrend']:.1f}x its normal level"
        if k == "risk" and m["volatility60"] is not None and m["volatility60"] <= 0.35:
            extra = f", beta {m['beta']:.2f}" if m.get("beta") is not None else ""
            return f"Calmer price swings ({_pct(m['volatility60'])} a year{extra})"
        if k == "quality":
            parts = []
            if m["profitMargin"] is not None and m["profitMargin"] >= 0.10:
                parts.append(f"profit margin {_pct(m['profitMargin'])}")
            if m["roe"] is not None and m["roe"] >= 0.15:
                parts.append(f"return on equity {_pct(m['roe'])}")
            return ("Strong business: " + ", ".join(parts)) if parts else None
        if k == "value" and m["pe"] and m["sectorPE"] and m["pe"] < m["sectorPE"]:
            return f"P/E {m['pe']:.1f} vs sector {m['sectorPE']:.1f}"
        if k == "growth":
            if m["revenueGrowth"] is not None and m["revenueGrowth"] >= 0.08:
                return f"Revenue growing {_pct(m['revenueGrowth'])} a year"
            if m["earningsGrowth"] is not None and m["earningsGrowth"] >= 0.10:
                return f"Earnings growing {_pct(m['earningsGrowth'])} a year"
    except (TypeError, KeyError):
        pass
    return None


def reasons(rec, cat=None, n=3):
    """Up to n plain-language reasons, strongest weighted factors first.
    cat=None uses the Trex Score weights."""
    w = config.WEIGHTS[cat] if cat else config.TREX_WEIGHTS
    have = [k for k in w if rec["factors"].get(k) is not None]
    ranked = sorted(have, key=lambda k: rec["factors"][k] * w[k], reverse=True)
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
