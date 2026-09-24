"""Risk Finder: Steady / Balanced / Bold.

Each level has hard filters (a stock either qualifies or not; filters are never
loosened to fill a list) and a Fit score (0-100) that ranks qualifying stocks
with weights suited to that level. Risk measures use fixed scales, so a number
means the same thing on any exchange.

The Trex Score is unchanged by any of this. Fits answer a different question:
"among stocks with this level of historical risk, which have the strongest profile?"
"""
from collections import Counter

from . import config

LEVELS = ["steady", "balanced", "bold"]
LABELS = {"steady": "Steady", "balanced": "Balanced", "bold": "Bold"}

# Hard filters. Caps and dollar volume are in the stock's own currency.
# debtToEquity from Yahoo is a percent: 200 means 2.0x.
FILTERS = {
    "steady": {
        "price": {"USD": 10, "CAD": 5}, "cap": {"USD": 10e9, "CAD": 5e9},
        "dollarVol": {"USD": 20e6, "CAD": 8e6}, "vol": {"USD": 0.35, "CAD": 0.38},
        "beta": 1.15, "maxDD": -0.30, "de": 200, "earningsExclude": 7,
    },
    "balanced": {
        "price": {"USD": 5, "CAD": 3}, "cap": {"USD": 2e9, "CAD": 1e9},
        "dollarVol": {"USD": 8e6, "CAD": 3e6}, "vol": {"USD": 0.50, "CAD": 0.52},
        "beta": 1.50, "maxDD": -0.45, "de": 300, "earningsExclude": 3,
    },
    "bold": {
        "price": {"USD": 3, "CAD": 2}, "cap": {"USD": 500e6, "CAD": 300e6},
        "dollarVol": {"USD": 3e6, "CAD": 1.5e6}, "vol": {"USD": 0.80, "CAD": 0.80},
        "beta": 2.20, "maxDD": -0.65, "de": None, "earningsExclude": None,
    },
}

# Fit weights (each row adds to 100)
WEIGHTS = {
    "steady": {"trend": 10, "relStrength": 10, "entry": 5, "volume": 5, "volRisk": 20,
               "betaRisk": 10, "ddRisk": 10, "quality": 15, "value": 5, "growth": 5,
               "dividend": 5},
    "balanced": {"trend": 15, "relStrength": 15, "entry": 8, "volume": 5, "volRisk": 10,
                 "betaRisk": 8, "ddRisk": 8, "quality": 12, "value": 5, "growth": 10,
                 "dividend": 4},
    "bold": {"trend": 20, "relStrength": 20, "entry": 8, "volume": 10, "volRisk": 5,
             "betaRisk": 3, "ddRisk": 4, "quality": 8, "value": 2, "growth": 20,
             "dividend": 0},
}

# Points off the Fit score by days until earnings: [(max_days, points)]
EARNINGS_PENALTY = {
    "steady": [(14, 8)],
    "balanced": [(7, 8), (14, 4)],
    "bold": [(1, 8), (7, 4), (14, 2)],
}

SECTOR_CAP = {"steady": 4, "balanced": 5, "bold": 6}   # for the default top-20 view
TOP_N = 20
MAX_LIST = 300          # "Show all" list length per level and market


def _clip(x):
    return max(0.0, min(100.0, x))


def vol_score(v):
    return None if v is None else _clip(100 * (0.80 - v) / (0.80 - 0.15))


def beta_score(b):
    return None if b is None else _clip(100 * (2.0 - max(b, 0.0)) / (2.0 - 0.60))


def dd_score(dd):
    return None if dd is None else _clip(100 * (0.70 - abs(dd)) / (0.70 - 0.10))


def dividend_score(yld, payout):
    """Dividends are never required; a non-payer is neutral (50), not risky."""
    if yld is None:
        return None
    if yld == 0:
        return 50.0
    if (payout is not None and payout > 1.0) or yld > 0.08:
        return 30.0
    if 0.015 <= yld <= 0.05 and payout is not None and 0 <= payout <= 0.80:
        return 100.0
    return 70.0


def _failed(rec, level):
    """Reasons a stock does not qualify for a level (empty list = qualifies)."""
    f, m, ccy = FILTERS[level], rec["metrics"], rec["currency"]
    out = []
    need = {"volatility": m.get("volatility60"), "beta": m.get("beta"),
            "drawdown": m.get("maxDrawdown"), "market cap": rec.get("marketCap"),
            "trading volume": m.get("avgDollarVol50")}
    missing = [k for k, v in need.items() if v is None]
    if missing or (m.get("bars") or 0) < 200:
        return ["not enough history or data"]
    if rec["price"] < f["price"][ccy]:
        out.append("price too low")
    if rec["marketCap"] < f["cap"][ccy]:
        out.append("company too small")
    if m["avgDollarVol50"] < f["dollarVol"][ccy]:
        out.append("not enough trading volume")
    if m["volatility60"] > f["vol"][ccy]:
        out.append("price swings too large")
    if m["beta"] > f["beta"]:
        out.append("moves too much with the market")
    if m["maxDrawdown"] < f["maxDD"]:
        out.append("fell too far in the past year")
    margin, roe, de = m.get("profitMargin"), m.get("roe"), m.get("debtToEquity")
    rev = m.get("revenueGrowth")
    fin = rec.get("sector") == "Financial Services"
    if level == "steady":
        if margin is None or margin <= 0 or roe is None or roe <= 0:
            out.append("not consistently profitable")
        if not fin and (de is None or de > f["de"]):
            out.append("too much debt")
    elif level == "balanced":
        if not ((margin is not None and margin > 0) or (rev is not None and rev >= 0.10)):
            out.append("neither profitable nor growing fast")
        if not fin and de is not None and de > f["de"]:
            out.append("too much debt")
    else:  # bold: exclude only clear financial stress
        if margin is not None and margin < 0 and de is not None and de > 400:
            out.append("losing money with heavy debt")
    dte, ex = rec.get("daysToEarnings"), f["earningsExclude"]
    if ex is not None and dte is not None and dte <= ex:
        out.append(f"earnings within {ex} days")
    return out


def _penalty(level, dte):
    if dte is None:
        return 0
    for max_days, pts in EARNINGS_PENALTY[level]:
        if dte <= max_days:
            return pts
    return 0


def fit(rec, level):
    """Fit score (0-100) for a stock that passed the level's filters, or None."""
    m, fac = rec["metrics"], rec["factors"]
    parts = {
        "trend": fac.get("trend"), "relStrength": fac.get("relStrength"),
        "entry": fac.get("entry"), "volume": fac.get("volume"),
        "quality": fac.get("quality"), "value": fac.get("value"), "growth": fac.get("growth"),
        "volRisk": vol_score(m.get("volatility60")), "betaRisk": beta_score(m.get("beta")),
        "ddRisk": dd_score(m.get("maxDrawdown")),
        "dividend": dividend_score(m.get("dividendYield"), m.get("payoutRatio")),
    }
    w = WEIGHTS[level]
    have = {k: v for k, v in parts.items() if v is not None and w[k] > 0}
    cov = sum(w[k] for k in have) / sum(v for v in w.values() if v > 0)
    if cov < config.MIN_SCORE_COVERAGE - 1e-9:
        return None
    score = sum(have[k] * w[k] for k in have) / sum(w[k] for k in have)
    return round(max(0.0, score - _penalty(level, rec.get("daysToEarnings"))), 1)


def assess(rec):
    """Fits for all three levels: {level: score or None} plus reasons for misses."""
    fits, why_not = {}, {}
    for lv in LEVELS:
        failed = _failed(rec, lv)
        if failed:
            fits[lv], why_not[lv] = None, failed
        else:
            fits[lv] = fit(rec, lv)
            if fits[lv] is None:
                why_not[lv] = ["not enough company data"]
    return fits, why_not


def attach(scored):
    """Add riskFits / riskWhyNot to every scored record (records are shared per stock)."""
    seen = {}
    for recs in scored.values():
        for s, r in recs.items():
            if s not in seen:
                seen[s] = assess(r)
            r["riskFits"], r["riskWhyNot"] = seen[s]
    return scored


def _row(r, names, lv):
    m = r["metrics"]
    return [r["yahoo"], names.get(r["yahoo"]) or r["symbol"], r["riskFits"][lv],
            r["trexScore"], r["sector"], r["marketCap"], r["price"],
            m.get("volatility60"), m.get("beta"), m.get("maxDrawdown"),
            m.get("dividendYield"), r.get("nextEarnings"), r.get("daysToEarnings"),
            m.get("marketCorr")]


ROW_FIELDS = ["yahoo", "name", "fit", "trexScore", "sector", "marketCap", "price",
              "volatility", "beta", "maxDrawdown", "dividendYield", "nextEarnings",
              "daysToEarnings", "marketCorr"]


def payload(scored, names, as_of):
    """Lists for the site: per level and market, the top 20 (sector-capped) and all."""
    uniq = {}
    for recs in scored.values():
        for s, r in recs.items():
            uniq.setdefault(s, r)
    out = {"asOf": as_of, "fields": ROW_FIELDS, "levels": {}, "filters": FILTERS,
           "counts": {}}
    for lv in LEVELS:
        out["levels"][lv], out["counts"][lv] = {}, {}
        for market, ccy in (("us", "USD"), ("ca", "CAD")):
            ok = sorted((r for r in uniq.values()
                         if r["currency"] == ccy and r.get("riskFits", {}).get(lv) is not None),
                        key=lambda r: r["riskFits"][lv], reverse=True)
            top, per_sector = [], Counter()
            for r in ok:
                if per_sector[r["sector"]] >= SECTOR_CAP[lv]:
                    continue
                top.append(_row(r, names, lv))
                per_sector[r["sector"]] += 1
                if len(top) == TOP_N:
                    break
            out["levels"][lv][market] = {
                "top": top,
                "all": [_row(r, names, lv) for r in ok[:MAX_LIST]],
            }
            out["counts"][lv][market] = len(ok)
    return out
