"""Track record: how each locked pick did 1 day, 1 week and 4 weeks later vs its index."""
from collections import defaultdict

from . import config

HORIZONS = {"1d": 1, "1w": 5, "4w": 20}


def _eval_one(closes, date_iso, entry, ratio_key, row):
    """Return {horizon: return} using closes after the pick date.

    The entry was a 10:30am price, not a close. We save entry/close-on-pick-day once
    so later stock splits (which rescale the whole close history) don't break the math.
    """
    if closes is None or entry is None:
        return {}
    dates = [d.date().isoformat() for d in closes.index]
    if date_iso not in dates:
        return {}
    i = dates.index(date_iso)
    base = float(closes.iloc[i])
    if ratio_key not in row:
        row[ratio_key] = entry / base if base else 1.0
    adj_entry = base * row[ratio_key]
    out = {}
    for h, n in HORIZONS.items():
        if i + n < len(closes):
            out[h] = float(closes.iloc[i + n]) / adj_entry - 1
    return out


def _series(df):
    """Dividend-adjusted closes, so a stock's dividends count toward its return."""
    if df is None:
        return None
    return df["AdjClose"] if "AdjClose" in df else df["Close"]


def _pick_day_close(df, date_iso):
    if df is None:
        return None
    for ts, v in df["Close"].items():
        if ts.date().isoformat() == date_iso:
            return float(v)
    return None


def evaluate(history, hist):
    for row in history:
        if all(h in row.get("results", {}) for h in HORIZONS):
            continue
        s = hist.get(row["yahoo"])
        b = hist.get(row["bench"])
        if "pickDayClose" not in row:
            # kept next to the 10:30am lock price, so both can be compared
            c = _pick_day_close(s, row["date"])
            if c is not None:
                row["pickDayClose"] = c
        # "adjRatio" keys: v2 uses dividend-adjusted closes (old rows used plain closes)
        sr = _eval_one(_series(s), row["date"], row["entryPrice"], "adjRatio", row)
        br = _eval_one(_series(b), row["date"], row.get("benchEntry"), "benchAdjRatio", row)
        res = row.setdefault("results", {})
        for h, r in sr.items():
            if h in res:
                continue  # never rewrite a result that was already graded
            res[h] = {"return": r, "bench": br.get(h),
                      "excess": (r - br[h]) if h in br else None}
    return history


def summary(history, recent_days=60):
    def agg(rows):
        out = {}
        for h in HORIZONS:
            done = [r["results"][h] for r in rows if h in r.get("results", {})]
            if not done:
                out[h] = {"count": 0}
                continue
            ex = [d["excess"] for d in done if d["excess"] is not None]
            out[h] = {
                "count": len(done),
                "avgReturn": sum(d["return"] for d in done) / len(done),
                "winRate": sum(d["return"] > 0 for d in done) / len(done),
                "avgExcess": (sum(ex) / len(ex)) if ex else None,
                "beatIndexRate": (sum(e > 0 for e in ex) / len(ex)) if ex else None,
            }
        return out

    # Picks made on the same day share the same market, so pick days (not picks)
    # are what count as separate observations.
    done_days = {r["date"] for r in history if "4w" in r.get("results", {})}
    evidence = None
    for limit, label in config.EVIDENCE_LABELS:
        if len(done_days) < limit:
            evidence = label
            break

    by_cat = defaultdict(list)
    for r in history:
        by_cat[r["category"]].append(r)
    dates = sorted({r["date"] for r in history}, reverse=True)[:recent_days]
    recent = [r for r in history if r["date"] in set(dates)]
    return {
        "since": min((r["date"] for r in history), default=None),
        "totalPicks": len(history),
        "pickDays": len({r["date"] for r in history}),
        "gradedPickDays4w": len(done_days),
        "evidence": evidence,
        "model": config.PICKS_MODEL,
        "models": sorted({r.get("model", "Daily Picks v1") for r in history}),
        "overall": agg(history),
        "byCategory": {k: agg(v) for k, v in by_cat.items()},
        "recent": sorted(
            ({**{k: r.get(k) for k in ("date", "index", "category", "symbol", "name",
                                       "sector", "score", "entryPrice", "pickDayClose",
                                       "results")},
              "model": r.get("model", "Daily Picks v1")}
             for r in recent),
            key=lambda r: r["date"], reverse=True),
        "note": ("Entry price is the price at pick time (about 10:30am ET). Stock returns "
                 "include dividends; index returns do not. Trading costs are not included."),
    }
