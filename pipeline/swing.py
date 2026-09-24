"""Swing Setups (2 to 10 trading days), tracked silently before launch.

Every Daily prep builds the list from the last completed daily candles (before the
market opens), saves it to data/state/swing_history.json, and grades earlier lists.
Nothing is shown on the site until the backtest and 4 weeks of silent tracking are
reviewed. Rules match the Product Gap Analysis, "Gap 2: Swing Setups".

Entry for grading = the open of the first trading day after the signal day (the
first price a reader could get). Outcomes use daily closes, highs and lows.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .util import load_state, save_state, log

MODEL = "Swing Setups v1 (silent)"
HISTORY_FILE = "swing_history.json"
PATTERNS = ("Breakout", "Pullback in uptrend")
HORIZONS = (1, 3, 5, 10, 20)
TOP_N, MAX_PER_SECTOR, MAX_PER_PATTERN = 10, 2, 6
WORK_ATR, WINDOW = 2.0, 10          # "worked" = +2 ATR before a close below the break level, within 10 days

ELIG = {
    "price": {"USD": 5.0, "CAD": 3.0},
    "medDollarVol20": {"USD": 15e6, "CAD": 5e6},
    "cap": {"USD": 500e6, "CAD": 300e6},
    "bars": 200, "atrPct": (0.015, 0.07), "gap5": 0.12, "earningsDays": 10,
}
WEIGHTS = {"strength": .30, "pattern": .30, "trend": .15, "nearHigh": .15, "volume": .10}

# Model lab: pre-declared alternatives (analysis doc, "Model lab", Sep 24, 2026).
# Tracked silently side by side; only adopted if they win on backtest AND new live data.
VARIANTS = {
    "main": {"patterns": PATTERNS},
    "S1_pullbacksOnly": {"patterns": ("Pullback in uptrend",)},
    "S2_supportiveOnly": {"patterns": PATTERNS, "backdrop": "Supportive"},
    "S3_calmerFirst": {"patterns": PATTERNS, "calm": 0.2},
    "S4_deeperDips": {"patterns": ("Pullback in uptrend",), "deep": True},
}
ALIASES = {"pullbackOnly": "S1_pullbacksOnly"}   # name used for the first few days


def lin(x, zero, full):
    """Linear 0-100 score: `zero` maps to 0, `full` to 100, clipped. Works on arrays too."""
    return np.clip((np.asarray(x, dtype=float) - zero) / (full - zero) * 100, 0, 100)


def breakout_quality(rvol, loc, tight_pre):
    return 0.4 * lin(rvol, 1.5, 3.0) + 0.3 * lin(loc, 0.5, 1.0) + 0.3 * lin(tight_pre, 6.0, 2.0)


def pullback_quality(depth, near, pb_vol, deep=False):
    d = np.asarray(depth, dtype=float)
    if deep:   # S4: deeper dips score higher, up to 4 ATR
        depth_s = np.where(d <= 4.0, lin(d, 1.5, 4.0), lin(d, 6.0, 4.0))
    else:
        depth_s = np.where(d <= 3.0, lin(d, 1.0, 1.5), lin(d, 5.0, 3.0))
    return 0.4 * depth_s + 0.3 * lin(near, 1.0, 0.0) + 0.3 * lin(pb_vol, 0.9, 0.5)


def calm_score(atr_pct):
    """S3: 100 for a 1.5% typical daily move, 0 for 7%."""
    return lin(atr_pct, 0.07, 0.015)


def score_parts(strength_pct, pattern_q, slope_pct, ext50, near_high, updown):
    trend = 0.5 * np.asarray(slope_pct, dtype=float) + 0.5 * np.where(
        np.asarray(ext50, dtype=float) <= 3, 100.0, lin(ext50, 6.0, 3.0))
    return (WEIGHTS["strength"] * np.asarray(strength_pct, dtype=float)
            + WEIGHTS["pattern"] * np.asarray(pattern_q, dtype=float)
            + WEIGHTS["trend"] * trend
            + WEIGHTS["nearHigh"] * lin(near_high, 0.75, 1.0)
            + WEIGHTS["volume"] * lin(updown, 0.8, 1.5))


def _eligible(rec, sw, setup):
    ccy = rec["currency"]
    m = rec["metrics"]
    dte = rec.get("daysToEarnings")
    return (setup in PATTERNS
            and rec["price"] >= ELIG["price"][ccy]
            and (sw.get("medDollarVol20") or 0) >= ELIG["medDollarVol20"][ccy]
            and (rec.get("marketCap") or 0) >= ELIG["cap"][ccy]
            and (sw.get("bars") or 0) >= ELIG["bars"]
            and ELIG["atrPct"][0] <= (sw.get("atrPct") or 0) <= ELIG["atrPct"][1]
            and (sw.get("gap5") or 0) <= ELIG["gap5"]
            and sw.get("above200")
            and (dte is None or dte > ELIG["earningsDays"])
            and m.get("high52"))


def build(scored, readings, tech, names, signal_date, variant="main", backdrop=None):
    """Today's ranked Swing Setups (list of dicts) for one variant (see VARIANTS)."""
    cfg = VARIANTS[variant]
    patterns = cfg["patterns"]
    uniq = {}
    for recs in scored.values():
        for s, r in recs.items():
            uniq.setdefault(s, r)
    bench = {"USD": tech.get("^GSPC", {}), "CAD": tech.get("^GSPTSE", {})}
    rows = []
    for s, r in uniq.items():
        rd = readings.get(s)
        if not rd:
            continue
        sw = rd.get("swing") or {}
        if rd["setup"] not in patterns or not _eligible(r, sw, rd["setup"]):
            continue
        if cfg.get("backdrop"):
            mkt = "ca" if r["currency"] == "CAD" else "us"
            if ((backdrop or {}).get(mkt) or {}).get("state") != cfg["backdrop"]:
                continue
        t, b = tech.get(s, {}), bench[r["currency"]]
        if t.get("ret63") is None or t.get("ret126_21") is None:
            continue
        strength = 0.6 * (t["ret63"] - (b.get("ret63") or 0)) + 0.4 * (t["ret126_21"] - (b.get("ret126_21") or 0))
        if rd["setup"] == "Breakout":
            bo = sw.get("breakout") or {}
            pq = float(breakout_quality(bo.get("rvol") or 0, bo.get("loc") or 0, bo.get("tightPre") or 9))
        else:
            pb = sw.get("pullback") or {}
            pq = float(pullback_quality(pb.get("depth") or 0, pb.get("near") or 9, pb.get("pbVol") or 9,
                                        deep=cfg.get("deep", False)))
        rows.append({"sym": s, "rec": r, "rd": rd, "strength": strength, "pattern": pq,
                     "slope50": sw.get("slope50") or 0})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["strengthPct"] = df["strength"].rank(pct=True) * 100
    df["slopePct"] = df["slope50"].rank(pct=True) * 100
    df["score"] = [float(score_parts(x.strengthPct, x.pattern, x.slopePct,
                                     x.rd["swing"].get("ext50") or 0,
                                     x.rec["price"] / x.rec["metrics"]["high52"],
                                     x.rd["swing"].get("updown") or 1))
                   for x in df.itertuples()]
    if cfg.get("calm"):
        w = cfg["calm"]
        df["score"] = [(1 - w) * sc + w * float(calm_score(x.rd["swing"].get("atrPct") or 0.07))
                       for sc, x in zip(df["score"], df.itertuples())]
    df = df.sort_values("score", ascending=False)
    pat_cap = MAX_PER_PATTERN if len(cfg["patterns"]) > 1 else TOP_N   # one-pattern variants fill all 10
    out, sectors, patterns = [], Counter(), Counter()
    for x in df.itertuples():
        r, rd = x.rec, x.rd
        if sectors[r["sector"]] >= MAX_PER_SECTOR or patterns[rd["setup"]] >= pat_cap:
            continue
        sectors[r["sector"]] += 1
        patterns[rd["setup"]] += 1
        out.append({
            "signalDate": signal_date, "yahoo": x.sym, "name": names.get(x.sym) or x.sym,
            "pattern": rd["setup"], "score": round(x.score, 1), "trexScore": r.get("trexScore"),
            "sector": r["sector"], "currency": r["currency"], "marketCap": r.get("marketCap"),
            "signalClose": r["price"], "breakLevel": rd.get("breakLevel"),
            "setupArea": rd.get("setupArea"), "ceiling": rd.get("ceiling"),
            "atr": rd["swing"].get("atr"), "model": MODEL, "variant": variant, "results": {},
        })
        if len(out) == TOP_N:
            break
    return out


def _grade(row, df, bdf):
    """Fill in results for one logged setup once enough days have passed."""
    if df is None or len(df) == 0:
        return
    dates = [d.date().isoformat() for d in df.index]
    after = [i for i, d in enumerate(dates) if d > row["signalDate"]]
    if not after:
        return
    e = after[0]
    entry = float(df["Open"].iloc[e]) if pd.notna(df["Open"].iloc[e]) else None
    if not entry:
        return
    row["entryDate"], row["entryOpen"] = dates[e], entry
    res = row.setdefault("results", {})
    bret = {}
    if bdf is not None:
        bd = [d.date().isoformat() for d in bdf.index]
        if dates[e] in bd:
            be = bd.index(dates[e])
            bo = float(bdf["Open"].iloc[be])
            for hz in HORIZONS:
                if be + hz - 1 < len(bdf) and bo:
                    bret[hz] = float(bdf["Close"].iloc[be + hz - 1]) / bo - 1
    for hz in HORIZONS:
        k = f"{hz}d"
        if k not in res and e + hz - 1 < len(df):
            r = float(df["Close"].iloc[e + hz - 1]) / entry - 1
            res[k] = {"return": r, "bench": bret.get(hz),
                      "excess": (r - bret[hz]) if hz in bret else None}
    if "outcome" not in row and e + WINDOW - 1 < len(df):
        win = df.iloc[e:e + WINDOW]
        atr = row.get("atr") or 0
        target = entry + WORK_ATR * atr
        brk = row.get("breakLevel")
        outcome = "Expired"
        for _, bar in win.iterrows():
            hit_up = atr and bar["High"] >= target
            hit_dn = brk is not None and bar["Close"] < brk
            if hit_dn:            # a day that touches both counts as failed (conservative)
                outcome = "Failed"
                break
            if hit_up:
                outcome = "Worked"
                break
        row["outcome"] = outcome
        row["mae"] = float(win["Low"].min()) / entry - 1
        row["mfe"] = float(win["High"].max()) / entry - 1


def update(scored, readings, tech, hist, names, signal_date, backdrop=None):
    """Grade earlier lists, then log today's list. Returns today's list."""
    history = load_state(HISTORY_FILE, []) or []
    for row in history:
        if "outcome" in row and all(f"{h}d" in row.get("results", {}) for h in HORIZONS):
            continue
        bench = "^GSPTSE" if row["currency"] == "CAD" else "^GSPC"
        try:
            _grade(row, hist.get(row["yahoo"]), hist.get(bench))
        except Exception as e:
            log(f"swing grading failed for {row['yahoo']}: {e}")
    today = []
    if not any(r["signalDate"] == signal_date for r in history):
        for v in VARIANTS:
            rows = build(scored, readings, tech, names, signal_date, variant=v, backdrop=backdrop)
            if v == "main":
                today = rows
            history.extend(rows)
    save_state(HISTORY_FILE, history)
    log(f"swing setups (silent): {len(today)} logged for {signal_date}: "
        + ", ".join(f"{r['yahoo']} ({r['pattern'].split()[0]})" for r in today))
    return today


def lab_summary(history=None):
    """Live comparison of the variants, from graded silent picks."""
    history = history if history is not None else (load_state(HISTORY_FILE, []) or [])
    out = {}
    for v in VARIANTS:
        rows = [r for r in history if ALIASES.get(r.get("variant", "main"), r.get("variant", "main")) == v]
        g5 = [r["results"]["5d"] for r in rows if "5d" in r.get("results", {})
              and r["results"]["5d"].get("excess") is not None]
        oc = [r["outcome"] for r in rows if r.get("outcome")]
        days = sorted({r["signalDate"] for r in rows})
        out[v] = {
            "picks": len(rows), "days": len(days), "graded5d": len(g5),
            "avgExcess5d": (sum(x["excess"] for x in g5) / len(g5)) if g5 else None,
            "beat5d": (sum(x["excess"] > 0 for x in g5) / len(g5)) if g5 else None,
            "worked": (oc.count("Worked") / len(oc)) if oc else None,
            "failed": (oc.count("Failed") / len(oc)) if oc else None,
            "avgWorstDip": (sum(r["mae"] for r in rows if "mae" in r) /
                            max(1, sum(1 for r in rows if "mae" in r))) if any("mae" in r for r in rows) else None,
        }
    return out
