"""TrexStocks backtest (research only; never publishes to the site).

    python -m pipeline.backtest            # S&P 500, NASDAQ-100 + mid-caps, TSX
    python -m pipeline.backtest --quick    # 200 stocks, for a fast trial run

Tests, on about 2 years of daily prices:
  1. Swing Setups vs two random baselines (1,000 draws each), split by pattern, market
     backdrop, size and sector, plus a doubled-cost check
  2. Model lab: the pre-declared Swing alternatives S1 to S4, each vs its own random baseline
  3. The price-based part of the Trex Score and the pre-declared alternatives T1 to T4
     (plus RSI v2 and no RSI): do higher score groups do better, by period and market?
  4. Chart Coach labels: what happened next, by label (description, not prediction)

Honest limits, printed in the report:
  - Uses today's index lists (survivorship bias: companies that dropped out are missing).
  - Uses today's market caps and sectors for size and sector rules.
  - Past earnings dates aren't available free, so days right after a large price gap
    (over 8%) are skipped as a stand-in.
  - Company numbers (quality, value, growth) can't be tested: free sources don't keep
    their history. Only the price-based parts are tested.
  - The pullback volume rule uses the last 5 days, not "since the high" (a close stand-in).
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from . import config, universe, market, fundamentals, swing
from .util import log, now_et

OUT_DIR = config.ROOT / "data" / "backtest"
HZ = (1, 3, 5, 10, 20)
DRAWS = 1000
SEED = 20260924
COST = {"large": 0.0005, "small": 0.0020}   # per side; spread unknown, so an assumption


# ---------------------------------------------------------------- per-stock features
def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(df, bench):
    """All daily measurements for one stock, as columns (no look-ahead: each row uses
    data up to and including that day's close; forward columns are outcomes)."""
    df = df.dropna(subset=["Close", "High", "Low", "Open"]).copy()
    if len(df) < 260:
        return None
    k = (df["AdjClose"] / df["Close"]).fillna(1.0) if "AdjClose" in df else 1.0
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"].fillna(0)
    oa, ha, la, ca = o * k, h * k, l * k, c * k          # dividend-adjusted, for returns
    s20, s50, s200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rsi = _rsi(c)
    slope50 = s50 / s50.shift(10) - 1
    up = (c > s50) & (s50 > s200) & (slope50 > 0)
    down = (c < s50) & (s50 < s200) & (slope50 < 0)
    avgv50 = v.shift(1).rolling(50).mean()
    rvol = v / avgv50
    prior20h = h.shift(1).rolling(20).max()
    rng = (h - l).replace(0, np.nan)
    loc = ((c - l) / rng).fillna(1.0)
    tight_pre = (h.shift(1).rolling(10).max() - l.shift(1).rolling(10).min()) / atr
    ext20, ext50 = (c - s20) / atr, (c - s50) / atr
    extended = up & ((ext20 >= 3) | (ext50 >= 6) | (rsi >= 78))
    breakout = (c > prior20h) & (rvol >= 1.5) & (loc >= 0.5) & ~down & ~extended
    new20 = (h >= prior20h).astype(float)
    recent = new20.rolling(15).max() > 0
    rhigh = h.rolling(15).max()
    depth = (rhigh - c) / atr
    tested = np.where((c - s20).abs() <= (c - s50).abs(), s20, s50)
    near = (c - tested).abs() / atr
    pbvol = v.rolling(5).mean() / avgv50
    pullback = (up & recent & (depth >= 1.5) & (near <= 1) & (c >= s50 - 0.5 * atr)
                & (rsi >= 38) & (rsi <= 55) & (pbvol < 0.9) & ~extended & ~breakout)
    gaps = (o / pc - 1).abs()
    gap5 = gaps.rolling(5).max()
    b = bench.reindex(df.index)
    bca = (b["AdjClose"] if "AdjClose" in b else b["Close"]).ffill()
    bo = b["Open"].ffill()
    r63, r126 = ca / ca.shift(63) - 1, ca.shift(21) / ca.shift(126) - 1
    br63, br126 = bca / bca.shift(63) - 1, bca.shift(21) / bca.shift(126) - 1
    r12_1, br12_1 = ca.shift(21) / ca.shift(252) - 1, bca.shift(21) / bca.shift(252) - 1
    bfull = bench["AdjClose"] if "AdjClose" in bench else bench["Close"]
    b_above200 = (bfull > bfull.rolling(200).mean()).reindex(df.index).ffill().fillna(False)
    daily, bdaily = ca.pct_change(), bca.pct_change()
    beta = daily.rolling(252, min_periods=200).cov(bdaily) / bdaily.rolling(252, min_periods=200).var()
    dd = (ca / ca.rolling(252, min_periods=200).max() - 1).rolling(252, min_periods=200).min()
    chg = c.diff()
    upv = (v * (chg > 0)).rolling(20).sum()
    dnv = (v * (chg < 0)).rolling(20).sum()
    f = pd.DataFrame({
        "close": c, "atrp": atr / c, "atr": atr, "medDV20": (c * v).rolling(20).median(),
        "gap5": gap5, "above200": (c > s200) & (s50 > s200), "has200": s200.notna(),
        "up": up, "down": down, "extended": extended, "breakout": breakout, "pullback": pullback,
        "rvol": rvol, "loc": loc, "tightPre": tight_pre, "depth": depth, "near": near, "pbvol": pbvol,
        "ext50": ext50, "slope50": slope50, "nearHigh": c / h.rolling(252, min_periods=200).max(),
        "updown": upv / dnv.replace(0, np.nan),
        "strength": 0.6 * (r63 - br63) + 0.4 * (r126 - br126),
        "rs3m": r63 - br63, "rs6m": r126 - br126, "mom12_1": r12_1 - br12_1,
        "ret1m": ca / ca.shift(21) - 1, "benchAbove200": b_above200.astype(bool),
        "dist50": c / s50 - 1, "dist200": c / s200 - 1, "rsi": rsi,
        "volTrend": v.rolling(10).mean() / v.rolling(50).mean(),
        "vol60": daily.rolling(60).std() * np.sqrt(252), "beta": beta, "maxDD": dd,
        "breakLevel": np.where(breakout, np.minimum(l, prior20h), l.rolling(10).min() - 0.25 * atr),
    })
    # ---- outcomes: enter at the next day's open
    entry = oa.shift(-1)
    bentry = bo.shift(-1)
    bca_ = bca
    for hz in HZ:
        f[f"r{hz}"] = ca.shift(-hz) / entry - 1
        f[f"b{hz}"] = bca_.shift(-hz) / bentry - 1
    f["entry"] = entry
    f["mae10"] = la[::-1].rolling(10, min_periods=10).min()[::-1].shift(-1) / entry - 1
    f["mfe10"] = ha[::-1].rolling(10, min_periods=10).max()[::-1].shift(-1) / entry - 1
    f["r60"] = ca.shift(-60) / entry - 1
    f["b60"] = bca_.shift(-60) / bentry - 1
    f["_h"], f["_c"], f["entryRaw"] = h, c, o.shift(-1)
    return f


def worked_first(f, idx):
    """For rows `idx` (positions): 'Worked' if +2 ATR is reached before a close below the
    break level within 10 days; same-day touches count as failed."""
    # raw (not dividend-adjusted) prices, the same basis as ATR and the break level
    H, C = f["_h"].values, f["_c"].values
    entry, atrs, brk = f["entryRaw"].values, f["atr"].values, f["breakLevel"].values
    out = {}
    n = len(f)
    for i in idx:
        if i + 10 >= n or np.isnan(entry[i]):
            continue
        tgt = entry[i] + swing.WORK_ATR * atrs[i]
        res = "Expired"
        for j in range(i + 1, i + 11):
            if C[j] < brk[i]:
                res = "Failed"
                break
            if H[j] >= tgt:
                res = "Worked"
                break
        out[i] = res
    return out


# ---------------------------------------------------------------- helpers
def _pct_rank(panel, col):
    return panel.groupby(level="date")[col].rank(pct=True) * 100


def _stats(df, prefix="r"):
    out = {}
    for hz in HZ:
        r, b = df[f"{prefix}{hz}"], df[f"b{hz}"]
        ok = r.notna() & b.notna()
        r, ex = r[ok], (r - b)[ok]
        if not len(r):
            continue
        out[f"{hz}d"] = {"n": int(len(r)), "avg": float(r.mean()), "median": float(r.median()),
                         "avgExcess": float(ex.mean()), "medianExcess": float(ex.median()),
                         "hit": float((r > 0).mean()), "beat": float((ex > 0).mean()),
                         "std": float(r.std()), "p5": float(r.quantile(0.05)), "worst": float(r.min())}
    return out


def _period(dates):
    end = dates.max()
    start = end - pd.DateOffset(years=2)
    dev_end, val_end = start + pd.DateOffset(months=12), start + pd.DateOffset(months=18)
    return start, dev_end, val_end, end


def _fmt(x, pct=True, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x * 100:+.{d}f}%" if pct else f"{x:.{d}f}"


# ---------------------------------------------------------------- main
def run(quick=False):
    c = universe.load()
    fund = fundamentals.load()
    syms = universe.all_symbols(c, include_wide=False)
    if quick:
        syms = syms[:200]
    log(f"backtest: downloading 5 years for {len(syms)} stocks")
    hist = market.history(syms + ["^GSPC", "^GSPTSE"], period="5y")
    frames = []
    for s in syms:
        df = hist.get(s)
        bench = hist.get("^GSPTSE" if s.endswith(".TO") else "^GSPC")
        if df is None or bench is None:
            continue
        try:
            f = features(df, bench)
        except Exception as e:
            log(f"features failed for {s}: {e}")
            continue
        if f is None:
            continue
        fd = fund.get(s) or {}
        f["sym"] = s
        f["ccy"] = "CAD" if s.endswith(".TO") else "USD"
        f["sector"] = fd.get("sector") or "Unknown"
        f["cap"] = fd.get("marketCap") or np.nan
        frames.append(f.reset_index().rename(columns={"index": "date", "Date": "date"}))
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    start, dev_end, val_end, end = _period(panel["date"])
    panel = panel[panel["date"] >= start].copy()
    panel["period"] = np.where(panel["date"] < dev_end, "build (months 1-12)",
                               np.where(panel["date"] < val_end, "check (13-18)", "final (19-24)"))
    panel = panel.set_index(["date", "sym"]).sort_index()
    log(f"backtest panel: {len(panel):,} stock-days, {panel.index.get_level_values('sym').nunique()} stocks, "
        f"{start.date()} to {end.date()}")
    report = {"generated": now_et().isoformat(), "start": str(start.date()), "end": str(end.date()),
              "stocks": int(panel.index.get_level_values("sym").nunique())}
    report["swing"] = _swing_test(panel)
    report["trex"] = _trex_test(panel)
    report["labels"] = _label_test(panel)
    _write(report)
    return report


# ---------------------------------------------------------------- 1. Swing Setups
PERIODS = ("build (months 1-12)", "check (13-18)", "final (19-24)")


def _backdrop_state(p):
    """Historical market backdrop per date and market (same rule as the live Chart Coach):
    Supportive = index above its 200-day average and 60%+ of stocks above their 50-day;
    Weak = index below and 40% or fewer; otherwise Mixed."""
    above50 = (p["dist50"] > 0).astype(float).where(p["dist50"].notna())
    share = above50.groupby([p.index.get_level_values("date"), p["ccy"]]).transform("mean") * 100
    b200 = p["benchAbove200"].astype(bool)
    return pd.Series(np.select([b200 & (share >= 60), ~b200 & (share <= 40)], ["Supportive", "Weak"],
                               default="Mixed"), index=p.index)


def _size_band(cap):
    return np.select([cap.isna(), cap < 2e9, cap < 10e9], ["unknown", "under $2B", "$2B to $10B"],
                     default="over $10B")


def _swing_prep(panel):
    """Every eligible stock-day with a pattern (the pool) and every eligible stock-day (the
    base), with the parts each variant needs, costs and worked / failed outcomes."""
    p = panel
    ccy = p["ccy"]
    cap_ok = p["cap"].isna() | (p["cap"] >= np.where(ccy == "CAD", 300e6, 500e6))
    elig = ((p["close"] >= np.where(ccy == "CAD", 3, 5))
            & (p["medDV20"] >= np.where(ccy == "CAD", 5e6, 15e6)) & cap_ok
            & p["has200"] & p["atrp"].between(0.015, 0.07)
            & (p["gap5"] <= 0.08)            # also the stand-in for "no earnings in the way"
            & p["above200"])
    backdrop = _backdrop_state(p)
    pool = p[elig & (p["breakout"] | p["pullback"])].copy()
    base = p[elig].copy()
    for df in (pool, base):
        df["backdrop"] = backdrop.reindex(df.index)
        df["kind"] = np.where(df["breakout"], "Breakout", np.where(df["pullback"], "Pullback", "None"))
        df["size"] = _size_band(df["cap"])
        df["cost"] = np.where(df["cap"] >= 2e9, 2 * COST["large"], 2 * COST["small"])
        for hz in HZ:
            df[f"n{hz}"] = df[f"r{hz}"] - df["cost"]
    log(f"swing: {len(base):,} eligible stock-days, {len(pool):,} with a pattern")
    bq = swing.breakout_quality(pool["rvol"], pool["loc"], pool["tightPre"].fillna(9))
    pool["pqMain"] = np.where(pool["breakout"], bq,
                              swing.pullback_quality(pool["depth"], pool["near"], pool["pbvol"].fillna(9)))
    pool["pqDeep"] = np.where(pool["breakout"], bq,
                              swing.pullback_quality(pool["depth"], pool["near"], pool["pbvol"].fillna(9), deep=True))
    pool["calm"] = swing.calm_score(pool["atrp"])
    # worked / failed / expired, once for the whole pool (all variants draw from it)
    outcome = pd.Series(index=pool.index, dtype=object)
    pool_syms = set(pool.index.get_level_values("sym"))
    for s, g in p.groupby(level="sym"):
        if s not in pool_syms:
            continue
        f = g.droplevel("sym")
        pos = {d: i for i, d in enumerate(f.index)}
        inv = {i: d for d, i in pos.items()}
        res = worked_first(f, [pos[d] for d in pool.xs(s, level="sym").index])
        for i, r in res.items():
            outcome.loc[(inv[i], s)] = r
    pool["outcome"] = outcome
    return pool, base


def _variant_pool(pool, base, variant):
    """Rows the variant may choose from, its score, and the matching filter-only base."""
    cfg = swing.VARIANTS[variant]
    kinds = {"Breakout" if k == "Breakout" else "Pullback" for k in cfg["patterns"]}
    pl = pool[pool["kind"].isin(kinds)].copy()
    bs = base
    if cfg.get("backdrop"):
        pl = pl[pl["backdrop"] == cfg["backdrop"]]
        bs = base[base["backdrop"] == cfg["backdrop"]]
    pl["strengthPct"] = _pct_rank(pl, "strength")
    pl["slopePct"] = _pct_rank(pl, "slope50")
    pq = pl["pqDeep"] if cfg.get("deep") else pl["pqMain"]
    pl["score"] = swing.score_parts(pl["strengthPct"], pq, pl["slopePct"], pl["ext50"].fillna(0),
                                    pl["nearHigh"].fillna(0.5), pl["updown"].fillna(1))
    if cfg.get("calm"):
        pl["score"] = (1 - cfg["calm"]) * pl["score"] + cfg["calm"] * pl["calm"]
    return pl, bs, (swing.MAX_PER_PATTERN if len(kinds) > 1 else swing.TOP_N)


def _daily_top(pl, pat_cap):
    """The daily list: 10, max 2 per sector, max 6 per pattern (one-pattern variants: 10)."""
    picks = []
    for _, g in pl.sort_values("score", ascending=False).groupby(level="date", sort=False):
        sec, pat = {}, {}
        for (dd, s), sector, kind in zip(g.index, g["sector"], g["kind"]):
            if sec.get(sector, 0) >= swing.MAX_PER_SECTOR or pat.get(kind, 0) >= pat_cap:
                continue
            sec[sector] = sec.get(sector, 0) + 1
            pat[kind] = pat.get(kind, 0) + 1
            picks.append((dd, s))
            if sum(sec.values()) == swing.TOP_N:
                break
    return pl.loc[picks].sort_index()


def _ex5(df, cost_mult=1.0):
    """5-day return vs index after costs (cost_mult=2 doubles the assumed costs)."""
    return df["n5"] - df["b5"] - (cost_mult - 1) * df["cost"]


def _swing_eval(pool, base, variant, rng, full=False):
    pl, bs, pat_cap = _variant_pool(pool, base, variant)
    top = _daily_top(pl, pat_cap)
    out = {}
    for per in PERIODS + ("all",):
        t = top if per == "all" else top[top["period"] == per]
        src = pl if per == "all" else pl[pl["period"] == per]
        # compare like with like: only rows whose 5-day outcome is already known
        t, src = (x[x["n5"].notna() & x["b5"].notna()] for x in (t, src))
        res = {"picks": int(len(t)), "days": int(t.index.get_level_values("date").nunique())}
        if not len(t):
            out[per] = res
            continue
        ex = _ex5(t)
        res["actualAvgExcess5d"] = float(ex.mean())
        res["avgExcess5d2xCost"] = float(_ex5(t, 2.0).mean())
        res["beat5d"] = float((ex > 0).mean())
        res["actualWorkedRate"] = float((t["outcome"] == "Worked").mean())
        res["outcomes"] = {k: float(v) for k, v in t["outcome"].value_counts(normalize=True).items()}
        res["mae10"], res["mfe10"] = float(t["mae10"].mean()), float(t["mfe10"].mean())
        res["byMarket"] = {m: {"n": int(len(g)), "avgExcess5d": float(_ex5(g).mean())}
                           for m, g in t.groupby("ccy")}
        draws5, worked = _random(t, src, rng, need_outcome=True)
        res["sameFilterRandom"] = {
            "avgExcess5dMedian": float(np.nanmedian(draws5)),
            "avgExcess5dP95": float(np.nanpercentile(draws5, 95)),
            "percentileOfActual": float((draws5 < res["actualAvgExcess5d"]).mean() * 100),
            "workedRateMedian": float(np.nanmedian(worked)) if worked is not None else None,
        }
        daily = ex.groupby(level="date").mean().dropna()
        if len(daily) > 10:
            idx = rng.integers(0, len(daily), (2000, len(daily)))
            bs_means = daily.values[idx].mean(axis=1)
            res["ci95Excess5d"] = [float(np.percentile(bs_means, 2.5)), float(np.percentile(bs_means, 97.5))]
        if full and per == "all":
            b = bs[bs["n5"].notna() & bs["b5"].notna()]
            d5, _ = _random(t, b, rng)
            res["filterOnlyRandom"] = {"avgExcess5dMedian": float(np.nanmedian(d5)),
                                       "percentileOfActual": float((d5 < res["actualAvgExcess5d"]).mean() * 100)}
            res["stats"] = _stats(t, "n")
            for grp, col in (("byPattern", "kind"), ("byBackdrop", "backdrop"), ("bySector", "sector"),
                             ("bySize", "size")):
                res[grp] = {str(k): {"n": int(len(g)), "avgExcess5d": float(_ex5(g).mean()),
                                     "beat5d": float((_ex5(g) > 0).mean()),
                                     "worked": float((g["outcome"] == "Worked").mean())}
                            for k, g in t.groupby(col)}
        out[per] = res
    fin = out["final (19-24)"]
    ci = fin.get("ci95Excess5d") or [np.nan, np.nan]
    if fin.get("picks"):
        v = {"beatsRandom95": fin["actualAvgExcess5d"] > fin["sameFilterRandom"]["avgExcess5dP95"],
             "ciAboveZero": bool(ci[0] > 0),
             "workedBeatsRandom": (fin["actualWorkedRate"] or 0) > (fin["sameFilterRandom"]["workedRateMedian"] or 0)}
        v["pass"] = all(v.values())
    else:
        v = {"beatsRandom95": False, "ciAboveZero": False, "workedBeatsRandom": False, "pass": False}
    out["verdict"] = v
    return out


def _swing_test(panel):
    pool, base = _swing_prep(panel)
    rng = np.random.default_rng(SEED)
    out = {}
    for v in swing.VARIANTS:
        log(f"swing: testing {v}")
        out[v] = _swing_eval(pool, base, v, rng, full=(v == "main"))
    # the pre-declared switch rule: beats the current version in every period and both markets
    m = out["main"]
    for v, r in out.items():
        if v == "main":
            continue
        per_ok = {per: (r[per].get("actualAvgExcess5d", -9) > m[per].get("actualAvgExcess5d", 9)) for per in PERIODS}
        mk_ok = {k: ((r["all"].get("byMarket") or {}).get(k, {}).get("avgExcess5d", -9)
                     > (m["all"].get("byMarket") or {}).get(k, {}).get("avgExcess5d", 9)) for k in ("USD", "CAD")}
        r["vsCurrent"] = {"periods": per_ok, "markets": mk_ok,
                          "beatsCurrentEverywhere": all(per_ok.values()) and all(mk_ok.values())}
    return out


def _random(top, src, rng, need_outcome=False):
    """1,000 draws: each day pick the same number of stocks at random from `src`."""
    counts = top.groupby(level="date").size()
    ex = _ex5(src)
    by_day = {d: g.values for d, g in ex.groupby(level="date")}
    oc_day = None
    if need_outcome and "outcome" in src:
        oc = (src["outcome"] == "Worked").astype(float)
        oc_day = {d: g.values for d, g in oc.groupby(level="date")}
    sums, n_tot = np.zeros(DRAWS), 0
    wsum = np.zeros(DRAWS) if oc_day else None
    for d, k in counts.items():
        vals = by_day.get(d)
        if vals is None or len(vals) == 0:
            continue
        k = min(k, len(vals))
        idx = np.argpartition(rng.random((DRAWS, len(vals))), k - 1, axis=1)[:, :k] if k < len(vals) \
            else np.tile(np.arange(len(vals)), (DRAWS, 1))
        sums += np.nansum(vals[idx], axis=1)
        if wsum is not None:
            wsum += oc_day[d][idx].sum(axis=1)
        n_tot += k
    if not n_tot:
        return np.array([np.nan]), None
    return sums / n_tot, (wsum / n_tot if wsum is not None else None)


# ---------------------------------------------------------------- 2. Trex Score (price part)
TREX_CURRENT = "Current (RSI v1)"


def _trex_test(panel):
    p = panel[panel["has200"] & (panel["medDV20"] > 1e6)].copy()
    dates = p.index.get_level_values("date").unique().sort_values()[::5]   # every 5th day
    p = p[p.index.get_level_values("date").isin(dates)].copy()
    rk = lambda col, asc=True: p.groupby(level="date")[col].rank(pct=True, ascending=asc) * 100
    trend = 0.35 * rk("dist50") + 0.35 * rk("dist200") + 0.30 * rk("slope50")
    rs = 0.6 * rk("rs3m") + 0.4 * rk("rs6m")
    vol = rk("volTrend")
    risk = (0.5 * swing.lin(p["vol60"], 0.80, 0.15) + 0.25 * swing.lin(p["beta"].clip(lower=0), 2.0, 0.6)
            + 0.25 * swing.lin(p["maxDD"].abs(), 0.70, 0.10))
    rsi = p["rsi"]
    entry_v1 = (100 - (rsi - 50).abs() * 4).clip(0, 100)
    entry_v2 = pd.Series(np.select(
        [rsi < 45, rsi < 55, rsi <= 65, rsi <= 75],
        [(60 - 3 * (45 - rsi)).clip(lower=0), 60 + 4 * (rsi - 45), 100.0, 100 - 4 * (rsi - 65)],
        default=(60 - 5 * (rsi - 75)).clip(lower=0)), index=p.index)
    mom = rk("mom12_1")
    rev = rk("ret1m", asc=False)          # last month's losers rank higher
    hi = rk("nearHigh")
    core = trend * 20 + rs * 20 + entry_v1 * 10 + vol * 5 + risk * 10
    variants = {
        TREX_CURRENT: core / 65,
        "RSI v2": (trend * 20 + rs * 20 + entry_v2 * 10 + vol * 5 + risk * 10) / 65,
        "No RSI factor": (trend * 20 + rs * 20 + vol * 5 + risk * 10) / 55,
        "T1 12-month momentum": (trend * 20 + mom * 20 + entry_v1 * 10 + vol * 5 + risk * 10) / 65,
        "T2 Reversal credit": (core + rev * 10) / 75,
        "T3 Low-volatility tilt": (core + risk * 15) / 80,
        "T4 Near 52-week high": (core + hi * 10) / 75,
    }
    out = {}
    for name, score in variants.items():
        p["score"] = score
        p["q"] = p.groupby(level="date")["score"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)

        def spread(mask, hz=20):
            ex = (p[f"r{hz}"] - p[f"b{hz}"])[mask]
            byq = ex.groupby(p["q"][mask]).mean()
            return float(byq.get(5, np.nan) - byq.get(1, np.nan))

        res = {}
        for hz, col in ((20, "r20"), (60, "r60")):
            ex = p[col] - p[f"b{hz}"]
            byq = ex.groupby(p["q"]).mean()
            res[f"{hz}d"] = {f"Q{int(k)}": float(v) for k, v in byq.items()}
            res[f"{hz}d_topMinusBottom"] = float(byq.get(5, np.nan) - byq.get(1, np.nan))
        res["20d_byMarket"] = {m: spread(p["ccy"] == m) for m in ("USD", "CAD")}
        res["20d_byPeriod"] = {per: spread(p["period"] == per) for per in PERIODS}
        out[name] = res
    cur = out[TREX_CURRENT]
    for name, r in out.items():
        if name == TREX_CURRENT:
            continue
        per_ok = {per: r["20d_byPeriod"][per] > cur["20d_byPeriod"][per] for per in PERIODS}
        mk_ok = {m: r["20d_byMarket"][m] > cur["20d_byMarket"][m] for m in ("USD", "CAD")}
        r["vsCurrent"] = {"periods": per_ok, "markets": mk_ok,
                          "beatsCurrentEverywhere": all(per_ok.values()) and all(mk_ok.values())}
    return out


# ---------------------------------------------------------------- 4. Chart Coach labels
def _label_test(panel):
    p = panel[panel["has200"] & (panel["medDV20"] > 5e6)]
    label = np.select([p["extended"], p["breakout"], p["pullback"], p["down"], p["up"]],
                      ["Extended", "Breakout", "Pullback in uptrend", "Downtrend", "Uptrend, other"],
                      default="Mixed / other")
    g = p.assign(label=label).groupby("label")
    out = {}
    for k, d in g:
        ex = d["r10"] - d["b10"]
        out[k] = {"n": int(len(d)), "avg10": float(d["r10"].mean()), "avgExcess10": float(ex.mean()),
                  "mae10": float(d["mae10"].mean()), "mfe10": float(d["mfe10"].mean())}
    return out


# ---------------------------------------------------------------- report
def _yn(x):
    return "yes" if x else "no"


def _write(rep):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    (OUT_DIR / f"results-{day}.json").write_text(json.dumps(rep, indent=1, default=float))
    L = [f"# TrexStocks backtest, {day}", "",
         f"Signals from {rep['start']} to {rep['end']}, {rep['stocks']} stocks (today's lists).",
         "Limits: survivorship bias (today's lists), today's sizes and sectors, earnings dates "
         "approximated by skipping days after big gaps, company numbers not tested. "
         "Entry at the next day's open; swing returns after estimated trading costs.", ""]
    allsw = rep["swing"]
    sw = allsw["main"]
    v = sw["verdict"]
    L += ["## 1. Swing Setups (current version) vs random", "",
          f"**Verdict on the untouched final 6 months: {'PASS' if v['pass'] else 'DID NOT PASS'}** "
          f"(beats 95% of random draws: {v['beatsRandom95']}; confidence interval above zero: {v['ciAboveZero']}; "
          f"worked-first rate beats random: {v['workedBeatsRandom']})", "",
          "| Period | Picks | Days | Avg 5-day vs index | Random (same filters) median / 95th | Percentile vs random | Worked / Failed / Expired | Avg worst dip (10d) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for per in PERIODS + ("all",):
        r = sw[per]
        if not r.get("picks"):
            L.append(f"| {per} | 0 | 0 | -- | -- | -- | -- | -- |")
            continue
        oc = r.get("outcomes", {})
        L.append(f"| {per} | {r['picks']} | {r['days']} | {_fmt(r['actualAvgExcess5d'])} | "
                 f"{_fmt(r['sameFilterRandom']['avgExcess5dMedian'])} / {_fmt(r['sameFilterRandom']['avgExcess5dP95'])} | "
                 f"{r['sameFilterRandom']['percentileOfActual']:.0f} | "
                 f"{oc.get('Worked', 0) * 100:.0f}% / {oc.get('Failed', 0) * 100:.0f}% / {oc.get('Expired', 0) * 100:.0f}% | "
                 f"{_fmt(r['mae10'])} |")
    a = sw["all"]
    if a.get("picks"):
        L += ["", f"With trading costs doubled: avg 5-day vs index {_fmt(a['avgExcess5d2xCost'])} (all periods).", "",
              "Returns by horizon (all periods, after costs):", "",
              "| Horizon | Picks | Avg | Median | Avg vs index | Hit rate | Beat index | 5th percentile |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for hz, s in a["stats"].items():
            L.append(f"| {hz} | {s['n']} | {_fmt(s['avg'])} | {_fmt(s['median'])} | {_fmt(s['avgExcess'])} | "
                     f"{s['hit'] * 100:.0f}% | {s['beat'] * 100:.0f}% | {_fmt(s['p5'])} |")
        for grp, title in (("byPattern", "pattern"), ("byBackdrop", "market backdrop on the signal day"),
                           ("bySize", "company size"), ("bySector", "sector")):
            L += ["", f"5-day, by {title} (all periods):", "",
                  "| Group | Picks | Avg vs index | Beat index | Worked first |", "| --- | --- | --- | --- | --- |"]
            items = sorted(a[grp].items(), key=lambda kv: -kv[1]["n"])
            for k, st in items:
                L.append(f"| {k} | {st['n']} | {_fmt(st['avgExcess5d'])} | {st['beat5d'] * 100:.0f}% | {st['worked'] * 100:.0f}% |")
        fr = a["filterOnlyRandom"]
        L += ["", f"Filter-only random (ignores patterns): median {_fmt(fr['avgExcess5dMedian'])}, "
                  f"actual beats {fr['percentileOfActual']:.0f}% of draws.", "",
              "Small groups (under about 200 picks) mean little on their own.", ""]

    L += ["## 2. Swing Setups: pre-declared alternatives (declared Sep 24, 2026)", "",
          "Each version is compared with its own random baseline (random picks from the same filtered pool). "
          "Switch rule: an alternative replaces the current version only if it beats it in all three periods "
          "and in both markets here, and also in live silent tracking (data/lab/live-report.md).", "",
          "| Version | Picks | Avg 5d vs index: build / check / final | US / Canada (all) | Final: percentile vs own random | Final verdict | Worked first (all) | Avg worst dip | 2x costs (all) | Beats current everywhere |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in allsw.items():
        a = r["all"]
        if not a.get("picks"):
            L.append(f"| {name} | 0 | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        pers = " / ".join(_fmt(r[p_].get("actualAvgExcess5d")) for p_ in PERIODS)
        bm = a.get("byMarket", {})
        fin = r["final (19-24)"]
        pct = fin.get("sameFilterRandom", {}).get("percentileOfActual")
        vc = r.get("vsCurrent")
        L.append(f"| {name} | {a['picks']} | {pers} | {_fmt((bm.get('USD') or {}).get('avgExcess5d'))} / "
                 f"{_fmt((bm.get('CAD') or {}).get('avgExcess5d'))} | {'--' if pct is None else f'{pct:.0f}'} | "
                 f"{'PASS' if r['verdict']['pass'] else 'did not pass'} | {a['actualWorkedRate'] * 100:.0f}% | "
                 f"{_fmt(a['mae10'])} | {_fmt(a['avgExcess5d2xCost'])} | "
                 f"{'(current)' if vc is None else _yn(vc['beatsCurrentEverywhere'])} |")

    L += ["", "## 3. Trex Score, price-based part: do higher groups do better?", "",
          "Stocks split into 5 equal groups each sampled day (Q5 = highest score). Average return vs index. "
          "Same switch rule: an alternative must beat the current version's Q5 minus Q1 in all three periods and both markets.", "",
          "| Version | 20d Q1 | Q2 | Q3 | Q4 | Q5 | Q5 minus Q1 (20d) | build / check / final (20d) | US / Canada (20d) | Q5 minus Q1 (60d) | Beats current everywhere |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in rep["trex"].items():
        q = r["20d"]
        vc = r.get("vsCurrent")
        L.append(f"| {name} | " + " | ".join(_fmt(q.get(f'Q{i}')) for i in range(1, 6)) +
                 f" | {_fmt(r['20d_topMinusBottom'])} | "
                 + " / ".join(_fmt(r["20d_byPeriod"].get(p_)) for p_ in PERIODS) +
                 f" | {_fmt(r['20d_byMarket'].get('USD'))} / {_fmt(r['20d_byMarket'].get('CAD'))} | "
                 f"{_fmt(r['60d_topMinusBottom'])} | {'(current)' if vc is None else _yn(vc['beatsCurrentEverywhere'])} |")
    L += ["", "## 4. Chart Coach labels: what happened over the next 10 days", "",
          "Description only; a difference here is not proof a label predicts anything.", "",
          "| Label | Stock-days | Avg 10d | Avg vs index | Avg worst dip | Avg best rise |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k, r in sorted(rep["labels"].items()):
        L.append(f"| {k} | {r['n']:,} | {_fmt(r['avg10'])} | {_fmt(r['avgExcess10'])} | {_fmt(r['mae10'])} | {_fmt(r['mfe10'])} |")
    (OUT_DIR / f"report-{day}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    log(f"backtest report saved: data/backtest/report-{day}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 stocks only, for a trial run")
    run(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
