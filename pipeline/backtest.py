"""TrexStocks backtest (research only; never publishes to the site).

    python -m pipeline.backtest            # S&P 500, NASDAQ-100 + mid-caps, TSX
    python -m pipeline.backtest --quick    # 200 stocks, for a fast trial run

Tests, on about 2 years of daily prices:
  1. Swing Setups vs two random baselines (1,000 draws each)
  2. The price-based part of the Trex Score: do higher score groups do better?
  3. Entry timing: RSI v1 vs v2 vs no RSI factor
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
        "rs3m": r63 - br63, "rs6m": r126 - br126,
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
def _swing_test(panel):
    p = panel
    ccy = p["ccy"]
    cap_ok = p["cap"].isna() | (p["cap"] >= np.where(ccy == "CAD", 300e6, 500e6))
    elig = ((p["close"] >= np.where(ccy == "CAD", 3, 5))
            & (p["medDV20"] >= np.where(ccy == "CAD", 5e6, 15e6)) & cap_ok
            & p["has200"] & p["atrp"].between(0.015, 0.07)
            & (p["gap5"] <= 0.08)            # also the stand-in for "no earnings in the way"
            & p["above200"])
    pattern = p["breakout"] | p["pullback"]
    pool = p[elig & pattern].copy()
    base = p[elig].copy()
    log(f"swing: {len(base):,} eligible stock-days, {len(pool):,} with a pattern")
    pool["strengthPct"] = _pct_rank(pool, "strength")
    pool["slopePct"] = _pct_rank(pool, "slope50")
    pq = np.where(pool["breakout"], swing.breakout_quality(pool["rvol"], pool["loc"], pool["tightPre"].fillna(9)),
                  swing.pullback_quality(pool["depth"], pool["near"], pool["pbvol"].fillna(9)))
    pool["score"] = swing.score_parts(pool["strengthPct"], pq, pool["slopePct"], pool["ext50"].fillna(0),
                                      pool["nearHigh"].fillna(0.5), pool["updown"].fillna(1))
    # costs: both sides, by size
    for df in (pool, base):
        cost = np.where(df["cap"] >= 2e9, 2 * COST["large"], 2 * COST["small"])
        for hz in HZ:
            df[f"n{hz}"] = df[f"r{hz}"] - cost

    # the daily top list: 10, max 2 per sector, max 6 per pattern
    picks = []
    for d, g in pool.groupby(level="date"):
        g = g.sort_values("score", ascending=False)
        sec, pat, n = {}, {}, 0
        for (dd, s), row in g.iterrows():
            kind = "Breakout" if row["breakout"] else "Pullback"
            if sec.get(row["sector"], 0) >= 2 or pat.get(kind, 0) >= 6:
                continue
            sec[row["sector"]] = sec.get(row["sector"], 0) + 1
            pat[kind] = pat.get(kind, 0) + 1
            picks.append((dd, s))
            n += 1
            if n == swing.TOP_N:
                break
    top = pool.loc[picks].copy()
    top["kind"] = np.where(top["breakout"], "Breakout", "Pullback")

    # worked / failed / expired for everything in the pool (needed for the random baseline)
    outcome = pd.Series(index=pool.index, dtype=object)
    for s, g in p.groupby(level="sym"):
        f = g.droplevel("sym")
        pos = {d: i for i, d in enumerate(f.index)}
        dates = [d for d in pool.xs(s, level="sym").index] if s in pool.index.get_level_values("sym") else []
        res = worked_first(f, [pos[d] for d in dates])
        inv = {i: d for d, i in pos.items()}
        for i, r in res.items():
            outcome.loc[(inv[i], s)] = r
    pool["outcome"] = outcome
    top["outcome"] = outcome.reindex(top.index)

    rng = np.random.default_rng(SEED)
    out = {}
    for per in ("build (months 1-12)", "check (13-18)", "final (19-24)", "all"):
        t = top if per == "all" else top[top["period"] == per]
        pl = pool if per == "all" else pool[pool["period"] == per]
        bs = base if per == "all" else base[base["period"] == per]
        # compare like with like: only rows whose 5-day outcome is already known
        t, pl, bs = (x[x["n5"].notna() & x["b5"].notna()] for x in (t, pl, bs))
        res = {"picks": int(len(t)), "days": int(t.index.get_level_values("date").nunique()),
               "stats": _stats(t, "n"), "byPattern": {k: _stats(g, "n") for k, g in t.groupby("kind")},
               "byMarket": {k: _stats(g, "n") for k, g in t.groupby("ccy")}}
        oc = t["outcome"].value_counts(normalize=True).to_dict()
        res["outcomes"] = {k: float(v) for k, v in oc.items()}
        res["mae10"], res["mfe10"] = float(t["mae10"].mean()), float(t["mfe10"].mean())
        # random baselines: same number of stocks per day, drawn from the pool / the eligible base
        for name, src in (("sameFilterRandom", pl), ("filterOnlyRandom", bs)):
            draws5, worked = _random(t, src, rng, need_outcome=(name == "sameFilterRandom"))
            actual5 = float((t["n5"] - t["b5"]).mean())
            res[name] = {
                "avgExcess5dMedian": float(np.nanmedian(draws5)),
                "avgExcess5dP95": float(np.nanpercentile(draws5, 95)),
                "percentileOfActual": float((draws5 < actual5).mean() * 100),
                "workedRateMedian": float(np.nanmedian(worked)) if worked is not None else None,
            }
        res["actualAvgExcess5d"] = float((t["n5"] - t["b5"]).mean())
        res["actualWorkedRate"] = float((t["outcome"] == "Worked").mean()) if len(t) else None
        # confidence interval: resample pick DAYS (not stocks)
        daily = (t["n5"] - t["b5"]).groupby(level="date").mean().dropna()
        if len(daily) > 10:
            bs_means = [daily.sample(len(daily), replace=True, random_state=int(i)).mean() for i in range(2000)]
            res["ci95Excess5d"] = [float(np.percentile(bs_means, 2.5)), float(np.percentile(bs_means, 97.5))]
        out[per] = res
    fin = out["final (19-24)"]
    ci = fin.get("ci95Excess5d") or [np.nan, np.nan]
    out["verdict"] = {
        "beatsRandom95": fin["actualAvgExcess5d"] > fin["sameFilterRandom"]["avgExcess5dP95"],
        "ciAboveZero": bool(ci[0] > 0),
        "workedBeatsRandom": (fin["actualWorkedRate"] or 0) > (fin["sameFilterRandom"]["workedRateMedian"] or 0),
    }
    out["verdict"]["pass"] = all(out["verdict"].values())
    return out


def _random(top, src, rng, need_outcome=False):
    """1,000 draws: each day pick the same number of stocks at random from `src`."""
    counts = top.groupby(level="date").size()
    ex = (src["n5"] - src["b5"])
    by_day = {d: g.values for d, g in ex.groupby(level="date")}
    oc = (src["outcome"] == "Worked").astype(float) if need_outcome and "outcome" in src else None
    oc_day = {d: g.values for d, g in oc.groupby(level="date")} if oc is not None else None
    sums, n_tot = np.zeros(DRAWS), 0
    wsum = np.zeros(DRAWS) if oc_day else None
    for d, k in counts.items():
        vals = by_day.get(d)
        if vals is None or len(vals) == 0:
            continue
        k = min(k, len(vals))
        idx = np.argsort(rng.random((DRAWS, len(vals))), axis=1)[:, :k]
        pickv = vals[idx]
        sums += np.nansum(pickv, axis=1)
        if wsum is not None:
            wsum += oc_day[d][idx].sum(axis=1)
        n_tot += k
    if not n_tot:
        return np.array([np.nan]), None
    return sums / n_tot, (wsum / n_tot if wsum is not None else None)


# ---------------------------------------------------------------- 2. Trex Score (price part)
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
    variants = {
        "RSI v1 (current)": (trend * 20 + rs * 20 + entry_v1 * 10 + vol * 5 + risk * 10) / 65,
        "RSI v2 (proposed)": (trend * 20 + rs * 20 + entry_v2 * 10 + vol * 5 + risk * 10) / 65,
        "No RSI factor": (trend * 20 + rs * 20 + vol * 5 + risk * 10) / 55,
    }
    out = {}
    for name, score in variants.items():
        p["score"] = score
        p["q"] = p.groupby(level="date")["score"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
        res = {}
        for hz, col in ((20, "r20"), (60, "r60")):
            ex = p[col] - p[f"b{hz}"]
            byq = ex.groupby(p["q"]).mean()
            res[f"{hz}d"] = {f"Q{int(k)}": float(v) for k, v in byq.items()}
            res[f"{hz}d_topMinusBottom"] = float(byq.get(5, np.nan) - byq.get(1, np.nan))
            res[f"{hz}d_byMarket"] = {m: float((ex[p["ccy"] == m].groupby(p["q"][p["ccy"] == m]).mean()).get(5, np.nan)
                                               - (ex[p["ccy"] == m].groupby(p["q"][p["ccy"] == m]).mean()).get(1, np.nan))
                                      for m in ("USD", "CAD")}
        out[name] = res
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
def _write(rep):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day = date.today().isoformat()
    (OUT_DIR / f"results-{day}.json").write_text(json.dumps(rep, indent=1, default=float))
    L = [f"# TrexStocks backtest, {day}", "",
         f"Signals from {rep['start']} to {rep['end']}, {rep['stocks']} stocks (today's lists).",
         "Limits: survivorship bias (today's lists), today's sizes and sectors, earnings dates "
         "approximated by skipping days after big gaps, company numbers not tested. "
         "Entry at the next day's open; swing returns after estimated trading costs.", ""]
    sw = rep["swing"]
    v = sw["verdict"]
    L += ["## 1. Swing Setups vs random", "",
          f"**Verdict on the untouched final 6 months: {'PASS' if v['pass'] else 'DID NOT PASS'}** "
          f"(beats 95% of random draws: {v['beatsRandom95']}; confidence interval above zero: {v['ciAboveZero']}; "
          f"worked-first rate beats random: {v['workedBeatsRandom']})", "",
          "| Period | Picks | Days | Avg 5-day vs index | Random (same filters) median / 95th | Percentile vs random | Worked / Failed / Expired | Avg worst dip (10d) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for per in ("build (months 1-12)", "check (13-18)", "final (19-24)", "all"):
        r = sw[per]
        oc = r.get("outcomes", {})
        L.append(f"| {per} | {r['picks']} | {r['days']} | {_fmt(r['actualAvgExcess5d'])} | "
                 f"{_fmt(r['sameFilterRandom']['avgExcess5dMedian'])} / {_fmt(r['sameFilterRandom']['avgExcess5dP95'])} | "
                 f"{r['sameFilterRandom']['percentileOfActual']:.0f} | "
                 f"{oc.get('Worked', 0) * 100:.0f}% / {oc.get('Failed', 0) * 100:.0f}% / {oc.get('Expired', 0) * 100:.0f}% | "
                 f"{_fmt(r['mae10'])} |")
    L += ["", "Returns by horizon (all periods, after costs):", "",
          "| Horizon | Picks | Avg | Median | Avg vs index | Hit rate | Beat index | 5th percentile |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for hz, s in sw["all"]["stats"].items():
        L.append(f"| {hz} | {s['n']} | {_fmt(s['avg'])} | {_fmt(s['median'])} | {_fmt(s['avgExcess'])} | "
                 f"{s['hit'] * 100:.0f}% | {s['beat'] * 100:.0f}% | {_fmt(s['p5'])} |")
    for grp in ("byPattern", "byMarket"):
        L += ["", f"5-day, {grp[2:].lower()} (all periods):", ""]
        for k, st in sw["all"][grp].items():
            s5 = st.get("5d")
            if s5:
                L.append(f"- {k}: {s5['n']} picks, avg vs index {_fmt(s5['avgExcess'])}, beat index {s5['beat'] * 100:.0f}%")
    fr = sw["all"]["filterOnlyRandom"]
    L += ["", f"Filter-only random (ignores patterns): median {_fmt(fr['avgExcess5dMedian'])}, "
              f"actual beats {fr['percentileOfActual']:.0f}% of draws.", ""]
    L += ["## 2. Trex Score, price-based part: do higher groups do better?", "",
          "Stocks split into 5 equal groups each sampled day (Q5 = highest score). Average return vs index.", "",
          "| Version | 20d Q1 | Q2 | Q3 | Q4 | Q5 | Q5 minus Q1 (20d) | Q5 minus Q1 (60d) | US / Canada (20d) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in rep["trex"].items():
        q = r["20d"]
        L.append(f"| {name} | " + " | ".join(_fmt(q.get(f'Q{i}')) for i in range(1, 6)) +
                 f" | {_fmt(r['20d_topMinusBottom'])} | {_fmt(r['60d_topMinusBottom'])} | "
                 f"{_fmt(r['20d_byMarket'].get('USD'))} / {_fmt(r['20d_byMarket'].get('CAD'))} |")
    L += ["", "## 3. Chart Coach labels: what happened over the next 10 days", "",
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
