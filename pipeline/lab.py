"""Model lab: live evidence, built up day by day. Research only; never shown on the site.

Writes data/lab/live-report.md (and .json) after every Daily prep:
  1. Swing Setups variants (pre-declared Sep 24, 2026), from graded silent picks.
  2. Factor monitor: once a week, for each daily score snapshot that is 20 trading days old, rank
     stocks by each Trex Score factor (and by the Trex Score itself), split into 5
     equal groups, and record each group's next-20-day return vs the average stock.
     This is the only way to test the company-number factors (quality, value, growth),
     because free sources don't keep their history.

Limits: returns use the snapshot closes (price only, no dividends); a stock missing from
the later snapshot is skipped; results need months before they mean much.
"""
import csv
import gzip
import json

import numpy as np

from . import config, swing
from .util import now_et, log

LAB_DIR = config.ROOT / "data" / "lab"
FORWARD = 20                      # trading days (snapshot files) ahead
STEP = 5                          # one start date a week
FACTORS = ["trexScore", "trend", "relStrength", "entry", "volume", "risk", "quality", "value", "growth"]


def _load_snapshot(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def factor_monitor():
    files = sorted(config.SNAPSHOT_DIR.glob("scores-*.csv.gz"))
    results = {f: {"USD": [], "CAD": []} for f in FACTORS}   # per start date: [Q1..Q5 excess]
    dates_used = []
    for i in range(0, len(files) - FORWARD, STEP):
        start, end = _load_snapshot(files[i]), _load_snapshot(files[i + FORWARD])
        if not start or "trend" not in start[0]:
            continue                      # snapshots from before factors were saved
        later = {r["ticker"]: _num(r["close"]) for r in end}
        dates_used.append(files[i].name[7:17])
        for ccy in ("USD", "CAD"):
            rows = []
            for r in start:
                if (r.get("currency") or ("CAD" if r["ticker"].endswith(".TO") else "USD")) != ccy:
                    continue
                c0, c1 = _num(r["close"]), later.get(r["ticker"])
                if c0 and c1:
                    rows.append((r, c1 / c0 - 1))
            if len(rows) < 50:
                continue
            avg = float(np.mean([ret for _, ret in rows]))
            for fac in FACTORS:
                pts = [(_num(r.get(fac)), ret) for r, ret in rows if _num(r.get(fac)) is not None]
                if len(pts) < 50:
                    continue
                pts.sort(key=lambda p: p[0])
                groups = np.array_split(np.array([ret for _, ret in pts]), 5)
                results[fac][ccy].append([float(g.mean()) - avg for g in groups])
    out = {"dates": len(dates_used), "first": dates_used[0] if dates_used else None,
           "last": dates_used[-1] if dates_used else None, "factors": {}}
    for fac, by in results.items():
        out["factors"][fac] = {}
        for ccy, series in by.items():
            if not series:
                continue
            arr = np.array(series)
            spread = arr[:, 4] - arr[:, 0]
            out["factors"][fac][ccy] = {
                "n": len(series), "quintiles": [float(x) for x in arr.mean(axis=0)],
                "topMinusBottom": float(spread.mean()),
                "shareOfDatesPositive": float((spread > 0).mean()),
            }
    return out


def _p(x, d=2):
    return "--" if x is None else f"{x * 100:+.{d}f}%"


def _share(x):
    return "--" if x is None else f"{x * 100:.0f}%"


def write_report():
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    sw = swing.lab_summary()
    fm = factor_monitor()
    rep = {"generated": now_et().isoformat(), "swing": sw, "factorMonitor": fm}
    (LAB_DIR / "live-report.json").write_text(json.dumps(rep, indent=1))
    L = [f"# TrexStocks model lab, live evidence ({now_et():%Y-%m-%d})", "",
         "Research only; nothing here is shown on the site. Alternatives were pre-declared on "
         "Sep 24, 2026 (see the analysis doc, \"Model lab\"). Small samples mean little: wait for "
         "at least 20 graded pick days before reading anything into Swing results.", "",
         "## Swing Setups: live silent tracking", "",
         "| Variant | Pick days | Picks | Graded (5d) | Avg 5d vs index | Beat index | Worked | Failed | Avg worst dip (10d) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v, r in sw.items():
        L.append(f"| {v} | {r['days']} | {r['picks']} | {r['graded5d']} | {_p(r['avgExcess5d'])} | "
                 f"{_share(r['beat5d'])} | {_share(r['worked'])} | {_share(r['failed'])} | {_p(r['avgWorstDip'])} |")
    L += ["", "## Factor monitor: next-20-day return vs the average stock, by factor group", ""]
    if not fm["dates"]:
        L.append(f"Not enough history yet: needs daily snapshots with factors that are {FORWARD} trading "
                 "days old. The first readings appear about 4 weeks after this build.")
    else:
        L += [f"Start dates used: {fm['dates']} ({fm['first']} to {fm['last']}). "
              "Q1 = lowest factor score, Q5 = highest. One start date a week; windows overlap, so treat as rough.", "",
              "| Factor | Market | Dates | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 minus Q1 | Dates Q5 beat Q1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for fac, by in fm["factors"].items():
            for ccy, r in by.items():
                q = r["quintiles"]
                L.append(f"| {fac} | {'US' if ccy == 'USD' else 'Canada'} | {r['n']} | "
                         + " | ".join(_p(x) for x in q)
                         + f" | {_p(r['topMinusBottom'])} | {r['shareOfDatesPositive'] * 100:.0f}% |")
    (LAB_DIR / "live-report.md").write_text("\n".join(L) + "\n")
    log(f"model lab report saved: data/lab/live-report.md ({fm['dates']} factor-monitor dates)")
