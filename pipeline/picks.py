"""Choose the 9 daily picks (3 categories x 3 index tabs) from the scored stocks.

Daily Picks v2:
  - No tab gets first choice. Each round, every open slot finds its best available
    stock, and the slot with the strongest candidate is filled first.
  - Pick score = category score minus an earnings penalty (see config).
  - A category is never loosened to fill a slot. If nothing qualifies, the slot
    stays empty and the site says so.
"""
from collections import Counter

from . import config
from .scoring import reasons, CATEGORIES, earnings_penalty
from .util import display_symbol

INDEX_ORDER = ["sp500", "nasdaq", "tsx"]
LABELS = config.CATEGORY_LABELS


def _recent_symbols(history, today_iso):
    dates = sorted({h["date"] for h in history if h["date"] < today_iso}, reverse=True)
    recent = set(dates[:config.NO_REPEAT_DAYS])
    return {h["yahoo"] for h in history if h["date"] in recent}


def pick_score(r, cat):
    """Category score after the earnings penalty, or None if it can't be picked."""
    sc = r["scores"].get(cat)
    pen = earnings_penalty(r.get("daysToEarnings"))
    if sc is None or pen is None:
        return None
    return round(sc - pen, 1)


def _candidates(scored, idx, cat, recent):
    out = []
    for r in scored.get(idx, {}).values():
        if r["filtersFailed"] or r["yahoo"] in recent or not r["eligible"][cat]:
            continue
        ps = pick_score(r, cat)
        if ps is not None:
            out.append((ps, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _sector_ok(sec, idx, tab_sectors, total_sectors, relaxed):
    if relaxed:
        return True
    return (tab_sectors[idx][sec] < config.MAX_PER_SECTOR_TAB
            and total_sectors[sec] < config.MAX_PER_SECTOR_DAILY)


def choose(scored, history, live, names, today_iso):
    recent = _recent_symbols(history, today_iso)
    cands = {(idx, cat): _candidates(scored, idx, cat, recent)
             for idx in INDEX_ORDER for cat in CATEGORIES}
    used = set()
    tab_sectors = {idx: Counter() for idx in INDEX_ORDER}
    total_sectors = Counter()
    chosen = {}
    notes = []

    for relaxed in (False, True):
        while True:
            best = None  # (pickScore, slot, record)
            for slot, lst in cands.items():
                if slot in chosen:
                    continue
                for ps, r in lst:
                    if r["yahoo"] in used:
                        continue
                    if not _sector_ok(r["sector"], slot[0], tab_sectors, total_sectors, relaxed):
                        continue
                    if best is None or ps > best[0]:
                        best = (ps, slot, r)
                    break
            if best is None:
                break
            ps, slot, r = best
            chosen[slot] = (ps, r, relaxed)
            used.add(r["yahoo"])
            tab_sectors[slot[0]][r["sector"]] += 1
            total_sectors[r["sector"]] += 1

    picks = {idx: {} for idx in INDEX_ORDER}
    for idx in INDEX_ORDER:
        for cat in CATEGORIES:
            if (idx, cat) not in chosen:
                picks[idx][cat] = None
                notes.append(f"{idx} {cat}: no qualifying {LABELS[cat]} stock today")
                continue
            ps, r, relaxed = chosen[(idx, cat)]
            if relaxed:
                notes.append(f"{idx} {cat}: sector limit relaxed")
            q = live.get(r["yahoo"], {})
            why = reasons(r, cat)
            picks[idx][cat] = {
                "symbol": r["symbol"], "yahoo": r["yahoo"],
                "name": names.get(r["yahoo"]) or r["symbol"],
                "category": LABELS[cat],
                "price": q.get("price", r["price"]),
                "change": q.get("change", 0.0),
                "changePercent": q.get("changePercent", 0.0),
                "volume": q.get("volume"),
                "currency": r["currency"], "sector": r["sector"],
                "marketCap": r["marketCap"],
                "score": r["scores"][cat], "pickScore": ps,
                "earningsPenalty": round(r["scores"][cat] - ps, 1),
                "trexScore": r["trexScore"], "trexPct": r["trexPct"], "tier": r["tier"],
                "bestFit": r["bestFit"], "limitedData": r["limitedData"],
                "factors": r["factors"],
                "metrics": r["metrics"],
                "reasons": why,
                "reasoning": ". ".join(why) + "." if why else "Top score in its category.",
                "earningsSoon": r["earningsSoon"], "earningsUnknown": r["earningsUnknown"],
                "nextEarnings": r["nextEarnings"], "daysToEarnings": r["daysToEarnings"],
                "model": config.PICKS_MODEL,
            }
    return picks, notes


def history_rows(picks, today_iso, bench_quotes):
    rows = []
    for idx, cats in picks.items():
        b = config.INDEX_SYMBOLS[idx]["yahoo"]
        for cat, p in cats.items():
            if not p:
                continue
            rows.append({
                "date": today_iso, "index": idx, "category": cat,
                "yahoo": p["yahoo"], "symbol": display_symbol(p["yahoo"]),
                "name": p["name"], "sector": p["sector"], "score": p["pickScore"],
                "trexScore": p["trexScore"], "model": config.PICKS_MODEL,
                "entryPrice": p["price"], "bench": b,
                "benchEntry": (bench_quotes.get(b) or {}).get("price"),
                "results": {},
            })
    return rows
