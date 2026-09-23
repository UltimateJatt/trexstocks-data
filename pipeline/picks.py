"""Choose the 9 daily picks (3 per index) from the scored stocks."""
from collections import Counter

from . import config
from .scoring import reasons, CATEGORIES
from .util import display_symbol

INDEX_ORDER = ["sp500", "nasdaq", "tsx"]
LABELS = {"blueChip": "Blue Chip", "speculative": "Speculative", "hiddenGem": "Hidden Gem"}


def _recent_symbols(history, today_iso):
    dates = sorted({h["date"] for h in history if h["date"] < today_iso}, reverse=True)
    recent = set(dates[:config.NO_REPEAT_DAYS])
    return {h["yahoo"] for h in history if h["date"] in recent}


def _gem_ladder(recs):
    """Hidden Gem rules, loosened step by step if an index has too few candidates."""
    strict = [r for r in recs if r["eligible"]["hiddenGem"]]
    yield "strict", strict
    cur = "CAD" if recs and recs[0]["index"] == "tsx" else "USD"
    lo, hi = config.GEM_CAP_RANGE[cur]
    yield "any analyst count", [r for r in recs if r["marketCap"] and lo <= r["marketCap"] <= hi]
    caps = sorted(r["marketCap"] for r in recs if r["marketCap"])
    if caps:
        cutoff = caps[len(caps) // 5]
        yield "smallest 20% of the index", [r for r in recs if r["marketCap"] and r["marketCap"] <= cutoff]


def choose(scored, history, live, names, today_iso):
    used = set()
    recent = _recent_symbols(history, today_iso)
    sector_count = Counter()
    picks = {k: {} for k in INDEX_ORDER}
    notes = []

    for idx in INDEX_ORDER:
        recs = [r for r in scored.get(idx, {}).values()
                if not r["filtersFailed"] and r["yahoo"] not in recent]
        idx_sectors = set()
        for cat in CATEGORIES:
            if cat == "hiddenGem":
                ladder = list(_gem_ladder(recs))
            else:
                ladder = [("strict", [r for r in recs if r["eligible"][cat]])]
            chosen, rule = None, None
            for relax_sector in (False, True):
                for rule_name, cands in ladder:
                    cands = sorted((r for r in cands if r["yahoo"] not in used),
                                   key=lambda r: r["scores"][cat], reverse=True)
                    for r in cands:
                        sec = r["sector"]
                        if not relax_sector and (sec in idx_sectors or
                                                 sector_count[sec] >= config.MAX_PER_SECTOR_DAILY):
                            continue
                        chosen, rule = r, rule_name
                        break
                    if chosen:
                        break
                if chosen:
                    if relax_sector:
                        notes.append(f"{idx} {cat}: sector limit relaxed")
                    break
            if not chosen:
                notes.append(f"{idx} {cat}: no stock passed the filters today")
                continue
            if rule != "strict":
                notes.append(f"{idx} {cat}: rule relaxed ({rule})")
            used.add(chosen["yahoo"])
            idx_sectors.add(chosen["sector"])
            sector_count[chosen["sector"]] += 1
            q = live.get(chosen["yahoo"], {})
            why = reasons(chosen, cat)
            picks[idx][cat] = {
                "symbol": chosen["symbol"], "yahoo": chosen["yahoo"],
                "name": names.get(chosen["yahoo"]) or chosen["symbol"],
                "category": LABELS[cat],
                "price": q.get("price", chosen["price"]),
                "change": q.get("change", 0.0),
                "changePercent": q.get("changePercent", 0.0),
                "volume": q.get("volume"),
                "currency": chosen["currency"], "sector": chosen["sector"],
                "marketCap": chosen["marketCap"],
                "score": chosen["scores"][cat],
                "trexScore": chosen["trexScore"], "bestFit": chosen["bestFit"],
                "factors": chosen["factors"],
                "metrics": chosen["metrics"],
                "reasons": why,
                "reasoning": ". ".join(why) + "." if why else "Top overall score in its category.",
                "earningsSoon": chosen["earningsSoon"],
                "nextEarnings": chosen["nextEarnings"],
                "ruleUsed": rule,
            }
    return picks, notes


def history_rows(picks, today_iso, bench_quotes):
    rows = []
    for idx, cats in picks.items():
        b = config.INDEX_SYMBOLS[idx]["yahoo"]
        for cat, p in cats.items():
            rows.append({
                "date": today_iso, "index": idx, "category": cat,
                "yahoo": p["yahoo"], "symbol": display_symbol(p["yahoo"]),
                "name": p["name"], "sector": p["sector"], "score": p["score"],
                "entryPrice": p["price"], "bench": b,
                "benchEntry": (bench_quotes.get(b) or {}).get("price"),
                "results": {},
            })
    return rows
