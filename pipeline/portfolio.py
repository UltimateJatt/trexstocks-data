"""TrexFolio: model portfolio stored in data/state/portfolio.json.

Values are tracked in USD. Canadian holdings convert at the CAD/USD rate on the
purchase date (to get the share count) and today's rate (to value them), so
currency moves show up in the return, like a real account.
"""
from datetime import date
from collections import Counter

from . import config
from .util import load_state, save_state, display_symbol, currency_of, log

STATE_FILE = "portfolio.json"
TYPE_LABEL = {"blueChip": "Blue Chip", "speculative": "Speculative", "hiddenGem": "Hidden Gem"}
LABEL_TYPE = {v: k for k, v in TYPE_LABEL.items()}


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(y, m, day)
        except ValueError:
            continue


def load():
    return load_state(STATE_FILE)


def current(state):
    return state["periods"][-1]


def symbols(state):
    return [h["yahoo"] for h in current(state)["holdings"]]


def _close_on(df, day_iso):
    if df is None:
        return None
    for ts, v in df["Close"].items():
        if ts.date().isoformat() == day_iso:
            return float(v)
    return None


def verify(state, hist):
    """Check hand-entered purchase prices against real closing prices on the start date,
    then lock in share counts. Runs once per period."""
    p = current(state)
    if p.get("verified"):
        return False
    start = p["start"]
    fx = _close_on(hist.get(config.FX_SYMBOL), start)
    if fx is None:
        log(f"portfolio verify: no CAD/USD rate for {start} yet, will retry")
        return False
    changed, review = [], []
    for h in p["holdings"]:
        actual = _close_on(hist.get(h["yahoo"]), start)
        if actual is None:
            log(f"portfolio verify: no close for {h['yahoo']} on {start}, will retry")
            return False
        gap = abs(actual / h["purchasePrice"] - 1)
        if gap > 0.25:  # too far off to be a typo; leave it for a human to check
            review.append(f"{h['yahoo']} entered {h['purchasePrice']} vs close {actual:.2f}")
        elif gap > config.PRICE_VERIFY_TOLERANCE:
            changed.append(f"{h['yahoo']} {h['purchasePrice']} -> {actual:.2f}")
            h["enteredPrice"] = h["purchasePrice"]
            h["purchasePrice"] = round(actual, 4)
    for h in p["holdings"]:
        if h["currency"] == "CAD":
            h["shares"] = h["allocationUSD"] / fx / h["purchasePrice"]
        else:
            h["shares"] = h["allocationUSD"] / h["purchasePrice"]
    p["fxAtStart"] = fx
    p["verified"] = True
    p["verifyNote"] = ("Prices checked against closing prices on the start date. "
                       + (f"Corrected: {', '.join(changed)}. " if changed else "")
                       + (f"NEEDS REVIEW (over 25% off, not changed): {', '.join(review)}" if review else "")
                       + ("All matched." if not changed and not review else ""))
    log("portfolio verified:", p["verifyNote"])
    save_state(STATE_FILE, state)
    return True


def _holding_value(h, q, fx_now):
    price = (q or {}).get("price") or h["purchasePrice"]
    shares = h.get("shares")
    if shares is None:  # not verified yet: estimate
        shares = h["allocationUSD"] / h["purchasePrice"] / (fx_now if h["currency"] == "CAD" else 1)
    local_value = shares * price
    usd = local_value * fx_now if h["currency"] == "CAD" else local_value
    return price, shares, usd


def value(state, live, fx_now, today=None):
    today = today or date.today()
    p = current(state)
    total, holdings = 0.0, []
    for h in p["holdings"]:
        q = live.get(h["yahoo"])
        price, shares, usd = _holding_value(h, q, fx_now)
        total += usd
        cost = h["allocationUSD"]
        holdings.append({
            "symbol": display_symbol(h["yahoo"]), "yahoo": h["yahoo"], "name": h["name"],
            "type": TYPE_LABEL[h["type"]], "shares": shares,
            "costBasis": h["purchasePrice"], "currentPrice": price,
            "currentValue": usd, "costBasisUSD": cost,
            "gainLoss": usd - cost, "gainLossPercent": (usd / cost - 1) * 100,
            "dailyChange": (q or {}).get("changePercent", 0.0),
            "currency": h["currency"], "sector": h.get("sector"),
        })
    start_value = p["startValueUSD"]
    inception = state["inception"]
    due = today >= date.fromisoformat(p["nextRebalance"])
    past = [{k: x.get(k) for k in ("id", "start", "end", "startValueUSD", "endValueUSD",
                                   "returnPct")} | {"holdings": [display_symbol(h["yahoo"]) for h in x["holdings"]]}
            for x in state["periods"][:-1]]
    return {
        "purchaseDate": p["start"],
        "nextRebalanceDate": p["nextRebalance"],
        "isRebalanceDue": due,
        "rebalanceMessage": "Rebalancing at 10:30am ET today." if due else None,
        "totalInvested": start_value,
        "currentValue": total,
        "totalGainLoss": total - start_value,
        "totalReturn": (total / start_value - 1) * 100,
        "exchangeRate": fx_now,
        "holdings": holdings,
        "period": p["id"],
        "pricesVerified": bool(p.get("verified")),
        "verifyNote": p.get("verifyNote"),
        "allTime": {
            "startDate": inception["date"], "startValue": inception["valueUSD"],
            "currentValue": total,
            "returnPct": (total / inception["valueUSD"] - 1) * 100,
        },
        "pastPeriods": past,
    }


def _select(candidates):
    """Pick 4 Blue Chip, 3 Speculative, 1 Hidden Gem; 2+ TSX; max 2 per sector."""
    total_slots = sum(config.PORTFOLIO_MIX.values())

    def fill(min_tsx, max_sector):
        chosen, used, sectors = [], set(), Counter()
        for cat, n in config.PORTFOLIO_MIX.items():
            pool = sorted((r for r in candidates if r["eligible"][cat] and not r["filtersFailed"]),
                          key=lambda r: r["scores"][cat], reverse=True)
            for _ in range(n):
                slots_left = total_slots - len(chosen)
                tsx_have = sum(c["currency"] == "CAD" for c, _ in chosen)
                must_tsx = min_tsx - tsx_have >= slots_left
                pick = next((r for r in pool if r["yahoo"] not in used
                             and sectors[r["sector"]] < max_sector
                             and (not must_tsx or r["currency"] == "CAD")), None)
                if pick:
                    chosen.append((pick, cat))
                    used.add(pick["yahoo"])
                    sectors[pick["sector"]] += 1
        return chosen

    # Try the full rules first; loosen only if the market leaves too few candidates.
    for min_tsx, max_sector in ((config.PORTFOLIO_MIN_TSX, config.PORTFOLIO_MAX_PER_SECTOR),
                                (config.PORTFOLIO_MIN_TSX, 3), (0, 3)):
        chosen = fill(min_tsx, max_sector)
        if len(chosen) == total_slots:
            if (min_tsx, max_sector) != (config.PORTFOLIO_MIN_TSX, config.PORTFOLIO_MAX_PER_SECTOR):
                log(f"rebalance: rules loosened (min TSX {min_tsx}, max/sector {max_sector})")
            return chosen
    return chosen


def rebalance(state, scored, live, fx_now, names, today):
    p = current(state)
    if today < date.fromisoformat(p["nextRebalance"]):
        return False
    val = value(state, live, fx_now, today)
    # merge all indices; a stock in two indices keeps its better scores
    merged = {}
    for idx_recs in scored.values():
        for s, r in idx_recs.items():
            if s not in merged or max(r["scores"].values()) > max(merged[s]["scores"].values()):
                merged[s] = r
    chosen = _select(list(merged.values()))
    if len(chosen) < sum(config.PORTFOLIO_MIX.values()):
        log(f"rebalance: only found {len(chosen)} holdings, postponing to next run")
        return False

    total = val["currentValue"]
    p["end"] = today.isoformat()
    p["endValueUSD"] = total
    p["returnPct"] = (total / p["startValueUSD"] - 1) * 100
    alloc = total / len(chosen)
    holdings = []
    for r, cat in chosen:
        price = (live.get(r["yahoo"]) or {}).get("price") or r["price"]
        ccy = currency_of(r["yahoo"])
        shares = alloc / price / (fx_now if ccy == "CAD" else 1)
        holdings.append({
            "yahoo": r["yahoo"], "name": names.get(r["yahoo"]) or r["symbol"], "type": cat,
            "allocationUSD": alloc, "purchasePrice": price, "currency": ccy,
            "shares": shares, "sector": r["sector"], "score": r["scores"][cat],
        })
    state["periods"].append({
        "id": p["id"] + 1, "start": today.isoformat(), "end": None,
        "nextRebalance": add_months(today, config.REBALANCE_MONTHS).isoformat(),
        "startValueUSD": total, "fxAtStart": fx_now, "verified": True,
        "verifyNote": "Bought automatically at the 10:30am ET price on rebalance day.",
        "holdings": holdings,
    })
    save_state(STATE_FILE, state)
    log(f"REBALANCED: period {p['id'] + 1} starts with ${total:,.2f}: "
        + ", ".join(h["yahoo"] for h in holdings))
    return True
