"""TrexStocks data pipeline.

    python -m pipeline.run prep       # daily, before the open: lists, fundamentals, 1y history
    python -m pipeline.run refresh    # every 15 min: prices, movers, portfolio, 10:30 picks

Options: --force (run outside market hours), --dry (don't send to Cloudflare)
"""
import argparse

import pandas as pd

from . import (config, universe, market, fundamentals, technicals, scoring, picks,
               track, portfolio, publish)
from .util import (now_et, today_et, in_window, display_symbol, load_state, save_state,
                   log, currency_of)

BENCH = [v["yahoo"] for v in config.INDEX_SYMBOLS.values()]
HISTORY_FILE = "picks_history.json"


# ---------------------------------------------------------------- prep
def prep(args):
    c = universe.refresh(force=args.force)
    state = portfolio.load()
    history = load_state(HISTORY_FILE, [])
    recent_picks = [h["yahoo"] for h in history[-9 * 40:]]
    stock_syms = sorted(set(universe.all_symbols(c)) | set(portfolio.symbols(state)))

    hist = market.history(stock_syms + recent_picks + BENCH + [config.FX_SYMBOL],
                          period="1y", auto_adjust=False)
    if len(hist) < 0.7 * len(stock_syms):
        raise RuntimeError(f"Only got prices for {len(hist)} of {len(stock_syms)} stocks; "
                           "Yahoo may be blocking. Keeping yesterday's data.")
    tech = technicals.compute_all(hist)
    save_state("technicals.json", tech)
    log(f"technicals saved for {len(tech)} symbols")

    portfolio.verify(state, hist)

    track.evaluate(history, hist)
    save_state(HISTORY_FILE, history)
    publish.put({"track": track.summary(history)}, dry=args.dry)

    # Slowest step last, so a Yahoo hiccup here never blocks prices
    fundamentals.refresh(stock_syms)


# ---------------------------------------------------------------- refresh helpers
def _volume_pace(q, avg_vol, t, today_iso):
    """Today's volume vs a normal full day, adjusted for how much of the day has passed."""
    if not avg_vol:
        return None
    frac = 1.0
    if q["date"] == today_iso:
        minutes = (t.hour * 60 + t.minute) - (9 * 60 + 30)
        if minutes < 30:  # too early in the day to judge volume fairly
            return None
        frac = min(1.0, minutes / 390)
    return q["volume"] / (avg_vol * frac)


def _stock_row(s, q, tech, fund, names, t, today_iso):
    tt = tech.get(s, {})
    f = fund.get(s, {})
    pace = _volume_pace(q, tt.get("avgVol50"), t, today_iso)
    return {
        "symbol": display_symbol(s), "yahoo": s,
        "name": names.get(s) or f.get("name") or display_symbol(s),
        "price": q["price"], "change": q["change"], "changePercent": q["changePercent"],
        "volume": q["volume"], "avgVolume": tt.get("avgVol50"), "volumeRatio": pace,
        "fiftyTwoWeekHigh": tt.get("high52"), "fiftyTwoWeekLow": tt.get("low52"),
        "marketCap": scoring.market_cap(s, fund, q["price"]),
        "sector": f.get("sector") or "Unknown", "currency": currency_of(s),
        "_dayHigh": q["high"], "_dayLow": q["low"],
    }


def _public(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _sectors(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    for sec, g in df.groupby("sector"):
        caps = pd.to_numeric(g["marketCap"], errors="coerce").fillna(0)
        chg = g["changePercent"]
        avg = float((chg * caps).sum() / caps.sum()) if caps.sum() > 0 else float(chg.mean())
        best, worst = g.loc[chg.idxmax()], g.loc[chg.idxmin()]
        out.append({"sector": sec, "changePercent": avg, "count": int(len(g)),
                    "best": {"symbol": best["symbol"], "changePercent": float(best["changePercent"])},
                    "worst": {"symbol": worst["symbol"], "changePercent": float(worst["changePercent"])}})
    return sorted(out, key=lambda x: x["changePercent"], reverse=True)


# ---------------------------------------------------------------- refresh
def refresh(args):
    t = now_et()
    today_iso = today_et().isoformat()
    if not args.force and (t.weekday() >= 5 or
                           not in_window(t, config.REFRESH_START, config.REFRESH_END)):
        log("outside market hours, nothing to do")
        return

    c = universe.load()
    tech = load_state("technicals.json", {})
    fund = fundamentals.load()
    state = portfolio.load()
    names = universe.names(c)
    for s, f in fund.items():
        if not names.get(s):
            names[s] = f.get("name")
    groups = universe.groups(c)

    movers_syms = set().union(*(g["movers"] for g in groups.values()))
    live = market.quotes(sorted(movers_syms | set(portfolio.symbols(state))
                                | set(BENCH) | {config.FX_SYMBOL}))
    if len(live) < 0.7 * len(movers_syms):
        raise RuntimeError(f"Only got {len(live)} quotes; Yahoo may be blocking. "
                           "Site keeps showing the last good data.")

    data_date = max((live[b]["date"] for b in BENCH if b in live), default=None)
    us_today = live.get("^GSPC", {}).get("date") == today_iso
    tsx_today = live.get("^GSPTSE", {}).get("date") == today_iso
    if not (us_today or tsx_today) and not args.force:
        meta = publish.get("meta")
        if meta and meta.get("dataDate") == data_date:
            log("markets closed today and data already published; skipping")
            return

    fx = (live.get(config.FX_SYMBOL) or {}).get("price")
    if not fx:
        fx = portfolio.current(state).get("fxAtStart") or 0.72
        log(f"WARNING: no CAD/USD quote, falling back to {fx}")

    payload = {}

    # Index cards (same shape as v3.8)
    payload["indices"] = {
        k: {"name": v["name"], "value": live[v["yahoo"]]["price"],
            "change": live[v["yahoo"]]["change"],
            "changePercent": live[v["yahoo"]]["changePercent"]}
        for k, v in config.INDEX_SYMBOLS.items() if v["yahoo"] in live}

    # Movers, sectors, unusual volume, 52-week highs/lows
    movers, extras = {}, {"sectors": {}, "unusualVolume": {}, "newHighs": {}, "newLows": {}}
    for idx, g in groups.items():
        rows = [_stock_row(s, live[s], tech, fund, names, t, today_iso)
                for s in g["movers"] if s in live]
        up = sorted((r for r in rows if r["changePercent"] > 0),
                    key=lambda r: r["changePercent"], reverse=True)
        down = sorted((r for r in rows if r["changePercent"] < 0),
                      key=lambda r: r["changePercent"])
        movers[idx] = {"gainers": [_public(r) for r in up[:config.TOP_MOVERS]],
                       "losers": [_public(r) for r in down[:config.TOP_MOVERS]],
                       "scanned": len(rows), "asOf": t.isoformat()}
        extras["sectors"][idx] = _sectors(rows)
        extras["unusualVolume"][idx] = [_public(r) for r in sorted(
            (r for r in rows if (r["volumeRatio"] or 0) >= config.UNUSUAL_VOLUME_PACE),
            key=lambda r: r["volumeRatio"], reverse=True)[:10]]
        extras["newHighs"][idx] = [_public(r) for r in rows
                                   if r["fiftyTwoWeekHigh"] and r["_dayHigh"] >= r["fiftyTwoWeekHigh"]]
        extras["newLows"][idx] = [_public(r) for r in rows
                                  if r["fiftyTwoWeekLow"] and r["_dayLow"] <= r["fiftyTwoWeekLow"]]
    payload["movers"] = movers
    payload["extras"] = extras | {"asOf": t.isoformat()}

    # Daily picks: once per US trading day, first run after 10:30am ET
    history = load_state(HISTORY_FILE, [])
    have_today = any(h["date"] == today_iso for h in history)
    picks_time = in_window(t, config.PICKS_TIME, (23, 59))
    if (args.picks or (picks_time and us_today)) and not have_today:
        if not tech:
            log("ERROR: no technicals yet; run the Daily prep workflow first")
        else:
            pool_syms = set().union(*(g["pool"] for g in groups.values()))
            extra = sorted(pool_syms - set(live))
            if extra:
                live.update(market.quotes(extra))
            scored = {idx: scoring.score_index(
                idx, g["pool"], tech, fund, tech.get(config.INDEX_SYMBOLS[idx]["yahoo"], {}),
                live, today=today_et()) for idx, g in groups.items()}
            chosen, notes = picks.choose(scored, history, live, names, today_iso)
            history.extend(picks.history_rows(chosen, today_iso, live))
            save_state(HISTORY_FILE, history)
            picks_payload = {
                "timestamp": t.isoformat(), "date": today_iso,
                "disclaimer": config.DISCLAIMER, "algorithm": config.ALGORITHM_NAME,
                "picks": chosen, "notes": notes,
                "universe": {idx: len(r) for idx, r in scored.items()},
            }
            save_state("picks_today.json", picks_payload)
            payload["picks"] = picks_payload
            # Lookup data is split by first letter so the Worker only reads a small piece
            table = _lookup_table(scored, names, t)
            for letter, chunk in _shard(table["stocks"]).items():
                payload[f"scores_{letter}"] = {"asOf": table["asOf"], "stocks": chunk}
            payload["symbols"] = {"asOf": table["asOf"], "count": table["count"],
                                  "symbols": [[v["yahoo"], v["name"], v["indexes"]]
                                              for v in table["stocks"].values()]}
            log("picks locked: " + ", ".join(
                p["symbol"] for cats in chosen.values() for p in cats.values())
                + (f" | notes: {notes}" if notes else ""))

            if portfolio.rebalance(state, scored, live, fx, names, today_et()):
                state = portfolio.load()

    payload["portfolio"] = portfolio.value(state, live, fx, today_et()) | {"asOf": t.isoformat()}
    payload["quotes"] = {s: [round(q["price"], 4), round(q["changePercent"], 3)]
                         for s, q in live.items()}
    payload["meta"] = {
        "version": config.VERSION, "asOf": t.isoformat(),
        "asOfET": t.strftime("%-I:%M %p ET"), "dataDate": data_date,
        "marketOpen": us_today and in_window(t, (9, 30), (16, 0)),
        "delayNote": "Prices are delayed about 15 minutes.",
        "scanned": {idx: m["scanned"] for idx, m in movers.items()},
        "constituentsUpdated": c.get("updated"),
    }
    publish.put(payload, dry=args.dry)


def _shard(stocks):
    out = {}
    for k, v in stocks.items():
        out.setdefault(k[0].upper(), {})[k] = v
    return out


def _lookup_table(scored, names, t):
    home = ["sp500", "nasdaq", "tsx"]  # which index's ranking to show for dual listings
    table = {}
    for idx in home:
        for s, r in scored.get(idx, {}).items():
            key = s  # Yahoo symbol, so TSX "T.TO" (Telus) and US "T" (AT&T) don't clash
            if key in table:
                table[key]["indexes"].append(idx)
                continue
            best = max(scoring.CATEGORIES, key=lambda k: r["scores"][k] if r["eligible"][k] else -1)
            table[key] = {
                "symbol": display_symbol(s), "yahoo": s, "name": names.get(s) or key,
                "indexes": [idx], "sector": r["sector"], "currency": r["currency"],
                "marketCap": r["marketCap"], "price": r["price"],
                "scores": r["scores"], "eligible": r["eligible"],
                "factors": r["factors"], "metrics": r["metrics"],
                "filtersFailed": r["filtersFailed"], "earningsSoon": r["earningsSoon"],
                "nextEarnings": r["nextEarnings"], "bestCategory": best,
                "reasons": scoring.reasons(r, best),
            }
    return {"asOf": t.isoformat(), "count": len(table), "stocks": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prep", "refresh"])
    ap.add_argument("--force", action="store_true", help="run even outside market hours")
    ap.add_argument("--picks", action="store_true", help="lock today's picks now (if not done)")
    ap.add_argument("--dry", action="store_true", help="don't send to Cloudflare")
    args = ap.parse_args()
    {"prep": prep, "refresh": refresh}[args.command](args)


if __name__ == "__main__":
    main()
