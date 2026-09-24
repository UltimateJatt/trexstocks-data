"""Price check: compare what Yahoo sends with what TrexStocks shows.

    python -m pipeline.check SHOP.TO CURA.TO AAPL

Prints, for each symbol, Yahoo's own quote fields and the last few daily rows,
next to the price and % change TrexStocks calculates. Used to track down
mismatches; it changes nothing and publishes nothing.
"""
import sys
from datetime import datetime, timezone

from . import market


def _t(ts):
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def show(sym):
    print(f"\n==================== {sym} ====================")
    for rng in ("1d", "5d"):
        j = market._fetch_chart(sym, rng)
        if not j:
            print(f"[{rng}] no data from Yahoo")
            continue
        res = j["chart"]["result"][0]
        m = res.get("meta") or {}
        print(f"[{rng}] exchange={m.get('exchangeName')} currency={m.get('currency')} "
              f"instrument={m.get('instrumentType')}")
        print(f"[{rng}] regularMarketPrice={m.get('regularMarketPrice')} at {_t(m.get('regularMarketTime'))}")
        print(f"[{rng}] previousClose={m.get('previousClose')} chartPreviousClose={m.get('chartPreviousClose')}")
        print(f"[{rng}] dayHigh={m.get('regularMarketDayHigh')} dayLow={m.get('regularMarketDayLow')} "
              f"volume={m.get('regularMarketVolume')}")
        ts = res.get("timestamp") or []
        q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        closes = q.get("close") or []
        for t, c in list(zip(ts, closes))[-4:]:
            print(f"[{rng}]   candle {_t(t)} close={c}")
    qt = market.quotes([sym]).get(sym)
    if qt:
        print(f"TrexStocks shows: price={qt['price']:.2f} prevClose={qt['prevClose']:.2f} "
              f"change={qt['changePercent']:+.2f}% date={qt['date']}")
    else:
        print("TrexStocks shows: no quote")


def beta_check(syms):
    """Beta diagnostics: are stock and index days lined up correctly?"""
    import pandas as pd
    benches = ["^GSPC", "^GSPTSE"]
    hist = market.history(list(syms) + benches, period="1y")
    for sym in syms:
        b = "^GSPTSE" if sym.endswith(".TO") else "^GSPC"
        s, x = hist.get(sym), hist.get(b)
        print(f"\n==================== BETA {sym} vs {b} ====================")
        if s is None or x is None:
            print("missing data")
            continue
        print(f"rows: stock={len(s)} index={len(x)}")
        print(f"only in stock: {[d.date().isoformat() for d in s.index.difference(x.index)][:10]}")
        print(f"only in index: {[d.date().isoformat() for d in x.index.difference(s.index)][:10]}")
        print("last 4 closes stock:", [(d.date().isoformat(), round(v, 2)) for d, v in s["Close"].tail(4).items()])
        print("last 4 closes index:", [(d.date().isoformat(), round(v, 2)) for d, v in x["Close"].tail(4).items()])
        for col in ("Close", "AdjClose"):
            rs, rx = s[col].pct_change(), x[col].pct_change()
            both = pd.concat([rs, rx], axis=1, join="inner").dropna()
            both.columns = ["s", "x"]
            corr = {lag: round(float(both["s"].corr(both["x"].shift(lag))), 3) for lag in (-1, 0, 1)}
            beta = both["s"].cov(both["x"]) / both["x"].var()
            wk = pd.concat([s[col].resample("W-FRI").last().pct_change(),
                            x[col].resample("W-FRI").last().pct_change()], axis=1).dropna()
            wbeta = wk.iloc[:, 0].cov(wk.iloc[:, 1]) / wk.iloc[:, 1].var()
            print(f"[{col}] n={len(both)} corr by lag {corr} daily beta={beta:.2f} weekly beta={wbeta:.2f}")


def main():
    syms = sys.argv[1:] or ["SHOP.TO", "CURA.TO", "RY.TO", "SHOP", "AAPL"]
    if syms and syms[0] == "beta":
        beta_check(syms[1:] or ["KO", "AAPL", "CNQ.TO", "BCE.TO"])
        return
    for s in syms:
        try:
            show(s)
        except Exception as e:
            print(f"{s}: error {e}")


if __name__ == "__main__":
    main()
