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


def main():
    syms = sys.argv[1:] or ["SHOP.TO", "CURA.TO", "RY.TO", "SHOP", "AAPL"]
    for s in syms:
        try:
            show(s)
        except Exception as e:
            print(f"{s}: error {e}")


if __name__ == "__main__":
    main()
