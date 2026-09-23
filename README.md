# TrexStocks data pipeline

Feeds https://trexstocks.com. Runs on GitHub Actions and publishes results to Cloudflare KV,
where the `trexstocks-api` Worker reads them.

| Workflow | When | What it does |
|---|---|---|
| Daily prep | 8:05am ET, weekdays | Refreshes index member lists (weekly), 1 year of prices, track record, fundamentals |
| Market refresh | Every 15 min, 9:25am to 4:45pm ET | Prices, top movers, sectors, unusual volume, portfolio value; locks daily picks after 10:30am; rebalances the portfolio on its due date |

Settings (filters, category rules, factor weights) live in `pipeline/config.py`.
Saved state (picks history, portfolio, member lists) lives in `data/state/`.

For educational purposes only. Not financial advice.
