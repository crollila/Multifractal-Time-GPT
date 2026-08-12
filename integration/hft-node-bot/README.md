# Wiring the regime overlay into the existing Node bot

This folder patches
[`High-Frequency-Trading-Algorithm-with-Instant-News-Sentiment-Analysis`](https://github.com/crollila/High-Frequency-Trading-Algorithm-with-Instant-News-Sentiment-Analysis)
so that every news trade is sized against an MSM volatility regime instead of a
fixed score bucket.

**Nothing in the existing bot is removed.** The overlay is additive and fails
open: if the Python service is slow or down, `s.mjs` falls back to exactly the
sizing it uses today.

---

## What actually changes

| Area | Today | After the patch |
|---|---|---|
| Position size | `equity/500 × bucket(score)`, ignores volatility | `risk_budget × equity / σ_horizon`, MSM forecast |
| Entry rule | hard cutoffs at 70 / 45 / 30 | continuous edge with a per-regime gate |
| Holding period | implicit, until a stop or trailing rule fires | explicit horizon returned per trade |
| Crisis-regime news | traded identically to calm news | down-weighted or declined on calibrated evidence |
| `scores.txt` | `id, TICKER, score` — **not backtestable** | adds ISO timestamp, headline and latency |

The last row matters more than it looks. Without a timestamp you cannot join a
signal to a price, so none of the current score history can be used to measure
anything. Fixing the log format is what makes every future calibration possible.

---

## Step 1 — start the regime service

From the repository root:

```bash
pip install -e ".[service]"
```

```bash
python -m mtgpt.cli serve --host 127.0.0.1 --port 8000
```

Verify it is up:

```bash
curl -s http://127.0.0.1:8000/health
```

## Step 2 — copy two files into the bot

Copy `regime.mjs` into `ExaltedBotV1.0/` next to `s.mjs`. It needs Node 18+ and
adds **no npm dependencies** — it uses the built-in `fetch`.

```bash
cp regime.mjs /path/to/High-Frequency-Trading-Algorithm.../ExaltedBotV1.0/
```

## Step 3 — warm up your universe at start-up

MSM estimation takes seconds, so it must happen once at boot, never inside the
news path. Add near the top of `s.mjs`, after the `alpaca` client is created:

```js
import { warmUp, isHealthy, pushBar } from './regime.mjs';

const REGIME_UNIVERSE = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT'];

async function initRegimes() {
  if (!(await isHealthy())) {
    console.warn('[regime] service unreachable - bot will use legacy sizing');
    return;
  }
  for (const symbol of REGIME_UNIVERSE) {
    await warmUp(symbol, { days: 30 });
  }
}
await initRegimes();
```

## Step 4 — keep the filter fed

The regime estimate is only as fresh as the last bar it saw. Add one line to
the existing `updateAndCheckPositions` loop, which already runs every 10s:

```js
// inside the Object.keys(currentPositions).map(async (symbol) => { ... }) body
pushBar(symbol, current_price).catch(() => {});   // fire and forget
```

## Step 5 — replace `executeTrade`

`executeTrade.patched.mjs` in this folder is a complete replacement. It keeps
the liquidity filter, the RegT cap, order type and extended-hours handling
byte-for-byte, and preserves the original bucket sizing as `legacySizing` for
the fallback path.

Either import it, or paste its body over the existing `executeTrade` in
`s.mjs`. If you import it, the call site becomes:

```js
import { executeTrade } from './executeTrade.patched.mjs';

await executeTrade(alpaca, ticker, score, {
  fetchLastTradingDayVolume,
  getCurrentStockPrice,
  sellPosition,
});
```

## Step 6 — make `scores.txt` backtestable

In `News/news.mjs`, replace `saveScores` with the version below. It adds the
publication timestamp, the headline and the measured news-to-score latency.

```js
function saveScores(tickers, sentimentScore, article) {
  const now = new Date();
  const published = article?.created ? new Date(article.created) : now;
  const latencyMs = now.getTime() - published.getTime();
  const headline = (article?.title || '').replace(/"/g, "'").replace(/[\r\n,]+/g, ' ').trim();

  const scoreEntries = sentimentScore.split('\n').map(line => {
    const parts = line.split(':');
    if (parts.length !== 2) return null;
    const ticker = parts[0].trim();
    const score = parseInt(parts[1].trim(), 10);
    if (ticker.includes('$') || isNaN(score)) return null;
    return `${currentScoreId++},${ticker},${score},${published.toISOString()},` +
           `"${headline}",benzinga,${latencyMs}\n`;
  }).filter(Boolean);

  try {
    fs.appendFileSync(SCORES_FILE, scoreEntries.join(''));
    saveCurrentScoreId(currentScoreId);
  } catch (error) {
    console.error('Error writing to scores file:', error);
  }
}
```

Then pass the article through at the call site:

```js
await saveScores(stockTickers, sentimentScore, news);
```

`s.mjs`'s `readScores` keeps working unchanged — it splits on commas and reads
the first three fields, which are still `id, ticker, score`. The parser on the
Python side (`mtgpt.signals.news.parse_score_line`) reads **both** the old and
new formats, so historical lines are not lost.

> One caveat on the quoted headline: `readScores` in `s.mjs` does a naive
> `line.split(',')`. A comma inside the quoted headline would shift later
> fields — harmless today because `s.mjs` only reads fields 0-2, but strip
> commas from headlines (as above) rather than relying on that.

---

## Step 7 — A/B it before you trust it

```bash
REGIME_ENABLED=false node s.mjs   # today's behaviour
REGIME_ENABLED=true  node s.mjs   # overlay on
```

Run paper accounts side by side. Do not judge on P&L until you have a few
hundred events **per regime** — see the power analysis in `docs/FINDINGS.md`,
which puts the requirement at roughly 550-850 events per regime to resolve a
meaningful difference. Below that you are reading noise.

---

## Failure behaviour, explicitly

| Condition | What happens |
|---|---|
| Service down | one 25ms timeout, then legacy sizing; breaker opens after 5 failures |
| Breaker open | no call at all for 30s, legacy sizing, zero added latency |
| Symbol not warmed up | HTTP 404 → treated as unavailable → legacy sizing |
| `side: "flat"` | trade **skipped** — this is a decision, not a failure |
| `failClosed: true` | throws instead of falling back, if you prefer to stand down |

Tune with `configureRegime({ timeoutMs, failClosed, breakerThreshold })` or the
`REGIME_SERVICE_URL`, `REGIME_TIMEOUT_MS` and `REGIME_ENABLED` env vars.

---

## Two things to fix in the bot regardless of this overlay

Both were noticed while reading `s.mjs` and are independent of the regime work:

1. **`main()` and `updateAndCheckPositions()` are each started twice.** At the
   bottom of `s.mjs`, `retryOperation(main)` is called at line ~74 and again at
   the end, and `updateAndCheckPositions()` runs immediately *and* on a 10s
   interval. Two `fs.watch` handlers on `scores.txt` means every score can be
   processed twice; `isProcessingScores` narrows the window but does not close
   it, because the guard is released before the second watcher fires.

2. **`readScores` filters on a stale `lastProcessedId`.** The module-level
   `lastProcessedId` is read once at import, while `processNewScores` re-reads
   the file's value into a local. The filter inside `readScores` uses the stale
   module-level copy, so the two disagree after the first batch.
