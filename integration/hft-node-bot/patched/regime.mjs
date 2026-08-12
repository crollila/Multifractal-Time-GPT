/**
 * regime.mjs - regime-conditioned sizing client for the news trading bot.
 *
 * Drop this file next to `s.mjs` and import it. No new npm dependencies:
 * it uses the global `fetch` and `AbortController` built into Node 18+.
 *
 * DESIGN RULE: FAIL OPEN.
 * ------------------------------------------------------------------
 * Every function here returns `null` on any problem - service down, slow,
 * malformed reply, symbol not warmed up - and the caller falls back to the
 * bot's existing sizing. An overlay that can halt trading is a worse risk than
 * the one it removes. If you would rather stand down than trade unsized, set
 * `failClosed: true` and handle the thrown error at the call site.
 *
 * LATENCY.
 * ------------------------------------------------------------------
 * The Python service answers /size in well under a millisecond; the default
 * 25ms timeout is budget for the network round trip, not for its compute. A
 * circuit breaker stops calling entirely after repeated failures so a dead
 * service costs one timeout, not one per headline.
 */

const DEFAULTS = {
  baseUrl: process.env.REGIME_SERVICE_URL || 'http://127.0.0.1:8000',
  timeoutMs: Number(process.env.REGIME_TIMEOUT_MS || 25),
  failClosed: false,
  breakerThreshold: 5,   // consecutive failures before the breaker opens
  breakerCooldownMs: 30_000,
  verbose: true,
};

let config = { ...DEFAULTS };
let consecutiveFailures = 0;
let breakerOpenUntil = 0;

/** Override defaults once at start-up. */
export function configureRegime(overrides = {}) {
  config = { ...config, ...overrides };
  return { ...config };
}

function log(...args) {
  if (config.verbose) console.log('[regime]', ...args);
}

function breakerIsOpen() {
  if (Date.now() < breakerOpenUntil) return true;
  if (breakerOpenUntil !== 0 && Date.now() >= breakerOpenUntil) {
    log('circuit breaker closing, retrying service');
    breakerOpenUntil = 0;
    consecutiveFailures = 0;
  }
  return false;
}

function recordFailure(reason) {
  consecutiveFailures += 1;
  if (consecutiveFailures >= config.breakerThreshold && breakerOpenUntil === 0) {
    breakerOpenUntil = Date.now() + config.breakerCooldownMs;
    log(`circuit breaker OPEN for ${config.breakerCooldownMs}ms after ` +
        `${consecutiveFailures} failures (last: ${reason})`);
  }
}

async function request(path, { method = 'GET', body = null } = {}) {
  if (breakerIsOpen()) return null;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const response = await fetch(`${config.baseUrl}${path}`, {
      method,
      signal: controller.signal,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      recordFailure(`HTTP ${response.status}`);
      return null;
    }
    consecutiveFailures = 0;
    return await response.json();
  } catch (error) {
    // AbortError means we blew the latency budget: treat it exactly like any
    // other failure and let the caller use its own sizing.
    recordFailure(error.name === 'AbortError' ? 'timeout' : error.message);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ask the service how large this news trade should be.
 *
 * @returns {Promise<object|null>} A decision object, or null to fall back.
 *   Shape: { side, target_qty, target_notional, limit_price, horizon_bars,
 *            regime, edge, edge_z, sigma_horizon, reason, latency_us }
 *   `side` is one of buy | sell | short | cover | flat.
 *   A `flat` side means the overlay actively declined the trade - that is a
 *   decision, not a failure, and should be respected.
 */
export async function getRegimeSizing({
  symbol,
  score,
  price,
  equity,
  currentQty = 0,
  grossExposure = 0,
  stalenessWeight = 1.0,
}) {
  if (!symbol || !Number.isFinite(price) || price <= 0 || !Number.isFinite(equity)) {
    return null;
  }
  const decision = await request('/size', {
    method: 'POST',
    body: {
      symbol,
      score,
      price,
      equity,
      current_qty: currentQty,
      gross_exposure: grossExposure,
      staleness_weight: stalenessWeight,
    },
  });

  if (decision === null) {
    if (config.failClosed) {
      throw new Error(`regime service unavailable for ${symbol} and failClosed is set`);
    }
    return null;
  }
  return decision;
}

/** Current regime for a symbol, for logging and dashboards. */
export async function getRegime(symbol) {
  return request(`/regime/${encodeURIComponent(symbol)}`);
}

/**
 * Advance a symbol's volatility filter with a new closing price.
 * Call this once per bar per symbol you hold or watch. Cheap and fire-and-forget.
 */
export async function pushBar(symbol, close) {
  return request(`/bars/${encodeURIComponent(symbol)}`, {
    method: 'POST',
    body: { close },
  });
}

/**
 * Fit MSM for a symbol. Slow (seconds) - run at start-up, never inside the
 * news path. `source: 'synthetic'` works with no market-data credentials.
 */
export async function warmUp(symbol, { source = 'alpaca', days = 30, kComponents = 5 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120_000);
  try {
    const response = await fetch(`${config.baseUrl}/warmup`, {
      method: 'POST',
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, source, days, k_components: kComponents }),
    });
    if (!response.ok) {
      log(`warmup failed for ${symbol}: HTTP ${response.status}`);
      return null;
    }
    const payload = await response.json();
    log(`warmed up ${symbol}: ${payload.bars_seen} bars, ` +
        `m0=${payload.msm_params.m0.toFixed(3)} sigma=${payload.msm_params.sigma.toExponential(2)}`);
    return payload;
  } catch (error) {
    log(`warmup error for ${symbol}: ${error.message}`);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** True if the service is reachable right now. */
export async function isHealthy() {
  const health = await request('/health');
  return health !== null && health.status === 'ok';
}

export function breakerStatus() {
  return {
    open: Date.now() < breakerOpenUntil,
    consecutiveFailures,
    reopensAt: breakerOpenUntil ? new Date(breakerOpenUntil).toISOString() : null,
  };
}
