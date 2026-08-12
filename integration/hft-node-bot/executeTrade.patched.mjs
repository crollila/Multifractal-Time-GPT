/**
 * Drop-in replacement for `executeTrade` in ExaltedBotV1.0/s.mjs.
 *
 * WHAT CHANGES
 * ------------
 * 1. Before sizing, ask the regime service how big this trade should be.
 * 2. If it answers, use its quantity, limit price and holding horizon.
 * 3. If it says `flat`, skip the trade - the overlay declined it on purpose.
 * 4. If it does not answer in time, fall through to the ORIGINAL logic
 *    unchanged. The bot never stops trading because the overlay is down.
 *
 * WHAT DOES NOT CHANGE
 * --------------------
 * The liquidity filter, the RegT cap, order type, extended-hours handling and
 * every existing safety check are preserved exactly. This is an overlay on
 * sizing, not a rewrite of execution.
 *
 * The original threshold logic is kept verbatim as `legacySizing` so the two
 * paths stay comparable and you can A/B them by flipping REGIME_ENABLED.
 */

import { getRegimeSizing } from './regime.mjs';

const REGIME_ENABLED = process.env.REGIME_ENABLED !== 'false';

// ---------------------------------------------------------------------------
// Original sizing rules, lifted unchanged from s.mjs so the fallback path is
// byte-for-byte the behaviour you have today.
// ---------------------------------------------------------------------------
function determineQuantity(score, Y, current_price, equity) {
  let factor = 0;
  if (score === 100) factor = 19;
  else if (score >= 90) factor = 14;
  else if (score >= 80) factor = 6;
  else if (score >= 70) factor = 3;
  const maxQuantity = (equity * 2 - equity) / current_price;
  return Math.min((Y * factor) / current_price, maxQuantity);
}

function determineShortQuantity(score, Y, current_price, multiplier, equity) {
  const maxQuantity = (equity * 2 - equity) / current_price;
  return Math.min(Math.floor((Y * multiplier) / current_price), maxQuantity);
}

function legacySizing(score, Y, current_price, equity) {
  if (score >= 70) {
    return {
      action: 'buy',
      quantity: determineQuantity(score, Y, current_price, equity),
      limit_price: Number((current_price * 1.007).toFixed(2)),
    };
  }
  if (score <= 30) {
    let multiplier = 1;
    if (score <= 30) multiplier = 2;
    if (score <= 20) multiplier = 4;
    if (score <= 10) multiplier = 9;
    if (score === 0) multiplier = 15;
    return {
      action: 'short',
      quantity: Math.floor(determineShortQuantity(score, Y, current_price, multiplier, equity)),
      limit_price: Number((current_price * 0.993).toFixed(2)),
    };
  }
  return { action: 'hold', quantity: 0, limit_price: current_price };
}

// ---------------------------------------------------------------------------
// Patched executeTrade
// ---------------------------------------------------------------------------
export async function executeTrade(alpaca, symbol, score, helpers) {
  const {
    fetchLastTradingDayVolume,
    getCurrentStockPrice,
    sellPosition,
    tradeValueThreshold = 5_000_000,
  } = helpers;

  if (!symbol.match(/^[A-Z]+$/)) {
    console.log(`Skipping ${symbol}: not a plain equity ticker.`);
    return;
  }
  if (Number.isNaN(score)) {
    console.log(`Score for ${symbol} is not a number. No trade executed.`);
    return;
  }

  // --- unchanged liquidity gate -------------------------------------------
  const lastDayVolume = await fetchLastTradingDayVolume(symbol);
  if (lastDayVolume === 0) {
    console.log(`No trading volume available for ${symbol}.`);
    return;
  }

  const account = await alpaca.getAccount();
  const equity = parseFloat(account.equity);
  const positionMarketValue = parseFloat(account.position_market_value) || 0;
  const Y = equity / 500;

  const current_price = await getCurrentStockPrice(symbol);
  if (!current_price) {
    console.error(`Error fetching price for ${symbol}`);
    return;
  }

  const tradeValue = lastDayVolume * current_price;
  if (tradeValue < tradeValueThreshold) {
    console.log(`Trade value of ${symbol} below threshold. No trade executed.`);
    return;
  }

  let currentQty = 0;
  try {
    const position = await alpaca.getPosition(symbol);
    currentQty = parseFloat(position.qty);
  } catch {
    currentQty = 0; // no existing position
  }

  // --- NEW: regime-conditioned sizing --------------------------------------
  let action = 'hold';
  let quantity = 0;
  let limit_price = current_price;
  let sizedBy = 'legacy';
  let horizonBars = null;

  const decision = REGIME_ENABLED
    ? await getRegimeSizing({
        symbol,
        score,
        price: current_price,
        equity,
        currentQty,
        grossExposure: positionMarketValue,
      })
    : null;

  if (decision) {
    if (decision.side === 'flat') {
      // The overlay actively declined. Respect it - this is the whole point:
      // not every headline deserves a trade.
      console.log(
        `[regime] ${symbol} SKIPPED in ${decision.regime} regime: ${decision.reason}`
      );
      return;
    }
    action = decision.side === 'cover' ? 'buy' : decision.side === 'sell' ? 'sell' : decision.side;
    quantity = Math.abs(Math.trunc(decision.target_qty));
    limit_price = Number(decision.limit_price.toFixed(2));
    horizonBars = decision.horizon_bars;
    sizedBy = 'regime';
    console.log(
      `[regime] ${symbol} regime=${decision.regime} ` +
      `sigma_h=${decision.sigma_horizon.toFixed(4)} edge_z=${decision.edge_z.toFixed(3)} ` +
      `-> ${action} ${quantity} @ ${limit_price} (hold ~${horizonBars} bars, ` +
      `${decision.latency_us.toFixed(0)}us)`
    );
  } else {
    const legacy = legacySizing(score, Y, current_price, equity);
    action = legacy.action;
    quantity = Math.abs(Math.trunc(legacy.quantity));
    limit_price = Number(legacy.limit_price.toFixed(2));
    if (REGIME_ENABLED) {
      console.log(`[regime] ${symbol} service unavailable - using legacy sizing`);
    }
  }

  if (action === 'hold' || quantity <= 0) {
    console.log(`No action for ${symbol} with score ${score} (sized by ${sizedBy}).`);
    return;
  }

  // A short means: flatten any long first, then sell short.
  if (action === 'short' && currentQty > 0) {
    await sellPosition(symbol, currentQty, Number((current_price * 0.99).toFixed(2)));
  }

  const side = action === 'buy' ? 'buy' : 'sell';
  try {
    await alpaca.createOrder({
      symbol,
      qty: quantity,
      side,
      type: 'limit',
      limit_price,
      time_in_force: 'day',
      extended_hours: true,
    });
    console.log(
      `Order successful (${sizedBy}): ${action} ${quantity} ${symbol} @ $${limit_price}`
    );
  } catch (error) {
    console.error(`Error executing ${action} for ${symbol}:`, error.message);
  }
}
