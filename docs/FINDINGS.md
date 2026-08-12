# Findings

Everything here comes from `mtgpt demo --bars 80000 --event-rate 30 --seed 7`.
Raw output is in [`demo_output.txt`](demo_output.txt), machine-readable summary
in [`demo_summary.json`](demo_summary.json).

> **These are synthetic results.** They demonstrate that the *machinery* works —
> that it finds a regime effect when one exists and refuses to invent one when
> it does not. They say nothing about whether real news markets behave this way.
> That question needs a real event tape, which is why fixing the `scores.txt`
> format is step one of the integration.

---

## The setup

Two scenarios, identical in every respect except the ground truth.

| | Scenario A | Scenario B (null) |
|---|---|---|
| CALM response | 1.10 | 0.70 |
| NORMAL response | 0.80 | 0.70 |
| TURBULENT response | 0.35 | 0.70 |
| CRISIS response | **−0.40** | 0.70 |

"Response" is the drift a maximally informative headline produces, in units of
horizon volatility. Scenario A encodes the hypothesis that news diffuses slowly
in quiet tape but overshoots and reverts in crisis. Scenario B says regime is
irrelevant. Both run 80,000 bars, ~2,400 events, walk-forward with the first
half for fitting and the second for evaluation.

---

## 1. Calibration recovers the effect — and correctly reports its uncertainty

Scenario A, fitted on training events only:

| regime | fitted β | raw β | s.e. | t | shrinkage | n | **true** |
|---|---|---|---|---|---|---|---|
| CALM | 0.786 | 0.875 | 0.170 | 5.15 | 0.82 | 149 | 1.10 |
| NORMAL | 0.507 | 0.521 | 0.117 | 4.45 | 0.91 | 339 | 0.80 |
| TURBULENT | 0.516 | 0.531 | 0.120 | 4.43 | 0.90 | 273 | 0.35 |
| CRISIS | **−0.022** | −0.059 | 0.112 | **−0.53** | 0.91 | 372 | −0.40 |

The ordering is right and, most importantly, **CRISIS is correctly identified as
having no reliable positive response** (t = −0.53). The estimate does not reach
the true −0.40 — with 372 events and a standard error of 0.11 it cannot — but it
gets close enough to zero that the strategy stops trading crisis news entirely.

Contrast the null:

| regime | fitted β | raw β | t | shrinkage | true |
|---|---|---|---|---|---|
| CALM | 0.623 | 0.602 | 3.95 | 0.35 | 0.70 |
| NORMAL | 0.540 | 0.427 | 3.48 | 0.46 | 0.70 |
| TURBULENT | 0.630 | 0.625 | 5.32 | 0.48 | 0.70 |
| CRISIS | 0.742 | 0.840 | 7.78 | 0.52 | 0.70 |

Spread collapses from 0.81 (A) to 0.20 (B), and the empirical-Bayes shrinkage
weights halve — 0.82–0.91 in A versus 0.35–0.52 in B — because the observed
between-regime spread is barely larger than sampling noise. The calibrator
recognises the null as a null.

---

## 2. Out-of-sample: the regime split earns its keep, but not the way expected

**Scenario A**

| strategy | trades | net P&L | Sharpe | 95% CI | hit | maxDD | ret/trade t |
|---|---|---|---|---|---|---|---|
| `legacy_threshold` | 741 | 9,487 | 5.55 | [2.26, 8.78] | 57.8% | 0.5% | 2.73 |
| `pooled_vol_target` | 939 | 21,821 | 4.17 | [0.95, 7.40] | 55.0% | 1.7% | 1.14 |
| **`regime_fixed_horizon`** | **611** | **30,181** | **7.87** | [4.72, 11.13] | 57.8% | 0.4% | **4.69** |
| `regime_conditioned` | 611 | 28,929 | 7.58 | [4.47, 10.87] | 57.1% | 0.4% | 4.53 |

**Scenario B (null)**

| strategy | trades | net P&L | Sharpe |
|---|---|---|---|
| `legacy_threshold` | 741 | 31,354 | 17.64 |
| `pooled_vol_target` | 939 | 85,315 | 15.23 |
| `regime_fixed_horizon` | 939 | **85,394** | 15.23 |
| `regime_conditioned` | 939 | 67,820 | 13.25 |

In the null, `regime_fixed_horizon` and `pooled_vol_target` differ by **0.09%**
in P&L and are identical in Sharpe to three significant figures. The regime
split adds nothing when there is nothing to find. That is the result that makes
Scenario A believable.

### The surprise: vol targeting alone made things *worse*

`pooled_vol_target` more than doubled P&L over legacy but **dropped Sharpe from
5.55 to 4.17** and tripled drawdown. It deployed 4x the notional and got paid
proportionally, not more than proportionally.

So the gain in `regime_fixed_horizon` is **not** mostly risk normalisation, which
is the opposite of what I expected going in. Decomposing:

- Vol targeting alone: Sharpe 5.55 → 4.17 (**worse**)
- Adding the regime split: 4.17 → 7.87 (**the entire gain, and then some**)

The mechanism is visible in the by-regime table. Legacy lost money in CRISIS
(−1,328 P&L, 263 trades, t = −0.69) while making it everywhere else. The
regime-conditioned version placed **zero crisis trades** and took 130 fewer
trades overall while making 3.2x the money. **The edge is in the trades it
declines to take, not in how it sizes the ones it takes.**

That is a direct answer to the original question, and it is a more useful answer
than "size by volatility" would have been.

---

## 3. Adaptive holding periods hurt

`regime_conditioned` (adaptive horizon) underperformed `regime_fixed_horizon` in
both scenarios, and in the null the shortfall was **statistically significant**
(difference −0.00106 per trade, 95% CI [−0.00183, −0.00030]).

The reason is a genuine modelling error worth stating plainly: **regime
persistence and alpha decay are different clocks.** MSM tells you when
volatility will mean-revert. It says nothing about when the news drift has
finished arriving. Cutting a 30-bar drift at bar 12 because the vol regime is
expected to break forfeits alpha while paying the identical round-trip cost.

`adapt_horizon_to_regime` therefore defaults to **`False`**. Turn it on only
after measuring your own alpha-decay curve per regime.

---

## 4. How much data you actually need

Inverting the calibration standard errors, to resolve a true β gap of 0.3
between two regimes at 80% power:

| regime | events observed | events required | shortfall |
|---|---|---|---|
| CALM | 149 | 751 | 602 |
| NORMAL | 339 | 811 | 472 |
| TURBULENT | 273 | 686 | 413 |
| CRISIS | 372 | 819 | 447 |

**Roughly 700–850 events per regime, so on the order of 3,000 scored events
overall.** At the upstream bot's rate that is months of logging — which is the
strongest practical argument for fixing the `scores.txt` timestamp format
immediately, whatever else gets built.

---

## 5. Honest caveats

1. **The headline comparison is not statistically significant.** The bootstrap
   on *per-trade* return gives +0.00029 with CI [−0.00035, +0.00092] — it
   straddles zero. The Sharpe improvement is real at the equity-curve level and
   the t-statistic on per-trade return roughly doubles (2.73 → 4.69), but at
   n ≈ 600 trades the per-trade means are not separable. Section 4 is the
   quantitative version of this caveat.

2. **The concentration cap binds on 83% of regime-conditioned trades**
   (100% in CALM, 98% in NORMAL, 62% in TURBULENT, per the attribution table).
   Vol targeting is switched off wherever the cap binds, so within calm and
   normal tape the strategy is effectively flat-sized. Raising
   `max_position_fraction` or lowering `risk_budget` would let the mechanism
   breathe, at the cost of more single-name concentration.

3. **Scenario B's Sharpe of 15–17 is not a real number.** It is one synthetic
   symbol with a strong embedded signal, no cross-sectional risk, no borrow
   constraints, no market impact beyond a fixed 3bp, and no regime in which the
   data-generating process changes. Read the *differences* between rows, never
   the levels.

4. **The null is not perfectly flat** — β ranges 0.54 to 0.74 where the truth is
   0.70. Part of this is sampling noise, but part is a real bias: the
   calibration normalises by the *forecast* volatility, and where MSM
   under-forecasts (typically crisis), realised/forecast is inflated. A
   realised-volatility normalisation would remove it at the cost of look-ahead.
   Worth revisiting.

5. **Regime shares drift from their targets** in any finite sample. MSM
   volatility is so persistent that 80,000 bars contain only a few dozen
   independent regime episodes; one 9,000-bar test sample came out 56% CRISIS.
   Regime frequencies are noisy even when the classifier is correct.

6. **MSM's `gamma_1` and `b` are only weakly identified** individually — they
   trade off against one another. Only the implied `gamma_k` schedule is pinned
   down, which is all the forecast uses, but do not read economic meaning into
   either parameter alone.

---

## What I would do next, in order

1. **Fix the log format and wait.** Nothing else can be validated on real data
   until timestamped scores exist. This is the highest-value change in the whole
   repo and it is ten lines of JavaScript.
2. **Re-run this exact ablation on the real tape** once ~3,000 events have
   accumulated. The framework is the deliverable; these synthetic numbers are
   only its unit test.
3. **Measure alpha decay per regime** directly, then reconsider adaptive
   horizons on that evidence rather than on volatility persistence.
4. **Check whether the crisis effect survives** as a liquidity story rather than
   an overreaction story — spreads widen in crisis, and some of what looks like
   reversal may just be paying the spread twice. Costs are already modelled, but
   with a flat 3bp that does not widen by regime. A regime-dependent cost model
   is the obvious refinement and could plausibly explain the whole CRISIS effect.
