# Method

The full chain, and why each link is built the way it is.

---

## 1. The volatility model: binomial MSM

Calvet & Fisher (2004). Returns are

```
r_t     = sigma_t * eps_t,              eps_t ~ N(0, 1)
sigma_t = sigma_bar * sqrt(M_1t * M_2t * ... * M_Kt)
```

Each multiplier `M_k` is drawn from `{m0, 2 - m0}` with equal probability, so
`E[M] = 1` and `sigma_bar` is the unconditional volatility. Component `k` is
redrawn each bar with probability

```
gamma_k = 1 - (1 - gamma_1) ** (b ** (k - 1)),      b > 1
```

so `k = 1` is the slowest scale and `k = K` the fastest. Four free parameters
(`m0`, `sigma_bar`, `gamma_1`, `b`) describe volatility dynamics across `K`
simultaneous time scales.

### Why not GARCH

GARCH has one memory parameter, so it has one decay rate. Volatility in real
tape does not have one decay rate — a Fed headline and a fat-finger print both
raise volatility, but one persists for days and the other for minutes. MSM
represents both at once, and the per-component posteriors let you tell them
apart in real time. For a news strategy that distinction is the difference
between "hold through this" and "get out".

### Estimation

Exact maximum likelihood via the Hamilton filter over `2**K` states. The
transition matrix is a Kronecker product of `2x2` blocks, which keeps `K <= 8`
(256 states) cheap: 20,000 observations filter in about 60ms.

`K` is not estimated — it is a modelling choice about how many scales matter.
Sweep it with `mtgpt fit --select-k` and take the BIC minimum.

Known identification caveat, confirmed by the recovery test in
`tests/test_msm.py`: `m0` and `sigma_bar` are recovered tightly, while
`gamma_1` and `b` trade off against each other and are only weakly identified
individually. What is identified is the *implied schedule* of `gamma_k`, which
is all the forecast depends on.

### The forecast that matters

```
E[sigma_{t+h}^2 | F_t]
```

is available in closed form. Since `A_k = (1-g)I + g*P` with `P` idempotent,

```
A_k^h = (1-g)^h I + (1 - (1-g)^h) P
```

so an `h`-step forecast is the one-step operator with `gamma_k` replaced by
`1 - (1-gamma_k)^h` — exact at any horizon, at `O(K * 2**K)` cost.

Better still, the conditional expectation factorises across components:

```
E[prod_k M_k(t+h) | state] = prod_k (a_k * M_k(state) + 1 - a_k),   a_k = (1-gamma_k)^h
```

which collapses the whole forecast to a **single cached dot product**. That is
what makes the live sizing endpoint sub-millisecond rather than ~2ms.

Position sizing uses the **cumulative** variance over the holding period,
`Var[r_{t+1} + ... + r_{t+h}]`, not `h` times the one-step variance. The two
differ sharply right after a news shock, which is exactly when we are trading.

---

## 2. Regimes without look-ahead

Four regimes — CALM, NORMAL, TURBULENT, CRISIS — split at the 25th, 60th and
85th percentiles of conditional volatility.

**Percentiles of what, exactly?** This is where regime studies usually leak the
future. Taking sample quantiles of the realised series labels every historical
bar using information from the whole sample, and any backtest conditioned on
those labels is inflated.

Here the cutoffs come from **simulating the fitted model**: draw a long path
from the estimated parameters, filter it, take quantiles of *that*. The only
input is the parameter vector, which in a walk-forward run was itself estimated
on strictly prior data. No realised future price touches a regime label.

Each snapshot also reports:

- **Survival probability** — `P(volatility still this elevated in h bars)`,
  exact from the `h`-step state distribution.
- **Expected regime bars** — when that probability first falls below one half.
- **Component posteriors** — `P(M_k = high)` per scale, slowest first. This is
  the fast-burst versus structural-shift diagnostic.

---

## 3. Is the asset even multifractal?

MSM assumes multifractality. `mtgpt diagnose` tests it with MF-DFA before you
commit, comparing the singularity-spectrum width `delta_alpha` against the same
series shuffled. Shuffling destroys temporal correlation while preserving the
fat-tailed marginal, so it isolates genuine multiscale structure from mere
kurtosis.

The unit tests pin both directions: MSM-simulated data is flagged multifractal,
and i.i.d. Gaussian noise is not (`delta_alpha` 0.08, excess width negative).

If `h(q)` comes out flat, MSM collapses toward a one-factor stochastic
volatility model and regime conditioning has little left to exploit. Better to
learn that in one command than after a month of paper trading.

---

## 4. The foundation-model layer

MSM says nothing about direction. TimeGPT supplies a conditional **mean** path,
optionally conditioned on the sentiment score as an exogenous regressor. The
two combine into

```
edge_z = expected cumulative return over horizon / MSM volatility over horizon
```

a forecast information ratio, comparable across symbols and regimes.

The default backend is **Theta**, offline and dependency-free. That is
deliberate: it makes the repo run on a fresh clone with no key, and it is the
control that tells you whether TimeGPT is earning its API call.
`mtgpt forecast-bench` runs the horse race on MASE and directional accuracy.

Directional accuracy is the metric to watch. On liquid intraday equity data a
random walk is very hard to beat, and a mean model near 50% directional
accuracy contributes nothing to a trading signal however good its MASE looks.
`foundation_weight` defaults to **0** for that reason — turn it up only when a
benchmark says to.

---

## 5. Fusion: three channels

Regime information reaches the order in three distinct places.

### Size (the main one)

```
notional = equity * risk_budget * clip(edge_z / reference_edge_z, -1, 1) / sigma_horizon
```

Notional is inversely proportional to forecast volatility, so every regime
contributes the same *risk* per trade instead of the same dollar exposure.

**Why a risk budget and not Kelly.** Full Kelly for a bet with expected move
`0.2 * sigma` implies notional of `0.2 / sigma` times equity — around 30x
leverage at intraday volatilities. Even quarter-Kelly pinned every position
against the 5% concentration cap in testing, which silently switched vol
targeting off and made all four regimes take identical size. Two strategies in
the ablation came out byte-identical before this was fixed. The backtest now
reports a **cap-binding rate** so this failure mode cannot hide again.

### Selection

A per-regime edge gate declines marginal signals in regimes where the fitted
response is weak. Defaults are identical across regimes; calibration moves them.

### Horizon

MSM says how long the regime should survive, so the hold can adapt.

**This is off by default, because measurement said it hurts.** See
`FINDINGS.md`. Regime persistence and alpha decay are different clocks:
volatility mean-reverting in 12 bars tells you nothing about whether the news
drift has finished arriving, and cutting a 30-bar drift at bar 12 forfeits
alpha while paying the same round trip.

---

## 6. Calibration, and refusing to invent an effect

Per regime, regress volatility-normalised realised return on the news edge,
through the origin:

```
realised_return / sigma_horizon  =  beta_regime * edge  +  noise
```

Through the origin because a news signal should carry no unconditional drift;
if it does, that is market beta and does not belong in a headline-response
coefficient.

Estimates are shrunk toward the pooled value with the James-Stein weight

```
w_i = tau^2 / (tau^2 + se_i^2)
```

where `tau^2` is the estimated between-regime variance of the true betas —
the observed spread minus the average sampling variance.

The property that matters: **if the spread between regimes is no larger than
estimation noise, `tau^2` goes to zero, every weight goes to zero, and all four
betas collapse to the pooled value.** A calibrator that manufactured
differences from noise would make every backtest in this repo meaningless, so
`test_calibration_returns_flat_betas_under_the_null` pins it directly, and the
synthetic generator ships a `flat_response()` scenario as the end-to-end null.

`power_analysis()` inverts the same standard errors to answer the question you
should ask before committing: **how many events per regime do I need?**

---

## 7. Backtest discipline

- Walk-forward. Parameters, cutoffs and betas are fitted on the first half and
  frozen; the test half is touched once.
- Latency is enforced: a signal inside bar `i` cannot fill before bar `i+1`.
  The synthetic generator front-loads news drift so being late genuinely costs.
- Costs on both sides — half-spread, slippage, commission, and borrow on shorts.
- Every headline number carries a bootstrap interval. Strategy returns are
  autocorrelated whenever positions span bars, so the Sharpe interval uses a
  **stationary block bootstrap**; i.i.d. resampling would report an interval far
  too tight.
- The four-way ablation attributes any gain to a specific channel rather than
  to "the new thing is better".
- `test_shuffled_scores_destroy_the_edge` detaches scores from their events and
  requires the measured response to become insignificant. If the engine were
  peeking at future prices, it would still find an edge in scrambled scores.

---

## References

- Calvet, L. and Fisher, A. (2004). How to Forecast Long-Run Volatility:
  Regime Switching and the Estimation of Multifractal Processes.
  *Journal of Financial Econometrics* 2(1), 49-83.
- Calvet, L. and Fisher, A. (2008). *Multifractal Volatility: Theory,
  Forecasting, and Pricing*. Academic Press.
- Kantelhardt, J. et al. (2002). Multifractal detrended fluctuation analysis of
  nonstationary time series. *Physica A* 316, 87-114.
- Assimakopoulos, V. and Nikolopoulos, K. (2000). The theta model.
  *International Journal of Forecasting* 16(4), 521-530.
- Politis, D. and Romano, J. (1994). The Stationary Bootstrap.
  *JASA* 89(428), 1303-1313.
