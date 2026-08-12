"""Regime x news fusion: the actual research question, made testable.

The hypothesis
--------------
Not all news should be traded the same way. The same sentiment score arriving in
a calm tape and in a crisis tape implies different expected moves, different
risk, and different holding periods. Conditioning on a volatility regime should
therefore beat treating every headline identically.

How this module keeps that honest
---------------------------------
It would be easy — and worthless — to hardcode "news works better when calm".
Instead every regime starts with the **same** parameters
(:func:`default_config`), so the out-of-the-box strategy is a regime-*agnostic*
vol-targeted news trader. That is the control. :func:`calibrate` then estimates a
separate response coefficient per regime **from training data only**, and the
backtest compares control against calibrated out-of-sample. If regime
conditioning adds nothing, the calibrated betas come out equal and the
comparison says so.

The three channels regime information flows through
---------------------------------------------------
1. **Size.** Position notional is proportional to ``1 / sigma_horizon``, where
   the volatility comes from MSM's exact multi-horizon forecast rather than
   trailing realised vol. This alone is most of the value: it equalises risk per
   trade across regimes instead of taking 5x the risk in a crisis.
2. **Selection.** A per-regime edge gate can decline to trade marginal signals
   in regimes where the fitted response is weak or noisy.
3. **Horizon.** MSM says how long the current regime is expected to survive, so
   the holding period adapts instead of being a fixed constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from ..models.foundation import Forecast
from ..models.regimes import Regime, RegimeClassifier, RegimeSnapshot

__all__ = [
    "RegimeRule",
    "FusionConfig",
    "TradeDecision",
    "RegimeConditionedSizer",
    "default_config",
    "calibrate",
    "CalibrationObservation",
]


@dataclass
class RegimeRule:
    """Per-regime parameters. Identical across regimes until calibrated."""

    beta: float = 0.30
    """Expected cumulative return over the horizon, per unit of edge, measured
    in units of horizon volatility. ``beta = 0.3`` means a maximally bullish
    headline (edge = 1) is worth 0.3 horizon-sigmas of expected move. This is
    the one number that carries the regime-conditional alpha, and it should come
    from :func:`calibrate`, not from intuition."""

    min_abs_edge: float = 0.20
    """Ignore scores closer to neutral than this (0.20 == score outside 40-60)."""

    size_multiplier: float = 1.0
    """Discretionary overlay on top of vol targeting. Left at 1.0 by default so
    that risk normalisation is not silently double counted."""

    horizon_bars: int = 30
    max_position_fraction: float = 0.05

    # Populated by calibrate(), for audit rather than for use.
    t_stat: float = float("nan")
    n_observations: int = 0
    raw_beta: float = float("nan")
    standard_error: float = float("nan")
    shrinkage_weight: float = float("nan")
    """Empirical-Bayes weight on this regime's own estimate versus the pooled
    one. Near 1 means the regime had enough clean evidence to speak for itself;
    near 0 means its apparent difference was indistinguishable from noise."""

    def to_dict(self) -> dict:
        return {
            "beta": self.beta,
            "min_abs_edge": self.min_abs_edge,
            "size_multiplier": self.size_multiplier,
            "horizon_bars": self.horizon_bars,
            "max_position_fraction": self.max_position_fraction,
            "t_stat": self.t_stat,
            "n_observations": self.n_observations,
            "raw_beta": self.raw_beta,
            "standard_error": self.standard_error,
            "shrinkage_weight": self.shrinkage_weight,
        }


@dataclass
class FusionConfig:
    rules: dict[Regime, RegimeRule] = field(default_factory=dict)

    risk_budget: float = 0.0003
    """Fraction of equity put at risk, at one horizon-sigma, by a
    maximum-conviction trade. 3bp sounds small; at a 30-bar sigma of 0.7% it
    corresponds to roughly 4% of equity in notional, which is already a large
    single-name concentration.

    Why a risk budget and not Kelly: full Kelly for a bet with expected move
    ``0.2 * sigma`` implies notional of ``0.2 / sigma`` times equity — order 30x
    leverage at intraday volatilities. Even quarter-Kelly pins every position
    against the concentration cap, which silently turns vol targeting off and
    makes every regime take the same size. A risk budget is the formulation
    that actually binds, and it is directly interpretable."""

    reference_edge_z: float = 0.15
    """Forecast information ratio at which the full risk budget is deployed.
    Conviction below this scales the position down linearly."""

    min_edge_z: float = 0.02
    """Minimum forecast information ratio to bother trading."""

    max_position_fraction: float = 0.05
    max_gross_leverage: float = 2.0
    """RegT caps a margin account at 2:1, matching the upstream bot."""

    min_notional: float = 100.0
    allow_short: bool = True

    adapt_horizon_to_regime: bool = False
    """Shorten the hold when MSM says the regime will not survive it.

    Off by default, because measurement says it hurts: regime persistence and
    alpha decay are different clocks. Volatility mean-reverting in 12 bars tells
    you nothing about whether the news drift has finished arriving, and cutting
    a 30-bar drift at bar 12 forfeits alpha while paying the same round-trip
    cost. Turn it on only once you have measured your own alpha-decay curve per
    regime. See ``docs/FINDINGS.md``."""

    news_half_life_seconds: float = 300.0

    foundation_weight: float = 0.0
    """Weight on the foundation-model mean forecast versus the calibrated
    sentiment response. 0 disables the foundation model entirely; the backtest
    ablation is what should set this."""

    def rule(self, regime: Regime) -> RegimeRule:
        return self.rules.get(regime, RegimeRule())

    def to_dict(self) -> dict:
        return {
            "risk_budget": self.risk_budget,
            "reference_edge_z": self.reference_edge_z,
            "min_edge_z": self.min_edge_z,
            "max_position_fraction": self.max_position_fraction,
            "max_gross_leverage": self.max_gross_leverage,
            "min_notional": self.min_notional,
            "allow_short": self.allow_short,
            "adapt_horizon_to_regime": self.adapt_horizon_to_regime,
            "news_half_life_seconds": self.news_half_life_seconds,
            "foundation_weight": self.foundation_weight,
            "rules": {r.name: self.rules[r].to_dict() for r in sorted(self.rules)},
        }


def default_config(**overrides) -> FusionConfig:
    """Neutral prior: every regime gets identical parameters.

    This is the regime-agnostic control the calibrated model has to beat.
    """
    cfg = FusionConfig(rules={regime: RegimeRule() for regime in Regime})
    return replace(cfg, **overrides) if overrides else cfg


@dataclass
class TradeDecision:
    """A sized, auditable trading instruction."""

    symbol: str
    side: str
    """One of ``buy``, ``sell``, ``short``, ``cover``, ``flat``."""

    target_notional: float
    target_qty: float
    limit_price: float
    horizon_bars: int

    regime: Regime
    edge: float
    edge_z: float
    expected_return: float
    sigma_horizon: float
    reason: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def is_trade(self) -> bool:
        return self.side != "flat" and abs(self.target_qty) > 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "target_notional": self.target_notional,
            "target_qty": self.target_qty,
            "limit_price": self.limit_price,
            "horizon_bars": self.horizon_bars,
            "regime": self.regime.name,
            "edge": self.edge,
            "edge_z": self.edge_z,
            "expected_return": self.expected_return,
            "sigma_horizon": self.sigma_horizon,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }


class RegimeConditionedSizer:
    """Turns (regime state, news score, account state) into a sized order."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or default_config()

    # -- horizon ---------------------------------------------------------
    def choose_horizon(self, snapshot: RegimeSnapshot, rule: RegimeRule) -> int:
        """Holding period, shortened when the regime is not expected to last.

        The volatility used for sizing is the *cumulative* forecast over exactly
        this many bars, so shortening the horizon and re-pricing risk stay
        consistent with each other.
        """
        base = int(rule.horizon_bars)
        if not self.config.adapt_horizon_to_regime:
            return base
        return max(1, min(base, int(snapshot.expected_regime_bars)))

    # -- the decision ----------------------------------------------------
    def decide(
        self,
        *,
        symbol: str,
        edge: float,
        classifier: RegimeClassifier,
        probs: np.ndarray,
        price: float,
        equity: float,
        forecast: Forecast | None = None,
        current_qty: float = 0.0,
        gross_exposure: float = 0.0,
        staleness_weight: float = 1.0,
        allow_fractional: bool = False,
    ) -> TradeDecision:
        cfg = self.config
        edge = float(np.clip(edge, -1.0, 1.0)) * float(np.clip(staleness_weight, 0.0, 1.0))

        # Classification depends only on the one-bar forecast, so the regime is
        # settled before the horizon is chosen and a single snapshot suffices.
        # (Taking a snapshot per pass cost ~2ms and blew the latency budget.)
        sigma_bar = math.sqrt(max(classifier.model.forecast_variance(probs, 1), 0.0))
        regime = classifier.classify(sigma_bar)
        rule = cfg.rule(regime)

        horizon = int(rule.horizon_bars)
        if cfg.adapt_horizon_to_regime:
            horizon = max(
                1, min(horizon, classifier.expected_regime_bars(probs, regime))
            )

        snapshot = classifier.snapshot(probs, horizon_bars=horizon)
        sigma_h = float(snapshot.sigma_horizon)

        def flat(reason: str) -> TradeDecision:
            return TradeDecision(
                symbol=symbol, side="flat", target_notional=0.0, target_qty=0.0,
                limit_price=price, horizon_bars=horizon, regime=regime, edge=edge,
                edge_z=0.0, expected_return=0.0, sigma_horizon=sigma_h,
                reason=reason, diagnostics=self._diagnostics(snapshot, rule, forecast),
            )

        if price <= 0 or equity <= 0:
            return flat("invalid price or equity")
        if sigma_h <= 0 or not math.isfinite(sigma_h):
            return flat("no usable volatility forecast")
        if abs(edge) < rule.min_abs_edge:
            return flat(
                f"|edge| {abs(edge):.3f} below {regime.name} gate {rule.min_abs_edge:.3f}"
            )

        # Expected cumulative return over the horizon, in return units.
        mu_news = rule.beta * edge * sigma_h
        mu = mu_news
        if cfg.foundation_weight > 0.0 and forecast is not None:
            mu_fm = forecast.cumulative_return(price)
            mu = (1.0 - cfg.foundation_weight) * mu_news + cfg.foundation_weight * mu_fm

        edge_z = mu / sigma_h
        if abs(edge_z) < cfg.min_edge_z:
            return flat(f"|edge_z| {abs(edge_z):.4f} below {cfg.min_edge_z:.4f}")

        # Risk-budget sizing: put `risk_budget` of equity at one horizon-sigma
        # when conviction is at the reference level, scaling down linearly
        # below it. Notional is proportional to 1/sigma_horizon, so every
        # regime contributes the same risk per trade rather than the same
        # dollar exposure -- this is the channel that does most of the work.
        conviction = float(np.clip(edge_z / cfg.reference_edge_z, -1.0, 1.0))
        notional = (
            equity * cfg.risk_budget * conviction / sigma_h * rule.size_multiplier
        )

        cap = equity * min(rule.max_position_fraction, cfg.max_position_fraction)
        capped = abs(notional) > cap
        notional = float(np.clip(notional, -cap, cap))

        # Respect the account-level gross limit.
        headroom = max(equity * cfg.max_gross_leverage - abs(gross_exposure), 0.0)
        if abs(notional) > headroom:
            notional = math.copysign(headroom, notional)

        if notional < 0 and not cfg.allow_short:
            return flat("short selling disabled")
        if abs(notional) < cfg.min_notional:
            return flat(f"notional ${abs(notional):.0f} below minimum")

        qty = notional / price
        if not allow_fractional:
            qty = float(np.trunc(qty))
            if abs(qty) < 1:
                return flat("rounds to zero shares")

        side = self._side(qty, current_qty)
        # Cross the spread slightly to get filled, in the direction of the trade.
        limit_price = price * (1.007 if qty > 0 else 0.993)

        diagnostics = self._diagnostics(snapshot, rule, forecast)
        diagnostics["position_cap_binding"] = capped
        diagnostics["conviction"] = conviction
        return TradeDecision(
            symbol=symbol, side=side, target_notional=qty * price, target_qty=qty,
            limit_price=round(limit_price, 2), horizon_bars=horizon, regime=regime,
            edge=edge, edge_z=edge_z, expected_return=mu, sigma_horizon=sigma_h,
            reason=f"{regime.name}: beta={rule.beta:.3f} edge={edge:+.2f} "
                   f"sigma_{horizon}={sigma_h:.4f}",
            diagnostics=diagnostics,
        )

    @staticmethod
    def _side(qty: float, current_qty: float) -> str:
        if qty > 0:
            return "cover" if current_qty < 0 else "buy"
        if qty < 0:
            return "sell" if current_qty > 0 else "short"
        return "flat"

    @staticmethod
    def _diagnostics(
        snapshot: RegimeSnapshot, rule: RegimeRule, forecast: Forecast | None
    ) -> dict:
        return {
            "regime_quantile": snapshot.quantile,
            "sigma_bar": snapshot.sigma_bar,
            "survival_probability": snapshot.survival_probability,
            "expected_regime_bars": snapshot.expected_regime_bars,
            "fast_share": snapshot.fast_share,
            "component_probabilities": snapshot.component_probabilities,
            "rule_beta": rule.beta,
            "rule_t_stat": rule.t_stat,
            "forecast_backend": None if forecast is None else forecast.backend,
        }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
@dataclass
class CalibrationObservation:
    """One realised news trade, used to fit the per-regime response."""

    regime: Regime
    edge: float
    realised_return: float
    """Cumulative return over the holding horizon, signed in price terms (not
    signed by the trade direction)."""

    sigma_horizon: float


def _fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Through-origin OLS slope and its standard error."""
    denom = float((x**2).sum())
    if denom <= 1e-12 or x.size < 3:
        return 0.0, float("nan")
    beta = float((x * y).sum() / denom)
    resid = y - beta * x
    dof = x.size - 1
    if dof <= 0:
        return beta, float("nan")
    se = math.sqrt(float((resid**2).sum()) / dof / denom)
    return beta, se


def calibrate(
    observations: Sequence[CalibrationObservation],
    *,
    base: FusionConfig | None = None,
    min_observations: int = 20,
) -> FusionConfig:
    """Estimate a per-regime response coefficient from realised outcomes.

    For each regime we regress the volatility-normalised realised return on the
    news edge, through the origin::

        realised_return / sigma_horizon  =  beta_regime * edge  +  noise

    Through the origin because a news signal should not carry an unconditional
    drift term; if it does, that is a market-beta artefact and does not belong
    in a headline-response coefficient.

    Empirical-Bayes shrinkage
    -------------------------
    Per-regime estimates are shrunk toward the pooled estimate with the
    James-Stein weight

        w_i = tau^2 / (tau^2 + se_i^2)

    where ``se_i`` is the regime's own standard error and ``tau^2`` is the
    estimated *between-regime* variance of the true betas, obtained by
    subtracting the average sampling variance from the observed spread. This
    replaces an arbitrary ``n / (n + k)`` rule with one that answers the actual
    question: is the spread between regimes larger than what pure estimation
    noise would produce? If it is not, ``tau^2`` collapses to zero, every weight
    goes to zero, and the calibration returns four identical betas — which is
    the correct answer when there is no regime effect, and the property that
    stops this module from inventing one.
    """
    if not observations:
        return base or default_config()

    cfg = base or default_config()
    edges = np.array([o.edge for o in observations], dtype=float)
    ys = np.array(
        [
            o.realised_return / o.sigma_horizon if o.sigma_horizon > 0 else 0.0
            for o in observations
        ],
        dtype=float,
    )
    regimes = np.array([int(o.regime) for o in observations])

    pooled_beta, pooled_se = _fit_through_origin(edges, ys)

    # Per-regime raw fits.
    raw: dict[Regime, tuple[float, float, int]] = {}
    for regime in Regime:
        mask = regimes == int(regime)
        n = int(mask.sum())
        if n < min_observations:
            continue
        beta, se = _fit_through_origin(edges[mask], ys[mask])
        if math.isfinite(se) and se > 0:
            raw[regime] = (beta, se, n)

    # Estimate the between-regime variance of the true betas.
    tau_squared = 0.0
    if len(raw) >= 2:
        betas = np.array([v[0] for v in raw.values()])
        ses = np.array([v[1] for v in raw.values()])
        observed_spread = float(betas.var(ddof=1))
        mean_sampling_var = float((ses**2).mean())
        tau_squared = max(0.0, observed_spread - mean_sampling_var)

    rules: dict[Regime, RegimeRule] = {}
    for regime in Regime:
        template = cfg.rule(regime)
        if regime not in raw:
            n = int((regimes == int(regime)).sum())
            rules[regime] = replace(
                template, beta=pooled_beta, n_observations=n,
                raw_beta=float("nan"), standard_error=pooled_se,
                t_stat=pooled_beta / pooled_se if pooled_se else float("nan"),
                shrinkage_weight=0.0,
            )
            continue
        raw_beta, se, n = raw[regime]
        weight = tau_squared / (tau_squared + se**2) if (tau_squared + se**2) > 0 else 0.0
        rules[regime] = replace(
            template,
            beta=weight * raw_beta + (1.0 - weight) * pooled_beta,
            t_stat=raw_beta / se,
            n_observations=n,
            raw_beta=raw_beta,
            standard_error=se,
            shrinkage_weight=weight,
        )

    return replace(cfg, rules=rules)


def power_analysis(
    cfg: FusionConfig,
    *,
    target_difference: float = 0.3,
    power: float = 0.80,
    alpha: float = 0.05,
) -> dict:
    """How many events per regime would be needed to resolve a real difference.

    The single most useful thing to know before committing to regime
    conditioning on live data. Standard errors shrink as ``1/sqrt(n)``, so from
    an observed ``(se, n)`` we can invert for the ``n`` at which a true beta gap
    of ``target_difference`` between two regimes becomes detectable::

        n_required = n_observed * (2 * (z_alpha + z_power)^2 * se^2)
                                  / target_difference^2

    The factor of two is because comparing two regimes doubles the variance of
    the difference.
    """
    # Normal quantiles without pulling in scipy at call time.
    z_alpha = 1.959963985 if abs(alpha - 0.05) < 1e-9 else _normal_quantile(1 - alpha / 2)
    z_power = 0.841621234 if abs(power - 0.80) < 1e-9 else _normal_quantile(power)

    out: dict[str, dict] = {}
    for regime in Regime:
        rule = cfg.rule(regime)
        se, n = rule.standard_error, rule.n_observations
        if not math.isfinite(se) or se <= 0 or n <= 0:
            out[regime.name] = {"n_observed": n, "required": None}
            continue
        required = n * (2.0 * (z_alpha + z_power) ** 2 * se**2) / (target_difference**2)
        out[regime.name] = {
            "n_observed": n,
            "standard_error": se,
            "required": int(math.ceil(required)),
            "shortfall": max(0, int(math.ceil(required)) - n),
        }
    return {
        "target_difference": target_difference,
        "power": power,
        "alpha": alpha,
        "per_regime": out,
    }


def _normal_quantile(p: float) -> float:
    """Acklam's rational approximation to the standard normal inverse CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def summarise_calibration(cfg: FusionConfig) -> str:
    lines = [
        f"{'regime':<10} {'beta':>8} {'raw':>8} {'se':>7} {'t':>7} "
        f"{'shrink':>7} {'n':>6}",
        "-" * 56,
    ]
    for regime in Regime:
        rule = cfg.rule(regime)
        raw = "     nan" if math.isnan(rule.raw_beta) else f"{rule.raw_beta:8.3f}"
        se = "    nan" if math.isnan(rule.standard_error) else f"{rule.standard_error:7.3f}"
        t = "    nan" if math.isnan(rule.t_stat) else f"{rule.t_stat:7.2f}"
        w = "    nan" if math.isnan(rule.shrinkage_weight) else f"{rule.shrinkage_weight:7.2f}"
        lines.append(
            f"{regime.name:<10} {rule.beta:8.3f} {raw} {se} {t} {w} "
            f"{rule.n_observations:6d}"
        )
    return "\n".join(lines)
