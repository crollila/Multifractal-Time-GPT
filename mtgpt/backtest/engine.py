"""Event-driven backtester for news trades, with regime attribution.

Design commitments
------------------
**Nothing sees the future.** MSM parameters are estimated on a training slice
and frozen; regime cutoffs come from simulating that fitted model, not from
sample quantiles; the Hamilton filter is causal by construction; per-regime
betas are calibrated on training events only. The test slice is touched exactly
once, at evaluation.

**Latency is real.** A signal timestamped inside bar ``i`` cannot fill at bar
``i``'s close. ``latency_bars`` defaults to 1, and the synthetic generator
front-loads news drift precisely so that being late costs you.

**Costs are charged on both sides**, including borrow on shorts. A news
strategy trading 30-bar holds at retail spreads is a cost story as much as an
alpha story, and a backtest that hides that is worse than no backtest.

**The comparison is an ablation, not a demo.** Four strategies run on identical
events so the gain can be attributed to a specific channel rather than to
"the new thing is better":

===========================  =========================================
``legacy_threshold``         the upstream bot's rule: score cutoffs at
                             70/45/30, bucketed fixed sizing
``pooled_vol_target``        MSM volatility targeting, one pooled beta,
                             no regime split
``regime_fixed_horizon``     per-regime betas and gates, fixed holding
``regime_conditioned``       the full model, adaptive holding period
===========================  =========================================

``pooled_vol_target`` is the control that matters. If it captures most of the
improvement over ``legacy_threshold``, then the win is risk normalisation and
the regime *split* is decoration — which is a real and useful finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from ..data.loaders import BarSeries
from ..models.msm import MSMModel
from ..models.regimes import Regime, RegimeClassifier
from ..signals.fusion import (
    CalibrationObservation,
    FusionConfig,
    RegimeConditionedSizer,
    TradeDecision,
    calibrate,
    default_config,
)
from ..signals.news import NewsEvent, decay_weight, deduplicate
from .metrics import PerformanceStats, compare, compute_stats

__all__ = [
    "CostModel",
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "ComparisonResult",
    "run_backtest",
    "run_comparison",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class CostModel:
    """Round-trip trading frictions."""

    half_spread_bps: float = 2.0
    """Half the quoted spread, paid on entry and again on exit."""

    slippage_bps: float = 1.0
    """Impact beyond the quoted spread. Marketable limit orders into a news
    print routinely do worse than this."""

    commission_per_share: float = 0.0
    borrow_bps_per_day: float = 30.0
    """Annualised-equivalent borrow charged pro rata while short."""

    def entry_cost(self, qty: float, price: float) -> float:
        notional = abs(qty) * price
        return notional * (self.half_spread_bps + self.slippage_bps) / 1e4 + abs(
            qty
        ) * self.commission_per_share

    def exit_cost(self, qty: float, price: float) -> float:
        return self.entry_cost(qty, price)

    def borrow_cost(self, qty: float, price: float, bars: int, bar_seconds: float) -> float:
        if qty >= 0:
            return 0.0
        days = bars * bar_seconds / 23_400.0  # 6.5-hour trading day
        return abs(qty) * price * (self.borrow_bps_per_day / 1e4) * days


@dataclass
class BacktestConfig:
    initial_equity: float = 1_000_000.0
    latency_bars: int = 1
    costs: CostModel = field(default_factory=CostModel)
    stop_loss: float = 0.035
    """Adverse move at which a position is closed, matching the upstream bot's
    3.5% emergency exit. Applied identically to every strategy so it cannot
    explain a difference between them."""

    max_concurrent: int = 25
    train_fraction: float = 0.5
    msm_k_components: int = 6
    msm_n_starts: int = 3
    legacy_horizon_bars: int = 30
    warmup_bars: int = 500
    seed: int = 0


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    entry_bar: int
    exit_bar: int
    qty: float
    entry_price: float
    exit_price: float
    regime: Regime
    edge: float
    edge_z: float
    sigma_horizon: float
    planned_bars: int
    costs: float
    exit_reason: str
    event_id: int | None = None
    cap_binding: bool = False
    """True when the concentration cap, not the risk budget, set the size. A
    high rate here means volatility targeting is switched off in practice."""

    @property
    def gross_pnl(self) -> float:
        return self.qty * (self.exit_price - self.entry_price)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.entry_price

    @property
    def net_return(self) -> float:
        """Return on notional deployed."""
        return self.net_pnl / self.notional if self.notional > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "regime": self.regime.name,
            "edge": self.edge,
            "edge_z": self.edge_z,
            "planned_bars": self.planned_bars,
            "costs": self.costs,
            "net_pnl": self.net_pnl,
            "net_return": self.net_return,
            "exit_reason": self.exit_reason,
            "event_id": self.event_id,
            "cap_binding": self.cap_binding,
        }


@dataclass
class BacktestResult:
    name: str
    trades: list[Trade]
    equity_curve: np.ndarray
    stats: PerformanceStats
    n_signals: int
    n_declined: int
    decline_reasons: dict[str, int] = field(default_factory=dict)

    def by_regime(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for regime in Regime:
            subset = [t for t in self.trades if t.regime == regime]
            if not subset:
                out[regime.name] = {"n": 0}
                continue
            rets = np.array([t.net_return for t in subset])
            pnl = np.array([t.net_pnl for t in subset])
            sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
            out[regime.name] = {
                "n": len(subset),
                "net_pnl": float(pnl.sum()),
                "mean_return": float(rets.mean()),
                "hit_rate": float((pnl > 0).mean()),
                "t_stat": float(rets.mean() / (sd / math.sqrt(rets.size))) if sd > 0 else float("nan"),
                "avg_notional": float(np.mean([t.notional for t in subset])),
                "cap_rate": float(np.mean([t.cap_binding for t in subset])),
                "avg_sigma_horizon": float(np.mean([t.sigma_horizon for t in subset])),
            }
        return out

    @property
    def cap_binding_rate(self) -> float:
        """Share of trades sized by the concentration cap rather than by risk."""
        if not self.trades:
            return 0.0
        return float(np.mean([t.cap_binding for t in self.trades]))

    def format_by_regime(self) -> str:
        lines = [
            f"{'regime':<10} {'n':>5} {'net P&L':>12} {'ret/trade':>10} {'hit':>7} "
            f"{'t':>7} {'avg $':>12} {'sig_h':>8} {'capped':>7}",
            "-" * 86,
        ]
        for name, row in self.by_regime().items():
            if row.get("n", 0) == 0:
                lines.append(f"{name:<10} {0:>5}" + " " * 20 + "(no trades)")
                continue
            lines.append(
                f"{name:<10} {row['n']:>5d} {row['net_pnl']:>12,.0f} "
                f"{row['mean_return']:>10.4%} {row['hit_rate']:>7.1%} "
                f"{row['t_stat']:>7.2f} {row['avg_notional']:>12,.0f} "
                f"{row['avg_sigma_horizon']:>8.4f} {row['cap_rate']:>7.0%}"
            )
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    results: dict[str, BacktestResult]
    calibration: FusionConfig
    msm_fit: object
    classifier: RegimeClassifier
    split_bar: int
    comparisons: dict[str, dict] = field(default_factory=dict)

    def format_table(self) -> str:
        lines = [PerformanceStats.header()]
        for name, result in self.results.items():
            lines.append(result.stats.format_row(name))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------
Decider = Callable[..., TradeDecision]


class LegacyThresholdStrategy:
    """Replica of the upstream bot's entry and sizing rule.

    Score >= 70 buys with a bucketed multiple of ``equity/500``; score <= 30
    shorts with a bucketed multiple; everything between is flat. Sizing ignores
    volatility entirely, which is the specific thing regime conditioning is
    meant to fix.
    """

    name = "legacy_threshold"

    def __init__(self, horizon_bars: int = 30, max_position_fraction: float = 0.05):
        self.horizon_bars = horizon_bars
        self.max_position_fraction = max_position_fraction

    @staticmethod
    def _long_factor(score: float) -> float:
        if score >= 100:
            return 19.0
        if score >= 90:
            return 14.0
        if score >= 80:
            return 6.0
        if score >= 70:
            return 3.0
        return 0.0

    @staticmethod
    def _short_factor(score: float) -> float:
        if score <= 0:
            return 15.0
        if score <= 10:
            return 9.0
        if score <= 20:
            return 4.0
        if score <= 30:
            return 2.0
        return 0.0

    def decide(self, *, symbol, score, price, equity, regime, sigma_horizon, **_) -> TradeDecision:
        base = equity / 500.0
        qty = 0.0
        if score >= 70:
            qty = self._long_factor(score) * base / price
        elif score <= 30:
            qty = -self._short_factor(score) * base / price

        cap = equity * self.max_position_fraction / price
        qty = float(np.trunc(np.clip(qty, -cap, cap)))
        side = "buy" if qty > 0 else ("short" if qty < 0 else "flat")
        return TradeDecision(
            symbol=symbol, side=side, target_notional=qty * price, target_qty=qty,
            limit_price=price, horizon_bars=self.horizon_bars, regime=regime,
            edge=(score - 50) / 50.0, edge_z=0.0, expected_return=0.0,
            sigma_horizon=sigma_horizon,
            reason="legacy threshold rule" if qty else f"score {score:g} inside 30-70 band",
        )


class FusionStrategy:
    """Wraps :class:`RegimeConditionedSizer` into the engine's interface."""

    def __init__(self, name: str, config: FusionConfig):
        self.name = name
        self.sizer = RegimeConditionedSizer(config)

    def decide(
        self, *, symbol, score, price, equity, classifier, probs,
        gross_exposure, staleness, **_
    ) -> TradeDecision:
        from ..signals.news import score_to_edge

        return self.sizer.decide(
            symbol=symbol, edge=score_to_edge(score), classifier=classifier,
            probs=probs, price=price, equity=equity,
            gross_exposure=gross_exposure, staleness_weight=staleness,
        )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
@dataclass
class _OpenPosition:
    trade_index: int
    qty: float
    entry_bar: int
    entry_price: float
    exit_bar: int
    stop_price: float
    planned_bars: int
    regime: Regime
    edge: float
    edge_z: float
    sigma_horizon: float
    entry_cost: float
    event_id: int | None
    cap_binding: bool = False


def run_backtest(
    strategy,
    bars: BarSeries,
    events: Sequence[NewsEvent],
    classifier: RegimeClassifier,
    filtered: np.ndarray,
    config: BacktestConfig,
    *,
    start_bar: int = 0,
    end_bar: int | None = None,
) -> BacktestResult:
    """Run one strategy over one symbol.

    ``filtered`` is the causal Hamilton-filter output for the whole series,
    produced once from the *training-fitted* model and shared across strategies
    so they see identical state.
    """
    close = bars.close
    n_bars = len(bars)
    end_bar = n_bars if end_bar is None else min(end_bar, n_bars)

    equity = config.initial_equity
    qty_by_bar = np.zeros(n_bars)
    cost_by_bar = np.zeros(n_bars)

    trades: list[Trade] = []
    open_positions: list[_OpenPosition] = []
    n_signals = 0
    decline_reasons: dict[str, int] = {}

    def close_position(pos: _OpenPosition, bar: int, price: float, reason: str) -> None:
        exit_cost = config.costs.exit_cost(pos.qty, price)
        borrow = config.costs.borrow_cost(
            pos.qty, pos.entry_price, bar - pos.entry_bar, bars.bar_seconds
        )
        total_costs = pos.entry_cost + exit_cost + borrow
        cost_by_bar[bar] += exit_cost + borrow
        trades.append(
            Trade(
                symbol=bars.symbol, entry_bar=pos.entry_bar, exit_bar=bar, qty=pos.qty,
                entry_price=pos.entry_price, exit_price=price, regime=pos.regime,
                edge=pos.edge, edge_z=pos.edge_z, sigma_horizon=pos.sigma_horizon,
                planned_bars=pos.planned_bars, costs=total_costs, exit_reason=reason,
                event_id=pos.event_id, cap_binding=pos.cap_binding,
            )
        )
        # We fill at bar ``entry_bar``'s close, so the position earns the price
        # change over bars entry_bar+1 .. bar inclusive. Getting this slice wrong
        # by one silently decouples the equity curve from the trade P&L.
        qty_by_bar[pos.entry_bar + 1 : bar + 1] += pos.qty

    def flush_due(up_to_bar: int) -> None:
        """Close positions whose exit or stop triggers at or before ``up_to_bar``."""
        still_open: list[_OpenPosition] = []
        for pos in open_positions:
            exit_bar, reason = pos.exit_bar, "horizon"
            # Walk forward looking for a stop breach before the planned exit.
            hi = min(pos.exit_bar, end_bar - 1)
            if config.stop_loss > 0 and hi > pos.entry_bar:
                window = close[pos.entry_bar + 1 : hi + 1]
                if pos.qty > 0:
                    breach = np.nonzero(window <= pos.stop_price)[0]
                else:
                    breach = np.nonzero(window >= pos.stop_price)[0]
                if breach.size:
                    exit_bar = pos.entry_bar + 1 + int(breach[0])
                    reason = "stop"
            if exit_bar <= up_to_bar:
                close_position(pos, exit_bar, float(close[exit_bar]), reason)
            else:
                still_open.append(pos)
        open_positions[:] = still_open

    ordered = sorted(deduplicate(events), key=lambda e: e.timestamp)
    for event in ordered:
        bar = bars.index_at_or_before(event.timestamp)
        entry_bar = bar + config.latency_bars
        if bar < max(start_bar, config.warmup_bars) or entry_bar >= end_bar - 1:
            continue

        flush_due(entry_bar)

        n_signals += 1
        price = float(close[entry_bar])
        gross = sum(abs(p.qty) * price for p in open_positions)
        realised = sum(t.net_pnl for t in trades)
        current_equity = config.initial_equity + realised

        if len(open_positions) >= config.max_concurrent:
            decline_reasons["max concurrent positions"] = (
                decline_reasons.get("max concurrent positions", 0) + 1
            )
            continue

        # Staleness: how old the news already is by the time we can act.
        age = (entry_bar - bar) * bars.bar_seconds + (event.latency_ms or 0.0) / 1000.0
        decision = strategy.decide(
            symbol=bars.symbol,
            score=event.score,
            price=price,
            equity=current_equity,
            classifier=classifier,
            probs=filtered[entry_bar],
            regime=Regime(int(classifier.label_path(filtered[entry_bar : entry_bar + 1])[0])),
            sigma_horizon=float(
                classifier.model.forecast_volatility(filtered[entry_bar], config.legacy_horizon_bars)
            ),
            gross_exposure=gross,
            staleness=decay_weight(age),
        )

        if not decision.is_trade:
            decline_reasons[decision.reason] = decline_reasons.get(decision.reason, 0) + 1
            continue

        qty = decision.target_qty
        entry_cost = config.costs.entry_cost(qty, price)
        cost_by_bar[entry_bar] += entry_cost
        stop = price * (1 - config.stop_loss) if qty > 0 else price * (1 + config.stop_loss)
        open_positions.append(
            _OpenPosition(
                trade_index=len(trades), qty=qty, entry_bar=entry_bar, entry_price=price,
                exit_bar=min(entry_bar + decision.horizon_bars, end_bar - 1),
                stop_price=stop, planned_bars=decision.horizon_bars,
                regime=decision.regime, edge=decision.edge, edge_z=decision.edge_z,
                sigma_horizon=decision.sigma_horizon, entry_cost=entry_cost,
                event_id=event.event_id,
                cap_binding=bool(decision.diagnostics.get("position_cap_binding", False)),
            )
        )

    flush_due(end_bar - 1)

    # Per-bar equity from held quantity times price change, less costs.
    price_change = np.zeros(n_bars)
    price_change[1:] = np.diff(close)
    bar_pnl = qty_by_bar * price_change - cost_by_bar
    equity_curve = config.initial_equity + np.cumsum(bar_pnl)

    window = slice(max(start_bar, config.warmup_bars), end_bar)
    curve = equity_curve[window]
    exposure = float(np.mean(np.abs(qty_by_bar[window]) * close[window] / config.initial_equity))

    stats = compute_stats(
        equity_curve=curve,
        trade_returns=[t.net_return for t in trades],
        trade_pnl=[t.net_pnl for t in trades],
        costs=float(cost_by_bar.sum()),
        periods_per_year=bars.annualisation_factor(),
        initial_equity=config.initial_equity,
        exposure_fraction=exposure,
        seed=config.seed,
    )

    return BacktestResult(
        name=getattr(strategy, "name", "strategy"),
        trades=trades,
        equity_curve=curve,
        stats=stats,
        n_signals=n_signals,
        n_declined=sum(decline_reasons.values()),
        decline_reasons=decline_reasons,
    )


def _build_calibration_observations(
    events: Sequence[NewsEvent],
    bars: BarSeries,
    classifier: RegimeClassifier,
    filtered: np.ndarray,
    config: BacktestConfig,
    horizon: int,
    *, end_bar: int,
) -> list[CalibrationObservation]:
    """Realised outcomes for training events, for :func:`calibrate`."""
    from ..signals.news import score_to_edge

    out: list[CalibrationObservation] = []
    close = bars.close
    for event in deduplicate(events):
        bar = bars.index_at_or_before(event.timestamp)
        entry = bar + config.latency_bars
        exit_bar = entry + horizon
        if bar < config.warmup_bars or exit_bar >= end_bar:
            continue
        realised = float(math.log(close[exit_bar] / close[entry]))
        sigma_h = float(classifier.model.forecast_volatility(filtered[entry], horizon))
        if sigma_h <= 0:
            continue
        regime = Regime(int(classifier.label_path(filtered[entry : entry + 1])[0]))
        out.append(
            CalibrationObservation(
                regime=regime,
                edge=score_to_edge(event.score),
                realised_return=realised,
                sigma_horizon=sigma_h,
            )
        )
    return out


def run_comparison(
    bars: BarSeries,
    events: Sequence[NewsEvent],
    config: BacktestConfig | None = None,
    *,
    base_fusion: FusionConfig | None = None,
    verbose: bool = True,
) -> ComparisonResult:
    """Walk-forward ablation: fit on the first half, evaluate on the second."""
    config = config or BacktestConfig()
    base_fusion = base_fusion or default_config()
    n_bars = len(bars)
    split = int(n_bars * config.train_fraction)
    if split <= config.warmup_bars + 100:
        raise ValueError("training slice is too short; supply more bars")

    returns = bars.log_returns  # aligned to bars[1:]

    if verbose:
        print(f"[1/4] fitting MSM on bars 0-{split} ({split} obs)...")
    fit = MSMModel.fit(
        returns[:split],
        k_components=config.msm_k_components,
        n_starts=config.msm_n_starts,
        seed=config.seed,
    )
    model = MSMModel(fit.params)
    if verbose:
        p = fit.params
        print(
            f"      m0={p.m0:.3f} sigma={p.sigma:.5f} gamma_1={p.gamma_1:.5f} "
            f"b={p.b:.2f} K={p.k_components} ll={fit.log_likelihood:,.0f}"
        )

    if verbose:
        print("[2/4] calibrating regime cutoffs from the fitted model...")
    classifier = RegimeClassifier.from_model(
        model, horizon_bars=config.legacy_horizon_bars, n_sim=20_000, seed=config.seed + 1
    )

    # Causal filter over the whole series. Row t uses returns up to t only.
    _, filtered_r = model.filter(returns, return_states=True)
    # Shift so that filtered[i] is the state after observing the bar-i close.
    filtered = np.vstack([model.stationary_distribution()[None, :], filtered_r])

    train_events = [e for e in events if bars.index_at_or_before(e.timestamp) < split]
    test_events = [e for e in events if bars.index_at_or_before(e.timestamp) >= split]
    if verbose:
        print(f"      {len(train_events)} training events / {len(test_events)} test events")

    if verbose:
        print("[3/4] calibrating per-regime response on training events only...")
    observations = _build_calibration_observations(
        train_events, bars, classifier, filtered, config,
        horizon=config.legacy_horizon_bars, end_bar=split,
    )
    calibrated = calibrate(observations, base=base_fusion)

    pooled_beta = float(
        np.mean([calibrated.rule(r).beta for r in Regime])
    )
    pooled = replace(
        base_fusion,
        rules={r: replace(base_fusion.rule(r), beta=pooled_beta) for r in Regime},
        adapt_horizon_to_regime=False,
    )
    regime_fixed = replace(calibrated, adapt_horizon_to_regime=False)
    regime_full = replace(calibrated, adapt_horizon_to_regime=True)

    strategies = [
        LegacyThresholdStrategy(horizon_bars=config.legacy_horizon_bars),
        FusionStrategy("pooled_vol_target", pooled),
        FusionStrategy("regime_fixed_horizon", regime_fixed),
        FusionStrategy("regime_conditioned", regime_full),
    ]

    if verbose:
        print(f"[4/4] evaluating {len(strategies)} strategies out-of-sample...")
    results: dict[str, BacktestResult] = {}
    for strategy in strategies:
        results[strategy.name] = run_backtest(
            strategy, bars, test_events, classifier, filtered, config,
            start_bar=split, end_bar=n_bars,
        )

    baseline = results["legacy_threshold"]
    comparisons = {
        name: compare(
            [t.net_return for t in baseline.trades],
            [t.net_return for t in result.trades],
            seed=config.seed,
        )
        for name, result in results.items()
        if name != "legacy_threshold"
    }

    return ComparisonResult(
        results=results,
        calibration=calibrated,
        msm_fit=fit,
        classifier=classifier,
        split_bar=split,
        comparisons=comparisons,
    )
