"""Backtester accounting, look-ahead control, and the live service."""

from __future__ import annotations

import numpy as np
import pytest

from mtgpt.backtest.engine import (
    BacktestConfig,
    CostModel,
    LegacyThresholdStrategy,
    run_backtest,
    run_comparison,
)
from mtgpt.backtest.metrics import bootstrap_sharpe_ci, compare, compute_stats
from mtgpt.data.synthetic import SyntheticConfig, flat_response, generate
from mtgpt.models.msm import MSMModel
from mtgpt.models.regimes import RegimeClassifier
from mtgpt.signals.fusion import default_config


@pytest.fixture(scope="module")
def dataset():
    return generate(SyntheticConfig(n_bars=9000, events_per_1000_bars=40.0, seed=5))


@pytest.fixture(scope="module")
def fitted(dataset):
    returns = dataset.bars.log_returns
    fit = MSMModel.fit(returns[:4000], k_components=4, n_starts=1)
    model = MSMModel(fit.params)
    classifier = RegimeClassifier.from_model(model, horizon_bars=30, n_sim=5000, seed=2)
    _, filtered_r = model.filter(returns)
    filtered = np.vstack([model.stationary_distribution()[None, :], filtered_r])
    return model, classifier, filtered


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------
def test_synthetic_dataset_is_well_formed(dataset):
    assert len(dataset.bars) == 9000
    assert (dataset.bars.close > 0).all()
    assert len(dataset.events) > 100
    assert dataset.true_regimes.shape == (9000,)
    assert set(np.unique(dataset.true_regimes)).issubset({0, 1, 2, 3})


def _event_outcomes(dataset, horizon: int = 30):
    """(edge, forward return, local vol, true regime) for every usable event."""
    close = dataset.bars.close
    rows = []
    for event in dataset.events:
        bar = dataset.bars.index_at_or_before(event.timestamp)
        if bar < 100 or bar + horizon >= len(close):
            continue
        rows.append((
            event.edge,
            float(np.log(close[bar + horizon] / close[bar])),
            float(dataset.true_volatility[bar]),
            int(dataset.true_regimes[bar]),
        ))
    return np.array(rows)


def _per_regime_correlations(dataset, min_events: int = 30):
    """corr(edge, vol-normalised forward return) for each well-populated regime."""
    rows = _event_outcomes(dataset)
    edges, forwards, vols, regimes = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    normalised = forwards / vols
    out = {}
    for regime in range(4):
        mask = regimes == regime
        if mask.sum() >= min_events:
            out[regime] = float(np.corrcoef(edges[mask], normalised[mask])[0, 1])
    return out, float(np.corrcoef(edges, normalised)[0, 1])


def test_synthetic_signal_matches_ground_truth_within_each_regime(dataset):
    """If the generator has no signal, every downstream test is vacuous.

    The signal must be checked *per regime*. Measured within a regime, the sign
    of the edge/return relationship has to agree with the ``regime_response``
    that generated it.
    """
    from mtgpt.models.regimes import Regime

    per_regime, _ = _per_regime_correlations(dataset)
    assert len(per_regime) >= 2, "need at least two populated regimes to test"
    for regime, correlation in per_regime.items():
        truth = dataset.config.regime_response[Regime(regime)]
        assert np.sign(correlation) == np.sign(truth), (
            f"{Regime(regime).name}: corr={correlation:+.3f} but ground-truth "
            f"response={truth:+.2f}"
        )
        assert abs(correlation) > 0.10, (Regime(regime).name, correlation)


def test_pooling_across_regimes_destroys_the_signal(dataset):
    """The central claim of the project, stated as a test.

    Within the quieter regimes the edge/return relationship is clearly positive.
    Pooled across all regimes it collapses — and in a crisis-heavy sample it
    inverts outright — because crisis events are both the most numerous by
    volatility weight and (in this scenario's ground truth) the ones that
    revert. Treating every headline identically averages a real edge away.
    """
    per_regime, pooled = _per_regime_correlations(dataset)
    best = max(per_regime.values())
    assert best > 0.15, per_regime
    assert best > pooled + 0.20, (per_regime, pooled)


def test_flat_response_scenario_has_equal_ground_truth():
    cfg = flat_response(n_bars=3000)
    assert len(set(cfg.regime_response.values())) == 1


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------
def test_equity_curve_reconciles_with_trade_pnl(dataset, fitted):
    _, classifier, filtered = fitted
    config = BacktestConfig(warmup_bars=300, seed=0)
    result = run_backtest(
        LegacyThresholdStrategy(), dataset.bars, dataset.events, classifier,
        filtered, config, start_bar=300,
    )
    assert result.trades, "expected at least one trade"
    curve_pnl = float(result.equity_curve[-1] - result.equity_curve[0])
    trade_pnl = sum(t.net_pnl for t in result.trades)
    assert curve_pnl == pytest.approx(trade_pnl, rel=1e-6, abs=1.0)


def test_costs_reduce_pnl(dataset, fitted):
    _, classifier, filtered = fitted
    free = BacktestConfig(warmup_bars=300, costs=CostModel(0.0, 0.0, 0.0, 0.0))
    expensive = BacktestConfig(warmup_bars=300, costs=CostModel(25.0, 25.0, 0.0, 0.0))
    kwargs = dict(bars=dataset.bars, events=dataset.events, classifier=classifier,
                  filtered=filtered, start_bar=300)
    cheap_result = run_backtest(LegacyThresholdStrategy(), config=free, **kwargs)
    dear_result = run_backtest(LegacyThresholdStrategy(), config=expensive, **kwargs)
    assert dear_result.stats.net_pnl < cheap_result.stats.net_pnl
    assert dear_result.stats.total_costs > cheap_result.stats.total_costs


def test_trades_never_start_before_the_signal(dataset, fitted):
    """Latency must be enforced: no fill on the bar the news landed in."""
    _, classifier, filtered = fitted
    config = BacktestConfig(warmup_bars=300, latency_bars=1)
    result = run_backtest(
        LegacyThresholdStrategy(), dataset.bars, dataset.events, classifier,
        filtered, config, start_bar=300,
    )
    by_id = {e.event_id: e for e in dataset.events}
    for trade in result.trades:
        signal_bar = dataset.bars.index_at_or_before(by_id[trade.event_id].timestamp)
        assert trade.entry_bar >= signal_bar + 1
        assert trade.exit_bar > trade.entry_bar


def test_stop_loss_caps_the_worst_trade(dataset, fitted):
    _, classifier, filtered = fitted
    config = BacktestConfig(warmup_bars=300, stop_loss=0.01,
                            costs=CostModel(0.0, 0.0, 0.0, 0.0))
    result = run_backtest(
        LegacyThresholdStrategy(), dataset.bars, dataset.events, classifier,
        filtered, config, start_bar=300,
    )
    stopped = [t for t in result.trades if t.exit_reason == "stop"]
    assert stopped, "a 1% stop should trigger somewhere"
    for trade in stopped:
        adverse = (trade.exit_price / trade.entry_price - 1.0) * np.sign(trade.qty)
        assert adverse < 0


# --------------------------------------------------------------------------
# The look-ahead control
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_shuffled_scores_destroy_the_edge():
    """Randomise the scores and the measured edge must vanish.

    The guard against a look-ahead bug. If the engine were peeking at future
    prices, calibration would still find a significant relationship after the
    scores have been detached from the events that produced them. Compared on
    t-statistics, which are scale free and directly interpretable.
    """
    import copy
    import random

    from mtgpt.models.regimes import Regime

    big = generate(SyntheticConfig(n_bars=30_000, events_per_1000_bars=40.0, seed=11))

    scrambled = copy.deepcopy(big.events)
    scores = [e.score for e in scrambled]
    random.Random(0).shuffle(scores)
    for event, score in zip(scrambled, scores):
        event.score = score

    config = BacktestConfig(msm_k_components=4, msm_n_starts=1, warmup_bars=300)
    real = run_comparison(big.bars, big.events, config, verbose=False)
    fake = run_comparison(big.bars, scrambled, config, verbose=False)

    def max_abs_t(result):
        values = [
            abs(result.calibration.rule(r).t_stat)
            for r in Regime
            if result.calibration.rule(r).n_observations >= 20
            and np.isfinite(result.calibration.rule(r).t_stat)
        ]
        return max(values) if values else 0.0

    real_t, fake_t = max_abs_t(real), max_abs_t(fake)
    assert real_t > 3.0, f"real data should show a clear response, got t={real_t:.2f}"
    assert fake_t < 3.0, f"shuffled scores produced t={fake_t:.2f} - look-ahead?"
    assert real_t > 2 * fake_t, (real_t, fake_t)


@pytest.mark.slow
def test_comparison_runs_all_four_strategies(dataset):
    result = run_comparison(
        dataset.bars, dataset.events,
        BacktestConfig(msm_k_components=4, msm_n_starts=1, warmup_bars=300),
        verbose=False,
    )
    assert set(result.results) == {
        "legacy_threshold", "pooled_vol_target",
        "regime_fixed_horizon", "regime_conditioned",
    }
    for res in result.results.values():
        assert res.n_signals > 0
        curve_pnl = float(res.equity_curve[-1] - res.equity_curve[0])
        assert curve_pnl == pytest.approx(sum(t.net_pnl for t in res.trades),
                                          rel=1e-6, abs=1.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_stats_on_a_flat_curve_are_zero():
    stats = compute_stats(
        equity_curve=np.full(500, 1e6), trade_returns=[], trade_pnl=[],
        costs=0.0, periods_per_year=98_280, initial_equity=1e6,
    )
    assert stats.sharpe == 0.0
    assert stats.max_drawdown == 0.0
    assert stats.n_trades == 0


def test_max_drawdown_is_measured_from_the_peak():
    curve = np.array([100.0, 120.0, 60.0, 90.0])
    stats = compute_stats(equity_curve=curve, trade_returns=[], trade_pnl=[],
                          costs=0.0, periods_per_year=252, initial_equity=100.0)
    assert stats.max_drawdown == pytest.approx(0.5)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(3000) * 0.001 + 0.00005
    lo, hi = bootstrap_sharpe_ci(returns, 252, n_boot=500)
    point = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert lo < point < hi


def test_compare_finds_no_difference_between_identical_samples():
    rng = np.random.default_rng(1)
    sample = rng.standard_normal(400) * 0.01
    result = compare(sample, sample.copy(), n_boot=500)
    assert not result["significant"]
    assert result["difference"] == pytest.approx(0.0, abs=1e-12)


def test_compare_detects_a_real_difference():
    rng = np.random.default_rng(2)
    worse = rng.standard_normal(2000) * 0.01
    better = rng.standard_normal(2000) * 0.01 + 0.01
    assert compare(worse, better, n_boot=500)["significant"]


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(dataset):
    from fastapi.testclient import TestClient

    from mtgpt.service.app import create_app
    from mtgpt.service.state import RegimeService

    service = RegimeService(default_config(), k_components=4)
    service.warm_up(dataset.bars.slice(0, 3000), k_components=4, n_starts=1, n_sim=3000)
    return TestClient(create_app(service))


def test_health_lists_registered_symbols(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert "SYNTH" in payload["symbols"]


def test_regime_endpoint_returns_a_full_snapshot(client):
    payload = client.get("/regime/SYNTH").json()
    assert payload["regime"] in {"CALM", "NORMAL", "TURBULENT", "CRISIS"}
    assert payload["sigma_horizon"] > 0
    assert 0.0 <= payload["survival_probability"] <= 1.0
    assert len(payload["component_probabilities"]) == 4


def test_unknown_symbol_returns_404(client):
    assert client.get("/regime/NOPE").status_code == 404


def test_size_endpoint_is_fast_enough(client):
    body = {"symbol": "SYNTH", "score": 88, "price": 100.0, "equity": 1_000_000.0}
    payload = client.post("/size", json=body).json()
    assert payload["side"] in {"buy", "sell", "short", "cover", "flat"}
    # The whole point of caching filter state: sizing must be sub-millisecond.
    assert payload["latency_us"] < 1000, payload["latency_us"]


def test_size_requires_a_signal(client):
    response = client.post("/size", json={"symbol": "SYNTH", "price": 100.0, "equity": 1e6})
    assert response.status_code == 400


def test_size_rejects_a_nonsense_price(client):
    body = {"symbol": "SYNTH", "score": 88, "price": -5.0, "equity": 1e6}
    assert client.post("/size", json=body).status_code == 422


def test_pushing_a_bar_advances_the_filter(client):
    before = client.get("/regime/SYNTH").json()["bars_seen"]
    client.post("/bars/SYNTH", json={"close": 101.0})
    after = client.get("/regime/SYNTH").json()["bars_seen"]
    assert after == before + 1


def test_metrics_report_latency_percentiles(client):
    for _ in range(20):
        client.post("/size", json={"symbol": "SYNTH", "score": 90,
                                   "price": 100.0, "equity": 1e6})
    summary = client.get("/metrics").json()["sizing_latency"]
    assert summary["n"] >= 20
    assert summary["p99_us"] < 5000
