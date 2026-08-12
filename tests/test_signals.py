"""Tests for regime classification, news parsing and regime x news fusion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mtgpt.models.mfdfa import mfdfa, multifractal_report
from mtgpt.models.msm import MSMModel, MSMParams
from mtgpt.models.regimes import (
    Regime,
    RegimeClassifier,
    component_high_probabilities,
)
from mtgpt.signals.fusion import (
    CalibrationObservation,
    calibrate,
    default_config,
    power_analysis,
)
from mtgpt.signals.news import (
    NewsEvent,
    decay_weight,
    deduplicate,
    parse_score_line,
    score_to_edge,
)
from mtgpt.signals.fusion import RegimeConditionedSizer

PARAMS = MSMParams(m0=1.45, sigma=0.012, gamma_1=0.003, b=3.0, k_components=5)


@pytest.fixture(scope="module")
def model():
    return MSMModel(PARAMS)


@pytest.fixture(scope="module")
def classifier(model):
    return RegimeClassifier.from_model(model, horizon_bars=30, n_sim=8000, seed=3)


@pytest.fixture(scope="module")
def filtered(model):
    returns, _ = model.simulate(6000, seed=17)
    _, probs = model.filter(returns)
    return returns, probs


# --------------------------------------------------------------------------
# Regime classification
# --------------------------------------------------------------------------
def test_regimes_are_ordered_by_realised_volatility(classifier, filtered):
    """The labels must actually separate quiet tape from wild tape."""
    returns, probs = filtered
    labels = classifier.label_path(probs)
    means = [
        np.abs(returns[labels == int(r)]).mean()
        for r in Regime
        if (labels == int(r)).sum() > 20
    ]
    assert len(means) >= 3
    assert all(b > a for a, b in zip(means, means[1:])), means


def test_regime_shares_track_the_target_quantiles(classifier, filtered):
    """Shares should be in the right neighbourhood of the target quantiles.

    The tolerance is wide on purpose. MSM volatility is extremely persistent, so
    6000 bars contain only a few dozen independent regime episodes and the
    realised share of any one regime is genuinely noisy. A tight bound here
    would be a flaky test, not a stronger guarantee - the real check that the
    labels mean something is
    ``test_regimes_are_ordered_by_realised_volatility``.
    """
    _, probs = filtered
    labels = classifier.label_path(probs)
    shares = np.bincount(labels, minlength=4) / labels.size
    assert (shares > 0.02).all(), f"every regime should occur: {shares}"
    # Targets implied by quantiles (0.25, 0.60, 0.85).
    for observed, target in zip(shares, (0.25, 0.35, 0.25, 0.15)):
        assert abs(observed - target) < 0.18, shares


def test_cutoffs_are_increasing(classifier):
    assert (np.diff(classifier.forecast_cutoffs) > 0).all()
    assert (np.diff(classifier.state_cutoffs) > 0).all()


def test_survival_probability_decays_monotonically(classifier, filtered):
    _, probs = filtered
    vols = classifier.model.conditional_volatility_path(probs)
    hot = probs[int(np.argmax(vols))]
    regime = classifier.classify(float(vols.max()))
    survival = [classifier.survival_probability(hot, regime, h) for h in (1, 5, 20, 100, 1000)]
    assert all(b <= a + 1e-12 for a, b in zip(survival, survival[1:])), survival
    assert 0.0 <= survival[-1] <= 1.0


def test_component_posteriors_are_probabilities(model, filtered):
    _, probs = filtered
    comp = component_high_probabilities(model, probs[-1])
    assert comp.shape == (PARAMS.k_components,)
    assert ((comp >= 0) & (comp <= 1)).all()


def test_snapshot_is_serialisable(classifier, filtered):
    _, probs = filtered
    snap = classifier.snapshot(probs[-1], horizon_bars=30)
    payload = snap.to_dict()
    assert payload["regime"] in {r.name for r in Regime}
    assert payload["sigma_horizon"] > payload["sigma_bar"]


def test_horizon_volatility_exceeds_one_bar(classifier, filtered):
    _, probs = filtered
    one = classifier.snapshot(probs[-1], horizon_bars=1)
    thirty = classifier.snapshot(probs[-1], horizon_bars=30)
    assert thirty.sigma_horizon > one.sigma_horizon


# --------------------------------------------------------------------------
# MF-DFA
# --------------------------------------------------------------------------
def test_mfdfa_flags_iid_gaussian_as_monofractal():
    noise = np.random.default_rng(0).standard_normal(4000) * 0.01
    report = multifractal_report(noise)
    assert not report["is_multifractal"]
    assert report["excess_width"] < 0.1


def test_mfdfa_flags_msm_data_as_multifractal(model):
    returns, _ = model.simulate(6000, seed=21)
    report = multifractal_report(returns)
    assert report["is_multifractal"]
    assert report["excess_width"] > 0.05


def test_mfdfa_hurst_of_random_walk_is_about_half():
    noise = np.random.default_rng(4).standard_normal(6000)
    assert mfdfa(noise).hurst == pytest.approx(0.5, abs=0.08)


def test_mfdfa_short_multifractal_sample_is_inconclusive_not_negative(model):
    """A short sample must not produce a confident false negative.

    MF-DFA needs long series. At a few thousand points a genuinely multifractal
    series routinely fails to clear the shuffled benchmark, and reporting
    "monofractal" there would send someone away from a model that would in fact
    have helped.
    """
    returns, _ = model.simulate(5000, seed=33)
    report = multifractal_report(returns)
    assert report["is_multifractal"] or report["inconclusive"], report["verdict"]
    if report["inconclusive"]:
        assert "short for MF-DFA" in report["verdict"]


def test_mfdfa_reports_sample_size(model):
    returns, _ = model.simulate(3000, seed=34)
    assert multifractal_report(returns)["n_observations"] == 3000


def test_mfdfa_rejects_short_series():
    with pytest.raises(ValueError):
        mfdfa(np.zeros(50))


# --------------------------------------------------------------------------
# News parsing
# --------------------------------------------------------------------------
def test_score_to_edge_is_centred_and_clipped():
    assert score_to_edge(50) == 0.0
    assert score_to_edge(100) == 1.0
    assert score_to_edge(0) == -1.0
    assert score_to_edge(75) == pytest.approx(0.5)


def test_parse_legacy_line_requires_a_default_time():
    assert parse_score_line("12, AAPL, 87") is None
    now = datetime(2024, 5, 1, tzinfo=timezone.utc)
    event = parse_score_line("12, AAPL, 87", default_time=now)
    assert event.symbol == "AAPL" and event.score == 87 and event.timestamp == now


def test_parse_extended_line():
    line = '7,TSLA,91,2024-05-01T13:30:00+00:00,"Beats on earnings",benzinga,42.5'
    event = parse_score_line(line)
    assert event.symbol == "TSLA"
    assert event.score == 91
    assert event.headline == "Beats on earnings"
    assert event.latency_ms == pytest.approx(42.5)


@pytest.mark.parametrize(
    "line",
    ["", "# comment", "1, $AAPL, 50", "1, NOTATICKERATALL, 50",
     "1, AAPL, notanumber", "1, AAPL, 150", "onlyonefield"],
)
def test_unusable_lines_are_skipped(line):
    assert parse_score_line(line, default_time=datetime.now(timezone.utc)) is None


def test_deduplicate_keeps_the_strongest_signal_in_a_window():
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    events = [
        NewsEvent(base, "AAPL", 72),
        NewsEvent(base + timedelta(seconds=10), "AAPL", 95),   # same story, stronger
        NewsEvent(base + timedelta(seconds=20), "MSFT", 60),
        NewsEvent(base + timedelta(seconds=600), "AAPL", 55),  # a genuinely new story
    ]
    kept = deduplicate(events, window_seconds=60)
    assert len(kept) == 3
    apple = [e for e in kept if e.symbol == "AAPL"]
    assert apple[0].score == 95


def test_decay_weight_halves_at_the_half_life():
    assert decay_weight(0) == 1.0
    assert decay_weight(300, 300) == pytest.approx(0.5)
    assert decay_weight(600, 300) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Fusion / sizing
# --------------------------------------------------------------------------
def test_default_config_is_regime_agnostic():
    """The out-of-the-box config must be the control, not a tuned strategy."""
    cfg = default_config()
    betas = {cfg.rule(r).beta for r in Regime}
    gates = {cfg.rule(r).min_abs_edge for r in Regime}
    assert len(betas) == 1 and len(gates) == 1


def test_position_size_is_inversely_proportional_to_volatility(classifier, filtered):
    """The core mechanism: same signal, calmer tape, bigger position."""
    _, probs = filtered
    vols = classifier.model.conditional_volatility_path(probs)
    calm, wild = probs[int(np.argmin(vols))], probs[int(np.argmax(vols))]

    # A generous cap so the risk budget, not the cap, sets the size.
    cfg = default_config()
    for rule in cfg.rules.values():
        rule.max_position_fraction = 1.0
    cfg.max_position_fraction = 1.0
    sizer = RegimeConditionedSizer(cfg)

    kwargs = dict(symbol="X", edge=0.8, classifier=classifier, price=100.0,
                  equity=1_000_000.0, allow_fractional=True)
    calm_decision = sizer.decide(probs=calm, **kwargs)
    wild_decision = sizer.decide(probs=wild, **kwargs)

    assert abs(calm_decision.target_notional) > abs(wild_decision.target_notional)
    # Notional should scale as 1/sigma, so the ratio tracks the vol ratio.
    ratio = abs(calm_decision.target_notional) / abs(wild_decision.target_notional)
    vol_ratio = wild_decision.sigma_horizon / calm_decision.sigma_horizon
    assert ratio == pytest.approx(vol_ratio, rel=0.05)


def test_weak_signals_are_declined(classifier, filtered):
    _, probs = filtered
    sizer = RegimeConditionedSizer(default_config())
    decision = sizer.decide(
        symbol="X", edge=0.01, classifier=classifier, probs=probs[-1],
        price=100.0, equity=1_000_000.0,
    )
    assert decision.side == "flat" and not decision.is_trade
    assert "below" in decision.reason


def test_negative_beta_flips_the_trade_direction(classifier, filtered):
    """If a regime's calibrated response is negative, good news must sell."""
    _, probs = filtered
    cfg = default_config()
    for rule in cfg.rules.values():
        rule.beta = -0.5
    decision = RegimeConditionedSizer(cfg).decide(
        symbol="X", edge=0.9, classifier=classifier, probs=probs[-1],
        price=100.0, equity=1_000_000.0,
    )
    assert decision.target_qty < 0


def test_shorting_can_be_disabled(classifier, filtered):
    from dataclasses import replace

    _, probs = filtered
    cfg = replace(default_config(), allow_short=False)
    decision = RegimeConditionedSizer(cfg).decide(
        symbol="X", edge=-0.9, classifier=classifier, probs=probs[-1],
        price=100.0, equity=1_000_000.0,
    )
    assert decision.side == "flat"


def test_gross_exposure_limit_is_respected(classifier, filtered):
    _, probs = filtered
    sizer = RegimeConditionedSizer(default_config())
    decision = sizer.decide(
        symbol="X", edge=0.9, classifier=classifier, probs=probs[-1],
        price=100.0, equity=1_000_000.0, gross_exposure=2_000_000.0,
    )
    assert decision.side == "flat"


def test_staleness_shrinks_the_edge(classifier, filtered):
    _, probs = filtered
    sizer = RegimeConditionedSizer(default_config())
    fresh = sizer.decide(symbol="X", edge=0.9, classifier=classifier, probs=probs[-1],
                         price=100.0, equity=1_000_000.0, staleness_weight=1.0)
    stale = sizer.decide(symbol="X", edge=0.9, classifier=classifier, probs=probs[-1],
                         price=100.0, equity=1_000_000.0, staleness_weight=0.3)
    assert abs(stale.edge) < abs(fresh.edge)


# --------------------------------------------------------------------------
# Calibration - the anti-overfitting guarantees
# --------------------------------------------------------------------------
def _observations(betas: dict[Regime, float], n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for regime, beta in betas.items():
        for _ in range(n):
            edge = float(rng.uniform(-1, 1))
            sigma = 0.01
            realised = (beta * edge + noise * rng.standard_normal()) * sigma
            out.append(CalibrationObservation(regime, edge, realised, sigma))
    return out


def test_calibration_recovers_a_real_regime_effect():
    truth = {Regime.CALM: 1.0, Regime.NORMAL: 0.6, Regime.TURBULENT: 0.2, Regime.CRISIS: -0.5}
    cfg = calibrate(_observations(truth, n=800, noise=0.3, seed=1))
    for regime, expected in truth.items():
        assert cfg.rule(regime).beta == pytest.approx(expected, abs=0.15)
    assert cfg.rule(Regime.CRISIS).beta < 0


def test_calibration_returns_flat_betas_under_the_null():
    """THE key test: no regime effect in, no regime effect out.

    A calibrator that manufactures differences when none exist would make every
    backtest in this repo meaningless.
    """
    truth = {r: 0.5 for r in Regime}
    cfg = calibrate(_observations(truth, n=600, noise=1.2, seed=2))
    betas = np.array([cfg.rule(r).beta for r in Regime])
    # The spread between regimes is what must collapse; the common level is only
    # as accurate as the pooled standard error allows.
    assert betas.std() < 0.05, betas
    assert np.allclose(betas, 0.5, atol=0.15), betas


def test_shrinkage_is_stronger_when_estimates_are_noisier():
    clean = calibrate(_observations({r: 0.5 for r in Regime}, n=400, noise=0.2, seed=3))
    noisy = calibrate(_observations({r: 0.5 for r in Regime}, n=400, noise=3.0, seed=3))
    clean_spread = np.std([clean.rule(r).beta for r in Regime])
    noisy_spread = np.std([noisy.rule(r).beta for r in Regime])
    assert noisy_spread <= clean_spread + 1e-9


def test_thin_regimes_fall_back_to_the_pooled_estimate():
    obs = _observations({Regime.NORMAL: 0.5}, n=200, noise=0.5, seed=4)
    obs += _observations({Regime.CRISIS: 5.0}, n=3, noise=0.5, seed=5)
    cfg = calibrate(obs, min_observations=20)
    assert cfg.rule(Regime.CRISIS).beta == pytest.approx(cfg.rule(Regime.NORMAL).beta, abs=0.2)
    assert cfg.rule(Regime.CRISIS).shrinkage_weight == 0.0


def test_calibration_on_empty_input_returns_the_base_config():
    cfg = calibrate([])
    assert cfg.rule(Regime.CALM).beta == default_config().rule(Regime.CALM).beta


def test_power_analysis_demands_more_data_for_smaller_effects():
    cfg = calibrate(_observations({r: 0.5 for r in Regime}, n=200, noise=1.0, seed=6))
    small = power_analysis(cfg, target_difference=0.1)["per_regime"]["NORMAL"]["required"]
    large = power_analysis(cfg, target_difference=0.5)["per_regime"]["NORMAL"]["required"]
    assert small > large
