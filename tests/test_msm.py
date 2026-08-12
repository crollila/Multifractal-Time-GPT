"""Correctness tests for the MSM engine.

These are mostly *identity* tests: things that must hold exactly regardless of
data, so a regression shows up as a hard failure rather than a slightly worse
backtest.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mtgpt.models.msm import MSMModel, MSMParams, select_k_components

PARAMS = MSMParams(m0=1.45, sigma=0.012, gamma_1=0.003, b=3.0, k_components=5)


@pytest.fixture(scope="module")
def model() -> MSMModel:
    return MSMModel(PARAMS)


@pytest.fixture(scope="module")
def simulated(model: MSMModel):
    return model.simulate(8000, seed=11)


# -- structural identities -------------------------------------------------
def test_transition_matrix_is_stochastic(model):
    A = model.transition_matrix
    assert A.shape == (32, 32)
    assert (A >= 0).all()
    assert np.allclose(A.sum(axis=1), 1.0)


def test_multipliers_have_unit_mean(model):
    """E[prod M_k] = 1, so sigma is genuinely the unconditional volatility."""
    assert model.state_multipliers.mean() == pytest.approx(1.0, abs=1e-12)
    pi = model.stationary_distribution()
    assert math.sqrt(pi @ model.state_variances) == pytest.approx(PARAMS.sigma, rel=1e-12)


def test_uniform_distribution_is_stationary(model):
    pi = model.stationary_distribution()
    assert np.allclose(pi @ model.transition_matrix, pi)


def test_h_step_matches_brute_force_matrix_power(model):
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(model.n_states))
    for h in (1, 2, 5, 13, 40):
        expected = probs @ np.linalg.matrix_power(model.transition_matrix, h)
        assert np.allclose(model.h_step_probs(probs, h), expected, atol=1e-12)


def test_long_horizon_converges_to_stationary(model):
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(model.n_states))
    assert np.allclose(
        model.h_step_probs(probs, 1_000_000), model.stationary_distribution(), atol=1e-9
    )


def test_collapsed_unconditional_matches_full_grid(model):
    vols, weights = model.unconditional_state_distribution()
    assert weights.sum() == pytest.approx(1.0)
    pi = model.stationary_distribution()
    assert (weights @ vols**2) == pytest.approx(pi @ model.state_variances, rel=1e-12)


def test_switching_probabilities_increase_with_frequency():
    gammas = PARAMS.switching_probabilities
    assert (np.diff(gammas) > 0).all(), "component 1 must be the slowest"
    assert gammas[0] == pytest.approx(PARAMS.gamma_1)


def test_invalid_parameters_are_rejected():
    for bad in (
        dict(m0=2.5), dict(m0=0.5), dict(sigma=-1.0),
        dict(gamma_1=0.0), dict(gamma_1=1.0), dict(b=0.9), dict(k_components=0),
    ):
        with pytest.raises(ValueError):
            MSMParams(**{**PARAMS.to_dict(), **bad})


# -- filtering -------------------------------------------------------------
def test_filtered_probabilities_are_valid(model, simulated):
    returns, _ = simulated
    ll, filtered = model.filter(returns)
    assert np.isfinite(ll)
    assert filtered.shape == (returns.size, model.n_states)
    assert np.allclose(filtered.sum(axis=1), 1.0)
    assert (filtered >= 0).all()


def test_incremental_filter_matches_batch(model, simulated):
    returns, _ = simulated
    state = model.filter_state()
    state.extend(returns[:400])
    _, batch = model.filter(returns[:400])
    assert np.allclose(state.probs, batch[-1], atol=1e-12)


def test_filter_recovers_latent_volatility(model, simulated):
    """The whole model is useless if the filter cannot see the hidden state."""
    returns, true_vol = simulated
    _, filtered = model.filter(returns)
    estimated = model.conditional_volatility_path(filtered)
    assert np.corrcoef(estimated, true_vol)[0, 1] > 0.6


def test_filter_survives_an_extreme_outlier(model):
    """A 50-sigma print must not produce NaNs anywhere in the state."""
    returns, _ = model.simulate(300, seed=3)
    returns[150] = 50 * PARAMS.sigma
    ll, filtered = model.filter(returns)
    assert np.isfinite(filtered).all()
    assert np.allclose(filtered.sum(axis=1), 1.0)


def test_empty_input_is_handled(model):
    ll, filtered = model.filter([])
    assert ll == 0.0
    assert filtered.shape == (0, model.n_states)


# -- forecasting -----------------------------------------------------------
def test_cumulative_variance_is_the_sum_of_per_bar_variances(model, simulated):
    returns, _ = simulated
    _, filtered = model.filter(returns)
    probs = filtered[-1]
    manual = sum(model.forecast_variance(probs, h) for h in range(1, 25))
    assert model.cumulative_variance(probs, 24) == pytest.approx(manual, rel=1e-12)


def test_forecast_variance_reverts_toward_unconditional(model, simulated):
    """From an extreme state, long-horizon variance must decay to sigma^2."""
    returns, _ = simulated
    _, filtered = model.filter(returns)
    hottest = filtered[int(np.argmax(model.conditional_volatility_path(filtered)))]
    near = model.forecast_variance(hottest, 1)
    far = model.forecast_variance(hottest, 500_000)
    assert near > PARAMS.sigma**2
    assert far == pytest.approx(PARAMS.sigma**2, rel=1e-6)


def test_cumulative_volatility_grows_with_horizon(model, simulated):
    returns, _ = simulated
    _, filtered = model.filter(returns)
    vols = [model.forecast_volatility(filtered[-1], h) for h in (1, 5, 20, 60)]
    assert all(b > a for a, b in zip(vols, vols[1:]))


# -- simulation and estimation --------------------------------------------
def test_simulation_matches_target_moments(model):
    returns, _ = model.simulate(60_000, seed=5)
    assert returns.std() == pytest.approx(PARAMS.sigma, rel=0.15)
    kurtosis = float(((returns - returns.mean()) ** 4).mean() / returns.var() ** 2)
    assert kurtosis > 4.0, "MSM must produce fat tails"


@pytest.mark.slow
def test_mle_recovers_known_parameters():
    truth = MSMParams(m0=1.45, sigma=0.012, gamma_1=0.003, b=3.5, k_components=4)
    returns, _ = MSMModel(truth).simulate(6000, seed=101)
    fit = MSMModel.fit(returns, k_components=4, n_starts=3, seed=0)
    # m0 and sigma are well identified; gamma_1 and b trade off against each
    # other and are only weakly identified, which is a known property of MSM.
    assert fit.params.m0 == pytest.approx(truth.m0, abs=0.08)
    assert fit.params.sigma == pytest.approx(truth.sigma, rel=0.20)


@pytest.mark.slow
def test_msm_beats_constant_volatility_on_msm_data():
    returns, _ = MSMModel(PARAMS).simulate(5000, seed=77)
    fitted = MSMModel.fit(returns, k_components=4, n_starts=2)
    flat = MSMParams(m0=1.0, sigma=float(returns.std()), gamma_1=0.5, b=2.0, k_components=4)
    assert fitted.log_likelihood > MSMModel(flat).log_likelihood(returns)


def test_fit_rejects_too_little_data():
    with pytest.raises(ValueError, match="at least 50"):
        MSMModel.fit(np.zeros(10))


@pytest.mark.slow
def test_select_k_returns_the_bic_minimiser():
    returns, _ = MSMModel(PARAMS).simulate(3000, seed=9)
    best, all_fits = select_k_components(returns, candidates=(2, 3, 4), n_starts=1)
    assert best is all_fits[min(all_fits, key=lambda k: all_fits[k].bic)]
