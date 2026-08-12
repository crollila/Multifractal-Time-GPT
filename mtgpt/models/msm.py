"""Binomial Markov-Switching Multifractal (MSM) volatility model.

Reference
---------
Calvet, L. and Fisher, A. (2004), "How to Forecast Long-Run Volatility:
Regime Switching and the Estimation of Multifractal Processes",
Journal of Financial Econometrics 2(1), 49-83.

Model
-----
    r_t     = sigma_t * eps_t,        eps_t ~ N(0, 1)
    sigma_t = sigma_bar * sqrt(prod_{k=1..K} M_{k,t})

Each volatility component ``M_{k,t}`` is drawn from the binomial distribution
``{m0, 2 - m0}`` with equal probability, so ``E[M] = 1`` and ``sigma_bar`` is
the unconditional volatility. At each step component ``k`` is redrawn with
probability

    gamma_k = 1 - (1 - gamma_1) ** (b ** (k - 1)),      b > 1

so ``k = 1`` is the slowest (most persistent) component and ``k = K`` the
fastest. The hidden state is the vector ``(M_1, ..., M_K)``, giving ``2**K``
states with a Kronecker-factored transition matrix.

Why this model for event-driven trading
---------------------------------------
Unlike GARCH, MSM carries volatility components at *many* time scales at once.
That matters for news trading because the same headline lands very differently
depending on which scales are currently hot: a slow component being high means
an elevated-vol regime that will persist for days, while a fast component being
high is a transient burst that decays within minutes. The multi-horizon
forecast is exact in closed form, so a trade can be sized against the
volatility expected over its *actual holding period* rather than against
trailing realised volatility.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

__all__ = [
    "MSMParams",
    "MSMModel",
    "MSMFilterState",
    "FitResult",
    "select_k_components",
]

_LOG_2PI = math.log(2.0 * math.pi)
_EPS = 1e-300


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MSMParams:
    """Structural parameters of the binomial MSM.

    Attributes
    ----------
    m0:
        High multiplier, in ``[1, 2)``. The low multiplier is ``2 - m0``.
        ``m0 = 1`` collapses the model to constant volatility; values near 2
        mean extreme multifractality.
    sigma:
        Unconditional volatility per period, in the same units as the returns
        passed to the model (e.g. per-bar log-return standard deviation).
    gamma_1:
        Switching probability of the *slowest* component, in ``(0, 1)``.
    b:
        Frequency ratio between adjacent components, ``> 1``.
    k_components:
        Number of multiplicative components ``K``. The state space is ``2**K``.
    """

    m0: float
    sigma: float
    gamma_1: float
    b: float
    k_components: int

    def __post_init__(self) -> None:
        if not 1.0 <= self.m0 < 2.0:
            raise ValueError(f"m0 must lie in [1, 2), got {self.m0}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if not 0.0 < self.gamma_1 < 1.0:
            raise ValueError(f"gamma_1 must lie in (0, 1), got {self.gamma_1}")
        if self.b <= 1.0:
            raise ValueError(f"b must exceed 1, got {self.b}")
        if self.k_components < 1:
            raise ValueError("k_components must be >= 1")

    @property
    def switching_probabilities(self) -> np.ndarray:
        """``gamma_k`` for ``k = 1..K``, slowest first."""
        k = np.arange(1, self.k_components + 1, dtype=float)
        gamma = 1.0 - (1.0 - self.gamma_1) ** (self.b ** (k - 1.0))
        # Guard the fast end: gamma_k -> 1 makes a component i.i.d., which is
        # legitimate, but exactly 1.0 breaks the (1-gamma)**h forecast recursion.
        return np.clip(gamma, 1e-12, 1.0 - 1e-12)

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected number of bars each component survives before redrawing."""
        return 1.0 / self.switching_probabilities

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitResult:
    params: MSMParams
    log_likelihood: float
    n_observations: int
    converged: bool
    n_starts: int

    @property
    def aic(self) -> float:
        # Four free parameters: m0, sigma, gamma_1, b (K is chosen, not fitted).
        return 2.0 * 4 - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return 4 * math.log(max(self.n_observations, 2)) - 2.0 * self.log_likelihood


# --------------------------------------------------------------------------
# Online filter state
# --------------------------------------------------------------------------
class MSMFilterState:
    """Incremental Hamilton filter, for streaming use in the live service.

    Holding this per symbol means a new bar costs one matrix-vector product
    (microseconds for the usual ``K <= 8``) instead of re-filtering the whole
    history, which is what keeps the sizing endpoint inside its latency budget.
    """

    __slots__ = ("model", "probs", "n_seen", "log_likelihood")

    def __init__(self, model: "MSMModel", probs: np.ndarray | None = None):
        self.model = model
        if probs is None:
            probs = np.full(model.n_states, 1.0 / model.n_states)
        self.probs = np.asarray(probs, dtype=float)
        self.n_seen = 0
        self.log_likelihood = 0.0

    def step(self, r: float) -> float:
        """Absorb one return. Returns the predictive log density of ``r``."""
        model = self.model
        predicted = self.probs @ model.transition_matrix
        weights = model.emission_weights(r)
        posterior = predicted * weights
        total = float(posterior.sum())
        if total <= _EPS or not np.isfinite(total):
            # Numerically impossible observation (e.g. a huge gap). Fall back to
            # the predictive distribution rather than propagating NaNs into a
            # live trading decision.
            self.probs = predicted
            self.n_seen += 1
            return float("-inf")
        self.probs = posterior / total
        self.n_seen += 1
        ll = math.log(total)
        self.log_likelihood += ll
        return ll

    def extend(self, returns: Iterable[float]) -> None:
        for r in returns:
            self.step(float(r))

    def forecast_variance(self, horizon: int = 1) -> float:
        return self.model.forecast_variance(self.probs, horizon)

    def forecast_volatility(self, horizon: int = 1) -> float:
        """Volatility of the *cumulative* return over the next ``horizon`` bars."""
        return self.model.forecast_volatility(self.probs, horizon)

    def copy(self) -> "MSMFilterState":
        clone = MSMFilterState(self.model, self.probs.copy())
        clone.n_seen = self.n_seen
        clone.log_likelihood = self.log_likelihood
        return clone


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MSMModel:
    """A fitted (or hand-specified) MSM, with filtering and forecasting."""

    def __init__(self, params: MSMParams):
        self.params = params
        n_comp = params.k_components
        self.n_states = 2**n_comp

        levels = np.array([params.m0, 2.0 - params.m0], dtype=float)
        # Kronecker order: component 1 (slowest) is the most significant index,
        # matching the transition matrix below and the (2,)*K tensor view.
        self.state_multipliers = functools.reduce(np.kron, [levels] * n_comp)
        self.state_variances = params.sigma**2 * self.state_multipliers
        self.state_volatilities = np.sqrt(self.state_variances)

        self._gammas = params.switching_probabilities
        self.transition_matrix = self._build_transition(self._gammas)

        # Per-component multiplier value in each state, shape (K, 2**K). Used by
        # the closed-form forecast weights below.
        components = np.empty((n_comp, self.n_states))
        for k in range(n_comp):
            shape = [1] * n_comp
            shape[k] = 2
            components[k] = np.broadcast_to(
                levels.reshape(shape), (2,) * n_comp
            ).reshape(-1)
        self._component_multipliers = components

        # Cached constants for the Gaussian emission density.
        self._inv_two_var = 0.5 / self.state_variances
        self._log_norm = -0.5 * (_LOG_2PI + np.log(self.state_variances))

        # Horizon -> weight vector caches (see _per_bar_weights).
        self._per_bar_cache: dict[int, np.ndarray] = {}
        self._cumulative_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _build_transition(gammas: np.ndarray) -> np.ndarray:
        blocks = [
            # Stay with prob 1-g; otherwise redraw uniformly from {m0, 2-m0}.
            np.array([[1 - g / 2, g / 2], [g / 2, 1 - g / 2]])
            for g in gammas
        ]
        return functools.reduce(np.kron, blocks)

    def stationary_distribution(self) -> np.ndarray:
        """Uniform over the ``2**K`` states, by construction of the binomial MSM."""
        return np.full(self.n_states, 1.0 / self.n_states)

    # -- emissions -------------------------------------------------------
    def emission_weights(self, r: float) -> np.ndarray:
        """Gaussian densities ``p(r | state)`` for every state."""
        return np.exp(self._log_norm - (r * r) * self._inv_two_var)

    def log_emission_matrix(self, returns: np.ndarray) -> np.ndarray:
        """``(T, 2**K)`` matrix of log densities."""
        r2 = np.asarray(returns, dtype=float) ** 2
        return self._log_norm[None, :] - r2[:, None] * self._inv_two_var[None, :]

    # -- filtering -------------------------------------------------------
    def filter(
        self, returns: Sequence[float] | np.ndarray, *, return_states: bool = True
    ) -> tuple[float, np.ndarray | None]:
        """Run the Hamilton filter over ``returns``.

        Returns ``(log_likelihood, filtered_probabilities)`` where the
        probability array has shape ``(T, 2**K)`` and row ``t`` is
        ``P(state_t | r_1..r_t)``.
        """
        r = np.asarray(returns, dtype=float).ravel()
        n_obs = r.size
        if n_obs == 0:
            return 0.0, (np.empty((0, self.n_states)) if return_states else None)

        transition = self.transition_matrix
        # Subtract the row max before exponentiating so that a very small or
        # very large return cannot underflow every state to zero at once.
        log_w = self.log_emission_matrix(r)
        row_max = log_w.max(axis=1)
        weights = np.exp(log_w - row_max[:, None])

        probs = np.full(self.n_states, 1.0 / self.n_states)
        out = np.empty((n_obs, self.n_states)) if return_states else None
        ll = 0.0
        for t in range(n_obs):
            predicted = probs @ transition
            posterior = predicted * weights[t]
            total = posterior.sum()
            if total <= _EPS or not np.isfinite(total):
                probs = predicted
                ll += -1e10  # hard penalty; keeps the optimiser out of this region
                if out is not None:
                    out[t] = probs
                continue
            probs = posterior / total
            ll += math.log(total) + row_max[t]
            if out is not None:
                out[t] = probs
        return float(ll), out

    def log_likelihood(self, returns: Sequence[float] | np.ndarray) -> float:
        return self.filter(returns, return_states=False)[0]

    def filter_state(
        self, returns: Sequence[float] | np.ndarray | None = None
    ) -> MSMFilterState:
        """Build an online filter state, optionally warmed up on history."""
        state = MSMFilterState(self)
        if returns is not None and len(returns) > 0:
            _, probs = self.filter(returns, return_states=True)
            state.probs = probs[-1].copy()
            state.n_seen = len(returns)
        return state

    # -- forecasting -----------------------------------------------------
    def h_step_probs(self, probs: np.ndarray, horizon: int) -> np.ndarray:
        """Exact ``h``-step-ahead state distribution.

        Uses ``A_k = (1-g)I + g*P`` with ``P`` idempotent, hence
        ``A_k**h = (1-g)**h I + (1-(1-g)**h) P``: an ``h``-step forecast is the
        one-step operator with ``gamma_k`` replaced by ``1-(1-gamma_k)**h``.
        Cost is ``O(K * 2**K)`` regardless of how far ahead we look.
        """
        p = np.asarray(probs, dtype=float)
        if horizon <= 0:
            return p
        tensor = p.reshape((2,) * self.params.k_components)
        effective = 1.0 - (1.0 - self._gammas) ** horizon
        for axis, g in enumerate(effective):
            tensor = (1.0 - g) * tensor + g * tensor.mean(axis=axis, keepdims=True)
        return tensor.reshape(-1)

    def _forecast_weights(self, horizon: int) -> np.ndarray:
        """Per-state weights ``w`` such that ``E[sigma_{t+h}^2|F_t] = probs @ w``.

        Because the components evolve independently, the ``h``-step conditional
        expectation of ``prod_k M_k`` factorises::

            E[prod_k M_k(t+h) | state] = prod_k (a_k * M_k(state) + 1 - a_k)

        with ``a_k = (1 - gamma_k)**h`` the probability that component ``k`` has
        not been redrawn. That turns an ``O(K * 2**K)`` tensor propagation
        followed by a dot product into a single cached dot product, which is
        what makes the live sizing endpoint sub-millisecond.
        """
        h = max(int(horizon), 1)
        cached = self._per_bar_cache.get(h)
        if cached is None:
            a = (1.0 - self._gammas) ** h
            terms = a[:, None] * self._component_multipliers + (1.0 - a)[:, None]
            cached = self.params.sigma**2 * terms.prod(axis=0)
            self._per_bar_cache[h] = cached
        return cached

    def _cumulative_weights(self, horizon: int) -> np.ndarray:
        """Weights ``w`` such that ``Var[sum of next h returns] = probs @ w``."""
        h = max(int(horizon), 1)
        cached = self._cumulative_cache.get(h)
        if cached is None:
            steps = np.arange(1, h + 1)
            a = (1.0 - self._gammas)[:, None] ** steps[None, :]  # (K, h)
            terms = (
                a[:, None, :] * self._component_multipliers[:, :, None]
                + (1.0 - a)[:, None, :]
            )
            cached = self.params.sigma**2 * terms.prod(axis=0).sum(axis=1)
            self._cumulative_cache[h] = cached
        return cached

    def forecast_variance(self, probs: np.ndarray, horizon: int = 1) -> float:
        """``E[sigma_{t+h}^2 | F_t]`` — the *per-bar* variance ``h`` bars out."""
        return float(np.asarray(probs, dtype=float) @ self._forecast_weights(horizon))

    def cumulative_variance(self, probs: np.ndarray, horizon: int = 1) -> float:
        """``Var[r_{t+1} + ... + r_{t+h} | F_t]``.

        Returns are conditionally uncorrelated under MSM, so the cumulative
        variance is the sum of the per-bar forecasts. This — not ``h`` times the
        one-step variance — is what a holding-period position should be sized
        against, and the two differ materially when volatility is mean
        reverting, which is exactly the case right after a news shock.
        """
        return float(np.asarray(probs, dtype=float) @ self._cumulative_weights(horizon))

    def forecast_volatility(self, probs: np.ndarray, horizon: int = 1) -> float:
        return math.sqrt(self.cumulative_variance(probs, horizon))

    def conditional_volatility_path(
        self, filtered: np.ndarray, horizon: int = 1
    ) -> np.ndarray:
        """Per-bar forecast volatility for every row of a filtered-prob matrix."""
        filtered = np.atleast_2d(np.asarray(filtered, dtype=float))
        if horizon == 1:
            ph = filtered @ self.transition_matrix
            return np.sqrt(ph @ self.state_variances)
        return np.array(
            [math.sqrt(self.forecast_variance(row, horizon)) for row in filtered]
        )

    # -- unconditional / model-implied quantities ------------------------
    def unconditional_state_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Distinct per-state volatilities and their stationary probabilities.

        ``prod_k M_k`` depends only on how many of the ``K`` multipliers are
        high, so the distribution collapses to ``K+1`` points with binomial
        weights.
        """
        n_comp = self.params.k_components
        j = np.arange(n_comp + 1)
        weights = comb(n_comp, j) / 2.0**n_comp
        vols = self.params.sigma * np.sqrt(
            self.params.m0**j * (2.0 - self.params.m0) ** (n_comp - j)
        )
        order = np.argsort(vols)
        return vols[order], weights[order]

    # -- simulation ------------------------------------------------------
    def simulate(
        self, n: int, *, seed: int | None = None, burn_in: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate ``(returns, true_volatility)`` from the model."""
        rng = np.random.default_rng(seed)
        n_comp = self.params.k_components
        gammas = self._gammas
        levels = np.array([self.params.m0, 2.0 - self.params.m0])

        total = n + burn_in
        multipliers = levels[rng.integers(0, 2, size=n_comp)]
        vols = np.empty(total)
        for t in range(total):
            redraw = rng.random(n_comp) < gammas
            if redraw.any():
                multipliers = np.where(
                    redraw, levels[rng.integers(0, 2, size=n_comp)], multipliers
                )
            vols[t] = self.params.sigma * math.sqrt(float(np.prod(multipliers)))
        rets = vols * rng.standard_normal(total)
        return rets[burn_in:], vols[burn_in:]

    # -- estimation ------------------------------------------------------
    @classmethod
    def fit(
        cls,
        returns: Sequence[float] | np.ndarray,
        *,
        k_components: int = 6,
        n_starts: int = 4,
        seed: int | None = 0,
        demean: bool = True,
        maxiter: int = 400,
    ) -> FitResult:
        """Maximum-likelihood estimation of ``(m0, sigma, gamma_1, b)``.

        ``k_components`` is held fixed — it is a modelling choice about how many
        time scales matter, best selected by comparing BIC across a few values
        (see :func:`select_k_components`).
        """
        r = np.asarray(returns, dtype=float).ravel()
        r = r[np.isfinite(r)]
        if r.size < 50:
            raise ValueError(f"need at least 50 returns to fit MSM, got {r.size}")
        if demean:
            r = r - r.mean()

        sample_sd = float(r.std(ddof=1))
        if sample_sd <= 0:
            raise ValueError("returns have zero variance")

        def unpack(x: np.ndarray) -> MSMParams:
            m0 = 1.0 + 0.98 / (1.0 + math.exp(-float(x[0])))
            sigma = math.exp(float(x[1]))
            gamma_1 = 1.0 / (1.0 + math.exp(-float(x[2])))
            b = 1.0 + math.exp(float(x[3]))
            return MSMParams(m0, sigma, gamma_1, b, k_components)

        def negative_ll(x: np.ndarray) -> float:
            try:
                model = cls(unpack(x))
            except (ValueError, FloatingPointError, OverflowError):
                return 1e12
            ll = model.log_likelihood(r)
            return 1e12 if not np.isfinite(ll) else -ll

        rng = np.random.default_rng(seed)
        # First start is an economic prior: moderate multifractality, sigma at
        # the sample level, slowest component lasting ~400 bars, b ~ 3.
        starts = [np.array([0.2, math.log(sample_sd), -6.0, math.log(2.0)])]
        for _ in range(max(n_starts - 1, 0)):
            starts.append(
                np.array(
                    [
                        rng.uniform(-1.0, 1.5),
                        math.log(sample_sd) + rng.uniform(-0.3, 0.3),
                        rng.uniform(-8.0, -3.0),
                        math.log(rng.uniform(1.0, 6.0)),
                    ]
                )
            )

        best, best_ll, converged = None, -np.inf, False
        for x0 in starts:
            res = minimize(
                negative_ll,
                x0,
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
            )
            if -res.fun > best_ll:
                best_ll, best, converged = -float(res.fun), res.x, bool(res.success)

        if best is None:
            raise RuntimeError("MSM estimation failed from every start")

        return FitResult(
            params=unpack(best),
            log_likelihood=float(best_ll),
            n_observations=int(r.size),
            converged=converged,
            n_starts=len(starts),
        )


def select_k_components(
    returns: Sequence[float] | np.ndarray,
    candidates: Sequence[int] = (3, 4, 5, 6, 7),
    **fit_kwargs,
) -> tuple[FitResult, dict[int, FitResult]]:
    """Fit MSM for several ``K`` and pick the best by BIC."""
    results: dict[int, FitResult] = {}
    for k in candidates:
        try:
            results[k] = MSMModel.fit(returns, k_components=k, **fit_kwargs)
        except Exception:  # noqa: BLE001 - one bad K must not kill the sweep
            continue
    if not results:
        raise RuntimeError("no candidate K produced a valid fit")
    best_k = min(results, key=lambda k: results[k].bic)
    return results[best_k], results
