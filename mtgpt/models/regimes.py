"""Turn MSM filtered state into a tradeable volatility regime.

The classifier answers three questions a news-driven strategy actually needs:

1. **How volatile is it right now, relative to this asset's own history?**
   Not in absolute vol units — a 40%-vol biotech and a 12%-vol utility need the
   same decision rule. We express the current conditional volatility as a
   quantile of the distribution the *fitted model itself* implies.

2. **Will it stay this way for as long as I intend to hold?**
   MSM gives an exact ``h``-step state distribution, so the probability that
   volatility is still elevated in ``h`` bars is a closed-form number rather
   than a guess.

3. **Is this a fast burst or a slow structural shift?**
   The per-component posteriors separate the two. A hot fast component decays
   in minutes; a hot slow component means the elevated regime outlives the
   trade. Two situations with identical current volatility, opposite correct
   holding periods.

Look-ahead
----------
Regime cutoffs come from **simulating the fitted model**, never from sample
quantiles of the realised series. Using in-sample quantiles would leak the
future into every historical regime label and inflate any backtest that
conditions on them. The only information used is the parameter vector, which in
a walk-forward run is itself estimated on strictly prior data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np

from .msm import MSMModel

__all__ = [
    "Regime",
    "RegimeClassifier",
    "RegimeSnapshot",
    "component_high_probabilities",
]

# Quantile boundaries between the four regimes, on the model-implied
# distribution of conditional volatility.
DEFAULT_QUANTILES: tuple[float, float, float] = (0.25, 0.60, 0.85)


class Regime(IntEnum):
    """Volatility regime, ordered from quietest to wildest."""

    CALM = 0
    NORMAL = 1
    TURBULENT = 2
    CRISIS = 3

    @property
    def label(self) -> str:
        return self.name

    @classmethod
    def from_name(cls, name: str) -> "Regime":
        try:
            return cls[name.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown regime {name!r}") from exc


def component_high_probabilities(model: MSMModel, probs: np.ndarray) -> np.ndarray:
    """``P(M_k = m0 | F_t)`` for each component, slowest first.

    Index 0 along every tensor axis is the high multiplier ``m0`` (see the
    Kronecker ordering in :class:`~mtgpt.models.msm.MSMModel`), so marginalising
    out the other axes and taking that slice gives the per-scale posterior.
    """
    n_comp = model.params.k_components
    tensor = np.asarray(probs, dtype=float).reshape((2,) * n_comp)
    out = np.empty(n_comp)
    for k in range(n_comp):
        axes = tuple(a for a in range(n_comp) if a != k)
        marginal = tensor.sum(axis=axes)
        out[k] = marginal[0]
    return out


@dataclass
class RegimeSnapshot:
    """Everything the sizing layer needs to know about the current state."""

    regime: Regime
    quantile: float
    """Where current conditional volatility sits in the model-implied
    distribution, in ``[0, 1]``. 0.9 means 'more volatile than 90% of this
    asset's own typical states'."""

    sigma_bar: float
    """One-bar-ahead conditional volatility forecast."""

    sigma_horizon: float
    """Volatility of the *cumulative* return over the planned holding period."""

    horizon_bars: int
    survival_probability: float
    """``P(latent volatility is still at least this elevated in horizon_bars)``."""

    expected_regime_bars: int
    """Bars until that survival probability first drops below one half."""

    component_probabilities: list[float] = field(default_factory=list)
    """``P(M_k = high)`` per component, slowest first."""

    fast_share: float = 0.0
    """Fraction of current excess volatility attributable to the faster half of
    the components. High means a transient burst, low means a structural shift."""

    n_observations: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime.name
        return d


class RegimeClassifier:
    """Maps MSM conditional volatility onto :class:`Regime` labels."""

    def __init__(
        self,
        model: MSMModel,
        forecast_cutoffs: Sequence[float],
        state_cutoffs: Sequence[float],
        *,
        horizon_bars: int = 1,
        reference_vols: np.ndarray | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ):
        if len(forecast_cutoffs) != 3 or len(state_cutoffs) != 3:
            raise ValueError("expected exactly three cutoffs (four regimes)")
        self.model = model
        self.forecast_cutoffs = np.asarray(forecast_cutoffs, dtype=float)
        self.state_cutoffs = np.asarray(state_cutoffs, dtype=float)
        self.horizon_bars = int(horizon_bars)
        self.quantiles = tuple(quantiles)
        # Sorted sample of model-implied conditional vols, used to report where
        # an observed vol falls without re-simulating.
        self._reference = (
            np.sort(np.asarray(reference_vols, dtype=float))
            if reference_vols is not None
            else None
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: MSMModel,
        *,
        horizon_bars: int = 1,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        n_sim: int = 40_000,
        seed: int | None = 7,
    ) -> "RegimeClassifier":
        """Calibrate cutoffs by simulating and filtering the fitted model.

        This is the look-ahead-free construction: it consumes the parameter
        vector and nothing else about the realised path.
        """
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantiles must lie strictly inside (0, 1)")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("quantiles must be increasing")

        sim_returns, _ = model.simulate(n_sim, seed=seed)
        _, filtered = model.filter(sim_returns, return_states=True)
        cond_vol = model.conditional_volatility_path(filtered, horizon=1)
        forecast_cutoffs = np.quantile(cond_vol, quantiles)

        # State-level cutoffs come out of the analytic stationary distribution,
        # no simulation required. They are the scale on which survival
        # probabilities are evaluated.
        vols, weights = model.unconditional_state_distribution()
        cdf = np.cumsum(weights)
        state_cutoffs = np.array(
            [vols[int(np.searchsorted(cdf, q, side="left"))] for q in quantiles]
        )

        return cls(
            model,
            forecast_cutoffs,
            state_cutoffs,
            horizon_bars=horizon_bars,
            reference_vols=cond_vol,
            quantiles=quantiles,
        )

    # -- classification --------------------------------------------------
    def classify(self, sigma_bar: float) -> Regime:
        return Regime(int(np.searchsorted(self.forecast_cutoffs, sigma_bar, side="right")))

    def quantile_of(self, sigma_bar: float) -> float:
        if self._reference is None or self._reference.size == 0:
            return float("nan")
        idx = int(np.searchsorted(self._reference, sigma_bar, side="right"))
        return idx / self._reference.size

    # -- survival --------------------------------------------------------
    def survival_probability(
        self, probs: np.ndarray, regime: Regime, horizon: int
    ) -> float:
        """``P(latent vol at t+h is still at or above this regime's floor)``.

        For :attr:`Regime.CALM` there is no floor, so this is 1 by definition.
        """
        if regime == Regime.CALM:
            return 1.0
        floor = self.state_cutoffs[int(regime) - 1]
        ph = self.model.h_step_probs(probs, max(int(horizon), 1))
        return float(ph[self.model.state_volatilities >= floor].sum())

    def expected_regime_bars(
        self, probs: np.ndarray, regime: Regime, *, max_bars: int = 512
    ) -> int:
        """First horizon at which the survival probability falls below 0.5."""
        if regime == Regime.CALM:
            return max_bars
        h = 1
        while h < max_bars:
            if self.survival_probability(probs, regime, h) < 0.5:
                return h
            h *= 2
        return max_bars

    # -- the main entry point -------------------------------------------
    def snapshot(
        self,
        probs: np.ndarray,
        *,
        horizon_bars: int | None = None,
        n_observations: int = 0,
    ) -> RegimeSnapshot:
        """Full regime description from a filtered state vector."""
        horizon = int(horizon_bars or self.horizon_bars)
        model = self.model

        sigma_bar = float(np.sqrt(model.forecast_variance(probs, 1)))
        sigma_horizon = float(model.forecast_volatility(probs, horizon))
        regime = self.classify(sigma_bar)

        comp = component_high_probabilities(model, probs)
        n_comp = comp.size
        # "Fast" = the faster half of the components (higher index = faster).
        split = n_comp // 2
        excess = np.clip(comp - 0.5, 0.0, None)
        total_excess = float(excess.sum())
        fast_share = float(excess[split:].sum() / total_excess) if total_excess > 1e-12 else 0.0

        return RegimeSnapshot(
            regime=regime,
            quantile=self.quantile_of(sigma_bar),
            sigma_bar=sigma_bar,
            sigma_horizon=sigma_horizon,
            horizon_bars=horizon,
            survival_probability=self.survival_probability(probs, regime, horizon),
            expected_regime_bars=self.expected_regime_bars(probs, regime),
            component_probabilities=[float(x) for x in comp],
            fast_share=fast_share,
            n_observations=int(n_observations),
        )

    def label_path(self, filtered: np.ndarray) -> np.ndarray:
        """Vectorised regime labels for a whole filtered-probability matrix."""
        vols = self.model.conditional_volatility_path(filtered, horizon=1)
        return np.searchsorted(self.forecast_cutoffs, vols, side="right")

    def describe(self) -> dict:
        return {
            "quantiles": list(self.quantiles),
            "forecast_cutoffs": self.forecast_cutoffs.tolist(),
            "state_cutoffs": self.state_cutoffs.tolist(),
            "horizon_bars": self.horizon_bars,
            "params": self.model.params.to_dict(),
        }
