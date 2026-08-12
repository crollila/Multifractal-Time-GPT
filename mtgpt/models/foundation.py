"""Time-series foundation-model layer (TimeGPT) with an offline baseline.

Division of labour
------------------
MSM is a pure *volatility* model — it says nothing about direction. The
foundation model supplies the missing half: a conditional **mean** path over the
holding period, optionally conditioned on the news sentiment score as an
exogenous regressor. Together they give the only quantity a sizing rule really
needs::

    edge_z = expected cumulative return over horizon / MSM volatility over horizon

which is a forecast information ratio, directly comparable across symbols and
across regimes.

Backends
--------
:class:`ThetaBackend` is the default and needs no network, no key, and no
install beyond numpy. It is the classical Theta method — the statistical
baseline that foundation-model papers benchmark against. Keeping it as the
default matters for two reasons: the repo has to run end-to-end for anyone who
clones it, and it is the control that tells you whether TimeGPT is *earning* its
API call. :func:`compare_backends` runs exactly that horse race.

:class:`TimeGPTBackend` calls Nixtla's hosted TimeGPT. Requires
``pip install nixtla`` and ``NIXTLA_API_KEY``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "Forecast",
    "ForecastBackend",
    "ThetaBackend",
    "DriftlessBackend",
    "TimeGPTBackend",
    "get_backend",
    "compare_backends",
]

logger = logging.getLogger(__name__)


@dataclass
class Forecast:
    """A conditional mean path plus an uncertainty band."""

    mean: np.ndarray
    """Forecast levels for ``t+1 .. t+h``."""

    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    backend: str = "unknown"
    level: float = 80.0
    metadata: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.mean.size)

    def cumulative_return(self, last_value: float) -> float:
        """Expected log return from ``last_value`` to the end of the horizon."""
        if last_value <= 0 or self.mean.size == 0:
            return 0.0
        terminal = float(self.mean[-1])
        if terminal <= 0:
            return 0.0
        return math.log(terminal / last_value)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "horizon": self.horizon,
            "mean": self.mean.tolist(),
            "lo": None if self.lo is None else self.lo.tolist(),
            "hi": None if self.hi is None else self.hi.tolist(),
        }


@runtime_checkable
class ForecastBackend(Protocol):
    """Anything that can turn a level series into a forward mean path."""

    name: str

    def forecast(
        self,
        y: Sequence[float] | np.ndarray,
        horizon: int,
        *,
        exog: np.ndarray | None = None,
        exog_future: np.ndarray | None = None,
        freq: str | None = None,
    ) -> Forecast: ...


# --------------------------------------------------------------------------
# Offline baselines
# --------------------------------------------------------------------------
class DriftlessBackend:
    """Random walk: the null hypothesis for any price series.

    Included because it is the benchmark that actually matters. On liquid
    intraday equity data a random walk is very hard to beat, and a strategy that
    only works when the mean model beats a flat line should know that about
    itself.
    """

    name = "random_walk"

    def forecast(self, y, horizon, *, exog=None, exog_future=None, freq=None) -> Forecast:
        arr = np.asarray(y, dtype=float).ravel()
        last = float(arr[-1])
        mean = np.full(int(horizon), last)
        sd = float(np.diff(arr).std(ddof=1)) if arr.size > 2 else 0.0
        band = 1.2816 * sd * np.sqrt(np.arange(1, horizon + 1))  # 80% normal band
        return Forecast(mean=mean, lo=mean - band, hi=mean + band, backend=self.name)


class ThetaBackend:
    """Classical Theta method (Assimakopoulos & Nikolopoulos, 2000).

    Simple exponential smoothing for the level, plus half the OLS trend slope
    extrapolated forward. This is the standard strong statistical baseline in
    forecasting competitions and the natural control for TimeGPT.
    """

    name = "theta"

    def __init__(self, alpha: float | None = None, damping: float = 1.0):
        self.alpha = alpha
        self.damping = float(damping)

    @staticmethod
    def _ses(arr: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
        """Simple exponential smoothing; returns final level and one-step errors."""
        level = float(arr[0])
        errors = np.empty(arr.size - 1)
        for i in range(1, arr.size):
            errors[i - 1] = arr[i] - level
            level += alpha * errors[i - 1]
        return level, errors

    def _fit_alpha(self, arr: np.ndarray) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        best_alpha, best_sse = 0.3, float("inf")
        for alpha in np.linspace(0.05, 0.95, 19):
            _, errors = self._ses(arr, float(alpha))
            sse = float((errors**2).sum())
            if sse < best_sse:
                best_alpha, best_sse = float(alpha), sse
        return best_alpha

    def forecast(self, y, horizon, *, exog=None, exog_future=None, freq=None) -> Forecast:
        arr = np.asarray(y, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        horizon = int(horizon)
        if arr.size < 5:
            return DriftlessBackend().forecast(arr, horizon)

        n = arr.size
        t = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, arr, 1)[0])

        alpha = self._fit_alpha(arr)
        level, errors = self._ses(arr, alpha)

        # Standard Theta point forecast: SES level plus half the linear drift.
        h = np.arange(1, horizon + 1, dtype=float)
        decay = (
            h if self.damping >= 1.0
            else np.cumsum(self.damping ** np.arange(1, horizon + 1))
        )
        offset = (1.0 - (1.0 - alpha) ** n) / alpha
        mean = level + (slope / 2.0) * (decay - 1.0 + offset)

        sd = float(errors.std(ddof=1)) if errors.size > 2 else 0.0
        band = 1.2816 * sd * np.sqrt(h)
        return Forecast(
            mean=mean, lo=mean - band, hi=mean + band, backend=self.name,
            metadata={"alpha": alpha, "slope": slope},
        )


# --------------------------------------------------------------------------
# TimeGPT
# --------------------------------------------------------------------------
class TimeGPTBackend:
    """Nixtla TimeGPT, with exogenous features and a graceful fallback.

    The sentiment score and the MSM volatility forecast are passed as exogenous
    regressors when supplied, which is the whole point of using a foundation
    model here rather than a univariate one: the model can learn that the same
    price history implies a different forward path depending on what just hit
    the tape and how volatile the tape currently is.

    Any failure — missing package, missing key, network error, malformed
    response — falls back to ``fallback`` and is logged. A live trading loop
    must never die because a forecasting API had a bad minute.
    """

    name = "timegpt"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "timegpt-1",
        fallback: ForecastBackend | None = None,
        level: float = 80.0,
        timeout: float = 10.0,
    ):
        self.api_key = api_key or os.environ.get("NIXTLA_API_KEY") or os.environ.get(
            "TIMEGPT_API_KEY"
        )
        self.model = model
        self.fallback = fallback or ThetaBackend()
        self.level = float(level)
        self.timeout = float(timeout)
        self._client = None
        self._unavailable_reason: str | None = None

    # -- lazy client -----------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._unavailable_reason is not None:
            return None
        if not self.api_key:
            self._unavailable_reason = "NIXTLA_API_KEY is not set"
            return None
        try:
            from nixtla import NixtlaClient  # noqa: PLC0415 - optional dependency
        except ImportError:
            self._unavailable_reason = "the 'nixtla' package is not installed"
            return None
        try:
            self._client = NixtlaClient(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = f"could not construct NixtlaClient: {exc}"
            return None
        return self._client

    @property
    def available(self) -> bool:
        return self._get_client() is not None

    @property
    def unavailable_reason(self) -> str | None:
        self._get_client()
        return self._unavailable_reason

    # -- forecasting -----------------------------------------------------
    def forecast(self, y, horizon, *, exog=None, exog_future=None, freq=None) -> Forecast:
        client = self._get_client()
        if client is None:
            logger.debug("TimeGPT unavailable (%s); using %s",
                         self._unavailable_reason, self.fallback.name)
            fc = self.fallback.forecast(y, horizon, exog=exog, exog_future=exog_future)
            fc.metadata["timegpt_fallback_reason"] = self._unavailable_reason
            return fc

        try:
            return self._call(client, y, int(horizon), exog, exog_future, freq)
        except Exception as exc:  # noqa: BLE001 - never let a forecast kill the loop
            logger.warning("TimeGPT call failed (%s); falling back to %s",
                           exc, self.fallback.name)
            fc = self.fallback.forecast(y, horizon, exog=exog, exog_future=exog_future)
            fc.metadata["timegpt_fallback_reason"] = str(exc)
            return fc

    def _call(self, client, y, horizon, exog, exog_future, freq) -> Forecast:
        import pandas as pd  # noqa: PLC0415 - only needed on this path

        arr = np.asarray(y, dtype=float).ravel()
        freq = freq or "min"
        history = pd.DataFrame(
            {
                "unique_id": "series",
                "ds": pd.date_range(end=pd.Timestamp.utcnow().floor("s"),
                                    periods=arr.size, freq=freq),
                "y": arr,
            }
        )

        future_df = None
        if exog is not None:
            exog_arr = np.atleast_2d(np.asarray(exog, dtype=float))
            if exog_arr.shape[0] != arr.size:
                exog_arr = exog_arr.T
            for j in range(exog_arr.shape[1]):
                history[f"x{j}"] = exog_arr[:, j]
            if exog_future is not None:
                fut = np.atleast_2d(np.asarray(exog_future, dtype=float))
                if fut.shape[0] != horizon:
                    fut = fut.T
                future_df = pd.DataFrame(
                    {
                        "unique_id": "series",
                        "ds": pd.date_range(
                            start=history["ds"].iloc[-1], periods=horizon + 1, freq=freq
                        )[1:],
                    }
                )
                for j in range(fut.shape[1]):
                    future_df[f"x{j}"] = fut[:, j]

        kwargs = dict(
            df=history, h=horizon, freq=freq, time_col="ds", target_col="y",
            level=[self.level], model=self.model,
        )
        if future_df is not None:
            kwargs["X_df"] = future_df

        result = client.forecast(**kwargs)

        # Column naming has varied across SDK versions, so locate them by shape
        # rather than hard-coding "TimeGPT".
        reserved = {"unique_id", "ds"}
        lo_col = next((c for c in result.columns if "-lo-" in str(c)), None)
        hi_col = next((c for c in result.columns if "-hi-" in str(c)), None)
        mean_col = next(
            c for c in result.columns
            if c not in reserved and "-lo-" not in str(c) and "-hi-" not in str(c)
        )
        return Forecast(
            mean=result[mean_col].to_numpy(dtype=float),
            lo=result[lo_col].to_numpy(dtype=float) if lo_col else None,
            hi=result[hi_col].to_numpy(dtype=float) if hi_col else None,
            backend=self.name,
            level=self.level,
            metadata={"model": self.model, "exogenous": exog is not None},
        )


# --------------------------------------------------------------------------
# Factory / evaluation
# --------------------------------------------------------------------------
def get_backend(name: str | None = None, **kwargs) -> ForecastBackend:
    """Resolve a backend by name, defaulting to ``MTGPT_FORECAST_BACKEND``.

    Defaults to the offline Theta baseline so that a fresh clone runs with no
    credentials.
    """
    name = (name or os.environ.get("MTGPT_FORECAST_BACKEND") or "theta").lower()
    if name in ("theta", "local", "baseline"):
        return ThetaBackend(**kwargs)
    if name in ("rw", "naive", "random_walk", "driftless"):
        return DriftlessBackend()
    if name in ("timegpt", "nixtla", "foundation"):
        return TimeGPTBackend(**kwargs)
    raise ValueError(f"unknown forecast backend {name!r}")


def compare_backends(
    y: Sequence[float] | np.ndarray,
    backends: Sequence[ForecastBackend],
    *,
    horizon: int = 10,
    n_folds: int = 20,
    step: int = 5,
) -> dict[str, dict[str, float]]:
    """Rolling-origin evaluation of several backends on one series.

    Two metrics, both chosen so they cannot flatter a backend:

    ``relative_mae``
        Mean absolute error divided by the random walk's, on identical folds.
        Below 1.0 beats a random walk; that is the only comparison that matters
        for a price series. (Plain MASE is also reported, but note its
        denominator is the *one-step* naive error while the forecast is
        ``horizon`` steps ahead, so its level is not meaningful across
        horizons — only its ordering across backends is.)

    ``directional_accuracy``
        Share of folds where the sign of the predicted move matched the actual.
        This is what a trading signal actually needs; a model can win on MAE and
        still be useless. Folds where a backend predicts *no* move contribute
        nothing, so a flat forecaster reports ``nan`` rather than a spurious 0%.
    """
    arr = np.asarray(y, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    min_train = max(50, arr.size - n_folds * step - horizon)
    if min_train < 20 or arr.size < min_train + horizon:
        raise ValueError("series is too short for the requested evaluation")

    origins = [o for o in range(min_train, arr.size - horizon + 1, step)][-n_folds:]
    scale = float(np.abs(np.diff(arr[:min_train])).mean()) or 1.0

    out: dict[str, dict[str, float]] = {}
    for backend in backends:
        abs_errors, hits = [], []
        for origin in origins:
            train, actual = arr[:origin], arr[origin : origin + horizon]
            forecast = backend.forecast(train, horizon)
            abs_errors.append(float(np.abs(forecast.mean - actual).mean()))
            predicted_move = float(forecast.mean[-1] - train[-1])
            actual_move = float(actual[-1] - train[-1])
            # A backend that forecasts no move is making no directional call.
            if abs(actual_move) > 1e-12 and abs(predicted_move) > 1e-12:
                hits.append(float(np.sign(predicted_move) == np.sign(actual_move)))
        out[backend.name] = {
            "mae": float(np.mean(abs_errors)),
            "mase": float(np.mean(abs_errors) / scale),
            "directional_accuracy": float(np.mean(hits)) if hits else float("nan"),
            "n_directional_calls": len(hits),
            "n_folds": len(origins),
        }

    # Anchor everything to the random walk, running one if it was not supplied.
    baseline = out.get(DriftlessBackend.name, {}).get("mae")
    if baseline is None:
        rw = DriftlessBackend()
        errors = [
            float(np.abs(rw.forecast(arr[:o], horizon).mean - arr[o : o + horizon]).mean())
            for o in origins
        ]
        baseline = float(np.mean(errors))
    for row in out.values():
        row["relative_mae"] = row["mae"] / baseline if baseline > 0 else float("nan")
    return out
