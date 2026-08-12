"""Multifractal Detrended Fluctuation Analysis (MF-DFA).

Reference
---------
Kantelhardt, J. et al. (2002), "Multifractal detrended fluctuation analysis of
nonstationary time series", Physica A 316, 87-114.

Purpose here
------------
MSM assumes the series *is* multifractal. This module tests that assumption
before you commit to the model, which is the honest order of operations: if
``h(q)`` comes out flat, the asset is (near-)monofractal, MSM collapses toward a
one-factor stochastic-volatility model, and regime conditioning has little left
to bite on. Run :func:`multifractal_report` on a candidate symbol first and let
the spectrum width decide whether it belongs in the universe.

Interpretation
--------------
``h(q)`` is the generalised Hurst exponent. ``h(2)`` is the classical Hurst
exponent: 0.5 is a random walk, above 0.5 is trending, below is mean reverting.
A *decreasing* ``h(q)`` is the signature of multifractality — large fluctuations
scale differently from small ones. The singularity-spectrum width
``delta_alpha`` summarises that in one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["MFDFAResult", "mfdfa", "multifractal_report"]


@dataclass
class MFDFAResult:
    q: np.ndarray
    h: np.ndarray
    """Generalised Hurst exponent for each ``q``."""

    tau: np.ndarray
    """Renyi scaling exponent ``tau(q) = q*h(q) - 1``."""

    alpha: np.ndarray
    """Holder exponent, ``d(tau)/dq``."""

    f_alpha: np.ndarray
    """Singularity spectrum, ``q*alpha - tau``."""

    scales: np.ndarray
    fluctuations: np.ndarray
    """``(len(q), len(scales))`` array of ``F_q(s)``."""

    r_squared: np.ndarray = field(default_factory=lambda: np.array([]))
    """Goodness of fit of each log-log scaling regression."""

    @property
    def delta_alpha(self) -> float:
        """Width of the singularity spectrum: the headline multifractality number."""
        return float(np.max(self.alpha) - np.min(self.alpha))

    @property
    def hurst(self) -> float:
        """``h(2)``, interpolated if ``q = 2`` was not on the grid."""
        return float(np.interp(2.0, self.q, self.h))

    @property
    def h_range(self) -> float:
        """``h(q_min) - h(q_max)``; positive for a multifractal series."""
        return float(self.h[0] - self.h[-1])

    def to_dict(self) -> dict:
        return {
            "q": self.q.tolist(),
            "h": self.h.tolist(),
            "hurst": self.hurst,
            "delta_alpha": self.delta_alpha,
            "h_range": self.h_range,
            "min_r_squared": float(self.r_squared.min()) if self.r_squared.size else None,
        }


def _segment_fluctuations(profile: np.ndarray, scale: int, order: int) -> np.ndarray:
    """Squared detrended fluctuation of every segment at one scale.

    Segments are cut from both ends so that a tail shorter than ``scale`` is not
    silently discarded.
    """
    n = profile.size
    n_seg = n // scale
    if n_seg == 0:
        return np.empty(0)

    x = np.arange(scale, dtype=float)
    # Vandermonde is shared by every segment at this scale, so build it once.
    vander = np.vander(x, order + 1)

    out = np.empty(2 * n_seg)
    for direction in (0, 1):
        if direction == 0:
            trimmed = profile[: n_seg * scale]
        else:
            trimmed = profile[n - n_seg * scale :]
        blocks = trimmed.reshape(n_seg, scale).T  # (scale, n_seg)
        coeffs, *_ = np.linalg.lstsq(vander, blocks, rcond=None)
        residuals = blocks - vander @ coeffs
        out[direction * n_seg : (direction + 1) * n_seg] = (residuals**2).mean(axis=0)
    return out


def mfdfa(
    series: Sequence[float] | np.ndarray,
    *,
    q_values: Sequence[float] | np.ndarray = (-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5),
    scales: Sequence[int] | np.ndarray | None = None,
    order: int = 1,
    already_profile: bool = False,
) -> MFDFAResult:
    """Run MF-DFA on a return series.

    Parameters
    ----------
    series:
        Returns (not prices). The integrated profile is built internally.
    q_values:
        Moment orders. ``q = 0`` is handled by the logarithmic limit; it is
        dropped from the grid if supplied since the direct formula is singular.
    scales:
        Segment lengths. Defaults to a log-spaced grid from 10 to ``N/10``,
        which keeps at least ten segments at the largest scale so the
        fluctuation estimate stays stable.
    order:
        Polynomial detrending order within each segment. 1 removes a linear
        trend (DFA-1) and is the usual choice.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 100:
        raise ValueError(f"MF-DFA needs at least 100 points, got {n}")

    profile = x if already_profile else np.cumsum(x - x.mean())

    if scales is None:
        hi = max(int(n // 10), 20)
        scales = np.unique(
            np.round(np.logspace(np.log10(10), np.log10(hi), 20)).astype(int)
        )
    scales = np.asarray([s for s in np.asarray(scales, dtype=int) if 2 * (order + 2) <= s <= n // 4])
    if scales.size < 4:
        raise ValueError("need at least 4 usable scales; series is too short")

    q = np.asarray([qq for qq in np.asarray(q_values, dtype=float) if abs(qq) > 1e-10])
    if q.size == 0:
        raise ValueError("q_values must contain at least one non-zero moment")

    fluct = np.empty((q.size, scales.size))
    for j, scale in enumerate(scales):
        f2 = _segment_fluctuations(profile, int(scale), order)
        f2 = np.clip(f2, 1e-300, None)
        for i, qq in enumerate(q):
            # F_q(s) = mean(F2^(q/2))^(1/q); computed in logs for stability at
            # large |q| where F2^(q/2) overflows or underflows outright.
            log_terms = (qq / 2.0) * np.log(f2)
            m = log_terms.max()
            log_mean = m + np.log(np.exp(log_terms - m).mean())
            fluct[i, j] = np.exp(log_mean / qq)

    log_s = np.log(scales.astype(float))
    h = np.empty(q.size)
    r2 = np.empty(q.size)
    for i in range(q.size):
        log_f = np.log(fluct[i])
        slope, intercept = np.polyfit(log_s, log_f, 1)
        h[i] = slope
        pred = slope * log_s + intercept
        ss_res = float(((log_f - pred) ** 2).sum())
        ss_tot = float(((log_f - log_f.mean()) ** 2).sum())
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tau = q * h - 1.0
    # alpha = d(tau)/dq via central differences on the (possibly uneven) q grid.
    alpha = np.gradient(tau, q)
    f_alpha = q * alpha - tau

    return MFDFAResult(
        q=q, h=h, tau=tau, alpha=alpha, f_alpha=f_alpha,
        scales=scales, fluctuations=fluct, r_squared=r2,
    )


def multifractal_report(
    series: Sequence[float] | np.ndarray,
    *,
    width_threshold: float = 0.15,
    n_shuffles: int = 5,
    min_reliable_n: int = 20_000,
    **kwargs,
) -> dict:
    """One-call verdict on whether a series is multifractal enough for MSM.

    ``width_threshold`` is a judgement call, not a p-value. The meaningful
    comparison is against the same series **shuffled**: shuffling destroys the
    temporal correlation that produces multifractality while leaving the
    (fat-tailed) marginal distribution untouched, so it isolates genuine
    multiscale structure from mere kurtosis. Several shuffles are averaged
    because a single one is itself noisy.

    The verdict is deliberately three-way. MF-DFA needs long samples: at a few
    thousand points a genuinely multifractal series routinely fails to clear the
    shuffled benchmark, so reporting a confident "monofractal" there would be
    wrong. Short samples that lean positive are returned as ``inconclusive``.
    """
    result = mfdfa(series, **kwargs)

    clean = np.asarray(series, dtype=float).ravel()
    clean = clean[np.isfinite(clean)]
    rng = np.random.default_rng(0)
    shuffled_widths = []
    for _ in range(max(1, n_shuffles)):
        permuted = clean.copy()
        rng.shuffle(permuted)
        shuffled_widths.append(mfdfa(permuted, **kwargs).delta_alpha)
    shuffled_width = float(np.mean(shuffled_widths))

    width = result.delta_alpha
    excess = width - shuffled_width
    n = int(clean.size)

    is_multifractal = bool(width > width_threshold and width > 1.5 * shuffled_width)
    leans_positive = excess > 0.0 and width > 0.5 * width_threshold
    underpowered = n < min_reliable_n

    if is_multifractal:
        verdict = "multifractal - MSM regime conditioning has structure to exploit"
    elif leans_positive and underpowered:
        verdict = (
            f"inconclusive - the spectrum is wider than the shuffled benchmark "
            f"but {n:,} points is short for MF-DFA. Re-run with at least "
            f"{min_reliable_n:,} before concluding either way."
        )
    else:
        verdict = (
            "near-monofractal - expect limited gain from MSM over a "
            "single-factor volatility model"
        )

    return {
        "n_observations": n,
        "hurst": result.hurst,
        "delta_alpha": width,
        "shuffled_delta_alpha": shuffled_width,
        "shuffled_std": float(np.std(shuffled_widths)) if len(shuffled_widths) > 1 else 0.0,
        "excess_width": excess,
        "h_range": result.h_range,
        "is_multifractal": is_multifractal,
        "inconclusive": bool(not is_multifractal and leans_positive and underpowered),
        "min_r_squared": float(result.r_squared.min()),
        "verdict": verdict,
    }
