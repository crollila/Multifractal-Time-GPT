"""Performance statistics, with the error bars attached.

A Sharpe ratio quoted without a confidence interval is close to meaningless on a
few hundred news trades. Every headline number here therefore ships with either
a t-statistic or a bootstrap interval, and :func:`compare` reports the
*difference* between two strategies with its own interval, because that — not
either strategy's absolute Sharpe — is what answers "does regime conditioning
help?".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["PerformanceStats", "compute_stats", "bootstrap_sharpe_ci", "compare"]


@dataclass
class PerformanceStats:
    n_trades: int
    total_return: float
    mean_trade_return: float
    """Mean per-trade return on notional deployed."""

    trade_return_t_stat: float
    hit_rate: float
    profit_factor: float
    sharpe: float
    """Annualised, computed on the per-bar equity series."""

    sharpe_ci: tuple[float, float]
    sortino: float
    max_drawdown: float
    calmar: float
    annualised_return: float
    annualised_volatility: float
    avg_win: float
    avg_loss: float
    total_costs: float
    gross_pnl: float
    net_pnl: float
    exposure_fraction: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["sharpe_ci"] = list(self.sharpe_ci)
        return d

    def format_row(self, label: str) -> str:
        lo, hi = self.sharpe_ci
        return (
            f"{label:<22} {self.n_trades:>6d} {self.net_pnl:>12,.0f} "
            f"{self.sharpe:>7.2f} [{lo:>5.2f},{hi:>5.2f}] "
            f"{self.hit_rate:>7.1%} {self.max_drawdown:>8.1%} "
            f"{self.mean_trade_return:>9.4%} {self.trade_return_t_stat:>7.2f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'strategy':<22} {'trades':>6} {'net P&L':>12} "
            f"{'Sharpe':>7} {'95% CI':>13} {'hit':>7} {'maxDD':>8} "
            f"{'ret/trade':>9} {'t':>7}\n" + "-" * 100
        )


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    return float(-dd.min()) if dd.size else 0.0


def bootstrap_sharpe_ci(
    bar_returns: np.ndarray,
    periods_per_year: float,
    *,
    n_boot: int = 2000,
    block: int = 20,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Stationary block-bootstrap confidence interval for the annualised Sharpe.

    Blocks rather than i.i.d. resampling because strategy returns are
    autocorrelated whenever positions span multiple bars, and i.i.d. bootstrap
    would report an interval far too tight.
    """
    r = np.asarray(bar_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30 or r.std() == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    # Wrap-around indexing keeps every block the same length.
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = r[idx.reshape(n_boot, -1)[:, :n]]

    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(sd > 0, mu / sd * math.sqrt(periods_per_year), np.nan)
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(sharpes, alpha / 2)),
        float(np.quantile(sharpes, 1 - alpha / 2)),
    )


def compute_stats(
    *,
    equity_curve: np.ndarray,
    trade_returns: Sequence[float],
    trade_pnl: Sequence[float],
    costs: float,
    periods_per_year: float,
    initial_equity: float,
    exposure_fraction: float = 0.0,
    seed: int = 0,
) -> PerformanceStats:
    equity = np.asarray(equity_curve, dtype=float)
    tr = np.asarray(list(trade_returns), dtype=float)
    pnl = np.asarray(list(trade_pnl), dtype=float)

    if equity.size > 1:
        bar_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    else:
        bar_returns = np.array([])

    mean_bar = float(bar_returns.mean()) if bar_returns.size else 0.0
    sd_bar = float(bar_returns.std(ddof=1)) if bar_returns.size > 1 else 0.0
    sharpe = mean_bar / sd_bar * math.sqrt(periods_per_year) if sd_bar > 0 else 0.0

    downside = bar_returns[bar_returns < 0]
    dd_sd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = mean_bar / dd_sd * math.sqrt(periods_per_year) if dd_sd > 0 else 0.0

    n_years = max(bar_returns.size / periods_per_year, 1e-9)
    total_return = float(equity[-1] / equity[0] - 1.0) if equity.size > 1 else 0.0
    ann_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if total_return > -1 else -1.0
    ann_vol = sd_bar * math.sqrt(periods_per_year)
    max_dd = _max_drawdown(equity)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    if tr.size > 1 and tr.std(ddof=1) > 0:
        t_stat = float(tr.mean() / (tr.std(ddof=1) / math.sqrt(tr.size)))
    else:
        t_stat = float("nan")

    return PerformanceStats(
        n_trades=int(tr.size),
        total_return=total_return,
        mean_trade_return=float(tr.mean()) if tr.size else 0.0,
        trade_return_t_stat=t_stat,
        hit_rate=float((pnl > 0).mean()) if pnl.size else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        sharpe=sharpe,
        sharpe_ci=bootstrap_sharpe_ci(bar_returns, periods_per_year, seed=seed),
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=(ann_return / max_dd) if max_dd > 1e-9 else float("nan"),
        annualised_return=ann_return,
        annualised_volatility=ann_vol,
        avg_win=float(wins.mean()) if wins.size else 0.0,
        avg_loss=float(losses.mean()) if losses.size else 0.0,
        total_costs=float(costs),
        gross_pnl=float(pnl.sum() + costs),
        net_pnl=float(pnl.sum()),
        exposure_fraction=float(exposure_fraction),
    )


def compare(
    baseline_trade_returns: Sequence[float],
    candidate_trade_returns: Sequence[float],
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Bootstrap the *difference* in mean per-trade return.

    The two strategies trade overlapping event sets, so this is an unpaired
    comparison of means with a bootstrap interval rather than a t-test that
    would assume independence and normality it does not have. If the interval
    straddles zero, regime conditioning did not demonstrably help — and saying
    so is the point of the function.
    """
    a = np.asarray(list(baseline_trade_returns), dtype=float)
    b = np.asarray(list(candidate_trade_returns), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 10 or b.size < 10:
        return {"difference": float("nan"), "ci": [float("nan")] * 2, "significant": False}

    rng = np.random.default_rng(seed)
    diffs = (
        b[rng.integers(0, b.size, (n_boot, b.size))].mean(axis=1)
        - a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    )
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    return {
        "baseline_mean": float(a.mean()),
        "candidate_mean": float(b.mean()),
        "difference": float(b.mean() - a.mean()),
        "ci": [lo, hi],
        "significant": bool(lo > 0 or hi < 0),
        "p_one_sided": float((diffs <= 0).mean()),
    }
