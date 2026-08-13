"""Generate the README figures from real pipeline runs.

Reruns the exact `mtgpt demo` configuration (seed 7, 80k bars) so every number
in the images matches docs/FINDINGS.md, then renders four figures into
docs/img/. Regenerate after any change to the engine with:

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtgpt.backtest.engine import BacktestConfig, run_comparison
from mtgpt.data.synthetic import SyntheticConfig, flat_response, generate
from mtgpt.models.regimes import Regime

matplotlib.use("Agg")

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference palette (validated; see the dataviz palette doc). Figures render on
# the light surface with a solid background so they read on both GitHub themes.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"

SERIES = {  # categorical slots 1-4, fixed order
    "legacy_threshold": "#2a78d6",
    "pooled_vol_target": "#eb6834",
    "regime_fixed_horizon": "#1baf7a",
    "regime_conditioned": "#eda100",
}
LABELS = {
    "legacy_threshold": "legacy threshold (current bot)",
    "pooled_vol_target": "vol target, no regime split",
    "regime_fixed_horizon": "regime-conditioned",
    "regime_conditioned": "regime + adaptive horizon",
}
# Ordinal ramp for the ordered severity scale (single hue, light->dark;
# lightest step >= documented 2:1 floor on the light surface).
REGIME_RAMP = {
    Regime.CALM: "#86b6ef",
    Regime.NORMAL: "#3987e5",
    Regime.TURBULENT: "#1c5cab",
    Regime.CRISIS: "#0d366b",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote docs/img/{name}")


# ---------------------------------------------------------------------------
# Figure 1: what regime detection looks like on tape
# ---------------------------------------------------------------------------
def fig_regimes() -> None:
    dataset = generate(SyntheticConfig(n_bars=12_000, seed=7))
    bars, model, classifier = dataset.bars, dataset.model, dataset.classifier
    returns = bars.log_returns
    _, filtered = model.filter(returns)
    filtered_vol = model.conditional_volatility_path(filtered)
    labels = classifier.label_path(filtered)

    # Labels churn bar-to-bar near the cutoffs; a centred 31-bar majority vote
    # makes the bands legible. Display only - the strategy uses the raw path.
    window = 31
    half = window // 2
    smoothed = labels.copy()
    for i in range(half, labels.size - half):
        smoothed[i] = np.bincount(labels[i - half : i + half + 1]).argmax()

    lo, hi = 2_000, 7_000  # a window with visible regime turnover
    x = np.arange(lo, hi)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.12},
    )

    # Regime bands behind the price.
    current, start = smoothed[lo], lo
    for i in range(lo + 1, hi + 1):
        if i == hi or smoothed[i] != current:
            for ax in (ax1, ax2):
                ax.axvspan(start, i, color=REGIME_RAMP[Regime(int(current))],
                           alpha=0.45, linewidth=0)
            if i < hi:
                current, start = smoothed[i], i
    ax1.plot(x, bars.close[lo + 1 : hi + 1], color=INK, linewidth=1.4)
    ax1.set_ylabel("price")
    ax1.set_title("MSM regime detection on tape (causal filter — no look-ahead; "
                  "bands 31-bar majority-smoothed for display)")
    ax1.grid(False)

    ann = np.sqrt(bars.annualisation_factor())
    ax2.plot(x, dataset.true_volatility[lo + 1 : hi + 1] * ann, color=INK_2,
             linewidth=1.2, linestyle="--", label="true latent vol (unobservable)")
    ax2.plot(x, filtered_vol[lo:hi] * ann, color=INK, linewidth=1.8,
             label="filtered vol (what the bot sees)")
    ax2.set_ylabel("annualised vol")
    ax2.set_xlabel("bar (1-minute)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(False)

    handles = [Patch(facecolor=c, alpha=0.45, label=r.name)
               for r, c in REGIME_RAMP.items()]
    ax1.legend(handles=handles, loc="upper left", ncol=4, fontsize=9)
    save(fig, "regime_detection.png")


# ---------------------------------------------------------------------------
# Figures 2-4 come from the exact FINDINGS.md runs
# ---------------------------------------------------------------------------
def run_scenarios():
    common = dict(n_bars=80_000, events_per_1000_bars=30.0, seed=7)
    config = BacktestConfig(msm_k_components=5, msm_n_starts=2,
                            warmup_bars=500, seed=7)
    print("running scenario A (regime-dependent)...")
    dataset_a = generate(SyntheticConfig(**common))
    result_a = run_comparison(dataset_a.bars, dataset_a.events, config, verbose=True)
    print("running scenario B (null)...")
    dataset_b = generate(flat_response(**common))
    result_b = run_comparison(dataset_b.bars, dataset_b.events, config, verbose=True)
    return dataset_a, result_a, dataset_b, result_b


def fig_equity(result_a) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    order = ["legacy_threshold", "pooled_vol_target",
             "regime_fixed_horizon", "regime_conditioned"]
    for name in order:
        curve = (result_a.results[name].equity_curve - 1_000_000.0) / 1_000.0
        x = np.arange(curve.size)
        ax.plot(x, curve, color=SERIES[name], linewidth=1.8)
        ax.annotate(
            f"  {LABELS[name]}  ({curve[-1]:+,.1f}k)",
            xy=(x[-1], curve[-1]), xytext=(4, 0), textcoords="offset points",
            color=INK, fontsize=9, va="center",
        )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlim(0, x[-1] * 1.42)  # room for the direct labels
    ax.set_xlabel("bar in out-of-sample window (1-minute)")
    ax.set_ylabel(r"P&L on \$1M, \$ thousands")
    ax.set_title("Out-of-sample equity, scenario A: same events, four sizing rules")
    save(fig, "equity_curves.png")


def fig_betas(result_a, dataset_a, result_b, dataset_b) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    panels = [
        (axes[0], result_a, dataset_a, "A: news impact depends on regime"),
        (axes[1], result_b, dataset_b, "B: null — impact identical everywhere"),
    ]
    for ax, result, dataset, title in panels:
        xs = np.arange(4)
        truth = [dataset.config.regime_response[r] for r in Regime]
        fitted = [result.calibration.rule(r).beta for r in Regime]
        ci = [1.96 * result.calibration.rule(r).standard_error for r in Regime]
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.errorbar(xs, fitted, yerr=ci, fmt="o", color="#2a78d6",
                    markersize=7, capsize=4, linewidth=1.8, zorder=3,
                    label="calibrated on training events (95% CI)")
        ax.scatter(xs, truth, marker="_", s=340, color=INK, linewidth=2.2,
                   zorder=4, label="ground truth")
        ax.set_xticks(xs, [r.name for r in Regime], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x")
    axes[0].set_ylabel("news response β (per unit edge, in horizon σ)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("The calibrator finds the regime effect — and refuses to invent one",
                 fontweight="bold", y=1.02)
    save(fig, "calibration.png")


def fig_attribution(result_a) -> None:
    legacy = result_a.results["legacy_threshold"].by_regime()
    regime = result_a.results["regime_fixed_horizon"].by_regime()

    xs = np.arange(4)
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.axhline(0, color=INK_2, linewidth=0.8)
    for offset, rows, name in (
        (-width / 2 - 0.01, legacy, "legacy_threshold"),
        (+width / 2 + 0.01, regime, "regime_fixed_horizon"),
    ):
        values = [rows[r.name].get("net_pnl", 0.0) / 1_000.0 for r in Regime]
        counts = [rows[r.name].get("n", 0) for r in Regime]
        ax.bar(xs + offset, values, width, color=SERIES[name],
               label=f"{LABELS[name]}")
        for x, v, n in zip(xs + offset, values, counts):
            text = f"{n} trades" if n else "0 trades\n(declined)"
            ax.annotate(text, xy=(x, v), xytext=(0, 6 if v >= 0 else -16),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=INK_2)
    ax.set_xticks(xs, [r.name for r in Regime])
    ax.set_ylabel("net P&L, $ thousands")
    ax.set_title("Where the money comes from — and the trades that were refused")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x")
    ax.margins(y=0.18)
    save(fig, "attribution.png")


if __name__ == "__main__":
    fig_regimes()
    dataset_a, result_a, dataset_b, result_b = run_scenarios()
    fig_equity(result_a)
    fig_betas(result_a, dataset_a, result_b, dataset_b)
    fig_attribution(result_a)
    print("done")
