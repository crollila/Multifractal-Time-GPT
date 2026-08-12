"""Command line interface: ``python -m mtgpt.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from .backtest.engine import BacktestConfig, run_comparison
from .data.loaders import load_csv
from .models.mfdfa import multifractal_report
from .models.msm import MSMModel, select_k_components
from .signals.fusion import power_analysis, summarise_calibration


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_bars(args):
    if args.csv:
        return load_csv(args.csv, symbol=args.symbol)
    if args.synthetic:
        from .data.synthetic import SyntheticConfig, generate

        return generate(SyntheticConfig(symbol=args.symbol or "SYNTH", n_bars=args.bars)).bars

    from .data.loaders import load_alpaca_bars

    if not args.symbol:
        raise SystemExit("--symbol is required when loading from Alpaca")
    end = datetime.now(timezone.utc)
    return load_alpaca_bars(
        args.symbol, start=end - timedelta(days=args.days), end=end,
        timeframe=args.timeframe,
    )


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_demo(args) -> int:
    """Run the full pipeline on both synthetic scenarios."""
    from .data.synthetic import flat_response, generate, regime_dependent

    scenarios = {
        "A: regime-dependent news impact": regime_dependent(
            n_bars=args.bars, events_per_1000_bars=args.event_rate, seed=args.seed
        ),
        "B: NULL - identical impact in every regime": flat_response(
            n_bars=args.bars, events_per_1000_bars=args.event_rate, seed=args.seed
        ),
    }

    summary = {}
    for title, config in scenarios.items():
        _rule(title)
        dataset = generate(config)
        print("ground truth:", json.dumps(dataset.summary()["ground_truth_response"]))
        print(f"{len(dataset.bars):,} bars, {len(dataset.events):,} news events")

        result = run_comparison(
            dataset.bars,
            dataset.events,
            BacktestConfig(
                msm_k_components=args.k, msm_n_starts=args.starts,
                warmup_bars=500, seed=args.seed,
            ),
        )

        _rule("calibrated per-regime response (training data only)")
        print(summarise_calibration(result.calibration))

        _rule("out-of-sample results")
        print(result.format_table())

        print("\nby-regime attribution, legacy bot:")
        print(result.results["legacy_threshold"].format_by_regime())
        print("\nby-regime attribution, regime-conditioned:")
        print(result.results["regime_fixed_horizon"].format_by_regime())

        print("\nvs legacy (bootstrap on per-trade return):")
        for name, comp in result.comparisons.items():
            flag = "SIGNIFICANT" if comp["significant"] else "not significant"
            print(
                f"  {name:<22} {comp['difference']:+.5f} "
                f"[{comp['ci'][0]:+.5f}, {comp['ci'][1]:+.5f}]  {flag}"
            )

        power = power_analysis(result.calibration, target_difference=args.target_diff)
        print(f"\nevents needed per regime to resolve a beta gap of {args.target_diff}:")
        for regime, row in power["per_regime"].items():
            if row.get("required") is None:
                print(f"  {regime:<10} insufficient data")
            else:
                print(
                    f"  {regime:<10} have {row['n_observed']:>4d}, "
                    f"need {row['required']:>5d}  (short by {row['shortfall']:>5d})"
                )

        summary[title] = {
            name: {
                "net_pnl": res.stats.net_pnl,
                "sharpe": res.stats.sharpe,
                "n_trades": res.stats.n_trades,
                "cap_binding_rate": res.cap_binding_rate,
            }
            for name, res in result.results.items()
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


def cmd_fit(args) -> int:
    bars = _load_bars(args)
    returns = bars.log_returns
    print(f"{bars.symbol}: {len(bars):,} bars, {returns.size:,} returns")

    if args.select_k:
        best, all_fits = select_k_components(returns, n_starts=args.starts)
        print(f"\n{'K':>3} {'logLik':>14} {'BIC':>14}")
        for k in sorted(all_fits):
            mark = " <-- best" if all_fits[k] is best else ""
            print(f"{k:>3} {all_fits[k].log_likelihood:>14,.1f} {all_fits[k].bic:>14,.1f}{mark}")
        fit = best
    else:
        fit = MSMModel.fit(returns, k_components=args.k, n_starts=args.starts)

    p = fit.params
    print(f"\nm0       = {p.m0:.4f}   (1 = no multifractality, 2 = extreme)")
    print(f"sigma    = {p.sigma:.6f} per bar")
    print(f"gamma_1  = {p.gamma_1:.6f}  (slowest component switches every "
          f"{1 / p.gamma_1:,.0f} bars)")
    print(f"b        = {p.b:.3f}")
    print(f"K        = {p.k_components}  ({2**p.k_components} states)")
    print(f"logLik   = {fit.log_likelihood:,.1f}   BIC = {fit.bic:,.1f}")
    print(f"converged = {fit.converged}")

    print("\ncomponent time scales (bars between switches):")
    for k, dur in enumerate(p.expected_durations, start=1):
        print(f"  component {k}: {dur:>12,.1f}")
    return 0


def cmd_diagnose(args) -> int:
    bars = _load_bars(args)
    report = multifractal_report(bars.log_returns)
    print(f"{bars.symbol}: {len(bars):,} bars\n")
    for key in ("hurst", "delta_alpha", "shuffled_delta_alpha", "shuffled_std",
                "excess_width", "h_range", "min_r_squared"):
        print(f"  {key:<24} {report[key]:.4f}")
    print(f"  {'is_multifractal':<24} {report['is_multifractal']}")
    print(f"  {'inconclusive':<24} {report['inconclusive']}")
    print(f"\n{report['verdict']}")
    return 0


def cmd_backtest(args) -> int:
    from .signals.news import read_event_tape

    bars = _load_bars(args)
    events = read_event_tape(args.events)
    print(f"{bars.symbol}: {len(bars):,} bars, {len(events):,} events")
    if not events:
        raise SystemExit(
            "no usable events. Legacy 'id, TICKER, score' lines have no timestamp "
            "and cannot be backtested - see integration/hft-node-bot/README.md step 6."
        )

    result = run_comparison(
        bars, events,
        BacktestConfig(msm_k_components=args.k, msm_n_starts=args.starts, seed=args.seed),
    )
    _rule("calibration")
    print(summarise_calibration(result.calibration))
    _rule("out-of-sample results")
    print(result.format_table())
    for name, res in result.results.items():
        print(f"\n--- {name} (cap-binding on {res.cap_binding_rate:.0%} of trades) ---")
        print(res.format_by_regime())
    return 0


def cmd_ingest(args) -> int:
    from .signals.news import read_event_tape, write_event_tape

    default_time = datetime.now(timezone.utc) if args.assume_now else None
    events = read_event_tape(args.source, default_time=default_time)
    if not events:
        print(
            "No events parsed. Legacy lines lack a timestamp; re-run with "
            "--assume-now to stamp them all with the current time (useful only "
            "for smoke-testing, never for backtesting)."
        )
        return 1
    n = write_event_tape(args.output, events)
    print(f"wrote {n:,} events to {args.output}")
    print(f"range: {events[0].timestamp.isoformat()} .. {events[-1].timestamp.isoformat()}")
    symbols = sorted({e.symbol for e in events})
    print(f"{len(symbols)} symbols: {', '.join(symbols[:20])}"
          + (" ..." if len(symbols) > 20 else ""))
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('uvicorn is not installed; pip install -e ".[service]"')
    uvicorn.run("mtgpt.service.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def cmd_forecast_bench(args) -> int:
    """Horse-race the foundation model against the offline baselines."""
    from .models.foundation import (
        DriftlessBackend, ThetaBackend, TimeGPTBackend, compare_backends,
    )

    bars = _load_bars(args)
    backends = [DriftlessBackend(), ThetaBackend()]
    timegpt = TimeGPTBackend()
    if timegpt.available:
        backends.append(timegpt)
    else:
        print(f"TimeGPT unavailable ({timegpt.unavailable_reason}); "
              "benchmarking offline baselines only.\n")

    results = compare_backends(
        bars.close, backends, horizon=args.horizon, n_folds=args.folds
    )
    print(f"{'backend':<16} {'rel.MAE':>9} {'dir.acc':>9} {'calls':>7} {'folds':>7}")
    print("-" * 54)
    for name, row in results.items():
        accuracy = row["directional_accuracy"]
        accuracy_text = "      n/a" if accuracy != accuracy else f"{accuracy:>9.1%}"
        print(f"{name:<16} {row['relative_mae']:>9.4f} {accuracy_text} "
              f"{row['n_directional_calls']:>7d} {row['n_folds']:>7d}")
    print("\nrel.MAE is versus the random walk on identical folds: below 1.00 beats")
    print("it. Directional accuracy near 50% means no usable sign information,")
    print("whatever the MAE says - and a flat forecaster makes no directional call")
    print("at all, reported as n/a rather than a spurious 0%.")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtgpt",
        description="Multifractal regime detection for event-driven trading.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p, *, synthetic_default=False):
        p.add_argument("--symbol")
        p.add_argument("--csv", help="load bars from a CSV instead of Alpaca")
        p.add_argument("--synthetic", action="store_true", default=synthetic_default)
        p.add_argument("--bars", type=int, default=20_000, help="synthetic bar count")
        p.add_argument("--days", type=int, default=30, help="Alpaca lookback")
        p.add_argument("--timeframe", default="1Min")

    p = sub.add_parser("demo", help="full pipeline on both synthetic scenarios")
    p.add_argument("--bars", type=int, default=60_000)
    p.add_argument("--event-rate", type=float, default=30.0,
                   help="news events per 1000 bars")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--target-diff", type=float, default=0.3)
    p.add_argument("--json", help="write a summary JSON here")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("fit", help="estimate MSM parameters")
    add_data_args(p)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--starts", type=int, default=3)
    p.add_argument("--select-k", action="store_true", help="sweep K and pick by BIC")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("diagnose", help="MF-DFA multifractality test")
    add_data_args(p)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("backtest", help="walk-forward ablation on a real event tape")
    add_data_args(p)
    p.add_argument("--events", required=True, help="path to a scores/event file")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--starts", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("ingest", help="convert a legacy scores.txt into an event tape")
    p.add_argument("source")
    p.add_argument("-o", "--output", default="events.csv")
    p.add_argument("--assume-now", action="store_true",
                   help="stamp undated legacy lines with the current time")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("serve", help="run the regime service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("forecast-bench", help="TimeGPT vs offline baselines")
    add_data_args(p, synthetic_default=False)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--folds", type=int, default=30)
    p.set_defaults(func=cmd_forecast_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.seterr(all="ignore")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
