"""In-memory per-symbol regime state.

Latency budget
--------------
The Node bot's whole point is news-to-order in tens of milliseconds, so the
regime lookup has to be effectively free. Everything expensive is done off the
hot path:

* **MSM estimation** (seconds) happens at warm-up and on a slow refit timer.
* **Regime cutoffs** (a 20k-step simulation) are computed once per refit.
* **Filtering** is incremental — one bar costs a single ``2**K``-dimensional
  matrix-vector product, microseconds for ``K <= 8``.

What is left on the request path is a handful of dot products, which is why
``/size`` answers in well under a millisecond. The Node client's timeout exists
for network hiccups, not for compute.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..data.loaders import BarSeries
from ..models.msm import MSMModel
from ..models.regimes import Regime, RegimeClassifier, RegimeSnapshot
from ..signals.fusion import FusionConfig, RegimeConditionedSizer, TradeDecision, default_config

__all__ = ["SymbolState", "RegimeService"]


@dataclass
class SymbolState:
    symbol: str
    model: MSMModel
    classifier: RegimeClassifier
    filter_state: object
    last_price: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bars_seen: int = 0
    fitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bar_seconds: float = 60.0

    def snapshot(self, horizon_bars: int | None = None) -> RegimeSnapshot:
        return self.classifier.snapshot(
            self.filter_state.probs,
            horizon_bars=horizon_bars,
            n_observations=self.bars_seen,
        )


class RegimeService:
    """Thread-safe registry of per-symbol MSM state."""

    def __init__(self, config: FusionConfig | None = None, *, k_components: int = 5):
        self._symbols: dict[str, SymbolState] = {}
        self._lock = threading.RLock()
        self.config = config or default_config()
        self.sizer = RegimeConditionedSizer(self.config)
        self.k_components = k_components
        self.latencies_us: list[float] = []

    # -- registration ----------------------------------------------------
    def warm_up(
        self,
        bars: BarSeries,
        *,
        k_components: int | None = None,
        n_starts: int = 2,
        n_sim: int = 20_000,
        horizon_bars: int = 30,
    ) -> SymbolState:
        """Fit MSM on ``bars`` and register the symbol. Slow; call off the hot path."""
        returns = bars.log_returns
        if returns.size < 200:
            raise ValueError(
                f"{bars.symbol}: need at least 200 bars to fit MSM, got {returns.size}"
            )
        fit = MSMModel.fit(
            returns,
            k_components=k_components or self.k_components,
            n_starts=n_starts,
        )
        model = MSMModel(fit.params)
        classifier = RegimeClassifier.from_model(
            model, horizon_bars=horizon_bars, n_sim=n_sim
        )
        state = SymbolState(
            symbol=bars.symbol,
            model=model,
            classifier=classifier,
            filter_state=model.filter_state(returns),
            last_price=float(bars.close[-1]),
            bars_seen=len(bars),
            bar_seconds=bars.bar_seconds,
        )
        with self._lock:
            self._symbols[bars.symbol.upper()] = state
        return state

    def get(self, symbol: str) -> SymbolState | None:
        with self._lock:
            return self._symbols.get(symbol.upper())

    def symbols(self) -> list[str]:
        with self._lock:
            return sorted(self._symbols)

    # -- streaming updates -----------------------------------------------
    def push_bar(self, symbol: str, close: float) -> SymbolState:
        """Absorb one new closing price. One matvec; safe to call per bar."""
        state = self.get(symbol)
        if state is None:
            raise KeyError(f"{symbol} is not registered; call warm_up first")
        if close <= 0:
            raise ValueError("close must be positive")
        with self._lock:
            if state.last_price > 0:
                state.filter_state.step(float(np.log(close / state.last_price)))
                state.bars_seen += 1
            state.last_price = float(close)
            state.last_update = datetime.now(timezone.utc)
        return state

    # -- the hot path ----------------------------------------------------
    def size(
        self,
        symbol: str,
        *,
        edge: float,
        price: float,
        equity: float,
        current_qty: float = 0.0,
        gross_exposure: float = 0.0,
        staleness_weight: float = 1.0,
    ) -> tuple[TradeDecision, float]:
        """Return a sized decision and the microseconds it took."""
        state = self.get(symbol)
        if state is None:
            raise KeyError(f"{symbol} is not registered")
        started = time.perf_counter()
        decision = self.sizer.decide(
            symbol=symbol,
            edge=edge,
            classifier=state.classifier,
            probs=state.filter_state.probs,
            price=price,
            equity=equity,
            current_qty=current_qty,
            gross_exposure=gross_exposure,
            staleness_weight=staleness_weight,
        )
        elapsed_us = (time.perf_counter() - started) * 1e6
        self.latencies_us.append(elapsed_us)
        if len(self.latencies_us) > 10_000:
            del self.latencies_us[:5_000]
        return decision, elapsed_us

    def latency_summary(self) -> dict:
        if not self.latencies_us:
            return {"n": 0}
        arr = np.array(self.latencies_us)
        return {
            "n": int(arr.size),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(arr.max()),
        }
