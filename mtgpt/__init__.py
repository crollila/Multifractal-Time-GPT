"""Multifractal Time-GPT: MSM volatility regimes for event-driven trading.

Quick start::

    from mtgpt import MSMModel, RegimeClassifier

    fit = MSMModel.fit(returns, k_components=5)
    model = MSMModel(fit.params)
    classifier = RegimeClassifier.from_model(model, horizon_bars=30)
    snapshot = classifier.snapshot(model.filter_state(returns).probs)
"""

__version__ = "0.1.0"

from .models.msm import FitResult, MSMFilterState, MSMModel, MSMParams
from .models.regimes import Regime, RegimeClassifier, RegimeSnapshot
from .signals.fusion import (
    FusionConfig,
    RegimeConditionedSizer,
    calibrate,
    default_config,
    power_analysis,
)
from .signals.news import NewsEvent, read_event_tape, score_to_edge

__all__ = [
    "__version__",
    "MSMModel",
    "MSMParams",
    "MSMFilterState",
    "FitResult",
    "Regime",
    "RegimeClassifier",
    "RegimeSnapshot",
    "FusionConfig",
    "RegimeConditionedSizer",
    "calibrate",
    "default_config",
    "power_analysis",
    "NewsEvent",
    "read_event_tape",
    "score_to_edge",
]
