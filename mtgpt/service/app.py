"""FastAPI regime service consumed by the Node HFT bot.

Endpoints
---------
``GET  /health``               liveness plus registered symbols
``GET  /regime/{symbol}``      current regime, volatility, persistence
``POST /size``                 the hot path: score in, sized order out
``POST /bars/{symbol}``        push a closing price to advance the filter
``POST /warmup``               fit a symbol from Alpaca or synthetic data
``GET  /metrics``              observed sizing latency percentiles

Failure policy
--------------
Every endpoint returns a structured error rather than a stack trace, and the
Node client is written to **fail open** — if this service is slow or down, the
bot falls back to its own sizing rather than halting. A regime overlay that can
take the trading system down with it is a worse risk than the one it removes.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..signals.news import score_to_edge
from .state import RegimeService

__all__ = ["create_app", "app"]


class SizeRequest(BaseModel):
    symbol: str
    price: float = Field(gt=0)
    equity: float = Field(gt=0)
    score: float | None = Field(default=None, ge=0, le=100)
    edge: float | None = Field(default=None, ge=-1, le=1)
    current_qty: float = 0.0
    gross_exposure: float = 0.0
    staleness_weight: float = 1.0


class BarRequest(BaseModel):
    close: float = Field(gt=0)


class WarmupRequest(BaseModel):
    symbol: str
    source: str = "alpaca"
    """``alpaca`` pulls real bars; ``synthetic`` generates a series, which is
    what makes the service demoable with no credentials."""

    days: int = 30
    timeframe: str = "1Min"
    k_components: int = 5
    horizon_bars: int = 30


def create_app(service: RegimeService | None = None) -> FastAPI:
    service = service or RegimeService()
    app = FastAPI(
        title="Multifractal Time-GPT regime service",
        version="0.1.0",
        description="MSM volatility regimes and regime-conditioned sizing for "
                    "news-driven execution.",
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "symbols": service.symbols(),
            "forecast_backend": os.environ.get("MTGPT_FORECAST_BACKEND", "theta"),
        }

    @app.get("/metrics")
    def metrics() -> dict:
        return {"sizing_latency": service.latency_summary()}

    @app.get("/regime/{symbol}")
    def regime(symbol: str, horizon_bars: int | None = None) -> dict:
        state = service.get(symbol)
        if state is None:
            raise HTTPException(404, f"{symbol} is not registered; POST /warmup first")
        snapshot = state.snapshot(horizon_bars)
        return {
            "symbol": state.symbol,
            "last_price": state.last_price,
            "bars_seen": state.bars_seen,
            "fitted_at": state.fitted_at.isoformat(),
            "last_update": state.last_update.isoformat(),
            "msm_params": state.model.params.to_dict(),
            **snapshot.to_dict(),
        }

    @app.post("/bars/{symbol}")
    def push_bar(symbol: str, body: BarRequest) -> dict:
        try:
            state = service.push_bar(symbol, body.close)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        snapshot = state.snapshot()
        return {
            "symbol": state.symbol,
            "bars_seen": state.bars_seen,
            "regime": snapshot.regime.name,
            "sigma_bar": snapshot.sigma_bar,
        }

    @app.post("/size")
    def size(body: SizeRequest) -> dict:
        if body.score is None and body.edge is None:
            raise HTTPException(400, "supply either 'score' (0-100) or 'edge' (-1..1)")
        edge = body.edge if body.edge is not None else score_to_edge(body.score)
        try:
            decision, elapsed_us = service.size(
                body.symbol,
                edge=edge,
                price=body.price,
                equity=body.equity,
                current_qty=body.current_qty,
                gross_exposure=body.gross_exposure,
                staleness_weight=body.staleness_weight,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        payload = decision.to_dict()
        payload["latency_us"] = round(elapsed_us, 1)
        return payload

    @app.post("/warmup")
    def warmup(body: WarmupRequest) -> dict:
        try:
            bars = _load_bars(body)
        except Exception as exc:  # noqa: BLE001 - surface the cause to the caller
            raise HTTPException(400, f"could not load bars: {exc}") from exc
        try:
            state = service.warm_up(
                bars, k_components=body.k_components, horizon_bars=body.horizon_bars
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "symbol": state.symbol,
            "bars_seen": state.bars_seen,
            "msm_params": state.model.params.to_dict(),
            "cutoffs": state.classifier.describe(),
        }

    return app


def _load_bars(body: WarmupRequest):
    if body.source == "synthetic":
        from ..data.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(symbol=body.symbol.upper(), n_bars=8_000))
        return dataset.bars

    from ..data.loaders import load_alpaca_bars

    end = datetime.now(timezone.utc)
    return load_alpaca_bars(
        body.symbol.upper(),
        start=end - timedelta(days=body.days),
        end=end,
        timeframe=body.timeframe,
    )


app = create_app()
