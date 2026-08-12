"""News events and the mapping from a sentiment score to a directional edge.

The upstream HFT bot emits lines of ``id, TICKER, score`` where ``score`` is a
0-100 LLM sentiment rating (0 = maximally bearish, 100 = maximally bullish).
That format has no timestamp, which makes it impossible to backtest — you cannot
join a signal to a price without knowing when it fired. :func:`parse_score_line`
accepts both the legacy 3-field format and the extended format this project
writes, and :func:`format_score_line` emits the extended one.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "NewsEvent",
    "parse_score_line",
    "format_score_line",
    "read_event_tape",
    "write_event_tape",
    "score_to_edge",
    "decay_weight",
    "deduplicate",
]

_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")


@dataclass
class NewsEvent:
    """A single scored news item."""

    timestamp: datetime
    symbol: str
    score: float
    """LLM sentiment, 0-100. 50 is neutral."""

    event_id: int | None = None
    headline: str = ""
    source: str = ""
    latency_ms: float | None = None
    """Wall-clock gap between publication and the score being available."""

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    @property
    def edge(self) -> float:
        """Directional edge in ``[-1, 1]``."""
        return score_to_edge(self.score)

    @property
    def is_tradeable_symbol(self) -> bool:
        return bool(_TICKER_RE.match(self.symbol))

    def to_row(self) -> list:
        return [
            self.event_id if self.event_id is not None else "",
            self.symbol,
            f"{self.score:g}",
            self.timestamp.astimezone(timezone.utc).isoformat(),
            self.headline.replace("\n", " ").strip(),
            self.source,
            "" if self.latency_ms is None else f"{self.latency_ms:.1f}",
        ]


def score_to_edge(score: float, *, neutral: float = 50.0, scale: float = 50.0) -> float:
    """Map a 0-100 sentiment score to a signed edge in ``[-1, 1]``.

    Deliberately linear. The upstream bot instead used hard thresholds (buy at
    >= 70, sell at <= 45, short at <= 30) which throws away the difference
    between a 71 and a 99 and makes position size a step function of an
    inherently noisy LLM output. Keeping the edge continuous lets the sizing
    layer decide how much conviction is worth, and makes the score calibratable
    against realised returns.
    """
    return float(np.clip((float(score) - neutral) / scale, -1.0, 1.0))


def decay_weight(age_seconds: float, half_life_seconds: float = 300.0) -> float:
    """Exponential staleness weight for an event that is no longer fresh.

    News alpha decays fast; a five-minute-old headline is largely priced. The
    default half-life of 300s is a starting point, not a measurement — calibrate
    it per venue with :mod:`mtgpt.backtest`.
    """
    if age_seconds <= 0:
        return 1.0
    if half_life_seconds <= 0:
        return 0.0
    return float(math.pow(0.5, age_seconds / half_life_seconds))


def parse_score_line(line: str, *, default_time: datetime | None = None) -> NewsEvent | None:
    """Parse one line of a scores file. Returns ``None`` for unusable lines.

    Accepts the legacy ``id, TICKER, score`` and the extended
    ``id, TICKER, score, iso8601, headline, source, latency_ms``.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = next(csv.reader([line]))
    parts = [p.strip() for p in parts]
    if len(parts) < 3:
        return None

    raw_id, symbol, raw_score = parts[0], parts[1], parts[2]
    # The upstream bot occasionally emits "$AAPL" or prose instead of the
    # requested "TICKER: score". A '$' means the model deviated from the output
    # contract, so the whole line is suspect - skip it rather than repair it and
    # trade on a guess. This matches the upstream bot's own behaviour.
    symbol = symbol.strip().upper()
    if not _TICKER_RE.match(symbol):
        return None
    try:
        score = float(raw_score)
    except ValueError:
        return None
    if not 0.0 <= score <= 100.0:
        return None

    try:
        event_id = int(raw_id)
    except ValueError:
        event_id = None

    timestamp = default_time
    if len(parts) >= 4 and parts[3]:
        try:
            timestamp = datetime.fromisoformat(parts[3].replace("Z", "+00:00"))
        except ValueError:
            timestamp = default_time
    if timestamp is None:
        return None

    latency = None
    if len(parts) >= 7 and parts[6]:
        try:
            latency = float(parts[6])
        except ValueError:
            latency = None

    return NewsEvent(
        timestamp=timestamp,
        symbol=symbol,
        score=score,
        event_id=event_id,
        headline=parts[4] if len(parts) >= 5 else "",
        source=parts[5] if len(parts) >= 6 else "",
        latency_ms=latency,
    )


def format_score_line(event: NewsEvent) -> str:
    import io

    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(event.to_row())
    return buf.getvalue()


def read_event_tape(
    path: str | Path, *, default_time: datetime | None = None
) -> list[NewsEvent]:
    """Read a scores file into a time-sorted event tape."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        event = parse_score_line(line, default_time=default_time)
        if event is not None:
            events.append(event)
    events.sort(key=lambda e: e.timestamp)
    return events


def write_event_tape(path: str | Path, events: Iterable[NewsEvent]) -> int:
    lines = [format_score_line(e) for e in events]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def deduplicate(
    events: Sequence[NewsEvent], *, window_seconds: float = 60.0
) -> list[NewsEvent]:
    """Collapse repeat scores for the same symbol inside a short window.

    Wire services routinely re-publish a story within seconds, and the upstream
    bot would happily trade each copy. Keeping the most extreme score in the
    window preserves the signal while removing the duplicate exposure.
    """
    ordered = sorted(events, key=lambda e: e.timestamp)
    kept: list[NewsEvent] = []
    last_by_symbol: dict[str, int] = {}
    for event in ordered:
        idx = last_by_symbol.get(event.symbol)
        if idx is not None:
            previous = kept[idx]
            age = (event.timestamp - previous.timestamp).total_seconds()
            if age <= window_seconds:
                if abs(event.edge) > abs(previous.edge):
                    kept[idx] = event
                continue
        last_by_symbol[event.symbol] = len(kept)
        kept.append(event)
    return kept


def iter_by_symbol(events: Sequence[NewsEvent]) -> Iterator[tuple[str, list[NewsEvent]]]:
    grouped: dict[str, list[NewsEvent]] = {}
    for event in events:
        grouped.setdefault(event.symbol, []).append(event)
    for symbol, items in sorted(grouped.items()):
        yield symbol, sorted(items, key=lambda e: e.timestamp)
