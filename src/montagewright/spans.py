"""The stretches of a take that may be cut into, as things with names.

Selection used to name a source and a second: `C8330`, `9.8`. A file and a
number, and the number was always expressible -- if 9.8 landed in the middle
of somebody saying "again", the plan was still well formed, and every layer
below it agreed. The card had said where the take was worth using and that
answer reached a line of prompt text and nothing else.

Bounding the number helps and does not finish the job, because one range can
only say "not yet, now, no longer". Rushes are usually not that shape. They
are: camera still settling, a good take, someone calls it, the frame is
reset, a second good take, somebody walks in to collect the props. Two
islands with water between them, and a single window has to either swallow
the water or throw away an island.

So the card cuts the take into segments and this turns the ones that survive
into spans with ids. `C8330:s03` is an object: it exists or it does not, and
a plan cannot name one that was never written down. That is the difference
between asking a planner not to choose a failed take and not giving it the
words to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# `1:07`, `0:00`, `12:04`. Also plain seconds, because a model told six times
# to write M:SS will occasionally write 67 anyway, and the reading is not in
# doubt when there is no colon to have been lost.
CLOCK = re.compile(r"^\s*(?:(\d{1,3}):)?([0-5]?\d(?:\.\d+)?)\s*$")

# A segment shorter than this is not a shot, it is a boundary that landed a
# little off. Kept out of the catalogue rather than repaired: a span nobody
# can cut into is worse than a gap, because it can be chosen.
LEAST_SECONDS = 0.4


@dataclass(frozen=True)
class Span:
    """One stretch of one source that a shot may be cut from."""

    span_id: str
    source_id: str
    starts_seconds: float
    ends_seconds: float
    why: str = ""

    @property
    def seconds(self) -> float:
        return max(0.0, self.ends_seconds - self.starts_seconds)

    def at(self, offset: float, wanted: float) -> tuple[float, float]:
        """An in and out point this far into the span, kept inside it.

        The planner says how far in to start and how long to run; both are
        answered against the span rather than against the file, so there is
        no arithmetic it can do here that leaves the usable take.
        """

        length = min(max(0.0, wanted), self.seconds) or self.seconds
        start = min(
            max(self.starts_seconds, self.starts_seconds + max(0.0, offset)),
            max(self.starts_seconds, self.ends_seconds - length),
        )
        return start, min(self.ends_seconds, start + length)


def seconds_of(value: Any) -> float | None:
    """Read a clock reading, or a number that was meant to be one.

    Asking for seconds and being answered in MM:SS is how a 71.1s take came
    back saying it was usable to 110.0 -- which is 1:10 with the colon
    dropped -- and a 113.4s take said 1.53, which is 1:53 with the colon
    turned into a decimal point. The second one is the dangerous reading,
    because it is smaller than the duration and so passes every range check
    while claiming two minutes of take is good for a second and a half.

    Asking for the notation the model already reads makes both unwritable.
    """

    if isinstance(value, (int, float)):
        return float(value)
    found = CLOCK.match(str(value or ""))
    if not found:
        return None
    minutes, seconds = found.groups()
    return int(minutes or 0) * 60 + float(seconds)


def spans_of(card: dict[str, Any] | None, source_id: str, duration: float) -> list[Span]:
    """Every stretch of this take a shot may be cut from, in order.

    A card with no segments answers with the whole clip. That is not a
    fallback so much as the truthful reading of "nobody said": the enforcement
    that matters is that a plan can only name a span, and a span covering
    everything is exactly as much as was known before any of this.
    """

    if not card or not card.get("usable", True):
        return []
    written = card.get("segments") or []
    kept: list[Span] = []
    for index, entry in enumerate(written):
        if str(entry.get("status", "")) != "eligible":
            continue
        first = seconds_of(entry.get("from"))
        last = seconds_of(entry.get("to"))
        if first is None or last is None:
            continue
        first = max(0.0, min(first, duration))
        last = max(0.0, min(last, duration))
        if last - first < LEAST_SECONDS:
            continue
        kept.append(
            Span(
                span_id=f"{source_id}:s{index:02d}",
                source_id=source_id,
                starts_seconds=round(first, 3),
                ends_seconds=round(last, 3),
                why=str(entry.get("why", "")),
            )
        )
    if not kept and not written and duration > LEAST_SECONDS:
        kept.append(
            Span(
                span_id=f"{source_id}:s00",
                source_id=source_id,
                starts_seconds=0.0,
                ends_seconds=round(duration, 3),
            )
        )
    return kept


def catalogue(
    cards: dict[str, dict[str, Any] | None], durations: dict[str, float]
) -> dict[str, Span]:
    """Every span across every take, by id."""

    found: dict[str, Span] = {}
    for source_id, card in cards.items():
        for span in spans_of(card, source_id, durations.get(source_id, 0.0)):
            found[span.span_id] = span
    return found
