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

# One part is seconds, two are minutes and seconds, three are hours, minutes
# and seconds. Counting the colons rather than pattern-matching the whole
# thing, because the alternative needs two optional groups in one expression
# and then depends on which way the engine happens to be greedy.
#
# All three are wanted. `72:15` and `1:12:15` are the same moment and both
# get written: minutes that keep counting past sixty is what a player shows,
# an hour field is what a model reaches for on a long file, and there is no
# reading of either that means anything else. An hour-long locked-off
# interview is not a strange thing to be handed, and it is not scene-split
# into pieces first, because it has no scene changes.
SECONDS = re.compile(r"^\d{1,2}(?:\.\d+)?$")
MINUTES = re.compile(r"^\d{1,3}$")

# A decimal point in a string with no colon in it. `1.53` is either a second
# and a half or 1:53 with the colon turned into a point, and both readings
# are plausible -- which is the ambiguity this whole notation exists to
# remove, so it is refused rather than guessed at. It cost a 113.4s take its
# entire usable range once, reading two minutes of material as 1.5 seconds
# while passing every range check on the way.
LOOSE = re.compile(r"^\s*\d+\.\d+\s*$")

# A segment shorter than this is not a shot, it is a boundary that landed a
# little off. Kept out of the catalogue rather than repaired: a span nobody
# can cut into is worse than a gap, because it can be chosen.
LEAST_SECONDS = 0.4

# Roles that mean the camera was being got ready rather than used. A segment
# carrying one of these is not offered, however clean its picture is.
REFUSED = frozenset({"setup_reframe", "disturbance", "unknown"})


@dataclass(frozen=True)
class Span:
    """One stretch of one source that a shot may be cut from."""

    span_id: str
    source_id: str
    starts_seconds: float
    ends_seconds: float
    why: str = ""
    # What the camera was doing here, as the card read it against the local
    # measurement. Carried because the executor has to know whether the
    # source is already moving before it adds a move of its own -- the
    # selection prompt has warned about stacking them since it was written
    # and nothing downstream held the fact needed to check.
    motion_role: str = "locked"

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

    # A real number is a resolved one: this reads its own output back, and
    # by then the ambiguity has been settled. Only what a model typed is
    # held to the notation.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    written = str(value or "").strip()
    if LOOSE.match(written):
        return None
    parts = written.split(":")
    if not parts or len(parts) > 3:
        return None
    # The last part is always seconds and is the only one allowed a decimal;
    # everything before it is a whole count. Sixty or more seconds means the
    # colon is not where it looks like it is, so the reading is refused.
    if not SECONDS.match(parts[-1]) or float(parts[-1]) >= 60.0:
        return None
    if any(not MINUTES.match(one) for one in parts[:-1]):
        return None
    if len(parts) == 3 and float(parts[1]) >= 60.0:
        return None
    total = float(parts[-1])
    for place, one in enumerate(reversed(parts[:-1]), start=1):
        total += int(one) * (60 ** place)
    return total


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
        # A camera moving from one unfinished framing to a ready one, or
        # recovering from a knock, is not a take -- it is the gap between
        # two of them, and the viewer gains nothing from watching it. Those
        # get no id, which is the only way a plan cannot ask for them.
        #
        # `unknown` is refused for the same reason and a different one: the
        # model said it could not tell, and choosing anyway would turn "I do
        # not know" into "yes".
        if str(entry.get("motion_role", "")) in REFUSED:
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
                motion_role=str(entry.get("motion_role", "locked") or "locked"),
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
