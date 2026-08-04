"""Per-asset facts, written once and reused forever.

A card describes one clip in isolation: what is in it, where the subjects sit,
how the frame is composed, whether the take failed. Everything here is true
without knowing what the clip will be used for, which is exactly why the brief
is not an input. A card written against one brief would have to be rewritten
for the next deliverable made from the same shoot; a card written against the
material alone is good until the material changes.

Subject boxes live here for the same reason. They are expensive to compute,
they never change, and asking for them once per render instead means paying
for the same answer on every rhythm tweak, every second aspect, and every
review round -- eleven calls a run for something that was already known.

What a card deliberately cannot carry is anything comparative or sequential.
Whether this take beats the other two, whether it serves the brief, how long
it should hold: none of that is visible from inside one clip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CARD_VERSION = "jascue-auto-clip-card-v1"


@dataclass(frozen=True)
class SubjectBox:
    """One nameable thing in the frame, with where it sits."""

    label: str
    centre_x: float
    centre_y: float
    width: float
    height: float
    moves: bool

    @property
    def is_horizontal(self) -> bool:
        return self.width > self.height


def card_schema() -> dict[str, Any]:
    """Flat, because a nested one was what made the old plan unservable."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "usable", "composition", "subjects"],
        "properties": {
            "summary": {
                "type": "string",
                "description": "What happens in this clip, in a line or two.",
            },
            "usable": {
                "type": "boolean",
                "description": (
                    "False only for a take that failed outright -- an action "
                    "cut off, a screen that slept, a misfire. Handheld "
                    "movement, soft focus and unusual framing are styles "
                    "available to an edit, not defects."
                ),
            },
            "unusable_reason": {"type": "string"},
            "composition": {
                "type": "string",
                "enum": ["horizontal", "vertical", "square", "mixed"],
                "description": (
                    "How the content is laid out in the frame. A row of "
                    "handsets across a table is horizontal and will fight a "
                    "vertical crop; a standing person is vertical and will "
                    "sit in one happily. Recorded here so the edit knows "
                    "before it commits an aspect."
                ),
            },
            "camera_moves": {
                "type": "boolean",
                "description": "Whether the camera itself moves in this take.",
            },
            "subjects": {
                "type": "array",
                "description": (
                    "The things worth framing, described so each can be told "
                    "from anything similar beside it, with where it sits."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "label",
                        "centre_x",
                        "centre_y",
                        "width",
                        "height",
                        "moves",
                    ],
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": (
                                "'the left, grey handset', not 'the handset'."
                            ),
                        },
                        "centre_x": {
                            "type": "number",
                            "description": (
                                "Fraction of frame width, 0.0 left to 1.0 "
                                "right. Never pixels."
                            ),
                        },
                        "centre_y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "moves": {
                            "type": "boolean",
                            "description": (
                                "Whether this subject moves within the shot."
                            ),
                        },
                    },
                },
            },
        },
    }


def load_card(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("version") == CARD_VERSION else None


def save_card(path: Path, card: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**card, "version": CARD_VERSION}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def subjects_from_card(card: dict[str, Any]) -> list[SubjectBox]:
    boxes: list[SubjectBox] = []
    for entry in card.get("subjects", []):
        try:
            boxes.append(
                SubjectBox(
                    label=str(entry["label"]),
                    centre_x=float(entry["centre_x"]),
                    centre_y=float(entry["centre_y"]),
                    width=float(entry["width"]),
                    height=float(entry["height"]),
                    moves=bool(entry.get("moves", False)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def find_subject(card: dict[str, Any], description: str) -> SubjectBox | None:
    """Match a planner's subject description to a box the card already holds.

    Exact wording will not match, so this looks for the card's label inside
    the description or the other way round. A miss returns None and the caller
    grounds the subject the expensive way -- a card that cannot answer is not
    a reason to fail, only a reason to pay.
    """

    boxes = subjects_from_card(card)
    lowered = description.lower()
    for box in boxes:
        label = box.label.lower()
        if label and (label in lowered or lowered in label):
            return box
    # Fall back to any shared distinguishing word of reasonable length.
    for box in boxes:
        for word in box.label.split():
            if len(word) >= 2 and word in description:
                return box
    return None
