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
class Beat:
    """One movement in a clip, and when it runs."""

    what: str
    starts_seconds: float
    ends_seconds: float


def action_beats(card: dict[str, Any]) -> list[Beat]:
    beats: list[Beat] = []
    for entry in card.get("action", []) or []:
        try:
            start = float(entry["starts_seconds"])
            end = float(entry["ends_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            beats.append(Beat(str(entry.get("what", "")), start, end))
    return sorted(beats, key=lambda beat: beat.starts_seconds)


def snap_to_action(
    card: dict[str, Any], wanted_start: float, duration: float
) -> tuple[float, str | None]:
    """Move a planned in-point onto the nearest action that contains it.

    A cut placed by arithmetic lands wherever the seconds fall, which is
    usually the middle of a gesture. An editor entering a shot goes in as the
    movement starts. This only moves the in-point when there is an action
    close enough to be the one meant -- half the shot's length -- so a static
    shot keeps the timing it was given.
    """

    beats = action_beats(card)
    if not beats:
        return wanted_start, None

    tolerance = max(0.5, duration / 2.0)
    nearest = min(beats, key=lambda beat: abs(beat.starts_seconds - wanted_start))
    drift = nearest.starts_seconds - wanted_start
    if abs(drift) > tolerance:
        return wanted_start, None
    return nearest.starts_seconds, (
        f"moved {drift:+.2f}s onto '{nearest.what}'" if abs(drift) > 0.05 else None
    )


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
        "required": [
            "summary",
            "usable",
            "composition",
            "subjects",
            "action",
        ],
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
            "action": {
                "type": "array",
                "description": (
                    "Where things actually happen in this clip. An editor "
                    "cuts on action -- into a gesture as it begins, out as it "
                    "completes -- and a cut placed by arithmetic lands "
                    "mid-movement, which reads as a mistake even to someone "
                    "who cannot say why. Empty for a static shot."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["what", "starts_seconds", "ends_seconds"],
                    "properties": {
                        "what": {
                            "type": "string",
                            "description": (
                                "The movement, in a few words: 'the hand "
                                "reaches in', 'the phone opens', 'the watch "
                                "is lowered'."
                            ),
                        },
                        "starts_seconds": {"type": "number"},
                        "ends_seconds": {"type": "number"},
                    },
                },
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


def describe_clip(
    proxy: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Watch one clip and write its card.

    Called once per asset, ever. The result is content-addressed, so a card
    survives every rerun, every second aspect and every review round -- which
    is the whole reason the subject boxes belong here rather than being
    grounded again on each render.
    """

    from jascue_auto.planner import MODEL_ID, Usage, _parse, _to_frame_fractions

    prompts = Path(__file__).resolve().parent / "prompts"
    instruction = (prompts / "clipcard_zh-TW.txt").read_text(encoding="utf-8")

    if cache is None:
        uploaded = client.files.upload(file=str(proxy))
        uri = uploaded.uri
    else:
        uri, _ = cache.uri_for(proxy, client, mime_type="video/mp4")

    interaction = client.interactions.create(
        model=model_id or MODEL_ID,
        store=False,
        input=[
            {"type": "text", "text": instruction},
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "media_resolution": "low",
            },
        ],
        generation_config={"thinking_level": "low", "max_output_tokens": 8192},
        response_format={
            "mime_type": "application/json",
            "schema": card_schema(),
        },
    )
    card = _parse(interaction, what="clip card")
    # The model answers boxes in its native 0..1000 space for some clips
    # whatever the field says, so the conversion happens on receipt.
    card["subjects"] = _to_frame_fractions(card.get("subjects", []))
    return card, Usage.from_interaction(interaction)


def build_library(
    proxies: dict[str, Path],
    card_dir: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write a card for every asset that does not already have one.

    Resumable by construction: an existing card of the current version is left
    alone. A run interrupted halfway costs nothing to restart, which matters
    when the alternative is seventy-four paid calls.
    """

    card_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    stats = {"written": 0, "reused": 0, "failed": 0, "input": 0, "output": 0}

    for source_id, proxy in sorted(proxies.items()):
        destination = card_dir / f"{source_id}.json"
        if load_card(destination) is not None:
            paths[source_id] = destination
            stats["reused"] += 1
            continue
        try:
            card, usage = describe_clip(
                proxy, client=client, cache=cache, model_id=model_id
            )
        except Exception:
            # One unreadable clip is not a reason to abandon the library.
            stats["failed"] += 1
            continue
        save_card(destination, card)
        paths[source_id] = destination
        stats["written"] += 1
        stats["input"] += usage.input_tokens
        stats["output"] += usage.output_tokens + usage.thought_tokens
    return paths, stats
