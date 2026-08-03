"""Ask Gemini for the editorial decisions, and only those.

This module owns the boundary the whole rebuild is arranged around. The model
decides what a shot means and how long it should be felt; local code decides
what frame that lands on. Nothing here computes a timestamp, and nothing
downstream second-guesses a judgement.

The rhythm pass is the first slice of that. It receives shots that are already
chosen and a grid that is already measured, and answers the one question
neither of those can: how long does each of these want to be, given what is
happening in it and what the track is doing underneath.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jascue_auto.grounding import BeatGrid
from jascue_auto.schema import EDL, Clip, MusicSync

PROMPTS = Path(__file__).resolve().parent / "prompts"
MODEL_ID = os.environ.get("JASCUE_AUTO_MODEL", "gemini-3.6-flash")

# 3.6 Flash deprecated the sampling knobs, so consistency comes from the
# response schema and the instructions rather than from temperature.
THINKING_HIGH = "high"


class PlannerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    thought_tokens: int

    @classmethod
    def from_interaction(cls, interaction: Any) -> "Usage":
        usage = getattr(interaction, "usage", None) or {}
        if not isinstance(usage, dict):
            usage = getattr(usage, "__dict__", {}) or {}
        return cls(
            input_tokens=int(usage.get("total_input_tokens") or 0),
            output_tokens=int(usage.get("total_output_tokens") or 0),
            thought_tokens=int(usage.get("total_thought_tokens") or 0),
        )


def _rhythm_schema(clip_ids: list[str]) -> dict[str, Any]:
    """One decision per shot, bound to the ids that were sent.

    The clip_id is an enum of what went in, which is the cheap structural way
    to stop an answer drifting onto a shot that does not exist. Vocabularies
    stay shallow here -- this schema is one array of small objects, nowhere
    near the nesting that made the previous plan schema unservable.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "clip_id",
                        "cut_on_beat",
                        "rhythm_reason",
                        "hold_seconds",
                    ],
                    "properties": {
                        "clip_id": {"type": "string", "enum": clip_ids},
                        "cut_on_beat": {
                            "type": "boolean",
                            "description": (
                                "True to land the cut on the nearest musical "
                                "event; false when the content governs."
                            ),
                        },
                        "hold_seconds": {
                            "type": "number",
                            "description": (
                                "How long this shot wants to be on screen, "
                                "judged from what happens in it. Local code "
                                "snaps this to a musical event when "
                                "cut_on_beat is true, so it is a judgement "
                                "about the material, not a final timing."
                            ),
                        },
                        "beats": {
                            "type": "integer",
                            "description": (
                                "Only when the shot wants a musical count. "
                                "Omit when the content decides its length."
                            ),
                        },
                        "sync_to": {
                            "type": "string",
                            "description": (
                                "Named point to coincide with, when one "
                                "matters. Omit otherwise."
                            ),
                        },
                        "rhythm_reason": {
                            "type": "string",
                            "description": (
                                "Why this length, in terms of what is visible "
                                "and audible. Not a beat count."
                            ),
                        },
                    },
                },
            }
        },
    }


def _describe_music(grid: BeatGrid) -> str:
    sections = sorted(grid.named_points.items(), key=lambda item: item[1])
    lines = [
        f"BPM {grid.bpm:g}, {grid.meter}/4, one beat is "
        f"{grid.seconds_per_beat:.3f}s, track runs "
        f"{grid.duration_seconds:.1f}s.",
        f"{len(grid.cuttable())} cuttable events were measured.",
    ]
    if sections:
        lines.append("Section boundaries the analyser found:")
        lines += [
            f"  {name} at {seconds:.2f}s" for name, seconds in sections
        ]
    return "\n".join(lines)


def _describe_clips(edl: EDL) -> str:
    lines = []
    for index, clip in enumerate(edl.clips, start=1):
        approx = clip.approx_out_seconds - clip.approx_in_seconds
        described = clip.in_looks_like or "(no description supplied)"
        lines.append(
            f"{index}. clip_id={clip.clip_id} source={clip.source_id} "
            f"available≈{approx:.1f}s energy={clip.energy_intent}\n"
            f"   content: {described}"
        )
    return "\n".join(lines)


def decide_rhythm(
    edl: EDL,
    grid: BeatGrid,
    *,
    intent: str,
    client: Any | None = None,
) -> tuple[EDL, Usage]:
    """Return the EDL with each clip's rhythm decided by the model.

    The returned clips keep their in-points and carry the model's judgement in
    `music_sync` plus an out-point reflecting the hold it asked for. Grounding
    turns that into frames.
    """

    if client is None:
        client = _default_client()

    clip_ids = [clip.clip_id for clip in edl.clips]
    prompt = (PROMPTS / "rhythm_zh-TW.txt").read_text(encoding="utf-8")
    request = {
        "model": MODEL_ID,
        "store": False,
        "input": [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n\n## 這支片要傳達什麼\n\n{intent}\n\n"
                    f"## 音樂\n\n{_describe_music(grid)}\n\n"
                    f"## 畫面（依序）\n\n{_describe_clips(edl)}\n"
                ),
            }
        ],
        "generation_config": {
            "thinking_level": THINKING_HIGH,
            "max_output_tokens": 8192,
        },
        "response_format": {
            "mime_type": "application/json",
            "schema": _rhythm_schema(clip_ids),
        },
    }

    interaction = client.interactions.create(**request)
    payload = _parse(interaction)
    decisions = {
        entry["clip_id"]: entry for entry in payload.get("decisions", [])
    }

    missing = set(clip_ids) - set(decisions)
    if missing:
        raise PlannerError(
            f"the rhythm pass skipped {sorted(missing)}; every shot needs a "
            "decision because a missing one silently keeps its placeholder "
            "length"
        )

    return _apply(edl, decisions), Usage.from_interaction(interaction)


def _apply(edl: EDL, decisions: dict[str, dict[str, Any]]) -> EDL:
    rewritten: list[Clip] = []
    for clip in edl.clips:
        decision = decisions[clip.clip_id]
        hold = float(decision["hold_seconds"])
        rewritten.append(
            clip.model_copy(
                update={
                    "approx_out_seconds": clip.approx_in_seconds + max(hold, 0.1),
                    "music_sync": MusicSync(
                        cut_on_beat=bool(decision["cut_on_beat"]),
                        beats=decision.get("beats"),
                        sync_to=decision.get("sync_to"),
                        rhythm_reason=str(decision.get("rhythm_reason", "")),
                    ),
                }
            )
        )
    return edl.model_copy(update={"clips": rewritten})


def _parse(interaction: Any) -> dict[str, Any]:
    text = getattr(interaction, "output_text", None)
    if not text:
        raise PlannerError("the rhythm pass returned no text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise PlannerError(
            f"the rhythm pass returned unparseable JSON: {error}"
        ) from error


def _default_client() -> Any:
    from google import genai  # imported lazily so tests need no key
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise PlannerError("GEMINI_API_KEY is required for a live rhythm pass")
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=10 * 60 * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
