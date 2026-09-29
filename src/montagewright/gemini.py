"""One adapter for the Gemini Interactions request contract."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any, Literal, TypedDict


VIDEO_PROCESSING_POLICY_VERSION = "video-processing-v1"
MOTION_STATIC_FPS = 4.0
MOTION_FAST_STATIC_FPS = 8.0
MOTION_FAST_MAX_SOURCE_SECONDS = 20.0
MOTION_FAST_EVENT_SECONDS = 1.0
MOTION_FAST_PEAK_VW_S = 0.08
MOTION_FAST_ZOOM_RATE_S = 0.04
MOTION_FAST_ROTATION_DEG_S = 3.0


class StaticVideoProcessing(TypedDict, total=False):
    type: Literal["static"]
    fps: float
    start_offset: str
    end_offset: str


VideoProcessing = Literal["agentic", "static"] | StaticVideoProcessing


def static_video_processing(
    fps: float, *, start_offset: str | None = None,
    end_offset: str | None = None,
) -> StaticVideoProcessing:
    """A validated fixed-rate Interactions video-processing request."""

    if not 0.0 < float(fps) <= 24.0:
        raise ValueError("static video FPS must be in (0, 24]")
    processing: StaticVideoProcessing = {
        "type": "static",
        "fps": float(fps),
    }
    if start_offset is not None:
        processing["start_offset"] = str(start_offset)
    if end_offset is not None:
        processing["end_offset"] = str(end_offset)
    return processing


def motion_video_processing(
    intervals: Iterable[Any] | None, *, source_seconds: float,
) -> VideoProcessing:
    """Choose dense sampling only when local facts say motion needs it.

    The whole-library semantic pass remains agentic for a still take. A take
    with measured motion is fixed-rate so the semantic role is based on the
    same observable frames on every run. Eight FPS is reserved for short
    sources containing a sub-second or fast movement; applying it to long
    rushes would spend context on unrelated seconds.
    """

    measured = tuple(intervals or ())
    interesting = [
        one for one in measured
        if str(getattr(one, "state", "")) in {"moving", "not_a_shift"}
    ]
    if not interesting:
        return "agentic"
    fast = source_seconds <= MOTION_FAST_MAX_SOURCE_SECONDS and any(
        (
            str(getattr(one, "state", "")) == "moving"
            and (
                float(getattr(one, "seconds", 0.0)) <= MOTION_FAST_EVENT_SECONDS
                or float(getattr(one, "peak_vw_s", 0.0))
                >= MOTION_FAST_PEAK_VW_S
                or abs(float(getattr(one, "zoom_rate_s", 0.0)))
                >= MOTION_FAST_ZOOM_RATE_S
                or abs(float(getattr(one, "rotation_deg_s", 0.0)))
                >= MOTION_FAST_ROTATION_DEG_S
            )
        )
        for one in interesting
    )
    return static_video_processing(
        MOTION_FAST_STATIC_FPS if fast else MOTION_STATIC_FPS
    )


def video_content(
    uri: str,
    *,
    mime_type: str = "video/mp4",
    resolution: Literal["low", "medium", "high", "ultra_high"] = "low",
    processing: VideoProcessing = "agentic",
) -> dict[str, Any]:
    """Build one Interactions video block with an explicit viewing policy."""

    if not uri:
        raise ValueError("video content requires a URI")
    if isinstance(processing, dict):
        if processing.get("type") != "static":
            raise ValueError("structured video processing must have type=static")
        processing = static_video_processing(
            float(processing.get("fps", 1.0)),
            start_offset=processing.get("start_offset"),
            end_offset=processing.get("end_offset"),
        )
    elif processing not in {"agentic", "static"}:
        raise ValueError(f"unsupported video processing {processing!r}")
    return {
        "type": "video",
        "mime_type": mime_type,
        "uri": uri,
        "resolution": resolution,
        "processing": processing,
    }


def structured_json(schema: dict[str, Any]) -> dict[str, Any]:
    """The current Interactions API structured-text response shape."""

    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }


def _count_contents(value: Any) -> Any:
    """Translate Interactions content parts to countTokens content parts."""

    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("budget preflight only supports text or content lists")

    from google.genai import types

    resolution_levels = {
        "low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
        "medium": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
        "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
        "ultra_high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH,
    }

    parts = []
    for part in value:
        if not isinstance(part, dict):
            raise TypeError("budget preflight content parts must be dictionaries")
        kind = part.get("type")
        if kind == "text":
            parts.append(types.Part.from_text(text=str(part.get("text", ""))))
            continue
        if kind in {"video", "audio", "image", "document"}:
            uri = str(part.get("uri") or "")
            if not uri:
                raise ValueError(f"{kind} content has no URI to count")
            resolution = part.get("resolution")
            counted_resolution = None
            if resolution is not None:
                try:
                    counted_resolution = resolution_levels[str(resolution).lower()]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported media resolution {resolution!r}"
                    ) from error
            counted_part = types.Part.from_uri(
                file_uri=uri,
                mime_type=part.get("mime_type"),
                media_resolution=counted_resolution,
            )
            processing = part.get("processing") if kind == "video" else None
            if isinstance(processing, dict):
                if processing.get("type") != "static":
                    raise ValueError(
                        "structured video processing must have type=static"
                    )
                fps = float(processing.get("fps", 1.0))
                if not 0.0 < fps <= 24.0:
                    raise ValueError("static video FPS must be in (0, 24]")
                counted_part.video_metadata = types.VideoMetadata(
                    fps=fps,
                    start_offset=processing.get("start_offset"),
                    end_offset=processing.get("end_offset"),
                )
            elif processing not in {None, "agentic", "static"}:
                raise ValueError(
                    f"unsupported video processing {processing!r}"
                )
            parts.append(counted_part)
            continue
        raise TypeError(f"budget preflight cannot count content type {kind!r}")
    return types.Content(role="user", parts=parts)


def count_request_tokens(
    client: Any,
    *,
    model: str,
    input_value: Any,
    response_format: Any = None,
) -> int:
    """Count paid input before dispatch, with room for the output contract.

    countTokens measures the media and prompt exactly. The response schema is
    request metadata rather than ``contents``, so reserve a deliberately
    conservative two bytes per token for its serialized representation and a
    five-percent envelope around the provider count.
    """

    models = getattr(client, "models", None)
    counter = getattr(models, "count_tokens", None)
    if counter is None:
        # Test doubles do not carry the SDK's models surface. Production
        # clients always do; this fallback only lets pure unit tests exercise
        # reservation behaviour without making a network request.
        encoded = json.dumps(input_value, ensure_ascii=False, default=str)
        counted = max(1, math.ceil(len(encoded.encode("utf-8")) / 2))
    else:
        try:
            result = counter(model=model, contents=_count_contents(input_value))
        except Exception as error:
            raise RuntimeError(
                "Gemini token counting failed; the paid interaction was not sent"
            ) from error
        counted = int(getattr(result, "total_tokens", 0) or 0)
        if counted <= 0:
            raise RuntimeError(
                "Gemini token counting returned no total; the paid interaction "
                "was not sent"
            )

    schema_bytes = len(
        json.dumps(response_format, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ) if response_format is not None else 0
    return math.ceil(counted * 1.05) + math.ceil(schema_bytes / 2)
