from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

from .clip_card_observations import (
    AssessmentStatus,
    EventObservationSupplement,
    ObservationBasis,
    event_fingerprint,
)
from .media import has_audio_stream, sha256_file
from .models import FullClipCard, FullClipEvent


CAPABILITY_NAMES = (
    "evidence_origin",
    "action_structure",
    "evidence_roles",
    "observable_beats",
    "readability",
    "audio_role",
)


def mmss_to_ms(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    if minutes < 0 or not 0 <= seconds <= 59:
        raise ValueError(f"invalid MM:SS value {value!r}")
    return (minutes * 60 + seconds) * 1000


def bounded_event_window_ms(
    card: FullClipCard,
    event: FullClipEvent,
    *,
    context_ms: int,
) -> tuple[int, int]:
    if context_ms < 0:
        raise ValueError("context_ms must be non-negative")
    start_ms = max(0, mmss_to_ms(event.start_mmss) - context_ms)
    event_end_ms = event.resolved_end_ms(card.duration_ms)
    end_ms = min(card.duration_ms, event_end_ms + context_ms)
    if end_ms <= start_ms:
        raise ValueError(f"event {event.event_id} has an empty bounded window")
    return start_ms, end_ms


def render_bounded_event_proxy(
    source_video: Path,
    output_video: Path,
    *,
    start_ms: int,
    end_ms: int,
    max_width: int = 1280,
) -> bool:
    source = source_video.expanduser().resolve(strict=True)
    if end_ms <= start_ms:
        raise ValueError("bounded proxy end must follow start")
    if max_width < 384:
        raise ValueError("max_width must be at least 384")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    audio_included = has_audio_stream(source)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-i",
        str(source),
        "-t",
        f"{(end_ms - start_ms) / 1000:.3f}",
        "-map",
        "0:v:0",
    ]
    if audio_included:
        command.extend(["-map", "0:a:0"])
    command.extend(
        [
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
        ]
    )
    if audio_included:
        command.extend(["-c:a", "aac", "-b:a", "96k"])
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", "-y", str(output_video)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return audio_included


def validate_requested_observation(
    observation: EventObservationSupplement,
    *,
    event: FullClipEvent,
    requested_capabilities: Iterable[str],
    audio_included: bool,
    expected_observation_basis: ObservationBasis | None = (
        ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO
    ),
) -> None:
    requested = set(requested_capabilities)
    unknown = requested - set(CAPABILITY_NAMES)
    if unknown:
        raise ValueError(f"unknown requested capabilities: {sorted(unknown)}")
    if observation.event_id != event.event_id:
        raise ValueError("observation changed immutable event_id")
    if observation.event_fingerprint != event_fingerprint(event):
        raise ValueError("observation changed immutable event fingerprint")
    if observation.audio_included != audio_included:
        raise ValueError("observation changed immutable audio inclusion flag")
    for capability in CAPABILITY_NAMES:
        status = getattr(observation.capabilities, capability)
        if capability not in requested and status != AssessmentStatus.NOT_ASSESSED:
            raise ValueError(
                f"unrequested capability {capability} must remain not_assessed"
            )
    assessed = {
        capability
        for capability in CAPABILITY_NAMES
        if getattr(observation.capabilities, capability)
        != AssessmentStatus.NOT_ASSESSED
    }
    if expected_observation_basis is None and (
        assessed or observation.observation_basis is not None
    ):
        raise ValueError(
            "capabilities cannot be assessed without bounded observation media"
        )
    if (
        expected_observation_basis is not None
        and observation.observation_basis != expected_observation_basis
    ):
        raise ValueError(
            "bounded observation changed immutable observation_basis"
        )
    origin_status = observation.capabilities.evidence_origin
    if (
        origin_status == AssessmentStatus.ASSESSED_PRESENT
        and observation.evidence_origin is not None
        and observation.evidence_origin.relation == "unknown"
    ):
        raise ValueError(
            "unknown evidence origin must remain not_assessed, not assessed_present"
        )
    if (
        origin_status
        in {
            AssessmentStatus.ASSESSED_ABSENT,
            AssessmentStatus.NOT_APPLICABLE,
        }
        and observation.evidence_provenance != "unknown"
    ):
        raise ValueError(
            "absent or inapplicable evidence origin conflicts with legacy provenance"
        )


def supplement_cache_key(
    *,
    source_video: Path,
    card: FullClipCard,
    event: FullClipEvent,
    requested_capabilities: Iterable[str],
    context_ms: int,
    model_id: str,
    prompt_sha256: str,
    response_schema_sha256: str,
) -> dict[str, object]:
    return {
        "contract_version": "clip-observation-supplement-cache-v2",
        "source_video_sha256": sha256_file(source_video),
        "source_asset_id": card.source_asset_id,
        "proxy_asset_id": card.proxy_asset_id,
        "base_card_sha256": hashlib.sha256(
            json.dumps(
                card.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "event_id": event.event_id,
        "event_fingerprint": event_fingerprint(event),
        "requested_capabilities": sorted(set(requested_capabilities)),
        "context_ms": context_ms,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "response_schema_sha256": response_schema_sha256,
    }
