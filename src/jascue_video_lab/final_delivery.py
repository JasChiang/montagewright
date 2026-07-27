from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Mapping

from .media import sha256_file
from .models import MusicAssemblyRenderManifest, MusicEditPlanV2
from .models import MusicEditRenderManifestV2
from .music_assembly import (
    MUSIC_ASSEMBLY_BINDING_FILENAME,
    MUSIC_ASSEMBLY_PLAN_FILENAME,
    MUSIC_ASSEMBLY_RENDER_FILENAME,
    MUSIC_EDIT_PLAN_V2_FILENAME,
    MUSIC_EDIT_RENDER_V2_FILENAME,
    MusicAssemblyError,
    load_music_assembly_artifacts,
    music_assembly_plan_sha256,
    music_edit_plan_v2_sha256,
)
from .storage import read_json, utc_now, write_json


class FinalDeliveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalDeliveryResult:
    output_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,"
            "sample_rate,channels,start_time,duration,duration_ts,time_base",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("format"), dict):
        raise FinalDeliveryError(f"ffprobe returned an invalid payload for {path}")
    return payload


def _duration_ms(payload: Mapping[str, Any]) -> int:
    try:
        return round(float(payload["format"]["duration"]) * 1000)
    except (KeyError, TypeError, ValueError) as error:
        raise FinalDeliveryError("ffprobe omitted a valid media duration") from error


def _single_stream(
    payload: Mapping[str, Any],
    codec_type: Literal["video", "audio"],
    *,
    source_label: str,
) -> dict[str, Any]:
    streams = [
        stream
        for stream in payload.get("streams") or []
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type
    ]
    if len(streams) != 1:
        raise FinalDeliveryError(
            f"{source_label} must contain exactly one {codec_type} stream; "
            f"found {len(streams)}"
        )
    return streams[0]


def _stream_duration_seconds(stream: Mapping[str, Any]) -> Fraction:
    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts is not None and time_base is not None:
        try:
            duration = int(str(duration_ts)) * Fraction(str(time_base))
            if duration > 0:
                return duration
        except (ValueError, ZeroDivisionError):
            pass
    try:
        duration = Fraction(str(stream["duration"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise FinalDeliveryError("ffprobe omitted a valid stream duration") from error
    if duration <= 0:
        raise FinalDeliveryError("ffprobe returned a non-positive stream duration")
    return duration


def _stream_duration_ms(stream: Mapping[str, Any]) -> int:
    return round(_stream_duration_seconds(stream) * 1000)


def _stream_start_ms(stream: Mapping[str, Any]) -> int:
    try:
        return round(Fraction(str(stream.get("start_time", "0"))) * 1000)
    except (ValueError, ZeroDivisionError) as error:
        raise FinalDeliveryError("ffprobe returned an invalid stream start") from error


def _stream_duration_samples(stream: Mapping[str, Any]) -> int:
    try:
        sample_rate = int(str(stream["sample_rate"]))
    except (KeyError, TypeError, ValueError) as error:
        raise FinalDeliveryError("ffprobe omitted a valid audio sample rate") from error
    return round(_stream_duration_seconds(stream) * sample_rate)


def _validate_music_assembly_evidence(
    *,
    music_path: Path,
    music_probe: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Validate a reviewed v1 or v2 plan → render → PCM evidence chain."""

    resolved_dir = artifact_dir.expanduser().resolve(strict=True)
    v2_plan_path = resolved_dir / MUSIC_EDIT_PLAN_V2_FILENAME
    v2_render_path = resolved_dir / MUSIC_EDIT_RENDER_V2_FILENAME
    if v2_plan_path.is_file() or v2_render_path.is_file():
        return _validate_music_edit_v2_evidence(
            music_path=music_path,
            music_probe=music_probe,
            artifact_dir=resolved_dir,
        )

    try:
        plan, binding = load_music_assembly_artifacts(resolved_dir)
        plan_path = resolved_dir / MUSIC_ASSEMBLY_PLAN_FILENAME
        binding_path = resolved_dir / MUSIC_ASSEMBLY_BINDING_FILENAME
        render_path = resolved_dir / MUSIC_ASSEMBLY_RENDER_FILENAME
        render = MusicAssemblyRenderManifest.model_validate(read_json(render_path))
    except (OSError, ValueError, MusicAssemblyError) as error:
        raise FinalDeliveryError(
            "music assembly evidence could not be loaded and validated"
        ) from error
    music_sha256 = sha256_file(music_path)
    errors: list[str] = []
    canonical_plan_sha256 = music_assembly_plan_sha256(plan)
    if render.assembly_id != plan.assembly_id:
        errors.append("render assembly ID does not match the plan")
    if render.assembly_plan_canonical_sha256 != canonical_plan_sha256:
        errors.append("render canonical plan hash does not match the saved plan")
    if Path(render.output_audio_path).expanduser().resolve() != music_path:
        errors.append("render manifest output path is not the supplied music file")
    if render.output_audio_sha256 != music_sha256:
        errors.append("render manifest output hash does not match the music file")
    if plan.music_id != f"sha256:{render.source_audio_sha256}":
        errors.append("render source hash is not the source music locked by the plan")
    if not render.qc_passed or render.qc_errors:
        errors.append("music render manifest did not pass its own QC")
    if render.internal_join_count != 0 or plan.join_count != 0:
        errors.append("single-interval delivery cannot contain music joins")
    if render.ending_policy != plan.ending_policy:
        errors.append("render ending policy does not match the assembly plan")

    music_stream = _single_stream(
        music_probe,
        "audio",
        source_label="rendered music",
    )
    try:
        probed_sample_rate = int(str(music_stream["sample_rate"]))
        probed_channels = int(str(music_stream["channels"]))
    except (KeyError, TypeError, ValueError) as error:
        raise FinalDeliveryError(
            "ffprobe omitted rendered-music channel metadata"
        ) from error
    probed_samples = _stream_duration_samples(music_stream)
    if probed_sample_rate != render.output_sample_rate:
        errors.append("rendered music sample rate differs from its manifest")
    if probed_channels != render.output_channels:
        errors.append("rendered music channel count differs from its manifest")
    if abs(probed_samples - render.probed_output_samples) > (
        render.duration_tolerance_samples
    ):
        errors.append("rendered music sample duration differs from its manifest")
    expected_plan_samples = round(
        Fraction(
            plan.output_duration_samples * render.output_sample_rate,
            plan.master_sample_rate,
        )
    )
    if render.expected_output_samples != expected_plan_samples:
        errors.append("render expected duration does not match the assembly plan")
    if errors:
        raise FinalDeliveryError(
            "music assembly evidence validation failed: " + "; ".join(errors)
        )
    span = plan.spans[0]
    return {
        "status": "validated",
        "artifact_dir": str(resolved_dir),
        "assembly_id": plan.assembly_id,
        "assembly_mode": plan.assembly_mode,
        "join_count": plan.join_count,
        "ending_policy": plan.ending_policy,
        "source_span": {
            "start_sample": span.source_start_sample,
            "end_sample": span.source_end_sample,
            "end_boundary_kind": span.end_boundary_kind,
            "source_interval_selection_applied": (
                span.source_start_sample != 0
                or span.source_end_sample != plan.source_duration_samples
            ),
        },
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "canonical_sha256": canonical_plan_sha256,
        },
        "binding": {
            "path": str(binding_path),
            "sha256": sha256_file(binding_path),
            "assembly_plan_sha256": binding.assembly_plan_sha256,
            "music_lock_sha256": binding.music_lock_sha256,
        },
        "render": {
            "path": str(render_path),
            "sha256": sha256_file(render_path),
            "render_id": render.render_id,
            "output_audio_sha256": render.output_audio_sha256,
            "sample_rate": render.output_sample_rate,
            "channels": render.output_channels,
            "expected_output_samples": render.expected_output_samples,
            "probed_output_samples": render.probed_output_samples,
            "duration_tolerance_samples": render.duration_tolerance_samples,
            "fade_out_samples": render.fade_out_samples,
            "qc_passed": render.qc_passed,
        },
    }


def _validate_music_edit_v2_evidence(
    *,
    music_path: Path,
    music_probe: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Validate the reviewed multi-passage plan and deterministic PCM render."""

    plan_path = artifact_dir / MUSIC_EDIT_PLAN_V2_FILENAME
    render_path = artifact_dir / MUSIC_EDIT_RENDER_V2_FILENAME
    try:
        plan = MusicEditPlanV2.model_validate(read_json(plan_path))
        render = MusicEditRenderManifestV2.model_validate(read_json(render_path))
    except (OSError, ValueError) as error:
        raise FinalDeliveryError(
            "music edit v2 evidence could not be loaded and validated"
        ) from error

    errors: list[str] = []
    music_sha256 = sha256_file(music_path)
    canonical_plan_sha256 = music_edit_plan_v2_sha256(plan)
    lock_path = Path(plan.music_lock_path).expanduser().resolve()
    if render.edit_id != plan.edit_id:
        errors.append("render edit ID does not match the plan")
    if render.edit_plan_canonical_sha256 != canonical_plan_sha256:
        errors.append("render canonical plan hash does not match the saved plan")
    if Path(render.output_audio_path).expanduser().resolve() != music_path:
        errors.append("render manifest output path is not the supplied music file")
    if render.output_audio_sha256 != music_sha256:
        errors.append("render manifest output hash does not match the music file")
    if plan.music_id != f"sha256:{render.source_audio_sha256}":
        errors.append("render source hash is not the source music locked by the plan")
    if not lock_path.is_file() or sha256_file(lock_path) != plan.music_lock_sha256:
        errors.append("reviewed music lock no longer matches the edit plan")
    if not render.qc_passed or render.qc_errors:
        errors.append("music edit render manifest did not pass its own QC")
    if render.internal_join_count != len(plan.joins):
        errors.append("render join count does not match the edit plan")

    music_stream = _single_stream(
        music_probe,
        "audio",
        source_label="rendered music",
    )
    try:
        probed_sample_rate = int(str(music_stream["sample_rate"]))
        probed_channels = int(str(music_stream["channels"]))
    except (KeyError, TypeError, ValueError) as error:
        raise FinalDeliveryError(
            "ffprobe omitted rendered-music channel metadata"
        ) from error
    probed_samples = _stream_duration_samples(music_stream)
    if probed_sample_rate != 48_000:
        errors.append("music edit v2 must render at 48 kHz")
    if probed_channels != 2:
        errors.append("music edit v2 must render stereo PCM")
    if abs(probed_samples - render.probed_output_samples) > (
        render.duration_tolerance_samples
    ):
        errors.append("rendered music sample duration differs from its manifest")
    if render.expected_output_samples != plan.output_duration_samples:
        errors.append("render expected duration does not match the edit plan")
    if errors:
        raise FinalDeliveryError(
            "music edit v2 evidence validation failed: " + "; ".join(errors)
        )

    return {
        "status": "validated",
        "artifact_dir": str(artifact_dir),
        "assembly_id": plan.edit_id,
        "assembly_mode": "reviewed_multi_passage_v2",
        "join_count": len(plan.joins),
        "ending_policy": plan.ending.mode,
        "source_spans": [
            {
                "span_id": span.span_id,
                "section_id": span.section_id,
                "source_start_sample": span.source_start_sample,
                "source_end_sample": span.source_end_sample,
                "output_start_sample": span.output_start_sample,
                "output_end_sample": span.output_end_sample,
            }
            for span in plan.spans
        ],
        "joins": [join.model_dump(mode="json") for join in plan.joins],
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "canonical_sha256": canonical_plan_sha256,
            "music_lock_path": str(lock_path),
            "music_lock_sha256": plan.music_lock_sha256,
        },
        "binding": None,
        "render": {
            "path": str(render_path),
            "sha256": sha256_file(render_path),
            "render_id": render.render_id,
            "output_audio_sha256": render.output_audio_sha256,
            "sample_rate": probed_sample_rate,
            "channels": probed_channels,
            "expected_output_samples": render.expected_output_samples,
            "probed_output_samples": render.probed_output_samples,
            "duration_tolerance_samples": render.duration_tolerance_samples,
            "fade_out_samples": render.fade_out_samples,
            "qc_passed": render.qc_passed,
        },
    }


def assemble_music_only_delivery(
    *,
    picture_path: Path,
    music_path: Path,
    output_path: Path,
    manifest_path: Path,
    music_assembly_artifact_dir: Path,
    aspect_ratio: Literal["16:9", "9:16"],
    artifact_bindings: Mapping[str, str] | None = None,
    duration_tolerance_ms: int = 100,
) -> FinalDeliveryResult:
    """Mux one reviewed soundtrack without altering editorial picture timing.

    This stage is intentionally not an editor. It refuses materially different
    picture/music durations instead of trimming music, repeating picture, or
    generating a hold. Source audio is excluded so adjacent rushes cannot
    overlap beneath the approved soundtrack.
    """

    if duration_tolerance_ms < 0:
        raise ValueError("duration tolerance must not be negative")
    picture = picture_path.expanduser().resolve(strict=True)
    music = music_path.expanduser().resolve(strict=True)
    output = output_path.expanduser().resolve()
    manifest_target = manifest_path.expanduser().resolve()
    picture_probe = _probe(picture)
    music_probe = _probe(music)
    picture_video_stream = _single_stream(
        picture_probe,
        "video",
        source_label="picture input",
    )
    music_audio_stream = _single_stream(
        music_probe,
        "audio",
        source_label="music input",
    )
    assembly_evidence = _validate_music_assembly_evidence(
        music_path=music,
        music_probe=music_probe,
        artifact_dir=music_assembly_artifact_dir,
    )
    picture_duration_ms = _stream_duration_ms(picture_video_stream)
    music_duration_ms = _stream_duration_ms(music_audio_stream)
    duration_delta_ms = picture_duration_ms - music_duration_ms
    if abs(duration_delta_ms) > duration_tolerance_ms:
        raise FinalDeliveryError(
            "picture and continuous music durations differ by "
            f"{duration_delta_ms} ms; revise the editorial or MusicAssembly plan "
            "instead of hard-cutting either stream"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(picture),
        "-i",
        str(music),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "-1",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise FinalDeliveryError("FFmpeg did not create a non-empty delivery")
    output_probe = _probe(output)
    output_video_stream = _single_stream(
        output_probe,
        "video",
        source_label="final delivery",
    )
    output_audio_stream = _single_stream(
        output_probe,
        "audio",
        source_label="final delivery",
    )
    output_duration_ms = _duration_ms(output_probe)
    output_video_duration_ms = _stream_duration_ms(output_video_stream)
    output_audio_duration_ms = _stream_duration_ms(output_audio_stream)
    output_video_start_ms = _stream_start_ms(output_video_stream)
    output_audio_start_ms = _stream_start_ms(output_audio_stream)
    per_stream_deltas = {
        "video_output_minus_picture_ms": (
            output_video_duration_ms - picture_duration_ms
        ),
        "audio_output_minus_music_ms": (
            output_audio_duration_ms - music_duration_ms
        ),
        "output_video_minus_audio_ms": (
            output_video_duration_ms - output_audio_duration_ms
        ),
        "output_video_start_minus_audio_start_ms": (
            output_video_start_ms - output_audio_start_ms
        ),
        "output_video_start_from_zero_ms": output_video_start_ms,
        "output_audio_start_from_zero_ms": output_audio_start_ms,
    }
    failed_measurements = {
        name: value
        for name, value in per_stream_deltas.items()
        if abs(value) > duration_tolerance_ms
    }
    if failed_measurements:
        raise FinalDeliveryError(
            "final delivery per-stream timing QC failed: "
            + ", ".join(
                f"{name}={value} ms"
                for name, value in sorted(failed_measurements.items())
            )
        )

    manifest: dict[str, Any] = {
        "contract_version": "final-music-delivery-v2",
        "generated_at": utc_now(),
        "aspect_ratio": aspect_ratio,
        "picture": {
            "path": str(picture),
            "sha256": sha256_file(picture),
            "duration_ms": picture_duration_ms,
            "duration_source": "video_stream",
            "probe": picture_probe,
        },
        "music": {
            "path": str(music),
            "sha256": sha256_file(music),
            "duration_ms": music_duration_ms,
            "duration_source": "audio_stream",
            "probe": music_probe,
            "assembly_mode": assembly_evidence["assembly_mode"],
            "join_count": assembly_evidence["join_count"],
            "internal_music_edits": assembly_evidence.get("joins", []),
            "assembly_evidence": assembly_evidence,
        },
        "synchronization": {
            "picture_minus_music_ms": duration_delta_ms,
            "duration_tolerance_ms": duration_tolerance_ms,
            "output_video_duration_ms": output_video_duration_ms,
            "output_audio_duration_ms": output_audio_duration_ms,
            "output_video_start_ms": output_video_start_ms,
            "output_audio_start_ms": output_audio_start_ms,
            "per_stream_deltas_ms": per_stream_deltas,
            "per_stream_qc_passed": True,
            "ffmpeg_shortest_used": False,
            "picture_extension_applied": False,
            "delivery_music_trim_applied": False,
            "music_time_stretch_applied": False,
        },
        "audio_policy": {
            "source_audio": "excluded",
            "delivery_audio": "continuous_music_only",
            "reason": (
                "prevent overlapping rush audio and preserve the approved "
                "reviewed soundtrack flow validated by immutable planning and "
                "render evidence"
            ),
        },
        "artifact_bindings": dict(artifact_bindings or {}),
        "ffmpeg": {
            "command": command,
            "video_codec_policy": "stream_copy",
            "audio_codec_policy": "aac_192k_48khz_stereo",
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "duration_ms": output_duration_ms,
            "video_stream_duration_ms": output_video_duration_ms,
            "audio_stream_duration_ms": output_audio_duration_ms,
            "probe": output_probe,
        },
    }
    write_json(manifest_target, manifest)
    return FinalDeliveryResult(
        output_path=output,
        manifest_path=manifest_target,
        manifest=manifest,
    )
