"""Deterministic source-FPS quality evidence for shortlisted video shots.

The scanner measures suspicious intervals.  It never decides that a rack focus,
locked camera, flash, or hold is editorially disposable.  Exact PTS stay local;
Gemini or a human may later classify the measured intent without inventing time.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageChops

from .media import sha256_file
from .models import (
    AspectCandidateCapacity,
    CandidateCapacity,
    QualityFrameEvidence,
    QualityRiskIntent,
    QualityRiskReason,
    QualityRiskSeverity,
    QualityRiskWindow,
    QualitySafeInterval,
    Rational,
    ShotQualityMap,
)
from .shots import ShotManifest, ShotSegment
from .storage import read_json, utc_now, write_json


SHOT_QUALITY_SCANNER_VERSION = "shot-quality-source-fps-v2"
_AUTO_EDGE_CLEAN_REASON_CODES = frozenset(
    {"focus_loss", "motion_blur", "camera_shake"}
)
_AUTO_EDGE_CLEAN_MAX_WINDOW_MS = 1_200
_AUTO_EDGE_TOUCH_TOLERANCE_MS = 250
_AUTO_EDGE_SETTLE_PADDING_MS = 200


@dataclass(frozen=True)
class _FrameMeta:
    global_index: int
    pts: int
    local_time_ms: int
    end_pts: int
    end_time_ms: int


@dataclass(frozen=True)
class _FrameMeasurement:
    meta: _FrameMeta
    analysis_sha256: str
    average_hash: int
    mean_luma: float
    black_fraction: float
    white_fraction: float
    focus_score: float
    mean_delta: float
    shift_x: int
    shift_y: int


@dataclass(frozen=True)
class _RiskCandidate:
    reason_code: QualityRiskReason
    severity: QualityRiskSeverity
    intent: QualityRiskIntent
    confidence: float
    frame_indexes: tuple[int, ...]
    metric_summary: dict[str, float]


def _canonical_sha256(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _probe_source_frames(
    video_path: Path,
) -> tuple[int, Fraction, list[_FrameMeta], list[str]]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=start_pts,time_base:"
                "frame=best_effort_timestamp,pkt_duration"
            ),
            "-of",
            "json",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffprobe could not enumerate decoded source PTS: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        source_start_pts = int(stream.get("start_pts") or 0)
        time_base = Fraction(stream["time_base"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("ffprobe returned unusable source timing metadata") from error
    raw_frames = payload.get("frames") or []
    warnings: list[str] = []
    pts_values: list[int] = []
    durations: list[int | None] = []
    for frame in raw_frames:
        if not isinstance(frame, dict) or frame.get("best_effort_timestamp") is None:
            warnings.append("ffprobe omitted PTS for at least one decoded frame")
            continue
        pts_values.append(int(frame["best_effort_timestamp"]))
        duration_value = frame.get("pkt_duration")
        durations.append(int(duration_value) if duration_value is not None else None)
    if not pts_values:
        raise ValueError("quality scan found no decoded video frame timestamps")
    positive_deltas = [
        current - previous
        for previous, current in zip(pts_values[:-1], pts_values[1:], strict=True)
        if current > previous
    ]
    fallback_duration = (
        round(statistics.median(positive_deltas)) if positive_deltas else 1
    )
    frames: list[_FrameMeta] = []
    for index, pts in enumerate(pts_values):
        duration_pts = durations[index] or (
            pts_values[index + 1] - pts
            if index + 1 < len(pts_values) and pts_values[index + 1] > pts
            else fallback_duration
        )
        duration_pts = max(1, duration_pts)
        end_pts = pts + duration_pts
        frames.append(
            _FrameMeta(
                global_index=index,
                pts=pts,
                local_time_ms=max(
                    0,
                    round(Fraction(pts - source_start_pts) * time_base * 1000),
                ),
                end_pts=end_pts,
                end_time_ms=max(
                    1,
                    round(
                        Fraction(end_pts - source_start_pts)
                        * time_base
                        * 1000
                    ),
                ),
            )
        )
    return source_start_pts, time_base, frames, warnings


def _decode_analysis_frames(
    video_path: Path,
    *,
    analysis_width: int,
    analysis_height: int,
    first_frame_index: int,
    last_frame_index: int,
    expected_count: int,
) -> tuple[list[bytes], str]:
    if expected_count == 0:
        return [], ""
    if (
        first_frame_index < 0
        or last_frame_index < first_frame_index
        or last_frame_index - first_frame_index + 1 != expected_count
    ):
        raise ValueError("shot-quality frame-index interval is inconsistent")
    frame_size = analysis_width * analysis_height
    filter_graph = (
        f"select='between(n\\,{first_frame_index}\\,{last_frame_index})',"
        f"scale={analysis_width}:{analysis_height}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={analysis_width}:{analysis_height}:(ow-iw)/2:(oh-ih)/2,"
        "format=gray"
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(expected_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to open FFmpeg shot-quality pipes")
    frames: list[bytes] = []
    while True:
        payload = process.stdout.read(frame_size)
        if not payload:
            break
        if len(payload) != frame_size:
            process.kill()
            raise RuntimeError("FFmpeg returned a truncated quality-analysis frame")
        frames.append(payload)
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg shot-quality decode failed: {stderr.strip()}")
    if len(frames) != expected_count:
        stderr = (
            f"{stderr}; " if stderr.strip() else ""
        ) + (
            "decoded raw-frame count differs from ffprobe frame count: "
            f"{len(frames)} != {expected_count}"
        )
    return frames, stderr.strip()


def _focus_score(image: Image.Image) -> float:
    width, height = image.size
    horizontal = ImageChops.difference(
        image.crop((1, 0, width, height)),
        image.crop((0, 0, width - 1, height)),
    )
    vertical = ImageChops.difference(
        image.crop((0, 1, width, height)),
        image.crop((0, 0, width, height - 1)),
    )

    def mean_square(value: Image.Image) -> float:
        histogram = value.histogram()
        count = max(1, value.width * value.height)
        return (
            sum((level * level) * frequency for level, frequency in enumerate(histogram))
            / count
            / (255 * 255)
        )

    return (mean_square(horizontal) + mean_square(vertical)) / 2


def _average_hash(image: Image.Image) -> int:
    sample = image.resize((8, 8), Image.Resampling.BILINEAR)
    values = list(sample.getdata())
    average = sum(values) / len(values)
    result = 0
    for index, value in enumerate(values):
        if value >= average:
            result |= 1 << index
    return result


def _estimate_translation(
    previous: Image.Image | None,
    current: Image.Image,
) -> tuple[int, int]:
    if previous is None:
        return 0, 0
    prior = previous.resize((48, 27), Image.Resampling.BILINEAR)
    now = current.resize((48, 27), Image.Resampling.BILINEAR)
    prior_pixels = prior.load()
    now_pixels = now.load()
    best = (math.inf, 0, 0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            x_start = max(0, -dx)
            x_end = min(48, 48 - dx)
            y_start = max(0, -dy)
            y_end = min(27, 27 - dy)
            if x_end <= x_start or y_end <= y_start:
                continue
            total = 0
            count = 0
            for y in range(y_start, y_end, 2):
                for x in range(x_start, x_end, 2):
                    total += abs(
                        int(prior_pixels[x, y])
                        - int(now_pixels[x + dx, y + dy])
                    )
                    count += 1
            score = total / max(1, count)
            candidate = (score, abs(dx) + abs(dy), dy * 10 + dx)
            if candidate < (best[0], abs(best[1]) + abs(best[2]), best[2] * 10 + best[1]):
                best = (score, dx, dy)
    return best[1], best[2]


def _measure_frames(
    metas: Sequence[_FrameMeta],
    payloads: Sequence[bytes],
    *,
    analysis_width: int,
    analysis_height: int,
) -> list[_FrameMeasurement]:
    measurements: list[_FrameMeasurement] = []
    previous: Image.Image | None = None
    for meta, payload in zip(metas, payloads, strict=True):
        image = Image.frombytes("L", (analysis_width, analysis_height), payload)
        histogram = image.histogram()
        pixel_count = analysis_width * analysis_height
        mean_luma = sum(
            level * frequency for level, frequency in enumerate(histogram)
        ) / (pixel_count * 255)
        black_fraction = sum(histogram[:13]) / pixel_count
        white_fraction = sum(histogram[243:]) / pixel_count
        if previous is None:
            mean_delta = 0.0
        else:
            delta_histogram = ImageChops.difference(previous, image).histogram()
            mean_delta = sum(
                level * frequency
                for level, frequency in enumerate(delta_histogram)
            ) / (pixel_count * 255)
        shift_x, shift_y = _estimate_translation(previous, image)
        measurements.append(
            _FrameMeasurement(
                meta=meta,
                analysis_sha256=hashlib.sha256(payload).hexdigest(),
                average_hash=_average_hash(image),
                mean_luma=mean_luma,
                black_fraction=black_fraction,
                white_fraction=white_fraction,
                focus_score=_focus_score(image),
                mean_delta=mean_delta,
                shift_x=shift_x,
                shift_y=shift_y,
            )
        )
        previous = image
    return measurements


def _contiguous_groups(
    indexes: Iterable[int],
    measurements: Sequence[_FrameMeasurement],
    *,
    maximum_gap_factor: float = 2.5,
) -> list[tuple[int, ...]]:
    ordered = sorted(set(indexes))
    if not ordered:
        return []
    frame_durations = [
        max(1, item.meta.end_time_ms - item.meta.local_time_ms)
        for item in measurements
    ]
    typical_duration = statistics.median(frame_durations)
    groups: list[list[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        previous = groups[-1][-1]
        gap_ms = (
            measurements[index].meta.local_time_ms
            - measurements[previous].meta.end_time_ms
        )
        if index == previous + 1 and gap_ms <= typical_duration * maximum_gap_factor:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [tuple(group) for group in groups]


def _candidate_groups(
    measurements: Sequence[_FrameMeasurement],
) -> list[_RiskCandidate]:
    if not measurements:
        return []
    candidates: list[_RiskCandidate] = []

    def add_groups(
        indexes: Iterable[int],
        *,
        reason: QualityRiskReason,
        severity: QualityRiskSeverity,
        confidence: float,
        minimum_duration_ms: int = 0,
        metric_name: str,
        metric_getter,
    ) -> None:
        for group in _contiguous_groups(indexes, measurements):
            start = measurements[group[0]].meta.local_time_ms
            end = measurements[group[-1]].meta.end_time_ms
            if end - start < minimum_duration_ms:
                continue
            values = [float(metric_getter(measurements[index])) for index in group]
            candidates.append(
                _RiskCandidate(
                    reason_code=reason,
                    severity=severity,
                    intent="unknown",
                    confidence=confidence,
                    frame_indexes=group,
                    metric_summary={
                        f"{metric_name}_min": round(min(values), 8),
                        f"{metric_name}_max": round(max(values), 8),
                        f"{metric_name}_mean": round(sum(values) / len(values), 8),
                    },
                )
            )

    add_groups(
        (
            index
            for index, item in enumerate(measurements)
            if item.mean_luma <= 0.035 and item.black_fraction >= 0.98
        ),
        reason="black",
        severity="trim_candidate",
        confidence=0.98,
        metric_name="black_fraction",
        metric_getter=lambda item: item.black_fraction,
    )
    add_groups(
        (
            index
            for index, item in enumerate(measurements)
            if item.mean_luma >= 0.965 and item.white_fraction >= 0.98
        ),
        reason="white_clip",
        severity="trim_candidate",
        confidence=0.98,
        metric_name="white_fraction",
        metric_getter=lambda item: item.white_fraction,
    )

    freeze_indexes: list[int] = []
    near_static_indexes: list[int] = []
    for index in range(1, len(measurements)):
        previous = measurements[index - 1]
        current = measurements[index]
        exact = current.analysis_sha256 == previous.analysis_sha256
        near = (
            (current.average_hash ^ previous.average_hash).bit_count() <= 1
            and current.mean_delta <= 0.00035
        )
        if exact:
            freeze_indexes.extend([index - 1, index])
        elif near:
            near_static_indexes.extend([index - 1, index])
    add_groups(
        freeze_indexes,
        reason="freeze",
        severity="trim_candidate",
        confidence=0.94,
        minimum_duration_ms=400,
        metric_name="mean_delta",
        metric_getter=lambda item: item.mean_delta,
    )
    add_groups(
        near_static_indexes,
        reason="freeze",
        severity="review",
        confidence=0.70,
        minimum_duration_ms=400,
        metric_name="mean_delta",
        metric_getter=lambda item: item.mean_delta,
    )

    focus_values = [item.focus_score for item in measurements]
    median_focus = statistics.median(focus_values)
    focus_threshold = max(0.00002, median_focus * 0.28)
    low_focus = [
        index
        for index, item in enumerate(measurements)
        if median_focus > 0.00005 and item.focus_score < focus_threshold
    ]
    add_groups(
        low_focus,
        reason="focus_loss",
        severity="review",
        confidence=0.78,
        minimum_duration_ms=120,
        metric_name="focus_score",
        metric_getter=lambda item: item.focus_score,
    )
    add_groups(
        (
            index
            for index in low_focus
            if measurements[index].mean_delta >= 0.025
        ),
        reason="motion_blur",
        severity="review",
        confidence=0.7,
        minimum_duration_ms=80,
        metric_name="mean_delta",
        metric_getter=lambda item: item.mean_delta,
    )

    shake_indexes: list[int] = []
    accelerations: dict[int, float] = {}
    for index in range(2, len(measurements)):
        prior = measurements[index - 1]
        current = measurements[index]
        acceleration = math.hypot(
            current.shift_x - prior.shift_x,
            current.shift_y - prior.shift_y,
        )
        accelerations[index] = acceleration
        if acceleration >= 3.5 and current.mean_delta >= 0.015:
            shake_indexes.append(index)
    for group in _contiguous_groups(shake_indexes, measurements):
        values = [accelerations[index] for index in group]
        candidates.append(
            _RiskCandidate(
                reason_code="camera_shake",
                severity="review",
                intent="unknown",
                confidence=0.64,
                frame_indexes=group,
                metric_summary={
                    "analysis_translation_acceleration_max": round(max(values), 6),
                    "analysis_translation_acceleration_mean": round(
                        sum(values) / len(values),
                        6,
                    ),
                },
            )
        )

    pts = [item.meta.pts for item in measurements]
    for index in range(1, len(pts)):
        if pts[index] <= pts[index - 1]:
            candidates.append(
                _RiskCandidate(
                    reason_code="duplicate_pts",
                    severity="hard_block",
                    intent="accidental",
                    confidence=1.0,
                    frame_indexes=(index - 1, index),
                    metric_summary={
                        "prior_pts": float(pts[index - 1]),
                        "current_pts": float(pts[index]),
                    },
                )
            )
    positive_deltas = [
        current - previous
        for previous, current in zip(pts[:-1], pts[1:], strict=True)
        if current > previous
    ]
    if positive_deltas:
        median_delta = statistics.median(positive_deltas)
        for index, delta in enumerate(positive_deltas, start=1):
            if delta > median_delta * 4:
                candidates.append(
                    _RiskCandidate(
                        reason_code="decoder_gap",
                        severity="hard_block",
                        intent="accidental",
                        confidence=0.98,
                        frame_indexes=(index - 1, index),
                        metric_summary={
                            "pts_delta": float(delta),
                            "median_pts_delta": float(median_delta),
                        },
                    )
                )
    return candidates


def _frame_id(
    source_sha256: str,
    frame: _FrameMeasurement,
    request_sha256: str,
) -> str:
    digest = hashlib.sha256(
        (
            source_sha256
            + ":"
            + str(frame.meta.pts)
            + ":"
            + frame.analysis_sha256
            + ":"
            + request_sha256
        ).encode("utf-8")
    ).hexdigest()
    return f"QF-{digest[:16]}"


def scan_shot_quality(
    video_path: Path,
    *,
    shot_manifest: ShotManifest,
    shot_id: str,
    analysis_width: int = 320,
    analysis_height: int = 180,
    output_path: Path | None = None,
) -> ShotQualityMap:
    """Measure one selected shot at source FPS and save exact PTS evidence."""

    resolved_video = video_path.expanduser().resolve(strict=True)
    if analysis_width <= 0 or analysis_height <= 0:
        raise ValueError("quality analysis dimensions must be positive")
    if Path(shot_manifest.video_path).expanduser().resolve(strict=True) != resolved_video:
        raise ValueError("shot manifest belongs to a different source video")
    shot = next((item for item in shot_manifest.shots if item.shot_id == shot_id), None)
    if shot is None:
        raise ValueError(f"unknown shot ID: {shot_id}")
    source_sha256 = sha256_file(resolved_video)
    source_start_pts, time_base, all_metas, warnings = _probe_source_frames(
        resolved_video
    )
    selected_metas = [
        meta
        for meta in all_metas
        if shot.start_time_ms <= meta.local_time_ms < shot.end_time_ms
    ]
    raw_frames, decode_warning = _decode_analysis_frames(
        resolved_video,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        first_frame_index=(
            selected_metas[0].global_index if selected_metas else 0
        ),
        last_frame_index=(
            selected_metas[-1].global_index if selected_metas else -1
        ),
        expected_count=len(selected_metas),
    )
    probed_frame_count = len(selected_metas)
    raw_decode_frame_count = len(raw_frames)
    paired_count = min(len(selected_metas), len(raw_frames))
    selected_metas = selected_metas[:paired_count]
    raw_frames = raw_frames[:paired_count]
    if decode_warning:
        warnings.append(decode_warning)
    selected_payloads = raw_frames
    request = {
        "scanner_version": SHOT_QUALITY_SCANNER_VERSION,
        "source_sha256": source_sha256,
        "shot_manifest": {
            "timeline_basis": shot_manifest.timeline_basis,
            "source_start_pts": shot_manifest.source_start_pts,
            "source_time_base": (
                shot_manifest.source_time_base.model_dump(mode="json")
                if shot_manifest.source_time_base is not None
                else None
            ),
            "shot": shot.model_dump(mode="json"),
        },
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "measurement_contract": {
            "black_luma_max": 0.035,
            "black_fraction_min": 0.98,
            "white_luma_min": 0.965,
            "white_fraction_min": 0.98,
            "freeze_min_duration_ms": 400,
            "focus_threshold": "per-shot median * 0.28",
            "source_fps_decode": True,
        },
    }
    request_sha256 = _canonical_sha256(request)
    measurements = _measure_frames(
        selected_metas,
        selected_payloads,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
    )
    candidates = _candidate_groups(measurements)
    if not measurements:
        candidates.append(
            _RiskCandidate(
                reason_code="decoder_gap",
                severity="hard_block",
                intent="accidental",
                confidence=1.0,
                frame_indexes=(),
                metric_summary={
                    "selected_decoded_frame_count": 0.0,
                },
            )
        )
        warnings.append(
            f"no decoded source frames were available inside shot {shot.shot_id}"
        )
    if raw_decode_frame_count != probed_frame_count:
        candidates.append(
            _RiskCandidate(
                reason_code="decoder_gap",
                severity="hard_block",
                intent="accidental",
                confidence=1.0,
                frame_indexes=tuple(range(min(2, len(measurements)))),
                metric_summary={
                    "ffprobe_frame_count": float(probed_frame_count),
                    "raw_decode_frame_count": float(raw_decode_frame_count),
                },
            )
        )

    referenced_indexes = sorted(
        {
            frame_index
            for candidate in candidates
            for frame_index in candidate.frame_indexes
            if 0 <= frame_index < len(measurements)
        }
    )
    evidence_by_index = {
        index: QualityFrameEvidence(
            frame_id=_frame_id(source_sha256, measurements[index], request_sha256),
            frame_pts=measurements[index].meta.pts,
            frame_time_ms=measurements[index].meta.local_time_ms,
            analysis_frame_sha256=measurements[index].analysis_sha256,
        )
        for index in referenced_indexes
    }
    windows: list[QualityRiskWindow] = []
    for window_index, candidate in enumerate(
        sorted(
            candidates,
            key=lambda item: (
                measurements[item.frame_indexes[0]].meta.local_time_ms
                if item.frame_indexes and measurements
                else shot.start_time_ms,
                item.reason_code,
            ),
        ),
        start=1,
    ):
        valid_indexes = [
            index
            for index in candidate.frame_indexes
            if 0 <= index < len(measurements)
        ]
        if valid_indexes:
            start_frame = measurements[min(valid_indexes)]
            end_frame = measurements[max(valid_indexes)]
            start_pts = start_frame.meta.pts
            end_pts = end_frame.meta.end_pts
            start_ms = max(shot.start_time_ms, start_frame.meta.local_time_ms)
            end_ms = min(shot.end_time_ms, end_frame.meta.end_time_ms)
            evidence_ids = [
                evidence_by_index[index].frame_id for index in valid_indexes
            ][:16]
        else:
            start_pts = shot.start_frame_pts or source_start_pts
            end_pts = start_pts + 1
            start_ms = shot.start_time_ms
            end_ms = min(shot.end_time_ms, shot.start_time_ms + 1)
            evidence_ids = []
        windows.append(
            QualityRiskWindow(
                risk_window_id=f"QRW-{window_index:04d}",
                source_asset_id=f"sha256:{source_sha256}",
                shot_id=shot.shot_id,
                start_pts=start_pts,
                end_pts=end_pts,
                start_ms=start_ms,
                end_ms=end_ms,
                reason_code=candidate.reason_code,
                severity=candidate.severity,
                intent=candidate.intent,
                confidence=candidate.confidence,
                evidence_frame_ids=list(dict.fromkeys(evidence_ids)),
                metric_summary=candidate.metric_summary,
            )
        )
    shot_start_pts = (
        shot.start_frame_pts
        if shot.start_frame_pts is not None
        else source_start_pts
        + round(Fraction(shot.start_time_ms, 1000) / time_base)
    )
    shot_end_pts = source_start_pts + round(
        Fraction(shot.end_time_ms, 1000) / time_base
    )
    result = ShotQualityMap(
        scanner_version=SHOT_QUALITY_SCANNER_VERSION,
        source_path=str(resolved_video),
        source_asset_id=f"sha256:{source_sha256}",
        shot_id=shot.shot_id,
        shot_start_pts=shot_start_pts,
        shot_end_pts=max(shot_start_pts + 1, shot_end_pts),
        shot_start_ms=shot.start_time_ms,
        shot_end_ms=shot.end_time_ms,
        source_time_base=Rational(
            numerator=time_base.numerator,
            denominator=time_base.denominator,
        ),
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        decoded_frame_count=len(measurements),
        request_sha256=request_sha256,
        evidence_frames=[evidence_by_index[index] for index in referenced_indexes],
        risk_windows=windows,
        warnings=[
            *warnings,
            (
                "Local measurements identify suspicious intervals only. "
                "Unknown intent requires Gemini or human review before deletion."
            ),
        ],
        generated_at=utc_now(),
    )
    if output_path is not None:
        write_json(output_path, result)
    return result


def load_shot_quality_map(path: Path) -> tuple[Path, ShotQualityMap]:
    resolved = path.expanduser().resolve(strict=True)
    return resolved, ShotQualityMap.model_validate(read_json(resolved))


def _pts_for_local_ms(quality_map: ShotQualityMap, time_ms: int) -> int:
    time_base = Fraction(
        quality_map.source_time_base.numerator,
        quality_map.source_time_base.denominator,
    )
    offset_ms = time_ms - quality_map.shot_start_ms
    return quality_map.shot_start_pts + round(
        Fraction(offset_ms, 1000) / time_base
    )


def build_quality_safe_intervals(
    quality_map: ShotQualityMap,
    *,
    quality_map_sha256: str | None = None,
    allowed_start_ms: int | None = None,
    allowed_end_ms: int | None = None,
) -> list[QualitySafeInterval]:
    """Subtract unresolved blocking/trim risks from one continuous source shot.

    Unknown trim candidates are excluded from automatic planning, not deleted.
    A later reviewed artifact may mark them intentional, at which point the same
    deterministic function will retain them.
    """

    start_ms = (
        quality_map.shot_start_ms
        if allowed_start_ms is None
        else allowed_start_ms
    )
    end_ms = (
        quality_map.shot_end_ms if allowed_end_ms is None else allowed_end_ms
    )
    if not (
        quality_map.shot_start_ms
        <= start_ms
        < end_ms
        <= quality_map.shot_end_ms
    ):
        raise ValueError("quality-safe bounds must stay inside the scanned shot")
    map_sha256 = quality_map_sha256 or _canonical_sha256(quality_map)
    exclusions = [
        window
        for window in quality_map.risk_windows
        if (
            window.severity == "hard_block"
            or (
                window.severity == "trim_candidate"
                and window.intent != "intentional"
            )
        )
        and window.start_ms < end_ms
        and start_ms < window.end_ms
    ]
    promoted_edge_review_ids: set[str] = set()
    for window in quality_map.risk_windows:
        duration_ms = window.end_ms - window.start_ms
        if (
            window.severity != "review"
            or window.intent == "intentional"
            or window.reason_code not in _AUTO_EDGE_CLEAN_REASON_CODES
            or duration_ms > _AUTO_EDGE_CLEAN_MAX_WINDOW_MS
        ):
            continue
        touches_leading_edge = (
            window.start_ms
            <= start_ms + _AUTO_EDGE_TOUCH_TOLERANCE_MS
        )
        touches_trailing_edge = (
            window.end_ms
            >= end_ms - _AUTO_EDGE_TOUCH_TOLERANCE_MS
        )
        if not touches_leading_edge and not touches_trailing_edge:
            continue
        promoted_start = max(start_ms, window.start_ms)
        promoted_end = min(end_ms, window.end_ms)
        if touches_leading_edge:
            promoted_end = min(
                end_ms,
                promoted_end + _AUTO_EDGE_SETTLE_PADDING_MS,
            )
        if touches_trailing_edge:
            promoted_start = max(
                start_ms,
                promoted_start - _AUTO_EDGE_SETTLE_PADDING_MS,
            )
        exclusions.append(
            window.model_copy(
                update={
                    "start_ms": promoted_start,
                    "end_ms": promoted_end,
                }
            )
        )
        promoted_edge_review_ids.add(window.risk_window_id)
    merged: list[tuple[int, int, list[str]]] = []
    for window in sorted(exclusions, key=lambda item: (item.start_ms, item.end_ms)):
        window_start = max(start_ms, window.start_ms)
        window_end = min(end_ms, window.end_ms)
        if not merged or window_start > merged[-1][1]:
            merged.append((window_start, window_end, [window.risk_window_id]))
        else:
            previous_start, previous_end, ids = merged[-1]
            merged[-1] = (
                previous_start,
                max(previous_end, window_end),
                [*ids, window.risk_window_id],
            )
    raw_safe: list[tuple[int, int, list[str]]] = []
    cursor = start_ms
    all_excluded_ids = [window.risk_window_id for window in exclusions]
    for excluded_start, excluded_end, _ in merged:
        if cursor < excluded_start:
            raw_safe.append((cursor, excluded_start, all_excluded_ids))
        cursor = max(cursor, excluded_end)
    if cursor < end_ms:
        raw_safe.append((cursor, end_ms, all_excluded_ids))
    review_windows = [
        window
        for window in quality_map.risk_windows
        if (
            window.severity == "review"
            and window.risk_window_id not in promoted_edge_review_ids
        )
    ]
    intervals: list[QualitySafeInterval] = []
    for index, (interval_start, interval_end, excluded_ids) in enumerate(
        raw_safe,
        start=1,
    ):
        overlapping_review_ids = [
            window.risk_window_id
            for window in review_windows
            if window.start_ms < interval_end and interval_start < window.end_ms
        ]
        intervals.append(
            QualitySafeInterval(
                interval_id=f"QSI-{index:04d}",
                source_asset_id=quality_map.source_asset_id,
                shot_id=quality_map.shot_id,
                start_pts=_pts_for_local_ms(quality_map, interval_start),
                end_pts=_pts_for_local_ms(quality_map, interval_end),
                start_ms=interval_start,
                end_ms=interval_end,
                excluded_risk_window_ids=sorted(set(excluded_ids)),
                review_risk_window_ids=sorted(set(overlapping_review_ids)),
                requires_human_review=bool(overlapping_review_ids),
                quality_map_sha256=map_sha256,
            )
        )
    return intervals


def build_candidate_capacity(
    *,
    candidate_id: str,
    quality_map_path: Path,
    preferred_duration: float,
    min_editorial_duration: float = 0.0,
    horizontal_geometry_status: str = "not_evaluated",
    vertical_geometry_status: str = "not_evaluated",
    horizontal_intervals: Sequence[QualitySafeInterval] | None = None,
    vertical_intervals: Sequence[QualitySafeInterval] | None = None,
) -> CandidateCapacity:
    """Build aspect-specific continuous capacity from one quality artifact."""

    resolved_path, quality_map = load_shot_quality_map(quality_map_path)
    quality_sha256 = sha256_file(resolved_path)
    base_intervals = build_quality_safe_intervals(
        quality_map,
        quality_map_sha256=quality_sha256,
    )
    horizontal = list(horizontal_intervals or base_intervals)
    vertical = list(vertical_intervals or base_intervals)

    def aspect_capacity(aspect: str, status: str, intervals):
        if status == "blocked":
            intervals = []
        maximum = max(
            (
                (interval.end_ms - interval.start_ms) / 1000
                for interval in intervals
            ),
            default=0.0,
        )
        return AspectCandidateCapacity(
            aspect=aspect,
            geometry_status=status,
            safe_intervals=intervals,
            maximum_continuous_seconds=round(maximum, 3),
            requires_human_review=any(
                interval.requires_human_review for interval in intervals
            ),
        )

    horizontal_capacity = aspect_capacity(
        "16:9", horizontal_geometry_status, horizontal
    )
    vertical_capacity = aspect_capacity(
        "9:16", vertical_geometry_status, vertical
    )
    max_editorial = min(
        horizontal_capacity.maximum_continuous_seconds,
        vertical_capacity.maximum_continuous_seconds,
    )
    if min_editorial_duration > max_editorial:
        raise ValueError(
            "candidate quality-safe continuous capacity is shorter than the "
            f"minimum editorial duration: {max_editorial:.3f}s < "
            f"{min_editorial_duration:.3f}s"
        )
    return CandidateCapacity(
        candidate_id=candidate_id,
        source_asset_id=quality_map.source_asset_id,
        shot_id=quality_map.shot_id,
        horizontal=horizontal_capacity,
        vertical=vertical_capacity,
        min_editorial_duration=min_editorial_duration,
        preferred_duration=min(preferred_duration, max_editorial),
        max_editorial_duration=max_editorial,
        quality_map_path=str(resolved_path),
        quality_map_sha256=quality_sha256,
    )


def build_render_quality_report(
    render_path: Path,
    *,
    scdet_threshold: float = 4.0,
    output_dir: Path,
) -> dict[str, object]:
    """Run the same local detector per rendered shot and fail on hard defects."""

    from .shots import detect_shots_ffmpeg

    resolved_render = render_path.expanduser().resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = detect_shots_ffmpeg(
        resolved_render,
        threshold=scdet_threshold,
        output_path=output_dir / "shots.json",
    )
    map_rows: list[dict[str, object]] = []
    hard_blocks: list[dict[str, object]] = []
    review_windows: list[dict[str, object]] = []
    for shot in manifest.shots:
        quality_path = output_dir / f"{shot.shot_id}.quality.json"
        quality_map = scan_shot_quality(
            resolved_render,
            shot_manifest=manifest,
            shot_id=shot.shot_id,
            output_path=quality_path,
        )
        map_rows.append(
            {
                "shot_id": shot.shot_id,
                "path": str(quality_path.resolve()),
                "sha256": sha256_file(quality_path),
                "risk_count": len(quality_map.risk_windows),
            }
        )
        for window in quality_map.risk_windows:
            row = window.model_dump(mode="json")
            if window.severity == "hard_block":
                hard_blocks.append(row)
            elif window.severity in {"trim_candidate", "review"}:
                review_windows.append(row)
    report = {
        "contract_version": "render-quality-report-v1",
        "render_path": str(resolved_render),
        "render_sha256": sha256_file(resolved_render),
        "scanner_version": SHOT_QUALITY_SCANNER_VERSION,
        "shot_manifest_path": str((output_dir / "shots.json").resolve()),
        "shot_manifest_sha256": sha256_file(output_dir / "shots.json"),
        "shot_quality_maps": map_rows,
        "technical_qc_passed": not hard_blocks,
        "requires_human_review": bool(review_windows),
        "hard_block_windows": hard_blocks,
        "review_windows": review_windows,
        "interpretation": (
            "Hard decoder/PTS defects block delivery. Visual trim/review "
            "candidates remain evidence for editorial review and are not "
            "silently removed from the completed render."
        ),
        "generated_at": utc_now(),
    }
    write_json(output_dir / "render-quality-report.json", report)
    return report
