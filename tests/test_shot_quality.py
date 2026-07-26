from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    QualityFrameEvidence,
    QualityRiskWindow,
    Rational,
    ShotQualityMap,
)
from jascue_video_lab.shot_quality import (
    _decode_analysis_frames,
    build_candidate_capacity,
    build_quality_safe_intervals,
    scan_shot_quality,
)
from jascue_video_lab.shots import detect_shots_ffmpeg
from jascue_video_lab.storage import write_json


def _render_three_part_fixture(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1:size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "color=black:duration=0.6:size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1:size=320x180:rate=30",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            str(path),
        ],
        check=True,
    )


def _render_locked_fixture(path: Path, *, temporal_noise: bool) -> None:
    source = "color=gray:duration=1:size=320x180:rate=30"
    if temporal_noise:
        source += ",noise=alls=2:allf=t+u"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _quality_map(tmp_path: Path) -> tuple[Path, ShotQualityMap]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    source_sha = sha256_file(source)
    evidence = [
        QualityFrameEvidence(
            frame_id="QF-" + character * 16,
            frame_pts=pts,
            frame_time_ms=time_ms,
            analysis_frame_sha256=character * 64,
        )
        for character, pts, time_ms in (
            ("a", 30, 1000),
            ("b", 60, 2000),
            ("c", 180, 6000),
        )
    ]
    quality_map = ShotQualityMap(
        scanner_version="test-v1",
        source_path=str(source),
        source_asset_id=f"sha256:{source_sha}",
        shot_id="shot-0001",
        shot_start_pts=0,
        shot_end_pts=300,
        shot_start_ms=0,
        shot_end_ms=10_000,
        source_time_base=Rational(numerator=1, denominator=30),
        analysis_width=320,
        analysis_height=180,
        decoded_frame_count=300,
        request_sha256="d" * 64,
        evidence_frames=evidence,
        risk_windows=[
            QualityRiskWindow(
                risk_window_id="QRW-0001",
                source_asset_id=f"sha256:{source_sha}",
                shot_id="shot-0001",
                start_pts=30,
                end_pts=60,
                start_ms=1000,
                end_ms=2000,
                reason_code="black",
                severity="trim_candidate",
                intent="unknown",
                confidence=0.99,
                evidence_frame_ids=[evidence[0].frame_id, evidence[1].frame_id],
                metric_summary={"black_fraction_mean": 1.0},
            ),
            QualityRiskWindow(
                risk_window_id="QRW-0002",
                source_asset_id=f"sha256:{source_sha}",
                shot_id="shot-0001",
                start_pts=150,
                end_pts=210,
                start_ms=5000,
                end_ms=7000,
                reason_code="focus_loss",
                severity="review",
                intent="unknown",
                confidence=0.7,
                evidence_frame_ids=[evidence[2].frame_id],
                metric_summary={"focus_score_mean": 0.001},
            ),
        ],
        warnings=["test"],
        generated_at="2026-07-26T00:00:00Z",
    )
    path = tmp_path / "quality.json"
    write_json(path, quality_map)
    return path, quality_map


def test_source_fps_scan_detects_black_and_freeze_without_editing(
    tmp_path: Path,
) -> None:
    video = tmp_path / "black.mp4"
    _render_three_part_fixture(video)
    shots = detect_shots_ffmpeg(video, threshold=100)
    assert len(shots.shots) == 1

    result = scan_shot_quality(
        video,
        shot_manifest=shots,
        shot_id="shot-0001",
    )

    reasons = {window.reason_code for window in result.risk_windows}
    assert "black" in reasons
    assert "freeze" in reasons
    black = next(
        window for window in result.risk_windows if window.reason_code == "black"
    )
    assert black.severity == "trim_candidate"
    assert black.intent == "unknown"
    assert black.evidence_frame_ids
    assert result.decoded_frame_count >= 70


def test_analysis_decode_emits_only_shortlisted_frame_interval(
    tmp_path: Path,
) -> None:
    video = tmp_path / "three-parts.mp4"
    _render_three_part_fixture(video)
    frames, warning = _decode_analysis_frames(
        video,
        analysis_width=160,
        analysis_height=90,
        first_frame_index=30,
        last_frame_index=47,
        expected_count=18,
    )
    assert len(frames) == 18
    assert warning == ""


def test_locked_camera_with_sensor_noise_is_not_exact_freeze(
    tmp_path: Path,
) -> None:
    exact = tmp_path / "exact.mp4"
    noisy = tmp_path / "noisy.mp4"
    _render_locked_fixture(exact, temporal_noise=False)
    _render_locked_fixture(noisy, temporal_noise=True)

    exact_result = scan_shot_quality(
        exact,
        shot_manifest=detect_shots_ffmpeg(exact, threshold=100),
        shot_id="shot-0001",
    )
    noisy_result = scan_shot_quality(
        noisy,
        shot_manifest=detect_shots_ffmpeg(noisy, threshold=100),
        shot_id="shot-0001",
    )

    assert any(
        window.reason_code == "freeze" for window in exact_result.risk_windows
    )
    assert not any(
        window.reason_code == "freeze" for window in noisy_result.risk_windows
    )


def test_quality_safe_intervals_exclude_unknown_trim_candidate_only(
    tmp_path: Path,
) -> None:
    path, quality_map = _quality_map(tmp_path)
    intervals = build_quality_safe_intervals(
        quality_map,
        quality_map_sha256=sha256_file(path),
    )

    assert [(item.start_ms, item.end_ms) for item in intervals] == [
        (0, 1000),
        (2000, 10_000),
    ]
    assert intervals[0].requires_human_review is False
    assert intervals[1].requires_human_review is True
    assert intervals[1].review_risk_window_ids == ["QRW-0002"]


def test_intentional_trim_candidate_is_retained(tmp_path: Path) -> None:
    path, quality_map = _quality_map(tmp_path)
    reviewed = quality_map.model_copy(
        update={
            "risk_windows": [
                quality_map.risk_windows[0].model_copy(
                    update={"intent": "intentional"}
                ),
                quality_map.risk_windows[1],
            ]
        }
    )
    intervals = build_quality_safe_intervals(
        reviewed,
        quality_map_sha256=sha256_file(path),
    )
    assert [(item.start_ms, item.end_ms) for item in intervals] == [(0, 10_000)]


def test_candidate_capacity_uses_longest_continuous_interval(
    tmp_path: Path,
) -> None:
    path, _ = _quality_map(tmp_path)
    capacity = build_candidate_capacity(
        candidate_id="candidate-a",
        quality_map_path=path,
        preferred_duration=9.0,
        min_editorial_duration=2.0,
    )

    assert capacity.horizontal.maximum_continuous_seconds == 8.0
    assert capacity.vertical.maximum_continuous_seconds == 8.0
    assert capacity.max_editorial_duration == 8.0
    assert capacity.preferred_duration == 8.0


def test_candidate_capacity_fails_when_minimum_exceeds_safe_interval(
    tmp_path: Path,
) -> None:
    path, _ = _quality_map(tmp_path)
    with pytest.raises(ValueError, match="shorter than the minimum"):
        build_candidate_capacity(
            candidate_id="candidate-a",
            quality_map_path=path,
            preferred_duration=9.0,
            min_editorial_duration=8.5,
        )


def test_geometry_blocked_aspect_exposes_no_executable_capacity(
    tmp_path: Path,
) -> None:
    path, _ = _quality_map(tmp_path)
    capacity = build_candidate_capacity(
        candidate_id="candidate-a",
        quality_map_path=path,
        preferred_duration=7.0,
        horizontal_geometry_status="blocked",
    )
    assert capacity.horizontal.safe_intervals == []
    assert capacity.horizontal.maximum_continuous_seconds == 0
    assert capacity.max_editorial_duration == 0


def test_quality_map_rejects_window_outside_shot(tmp_path: Path) -> None:
    _, quality_map = _quality_map(tmp_path)
    invalid = quality_map.model_dump(mode="json")
    invalid["risk_windows"][0]["end_ms"] = 11_000
    with pytest.raises(ValidationError, match="outside its shot"):
        ShotQualityMap.model_validate(invalid)
