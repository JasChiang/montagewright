from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.media import sha256_file
from jascue_video_lab.music import (
    CuePriority,
    LockedMusicCue,
    MusicMapLock,
    MusicMapReview,
    MusicSectionCandidate,
)
from jascue_video_lab.music_assembly import (
    MusicAssemblyError,
    load_music_assembly_artifacts,
    plan_single_interval_music_assembly,
    render_single_interval_music_assembly,
    write_music_assembly_artifacts,
)
from jascue_video_lab.models import MusicAssemblyPlan, MusicAssemblyRenderManifest
from jascue_video_lab.storage import read_json, write_json


SAMPLE_RATE = 48_000


def _music_lock(
    *,
    duration_seconds: float = 41,
    music_id: str | None = None,
    section_start_sample: int | None = None,
) -> MusicMapLock:
    duration_samples = round(SAMPLE_RATE * duration_seconds)
    first_downbeat = SAMPLE_RATE
    beat_period = SAMPLE_RATE // 2
    cues: list[LockedMusicCue] = []
    for index, sample in enumerate(
        range(first_downbeat, duration_samples, beat_period),
        start=1,
    ):
        beat_index = index - 1
        cues.append(
            LockedMusicCue(
                cue_id=f"locked-cue-{index:05d}",
                kind="downbeat" if beat_index % 4 == 0 else "beat",
                sample_index=sample,
                time_ms=round(sample * 1000 / SAMPLE_RATE),
                strength=0.9 if beat_index % 4 == 0 else 0.6,
                priority=(
                    CuePriority.PREFERRED
                    if beat_index % 4 == 0
                    else CuePriority.OPTIONAL
                ),
            )
        )
    proposal_sha256 = "a" * 64
    review = MusicMapReview(
        proposal_sha256=proposal_sha256,
        reviewer="test-editor",
        reviewed_at="2026-07-23T00:00:00+00:00",
        decision="approved",
        bpm=120.0,
        first_downbeat_sample=first_downbeat,
        meter=4,
    )
    if section_start_sample is None:
        sections = [
            MusicSectionCandidate(
                section_id="section-001",
                start_sample=0,
                end_sample=duration_samples,
                label="section_001",
                boundary_source="whole_track",
                confidence=0.5,
            )
        ]
    else:
        sections = [
            MusicSectionCandidate(
                section_id="section-001",
                start_sample=0,
                end_sample=section_start_sample,
                label="section_001",
                boundary_source="energy_change",
                confidence=0.8,
            ),
            MusicSectionCandidate(
                section_id="section-002",
                start_sample=section_start_sample,
                end_sample=duration_samples,
                label="section_002",
                boundary_source="energy_change",
                confidence=0.9,
            ),
        ]
    return MusicMapLock(
        music_id=music_id or f"sha256:{'b' * 64}",
        proposal_path="/not/needed/by/this/contract.json",
        proposal_sha256=proposal_sha256,
        review=review,
        master_sample_rate=SAMPLE_RATE,
        duration_samples=duration_samples,
        duration_ms=round(duration_seconds * 1000),
        bpm=120.0,
        meter=4,
        first_downbeat_sample=first_downbeat,
        cues=cues,
        sections=sections,
        definition_sha256="c" * 64,
    )


def _saved_lock(
    tmp_path: Path,
    *,
    duration_seconds: float = 41,
    music_id: str | None = None,
    section_start_sample: int | None = None,
) -> tuple[MusicMapLock, Path]:
    music_lock = _music_lock(
        duration_seconds=duration_seconds,
        music_id=music_id,
        section_start_sample=section_start_sample,
    )
    lock_path = tmp_path / "music-map.lock.json"
    write_json(lock_path, music_lock)
    return music_lock, lock_path


def _plan(tmp_path: Path) -> MusicAssemblyPlan:
    music_lock, lock_path = _saved_lock(tmp_path)
    return plan_single_interval_music_assembly(
        music_lock,
        music_lock_path=lock_path,
        target_duration_ms=19_000,
        minimum_duration_ms=14_000,
        maximum_duration_ms=20_000,
        preferred_phrase_bars=(8,),
    )


def _write_tone(path: Path, *, duration_seconds: float) -> int:
    frame_count = round(SAMPLE_RATE * duration_seconds)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    round(
                        12_000
                        * math.cos(2 * math.pi * 437 * index / SAMPLE_RATE)
                    ),
                )
                for index in range(frame_count)
            )
        )
    return frame_count


def test_planner_selects_one_phrase_aligned_continuous_interval(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    assert plan.assembly_mode == "single_continuous_interval"
    assert plan.join_count == 0
    assert len(plan.spans) == 1
    span = plan.spans[0]
    assert span.source_start_sample == SAMPLE_RATE
    assert span.source_end_sample == SAMPLE_RATE * 17
    assert span.output_start_sample == 0
    assert span.output_end_sample == SAMPLE_RATE * 16
    assert span.start_boundary_kind == "phrase_grid"
    assert span.end_boundary_kind == "phrase_grid"
    assert plan.ending_policy == "short_fade_at_phrase_grid_boundary"
    assert span.bar_count == 8
    assert span.phrase_bar_multiple == 8
    assert plan.output_duration_samples == SAMPLE_RATE * 16
    assert plan.target_duration_error_samples == SAMPLE_RATE * 3

    assert plan.cue_instances[0].source_sample_index == SAMPLE_RATE
    assert plan.cue_instances[0].output_sample_index == 0
    assert all(
        cue.output_sample_index
        == cue.source_sample_index - span.source_start_sample
        for cue in plan.cue_instances
    )


def test_planner_prefers_reviewed_section_through_natural_track_end(
    tmp_path: Path,
) -> None:
    section_start = round(SAMPLE_RATE * 2.2)
    music_lock, lock_path = _saved_lock(
        tmp_path,
        duration_seconds=6.2,
        section_start_sample=section_start,
    )

    plan = plan_single_interval_music_assembly(
        music_lock,
        music_lock_path=lock_path,
        target_duration_ms=4_000,
        minimum_duration_ms=3_500,
        maximum_duration_ms=4_500,
        preferred_phrase_bars=(2, 1),
    )

    span = plan.spans[0]
    assert span.source_start_sample == section_start
    assert span.source_end_sample == music_lock.duration_samples
    assert span.start_boundary_kind == "section_boundary"
    assert span.end_boundary_kind == "natural_track_end"
    assert plan.ending_policy == "preserve_natural_track_end_no_fade_out"
    assert span.start_bar_index is None
    assert span.end_bar_index is None
    assert span.bar_count is None
    assert span.phrase_bar_multiple is None
    assert span.end_boundary_cue_id is None


def test_planner_keeps_a_fractional_natural_tail_after_the_last_full_bar(
    tmp_path: Path,
) -> None:
    section_start = round(SAMPLE_RATE * 80.290)
    music_lock, lock_path = _saved_lock(
        tmp_path,
        duration_seconds=154.828,
        section_start_sample=section_start,
    )

    plan = plan_single_interval_music_assembly(
        music_lock,
        music_lock_path=lock_path,
        target_duration_ms=75_000,
        minimum_duration_ms=60_000,
        maximum_duration_ms=90_000,
    )

    span = plan.spans[0]
    assert span.source_start_sample == section_start
    assert span.source_end_sample == round(SAMPLE_RATE * 154.828)
    assert span.output_end_sample == round(SAMPLE_RATE * 74.538)
    assert span.start_boundary_kind == "section_boundary"
    assert span.end_boundary_kind == "natural_track_end"
    assert span.end_bar_index is None


def test_plan_contract_rejects_a_second_music_span(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    payload = plan.model_dump(mode="json")
    second = payload["spans"][0].copy()
    second["span_id"] = "music-span-002"
    payload["spans"].append(second)

    with pytest.raises(ValidationError, match="at most 1 item"):
        MusicAssemblyPlan.model_validate(payload)


def test_planner_fails_closed_when_no_continuous_interval_fits(
    tmp_path: Path,
) -> None:
    music_lock, lock_path = _saved_lock(tmp_path, duration_seconds=10)

    with pytest.raises(MusicAssemblyError, match="no single reviewed-boundary"):
        plan_single_interval_music_assembly(
            music_lock,
            music_lock_path=lock_path,
            target_duration_ms=17_000,
            minimum_duration_ms=14_000,
            maximum_duration_ms=20_000,
        )


def test_music_assembly_artifacts_round_trip_and_bind_the_lock(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    output_dir = tmp_path / "assembly"
    paths = write_music_assembly_artifacts(plan, output_dir=output_dir)

    assert paths.plan_path.exists()
    assert paths.binding_path.exists()
    loaded_plan, binding = load_music_assembly_artifacts(output_dir)
    assert loaded_plan == plan
    assert binding.assembly_id == plan.assembly_id
    assert binding.music_lock_sha256 == plan.music_lock_sha256

    different = MusicAssemblyPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "generated_at": "2026-07-23T01:00:00+00:00",
        }
    )
    with pytest.raises(FileExistsError, match="different music assembly plan"):
        write_music_assembly_artifacts(different, output_dir=output_dir)


def test_loading_artifacts_detects_a_changed_music_lock(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output_dir = tmp_path / "assembly"
    write_music_assembly_artifacts(plan, output_dir=output_dir)

    lock_path = Path(plan.music_lock_path)
    changed = read_json(lock_path)
    changed["definition_sha256"] = "d" * 64
    write_json(lock_path, changed)

    with pytest.raises(MusicAssemblyError, match="hash no longer matches"):
        load_music_assembly_artifacts(output_dir)


def test_render_single_interval_is_48k_stereo_and_preserves_natural_end(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-tone.wav"
    _write_tone(source, duration_seconds=6.2)
    source_id = f"sha256:{sha256_file(source)}"
    section_start = round(SAMPLE_RATE * 2.2)
    music_lock, lock_path = _saved_lock(
        tmp_path,
        duration_seconds=6.2,
        music_id=source_id,
        section_start_sample=section_start,
    )
    plan = plan_single_interval_music_assembly(
        music_lock,
        music_lock_path=lock_path,
        target_duration_ms=4_000,
        minimum_duration_ms=3_500,
        maximum_duration_ms=4_500,
        preferred_phrase_bars=(2, 1),
    )
    output = tmp_path / "rendered" / "music.wav"
    result = render_single_interval_music_assembly(
        source,
        plan,
        output,
        tmp_path / "artifacts",
    )

    assert result.output_audio_path == output.resolve()
    assert result.manifest.qc_passed is True
    assert result.manifest.output_sample_rate == SAMPLE_RATE
    assert result.manifest.output_channels == 2
    assert result.manifest.expected_output_samples == SAMPLE_RATE * 4
    assert result.manifest.probed_output_samples == SAMPLE_RATE * 4
    assert result.manifest.duration_delta_samples == 0
    assert result.manifest.fade_in_samples == 480
    assert result.manifest.fade_out_samples == 0
    assert (
        result.manifest.ending_policy
        == "preserve_natural_track_end_no_fade_out"
    )
    assert result.manifest.internal_join_count == 0
    assert result.manifest.natural_track_end_preserved is True
    assert "concat" not in result.manifest.ffmpeg_filter_graph
    saved_manifest = MusicAssemblyRenderManifest.model_validate(
        read_json(result.manifest_path)
    )
    assert saved_manifest == result.manifest
    assert saved_manifest.output_audio_sha256 == sha256_file(output)

    with wave.open(str(output), "rb") as output_wave:
        assert output_wave.getframerate() == SAMPLE_RATE
        assert output_wave.getnchannels() == 2
        assert output_wave.getnframes() == SAMPLE_RATE * 4
        first_left, first_right = struct.unpack("<hh", output_wave.readframes(1))
        assert first_left == first_right == 0
        output_wave.setpos(output_wave.getnframes() - 1)
        last_left, last_right = struct.unpack("<hh", output_wave.readframes(1))
    assert last_left == last_right
    assert abs(last_left) > 1_000


def test_render_phrase_grid_ending_applies_short_fade_and_ends_near_zero(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-tone.wav"
    _write_tone(source, duration_seconds=41)
    source_id = f"sha256:{sha256_file(source)}"
    music_lock, lock_path = _saved_lock(
        tmp_path,
        music_id=source_id,
    )
    plan = plan_single_interval_music_assembly(
        music_lock,
        music_lock_path=lock_path,
        target_duration_ms=19_000,
        minimum_duration_ms=14_000,
        maximum_duration_ms=20_000,
        preferred_phrase_bars=(8,),
    )
    output = tmp_path / "rendered" / "music.wav"
    result = render_single_interval_music_assembly(
        source,
        plan,
        output,
        tmp_path / "artifacts",
        phrase_grid_fade_out_ms=10,
    )

    assert plan.spans[0].end_boundary_kind == "phrase_grid"
    assert result.manifest.fade_out_samples == 480
    assert (
        result.manifest.ending_policy
        == "short_fade_at_phrase_grid_boundary"
    )
    assert result.manifest.natural_track_end_preserved is False
    assert "afade=t=out" in result.manifest.ffmpeg_filter_graph
    with wave.open(str(output), "rb") as output_wave:
        output_wave.setpos(output_wave.getnframes() - 1)
        last_left, last_right = struct.unpack("<hh", output_wave.readframes(1))
    assert last_left == last_right
    assert abs(last_left) < 100
    invalid_manifest = result.manifest.model_dump(mode="json")
    invalid_manifest["fade_out_samples"] = 0
    with pytest.raises(ValidationError, match="non-zero fade-out"):
        MusicAssemblyRenderManifest.model_validate(invalid_manifest)


def test_render_refuses_source_not_bound_to_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = tmp_path / "unrelated.wav"
    _write_tone(source, duration_seconds=41)

    with pytest.raises(MusicAssemblyError, match="source audio hash"):
        render_single_interval_music_assembly(
            source,
            plan,
            tmp_path / "out.wav",
            tmp_path / "render-artifacts",
        )
