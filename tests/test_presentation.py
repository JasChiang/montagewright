from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image, ImageDraw

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
)
from jascue_video_lab.editing_capabilities import (
    autonomous_production_capability_catalog,
)
from jascue_video_lab.event_lock import (
    ExactEventLockV2,
    ExactEventResolverProvenance,
)
from jascue_video_lab.fixtures import HEIGHT, WIDTH, _encode
from jascue_video_lab.gemini import GeminiLabClient, MODEL_ID
from jascue_video_lab.media import probe_video, sha256_file
from jascue_video_lab.models import (
    SegmentationModelProvenance,
    SegmentationSample,
    SegmentationTrack,
    SemanticIdentityStatus,
    TrackingState,
)
from jascue_video_lab.presentation import (
    GroundingTargetRequest,
    PresentationTarget,
    SceneFacts,
    SourceCameraMotionEvidence,
    _vertical_intentional_freeze_filter,
    choose_two_panel_layout,
    compile_intentional_freeze,
    compile_minimal_camera_motion,
    compile_presentation,
    measure_source_camera_motion,
    shared_sam_seeds_from_grounding,
    static_full_bleed_crop_filter,
    two_panel_ffmpeg_filter,
)


def _policy() -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=85_000,
            min_ms=75_000,
            max_ms=90_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
    )


def _target(
    target_id: str,
    box: tuple[int, int, int, int],
    *,
    asset: str = "sha256:" + "a" * 64,
    pts: int = 30,
    track_boxes_by_pts: tuple[
        tuple[int, tuple[int, int, int, int]],
        ...,
    ] = (),
) -> PresentationTarget:
    return PresentationTarget(
        target_id=target_id,
        source_asset_id=asset,
        source_pts=pts,
        box_2d=box,
        track_boxes_by_pts=track_boxes_by_pts,
    )


def _event_lock(
    *,
    frame_hash: str = "b" * 64,
    event_type: str = "group_laugh_reaction_peak",
) -> ExactEventLockV2:
    return ExactEventLockV2(
        event_id="event-1",
        event_type=event_type,
        source_asset_id="sha256:" + "a" * 64,
        source_frame_id="DF000004",
        source_pts=120,
        source_time_ms=4_000,
        source_frame_hash=frame_hash,
        support_window_start_frame_id="DF000003",
        support_window_end_frame_id="DF000005",
        support_window_start_ms=3_900,
        support_window_end_ms=4_100,
        confidence=0.9,
        resolver=ExactEventResolverProvenance(
            local_bracket_method="frame_difference",
            sampling_fps=8,
            gemini_interaction_id="exact-1",
            contact_sheet_hashes=("c" * 64,),
        ),
        input_artifact_hashes=("sha256:" + "d" * 64,),
        generated_at="now",
    )


def _textured_background(width: int) -> Image.Image:
    image = Image.new("RGB", (width, HEIGHT), "#203048")
    draw = ImageDraw.Draw(image)
    randomizer = random.Random(20260729)
    for index in range(220):
        x = randomizer.randrange(5, width - 30)
        y = randomizer.randrange(5, HEIGHT - 30)
        size = randomizer.randrange(5, 20)
        color = (
            randomizer.randrange(80, 255),
            randomizer.randrange(80, 255),
            randomizer.randrange(80, 255),
        )
        if index % 2:
            draw.rectangle((x, y, x + size, y + size), fill=color)
        else:
            draw.ellipse((x, y, x + size, y + size), fill=color)
    return image


def _source_pan_video(path: Path) -> Path:
    background = _textured_background(WIDTH + 240)

    def render(time_seconds: float) -> Image.Image:
        camera_x = min(220, round(time_seconds * 90))
        return background.crop(
            (camera_x, 0, camera_x + WIDTH, HEIGHT)
        )

    _encode(path, 2.0, render)
    return path


def _source_head_jolt_video(path: Path) -> Path:
    background = _textured_background(WIDTH + 160)

    def render(time_seconds: float) -> Image.Image:
        if 0.10 <= time_seconds < 0.20:
            camera_x = 64
        elif 0.20 <= time_seconds < 0.30:
            camera_x = 20
        else:
            camera_x = 0
        return background.crop(
            (camera_x, 0, camera_x + WIDTH, HEIGHT)
        )

    _encode(path, 2.0, render)
    return path


def _source_tail_jolt_video(path: Path) -> Path:
    background = _textured_background(WIDTH + 160)

    def render(time_seconds: float) -> Image.Image:
        if 1.70 <= time_seconds < 1.80:
            camera_x = 64
        elif 1.80 <= time_seconds < 1.90:
            camera_x = 20
        else:
            camera_x = 0
        return background.crop(
            (camera_x, 0, camera_x + WIDTH, HEIGHT)
        )

    _encode(path, 2.0, render)
    return path


def _static_video_with_moving_foreground(path: Path) -> Path:
    background = _textured_background(WIDTH)

    def render(time_seconds: float) -> Image.Image:
        frame = background.copy()
        draw = ImageDraw.Draw(frame)
        x = round(430 + time_seconds * 140)
        draw.rectangle(
            (x, 150, x + 320, 620),
            fill="#f8f8f8",
            outline="#101010",
            width=12,
        )
        for offset in range(0, 300, 35):
            draw.line(
                (x + offset, 155, x + offset + 20, 615),
                fill="#101010",
                width=8,
            )
        return frame

    _encode(path, 2.0, render)
    return path


def _sam_subject_track(root: Path, source: Path) -> SegmentationTrack:
    media = probe_video(source)
    mask_path = root / "subject-mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(mask).rectangle(
        (350, 90, 900, 680),
        fill=255,
    )
    mask.save(mask_path)
    mask_hash = sha256_file(mask_path)
    box = [270, 120, 730, 950]
    samples = [
        SegmentationSample(
            sample_index=index,
            analysis_sample_time_ms=time_ms,
            source_pts=None,
            timing_basis="uniform_ffmpeg_analysis_sample",
            mask_path=str(mask_path),
            mask_sha256=mask_hash,
            mask_area_pixels=550 * 590,
            mask_area_ratio=0.35,
            connected_components=1,
            derived_tracking_box=box,
            center_2d=[500.0, 535.0],
            mean_positive_probability=0.95,
            scene_cut_score=None,
            shot_boundary=False,
            tracking_state=TrackingState.TRACKED,
            state_reasons=[],
            semantic_identity_status=(
                SemanticIdentityStatus.NOT_REVALIDATED
            ),
        )
        for index, time_ms in enumerate((0, 500, 1_000, 1_500, 1_900))
    ]
    return SegmentationTrack(
        method="bbox_seed_sam2_video_mask_propagation",
        asset_id=media.asset_id,
        video_path=str(source),
        target_description="moving foreground subject",
        seed_source="test",
        seed_time_ms=1_000,
        seed_sample_index=2,
        semantic_seed_box=box,
        seed_prompt_type="box",
        sam_prompt_box=box,
        seed_box_padding_ratio=0.0,
        refined_seed_mask_path=str(mask_path),
        analysis_fps=2.0,
        analysis_width=WIDTH,
        analysis_height=HEIGHT,
        analysis_start_ms=0,
        analysis_end_ms=2_000,
        timing_warning="synthetic test timing",
        semantic_warning="synthetic test identity",
        total_samples=len(samples),
        state_counts={TrackingState.TRACKED: len(samples)},
        elapsed_seconds=0.0,
        effective_fps=2.0,
        model_provenance=SegmentationModelProvenance(
            model_id="sam2.1_hiera_tiny",
            implementation="synthetic-test",
            implementation_revision="v1",
            checkpoint_sha256="a" * 64,
            device="cpu",
            torch_version="test",
            generated_at="2026-07-29T00:00:00Z",
        ),
        samples=samples,
    )


def test_static_subject_compiles_without_synthetic_motion() -> None:
    compilation = compile_presentation(
        targets=[_target("device", (450, 300, 550, 700))],
        source_width=1920,
        source_height=1080,
        relation_mode="single_subject",
        policy=_policy(),
    )

    assert compilation.mode == "static_full_bleed_crop"
    assert compilation.static_crop_box_2d is not None

    assert not any(
        "motion" in code for code in compilation.decision_codes
    )
    assert compilation.paid_model_calls_added == 0


def test_tracker_jitter_stays_inside_deadband_without_drift() -> None:
    decision = compile_minimal_camera_motion(
        (0.50, 0.505, 0.498, 0.502),
        movement_motivated=True,
    )

    assert decision.mode == "hold"
    assert decision.synthetic_reversal_count == 0
    assert decision.settle_required is False


def test_optimizable_center_left_right_avoids_initial_reversal() -> None:
    decision = compile_minimal_camera_motion(
        (0.50, 0.20, 0.80),
        movement_motivated=True,
        initial_position_optimizable=True,
    )

    assert decision.mode == "minimal_monotonic_move"
    assert decision.normalized_x_values == (0.20, 0.20, 0.80)
    assert decision.synthetic_reversal_count == 0
    assert decision.settle_required is True


def test_semantic_order_is_not_reordered_to_hide_a_reversal() -> None:
    decision = compile_minimal_camera_motion(
        (0.50, 0.20, 0.80),
        movement_motivated=True,
        initial_position_optimizable=False,
    )

    assert decision.mode == "hard_cut"
    assert decision.normalized_x_values == (0.50, 0.20, 0.80)
    assert decision.synthetic_reversal_count == 1


def test_existing_source_pan_suppresses_synthetic_camera_motion() -> None:
    decision = compile_minimal_camera_motion(
        (0.20, 0.80),
        source_camera_x_values=(0.15, 0.85),
        movement_motivated=True,
    )

    assert decision.mode == "hold"
    assert decision.source_motion.direction == "right"
    assert decision.movement_motivation == "none"


def test_source_camera_pan_is_measured_from_background_geometry(
    tmp_path: Path,
) -> None:
    source = _source_pan_video(tmp_path / "source-pan.mp4")
    media = probe_video(source)

    evidence = measure_source_camera_motion(
        source_path=source,
        source_asset_id=media.asset_id,
        window_start_ms=0,
        window_end_ms=2_000,
        subject_tracks=(),
        output_dir=tmp_path / "motion",
    )

    assert evidence.reliable is True
    assert evidence.classification == "pan_right"
    assert evidence.normalized_translation_x_per_second > 0.03
    assert evidence.confidence > 0.35
    assert len(evidence.sample_frame_hashes) >= 6
    assert (next((tmp_path / "motion").glob("*/evidence.json"))).exists()

    decision = compile_minimal_camera_motion(
        (0.2, 0.8),
        source_camera_motion=evidence.as_motion_estimate(),
        movement_motivated=True,
    )
    assert decision.mode == "hold"
    assert decision.source_motion.direction == "right"

    compilation = compile_presentation(
        targets=[
            _target("first", (430, 250, 500, 750)),
            _target("second", (500, 250, 570, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="sequential_focus",
        policy=_policy(),
        required_x_values=(0.465, 0.535),
        source_camera_motion_evidence=evidence,
        movement_motivated=True,
        preferred_capability_ids=("phase_virtual_camera",),
    )
    assert compilation.mode == "static_full_bleed_crop"
    assert compilation.static_crop_box_2d is not None
    unmotivated = compile_presentation(
        targets=[_target("subject", (430, 250, 570, 750))],
        source_width=1920,
        source_height=1080,
        relation_mode="single_subject",
        policy=_policy(),
        source_camera_motion_evidence=evidence,
        movement_motivated=False,
    )
    assert unmotivated.mode == "blocked"
    assert unmotivated.selection is not None
    assert any(
        "unmotivated_source_camera_motion" in failures
        for failures in unmotivated.selection.rejected_options.values()
    )
    filter_graph = static_full_bleed_crop_filter(
        compilation.static_crop_box_2d
    )
    assert "crop=w=" in filter_graph
    assert "scale=1080:1920" in filter_graph
    rendered = tmp_path / "static-full-bleed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_graph,
            "-map",
            "[base]",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(rendered),
        ],
        check=True,
    )
    rendered_media = probe_video(rendered)
    assert rendered_media.video.display_width == 1080
    assert rendered_media.video.display_height == 1920


def test_sam_subject_exclusion_prevents_foreground_motion_from_becoming_pan(
    tmp_path: Path,
) -> None:
    source = _static_video_with_moving_foreground(
        tmp_path / "moving-subject.mp4"
    )
    media = probe_video(source)
    track = _sam_subject_track(tmp_path / "track", source)

    evidence = measure_source_camera_motion(
        source_path=source,
        source_asset_id=media.asset_id,
        window_start_ms=0,
        window_end_ms=2_000,
        subject_tracks=(track,),
        output_dir=tmp_path / "motion",
    )

    assert evidence.reliable is True
    assert evidence.classification == "static"
    assert evidence.subject_exclusion_mode == "sam_track_boxes"
    assert evidence.mean_excluded_area_fraction > 0.25


def test_source_motion_reuses_and_supplements_sparse_sam_analysis_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _static_video_with_moving_foreground(
        tmp_path / "reuse-source.mp4"
    )
    media = probe_video(source)
    track_root = tmp_path / "reused-track"
    original_track = _sam_subject_track(track_root, source)
    track = original_track.model_copy(
        update={
            "seed_source": str(track_root / "seed-selection.json"),
            "samples": [
                sample.model_copy(
                    update={"source_pts": sample.sample_index}
                )
                for sample in original_track.samples
            ],
        }
    )
    frames_dir = track_root / "analysis-frames"
    frames_dir.mkdir(parents=True)
    background = _textured_background(WIDTH)
    for sample in track.samples:
        background.save(
            frames_dir / f"{sample.sample_index:06d}.jpg",
            quality=92,
        )

    from jascue_video_lab import presentation

    original_extract_frame = presentation.extract_frame
    decode_times: list[int] = []

    def record_decode(*args: Any, **kwargs: Any):
        decode_times.append(int(args[1]))
        return original_extract_frame(*args, **kwargs)

    monkeypatch.setattr(
        "jascue_video_lab.presentation.extract_frame",
        record_decode,
    )
    evidence = measure_source_camera_motion(
        source_path=source,
        source_asset_id=media.asset_id,
        window_start_ms=0,
        window_end_ms=2_000,
        subject_tracks=(track,),
        output_dir=tmp_path / "motion",
    )

    assert evidence.reliable is True
    assert evidence.classification == "static"
    assert decode_times
    assert len(decode_times) < len(evidence.sample_frame_pts)
    assert evidence.sampling_version == "hybrid-edge-dense-bounded-gap-v2"
    assert evidence.actual_max_sample_gap_ms is not None
    assert (
        evidence.actual_max_sample_gap_ms
        <= evidence.requested_max_sample_gap_ms + 100
    )


def test_edge_dense_sampling_preserves_short_opening_jolt(
    tmp_path: Path,
) -> None:
    source = _source_head_jolt_video(tmp_path / "head-jolt.mp4")
    media = probe_video(source)

    evidence = measure_source_camera_motion(
        source_path=source,
        source_asset_id=media.asset_id,
        window_start_ms=0,
        window_end_ms=2_000,
        subject_tracks=(),
        output_dir=tmp_path / "motion",
    )

    assert evidence.contract_version == "source-camera-motion-evidence-v2"
    assert evidence.estimator_version == "background-gftt-lk-ransac-affine-v2"
    assert evidence.sampling_version == "hybrid-edge-dense-bounded-gap-v2"
    assert evidence.head_sample_coverage_ms is not None
    assert evidence.head_sample_coverage_ms <= 100
    assert evidence.isolated_jolt_count >= 1
    assert evidence.dirty_head is True
    assert evidence.clean_head_start_ms is not None
    assert evidence.clean_head_start_ms >= 300
    assert evidence.max_translation_speed_per_second > (
        evidence.normalized_travel / 2.0
    )
    assert "isolated_source_camera_jolt_detected" in evidence.reason_codes
    assert any(pair.isolated_jolt for pair in evidence.pairs)
    compilation = compile_presentation(
        targets=[_target("device", (430, 250, 570, 750))],
        source_width=WIDTH,
        source_height=HEIGHT,
        relation_mode="single_subject",
        policy=_policy(),
        source_camera_motion_evidence=evidence,
    )
    assert compilation.mode == "blocked"
    assert compilation.selection is not None
    assert any(
        "unresolved_source_camera_jolt" in failures
        for failures in compilation.selection.rejected_options.values()
    )


def test_edge_dense_sampling_reports_clean_tail_before_closing_jolt(
    tmp_path: Path,
) -> None:
    source = _source_tail_jolt_video(tmp_path / "tail-jolt.mp4")
    media = probe_video(source)

    evidence = measure_source_camera_motion(
        source_path=source,
        source_asset_id=media.asset_id,
        window_start_ms=0,
        window_end_ms=2_000,
        subject_tracks=(),
        output_dir=tmp_path / "motion",
    )

    assert evidence.isolated_jolt_count >= 1
    assert evidence.dirty_tail is True
    assert evidence.clean_tail_end_ms is not None
    assert evidence.clean_tail_end_ms <= 1_700


def test_legacy_source_motion_artifact_remains_loadable() -> None:
    payload = {
        "contract_version": "source-camera-motion-evidence-v1",
        "estimator_version": "background-gftt-lk-ransac-affine-v1",
        "source_asset_id": "sha256:" + "a" * 64,
        "window_start_ms": 0,
        "window_end_ms": 2_000,
        "sample_times_ms": [0, 1_900],
        "sample_frame_pts": [0, 57],
        "sample_frame_hashes": ["b" * 64, "c" * 64],
        "subject_exclusion_mode": "none",
        "mean_excluded_area_fraction": 0.0,
        "pairs": [],
        "classification": "static",
        "reliable": True,
        "confidence": 0.8,
        "normalized_translation_x_per_second": 0.0,
        "normalized_translation_y_per_second": 0.0,
        "scale_rate_per_second": 0.0,
        "rotation_degrees_per_second": 0.0,
        "normalized_travel": 0.0,
        "reversal_count": 0,
        "reason_codes": ["legacy"],
        "cache_key_sha256": "d" * 64,
    }
    legacy = SourceCameraMotionEvidence.model_validate(payload)
    expected_sha = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert legacy.contract_version == "source-camera-motion-evidence-v1"
    assert legacy.sampling_version == "legacy-uniform-or-sam-v1"
    assert legacy.isolated_jolt_count == 0
    assert legacy.definition_sha256 == expected_sha


def test_unreliable_source_motion_forbids_virtual_pan_but_allows_hard_cut() -> None:
    evidence = SourceCameraMotionEvidence(
        source_asset_id="sha256:" + "a" * 64,
        window_start_ms=0,
        window_end_ms=2_000,
        sample_times_ms=(),
        sample_frame_pts=(),
        sample_frame_hashes=(),
        subject_exclusion_mode="none",
        mean_excluded_area_fraction=0.0,
        pairs=(),
        classification="unreliable",
        reliable=False,
        confidence=0.0,
        normalized_translation_x_per_second=0.0,
        normalized_translation_y_per_second=0.0,
        scale_rate_per_second=0.0,
        rotation_degrees_per_second=0.0,
        normalized_travel=0.0,
        reversal_count=0,
        reason_codes=("insufficient_background_features",),
        cache_key_sha256="b" * 64,
    )

    compilation = compile_presentation(
        targets=[
            _target("first", (100, 250, 300, 750)),
            _target("second", (700, 250, 900, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="sequential_focus",
        policy=_policy(),
        required_x_values=(0.2, 0.8),
        source_camera_motion_evidence=evidence,
        movement_motivated=True,
        preferred_capability_ids=("phase_virtual_camera",),
    )

    assert compilation.mode != "phase_virtual_camera"
    assert compilation.scene_facts is not None
    assert compilation.scene_facts.source_camera_motion_measured is True
    assert compilation.scene_facts.source_camera_motion_reliable is False


def test_impossible_two_device_crop_uses_scale_locked_two_panel() -> None:
    compilation = compile_presentation(
        targets=[
            _target("device-a", (20, 250, 270, 750)),
            _target("device-b", (730, 250, 980, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        physical_scale_comparison=True,
    )

    assert compilation.mode == "two_panel_layout"
    assert compilation.panel_layout is not None
    assert compilation.panel_layout.relative_scale_policy == "locked"
    first, second = compilation.panel_layout.panels
    assert first.source_pts == second.source_pts
    assert first.source_asset_id == second.source_asset_id
    assert first.crop_box_2d[2] - first.crop_box_2d[0] == (
        second.crop_box_2d[2] - second.crop_box_2d[0]
    )
    assert compilation.paid_model_calls_added == 0
    filter_graph = two_panel_ffmpeg_filter(compilation.panel_layout)
    assert "[0:v]split=2" in filter_graph
    assert filter_graph.count("[0:v]") == 1
    assert "force_original_aspect_ratio=decrease" not in filter_graph
    assert ",pad=" not in filter_graph


def test_nested_context_targets_do_not_create_panels_from_target_count() -> None:
    compilation = compile_presentation(
        targets=[
            _target("person", (380, 120, 620, 980)),
            _target("phone", (470, 420, 540, 610)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="context_detail",
        policy=_policy(),
        allow_static_full_bleed=False,
        panel_semantically_admissible=False,
    )

    assert compilation.mode != "two_panel_layout"
    assert compilation.scene_facts is not None
    assert compilation.scene_facts.nested_target_pairs == (
        ("person", "phone"),
    )
    assert compilation.paid_model_calls_added == 0


def test_explicit_context_detail_may_split_nested_targets() -> None:
    compilation = compile_presentation(
        targets=[
            _target("person", (380, 120, 620, 980)),
            _target("phone", (470, 420, 540, 610)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="context_detail",
        policy=_policy(),
        allow_static_full_bleed=False,
        tracking_available=False,
        acceptable_capability_ids=("two_panel_layout",),
        panel_semantically_admissible=True,
    )

    assert compilation.mode == "two_panel_layout"
    assert compilation.panel_layout is not None


def test_two_panels_use_semantic_groups_not_bbox_count() -> None:
    targets = [
        _target("person", (40, 100, 280, 900)),
        _target("held_phone", (170, 350, 250, 600)),
        _target("comparison_phone", (720, 250, 950, 750)),
    ]
    without_groups = compile_presentation(
        targets=targets,
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        allow_static_full_bleed=False,
        tracking_available=False,
        acceptable_capability_ids=("two_panel_layout",),
        panel_semantically_admissible=True,
    )
    with_groups = compile_presentation(
        targets=targets,
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        allow_static_full_bleed=False,
        tracking_available=False,
        acceptable_capability_ids=("two_panel_layout",),
        panel_semantically_admissible=True,
        panel_target_groups=(
            ("person", "held_phone"),
            ("comparison_phone",),
        ),
    )

    assert without_groups.mode == "blocked"
    assert with_groups.mode == "two_panel_layout"
    assert with_groups.panel_layout is not None
    assert with_groups.panel_layout.panels[0].target_ids == (
        "person",
        "held_phone",
    )
    assert with_groups.panel_layout.panels[1].target_ids == (
        "comparison_phone",
    )


def test_panel_group_cannot_hide_one_unreadable_required_target() -> None:
    compilation = compile_presentation(
        targets=[
            _target("person", (40, 100, 300, 900)),
            _target("tiny_ui", (180, 460, 190, 470)),
            _target("comparison_phone", (720, 250, 950, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        allow_static_full_bleed=False,
        tracking_available=False,
        acceptable_capability_ids=("two_panel_layout",),
        required_readability_by_target={
            "person": 0.5,
            "tiny_ui": 0.9,
            "comparison_phone": 0.5,
        },
        panel_semantically_admissible=True,
        panel_target_groups=(
            ("person", "tiny_ui"),
            ("comparison_phone",),
        ),
    )

    assert compilation.mode == "blocked"
    assert compilation.selection is not None
    rejected_codes = {
        code
        for reasons in compilation.selection.rejected_options.values()
        for code in reasons
    }
    assert "minimum_readability_failed" in rejected_codes


def test_common_motion_and_tracked_relation_are_measured_per_pts() -> None:
    shared_samples_a = (
        (0, (100, 200, 300, 800)),
        (30, (120, 200, 320, 800)),
        (60, (140, 200, 340, 800)),
    )
    shared_samples_b = (
        (0, (280, 300, 380, 700)),
        (30, (300, 300, 400, 700)),
        (60, (320, 300, 420, 700)),
    )
    compilation = compile_presentation(
        targets=[
            _target(
                "context",
                (100, 200, 340, 800),
                track_boxes_by_pts=shared_samples_a,
            ),
            _target(
                "detail",
                (280, 300, 420, 700),
                track_boxes_by_pts=shared_samples_b,
            ),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        allow_static_full_bleed=False,
        acceptable_capability_ids=("tracked_full_bleed_crop",),
    )

    assert compilation.mode == "tracked_full_bleed_crop"
    assert compilation.scene_facts is not None
    assert compilation.scene_facts.shared_tracked_crop_feasible is True
    assert compilation.scene_facts.common_motion_pairs == (
        ("context", "detail"),
    )
    assert (
        compilation.scene_facts.aligned_track_sample_count_matrix[0][1]
        == 3
    )

    panel = compile_presentation(
        targets=[
            _target(
                "context",
                (100, 200, 340, 800),
                track_boxes_by_pts=shared_samples_a,
            ),
            _target(
                "detail",
                (280, 300, 420, 700),
                track_boxes_by_pts=shared_samples_b,
            ),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        physical_scale_comparison=True,
        allow_static_full_bleed=False,
        tracking_available=False,
        acceptable_capability_ids=("two_panel_layout",),
        panel_semantically_admissible=True,
        panel_target_groups=(("context",), ("detail",)),
    )

    assert panel.mode == "two_panel_layout"
    assert panel.panel_layout is not None
    assert panel.panel_layout.temporal_relation == "same_source_same_pts"
    assert panel.panel_layout.relative_scale_policy == "locked"


def test_scene_facts_v1_artifact_remains_loadable() -> None:
    compilation = compile_presentation(
        targets=[_target("subject", (300, 100, 700, 900))],
        source_width=1920,
        source_height=1080,
        relation_mode="single_subject",
        policy=_policy(),
    )
    assert compilation.scene_facts is not None
    payload = compilation.scene_facts.model_dump(mode="json")
    payload["contract_version"] = "presentation-scene-facts-v1"
    for field_name in (
        "target_center_y",
        "target_height_fractions",
        "normalized_xy_distance_matrix",
        "intersection_over_union_matrix",
        "containment_fraction_matrix",
        "nested_target_pairs",
        "aligned_track_sample_count_matrix",
        "co_visibility_fraction_matrix",
        "common_motion_residual_matrix",
        "common_motion_pairs",
        "shared_tracked_crop_feasible",
        "target_short_edge_pixels_by_mode",
    ):
        payload.pop(field_name)

    expected_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    restored = SceneFacts.model_validate(payload)
    assert restored.contract_version == "presentation-scene-facts-v1"
    assert restored.definition_sha256 == expected_sha256


def test_intentional_freeze_composes_after_selected_presentation() -> None:
    base = (
        "[0:v]crop=720:1080:100:0,"
        "scale=1080:1920,setsar=1[base]"
    )
    graph = _vertical_intentional_freeze_filter(
        base,
        freeze_start_seconds=2.0,
        total_duration_seconds=3.0,
    )

    assert "crop=720:1080:100:0" in graph
    assert "[presented]trim=end=2.000000" in graph
    assert "force_original_aspect_ratio=decrease" not in graph


def test_different_sources_cannot_claim_physical_simultaneity() -> None:
    result = choose_two_panel_layout(
        _target("a", (100, 200, 300, 800), asset="sha256:" + "a" * 64),
        _target("b", (700, 200, 900, 800), asset="sha256:" + "b" * 64),
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        physical_scale_comparison=True,
        allow_conceptual_different_source=True,
        allowed_modes=("top_bottom", "side_by_side"),
    )

    assert result is None


def test_landscape_two_panel_uses_fill_crops_without_internal_letterbox() -> None:
    result = choose_two_panel_layout(
        _target("a", (100, 300, 400, 700)),
        _target("b", (600, 300, 900, 700)),
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        physical_scale_comparison=True,
        allow_conceptual_different_source=False,
        allowed_modes=("top_bottom", "side_by_side"),
    )

    assert result is not None
    assert result.layout_mode == "top_bottom"
    for panel in result.panels:
        crop_width = panel.crop_box_2d[2] - panel.crop_box_2d[0]
        crop_height = panel.crop_box_2d[3] - panel.crop_box_2d[1]
        crop_pixel_aspect = (crop_width * 1920) / (crop_height * 1080)
        panel_pixel_aspect = (
            (panel.output_rect.width * 1080)
            / (panel.output_rect.height * 1920)
        )
        assert crop_pixel_aspect == pytest.approx(
            panel_pixel_aspect,
            abs=0.01,
        )
    filter_graph = two_panel_ffmpeg_filter(result)
    assert "force_original_aspect_ratio" not in filter_graph
    assert "pad=" not in filter_graph


def test_three_person_group_never_creates_three_panels() -> None:
    compilation = compile_presentation(
        targets=[
            _target("first", (10, 200, 210, 800)),
            _target("second", (400, 200, 600, 800)),
            _target("third", (790, 200, 990, 800)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
    )

    assert compilation.mode == "solid_matte_fit"
    assert compilation.panel_layout is None


def test_intentional_freeze_requires_exact_event_and_policy_duration() -> None:
    freeze = compile_intentional_freeze(
        _event_lock(),
        cue_id="locked-cue-00012",
        duration_ms=1_000,
        policy=_policy(),
    )

    assert freeze.source_pts == 120
    assert freeze.motivation == "brief_authorized_phrase_ending"


def test_autonomous_catalog_adds_only_policy_gated_presentations() -> None:
    catalog = autonomous_production_capability_catalog()
    ids = {item.capability_id for item in catalog.capabilities}

    assert {
        "two_panel_layout",
        "solid_matte_fit",
        "intentional_freeze",
    } <= ids
    assert catalog.prohibited_automatic_delivery == [
        "blurred_background"
    ]


def test_multi_target_grounding_is_one_call_and_builds_one_sam_session(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "exact.png"
    Image.new("RGB", (640, 360), color=(20, 30, 40)).save(frame)
    frame_hash = hashlib.sha256(frame.read_bytes()).hexdigest()
    lock = _event_lock(frame_hash=frame_hash)
    target_requests = (
        GroundingTargetRequest(
            target_id="device-a",
            target_description="left device",
        ),
        GroundingTargetRequest(
            target_id="device-b",
            target_description="right device",
        ),
    )
    output = json.dumps(
        {
            "source_asset_id": lock.source_asset_id,
            "event_lock_id": lock.event_id,
            "source_frame_id": lock.source_frame_id,
            "source_frame_hash": frame_hash,
            "source_width": 640,
            "source_height": 360,
            "targets": [
                {
                    "target_id": "device-a",
                    "visible": True,
                    "candidates": [
                        {
                            "box_2d_yxyx": [100, 50, 900, 400],
                            "confidence": 0.9,
                            "disambiguation_reason": "left instance",
                        }
                    ],
                },
                {
                    "target_id": "device-b",
                    "visible": True,
                    "candidates": [
                        {
                            "box_2d_yxyx": [100, 600, 900, 950],
                            "confidence": 0.9,
                            "disambiguation_reason": "right instance",
                        }
                    ],
                },
            ],
        }
    )
    requests: list[dict[str, Any]] = []

    class Interaction:
        id = "ground-1"
        output_text = output

        def model_dump(
            self,
            *,
            mode: str,
            exclude_none: bool,
        ) -> dict[str, object]:
            return {
                "id": self.id,
                "model": MODEL_ID,
                "output_text": self.output_text,
                "usage": {},
            }

    def create(**request: Any) -> Interaction:
        requests.append(request)
        return Interaction()

    client = object.__new__(GeminiLabClient)
    client.model_id = MODEL_ID
    client.client = SimpleNamespace(
        interactions=SimpleNamespace(create=create)
    )

    group = client.ground_multi_target_exact_frame(
        event_lock=lock,
        frame_path=frame,
        targets=target_requests,
        output_dir=tmp_path / "grounding",
    )
    seeds = shared_sam_seeds_from_grounding(
        group,
        target_requests=target_requests,
        seed_time_ms=4_000,
        seed_frame_pts=120,
    )

    assert len(requests) == 1
    assert requests[0]["input"][1]["media_resolution"] == "high"
    assert len(seeds) == 2
    assert {seed.target_id for seed in seeds} == {
        "device-a",
        "device-b",
    }
