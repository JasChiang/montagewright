from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
import jascue_video_lab.feature_cut as feature_cut_module

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
)
from jascue_video_lab.editing_capabilities import (
    simple_production_capability_catalog,
)
from scripts.plan_clip_card_open_edit import (
    OpenEditCandidate,
    OpenEditPlan,
    OpenEditShot,
    _assert_fresh_open_edit_namespace_empty,
    _verified_open_edit_raw_output_text,
    _write_open_edit_normalization_artifacts,
    canonicalize_open_edit_output,
    project_feature_contracts,
)

from jascue_video_lab.feature_cut import (
    _chapter_bounds_with_approved_trim,
    _audit_feature_plan_candidate_recall,
    _audit_requested_candidate_recall,
    _build_feature_cut_eligibility_report,
    _audit_render_source_reuse,
    _autonomous_panel_fallback_eligible,
    _runtime_candidate_reuse_violation,
    _candidate_asset_reference_matches,
    _feature_vertical_candidate_from_runtime_option,
    _cover_transform,
    _current_feature_plan_binding,
    _current_external_projection_binding,
    _has_complete_cached_primary_track,
    _horizontal_filter_from_track,
    _is_exhausted_model_quota_error,
    _is_non_retryable_spending_cap_error,
    _load_trim_decisions,
    _migrate_legacy_feature_plan_binding,
    _piecewise_expression,
    _project_locked_cues_to_music_output,
    _production_review_preflight_failures,
    _prompt_binds_sha256,
    _concat_segments,
    _controlled_primary_center_preview_allowed,
    _render_source_segment,
    _render_text_layer,
    _requested_render_aspects,
    _required_track_union,
    _resolve_editorial_chapter_durations,
    _resolve_vertical_camera_phases,
    _resolve_vertical_candidate_intent,
    _refine_selected_vertical_candidate,
    _segment_variant_fingerprint,
    _selected_source_capacity_seconds,
    _should_refine_selected_vertical_candidate,
    _soft_extent_visibility_audit,
    _summarize_automatic_reframe,
    _tracking_seed_request_ms,
    _tracked_crop_geometry,
    _usable_track_centers,
    _validate_autonomous_plan_reuse_flags,
    _validate_selected_framing_coverage_invariant,
    _validate_feature_plan_binding,
    _validate_shared_sam_session_cache,
    _vertical_crop_geometry,
    _vertical_center_crop_filter,
    _vertical_delivery_fallback,
    _vertical_filter_from_track,
    _vertical_fit_filter,
    _vertical_virtual_camera_filter_from_tracks,
    _vertical_required_scope_fit_filter,
    _vertical_runtime_candidate_options,
    _vertical_target_fits_crop,
    _write_incremental_pricing,
    run_feature_cut_experiment,
    write_external_feature_plan_projection,
)
from jascue_video_lab.auto_reframe import FailureCode


def test_simultaneous_relation_enables_local_two_panel_fallback() -> None:
    policy = AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=75_000,
            min_ms=60_000,
            max_ms=90_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
    )

    assert _autonomous_panel_fallback_eligible(
        hard_region_count=2,
        presentation_preference="tracked_full_bleed",
        relation_mode="simultaneous",
        policy=policy,
    )
    assert not _autonomous_panel_fallback_eligible(
        hard_region_count=3,
        presentation_preference="tracked_full_bleed",
        relation_mode="simultaneous",
        policy=policy,
    )
from jascue_video_lab.cli import build_parser
from jascue_video_lab.models import (
    FeatureCutExecutionProfile,
    FeatureCutRunState,
    FeatureChapterBrief,
    FramingRegionIntent,
    SelectedVerticalFramingProposal,
    VerticalVirtualCameraPhase,
)


def test_feature_cut_aspect_gate_and_cli_defaults() -> None:
    assert _requested_render_aspects("both") == (True, True)
    assert _requested_render_aspects("16x9") == (True, False)
    assert _requested_render_aspects("9x16") == (False, True)
    with pytest.raises(ValueError, match="aspect must be one of"):
        _requested_render_aspects("square")

    defaults = build_parser().parse_args(
        [
            "feature-cut",
            "catalog.json",
            "brief.json",
            "--sam-checkpoint",
            "sam.pt",
            "--output-dir",
            "output",
        ]
    )
    assert defaults.aspect == "both"
    assert defaults.shot_quality_map == []
    assert defaults.post_render_quality_qc is True
    assert defaults.rhythm_style == "standard"
    assert defaults.allow_shorter_within_delivery_range is False
    assert defaults.auto_vertical_framing is True
    assert defaults.execution_profile == "review_preview"
    vertical = build_parser().parse_args(
        [
            "feature-cut",
            "catalog.json",
            "brief.json",
            "--sam-checkpoint",
            "sam.pt",
            "--aspect",
            "9x16",
            "--output-dir",
            "output",
        ]
    )
    assert vertical.aspect == "9x16"


def test_direct_video_v2_never_reopens_selected_clip_semantics() -> None:
    assert _should_refine_selected_vertical_candidate(
        auto_vertical_framing=True,
        human_reframe_policy_requested=False,
        feature_plan_origin="external_projection",
        external_projection_contract_id="direct-video-edit-plan-v1",
        option_data={
            "virtual_camera_proposal": None,
            "strategy": "fit_with_background",
            "regions": [],
        },
    )
    assert not _should_refine_selected_vertical_candidate(
        auto_vertical_framing=True,
        human_reframe_policy_requested=False,
        feature_plan_origin="external_projection",
        external_projection_contract_id="direct-video-edit-plan-v2",
        option_data={
            "virtual_camera_proposal": None,
            "strategy": "tracked_crop",
            "regions": [{"role": "required"}],
        },
    )
    assert not _should_refine_selected_vertical_candidate(
        auto_vertical_framing=True,
        human_reframe_policy_requested=False,
        feature_plan_origin="external_projection",
        external_projection_contract_id="direct-video-edit-plan-v2",
        option_data={
            "virtual_camera_proposal": {"composition_mode": "single_anchor_hold"},
            "strategy": "tracked_crop",
            "regions": [{"role": "required"}],
        },
    )


def test_legacy_plan_can_still_request_missing_selected_clip_framing() -> None:
    assert _should_refine_selected_vertical_candidate(
        auto_vertical_framing=True,
        human_reframe_policy_requested=False,
        feature_plan_origin="generated",
        external_projection_contract_id=None,
        option_data={"virtual_camera_proposal": None},
    )
    assert not _should_refine_selected_vertical_candidate(
        auto_vertical_framing=True,
        human_reframe_policy_requested=True,
        feature_plan_origin="generated",
        external_projection_contract_id=None,
        option_data={"virtual_camera_proposal": None},
    )


def test_autonomous_profile_rejects_raw_output_reuse() -> None:
    with pytest.raises(ValueError, match="forbid.*raw-output"):
        _validate_autonomous_plan_reuse_flags(
            FeatureCutExecutionProfile.AUTONOMOUS_STRICT,
            reuse_feature_plan=False,
            reuse_feature_plan_raw_output=True,
        )


def test_autonomous_profile_rejects_unbound_plan_reuse() -> None:
    with pytest.raises(ValueError, match="requires the current output"):
        _validate_autonomous_plan_reuse_flags(
            FeatureCutExecutionProfile.AUTONOMOUS_STRICT,
            reuse_feature_plan=True,
            reuse_feature_plan_raw_output=False,
        )


def test_review_profile_can_explicitly_reuse_editorial_plan() -> None:
    _validate_autonomous_plan_reuse_flags(
        FeatureCutExecutionProfile.PRODUCTION_REVIEW,
        reuse_feature_plan=True,
        reuse_feature_plan_raw_output=False,
    )


def test_autonomous_profile_rejects_plan_with_legacy_capability_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    plan_dir = output_dir / "gemini-plan"
    plan_dir.mkdir(parents=True)
    policy_path = tmp_path / "policy.json"
    policy = AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=75_000,
            min_ms=60_000,
            max_ms=90_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
    )
    write_json(policy_path, policy)
    capability_path = plan_dir / "editing-capability-catalog.json"
    write_json(
        capability_path,
        simple_production_capability_catalog(),
    )
    record = {
        "projection_contract_id": "direct-video-edit-plan-v2",
        "source_artifacts": [
            {
                "role": "autonomous_policy",
                "path": str(policy_path.resolve()),
                "sha256": sha256_file(policy_path),
            },
            {
                "role": "editing_capability_catalog",
                "path": str(capability_path.resolve()),
                "sha256": sha256_file(capability_path),
            },
        ],
    }
    monkeypatch.setattr(
        feature_cut_module,
        "load_external_feature_plan_projection",
        lambda _plan_dir: (None, None, record),
    )

    with pytest.raises(ValueError, match="predates.*presentation catalog"):
        _validate_autonomous_plan_reuse_flags(
            FeatureCutExecutionProfile.AUTONOMOUS_STRICT,
            reuse_feature_plan=True,
            reuse_feature_plan_raw_output=False,
            output_dir=output_dir,
            autonomous_policy_path=policy_path,
        )


def test_feature_cut_failure_writes_terminal_run_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_impl(**_kwargs: object) -> dict[str, object]:
        raise ValueError("production_review preflight failed")

    monkeypatch.setattr(
        feature_cut_module,
        "_run_feature_cut_experiment_impl",
        fail_impl,
    )

    with pytest.raises(ValueError, match="production_review preflight"):
        run_feature_cut_experiment(
            catalog_path=tmp_path / "catalog.json",
            brief_path=tmp_path / "brief.json",
            checkpoint_path=tmp_path / "model.pt",
            output_dir=tmp_path / "output",
            plan_prompt="plan",
            grounding_prompt="ground",
            execution_profile="production_review",
        )

    status = json.loads(
        (tmp_path / "output" / "run-status.json").read_text(encoding="utf-8")
    )
    assert status["terminal"] is True
    assert status["run_state"] == "failed"
    assert status["delivery_eligible"] is False
    assert status["error"]["type"] == "ValueError"


def test_production_review_preflight_requires_candidates_and_quality() -> None:
    assert _production_review_preflight_failures(
        {"complete": False},
        {"complete": False},
    ) == [
        "candidate_recall_incomplete",
        "quality_map_coverage_incomplete",
    ]
    assert _production_review_preflight_failures(
        {"complete": True},
        {"complete": True},
    ) == []


def test_prompt_sha256_binding_accepts_delimiters_but_not_near_matches() -> None:
    digest = "a" * 64
    assert _prompt_binds_sha256(f"music_sha256={digest}", "music_sha256", digest)
    assert _prompt_binds_sha256(
        f"music_sha256：{digest}",
        "music_sha256",
        digest,
    )
    assert _prompt_binds_sha256(
        f"music_sha256 必須原樣回傳：{digest}",
        "music_sha256",
        digest,
    )
    assert not _prompt_binds_sha256(
        f"other_music_sha256={digest}",
        "music_sha256",
        digest,
    )
    assert not _prompt_binds_sha256(
        f"music_sha256={'b' * 64}",
        "music_sha256",
        digest,
    )


def test_editorial_dwell_reconciles_short_source_without_synthetic_hold() -> None:
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id=feature_id,
                title=feature_id,
                detail_lines=[],
                target_duration_seconds=10,
            )
            for feature_id in (
                "opening",
                "action",
                "detail",
                "comparison",
                "result",
                "closing",
            )
        ],
    )
    chapters = [
        FeatureChapterSelect(
            feature_id=feature_id,
            evidence_status="supported",
            observed_visual_evidence=f"Observable {feature_id}.",
            selection_reason=f"Selected {feature_id}.",
            horizontal_frame_id=f"RF{index:06d}",
            horizontal_strategy="original",
            horizontal_zoom_intent="none",
            horizontal_target_description=None,
            vertical_frame_id=f"RF{index:06d}",
            vertical_strategy="fit_with_background",
            vertical_target_description=None,
            recommended_duration_seconds=10,
            duration_rationale="Relative information and action judgment.",
            quality_risks=[],
            confidence=0.9,
        )
        for index, feature_id in enumerate(
            (
                "opening",
                "action",
                "detail",
                "comparison",
                "result",
                "closing",
            ),
            start=1,
        )
    ]
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id="generic-catalog",
        title=brief.title,
        chapters=chapters,
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    durations, audit = _resolve_editorial_chapter_durations(
        brief,
        plan,
        source_capacity_seconds={
            "opening": 2.5,
            "action": 20,
            "closing": 20,
        },
    )

    assert durations == {
        "opening": 2.5,
        "action": 11.5,
        "detail": 11.5,
        "comparison": 11.5,
        "result": 11.5,
        "closing": 11.5,
    }
    assert sum(durations.values()) == 60
    assert audit["capacity_reconciliation_applied"] is True
    opening = next(
        row for row in audit["chapters"] if row["feature_id"] == "opening"
    )
    assert opening["source_capacity_applied"] is True
    assert opening["source_capacity_seconds"] == 2.5


def test_editorial_dwell_locks_human_approved_trim_before_allocation() -> None:
    feature_ids = ("opening", "action", "detail", "comparison", "result", "closing")
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id=feature_id,
                title=feature_id,
                detail_lines=[],
                target_duration_seconds=10,
            )
            for feature_id in feature_ids
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id="generic-catalog",
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id=feature_id,
                evidence_status="supported",
                observed_visual_evidence=f"Observable {feature_id}.",
                selection_reason=f"Selected {feature_id}.",
                horizontal_frame_id=f"RF{index:06d}",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_frame_id=f"RF{index:06d}",
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                recommended_duration_seconds=10,
                duration_rationale="Relative information and action judgment.",
                quality_risks=[],
                confidence=0.9,
            )
            for index, feature_id in enumerate(feature_ids, start=1)
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    durations, audit = _resolve_editorial_chapter_durations(
        brief,
        plan,
        source_capacity_seconds={
            "opening": 4.2,
            **{feature_id: 20 for feature_id in feature_ids[1:]},
        },
        fixed_duration_seconds={"opening": 4.2},
    )

    assert durations["opening"] == 4.2
    assert sum(durations.values()) == 60
    assert set(durations[feature_id] for feature_id in feature_ids[1:]) == {
        11.16
    }
    opening = next(
        row for row in audit["chapters"] if row["feature_id"] == "opening"
    )
    assert opening["fixed_duration_authority"] == (
        "human_approved_trim_exact_pts"
    )
    assert audit["fixed_approved_trim_duration_seconds"] == {"opening": 4.2}


def test_editorial_dwell_can_snap_to_longer_music_prefix_when_authorized() -> None:
    feature_ids = ("opening", "action", "detail", "comparison", "result", "closing")
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id=feature_id,
                title=feature_id,
                detail_lines=[],
                target_duration_seconds=10,
            )
            for feature_id in feature_ids
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id="generic-catalog",
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id=feature_id,
                evidence_status="supported",
                observed_visual_evidence="Observable evidence.",
                selection_reason="Selected evidence.",
                horizontal_frame_id=f"RF{index:06d}",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_frame_id=f"RF{index:06d}",
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                recommended_duration_seconds=10,
                duration_rationale="Relative information and action judgment.",
                quality_risks=[],
                confidence=0.9,
            )
            for index, feature_id in enumerate(feature_ids, start=1)
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    music_lock = MusicMapLock(
        music_id=f"sha256:{'a' * 64}",
        proposal_path="/test/music-map.proposal.json",
        proposal_sha256="b" * 64,
        review=MusicMapReview(
            proposal_sha256="b" * 64,
            reviewer="test",
            reviewed_at="2026-07-26T00:00:00+00:00",
            decision="approved",
            bpm=120,
            first_downbeat_sample=0,
            meter=4,
        ),
        master_sample_rate=48_000,
        duration_samples=61 * 48_000,
        duration_ms=61_000,
        bpm=120,
        meter=4,
        first_downbeat_sample=0,
        cues=[
            LockedMusicCue(
                cue_id=f"locked-cue-{index:05d}",
                kind="downbeat",
                sample_index=time_ms * 48,
                time_ms=time_ms,
                strength=0.9,
                priority=CuePriority.PREFERRED,
            )
            for index, time_ms in enumerate(
                (10_100, 20_100, 30_100, 40_100, 50_100),
                start=1,
            )
        ],
        sections=[
            MusicSectionCandidate(
                section_id="section-001",
                start_sample=0,
                end_sample=61 * 48_000,
                label="section_001",
                boundary_source="whole_track",
                confidence=1.0,
            )
        ],
        definition_sha256="c" * 64,
    )

    durations, audit = _resolve_editorial_chapter_durations(
        brief,
        plan,
        music_lock=music_lock,
        project_duration_seconds=60,
        allow_music_lock_prefix=True,
    )

    assert sum(durations.values()) == 60
    assert durations["opening"] == 10.1
    assert audit["music_lock_prefix_used"] is True
    assert audit["music_lock_duration_ms"] == 61_000
    assert audit["project_timeline_end_ms"] == 60_000
    assert audit["music_boundary_refinements"][0]["music_snap_applied"] is True


def test_editorial_dwell_refuses_approved_trim_beyond_safe_capacity() -> None:
    feature_ids = ("opening", "action", "detail", "comparison", "result", "closing")
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id=feature_id,
                title=feature_id,
                detail_lines=[],
                target_duration_seconds=10,
            )
            for feature_id in feature_ids
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id="generic-catalog",
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id=feature_id,
                evidence_status="supported",
                observed_visual_evidence="Observable evidence.",
                selection_reason="Selected evidence.",
                horizontal_frame_id=f"RF{index:06d}",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_frame_id=f"RF{index:06d}",
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                recommended_duration_seconds=10,
                duration_rationale="Relative information and action judgment.",
                quality_risks=[],
                confidence=0.9,
            )
            for index, feature_id in enumerate(feature_ids, start=1)
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    with pytest.raises(ValueError, match="exceeds its QualitySafeInterval"):
        _resolve_editorial_chapter_durations(
            brief,
            plan,
            source_capacity_seconds={
                "opening": 3.0,
                **{feature_id: 20 for feature_id in feature_ids[1:]},
            },
            fixed_duration_seconds={"opening": 4.2},
        )


def test_source_capacity_uses_only_requested_aspect() -> None:
    plan = FeatureEditPlan(
        project_id="capacity",
        catalog_id="catalog",
        title="Capacity",
        chapters=[
            FeatureChapterSelect(
                feature_id="chapter",
                evidence_status="supported",
                horizontal_frame_id="RF000001",
                vertical_frame_id="RF000002",
                observed_visual_evidence="Visible evidence.",
                selection_reason="Compare requested aspect capacity.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=[],
                confidence=0.9,
            )
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    clips = {
        "short": RushClip(
            clip_id="short",
            path="/tmp/short.mp4",
            sha256="a" * 64,
            duration_ms=2_000,
            width=1920,
            height=1080,
            frame_rate="30/1",
            size_bytes=1,
        ),
        "long": RushClip(
            clip_id="long",
            path="/tmp/long.mp4",
            sha256="b" * 64,
            duration_ms=10_000,
            width=1920,
            height=1080,
            frame_rate="30/1",
            size_bytes=1,
        ),
    }
    frames = {
        "RF000001": RushFrame(
            frame_id="RF000001",
            clip_id="short",
            requested_time_ms=1_000,
            image_path="/tmp/a.jpg",
        ),
        "RF000002": RushFrame(
            frame_id="RF000002",
            clip_id="long",
            requested_time_ms=5_000,
            image_path="/tmp/b.jpg",
        ),
    }
    shot_cache = {
        "short": SimpleNamespace(
            shots=[
                SimpleNamespace(
                    shot_id="short-shot",
                    start_time_ms=0,
                    end_time_ms=2_000,
                )
            ]
        ),
        "long": SimpleNamespace(
            shots=[
                SimpleNamespace(
                    shot_id="long-shot",
                    start_time_ms=0,
                    end_time_ms=10_000,
                )
            ]
        ),
    }

    vertical = _selected_source_capacity_seconds(
        plan,
        aspect="9x16",
        frames=frames,
        clips=clips,
        shot_cache=shot_cache,
        shots_dir=Path("/tmp"),
        scdet_threshold=4,
    )
    both = _selected_source_capacity_seconds(
        plan,
        aspect="both",
        frames=frames,
        clips=clips,
        shot_cache=shot_cache,
        shots_dir=Path("/tmp"),
        scdet_threshold=4,
    )

    assert vertical["chapter"] == 10
    assert both["chapter"] == 2


def test_source_capacity_requires_safe_interval_containing_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    source_sha = hashlib.sha256(b"source").hexdigest()
    quality_path = tmp_path / "quality.json"
    quality_path.write_text("{}", encoding="utf-8")
    quality_map = SimpleNamespace(
        source_asset_id=f"sha256:{source_sha}",
        source_path=str(source),
        shot_id="shot",
    )
    monkeypatch.setattr(
        feature_cut_module,
        "build_quality_safe_intervals",
        lambda *_args, **_kwargs: [
            SimpleNamespace(start_ms=0, end_ms=6_000),
            SimpleNamespace(start_ms=7_000, end_ms=10_000),
        ],
    )
    plan = FeatureEditPlan(
        project_id="capacity",
        catalog_id="catalog",
        title="Capacity",
        chapters=[
            FeatureChapterSelect(
                feature_id="chapter",
                evidence_status="supported",
                horizontal_frame_id="RF000001",
                vertical_frame_id="RF000001",
                observed_visual_evidence="Visible evidence.",
                selection_reason="Anchor is in the shorter safe pocket.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=[],
                confidence=0.9,
            )
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    clip = RushClip(
        clip_id="clip",
        path=str(source),
        sha256=source_sha,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=len(b"source"),
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id="clip",
        requested_time_ms=8_000,
        image_path="/tmp/frame.jpg",
    )

    capacity = _selected_source_capacity_seconds(
        plan,
        aspect="9x16",
        frames={frame.frame_id: frame},
        clips={clip.clip_id: clip},
        shot_cache={
            clip.clip_id: SimpleNamespace(
                shots=[
                    SimpleNamespace(
                        shot_id="shot",
                        start_time_ms=0,
                        end_time_ms=10_000,
                    )
                ]
            )
        },
        shots_dir=tmp_path,
        scdet_threshold=4,
        quality_maps=[(quality_path, quality_map)],
    )

    assert capacity["chapter"] == 3


def test_editorial_dwell_saves_generic_shortfall_audit_before_fail_closed(
    tmp_path: Path,
) -> None:
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="only",
                title="Only",
                detail_lines=[],
                target_duration_seconds=10,
            )
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id="generic-catalog",
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id="only",
                evidence_status="supported",
                observed_visual_evidence="Observable action.",
                selection_reason="Selected evidence.",
                horizontal_frame_id="RF000001",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_frame_id="RF000001",
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                recommended_duration_seconds=10,
                duration_rationale="Relative information and action judgment.",
                quality_risks=[],
                confidence=0.9,
            )
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    audit_path = tmp_path / "editorial-duration-capacity-shortfall.json"
    with pytest.raises(ValueError, match="cannot satisfy"):
        _resolve_editorial_chapter_durations(
            brief,
            plan,
            source_capacity_seconds={"only": 2.5},
            shortfall_audit_path=audit_path,
        )
    audit = read_json(audit_path)
    assert audit["contract_version"] == (
        "editorial-duration-capacity-shortfall-v1"
    )
    assert audit["status"] == "blocked"
    assert audit["failure_policy"] == "fail_closed_before_render"
    assert audit["preferred_total_seconds"] == 60
    assert audit["feasible_total_seconds"] == 2.5
    assert audit["shortfall_seconds"] == 57.5
    assert audit["user_duration_range"] == {
        "minimum_seconds": 60.0,
        "preferred_seconds": 60.0,
        "maximum_seconds": 90.0,
        "source": (
            "FeatureEditBrief.target_duration_seconds contract and user brief"
        ),
    }
    assert audit["chapter_capacities"] == [
        {
            "feature_id": "only",
            "preferred_weight_seconds": 10.0,
            "preferred_weight_authority": "gemini_relative_dwell",
            "feasible_capacity_seconds": 2.5,
            "capacity_evidence": "selected_shot_boundary",
        }
    ]
    action_by_id = {
        action["action_id"]: action for action in audit["next_actions"]
    }
    assert set(action_by_id) == {
        "select_alternate_candidates",
        "provide_additional_source",
        "approve_shorter_project_duration",
        "revise_required_scope",
    }
    assert (
        action_by_id["approve_shorter_project_duration"]["available"] is False
    )
    assert "repeat_selected_footage" in audit["prohibited_automatic_actions"]


@pytest.mark.parametrize(
    ("aspect", "expected_dimensions", "requested_key", "skipped_key"),
    [
        ("9x16", (1080, 1920), "vertical", "horizontal"),
        ("16x9", (1920, 1080), "horizontal", "vertical"),
    ],
)
def test_feature_cut_single_aspect_skips_unrequested_segments_and_concat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aspect: str,
    expected_dimensions: tuple[int, int],
    requested_key: str,
    skipped_key: str,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    output_dir = tmp_path / "output"
    plan_dir = output_dir / "gemini-plan"
    plan_path = plan_dir / "feature_edit_plan.json"
    request_path = plan_dir / "feature_edit_plan.request.json"
    catalog = RushesCatalog(
        catalog_id="generic-catalog",
        source_directory="/generic-source",
        sample_interval_ms=2000,
        total_duration_ms=60000,
        clips=[],
        frames=[],
        analysis_reel_path=str(tmp_path / "analysis-reel.mp4"),
        generated_at="test",
    )
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="missing-scene",
                title="Missing scene",
                detail_lines=[],
                target_duration_seconds=3,
            )
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id="missing-scene",
                evidence_status="not_found",
                observed_visual_evidence="No direct evidence.",
                selection_reason="No matching catalog evidence.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=["missing evidence"],
                confidence=0.0,
            )
        ],
        uncertainties=["missing evidence"],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    write_json(catalog_path, catalog)
    write_json(brief_path, brief)
    write_json(plan_path, plan)
    write_json(request_path, {"request": "bound test request"})
    write_json(
        plan_dir / "feature-plan.binding.json",
        _current_feature_plan_binding(
            catalog_path=catalog_path,
            catalog_reel_sha256="a" * 64,
            brief_path=brief_path,
            plan_path=plan_path,
            plan_prompt="plan",
            music_sha256=None,
            request_path=request_path,
            created_at="test",
            origin="generated",
        ),
    )

    rendered_dimensions: list[tuple[int, int]] = []
    concat_outputs: list[Path] = []

    class FakeClient:
        def close(self) -> None:
            return None

    def fake_render_missing(
        chapter: FeatureChapterBrief,
        output_path: Path,
        overlay_path: Path,
        dimensions: tuple[int, int],
    ) -> None:
        del chapter, overlay_path
        rendered_dimensions.append(dimensions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")

    def fake_concat(
        segments: list[Path],
        output_path: Path,
        **_kwargs,
    ) -> None:
        assert len(segments) == 1
        concat_outputs.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"render")

    monkeypatch.setattr(feature_cut_module, "GeminiLabClient", FakeClient)
    monkeypatch.setattr(
        feature_cut_module,
        "probe_video",
        lambda _path: SimpleNamespace(sha256="a" * 64),
    )
    monkeypatch.setattr(feature_cut_module, "_segment_is_valid", lambda *a, **k: False)
    monkeypatch.setattr(feature_cut_module, "_render_missing_segment", fake_render_missing)
    monkeypatch.setattr(feature_cut_module, "_concat_segments", fake_concat)
    monkeypatch.setattr(
        feature_cut_module,
        "_output_media_metadata",
        lambda _path: {"duration_ms": 3000},
    )

    result = run_feature_cut_experiment(
        catalog_path=catalog_path,
        brief_path=brief_path,
        checkpoint_path=tmp_path / "unused.pt",
        output_dir=output_dir,
        plan_prompt="plan",
        grounding_prompt="ground",
        vertical_framing_prompt="framing",
        reuse_feature_plan=True,
        aspect=aspect,
        post_render_quality_qc=False,
        auto_vertical_framing=False,
    )

    manifest = read_json(output_dir / "render-manifest.json")
    assert rendered_dimensions == [expected_dimensions]
    assert len(concat_outputs) == 1
    assert manifest[requested_key]["status"] == "rendered"
    assert len(manifest[requested_key]["chapters"]) == 1
    assert manifest[skipped_key] == {
        "requested": False,
        "status": "not_requested",
        "chapters": [],
    }
    assert result[f"{requested_key}_output"] is not None
    assert result[f"{skipped_key}_output"] is None
    assert result["media_rendered"] is True
    assert result["run_state"] == "partial"
    assert result["delivery_eligible"] is False


@pytest.mark.parametrize(
    ("aspect", "expected_build_track_calls", "expected_vertical_geometry_calls"),
    [
        ("9x16", 0, 1),
        ("16x9", 1, 0),
    ],
)
def test_feature_cut_single_aspect_found_evidence_never_runs_other_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aspect: str,
    expected_build_track_calls: int,
    expected_vertical_geometry_calls: int,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    output_dir = tmp_path / "output"
    plan_dir = output_dir / "gemini-plan"
    plan_path = plan_dir / "feature_edit_plan.json"
    request_path = plan_dir / "feature_edit_plan.request.json"
    source_path = tmp_path / "source.mp4"
    clip = RushClip(
        clip_id="clip-1",
        path=str(source_path),
        sha256="b" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=5000,
        image_path=str(tmp_path / "frame.jpg"),
    )
    catalog = RushesCatalog(
        catalog_id="generic-catalog",
        source_directory=str(tmp_path),
        sample_interval_ms=2000,
        total_duration_ms=clip.duration_ms,
        clips=[clip],
        frames=[frame],
        analysis_reel_path=str(tmp_path / "analysis-reel.mp4"),
        generated_at="test",
    )
    brief = FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="visible-scene",
                title="Visible scene",
                detail_lines=[],
                target_duration_seconds=3,
            )
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id="visible-scene",
                evidence_status="supported",
                horizontal_frame_id=frame.frame_id,
                vertical_frame_id=frame.frame_id,
                observed_visual_evidence="One directly visible subject.",
                selection_reason="Representative evidence frame.",
                horizontal_strategy="tracked_reframe",
                horizontal_zoom_intent="subtle",
                horizontal_target_description="the directly visible subject",
                vertical_strategy="tracked_crop",
                vertical_target_description="the directly visible subject",
                quality_risks=[],
                confidence=0.9,
            )
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    write_json(catalog_path, catalog)
    write_json(brief_path, brief)
    write_json(plan_path, plan)
    write_json(request_path, {"request": "bound test request"})
    write_json(
        plan_dir / "feature-plan.binding.json",
        _current_feature_plan_binding(
            catalog_path=catalog_path,
            catalog_reel_sha256="a" * 64,
            brief_path=brief_path,
            plan_path=plan_path,
            plan_prompt="plan",
            music_sha256=None,
            request_path=request_path,
            created_at="test",
            origin="generated",
        ),
    )

    build_track_calls = 0
    vertical_geometry_calls = 0
    concat_outputs: list[Path] = []

    class FakeClient:
        def close(self) -> None:
            return None

    class FakeRatio:
        numerator = 1
        denominator = 1

        def model_dump(self, *, mode: str) -> dict[str, int]:
            assert mode == "json"
            return {"numerator": 1, "denominator": 1}

    class FakeTrack:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"track": "test"}

    source_media = SimpleNamespace(
        video=SimpleNamespace(
            sample_aspect_ratio=FakeRatio(),
            display_sample_aspect_ratio=FakeRatio(),
        )
    )

    def fake_probe(path: Path) -> object:
        if Path(path) == Path(catalog.analysis_reel_path):
            return SimpleNamespace(sha256="a" * 64)
        assert Path(path) == source_path
        return source_media

    def fake_build_track(**kwargs: object) -> tuple[object, FakeTrack]:
        nonlocal build_track_calls
        build_track_calls += 1
        assert kwargs["clip"] == clip
        return object(), FakeTrack()

    def fake_vertical_candidate_geometry(
        **kwargs: object,
    ) -> tuple[str, dict[str, object], list[Path], str]:
        nonlocal vertical_geometry_calls
        vertical_geometry_calls += 1
        assert kwargs["clip"] == clip
        return (
            "test-vertical-filter",
            {
                "applied_strategy": "tracked_crop",
                "fallback_reason": None,
                "source_geometry_lineage_passed": True,
                "tracking_confidence_gate_passed": True,
                "coverage_passed": True,
            },
            [],
            "c" * 64,
        )

    def fake_render_source_segment(**kwargs: object) -> None:
        output_path = Path(str(kwargs["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")

    def fake_concat(
        segments: list[Path],
        output_path: Path,
        **_kwargs,
    ) -> None:
        assert len(segments) == 1
        concat_outputs.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"render")

    monkeypatch.setattr(feature_cut_module, "GeminiLabClient", FakeClient)
    monkeypatch.setattr(
        feature_cut_module,
        "validate_rushes_catalog_sources",
        lambda _catalog: {"status": "validated_test_fixture"},
    )
    monkeypatch.setattr(feature_cut_module, "probe_video", fake_probe)
    monkeypatch.setattr(feature_cut_module, "has_audio_stream", lambda _path: False)
    monkeypatch.setattr(
        feature_cut_module,
        "_selected_source_capacity_seconds",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        feature_cut_module,
        "_chapter_bounds_with_approved_trim",
        lambda *args, **kwargs: (
            3500,
            6500,
            "shot-1",
            {"trim_method": "test_bounds"},
        ),
    )
    monkeypatch.setattr(feature_cut_module, "_build_track", fake_build_track)
    monkeypatch.setattr(
        feature_cut_module,
        "_horizontal_filter_from_track",
        lambda *args, **kwargs: (
            "test-horizontal-filter",
            {
                "applied_zoom": 1.1,
                "fallback_reason": None,
                "risk_codes": [],
                "requires_gemini_review": False,
            },
        ),
    )
    monkeypatch.setattr(
        feature_cut_module,
        "_vertical_candidate_geometry",
        fake_vertical_candidate_geometry,
    )
    monkeypatch.setattr(feature_cut_module, "_segment_is_valid", lambda *a, **k: False)
    monkeypatch.setattr(
        feature_cut_module,
        "_render_source_segment",
        fake_render_source_segment,
    )
    monkeypatch.setattr(
        feature_cut_module,
        "_exact_render_source_interval",
        lambda **_kwargs: {
            "contract_version": "source-pts-interval-v1",
            "source_start_pts": 3500,
            "source_end_pts_exclusive": 6500,
            "time_base": {"numerator": 1, "denominator": 1000},
            "display_start_ms": 3500,
            "display_end_ms": 6500,
        },
    )
    monkeypatch.setattr(
        feature_cut_module,
        "_write_render_boundary_lineage",
        lambda **_kwargs: {
            "contract_version": "render-boundary-lineage-v1",
            "status": "validated_test_fixture",
        },
    )
    monkeypatch.setattr(feature_cut_module, "_concat_segments", fake_concat)
    monkeypatch.setattr(
        feature_cut_module,
        "_output_media_metadata",
        lambda _path: {"duration_ms": 3000},
    )

    result = run_feature_cut_experiment(
        catalog_path=catalog_path,
        brief_path=brief_path,
        checkpoint_path=tmp_path / "unused.pt",
        output_dir=output_dir,
        plan_prompt="plan",
        grounding_prompt="ground",
        vertical_framing_prompt="framing",
        reuse_feature_plan=True,
        aspect=aspect,
        post_render_quality_qc=False,
        auto_vertical_framing=False,
    )

    assert build_track_calls == expected_build_track_calls
    assert vertical_geometry_calls == expected_vertical_geometry_calls
    assert len(concat_outputs) == 1
    expected_output_key = "vertical_output" if aspect == "9x16" else "horizontal_output"
    skipped_output_key = "horizontal_output" if aspect == "9x16" else "vertical_output"
    assert result[expected_output_key] is not None
    assert result[skipped_output_key] is None


def test_no_brief_topk_rejects_candidate_aliases_of_same_evidence_frame() -> None:
    base = {
        "source_asset_id": "sha256:" + "a" * 64,
        "event_id": "event-generic",
        "frame_id": "RF000001",
        "observed_visual_evidence": "One directly visible instance.",
        "selection_reason": "Representative evidence.",
        "quality_risks": [],
        "horizontal_strategy": "original",
        "horizontal_zoom_intent": "none",
        "horizontal_target_description": None,
        "vertical_strategy": "fit_with_background",
        "vertical_target_description": None,
        "vertical_crop_mode": "strict",
        "confidence": 0.8,
    }
    candidates = [
        OpenEditCandidate(candidate_id="alias-a", **base),
        OpenEditCandidate(candidate_id="alias-b", **base),
    ]

    with pytest.raises(ValidationError, match="distinct evidence frames"):
        OpenEditShot(
            feature_id="generic_slot",
            title="Generic slot",
            editorial_role="action",
            intended_effect="Advance the observable sequence.",
            target_duration_seconds=6,
            candidates=candidates,
            horizontal_candidate_id="alias-a",
            vertical_candidate_id="alias-b",
        )
from jascue_video_lab.gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
)
from jascue_video_lab.models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FeatureVerticalCandidate,
    FramingRegionIntent,
    ModelProvenance,
    TrimIntentDecision,
    RushClip,
    RushFrame,
    RushesCatalog,
    SegmentationModelProvenance,
    SegmentationSample,
    SegmentationTrack,
    SelectedVerticalFramingProposal,
    SemanticIdentityStatus,
    SharedSam21AnalysisFrame,
    SharedSam21BBoxSeed,
    SharedSam21SessionManifest,
    SharedSam21SessionTarget,
    SharedSam21SessionTiming,
    TrackingState,
    VerticalVirtualCameraProposal,
    VerticalVirtualCameraProposalPhase,
)
from jascue_video_lab.music import (
    CuePriority,
    LockedMusicCue,
    MusicMapLock,
    MusicMapReview,
    MusicSectionCandidate,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.sam_tracking import (
    SAM21_IMPLEMENTATION_REVISION,
    SAM21_TINY_MODEL_ID,
    pad_normalized_box,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.shots import ShotManifest, ShotSegment
from jascue_video_lab.storage import read_json, write_json


def _portrait_presentation_options(
    *,
    single: str = "not_feasible",
    sequential: str = "not_feasible",
    controlled: str = "not_feasible",
    fit: str = "not_feasible",
) -> list[dict[str, str]]:
    verdicts = {
        "single_full_bleed_crop": single,
        "sequential_virtual_camera": sequential,
        "controlled_clipping": controlled,
        "fit_or_layout": fit,
    }
    return [
        {
            "mode": mode,
            "verdict": verdict,
            "observable_reason": f"{mode} is {verdict} in this fixture.",
        }
        for mode, verdict in verdicts.items()
    ]


def test_open_edit_hard_region_canonicalization_preserves_soft_visibility() -> None:
    payload = {
        "shots": [
            {
                "candidates": [
                    {
                        "vertical_regions": [
                            {
                                "role": "required",
                                "atomic": False,
                                "minimum_visible_fraction": 0.8,
                            },
                            {
                                "role": "preferred",
                                "atomic": True,
                                "minimum_visible_fraction": 0.7,
                            },
                            {
                                "role": "preferred",
                                "atomic": False,
                                "minimum_visible_fraction": 0.6,
                            },
                        ]
                    }
                ]
            }
        ]
    }
    original = json.dumps(payload)

    canonical_text, changes = canonicalize_open_edit_output(original)
    regions = json.loads(canonical_text)["shots"][0]["candidates"][0][
        "vertical_regions"
    ]

    assert [region["minimum_visible_fraction"] for region in regions] == [1.0, 1.0, 0.6]
    assert len(changes) == 2
    assert all(
        change["rule"] == "required_or_atomic_region_is_fully_visible"
        for change in changes
    )


def test_open_edit_normalization_audit_does_not_overwrite_raw_output(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "open-edit.raw_output.json"
    raw_text = json.dumps(
        {
            "shots": [
                {
                    "candidates": [
                        {
                            "vertical_regions": [
                                {
                                    "role": "required",
                                    "minimum_visible_fraction": 0.75,
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    )
    write_json(raw_path, {"output_text": raw_text})
    original = raw_path.read_bytes()

    _, canonical_path, audit_path = _write_open_edit_normalization_artifacts(
        output_dir=tmp_path,
        raw_output_path=raw_path,
        raw_output_text=raw_text,
    )

    assert raw_path.read_bytes() == original
    assert canonical_path.exists()
    audit = read_json(audit_path)
    assert audit["change_count"] == 1
    assert audit["raw_output_artifact_sha256"] == hashlib.sha256(original).hexdigest()


def test_open_edit_reuse_rejects_mismatched_raw_response_copies() -> None:
    with pytest.raises(ValueError, match="does not exactly match"):
        _verified_open_edit_raw_output_text(
            raw_output={"output_text": "first"},
            raw_interaction={"output_text": "second"},
        )


def test_fresh_open_edit_run_refuses_existing_paid_namespace(tmp_path: Path) -> None:
    write_json(tmp_path / "open-edit.raw_output.json", {"output_text": "already paid"})
    with pytest.raises(FileExistsError, match="new output directory"):
        _assert_fresh_open_edit_namespace_empty(tmp_path)


def test_feature_plan_binding_rejects_changed_causal_inputs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    brief = tmp_path / "brief.json"
    plan = tmp_path / "plan.json"
    request = tmp_path / "request.json"
    catalog.write_text('{"catalog":"original"}\n', encoding="utf-8")
    brief.write_text('{"brief":"original"}\n', encoding="utf-8")
    plan.write_text('{"plan":"original"}\n', encoding="utf-8")
    request.write_text('{"request":"original"}\n', encoding="utf-8")

    saved = _current_feature_plan_binding(
        catalog_path=catalog,
        catalog_reel_sha256="a" * 64,
        brief_path=brief,
        plan_path=plan,
        plan_prompt="generic editorial prompt",
        music_sha256=None,
        request_path=request,
        created_at="2026-01-01T00:00:00+00:00",
        origin="generated",
    )
    current = _current_feature_plan_binding(
        catalog_path=catalog,
        catalog_reel_sha256="a" * 64,
        brief_path=brief,
        plan_path=plan,
        plan_prompt="generic editorial prompt",
        music_sha256=None,
        request_path=request,
        created_at="2026-01-02T00:00:00+00:00",
        origin="generated",
    )
    _validate_feature_plan_binding(saved, current)
    causal_hashes = {
        "catalog_sha256",
        "catalog_reel_sha256",
        "brief_sha256",
        "plan_prompt_sha256",
        "system_instruction_sha256",
        "model_id_sha256",
        "response_schema_sha256",
        "plan_sha256",
        "request_sha256",
    }
    assert causal_hashes <= saved.keys()
    assert "music_sha256" in saved
    assert saved["music_sha256"] is None

    for key in causal_hashes:
        changed = dict(current)
        changed[key] = "0" * 64
        with pytest.raises(ValueError, match=key):
            _validate_feature_plan_binding(saved, changed)
    changed_music = dict(current)
    changed_music["music_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="music_sha256"):
        _validate_feature_plan_binding(saved, changed_music)


def test_legacy_feature_plan_reuse_migrates_without_overwriting_evidence(
    tmp_path: Path,
) -> None:
    plan_dir = tmp_path / "gemini-plan"
    plan_dir.mkdir()
    catalog = tmp_path / "catalog.json"
    brief = tmp_path / "brief.json"
    plan = plan_dir / "feature_edit_plan.json"
    prompt = "Use direct evidence to select reusable footage."
    catalog.write_text('{"catalog":"v1"}\n', encoding="utf-8")
    brief.write_text('{"brief":"v1"}\n', encoding="utf-8")
    plan.write_text('{"plan":"v1"}\n', encoding="utf-8")
    request = {
        "model": MODEL_ID,
        "system_instruction": EDITORIAL_SYSTEM_INSTRUCTION,
        "input": [
            {"type": "video", "uri": "files/example", "mime_type": "video/mp4"},
            {
                "type": "text",
                "text": prompt + "\n\n## 本次不可變 metadata\nproject_id: test",
            },
        ],
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(FeatureEditPlan),
        },
    }
    write_json(plan_dir / "feature_edit_plan.request.json", request)
    legacy = {
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "current_catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "current_brief_sha256": hashlib.sha256(brief.read_bytes()).hexdigest(),
        "current_plan_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "model_id": MODEL_ID,
        # This was the known legacy bug; the original request proves the
        # actual editorial instruction before migration is accepted.
        "system_instruction_sha256": hashlib.sha256(
            VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
    }
    legacy_path = plan_dir / "feature-plan.reuse.json"
    write_json(legacy_path, legacy)
    original_legacy_bytes = legacy_path.read_bytes()

    migrated = _migrate_legacy_feature_plan_binding(
        plan_dir=plan_dir,
        catalog_path=catalog,
        catalog_reel_sha256="a" * 64,
        brief_path=brief,
        plan_path=plan,
        plan_prompt=prompt,
        music_sha256=None,
    )

    assert migrated["origin"] == "migrated_legacy_reuse"
    assert migrated["system_instruction_sha256"] == hashlib.sha256(
        EDITORIAL_SYSTEM_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    assert legacy_path.read_bytes() == original_legacy_bytes

    legacy["current_catalog_sha256"] = "f" * 64
    write_json(legacy_path, legacy)
    with pytest.raises(ValueError, match="current_catalog_sha256"):
        _migrate_legacy_feature_plan_binding(
            plan_dir=plan_dir,
            catalog_path=catalog,
            catalog_reel_sha256="a" * 64,
            brief_path=brief,
            plan_path=plan,
            plan_prompt=prompt,
            music_sha256=None,
        )


@pytest.mark.parametrize(
    ("projection_contract_id", "preserve_runtime_candidates"),
    [
        ("clip-card-open-edit-v1", False),
        ("clip-card-open-edit-v2", True),
    ],
)
def test_external_projection_binding_verifies_source_request_plan_and_artifacts(
    tmp_path: Path,
    projection_contract_id: str,
    preserve_runtime_candidates: bool,
) -> None:
    raw_provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="test",
        run_id="run-test",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    source_provenance = raw_provenance.model_copy(
        update={"interaction_id": "interaction-test"}
    )
    catalog = RushesCatalog(
        catalog_id="catalog-test",
        source_directory=str(tmp_path / "sources"),
        sample_interval_ms=1000,
        total_duration_ms=20000,
        clips=[
            RushClip(
                clip_id="clip-1",
                path=str(tmp_path / "source.mp4"),
                sha256="a" * 64,
                duration_ms=20000,
                width=1920,
                height=1080,
                frame_rate="30/1",
                size_bytes=1,
            )
        ],
        frames=[
            RushFrame(
                frame_id=f"RF{index:06d}",
                clip_id="clip-1",
                requested_time_ms=(index - 1) * 500,
                image_path=str(tmp_path / f"frame-{index}.jpg"),
            )
            for index in range(1, 21)
        ],
        analysis_reel_path=str(tmp_path / "reel.mp4"),
        generated_at="2026-01-01T00:00:00+00:00",
    )
    shots: list[OpenEditShot] = []
    for index in range(10):
        candidates = [
            OpenEditCandidate(
                candidate_id=f"candidate-{index}-{offset}",
                source_asset_id="sha256:" + "a" * 64,
                event_id=f"event-{index}",
                frame_id=f"RF{index * 2 + offset + 1:06d}",
                observed_visual_evidence="One directly visible subject.",
                selection_reason="Clear representative state.",
                quality_risks=[],
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                vertical_crop_mode="strict",
                confidence=0.8,
            )
            for offset in range(2)
        ]
        shots.append(
            OpenEditShot(
                feature_id=f"scene_{index}",
                title=f"Scene {index}",
                editorial_role=(
                    "hook" if index == 0 else "closing" if index == 9 else "action"
                ),
                intended_effect="Maintain visible narrative progression.",
                target_duration_seconds=6,
                candidates=candidates,
                horizontal_candidate_id=candidates[0].candidate_id,
                vertical_candidate_id=candidates[0].candidate_id,
            )
        )
    raw_source_plan = OpenEditPlan(
        project_id="project-test",
        catalog_id=catalog.catalog_id,
        inferred_title="Generic edit",
        inferred_theme="Observable sequence",
        intended_audience_hypothesis="General audience",
        story_arc="Opening, progression, closing",
        shots=shots,
        excluded_patterns=[],
        uncertainties=[],
        model_provenance=raw_provenance,
    )
    source_plan = raw_source_plan.model_copy(
        update={"model_provenance": source_provenance}
    )
    # Exercise both deterministic projection generations.  v1 predates
    # runtime Top-K candidates; v2 preserves them for automatic recovery.
    brief, feature_plan, _ = project_feature_contracts(
        source_plan,
        preserve_runtime_candidates=preserve_runtime_candidates,
    )
    assert all(
        selected.recommended_duration_seconds == 6
        for selected in feature_plan.chapters
    )
    assert all(
        selected.duration_rationale == "Maintain visible narrative progression."
        for selected in feature_plan.chapters
    )
    assert all(
        selected.horizontal_camera_intent == "hold"
        for selected in feature_plan.chapters
    )
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    plan_dir = tmp_path / "gemini-plan"
    feature_plan_path = plan_dir / "feature_edit_plan.json"
    source_plan_path = tmp_path / "source-plan.json"
    request_path = tmp_path / "source.request.json"
    raw_output_path = tmp_path / "source.raw_output.json"
    write_json(catalog_path, catalog)
    write_json(brief_path, brief)
    write_json(feature_plan_path, feature_plan)
    write_json(source_plan_path, source_plan)
    write_json(
        request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only the supplied media evidence.",
            "input": [{"type": "text", "text": "Select a coherent sequence."}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(OpenEditPlan),
            },
        },
    )
    raw_interaction_path = tmp_path / "source.raw_interaction.json"
    write_json(raw_output_path, {"output_text": raw_source_plan.model_dump_json()})
    write_json(raw_interaction_path, {"id": "interaction-test"})

    invalid_request = read_json(request_path)
    invalid_request["response_format"]["schema"] = {"type": "object"}
    write_json(request_path, invalid_request)
    with pytest.raises(ValueError, match="registered model"):
        write_external_feature_plan_projection(
            plan_dir=plan_dir,
                projection_contract_id=projection_contract_id,
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=feature_plan_path,
            source_plan_path=source_plan_path,
            source_request_path=request_path,
            source_artifacts={
                "source_raw_output": raw_output_path,
                "source_raw_interaction": raw_interaction_path,
            },
        )
    invalid_request["response_format"]["schema"] = gemini_response_schema(
        OpenEditPlan
    )
    write_json(request_path, invalid_request)

    fabricated_source = source_plan.model_copy(
        update={"inferred_title": "Fabricated source title"}
    )
    write_json(source_plan_path, fabricated_source)
    with pytest.raises(ValueError, match="raw model output"):
        write_external_feature_plan_projection(
            plan_dir=plan_dir,
                projection_contract_id=projection_contract_id,
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=feature_plan_path,
            source_plan_path=source_plan_path,
            source_request_path=request_path,
            source_artifacts={
                "source_raw_output": raw_output_path,
                "source_raw_interaction": raw_interaction_path,
            },
        )
    write_json(source_plan_path, source_plan)

    fabricated_plan = feature_plan.model_copy(update={"title": "Fabricated title"})
    write_json(feature_plan_path, fabricated_plan)
    with pytest.raises(ValueError, match="deterministic projector"):
        write_external_feature_plan_projection(
            plan_dir=plan_dir,
                projection_contract_id=projection_contract_id,
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=feature_plan_path,
            source_plan_path=source_plan_path,
            source_request_path=request_path,
            source_artifacts={
                "source_raw_output": raw_output_path,
                "source_raw_interaction": raw_interaction_path,
            },
        )
    write_json(feature_plan_path, feature_plan)

    pointer_path = write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id=projection_contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=request_path,
        source_artifacts={
            "source_raw_output": raw_output_path,
            "source_raw_interaction": raw_interaction_path,
        },
    )
    binding = _current_external_projection_binding(
        plan_dir=plan_dir,
        catalog_path=catalog_path,
        catalog_reel_sha256="a" * 64,
        brief_path=brief_path,
        plan_path=feature_plan_path,
        music_sha256=None,
        created_at="2026-01-02T00:00:00+00:00",
    )

    assert pointer_path.name == "feature-plan.external-projection.json"
    assert binding["origin"] == "external_projection"
    assert binding["external_projection_contract_id"] == projection_contract_id
    _validate_feature_plan_binding(binding, dict(binding))

    write_json(raw_output_path, {"output_text": '{"changed":true}'})
    with pytest.raises(ValueError, match="source artifact changed"):
        _current_external_projection_binding(
            plan_dir=plan_dir,
            catalog_path=catalog_path,
            catalog_reel_sha256="a" * 64,
            brief_path=brief_path,
            plan_path=feature_plan_path,
            music_sha256=None,
            created_at="2026-01-02T00:00:00+00:00",
        )


def test_incremental_pricing_names_changed_error_artifacts_honestly(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "grounding" / "grounding.raw_interaction.json",
        {
            "model": MODEL_ID,
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 10,
                "total_thought_tokens": 2,
            }
        },
    )
    write_json(tmp_path / "grounding" / "errors.json", [{"message": "review"}])

    result = _write_incremental_pricing(
        output_dir=tmp_path,
        prior_interaction_hashes={},
        prior_error_hashes={},
    )

    assert result["request_count"] == 1
    assert result["changed_error_artifact_count"] == 1
    assert "failed_request_artifact_count" not in result
    assert (tmp_path / "pricing.incremental.json").exists()


def test_feature_brief_requires_unique_chapter_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FeatureEditBrief(
            project_id="test",
            title="test",
            target_duration_seconds=60,
            chapters=[
                FeatureChapterBrief(
                    feature_id="same",
                    title="one",
                    detail_lines=[],
                    target_duration_seconds=4,
                ),
                FeatureChapterBrief(
                    feature_id="same",
                    title="two",
                    detail_lines=[],
                    target_duration_seconds=4,
                ),
            ],
        )


def test_segment_cache_key_changes_with_source_or_tracking_geometry() -> None:
    base = {
        "source_sha256": "a" * 64,
        "start_ms": 1000,
        "end_ms": 3000,
        "filter_graph": "crop=1080:1920:x=100:y=0",
        "geometry": {"applied_strategy": "tracked_crop"},
        "track_fingerprint": "b" * 64,
    }
    original = _segment_variant_fingerprint(**base)
    assert original != _segment_variant_fingerprint(
        **{**base, "track_fingerprint": "c" * 64}
    )
    assert original != _segment_variant_fingerprint(
        **{**base, "source_sha256": "d" * 64}
    )


def test_vertical_crop_geometry_preserves_rendered_x_audit_keyframes() -> None:
    x_values, audit = _vertical_crop_geometry(
        [0.0, 1.0, 2.0],
        [200.0, 500.0, 900.0],
        [[100, 100, 300, 900], [400, 100, 600, 900], [800, 100, 1000, 900]],
    )

    assert len(x_values) == 3
    assert [item["crop_x_pixels"] for item in audit["crop_keyframes"]] == [
        round(value, 3) for value in x_values
    ]
    coordinate_space = audit["crop_coordinate_space"]
    assert coordinate_space["contract_version"] == "aspect-preserving-cover-v1"
    assert coordinate_space["orientation_basis"] == "ffmpeg_autorotated_display"
    assert coordinate_space["scale_policy"] == "aspect_preserving_cover"
    assert coordinate_space["source_display_width"] == 1920
    assert coordinate_space["source_display_height"] == 1080
    assert coordinate_space["scaled_width"] == 3414
    assert coordinate_space["scaled_height"] == 1920
    assert coordinate_space["crop_width"] == 1080
    assert coordinate_space["crop_height"] == 1920
    assert coordinate_space["active_pan_axes"] == ["x"]
    assert audit["crop_width_normalized"] == pytest.approx(316.3445)
    assert audit["max_target_width_normalized"] == 200
    assert x_values == sorted(x_values)


def test_soft_extent_visibility_is_measured_without_relaxing_hard_containment() -> None:
    _, _, crop_audit = _tracked_crop_geometry(
        [0.0, 0.5],
        [500.0, 500.0],
        [[450, 200, 550, 800], [450, 200, 550, 800]],
        source_width=1920,
        source_height=1080,
        output_width=1080,
        output_height=1920,
    )
    soft_track = SimpleNamespace(
        analysis_start_ms=0,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=time_ms,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=[0, 200, 400, 800],
            )
            for time_ms in (0, 500)
        ],
    )
    permissive = FramingRegionIntent(
        region_id="context",
        target_description="visible surrounding context",
        role="preferred",
        minimum_visible_fraction=0.1,
    )
    strict = permissive.model_copy(update={"minimum_visible_fraction": 0.9})

    accepted = _soft_extent_visibility_audit(
        tracks=[soft_track],  # type: ignore[list-item]
        regions=[permissive],
        crop_audit=crop_audit,
    )
    rejected = _soft_extent_visibility_audit(
        tracks=[soft_track],  # type: ignore[list-item]
        regions=[strict],
        crop_audit=crop_audit,
    )

    assert crop_audit["containment_failure_count"] == 0
    assert accepted["soft_extent_visibility_passed"] is True
    assert rejected["soft_extent_visibility_passed"] is False
    assert rejected["soft_extent_regions"][0]["minimum_visible_area_fraction"] < 0.9


def test_controlled_primary_center_preview_is_bounded_and_non_atomic() -> None:
    subject = FramingRegionIntent(
        region_id="primary-subject",
        target_description="the selected visible subject",
        kind="subject",
        role="required",
    )
    geometry = {
        "applied_strategy": "tracked_crop",
        "fallback_reason": None,
        "minimum_visible_required_area_fraction": 0.94,
    }
    failures = [
        FailureCode.HARD_CORE_NOT_FULLY_RETAINED,
        FailureCode.IDENTITY_VERIFICATION_PENDING,
    ]

    assert _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry=geometry,
        regions=[subject],
        failure_codes=failures,
    )
    assert not _controlled_primary_center_preview_allowed(
        crop_mode="strict",
        geometry=geometry,
        regions=[subject],
        failure_codes=failures,
    )
    assert not _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry={
            **geometry,
            "minimum_visible_required_area_fraction": 0.89,
        },
        regions=[subject],
        failure_codes=failures,
    )
    assert _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry={
            **geometry,
            "applied_strategy": "phase_virtual_camera",
            "minimum_visible_required_area_fraction": 2 / 3,
        },
        regions=[subject],
        failure_codes=failures,
    )
    assert not _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry=geometry,
        regions=[
            subject.model_copy(
                update={"kind": "text_region", "atomic": True}
            )
        ],
        failure_codes=failures,
    )


def test_controlled_preview_allows_only_bound_atomic_relation_carrier() -> None:
    relation_core = FramingRegionIntent(
        region_id="relation-core",
        target_description="the directly visible relation zone",
        kind="subject",
        evidence_role="relation_carrier",
        role="required",
        atomic=True,
        observable_relations=["participant A visibly contacts participant B"],
    )
    participants = [
        FramingRegionIntent(
            region_id=f"participant-{suffix}",
            entity_id=f"entity-{suffix}",
            target_description=f"visible participant {suffix}",
            evidence_role="relation_participant",
            role="preferred",
            minimum_visible_fraction=0.8,
        )
        for suffix in ("a", "b")
    ]
    geometry = {
        "applied_strategy": "tracked_crop",
        "fallback_reason": None,
        "minimum_visible_required_area_fraction": 0.84,
    }

    assert _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry=geometry,
        regions=[relation_core, *participants],
        failure_codes=[FailureCode.ATOMIC_REGION_CLIPPED],
    )
    assert not _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry=geometry,
        regions=[relation_core, participants[0]],
        failure_codes=[FailureCode.ATOMIC_REGION_CLIPPED],
    )
    assert not _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry=geometry,
        regions=[
            relation_core.model_copy(update={"kind": "text_region"}),
            *participants,
        ],
        failure_codes=[FailureCode.ATOMIC_REGION_CLIPPED],
    )


def test_controlled_preview_allows_bound_relation_group_outer_clipping() -> None:
    participants = [
        FramingRegionIntent(
            region_id=f"participant-{suffix}",
            entity_id=f"entity-{suffix}",
            target_description=f"visible participant {suffix}",
            evidence_role="relation_participant",
            role="required",
            atomic=False,
            observable_relations=["visibly forms the selected relation"],
        )
        for suffix in ("a", "b")
    ]
    geometry = {
        "applied_strategy": "tracked_crop",
        "fallback_reason": None,
        "minimum_visible_required_area_fraction": 0.77,
    }

    assert _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry=geometry,
        regions=participants,
        failure_codes=[FailureCode.HARD_CORE_NOT_FULLY_RETAINED],
    )
    assert not _controlled_primary_center_preview_allowed(
        crop_mode="primary_center",
        geometry={
            **geometry,
            "minimum_visible_required_area_fraction": 0.65,
        },
        regions=participants,
        failure_codes=[FailureCode.HARD_CORE_NOT_FULLY_RETAINED],
    )


def test_ranked_candidate_intent_is_not_overridden_by_generic_brief_target() -> None:
    regions, target = _resolve_vertical_candidate_intent(
        option_regions=[],
        option_target_description="the rightmost visible instance beside the sign",
        selected_target_description="the leftmost selected instance",
        brief_primary_target_description="the main object",
        brief_regions=[],
        inherit_reviewed_brief_intent=False,
    )

    assert regions == []
    assert target == "the rightmost visible instance beside the sign"


def test_legacy_or_human_candidate_can_inherit_reviewed_brief_regions() -> None:
    reviewed = FramingRegionIntent(
        region_id="reviewed-core",
        target_description="reviewed visible core",
        role="required",
    )

    regions, target = _resolve_vertical_candidate_intent(
        option_regions=[],
        option_target_description=None,
        selected_target_description=None,
        brief_primary_target_description="reviewed target",
        brief_regions=[reviewed],
        inherit_reviewed_brief_intent=True,
    )

    assert regions == [reviewed]
    assert target == "reviewed visible core"


def test_runtime_candidates_preserve_rank_but_human_binding_disables_switching() -> None:
    candidates = [
        FeatureVerticalCandidate(
            candidate_id=f"take-{rank}",
            rank=rank,
            source_asset_id="sha256:" + ("a" if rank == 1 else "b") * 64,
            event_id=f"event-{rank}",
            frame_id=f"RF{rank:06d}",
            observed_visual_evidence=f"Visible evidence {rank}",
            selection_reason=f"Reason {rank}",
            strategy="fit_with_background",
            target_description=None,
            confidence=0.8,
        )
        for rank in (1, 2)
    ]
    selected = FeatureChapterSelect(
        feature_id="scene",
        evidence_status="supported",
        horizontal_frame_id="RF000001",
        vertical_frame_id="RF000001",
        observed_visual_evidence="Visible evidence 1",
        selection_reason="Reason 1",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        vertical_coverage_intent="group_coverage",
        vertical_coverage_target_descriptions=[
            "the first visible group member",
            "the second visible group member",
        ],
        quality_risks=[],
        confidence=0.8,
        vertical_candidates=candidates,
    )

    automatic = _vertical_runtime_candidate_options(
        selected, human_policy_binding_present=False
    )
    reviewed = _vertical_runtime_candidate_options(
        selected, human_policy_binding_present=True
    )

    assert [item["candidate_id"] for item in automatic] == ["take-1", "take-2"]
    assert automatic[0]["coverage_intent"] == "group_coverage"
    assert automatic[0]["coverage_target_descriptions"] == [
        "the first visible group member",
        "the second visible group member",
    ]
    assert automatic[1]["coverage_intent"] == "single_primary"
    assert automatic[1]["coverage_target_descriptions"] == []
    assert reviewed[0]["candidate_id"] == "legacy-primary"
    assert reviewed[0]["frame_id"] == "RF000001"
    assert reviewed[0]["target_description"] is None
    assert reviewed[0]["coverage_intent"] == "group_coverage"


def test_runtime_fallback_binds_its_own_clip_card_entities() -> None:
    candidates = [
        FeatureVerticalCandidate(
            candidate_id=f"take-{rank}",
            rank=rank,
            source_asset_id="sha256:" + ("a" if rank == 1 else "b") * 64,
            event_id=f"event-{rank}",
            frame_id=f"RF{rank:06d}",
            observed_visual_evidence=f"Visible evidence {rank}",
            selection_reason=f"Reason {rank}",
            strategy="fit_with_background",
            target_description=None,
            confidence=0.8,
        )
        for rank in (1, 2)
    ]
    selected = FeatureChapterSelect(
        feature_id="scene",
        evidence_status="supported",
        horizontal_frame_id="RF000001",
        vertical_frame_id="RF000001",
        observed_visual_evidence="The first take shows two unrelated subjects.",
        selection_reason="Rank one was selected first.",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        vertical_coverage_intent="group_coverage",
        vertical_coverage_target_descriptions=["rank-one A", "rank-one B"],
        quality_risks=[],
        confidence=0.8,
        vertical_candidates=candidates,
    )
    fallback_event = {
        "source_asset_id": candidates[1].source_asset_id,
        "event_id": candidates[1].event_id,
        "required_entity_ids": ["entity-object", "entity-reference"],
        "primary_entity_ids": ["entity-object", "entity-reference"],
        "entities": [
            {
                "entity_id": "entity-object",
                "kind": "device",
                "label": "visible object",
                "distinguishing_features": "the larger visible object",
            },
            {
                "entity_id": "entity-reference",
                "kind": "object",
                "label": "visible reference",
                "distinguishing_features": "the smaller comparison reference",
            },
        ],
        "grounding_targets": [
            {
                "entity_id": "entity-object",
                "target_description": "the larger visible object",
            },
            {
                "entity_id": "entity-reference",
                "target_description": "the smaller visible comparison reference",
            },
        ],
    }

    options = _vertical_runtime_candidate_options(
        selected,
        human_policy_binding_present=False,
        candidate_evidence_events={
            (candidates[1].source_asset_id, candidates[1].event_id): fallback_event
        },
    )

    fallback = options[1]
    assert fallback["coverage_intent"] == "single_primary"
    assert fallback["coverage_target_descriptions"] == [
        "the larger visible object",
    ]
    assert {
        region["entity_id"] for region in fallback["regions"]
    } == {"entity-object", "entity-reference"}
    roles = {
        region["entity_id"]: (
            region["role"],
            region["evidence_role"],
        )
        for region in fallback["regions"]
    }
    assert roles["entity-object"] == ("required", "primary_subject")
    assert roles["entity-reference"] == ("preferred", "context_reference")
    assert "rank-one A" not in json.dumps(fallback, ensure_ascii=False)


def test_candidate_asset_reference_accepts_catalog_id_or_sha256() -> None:
    clip = RushClip(
        clip_id="C0001",
        path="/tmp/visible.mp4",
        sha256="a" * 64,
        duration_ms=1000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )

    assert _candidate_asset_reference_matches("C0001", clip)
    assert _candidate_asset_reference_matches("sha256:" + "a" * 64, clip)
    assert _candidate_asset_reference_matches(None, clip)
    assert not _candidate_asset_reference_matches("C0002", clip)


def test_gemini_virtual_camera_proposal_preserves_observed_anchor_order() -> None:
    regions = [
        FramingRegionIntent(
            region_id="result",
            target_description="the visible result area on the right",
        ),
        FramingRegionIntent(
            region_id="performer",
            target_description="the visible performer on the left",
        ),
    ]
    proposal = VerticalVirtualCameraProposal(
        composition_mode="sequential_focus",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="result-first",
                start_progress=0.0,
                end_progress=0.4,
                anchor_region_ids=["result"],
                camera_behavior="hold",
                observable_predicate="The result is already fully visible.",
                transition_condition="The performer begins the next visible action.",
                editorial_reason="Let the viewer identify the result before the hand-off.",
            ),
            VerticalVirtualCameraProposalPhase(
                phase_id="performer-second",
                start_progress=0.4,
                end_progress=1.0,
                anchor_region_ids=["performer"],
                camera_behavior="follow",
                transition_in="smoothstep",
                transition_duration_fraction=0.25,
                observable_predicate="The performer carries the next visible action.",
                transition_condition="Hold through the end of that visible action.",
                editorial_reason="Follow the action rather than a fixed screen direction.",
            ),
        ],
        proposal_reason="Visible information passes from the result to the performer.",
    )
    candidate = FeatureVerticalCandidate(
        candidate_id="generic-take",
        rank=1,
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-generic",
        frame_id="RF000001",
        observed_visual_evidence="A result and performer are visible in one shot.",
        selection_reason="The visible hand-off supports a sequential portrait crop.",
        strategy="tracked_crop",
        regions=regions,
        virtual_camera_proposal=proposal,
        confidence=0.8,
    )

    phases, origin = _resolve_vertical_camera_phases(
        option_data=candidate.model_dump(mode="python"),
        reviewed_phases=[],
    )

    assert origin == "gemini_proposed"
    assert [phase.anchor_region_ids for phase in phases] == [
        ["result"],
        ["performer"],
    ]
    assert all(phase.minimum_anchor_visible_fraction == 1.0 for phase in phases)
    assert "Visible predicate" in phases[0].editorial_reason

    controlled_phases, controlled_origin = _resolve_vertical_camera_phases(
        option_data=candidate.model_copy(
            update={
                "crop_mode": "primary_center",
                "allow_controlled_clip": True,
            }
        ).model_dump(mode="python"),
        reviewed_phases=[],
        regions=regions,
        allow_controlled_clip=True,
    )

    assert controlled_origin == "gemini_proposed"
    assert all(
        phase.minimum_anchor_visible_fraction == 2 / 3
        for phase in controlled_phases
    )


def test_selected_framing_allows_scale_locked_sequential_comparison() -> None:
    sequential = VerticalVirtualCameraProposal(
        composition_mode="sequential_focus",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="first",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["subject-a"],
                observable_predicate="The first subject is visible.",
                transition_condition="Attention moves to the second subject.",
                editorial_reason="Show the first subject.",
            ),
                VerticalVirtualCameraProposalPhase(
                    phase_id="second",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["subject-b"],
                observable_predicate="The second subject is visible.",
                    transition_condition="Hold to the end.",
                    cut_admissible=True,
                    editorial_reason="Show the second subject.",
            ),
        ],
        proposal_reason="Show both comparison subjects at one consistent scale.",
    )
    proposal = SelectedVerticalFramingProposal(
        candidate_id="candidate-a",
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-a",
        frame_id="RF000001",
        semantic_requirement="simultaneous_relation",
        relation_temporal_mode="sequentially_reconstructable",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            sequential="feasible",
        ),
        regions=[
            {
                "region_id": "subject-a",
                "target_description": "the first visible subject",
                "evidence_role": "relation_participant",
                "role": "required",
            },
            {
                "region_id": "subject-b",
                "target_description": "the second visible subject",
                "evidence_role": "relation_participant",
                "role": "required",
            },
        ],
        sequential_reconstruction={
            "linkage_type": "scale_locked_comparison",
            "linkage_region_ids": ["subject-a", "subject-b"],
            "preserve_scale": True,
            "observable_reason": "Both subjects share one source-camera scale.",
        },
        virtual_camera_proposal=sequential,
        observed_evidence=["Both subjects are visible in the same source shot."],
        decision_reason="The same-scale views support a sequential comparison.",
        confidence=0.8,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    assert proposal.virtual_camera_proposal == sequential


def test_sequential_comparison_rejects_scale_changing_phase() -> None:
    sequential = VerticalVirtualCameraProposal(
        composition_mode="sequential_focus",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="first",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["subject-a"],
                observable_predicate="The first subject is visible.",
                transition_condition="Attention moves to the second subject.",
                editorial_reason="Show the first subject.",
                camera_behavior="push_in",
            ),
                VerticalVirtualCameraProposalPhase(
                    phase_id="second",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["subject-b"],
                observable_predicate="The second subject is visible.",
                    transition_condition="Hold to the end.",
                    cut_admissible=True,
                    editorial_reason="Show the second subject.",
            ),
        ],
        proposal_reason="An invalid comparison that changes scale.",
    )
    with pytest.raises(ValidationError, match="scale-preserving"):
        SelectedVerticalFramingProposal(
            candidate_id="candidate-scale-change",
            source_asset_id="sha256:" + "e" * 64,
            event_id="event-scale-change",
            frame_id="RF000001",
            semantic_requirement="simultaneous_relation",
            relation_temporal_mode="sequentially_reconstructable",
            recommended_action="tracked_crop",
            presentation_options=_portrait_presentation_options(
                sequential="feasible",
            ),
            regions=[
                {
                    "region_id": region_id,
                    "target_description": f"the {region_id} visible subject",
                    "evidence_role": "relation_participant",
                    "role": "required",
                }
                for region_id in ("subject-a", "subject-b")
            ],
            sequential_reconstruction={
                "linkage_type": "scale_locked_comparison",
                "linkage_region_ids": ["subject-a", "subject-b"],
                "preserve_scale": True,
                "observable_reason": "Both subjects share one source-camera scale.",
            },
            virtual_camera_proposal=sequential,
            observed_evidence=["Both subjects are visible."],
            decision_reason="Changing scale would invalidate the comparison.",
            confidence=0.8,
            model_provenance=ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="test",
                generated_at="test",
            ),
        )


def test_selected_framing_allows_overlapping_multi_subject_handoff() -> None:
    sequential = VerticalVirtualCameraProposal(
        composition_mode="sequential_focus",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="left-center",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["left", "center"],
                observable_predicate="The left and center subjects are visible.",
                transition_condition="Attention moves toward the right.",
                editorial_reason="Establish the group with an overlapping anchor.",
            ),
                VerticalVirtualCameraProposalPhase(
                    phase_id="center-right",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["center", "right"],
                observable_predicate="The center and right subjects are visible.",
                    transition_condition="Hold to the end.",
                    cut_admissible=True,
                    editorial_reason="Preserve the center subject across the handoff.",
            ),
        ],
        proposal_reason="The shared center anchor preserves the group relation.",
    )
    proposal = SelectedVerticalFramingProposal(
        candidate_id="candidate-overlap",
        source_asset_id="sha256:" + "d" * 64,
        event_id="event-overlap",
        frame_id="RF000001",
        semantic_requirement="simultaneous_relation",
        relation_temporal_mode="sequentially_reconstructable",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            sequential="feasible",
        ),
        regions=[
            {
                "region_id": region_id,
                "target_description": f"the {region_id} visible subject",
                "evidence_role": "relation_participant",
                "role": "required",
            }
            for region_id in ("left", "center", "right")
        ],
        sequential_reconstruction={
            "linkage_type": "shared_tracked_anchor",
            "linkage_region_ids": ["center"],
            "observable_reason": "The center subject appears in both phases.",
        },
        virtual_camera_proposal=sequential,
        observed_evidence=["Three subjects form one visible group."],
        decision_reason="Overlapping phases preserve the group relationship.",
        confidence=0.8,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    assert proposal.virtual_camera_proposal == sequential


def test_selected_framing_group_coverage_requires_multiple_hard_members() -> None:
    single = VerticalVirtualCameraProposal(
        composition_mode="single_anchor_hold",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="only",
                start_progress=0.0,
                end_progress=1.0,
                anchor_region_ids=["center"],
                observable_predicate="The center member is visible.",
                transition_condition="Hold to the end.",
                editorial_reason="An invalid group proposal that covers one member.",
            )
        ],
        proposal_reason="This incorrectly collapses a group into one anchor.",
    )
    with pytest.raises(ValidationError, match="at least two hard-core member"):
        SelectedVerticalFramingProposal(
            candidate_id="candidate-collapsed-group",
            source_asset_id="sha256:" + "f" * 64,
            event_id="event-collapsed-group",
            frame_id="RF000001",
            semantic_requirement="group_coverage",
            relation_temporal_mode="not_applicable",
            recommended_action="tracked_crop",
            presentation_options=_portrait_presentation_options(
                single="feasible",
            ),
            regions=[
                {
                    "region_id": "center",
                    "target_description": "the center visible group member",
                    "evidence_role": "primary_subject",
                    "role": "required",
                },
                {
                    "region_id": "flanking-members",
                    "target_description": "the other visible group members",
                    "evidence_role": "relation_participant",
                    "role": "preferred",
                    "minimum_visible_fraction": 0.3,
                },
            ],
            virtual_camera_proposal=single,
            observed_evidence=["Multiple members form one visible lineup."],
            decision_reason="Convenient centering does not satisfy group coverage.",
            confidence=0.8,
            model_provenance=ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="test",
                generated_at="test",
            ),
        )


def test_selected_framing_group_coverage_accepts_atomic_compound_group() -> None:
    hold = VerticalVirtualCameraProposal(
        composition_mode="single_anchor_hold",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="whole-group",
                start_progress=0.0,
                end_progress=1.0,
                anchor_region_ids=["compound-group"],
                observable_predicate="The indivisible group remains visible.",
                transition_condition="Hold to the end.",
                editorial_reason="Preserve the complete compound group.",
            )
        ],
        proposal_reason="The group is one indivisible visible composition.",
    )
    proposal = SelectedVerticalFramingProposal(
        candidate_id="candidate-compound-group",
        source_asset_id="sha256:" + "e" * 64,
        event_id="event-compound-group",
        frame_id="RF000001",
        semantic_requirement="group_coverage",
        relation_temporal_mode="not_applicable",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            single="feasible",
        ),
        regions=[
            {
                "region_id": "compound-group",
                "target_description": "the complete indivisible visible group",
                "evidence_role": "primary_subject",
                "role": "required",
                "atomic": True,
            }
        ],
        virtual_camera_proposal=hold,
        observed_evidence=["The complete group forms one bounded composition."],
        decision_reason="Partial clipping would change the group meaning.",
        confidence=0.8,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    assert proposal.regions[0].atomic is True


def test_selected_framing_group_coverage_accepts_overlapping_sequence() -> None:
    sequential = VerticalVirtualCameraProposal(
        composition_mode="sequential_focus",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="first-pair",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["first", "middle"],
                observable_predicate="The first and middle members are visible.",
                transition_condition="Attention passes toward the final member.",
                editorial_reason="Cover the first part of the group.",
            ),
                VerticalVirtualCameraProposalPhase(
                    phase_id="second-pair",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["middle", "last"],
                observable_predicate="The middle and last members are visible.",
                    transition_condition="Hold through the end.",
                    cut_admissible=True,
                    editorial_reason="Complete coverage with an overlapping anchor.",
            ),
        ],
        proposal_reason="Every meaning-bearing member is covered once.",
    )
    proposal = SelectedVerticalFramingProposal(
        candidate_id="candidate-group",
        source_asset_id="sha256:" + "c" * 64,
        event_id="event-group",
        frame_id="RF000001",
        semantic_requirement="group_coverage",
        relation_temporal_mode="sequentially_reconstructable",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            sequential="feasible",
        ),
        regions=[
            {
                "region_id": region_id,
                "target_description": f"the {region_id} visible member",
                "evidence_role": "relation_participant",
                "role": "required",
            }
            for region_id in ("first", "middle", "last")
        ],
        sequential_reconstruction={
            "linkage_type": "shared_tracked_anchor",
            "linkage_region_ids": ["middle"],
            "observable_reason": "The middle member appears in both phases.",
        },
        virtual_camera_proposal=sequential,
        observed_evidence=["Three distinct members form one visible group."],
        decision_reason="The portrait camera covers the complete group over time.",
        confidence=0.8,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    assert proposal.virtual_camera_proposal == sequential


def test_selected_vertical_framing_runs_once_then_reuses_content_cache(
    tmp_path: Path,
) -> None:
    proposal = SelectedVerticalFramingProposal(
        candidate_id="legacy-primary",
        source_asset_id="sha256:" + "b" * 64,
        event_id="catalog-scene",
        frame_id="RF000001",
        semantic_requirement="single_primary",
        relation_temporal_mode="not_applicable",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            single="feasible",
        ),
        regions=[
            {
                "region_id": "primary",
                "target_description": "the directly visible primary subject",
                "evidence_role": "primary_subject",
                "role": "required",
            }
        ],
        virtual_camera_proposal=VerticalVirtualCameraProposal(
            composition_mode="single_anchor_follow",
            phases=[
                VerticalVirtualCameraProposalPhase(
                    phase_id="follow",
                    start_progress=0.0,
                    end_progress=1.0,
                    anchor_region_ids=["primary"],
                    camera_behavior="follow_deadband",
                    observable_predicate="The primary subject remains visible.",
                    transition_condition="No attention hand-off is observed.",
                    editorial_reason="Follow meaningful movement without micro-jitter.",
                )
            ],
            proposal_reason="One moving primary subject carries the shot.",
        ),
        observed_evidence=["One subject carries the visible action."],
        decision_reason="A tracked portrait crop can retain the observed action.",
        confidence=0.9,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    class FakeFramingClient:
        model_id = MODEL_ID
        upload_calls = 0
        proposal_calls = 0

        def ensure_video_upload(
            self, _path: Path, _upload_dir: Path
        ) -> tuple[SimpleNamespace, bool]:
            self.upload_calls += 1
            return (
                SimpleNamespace(uri="files/test", mime_type="video/mp4"),
                False,
            )

        def propose_selected_vertical_framing(
            self, **kwargs: object
        ) -> SelectedVerticalFramingProposal:
            self.proposal_calls += 1
            write_json(
                Path(str(kwargs["run_dir"])) / "selected_vertical_framing.json",
                proposal,
            )
            return proposal

    client = FakeFramingClient()
    clip = RushClip(
        clip_id="clip-1",
        path=str(tmp_path / "source.mp4"),
        sha256="b" * 64,
        duration_ms=5000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=1000,
        image_path=str(tmp_path / "frame.jpg"),
    )
    chapter = FeatureChapterBrief(
        feature_id="scene",
        title="Visible scene",
        detail_lines=[],
        target_duration_seconds=5,
    )
    option = {
        "candidate_id": "legacy-primary",
        "rank": 1,
        "source_asset_id": None,
        "event_id": None,
        "frame_id": frame.frame_id,
        "observed_visual_evidence": "One visible subject moves.",
        "selection_reason": "The action supports the chapter.",
        "strategy": "fit_with_background",
        "crop_mode": "strict",
        "target_description": None,
        "regions": [],
        "quality_risks": [],
        "confidence": 0.8,
    }

    first, first_proposal, first_reused = _refine_selected_vertical_candidate(
        client=client,  # type: ignore[arg-type]
        option_data=option,
        chapter=chapter,
        clip=clip,
        frame=frame,
        prompt_template="generic framing prompt",
        catalog_path=tmp_path / "catalog.json",
        output_dir=tmp_path / "artifacts",
        vertical_fallback_strategy="center_crop",
    )
    second, second_proposal, second_reused = _refine_selected_vertical_candidate(
        client=client,  # type: ignore[arg-type]
        option_data=option,
        chapter=chapter,
        clip=clip,
        frame=frame,
        prompt_template="generic framing prompt",
        catalog_path=tmp_path / "catalog.json",
        output_dir=tmp_path / "artifacts",
        vertical_fallback_strategy="center_crop",
    )

    assert first["strategy"] == "tracked_crop"
    assert first["virtual_camera_proposal"]["composition_mode"] == (
        "single_anchor_follow"
    )
    assert second == first | {
        "framing_refinement": {
            **first["framing_refinement"],
            "reused": True,
        }
    }
    assert first_proposal == second_proposal == proposal
    assert first_reused is False
    assert second_reused is True
    assert client.upload_calls == 1
    assert client.proposal_calls == 1


def test_fit_framing_preserves_but_does_not_execute_surplus_camera_proposal() -> None:
    surplus = VerticalVirtualCameraProposal(
        composition_mode="joint_relation",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="hold",
                start_progress=0.0,
                end_progress=1.0,
                anchor_region_ids=["left", "right"],
                observable_predicate="Both subjects are visible.",
                transition_condition="No transition is requested.",
                editorial_reason="A redundant camera idea accompanies a fit decision.",
            )
        ],
        proposal_reason="Surplus evidence that must not override fit_or_layout.",
    )
    proposal = SelectedVerticalFramingProposal(
        candidate_id="candidate-fit",
        source_asset_id="sha256:" + "c" * 64,
        event_id="event-fit",
        frame_id="RF000001",
        semantic_requirement="simultaneous_relation",
        relation_temporal_mode="simultaneous_required",
        recommended_action="fit_or_layout",
        presentation_options=_portrait_presentation_options(
            fit="feasible",
        ),
        regions=[
            {
                "region_id": "left",
                "target_description": "the left visible subject",
                "evidence_role": "relation_participant",
                "role": "required",
            },
            {
                "region_id": "right",
                "target_description": "the right visible subject",
                "evidence_role": "relation_participant",
                "role": "required",
            },
        ],
        virtual_camera_proposal=surplus,
        observed_evidence=["Both subjects form one comparison."],
        decision_reason="A portrait crop cannot preserve the relation.",
        confidence=0.8,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    assert proposal.recommended_action == "fit_or_layout"
    assert proposal.virtual_camera_proposal == surplus


def test_fit_framing_cannot_bypass_feasible_sequential_virtual_camera() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot bypass a feasible full-bleed presentation option",
    ):
        SelectedVerticalFramingProposal(
            candidate_id="candidate-premature-fit",
            source_asset_id="sha256:" + "c" * 64,
            event_id="event-comparison",
            frame_id="RF000001",
            semantic_requirement="simultaneous_relation",
            relation_temporal_mode="sequentially_reconstructable",
            recommended_action="fit_or_layout",
            regions=[
                {
                    "region_id": "first",
                    "target_description": "the first visible comparison subject",
                    "evidence_role": "relation_participant",
                    "role": "required",
                },
                {
                    "region_id": "second",
                    "target_description": "the second visible comparison subject",
                    "evidence_role": "relation_participant",
                    "role": "required",
                },
            ],
            presentation_options=_portrait_presentation_options(
                sequential="feasible",
                fit="feasible",
            ),
            observed_evidence=[
                "Both subjects can be inspected independently in one source shot."
            ],
            decision_reason="Sequential viewing remains a full-bleed option.",
            confidence=0.8,
            model_provenance=ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="test",
                generated_at="test",
            ),
        )


def test_strictly_simultaneous_relation_rejects_sequential_focus() -> None:
    with pytest.raises(
        ValidationError,
        match="strictly simultaneous relation cannot use sequential focus",
    ):
        SelectedVerticalFramingProposal(
            candidate_id="candidate-contact",
            source_asset_id="sha256:" + "d" * 64,
            event_id="event-contact",
            frame_id="RF000001",
            semantic_requirement="simultaneous_relation",
            relation_temporal_mode="simultaneous_required",
            recommended_action="tracked_crop",
            regions=[
                {
                    "region_id": "giver",
                    "target_description": "the visible giver at contact",
                    "evidence_role": "relation_participant",
                    "role": "required",
                },
                {
                    "region_id": "receiver",
                    "target_description": "the visible receiver at contact",
                    "evidence_role": "relation_participant",
                    "role": "required",
                },
            ],
            virtual_camera_proposal=VerticalVirtualCameraProposal(
                composition_mode="sequential_focus",
                phases=[
                    VerticalVirtualCameraProposalPhase(
                        phase_id="giver",
                        start_progress=0.0,
                        end_progress=0.5,
                        anchor_region_ids=["giver"],
                        observable_predicate="The giver is visible.",
                        transition_condition="The receiver becomes relevant.",
                        editorial_reason="Show the giver.",
                    ),
                        VerticalVirtualCameraProposalPhase(
                            phase_id="receiver",
                        start_progress=0.5,
                        end_progress=1.0,
                        anchor_region_ids=["receiver"],
                        observable_predicate="The receiver is visible.",
                            transition_condition="Hold to the end.",
                            cut_admissible=True,
                            editorial_reason="Show the receiver.",
                    ),
                ],
                proposal_reason="This would hide the required contact relation.",
            ),
            presentation_options=_portrait_presentation_options(
                sequential="feasible",
            ),
            observed_evidence=["The transfer exists only while both people touch it."],
            decision_reason="The exact contact must remain simultaneous.",
            confidence=0.8,
            model_provenance=ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="test",
                generated_at="test",
            ),
        )


def test_phase_mixed_relation_requires_joint_and_single_anchor_phases() -> None:
    proposal = SelectedVerticalFramingProposal(
        candidate_id="candidate-mixed-interaction",
        source_asset_id="sha256:" + "f" * 64,
        event_id="event-mixed-interaction",
        frame_id="RF000001",
        semantic_requirement="simultaneous_relation",
        relation_temporal_mode="phase_mixed",
        recommended_action="tracked_crop",
        regions=[
            {
                "region_id": "speaker",
                "target_description": "the visible speaking participant",
                "evidence_role": "relation_participant",
                "role": "required",
            },
            {
                "region_id": "listener",
                "target_description": "the visible reacting participant",
                "evidence_role": "relation_participant",
                "role": "required",
            },
        ],
        virtual_camera_proposal=VerticalVirtualCameraProposal(
            composition_mode="mixed_relation",
            phases=[
                VerticalVirtualCameraProposalPhase(
                    phase_id="shared-reaction",
                    start_progress=0.0,
                    end_progress=0.4,
                    anchor_region_ids=["speaker", "listener"],
                    observable_predicate="Both participants visibly react together.",
                    transition_condition="One participant begins a solo answer.",
                    editorial_reason="Preserve the simultaneous reaction.",
                ),
                    VerticalVirtualCameraProposalPhase(
                        phase_id="solo-answer",
                    start_progress=0.4,
                    end_progress=1.0,
                    anchor_region_ids=["speaker"],
                    observable_predicate="The speaker continues the answer.",
                        transition_condition="Hold through the end.",
                        cut_admissible=True,
                        editorial_reason="Use a readable single-person portrait.",
                ),
            ],
            proposal_reason="The interaction mixes joint and solo evidence.",
        ),
        sequential_reconstruction={
            "linkage_type": "shared_tracked_anchor",
            "linkage_region_ids": ["speaker"],
            "observable_reason": "The speaker remains visible across both phases.",
        },
        presentation_options=_portrait_presentation_options(
            sequential="feasible",
        ),
        observed_evidence=[
            "A shared reaction is followed by a solo answer in the same shot."
        ],
        decision_reason="The temporal requirement changes between phases.",
        confidence=0.8,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    assert proposal.virtual_camera_proposal is not None
    assert proposal.virtual_camera_proposal.composition_mode == "mixed_relation"


def test_uncertain_relation_cannot_authorize_tracked_crop() -> None:
    with pytest.raises(
        ValidationError,
        match="uncertain temporal relation cannot authorize tracked crop",
    ):
        SelectedVerticalFramingProposal(
            candidate_id="candidate-uncertain",
            source_asset_id="sha256:" + "0" * 64,
            event_id="event-uncertain",
            frame_id="RF000001",
            semantic_requirement="simultaneous_relation",
            relation_temporal_mode="uncertain",
            recommended_action="tracked_crop",
            regions=[
                {
                    "region_id": "subject",
                    "target_description": "the visible subject",
                    "evidence_role": "primary_subject",
                    "role": "required",
                }
            ],
            virtual_camera_proposal=VerticalVirtualCameraProposal(
                composition_mode="single_anchor_hold",
                phases=[
                    VerticalVirtualCameraProposalPhase(
                        phase_id="hold",
                        start_progress=0.0,
                        end_progress=1.0,
                        anchor_region_ids=["subject"],
                        observable_predicate="The subject remains visible.",
                        transition_condition="No reliable transition is visible.",
                        editorial_reason="This crop must not be authorized.",
                    )
                ],
                proposal_reason="The relation evidence is insufficient.",
            ),
            presentation_options=_portrait_presentation_options(
                single="feasible",
            ),
            observed_evidence=["The wider relationship cannot be confirmed."],
            decision_reason="Insufficient evidence requires review.",
            confidence=0.5,
            model_provenance=ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="test",
                generated_at="test",
            ),
        )


def test_virtual_camera_proposal_cannot_reference_untracked_or_fit_regions() -> None:
    proposal = VerticalVirtualCameraProposal(
        composition_mode="single_anchor_hold",
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id="only",
                start_progress=0.0,
                end_progress=1.0,
                anchor_region_ids=["missing"],
                observable_predicate="The visible subject remains the focus.",
                transition_condition="No hand-off is required.",
                editorial_reason="Keep a stable composition.",
            )
        ],
        proposal_reason="One stable visible subject.",
    )
    base = {
        "candidate_id": "generic-take",
        "rank": 1,
        "source_asset_id": "sha256:" + "a" * 64,
        "event_id": "event-generic",
        "frame_id": "RF000001",
        "observed_visual_evidence": "One visible subject.",
        "selection_reason": "Stable composition.",
        "regions": [
            FramingRegionIntent(
                region_id="visible",
                target_description="the directly visible subject",
            )
        ],
        "virtual_camera_proposal": proposal,
        "confidence": 0.8,
    }
    with pytest.raises(ValidationError, match="references unknown regions"):
        FeatureVerticalCandidate(strategy="tracked_crop", **base)
    with pytest.raises(ValidationError, match="tracked_crop"):
        FeatureVerticalCandidate(strategy="fit_with_background", **base)


def test_vertical_candidate_allows_controlled_required_clipping_and_fit_regions() -> None:
    common = {
        "candidate_id": "generic-take",
        "rank": 1,
        "source_asset_id": "sha256:" + "a" * 64,
        "event_id": "event-generic",
        "frame_id": "RF000001",
        "observed_visual_evidence": "One visible subject.",
        "selection_reason": "Stable composition.",
        "regions": [
            FramingRegionIntent(
                region_id="visible",
                target_description="the directly visible non-atomic subject",
                role="required",
                atomic=False,
                minimum_visible_fraction=0.8,
            )
        ],
        "confidence": 0.8,
    }

    controlled = FeatureVerticalCandidate(
        strategy="tracked_crop",
        crop_mode="primary_center",
        **common,
    )
    assert controlled.regions[0].effective_minimum_visible_fraction == 0.8
    with pytest.raises(ValidationError, match="strict tracked crops"):
        FeatureVerticalCandidate(
            strategy="tracked_crop",
            crop_mode="strict",
            **common,
        )
    fit = FeatureVerticalCandidate(
        strategy="fit_with_background",
        crop_mode="primary_center",
        **common,
    )
    assert fit.regions[0].region_id == "visible"


def test_runtime_vertical_candidate_ignores_transient_audit_fields() -> None:
    candidate = FeatureVerticalCandidate(
        candidate_id="generic-take",
        rank=1,
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-generic",
        frame_id="RF000001",
        observed_visual_evidence="One directly visible subject.",
        selection_reason="The selected evidence supports a tracked portrait crop.",
        strategy="tracked_crop",
        regions=[
            FramingRegionIntent(
                region_id="visible",
                target_description="the directly visible subject",
            )
        ],
        confidence=0.8,
    )
    runtime_option = candidate.model_dump(mode="python")
    runtime_option.update(
        {
            "coverage_intent": "single_primary",
            "coverage_target_descriptions": ["the directly visible subject"],
            "framing_refinement": {"status": "accepted"},
        }
    )

    parsed = _feature_vertical_candidate_from_runtime_option(runtime_option)

    assert parsed == candidate


def test_canonical_feature_plan_rejects_topk_aliases_of_same_evidence() -> None:
    candidates = [
        FeatureVerticalCandidate(
            candidate_id=f"alias-{rank}",
            rank=rank,
            source_asset_id="sha256:" + "a" * 64,
            event_id="event-shared",
            frame_id="RF000001",
            observed_visual_evidence="Same directly visible evidence.",
            selection_reason="Alias should not create another paid attempt.",
            strategy="fit_with_background",
            target_description=None,
            confidence=0.8,
        )
        for rank in (1, 2)
    ]

    with pytest.raises(ValidationError, match="distinct evidence frames"):
        FeatureChapterSelect(
            feature_id="scene",
            evidence_status="supported",
            horizontal_frame_id="RF000001",
            vertical_frame_id="RF000001",
            observed_visual_evidence="Same directly visible evidence.",
            selection_reason="One evidence frame cannot become two Top-K choices.",
            horizontal_strategy="original",
            horizontal_zoom_intent="none",
            horizontal_target_description=None,
            vertical_strategy="fit_with_background",
            vertical_target_description=None,
            quality_risks=[],
            confidence=0.8,
            vertical_candidates=candidates,
        )


@pytest.mark.parametrize(
    (
        "source_dimensions",
        "output_dimensions",
        "expected_scaled_dimensions",
        "expected_pan_axes",
    ),
    [
        ((1440, 1080), (1080, 1920), (2560, 1920), ["x"]),
        ((1080, 2400), (1080, 1920), (1080, 2400), ["y"]),
        ((1080, 1920), (1080, 1920), (1080, 1920), []),
        ((1440, 1080), (1920, 1080), (1920, 1440), ["y"]),
        ((2560, 1080), (1920, 1080), (2560, 1080), ["x"]),
    ],
)
def test_cover_transform_preserves_source_aspect_for_general_source_shapes(
    source_dimensions: tuple[int, int],
    output_dimensions: tuple[int, int],
    expected_scaled_dimensions: tuple[int, int],
    expected_pan_axes: list[str],
) -> None:
    transform = _cover_transform(*source_dimensions, *output_dimensions)

    assert (transform["scaled_width"], transform["scaled_height"]) == (
        expected_scaled_dimensions
    )
    assert transform["active_pan_axes"] == expected_pan_axes
    assert transform["aspect_ratio_relative_error"] < 0.001
    assert transform["normalized_track_space"] == (
        "orientation_corrected_source_0_1000"
    )


def test_portrait_source_uses_y_crop_and_preserves_required_region() -> None:
    boxes = [[400, 650, 600, 850], [400, 100, 600, 300]]
    x_values, y_values, audit = _tracked_crop_geometry(
        [0.0, 1.0],
        [500.0, 500.0],
        boxes,
        source_width=1080,
        source_height=2400,
        output_width=1080,
        output_height=1920,
    )

    assert x_values == [0.0, 0.0]
    assert y_values[0] > y_values[1]
    assert audit["crop_coordinate_space"]["active_pan_axes"] == ["y"]
    assert audit["crop_width_normalized"] == 1000
    assert audit["crop_height_normalized"] == 800
    assert audit["containment_failure_count"] == 0
    assert all(
        keyframe["required_union_contained"]
        for keyframe in audit["crop_keyframes"]
    )


def test_four_by_three_source_uses_x_crop_without_stretching() -> None:
    boxes = [[80, 200, 280, 800], [720, 200, 920, 800]]
    x_values, y_values, audit = _tracked_crop_geometry(
        [0.0, 1.0],
        [180.0, 820.0],
        boxes,
        source_width=1440,
        source_height=1080,
        output_width=1080,
        output_height=1920,
    )

    assert x_values[0] < x_values[1]
    assert y_values == [0.0, 0.0]
    coordinate_space = audit["crop_coordinate_space"]
    assert (coordinate_space["scaled_width"], coordinate_space["scaled_height"]) == (
        2560,
        1920,
    )
    assert coordinate_space["active_pan_axes"] == ["x"]
    assert coordinate_space["aspect_ratio_relative_error"] == 0
    assert audit["containment_failure_count"] == 0


def test_preserve_all_rejects_required_region_too_tall_for_viewport() -> None:
    _, _, audit = _tracked_crop_geometry(
        [0.0, 1.0],
        [500.0, 500.0],
        [[400, 50, 600, 950], [400, 50, 600, 950]],
        source_width=1080,
        source_height=2400,
        output_width=1080,
        output_height=1920,
    )

    assert audit["crop_height_normalized"] == 800
    assert audit["full_containment_feasible"] is False
    assert audit["geometry_feasible"] is False
    assert audit["containment_failure_count"] == 2


def test_projected_vertical_crop_contains_fast_moving_required_region() -> None:
    boxes = [
        [80, 100, 220, 900],
        [720, 100, 860, 900],
        [120, 100, 260, 900],
    ]
    x_values, audit = _vertical_crop_geometry(
        [0.0, 0.5, 1.0],
        [150.0, 790.0, 190.0],
        boxes,
        safety_multiplier=1.08,
    )

    assert len(x_values) == len(boxes)
    assert audit["geometry_feasible"] is True
    assert audit["full_containment_feasible"] is True
    assert audit["containment_failure_count"] == 0
    assert all(
        keyframe["required_union_contained"]
        for keyframe in audit["crop_keyframes"]
    )


def test_vertical_crop_audit_flags_source_boundary_contact() -> None:
    _, audit = _vertical_crop_geometry(
        [0.0, 0.5],
        [500.0, 500.0],
        [[100, 0, 300, 900], [700, 100, 1000, 900]],
        overflow_policy="controlled_clip",
    )

    assert audit["source_boundary_contact_count"] == 2
    assert audit["source_x_edge_contact_count"] == 1
    assert audit["source_y_edge_contact_count"] == 1
    assert audit["source_boundary_contact_ratio"] == 1.0


def test_cached_primary_track_requires_both_grounding_and_track(tmp_path: Path) -> None:
    root = tmp_path / "primary"
    assert _has_complete_cached_primary_track(root) is False
    grounding = root / "grounding" / "bbox-a" / "grounding.json"
    grounding.parent.mkdir(parents=True)
    grounding.write_text("{}", encoding="utf-8")
    assert _has_complete_cached_primary_track(root) is False
    track = root / "sam21" / "bbox-b" / "segmentation-track.json"
    track.parent.mkdir(parents=True)
    track.write_text("{}", encoding="utf-8")
    assert _has_complete_cached_primary_track(root) is True


def test_shared_sam_cache_revalidates_hashed_track_and_frame_lineage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    session_dir = tmp_path / "session"
    frames_dir = session_dir / "analysis-frames"
    frames_dir.mkdir(parents=True)
    frame_records: list[SharedSam21AnalysisFrame] = []
    for index, time_ms in enumerate((0, 500)):
        path = frames_dir / f"{index:06d}.jpg"
        path.write_bytes(f"frame-{index}".encode())
        frame_records.append(
            SharedSam21AnalysisFrame(
                sample_index=index,
                analysis_sample_time_ms=time_ms,
                source_pts=time_ms,
                path=f"analysis-frames/{path.name}",
                sha256=sha256_file(path),
            )
        )
    frames_manifest_path = session_dir / "analysis-frames-manifest.json"
    write_json(
        frames_manifest_path,
        {
            "timing_basis": "decoded_source_pts",
            "frames": [frame.model_dump(mode="json") for frame in frame_records],
        },
    )
    frames_manifest_sha256 = sha256_file(frames_manifest_path)
    checkpoint_sha256 = "c" * 64
    provenance = SegmentationModelProvenance(
        model_id=SAM21_TINY_MODEL_ID,
        implementation="facebookresearch/sam2",
        implementation_revision=SAM21_IMPLEMENTATION_REVISION,
        checkpoint_sha256=checkpoint_sha256,
        device="cpu",
        torch_version="test",
        generated_at="2026-07-22T00:00:00Z",
    )
    seeds = [
        SharedSam21BBoxSeed(
            target_id=f"region-{index}",
            target_description=f"required region {index}",
            seed_source=f"grounding-{index}.json",
            seed_time_ms=0,
            seed_frame_pts=0,
            seed_frame_sha256=str(index) * 64,
            seed_source_width=1920,
            seed_source_height=1080,
            seed_box_2d=box,
        )
        for index, box in enumerate(
            ([100, 100, 300, 800], [600, 100, 800, 800]), start=1
        )
    ]
    members: list[SharedSam21SessionTarget] = []
    for seed in seeds:
        samples = [
            SegmentationSample(
                sample_index=index,
                analysis_sample_time_ms=time_ms,
                source_pts=time_ms,
                timing_basis="decoded_source_pts",
                mask_path=f"masks/{index:06d}.png",
                mask_sha256="d" * 64,
                mask_area_pixels=100,
                mask_area_ratio=0.01,
                connected_components=1,
                derived_tracking_box=seed.seed_box_2d,
                center_2d=[
                    (seed.seed_box_2d[0] + seed.seed_box_2d[2]) / 2,
                    (seed.seed_box_2d[1] + seed.seed_box_2d[3]) / 2,
                ],
                mean_positive_probability=0.9,
                scene_cut_score=None,
                shot_boundary=False,
                tracking_state=TrackingState.TRACKED,
                state_reasons=[],
                semantic_identity_status=SemanticIdentityStatus.NOT_REVALIDATED,
            )
            for index, time_ms in enumerate((0, 500))
        ]
        track = SegmentationTrack(
            method="bbox_seed_sam2_video_mask_propagation",
            asset_id="sha256:" + "a" * 64,
            video_path=str(source.resolve()),
            target_description=seed.target_description,
            seed_source=seed.seed_source,
            seed_time_ms=seed.seed_time_ms,
            seed_sample_index=0,
            seed_frame_pts=seed.seed_frame_pts,
            seed_frame_sha256=seed.seed_frame_sha256,
            seed_source_width=seed.seed_source_width,
            seed_source_height=seed.seed_source_height,
            semantic_seed_box=seed.seed_box_2d,
            seed_prompt_type="box",
            sam_prompt_box=pad_normalized_box(seed.seed_box_2d, 0.04),
            sam_prompt_mask_polygon_xy=None,
            seed_box_padding_ratio=0.04,
            refined_seed_mask_path="masks/000000.png",
            analysis_fps=2,
            analysis_width=320,
            analysis_height=180,
            analysis_start_ms=0,
            analysis_end_ms=1000,
            source_start_pts=0,
            source_time_base={"numerator": 1, "denominator": 1000},
            timing_warning="test",
            semantic_warning="test",
            total_samples=2,
            state_counts={TrackingState.TRACKED: 2},
            elapsed_seconds=0,
            effective_fps=2,
            model_provenance=provenance,
            samples=samples,
            target_id=seed.target_id,
            shared_session_id="session-1",
            analysis_frames_manifest_sha256=frames_manifest_sha256,
        )
        track_path = (
            session_dir / "targets" / seed.target_id / "segmentation-track.json"
        )
        track_path.parent.mkdir(parents=True)
        write_json(track_path, track)
        members.append(
            SharedSam21SessionTarget(
                target_id=seed.target_id,
                target_description=seed.target_description,
                seed_time_ms=seed.seed_time_ms,
                seed_sample_index=0,
                seed_frame_pts=seed.seed_frame_pts,
                seed_frame_sha256=seed.seed_frame_sha256,
                seed_source_width=seed.seed_source_width,
                seed_source_height=seed.seed_source_height,
                track_path=str(track_path.relative_to(session_dir)),
                track_sha256=sha256_file(track_path),
                state_counts={TrackingState.TRACKED: 2},
            )
        )
    mismatched_state_counts = track.model_dump(mode="json")
    mismatched_state_counts["state_counts"] = {"low_confidence": 2}
    with pytest.raises(
        ValidationError, match="state_counts must match sample tracking_state values"
    ):
        SegmentationTrack.model_validate(mismatched_state_counts)
    manifest = SharedSam21SessionManifest(
        artifact_type="shared_sam21_multi_object_tracking_session",
        method="bbox_seed_shared_sam2_video_mask_propagation",
        session_id="session-1",
        asset_id="sha256:" + "a" * 64,
        video_path=str(source.resolve()),
        shot_id="shot-1",
        analysis_fps=2,
        analysis_width=320,
        analysis_height=180,
        analysis_start_ms=0,
        analysis_end_ms=1000,
        source_start_pts=0,
        source_time_base={"numerator": 1, "denominator": 1000},
        analysis_frames_path=frames_manifest_path.name,
        analysis_frames_manifest_sha256=frames_manifest_sha256,
        analysis_frames=frame_records,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        target_count=2,
        targets=members,
        model_provenance=provenance,
        timing=SharedSam21SessionTiming(
            shot_detection_seconds=0,
            analysis_frame_extraction_seconds=0,
            predictor_initialization_seconds=0,
            prompt_seconds=0,
            forward_propagation_seconds=0,
            reverse_propagation_seconds=0,
            target_artifact_seconds=0,
            total_seconds=0,
        ),
        warning="test",
        generated_at="2026-07-22T00:00:00Z",
    )

    tracks = _validate_shared_sam_session_cache(
        manifest=manifest,
        session_dir=session_dir,
        video_path=source,
        asset_id=manifest.asset_id,
        start_ms=0,
        end_ms=1000,
        analysis_fps=2,
        analysis_max_side=960,
        checkpoint_sha256=checkpoint_sha256,
        seeds=seeds,
        seed_box_padding_ratio=0.04,
    )
    assert [track.target_id for track in tracks] == ["region-1", "region-2"]

    first_track_path = session_dir / manifest.targets[0].track_path
    first_track_path.write_bytes(first_track_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="track hash mismatch"):
        _validate_shared_sam_session_cache(
            manifest=manifest,
            session_dir=session_dir,
            video_path=source,
            asset_id=manifest.asset_id,
            start_ms=0,
            end_ms=1000,
            analysis_fps=2,
            analysis_max_side=960,
            checkpoint_sha256=checkpoint_sha256,
            seeds=seeds,
            seed_box_padding_ratio=0.04,
        )


def test_only_non_retryable_spending_cap_errors_trip_geometry_circuit_breaker() -> None:
    assert _is_non_retryable_spending_cap_error(
        RuntimeError("project exceeded its monthly spending cap")
    )
    assert not _is_non_retryable_spending_cap_error(
        RuntimeError("429 transient requests per minute quota")
    )
    assert _is_exhausted_model_quota_error(
        RuntimeError("429 transient requests per minute quota")
    )
    assert _is_exhausted_model_quota_error(
        RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")
    )
    assert not _is_exhausted_model_quota_error(
        ValueError("the selected feature429 marker is not visible")
    )
    assert not _is_exhausted_model_quota_error(
        RuntimeError("the selected entity is not visible in this frame")
    )


def test_controlled_clip_can_preserve_trailing_edge_without_claiming_containment() -> None:
    x_values, audit = _vertical_crop_geometry(
        [0.0, 0.5],
        [500.0, 500.0],
        [[100, 100, 900, 900], [100, 100, 900, 900]],
        overflow_policy="controlled_clip",
        edge_priority="preserve_end",
    )
    crop_width = audit["crop_width_normalized"]
    crop_right = x_values[0] * 1000 / 3414 + crop_width

    assert crop_right == pytest.approx(900, abs=0.01)
    assert audit["controlled_clip_applied"] is True
    assert audit["full_containment_feasible"] is False
    assert audit["containment_failure_count"] == 2
    assert 0 < audit["minimum_visible_required_width_fraction"] < 1


def test_incomplete_tracking_can_hold_grounded_seed_without_centering_the_source() -> None:
    track = SimpleNamespace(
        seed_time_ms=500,
        semantic_seed_box=[300, 100, 730, 800],
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        analysis_start_ms=0,
        analysis_end_ms=2000,
        analysis_fps=2.0,
        target_description="the complete visible required region",
        state_counts={"drift_suspected": 4},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.DRIFT_SUSPECTED,
                derived_tracking_box=[300, 100, 730, 800],
            )
            for index in range(4)
        ],
    )

    filter_graph, audit = _vertical_filter_from_track(  # type: ignore[arg-type]
        [track],
        allow_subject_clipping=True,
        overflow_policy="controlled_clip",
        edge_priority="preserve_end",
        fallback_strategy="center_crop",
    )

    expected_crop_left = 730 - audit["crop_width_normalized"]
    actual_crop_left = (
        audit["crop_keyframes"][0]["crop_x_pixels"] * 1000 / 3414
    )
    assert "x='" in filter_graph
    assert audit["applied_strategy"] == "seed_anchor_crop"
    assert audit["coverage_passed"] is False
    assert audit["requires_gemini_review"] is True
    assert "motion_outside_seed_unverified" in audit["risk_codes"]
    assert actual_crop_left == pytest.approx(expected_crop_left, abs=0.01)


def test_required_track_union_combines_independent_regions_and_flags_missing_samples() -> None:
    def track(target: str, boxes: list[list[int] | None]) -> SimpleNamespace:
        samples = []
        for index, box in enumerate(boxes):
            samples.append(
                SimpleNamespace(
                    analysis_sample_time_ms=index * 500,
                    tracking_state="tracked" if box is not None else "lost",
                    derived_tracking_box=box,
                )
            )
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=2000,
            analysis_fps=2.0,
            target_description=target,
            state_counts={"tracked": sum(box is not None for box in boxes)},
            samples=samples,
        )

    left = track("left performer", [[100, 100, 250, 900]] * 4)
    right = track("right performer", [[600, 100, 760, 900]] * 4)
    times, centers, boxes, coverage = _required_track_union(  # type: ignore[arg-type]
        [left, right], region_ids=["left", "right"]
    )
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert centers == [430.0] * 4
    assert boxes == [[100, 100, 760, 900]] * 4
    assert coverage["coverage_passed"] is True
    assert coverage["expected_sample_interval_ms"] == 500.0

    missing = track(
        "right performer",
        [[600, 100, 760, 900], None, [600, 100, 760, 900], None],
    )
    _, _, _, failed = _required_track_union(  # type: ignore[arg-type]
        [left, missing], region_ids=["left", "right"]
    )
    assert failed["coverage_passed"] is False
    assert failed["unavailable_required_sample_count"] == 2


def test_required_track_union_fails_closed_on_any_low_confidence_sample() -> None:
    samples = [
        SimpleNamespace(
            analysis_sample_time_ms=index * 500,
            tracking_state=(
                TrackingState.LOW_CONFIDENCE
                if index == 5
                else TrackingState.TRACKED
            ),
            derived_tracking_box=[300, 100, 600, 900],
        )
        for index in range(10)
    ]
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=5000,
        analysis_fps=2.0,
        seed_time_ms=0,
        semantic_seed_box=[300, 100, 600, 900],
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        target_description="the required visible region",
        state_counts={"tracked": 9, "low_confidence": 1},
        samples=samples,
    )

    _, _, _, coverage = _required_track_union([track])  # type: ignore[arg-type]
    _, audit = _vertical_filter_from_track([track])  # type: ignore[list-item]

    assert coverage["unavailable_required_sample_ratio"] == pytest.approx(0.1)
    assert coverage["tracking_confidence_gate_passed"] is False
    assert coverage["low_confidence_required_sample_count"] == 1
    assert coverage["coverage_passed"] is False
    assert audit["applied_strategy"] == "fit_with_background"
    assert audit["fallback_reason"] == "required_region_tracking_confidence_failed"
    assert "required_region_low_confidence" in audit["risk_codes"]
    assert audit["requires_gemini_review"] is True


def test_primary_center_relaxes_margin_but_never_clips_primary_target() -> None:
    strict_fits, strict_margin = _vertical_target_fits_crop(
        310.0, 316.3445, primary_center=False
    )
    primary_fits, primary_margin = _vertical_target_fits_crop(
        310.0, 316.3445, primary_center=True
    )
    too_wide, _ = _vertical_target_fits_crop(
        320.0, 316.3445, primary_center=True
    )

    assert strict_fits is False
    assert strict_margin == 1.08
    assert primary_fits is True
    assert primary_margin == 1.0
    assert too_wide is False


def test_tracking_seed_moves_inside_a_trim_that_excludes_catalog_anchor() -> None:
    frame = RushFrame(
        frame_id="RF000001",
        clip_id="clip-1",
        requested_time_ms=7500,
        image_path="/tmp/frame.jpg",
    )

    assert _tracking_seed_request_ms(frame, 1000, 4000) == (2500, "trim_midpoint")
    assert _tracking_seed_request_ms(frame, 7000, 8000) == (7500, "catalog_anchor")


def test_feature_cut_refuses_unreviewed_trim_decision(tmp_path) -> None:
    path = tmp_path / "proposed.json"
    decision = TrimIntentDecision(
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        shot_id="shot-0001",
        usable=False,
        first_included_frame=None,
        last_included_frame=None,
        exclusive_out_frame=None,
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=None,
        source_out_ms=None,
        source_in_pts=None,
        source_out_pts=None,
        handle_in_ms=None,
        handle_out_ms=None,
        tail_intent="uncertain",
        proposal_path="/tmp/proposal.json",
        catalog_path="/tmp/catalog.json",
    )
    write_json(path, decision)

    with pytest.raises(ValueError, match="human-approved"):
        _load_trim_decisions([path])


def test_feature_cut_preview_flag_still_refuses_unusable_proposal(tmp_path) -> None:
    path = tmp_path / "proposed.json"
    proposal = TrimIntentDecision(
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        shot_id="shot-0001",
        usable=False,
        first_included_frame=None,
        last_included_frame=None,
        exclusive_out_frame=None,
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=None,
        source_out_ms=None,
        source_in_pts=None,
        source_out_pts=None,
        handle_in_ms=None,
        handle_out_ms=None,
        tail_intent="uncertain",
        proposal_path="/tmp/proposal.json",
        catalog_path="/tmp/catalog.json",
    )
    write_json(path, proposal)

    with pytest.raises(ValueError, match="unreviewed proposed"):
        _load_trim_decisions([path], allow_proposed_preview=True)


def test_feature_cut_applies_only_matching_approved_trim_bounds(tmp_path) -> None:
    clip = RushClip(
        clip_id="clip-1",
        path="/tmp/source.mp4",
        sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=5000,
        image_path="/tmp/frame.jpg",
    )
    evidence = {
        "frame_id": "DF000001",
        "requested_time_ms": 3000,
        "frame_time_ms": 3003,
        "frame_pts": 90,
        "frame_hash": "b" * 64,
    }
    decision = TrimIntentDecision.model_validate(
        {
            "source_asset_id": "sha256:" + clip.sha256,
            "event_id": "event-1",
            "shot_id": "shot-0001",
            "usable": True,
            "first_included_frame": evidence,
            "last_included_frame": {**evidence, "frame_id": "DF000002", "frame_time_ms": 7007},
            "exclusive_out_frame": {
                **evidence,
                "frame_id": "DF000003",
                "frame_time_ms": 7250,
                "frame_pts": 220,
            },
            "hold_start_frame": None,
            "hold_end_frame": None,
            "source_in_ms": 3003,
            "source_out_ms": 7250,
            "source_in_pts": 90,
            "source_out_pts": 220,
            "handle_in_ms": 2250,
            "handle_out_ms": 8250,
            "tail_intent": "natural_pause",
            "approval_status": "approved",
            "requires_human_review": False,
            "human_review": {
                "reviewer": "reviewer",
                "reviewed_at": "2026-07-21T00:00:00Z",
                "decision": "approved",
                "notes": "verified",
            },
            "proposal_path": "/tmp/proposal.json",
            "catalog_path": "/tmp/catalog.json",
        }
    )
    proposed = TrimIntentDecision.model_validate(
        decision.model_copy(
            update={
                "approval_status": "proposed",
                "requires_human_review": True,
                "human_review": None,
            }
        ).model_dump(mode="json")
    )
    proposed_path = tmp_path / "usable-proposed.json"
    write_json(proposed_path, proposed)
    accepted = _load_trim_decisions(
        [proposed_path],
        allow_proposed_preview=True,
    )
    assert accepted[0][1].approval_status == "proposed"
    shot_cache = {
        clip.clip_id: ShotManifest(
            video_path=clip.path,
            duration_ms=clip.duration_ms,
            detector="test",
            threshold=4,
            generated_at="2026-07-21T00:00:00Z",
            boundaries=[],
            shots=[
                ShotSegment(
                    shot_id="shot-0001",
                    start_time_ms=0,
                    end_time_ms=10_000,
                    start_frame_pts=0,
                    boundary_source="video_start",
                    boundary_score=None,
                )
            ],
        )
    }

    start_ms, end_ms, shot_id, audit = _chapter_bounds_with_approved_trim(
        frame,
        clip,
        2.0,
        shot_cache,
        tmp_path,
        4.0,
        [(tmp_path / "approved.json", decision)],
    )

    assert (start_ms, end_ms, shot_id) == (3003, 7250, "shot-0001")
    assert audit["trim_method"] == "human_approved_frame_id_pts"
    assert audit["trim_event_id"] == "event-1"


def test_trim_decision_can_select_a_better_range_away_from_catalog_anchor(tmp_path) -> None:
    clip = RushClip(
        clip_id="clip-1",
        path="/tmp/source.mp4",
        sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=7500,
        image_path="/tmp/frame.jpg",
    )
    evidence = {
        "frame_id": "DF000001",
        "requested_time_ms": 1000,
        "frame_time_ms": 1001,
        "frame_pts": 30,
        "frame_hash": "b" * 64,
    }
    decision = TrimIntentDecision.model_validate(
        {
            "source_asset_id": "sha256:" + clip.sha256,
            "event_id": "event-1",
            "shot_id": "shot-0001",
            "usable": True,
            "first_included_frame": evidence,
            "last_included_frame": None,
            "exclusive_out_frame": {
                **evidence,
                "frame_id": "DF000002",
                "requested_time_ms": 4000,
                "frame_time_ms": 4004,
                "frame_pts": 120,
            },
            "hold_start_frame": None,
            "hold_end_frame": None,
            "source_in_ms": 1001,
            "source_out_ms": 4004,
            "source_in_pts": 30,
            "source_out_pts": 120,
            "handle_in_ms": 0,
            "handle_out_ms": 5000,
            "tail_intent": "natural_pause",
            "approval_status": "approved",
            "requires_human_review": False,
            "human_review": {
                "reviewer": "reviewer",
                "reviewed_at": "2026-07-21T00:00:00Z",
                "decision": "approved",
                "notes": "representative select precedes the coarse catalog anchor",
            },
            "proposal_path": "/tmp/proposal.json",
            "catalog_path": "/tmp/catalog.json",
        }
    )
    shot_cache = {
        clip.clip_id: ShotManifest(
            video_path=clip.path,
            duration_ms=clip.duration_ms,
            detector="test",
            threshold=4,
            generated_at="2026-07-21T00:00:00Z",
            boundaries=[],
            shots=[
                ShotSegment(
                    shot_id="shot-0001",
                    start_time_ms=0,
                    end_time_ms=10_000,
                    start_frame_pts=0,
                    boundary_source="video_start",
                    boundary_score=None,
                )
            ],
        )
    }

    start_ms, end_ms, _, _ = _chapter_bounds_with_approved_trim(
        frame,
        clip,
        2.0,
        shot_cache,
        tmp_path,
        4.0,
        [(tmp_path / "approved.json", decision)],
    )

    assert (start_ms, end_ms) == (1001, 4004)


def test_feature_brief_can_disable_titles_and_choose_primary_center_crop() -> None:
    brief = FeatureEditBrief(
        project_id="clean-cut",
        title="clean",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="hero",
                title="hero",
                detail_lines=[],
                target_duration_seconds=6,
                vertical_primary_target_description="reviewer-selected foreground subject",
                vertical_crop_mode="primary_center",
            )
        ],
    )
    assert brief.render_title_overlays is False
    assert brief.chapters[0].vertical_crop_mode == "primary_center"


def test_feature_brief_supports_generic_required_text_and_subject_regions() -> None:
    brief = FeatureChapterBrief(
        feature_id="mixed_scene",
        title="Preserve evidence",
        detail_lines=[],
        target_duration_seconds=4,
        vertical_regions=[
            FramingRegionIntent(
                region_id="speaker",
                target_description="the presenter nearest the lectern",
                kind="subject",
            ),
            FramingRegionIntent(
                region_id="heading",
                target_description="the complete visible heading on the sign",
                kind="text_region",
            ),
        ],
    )
    assert [region.kind for region in brief.vertical_regions] == [
        "subject",
        "text_region",
    ]

    with pytest.raises(ValidationError, match="edge priority"):
        FeatureChapterBrief(
            feature_id="invalid",
            title="invalid",
            detail_lines=[],
            target_duration_seconds=4,
            vertical_edge_priority="preserve_end",
        )


def test_feature_brief_can_forbid_blurred_vertical_fallback() -> None:
    brief = FeatureEditBrief(
        project_id="clean-cut",
        title="clean",
        target_duration_seconds=60,
        vertical_fallback_strategy="center_crop",
        chapters=[
            FeatureChapterBrief(
                feature_id="hero",
                title="hero",
                detail_lines=[],
                target_duration_seconds=6,
            )
        ],
    )
    assert brief.vertical_fallback_strategy == "center_crop"


def test_tracked_reframe_requires_target_and_nonzero_intent() -> None:
    payload = {
        "feature_id": "ui",
        "evidence_status": "supported",
        "horizontal_frame_id": "RF000001",
        "vertical_frame_id": "RF000002",
        "observed_visual_evidence": "selected subject remains visible",
        "selection_reason": "clear",
        "horizontal_strategy": "tracked_reframe",
        "horizontal_zoom_intent": "none",
        "horizontal_target_description": None,
        "vertical_strategy": "fit_with_background",
        "vertical_target_description": None,
        "quality_risks": [],
        "confidence": 0.9,
    }
    with pytest.raises(ValidationError, match="requires a zoom intent"):
        FeatureChapterSelect.model_validate(payload)


def test_piecewise_expression_is_ffmpeg_escaped() -> None:
    expression = _piecewise_expression([0.0, 0.5, 1.0], [100.0, 150.0, 130.0])
    assert "lt(t\\,0.500)" in expression
    assert "if(" in expression


def test_track_centers_are_rebased_and_exclude_low_confidence_geometry() -> None:
    track = SimpleNamespace(
        analysis_start_ms=5000,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=5100,
                tracking_state="tracked",
                center_2d=[300.0, 500.0],
                derived_tracking_box=[200, 200, 400, 800],
            ),
            SimpleNamespace(
                analysis_sample_time_ms=6100,
                tracking_state="low_confidence",
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 200, 600, 800],
            ),
        ],
    )
    times, centers, boxes = _usable_track_centers(track)  # type: ignore[arg-type]
    assert times == pytest.approx([0.1])
    assert centers == [300.0]
    assert boxes == [[200, 200, 400, 800]]


def test_horizontal_reframe_fails_closed_on_low_confidence_geometry() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=(
                    TrackingState.LOW_CONFIDENCE
                    if index == 1
                    else TrackingState.TRACKED
                ),
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 300, 600, 700],
            )
            for index in range(3)
        ],
    )

    filter_graph, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track, "subtle"
    )

    assert "scale=1920:1080" in filter_graph
    assert audit["applied_zoom"] == 1.0
    assert audit["fallback_reason"] == "tracking_confidence_gate_failed"
    assert audit["tracking_confidence_gate_passed"] is False
    assert audit["low_confidence_sample_count"] == 1
    assert audit["risk_codes"] == [
        "tracking_low_confidence",
        "requested_tracked_reframe_not_applied",
    ]
    assert audit["requires_gemini_review"] is True


def test_horizontal_reframe_fails_closed_on_source_lineage_mismatch() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=640,
        analysis_height=480,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 300, 600, 700],
            )
            for index in range(3)
        ],
    )

    _, audit = _horizontal_filter_from_track(track, "subtle")  # type: ignore[arg-type]

    assert audit["applied_zoom"] == 1.0
    assert audit["source_geometry_lineage_passed"] is False
    assert audit["fallback_reason"].endswith("analysis_aspect_disagrees")
    assert "track_source_geometry_mismatch" in audit["risk_codes"]
    assert audit["requires_gemini_review"] is True


def test_horizontal_four_by_three_reframe_tracks_in_both_crop_axes() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1440,
        seed_source_height=1080,
        analysis_width=640,
        analysis_height=480,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 350.0 + index * 100],
                derived_tracking_box=[400, 250 + index * 100, 600, 450 + index * 100],
            )
            for index in range(3)
        ],
    )

    filter_graph, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track, "subtle"
    )

    assert "scale=2152:1614" in filter_graph
    assert ":x='" in filter_graph and ":y='" in filter_graph
    assert audit["fallback_reason"] is None
    assert audit["full_containment_feasible"] is True
    assert audit["crop_coordinate_space"]["source_display_width"] == 1440
    assert audit["crop_coordinate_space"]["active_pan_axes"] == ["x", "y"]


def test_horizontal_push_in_builds_keyframed_virtual_camera() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        target_id="generic-subject",
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[420, 300, 580, 700],
            )
            for index in range(3)
        ],
    )

    filter_graph, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track,
        "subtle",
        camera_intent="push_in",
    )

    camera = audit["virtual_camera_plan"]
    assert "eval=frame" in filter_graph
    assert camera["requested_intent"] == "push_in"
    assert camera["applied_intent"] == "push_in"
    assert camera["execution_status"] == "applied"
    assert camera["anchor_target_ids"] == ["generic-subject"]
    assert camera["keyframes"][0]["scale"] == pytest.approx(1.0)
    assert camera["keyframes"][-1]["scale"] == pytest.approx(1.12)


def test_horizontal_hold_uses_one_static_safe_camera_when_feasible() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        target_id="moving-subject",
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                center_2d=[450.0 + index * 50, 500.0],
                derived_tracking_box=[
                    380 + index * 50,
                    300,
                    520 + index * 50,
                    700,
                ],
            )
            for index in range(3)
        ],
    )

    _, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track,
        "subtle",
        camera_intent="hold",
    )

    camera = audit["virtual_camera_plan"]
    assert camera["execution_status"] == "applied"
    assert camera["applied_intent"] == "hold"
    assert audit["stable_hold_feasible"] is True
    assert len(set(audit["crop_x_values_pixels"])) == 1
    assert len(set(audit["crop_y_values_pixels"])) == 1
    assert camera["max_velocity"] == 0


def test_pan_reveal_without_two_locks_falls_back_with_evidence() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        target_id="first-anchor-only",
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[420, 300, 580, 700],
            )
            for index in range(3)
        ],
    )

    _, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track,
        "subtle",
        camera_intent="pan_reveal",
    )

    camera = audit["virtual_camera_plan"]
    assert camera["requested_intent"] == "pan_reveal"
    assert camera["applied_intent"] == "follow"
    assert camera["execution_status"] == "fallback"
    assert camera["fallback_reason"] == (
        "pan_reveal_requires_two_independently_locked_anchors"
    )
    assert audit["requires_gemini_review"] is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_virtual_camera_filter_renders_a_playable_segment(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "virtual-camera.mp4"
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
            "testsrc2=size=640x360:rate=30:duration=0.6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=640,
        seed_source_height=360,
        analysis_width=640,
        analysis_height=360,
        target_id="render-subject",
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 300,
                source_pts=index * 9,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[420, 300, 580, 700],
            )
            for index in range(3)
        ],
    )
    filter_graph, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track,
        "subtle",
        camera_intent="push_in",
    )

    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=600,
        overlay_path=None,
        base_filter=filter_graph,
        output_path=output,
        source_has_audio=False,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert audit["virtual_camera_plan"]["execution_status"] == "applied"


def test_vertical_portrait_reframe_uses_track_driven_y_crop() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=1000,
        analysis_fps=2.0,
        seed_time_ms=0,
        semantic_seed_box=[400, 650, 600, 850],
        seed_source_width=1080,
        seed_source_height=2400,
        analysis_width=432,
        analysis_height=960,
        target_description="required visible region",
        state_counts={"tracked": 2},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=(
                    [400, 650, 600, 850]
                    if index == 0
                    else [400, 100, 600, 300]
                ),
            )
            for index in range(2)
        ],
    )

    filter_graph, audit = _vertical_filter_from_track(  # type: ignore[arg-type]
        [track]
    )

    assert "scale=1080:2400" in filter_graph
    assert ":x='" in filter_graph and ":y='" in filter_graph
    assert audit["applied_strategy"] == "tracked_crop"
    assert audit["crop_coordinate_space"]["active_pan_axes"] == ["y"]
    assert audit["crop_keyframes"][0]["crop_y_pixels"] > (
        audit["crop_keyframes"][1]["crop_y_pixels"]
    )
    assert audit["containment_failure_count"] == 0


def test_vertical_multi_region_reframe_rejects_disagreeing_seed_dimensions() -> None:
    def track(
        target: str,
        source_dimensions: tuple[int, int],
        analysis_dimensions: tuple[int, int],
        box: list[int],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=1000,
            analysis_fps=2.0,
            seed_time_ms=0,
            semantic_seed_box=box,
            seed_source_width=source_dimensions[0],
            seed_source_height=source_dimensions[1],
            analysis_width=analysis_dimensions[0],
            analysis_height=analysis_dimensions[1],
            target_description=target,
            state_counts={"tracked": 2},
            samples=[
                SimpleNamespace(
                    analysis_sample_time_ms=index * 500,
                    tracking_state=TrackingState.TRACKED,
                    derived_tracking_box=box,
                )
                for index in range(2)
            ],
        )

    _, audit = _vertical_filter_from_track(  # type: ignore[list-item]
        [
            track("left", (1920, 1080), (960, 540), [100, 100, 300, 900]),
            track("right", (1440, 1080), (640, 480), [600, 100, 800, 900]),
        ],
        region_ids=["left", "right"],
    )

    assert audit["applied_strategy"] == "fit_with_background"
    assert audit["source_geometry_lineage_passed"] is False
    assert audit["fallback_reason"].endswith("required_tracks_disagree")
    assert audit["risk_codes"] == ["track_source_geometry_mismatch"]
    assert audit["requires_gemini_review"] is True


def test_vertical_camera_phases_require_contiguous_known_region_anchors() -> None:
    regions = [
        FramingRegionIntent(
            region_id="left",
            target_description="the visible subject on the left",
            role="required",
        ),
        FramingRegionIntent(
            region_id="right",
            target_description="the visible subject on the right",
            role="required",
        ),
    ]
    phases = [
        VerticalVirtualCameraPhase(
            phase_id="right-first",
            start_progress=0.0,
            end_progress=0.45,
            anchor_region_ids=["right"],
            camera_behavior="hold",
            editorial_reason="Establish the result first.",
        ),
        VerticalVirtualCameraPhase(
            phase_id="left-second",
            start_progress=0.45,
            end_progress=1.0,
            anchor_region_ids=["left"],
            camera_behavior="follow",
            cut_admissible=True,
            transition_in="smoothstep",
            transition_duration_fraction=0.4,
            editorial_reason="Reveal the performer after the result.",
        ),
    ]

    chapter = FeatureChapterBrief(
        feature_id="generic_scene",
        title="Generic scene",
        detail_lines=[],
        target_duration_seconds=5,
        vertical_regions=regions,
        vertical_camera_phases=phases,
    )

    assert [phase.phase_id for phase in chapter.vertical_camera_phases] == [
        "right-first",
        "left-second",
    ]
    with pytest.raises(
        ValidationError,
        match="vertical camera phases reference unknown regions",
    ):
        FeatureChapterBrief(
            feature_id="generic_scene",
            title="Generic scene",
            detail_lines=[],
            target_duration_seconds=5,
            vertical_regions=regions,
            vertical_camera_phases=[
                phases[0],
                phases[1].model_copy(
                    update={"anchor_region_ids": ["missing"]}
                ),
            ],
        )
    with pytest.raises(
        ValidationError,
        match="cannot relax visibility for atomic",
    ):
        FeatureChapterBrief(
            feature_id="generic_scene",
            title="Generic scene",
            detail_lines=[],
            target_duration_seconds=5,
            vertical_regions=[
                regions[0].model_copy(update={"atomic": True}),
                regions[1],
            ],
            vertical_camera_phases=[
                phases[0],
                phases[1].model_copy(
                    update={"minimum_anchor_visible_fraction": 0.8}
                ),
            ],
        )


def test_phase_virtual_camera_moves_between_independent_tracked_anchors() -> None:
    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        samples = [
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=box,
            )
            for index in range(9)
        ]
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=4000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=samples,
            model_dump=lambda *, mode: {
                "target_id": target_id,
                "box": box,
                "mode": mode,
            },
        )

    phases = [
        VerticalVirtualCameraPhase(
            phase_id="right-first",
            start_progress=0.0,
            end_progress=0.5,
            anchor_region_ids=["right"],
            camera_behavior="hold",
            editorial_reason="Establish the right-side evidence.",
        ),
        VerticalVirtualCameraPhase(
            phase_id="left-second",
            start_progress=0.5,
            end_progress=1.0,
            anchor_region_ids=["left"],
            camera_behavior="hold",
            cut_admissible=True,
            transition_in="smoothstep",
            transition_duration_fraction=0.5,
            editorial_reason="Pan to the left-side evidence.",
        ),
    ]
    filter_graph, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "left": track("left", [100, 180, 260, 820]),
            "right": track("right", [400, 180, 560, 820]),
        },
        phases=phases,
    )

    assert "crop=1080:1920" in filter_graph
    assert audit["applied_strategy"] == "phase_virtual_camera"
    assert audit["minimum_visible_required_area_fraction"] == 1.0
    assert audit["transition_sample_count"] == 0
    assert audit["distance_aware_transition_audit"][0]["disposition"] == (
        "converted_to_cut_unmotivated"
    )
    assert audit["requires_gemini_review"] is True
    keyframes = audit["crop_keyframes"]
    assert keyframes[0]["phase_id"] == "right-first"
    assert keyframes[-1]["phase_id"] == "left-second"
    assert keyframes[0]["crop_x_pixels"] > keyframes[-1]["crop_x_pixels"]
    plan = audit["phase_virtual_camera_plan"]
    assert plan["anchor_region_ids"] == ["right", "left"]
    assert plan["execution_status"] == "applied"
    assert plan["max_velocity"] == 0.0
    assert plan["max_acceleration"] == 0.0
    assert plan["max_jerk"] == 0.0


def test_spatially_optimizable_virtual_camera_preserves_temporal_order() -> None:
    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        samples = [
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=box,
            )
            for index in range(13)
        ]
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=6000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=samples,
            model_dump=lambda *, mode: {
                "target_id": target_id,
                "box": box,
                "mode": mode,
            },
        )

    phases = [
        VerticalVirtualCameraPhase(
            phase_id="center",
            start_progress=0.0,
            end_progress=1 / 3,
            anchor_region_ids=["center"],
            camera_behavior="hold",
            movement_motivation="none",
            traversal_policy="spatially_optimizable",
            editorial_reason="Independent center subject.",
        ),
        VerticalVirtualCameraPhase(
            phase_id="left",
            start_progress=1 / 3,
            end_progress=2 / 3,
            anchor_region_ids=["left"],
            camera_behavior="hold",
            movement_motivation="attention_handoff",
            traversal_policy="spatially_optimizable",
            cut_admissible=True,
            transition_in="smoothstep",
            transition_duration_fraction=0.25,
            editorial_reason="Independent left subject.",
        ),
        VerticalVirtualCameraPhase(
            phase_id="right",
            start_progress=2 / 3,
            end_progress=1.0,
            anchor_region_ids=["right"],
            camera_behavior="hold",
            movement_motivation="attention_handoff",
            traversal_policy="spatially_optimizable",
            cut_admissible=True,
            transition_in="smoothstep",
            transition_duration_fraction=0.25,
            editorial_reason="Independent right subject.",
        ),
    ]

    _, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "left": track("left", [80, 300, 260, 520]),
            "center": track("center", [410, 300, 590, 520]),
            "right": track("right", [740, 300, 920, 520]),
        },
        phases=phases,
    )

    traversal = audit["traversal_audit"]
    assert traversal["original_phase_order"] == ["center", "left", "right"]
    assert traversal["effective_phase_order"] == ["center", "left", "right"]
    assert traversal["reordered"] is False
    assert (
        audit["motion_quality_audit"][
            "meaningful_direction_reversal_count"
        ]
        == 0
    )


def test_virtual_camera_suppresses_subperceptual_attention_handoff() -> None:
    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        samples = [
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=box,
            )
            for index in range(9)
        ]
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=4000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=samples,
            model_dump=lambda *, mode: {"target_id": target_id, "mode": mode},
        )

    _, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "a": track("a", [430, 300, 530, 520]),
            "b": track("b", [445, 300, 545, 520]),
        },
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="a",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["a"],
                camera_behavior="hold",
                editorial_reason="First nearby subject.",
            ),
            VerticalVirtualCameraPhase(
                phase_id="b",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["b"],
                camera_behavior="hold",
                movement_motivation="attention_handoff",
                cut_admissible=True,
                transition_in="smoothstep",
                transition_duration_fraction=0.25,
                editorial_reason="Second nearby subject.",
            ),
        ],
    )

    assert audit["transition_sample_count"] == 0
    assert audit["distance_aware_transition_audit"][0]["disposition"] == (
        "shared_hold_small_displacement"
    )
    assert audit["motion_quality_audit"]["no_gratuitous_motion_passed"] is True


def test_virtual_camera_uses_feasible_regions_instead_of_greedy_centers() -> None:
    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        samples = [
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=box,
            )
            for index in range(13)
        ]
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=6000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=samples,
            model_dump=lambda *, mode: {"target_id": target_id, "mode": mode},
        )

    _, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "establish": track("establish", [450, 300, 550, 520]),
            "left": track("left", [300, 300, 400, 520]),
            "right": track("right", [700, 300, 800, 520]),
        },
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="establish",
                start_progress=0.0,
                end_progress=1 / 3,
                anchor_region_ids=["establish"],
                camera_behavior="hold",
                editorial_reason="Establish context without mandatory centering.",
            ),
            VerticalVirtualCameraPhase(
                phase_id="left",
                start_progress=1 / 3,
                end_progress=2 / 3,
                anchor_region_ids=["left"],
                camera_behavior="hold",
                movement_motivation="attention_handoff",
                cut_admissible=True,
                transition_in="smoothstep",
                transition_duration_fraction=0.25,
                editorial_reason="Attend to the left evidence.",
            ),
            VerticalVirtualCameraPhase(
                phase_id="right",
                start_progress=2 / 3,
                end_progress=1.0,
                anchor_region_ids=["right"],
                camera_behavior="hold",
                movement_motivation="attention_handoff",
                cut_admissible=True,
                transition_in="smoothstep",
                transition_duration_fraction=0.25,
                editorial_reason="Attend to the right evidence.",
            ),
        ],
    )

    targets = {
        item["phase_id"]: item["optimized_camera_center_normalized"][0]
        for item in audit["traversal_audit"]["phase_target_audit"]
    }
    assert targets["establish"] == targets["left"]
    assert targets["establish"] < targets["right"]
    assert audit["traversal_audit"]["solver"] == (
        "minimum_variation_feasible_region_path_v1"
    )
    assert (
        audit["distance_aware_transition_audit"][0]["disposition"]
        == "shared_hold_small_displacement"
    )
    assert (
        audit["motion_quality_audit"][
            "meaningful_direction_reversal_count"
        ]
        == 0
    )


def test_small_required_camera_move_is_not_forced_into_hold() -> None:
    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=4000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=[
                SimpleNamespace(
                    analysis_sample_time_ms=index * 500,
                    source_pts=index * 15,
                    tracking_state=TrackingState.TRACKED,
                    derived_tracking_box=box,
                )
                for index in range(9)
            ],
            model_dump=lambda *, mode: {"target_id": target_id, "mode": mode},
        )

    _, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "a": track("a", [400, 300, 500, 520]),
            "b": track("b", [626, 300, 726, 520]),
        },
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="a",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["a"],
                camera_behavior="hold",
                editorial_reason="First hard anchor.",
            ),
            VerticalVirtualCameraPhase(
                phase_id="b",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["b"],
                camera_behavior="hold",
                movement_motivation="attention_handoff",
                cut_admissible=True,
                transition_in="smoothstep",
                transition_duration_fraction=0.5,
                editorial_reason="Second hard anchor requires a small correction.",
            ),
        ],
    )

    transition = audit["distance_aware_transition_audit"][0]
    assert transition["distance_pixels"] < transition[
        "minimum_perceptual_move_pixels"
    ]
    assert transition["shared_static_composition_feasible"] is False
    assert transition["disposition"] != "shared_hold_small_displacement"
    assert audit["crop_keyframes"][0]["crop_x_pixels"] != (
        audit["crop_keyframes"][-1]["crop_x_pixels"]
    )


def test_compiler_rejects_unapproved_automatic_hard_cut() -> None:
    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=4000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=[
                SimpleNamespace(
                    analysis_sample_time_ms=index * 500,
                    source_pts=index * 15,
                    tracking_state=TrackingState.TRACKED,
                    derived_tracking_box=box,
                )
                for index in range(9)
            ],
            model_dump=lambda *, mode: {"target_id": target_id, "mode": mode},
        )

    with pytest.raises(
        ValueError,
        match="did not prove that cut admissible",
    ):
        _vertical_virtual_camera_filter_from_tracks(
            tracks_by_region={
                "a": track("a", [100, 300, 220, 520]),
                "b": track("b", [760, 300, 880, 520]),
            },
            phases=[
                VerticalVirtualCameraPhase(
                    phase_id="a",
                    start_progress=0.0,
                    end_progress=0.5,
                    anchor_region_ids=["a"],
                    camera_behavior="hold",
                    editorial_reason="First hard anchor.",
                ),
                VerticalVirtualCameraPhase(
                    phase_id="b",
                    start_progress=0.5,
                    end_progress=1.0,
                    anchor_region_ids=["b"],
                    camera_behavior="hold",
                    movement_motivation="none",
                    cut_admissible=False,
                    transition_in="smoothstep",
                    transition_duration_fraction=0.25,
                    editorial_reason=(
                        "No semantic evidence authorizes cutting this boundary."
                    ),
                ),
            ],
        )


def test_motion_extrema_excludes_discontinuities_at_editorial_cuts() -> None:
    with_cut = feature_cut_module._motion_extrema(
        [0.0, 0.5, 1.0, 1.5],
        [100.0, 100.0, 900.0, 900.0],
        [200.0, 200.0, 200.0, 200.0],
        [1.0, 1.0, 1.0, 1.0],
        cut_before_indexes={2},
    )
    without_cut = feature_cut_module._motion_extrema(
        [0.0, 0.5, 1.0, 1.5],
        [100.0, 100.0, 900.0, 900.0],
        [200.0, 200.0, 200.0, 200.0],
        [1.0, 1.0, 1.0, 1.0],
    )

    assert with_cut == (0.0, 0.0, 0.0)
    assert without_cut[0] > 0.0


def test_virtual_camera_deadband_ignores_small_tracker_motion() -> None:
    prior = (500.0, 500.0)

    assert feature_cut_module._deadband_center(
        prior,
        (506.0, 496.0),
        deadband_x=10.0,
        deadband_y=10.0,
    ) == prior
    assert feature_cut_module._deadband_center(
        prior,
        (530.0, 500.0),
        deadband_x=10.0,
        deadband_y=10.0,
    ) == (520.0, 500.0)


def test_phase_virtual_camera_push_in_uses_resolution_bounded_scale() -> None:
    samples = [
        SimpleNamespace(
            analysis_sample_time_ms=index * 500,
            source_pts=index * 15,
            tracking_state=TrackingState.TRACKED,
            derived_tracking_box=[380, 220, 620, 780],
        )
        for index in range(7)
    ]
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=3000,
        analysis_fps=2.0,
        seed_source_width=3840,
        seed_source_height=2160,
        analysis_width=960,
        analysis_height=540,
        target_description="generic visible subject",
        target_id="subject",
        samples=samples,
        model_dump=lambda *, mode: {"target_id": "subject", "mode": mode},
    )

    filter_graph, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={"subject": track},
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="detail",
                start_progress=0.0,
                end_progress=1.0,
                anchor_region_ids=["subject"],
                camera_behavior="push_in",
                editorial_reason="Emphasize the observed result.",
            )
        ],
    )

    assert "eval=frame" in filter_graph
    assert audit["maximum_applied_zoom"] == pytest.approx(1.12)
    assert audit["crop_scale_values"][0] == pytest.approx(1.0)
    assert audit["crop_scale_values"][-1] == pytest.approx(1.12)
    assert audit["max_crop_speed_pixels_per_second"] <= 720


def test_phase_virtual_camera_allows_reviewed_non_atomic_clipping() -> None:
    samples = [
        SimpleNamespace(
            analysis_sample_time_ms=index * 500,
            source_pts=index * 15,
            tracking_state=TrackingState.TRACKED,
            derived_tracking_box=[250, 100, 650, 900],
        )
        for index in range(5)
    ]
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=2000,
        analysis_fps=2.0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        target_description="wide non-atomic subject",
        target_id="subject",
        samples=samples,
        model_dump=lambda *, mode: {"target_id": "subject", "mode": mode},
    )

    _, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={"subject": track},
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="reviewed-clip",
                start_progress=0.0,
                end_progress=1.0,
                anchor_region_ids=["subject"],
                minimum_anchor_visible_fraction=0.75,
                editorial_reason="The reviewed portrait composition may trim shoulders.",
            )
        ],
    )

    assert audit["applied_strategy"] == "phase_virtual_camera"
    assert audit["subject_clipping_allowed"] is True
    assert audit["full_containment_feasible"] is False
    assert audit["minimum_visible_required_area_fraction"] >= 0.75
    assert "phase_intentional_anchor_clipping" in audit["risk_codes"]


def test_phase_virtual_camera_interpolates_one_short_track_gap() -> None:
    def track(target_id: str, *, missing_index: int | None = None) -> SimpleNamespace:
        samples = [
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                source_pts=index * 15,
                tracking_state=(
                    TrackingState.LOW_CONFIDENCE
                    if index == missing_index
                    else TrackingState.TRACKED
                ),
                derived_tracking_box=(
                    None
                    if index == missing_index
                    else [100 + index, 180, 260 + index, 820]
                ),
            )
            for index in range(9)
        ]
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=4000,
            analysis_fps=2.0,
            seed_source_width=1920,
            seed_source_height=1080,
            analysis_width=960,
            analysis_height=540,
            target_description=target_id,
            target_id=target_id,
            samples=samples,
            model_dump=lambda *, mode: {"target_id": target_id, "mode": mode},
        )

    _, audit = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "first": track("first"),
            "second": track("second", missing_index=5),
        },
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="first",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["first"],
                editorial_reason="First subject.",
            ),
            VerticalVirtualCameraPhase(
                phase_id="second",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["second"],
                cut_admissible=True,
                transition_in="smoothstep",
                transition_duration_fraction=0.25,
                editorial_reason="Second subject.",
            ),
        ],
    )

    assert audit["interpolated_anchor_sample_count"] == 1
    assert audit["interpolated_anchor_sample_ratio"] < 0.15
    assert "phase_anchor_track_gap_interpolated" in audit["risk_codes"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_phase_virtual_camera_filter_renders_playable_portrait(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "portrait-pan.mp4"
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
            "testsrc2=size=640x360:rate=30:duration=1.1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    def track(target_id: str, box: list[int]) -> SimpleNamespace:
        samples = [
            SimpleNamespace(
                analysis_sample_time_ms=index * 250,
                source_pts=index * 8,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=box,
            )
            for index in range(5)
        ]
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=1000,
            analysis_fps=4.0,
            seed_source_width=640,
            seed_source_height=360,
            analysis_width=640,
            analysis_height=360,
            target_description=target_id,
            target_id=target_id,
            samples=samples,
            model_dump=lambda *, mode: {
                "target_id": target_id,
                "box": box,
                "mode": mode,
            },
        )

    filter_graph, _ = _vertical_virtual_camera_filter_from_tracks(
        tracks_by_region={
            "left": track("left", [120, 200, 280, 800]),
            "right": track("right", [420, 200, 580, 800]),
        },
        phases=[
            VerticalVirtualCameraPhase(
                phase_id="right",
                start_progress=0.0,
                end_progress=0.5,
                anchor_region_ids=["right"],
                editorial_reason="Start right.",
            ),
            VerticalVirtualCameraPhase(
                phase_id="left",
                start_progress=0.5,
                end_progress=1.0,
                anchor_region_ids=["left"],
                cut_admissible=True,
                transition_in="smoothstep",
                transition_duration_fraction=0.5,
                editorial_reason="Move left.",
            ),
        ],
    )
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=1000,
        overlay_path=None,
        base_filter=filter_graph,
        output_path=output,
        source_has_audio=False,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_vertical_fallback_filters_are_aspect_preserving_on_tall_sources() -> None:
    fit_filter = _vertical_fit_filter()
    center_filter = _vertical_center_crop_filter()

    assert "force_original_aspect_ratio=decrease" in fit_filter
    assert "pad=1080:1920" in fit_filter
    assert "gblur" not in fit_filter
    assert "color=0x0b0e12" in fit_filter
    assert "y=(ih-oh)/2" in center_filter


def test_explicit_center_crop_delivery_fallback_stays_full_bleed_and_audited() -> None:
    filter_graph, audit = _vertical_delivery_fallback(
        "center_crop",
        reason="all_automatic_candidates_exhausted",
    )

    assert "force_original_aspect_ratio=increase" in filter_graph
    assert "pad=1080:1920" not in filter_graph
    assert audit["applied_strategy"] == "full_bleed_center_crop_review"
    assert audit["full_bleed"] is True
    assert audit["requires_gemini_review"] is True
    assert "unverified_center_crop" in audit["risk_codes"]


def test_scope_preserving_delivery_fallback_remains_available_by_request() -> None:
    filter_graph, audit = _vertical_delivery_fallback(
        "fit_with_background",
        reason="atomic_relation_cannot_be_cropped",
    )

    assert "force_original_aspect_ratio=decrease" in filter_graph
    assert "pad=1080:1920" in filter_graph
    assert audit["applied_strategy"] == "fit_with_solid_matte"
    assert audit["full_bleed"] is False


def test_required_scope_fit_removes_only_space_outside_all_sampled_unions() -> None:
    result = _vertical_required_scope_fit_filter(
        {
            "source_display_width": 3840,
            "source_display_height": 2160,
            "source_geometry_lineage_passed": True,
            "tracking_confidence_gate_passed": True,
            "crop_keyframes": [
                {"required_union_box": [210, 239, 761, 785]},
                {"required_union_box": [212, 241, 764, 783]},
            ],
        }
    )

    assert result is not None
    filter_graph, audit = result
    assert "gblur" not in filter_graph
    assert "color=0x0b0e12" in filter_graph
    assert "crop=" in filter_graph
    assert audit["applied_strategy"] == "required_scope_solid_fit"
    assert audit["required_envelope_contained"] is True
    assert audit["scope_envelope_box_2d"] == [165.0, 194.0, 809.0, 830.0]
    assert audit["source_geometry_lineage_passed"] is True
    assert audit["tracking_confidence_gate_passed"] is True
    crop = audit["scope_crop_pixels"]
    assert 0 < crop["x"] < 3840
    assert 0 < crop["y"] < 2160
    assert crop["width"] < 3840
    assert crop["height"] < 2160


def test_required_scope_fit_can_be_policy_authorized_for_delivery() -> None:
    result = _vertical_required_scope_fit_filter(
        {
            "source_display_width": 1920,
            "source_display_height": 1080,
            "source_geometry_lineage_passed": True,
            "tracking_confidence_gate_passed": True,
            "crop_keyframes": [
                {"required_union_box": [100, 100, 900, 900]},
                {"required_union_box": [120, 100, 880, 900]},
            ],
        },
        autonomous_policy_reference="sha256:" + "a" * 64,
    )

    assert result is not None
    _, audit = result
    assert audit["requires_gemini_review"] is False
    assert "auto_policy_authorized" in audit["risk_codes"]
    assert audit["autonomous_policy_reference"] == "sha256:" + "a" * 64


def test_non_square_pixel_source_fails_closed_to_sar_normalized_static_reframe() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=1000,
        analysis_fps=2.0,
        seed_time_ms=0,
        semantic_seed_box=[350, 200, 650, 800],
        seed_source_width=720,
        seed_source_height=576,
        analysis_width=720,
        analysis_height=576,
        target_description="required visible region",
        state_counts={"tracked": 2},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[350, 200, 650, 800],
            )
            for index in range(2)
        ],
    )

    horizontal_filter, horizontal_audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track,
        "subtle",
        display_sample_aspect_ratio=16 / 15,
    )
    vertical_filter, vertical_audit = _vertical_filter_from_track(  # type: ignore[list-item]
        [track],
        fallback_strategy="center_crop",
        display_sample_aspect_ratio=16 / 15,
    )

    for filter_graph in (horizontal_filter, vertical_filter):
        assert "iw*sar" in filter_graph
        assert "setsar=1" in filter_graph
    for audit in (horizontal_audit, vertical_audit):
        assert audit["fallback_reason"] == (
            "non_square_pixel_aspect_ratio_requires_static_reframe"
        )
        assert audit["risk_codes"][0] == (
            "non_square_pixel_aspect_ratio_requires_static_reframe"
        )
        assert audit["requires_gemini_review"] is True
        assert audit["source_display_sample_aspect_ratio"] == pytest.approx(16 / 15)
        assert audit["sample_aspect_ratio_normalized_by_ffmpeg"] is True


def test_four_by_three_tracked_filter_renders_without_aspect_stretch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "four-by-three.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=10:d=0.3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=300,
        analysis_fps=10.0,
        seed_time_ms=0,
        semantic_seed_box=[400, 200, 600, 800],
        seed_source_width=320,
        seed_source_height=240,
        analysis_width=320,
        analysis_height=240,
        target_description="required visible region",
        state_counts={"tracked": 3},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 100,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=[400, 200, 600, 800],
            )
            for index in range(3)
        ],
    )
    filter_graph, audit = _vertical_filter_from_track(  # type: ignore[arg-type]
        [track]
    )
    output = tmp_path / "four-by-three-vertical.mp4"

    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=300,
        overlay_path=None,
        base_filter=filter_graph,
        output_path=output,
        source_has_audio=False,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    video = json.loads(completed.stdout)["streams"][0]

    assert (video["width"], video["height"]) == (1080, 1920)
    assert "scale=2560:1920" in filter_graph
    assert audit["crop_coordinate_space"]["aspect_ratio_relative_error"] == 0
    assert audit["containment_failure_count"] == 0


def test_dynamic_crop_filter_renders_video_and_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    chapter = FeatureChapterBrief(
        feature_id="demo",
        title="動態安全裁切",
        detail_lines=["保留指定主體"],
        target_duration_seconds=3,
    )
    overlay = tmp_path / "overlay.png"
    _render_text_layer(chapter, overlay, dimensions=(1080, 1920))
    expression = _piecewise_expression([0.0, 1.0, 2.0], [400.0, 900.0, 500.0])
    output = tmp_path / "vertical.mp4"
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=2000,
        overlay_path=overlay,
        base_filter=(
            "[0:v]fps=30,scale=3414:1920,"
            f"crop=1080:1920:x='{expression}':y=0,setsar=1[base]"
        ),
        output_path=output,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(stream["codec_type"] == "audio" for stream in streams)

    clean_output = tmp_path / "vertical-clean.mp4"
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=500,
        overlay_path=None,
        base_filter=(
            "[0:v]fps=30,scale=3414:1920,"
            "crop=1080:1920:x=500:y=0,setsar=1[base]"
        ),
        output_path=clean_output,
    )
    assert clean_output.exists()
    assert not (tmp_path / ".vertical-clean.partial.mp4").exists()


def test_concat_decodes_each_mp4_instead_of_stream_copy(tmp_path: Path) -> None:
    segments: list[Path] = []
    for index, frequency in enumerate((440, 660)):
        segment = tmp_path / f"segment-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={'red' if index == 0 else 'blue'}:s=320x180:r=30:d=1",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                str(segment),
            ],
            check=True,
        )
        segments.append(segment)
    output = tmp_path / "joined.mp4"
    _concat_segments(segments, output)
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout) == pytest.approx(2.0, abs=0.08)


def test_concat_normalizes_mixed_frame_rates_to_editorial_durations(
    tmp_path: Path,
) -> None:
    segments: list[Path] = []
    for index, frame_rate in enumerate((25, 30)):
        segment = tmp_path / f"mixed-rate-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s=320x180:r={frame_rate}:d=0.93",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-t",
                "0.95",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                str(segment),
            ],
            check=True,
        )
        segments.append(segment)

    output = tmp_path / "normalized.mp4"
    _concat_segments(
        segments,
        output,
        segment_durations_seconds=(1.0, 1.0),
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    video = next(
        stream
        for stream in metadata["streams"]
        if stream["codec_type"] == "video"
    )
    assert float(metadata["format"]["duration"]) == pytest.approx(2.0, abs=0.04)
    assert video["avg_frame_rate"] == "30/1"
    assert int(video["nb_frames"]) == 60


def test_video_only_source_gets_explicit_synthetic_silence(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "review-segment.mp4"
    audio_origin = _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=1000,
        overlay_path=None,
        base_filter="[0:v]fps=30,scale=320:180,setsar=1[base]",
        output_path=output,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {stream["codec_type"] for stream in json.loads(completed.stdout)["streams"]}
    assert audio_origin == "synthetic_silence"
    assert stream_types == {"video", "audio"}


def test_automatic_reframe_summary_preserves_switch_and_failure_audit() -> None:
    summary = _summarize_automatic_reframe(
        [
            {
                "feature_id": "opening",
                "applied_strategy": "tracked_crop",
                "requires_gemini_review": False,
                "automatic_candidate_selection": {
                    "enabled": True,
                    "planned_candidate_count": 2,
                    "selected_candidate_id": "take-b",
                    "selected_candidate_rank": 2,
                    "attempts": [
                        {
                            "candidate_id": "take-a",
                            "failure_codes": ["hard_core_not_fully_retained"],
                        },
                        {"candidate_id": "take-b", "failure_codes": []},
                    ],
                },
            },
            {
                "feature_id": "closing",
                "applied_strategy": "policy_blocked_preview_solid_fit",
                "requires_gemini_review": True,
                "automatic_candidate_selection": {
                    "enabled": True,
                    "planned_candidate_count": 1,
                    "selected_candidate_id": "take-c",
                    "selected_candidate_rank": 1,
                    "attempts": [
                        {
                            "candidate_id": "take-c",
                            "failure_codes": ["track_coverage_below_minimum"],
                        }
                    ],
                },
            },
        ]
    )

    assert summary["candidate_attempt_count"] == 3
    assert summary["candidate_switch_count"] == 1
    assert summary["candidate_recall_incomplete_chapter_count"] == 1
    assert summary["portrait_crop_chapter_count"] == 1
    assert summary["scope_preserving_fit_chapter_count"] == 1
    assert summary["policy_blocked_chapter_count"] == 1
    assert summary["review_required_chapter_count"] == 1
    assert summary["failure_code_counts"] == {
        "hard_core_not_fully_retained": 1,
        "track_coverage_below_minimum": 1,
    }
    assert len(summary["summary_sha256"]) == 64


def test_feature_plan_candidate_audit_exposes_rank_one_and_unexplained_reuse() -> None:
    chapters = [
        FeatureChapterSelect(
            feature_id=feature_id,
            evidence_status="supported",
            observed_visual_evidence=f"Observable {feature_id}.",
            selection_reason=f"Selected {feature_id}.",
            horizontal_frame_id=f"RF{index:06d}",
            horizontal_strategy="original",
            horizontal_zoom_intent="none",
            horizontal_target_description=None,
            vertical_frame_id=f"RF{index:06d}",
            vertical_strategy="fit_with_background",
            vertical_target_description=None,
            quality_risks=[],
            confidence=0.9,
        )
        for index, feature_id in enumerate(("opening", "closing"), start=1)
    ]
    plan = FeatureEditPlan(
        project_id="generic-project",
        catalog_id="generic-catalog",
        title="Generic",
        chapters=chapters,
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    audit = _audit_feature_plan_candidate_recall(
        plan,
        frame_source_assets={
            "RF000001": "sha256:" + "a" * 64,
            "RF000002": "sha256:" + "a" * 64,
        },
    )

    assert audit["candidate_recall_complete"] is False
    assert audit["rank_one_only_chapter_count"] == 2
    assert audit["selection_repetition_review_required"] is True
    assert audit["reuse_groups"]["9x16"][0]["feature_ids"] == [
        "opening",
        "closing",
    ]
    assert len(audit["audit_sha256"]) == 64


def test_requested_candidate_recall_ignores_unrequested_aspect() -> None:
    chapter = FeatureChapterSelect(
        feature_id="chapter",
        evidence_status="supported",
        observed_visual_evidence="Observable evidence.",
        selection_reason="Selected evidence.",
        horizontal_frame_id="RF000001",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_frame_id="RF000002",
        vertical_strategy="tracked_crop",
        vertical_target_description="the visible subject",
        vertical_candidates=[
            {
                "candidate_id": "vertical-a",
                "rank": 1,
                "source_asset_id": "asset-a",
                "event_id": "event-a",
                "frame_id": "RF000002",
                "observed_visual_evidence": "First vertical option.",
                "selection_reason": "Primary.",
                "strategy": "tracked_crop",
                "target_description": "the visible subject",
                "confidence": 0.9,
            },
            {
                "candidate_id": "vertical-b",
                "rank": 2,
                "source_asset_id": "asset-b",
                "event_id": "event-b",
                "frame_id": "RF000003",
                "observed_visual_evidence": "Second vertical option.",
                "selection_reason": "Fallback.",
                "strategy": "tracked_crop",
                "target_description": "the other visible subject",
                "confidence": 0.8,
            },
        ],
        quality_risks=[],
        confidence=0.9,
    )
    plan = FeatureEditPlan(
        project_id="generic-project",
        catalog_id="generic-catalog",
        title="Generic",
        chapters=[chapter],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    vertical = _audit_requested_candidate_recall(plan, aspect="9x16")
    horizontal = _audit_requested_candidate_recall(plan, aspect="16x9")

    assert vertical["complete"] is True
    assert horizontal["complete"] is False
    assert vertical["rows"][0]["candidate_counts"] == {"9x16": 2}


def test_missing_evidence_render_is_partial_not_delivery_success() -> None:
    manifest = {
        "horizontal": {
            "requested": True,
            "status": "rendered",
            "chapters": [
                {
                    "feature_id": "missing",
                    "source_clip_id": None,
                    "fallback_reason": "catalog_evidence_not_found",
                }
            ],
        },
        "vertical": {"requested": False, "status": "not_requested", "chapters": []},
        "requested_candidate_recall_audit": {"complete": True},
        "quality_map_coverage_audit": {"complete": True},
        "reframe_policy_binding": None,
        "post_render_quality_qc": {
            "requested": True,
            "technical_qc_passed": True,
        },
    }

    report = _build_feature_cut_eligibility_report(
        manifest,
        execution_profile=FeatureCutExecutionProfile.PRODUCTION_REVIEW,
    )

    assert report.media_rendered is True
    assert report.run_state == FeatureCutRunState.PARTIAL
    assert report.delivery_eligible is False
    assert report.ready_for_human_review is False
    assert "required_evidence_incomplete" in report.blocking_reasons


def test_review_preview_exposes_unfinished_gates_without_claiming_delivery() -> None:
    manifest = {
        "horizontal": {
            "requested": True,
            "status": "rendered",
            "chapters": [
                {
                    "feature_id": "supported",
                    "source_clip_id": "clip-a",
                    "source_in_ms": 0,
                    "source_out_ms": 3000,
                    "fallback_reason": None,
                    "risk_codes": [],
                }
            ],
        },
        "vertical": {"requested": False, "status": "not_requested", "chapters": []},
        "requested_candidate_recall_audit": {"complete": False},
        "quality_map_coverage_audit": {"complete": False},
        "reframe_policy_binding": None,
        "post_render_quality_qc": {
            "requested": True,
            "technical_qc_passed": True,
        },
    }

    review = _build_feature_cut_eligibility_report(
        manifest,
        execution_profile=FeatureCutExecutionProfile.REVIEW_PREVIEW,
    )
    production = _build_feature_cut_eligibility_report(
        manifest,
        execution_profile=FeatureCutExecutionProfile.PRODUCTION_REVIEW,
    )

    assert review.run_state == FeatureCutRunState.REVIEW_PREVIEW
    assert review.ready_for_human_review is False
    assert review.delivery_eligible is False
    assert "candidate_recall_incomplete" in review.review_reasons
    assert production.run_state == FeatureCutRunState.REVIEW_PREVIEW
    assert production.ready_for_human_review is False


def test_all_automatic_gates_only_reach_ready_for_human_review() -> None:
    manifest = {
        "horizontal": {
            "requested": True,
            "status": "rendered",
            "chapters": [
                {
                    "feature_id": "supported",
                    "source_clip_id": "clip-a",
                    "source_in_ms": 0,
                    "source_out_ms": 3000,
                    "fallback_reason": None,
                    "risk_codes": [],
                }
            ],
        },
        "vertical": {"requested": False, "status": "not_requested", "chapters": []},
        "requested_candidate_recall_audit": {"complete": True},
        "quality_map_coverage_audit": {"complete": True},
        "reframe_policy_binding": None,
        "post_render_quality_qc": {
            "requested": True,
            "technical_qc_passed": True,
        },
    }

    report = _build_feature_cut_eligibility_report(
        manifest,
        execution_profile=FeatureCutExecutionProfile.PRODUCTION_REVIEW,
    )

    assert report.run_state == FeatureCutRunState.READY_FOR_HUMAN_REVIEW
    assert report.ready_for_human_review is True
    assert report.delivery_eligible is False
    assert report.editorial_contract.final_sequence_qa_passed == "not_run"
    assert report.editorial_contract.human_approval_passed == "not_run"


def test_unauthorized_source_overlap_keeps_review_media_but_blocks_readiness() -> None:
    manifest = {
        "horizontal": {
            "requested": True,
            "status": "rendered",
            "chapters": [
                {
                    "feature_id": "opening",
                    "source_clip_id": "clip-a",
                    "source_in_ms": 0,
                    "source_out_ms": 3000,
                    "fallback_reason": None,
                    "risk_codes": [],
                }
            ],
        },
        "vertical": {"requested": False, "status": "not_requested", "chapters": []},
        "requested_candidate_recall_audit": {"complete": True},
        "quality_map_coverage_audit": {"complete": True},
        "source_reuse_contract_passed": False,
        "reframe_policy_binding": None,
        "post_render_quality_qc": {
            "requested": True,
            "technical_qc_passed": True,
        },
    }

    report = _build_feature_cut_eligibility_report(
        manifest,
        execution_profile=FeatureCutExecutionProfile.PRODUCTION_REVIEW,
    )

    assert report.media_rendered is True
    assert report.run_state == FeatureCutRunState.REVIEW_PREVIEW
    assert report.ready_for_human_review is False
    assert "source_reuse_contract_failed" in report.blocking_reasons


def test_human_intent_does_not_replace_execution_verification() -> None:
    manifest = {
        "horizontal": {"requested": False, "status": "not_requested", "chapters": []},
        "vertical": {
            "requested": True,
            "status": "rendered",
            "chapters": [
                {
                    "feature_id": "reviewed-intent",
                    "source_clip_id": "clip-a",
                    "source_in_ms": 0,
                    "source_out_ms": 3000,
                    "fallback_reason": None,
                    "risk_codes": [],
                    "human_policy_execution_verified": False,
                }
            ],
        },
        "requested_candidate_recall_audit": {"complete": True},
        "quality_map_coverage_audit": {"complete": True},
        "reframe_policy_binding": {"policy_id": "human-intent"},
        "post_render_quality_qc": {
            "requested": True,
            "technical_qc_passed": True,
        },
    }

    report = _build_feature_cut_eligibility_report(
        manifest,
        execution_profile=FeatureCutExecutionProfile.PRODUCTION_REVIEW,
    )

    assert report.run_state == FeatureCutRunState.REVIEW_PREVIEW
    assert "human_intent_execution_not_verified" in report.blocking_reasons


def test_render_source_reuse_allows_audited_reprise_but_not_silent_padding() -> None:
    chapters = [
        FeatureChapterSelect(
            feature_id="opening",
            evidence_status="supported",
            observed_visual_evidence="A visible establishing view.",
            selection_reason="Establishes the location.",
            horizontal_frame_id="RF000001",
            horizontal_strategy="original",
            horizontal_zoom_intent="none",
            horizontal_target_description=None,
            vertical_frame_id="RF000001",
            vertical_strategy="fit_with_background",
            vertical_target_description=None,
            quality_risks=[],
            confidence=0.9,
        ),
        FeatureChapterSelect(
            feature_id="closing",
            evidence_status="supported",
            observed_visual_evidence="The establishing view returns.",
            selection_reason="Closes the sequence.",
            horizontal_frame_id="RF000002",
            horizontal_strategy="original",
            horizontal_zoom_intent="none",
            horizontal_target_description=None,
            vertical_frame_id="RF000002",
            vertical_strategy="fit_with_background",
            vertical_target_description=None,
            quality_risks=[],
            confidence=0.9,
            source_reuse_mode="editorial_reprise",
            source_reuse_justification=(
                "The return forms an observable opening/closing callback."
            ),
        ),
    ]
    plan = FeatureEditPlan(
        project_id="generic-project",
        catalog_id="generic-catalog",
        title="Generic",
        chapters=chapters,
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    rendered = [
        {
            "feature_id": "opening",
            "source_clip_id": "clip-a",
            "source_in_ms": 1000,
            "source_out_ms": 3000,
            "segment_render_fingerprint": "same",
        },
        {
            "feature_id": "closing",
            "source_clip_id": "clip-a",
            "source_in_ms": 1000,
            "source_out_ms": 3000,
            "segment_render_fingerprint": "same",
        },
    ]

    audit = _audit_render_source_reuse(plan, rendered, aspect="9x16")

    assert audit["status"] == "passed"
    assert audit["requires_human_review"] is True
    assert audit["rows"][0]["exact_interval_repeat"] is True
    assert audit["rows"][0]["reuse_mode"] == "editorial_reprise"

    blocked_plan = plan.model_copy(
        update={
            "chapters": [
                chapters[0],
                chapters[1].model_copy(
                    update={
                        "source_reuse_mode": "none",
                        "source_reuse_justification": None,
                    }
                ),
            ]
        }
    )
    blocked = _audit_render_source_reuse(
        blocked_plan,
        rendered,
        aspect="9x16",
    )
    assert blocked["status"] == "blocked"
    assert blocked["violations"][0]["feature_id"] == "closing"


def test_render_source_reuse_distinct_interval_cannot_overlap() -> None:
    first = FeatureChapterSelect(
        feature_id="first",
        evidence_status="supported",
        observed_visual_evidence="First action.",
        selection_reason="Introduces the action.",
        horizontal_frame_id="RF000001",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_frame_id="RF000001",
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        quality_risks=[],
        confidence=0.9,
    )
    second = first.model_copy(
        update={
            "feature_id": "second",
            "horizontal_frame_id": "RF000002",
            "vertical_frame_id": "RF000002",
            "source_reuse_mode": "distinct_interval",
            "source_reuse_justification": "Shows a different later action.",
        }
    )
    plan = FeatureEditPlan(
        project_id="generic-project",
        catalog_id="generic-catalog",
        title="Generic",
        chapters=[first, second],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    rendered = [
        {
            "feature_id": "first",
            "source_clip_id": "clip-a",
            "source_in_ms": 1000,
            "source_out_ms": 3000,
            "segment_render_fingerprint": "first",
        },
        {
            "feature_id": "second",
            "source_clip_id": "clip-a",
            "source_in_ms": 2500,
            "source_out_ms": 4000,
            "segment_render_fingerprint": "second",
        },
    ]

    audit = _audit_render_source_reuse(plan, rendered, aspect="16x9")

    assert audit["status"] == "blocked"
    assert audit["violations"][0]["overlap_ms"] == 500


def test_runtime_candidate_reuse_blocks_before_grounding() -> None:
    selected = FeatureChapterSelect(
        feature_id="second",
        evidence_status="supported",
        observed_visual_evidence="Second feature.",
        selection_reason="Shows another feature.",
        horizontal_frame_id="RF000002",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_frame_id="RF000002",
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        quality_risks=[],
        confidence=0.9,
    )
    prior = [
        {
            "feature_id": "first",
            "source_clip_id": "clip-a",
            "source_in_ms": 1000,
            "source_out_ms": 8000,
        }
    ]

    violation = _runtime_candidate_reuse_violation(
        selected,
        prior,
        source_clip_id="clip-a",
        source_in_ms=1000,
        source_out_ms=8000,
    )

    assert violation is not None
    assert violation["prior_feature_id"] == "first"
    assert violation["overlap_ms"] == 7000
    assert violation["reuse_mode"] == "none"


def test_runtime_candidate_reuse_allows_authorized_distinct_interval() -> None:
    selected = FeatureChapterSelect(
        feature_id="second",
        evidence_status="supported",
        observed_visual_evidence="A later, distinct action.",
        selection_reason="Shows a distinct event interval.",
        horizontal_frame_id="RF000002",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_frame_id="RF000002",
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        quality_risks=[],
        confidence=0.9,
        source_reuse_mode="distinct_interval",
        source_reuse_justification="A visibly separate event in the same take.",
    )
    prior = [
        {
            "feature_id": "first",
            "source_clip_id": "clip-a",
            "source_in_ms": 1000,
            "source_out_ms": 3000,
        }
    ]

    assert (
        _runtime_candidate_reuse_violation(
            selected,
            prior,
            source_clip_id="clip-a",
            source_in_ms=4000,
            source_out_ms=6000,
        )
        is None
    )
    assert (
        _runtime_candidate_reuse_violation(
            selected,
            prior,
            source_clip_id="clip-a",
            source_in_ms=2500,
            source_out_ms=4500,
        )
        is not None
    )


def test_runtime_candidate_reuse_blocks_third_use_even_with_authority() -> None:
    selected = FeatureChapterSelect(
        feature_id="third",
        evidence_status="supported",
        observed_visual_evidence="A third distinct interval.",
        selection_reason="A later event.",
        horizontal_frame_id="RF000003",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_frame_id="RF000003",
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        quality_risks=[],
        confidence=0.9,
        source_reuse_mode="distinct_interval",
        source_reuse_justification="A different event interval.",
    )
    prior = [
        {
            "feature_id": feature_id,
            "source_clip_id": "clip-a",
            "source_in_ms": start,
            "source_out_ms": start + 1000,
        }
        for feature_id, start in (("first", 0), ("second", 2000))
    ]

    violation = _runtime_candidate_reuse_violation(
        selected,
        prior,
        source_clip_id="clip-a",
        source_in_ms=4000,
        source_out_ms=5000,
    )

    assert violation is not None
    assert violation["reason_code"] == (
        "source_use_count_exceeds_bounded_v1_limit"
    )


def test_music_cues_are_projected_from_source_to_assembly_timeline() -> None:
    cue = LockedMusicCue(
        cue_id="locked-cue-00001",
        kind="downbeat",
        sample_index=72_000,
        time_ms=1_500,
        strength=0.9,
        priority=CuePriority.PREFERRED,
    )
    lock = SimpleNamespace(master_sample_rate=48_000, cues=[cue])
    span = SimpleNamespace(
        source_start_sample=48_000,
        source_end_sample=96_000,
        output_start_sample=0,
    )

    projected = _project_locked_cues_to_music_output(
        lock,
        spans=[span],
    )

    assert projected[0].cue_id == cue.cue_id
    assert projected[0].sample_index == 24_000
    assert projected[0].time_ms == 500


def test_simultaneous_relation_rejects_one_unbound_atomic_core() -> None:
    with pytest.raises(
        ValidationError,
        match="two independently grounded hard-core participant regions",
    ):
        SelectedVerticalFramingProposal(
            candidate_id="generic-comparison",
            source_asset_id="sha256:" + "a" * 64,
            event_id="comparison",
            frame_id="RF000001",
            semantic_requirement="simultaneous_relation",
            relation_temporal_mode="simultaneous_required",
            recommended_action="tracked_crop",
            presentation_options=_portrait_presentation_options(
                single="feasible",
            ),
            regions=[
                {
                    "region_id": "contact-core",
                    "target_description": "the visible contact boundary",
                    "evidence_role": "relation_carrier",
                    "role": "required",
                    "atomic": True,
                    "observable_relations": ["reference touches subject edge"],
                }
            ],
            virtual_camera_proposal={
                "composition_mode": "single_anchor_hold",
                "phases": [
                    {
                        "phase_id": "hold",
                        "start_progress": 0,
                        "end_progress": 1,
                        "anchor_region_ids": ["contact-core"],
                        "observable_predicate": "The relation is directly visible.",
                        "transition_condition": "Hold while directly visible.",
                        "editorial_reason": "Preserve the minimal relational evidence.",
                        "camera_behavior": "hold",
                        "transition_in": "cut",
                        "transition_duration_fraction": 0,
                    }
                ],
                "proposal_reason": "One unbound compound region is insufficient.",
            },
            observed_evidence=[
                "A visible reference touches a visible subject edge."
            ],
            decision_reason=(
                "The relation participants are not independently grounded."
            ),
            confidence=0.9,
            model_provenance=ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="test",
                generated_at="test",
            ),
        )


def test_simultaneous_relation_accepts_bound_atomic_relation_core() -> None:
    proposal = SelectedVerticalFramingProposal(
        candidate_id="generic-comparison",
        source_asset_id="sha256:" + "a" * 64,
        event_id="comparison",
        frame_id="RF000001",
        semantic_requirement="simultaneous_relation",
        relation_temporal_mode="simultaneous_required",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            single="feasible",
        ),
        regions=[
            {
                "region_id": "contact-core",
                "target_description": "the indivisible visible relation zone",
                "evidence_role": "relation_carrier",
                "role": "required",
                "atomic": True,
                "observable_relations": [
                    "the two participants form a directly visible relation"
                ],
            },
            {
                "region_id": "participant-a",
                "entity_id": "entity-a",
                "target_description": "the first visible participant",
                "evidence_role": "relation_participant",
                "role": "preferred",
            },
            {
                "region_id": "participant-b",
                "entity_id": "entity-b",
                "target_description": "the second visible participant",
                "evidence_role": "relation_participant",
                "role": "preferred",
            },
        ],
        virtual_camera_proposal={
            "composition_mode": "single_anchor_hold",
            "phases": [
                {
                    "phase_id": "hold",
                    "start_progress": 0,
                    "end_progress": 1,
                    "anchor_region_ids": ["contact-core"],
                    "observable_predicate": (
                        "Both participants and their relation remain visible."
                    ),
                    "transition_condition": "Hold while the relation is visible.",
                    "editorial_reason": (
                        "Preserve the smallest complete relational evidence."
                    ),
                    "camera_behavior": "hold",
                    "transition_in": "cut",
                    "transition_duration_fraction": 0,
                }
            ],
            "proposal_reason": (
                "The compound carrier preserves the bound relation."
            ),
        },
        observed_evidence=[
            "Both bound participants and their relation are directly visible."
        ],
        decision_reason=(
            "The compound carrier is smaller than either complete participant."
        ),
        confidence=0.9,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    _validate_selected_framing_coverage_invariant(
        upstream_intent="simultaneous_relation",
        upstream_target_descriptions=["first participant", "second participant"],
        proposal=proposal,
    )


def test_selected_framing_cannot_weaken_group_coverage() -> None:
    proposal = SelectedVerticalFramingProposal(
        candidate_id="group",
        source_asset_id="sha256:" + "a" * 64,
        event_id="group",
        frame_id="RF000001",
        semantic_requirement="single_primary",
        relation_temporal_mode="not_applicable",
        recommended_action="tracked_crop",
        presentation_options=_portrait_presentation_options(
            single="feasible",
        ),
        regions=[
            {
                "region_id": "one",
                "target_description": "one visible participant",
                "evidence_role": "primary_subject",
                "role": "required",
            }
        ],
        virtual_camera_proposal={
            "composition_mode": "single_anchor_hold",
            "phases": [
                {
                    "phase_id": "hold",
                    "start_progress": 0,
                    "end_progress": 1,
                    "anchor_region_ids": ["one"],
                    "observable_predicate": "One participant is visible.",
                    "transition_condition": "Hold.",
                    "editorial_reason": "Follow one participant.",
                    "camera_behavior": "hold",
                    "transition_in": "cut",
                    "transition_duration_fraction": 0,
                }
            ],
            "proposal_reason": "Single participant crop.",
        },
        observed_evidence=["One participant is visible."],
        decision_reason="Center one participant.",
        confidence=0.9,
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    with pytest.raises(ValueError, match="weakens the upstream coverage"):
        _validate_selected_framing_coverage_invariant(
            upstream_intent="group_coverage",
            upstream_target_descriptions=["participant A", "participant B"],
            proposal=proposal,
        )
