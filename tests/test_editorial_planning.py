from __future__ import annotations

import pytest

from jascue_video_lab.editorial_planning import (
    build_attention_profile,
    build_rhythm_plan,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    AttentionObservation,
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    ModelProvenance,
)
from jascue_video_lab.storage import write_json


def _brief() -> FeatureEditBrief:
    return FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id=f"chapter-{index}",
                title=f"Chapter {index}",
                detail_lines=[],
                target_duration_seconds=10,
            )
            for index in range(1, 7)
        ],
    )


def _plan(*, with_attention: bool) -> FeatureEditPlan:
    observation = (
        AttentionObservation(
            semantic_novelty=0.8,
            action_progress=0.9,
            visual_motion=0.5,
            composition_change=0.4,
            reading_load=0.2,
            unresolved_tension=0.1,
            emotional_hold_value=0.2,
            repetition_pressure=0.7,
            music_transition_opportunity=0.8,
            minimum_dwell_seconds=3,
            maximum_dwell_seconds=12,
            rationale="The observable action resolves before the transition.",
            uncertainties=[],
            requires_human_review=True,
        )
        if with_attention
        else None
    )
    return FeatureEditPlan(
        project_id="generic-project",
        catalog_id="generic-catalog",
        title="Generic edit",
        chapters=[
            FeatureChapterSelect(
                feature_id=f"chapter-{index}",
                evidence_status="supported",
                horizontal_frame_id=f"RF{index:06d}",
                vertical_frame_id=f"RF{index:06d}",
                observed_visual_evidence="Observable generic action.",
                selection_reason="Representative evidence.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=[],
                confidence=0.9,
                recommended_duration_seconds=10,
                duration_rationale="Relative observable information.",
                attention_observation=observation,
            )
            for index in range(1, 7)
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-test",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )


def test_attention_profile_clamps_model_maximum_to_quality_capacity(
    tmp_path,
) -> None:
    brief = _brief()
    plan = _plan(with_attention=True)
    brief_path = tmp_path / "brief.json"
    plan_path = tmp_path / "plan.json"
    write_json(brief_path, brief)
    write_json(plan_path, plan)

    profile = build_attention_profile(
        brief,
        plan,
        source_brief_sha256=sha256_file(brief_path),
        source_feature_plan_sha256=sha256_file(plan_path),
        quality_safe_capacity_seconds={
            f"chapter-{index}": 8.0 for index in range(1, 7)
        },
    )

    assert all(chapter.minimum_dwell_seconds == 3 for chapter in profile.chapters)
    assert all(chapter.preferred_dwell_seconds == 8 for chapter in profile.chapters)
    assert all(chapter.maximum_dwell_seconds == 8 for chapter in profile.chapters)
    assert all(
        chapter.evidence_authority == "gemini_attention_observation"
        for chapter in profile.chapters
    )


def test_legacy_profile_preserves_unknown_attention_as_null(tmp_path) -> None:
    brief = _brief()
    plan = _plan(with_attention=False)
    brief_path = tmp_path / "brief.json"
    plan_path = tmp_path / "plan.json"
    write_json(brief_path, brief)
    write_json(plan_path, plan)

    profile = build_attention_profile(
        brief,
        plan,
        source_brief_sha256=sha256_file(brief_path),
        source_feature_plan_sha256=sha256_file(plan_path),
        quality_safe_capacity_seconds={
            f"chapter-{index}": 12.0 for index in range(1, 7)
        },
    )

    assert profile.chapters[0].semantic_novelty is None
    assert profile.chapters[0].evidence_authority == (
        "gemini_relative_dwell_legacy"
    )
    assert profile.chapters[0].uncertainties == [
        "attention_vector_unavailable_for_legacy_plan"
    ]


def test_rhythm_plan_uses_vector_without_inventing_cut_timestamp(
    tmp_path,
) -> None:
    brief = _brief()
    plan = _plan(with_attention=True)
    brief_path = tmp_path / "brief.json"
    plan_path = tmp_path / "plan.json"
    attention_path = tmp_path / "attention.json"
    write_json(brief_path, brief)
    write_json(plan_path, plan)
    profile = build_attention_profile(
        brief,
        plan,
        source_brief_sha256=sha256_file(brief_path),
        source_feature_plan_sha256=sha256_file(plan_path),
        quality_safe_capacity_seconds={
            f"chapter-{index}": 12.0 for index in range(1, 7)
        },
    )
    write_json(attention_path, profile)

    rhythm = build_rhythm_plan(
        profile,
        target_duration_seconds=60,
        attention_profile_sha256=sha256_file(attention_path),
        style_profile="energetic",
    )

    assert rhythm.interpretation == (
        "attention_bounds_and_boundary_pressure_not_frame_accurate_cuts"
    )
    assert all(chapter.cut_pressure is not None for chapter in rhythm.chapters)
    assert all(
        chapter.boundary_priority == "high" for chapter in rhythm.chapters
    )
    assert all(
        "action_or_result_complete" in chapter.transition_reasons
        for chapter in rhythm.chapters
    )


def test_attention_profile_rejects_capacity_shorter_than_minimum(
    tmp_path,
) -> None:
    brief = _brief()
    plan = _plan(with_attention=True)
    brief_path = tmp_path / "brief.json"
    plan_path = tmp_path / "plan.json"
    write_json(brief_path, brief)
    write_json(plan_path, plan)

    with pytest.raises(ValueError, match="shorter than its minimum"):
        build_attention_profile(
            brief,
            plan,
            source_brief_sha256=sha256_file(brief_path),
            source_feature_plan_sha256=sha256_file(plan_path),
            quality_safe_capacity_seconds={
                "chapter-1": 2.5,
                **{f"chapter-{index}": 8.0 for index in range(2, 7)},
            },
        )
