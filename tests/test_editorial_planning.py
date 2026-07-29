from __future__ import annotations

import pytest

from jascue_video_lab.editorial_planning import (
    build_attention_profile,
    build_rhythm_plan,
    reconcile_attention_delivery_floor,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    AttentionObservation,
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    ModelProvenance,
    ShotFlowIntent,
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
    assert profile.chapters[0].minimum_dwell_seconds == 1
    assert profile.chapters[0].preferred_dwell_seconds == 10
    assert profile.chapters[0].maximum_dwell_seconds == 12
    assert (
        profile.chapters[0].maximum_dwell_seconds
        != brief.target_duration_seconds
    )


def test_preferred_six_seven_eight_second_dwells_remain_bounded_options(
    tmp_path,
) -> None:
    brief = _brief()
    plan = _plan(with_attention=True)
    preferred_values = [6.0, 7.0, 8.0, 6.0, 7.0, 8.0]
    capacity_values = [5.5, 6.5, 7.5, 5.25, 6.25, 7.25]
    revised_chapters = []
    for selected, preferred in zip(
        plan.chapters,
        preferred_values,
        strict=True,
    ):
        observation = selected.attention_observation
        assert observation is not None
        revised_chapters.append(
            selected.model_copy(
                update={
                    "recommended_duration_seconds": preferred,
                    "attention_observation": observation.model_copy(
                        update={
                            "minimum_dwell_seconds": 2.0,
                            "maximum_dwell_seconds": 9.0,
                        }
                    ),
                }
            )
        )
    plan = plan.model_copy(update={"chapters": revised_chapters})
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
            f"chapter-{index}": capacity
            for index, capacity in enumerate(capacity_values, start=1)
        },
    )
    write_json(attention_path, profile)
    rhythm = build_rhythm_plan(
        profile,
        target_duration_seconds=sum(capacity_values),
        attention_profile_sha256=sha256_file(attention_path),
    )

    assert [chapter.minimum_duration_seconds for chapter in rhythm.chapters] == [
        2.0
    ] * 6
    assert [
        chapter.preferred_duration_seconds for chapter in rhythm.chapters
    ] == capacity_values
    assert [
        chapter.maximum_duration_seconds for chapter in rhythm.chapters
    ] == capacity_values
    assert [
        chapter.preferred_duration_seconds for chapter in rhythm.chapters
    ] != preferred_values


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


def test_flow_intent_can_lock_content_boundary_without_exact_time(
    tmp_path,
) -> None:
    brief = _brief()
    plan = _plan(with_attention=True)
    first = plan.chapters[0].model_copy(
        update={
            "flow_intent": ShotFlowIntent(
                narrative_role="proof",
                energy_role="rise",
                relation_to_previous="continue_action",
                boundary_alignment="content_locked",
                visual_sync_event=None,
                visual_sync_predicate=None,
                music_target=None,
            )
        }
    )
    plan = plan.model_copy(update={"chapters": [first, *plan.chapters[1:]]})
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
    )

    assert rhythm.chapters[0].boundary_alignment == "content_locked"
    assert rhythm.chapters[0].boundary_priority == "low"
    assert rhythm.chapters[0].flow_intent == first.flow_intent


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


def test_attention_floor_reconciliation_uses_only_small_safe_headroom(
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
            f"chapter-{index}": 12.0 for index in range(1, 7)
        },
    )
    shortened = profile.model_copy(
        update={
            "chapters": [
                chapter.model_copy(
                    update={
                        "maximum_dwell_seconds": (
                            9.9 if index == 0 else 10.0
                        ),
                        "preferred_dwell_seconds": min(
                            chapter.preferred_dwell_seconds,
                            9.9 if index == 0 else 10.0,
                        ),
                    }
                )
                for index, chapter in enumerate(profile.chapters)
            ]
        }
    )

    resolved, audit = reconcile_attention_delivery_floor(
        shortened,
        delivery_floor_seconds=60.0,
        maximum_shortfall_tolerance_seconds=1.0,
    )

    assert sum(chapter.maximum_dwell_seconds for chapter in resolved.chapters) == 60
    assert audit["applied"] is True
    assert audit["shortfall_seconds"] == 0.1
    assert audit["adjustments"]
    assert any(
        "maximum_dwell_extended_by_local_delivery_floor_reconciliation"
        in chapter.uncertainties
        for chapter in resolved.chapters
    )


def test_attention_floor_reconciliation_rejects_material_shortfall(
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
            f"chapter-{index}": 12.0 for index in range(1, 7)
        },
    )
    shortened = profile.model_copy(
        update={
            "chapters": [
                chapter.model_copy(
                    update={
                        "maximum_dwell_seconds": 9.0,
                        "preferred_dwell_seconds": 9.0,
                    }
                )
                for chapter in profile.chapters
            ]
        }
    )

    with pytest.raises(ValueError, match="exceeds local reconciliation tolerance"):
        reconcile_attention_delivery_floor(
            shortened,
            delivery_floor_seconds=60.0,
            maximum_shortfall_tolerance_seconds=1.0,
        )
