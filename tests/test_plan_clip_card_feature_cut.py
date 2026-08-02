from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
)
from jascue_video_lab.clip_card_retrieval import (
    FeatureChapterShortlist,
    FeatureShortlistCandidate,
    FeatureShortlistPlan,
    normalize_shortlist_event_ids,
)
from jascue_video_lab.clip_card_observations import (
    AssessmentStatus,
    ClipObservationSupplement,
    EventCapabilityManifest,
    EventObservationSupplement,
    EvidenceRoleMap,
    ObservationBasis,
    ObservableBeat,
    clip_card_sha256,
    event_fingerprint,
)
from jascue_video_lab.models import (
    AttentionObservation,
    BoundaryPrecision,
    CardOpportunity,
    Entity,
    EntityKind,
    EvidenceOriginObservation,
    EvidenceModality,
    FeatureChapterBrief,
    FeatureEditBrief,
    FeatureEditPlan,
    FullClipCard,
    FullClipAttentionPhase,
    FullClipEvent,
    FullClipGroundingTarget,
    ModelProvenance,
    RushClip,
    RushFrame,
    RushesCatalog,
    ShotFlowIntent,
)
from jascue_video_lab.editing_capabilities import (
    simple_production_capability_catalog,
)
from jascue_video_lab.event_lock import EditorialBeatContract
from jascue_video_lab.feature_cut import (
    _current_external_projection_binding,
    write_external_feature_plan_projection,
)
from jascue_video_lab.gemini import MODEL_ID
from jascue_video_lab.media import sha256_file
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import write_json
from scripts.plan_clip_card_feature_cut import (
    CandidateVideoBudgetPreflightError,
    ClipCardFeatureCandidate,
    ClipCardFeatureCandidateV3,
    ClipCardVirtualCameraPhaseV1,
    ClipCardVirtualCameraProposalV1,
    ClipCardFeaturePlan,
    ClipCardFeaturePlanV2,
    ClipCardFeaturePlanV3,
    ClipCardFeatureSelect,
    ClipCardFeatureSelectV2,
    ClipCardFeatureSelectV3,
    DirectVideoAttentionStep,
    DirectVideoChapterDecision,
    DirectVideoEditPlan,
    DirectVideoHorizontalDecision,
    DirectVideoVerticalDecision,
    ResolvedEntityRef,
    ResolvedFramingRegion,
    SelectedClipCardEvidence,
    _assert_fresh_feature_namespace_empty,
    _resolve_latest_failed_feature_plan_attempt,
    _resolve_feature_reuse_artifacts,
    _verified_feature_raw_output_text,
    _write_feature_normalization_artifacts,
    audit_editorial_freshness,
    autonomous_content_mode_instructions,
    build_selected_clip_card_evidence,
    canonicalize_direct_video_edit_plan_output,
    canonicalize_feature_plan_output,
    compact_card,
    compact_card_v3,
    fit_candidate_video_windows_to_budget,
    project_feature_contracts,
    project_feature_contracts_v3,
    reproject_external_feature_plan,
    reproject_external_feature_plan_v2,
    reproject_external_feature_plan_v3,
    validate_plan_contract,
    validate_plan_contract_v3,
    main as feature_planner_main,
    planning_candidate_id,
    planning_candidate_slice,
    planning_capability_catalog,
    prior_vertical_selections,
    project_direct_video_edit_plan,
    validate_candidate_video_budget,
    validate_autonomous_planner_requested_aspects,
    direct_video_aspect_contract_instructions,
    direct_video_horizontal_only_prompt_instructions,
    direct_video_response_schema,
    project_direct_video_prompt_payload,
    validate_direct_video_aspect_contract,
    validate_direct_video_plan_fulfillment,
    validate_direct_video_source_allocation,
    validate_hard_shortlist_provenance,
)
from scripts.shortlist_clip_card_feature_candidates import (
    _shortlist_input_binding,
    validate_shortlist_fulfillment_minimums,
)


ASSET_ID = "sha256:" + "a" * 64


def test_shortlist_event_id_normalization_repairs_only_unique_zero_padding() -> None:
    card = _card()
    card = card.model_copy(
        update={
            "events": [
                card.events[0].model_copy(update={"event_id": "evt_001"})
            ]
        }
    )
    payload = {
        "chapters": [
            {
                "feature_id": "feature-1",
                "candidates": [
                    {
                        "source_asset_id": ASSET_ID,
                        "event_id": "evt_01",
                    }
                ],
            }
        ]
    }

    changes = normalize_shortlist_event_ids(
        payload,
        cards={ASSET_ID: card},
    )

    assert payload["chapters"][0]["candidates"][0]["event_id"] == "evt_001"
    assert changes == [
        {
            "feature_id": "feature-1",
            "source_asset_id": ASSET_ID,
            "field": "event_id",
            "from": "evt_01",
            "to": "evt_001",
            "reason": "unique_numeric_padding_equivalence",
        }
    ]


def test_shortlist_event_id_normalization_leaves_ambiguous_ids_to_fail() -> None:
    card = _card()
    first = card.events[0].model_copy(update={"event_id": "evt_001"})
    second = card.events[0].model_copy(update={"event_id": "evt_0001"})
    card = card.model_copy(update={"events": [first, second]})
    payload = {
        "chapters": [
            {
                "feature_id": "feature-1",
                "candidates": [
                    {
                        "source_asset_id": ASSET_ID,
                        "event_id": "evt_01",
                    }
                ],
            }
        ]
    }

    changes = normalize_shortlist_event_ids(
        payload,
        cards={ASSET_ID: card},
    )

    assert changes == []
    assert payload["chapters"][0]["candidates"][0]["event_id"] == "evt_01"


def test_shortlist_reuse_binding_changes_with_effective_evidence() -> None:
    common = {
        "catalog": _catalog(),
        "brief": _brief(),
        "editorial_contracts": (),
        "thinking_level": "low",
    }
    first = _shortlist_input_binding(
        **common,
        evidence=[{"event_id": "demo", "evidence_origin": "direct_source_event"}],
    )
    second = _shortlist_input_binding(
        **common,
        evidence=[{"event_id": "demo", "evidence_origin": "mediated_depiction"}],
    )

    assert first["effective_evidence_sha256"] != second[
        "effective_evidence_sha256"
    ]


def test_illustrative_coverage_policy_is_opt_in_and_binds_shortlist_input() -> None:
    disabled = _brief()
    enabled = disabled.model_copy(
        update={
            "illustrative_coverage_policy": (
                "related_product_or_environment_when_direct_absent"
            )
        }
    )
    disabled_dump = disabled.model_dump(mode="json")
    enabled_dump = enabled.model_dump(mode="json")

    assert "illustrative_coverage_policy" not in disabled_dump
    assert enabled_dump["illustrative_coverage_policy"] == (
        "related_product_or_environment_when_direct_absent"
    )
    assert "illustrative_coverage_policy" in enabled.model_dump_json()
    common = {
        "catalog": _catalog(),
        "editorial_contracts": (),
        "evidence": [{"event_id": "demo"}],
        "thinking_level": "low",
    }
    disabled_binding = _shortlist_input_binding(brief=disabled, **common)
    enabled_binding = _shortlist_input_binding(brief=enabled, **common)

    assert disabled_binding["brief_sha256"] != enabled_binding["brief_sha256"]


def test_autonomous_planner_catalog_exposes_policy_gated_presentations() -> None:
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

    review_ids = {
        item.capability_id
        for item in planning_capability_catalog(None).capabilities
    }
    autonomous_ids = {
        item.capability_id
        for item in planning_capability_catalog(policy).capabilities
    }

    assert "two_panel_layout" not in review_ids
    assert {
        "two_panel_layout",
        "solid_matte_fit",
        "intentional_freeze",
    } <= autonomous_ids

    restricted = policy.model_copy(
        update={
            "presentation": policy.presentation.model_copy(
                update={
                    "allow_solid_matte_fit": False,
                    "allow_intentional_freeze": False,
                }
            )
        }
    )
    restricted_catalog = planning_capability_catalog(restricted)
    restricted_ids = {
        item.capability_id for item in restricted_catalog.capabilities
    }
    assert "two_panel_layout" in restricted_ids
    assert "solid_matte_fit" not in restricted_ids
    assert "intentional_freeze" not in restricted_ids
    assert "solid_fit" in restricted_catalog.prohibited_automatic_delivery


def test_autonomous_content_mode_is_visible_to_the_semantic_planner() -> None:
    music_led = AutonomousEditPolicy(
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
    visual_demo = music_led.model_copy(
        update={"content_mode": "visual_demo"}
    )

    music_text = autonomous_content_mode_instructions(music_led)
    demo_text = autonomous_content_mode_instructions(visual_demo)

    assert "content_mode=music_led_feature" in music_text
    assert "沒有音樂時" in music_text
    assert "content_mode=visual_demo" in demo_text
    assert "操作、狀態轉換" in demo_text
    assert music_text != demo_text


@pytest.mark.parametrize(
    "requested_aspects",
    [
        ("16:9",),
        ("9:16",),
        ("16:9", "9:16"),
    ],
)
def test_autonomous_direct_planner_accepts_every_supported_aspect_shape(
    requested_aspects: tuple[str, ...],
) -> None:
    policy = AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=requested_aspects,
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

    assert validate_autonomous_planner_requested_aspects(policy) == frozenset(
        requested_aspects
    )


def test_direct_video_planning_only_selects_candidates_actually_attached() -> None:
    candidates = ["rank-1", "rank-2", "rank-3"]

    assert planning_candidate_slice(
        candidates,
        direct_video_evidence=True,
        depth=2,
    ) == ["rank-1", "rank-2"]
    assert planning_candidate_slice(
        candidates,
        direct_video_evidence=False,
        depth=2,
    ) == candidates
    assert planning_candidate_id(1) == "rank-01"
    assert planning_candidate_id(4) == "rank-04"


def test_candidate_video_budget_fails_before_upload_or_paid_planning() -> None:
    validate_candidate_video_budget(
        total_duration_ms=359_999,
        maximum_duration_ms=360_000,
    )
    with pytest.raises(ValueError, match="before upload or paid planning"):
        validate_candidate_video_budget(
            total_duration_ms=360_001,
            maximum_duration_ms=360_000,
        )


def test_candidate_video_budget_reduces_context_before_omitting_evidence() -> None:
    card = _card()
    event = card.events[0].model_copy(
        update={"start_mmss": "00:02", "end_mmss": "00:08"}
    )
    card = card.model_copy(update={"events": [event]})
    rows = [{"source_asset_id": ASSET_ID, "event_id": event.event_id}]

    context_ms, total_ms = fit_candidate_video_windows_to_budget(
        rows=rows,
        cards={ASSET_ID: card},
        requested_context_ms=1_000,
        maximum_total_ms=7_000,
    )

    assert context_ms == 500
    assert total_ms == 7_000
    assert rows[0]["start_ms"] == 1_500
    assert rows[0]["end_ms"] == 8_500


def test_candidate_video_budget_fits_363_seconds_to_exact_cap() -> None:
    card = _card()
    event = card.events[0].model_copy(
        update={"start_mmss": "00:02", "end_mmss": "00:37"}
    )
    card = card.model_copy(update={"duration_ms": 40_000, "events": [event]})
    rows = [
        {"source_asset_id": ASSET_ID, "event_id": event.event_id}
        for _ in range(10)
    ]

    context_ms, total_ms = fit_candidate_video_windows_to_budget(
        rows=rows,
        cards={ASSET_ID: card},
        requested_context_ms=650,
        maximum_total_ms=360_000,
    )

    assert context_ms == 500
    assert total_ms == 360_000
    assert all(row["duration_ms"] == 36_000 for row in rows)


def test_candidate_video_budget_reports_irreducible_window_minima() -> None:
    card = _card()
    event = card.events[0].model_copy(
        update={"start_mmss": "00:00", "end_mmss": "06:03"}
    )
    card = card.model_copy(update={"duration_ms": 363_000, "events": [event]})
    rows = [
        {
            "source_asset_id": ASSET_ID,
            "event_id": event.event_id,
            "references": [{"feature_id": "feature-1", "rank": 1}],
        }
    ]

    with pytest.raises(CandidateVideoBudgetPreflightError) as raised:
        fit_candidate_video_windows_to_budget(
            rows=rows,
            cards={ASSET_ID: card},
            requested_context_ms=1_000,
            maximum_total_ms=360_000,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic["reason_code"] == (
        "candidate_video_budget_irreducible_minimum"
    )
    assert diagnostic["minimum_total_ms"] == 363_000
    assert diagnostic["maximum_total_ms"] == 360_000
    assert diagnostic["per_window_minima"] == [
        {
            "source_asset_id": ASSET_ID,
            "event_id": event.event_id,
            "minimum_start_ms": 0,
            "minimum_end_ms": 363_000,
            "minimum_duration_ms": 363_000,
            "references": [{"feature_id": "feature-1", "rank": 1}],
        }
    ]


def test_alternate_edit_freshness_changes_only_substitutable_events() -> None:
    prior = project_feature_contracts(
        _v2_plan(),
        brief=_brief(),
        catalog=_catalog(),
    )
    unchanged_payload = prior.model_dump(mode="json")
    unchanged_payload["chapters"][0]["vertical_candidates"][1][
        "event_id"
    ] = "alternate-demo"
    unchanged = FeatureEditPlan.model_validate(unchanged_payload)

    failed = audit_editorial_freshness(
        unchanged,
        prior=prior,
        minimum_change_fraction=0.5,
    )

    assert failed["passed"] is False
    assert failed["substitutable_beat_count"] == 1
    assert failed["changed_substitutable_beat_count"] == 0
    assert prior_vertical_selections(prior) == {
        "feature-1": (ASSET_ID, "demo")
    }

    changed_payload = unchanged.model_dump(mode="json")
    changed_payload["chapters"][0]["vertical_candidates"][0][
        "event_id"
    ] = "alternate-demo"
    changed_payload["chapters"][0]["vertical_candidates"][1][
        "event_id"
    ] = "demo"
    changed = FeatureEditPlan.model_validate(changed_payload)
    passed = audit_editorial_freshness(
        changed,
        prior=prior,
        minimum_change_fraction=0.5,
    )

    assert passed["passed"] is True
    assert passed["changed_substitutable_beat_count"] == 1
    assert passed["rows"][0]["changed"] is True


def test_alternate_edit_never_forces_hard_evidence_to_change() -> None:
    prior = project_feature_contracts(
        _v2_plan(),
        brief=_brief(),
        catalog=_catalog(),
    )
    payload = prior.model_dump(mode="json")
    payload["chapters"][0]["vertical_candidates"][1][
        "event_id"
    ] = "alternate-demo"
    current = FeatureEditPlan.model_validate(payload)

    audit = audit_editorial_freshness(
        current,
        prior=prior,
        hard_protected_feature_ids=frozenset({"feature-1"}),
        minimum_change_fraction=1.0,
    )

    assert audit["passed"] is True
    assert audit["substitutable_beat_count"] == 0
    assert audit["rows"][0]["hard_evidence_protected"] is True


def test_direct_video_response_uses_integer_ranks_and_projects_ids_locally() -> None:
    card = _card()
    card = card.model_copy(
        update={
            "events": [
                card.events[0].model_copy(
                    update={"primary_entity_ids": ["sign-1"]}
                )
            ]
        }
    )
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="Visible demonstration.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    direct = DirectVideoEditPlan(
        contract_version="direct-video-edit-plan-v2",
        capability_catalog_sha256=(
            simple_production_capability_catalog().definition_sha256()
        ),
        title="Generic feature cut",
        strategy_summary="Use the visible demonstration.",
        chapters=[
            DirectVideoChapterDecision(
                chapter_index=1,
                evidence_status="partial",
                observed_visual_evidence="The subject demonstrates beside a sign.",
                selection_reason="The action is directly visible.",
                quality_risks=[],
                horizontal=DirectVideoHorizontalDecision(
                    candidate_rank=1,
                    strategy="original",
                    zoom_intent="none",
                    camera_intent="hold",
                ),
                horizontal_alternates=[],
                vertical=DirectVideoVerticalDecision(
                    candidate_rank=1,
                    strategy="tracked_crop",
                    crop_mode="primary_center",
                    coverage_mode="sequential",
                    source_camera_motion_role="editorially_useful",
                    source_camera_motion_reason=(
                        "The bounded candidate visibly reveals the complete "
                        "demonstration while moving."
                    ),
                    allow_controlled_clip=True,
                    framing_intent="Observe the subject, then the sign.",
                    required_entity_indices=[1, 2],
                    preferred_entity_indices=[],
                    sacrificable_entity_indices=[],
                    attention_sequence=[
                        DirectVideoAttentionStep(
                            start_progress=0,
                            end_progress=0.5,
                            anchor_entity_indices=[1],
                            camera_behavior="hold",
                        ),
                            DirectVideoAttentionStep(
                                start_progress=0.5,
                                end_progress=1,
                                anchor_entity_indices=[2],
                                camera_behavior="hold",
                                cut_admissible=True,
                                transition_preference="cut",
                            ),
                    ],
                ),
                vertical_alternates=[],
                recommended_duration_seconds=6,
                duration_rationale="Allow the complete action to read.",
                attention_observation=AttentionObservation(
                    semantic_novelty=0.7,
                    action_progress=0.9,
                    visual_motion=0.4,
                    composition_change=0.2,
                    reading_load=0.3,
                    unresolved_tension=0.1,
                    emotional_hold_value=0.4,
                    repetition_pressure=0.1,
                    music_transition_opportunity=0.6,
                    minimum_dwell_seconds=4,
                    maximum_dwell_seconds=8,
                    rationale="The complete visible action needs a short hold.",
                    uncertainties=[],
                    requires_human_review=True,
                ),
                flow_intent=ShotFlowIntent(
                    narrative_role="proof",
                    energy_role="rise",
                    relation_to_previous="continue_action",
                    boundary_alignment="phrase_preferred",
                    visual_sync_event="result_state",
                    visual_sync_predicate="The visible result becomes stable.",
                    music_target="phrase_end",
                ),
                confidence=0.8,
            )
        ],
        uncertainties=[],
    )

    projected = project_direct_video_edit_plan(
        direct,
        shortlist=shortlist,
        candidate_depth=2,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: card},
        provenance=_provenance(),
    )

    candidate = projected.chapters[0].candidates[0]
    assert candidate.candidate_id == "rank-01"
    assert candidate.source_asset_id == ASSET_ID
    assert candidate.event_id == "demo"
    assert candidate.frame_id == "RF000001"
    assert candidate.source_camera_motion_role == "editorially_useful"
    assert projected.chapters[0].horizontal_candidate_id == "rank-01"
    assert projected.chapters[0].vertical_candidate_id == "rank-01"
    assert (
        projected.chapters[0].attention_observation is not None
    )
    executable = project_feature_contracts_v3(
        projected,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=build_selected_clip_card_evidence(
            projected,
            cards={ASSET_ID: card},
        ),
    )
    minimum_visibility = {
        region.entity_id: region.minimum_visible_fraction
        for region in executable.chapters[0].vertical_candidates[0].regions
    }
    assert minimum_visibility == {
        "subject-1": 0.6,
        "sign-1": 1.0,
    }
    virtual_camera = projected.chapters[0].candidates[0].virtual_camera_proposal
    assert virtual_camera is not None
    assert virtual_camera.phases[1].transition_in == "cut"
    stale_plan = direct.model_copy(
        update={"capability_catalog_sha256": "b" * 64}
    )
    with pytest.raises(ValueError, match="capability catalog differs"):
        project_direct_video_edit_plan(
            stale_plan,
            shortlist=shortlist,
            candidate_depth=2,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
            provenance=_provenance(),
        )
    direct_schema = json.dumps(gemini_response_schema(DirectVideoEditPlan))
    assert '"candidate_rank"' in direct_schema
    assert '"chapter_index"' in direct_schema
    assert '"entity_index"' not in direct_schema
    assert '"required_entity_indices"' in direct_schema
    assert '"vertical_alternates"' in direct_schema
    assert '"source_camera_motion_role"' in direct_schema
    assert '"candidate_id"' not in direct_schema
    assert '"source_asset_id"' not in direct_schema
    assert '"frame_id"' not in direct_schema
    assert '"project_id"' not in direct_schema
    assert '"model_provenance"' not in direct_schema


def test_vertical_only_direct_plan_projects_neutral_unrequested_horizontal_shadow() -> None:
    """A requested 9:16-only plan must not be rejected for omitting 16:9."""

    card = _card()
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="Visible demonstration.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    direct = DirectVideoEditPlan(
        contract_version="direct-video-edit-plan-v2",
        capability_catalog_sha256=(
            simple_production_capability_catalog().definition_sha256()
        ),
        title="Vertical only",
        strategy_summary="Use the visible vertical candidate.",
        chapters=[
            DirectVideoChapterDecision(
                chapter_index=1,
                evidence_status="partial",
                observed_visual_evidence="A subject is visibly demonstrated.",
                selection_reason="The action is visible.",
                horizontal=None,
                horizontal_alternates=[],
                vertical=DirectVideoVerticalDecision(
                    candidate_rank=1,
                    strategy="tracked_crop",
                    crop_mode="primary_center",
                    coverage_mode="independent_detail",
                    allow_controlled_clip=True,
                    framing_intent="Keep the visible subject readable.",
                    required_entity_indices=[1],
                    preferred_entity_indices=[],
                    sacrificable_entity_indices=[],
                    attention_sequence=[],
                ),
                vertical_alternates=[],
                recommended_duration_seconds=4,
                duration_rationale="Enough time for the visible action.",
                attention_observation=AttentionObservation(
                    semantic_novelty=0.6,
                    action_progress=0.7,
                    visual_motion=0.3,
                    composition_change=0.2,
                    reading_load=0.2,
                    unresolved_tension=0.1,
                    emotional_hold_value=0.2,
                    repetition_pressure=0.1,
                    music_transition_opportunity=0.5,
                    minimum_dwell_seconds=3,
                    maximum_dwell_seconds=6,
                    rationale="The visible action is brief.",
                    uncertainties=[],
                    requires_human_review=True,
                ),
                flow_intent=ShotFlowIntent(
                    narrative_role="proof",
                    energy_role="low_hold",
                    relation_to_previous="new_context",
                    boundary_alignment="phrase_preferred",
                    visual_sync_event="intentional_hold",
                    visual_sync_predicate="The visible subject holds briefly.",
                    music_target="phrase_end",
                ),
                confidence=0.8,
            )
        ],
        uncertainties=[],
    )

    projected = project_direct_video_edit_plan(
        direct,
        shortlist=shortlist,
        candidate_depth=2,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: card},
        provenance=_provenance(),
    )

    assert direct.chapters[0].horizontal is None
    unsuitable = direct.model_copy(
        update={
            "chapters": [
                direct.chapters[0].model_copy(
                    update={
                        "vertical": direct.chapters[0].vertical.model_copy(
                            update={"aspect_suitability": "unsuitable"}
                        )
                    }
                )
            ]
        }
    )
    with pytest.raises(
        ValueError,
        match="no Gemini-authorised executable 9:16 candidate",
    ):
        validate_direct_video_plan_fulfillment(
            unsuitable,
            shortlist=shortlist,
            cards={ASSET_ID: card},
            contracts=(),
            candidate_depth=2,
            requested_aspects=("9:16",),
        )
    assert projected.chapters[0].vertical_candidate_id == "rank-01"
    assert projected.chapters[0].horizontal_candidate_id == "rank-01"
    shadow = projected.chapters[0].candidates[0]
    assert shadow.horizontal_strategy == "original"
    assert shadow.horizontal_zoom_intent == "none"
    assert shadow.horizontal_camera_intent == "hold"


def test_direct_video_aspect_contract_is_required_nullable_and_fail_closed() -> None:
    horizontal = DirectVideoHorizontalDecision(
        candidate_rank=1,
        strategy="original",
        zoom_intent="none",
        camera_intent="hold",
    )
    vertical = DirectVideoVerticalDecision(
        candidate_rank=1,
        strategy="tracked_crop",
        crop_mode="primary_center",
        coverage_mode="independent_detail",
        allow_controlled_clip=True,
        framing_intent="Keep the visible subject readable.",
        required_entity_indices=[1],
        preferred_entity_indices=[],
        sacrificable_entity_indices=[],
        attention_sequence=[],
    )
    horizontal_only = DirectVideoEditPlan(
        contract_version="direct-video-edit-plan-v2",
        capability_catalog_sha256=(
            simple_production_capability_catalog().definition_sha256()
        ),
        title="Horizontal only",
        strategy_summary="Use the verified landscape take.",
        chapters=[
            _direct_chapter_for_aspects(horizontal=horizontal, vertical=None)
        ],
        uncertainties=[],
    )
    assert validate_direct_video_aspect_contract(
        horizontal_only,
        requested_aspects=("16:9",),
    ) == ("16:9",)
    with pytest.raises(ValueError, match="omitted requested aspect 9:16"):
        validate_direct_video_aspect_contract(
            horizontal_only,
            requested_aspects=("9:16",),
        )

    vertical_only = horizontal_only.model_copy(
        update={
            "chapters": [
                _direct_chapter_for_aspects(horizontal=None, vertical=vertical)
            ]
        }
    )
    assert validate_direct_video_aspect_contract(
        vertical_only,
        requested_aspects=("9:16",),
    ) == ("9:16",)
    with pytest.raises(ValueError, match="supplied unrequested aspect 9:16"):
        validate_direct_video_aspect_contract(
            horizontal_only.model_copy(
                update={
                    "chapters": [
                        _direct_chapter_for_aspects(
                            horizontal=horizontal,
                            vertical=vertical,
                        )
                    ]
                }
            ),
            requested_aspects=("16:9",),
        )

    chapter_schema = direct_video_response_schema(("16:9",))["$defs"][
        "DirectVideoChapterDecision"
    ]
    assert {
        "horizontal",
        "horizontal_alternates",
        "vertical",
        "vertical_alternates",
        "recommended_duration_seconds",
        "duration_rationale",
        "attention_observation",
        "flow_intent",
    }.issubset(chapter_schema["required"])
    assert "Must be null because 9:16 is not requested." in chapter_schema[
        "properties"
    ]["vertical"]["description"]
    assert "non-null 9:16 primary" in direct_video_response_schema(("9:16",))["$defs"][
        "DirectVideoChapterDecision"
    ]["properties"]["vertical"]["description"]
    assert "只要求：16:9" in direct_video_aspect_contract_instructions(("16:9",))
    assert "horizontal_alternates` 必須是 []" in (
        direct_video_aspect_contract_instructions(("9:16",))
    )
    assert "horizontal_alternates` 與 `vertical_alternates`" in (
        direct_video_aspect_contract_instructions(("16:9", "9:16"))
    )


def test_horizontal_only_prompt_projects_out_inactive_aspect_context() -> None:
    projected_brief = project_direct_video_prompt_payload(
        _brief().model_dump(mode="json"),
        requested_aspects=("16:9",),
    )
    chapter = projected_brief["chapters"][0]
    assert "vertical_primary_target_description" not in chapter
    instructions = direct_video_horizontal_only_prompt_instructions().lower()
    assert "vertical" not in instructions
    assert "9:16" not in instructions
    assert "two_panel" not in instructions
    assert "horizontal_alternates" in instructions


def test_horizontal_decision_keeps_motion_and_reuse_authority_candidate_scoped() -> None:
    with pytest.raises(ValidationError, match="observable reason"):
        DirectVideoHorizontalDecision(
            candidate_rank=1,
            strategy="original",
            zoom_intent="none",
            camera_intent="hold",
            source_camera_motion_role="editorially_useful",
        )
    decision = DirectVideoHorizontalDecision(
        candidate_rank=1,
        strategy="original",
        zoom_intent="none",
        camera_intent="hold",
        source_camera_motion_role="incidental_or_unwanted",
        source_camera_motion_reason="The bounded take visibly starts with a shake.",
        source_reuse_mode="alternate_presentation",
        source_reuse_justification="The alternate wide take has a distinct reading purpose.",
    )
    assert decision.source_camera_motion_role == "incidental_or_unwanted"
    assert decision.source_reuse_mode == "alternate_presentation"


def test_direct_video_response_schema_requires_candidate_semantic_authority() -> None:
    schema = direct_video_response_schema(("16:9",))
    for decision_name in (
        "DirectVideoHorizontalDecision",
        "DirectVideoVerticalDecision",
    ):
        required = set(schema["$defs"][decision_name]["required"])
        assert {
            "source_camera_motion_role",
            "source_camera_motion_reason",
            "source_reuse_mode",
            "source_reuse_justification",
        } <= required


def test_horizontal_only_projection_keeps_vertical_shadow_inactive() -> None:
    horizontal = DirectVideoHorizontalDecision(
        candidate_rank=1,
        strategy="original",
        zoom_intent="none",
        camera_intent="hold",
    )
    direct = DirectVideoEditPlan(
        contract_version="direct-video-edit-plan-v2",
        capability_catalog_sha256=(
            simple_production_capability_catalog().definition_sha256()
        ),
        title="Horizontal only",
        strategy_summary="Use the verified landscape take.",
        chapters=[
            _direct_chapter_for_aspects(horizontal=horizontal, vertical=None)
        ],
        uncertainties=[],
    )
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="The action is visible.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    projected = project_direct_video_edit_plan(
        direct,
        shortlist=shortlist,
        candidate_depth=1,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: _card()},
        provenance=_provenance(),
        requested_aspects=("16:9",),
    )
    candidate = projected.chapters[0].candidates[0]
    assert candidate.horizontal_authorized is True
    assert candidate.vertical_authorized is False
    executable = project_feature_contracts_v3(
        projected,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=build_selected_clip_card_evidence(
            projected,
            cards={ASSET_ID: _card()},
        ),
    )
    chapter = executable.chapters[0]
    assert chapter.inactive_aspects == ["9:16"]
    assert len(chapter.horizontal_candidates) == 1
    assert chapter.vertical_candidates == []
    assert "unrequested_9_16_projection_shadow" not in chapter.quality_risks


def test_active_aspect_confidence_ignores_inactive_shadow_primary() -> None:
    plan = _v3_plan()
    chapter = plan.chapters[0]
    plan = plan.model_copy(
        update={
            "chapters": [
                chapter.model_copy(
                    update={
                        "candidates": [
                            candidate.model_copy(
                                update={"vertical_authorized": False}
                            )
                            for candidate in chapter.candidates
                        ]
                    }
                )
            ]
        }
    )
    executable = project_feature_contracts_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=build_selected_clip_card_evidence(
            plan,
            cards={ASSET_ID: _card()},
        ),
    )
    assert executable.chapters[0].inactive_aspects == ["9:16"]
    assert executable.chapters[0].confidence == 0.8


def test_direct_source_allocation_requires_explicit_reuse_but_has_no_count_cap() -> None:
    base = DirectVideoChapterDecision(
        chapter_index=1,
        evidence_status="partial",
        observed_visual_evidence="A product is visibly present.",
        selection_reason="The visible product supports this chapter.",
        horizontal=DirectVideoHorizontalDecision(
            candidate_rank=1,
            strategy="original",
            zoom_intent="none",
            camera_intent="hold",
        ),
        horizontal_alternates=[],
        vertical=DirectVideoVerticalDecision(
            candidate_rank=1,
            strategy="tracked_crop",
            crop_mode="primary_center",
            coverage_mode="independent_detail",
            allow_controlled_clip=True,
            framing_intent="Hold the visible product detail.",
            required_entity_indices=[1],
            preferred_entity_indices=[],
            sacrificable_entity_indices=[],
            attention_sequence=[],
        ),
        vertical_alternates=[],
        recommended_duration_seconds=4,
        duration_rationale="The product detail remains readable.",
        attention_observation=AttentionObservation(
            semantic_novelty=0.5,
            action_progress=1.0,
            visual_motion=0.0,
            composition_change=0.0,
            reading_load=0.2,
            unresolved_tension=0.0,
            emotional_hold_value=0.2,
            repetition_pressure=0.1,
            music_transition_opportunity=0.3,
            minimum_dwell_seconds=3,
            maximum_dwell_seconds=5,
            rationale="A short product hold is readable.",
            uncertainties=[],
            requires_human_review=True,
        ),
        flow_intent=ShotFlowIntent(
            narrative_role="development",
            energy_role="low_hold",
            relation_to_previous="new_context",
            boundary_alignment="phrase_preferred",
        ),
        confidence=0.8,
    )
    alternate = base.model_copy(
        update={
            "chapter_index": 2,
            "source_reuse_mode": "alternate_presentation",
            "source_reuse_justification": "A tighter view has a distinct reading purpose.",
        }
    )
    reprise = base.model_copy(
        update={
            "chapter_index": 3,
            "source_reuse_mode": "editorial_reprise",
            "source_reuse_justification": "Return to the same product for the ending.",
        }
    )
    plan = DirectVideoEditPlan(
        contract_version="direct-video-edit-plan-v2",
        capability_catalog_sha256="a" * 64,
        title="Three authorized views",
        strategy_summary="One source deliberately serves distinct editorial roles.",
        chapters=[base, alternate, reprise],
        uncertainties=[],
    )
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id=feature_id,
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="The product is visible.",
                    )
                ],
            )
            for feature_id in ("opening", "detail", "closing")
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )

    validate_direct_video_source_allocation(
        plan,
        shortlist=shortlist,
        candidate_depth=1,
    )

    untyped_repeat = plan.model_copy(
        update={
            "chapters": [base, alternate, base.model_copy(update={"chapter_index": 3})]
        }
    )
    with pytest.raises(ValueError, match="without Gemini reuse authority"):
        validate_direct_video_source_allocation(
            untyped_repeat,
            shortlist=shortlist,
            candidate_depth=1,
        )


def test_direct_video_canonicalization_preserves_phase_local_anchors() -> None:
    payload = {
        "contract_version": "direct-video-edit-plan-v2",
        "capability_catalog_sha256": "a" * 64,
        "chapters": [
            {
                "evidence_status": "supported",
                "horizontal": {
                    "candidate_rank": 1,
                    "strategy": "original",
                    "zoom_intent": "none",
                    "camera_intent": "hold",
                    "focus_entity_index": None,
                },
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "primary_center",
                    "coverage_mode": "sequential",
                    "allow_controlled_clip": False,
                    "framing_intent": "Observe A and then B.",
                    "required_entity_indices": [1, 2],
                    "preferred_entity_indices": [],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [
                        {
                            "start_progress": 0.0,
                            "end_progress": 0.5,
                            "anchor_entity_indices": [1],
                            "camera_behavior": "hold",
                        },
                        {
                            "start_progress": 0.5,
                            "end_progress": 1.0,
                            "anchor_entity_indices": [2],
                            "camera_behavior": "follow_deadband",
                        },
                    ],
                },
            }
        ],
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]

    assert vertical["attention_sequence"][0]["anchor_entity_indices"] == [1]
    assert vertical["attention_sequence"][1]["anchor_entity_indices"] == [2]
    assert all(
        change["rule"] != "required_entities_remain_visible_in_attention_phase"
        for change in changes
    )


def test_direct_video_canonicalization_demotes_unexplained_source_motion() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "vertical": {
                    "candidate_rank": 1,
                    "source_camera_motion_role": "editorially_useful",
                    "source_camera_motion_reason": None,
                },
                "vertical_alternates": [
                    {
                        "candidate_rank": 2,
                        "strategy": "tracked_crop",
                        "crop_mode": "strict",
                        "coverage_mode": "relation_core",
                        "allow_controlled_clip": True,
                        "source_camera_motion_role": "static_or_negligible",
                        "source_camera_motion_reason": "",
                        "required_entity_indices": [1, 2],
                        "preferred_entity_indices": [3],
                        "sacrificable_entity_indices": [3],
                        "attention_sequence": [
                            {
                                "start_progress": 0.0,
                                "end_progress": 1.0,
                                "anchor_entity_indices": [1],
                                "camera_behavior": "hold",
                            }
                        ],
                    }
                ],
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    chapter = json.loads(canonical)["chapters"][0]

    assert chapter["vertical"]["source_camera_motion_role"] == "unknown"
    assert (
        chapter["vertical_alternates"][0]["source_camera_motion_role"]
        == "unknown"
    )
    assert chapter["vertical_alternates"][0]["crop_mode"] == "primary_center"
    assert chapter["vertical_alternates"][0]["sacrificable_entity_indices"] == []
    assert chapter["vertical_alternates"][0]["coverage_mode"] == "relation_core"
    assert chapter["vertical_alternates"][0]["attention_sequence"][0][
        "anchor_entity_indices"
    ] == [1]
    assert "aspect_suitability" not in chapter["vertical_alternates"][0]
    assert "suitability_risks" not in chapter["vertical_alternates"][0]
    assert {
        change["rule"] for change in changes
    } >= {
        "unexplained_source_camera_motion_classification_fails_safe_to_unknown",
        "explicit_controlled_clip_uses_primary_center_representation",
        "entity_role_precedence_required_preferred_sacrificable",
    }


def test_direct_video_canonicalization_only_removes_incomplete_optional_sync() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "flow_intent": {
                    "narrative_role": "proof",
                    "energy_role": "rise",
                    "relation_to_previous": "contrast",
                    "boundary_alignment": "phrase_preferred",
                    "visual_sync_event": None,
                    "visual_sync_predicate": None,
                    "music_target": "downbeat",
                },
                "horizontal": {
                    "candidate_rank": 1,
                    "strategy": "original",
                    "zoom_intent": "none",
                    "camera_intent": "hold",
                    "focus_entity_index": None,
                },
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "strict",
                    "coverage_mode": "primary_with_context",
                    "allow_controlled_clip": True,
                    "required_entity_indices": [1],
                    "preferred_entity_indices": [2],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    chapter = json.loads(canonical)["chapters"][0]

    assert chapter["vertical"]["crop_mode"] == "primary_center"
    assert chapter["flow_intent"]["visual_sync_event"] is None
    assert chapter["flow_intent"]["visual_sync_predicate"] is None
    assert chapter["flow_intent"]["music_target"] is None
    assert {
        change["rule"] for change in changes
    } >= {
        "explicit_controlled_clip_uses_primary_center_representation",
        "incomplete_optional_visual_sync_is_removed_rather_than_invented",
    }


def test_direct_video_canonicalization_derives_clip_permission_from_explicit_primary_center_semantic_core() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "primary_center",
                    "coverage_mode": "primary_with_context",
                    "allow_controlled_clip": False,
                    "required_entity_indices": [1],
                    "preferred_entity_indices": [],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [
                        {
                            "start_progress": 0.0,
                            "end_progress": 1.0,
                            "anchor_entity_indices": [1],
                            "camera_behavior": "hold",
                        }
                    ],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]

    assert vertical["allow_controlled_clip"] is True
    assert {
        change["rule"] for change in changes
    } >= {"explicit_primary_center_semantic_core_authorizes_controlled_clipping"}


def test_direct_video_canonicalization_keeps_static_fit_without_camera_attention() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "fit_with_background",
                    "crop_mode": "strict",
                    "coverage_mode": "simultaneous",
                    "allow_controlled_clip": False,
                    "required_entity_indices": [1, 2],
                    "preferred_entity_indices": [],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [],
                },
            }
        ]
    }

    canonical, _changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]

    assert vertical.get("aspect_suitability") != "unsuitable"
    assert vertical["attention_sequence"] == []


def test_direct_video_canonicalization_disables_clipping_for_fit() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "horizontal": {
                    "candidate_rank": 1,
                    "strategy": "original",
                    "zoom_intent": "none",
                    "camera_intent": "hold",
                    "focus_entity_index": None,
                },
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "fit_with_background",
                    "crop_mode": "strict",
                    "coverage_mode": "simultaneous",
                    "allow_controlled_clip": True,
                    "required_entity_indices": [1, 2],
                    "preferred_entity_indices": [],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]

    assert vertical["allow_controlled_clip"] is False
    assert any(
        change["json_path"]
        == "chapters[0].vertical.allow_controlled_clip"
        and change["rule"]
        == "fit_with_background_preserves_scope_without_controlled_clipping"
        for change in changes
    )


def test_direct_video_canonicalization_repairs_bounded_representation_conflicts() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "observed_visual_evidence": "A second, separate action is visible.",
                "selection_reason": "A second, separate action is visible.",
                "source_reuse_mode": "distinct_interval",
                "source_reuse_justification": None,
                "horizontal": {
                    "candidate_rank": 1,
                    "strategy": "original",
                    "zoom_intent": "none",
                    "camera_intent": "hold",
                    "focus_entity_index": None,
                },
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "primary_center",
                    "coverage_mode": "sequential",
                    "presentation_preference": "two_panel_layout",
                    "allow_controlled_clip": True,
                    "framing_intent": "Observe each subject in sequence.",
                    "required_entity_indices": [1, 2],
                    "preferred_entity_indices": [3, 4, 5, 6, 7],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [
                        {
                            "start_progress": 0.0,
                            "end_progress": 0.5,
                            "anchor_entity_indices": [1],
                            "camera_behavior": "hold",
                        },
                        {
                            "start_progress": 0.5,
                            "end_progress": 1.0,
                            "anchor_entity_indices": [2],
                            "camera_behavior": "follow_deadband",
                        },
                    ],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    chapter = json.loads(canonical)["chapters"][0]

    assert chapter["source_reuse_mode"] == "distinct_interval"
    assert chapter["source_reuse_justification"] == "A second, separate action is visible."
    assert chapter["vertical"]["presentation_preference"] == "two_panel_layout"
    assert chapter["vertical"]["preferred_entity_indices"] == [3, 4, 5, 6]
    assert chapter["vertical"]["aspect_suitability"] == "unsuitable"
    assert {
        change["rule"] for change in changes
    } >= {
        "model_selection_reason_projects_to_missing_reuse_audit_field",
        "planner_contract_candidate_marked_unsuitable",
    }


def test_direct_video_reuse_without_reason_removes_unproven_authority() -> None:
    """A normalizer preserves a take but cannot invent reuse authority."""

    with pytest.raises(ValidationError, match="observable justification"):
        DirectVideoChapterDecision.model_validate(
            {
                "chapter_index": 1,
                "evidence_status": "supported",
                "observed_visual_evidence": "A product is visible.",
                    "selection_reason": "A deliberate return is requested.",
                    "horizontal": None,
                    "horizontal_alternates": [],
                    "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "primary_center",
                    "coverage_mode": "primary_with_context",
                    "aspect_suitability": "unsuitable",
                    "framing_intent": "Keep the product visible.",
                    "required_entity_indices": [],
                    "preferred_entity_indices": [],
                    "sacrificable_entity_indices": [],
                        "attention_sequence": [],
                    },
                    "vertical_alternates": [],
                "recommended_duration_seconds": 4.0,
                "duration_rationale": "Enough time to read the closing image.",
                "attention_observation": None,
                "flow_intent": None,
                "source_reuse_mode": "editorial_reprise",
                "source_reuse_justification": None,
                "confidence": 0.9,
            }
        )

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "evidence_status": "supported",
                        "vertical": {
                            "candidate_rank": 1,
                            "source_reuse_mode": "editorial_reprise",
                            "source_reuse_justification": None,
                        },
                    }
                ]
            }
        )
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]
    assert vertical["source_reuse_mode"] == "none"
    assert vertical["source_reuse_justification"] is None
    assert "unexplained_candidate_reuse_authority_removed_fail_closed" in {
        change["rule"] for change in changes
    }


def test_hard_shortlist_provenance_blocks_prerecorded_playback() -> None:
    card = _card().model_copy(
        update={
            "events": [
                _card().events[0].model_copy(
                    update={"evidence_provenance": "prerecorded_screen_playback"}
                )
            ]
        }
    )
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="A nested demonstration is visible.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    contract = EditorialBeatContract.model_validate(
        {
            "beat_id": "hard-result",
            "feature_id": "feature-1",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["result"],
            "allowed_evidence_provenance": ["direct_result"],
            "narrative_function": "global_energy_peak",
            "visual_events": [
                {
                    "event_type": "result_stable_start",
                    "cue_relation": "principal_downbeat",
                    "tolerance_frames": 2,
                }
            ],
            "duration": {
                "minimum_readable_frames": 12,
                "preferred_frames": 24,
                "maximum_frames": 48,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )

    with pytest.raises(
        ValueError,
        match="blocked before candidate-video upload",
    ):
        validate_hard_shortlist_provenance(
            shortlist,
            cards={ASSET_ID: card},
            contracts=(contract,),
            candidate_depth=3,
            direct_video_evidence=True,
        )


def test_hard_shortlist_accepts_authorized_contextual_minimum() -> None:
    card = _card().model_copy(
        update={
            "events": [
                _card().events[0].model_copy(
                    update={
                        "evidence_provenance": (
                            "prerecorded_screen_playback"
                        )
                    }
                )
            ]
        }
    )
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="Illustrative playback is visible.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    contract = EditorialBeatContract.model_validate(
        {
            "beat_id": "hard-chapter",
            "feature_id": "feature-1",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["product"],
            "narrative_function": "feature_evidence",
            "visual_events": [
                {
                    "event_type": "result_stable_start",
                    "cue_relation": "principal_downbeat",
                    "tolerance_frames": 2,
                }
            ],
            "minimum_fulfillment_level": "contextual_identity",
            "fulfillment_alternatives": [
                {
                    "fulfillment_level": "direct_demonstration",
                    "accepted_evidence_provenance": ["direct_result"],
                    "claim_support_level": "direct",
                    "exact_event_requirement": "required_when_selected",
                    "visual_events": [
                        {
                            "event_type": "result_stable_start",
                            "cue_relation": "principal_downbeat",
                            "tolerance_frames": 2,
                        }
                    ],
                },
                {
                    "fulfillment_level": "contextual_identity",
                    "accepted_evidence_provenance": [
                        "prerecorded_screen_playback"
                    ],
                    "claim_support_level": "illustrative_only",
                    "exact_event_requirement": "none",
                    "degradation_codes": [
                        "contextual_visual_substitution"
                    ],
                    "copy_suppression_codes": [
                        "specific_claim_copy_suppressed"
                    ],
                },
            ],
            "duration": {
                "minimum_readable_frames": 12,
                "preferred_frames": 24,
                "maximum_frames": 48,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )

    validate_hard_shortlist_provenance(
        shortlist,
        cards={ASSET_ID: card},
        contracts=(contract,),
        candidate_depth=3,
        direct_video_evidence=True,
    )
    validate_shortlist_fulfillment_minimums(
        shortlist,
        cards={ASSET_ID: card},
        contracts=(contract,),
    )


def test_direct_video_plan_cannot_drop_hard_fulfillment_chapter() -> None:
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="A candidate is available.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    direct = DirectVideoEditPlan(
        contract_version="direct-video-edit-plan-v2",
        capability_catalog_sha256="1" * 64,
        title="Plan",
        strategy_summary="No supported chapter.",
        chapters=[
            DirectVideoChapterDecision(
                chapter_index=1,
                evidence_status="not_found",
                    observed_visual_evidence="",
                    selection_reason="No selection.",
                    horizontal=None,
                    horizontal_alternates=[],
                    vertical=None,
                    vertical_alternates=[],
                    recommended_duration_seconds=None,
                    duration_rationale=None,
                    attention_observation=None,
                flow_intent=None,
                confidence=0.0,
            )
        ],
        uncertainties=[],
    )
    contract = EditorialBeatContract.model_validate(
        {
            "beat_id": "hard-result",
            "feature_id": "feature-1",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["result"],
            "allowed_evidence_provenance": ["direct_result"],
            "narrative_function": "feature_evidence",
            "visual_events": [
                {
                    "event_type": "result_stable_start",
                    "cue_relation": "principal_downbeat",
                    "tolerance_frames": 2,
                }
            ],
            "duration": {
                "minimum_readable_frames": 12,
                "preferred_frames": 24,
                "maximum_frames": 48,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )

    with pytest.raises(
        ValueError,
        match="hard beat cannot use a not-found",
    ):
        validate_direct_video_plan_fulfillment(
            direct,
            shortlist=shortlist,
            cards={ASSET_ID: _card()},
            contracts=(contract,),
            candidate_depth=3,
        )


def test_hard_shortlist_and_selected_snapshot_use_effective_origin() -> None:
    card = _card().model_copy(
        update={
            "events": [
                _card().events[0].model_copy(
                    update={"evidence_provenance": "direct_result"}
                )
            ]
        }
    )
    event = card.events[0]
    supplement = ClipObservationSupplement(
        supplement_id="origin-correction",
        source_asset_id=card.source_asset_id,
        proxy_asset_id=card.proxy_asset_id,
        base_card_sha256=clip_card_sha256(card),
        supplement_prompt_sha256="c" * 64,
        response_schema_sha256="d" * 64,
        event_observations=[
            EventObservationSupplement(
                event_id=event.event_id,
                event_fingerprint=event_fingerprint(event),
                observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
                evidence_provenance="prerecorded_screen_playback",
                evidence_origin=EvidenceOriginObservation(
                    relation="mediated_depiction",
                    observable_reason=(
                        "The source scene shows a display playing another video."
                    ),
                ),
                capabilities=EventCapabilityManifest(
                    evidence_origin=AssessmentStatus.ASSESSED_PRESENT,
                ),
            )
        ],
        model_provenance=card.model_provenance,
    )
    shortlist = FeatureShortlistPlan(
        project_id="project-1",
        catalog_id="catalog-1",
        chapters=[
            FeatureChapterShortlist(
                feature_id="feature-1",
                evidence_status="partial",
                candidates=[
                    FeatureShortlistCandidate(
                        source_asset_id=ASSET_ID,
                        event_id="demo",
                        retrieval_reason="A displayed result is visible.",
                    )
                ],
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )
    contract = EditorialBeatContract.model_validate(
        {
            "beat_id": "hard-result",
            "feature_id": "feature-1",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["result"],
            "allowed_evidence_provenance": ["direct_result"],
            "narrative_function": "global_energy_peak",
            "visual_events": [
                {
                    "event_type": "result_stable_start",
                    "cue_relation": "principal_downbeat",
                    "tolerance_frames": 2,
                }
            ],
            "duration": {
                "minimum_readable_frames": 12,
                "preferred_frames": 24,
                "maximum_frames": 48,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )

    with pytest.raises(ValueError, match="blocked before candidate-video upload"):
        validate_hard_shortlist_provenance(
            shortlist,
            cards={ASSET_ID: card},
            contracts=(contract,),
            candidate_depth=3,
            direct_video_evidence=True,
            supplements={ASSET_ID: [supplement]},
        )

    evidence = build_selected_clip_card_evidence(
        _v3_plan(),
        cards={ASSET_ID: card},
        supplements={ASSET_ID: [supplement]},
    )
    selected = evidence.events[0]
    assert evidence.contract_version == (
        "clip-card-feature-cut-selected-evidence-v3"
    )
    assert selected.evidence_provenance == "prerecorded_screen_playback"
    assert selected.evidence_origin is not None
    assert selected.evidence_origin.relation == "mediated_depiction"
    assert selected.effective_observation_sha256 is not None


def test_direct_video_canonicalization_keeps_attention_distinct_from_visibility() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "recommended_duration_seconds": None,
                "attention_observation": {
                    "minimum_dwell_seconds": 4.0,
                    "maximum_dwell_seconds": 8.0,
                },
                "horizontal": {
                    "candidate_rank": 1,
                    "strategy": "original",
                    "zoom_intent": "none",
                    "camera_intent": "hold",
                    "focus_entity_index": None,
                },
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "primary_center",
                    "coverage_mode": "sequential",
                    "allow_controlled_clip": True,
                    "framing_intent": "Observe all required subjects.",
                    "required_entity_indices": [1, 2, 3],
                    "preferred_entity_indices": [4],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [
                        {
                            "start_progress": 0.0,
                            "end_progress": 0.5,
                            "anchor_entity_indices": [1],
                            "camera_behavior": "hold",
                        },
                        {
                            "start_progress": 0.5,
                            "end_progress": 1.0,
                            "anchor_entity_indices": [2],
                            "camera_behavior": "follow",
                        },
                    ],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    chapter = json.loads(canonical)["chapters"][0]

    assert chapter["recommended_duration_seconds"] == 6.0
    assert chapter["vertical"]["coverage_mode"] == "sequential"
    assert chapter["vertical"].get("traversal_policy") is None
    assert [
        phase["anchor_entity_indices"]
        for phase in chapter["vertical"]["attention_sequence"]
    ] == [[1], [2]]
    assert "aspect_suitability" not in chapter["vertical"]
    assert {
        change["rule"] for change in changes
    } >= {
        "missing_recommended_duration_uses_model_attention_midpoint",
    }


def test_direct_video_canonicalization_preserves_locked_attention_sequence() -> None:
    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "recommended_duration_seconds": 5.0,
                "horizontal": {
                    "candidate_rank": 1,
                    "strategy": "original",
                    "zoom_intent": "none",
                    "camera_intent": "hold",
                    "focus_entity_index": None,
                },
                "vertical": {
                    "candidate_rank": 1,
                    "strategy": "tracked_crop",
                    "crop_mode": "primary_center",
                    "coverage_mode": "sequential",
                    "traversal_policy": "semantic_order_locked",
                    "allow_controlled_clip": True,
                    "framing_intent": "Observe subjects in declared order.",
                    "required_entity_indices": [1, 2, 3],
                    "preferred_entity_indices": [],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [
                        {
                            "start_progress": 0.0,
                            "end_progress": 0.5,
                            "anchor_entity_indices": [1],
                            "camera_behavior": "hold",
                        },
                        {
                            "start_progress": 0.5,
                            "end_progress": 1.0,
                            "anchor_entity_indices": [2],
                            "camera_behavior": "follow",
                        },
                    ],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]

    assert vertical["coverage_mode"] == "sequential"
    assert vertical["traversal_policy"] == "semantic_order_locked"
    assert [
        phase["anchor_entity_indices"]
        for phase in vertical["attention_sequence"]
    ] == [[1], [2]]
    assert "aspect_suitability" not in vertical
    assert {
        change["rule"] for change in changes
    } == {"missing_required_nullable_key_is_explicit"}


def test_unsuitable_direct_candidate_is_bounded_before_schema_validation() -> None:
    """An impossible alternate remains reported, rather than aborting the plan."""

    payload = {
        "chapters": [
            {
                "evidence_status": "supported",
                "vertical": {
                    "strategy": "tracked_crop",
                    "crop_mode": "strict",
                    "coverage_mode": "simultaneous",
                    "aspect_suitability": "reconstructable",
                    "presentation_preference": "tracked_full_bleed",
                    "required_entity_indices": [1, 2, 3, 4, 5, 6],
                    "preferred_entity_indices": [7, 8, 9, 10, 11],
                    "sacrificable_entity_indices": [],
                    "attention_sequence": [
                        {
                            "anchor_entity_indices": [1, 2, 3, 4, 5, 6],
                        }
                    ],
                },
            }
        ]
    }

    canonical, changes = canonicalize_direct_video_edit_plan_output(
        json.dumps(payload)
    )
    vertical = json.loads(canonical)["chapters"][0]["vertical"]

    assert vertical["aspect_suitability"] == "unsuitable"
    assert len(vertical["required_entity_indices"]) == 4
    assert len(vertical["preferred_entity_indices"]) == 4
    assert len(vertical["attention_sequence"][0]["anchor_entity_indices"]) == 4
    assert "unsuitable_candidate_execution_indices_bounded_without_authorizing_a_crop" in {
        change["rule"] for change in changes
    }


def test_failed_plan_resume_selects_latest_complete_paid_attempt(
    tmp_path: Path,
) -> None:
    for attempt in (1, 2):
        stem = f"clip-card-feature-plan.attempt-{attempt:02d}"
        write_json(
            tmp_path / f"{stem}.request.json",
            {"model": "gemini-3.6-flash", "input": []},
        )
        write_json(
            tmp_path / f"{stem}.raw_output.json",
            {"output_text": "{}"},
        )
        write_json(
            tmp_path / f"{stem}.raw_interaction.json",
            {"output_text": "{}"},
        )
        write_json(
            tmp_path / f"{stem}.schema-validation.json",
            {"ok": False, "error": f"attempt {attempt} failed"},
        )

    resolved = _resolve_latest_failed_feature_plan_attempt(tmp_path)

    assert resolved["attempt_number"] == 2
    assert resolved["error"] == "attempt 2 failed"
    assert Path(resolved["request"]).name.endswith("attempt-02.request.json")


def test_simultaneous_coverage_cannot_be_split_across_phases() -> None:
    with pytest.raises(ValidationError, match="simultaneous coverage"):
        DirectVideoVerticalDecision(
            candidate_rank=1,
            strategy="tracked_crop",
            crop_mode="strict",
            coverage_mode="simultaneous",
            framing_intent="Keep the relation visible.",
            required_entity_indices=[1, 2],
            preferred_entity_indices=[],
            sacrificable_entity_indices=[],
            attention_sequence=[
                DirectVideoAttentionStep(
                    start_progress=0,
                    end_progress=0.5,
                    anchor_entity_indices=[1, 2],
                    camera_behavior="hold",
                ),
                DirectVideoAttentionStep(
                    start_progress=0.5,
                    end_progress=1,
                    anchor_entity_indices=[1, 2],
                    camera_behavior="hold",
                ),
            ],
        )


def test_phase_virtual_camera_requires_gemini_described_sequential_phases() -> None:
    with pytest.raises(ValidationError, match="phase virtual camera requires"):
        DirectVideoVerticalDecision(
            candidate_rank=1,
            strategy="tracked_crop",
            crop_mode="strict",
            coverage_mode="simultaneous",
            presentation_preference="phase_virtual_camera",
            framing_intent="Hold the one visible subject.",
            required_entity_indices=[1],
            preferred_entity_indices=[],
            sacrificable_entity_indices=[],
            attention_sequence=[],
        )


def _provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-3.6-flash",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="1.0",
        interaction_id="interaction-1",
        run_id="run-1",
        generated_at="2026-07-22T00:00:00+00:00",
    )


def _card() -> FullClipCard:
    return FullClipCard(
        source_asset_id=ASSET_ID,
        proxy_asset_id="sha256:" + "b" * 64,
        duration_ms=10_000,
        summary="A visible subject demonstrates an object beside a sign.",
        content_type="generic demonstration",
        entities=[
            Entity(
                entity_id="subject-1",
                kind=EntityKind.PERSON,
                label="visible subject",
                distinguishing_features="standing at frame center",
                evidence="visible throughout the event",
            ),
            Entity(
                entity_id="sign-1",
                kind=EntityKind.TEXT_REGION,
                label="foreground sign",
                distinguishing_features="wide line of visible text",
                evidence="visible along the lower edge",
            ),
        ],
        events=[
            FullClipEvent(
                event_id="demo",
                start_mmss="00:00",
                end_mmss="00:10",
                recommended_keyframe_mmss="00:02",
                label="demonstration",
                description="The subject demonstrates an object beside a sign.",
                observable_evidence="A subject and a foreground sign are visible.",
                evidence_modalities=EvidenceModality.VISUAL,
                entity_ids=["subject-1", "sign-1"],
                primary_entity_ids=["subject-1"],
                required_entity_ids=["subject-1", "sign-1"],
                optional_entity_ids=[],
                avoid_overlay_entity_ids=["sign-1"],
                keyframe_reason="Both required regions are clear.",
                boundary_precision=BoundaryPrecision.SECOND_LEVEL,
                confidence=0.8,
                action_completeness="complete",
                editing_uses=["demo"],
                quality_risks=[],
                framing_intent="Keep the subject and sign visible.",
                card_opportunities=[
                    CardOpportunity(
                        kind="object_callout",
                        rationale="The sign can be referenced without covering it.",
                        entity_ids=["sign-1"],
                    )
                ],
                dense_refinement="not_needed",
                dense_refinement_reasons=[],
                grounding_targets=[
                    FullClipGroundingTarget(
                        entity_id="subject-1",
                        target_kind=EntityKind.PERSON,
                        target_description="the visible subject",
                        purpose="reframe",
                    )
                ],
            )
        ],
        clip_uses=["demo"],
        portrait_reframe_feasibility="good",
        uncertainties=[],
        model_provenance=_provenance(),
    )


def _catalog() -> RushesCatalog:
    return RushesCatalog(
        catalog_id="catalog-1",
        source_directory="/source",
        sample_interval_ms=1_000,
        total_duration_ms=10_000,
        clips=[
            RushClip(
                clip_id="clip-1",
                path="/source/clip.mp4",
                sha256="a" * 64,
                duration_ms=10_000,
                width=1920,
                height=1080,
                frame_rate="30/1",
                size_bytes=1,
            )
        ],
        frames=[
            RushFrame(
                frame_id="RF000001",
                clip_id="clip-1",
                requested_time_ms=2_000,
                image_path="/frames/1.jpg",
            ),
            RushFrame(
                frame_id="RF000002",
                clip_id="clip-1",
                requested_time_ms=3_000,
                image_path="/frames/2.jpg",
            ),
        ],
        analysis_reel_path="/analysis.mp4",
        generated_at="2026-07-22T00:00:00+00:00",
    )


def _camera_supplement(card: FullClipCard) -> ClipObservationSupplement:
    event = card.events[0]
    return ClipObservationSupplement(
        supplement_id="camera-observation",
        source_asset_id=card.source_asset_id,
        proxy_asset_id=card.proxy_asset_id,
        base_card_sha256=clip_card_sha256(card),
        supplement_prompt_sha256="c" * 64,
        response_schema_sha256="d" * 64,
        event_observations=[
            EventObservationSupplement(
                event_id=event.event_id,
                event_fingerprint=event_fingerprint(event),
                observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
                capabilities=EventCapabilityManifest(
                    evidence_roles=AssessmentStatus.ASSESSED_PRESENT,
                    observable_beats=AssessmentStatus.ASSESSED_PRESENT,
                ),
                evidence_roles=EvidenceRoleMap(
                    primary_subject_ids=["subject-1"],
                    context_anchor_ids=["sign-1"],
                    observable_reason="The sign establishes context before the action.",
                ),
                observable_beats=[
                    ObservableBeat(
                        beat_id="sign-first",
                        kind="setup",
                        entity_ids=["sign-1"],
                        relation_mode="sequentially_reconstructable",
                        observable_predicate="The sign is directly visible first.",
                        transition_condition="The subject begins the visible action.",
                    ),
                    ObservableBeat(
                        beat_id="subject-second",
                        kind="interaction",
                        entity_ids=["subject-1"],
                        relation_mode="sequentially_reconstructable",
                        observable_predicate="The subject performs the visible action.",
                        transition_condition="The visible action reaches a result.",
                    ),
                ],
            )
        ],
        model_provenance=card.model_provenance,
    )


def _brief() -> FeatureEditBrief:
    return FeatureEditBrief(
        project_id="project-1",
        title="Generic feature cut",
        target_duration_seconds=60,
        chapters=[
            FeatureChapterBrief(
                feature_id="feature-1",
                title="Visible demonstration",
                detail_lines=["Show the directly observable demonstration."],
                target_duration_seconds=6,
                vertical_primary_target_description="the visible subject",
            )
        ],
    )


def _direct_chapter_for_aspects(
    *,
    horizontal: DirectVideoHorizontalDecision | None,
    vertical: DirectVideoVerticalDecision | None,
    horizontal_alternates: list[DirectVideoHorizontalDecision] | None = None,
    vertical_alternates: list[DirectVideoVerticalDecision] | None = None,
) -> DirectVideoChapterDecision:
    return DirectVideoChapterDecision(
        chapter_index=1,
        evidence_status="partial",
        observed_visual_evidence="The visible subject completes a demonstration.",
        selection_reason="The bounded evidence shows the requested action.",
        quality_risks=[],
        horizontal=horizontal,
        horizontal_alternates=horizontal_alternates or [],
        vertical=vertical,
        vertical_alternates=vertical_alternates or [],
        recommended_duration_seconds=4,
        duration_rationale="A short hold preserves the observable action.",
        attention_observation=AttentionObservation(
            semantic_novelty=0.6,
            action_progress=0.8,
            visual_motion=0.2,
            composition_change=0.1,
            reading_load=0.2,
            unresolved_tension=0.1,
            emotional_hold_value=0.2,
            repetition_pressure=0.1,
            music_transition_opportunity=0.3,
            minimum_dwell_seconds=3,
            maximum_dwell_seconds=6,
            rationale="The action reads within a short bounded hold.",
            uncertainties=[],
            requires_human_review=True,
        ),
        flow_intent=ShotFlowIntent(
            narrative_role="proof",
            energy_role="low_hold",
            relation_to_previous="new_context",
            boundary_alignment="phrase_preferred",
        ),
        confidence=0.8,
    )


def _region(
    *,
    region_id: str,
    entity_id: str,
    event_relation: str,
    constraint_role: str = "hard_core",
) -> ResolvedFramingRegion:
    return ResolvedFramingRegion.model_validate(
        {
            "region_id": region_id,
            "target_description": f"visible region for {entity_id}",
            "kind": "text_region" if entity_id == "sign-1" else "subject",
            "constraint_role": constraint_role,
            "composition": "atomic",
            "entity_refs": [
                {"entity_id": entity_id, "event_relation": event_relation}
            ],
            "observable_relation": "The region is directly visible in the event.",
        }
    )


def _candidate(
    candidate_id: str, frame_id: str, *, vertical_strategy: str
) -> ClipCardFeatureCandidate:
    resolved_regions = (
        [
            _region(
                region_id=f"{candidate_id}-subject",
                entity_id="subject-1",
                event_relation="required",
            ),
            _region(
                region_id=f"{candidate_id}-sign",
                entity_id="sign-1",
                event_relation="avoid_overlay",
                constraint_role="overlay_keepout",
            ),
        ]
        if vertical_strategy == "tracked_crop"
        else []
    )
    return ClipCardFeatureCandidate(
        candidate_id=candidate_id,
        source_asset_id=ASSET_ID,
        event_id="demo",
        frame_id=frame_id,
        observed_visual_evidence="The subject and sign are both visible.",
        selection_reason="Complete visible action with auditable regions.",
        quality_risks=[],
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_strategy=vertical_strategy,
        vertical_target_description=(
            "the directly visible subject"
            if vertical_strategy == "tracked_crop"
            else None
        ),
        resolved_regions=resolved_regions,
        confidence=0.8,
    )


def _v2_plan() -> ClipCardFeaturePlanV2:
    first = _candidate(
        "candidate-a", "RF000001", vertical_strategy="tracked_crop"
    )
    second = _candidate(
        "candidate-b", "RF000002", vertical_strategy="fit_with_background"
    )
    return ClipCardFeaturePlanV2(
        contract_version="clip-card-feature-cut-v2",
        project_id="project-1",
        catalog_id="catalog-1",
        title="Generic feature cut",
        strategy_summary="Preserve alternatives and evidence-bound regions.",
        chapters=[
            ClipCardFeatureSelectV2(
                feature_id="feature-1",
                evidence_status="supported",
                horizontal_source_asset_id=ASSET_ID,
                horizontal_event_id="demo",
                horizontal_frame_id="RF000001",
                vertical_source_asset_id=ASSET_ID,
                vertical_event_id="demo",
                vertical_frame_id="RF000002",
                observed_visual_evidence="The subject and sign are visible.",
                selection_reason="Each aspect uses a validated candidate.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                # A brief primary target no longer forces tracked_crop.
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=[],
                confidence=0.8,
                candidates=[first, second],
                horizontal_candidate_id="candidate-a",
                vertical_candidate_id="candidate-b",
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )


def _v3_plan() -> ClipCardFeaturePlanV3:
    first = ClipCardFeatureCandidateV3(
        candidate_id="candidate-a",
        source_asset_id=ASSET_ID,
        event_id="demo",
        frame_id="RF000001",
        observed_visual_evidence="The subject and sign are both visible.",
        selection_reason="Complete action with enough room for a portrait crop.",
        quality_risks=[],
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_focus_entity_id=None,
        vertical_strategy="tracked_crop",
        vertical_crop_mode="strict",
        framing_intent="Prioritize the subject; retain the sign as useful context.",
        required_entity_ids=["subject-1"],
        preferred_entity_ids=["sign-1"],
        sacrificable_entity_ids=[],
        confidence=0.8,
    )
    second = ClipCardFeatureCandidateV3(
        candidate_id="candidate-b",
        source_asset_id=ASSET_ID,
        event_id="demo",
        frame_id="RF000002",
        observed_visual_evidence="The subject is clear and the sign is peripheral.",
        selection_reason="Stable wide composition can be fit without tracking.",
        quality_risks=["The sign is near the edge."],
        horizontal_strategy="tracked_reframe",
        horizontal_zoom_intent="subtle",
        horizontal_focus_entity_id="subject-1",
        vertical_strategy="fit_with_background",
        vertical_crop_mode="primary_center",
        framing_intent="Preserve the whole source; the sign may be sacrificed.",
        required_entity_ids=["subject-1"],
        preferred_entity_ids=[],
        sacrificable_entity_ids=["sign-1"],
        confidence=0.7,
    )
    return ClipCardFeaturePlanV3(
        contract_version="clip-card-feature-cut-v3",
        project_id="project-1",
        catalog_id="catalog-1",
        title="Generic feature cut",
        strategy_summary="Select alternatives; derive geometry from local evidence.",
        chapters=[
            ClipCardFeatureSelectV3(
                feature_id="feature-1",
                evidence_status="supported",
                candidates=[first, second],
                horizontal_candidate_id="candidate-a",
                vertical_candidate_id="candidate-b",
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )


def test_feature_output_canonicalization_is_narrow_ordered_and_auditable() -> None:
    payload = _v3_plan().model_dump(mode="json")
    first, second = payload["chapters"][0]["candidates"]
    first.update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "detail",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    second.update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "none",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    original = json.dumps(payload)

    canonical_text, changes = canonicalize_feature_plan_output(original)
    canonical = json.loads(canonical_text)

    assert original == json.dumps(payload)
    assert canonical["chapters"][0]["candidates"][0]["horizontal_strategy"] == "original"
    assert canonical["chapters"][0]["candidates"][0]["horizontal_zoom_intent"] == "none"
    assert canonical["chapters"][0]["candidates"][0][
        "horizontal_focus_entity_id"
    ] is None
    assert canonical["chapters"][0]["candidates"][1][
        "horizontal_focus_entity_id"
    ] is None
    assert [change["rule"] for change in changes] == [
        "explicit_original_strategy_disables_zoom",
        "explicit_original_strategy_has_no_focus_entity",
        "explicit_original_strategy_has_no_focus_entity",
    ]
    ClipCardFeaturePlanV3.model_validate_json(canonical_text)


def test_feature_output_canonicalization_zero_pads_short_rf_identifier() -> None:
    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0]["frame_id"] = "RF00030"

    canonical_text, changes = canonicalize_feature_plan_output(json.dumps(payload))
    canonical = json.loads(canonical_text)

    assert canonical["chapters"][0]["candidates"][0]["frame_id"] == "RF000030"
    assert changes[0]["rule"] == "fixed_width_rf_identifier_zero_padding"
    ClipCardFeaturePlanV3.model_validate_json(canonical_text)


def test_feature_reuse_rejects_mismatched_raw_response_copies() -> None:
    with pytest.raises(ValueError, match="does not exactly match"):
        _verified_feature_raw_output_text(
            raw_output={"output_text": "first"},
            raw_interaction={"output_text": "second"},
        )


def test_fresh_feature_run_refuses_existing_paid_namespace(tmp_path: Path) -> None:
    write_json(
        tmp_path / "clip-card-feature-plan.attempt-01.raw_output.json",
        {"output_text": "already paid"},
    )
    with pytest.raises(FileExistsError, match="new output directory"):
        _assert_fresh_feature_namespace_empty(tmp_path)


def test_feature_raw_reuse_resolves_complete_set_and_preserves_paid_artifact(
    tmp_path: Path,
) -> None:
    stem = "clip-card-feature-plan.attempt-01"
    paths = {
        "request": tmp_path / f"{stem}.request.json",
        "raw_output": tmp_path / f"{stem}.raw_output.json",
        "raw_interaction": tmp_path / f"{stem}.raw_interaction.json",
    }
    write_json(paths["request"], {"model": MODEL_ID})
    raw_payload = _v3_plan().model_dump(mode="json")
    raw_payload["chapters"][0]["candidates"][0].update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "subtle",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    write_json(paths["raw_output"], {"output_text": json.dumps(raw_payload)})
    write_json(paths["raw_interaction"], {"model": MODEL_ID, "id": "paid-1"})
    original_bytes = paths["raw_output"].read_bytes()

    resolved = _resolve_feature_reuse_artifacts(tmp_path)
    canonical_text, canonical_path, audit_path = _write_feature_normalization_artifacts(
        output_dir=tmp_path,
        artifact_stem="clip-card-feature-plan",
        raw_output_path=resolved["raw_output"],
        raw_output_text=json.loads(original_bytes)["output_text"],
    )

    assert resolved["kind"] == "attempt-01"
    assert paths["raw_output"].read_bytes() == original_bytes
    assert canonical_path.exists() and audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["change_count"] == 2
    assert audit["raw_output_artifact_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    ClipCardFeaturePlanV3.model_validate_json(canonical_text)


def test_feature_raw_reuse_prefers_latest_complete_repaired_attempt(
    tmp_path: Path,
) -> None:
    for attempt in (1, 2):
        stem = f"clip-card-feature-plan.attempt-{attempt:02d}"
        write_json(tmp_path / f"{stem}.request.json", {"model": MODEL_ID})
        write_json(
            tmp_path / f"{stem}.raw_output.json",
            {"output_text": f"response-{attempt}"},
        )
        write_json(
            tmp_path / f"{stem}.raw_interaction.json",
            {"model": MODEL_ID, "id": f"paid-{attempt}"},
        )

    resolved = _resolve_feature_reuse_artifacts(tmp_path)

    assert resolved["kind"] == "attempt-02"
    assert resolved["raw_output"].name.endswith("attempt-02.raw_output.json")


def test_feature_reuse_binding_keeps_original_paid_request_as_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    library = tmp_path / "library"
    output_dir = tmp_path / "plan"
    card_path = (
        library / "clips" / ("a" * 16) / "gemini" / "clip-card" / "clip_card.json"
    )
    write_json(catalog_path, _catalog())
    write_json(brief_path, _brief())
    write_json(card_path, _card())
    output_dir.mkdir()

    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0].update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "subtle",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    output_text = json.dumps(payload)
    stem = output_dir / "clip-card-feature-plan.attempt-01"
    paid_request_path = Path(f"{stem}.request.json")
    paid_raw_output_path = Path(f"{stem}.raw_output.json")
    paid_raw_interaction_path = Path(f"{stem}.raw_interaction.json")
    write_json(
        paid_request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only supplied evidence.",
            "input": [{"type": "text", "text": "Paid request."}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(ClipCardFeaturePlanV3),
            },
        },
    )
    write_json(paid_raw_output_path, {"output_text": output_text})
    write_json(
        paid_raw_interaction_path,
        {
            "model": MODEL_ID,
            "id": "interaction-1",
            "output_text": output_text,
        },
    )
    original_request_hash = hashlib.sha256(paid_request_path.read_bytes()).hexdigest()
    original_raw_bytes = paid_raw_output_path.read_bytes()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_clip_card_feature_cut.py",
            str(catalog_path),
            str(brief_path),
            str(library),
            str(output_dir),
            "--reuse-raw-output",
        ],
    )

    assert feature_planner_main() == 0

    pointer = json.loads(
        (output_dir / "feature-plan.external-projection.json").read_text(
            encoding="utf-8"
        )
    )
    record = json.loads(
        (output_dir / pointer["record_path"]).read_text(encoding="utf-8")
    )
    assert record["source_request_sha256"] == original_request_hash
    assert paid_raw_output_path.read_bytes() == original_raw_bytes

    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    monkeypatch.setattr(
        "scripts.plan_clip_card_feature_cut.genai.Client",
        lambda **_: pytest.fail("fresh rerun must fail before constructing an API client"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_clip_card_feature_cut.py",
            str(catalog_path),
            str(brief_path),
            str(library),
            str(output_dir),
        ],
    )
    with pytest.raises(FileExistsError, match="new output directory"):
        feature_planner_main()
def test_compact_card_preserves_entities_roles_and_relations() -> None:
    compact = compact_card(_card())

    assert [entity["entity_id"] for entity in compact["entities"]] == [
        "subject-1",
        "sign-1",
    ]
    event = compact["events"][0]
    assert event["primary_entity_ids"] == ["subject-1"]
    assert event["required_entity_ids"] == ["subject-1", "sign-1"]
    assert event["optional_entity_ids"] == []
    assert event["avoid_overlay_entity_ids"] == ["sign-1"]
    relations = {
        item["entity_id"]: item["relations"] for item in event["entity_relations"]
    }
    assert relations["subject-1"] == [
        "event_member",
        "primary",
        "required",
        "grounding_target",
    ]
    assert relations["sign-1"] == ["event_member", "required", "avoid_overlay"]


def test_v1_source_schema_remains_the_exact_single_selection_shape() -> None:
    legacy_schema = gemini_response_schema(ClipCardFeaturePlan)
    current_schema = gemini_response_schema(ClipCardFeaturePlanV2)
    legacy_schema_hash = hashlib.sha256(
        json.dumps(
            legacy_schema,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    # Canonical schema from commit 7a3d686. Existing v1 raw request sidecars
    # are checked by exact schema equality, so this fingerprint is contractual.
    assert legacy_schema_hash == (
        "112584f2b89bee80869d81b63597bf68fe1552edf9c11e7ef4ecb6197145797c"
    )
    assert "contract_version" not in legacy_schema["properties"]
    assert "contract_version" in current_schema["properties"]
    legacy_chapter = legacy_schema["$defs"]["ClipCardFeatureSelect"]["properties"]
    current_chapter = current_schema["$defs"]["ClipCardFeatureSelectV2"][
        "properties"
    ]
    assert "candidates" not in legacy_chapter
    assert "horizontal_candidate_id" not in legacy_chapter
    assert "vertical_candidate_id" not in legacy_chapter
    assert "candidates" in current_chapter


def test_v2_preserves_top_k_and_projects_legacy_feature_plan() -> None:
    plan = _v2_plan()

    validate_plan_contract(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: _card()},
    )
    projected = project_feature_contracts(plan, brief=_brief(), catalog=_catalog())

    assert len(plan.chapters[0].candidates) == 2
    assert projected.chapters[0].horizontal_frame_id == "RF000001"
    assert projected.chapters[0].vertical_frame_id == "RF000002"
    assert projected.chapters[0].vertical_strategy == "fit_with_background"
    assert [candidate.candidate_id for candidate in projected.chapters[0].horizontal_candidates] == [
        "candidate-a",
        "candidate-b",
    ]
    assert [candidate.candidate_id for candidate in projected.chapters[0].vertical_candidates] == [
        "candidate-b",
        "candidate-a",
    ]
    assert projected.chapters[0].horizontal_candidates[0].rank == 1
    assert projected.chapters[0].vertical_candidates[0].rank == 1
    assert projected.chapters[0].vertical_candidates[1].regions[0].entity_id == "subject-1"
    assert projected.chapters[0].vertical_candidates[1].regions[0].role == "required"


def test_legacy_single_selection_json_remains_readable() -> None:
    payload = _v2_plan().model_dump(mode="json")
    payload.pop("contract_version")
    for chapter in payload["chapters"]:
        chapter.pop("candidates")
        chapter.pop("horizontal_candidate_id")
        chapter.pop("vertical_candidate_id")

    legacy = ClipCardFeaturePlan.model_validate(payload)
    projected = project_feature_contracts(legacy, brief=_brief(), catalog=_catalog())

    assert not hasattr(legacy, "contract_version")
    assert not hasattr(legacy.chapters[0], "candidates")
    assert projected.chapters[0].vertical_frame_id == "RF000002"
    assert projected.chapters[0].horizontal_candidates == []
    assert projected.chapters[0].vertical_candidates == []


def test_projection_entrypoints_keep_v1_candidate_free_and_v2_top_k() -> None:
    v2 = _v2_plan()
    payload = v2.model_dump(mode="json")
    payload.pop("contract_version")
    for chapter in payload["chapters"]:
        chapter.pop("candidates")
        chapter.pop("horizontal_candidate_id")
        chapter.pop("vertical_candidate_id")
    legacy = ClipCardFeaturePlan.model_validate(payload)

    _, legacy_projection = reproject_external_feature_plan(
        source_plan=legacy,
        catalog=_catalog(),
        brief=_brief(),
        source_artifacts={},
    )
    _, v2_projection = reproject_external_feature_plan_v2(
        source_plan=v2,
        catalog=_catalog(),
        brief=_brief(),
        source_artifacts={},
    )

    assert legacy_projection.chapters[0].horizontal_candidates == []
    assert legacy_projection.chapters[0].vertical_candidates == []
    assert [
        candidate.candidate_id
        for candidate in v2_projection.chapters[0].horizontal_candidates
    ] == ["candidate-a", "candidate-b"]
    assert [
        candidate.candidate_id
        for candidate in v2_projection.chapters[0].vertical_candidates
    ] == ["candidate-b", "candidate-a"]


def test_projection_entrypoints_reject_cross_version_source_plans() -> None:
    v2 = _v2_plan()
    payload = v2.model_dump(mode="json")
    payload.pop("contract_version")
    for chapter in payload["chapters"]:
        chapter.pop("candidates")
        chapter.pop("horizontal_candidate_id")
        chapter.pop("vertical_candidate_id")
    legacy = ClipCardFeaturePlan.model_validate(payload)

    with pytest.raises(ValueError, match="v1 requires its exact legacy source schema"):
        reproject_external_feature_plan(
            source_plan=v2,
            catalog=_catalog(),
            brief=_brief(),
            source_artifacts={},
        )
    with pytest.raises(ValueError, match="v2 requires a clip-card-feature-cut-v2"):
        reproject_external_feature_plan_v2(
            source_plan=legacy,
            catalog=_catalog(),
            brief=_brief(),
            source_artifacts={},
        )


@pytest.mark.parametrize(
    ("projection_contract_id", "source_model"),
    [
        ("clip-card-feature-cut-v1", ClipCardFeaturePlan),
        ("clip-card-feature-cut-v2", ClipCardFeaturePlanV2),
    ],
)
def test_projection_sidecar_validates_both_exact_source_schemas(
    tmp_path: Path,
    projection_contract_id: str,
    source_model: type[ClipCardFeaturePlan] | type[ClipCardFeaturePlanV2],
) -> None:
    v2 = _v2_plan()
    if source_model is ClipCardFeaturePlan:
        payload = v2.model_dump(mode="json")
        payload.pop("contract_version")
        for chapter in payload["chapters"]:
            chapter.pop("candidates")
            chapter.pop("horizontal_candidate_id")
            chapter.pop("vertical_candidate_id")
        source_plan = ClipCardFeaturePlan.model_validate(payload)
    else:
        source_plan = v2
    feature_plan = project_feature_contracts(
        source_plan,
        brief=_brief(),
        catalog=_catalog(),
    )
    plan_dir = tmp_path / "gemini-plan"
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    feature_plan_path = plan_dir / "feature_edit_plan.json"
    source_plan_path = plan_dir / "source-plan.json"
    source_request_path = plan_dir / "source.request.json"
    raw_output_path = plan_dir / "source.raw_output.json"
    raw_interaction_path = plan_dir / "source.raw_interaction.json"
    write_json(catalog_path, _catalog())
    write_json(brief_path, _brief())
    write_json(feature_plan_path, feature_plan)
    write_json(source_plan_path, source_plan)
    write_json(
        source_request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only the supplied evidence.",
            "input": [{"type": "text", "text": "Select auditable evidence."}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(source_model),
            }
        },
    )
    write_json(raw_output_path, {"output_text": source_plan.model_dump_json()})
    write_json(raw_interaction_path, {"id": "interaction-1"})

    pointer = write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id=projection_contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=source_request_path,
        source_artifacts={
            "source_raw_output": raw_output_path,
            "source_raw_interaction": raw_interaction_path,
        },
    )

    assert pointer.name == "feature-plan.external-projection.json"


def test_external_top_k_projection_binds_actual_music_to_paid_request(
    tmp_path: Path,
) -> None:
    source_plan = _v2_plan()
    feature_plan = project_feature_contracts(
        source_plan,
        brief=_brief(),
        catalog=_catalog(),
    )
    plan_dir = tmp_path / "gemini-plan"
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    feature_plan_path = plan_dir / "feature_edit_plan.json"
    source_plan_path = plan_dir / "source-plan.json"
    source_request_path = plan_dir / "source.request.json"
    raw_output_path = plan_dir / "source.raw_output.json"
    raw_interaction_path = plan_dir / "source.raw_interaction.json"
    music_path = tmp_path / "music.wav"
    music_path.write_bytes(b"audible music fixture")
    music_sha256 = sha256_file(music_path)
    write_json(catalog_path, _catalog())
    write_json(brief_path, _brief())
    write_json(feature_plan_path, feature_plan)
    write_json(source_plan_path, source_plan)
    write_json(
        source_request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only the supplied evidence.",
            "input": [
                {
                    "type": "text",
                    "text": f"Select evidence. music_sha256={music_sha256}",
                },
                {
                    "type": "audio",
                    "uri": "https://example.invalid/files/music",
                    "mime_type": "audio/wav",
                },
            ],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(ClipCardFeaturePlanV2),
            },
        },
    )
    write_json(raw_output_path, {"output_text": source_plan.model_dump_json()})
    write_json(raw_interaction_path, {"id": "interaction-music"})
    write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id="clip-card-feature-cut-v2",
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=source_request_path,
        source_artifacts={
            "source_raw_output": raw_output_path,
            "source_raw_interaction": raw_interaction_path,
            "source_music": music_path,
        },
    )

    binding = _current_external_projection_binding(
        plan_dir=plan_dir,
        catalog_path=catalog_path,
        catalog_reel_sha256="b" * 64,
        brief_path=brief_path,
        plan_path=feature_plan_path,
        music_sha256=music_sha256,
        created_at="2026-07-26T00:00:00+00:00",
    )

    assert binding["music_sha256"] == music_sha256
    with pytest.raises(ValueError, match="music differs"):
        _current_external_projection_binding(
            plan_dir=plan_dir,
            catalog_path=catalog_path,
            catalog_reel_sha256="b" * 64,
            brief_path=brief_path,
            plan_path=feature_plan_path,
            music_sha256="f" * 64,
            created_at="2026-07-26T00:00:00+00:00",
        )


def test_v2_rejects_single_candidate() -> None:
    payload = _v2_plan().model_dump(mode="json")
    chapter = payload["chapters"][0]
    chapter["candidates"] = chapter["candidates"][:1]
    chapter["vertical_candidate_id"] = "candidate-a"
    chapter["vertical_frame_id"] = "RF000001"
    chapter["vertical_strategy"] = "tracked_crop"
    chapter["vertical_target_description"] = "the directly visible subject"

    with pytest.raises(ValidationError, match="Top-K 2-4"):
        ClipCardFeaturePlanV2.model_validate(payload)


def test_v3_allows_a_unique_supported_candidate() -> None:
    payload = _v3_plan().model_dump(mode="json")
    chapter = payload["chapters"][0]
    chapter["candidates"] = chapter["candidates"][:1]
    chapter["horizontal_candidate_id"] = "candidate-a"
    chapter["vertical_candidate_id"] = "candidate-a"

    plan = ClipCardFeaturePlanV3.model_validate(payload)

    assert len(plan.chapters[0].candidates) == 1


def test_selected_tracked_crop_can_use_resolved_hard_core_without_fuzzy_target() -> None:
    payload = _v2_plan().model_dump(mode="json")
    chapter = payload["chapters"][0]
    first = chapter["candidates"][0]
    first["vertical_target_description"] = None
    chapter["vertical_candidate_id"] = "candidate-a"
    chapter["vertical_frame_id"] = "RF000001"
    chapter["vertical_strategy"] = "tracked_crop"
    chapter["vertical_target_description"] = None
    plan = ClipCardFeaturePlanV2.model_validate(payload)

    validate_plan_contract(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: _card()},
    )
    projected = project_feature_contracts(plan, brief=_brief(), catalog=_catalog())

    rank_one = projected.chapters[0].vertical_candidates[0]
    assert rank_one.target_description is None
    assert rank_one.regions[0].role == "required"
    assert rank_one.regions[0].entity_id == "subject-1"


@pytest.mark.parametrize(
    ("entity_id", "event_relation", "error"),
    [
        ("missing-entity", "event_member", "unknown entity"),
        ("subject-1", "optional", "relation is not backed"),
    ],
)
def test_local_validation_rejects_unverifiable_region_lineage(
    entity_id: str, event_relation: str, error: str
) -> None:
    payload = _v2_plan().model_dump(mode="json")
    ref = payload["chapters"][0]["candidates"][0]["resolved_regions"][0][
        "entity_refs"
    ][0]
    ref["entity_id"] = entity_id
    ref["event_relation"] = event_relation
    plan = ClipCardFeaturePlanV2.model_validate(payload)

    with pytest.raises(ValueError, match=error):
        validate_plan_contract(
            plan,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
        )


def test_resolved_atomic_region_requires_one_entity() -> None:
    with pytest.raises(ValidationError, match="exactly one entity"):
        ResolvedFramingRegion(
            region_id="invalid",
            target_description="two entities incorrectly presented as atomic",
            kind="subject",
            constraint_role="hard_core",
            composition="atomic",
            entity_refs=[
                ResolvedEntityRef(entity_id="subject-1", event_relation="required"),
                ResolvedEntityRef(entity_id="sign-1", event_relation="required"),
            ],
            observable_relation="Both are visible.",
        )


def test_v3_model_schema_contains_choices_not_projection_mirrors() -> None:
    schema = gemini_response_schema(ClipCardFeaturePlanV3)
    chapter = schema["$defs"]["ClipCardFeatureSelectV3"]["properties"]
    candidate = schema["$defs"]["ClipCardFeatureCandidateV3"]["properties"]

    assert "candidates" in chapter
    assert "horizontal_frame_id" not in chapter
    assert "vertical_frame_id" not in chapter
    assert "observed_visual_evidence" not in chapter
    assert "resolved_regions" not in candidate
    assert "horizontal_target_description" not in candidate
    assert "vertical_target_description" not in candidate
    assert "required_entity_ids" in candidate
    assert "preferred_entity_ids" in candidate
    assert "sacrificable_entity_ids" in candidate
    assert "framing_intent" in candidate


def test_v3_compact_card_omits_redundant_relation_expansion() -> None:
    compact = compact_card_v3(_card())
    event = compact["events"][0]

    assert "evidence" not in compact["entities"][0]
    assert "entity_relations" not in event
    assert "card_opportunities" not in event
    assert event["primary_entity_ids"] == ["subject-1"]
    assert event["required_entity_ids"] == ["subject-1", "sign-1"]
    assert event["grounding_target_entity_ids"] == ["subject-1"]


def test_v3_compact_card_preserves_observation_but_drops_camera_advice() -> None:
    card = _card()
    card.events[0].portrait_attention_sequence = [
        FullClipAttentionPhase(
            phase_id="context-first",
            anchor_entity_ids=["sign-1"],
            relation_mode="single_focus",
            suggested_camera_behavior="hold",
            observable_predicate="The sign is directly visible before the action.",
            transition_condition="The subject begins the directly visible action.",
        ),
        FullClipAttentionPhase(
            phase_id="action-second",
            anchor_entity_ids=["subject-1"],
            relation_mode="single_focus",
            suggested_camera_behavior="follow",
            observable_predicate="The subject performs the visible action.",
            transition_condition="The visible result is complete.",
        ),
    ]

    event = compact_card_v3(card)["events"][0]

    assert [
        beat["entity_ids"]
        for beat in event["observable_beats"]
    ] == [["sign-1"], ["subject-1"]]
    assert event["observation_capabilities"]["observable_beats"] == "assessed_present"
    assert all("suggested_camera_behavior" not in beat for beat in event["observable_beats"])


def test_v3_projects_local_descriptions_and_regions_from_selected_evidence() -> None:
    plan = _v3_plan()
    cards = {ASSET_ID: _card()}
    validate_plan_contract_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards=cards,
    )
    evidence = build_selected_clip_card_evidence(plan, cards=cards)
    projected = project_feature_contracts_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=evidence,
    )

    chapter = projected.chapters[0]
    assert chapter.horizontal_frame_id == "RF000001"
    assert chapter.vertical_frame_id == "RF000002"
    assert chapter.vertical_strategy == "fit_with_background"
    assert [item.candidate_id for item in chapter.horizontal_candidates] == [
        "candidate-a",
        "candidate-b",
    ]
    assert [item.candidate_id for item in chapter.vertical_candidates] == [
        "candidate-b",
        "candidate-a",
    ]
    assert chapter.horizontal_candidates[1].target_description == "the visible subject"
    tracked = chapter.vertical_candidates[1]
    assert tracked.target_description == "the visible subject"
    assert [(region.entity_id, region.role) for region in tracked.regions] == [
        ("subject-1", "required"),
        ("sign-1", "preferred"),
    ]
    assert tracked.regions[0].target_description == "the visible subject"
    assert tracked.regions[1].target_description.startswith("foreground sign;")
    assert any(
        relation.startswith("editorial_framing_intent=")
        for relation in tracked.regions[0].observable_relations
    )


def test_v3_projects_entity_order_into_virtual_camera_region_order() -> None:
    payload = _v3_plan().model_dump(mode="python")
    chapter = payload["chapters"][0]
    chapter["vertical_candidate_id"] = "candidate-a"
    chapter["candidates"][0]["virtual_camera_proposal"] = (
        ClipCardVirtualCameraProposalV1(
            composition_mode="sequential_focus",
            phases=[
                ClipCardVirtualCameraPhaseV1(
                    phase_id="sign-first",
                    start_progress=0.0,
                    end_progress=0.45,
                    anchor_entity_ids=["sign-1"],
                    camera_behavior="hold",
                    observable_predicate="The sign is directly visible first.",
                    transition_condition="The subject begins the visible action.",
                    editorial_reason="Establish context before following the action.",
                ),
                ClipCardVirtualCameraPhaseV1(
                    phase_id="subject-second",
                    start_progress=0.45,
                    end_progress=1.0,
                    anchor_entity_ids=["subject-1"],
                    camera_behavior="follow",
                    transition_in="smoothstep",
                    transition_duration_fraction=0.25,
                    observable_predicate="The subject performs the visible action.",
                    transition_condition="Hold until the visible result is complete.",
                    editorial_reason="Follow the evidence order, not screen direction.",
                ),
            ],
            proposal_reason="Visible information hands off from context to action.",
        ).model_dump(mode="python")
    )
    plan = ClipCardFeaturePlanV3.model_validate(payload)
    cards = {ASSET_ID: _card()}
    validate_plan_contract_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards=cards,
        supplements={ASSET_ID: [_camera_supplement(cards[ASSET_ID])]},
    )
    evidence = build_selected_clip_card_evidence(plan, cards=cards)
    projected = project_feature_contracts_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=evidence,
    )

    proposal = projected.chapters[0].vertical_candidates[0].virtual_camera_proposal
    assert proposal is not None
    assert [phase.anchor_region_ids for phase in proposal.phases] == [
        ["candidate-a.preferred.sign-1"],
        ["candidate-a.required.subject-1"],
    ]
    assert proposal.phases[0].observable_predicate.startswith("The sign")


def test_v3_rejects_sequential_camera_without_observation_supplement() -> None:
    payload = _v3_plan().model_dump(mode="python")
    chapter = payload["chapters"][0]
    chapter["vertical_candidate_id"] = "candidate-a"
    chapter["candidates"][0]["virtual_camera_proposal"] = (
        ClipCardVirtualCameraProposalV1(
            composition_mode="sequential_focus",
            phases=[
                ClipCardVirtualCameraPhaseV1(
                    phase_id="sign-first",
                    start_progress=0.0,
                    end_progress=0.5,
                    anchor_entity_ids=["sign-1"],
                    camera_behavior="hold",
                    observable_predicate="The sign is directly visible.",
                    transition_condition="The subject begins the action.",
                    editorial_reason="Establish context.",
                ),
                ClipCardVirtualCameraPhaseV1(
                    phase_id="subject-second",
                    start_progress=0.5,
                    end_progress=1.0,
                    anchor_entity_ids=["subject-1"],
                    camera_behavior="follow",
                    transition_in="smoothstep",
                    transition_duration_fraction=0.25,
                    observable_predicate="The subject performs the action.",
                    transition_condition="The action reaches its result.",
                    editorial_reason="Follow the visible action.",
                ),
            ],
            proposal_reason="Present context before action.",
        ).model_dump(mode="python")
    )
    plan = ClipCardFeaturePlanV3.model_validate(payload)
    with pytest.raises(ValueError, match="lacks assessed observation evidence"):
        validate_plan_contract_v3(
            plan,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
        )


def test_v3_projection_is_reproducible_from_hash_bound_evidence(
    tmp_path: Path,
) -> None:
    plan = _v3_plan()
    evidence = build_selected_clip_card_evidence(plan, cards={ASSET_ID: _card()})
    evidence_path = tmp_path / "selected-evidence.json"
    write_json(evidence_path, evidence)
    expected = project_feature_contracts_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=evidence,
    )

    _, reproduced = reproject_external_feature_plan_v3(
        source_plan=plan,
        catalog=_catalog(),
        brief=_brief(),
        source_artifacts={"selected_clip_card_evidence": evidence_path},
    )

    assert reproduced.model_dump(mode="json") == expected.model_dump(mode="json")
    assert SelectedClipCardEvidence.model_validate_json(
        evidence.model_dump_json()
    ) == evidence


def test_v3_rejects_unbacked_but_allows_brief_specific_entity_subset() -> None:
    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0]["required_entity_ids"] = [
        "missing-entity"
    ]
    plan = ClipCardFeaturePlanV3.model_validate(payload)
    with pytest.raises(ValueError, match="not backed by its event"):
        validate_plan_contract_v3(
            plan,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
        )

    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0]["preferred_entity_ids"] = []
    plan = ClipCardFeaturePlanV3.model_validate(payload)
    validate_plan_contract_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: _card()},
    )
