from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.clip_card_retrieval import (
    FeatureChapterShortlist,
    FeatureShortlistCandidate,
    FeatureShortlistPlan,
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
    EvidenceModality,
    FeatureChapterBrief,
    FeatureEditBrief,
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
from jascue_video_lab.feature_cut import (
    _current_external_projection_binding,
    write_external_feature_plan_projection,
)
from jascue_video_lab.gemini import MODEL_ID
from jascue_video_lab.media import sha256_file
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import write_json
from scripts.plan_clip_card_feature_cut import (
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
    build_selected_clip_card_evidence,
    canonicalize_direct_video_edit_plan_output,
    canonicalize_feature_plan_output,
    compact_card,
    compact_card_v3,
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
    project_direct_video_edit_plan,
    validate_candidate_video_budget,
)


ASSET_ID = "sha256:" + "a" * 64


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


def test_direct_video_response_uses_integer_ranks_and_projects_ids_locally() -> None:
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
                vertical=DirectVideoVerticalDecision(
                    candidate_rank=1,
                    strategy="tracked_crop",
                    crop_mode="strict",
                    coverage_mode="sequential",
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
        cards={ASSET_ID: _card()},
        provenance=_provenance(),
    )

    candidate = projected.chapters[0].candidates[0]
    assert candidate.candidate_id == "rank-01"
    assert candidate.source_asset_id == ASSET_ID
    assert candidate.event_id == "demo"
    assert candidate.frame_id == "RF000001"
    assert projected.chapters[0].horizontal_candidate_id == "rank-01"
    assert projected.chapters[0].vertical_candidate_id == "rank-01"
    assert (
        projected.chapters[0].attention_observation is not None
    )
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
    assert '"candidate_id"' not in direct_schema
    assert '"source_asset_id"' not in direct_schema
    assert '"frame_id"' not in direct_schema
    assert '"project_id"' not in direct_schema
    assert '"model_provenance"' not in direct_schema


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


def test_direct_video_canonicalization_fails_safe_for_missing_duration_and_attention() -> None:
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
    assert chapter["vertical"]["coverage_mode"] == "simultaneous"
    assert chapter["vertical"]["traversal_policy"] == "no_continuous_traversal"
    assert chapter["vertical"]["attention_sequence"] == [
        {
            "start_progress": 0.0,
            "end_progress": 1.0,
            "anchor_entity_indices": [1, 2, 3],
            "camera_behavior": "hold",
            "movement_motivation": "none",
            "cut_admissible": False,
            "transition_preference": "auto",
        }
    ]
    assert {
        change["rule"] for change in changes
    } >= {
        "missing_recommended_duration_uses_model_attention_midpoint",
        "incomplete_required_attention_fails_safe_to_joint_static_hold",
        "joint_static_hold_disables_synthetic_traversal",
    }


def test_direct_video_canonicalization_completes_locked_required_suffix() -> None:
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
    ] == [[1], [2], [3]]
    assert vertical["attention_sequence"][-1]["camera_behavior"] == "hold"
    assert vertical["attention_sequence"][-1]["cut_admissible"] is True
    assert vertical["attention_sequence"][-1]["transition_preference"] == "cut"
    assert {
        change["rule"] for change in changes
    } >= {
        "semantic_order_locked_missing_required_suffix_is_completed_without_reordering"
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
