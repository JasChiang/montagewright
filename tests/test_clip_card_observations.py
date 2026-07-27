from __future__ import annotations

from jascue_video_lab.clip_card_observations import (
    AssessmentStatus,
    ClaimDecision,
    ClipObservationSupplement,
    EditingClaim,
    EventCapabilityManifest,
    EventObservationSupplement,
    EvidenceRoleMap,
    ObservationBasis,
    ObservableBeat,
    assess_editing_claim,
    clip_card_sha256,
    effective_event_observations,
    event_fingerprint,
    plan_supplement_needs,
)
from jascue_video_lab.clip_card_retrieval import compact_retrieval_card
from jascue_video_lab.clip_card_supplement_runner import (
    bounded_event_window_ms,
    validate_requested_observation,
)
from jascue_video_lab.models import (
    BoundaryPrecision,
    Entity,
    EntityKind,
    EvidenceModality,
    FullClipAttentionPhase,
    FullClipCard,
    FullClipEvent,
    ModelProvenance,
)


def _card(*, legacy_attention: bool = False) -> FullClipCard:
    phases = (
        [
            FullClipAttentionPhase(
                phase_id="person-first",
                anchor_entity_ids=["person"],
                relation_mode="single_focus",
                suggested_camera_behavior="push_in",
                observable_predicate="The person visibly begins the action.",
                transition_condition="The visible result appears on the screen.",
            )
        ]
        if legacy_attention
        else []
    )
    return FullClipCard(
        source_asset_id="sha256:" + "a" * 64,
        proxy_asset_id="sha256:" + "b" * 64,
        duration_ms=10_000,
        summary="A person operates a visible screen.",
        content_type="generic demonstration",
        entities=[
            Entity(
                entity_id="person",
                kind=EntityKind.PERSON,
                label="person",
                distinguishing_features="visible operator",
                evidence="visible in the event",
            ),
            Entity(
                entity_id="screen",
                kind=EntityKind.SCREEN,
                label="screen",
                distinguishing_features="visible stateful display",
                evidence="visible beside the person",
            ),
        ],
        events=[
            FullClipEvent(
                event_id="event",
                start_mmss="00:00",
                end_mmss="00:10",
                recommended_keyframe_mmss="00:05",
                label="visible operation",
                description="The person operates the screen.",
                observable_evidence="A person and changing screen are visible.",
                evidence_modalities=EvidenceModality.VISUAL_AND_AUDIO,
                entity_ids=["person", "screen"],
                primary_entity_ids=["person", "screen"],
                required_entity_ids=["person", "screen"],
                optional_entity_ids=[],
                avoid_overlay_entity_ids=["screen"],
                keyframe_reason="Both visible entities are clear.",
                boundary_precision=BoundaryPrecision.COARSE,
                confidence=0.8,
                action_completeness="uncertain",
                editing_uses=["demo"],
                quality_risks=[],
                framing_intent="Keep visible evidence.",
                card_opportunities=[],
                dense_refinement="required",
                dense_refinement_reasons=["Transient state may be missed."],
                grounding_targets=[],
                portrait_attention_sequence=phases,
            )
        ],
        clip_uses=["demo"],
        portrait_reframe_feasibility="conditional",
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-3.6-flash",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="1",
            run_id="run",
            generated_at="2026-07-27T00:00:00+00:00",
            interaction_id=None,
        ),
    )


def _supplement(card: FullClipCard) -> ClipObservationSupplement:
    event = card.events[0]
    return ClipObservationSupplement(
        supplement_id="observation-001",
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
                    action_structure=AssessmentStatus.ASSESSED_PRESENT,
                    evidence_roles=AssessmentStatus.ASSESSED_PRESENT,
                    observable_beats=AssessmentStatus.ASSESSED_PRESENT,
                ),
                source_action_completeness="complete",
                clean_entry="stable",
                clean_exit="stable_result",
                evidence_roles=EvidenceRoleMap(
                    primary_subject_ids=["person"],
                    result_evidence_ids=["screen"],
                    shared_context_required=True,
                    observable_reason="The operation and result share visible context.",
                ),
                observable_beats=[
                    ObservableBeat(
                        beat_id="operation",
                        kind="interaction",
                        entity_ids=["person", "screen"],
                        relation_mode="simultaneous_required",
                        observable_predicate="The person operates the visible screen.",
                        transition_condition="The visible screen state changes.",
                    )
                ],
            )
        ],
        model_provenance=card.model_provenance,
    )


def test_legacy_empty_attention_is_not_assessed_not_absent() -> None:
    observation = effective_event_observations(_card())["event"]
    assert observation.capabilities.observable_beats == AssessmentStatus.NOT_ASSESSED
    assert observation.observable_beats == []


def test_legacy_attention_keeps_observation_but_drops_camera_behavior() -> None:
    observation = effective_event_observations(_card(legacy_attention=True))["event"]
    assert observation.capabilities.observable_beats == AssessmentStatus.ASSESSED_PRESENT
    assert observation.observable_beats[0].observable_predicate.startswith("The person")
    assert "camera" not in observation.model_dump(mode="json")


def test_stale_supplement_is_rejected_when_event_changes() -> None:
    card = _card()
    supplement = _supplement(card)
    changed = card.model_copy(deep=True)
    changed.events[0].observable_evidence = "Different directly visible evidence."
    try:
        effective_event_observations(changed, [supplement])
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale event supplement was accepted")


def test_compact_retrieval_carries_capabilities_and_evidence_roles() -> None:
    card = _card()
    compact = compact_retrieval_card(card, [_supplement(card)])
    event = compact["events"][0]
    assert event["capabilities"]["evidence_roles"] == "assessed_present"
    assert event["evidence_roles"]["result_evidence_ids"] == ["screen"]
    assert event["observable_beats"][0]["relation_mode"] == "simultaneous_required"
    assert "portrait_reframe_feasibility" not in compact


def test_generic_supplement_triggers_do_not_depend_on_product_labels() -> None:
    card = _card()
    needs = plan_supplement_needs(
        card,
        frontier_event_ids={"event"},
    )
    assert needs[0].required_capabilities == [
        "action_structure",
        "evidence_roles",
        "observable_beats",
        "readability",
        "audio_role",
    ]


def test_soft_supplement_triggers_wait_until_event_enters_frontier() -> None:
    card = _card()
    event = card.events[0]
    event.action_completeness = "complete"
    event.dense_refinement = "not_needed"
    event.dense_refinement_reasons = []
    assert plan_supplement_needs(card) == []
    needs = plan_supplement_needs(card, frontier_event_ids={"event"})
    assert needs[0].required_capabilities == [
        "evidence_roles",
        "observable_beats",
        "readability",
        "audio_role",
    ]


def test_frames_cannot_prove_capability_absence() -> None:
    card = _card()
    event = card.events[0]
    try:
        EventObservationSupplement(
            event_id=event.event_id,
            event_fingerprint=event_fingerprint(event),
            observation_basis=ObservationBasis.SELECTED_FRAMES,
            capabilities=EventCapabilityManifest(
                readability=AssessmentStatus.ASSESSED_ABSENT
            ),
        )
    except ValueError as error:
        assert "assessed_absent" in str(error)
    else:
        raise AssertionError("selected frames incorrectly proved absence")


def test_claim_gate_is_scoped_and_does_not_block_retrieval() -> None:
    observation = effective_event_observations(_card())["event"]
    topical = assess_editing_claim(
        observation,
        EditingClaim.TOPICAL_RELEVANCE,
    )
    camera = assess_editing_claim(
        observation,
        EditingClaim.SEQUENTIAL_VIRTUAL_CAMERA,
    )
    assert topical.decision == ClaimDecision.READY
    assert camera.decision == ClaimDecision.REQUEST_SUPPLEMENT
    assert camera.missing_capabilities == ["evidence_roles", "observable_beats"]


def test_conflicting_active_supplements_fail_closed() -> None:
    card = _card()
    first = _supplement(card)
    second = first.model_copy(
        update={
            "supplement_id": "observation-002",
            "event_observations": [
                first.event_observations[0].model_copy(
                    update={"clean_exit": "reset"}
                )
            ],
        }
    )
    try:
        effective_event_observations(card, [first, second])
    except ValueError as error:
        assert "conflicting action_structure" in str(error)
    else:
        raise AssertionError("conflicting active supplements were merged")


def test_explicit_supersedes_resolves_prior_conflict() -> None:
    card = _card()
    first = _supplement(card)
    second = first.model_copy(
        update={
            "supplement_id": "observation-002",
            "supersedes": [first.supplement_id],
            "event_observations": [
                first.event_observations[0].model_copy(
                    update={"clean_exit": "reset"}
                )
            ],
        }
    )
    observation = effective_event_observations(card, [first, second])["event"]
    assert observation.clean_exit == "reset"


def test_bounded_event_window_uses_mmss_only_for_coarse_context() -> None:
    card = _card()
    start_ms, end_ms = bounded_event_window_ms(
        card,
        card.events[0],
        context_ms=2_000,
    )
    assert (start_ms, end_ms) == (0, 10_000)


def test_unrequested_capability_cannot_leak_from_supplement_call() -> None:
    card = _card()
    observation = _supplement(card).event_observations[0]
    try:
        validate_requested_observation(
            observation,
            event=card.events[0],
            requested_capabilities=["action_structure"],
            audio_included=False,
        )
    except ValueError as error:
        assert "unrequested capability evidence_roles" in str(error)
    else:
        raise AssertionError("unrequested assessed capability leaked through")
