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
    build_supplement_request_binding,
    clip_card_sha256,
    effective_event_observation_sha256,
    effective_event_observations,
    effective_event_observations_sha256,
    event_fingerprint,
    plan_supplement_needs,
    validate_supplement,
)
from jascue_video_lab.clip_card_retrieval import compact_retrieval_card
from jascue_video_lab.clip_card_supplement_runner import (
    CAPABILITY_NAMES,
    bounded_event_window_ms,
    current_supplement_request_binding,
    supplement_cache_key,
    validate_requested_observation,
)
from jascue_video_lab.models import (
    BoundaryPrecision,
    Entity,
    EntityKind,
    EvidenceOriginObservation,
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


def test_legacy_provenance_migrates_to_generic_origin() -> None:
    card = _card()
    card.events[0].evidence_provenance = "prerecorded_screen_playback"

    observation = effective_event_observations(card)["event"]

    assert (
        observation.capabilities.evidence_origin
        == AssessmentStatus.ASSESSED_PRESENT
    )
    assert observation.evidence_origin is not None
    assert observation.evidence_origin.relation == "mediated_depiction"
    assert observation.observation_basis == ObservationBasis.FULL_CLIP_VIDEO
    compact_event = compact_retrieval_card(card)["events"][0]
    assert compact_event["evidence_origin"]["relation"] == "mediated_depiction"
    assert compact_event["observation_basis"] == "full_clip_video"


def test_base_clip_event_accepts_generic_origin_without_legacy_enum() -> None:
    card = _card()
    card.events[0].evidence_origin = EvidenceOriginObservation(
        relation="mediated_depiction",
        observable_reason="The source scene visibly contains replayed media.",
    )

    observation = effective_event_observations(card)["event"]

    assert observation.evidence_origin == card.events[0].evidence_origin
    assert observation.evidence_provenance == "unknown"
    assert event_fingerprint(card.events[0]) != event_fingerprint(_card().events[0])


def test_base_clip_event_rejects_conflicting_legacy_and_generic_origin() -> None:
    payload = _card().events[0].model_dump(mode="json")
    payload["evidence_provenance"] = "direct_result"
    payload["evidence_origin"] = {
        "relation": "mediated_depiction",
        "observable_reason": "Only a replayed depiction is visible.",
    }

    try:
        FullClipEvent.model_validate(payload)
    except ValueError as error:
        assert "conflicts with generic evidence_origin" in str(error)
    else:
        raise AssertionError("conflicting base evidence origins were accepted")


def test_legacy_v1_supplement_remains_readable() -> None:
    card = _card()
    payload = _supplement(card).model_dump(mode="json")
    payload["contract_version"] = "clip-observation-supplement-v1"

    restored = ClipObservationSupplement.model_validate(payload)

    assert restored.contract_version == "clip-observation-supplement-v1"
    assert restored.event_observations[0].evidence_origin is None
    assert _supplement(card).contract_version == "clip-observation-supplement-v2"


def _request_binding(
    *,
    model_id: str = "gemini-3.6-flash",
    media_resolution: str = "low",
):
    return build_supplement_request_binding(
        model_id=model_id,
        system_instruction_sha256="e" * 64,
        prompt_sha256="c" * 64,
        response_schema_sha256="d" * 64,
        media_resolution=media_resolution,
        thinking_level="low",
        max_output_tokens=2_048,
    )


def test_v3_supplement_requires_hash_bound_request_lineage() -> None:
    card = _card()
    payload = _supplement(card).model_dump(mode="json")
    payload["contract_version"] = "clip-observation-supplement-v3"

    try:
        ClipObservationSupplement.model_validate(payload)
    except ValueError as error:
        assert "request_binding" in str(error)
    else:
        raise AssertionError("v3 supplement without request lineage was accepted")


def test_autonomous_lineage_rejects_legacy_review_supplement() -> None:
    card = _card()

    try:
        validate_supplement(
            card,
            _supplement(card),
            expected_request_binding=_request_binding(),
            require_current_lineage=True,
        )
    except ValueError as error:
        assert "review-only" in str(error)
    else:
        raise AssertionError("legacy supplement authorized autonomous reuse")


def test_autonomous_lineage_rejects_changed_media_or_model_config() -> None:
    card = _card()
    current = _request_binding()
    supplement = _supplement(card).model_copy(
        update={
            "contract_version": "clip-observation-supplement-v3",
            "request_binding": current,
        }
    )
    validate_supplement(
        card,
        supplement,
        expected_request_binding=current,
        require_current_lineage=True,
    )

    for stale in (
        _request_binding(media_resolution="high"),
        _request_binding(model_id="gemini-other"),
    ):
        try:
            validate_supplement(
                card,
                supplement,
                expected_request_binding=stale,
                require_current_lineage=True,
            )
        except ValueError as error:
            assert "stale for the current model" in str(error)
        else:
            raise AssertionError("stale supplement request lineage was accepted")


def test_request_binding_rejects_tampered_hash() -> None:
    payload = _request_binding().model_dump(mode="json")
    payload["binding_sha256"] = "f" * 64

    try:
        type(_request_binding()).model_validate(payload)
    except ValueError as error:
        assert "binding hash is invalid" in str(error)
    else:
        raise AssertionError("tampered request binding hash was accepted")


def test_supplement_cache_key_includes_complete_request_binding(
    tmp_path,
    monkeypatch,
) -> None:
    card = _card()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        "jascue_video_lab.clip_card_supplement_runner.sha256_file",
        lambda path: "1" * 64,
    )
    low = current_supplement_request_binding(
        model_id="gemini-3.6-flash",
        prompt_sha256="c" * 64,
        response_schema_sha256="d" * 64,
    )
    high = _request_binding(media_resolution="high")

    low_key = supplement_cache_key(
        source_video=source,
        card=card,
        event=card.events[0],
        requested_capabilities=["readability"],
        context_ms=2_000,
        model_id="gemini-3.6-flash",
        prompt_sha256="c" * 64,
        response_schema_sha256="d" * 64,
        request_binding=low,
    )
    high_key = supplement_cache_key(
        source_video=source,
        card=card,
        event=card.events[0],
        requested_capabilities=["readability"],
        context_ms=2_000,
        model_id="gemini-3.6-flash",
        prompt_sha256="c" * 64,
        response_schema_sha256="d" * 64,
        request_binding=high,
    )

    assert low_key["contract_version"] == "clip-observation-supplement-cache-v3"
    assert low_key["request_binding_sha256"] != high_key["request_binding_sha256"]


def test_generic_origin_override_replaces_stale_legacy_directness() -> None:
    card = _card()
    card.events[0].evidence_provenance = "direct_result"
    supplement = _supplement(card)
    incoming = supplement.event_observations[0]
    supplement = supplement.model_copy(
        update={
            "event_observations": [
                incoming.model_copy(
                    update={
                        "capabilities": incoming.capabilities.model_copy(
                            update={
                                "evidence_origin": (
                                    AssessmentStatus.ASSESSED_PRESENT
                                )
                            }
                        ),
                        "evidence_origin": EvidenceOriginObservation(
                            relation="mediated_depiction",
                            observable_reason=(
                                "The bounded media shows a display replaying footage."
                            ),
                        ),
                        "evidence_provenance": "prerecorded_screen_playback",
                    }
                )
            ]
        }
    )

    observation = effective_event_observations(card, [supplement])["event"]

    assert observation.evidence_origin is not None
    assert observation.evidence_origin.relation == "mediated_depiction"
    assert observation.evidence_provenance == "prerecorded_screen_playback"


def test_conflicting_generic_origins_fail_closed() -> None:
    card = _card()
    first = _supplement(card)
    incoming = first.event_observations[0]
    first = first.model_copy(
        update={
            "event_observations": [
                incoming.model_copy(
                    update={
                        "capabilities": incoming.capabilities.model_copy(
                            update={
                                "evidence_origin": (
                                    AssessmentStatus.ASSESSED_PRESENT
                                )
                            }
                        ),
                        "evidence_origin": EvidenceOriginObservation(
                            relation="direct_source_event",
                            observable_reason="The source event itself is visible.",
                        ),
                    }
                )
            ]
        }
    )
    second_observation = first.event_observations[0].model_copy(
        update={
            "evidence_origin": EvidenceOriginObservation(
                relation="mediated_depiction",
                observable_reason="Only a replayed depiction is visible.",
            ),
            "evidence_provenance": "prerecorded_screen_playback",
        }
    )
    second = first.model_copy(
        update={
            "supplement_id": "observation-002",
            "event_observations": [second_observation],
        }
    )

    try:
        effective_event_observations(card, [first, second])
    except ValueError as error:
        assert "conflicting evidence_origin" in str(error)
    else:
        raise AssertionError("conflicting evidence origins were merged")


def test_effective_observation_hash_is_order_stable_and_content_bound() -> None:
    card = _card()
    first = _supplement(card)
    second = first.model_copy(
        update={
            "supplement_id": "observation-002",
            "supersedes": [first.supplement_id],
        }
    )
    forward = effective_event_observations_sha256(card, [first, second])
    reverse = effective_event_observations_sha256(card, [second, first])
    merged = effective_event_observations(card, [first, second])["event"]

    assert forward == reverse
    assert effective_event_observation_sha256(merged) != (
        effective_event_observation_sha256(
            merged.model_copy(update={"clean_exit": "reset"})
        )
    )


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


def test_evidence_origin_is_a_requestable_bounded_capability() -> None:
    card = _card()
    event = card.events[0]
    observation = EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
        capabilities=EventCapabilityManifest(
            evidence_origin=AssessmentStatus.ASSESSED_PRESENT,
        ),
        evidence_origin=EvidenceOriginObservation(
            relation="mediated_depiction",
            observable_reason="The bounded video visibly contains replayed media.",
        ),
    )

    validate_requested_observation(
        observation,
        event=event,
        requested_capabilities=["evidence_origin"],
        audio_included=False,
    )

    assert "evidence_origin" in CAPABILITY_NAMES


def test_runner_accepts_v1_legacy_origin_for_requested_capability() -> None:
    card = _card()
    event = card.events[0]
    legacy = EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
        evidence_provenance="prerecorded_screen_playback",
        capabilities=EventCapabilityManifest(),
    )

    validate_requested_observation(
        legacy,
        event=event,
        requested_capabilities=["evidence_origin"],
        audio_included=False,
    )

    merged = effective_event_observations(
        card,
        [
            ClipObservationSupplement(
                contract_version="clip-observation-supplement-v1",
                supplement_id="legacy-origin",
                source_asset_id=card.source_asset_id,
                proxy_asset_id=card.proxy_asset_id,
                base_card_sha256=clip_card_sha256(card),
                supplement_prompt_sha256="c" * 64,
                response_schema_sha256="d" * 64,
                event_observations=[legacy],
                model_provenance=card.model_provenance,
            )
        ],
    )["event"]
    assert merged.evidence_origin is not None
    assert merged.evidence_origin.relation == "mediated_depiction"


def test_unrequested_evidence_origin_cannot_leak_from_bounded_call() -> None:
    card = _card()
    event = card.events[0]
    observation = EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
        capabilities=EventCapabilityManifest(
            evidence_origin=AssessmentStatus.ASSESSED_PRESENT,
        ),
        evidence_origin=EvidenceOriginObservation(
            relation="direct_source_event",
            observable_reason="The source event itself is directly visible.",
        ),
    )

    try:
        validate_requested_observation(
            observation,
            event=event,
            requested_capabilities=["action_structure"],
            audio_included=False,
        )
    except ValueError as error:
        assert "unrequested capability evidence_origin" in str(error)
    else:
        raise AssertionError("unrequested evidence origin leaked through")


def test_no_bounded_media_keeps_unknown_origin_not_assessed() -> None:
    card = _card()
    event = card.events[0]
    observation = EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        capabilities=EventCapabilityManifest(),
    )

    validate_requested_observation(
        observation,
        event=event,
        requested_capabilities=["evidence_origin"],
        audio_included=False,
        expected_observation_basis=None,
    )

    assert (
        observation.capabilities.evidence_origin
        == AssessmentStatus.NOT_ASSESSED
    )
    assert observation.evidence_origin is None


def test_no_bounded_media_cannot_claim_evidence_origin_absent() -> None:
    card = _card()
    event = card.events[0]
    observation = EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
        capabilities=EventCapabilityManifest(
            evidence_origin=AssessmentStatus.ASSESSED_ABSENT,
        ),
    )

    try:
        validate_requested_observation(
            observation,
            event=event,
            requested_capabilities=["evidence_origin"],
            audio_included=False,
            expected_observation_basis=None,
        )
    except ValueError as error:
        assert "without bounded observation media" in str(error)
    else:
        raise AssertionError("missing bounded media proved evidence absence")


def test_assessed_unknown_origin_must_remain_not_assessed() -> None:
    card = _card()
    event = card.events[0]
    observation = EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        observation_basis=ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
        capabilities=EventCapabilityManifest(
            evidence_origin=AssessmentStatus.ASSESSED_PRESENT,
        ),
        evidence_origin=EvidenceOriginObservation(
            relation="unknown",
            observable_reason="The bounded media does not support a classification.",
        ),
    )

    try:
        validate_requested_observation(
            observation,
            event=event,
            requested_capabilities=["evidence_origin"],
            audio_included=False,
        )
    except ValueError as error:
        assert "unknown evidence origin must remain not_assessed" in str(error)
    else:
        raise AssertionError("unknown evidence origin was marked assessed")
