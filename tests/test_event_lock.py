from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
    authorize_decision,
)
from jascue_video_lab.event_lock import (
    EditorialBeatContract,
    EvidenceFulfillmentObservation,
    ExactEventSelection,
    bracket_dense_frames_by_difference,
    build_cue_alignment_evidence,
    authorize_trim_intent_decision,
    bind_editorial_contract_to_selected_evidence,
    bind_grouped_event_lock_ids,
    bind_selected_fulfillment,
    compile_illustrative_coverage_contracts,
    exact_event_resolver_binding_sha256,
    hard_exact_event_requirements_satisfied,
    illustrative_coverage_planning_instruction,
    load_editorial_beat_contracts,
    resolve_exact_event_locks,
    select_strongest_evidence_fulfillment,
    write_exact_event_bundle,
)
from jascue_video_lab.gemini import GeminiLabClient, MODEL_ID
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    DenseFrame,
    DenseFrameCatalog,
    TrimFrameEvidence,
    TrimIntentDecision,
)


def _catalog(tmp_path: Path, *, count: int = 16) -> DenseFrameCatalog:
    frames: list[DenseFrame] = []
    for index in range(count):
        path = tmp_path / f"{index:02d}.png"
        color = 0 if index < count // 2 else 255
        Image.new("L", (32, 18), color=color).save(path)
        frame_hash = sha256_file(path)
        frames.append(
            DenseFrame(
                frame_id=f"DF{index:06d}",
                event_id="shot-window",
                requested_time_ms=index * 125,
                frame_time_ms=index * 125,
                frame_pts=index * 4,
                frame_hash=frame_hash,
                width=32,
                height=18,
                image_path=str(path),
                transport_image_path=str(path),
                transport_image_hash=frame_hash,
            )
        )
    contact_path = tmp_path / "contact.jpg"
    Image.new("RGB", (64, 36), color=(32, 32, 32)).save(contact_path)
    return DenseFrameCatalog(
        source_asset_id="sha256:" + "a" * 64,
        event_id="shot-window",
        sampling_fps=8,
        source_start_ms=0,
        source_end_ms=count * 125,
        frames=frames,
        contact_sheet_paths=[str(contact_path)],
        contact_sheet_hashes=[sha256_file(contact_path)],
        generated_at="now",
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


def test_difference_bracket_is_bounded_and_keeps_change_frontier(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)

    bracket = bracket_dense_frames_by_difference(catalog, max_frames=8)

    assert len(bracket) == 8
    assert bracket[0].frame_id == "DF000000"
    assert bracket[-1].frame_id == "DF000015"
    assert {"DF000007", "DF000008"} <= {
        frame.frame_id for frame in bracket
    }


def test_exact_event_binding_rehashes_dense_evidence_files(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    contract = EditorialBeatContract(
        beat_id="gesture",
        priority="hard",
        evidence_query_lock_sha256="1" * 64,
        required_target_ids=("phone",),
        narrative_function="feature_evidence",
        visual_events=(
            {
                "event_type": "camera_gesture_apex",
                "cue_relation": "accent",
                "tolerance_frames": 2,
            },
        ),
        duration={
            "minimum_readable_frames": 12,
            "preferred_frames": 24,
            "maximum_frames": 48,
        },
        relation_mode="single_subject",
        allowed_reconstruction=("continuous",),
    )

    binding = exact_event_resolver_binding_sha256(
        catalog=catalog,
        contracts=(contract,),
        model_id=MODEL_ID,
    )
    assert len(binding) == 64

    Path(catalog.contact_sheet_paths[0]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="contact sheet integrity mismatch"):
        exact_event_resolver_binding_sha256(
            catalog=catalog,
            contracts=(contract,),
            model_id=MODEL_ID,
        )


def test_exact_event_selection_maps_only_existing_ids_to_pts(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    selection = ExactEventSelection(
        event_id="generation-result",
        event_type="generation_result_stable_start",
        selected_frame_id="DF000008",
        support_start_frame_id="DF000007",
        support_end_frame_id="DF000010",
        confidence=0.91,
    )

    locks = resolve_exact_event_locks(
        catalog,
        [selection],
        gemini_interaction_id="interaction-1",
        input_artifact_hashes=("sha256:" + "c" * 64,),
    )

    assert locks[0].source_pts == 32
    assert locks[0].source_time_ms == 1_000
    assert locks[0].source_frame_id == "DF000008"

    with pytest.raises(ValueError, match="outside the dense catalog"):
        resolve_exact_event_locks(
            catalog,
            [
                selection.model_copy(
                    update={"selected_frame_id": "DF999999"}
                )
            ],
            gemini_interaction_id="interaction-2",
            input_artifact_hashes=("sha256:" + "c" * 64,),
        )


def test_selected_window_bundle_binds_runtime_query_and_persists(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "beats.json"
    template_path.write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "beat_id": "watch-ui",
                        "feature_id": "watch9",
                        "priority": "hard",
                        "evidence_query_lock_sha256": "1" * 64,
                        "required_target_ids": ["placeholder"],
                        "narrative_function": "feature_evidence",
                        "visual_events": [
                            {
                                "event_type": "watch_ui_state_change",
                                "cue_relation": "music_emphasis",
                                "tolerance_frames": 2,
                            }
                        ],
                        "duration": {
                            "minimum_readable_frames": 18,
                            "preferred_frames": 42,
                            "maximum_frames": 72,
                        },
                        "relation_mode": "context_detail",
                        "allowed_reconstruction": ["continuous"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    template = load_editorial_beat_contracts(template_path)[0]
    bound = bind_editorial_contract_to_selected_evidence(
        template,
        evidence_query_lock_sha256="a" * 64,
        required_target_ids=("watch-face", "watch-face"),
    )
    lock = resolve_exact_event_locks(
        _catalog(tmp_path),
        [
            ExactEventSelection(
                event_id="watch-state",
                event_type="watch_ui_state_change",
                selected_frame_id="DF000008",
                support_start_frame_id="DF000007",
                support_end_frame_id="DF000009",
                confidence=0.9,
            )
        ],
        gemini_interaction_id="interaction-1",
        input_artifact_hashes=("sha256:" + "b" * 64,),
    )[0]

    paths = write_exact_event_bundle(
        tmp_path / "context",
        contracts=(bound,),
        locks=(lock,),
        selected_windows=(
            {
                "feature_id": "watch9",
                "source_in_ms": 0,
                "source_out_ms": 2_000,
            },
        ),
    )

    assert bound.evidence_query_lock_sha256 == "a" * 64
    assert bound.required_target_ids == ("watch-face",)
    assert set(paths) == {
        "editorial_beat_contracts",
        "exact_event_locks",
    }
    saved = json.loads(paths["exact_event_locks"].read_text("utf-8"))
    assert saved["locks"][0]["source_frame_id"] == "DF000008"
    assert saved["selected_windows"][0]["feature_id"] == "watch9"


def test_grouped_event_ids_bind_by_type_when_model_reorders(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    locks = resolve_exact_event_locks(
        catalog,
        [
            ExactEventSelection(
                event_id="model-freeze",
                event_type="freeze_start",
                selected_frame_id="DF000009",
                support_start_frame_id="DF000008",
                support_end_frame_id="DF000010",
                confidence=0.9,
            ),
            ExactEventSelection(
                event_id="model-reaction",
                event_type="group_laugh_reaction_peak",
                selected_frame_id="DF000008",
                support_start_frame_id="DF000007",
                support_end_frame_id="DF000009",
                confidence=0.9,
            ),
        ],
        gemini_interaction_id="interaction-1",
        input_artifact_hashes=("sha256:" + "b" * 64,),
    )
    contract = EditorialBeatContract(
        beat_id="closing",
        feature_id="closing",
        priority="preferred",
        evidence_query_lock_sha256="a" * 64,
        required_target_ids=("group",),
        narrative_function="closing",
        visual_events=(
            {
                "event_type": "group_laugh_reaction_peak",
                "cue_relation": "phrase_ending",
                "tolerance_frames": 2,
            },
            {
                "event_type": "freeze_start",
                "cue_relation": "phrase_ending",
                "tolerance_frames": 2,
            },
        ),
        duration={
            "minimum_readable_frames": 18,
            "preferred_frames": 45,
            "maximum_frames": 75,
        },
        relation_mode="simultaneous_relation",
        allowed_reconstruction=("continuous", "intentional_freeze"),
    )

    bound = bind_grouped_event_lock_ids(locks, (contract,))

    assert [lock.event_id for lock in bound] == [
        "closing:freeze_start",
        "closing:group_laugh_reaction_peak",
    ]


def test_repeated_event_type_requires_beat_qualified_lock_ids(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    locks = resolve_exact_event_locks(
        catalog,
        [
            ExactEventSelection(
                event_id="ambiguous-one",
                event_type="state_change",
                selected_frame_id="DF000007",
                support_start_frame_id="DF000006",
                support_end_frame_id="DF000008",
                confidence=0.9,
            ),
            ExactEventSelection(
                event_id="ambiguous-two",
                event_type="state_change",
                selected_frame_id="DF000009",
                support_start_frame_id="DF000008",
                support_end_frame_id="DF000010",
                confidence=0.9,
            ),
        ],
        gemini_interaction_id="interaction-1",
        input_artifact_hashes=("sha256:" + "b" * 64,),
    )
    contracts = tuple(
        EditorialBeatContract(
            beat_id=beat_id,
            feature_id=beat_id,
            priority="hard",
            evidence_query_lock_sha256="a" * 64,
            required_target_ids=(beat_id,),
            narrative_function="feature_evidence",
            visual_events=(
                {
                    "event_type": "state_change",
                    "cue_relation": "music_emphasis",
                    "tolerance_frames": 2,
                },
            ),
            duration={
                "minimum_readable_frames": 12,
                "preferred_frames": 24,
                "maximum_frames": 48,
            },
            relation_mode="single_subject",
            allowed_reconstruction=("continuous",),
        )
        for beat_id in ("first-beat", "second-beat")
    )

    with pytest.raises(
        ValueError,
        match="ambiguous legacy ExactEventLock ID",
    ):
        bind_grouped_event_lock_ids(locks, contracts)

    qualified = (
        locks[0].model_copy(update={"event_id": "first-beat:state_change"}),
        locks[1].model_copy(update={"event_id": "second-beat:state_change"}),
    )
    bound = bind_grouped_event_lock_ids(qualified, contracts)

    assert {lock.event_id for lock in bound} == {
        "first-beat:state_change",
        "second-beat:state_change",
    }
    assert hard_exact_event_requirements_satisfied(contracts, bound)
    assert not hard_exact_event_requirements_satisfied(
        contracts,
        (bound[0],),
    )


def test_trim_authority_requires_exact_event_inside_immutable_trim(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    event_lock = resolve_exact_event_locks(
        catalog,
        [
            ExactEventSelection(
                event_id="gesture",
                event_type="camera_gesture_apex",
                selected_frame_id="DF000008",
                support_start_frame_id="DF000007",
                support_end_frame_id="DF000009",
                confidence=0.9,
            )
        ],
        gemini_interaction_id="interaction-1",
        input_artifact_hashes=("sha256:" + "c" * 64,),
    )[0]
    first = TrimFrameEvidence(
        frame_id="DF000004",
        requested_time_ms=500,
        frame_time_ms=500,
        frame_pts=16,
        frame_hash="5" * 64,
    )
    decision = TrimIntentDecision(
        source_asset_id=catalog.source_asset_id,
        event_id="shot-window",
        shot_id="shot-1",
        usable=True,
        first_included_frame=first,
        last_included_frame=first,
        exclusive_out_frame=None,
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=500,
        source_out_ms=1_500,
        source_in_pts=16,
        source_out_pts=48,
        handle_in_ms=0,
        handle_out_ms=2_000,
        tail_intent="natural_pause",
        proposal_path=str(tmp_path / "proposal.json"),
        catalog_path=str(tmp_path / "catalog.json"),
    )
    policy = _policy()
    authority = authorize_decision(
        policy,
        decision_scope="trim_intent",
        input_artifact_hashes=("sha256:" + "d" * 64,),
        deterministic_gate_results={
            "trim_bounds": "passed",
            "event_inside_trim": "passed",
        },
        decision_codes=("exact_event_trim_bound",),
    )

    authorized = authorize_trim_intent_decision(
        decision,
        exact_event_locks=[event_lock],
        authority=authority,
        policy=policy,
    )

    assert authorized.requires_human_review is False
    assert authorized.approval_status == "approved"

    contextual_authorized = authorize_trim_intent_decision(
        decision,
        exact_event_locks=[],
        authority=authority,
        policy=policy,
    )
    assert contextual_authorized.exact_event_lock_sha256s == ()

    with pytest.raises(ValueError, match="outside immutable trim"):
        authorize_trim_intent_decision(
            decision.model_copy(
                update={"source_out_ms": 900, "source_out_pts": 28}
            ),
            exact_event_locks=[event_lock],
            authority=authority,
            policy=policy,
        )


def test_cue_alignment_uses_actual_frame_delta(tmp_path: Path) -> None:
    event_lock = resolve_exact_event_locks(
        _catalog(tmp_path),
        [
            ExactEventSelection(
                event_id="ui-change",
                event_type="watch_ui_state_change",
                selected_frame_id="DF000008",
                support_start_frame_id="DF000007",
                support_end_frame_id="DF000009",
                confidence=0.9,
            )
        ],
        gemini_interaction_id="interaction-1",
        input_artifact_hashes=("sha256:" + "c" * 64,),
    )[0]

    evidence = build_cue_alignment_evidence(
        event_lock,
        cue_id="locked-cue-00001",
        cue_sample_index=48_000,
        music_sample_rate=48_000,
        project_event_time_ms=1_067,
        fps_numerator=30,
        tolerance_frames=2,
    )

    assert evidence.delta_frames == 2
    assert evidence.passed is True


def test_samsung_fixture_encodes_exact_event_music_relations() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "autonomous"
        / "samsung-editorial-beats.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    beats = [
        EditorialBeatContract.model_validate(beat)
        for beat in payload["beats"]
    ]
    mapping = {
        event.event_type: (
            event.cue_relation,
            beat.narrative_function,
        )
        for beat in beats
        for event in beat.visual_events
    }

    assert mapping["camera_gesture_apex"][0] == "accent"
    assert mapping["generation_result_stable_start"] == (
        "principal_downbeat",
        "global_energy_peak",
    )
    assert mapping["watch_ui_state_change"][0] == "music_emphasis"
    assert mapping["underwater_lift_apex"][0] == "music_emphasis"
    assert mapping["group_laugh_reaction_peak"][0] == "phrase_ending"
    assert mapping["freeze_start"][0] == "phrase_ending"
    by_beat = {beat.beat_id: beat for beat in beats}
    assert (
        by_beat["ai_generation_payoff"].minimum_fulfillment_level
        == "contextual_identity"
    )
    assert by_beat["watch_ui"].minimum_fulfillment_level == (
        "contextual_identity"
    )


def test_legacy_editorial_contract_remains_direct_exact_compatible() -> None:
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "legacy-result",
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

    selected = select_strongest_evidence_fulfillment(
        beat,
        [
            EvidenceFulfillmentObservation(
                candidate_id="candidate-direct",
                evidence_provenance="direct_result",
                available_visual_event_types=("result_stable_start",),
            )
        ],
    )

    assert selected.fulfillment_level == "direct_demonstration"
    assert selected.exact_event_required is True
    assert selected.visual_events == beat.visual_events


def test_illustrative_policy_compiles_only_preferred_contextual_fallbacks() -> None:
    def legacy_contract(priority: str) -> EditorialBeatContract:
        return EditorialBeatContract.model_validate(
            {
                "beat_id": f"{priority}-camera",
                "feature_id": f"{priority}-feature",
                "priority": priority,
                "evidence_query_lock_sha256": "1" * 64,
                "required_target_ids": ["product"],
                "allowed_evidence_provenance": ["direct_physical_action"],
                "narrative_function": "feature_evidence",
                "visual_events": [
                    {
                        "event_type": "camera_gesture_apex",
                        "cue_relation": "accent",
                        "tolerance_frames": 2,
                    }
                ],
                "duration": {
                    "minimum_readable_frames": 12,
                    "preferred_frames": 24,
                    "maximum_frames": 48,
                },
                "relation_mode": "single_subject",
                "allowed_reconstruction": ["continuous", "solid_fit"],
            }
        )

    hard = legacy_contract("hard")
    preferred = legacy_contract("preferred")
    compiled = compile_illustrative_coverage_contracts(
        (hard, preferred),
        policy="related_product_or_environment_when_direct_absent",
    )

    assert compiled[0] is hard
    assert compiled[1].minimum_fulfillment_level == "contextual_identity"
    direct_selection = select_strongest_evidence_fulfillment(
        compiled[1],
        [
            EvidenceFulfillmentObservation(
                candidate_id="related-product-empty-shot",
                evidence_provenance="context_only",
                available_visual_event_types=(),
            ),
            EvidenceFulfillmentObservation(
                candidate_id="direct-camera-gesture",
                evidence_provenance="direct_physical_action",
                available_visual_event_types=("camera_gesture_apex",),
            ),
        ],
    )
    assert direct_selection.candidate_id == "direct-camera-gesture"
    assert direct_selection.fulfillment_level == "direct_demonstration"
    selection = select_strongest_evidence_fulfillment(
        compiled[1],
        [
            EvidenceFulfillmentObservation(
                candidate_id="related-product-empty-shot",
                evidence_provenance="context_only",
                available_visual_event_types=(),
            )
        ],
    )
    assert selection.fulfillment_level == "contextual_identity"
    assert selection.claim_support_level == "illustrative_only"
    assert "contextual_visual_substitution" in selection.degradation_codes
    assert "specific_claim_copy_suppressed" in selection.copy_suppression_codes
    with pytest.raises(ValueError, match="hard-camera/direct_demonstration"):
        select_strongest_evidence_fulfillment(
            compiled[0],
            [
                EvidenceFulfillmentObservation(
                    candidate_id="related-product-empty-shot",
                    evidence_provenance="context_only",
                    available_visual_event_types=(),
                )
            ],
        )

    instruction = illustrative_coverage_planning_instruction(
        "related_product_or_environment_when_direct_absent"
    )
    assert "產品空景或環境空景" in instruction
    assert "不得取代 hard evidence" in instruction


def test_fulfillment_chooses_direct_before_contextual_fallback() -> None:
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "feature-chapter",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["product"],
            "narrative_function": "feature_evidence",
            "minimum_fulfillment_level": "contextual_identity",
            "fulfillment_alternatives": [
                {
                    "fulfillment_level": "direct_demonstration",
                    "accepted_evidence_provenance": [
                        "direct_ui_interaction",
                        "direct_result",
                    ],
                    "required_observable_predicates": ["visible_result"],
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
                        "context_only",
                        "prerecorded_screen_playback",
                    ],
                    "claim_support_level": "illustrative_only",
                    "exact_event_requirement": "none",
                    "degradation_codes": [
                        "direct_demonstration_unavailable",
                        "contextual_visual_substitution",
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
            "allowed_reconstruction": ["continuous", "solid_fit"],
        }
    )

    selected = select_strongest_evidence_fulfillment(
        beat,
        [
            EvidenceFulfillmentObservation(
                candidate_id="candidate-context",
                evidence_provenance="context_only",
                available_visual_event_types=(),
            ),
            EvidenceFulfillmentObservation(
                candidate_id="candidate-direct",
                evidence_provenance="direct_result",
                observable_predicates=("visible_result",),
                available_visual_event_types=("result_stable_start",),
            ),
        ],
    )

    assert selected.candidate_id == "candidate-direct"
    assert selected.fulfillment_level == "direct_demonstration"
    assert selected.claim_support_level == "direct"
    assert selected.degradation_codes == ()


def test_missing_exact_event_uses_context_without_fabricating_event() -> None:
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "feature-chapter",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["product"],
            "narrative_function": "feature_evidence",
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
                        "direct_result",
                        "prerecorded_screen_playback",
                        "context_only",
                    ],
                    "claim_support_level": "illustrative_only",
                    "exact_event_requirement": "none",
                    "degradation_codes": [
                        "direct_demonstration_unavailable",
                        "contextual_visual_substitution",
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

    selected = select_strongest_evidence_fulfillment(
        beat,
        [
            EvidenceFulfillmentObservation(
                candidate_id="candidate-playback",
                evidence_provenance="prerecorded_screen_playback",
                available_visual_event_types=("state_change",),
            ),
            EvidenceFulfillmentObservation(
                candidate_id="candidate-no-result",
                evidence_provenance="direct_result",
                available_visual_event_types=(),
            ),
        ],
    )
    executable = bind_selected_fulfillment(beat, selected)

    assert selected.fulfillment_level == "contextual_identity"
    assert selected.claim_support_level == "illustrative_only"
    assert selected.exact_event_required is False
    assert selected.visual_events == ()
    assert "contextual_visual_substitution" in selected.degradation_codes
    assert (
        "specific_claim_copy_suppressed"
        in selected.copy_suppression_codes
    )
    assert executable.visual_events == ()
    assert executable.allowed_evidence_provenance == (
        "prerecorded_screen_playback",
    )


def test_contextual_fallback_cannot_omit_degradation_or_copy_suppression() -> None:
    payload = {
        "beat_id": "feature-chapter",
        "priority": "hard",
        "evidence_query_lock_sha256": "1" * 64,
        "required_target_ids": ["product"],
        "narrative_function": "feature_evidence",
        "minimum_fulfillment_level": "contextual_identity",
        "fulfillment_alternatives": [
            {
                "fulfillment_level": "contextual_identity",
                "accepted_evidence_provenance": ["context_only"],
                "claim_support_level": "illustrative_only",
                "exact_event_requirement": "none",
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

    with pytest.raises(
        ValueError,
        match="contextual fallback must record its substitution",
    ):
        EditorialBeatContract.model_validate(payload)


def test_direct_minimum_rejects_contextual_only_candidate() -> None:
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "hard-proof",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["result"],
            "narrative_function": "feature_evidence",
            "minimum_fulfillment_level": "direct_demonstration",
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
        match="no evidence candidate satisfies",
    ):
        select_strongest_evidence_fulfillment(
            beat,
            [
                EvidenceFulfillmentObservation(
                    candidate_id="candidate-context",
                    evidence_provenance="context_only",
                    available_visual_event_types=(),
                )
            ],
        )


def test_grouped_exact_event_call_uses_high_stills_and_no_time_schema(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path, count=8)
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "ai-payoff",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["generation_result"],
            "narrative_function": "global_energy_peak",
            "visual_events": [
                {
                    "event_type": "generation_result_stable_start",
                    "cue_relation": "principal_downbeat",
                    "tolerance_frames": 2,
                }
            ],
            "duration": {
                "minimum_readable_frames": 24,
                "preferred_frames": 54,
                "maximum_frames": 90,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )
    output = json.dumps(
        {
            "source_asset_id": catalog.source_asset_id,
            "catalog_event_id": catalog.event_id,
            "selections": [
                {
                    "event_id": "generation-result",
                    "event_type": "generation_result_stable_start",
                    "selected_frame_id": "DF000004",
                    "support_start_frame_id": "DF000003",
                    "support_end_frame_id": "DF000005",
                    "confidence": 0.9,
                }
            ],
        }
    )
    requests: list[dict[str, Any]] = []

    class Interaction:
        id = "exact-1"
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

    locks = client.select_exact_event_locks(
        catalog=catalog,
        beat_contracts=[beat],
        run_dir=tmp_path / "exact",
        input_artifact_hashes=("sha256:" + "c" * 64,),
        max_bracket_frames=8,
    )

    assert len(locks) == 1
    assert len(requests) == 1
    images = [
        item
        for item in requests[0]["input"]
        if item["type"] == "image"
    ]
    assert len(images) == 8
    assert {item["media_resolution"] for item in images} == {"high"}
    schema_text = json.dumps(requests[0]["response_format"]["schema"])
    assert "source_time_ms" not in schema_text
    assert "source_pts" not in schema_text
    prompt_text = requests[0]["input"][0]["text"]
    assert (
        '"required_event_id": "ai-payoff:generation_result_stable_start"'
        in prompt_text
    )


def test_screen_playback_cannot_trigger_direct_exact_event_paid_call(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path, count=8)
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "ai-payoff",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["generation_result"],
            "allowed_evidence_provenance": [
                "direct_ui_interaction",
                "direct_result",
            ],
            "narrative_function": "global_energy_peak",
            "visual_events": [
                {
                    "event_type": "generation_result_stable_start",
                    "cue_relation": "principal_downbeat",
                    "tolerance_frames": 2,
                }
            ],
            "duration": {
                "minimum_readable_frames": 24,
                "preferred_frames": 54,
                "maximum_frames": 90,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )
    requests: list[dict[str, Any]] = []
    client = object.__new__(GeminiLabClient)
    client.model_id = MODEL_ID
    client.client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=lambda **request: requests.append(request)
        )
    )

    with pytest.raises(
        ValueError,
        match="provenance cannot satisfy exact-event contracts",
    ):
        client.select_exact_event_locks(
            catalog=catalog,
            beat_contracts=[beat],
            run_dir=tmp_path / "exact",
            input_artifact_hashes=("sha256:" + "c" * 64,),
            evidence_provenance="prerecorded_screen_playback",
            max_bracket_frames=8,
        )

    assert requests == []


def test_grouped_exact_event_empty_selection_is_persisted_fail_closed(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path, count=8)
    beat = EditorialBeatContract.model_validate(
        {
            "beat_id": "watch-state",
            "priority": "hard",
            "evidence_query_lock_sha256": "1" * 64,
            "required_target_ids": ["watch_ui"],
            "narrative_function": "feature_evidence",
            "visual_events": [
                {
                    "event_type": "watch_ui_state_change",
                    "cue_relation": "music_emphasis",
                    "tolerance_frames": 2,
                }
            ],
            "duration": {
                "minimum_readable_frames": 24,
                "preferred_frames": 54,
                "maximum_frames": 90,
            },
            "relation_mode": "single_subject",
            "allowed_reconstruction": ["continuous"],
        }
    )
    output = json.dumps(
        {
            "source_asset_id": catalog.source_asset_id,
            "catalog_event_id": catalog.event_id,
            "selections": [],
        }
    )
    requests: list[dict[str, Any]] = []

    class Interaction:
        id = "exact-empty"
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

    run_dir = tmp_path / "exact-empty"
    locks = client.select_exact_event_locks(
        catalog=catalog,
        beat_contracts=[beat],
        run_dir=run_dir,
        input_artifact_hashes=("sha256:" + "c" * 64,),
        max_bracket_frames=8,
    )

    assert locks == ()
    assert len(requests) == 1
    persisted = json.loads(
        (run_dir / "exact_event_locks.json").read_text("utf-8")
    )
    assert persisted["locks"] == []
    assert persisted["fail_closed"] is True
    assert persisted["unresolved_events"] == [
        {
            "event_type": "watch_ui_state_change",
            "reason_code": "insufficient_exact_frame_evidence",
        }
    ]
    validation = json.loads(
        (run_dir / "exact_event.schema_validation.json").read_text("utf-8")
    )
    assert validation["semantic_status"] == "unresolved_fail_closed"
