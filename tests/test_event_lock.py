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
    ExactEventSelection,
    bracket_dense_frames_by_difference,
    build_cue_alignment_evidence,
    authorize_trim_intent_decision,
    bind_editorial_contract_to_selected_evidence,
    bind_grouped_event_lock_ids,
    load_editorial_beat_contracts,
    resolve_exact_event_locks,
    write_exact_event_bundle,
)
from jascue_video_lab.gemini import GeminiLabClient, MODEL_ID
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
        frames.append(
            DenseFrame(
                frame_id=f"DF{index:06d}",
                event_id="shot-window",
                requested_time_ms=index * 125,
                frame_time_ms=index * 125,
                frame_pts=index * 4,
                frame_hash=f"{index + 1:064x}",
                width=32,
                height=18,
                image_path=str(path),
                transport_image_path=str(path),
                transport_image_hash=f"{index + 1:064x}",
            )
        )
    return DenseFrameCatalog(
        source_asset_id="sha256:" + "a" * 64,
        event_id="shot-window",
        sampling_fps=8,
        source_start_ms=0,
        source_end_ms=count * 125,
        frames=frames,
        contact_sheet_paths=[str(tmp_path / "contact.jpg")],
        contact_sheet_hashes=["b" * 64],
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
