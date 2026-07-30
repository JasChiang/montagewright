from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import jascue_video_lab.delivery_pipeline as delivery_pipeline
import jascue_video_lab.feature_cut as feature_cut
from jascue_video_lab.autonomous_policy import (
    AutonomousDegradationManifest,
    AutonomousEditPolicy,
    BudgetPolicy,
    DegradationRecord,
    DurationPolicy,
    authorize_decision,
    omissions_are_policy_authorized,
    sync_tolerance_for_priority,
    validate_authority_binding,
)
from jascue_video_lab.billing import (
    BudgetExceeded,
    BudgetLedger,
    estimate_paid_call,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.storage import read_json, write_json


def _policy(**updates: object) -> AutonomousEditPolicy:
    values: dict[str, object] = {
        "execution_profile": "autonomous_strict",
        "content_mode": "music_led_feature",
        "requested_aspects": ("9:16",),
        "duration": DurationPolicy(
            target_ms=85_000,
            min_ms=75_000,
            max_ms=90_000,
        ),
        "budget": BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
    }
    values.update(updates)
    return AutonomousEditPolicy(**values)


def _authorize_context_cue_plan(
    context_paths: dict[str, Path] | dict[str, str],
    policy: AutonomousEditPolicy,
) -> None:
    paths = {key: Path(value) for key, value in context_paths.items()}
    feature_cut._write_authorized_selected_window_cue_plan(
        paths["cue_plan"],
        proposal={
            "contract_version": "selected-window-cue-plan-v2",
            "aspect": "9:16",
            "music_map_sha256": sha256_file(paths["music_map"]),
            "music_supplied": False,
            "cue_timeline": "locked_source_timeline",
            "music_output_timeline_sha256": None,
            "alignments": [],
        },
        authority_inputs={
            "music_map": paths["music_map"],
            "editorial_beat_contracts": paths[
                "editorial_beat_contracts"
            ],
            "exact_event_locks": paths["exact_event_locks"],
        },
        policy=policy,
    )


def test_solid_fit_degradation_is_not_an_editorial_omission() -> None:
    policy = _policy()

    assert omissions_are_policy_authorized(
        policy,
        (
            DegradationRecord(
                beat_id="opening",
                action="solid_matte_fit_used",
                reason_code="required_scope_preserved",
            ),
        ),
    )


def test_contextual_degradation_requires_copy_suppression() -> None:
    with pytest.raises(
        ValidationError,
        match="must suppress specific claim copy",
    ):
        DegradationRecord(
            beat_id="contextual-feature",
            action="contextual_visual_substitution",
            reason_code="direct_demonstration_unavailable",
        )

    record = DegradationRecord(
        beat_id="contextual-feature",
        action="contextual_visual_substitution",
        reason_code="direct_demonstration_unavailable",
        copy_suppression_codes=("specific_claim_copy_suppressed",),
    )

    assert record.copy_suppression_codes == (
        "specific_claim_copy_suppressed",
    )


def test_optional_omission_requires_policy_authority() -> None:
    policy = _policy(
        editorial=_policy().editorial.model_copy(
            update={"allow_optional_beat_omission": False}
        )
    )

    assert not omissions_are_policy_authorized(
        policy,
        (
            DegradationRecord(
                beat_id="optional-detail",
                action="optional_beat_omitted",
                reason_code="no_candidate",
            ),
        ),
    )


def test_policy_hash_is_canonical_and_binds_authority() -> None:
    first = _policy()
    second = AutonomousEditPolicy.model_validate(
        first.model_dump(mode="json")
    )
    assert first.definition_sha256() == second.definition_sha256()
    assert first.policy_reference.startswith("sha256:")

    authority = authorize_decision(
        first,
        decision_scope="final_delivery",
        input_artifact_hashes=("sha256:" + "a" * 64,),
        deterministic_gate_results={
            "hard_evidence": "passed",
            "music_sync": "passed",
            "final_qa": "passed",
        },
        decision_codes=(
            "hard_evidence_passed",
            "music_sync_passed",
            "final_qa_passed",
        ),
        gemini_interaction_ids=("qa-1",),
    )

    validate_authority_binding(authority, first)
    with pytest.raises(ValueError, match="not bound"):
        validate_authority_binding(
            authority,
            _policy(content_mode="visual_demo"),
        )


def test_policy_cannot_authorize_hard_evidence_omission() -> None:
    payload = _policy().model_dump(mode="json")
    payload["editorial"]["allow_hard_evidence_omission"] = True

    with pytest.raises(ValidationError):
        AutonomousEditPolicy.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "unsupported_value"),
    (
        ("editorial", "allow_preferred_beat_substitution", True),
        (
            "recovery",
            "allow_deterministic_delivery_when_semantic_qa_unavailable",
            True,
        ),
        ("recovery", "max_request_failure_retries", 1),
        ("media_resolution", "base_clip_card", "medium"),
        ("media_resolution", "bounded_event_video", "high"),
        ("media_resolution", "text_heavy_video", "medium"),
    ),
)
def test_policy_rejects_declared_but_unsupported_execution_options(
    section: str,
    field: str,
    unsupported_value: object,
) -> None:
    payload = _policy().model_dump(mode="json")
    payload[section][field] = unsupported_value

    with pytest.raises(ValidationError):
        AutonomousEditPolicy.model_validate(payload)


def test_worker_values_are_concurrency_ceilings() -> None:
    policy = _policy()

    assert all(
        1 <= ceiling
        for ceiling in (
            policy.workers.ffmpeg_workers,
            policy.workers.proxy_workers,
            policy.workers.sam_workers,
            policy.workers.cold_clip_card_workers,
        )
    )


def test_policy_rejects_custom_scoped_replan_limits_until_route_exists() -> None:
    payload = _policy().model_dump(mode="json")
    payload["gemini_limits"]["scoped_replan"]["max_output_tokens"] = 8_192

    with pytest.raises(ValidationError, match="scoped semantic replan"):
        AutonomousEditPolicy.model_validate(payload)


def test_sync_policy_has_distinct_hard_and_preferred_ceilings() -> None:
    policy = _policy()

    assert sync_tolerance_for_priority(policy, "hard") == 2
    assert sync_tolerance_for_priority(policy, "preferred") == 4
    assert sync_tolerance_for_priority(policy, "optional") == 4


def test_budget_reserves_recovery_and_blocks_before_paid_call() -> None:
    ledger = BudgetLedger(
        max_cost_usd=0.10,
        max_interactions=5,
        reserved_recovery_fraction=0.20,
    )
    estimate = estimate_paid_call(
        stage="candidate_reel_plan",
        model_id="gemini-3.6-flash",
        media_duration_ms=120_000,
        media_resolution="low",
        text_input_tokens=2_000,
        max_output_tokens=12_288,
        thinking_level="low",
    )
    assert estimate.worst_case_cost_usd > 0

    with pytest.raises(BudgetExceeded, match="blocked before request"):
        ledger.reserve(estimate)
    assert ledger.committed_interactions == 0
    assert ledger.report()["stages"] == {}


def test_budget_reconciles_cached_input_and_stage_breakdown() -> None:
    ledger = BudgetLedger(max_cost_usd=1.25, max_interactions=25)
    estimate = estimate_paid_call(
        stage="final_qa",
        model_id="gemini-3.6-flash",
        media_duration_ms=90_000,
        media_resolution="low",
        text_input_tokens=2_000,
        max_output_tokens=8_192,
        thinking_level="low",
    )
    reservation = ledger.reserve(estimate, recovery_call=True)
    ledger.reconcile(
        reservation.reservation_id,
        usage={
            "total_input_tokens": 11_000,
            "total_cached_tokens": 8_000,
            "total_output_tokens": 2_000,
            "total_thought_tokens": 100,
        },
    )

    report = ledger.report()
    assert report["committed_interactions"] == 1
    assert report["stages"]["final_qa"]["actual_cached_input_tokens"] == 8_000
    assert report["actual_cost_usd"] > 0


def test_autonomous_preflight_rejects_aspect_mismatch_before_feature_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    called = False

    def forbidden(**_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("feature-cut must not start before preflight")

    monkeypatch.setattr(
        delivery_pipeline,
        "run_feature_cut_experiment",
        forbidden,
    )

    with pytest.raises(
        delivery_pipeline.DeliveryPipelineBlocked,
        match="requested aspect",
    ):
        delivery_pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={"aspect": "16x9"},
            brief_path=tmp_path / "brief.json",
            music_path=tmp_path / "music.wav",
            music_lock_path=tmp_path / "music-lock.json",
            output_dir=tmp_path / "delivery",
            execution_profile="autonomous_strict",
            autonomous_policy_path=policy_path,
        )
    assert called is False


def test_autonomous_preflight_rejects_runtime_model_override_before_paid_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    called = False

    def forbidden(**_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("feature-cut must not start with a foreign model")

    monkeypatch.setattr(
        delivery_pipeline,
        "run_feature_cut_experiment",
        forbidden,
    )

    with pytest.raises(
        delivery_pipeline.DeliveryPipelineBlocked,
        match="model does not match",
    ):
        delivery_pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={"aspect": "9x16"},
            brief_path=tmp_path / "brief.json",
            music_path=None,
            music_lock_path=None,
            output_dir=tmp_path / "delivery",
            model_id="different-model",
            execution_profile="autonomous_strict",
            autonomous_policy_path=policy_path,
        )
    assert called is False


def test_autonomous_preflight_rejects_hard_gate_before_paid_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    context_paths: dict[str, Path] = {}
    for key, payload in {
        "editorial_beat_contracts": [],
        "music_map": {},
        "cue_plan": {},
        "exact_event_locks": [],
            "sequence_optimization": {"result": {"outcome": "selected"}},
            "resolved_timeline": {
                "aspect": "9:16",
                "definition_sha256": "f" * 64,
            },
            "reuse_degradation": AutonomousDegradationManifest(
            policy_reference=policy.policy_reference,
            generated_at="now",
        ).model_dump(mode="json"),
    }.items():
        path = tmp_path / f"{key}.json"
        write_json(path, payload)
        context_paths[key] = path
    _authorize_context_cue_plan(context_paths, policy)
    deterministic = tmp_path / "deterministic.json"
    write_json(
        deterministic,
        {
            "media_playable": True,
            "pts_valid": True,
            "unexpected_freeze_count": 0,
            "containment_passed": True,
            "identity_passed": True,
            "relation_passed": True,
            "panel_same_pts_passed": True,
            "relative_scale_lock_passed": True,
            "cue_delta_frames": {"hard-event": 3},
            "synthetic_motion_motivated": True,
            "synthetic_reversal_count": 0,
            "settle_passed": True,
            "readability_passed": True,
            "reuse_authorized": True,
            "omissions_authorized": True,
            "hard_evidence_passed": True,
            "sequence_optimization_audited": True,
            "sequence_optimization_passed": True,
        },
    )
    called = False

    def forbidden(**_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("feature-cut must not start after a hard failure")

    monkeypatch.setattr(
        delivery_pipeline,
        "run_feature_cut_experiment",
        forbidden,
    )

    with pytest.raises(
        delivery_pipeline.DeliveryPipelineBlocked,
        match="failed before paid work.*cue_sync",
    ):
        delivery_pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={"aspect": "9x16"},
            brief_path=tmp_path / "brief.json",
            music_path=tmp_path / "music.wav",
            music_lock_path=tmp_path / "music-lock.json",
            output_dir=tmp_path / "delivery",
            execution_profile="autonomous_strict",
            autonomous_policy_path=policy_path,
            autonomous_context_paths=context_paths,
            deterministic_delivery_evidence_path=deterministic,
        )
    assert called is False


def test_autonomous_pipeline_discovers_selected_window_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    beat_templates = tmp_path / "beats.json"
    write_json(policy_path, policy)
    write_json(beat_templates, {"beats": []})
    context_paths: dict[str, str] = {}
    for key, payload in {
        "editorial_beat_contracts": {"beats": []},
        "music_map": {},
        "cue_plan": {},
        "exact_event_locks": {"locks": []},
            "sequence_optimization": {"result": {"outcome": "selected"}},
            "resolved_timeline": {
                "aspect": "9:16",
                "definition_sha256": "f" * 64,
            },
            "reuse_degradation": AutonomousDegradationManifest(
            policy_reference=policy.policy_reference,
            generated_at="now",
        ).model_dump(mode="json"),
    }.items():
        path = tmp_path / f"{key}.json"
        write_json(path, payload)
        context_paths[key] = str(path)
    _authorize_context_cue_plan(context_paths, policy)
    deterministic = tmp_path / "generated-deterministic.json"
    write_json(
        deterministic,
        {
            "media_playable": True,
            "pts_valid": True,
            "unexpected_freeze_count": 0,
            "containment_passed": True,
            "identity_passed": True,
            "relation_passed": True,
            "panel_same_pts_passed": True,
            "relative_scale_lock_passed": True,
            "cue_delta_frames": {"hard-event": 3},
            "synthetic_motion_motivated": True,
            "synthetic_reversal_count": 0,
            "settle_passed": True,
                "readability_passed": True,
                "reuse_authorized": True,
                "omissions_authorized": True,
                "hard_evidence_passed": True,
                "sequence_optimization_audited": True,
                "sequence_optimization_passed": True,
            },
        )
    preflight_deterministic = tmp_path / "preflight-deterministic.json"
    write_json(
        preflight_deterministic,
        {
            "media_playable": True,
            "pts_valid": True,
            "unexpected_freeze_count": 0,
            "containment_passed": True,
            "identity_passed": True,
            "relation_passed": True,
            "panel_same_pts_passed": True,
            "relative_scale_lock_passed": True,
            "cue_delta_frames": {"hard-event": 0},
            "cue_tolerance_frames": {"hard-event": 2},
            "cue_id_by_event": {"hard-event": "cue-1"},
            "required_cue_event_ids": ["hard-event"],
            "cue_boundary_coverage_audited": True,
            "music_edit_boundary_coverage_passed": True,
            "synthetic_motion_motivated": True,
            "synthetic_reversal_count": 0,
            "settle_passed": True,
            "source_camera_motion_audited": True,
            "unwanted_source_camera_motion_count": 0,
            "dwell_bounds_audited": True,
            "excessive_dwell_count": 0,
            "dead_air_audited": True,
            "dead_air_count": 0,
            "concat_padding_audited": True,
            "unauthorized_concat_padding_count": 0,
            "readability_passed": True,
            "reuse_authorized": True,
            "omissions_authorized": True,
            "hard_evidence_passed": True,
            "sequence_optimization_audited": True,
            "sequence_optimization_passed": True,
        },
    )
    picture = tmp_path / "picture.mp4"
    picture.write_bytes(b"picture")
    presentation_proposal = tmp_path / "presentation.proposal.json"
    write_json(
        presentation_proposal,
        {
            "contract_version": "presentation-compilation-proposal-v2",
            "aspect": "9:16",
            "final_output_path": str(picture),
            "final_output_sha256": sha256_file(picture),
            "chapters": [
                {
                    "feature_id": "fixture",
                    "segment_path": str(picture),
                    "segment_sha256": sha256_file(picture),
                }
            ],
        },
    )
    presentation_authority = (
        feature_cut._write_policy_decision_artifact(
            tmp_path / "presentation-authority.json",
            proposal_path=presentation_proposal,
            authority_inputs={
                "exact_event_locks": Path(
                    context_paths["exact_event_locks"]
                )
            },
            additional_input_hashes=(
                f"sha256:{sha256_file(picture)}",
            ),
            policy=policy,
            decision_scope="reframe",
            aspect="9:16",
            deterministic_gate_results={"presentation": "passed"},
            decision_codes=("presentation_bound",),
        )
    )
    eligibility = tmp_path / "delivery-eligibility.json"
    write_json(
        eligibility,
        feature_cut._build_feature_cut_eligibility_report(
            {
                "horizontal": {
                    "requested": False,
                    "status": "not_requested",
                    "chapters": [],
                },
                "vertical": {
                    "requested": True,
                    "status": "rendered",
                    "chapters": [
                        {
                            "feature_id": "fixture",
                            "source_clip_id": "clip",
                            "fallback_reason": None,
                            "risk_codes": [],
                        }
                    ],
                },
                "requested_candidate_recall_audit": {"complete": True},
                "quality_map_coverage_audit": {"complete": True},
                "reframe_policy_binding": None,
                "post_render_quality_qc": {
                    "requested": True,
                    "technical_qc_passed": True,
                },
            },
            execution_profile=policy.execution_profile.value,
        ),
    )
    feature_authority = feature_cut._write_policy_decision_artifact(
        tmp_path / "feature-cut-authority.json",
        proposal_path=eligibility,
        authority_inputs={
            "delivery_eligibility": eligibility,
            "presentation_authority_9x16": presentation_authority,
        },
        additional_input_hashes=(),
        policy=policy,
        decision_scope="feature_cut",
        aspect=None,
        deterministic_gate_results={"feature_cut": "passed"},
        decision_codes=("feature_cut_bound",),
    )
    render_manifest = tmp_path / "render-manifest.json"
    write_json(
        render_manifest,
        {
            "presentation_authority_by_aspect": {
                "9:16": {
                    "path": str(presentation_authority),
                    "sha256": sha256_file(presentation_authority),
                }
            },
            "feature_cut_authority": {
                "path": str(feature_authority),
                "sha256": sha256_file(feature_authority),
            },
        },
    )
    # This test targets generated deterministic-gate discovery. Authority
    # validation has dedicated tamper/pass tests, so keep the historical
    # missing-picture condition without short-circuiting on that other layer.
    picture.unlink()
    monkeypatch.setattr(
        delivery_pipeline,
        "validate_policy_decision_artifact",
        lambda path, **_kwargs: read_json(path),
    )
    received: dict[str, object] = {}

    def generated(**kwargs: object) -> object:
        received.update(kwargs)
        return {
            "autonomous_context_paths": context_paths,
            "deterministic_delivery_evidence_path": str(deterministic),
            "media_rendered": True,
            "vertical_output": str(picture),
            "manifest_path": str(render_manifest),
            "presentation_authority_paths_by_aspect": {
                "9:16": str(presentation_authority)
            },
            "feature_cut_authority_path": str(feature_authority),
        }

    monkeypatch.setattr(
        delivery_pipeline,
        "run_feature_cut_experiment",
        generated,
    )
    music_path = tmp_path / "music.wav"
    music_lock_path = tmp_path / "music-lock.json"
    music_path.write_bytes(b"music")
    write_json(music_lock_path, {})
    monkeypatch.setattr(
        delivery_pipeline,
        "_bind_music_lock_to_autonomous_policy",
        lambda **_kwargs: music_lock_path,
    )

    with pytest.raises(
        delivery_pipeline.DeliveryPipelineBlocked,
        match="generated deterministic autonomous gates failed.*cue_sync",
    ):
        delivery_pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={
                "aspect": "9x16",
                # This fixture stubs feature-cut with already-selected-window
                # artifacts; do not invoke the fresh paid planner preflight.
                "reuse_feature_plan": True,
            },
            brief_path=tmp_path / "brief.json",
            music_path=music_path,
            music_lock_path=music_lock_path,
            output_dir=tmp_path / "delivery",
            execution_profile="autonomous_strict",
            autonomous_policy_path=policy_path,
            autonomous_context_paths={
                key: Path(path) for key, path in context_paths.items()
            },
            deterministic_delivery_evidence_path=preflight_deterministic,
            editorial_beat_contracts_path=beat_templates,
        )

    assert received["autonomous_policy_path"] == policy_path
    assert received["editorial_beat_contracts_path"] == beat_templates
