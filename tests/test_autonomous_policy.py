from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import jascue_video_lab.delivery_pipeline as delivery_pipeline
from jascue_video_lab.autonomous_policy import (
    AutonomousDegradationManifest,
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
    authorize_decision,
    validate_authority_binding,
)
from jascue_video_lab.billing import (
    BudgetExceeded,
    BudgetLedger,
    estimate_paid_call,
)
from jascue_video_lab.storage import write_json


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
        "reuse_degradation": AutonomousDegradationManifest(
            policy_reference=policy.policy_reference,
            generated_at="now",
        ).model_dump(mode="json"),
    }.items():
        path = tmp_path / f"{key}.json"
        write_json(path, payload)
        context_paths[key] = path
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
