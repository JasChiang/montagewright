from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jascue_video_lab.delivery_pipeline as pipeline
import pytest
from jascue_video_lab.autonomous_policy import (
    AutonomousDegradationManifest,
    AutonomousEditPolicy,
    BudgetPolicy,
    DegradationRecord,
    DurationPolicy,
)
from jascue_video_lab.final_edit_qa import (
    AutonomousFinalEditQa,
    DeterministicDeliveryEvidence,
    run_deterministic_delivery_qa,
)
from jascue_video_lab.media import sha256_file


@pytest.mark.parametrize(
    ("picture_ready", "expected_state"),
    [
        (True, "ready_for_human_review"),
        (False, "review_required"),
    ],
)
def test_delivery_pipeline_hash_binds_mux_and_runs_qa_on_final_media(
    tmp_path: Path,
    monkeypatch,
    picture_ready: bool,
    expected_state: str,
) -> None:
    brief = tmp_path / "brief.json"
    music = tmp_path / "music.wav"
    lock = tmp_path / "music-lock.json"
    picture = tmp_path / "picture.mp4"
    render_manifest = tmp_path / "render-manifest.json"
    for path, payload in (
        (brief, b"{}"),
        (music, b"music"),
        (lock, b"{}"),
        (picture, b"picture"),
        (render_manifest, b"{}"),
    ):
        path.write_bytes(payload)

    monkeypatch.setattr(
        pipeline,
        "run_feature_cut_experiment",
        lambda **_kwargs: {
            "ready_for_human_review": picture_ready,
            "media_rendered": True,
            "run_state": (
                "ready_for_human_review"
                if picture_ready
                else "review_preview"
            ),
            "horizontal_output": str(picture),
            "vertical_output": None,
            "manifest_path": str(render_manifest),
        },
    )
    fake_lock = SimpleNamespace(music_id=f"sha256:{sha256_file(music)}")
    monkeypatch.setattr(
        pipeline.MusicMapLock,
        "model_validate",
        lambda _payload: fake_lock,
    )
    monkeypatch.setattr(pipeline, "read_json", lambda _path: {})
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda _path: SimpleNamespace(duration_ms=60_000),
    )
    plan = SimpleNamespace()
    monkeypatch.setattr(
        pipeline,
        "plan_single_interval_music_assembly",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        pipeline,
        "write_music_assembly_artifacts",
        lambda *_args, **_kwargs: None,
    )

    def fake_music_render(_source, _plan, output, output_dir):
        del output_dir
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered-music")
        manifest = output.parent / "music-assembly-render.json"
        manifest.write_bytes(b"{}")
        return SimpleNamespace(output_audio_path=output, manifest_path=manifest)

    monkeypatch.setattr(
        pipeline,
        "render_single_interval_music_assembly",
        fake_music_render,
    )

    def fake_delivery(**kwargs):
        output = kwargs["output_path"]
        manifest = kwargs["manifest_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"final-mux")
        manifest.write_bytes(b"{}")
        return SimpleNamespace(output_path=output, manifest_path=manifest)

    monkeypatch.setattr(pipeline, "assemble_music_only_delivery", fake_delivery)
    prepared = SimpleNamespace(
        proxy_path=tmp_path / "qa-proxy.mp4",
        input_hashes={"proxy_sha256": "a" * 64},
    )
    prepared.proxy_path.write_bytes(b"proxy")
    monkeypatch.setattr(
        pipeline,
        "prepare_final_edit_qa",
        lambda **_kwargs: prepared,
    )
    qa = SimpleNamespace(
        result=SimpleNamespace(
            global_review=SimpleNamespace(
                disposition="ready_for_human_review"
            )
        ),
        run_dir=tmp_path / "qa-run",
        cache_hit=False,
    )
    monkeypatch.setattr(
        pipeline,
        "execute_final_edit_qa",
        lambda **_kwargs: qa,
    )

    class FakeClient:
        client = object()

        def __init__(self, *, model_id: str) -> None:
            assert model_id

        def ensure_video_upload(self, path: Path, artifact_dir: Path):
            assert path == prepared.proxy_path
            assert artifact_dir.name == "a" * 64
            return {"uri": "files/qa", "mime_type": "video/mp4"}, True

        def close(self) -> None:
            return None

    monkeypatch.setattr(pipeline, "GeminiLabClient", FakeClient)

    result = pipeline.run_feature_delivery_pipeline(
        feature_cut_kwargs={"catalog_path": tmp_path / "catalog.json"},
        brief_path=brief,
        music_path=music,
        music_lock_path=lock,
        output_dir=tmp_path / "delivery",
    )

    assert result["state"] == expected_state
    assert result["picture_ready_for_human_review"] is picture_ready
    assert result["delivery_eligible"] is False
    assert result["human_approval_status"] == "not_run"
    assert Path(result["aspects"]["horizontal"]["final_output"]).read_bytes() == (
        b"final-mux"
    )


def test_delivery_pipeline_stops_before_music_when_picture_media_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "run_feature_cut_experiment",
        lambda **_kwargs: {
            "ready_for_human_review": False,
            "media_rendered": False,
            "run_state": "partial",
        },
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("music stage must not run")

    monkeypatch.setattr(
        pipeline,
        "plan_single_interval_music_assembly",
        forbidden,
    )
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="did not produce reviewable picture media",
    ):
        pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={},
            brief_path=tmp_path / "brief.json",
            music_path=tmp_path / "music.wav",
            music_lock_path=tmp_path / "music-lock.json",
            output_dir=tmp_path / "delivery",
        )
    assert called is False


def _autonomous_policy(profile: str = "autonomous_strict") -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile=profile,
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=60_000,
            min_ms=50_000,
            max_ms=70_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
    )


def test_autonomous_music_lock_refreshes_stale_policy_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    music = tmp_path / "music.wav"
    proposal = tmp_path / "music-map.proposal.json"
    saved_lock_path = tmp_path / "music-map.lock.v2.json"
    music.write_bytes(b"music")
    proposal.write_text("{}")
    saved_lock_path.write_text('{"lock": true}')
    music_id = f"sha256:{sha256_file(music)}"
    saved_lock = SimpleNamespace(
        music_id=music_id,
        proposal_path=str(proposal),
        proposal_sha256=sha256_file(proposal),
        authority=SimpleNamespace(policy_reference="sha256:" + "0" * 64),
        bpm=117.0,
        first_downbeat_sample=13_940,
        meter=4,
    )
    monkeypatch.setattr(
        pipeline.MusicMapLock,
        "model_validate",
        lambda _payload: saved_lock,
    )
    parsed_proposal = SimpleNamespace(music_id=music_id)
    monkeypatch.setattr(
        pipeline.MusicMapProposal,
        "model_validate",
        lambda _payload: parsed_proposal,
    )
    captured = {}

    def fake_lock(proposal_value, **kwargs):
        captured["proposal"] = proposal_value
        captured.update(kwargs)
        return {"contract_version": "music-map-lock-v2"}

    monkeypatch.setattr(
        pipeline,
        "lock_music_map_with_auto_policy",
        fake_lock,
    )
    policy = _autonomous_policy()
    refreshed = pipeline._bind_music_lock_to_autonomous_policy(
        music_path=music,
        music_lock_path=saved_lock_path,
        policy=policy,
        output_dir=tmp_path / "delivery",
    )

    assert refreshed.is_file()
    assert captured["proposal"] is parsed_proposal
    assert captured["authority"].policy_reference == policy.policy_reference
    assert captured["bpm"] == 117.0
    assert saved_lock_path != refreshed


def _passing_deterministic_report(policy: AutonomousEditPolicy):
    return run_deterministic_delivery_qa(
        DeterministicDeliveryEvidence(
            media_playable=True,
            pts_valid=True,
            unexpected_freeze_count=0,
            containment_passed=True,
            identity_passed=True,
            relation_passed=True,
            panel_same_pts_passed=True,
            relative_scale_lock_passed=True,
            cue_delta_frames={"event": 2},
            cue_tolerance_frames={"event": 2},
            cue_id_by_event={"event": "cue-1"},
            required_cue_event_ids=("event",),
            cue_boundary_coverage_audited=True,
            music_edit_boundary_coverage_passed=True,
            synthetic_motion_motivated=True,
            synthetic_reversal_count=0,
            settle_passed=True,
            source_camera_motion_audited=True,
            unwanted_source_camera_motion_count=0,
            dwell_bounds_audited=True,
            excessive_dwell_count=0,
            dead_air_audited=True,
            dead_air_count=0,
            concat_padding_audited=True,
            unauthorized_concat_padding_count=0,
            readability_passed=True,
            reuse_authorized=True,
            omissions_authorized=True,
            hard_evidence_passed=True,
        ),
        policy=policy,
    )


def _passing_autonomous_qa() -> AutonomousFinalEditQa:
    return AutonomousFinalEditQa(
        mode="autonomous_final_9x16",
        render_sha256="a" * 64,
        proxy_sha256="b" * 64,
        manifest_sha256="c" * 64,
        brief_sha256="d" * 64,
        context_hashes={"editorial_beat_contracts": "e" * 64},
        issues=[],
        opening_observation="The subject is established.",
        ending_observation="The final result resolves with music.",
        sequence_observation="Required beats remain understandable.",
        qa_observation_status="no_blocking_observation",
        limitations=[],
    )


def test_autonomous_strict_reaches_delivery_without_human_artifact() -> None:
    policy = _autonomous_policy()
    degradation = AutonomousDegradationManifest(
        policy_reference=policy.policy_reference,
        generated_at="now",
    )

    state, authority = pipeline.authorize_autonomous_delivery(
        policy=policy,
        deterministic_qa=_passing_deterministic_report(policy),
        qa_results={"9:16": _passing_autonomous_qa()},
        degradation=degradation,
        input_artifact_hashes=("sha256:" + "f" * 64,),
        gemini_interaction_ids=("qa-observation-1",),
    )

    assert state == "delivery_eligible"
    assert authority.authority_type == "auto_policy"
    assert authority.policy_reference == policy.policy_reference
    assert authority.decision_scope == "final_delivery"
    assert authority.gemini_interaction_ids == ("qa-observation-1",)


def test_best_effort_records_degradation_but_never_hard_omission() -> None:
    policy = _autonomous_policy("autonomous_best_effort")
    degradation = AutonomousDegradationManifest(
        policy_reference=policy.policy_reference,
        records=(
            DegradationRecord(
                beat_id="optional-beat",
                action="optional_beat_omitted",
                reason_code="duration_reconciliation",
            ),
        ),
        generated_at="now",
    )

    state, authority = pipeline.authorize_autonomous_delivery(
        policy=policy,
        deterministic_qa=_passing_deterministic_report(policy),
        qa_results={"9:16": _passing_autonomous_qa()},
        degradation=degradation,
        input_artifact_hashes=("sha256:" + "f" * 64,),
    )

    assert state == "best_effort_complete"
    assert "authorized_degradations_recorded" in authority.decision_codes
