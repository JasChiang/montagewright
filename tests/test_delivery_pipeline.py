from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jascue_video_lab.delivery_pipeline as pipeline
import jascue_video_lab.feature_cut as feature_cut
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
    AutonomousQaIssue,
    AutonomousRecoveryPlan,
    AutonomousRepairAction,
    DeterministicDeliveryEvidence,
    run_deterministic_delivery_qa,
)
from jascue_video_lab.billing import (
    BudgetExceeded,
    BudgetLedger,
    estimate_paid_call,
)
from jascue_video_lab.cli import build_parser
from jascue_video_lab.media import sha256_file
from jascue_video_lab.storage import read_json, write_json


def test_planner_and_delivery_share_full_clip_card_cache_contract() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plan_clip_card_feature_cut.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plan_clip_card_feature_cut_contract_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    card = SimpleNamespace(
        source_asset_id="sha256:source",
        proxy_asset_id="sha256:proxy",
    )
    prompt = "canonical prompt"

    planner_key = module._expected_clip_card_cache_key(card, prompt)
    shared_key = module.current_full_clip_card_cache_key(
        prompt=prompt,
        source_asset_id=card.source_asset_id,
        proxy_asset_id=card.proxy_asset_id,
    )

    assert planner_key == shared_key
    assert planner_key["thinking_level"] == "low"
    assert planner_key["max_output_tokens"] == 4_096


def test_budgeted_planning_failure_preserves_subprocess_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage_dir = tmp_path / "planner"

    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            17,
            ["planner"],
            output="planner stdout",
            stderr="planner stderr",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    ledger = BudgetLedger(max_cost_usd=1.0, max_interactions=3)

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="immutable subprocess artifacts",
    ):
        pipeline._run_budgeted_planning_stage(
            command=["planner"],
            stage="candidate_reel_plan",
            stage_dir=stage_dir,
            budget_ledger=ledger,
            estimated_text_tokens=10,
        )

    orchestration = read_json(stage_dir / "orchestration.json")
    assert orchestration["returncode"] == 17
    assert orchestration["stdout"] == "planner stdout"
    assert orchestration["stderr"] == "planner stderr"
    assert ledger.committed_interactions == 0


def test_budgeted_planning_reconciles_text_only_repair_separately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage_dir = tmp_path / "planner"
    first_stem = "clip-card-feature-plan.attempt-01"
    write_json(
        stage_dir / f"{first_stem}.request.json",
        {
            "model": "gemini-3.6-flash",
            "input": [
                {"type": "text", "text": "editorial contract"},
                {"type": "video", "uri": "file-api://candidate"},
            ],
        },
    )
    first_raw_path = stage_dir / f"{first_stem}.raw_interaction.json"
    write_json(
        first_raw_path,
        {
            "model": "gemini-3.6-flash",
            "usage": {
                "total_input_tokens": 1_000,
                "total_cached_tokens": 0,
                "total_output_tokens": 200,
                "total_thought_tokens": 0,
            },
        },
    )

    def complete(*_args, **_kwargs):
        stage_dir.mkdir(parents=True, exist_ok=True)
        repair_stem = "clip-card-feature-plan.attempt-02"
        write_json(
            stage_dir / f"{repair_stem}.request.json",
            {
                "model": "gemini-3.6-flash",
                "input": [{"type": "text", "text": "repair only"}],
                "generation_config": {
                    "max_output_tokens": 512,
                    "thinking_level": "minimal",
                },
            },
        )
        repair_raw = {
            "model": "gemini-3.6-flash",
            "usage": {
                "total_input_tokens": 500,
                "total_cached_tokens": 0,
                "total_output_tokens": 80,
                "total_thought_tokens": 0,
            },
        }
        write_json(
            stage_dir / f"{repair_stem}.raw_interaction.json",
            repair_raw,
        )
        # The planner writes a canonical alias of its successful repair.  It
        # is evidence, not a third paid interaction.
        write_json(
            stage_dir / "clip-card-feature-plan.raw_interaction.json",
            repair_raw,
        )
        return subprocess.CompletedProcess(["planner"], 0, "ok", "")

    monkeypatch.setattr(pipeline.subprocess, "run", complete)
    ledger = BudgetLedger(max_cost_usd=1.0, max_interactions=5)

    usage = pipeline._run_budgeted_planning_stage(
        command=["planner"],
        stage="autonomous_direct_video_edit_plan_text_only_repair",
        stage_dir=stage_dir,
        budget_ledger=ledger,
        estimated_text_tokens=10_000,
        max_output_tokens=512,
        thinking_level="minimal",
        exclude_existing_raw_interaction_paths=(first_raw_path,),
    )

    assert usage["request_count"] == 1
    assert ledger.committed_interactions == 1
    report = ledger.report()["stages"]
    assert (
        report["autonomous_direct_video_edit_plan_text_only_repair"]
        ["actual_input_tokens"]
        == 500
    )
    repair_journals = sorted(stage_dir.glob("*.paid_dispatch.json"))
    assert len(repair_journals) == 1
    assert read_json(repair_journals[0])["raw_artifact_path"].endswith(
        "attempt-02.raw_interaction.json"
    )


def test_planning_dispatch_migration_never_relabels_prior_stage_raw(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage_dir = tmp_path / "planner"
    raw_path = stage_dir / "attempt-01.raw_interaction.json"
    request_path = stage_dir / "attempt-01.request.json"
    write_json(raw_path, {"usage": {}})
    write_json(request_path, {})
    write_json(
        stage_dir / "original.paid_dispatch.json",
        {"raw_artifact_path": str(raw_path.resolve())},
    )
    monkeypatch.setattr(
        pipeline,
        "migrate_completed_legacy_paid_dispatch",
        lambda **_kwargs: pytest.fail("a prior raw interaction was relabeled"),
    )

    assert pipeline._migrate_completed_planning_dispatches(
        stage_dir=stage_dir,
        stage="text_repair",
    ) == ()


def test_warm_dispatch_migration_counts_attempt_and_alias_once(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "picture" / "gemini-plan"
    write_json(
        stage_dir / "orchestration.json",
        {
            "contract_version": "autonomous-planning-orchestration-v1",
            "stage": "autonomous_direct_video_edit_plan",
        },
    )
    request = {
        "model": "gemini-3.6-flash",
        "input": [{"type": "text", "text": "plan"}],
        "generation_config": {
            "max_output_tokens": 512,
            "thinking_level": "low",
        },
    }
    raw = {
        "id": "interaction-plan-1",
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 100,
            "total_cached_tokens": 0,
            "total_output_tokens": 20,
            "total_thought_tokens": 4,
        },
    }
    for stem in (
        "clip-card-feature-plan.attempt-01",
        "clip-card-feature-plan",
    ):
        write_json(stage_dir / f"{stem}.request.json", request)
        write_json(stage_dir / f"{stem}.raw_interaction.json", raw)

    migrated = pipeline._migrate_completed_warm_dispatches(
        root=tmp_path,
        allowed_top_level={"picture"},
    )
    assert len(migrated) == 1

    ledger = BudgetLedger(max_cost_usd=1.25, max_interactions=25)
    adopted, raw_paths = (
        pipeline.adopt_paid_dispatch_journal_state(
            budget_ledger=ledger,
            root=tmp_path,
            allowed_top_level={"picture"},
        )
    )
    assert len(adopted) == 1
    assert ledger.committed_interactions == 1
    assert not pipeline._find_unjournaled_warm_paid_artifacts(
        root=tmp_path,
        allowed_top_level={"picture"},
        journaled_raw_paths=raw_paths,
    )

    orphan = stage_dir / "other.raw_interaction.json"
    write_json(
        orphan,
        {
            **raw,
            "id": "interaction-plan-2",
            "usage": {
                **raw["usage"],
                "total_input_tokens": 101,
            },
        },
    )
    assert pipeline._find_unjournaled_warm_paid_artifacts(
        root=tmp_path,
        allowed_top_level={"picture"},
        journaled_raw_paths=raw_paths,
    ) == (orphan,)


def test_mandatory_budget_holds_are_derived_from_contract_graph(
    tmp_path: Path,
) -> None:
    policy = AutonomousEditPolicy(
        execution_profile="autonomous_strict",
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
    contracts = tmp_path / "contracts.json"
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "autonomous"
        / "samsung-editorial-beats.json"
    )
    contracts.write_bytes(source.read_bytes())

    minimums = pipeline._mandatory_paid_stage_minimums(
        policy=policy,
        editorial_beat_contracts_path=contracts,
    )

    assert minimums["final_qa"] == 2
    assert {
        key: value
        for key, value in minimums.items()
        if key.startswith("exact_event_group:")
    } == {
        "exact_event_group:closing": 1,
        "exact_event_group:fold8_camera": 1,
        "exact_event_group:galaxy_ai": 1,
        "exact_event_group:watch9": 1,
        "exact_event_group:watch_ultra2": 1,
    }
    assert {
        key: value
        for key, value in minimums.items()
        if key.startswith("multi_target_grounding:")
    } == {
        "multi_target_grounding:galaxy_ai": 1,
        "multi_target_grounding:watch9": 1,
    }

    cost_holds = pipeline._mandatory_paid_stage_cost_holds(
        policy=policy,
        editorial_beat_contracts_path=contracts,
    )
    assert set(cost_holds) == set(minimums)
    assert 0 < sum(cost_holds.values()) < policy.budget.max_gemini_cost_usd


def test_multi_aspect_qa_reserve_is_per_aspect_and_per_pass() -> None:
    policy = _autonomous_policy(
        requested_aspects=("16:9", "9:16"),
    )

    minimums = pipeline._mandatory_paid_stage_minimums(
        policy=policy,
        editorial_beat_contracts_path=None,
    )

    assert minimums["final_qa"] == 4


def test_autonomous_delivery_cli_accepts_explicit_no_music_mode() -> None:
    args = build_parser().parse_args(
        [
            "feature-delivery",
            "catalog.json",
            "brief.json",
            "--sam-checkpoint",
            "sam.pt",
            "--execution-profile",
            "autonomous_strict",
            "--autonomous-policy",
            "policy.json",
            "--editorial-beat-contracts",
            "contracts.json",
            "--prepared-clip-cards",
            "clip-cards",
            "--output-dir",
            "delivery",
        ]
    )

    assert args.music is None
    assert args.music_map_lock is None
    assert args.prepared_clip_cards == Path("clip-cards")


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


def _autonomous_policy(
    profile: str = "autonomous_strict",
    *,
    requested_aspects: tuple[str, ...] = ("9:16",),
) -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile=profile,
        content_mode="music_led_feature",
        requested_aspects=requested_aspects,
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


def test_multi_aspect_preflight_rejects_cross_aspect_context_reuse(
    tmp_path: Path,
) -> None:
    policy = _autonomous_policy(
        requested_aspects=("16:9", "9:16"),
    )
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)

    contexts_by_aspect: dict[str, dict[str, Path]] = {}
    for aspect in ("16:9", "9:16"):
        aspect_dir = tmp_path / aspect.replace(":", "x")
        aspect_dir.mkdir()
        paths: dict[str, Path] = {}
        for key in (
            "editorial_beat_contracts",
            "music_map",
            "cue_plan",
            "exact_event_locks",
            "sequence_optimization",
            "reuse_degradation",
            "resolved_timeline",
        ):
            path = aspect_dir / f"{key}.json"
            if key == "reuse_degradation":
                payload = AutonomousDegradationManifest(
                    policy_reference=policy.policy_reference,
                    aspect=aspect,
                    generated_at="now",
                ).model_dump(mode="json")
            else:
                payload = {"aspect": aspect}
            write_json(path, payload)
            paths[key] = path
        contexts_by_aspect[aspect] = paths

    # Model the historical bug: the 16:9 slot points at an artifact that
    # explicitly belongs to the 9:16 selected-window run.
    contexts_by_aspect["16:9"]["exact_event_locks"] = (
        contexts_by_aspect["9:16"]["exact_event_locks"]
    )

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="16:9 autonomous QA cannot consume 9:16",
    ):
        pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={"aspect": "both"},
            brief_path=tmp_path / "brief.json",
            music_path=None,
            music_lock_path=None,
            output_dir=tmp_path / "delivery",
            execution_profile="autonomous_strict",
            autonomous_policy_path=policy_path,
            autonomous_context_paths_by_aspect=contexts_by_aspect,
        )


def test_both_aspects_reserve_run_global_initial_qa_before_repair() -> None:
    # With the policy maximum of two full final-QA observations, horizontal
    # and vertical each consume one initial observation. A repair after either
    # one therefore has no legal follow-up-QA slot and must fail closed.
    assert pipeline._remaining_run_global_followup_qa_slots(
        maximum_full_final_qa_calls=2,
        completed_full_final_qa_calls=1,
        requested_initial_aspects=("16:9", "9:16"),
        started_initial_aspects={"16:9"},
    ) == 0
    assert pipeline._remaining_run_global_followup_qa_slots(
        maximum_full_final_qa_calls=2,
        completed_full_final_qa_calls=2,
        requested_initial_aspects=("16:9", "9:16"),
        started_initial_aspects={"16:9", "9:16"},
    ) == 0
    # A single-aspect run still has the one bounded repair verification pass.
    assert pipeline._remaining_run_global_followup_qa_slots(
        maximum_full_final_qa_calls=2,
        completed_full_final_qa_calls=1,
        requested_initial_aspects=("9:16",),
        started_initial_aspects={"9:16"},
    ) == 1


@pytest.mark.parametrize("same_policy", [False, True])
def test_autonomous_music_lock_always_revalidates_and_refreshes_authority(
    tmp_path: Path,
    monkeypatch,
    same_policy: bool,
) -> None:
    music = tmp_path / "music.wav"
    proposal = tmp_path / "music-map.proposal.json"
    saved_lock_path = tmp_path / "music-map.lock.v2.json"
    music.write_bytes(b"music")
    proposal.write_text("{}")
    saved_lock_path.write_text('{"lock": true}')
    music_id = f"sha256:{sha256_file(music)}"
    policy = _autonomous_policy()
    saved_lock = SimpleNamespace(
        music_id=music_id,
        proposal_path=str(proposal),
        proposal_sha256=sha256_file(proposal),
        authority=SimpleNamespace(
            policy_reference=(
                policy.policy_reference
                if same_policy
                else "sha256:" + "0" * 64
            )
        ),
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
    validated: list[object] = []
    monkeypatch.setattr(
        pipeline,
        "validate_music_map_lock_integrity",
        lambda *_args, **_kwargs: (
            validated.append((_args, _kwargs)) or parsed_proposal
        ),
    )
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
    # Once before issuing authority and once against the newly persisted lock.
    assert len(validated) == 2


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
            sequence_optimization_audited=True,
            sequence_optimization_passed=True,
            readability_passed=True,
            reuse_authorized=True,
            omissions_authorized=True,
            hard_evidence_passed=True,
        ),
        policy=policy,
    )


def _passing_autonomous_qa(
    mode: str = "autonomous_final_9x16",
) -> AutonomousFinalEditQa:
    return AutonomousFinalEditQa(
        mode=mode,
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
        qa_context_hashes_by_aspect={
            "9:16": {"editorial_beat_contracts": "e" * 64}
        },
        degradation=degradation,
        input_artifact_hashes=("sha256:" + "f" * 64,),
        final_render_sha256_by_aspect={"9:16": "a" * 64},
        final_manifest_sha256="c" * 64,
        brief_sha256="d" * 64,
        gemini_interaction_ids=("qa-observation-1",),
    )

    assert state == "delivery_eligible"
    assert authority.authority_type == "auto_policy"
    assert authority.policy_reference == policy.policy_reference
    assert authority.decision_scope == "final_delivery"
    assert authority.gemini_interaction_ids == ("qa-observation-1",)


def test_autonomous_horizontal_requires_typed_autonomous_qa_authority() -> None:
    policy = _autonomous_policy(requested_aspects=("16:9",))
    degradation = AutonomousDegradationManifest(
        policy_reference=policy.policy_reference,
        generated_at="now",
    )
    canonical_review = SimpleNamespace(
        global_review=SimpleNamespace(
            disposition="ready_for_human_review"
        )
    )

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="wrong semantic QA mode",
    ):
        pipeline.authorize_autonomous_delivery(
            policy=policy,
            deterministic_qa=_passing_deterministic_report(policy),
            qa_results={"16:9": canonical_review},
            qa_context_hashes_by_aspect={
                "16:9": {"editorial_beat_contracts": "e" * 64}
            },
            degradation=degradation,
            input_artifact_hashes=("sha256:" + "f" * 64,),
            final_render_sha256_by_aspect={"16:9": "a" * 64},
            final_manifest_sha256="c" * 64,
            brief_sha256="d" * 64,
        )

    state, _ = pipeline.authorize_autonomous_delivery(
        policy=policy,
        deterministic_qa=_passing_deterministic_report(policy),
        qa_results={
            "16:9": _passing_autonomous_qa("autonomous_final_16x9")
        },
        qa_context_hashes_by_aspect={
            "16:9": {"editorial_beat_contracts": "e" * 64}
        },
        degradation=degradation,
        input_artifact_hashes=("sha256:" + "f" * 64,),
        final_render_sha256_by_aspect={"16:9": "a" * 64},
        final_manifest_sha256="c" * 64,
        brief_sha256="d" * 64,
    )
    assert state == "delivery_eligible"


def test_autonomous_authority_rejects_cross_aspect_or_stale_render_qa() -> None:
    policy = _autonomous_policy(requested_aspects=("16:9",))
    degradation = AutonomousDegradationManifest(
        policy_reference=policy.policy_reference,
        generated_at="now",
    )
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="wrong semantic QA mode",
    ):
        pipeline.authorize_autonomous_delivery(
            policy=policy,
            deterministic_qa=_passing_deterministic_report(policy),
            qa_results={"16:9": _passing_autonomous_qa()},
            qa_context_hashes_by_aspect={
                "16:9": {"editorial_beat_contracts": "e" * 64}
            },
            degradation=degradation,
            input_artifact_hashes=("sha256:" + "f" * 64,),
            final_render_sha256_by_aspect={"16:9": "a" * 64},
            final_manifest_sha256="c" * 64,
            brief_sha256="d" * 64,
        )
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="does not bind the current final render",
    ):
        pipeline.authorize_autonomous_delivery(
            policy=policy,
            deterministic_qa=_passing_deterministic_report(policy),
            qa_results={
                "16:9": _passing_autonomous_qa(
                    "autonomous_final_16x9"
                )
            },
            qa_context_hashes_by_aspect={
                "16:9": {"editorial_beat_contracts": "e" * 64}
            },
            degradation=degradation,
            input_artifact_hashes=("sha256:" + "f" * 64,),
            final_render_sha256_by_aspect={"16:9": "9" * 64},
            final_manifest_sha256="c" * 64,
            brief_sha256="d" * 64,
        )
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="does not bind the current autonomous context",
    ):
        pipeline.authorize_autonomous_delivery(
            policy=policy,
            deterministic_qa=_passing_deterministic_report(policy),
            qa_results={
                "16:9": _passing_autonomous_qa(
                    "autonomous_final_16x9"
                )
            },
            qa_context_hashes_by_aspect={
                "16:9": {"editorial_beat_contracts": "9" * 64}
            },
            degradation=degradation,
            input_artifact_hashes=("sha256:" + "f" * 64,),
            final_render_sha256_by_aspect={"16:9": "a" * 64},
            final_manifest_sha256="c" * 64,
            brief_sha256="d" * 64,
        )
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="does not bind the current render manifest",
    ):
        pipeline.authorize_autonomous_delivery(
            policy=policy,
            deterministic_qa=_passing_deterministic_report(policy),
            qa_results={
                "16:9": _passing_autonomous_qa(
                    "autonomous_final_16x9"
                )
            },
            qa_context_hashes_by_aspect={
                "16:9": {"editorial_beat_contracts": "e" * 64}
            },
            degradation=degradation,
            input_artifact_hashes=("sha256:" + "f" * 64,),
            final_render_sha256_by_aspect={"16:9": "a" * 64},
            final_manifest_sha256="9" * 64,
            brief_sha256="d" * 64,
        )


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
        qa_context_hashes_by_aspect={
            "9:16": {"editorial_beat_contracts": "e" * 64}
        },
        degradation=degradation,
        input_artifact_hashes=("sha256:" + "f" * 64,),
        final_render_sha256_by_aspect={"9:16": "a" * 64},
        final_manifest_sha256="c" * 64,
        brief_sha256="d" * 64,
    )

    assert state == "best_effort_complete"
    assert "authorized_degradations_recorded" in authority.decision_codes


@pytest.mark.parametrize(
    ("profile", "requested_aspects", "music_supplied"),
    [
        (profile, requested_aspects, music_supplied)
        for profile in ("autonomous_strict", "autonomous_best_effort")
        for requested_aspects in (
            ("9:16",),
            ("16:9",),
            ("16:9", "9:16"),
        )
        for music_supplied in (False, True)
    ],
)
def test_autonomous_authority_mode_matrix(
    profile: str,
    requested_aspects: tuple[str, ...],
    music_supplied: bool,
) -> None:
    policy = _autonomous_policy(
        profile,
        requested_aspects=requested_aspects,
    )
    records = (
        (
            DegradationRecord(
                beat_id="optional-beat",
                action="optional_beat_omitted",
                reason_code="evidence_not_found",
            ),
        )
        if profile == "autonomous_best_effort"
        else ()
    )
    degradation = AutonomousDegradationManifest(
        policy_reference=policy.policy_reference,
        records=records,
        generated_at="now",
    )
    qa_results = {
        aspect: _passing_autonomous_qa(
            "autonomous_final_16x9"
            if aspect == "16:9"
            else "autonomous_final_9x16"
        )
        for aspect in requested_aspects
    }
    context_hashes = {
        aspect: {"editorial_beat_contracts": "e" * 64}
        for aspect in requested_aspects
    }

    state, authority = pipeline.authorize_autonomous_delivery(
        policy=policy,
        deterministic_qa=_passing_deterministic_report(policy),
        qa_results=qa_results,
        qa_context_hashes_by_aspect=context_hashes,
        degradation=degradation,
        input_artifact_hashes=("sha256:" + "f" * 64,),
        final_render_sha256_by_aspect={
            aspect: "a" * 64 for aspect in requested_aspects
        },
        final_manifest_sha256="c" * 64,
        brief_sha256="d" * 64,
        music_supplied=music_supplied,
    )

    assert state == (
        "best_effort_complete"
        if profile == "autonomous_best_effort"
        else "delivery_eligible"
    )
    assert (
        "music_sync_passed"
        if music_supplied
        else "semantic_visual_cadence_passed"
    ) in authority.decision_codes


def test_final_authority_rechecks_degradation_policy_independently() -> None:
    policy = _autonomous_policy().model_copy(
        update={
            "editorial": _autonomous_policy().editorial.model_copy(
                update={"allow_optional_beat_omission": False}
            )
        }
    )
    degradation = AutonomousDegradationManifest(
        policy_reference=policy.policy_reference,
        records=(
            DegradationRecord(
                beat_id="optional-beat",
                action="optional_beat_omitted",
                reason_code="evidence_not_found",
            ),
        ),
        generated_at="now",
    )

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="policy did not authorize",
    ):
        pipeline.authorize_autonomous_delivery(
            policy=policy,
            deterministic_qa=_passing_deterministic_report(policy),
            qa_results={"9:16": _passing_autonomous_qa()},
            qa_context_hashes_by_aspect={
                "9:16": {"editorial_beat_contracts": "e" * 64}
            },
            degradation=degradation,
            input_artifact_hashes=("sha256:" + "f" * 64,),
            final_render_sha256_by_aspect={"9:16": "a" * 64},
            final_manifest_sha256="c" * 64,
            brief_sha256="d" * 64,
        )


def _recovery_fixture_paths(
    tmp_path: Path,
    policy: AutonomousEditPolicy,
) -> dict[str, object]:
    input_render = tmp_path / "input.mp4"
    input_render.write_bytes(b"input-render")
    input_qa = tmp_path / "validated.json"
    write_json(input_qa, {"qa": "issues_observed"})
    render_manifest = tmp_path / "render-manifest.json"
    delivery_manifest = tmp_path / "delivery-manifest.json"
    media_manifest = tmp_path / "media-manifest.json"
    for path in (render_manifest, delivery_manifest, media_manifest):
        write_json(path, {"path": path.name})
    contexts: dict[str, Path] = {}
    for key in (
        "editorial_beat_contracts",
        "music_map",
        "cue_plan",
        "exact_event_locks",
        "sequence_optimization",
        "reuse_degradation",
        "resolved_timeline",
    ):
        path = tmp_path / f"{key}.json"
        payload: object = (
            []
            if key == "editorial_beat_contracts"
            else {"aspect": "9:16"}
            if key == "resolved_timeline"
            else {}
        )
        if key == "reuse_degradation":
            payload = AutonomousDegradationManifest(
                policy_reference=policy.policy_reference,
                generated_at="now",
            )
        write_json(path, payload)
        contexts[key] = path
    feature_cut._write_authorized_selected_window_cue_plan(
        contexts["cue_plan"],
        proposal={
            "contract_version": "selected-window-cue-plan-v2",
            "aspect": "9:16",
            "music_map_sha256": sha256_file(contexts["music_map"]),
            "music_supplied": False,
            "cue_timeline": "locked_source_timeline",
            "music_output_timeline_sha256": None,
            "alignments": [],
        },
        authority_inputs={
            "music_map": contexts["music_map"],
            "editorial_beat_contracts": contexts[
                "editorial_beat_contracts"
            ],
            "exact_event_locks": contexts["exact_event_locks"],
        },
        policy=policy,
    )
    evidence = DeterministicDeliveryEvidence(
        media_playable=True,
        pts_valid=True,
        unexpected_freeze_count=0,
        containment_passed=True,
        identity_passed=True,
        relation_passed=True,
        panel_same_pts_passed=True,
        relative_scale_lock_passed=True,
        cue_delta_frames={"event": 0},
        cue_tolerance_frames={"event": 2},
        cue_id_by_event={"event": "cue"},
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
        sequence_optimization_audited=True,
        sequence_optimization_passed=True,
        readability_passed=True,
        reuse_authorized=True,
        omissions_authorized=True,
        hard_evidence_passed=True,
    )
    evidence_path = tmp_path / "deterministic.json"
    write_json(evidence_path, evidence)
    return {
        "input_render_path": input_render,
        "input_qa_path": input_qa,
        "input_render_manifest_path": render_manifest,
        "input_delivery_manifest_path": delivery_manifest,
        "input_music_assembly_manifest_path": media_manifest,
        "autonomous_context_paths": contexts,
        "deterministic_delivery_evidence_path": evidence_path,
        "evidence": evidence,
    }


def _mock_feature_cut_authorities(
    tmp_path: Path,
    *,
    policy: AutonomousEditPolicy,
    picture: Path,
    render_manifest: Path,
    context_paths: dict[str, Path],
) -> tuple[Path, dict[str, str]]:
    proposal = tmp_path / "mock-presentation.proposal.json"
    write_json(
        proposal,
        {
            "contract_version": "presentation-compilation-proposal-v2",
            "aspect": "9:16",
            "final_output_path": str(picture),
            "final_output_sha256": sha256_file(picture),
            "chapters": [
                {
                    "feature_id": "mock",
                    "segment_path": str(picture),
                    "segment_sha256": sha256_file(picture),
                }
            ],
        },
    )
    presentation = feature_cut._write_policy_decision_artifact(
        tmp_path / "mock-presentation-authority.json",
        proposal_path=proposal,
        authority_inputs={
            "exact_event_locks": context_paths["exact_event_locks"]
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
    eligibility = tmp_path / "mock-delivery-eligibility.json"
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
                            "feature_id": "mock",
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
        tmp_path / "mock-feature-cut-authority.json",
        proposal_path=eligibility,
        authority_inputs={
            "delivery_eligibility": eligibility,
            "presentation_authority_9x16": presentation,
        },
        additional_input_hashes=(),
        policy=policy,
        decision_scope="feature_cut",
        aspect=None,
        deterministic_gate_results={"feature_cut": "passed"},
        decision_codes=("feature_cut_bound",),
    )
    manifest = read_json(render_manifest)
    manifest["presentation_authority_by_aspect"] = {
        "9:16": {
            "path": str(presentation),
            "sha256": sha256_file(presentation),
        }
    }
    manifest["feature_cut_authority"] = {
        "path": str(feature_authority),
        "sha256": sha256_file(feature_authority),
    }
    write_json(render_manifest, manifest)
    return feature_authority, {"9:16": str(presentation)}


def test_selected_window_cue_plan_rejects_changed_authority_input(
    tmp_path: Path,
) -> None:
    policy = _autonomous_policy()
    fixture = _recovery_fixture_paths(tmp_path, policy)
    contexts = fixture["autonomous_context_paths"]
    feature_cut.validate_authorized_selected_window_cue_plan(
        contexts["cue_plan"],
        policy=policy,
        expected_aspect="9:16",
    )

    write_json(contexts["exact_event_locks"], {"changed": True})
    with pytest.raises(ValueError, match="exact_event_locks changed"):
        feature_cut.validate_authorized_selected_window_cue_plan(
            contexts["cue_plan"],
            policy=policy,
            expected_aspect="9:16",
        )


def test_multi_aspect_preflight_rejects_cross_aspect_deterministic_evidence(
    tmp_path: Path,
) -> None:
    policy = _autonomous_policy(
        requested_aspects=("16:9", "9:16"),
    )
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    contexts_by_aspect: dict[str, dict[str, Path]] = {}
    evidence_paths: dict[str, Path] = {}
    fixture = _recovery_fixture_paths(tmp_path, policy)
    for aspect in ("16:9", "9:16"):
        aspect_dir = tmp_path / aspect.replace(":", "x")
        aspect_dir.mkdir()
        paths: dict[str, Path] = {}
        for key, source_path in fixture[
            "autonomous_context_paths"
        ].items():
            payload = read_json(source_path)
            if isinstance(payload, dict):
                payload["aspect"] = aspect
            target = aspect_dir / f"{key}.json"
            write_json(target, payload)
            paths[key] = target
        feature_cut._write_authorized_selected_window_cue_plan(
            paths["cue_plan"],
            proposal={
                "contract_version": "selected-window-cue-plan-v2",
                "aspect": aspect,
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
        contexts_by_aspect[aspect] = paths
        evidence_path = aspect_dir / "deterministic.json"
        write_json(
            evidence_path,
            fixture["evidence"].model_copy(
                update={
                    # Intentionally misbind only the horizontal slot.
                    "aspect": "9:16" if aspect == "16:9" else aspect,
                }
            ),
        )
        evidence_paths[aspect] = evidence_path

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="16:9 autonomous QA cannot consume 9:16 deterministic evidence",
    ):
        pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={"aspect": "both"},
            brief_path=tmp_path / "brief.json",
            music_path=None,
            music_lock_path=None,
            output_dir=tmp_path / "delivery",
            execution_profile="autonomous_strict",
            autonomous_policy_path=policy_path,
            autonomous_context_paths_by_aspect=contexts_by_aspect,
            deterministic_delivery_evidence_paths_by_aspect=(
                evidence_paths
            ),
        )


def test_recovery_without_verified_executor_fails_closed_with_artifact(
    tmp_path: Path,
) -> None:
    policy = _autonomous_policy()
    fixture = _recovery_fixture_paths(tmp_path, policy)
    plan = AutonomousRecoveryPlan(
        qa_passes_completed=1,
        semantic_replans_used=0,
        actions=(
            AutonomousRepairAction(
                issue_id="issue-1",
                segment_id="s2",
                beat_id="beat-2",
                action="hold",
            ),
        ),
        requires_another_qa=True,
        outcome="repair",
        decision_codes=("deterministic_repairs_prioritized",),
    )
    ledger = BudgetLedger(
        max_cost_usd=1.25,
        max_interactions=25,
        reserved_recovery_fraction=0.20,
    )

    execution, repaired = pipeline._execute_autonomous_recovery_plan(
        plan=plan,
        policy=policy,
        **{key: value for key, value in fixture.items() if key != "evidence"},
        segment_contract=(
            {"segment_id": "s1"},
            {"segment_id": "s2"},
        ),
        output_dir=tmp_path / "recovery",
        budget_ledger=ledger,
        executor=None,
    )

    assert repaired is None
    assert execution.status == "unavailable"
    assert execution.reason_code == "no_verified_autonomous_repair_executor"
    saved = read_json(
        tmp_path / "recovery" / "recovery-execution.json"
    )
    assert saved["status"] == "unavailable"
    assert saved["changed_segment_ids"] == []
    assert ledger.report()["committed_interactions"] == 0


def test_recovery_rejects_copied_all_true_evidence_without_causal_binding(
    tmp_path: Path,
) -> None:
    policy = _autonomous_policy()
    fixture = _recovery_fixture_paths(tmp_path, policy)
    plan = AutonomousRecoveryPlan(
        qa_passes_completed=1,
        semantic_replans_used=0,
        actions=(
            AutonomousRepairAction(
                issue_id="issue-1",
                segment_id="s2",
                beat_id="beat-2",
                action="hold",
            ),
        ),
        requires_another_qa=True,
        outcome="repair",
        decision_codes=("deterministic_repairs_prioritized",),
    )
    ledger = BudgetLedger(
        max_cost_usd=1.25,
        max_interactions=25,
        reserved_recovery_fraction=0.20,
    )

    def executor(**kwargs):
        executor_dir = kwargs["output_dir"]
        executor_dir.mkdir(parents=True)
        render = executor_dir / "repaired.mp4"
        render.write_bytes(b"repaired-render")
        render_manifest = executor_dir / "render-manifest.json"
        delivery_manifest = executor_dir / "delivery-manifest.json"
        media_manifest = executor_dir / "media-manifest.json"
        for path in (render_manifest, delivery_manifest, media_manifest):
            write_json(path, {"repaired": True})
        evidence_path = executor_dir / "deterministic.json"
        write_json(evidence_path, fixture["evidence"])
        return {
            "render_path": render,
            "render_manifest_path": render_manifest,
            "delivery_manifest_path": delivery_manifest,
            "music_assembly_manifest_path": media_manifest,
            "autonomous_context_paths": fixture[
                "autonomous_context_paths"
            ],
            "deterministic_delivery_evidence_path": evidence_path,
            "changed_segment_ids": ("s2",),
            "reused_segment_ids": ("s1",),
            "semantic_replan_interaction_ids": (),
        }

    execution, repaired = pipeline._execute_autonomous_recovery_plan(
        plan=plan,
        policy=policy,
        **{key: value for key, value in fixture.items() if key != "evidence"},
        segment_contract=(
            {"segment_id": "s1"},
            {"segment_id": "s2"},
        ),
        output_dir=tmp_path / "recovery",
        budget_ledger=ledger,
        executor=executor,
    )

    assert execution.status == "unavailable"
    assert execution.reason_code == "post_repair_deterministic_evidence_invalid"
    assert repaired is None
    assert ledger.report()["committed_interactions"] == 0


def test_deterministic_evidence_for_render_a_cannot_authorize_render_b(
    tmp_path: Path,
) -> None:
    policy = _autonomous_policy()
    fixture = _recovery_fixture_paths(tmp_path, policy)
    evidence_path = fixture["deterministic_delivery_evidence_path"]
    segment = tmp_path / "segment.mp4"
    segment.write_bytes(b"segment-a")
    render_manifest = tmp_path / "render-manifest.json"
    write_json(
        render_manifest,
        {
            "vertical": {
                "chapters": [
                    {
                        "segment_id": "s1",
                        "segment_path": str(segment),
                    }
                ]
            },
            "deterministic_delivery_evidence": {
                "path": str(evidence_path.resolve()),
                "sha256": sha256_file(evidence_path),
            },
        },
    )
    render_a = tmp_path / "render-a.mp4"
    render_b = tmp_path / "render-b.mp4"
    render_a.write_bytes(b"render-a")
    render_b.write_bytes(b"render-b")
    delivery_manifest = tmp_path / "delivery.json"
    music_manifest = tmp_path / "music.json"
    write_json(delivery_manifest, {"render": "a"})
    write_json(music_manifest, {"music": "a"})

    pipeline._feature_manifest_attests_deterministic_evidence(
        render_manifest_path=render_manifest,
        aspect="9:16",
        evidence_path=evidence_path,
    )
    bound, _ = pipeline._bind_deterministic_evidence_to_delivery(
        evidence=fixture["evidence"],
        aspect="9:16",
        policy=policy,
        render_path=render_a,
        render_manifest_path=render_manifest,
        delivery_manifest_path=delivery_manifest,
        music_assembly_manifest_path=music_manifest,
        autonomous_context_paths=fixture["autonomous_context_paths"],
        output_path=tmp_path / "bound-evidence.json",
    )
    assert bound.causal_binding is not None
    assert bound.causal_binding.render_sha256 == sha256_file(render_a)
    with pytest.raises(
        ValueError,
        match="causal binding does not match",
    ):
        pipeline.validate_deterministic_evidence_causal_binding(
            bound,
            render_sha256=sha256_file(render_b),
            render_manifest_sha256=sha256_file(render_manifest),
            policy_reference=policy.policy_reference,
            context_hashes={
                key: sha256_file(path)
                for key, path in fixture[
                    "autonomous_context_paths"
                ].items()
            },
            delivery_manifest_sha256=sha256_file(delivery_manifest),
            music_assembly_manifest_sha256=sha256_file(music_manifest),
            segment_content_hashes={"s1": sha256_file(segment)},
        )


def test_autonomous_delivery_default_feature_repair_runs_at_most_two_qa_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = _autonomous_policy()
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    fixture = _recovery_fixture_paths(tmp_path, policy)
    contexts = fixture["autonomous_context_paths"]
    evidence_path = fixture["deterministic_delivery_evidence_path"]
    brief = tmp_path / "brief.json"
    music = tmp_path / "music.wav"
    music_lock = tmp_path / "music-lock.json"
    picture = tmp_path / "picture.mp4"
    render_manifest = tmp_path / "render-manifest.json"
    for path, payload in (
        (brief, b"{}"),
        (music, b"music"),
        (music_lock, b"{}"),
        (picture, b"picture"),
        (render_manifest, b"{}"),
    ):
        path.write_bytes(payload)
    initial_segments = {}
    for segment_id in ("s1", "s2"):
        segment = tmp_path / f"{segment_id}.mp4"
        segment.write_bytes(segment_id.encode())
        initial_segments[segment_id] = segment
    write_json(
        render_manifest,
        {
            "vertical": {
                "chapters": [
                    {
                        "segment_id": segment_id,
                        "segment_path": str(segment),
                    }
                    for segment_id, segment in initial_segments.items()
                ]
            },
            "deterministic_delivery_evidence": {
                "path": str(evidence_path.resolve()),
                "sha256": sha256_file(evidence_path),
            },
            },
        )
    feature_authority, presentation_authorities = (
        _mock_feature_cut_authorities(
            tmp_path,
            policy=policy,
            picture=picture,
            render_manifest=render_manifest,
            context_paths=contexts,
        )
    )

    monkeypatch.setattr(
        pipeline,
        "_bind_music_lock_to_autonomous_policy",
        lambda **_kwargs: music_lock,
    )
    monkeypatch.setattr(
        pipeline,
        "run_feature_cut_experiment",
        lambda **_kwargs: {
            "ready_for_human_review": False,
            "media_rendered": True,
            "run_state": "autonomous_ready",
            "horizontal_output": None,
            "vertical_output": str(picture),
            "manifest_path": str(render_manifest),
            "feature_cut_authority_path": str(feature_authority),
            "presentation_authority_paths_by_aspect": (
                presentation_authorities
            ),
            "autonomous_context_paths": {
                key: str(path) for key, path in contexts.items()
            },
            "deterministic_delivery_evidence_path": str(evidence_path),
        },
    )
    fake_lock = SimpleNamespace(music_id=f"sha256:{sha256_file(music)}")
    monkeypatch.setattr(
        pipeline.MusicMapLock,
        "model_validate",
        lambda _payload: fake_lock,
    )
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda _path: SimpleNamespace(duration_ms=60_000),
    )
    monkeypatch.setattr(
        pipeline,
        "plan_single_interval_music_assembly",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        pipeline,
        "write_music_assembly_artifacts",
        lambda *_args, **_kwargs: None,
    )

    def fake_music_render(_source, _plan, output, output_dir):
        del output_dir
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"music-render")
        manifest = output.parent / "music-render.json"
        write_json(manifest, {})
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
        output.write_bytes(
            b"second-final"
            if "repaired" in output.name
            else b"first-final"
        )
        write_json(manifest, {})
        return SimpleNamespace(output_path=output, manifest_path=manifest)

    monkeypatch.setattr(
        pipeline,
        "assemble_music_only_delivery",
        fake_delivery,
    )

    prepared_count = 0

    def fake_prepare(**kwargs):
        nonlocal prepared_count
        prepared_count += 1
        proxy = tmp_path / f"proxy-{prepared_count}.mp4"
        proxy.write_bytes(f"proxy-{prepared_count}".encode())
        return SimpleNamespace(
            render_path=kwargs["render_path"],
            proxy_path=proxy,
            input_hashes={
                "proxy_sha256": sha256_file(proxy),
            },
            autonomous_context_hashes={"cue_plan": "e" * 64},
            manifest_sha256=sha256_file(kwargs["manifest_path"]),
            brief_sha256=sha256_file(kwargs["brief_path"]),
            segment_contract=(
                {"segment_id": "s1"},
                {"segment_id": "s2"},
            ),
            mode=kwargs["mode"],
        )

    monkeypatch.setattr(pipeline, "prepare_final_edit_qa", fake_prepare)
    first_qa = AutonomousFinalEditQa(
        mode="autonomous_final_9x16",
        render_sha256="a" * 64,
        proxy_sha256="b" * 64,
        manifest_sha256="c" * 64,
        brief_sha256="d" * 64,
        context_hashes={"cue_plan": "e" * 64},
        issues=[
            AutonomousQaIssue(
                issue_id="issue-1",
                issue_type="unmotivated_motion",
                severity="high",
                segment_id="s2",
                beat_id="beat-2",
                observation="Motion has no visible motivation.",
                evidence_modality="visual",
                repair_class="hold",
            )
        ],
        opening_observation="Opening is clear.",
        ending_observation="Ending is clear.",
        sequence_observation="One local motion issue remains.",
        qa_observation_status="issues_observed",
        limitations=[],
    )
    second_qa = _passing_autonomous_qa().model_copy(
        update={"context_hashes": {"cue_plan": "e" * 64}}
    )
    recovery_flags: list[bool] = []

    def fake_execute(**kwargs):
        recovery_flags.append(kwargs["recovery_call"])
        run_dir = tmp_path / f"qa-run-{len(recovery_flags)}"
        run_dir.mkdir()
        result = first_qa if len(recovery_flags) == 1 else second_qa
        result = result.model_copy(
            update={
                "render_sha256": sha256_file(
                    kwargs["prepared"].render_path
                ),
                "manifest_sha256": kwargs["prepared"].manifest_sha256,
                "brief_sha256": kwargs["prepared"].brief_sha256,
                "context_hashes": (
                    kwargs["prepared"].autonomous_context_hashes
                ),
            }
        )
        for filename, payload in (
            ("input_hashes.json", {}),
            ("schema_validation.json", {"ok": True}),
            ("validated.json", result.model_dump(mode="json")),
        ):
            write_json(run_dir / filename, payload)
        return SimpleNamespace(
            result=result,
            run_dir=run_dir,
            cache_hit=False,
        )

    monkeypatch.setattr(pipeline, "execute_final_edit_qa", fake_execute)

    class FakeClient:
        client = object()

        def __init__(self, *, model_id: str) -> None:
            assert model_id

        def ensure_video_upload(self, path: Path, _artifact_dir: Path):
            return {"uri": str(path), "mime_type": "video/mp4"}, False

        def close(self) -> None:
            return None

    monkeypatch.setattr(pipeline, "GeminiLabClient", FakeClient)

    def fake_compile_repair_request(**kwargs):
        compile_dir = kwargs["output_dir"]
        compile_dir.mkdir(parents=True)
        request_path = compile_dir / "compiled-repair-request.json"
        write_json(request_path, {"compiled": True})
        return {
            "status": "compiled",
            "path": str(request_path),
            "sha256": sha256_file(request_path),
            "blockers": [],
        }

    monkeypatch.setattr(
        pipeline,
        "compile_repair_request",
        fake_compile_repair_request,
    )

    def fake_render_changed_segments(**kwargs):
        render_dir = kwargs["output_dir"]
        render_dir.mkdir(parents=True)
        repaired_picture = render_dir / "repaired-picture.mp4"
        repaired_picture.write_bytes(b"repaired-picture")
        repaired_render_manifest = render_dir / "render-manifest.json"
        repaired_segments = {}
        for segment_id in ("s1", "s2"):
            segment = render_dir / f"{segment_id}.mp4"
            segment.write_bytes(f"repaired-{segment_id}".encode())
            repaired_segments[segment_id] = segment
        write_json(
            repaired_render_manifest,
            {
                "repaired": True,
                "vertical": {
                    "chapters": [
                        {
                            "segment_id": segment_id,
                            "segment_path": str(segment),
                        }
                        for segment_id, segment in repaired_segments.items()
                    ]
                },
            },
        )
        repaired_evidence = render_dir / "deterministic.json"
        write_json(repaired_evidence, fixture["evidence"])
        return {
            "picture_path": repaired_picture,
            "render_manifest_path": repaired_render_manifest,
            "autonomous_context_paths": contexts,
            "deterministic_delivery_evidence_path": repaired_evidence,
            "changed_segment_ids": ("s2",),
            "reused_segment_ids": ("s1",),
            "semantic_replan_interaction_ids": (),
        }

    monkeypatch.setattr(
        pipeline,
        "render_changed_segments_and_concat",
        fake_render_changed_segments,
    )

    result = pipeline.run_feature_delivery_pipeline(
        feature_cut_kwargs={
            "catalog_path": tmp_path / "catalog.json",
            "aspect": "9x16",
            "reuse_feature_plan": True,
        },
        brief_path=brief,
        music_path=music,
        music_lock_path=music_lock,
        output_dir=tmp_path / "delivery",
        execution_profile="autonomous_strict",
        autonomous_policy_path=policy_path,
        autonomous_context_paths=contexts,
        deterministic_delivery_evidence_path=evidence_path,
        editorial_beat_contracts_path=contexts[
            "editorial_beat_contracts"
        ],
    )

    assert recovery_flags == [False, True]
    assert result["delivery_eligible"] is True
    assert result["aspects"]["vertical"]["qa_pass_count"] == 2
    recovery = result["aspects"]["vertical"]["autonomous_recovery"]
    assert recovery["first_status"] == "executed"
    assert recovery["second_status"] == "not_required"
    assert Path(result["aspects"]["vertical"]["final_output"]).read_bytes() == (
        b"second-final"
    )


def test_fresh_autonomous_planning_uses_shortlist_then_direct_video_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "catalog.json"
    brief = tmp_path / "brief.json"
    policy_path = tmp_path / "policy.json"
    contracts = tmp_path / "contracts.json"
    music = tmp_path / "music.wav"
    library = tmp_path / "clip-cards"
    for path in (catalog, brief, policy_path, contracts):
        write_json(path, {})
    write_json(policy_path, _autonomous_policy())
    music.write_bytes(b"music")
    library.mkdir()
    monkeypatch.setattr(
        pipeline,
        "_resolve_prepared_clip_card_library",
        lambda **_kwargs: (
            library,
            SimpleNamespace(),
            (),
            1_000,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_refresh_stale_clip_cards",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        pipeline,
        "_archive_stale_clip_card_supplements",
        lambda **_kwargs: (),
    )
    stages: list[tuple[str, list[str]]] = []

    def fake_stage(**kwargs):
        stage = kwargs["stage"]
        command = kwargs["command"]
        stages.append((stage, command))
        if stage == "autonomous_clip_card_shortlist":
            write_json(
                tmp_path
                / "delivery"
                / "retrieval"
                / "feature-shortlist.json",
                {},
            )
        else:
            plan_dir = (
                tmp_path / "delivery" / "picture" / "gemini-plan"
            )
            for name in (
                "feature_edit_plan.json",
                "selected-clip-card-evidence.json",
                "feature-plan.external-projection.json",
            ):
                write_json(plan_dir / name, {})
        return {"request_count": 1}

    monkeypatch.setattr(
        pipeline,
        "_run_budgeted_planning_stage",
        fake_stage,
    )
    ledger = BudgetLedger(
        max_cost_usd=1.25,
        max_interactions=25,
    )

    result = pipeline._prepare_fresh_autonomous_direct_plan(
        catalog_path=catalog,
        brief_path=brief,
        music_path=music,
        music_duration_ms=90_000,
        policy_path=policy_path,
        editorial_contracts_path=contracts,
        prepared_library_path=library,
        output_dir=tmp_path / "delivery",
        budget_ledger=ledger,
    )

    assert [stage for stage, _ in stages] == [
        "autonomous_clip_card_shortlist",
        "autonomous_direct_video_edit_plan",
    ]
    direct_command = stages[1][1]
    assert "--candidate-video-evidence" in direct_command
    assert "--repair-attempts" in direct_command
    assert direct_command[direct_command.index("--repair-attempts") + 1] == "0"
    assert "--music" in direct_command
    assert result["plan_dir"].endswith("picture/gemini-plan")


def test_fresh_autonomous_planning_budgets_text_repair_after_typed_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "catalog.json"
    brief = tmp_path / "brief.json"
    policy_path = tmp_path / "policy.json"
    contracts = tmp_path / "contracts.json"
    music = tmp_path / "music.wav"
    library = tmp_path / "clip-cards"
    for path in (catalog, brief, policy_path, contracts):
        write_json(path, {})
    write_json(policy_path, _autonomous_policy())
    music.write_bytes(b"music")
    library.mkdir()
    monkeypatch.setattr(
        pipeline,
        "_resolve_prepared_clip_card_library",
        lambda **_kwargs: (library, SimpleNamespace(), (), 1_000),
    )
    monkeypatch.setattr(pipeline, "_refresh_stale_clip_cards", lambda **_kwargs: ())
    monkeypatch.setattr(
        pipeline,
        "_archive_stale_clip_card_supplements",
        lambda **_kwargs: (),
    )
    stages: list[tuple[str, list[str], dict]] = []

    def fake_stage(**kwargs):
        stage = kwargs["stage"]
        command = kwargs["command"]
        stages.append((stage, command, kwargs))
        delivery = tmp_path / "delivery"
        if stage == "autonomous_clip_card_shortlist":
            write_json(delivery / "retrieval" / "feature-shortlist.json", {})
        elif stage == "autonomous_direct_video_edit_plan":
            write_json(
                delivery
                / "picture"
                / "gemini-plan"
                / "clip-card-feature-plan.attempt-01.schema-validation.json",
                {"ok": False, "error": "typed contract failed"},
            )
        else:
            for name in (
                "feature_edit_plan.json",
                "selected-clip-card-evidence.json",
                "feature-plan.external-projection.json",
            ):
                write_json(delivery / "picture" / "gemini-plan" / name, {})
        return {"request_count": 1}

    monkeypatch.setattr(pipeline, "_run_budgeted_planning_stage", fake_stage)

    result = pipeline._prepare_fresh_autonomous_direct_plan(
        catalog_path=catalog,
        brief_path=brief,
        music_path=music,
        music_duration_ms=90_000,
        policy_path=policy_path,
        editorial_contracts_path=contracts,
        prepared_library_path=library,
        output_dir=tmp_path / "delivery",
        budget_ledger=BudgetLedger(max_cost_usd=1.25, max_interactions=25),
    )

    assert [stage for stage, _command, _kwargs in stages] == [
        "autonomous_clip_card_shortlist",
        "autonomous_direct_video_edit_plan",
        "autonomous_direct_video_edit_plan_text_only_repair",
    ]
    initial_kwargs = stages[1][2]
    assert initial_kwargs["raise_on_subprocess_error"] is False
    repair_command = stages[2][1]
    assert "--resume-failed-plan" in repair_command
    assert result["planning_usage"]["request_count"] == 2


def test_prepared_clip_card_stale_model_blocks_before_paid_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    digest = "a" * 64
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, {})
    library = tmp_path / "clip-cards"
    clip_root = library / "clips" / digest[:16]
    card_dir = clip_root / "gemini" / "clip-card"
    write_json(card_dir / "clip_card.json", {})
    (clip_root / "analysis-proxy.mp4").write_bytes(b"proxy")
    clip = SimpleNamespace(
        sha256=digest,
        duration_ms=10_000,
    )
    card = SimpleNamespace(
        source_asset_id=f"sha256:{digest}",
        proxy_asset_id="sha256:" + "b" * 64,
        duration_ms=10_000,
        model_provenance=SimpleNamespace(model_id="stale-model"),
    )
    monkeypatch.setattr(
        pipeline,
        "RushesCatalog",
        SimpleNamespace(
            model_validate=lambda _payload: SimpleNamespace(clips=(clip,))
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "FullClipCard",
        SimpleNamespace(model_validate=lambda _payload: card),
    )
    monkeypatch.setattr(
        pipeline,
        "gemini_response_schema",
        lambda _model: {},
    )
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda _path: SimpleNamespace(
            asset_id=card.proxy_asset_id,
            duration_ms=10_000,
        ),
    )

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="model lineage is stale",
    ):
        pipeline._resolve_prepared_clip_card_library(
            catalog_path=catalog_path,
            prepared_library_path=library,
        )


def test_stale_clip_card_refresh_archives_prior_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    digest = "a" * 64
    clip = SimpleNamespace(
        sha256=digest,
        duration_ms=10_000,
        path=str(source),
    )
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, {})
    library = tmp_path / "clip-cards"
    old_root = library / "clips" / digest[:16]
    old_root.mkdir(parents=True)
    (old_root / "old-lineage.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "RushesCatalog",
        SimpleNamespace(
            model_validate=lambda _payload: SimpleNamespace(clips=(clip,))
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepared_clip_card_library_root",
        lambda **_kwargs: library,
    )
    monkeypatch.setattr(
        pipeline,
        "_clip_card_entry_stale_reason",
        lambda *, clip, clip_root: (
            None if ".refresh" in clip_root.parts else "stale_prompt"
        ),
    )

    def fake_stage(**kwargs):
        refresh_root = kwargs["stage_dir"]
        write_json(
            refresh_root
            / "gemini"
            / "clip-card"
            / "clip_card.raw_interaction.json",
            {"model": pipeline.MODEL_ID, "usage": {}},
        )
        write_json(
            refresh_root / "orchestration.json",
            {"usage": {"request_count": 1}},
        )
        return {"request_count": 1}

    monkeypatch.setattr(
        pipeline,
        "_run_budgeted_planning_stage",
        fake_stage,
    )
    records = pipeline._refresh_stale_clip_cards(
        catalog_path=catalog_path,
        prepared_library_path=library,
        output_dir=tmp_path / "delivery",
        max_cold_ingest_cost_usd=1.25,
    )
    assert records[0]["reason_code"] == "stale_prompt"
    assert (old_root / "orchestration.json").is_file()
    archived = Path(records[0]["archived_path"])
    assert (archived / "old-lineage.txt").read_text(encoding="utf-8") == "old"
    assert (
        tmp_path
        / "delivery"
        / "cold-ingest"
        / digest[:16]
        / "refresh-record.json"
    ).is_file()


def test_stale_clip_card_refresh_requires_explicit_cold_budget_before_paid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    digest = "c" * 64
    clip = SimpleNamespace(
        sha256=digest,
        duration_ms=10_000,
        path=str(source),
    )
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, {})
    library = tmp_path / "clip-cards"
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "RushesCatalog",
        SimpleNamespace(
            model_validate=lambda _payload: SimpleNamespace(clips=(clip,))
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepared_clip_card_library_root",
        lambda **_kwargs: library,
    )
    monkeypatch.setattr(
        pipeline,
        "_clip_card_entry_stale_reason",
        lambda **_kwargs: "missing",
    )
    monkeypatch.setattr(
        pipeline,
        "_run_budgeted_planning_stage",
        lambda **_kwargs: calls.append("paid"),
    )

    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="max_cold_ingest_cost_usd",
    ):
        pipeline._refresh_stale_clip_cards(
            catalog_path=catalog_path,
            prepared_library_path=library,
            output_dir=tmp_path / "delivery",
            max_cold_ingest_cost_usd=None,
        )

    assert calls == []


def test_planning_subprocess_environment_forwards_workspace_gemini_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh children receive configuration without exposing it in artifacts."""

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "OTHER=value\nGEMINI_API_KEY='test-only-key'\n",
        encoding="utf-8",
    )

    environment = pipeline._planning_subprocess_environment(project_root=tmp_path)

    assert environment["GEMINI_API_KEY"] == "test-only-key"
    assert "OTHER" not in environment


def test_cold_refresh_resume_adopts_failed_dispatch_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    digest = "d" * 64
    clip = SimpleNamespace(
        sha256=digest,
        duration_ms=10_000,
        path=str(source),
    )
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, {})
    library = tmp_path / "clip-cards"
    output_dir = tmp_path / "delivery"
    write_json(
        output_dir
        / "cold-ingest"
        / digest[:16]
        / "failed-dispatch-record.json",
        {
            "charged_interaction": True,
            "usage": {
                "request_count": 0,
                "usage_status": "dispatch_recorded_usage_unavailable",
                "total_input_tokens": 10_000,
                "total_cached_input_tokens": 0,
                "total_output_tokens": 4_096,
                "total_thought_tokens": 1_024,
            },
        },
    )
    monkeypatch.setattr(
        pipeline,
        "RushesCatalog",
        SimpleNamespace(
            model_validate=lambda _payload: SimpleNamespace(clips=(clip,))
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepared_clip_card_library_root",
        lambda **_kwargs: library,
    )
    monkeypatch.setattr(
        pipeline,
        "_clip_card_entry_stale_reason",
        lambda **_kwargs: "missing",
    )

    def assert_no_regained_dispatch(**kwargs):
        estimate = estimate_paid_call(
            stage="autonomous_base_clip_card_refresh",
            model_id=pipeline.MODEL_ID,
            media_duration_ms=10_000,
            media_resolution="low",
            text_input_tokens=1_000,
            max_output_tokens=256,
            thinking_level="minimal",
        )
        kwargs["budget_ledger"].reserve(estimate)

    monkeypatch.setattr(
        pipeline,
        "_run_budgeted_planning_stage",
        assert_no_regained_dispatch,
    )

    with pytest.raises(BudgetExceeded, match="interaction reserve"):
        pipeline._refresh_stale_clip_cards(
            catalog_path=catalog_path,
            prepared_library_path=library,
            output_dir=output_dir,
            max_cold_ingest_cost_usd=1.25,
        )


def test_direct_plan_reuse_requires_exact_current_shortlist_binding(
    tmp_path: Path,
) -> None:
    shortlist = tmp_path / "retrieval" / "feature-shortlist.json"
    plan_dir = tmp_path / "picture" / "gemini-plan"
    write_json(shortlist, {"candidate_ids": ["candidate-a"]})
    record = plan_dir / "feature-plan-projections" / "projection.json"
    write_json(
        record,
        {
            "source_artifacts": [
                {
                    "role": "feature_shortlist",
                    "path": str(shortlist.resolve()),
                    "sha256": sha256_file(shortlist),
                }
            ]
        },
    )
    write_json(
        plan_dir / "feature-plan.external-projection.json",
        {
            "record_path": str(record.relative_to(plan_dir)),
            "record_sha256": sha256_file(record),
        },
    )

    assert pipeline._direct_plan_binds_current_shortlist(
        plan_dir=plan_dir,
        shortlist_path=shortlist,
    )

    write_json(shortlist, {"candidate_ids": ["candidate-b"]})
    assert not pipeline._direct_plan_binds_current_shortlist(
        plan_dir=plan_dir,
        shortlist_path=shortlist,
    )


def test_stale_planning_stage_moves_out_of_active_namespace(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "delivery"
    stage_dir = output_dir / "retrieval"
    artifact = stage_dir / "feature-shortlist.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("stale", encoding="utf-8")

    archived = pipeline._archive_stale_planning_stage(
        stage_dir,
        output_dir=output_dir,
        reason_code="clip_card_lineage_changed",
    )

    assert archived is not None
    assert not stage_dir.exists()
    assert (archived / artifact.name).read_text(encoding="utf-8") == "stale"
    record = read_json(archived / "archive-record.json")
    assert record["reason_code"] == "clip_card_lineage_changed"


def test_review_delivery_api_still_rejects_missing_music(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="review feature-delivery requires music",
    ):
        pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={},
            brief_path=tmp_path / "unused-brief.json",
            music_path=None,
            music_lock_path=None,
            output_dir=tmp_path / "delivery",
            execution_profile="production_review",
        )
