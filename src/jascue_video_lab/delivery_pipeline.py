from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .autonomous_policy import (
    AutonomousDegradationManifest,
    AutonomousEditPolicy,
    AutonomousExecutionProfile,
    DecisionAuthorityV2,
    authorize_decision,
)
from .billing import BudgetLedger, summarize_usage_and_list_price
from .feature_cut import run_feature_cut_experiment
from .final_delivery import assemble_music_only_delivery
from .final_edit_qa import (
    AutonomousFinalEditQa,
    DeterministicDeliveryEvidence,
    DeterministicDeliveryQaReport,
    execute_final_edit_qa,
    prepare_final_edit_qa,
    run_deterministic_delivery_qa,
)
from .gemini import GeminiLabClient, MODEL_ID
from .media import probe_video, sha256_file
from .music import (
    MusicMapLock,
    MusicMapProposal,
    lock_music_map_with_auto_policy,
)
from .music_assembly import (
    MusicAssemblyError,
    plan_contiguous_reviewed_music_edit_v2,
    plan_single_interval_music_assembly,
    render_reviewed_music_edit_v2,
    render_single_interval_music_assembly,
    write_music_assembly_artifacts,
)
from .models import (
    FeatureCutExecutionProfile,
    MusicAssemblyPlan,
    MusicEditPlanV2,
)
from .storage import read_json, utc_now, write_json


class DeliveryPipelineBlocked(RuntimeError):
    """The pipeline preserved review artifacts but cannot continue safely."""


def _bind_music_lock_to_autonomous_policy(
    *,
    music_path: Path,
    music_lock_path: Path,
    policy: AutonomousEditPolicy,
    output_dir: Path,
) -> Path:
    """Reuse deterministic analysis while issuing fresh run-policy authority."""

    resolved_music = music_path.expanduser().resolve(strict=True)
    resolved_lock = music_lock_path.expanduser().resolve(strict=True)
    saved_lock = MusicMapLock.model_validate(read_json(resolved_lock))
    music_digest = sha256_file(resolved_music)
    if saved_lock.music_id != f"sha256:{music_digest}":
        raise DeliveryPipelineBlocked(
            "MusicMap lock does not bind the supplied soundtrack"
        )
    if (
        saved_lock.authority is not None
        and saved_lock.authority.policy_reference == policy.policy_reference
    ):
        return resolved_lock
    proposal_path = Path(saved_lock.proposal_path).expanduser().resolve(
        strict=True
    )
    if sha256_file(proposal_path) != saved_lock.proposal_sha256:
        raise DeliveryPipelineBlocked(
            "cannot refresh MusicMap authority because its proposal changed"
        )
    proposal = MusicMapProposal.model_validate(read_json(proposal_path))
    if proposal.music_id != saved_lock.music_id:
        raise DeliveryPipelineBlocked(
            "cannot refresh MusicMap authority because source identity changed"
        )
    authority = authorize_decision(
        policy,
        decision_scope="music_map",
        input_artifact_hashes=(
            f"sha256:{sha256_file(proposal_path)}",
            f"sha256:{music_digest}",
            f"sha256:{sha256_file(resolved_lock)}",
        ),
        deterministic_gate_results={
            "proposal_integrity": "passed",
            "music_source_hash": "passed",
            "tempo_and_meter_resolved": "passed",
        },
        decision_codes=(
            "music_source_bound",
            "deterministic_music_analysis_reused",
            "authority_refreshed_for_current_policy",
        ),
    )
    refreshed = lock_music_map_with_auto_policy(
        proposal,
        proposal_path=proposal_path,
        authority=authority,
        policy=policy,
        bpm=saved_lock.bpm,
        first_downbeat_sample=saved_lock.first_downbeat_sample,
        meter=saved_lock.meter,
    )
    refreshed_path = output_dir / "music" / "music-map.lock.v2.json"
    write_json(refreshed_path, refreshed)
    return refreshed_path.resolve(strict=True)


def _write_status(
    path: Path,
    *,
    stage: str,
    terminal: bool,
    state: str,
    error: BaseException | None = None,
    outputs: Mapping[str, Any] | None = None,
    delivery_eligible: bool = False,
) -> None:
    write_json(
        path,
        {
            "contract_version": "feature-delivery-run-status-v1",
            "stage": stage,
            "terminal": terminal,
            "state": state,
            "delivery_eligible": delivery_eligible,
            "error": (
                None
                if error is None
                else {"type": type(error).__name__, "message": str(error)}
            ),
            "outputs": dict(outputs or {}),
            "updated_at": utc_now(),
        },
    )


def _qa_disposition(execution: Any) -> str:
    review = execution.result.global_review
    disposition = getattr(review, "disposition", None)
    if not isinstance(disposition, str):
        raise ValueError("FinalEditQA omitted its typed global disposition")
    return disposition


def _autonomous_semantic_qa_passed(result: Any) -> bool:
    if isinstance(result, AutonomousFinalEditQa):
        return (
            result.qa_observation_status == "no_blocking_observation"
            and not result.issues
        )
    review = getattr(result, "global_review", None)
    return getattr(review, "disposition", None) == "ready_for_human_review"


def authorize_autonomous_delivery(
    *,
    policy: AutonomousEditPolicy,
    deterministic_qa: DeterministicDeliveryQaReport,
    qa_results: Mapping[str, Any],
    degradation: AutonomousDegradationManifest,
    input_artifact_hashes: tuple[str, ...],
    gemini_interaction_ids: tuple[str, ...] = (),
) -> tuple[str, DecisionAuthorityV2]:
    """Grant delivery only from local gates bound to immutable artifacts."""

    if not deterministic_qa.passed:
        raise DeliveryPipelineBlocked(
            "deterministic autonomous delivery gates did not pass"
        )
    if set(qa_results) != set(policy.requested_aspects):
        raise DeliveryPipelineBlocked(
            "semantic QA results do not cover every requested aspect"
        )
    if not all(
        _autonomous_semantic_qa_passed(result)
        for result in qa_results.values()
    ):
        raise DeliveryPipelineBlocked(
            "final semantic QA reported a blocking observation"
        )
    if degradation.policy_reference != policy.policy_reference:
        raise DeliveryPipelineBlocked(
            "degradation manifest is not bound to the autonomous policy"
        )
    gate_results = {
        **{
            f"deterministic_{name}": "passed"
            for name, status in deterministic_qa.gate_results.items()
            if status == "passed"
        },
        "semantic_final_qa": "passed",
        "hard_evidence_omission_forbidden": "passed",
        "policy_binding": "passed",
    }
    decision_codes = [
        "hard_evidence_passed",
        "music_sync_passed",
        "geometry_passed",
        "final_qa_passed",
    ]
    if degradation.records:
        decision_codes.append("authorized_degradations_recorded")
    authority = authorize_decision(
        policy,
        decision_scope="final_delivery",
        input_artifact_hashes=input_artifact_hashes,
        deterministic_gate_results=gate_results,
        decision_codes=tuple(decision_codes),
        gemini_interaction_ids=gemini_interaction_ids,
    )
    state = (
        "best_effort_complete"
        if (
            policy.execution_profile
            == AutonomousExecutionProfile.BEST_EFFORT
            and degradation.records
        )
        else "delivery_eligible"
    )
    return state, authority


def _picture_video_duration_ms(path: Path) -> int:
    """Return authoritative picture duration from the video stream timeline."""

    media = probe_video(path)
    video = getattr(media, "video", None)
    duration_ts = getattr(video, "duration_ts", None)
    if duration_ts is None:
        return media.duration_ms
    time_base = Fraction(
        video.time_base.numerator,
        video.time_base.denominator,
    )
    return round(duration_ts * time_base * 1000)


def _load_reusable_picture_result(
    picture_dir: Path,
    *,
    catalog_path: Path,
    brief_path: Path,
    music_path: Path,
) -> dict[str, Any]:
    """Resume downstream delivery only from a completed, hash-bound picture run."""

    result_path = picture_dir / "result.json"
    status_path = picture_dir / "run-status.json"
    manifest_path = picture_dir / "render-manifest.json"
    result = read_json(result_path)
    status = read_json(status_path)
    manifest = read_json(manifest_path)
    if not isinstance(result, dict) or not isinstance(status, dict):
        raise DeliveryPipelineBlocked("reusable picture artifacts are malformed")
    if (
        status.get("stage") != "completed"
        or status.get("terminal") is not True
        or status.get("media_rendered") is not True
        or result.get("media_rendered") is not True
    ):
        raise DeliveryPipelineBlocked(
            "picture resume requires a completed, rendered feature-cut run"
        )
    binding_path = Path(str(manifest["feature_plan_binding"])).resolve(strict=True)
    binding = read_json(binding_path)
    expected_hashes = {
        "catalog_sha256": sha256_file(catalog_path.expanduser().resolve(strict=True)),
        "brief_sha256": sha256_file(brief_path.expanduser().resolve(strict=True)),
        "music_sha256": sha256_file(music_path.expanduser().resolve(strict=True)),
    }
    for field, expected in expected_hashes.items():
        if binding.get(field) != expected:
            raise DeliveryPipelineBlocked(
                f"picture resume rejected because {field} changed"
            )
    if Path(str(result["manifest_path"])).resolve(strict=True) != manifest_path:
        raise DeliveryPipelineBlocked(
            "picture result references an unexpected render manifest"
        )
    for aspect_key, manifest_key in (
        ("horizontal", "horizontal"),
        ("vertical", "vertical"),
    ):
        output_value = result.get(f"{aspect_key}_output")
        if output_value is None:
            continue
        output_path = Path(str(output_value)).resolve(strict=True)
        media = manifest.get(manifest_key, {}).get("media", {})
        if media.get("sha256") != sha256_file(output_path):
            raise DeliveryPipelineBlocked(
                f"picture resume rejected because {aspect_key} output changed"
            )
    saved_context = manifest.get("autonomous_context")
    result_context = result.get("autonomous_context_paths")
    if saved_context is not None or result_context is not None:
        if not isinstance(saved_context, Mapping) or not isinstance(
            result_context, Mapping
        ):
            raise DeliveryPipelineBlocked(
                "picture resume autonomous context is malformed"
            )
        if set(saved_context) != set(result_context):
            raise DeliveryPipelineBlocked(
                "picture resume autonomous context keys changed"
            )
        for key, value in result_context.items():
            context_path = Path(str(value)).resolve(strict=True)
            manifest_row = saved_context.get(key)
            if (
                not isinstance(manifest_row, Mapping)
                or manifest_row.get("path") != str(context_path)
                or manifest_row.get("sha256") != sha256_file(context_path)
            ):
                raise DeliveryPipelineBlocked(
                    f"picture resume rejected because {key} context changed"
                )
    saved_deterministic = manifest.get(
        "deterministic_delivery_evidence"
    )
    result_deterministic = result.get(
        "deterministic_delivery_evidence_path"
    )
    if saved_deterministic is not None or result_deterministic is not None:
        if not isinstance(saved_deterministic, Mapping) or not isinstance(
            result_deterministic, str
        ):
            raise DeliveryPipelineBlocked(
                "picture resume deterministic evidence is malformed"
            )
        evidence_path = Path(result_deterministic).resolve(strict=True)
        if (
            saved_deterministic.get("path") != str(evidence_path)
            or saved_deterministic.get("sha256")
            != sha256_file(evidence_path)
        ):
            raise DeliveryPipelineBlocked(
                "picture resume rejected because deterministic evidence changed"
            )
    saved_bundle = manifest.get("autonomous_evidence_bundle")
    result_bundle = result.get("autonomous_evidence_bundle_path")
    if saved_bundle is not None or result_bundle is not None:
        if not isinstance(saved_bundle, Mapping) or not isinstance(
            result_bundle, str
        ):
            raise DeliveryPipelineBlocked(
                "picture resume autonomous evidence bundle is malformed"
            )
        bundle_path = Path(result_bundle).resolve(strict=True)
        if (
            saved_bundle.get("path") != str(bundle_path)
            or saved_bundle.get("sha256") != sha256_file(bundle_path)
        ):
            raise DeliveryPipelineBlocked(
                "picture resume rejected because evidence bundle changed"
            )
    return result


def run_feature_delivery_pipeline(
    *,
    feature_cut_kwargs: Mapping[str, Any],
    brief_path: Path,
    music_path: Path,
    music_lock_path: Path,
    output_dir: Path,
    model_id: str = MODEL_ID,
    execution_profile: str = "production_review",
    reuse_picture_result: bool = False,
    autonomous_policy_path: Path | None = None,
    max_gemini_cost_usd: float | None = None,
    autonomous_context_paths: Mapping[str, Path] | None = None,
    deterministic_delivery_evidence_path: Path | None = None,
    editorial_beat_contracts_path: Path | None = None,
) -> dict[str, Any]:
    """Run picture → continuous music → final mux → final QA as one chain.

    Review profiles stop at immutable QA packages for human approval.
    Autonomous profiles may grant delivery eligibility only after deterministic
    gates, semantic final QA, policy-authorized degradation checks, and a
    separately hash-bound DecisionAuthorityV2 all pass.
    """

    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    profile = FeatureCutExecutionProfile(execution_profile)
    policy: AutonomousEditPolicy | None = None
    budget_ledger: BudgetLedger | None = None
    deterministic_evidence: DeterministicDeliveryEvidence | None = None
    deterministic_report: DeterministicDeliveryQaReport | None = None
    resolved_autonomous_context: dict[str, Path] = {}
    if profile in {
        FeatureCutExecutionProfile.AUTONOMOUS_STRICT,
        FeatureCutExecutionProfile.AUTONOMOUS_BEST_EFFORT,
    }:
        if autonomous_policy_path is None:
            raise DeliveryPipelineBlocked(
                "autonomous delivery requires an AutonomousEditPolicy"
            )
        resolved_policy = autonomous_policy_path.expanduser().resolve(
            strict=True
        )
        policy = AutonomousEditPolicy.model_validate(read_json(resolved_policy))
        expected_profile = AutonomousExecutionProfile(profile.value)
        if policy.execution_profile != expected_profile:
            raise DeliveryPipelineBlocked(
                "execution profile does not match the autonomous policy"
            )
        requested = set(policy.requested_aspects)
        aspect_argument = str(feature_cut_kwargs.get("aspect", "both"))
        expected_aspects = (
            {"16:9", "9:16"}
            if aspect_argument == "both"
            else {aspect_argument.replace("x", ":")}
        )
        if requested != expected_aspects:
            raise DeliveryPipelineBlocked(
                "requested aspect does not match the autonomous policy"
            )
        effective_cap = policy.budget.max_gemini_cost_usd
        if max_gemini_cost_usd is not None:
            if max_gemini_cost_usd <= 0:
                raise DeliveryPipelineBlocked(
                    "Gemini cost cap must be positive"
                )
            if max_gemini_cost_usd > effective_cap:
                raise DeliveryPipelineBlocked(
                    "runtime Gemini cost cap cannot loosen policy authority"
                )
            effective_cap = max_gemini_cost_usd
        budget_ledger = BudgetLedger(
            max_cost_usd=effective_cap,
            max_interactions=policy.budget.max_paid_interactions,
            reserved_recovery_fraction=(
                policy.budget.reserved_recovery_fraction
            ),
        )
        prior_usage = summarize_usage_and_list_price(resolved_output)
        policy_scoped_prior_requests = [
            request
            for request in prior_usage["requests"]
            if str(request["path"]).startswith(
                ("picture/", "aspects/", "audition/")
            )
        ]
        for request in policy_scoped_prior_requests:
            budget_ledger.adopt_reconciled_usage(
                stage="resumed_artifact",
                model_id=str(request["model"]),
                usage={
                    "total_input_tokens": request["input_tokens"],
                    "total_cached_tokens": request[
                        "cached_input_tokens"
                    ],
                    "total_output_tokens": request["output_tokens"],
                    "total_thought_tokens": request["thought_tokens"],
                },
            )
        required_context_keys = {
            "editorial_beat_contracts",
            "music_map",
            "cue_plan",
            "exact_event_locks",
            "reuse_degradation",
        }
        if autonomous_context_paths is not None:
            if set(autonomous_context_paths) != required_context_keys:
                raise DeliveryPipelineBlocked(
                    "autonomous final-QA context keys are incomplete or unknown"
                )
            resolved_autonomous_context = {
                key: path.expanduser().resolve(strict=True)
                for key, path in autonomous_context_paths.items()
            }
            preflight_degradation = (
                AutonomousDegradationManifest.model_validate(
                    read_json(
                        resolved_autonomous_context["reuse_degradation"]
                    )
                )
            )
            if preflight_degradation.policy_reference != policy.policy_reference:
                raise DeliveryPipelineBlocked(
                    "degradation manifest does not bind the autonomous policy"
                )
        elif editorial_beat_contracts_path is None:
            raise DeliveryPipelineBlocked(
                "autonomous delivery requires editorial beat contracts so "
                "feature-cut can generate selected-window context"
            )
        resolved_deterministic_evidence: Path | None = None
        if deterministic_delivery_evidence_path is not None:
            resolved_deterministic_evidence = (
                deterministic_delivery_evidence_path.expanduser().resolve(
                    strict=True
                )
            )
            deterministic_evidence = (
                DeterministicDeliveryEvidence.model_validate(
                    read_json(resolved_deterministic_evidence)
                )
            )
            deterministic_report = run_deterministic_delivery_qa(
                deterministic_evidence,
                policy=policy,
            )
            if not deterministic_report.passed:
                raise DeliveryPipelineBlocked(
                    "deterministic autonomous gates failed before paid work: "
                    + ", ".join(deterministic_report.failure_codes)
                )
        write_json(
            resolved_output / "autonomous-preflight.json",
            {
                "contract_version": "autonomous-delivery-preflight-v1",
                "execution_profile": profile.value,
                "policy_path": str(resolved_policy),
                "policy_reference": policy.policy_reference,
                "requested_aspects": list(policy.requested_aspects),
                "budget": budget_ledger.report(),
                "prior_usage_scope": {
                    "included_policy_scoped_requests": len(
                        policy_scoped_prior_requests
                    ),
                    "excluded_pre_policy_requests": (
                        prior_usage["request_count"]
                        - len(policy_scoped_prior_requests)
                    ),
                    "interpretation": (
                        "retrieval and cold-ingest calls created before the "
                        "policy-bound direct edit are reported separately and "
                        "cannot consume or reset the selected-edit ledger"
                    ),
                },
                "autonomous_context_paths": {
                    key: str(path)
                    for key, path in resolved_autonomous_context.items()
                },
                "context_source": (
                    "supplied"
                    if resolved_autonomous_context
                    else "selected_window_generation_pending"
                ),
                "deterministic_delivery_evidence_path": (
                    str(resolved_deterministic_evidence)
                    if resolved_deterministic_evidence is not None
                    else None
                ),
                "paid_call_started": False,
                "generated_at": utc_now(),
            },
        )
    status_path = resolved_output / "run-status.json"
    started_at = utc_now()
    _write_status(
        status_path,
        stage="feature_cut",
        terminal=False,
        state="running",
    )
    client: GeminiLabClient | None = None
    outputs: dict[str, Any] = {}
    deterministic_failure_codes: tuple[str, ...] = ()
    try:
        effective_music_lock_path = music_lock_path
        if (
            policy is not None
            and music_path.expanduser().is_file()
            and music_lock_path.expanduser().is_file()
        ):
            effective_music_lock_path = _bind_music_lock_to_autonomous_policy(
                music_path=music_path,
                music_lock_path=music_lock_path,
                policy=policy,
                output_dir=resolved_output,
            )
        kwargs = dict(feature_cut_kwargs)
        kwargs["output_dir"] = resolved_output / "picture"
        kwargs["brief_path"] = brief_path
        kwargs["music_path"] = music_path
        kwargs["music_lock_path"] = effective_music_lock_path
        kwargs["execution_profile"] = profile.value
        if policy is not None:
            kwargs["autonomous_policy_path"] = autonomous_policy_path
            kwargs["budget_ledger"] = budget_ledger
            kwargs["editorial_beat_contracts_path"] = (
                editorial_beat_contracts_path
                or resolved_autonomous_context.get(
                    "editorial_beat_contracts"
                )
            )
        if reuse_picture_result:
            feature_result = _load_reusable_picture_result(
                resolved_output / "picture",
                catalog_path=Path(kwargs["catalog_path"]),
                brief_path=brief_path,
                music_path=music_path,
            )
        else:
            feature_result = run_feature_cut_experiment(**kwargs)
        outputs["feature_cut"] = feature_result
        if policy is not None and not resolved_autonomous_context:
            generated_context = feature_result.get(
                "autonomous_context_paths"
            )
            if not isinstance(generated_context, Mapping):
                raise DeliveryPipelineBlocked(
                    "feature-cut did not persist selected-window context"
                )
            if set(generated_context) != required_context_keys:
                raise DeliveryPipelineBlocked(
                    "generated autonomous context is incomplete"
                )
            resolved_autonomous_context = {
                str(key): Path(str(path)).expanduser().resolve(strict=True)
                for key, path in generated_context.items()
            }
        if policy is not None and deterministic_evidence is None:
            generated_evidence = feature_result.get(
                "deterministic_delivery_evidence_path"
            )
            if not isinstance(generated_evidence, str):
                raise DeliveryPipelineBlocked(
                    "feature-cut did not persist deterministic delivery evidence"
                )
            resolved_deterministic_evidence = Path(
                generated_evidence
            ).expanduser().resolve(strict=True)
            deterministic_evidence = (
                DeterministicDeliveryEvidence.model_validate(
                    read_json(resolved_deterministic_evidence)
                )
            )
            deterministic_report = run_deterministic_delivery_qa(
                deterministic_evidence,
                policy=policy,
            )
            if not deterministic_report.passed:
                deterministic_failure_codes = tuple(
                    deterministic_report.failure_codes
                )
        picture_media_rendered = bool(feature_result.get("media_rendered"))
        picture_outputs_present = any(
            feature_result.get(f"{aspect_key}_output") is not None
            for aspect_key in ("horizontal", "vertical")
        )
        if not picture_media_rendered or not picture_outputs_present:
            raise DeliveryPipelineBlocked(
                "feature-cut did not produce reviewable picture media"
            )
        if deterministic_failure_codes and not any(
            Path(str(feature_result[f"{aspect_key}_output"])).is_file()
            for aspect_key in ("horizontal", "vertical")
            if feature_result.get(f"{aspect_key}_output") is not None
        ):
            raise DeliveryPipelineBlocked(
                "generated deterministic autonomous gates failed before "
                "final QA: "
                + ", ".join(deterministic_failure_codes)
            )
        picture_ready_for_review = bool(
            feature_result.get("ready_for_human_review")
        )

        resolved_music = music_path.expanduser().resolve(strict=True)
        resolved_lock_path = effective_music_lock_path.expanduser().resolve(
            strict=True
        )
        music_lock = MusicMapLock.model_validate(read_json(resolved_lock_path))
        if music_lock.music_id != f"sha256:{sha256_file(resolved_music)}":
            raise DeliveryPipelineBlocked(
                "reviewed MusicMap lock does not bind the supplied soundtrack"
            )

        render_manifest_path = Path(feature_result["manifest_path"]).resolve(
            strict=True
        )
        final_results: dict[str, Any] = {}
        qa_results_by_aspect: dict[str, Any] = {}
        qa_interaction_ids: list[str] = []
        if not deterministic_failure_codes:
            client = GeminiLabClient(model_id=model_id)
        for aspect_key, aspect_ratio, qa_mode in (
            ("horizontal", "16:9", "canonical_16x9"),
            ("vertical", "9:16", "crop_only_9x16"),
        ):
            picture_value = feature_result.get(f"{aspect_key}_output")
            if picture_value is None:
                continue
            picture = Path(str(picture_value)).resolve(strict=True)
            picture_duration_ms = _picture_video_duration_ms(picture)
            music_timeline_path = (
                render_manifest_path.parent
                / "editorial-planning"
                / "music-output-timeline.json"
            )
            expected_timeline = (
                read_json(music_timeline_path)
                if music_timeline_path.is_file()
                else None
            )
            aspect_dir = resolved_output / (
                "audition" if deterministic_failure_codes else "aspects"
            ) / aspect_key
            assembly_key = hashlib.sha256(
                (
                    "feature-delivery-music-assembly-v2:"
                    f"{sha256_file(picture)}:"
                    f"{picture_duration_ms}:"
                    f"{sha256_file(resolved_lock_path)}:"
                    f"{sha256_file(music_timeline_path) if expected_timeline else 'unplanned'}"
                ).encode("utf-8")
            ).hexdigest()
            assembly_dir = (
                aspect_dir / "music-assembly" / "runs" / assembly_key
            )
            if expected_timeline is not None:
                expected_plan = expected_timeline.get("plan_definition", {})
                plan_contract_version = expected_timeline.get(
                    "plan_contract_version"
                )
                if plan_contract_version == "music-assembly-plan-v1":
                    selected_music_plan = MusicAssemblyPlan.model_validate(
                        expected_plan
                    )
                    write_music_assembly_artifacts(
                        selected_music_plan,
                        output_dir=assembly_dir,
                    )
                    rendered_music = render_single_interval_music_assembly(
                        resolved_music,
                        selected_music_plan,
                        assembly_dir / "music.wav",
                        assembly_dir,
                    )
                elif plan_contract_version == "music-edit-plan-v2":
                    selected_music_plan = MusicEditPlanV2.model_validate(
                        expected_plan
                    )
                    rendered_music = render_reviewed_music_edit_v2(
                        resolved_music,
                        selected_music_plan,
                        assembly_dir / "music.wav",
                        assembly_dir,
                    )
                else:
                    raise DeliveryPipelineBlocked(
                        "picture music timeline uses an unsupported plan contract"
                    )
            else:
                try:
                    selected_music_plan = plan_single_interval_music_assembly(
                        music_lock,
                        music_lock_path=resolved_lock_path,
                        target_duration_ms=picture_duration_ms,
                        minimum_duration_ms=max(1, picture_duration_ms - 100),
                        maximum_duration_ms=picture_duration_ms + 100,
                    )
                    write_music_assembly_artifacts(
                        selected_music_plan,
                        output_dir=assembly_dir,
                    )
                    rendered_music = render_single_interval_music_assembly(
                        resolved_music,
                        selected_music_plan,
                        assembly_dir / "music.wav",
                        assembly_dir,
                    )
                except MusicAssemblyError:
                    selected_music_plan = plan_contiguous_reviewed_music_edit_v2(
                        music_lock,
                        music_lock_path=resolved_lock_path,
                        target_duration_ms=picture_duration_ms,
                        minimum_duration_ms=max(1, picture_duration_ms - 100),
                        maximum_duration_ms=picture_duration_ms + 100,
                    )
                    rendered_music = render_reviewed_music_edit_v2(
                        resolved_music,
                        selected_music_plan,
                        assembly_dir / "music.wav",
                        assembly_dir,
                    )
            delivery = assemble_music_only_delivery(
                picture_path=picture,
                music_path=rendered_music.output_audio_path,
                output_path=aspect_dir / (
                    f"audition-{aspect_key}.mp4"
                    if deterministic_failure_codes
                    else f"final-{aspect_key}.mp4"
                ),
                manifest_path=aspect_dir / (
                    "audition-delivery.json"
                    if deterministic_failure_codes
                    else "final-delivery.json"
                ),
                music_assembly_artifact_dir=assembly_dir,
                aspect_ratio=aspect_ratio,
                artifact_bindings={
                    "feature_render_manifest_sha256": sha256_file(
                        render_manifest_path
                    ),
                    "music_map_lock_sha256": sha256_file(resolved_lock_path),
                },
            )
            if deterministic_failure_codes:
                final_results[aspect_key] = {
                    "audition_output": str(delivery.output_path),
                    "audition_output_sha256": sha256_file(
                        delivery.output_path
                    ),
                    "audition_manifest": str(delivery.manifest_path),
                    "music_assembly_manifest": str(
                        rendered_music.manifest_path
                    ),
                    "delivery_eligible": False,
                    "interpretation": (
                        "music-backed review artifact preserved after local "
                        "autonomous gates blocked delivery"
                    ),
                }
                continue
            assert client is not None
            qa_dir = aspect_dir / "final-qa"
            if policy is not None and aspect_ratio == "9:16":
                qa_mode = "autonomous_final_9x16"
            prepared = prepare_final_edit_qa(
                mode=qa_mode,
                render_path=delivery.output_path,
                manifest_path=render_manifest_path,
                output_dir=qa_dir,
                model_id=model_id,
                brief_path=(
                    brief_path
                    if qa_mode
                    in {"canonical_16x9", "autonomous_final_9x16"}
                    else None
                ),
                crop_include_audio=qa_mode != "crop_only_9x16",
                autonomous_context_paths=(
                    resolved_autonomous_context
                    if qa_mode == "autonomous_final_9x16"
                    else None
                ),
            )
            uploaded, file_reused = client.ensure_video_upload(
                prepared.proxy_path,
                qa_dir / "file-api" / prepared.input_hashes["proxy_sha256"],
            )
            qa = execute_final_edit_qa(
                prepared=prepared,
                client=client.client,
                uploaded_video=uploaded,
                output_dir=qa_dir,
                budget_ledger=budget_ledger,
                recovery_call=policy is not None,
            )
            qa_results_by_aspect[aspect_ratio] = qa.result
            raw_path = qa.run_dir / "raw_interaction.json"
            if raw_path.is_file():
                interaction_id = read_json(raw_path).get("id")
                if isinstance(interaction_id, str) and interaction_id:
                    qa_interaction_ids.append(interaction_id)
            qa_disposition = (
                qa.result.qa_observation_status
                if isinstance(qa.result, AutonomousFinalEditQa)
                else _qa_disposition(qa)
            )
            final_results[aspect_key] = {
                "final_output": str(delivery.output_path),
                "final_output_sha256": sha256_file(delivery.output_path),
                "delivery_manifest": str(delivery.manifest_path),
                "music_assembly_manifest": str(rendered_music.manifest_path),
                "qa_run_dir": str(qa.run_dir),
                "qa_disposition": qa_disposition,
                "qa_cache_hit": qa.cache_hit,
                "file_api_reused": file_reused,
            }

        if deterministic_failure_codes:
            outputs["audition_music_mix"] = final_results
            if deterministic_report is not None:
                write_json(
                    resolved_output / "deterministic-delivery-qa.json",
                    deterministic_report,
                )
            write_json(
                resolved_output / "audition" / "audition-manifest.json",
                {
                    "contract_version": "blocked-audition-mix-v1",
                    "delivery_eligible": False,
                    "failure_codes": list(deterministic_failure_codes),
                    "aspects": final_results,
                    "generated_at": utc_now(),
                },
            )
            raise DeliveryPipelineBlocked(
                "generated deterministic autonomous gates failed before "
                "final QA; a music-backed audition mix was preserved: "
                + ", ".join(deterministic_failure_codes)
            )
        if not final_results:
            raise DeliveryPipelineBlocked(
                "feature-cut did not produce any requested picture output"
            )
        delivery_authority: DecisionAuthorityV2 | None = None
        if policy is not None:
            assert deterministic_evidence is not None
            assert deterministic_report is not None
            degradation_path = resolved_autonomous_context.get(
                "reuse_degradation"
            )
            if degradation_path is None:
                raise DeliveryPipelineBlocked(
                    "autonomous context omitted reuse_degradation"
                )
            degradation = AutonomousDegradationManifest.model_validate(
                read_json(degradation_path)
            )
            authority_hashes = {
                f"sha256:{policy.definition_sha256()}",
                *(
                    f"sha256:{sha256_file(path)}"
                    for path in resolved_autonomous_context.values()
                ),
                *(
                    f"sha256:{row['final_output_sha256']}"
                    for row in final_results.values()
                ),
            }
            state, delivery_authority = authorize_autonomous_delivery(
                policy=policy,
                deterministic_qa=deterministic_report,
                qa_results=qa_results_by_aspect,
                degradation=degradation,
                input_artifact_hashes=tuple(sorted(authority_hashes)),
                gemini_interaction_ids=tuple(
                    dict.fromkeys(qa_interaction_ids)
                ),
            )
            write_json(
                resolved_output / "deterministic-delivery-qa.json",
                deterministic_report,
            )
            write_json(
                resolved_output / "decision-authority.json",
                delivery_authority,
            )
            delivery_eligible = True
            human_approval_status = "not_required_auto_policy"
        else:
            dispositions = {
                row["qa_disposition"] for row in final_results.values()
            }
            state = (
                "ready_for_human_review"
                if (
                    picture_ready_for_review
                    and dispositions == {"ready_for_human_review"}
                )
                else "review_required"
            )
            delivery_eligible = False
            human_approval_status = "not_run"
        result = {
            "contract_version": "feature-delivery-result-v1",
            "started_at": started_at,
            "completed_at": utc_now(),
            "state": state,
            "media_rendered": True,
            "final_sequence_qa_completed": True,
            "human_approval_status": human_approval_status,
            "delivery_eligible": delivery_eligible,
            "autonomous_policy_reference": (
                policy.policy_reference if policy is not None else None
            ),
            "budget": (
                budget_ledger.report()
                if budget_ledger is not None
                else None
            ),
            "decision_authority": (
                delivery_authority.model_dump(mode="json")
                if delivery_authority is not None
                else None
            ),
            "deterministic_delivery_qa": (
                deterministic_report.model_dump(mode="json")
                if deterministic_report is not None
                else None
            ),
            "picture_ready_for_human_review": picture_ready_for_review,
            "picture_run_state": feature_result.get("run_state"),
            "feature_cut": feature_result,
            "aspects": final_results,
        }
        write_json(resolved_output / "result.json", result)
        _write_status(
            status_path,
            stage="completed",
            terminal=True,
            state=state,
            outputs=final_results,
            delivery_eligible=delivery_eligible,
        )
        return result
    except Exception as error:
        _write_status(
            status_path,
            stage="blocked",
            terminal=True,
            state="blocked",
            error=error,
            outputs=outputs,
        )
        raise
    finally:
        if client is not None:
            client.close()
