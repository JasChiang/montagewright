from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from time import monotonic
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

from .autonomous_policy import (
    AutonomousDegradationManifest,
    AutonomousEditPolicy,
    AutonomousExecutionProfile,
    DecisionAuthorityV2,
    authorize_decision,
    omissions_are_policy_authorized,
)
from .billing import (
    BudgetLedger,
    adopt_paid_dispatch_journal_state,
    estimate_paid_call,
    migrate_completed_legacy_paid_dispatch,
    summarize_usage_and_list_price,
)
from .clip_card_observations import (
    ClipObservationSupplement,
    EventObservationSupplement,
    validate_supplement,
)
from .clip_card_supplement_runner import (
    current_supplement_request_binding,
)
from .clip_card_retrieval import compact_retrieval_card
from .clip_card_retrieval import FeatureShortlistPlan
from .feature_cut import (
    compile_repair_request,
    render_changed_segments_and_concat,
    run_feature_cut_experiment,
    validate_authorized_selected_window_cue_plan,
    validate_policy_decision_artifact,
)
from .event_lock import load_editorial_beat_contracts
from .final_delivery import (
    assemble_music_only_delivery,
    assemble_picture_only_delivery,
)
from .final_edit_qa import (
    AutonomousFinalEditQa,
    AutonomousRecoveryExecution,
    AutonomousRecoveryPlan,
    DeterministicEvidenceCausalBinding,
    DeterministicDeliveryEvidence,
    DeterministicDeliveryQaReport,
    execute_final_edit_qa,
    plan_autonomous_recovery,
    prepare_final_edit_qa,
    run_deterministic_delivery_qa,
    validate_deterministic_evidence_causal_binding,
)
from .full_v1 import current_full_clip_card_cache_key
from .gemini import (
    GeminiLabClient,
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
)
from .media import probe_video, sha256_file
from .music import (
    MusicMapLock,
    MusicMapProposal,
    lock_music_map_with_auto_policy,
    validate_music_map_lock_integrity,
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
    FullClipCard,
    MusicAssemblyPlan,
    MusicEditPlanV2,
    RushesCatalog,
)
from .schema import gemini_response_schema
from .storage import read_json, utc_now, write_json


class DeliveryPipelineBlocked(RuntimeError):
    """The pipeline preserved review artifacts but cannot continue safely."""


def _planning_subprocess_environment(*, project_root: Path) -> dict[str, str]:
    """Pass a locally configured Gemini credential to owned CLI children.

    ``uv run`` can load a workspace ``.env`` for its direct command without
    exporting that value into this process.  Cold Clip Card refreshes are
    deliberately isolated child CLI invocations, so relying on inheritance
    alone turns a legitimate stale-card refresh into a misleading blocked
    delivery before its first paid request.  Copy only the two supported
    credential names from the workspace file when the parent has neither;
    never persist or log the resulting environment.
    """

    environment = os.environ.copy()
    if environment.get("GEMINI_API_KEY") or environment.get("GOOGLE_API_KEY"):
        return environment
    env_path = project_root / ".env"
    if not env_path.is_file():
        return environment
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, raw_value = line.partition("=")
        if (
            not separator
            or key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}
            or not raw_value.strip()
        ):
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        environment.setdefault(key, value)
    return environment


def _maximum_full_final_qa_calls(policy: AutonomousEditPolicy) -> int:
    """Reserve every policy-reachable QA pass independently per aspect."""

    return (
        len(policy.requested_aspects)
        * policy.budget.max_final_qa_passes
    )


def _mandatory_paid_stage_minimums(
    *,
    policy: AutonomousEditPolicy,
    editorial_beat_contracts_path: Path | None,
) -> dict[str, int]:
    """Reserve graph-derived hard work instead of a guessed call count."""

    minimums = {
        # Reserve both allowed QA passes. The second remains unused unless a
        # bounded repair is actually necessary.
        "final_qa": _maximum_full_final_qa_calls(policy),
    }
    if editorial_beat_contracts_path is None:
        return minimums
    contracts = load_editorial_beat_contracts(
        editorial_beat_contracts_path.expanduser().resolve(strict=True)
    )
    hard_contracts = tuple(
        contract for contract in contracts if contract.priority == "hard"
    )
    exact_event_features = {
        contract.feature_id
        for contract in contracts
        if contract.visual_events
    }
    for feature_id in sorted(exact_event_features):
        minimums[f"exact_event_group:{feature_id}"] = 1
    hard_grounding_features = {
        contract.feature_id or contract.beat_id
        for contract in hard_contracts
        if contract.required_target_ids
    }
    for feature_id in sorted(hard_grounding_features):
        minimums[f"multi_target_grounding:{feature_id}"] = 1
    return minimums


def _mandatory_paid_stage_cost_holds(
    *,
    policy: AutonomousEditPolicy,
    editorial_beat_contracts_path: Path | None,
) -> dict[str, float]:
    """Price the mandatory completion path before the first paid dispatch.

    These are conservative admission ceilings, not forecasts.  Provider cache
    hits are deliberately ignored.  Each configured cost hold has a matching
    interaction hold so a completed node consumes both forms of escrow.
    """

    limits = policy.gemini_limits
    holds: dict[str, float] = {}
    qa_calls = _maximum_full_final_qa_calls(policy)
    qa_estimate = estimate_paid_call(
        stage="final_qa",
        model_id=MODEL_ID,
        media_duration_ms=policy.duration.max_ms,
        media_resolution=policy.media_resolution.final_video_qa,
        text_input_tokens=8_000,
        max_output_tokens=limits.final_qa.max_output_tokens,
        thinking_level=limits.final_qa.thinking_level,
        retry_allowance=0,
    )
    holds["final_qa"] = round(
        qa_estimate.worst_case_cost_usd * qa_calls,
        8,
    )
    if editorial_beat_contracts_path is None:
        return holds
    contracts = load_editorial_beat_contracts(
        editorial_beat_contracts_path.expanduser().resolve(strict=True)
    )
    hard_contracts = tuple(
        contract for contract in contracts if contract.priority == "hard"
    )
    exact_event_features = {
        contract.feature_id
        for contract in contracts
        if contract.visual_events
    }
    for feature_id in sorted(exact_event_features):
        estimate = estimate_paid_call(
            stage=f"exact_event_group:{feature_id}",
            model_id=MODEL_ID,
            media_resolution=policy.media_resolution.exact_event_image,
            image_count=12,
            text_input_tokens=4_000,
            max_output_tokens=limits.exact_event_group.max_output_tokens,
            thinking_level=limits.exact_event_group.thinking_level,
            retry_allowance=0,
        )
        holds[estimate.stage] = estimate.worst_case_cost_usd
    hard_grounding_features = {
        contract.feature_id or contract.beat_id
        for contract in hard_contracts
        if contract.required_target_ids
    }
    if hard_grounding_features:
        grounding_estimate = estimate_paid_call(
            stage="multi_target_grounding",
            model_id=MODEL_ID,
            media_resolution=(
                policy.media_resolution.exact_frame_grounding_image
            ),
            image_count=1,
            text_input_tokens=4_000,
            max_output_tokens=(
                limits.multi_target_grounding.max_output_tokens
            ),
            thinking_level=(
                limits.multi_target_grounding.thinking_level
            ),
            retry_allowance=0,
        )
        for feature_id in sorted(hard_grounding_features):
            holds[f"multi_target_grounding:{feature_id}"] = (
                grounding_estimate.worst_case_cost_usd
            )
    return holds


def _prepared_clip_card_library_root(
    *,
    catalog_path: Path,
    prepared_library_path: Path | None,
    create_explicit: bool = False,
) -> Path:
    resolved_catalog = catalog_path.expanduser().resolve(strict=True)
    if prepared_library_path is not None:
        explicit = prepared_library_path.expanduser().resolve()
        if create_explicit:
            (explicit / "clips").mkdir(parents=True, exist_ok=True)
        if (explicit / "clips").is_dir():
            return explicit
        raise DeliveryPipelineBlocked(
            "prepared Clip Card library has no clips directory"
        )
    candidates = (
        resolved_catalog.parent.parent / "clip-cards",
        resolved_catalog.parent / "clip-cards",
    )
    library = next(
        (candidate for candidate in candidates if (candidate / "clips").is_dir()),
        None,
    )
    if library is None:
        raise DeliveryPipelineBlocked(
            "fresh autonomous planning requires a prepared Base Clip Card "
            "library; pass --prepared-clip-cards or place clip-cards beside "
            "the catalog artifacts"
        )
    return library.resolve()


def _expected_clip_card_cache_key(
    *,
    source_asset_id: str,
    proxy_asset_id: str,
) -> dict[str, Any]:
    card_prompt = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "full_clip_card_mmss_zh-TW.txt"
    ).read_text(encoding="utf-8")
    return current_full_clip_card_cache_key(
        prompt=card_prompt,
        source_asset_id=source_asset_id,
        proxy_asset_id=proxy_asset_id,
    )


def _clip_card_entry_stale_reason(
    *,
    clip: Any,
    clip_root: Path,
) -> str | None:
    card_dir = clip_root / "gemini" / "clip-card"
    card_path = card_dir / "clip_card.json"
    proxy_path = clip_root / "analysis-proxy.mp4"
    cache_key_path = card_dir / "cache-key.json"
    if not card_path.is_file():
        return "missing_clip_card"
    if not proxy_path.is_file():
        return "missing_analysis_proxy"
    if not cache_key_path.is_file():
        return "missing_cache_binding"
    try:
        card = FullClipCard.model_validate(read_json(card_path))
        proxy = probe_video(proxy_path)
    except Exception:
        return "invalid_clip_card_or_proxy"
    if card.source_asset_id != f"sha256:{clip.sha256}":
        return "source_identity_changed"
    if (
        card.proxy_asset_id != proxy.asset_id
        or card.duration_ms != clip.duration_ms
        or abs(proxy.duration_ms - card.duration_ms) > 100
        or card.model_provenance.model_id != MODEL_ID
    ):
        return "proxy_duration_or_model_lineage_changed"
    expected = _expected_clip_card_cache_key(
        source_asset_id=card.source_asset_id,
        proxy_asset_id=card.proxy_asset_id,
    )
    if read_json(cache_key_path) != expected:
        return "prompt_schema_or_request_binding_changed"
    return None


def _resolve_prepared_clip_card_library(
    *,
    catalog_path: Path,
    prepared_library_path: Path | None,
) -> tuple[Path, RushesCatalog, tuple[Path, ...], int]:
    """Resolve and validate the reusable Base Clip Card library before paid work."""

    resolved_catalog = catalog_path.expanduser().resolve(strict=True)
    catalog = RushesCatalog.model_validate(read_json(resolved_catalog))
    library = _prepared_clip_card_library_root(
        catalog_path=resolved_catalog,
        prepared_library_path=prepared_library_path,
    )
    compact_evidence: list[dict[str, object]] = []
    cards_by_asset: dict[str, FullClipCard] = {}
    for clip in catalog.clips:
        clip_root = library / "clips" / clip.sha256[:16]
        card_dir = (
            clip_root
            / "gemini"
            / "clip-card"
        )
        card_path = (
            card_dir
            / "clip_card.json"
        )
        if not card_path.is_file():
            raise DeliveryPipelineBlocked(
                "prepared Clip Card library is incomplete for source "
                f"sha256:{clip.sha256}"
            )
        card = FullClipCard.model_validate(read_json(card_path))
        if card.source_asset_id != f"sha256:{clip.sha256}":
            raise DeliveryPipelineBlocked(
                "prepared Clip Card source identity differs from the catalog"
            )
        proxy_path = clip_root / "analysis-proxy.mp4"
        if not proxy_path.is_file():
            raise DeliveryPipelineBlocked(
                "prepared Clip Card is missing its immutable analysis proxy"
            )
        proxy = probe_video(proxy_path)
        if (
            card.proxy_asset_id != proxy.asset_id
            or card.duration_ms != clip.duration_ms
            or abs(proxy.duration_ms - card.duration_ms) > 100
            or card.model_provenance.model_id != MODEL_ID
        ):
            raise DeliveryPipelineBlocked(
                "prepared Clip Card proxy, duration, or model lineage is stale"
            )
        cache_key_path = card_dir / "cache-key.json"
        expected_cache_key = _expected_clip_card_cache_key(
            source_asset_id=card.source_asset_id,
            proxy_asset_id=card.proxy_asset_id,
        )
        if (
            not cache_key_path.is_file()
            or read_json(cache_key_path) != expected_cache_key
        ):
            raise DeliveryPipelineBlocked(
                "prepared Clip Card cache binding is stale; regenerate it "
                "before autonomous planning"
            )
        cards_by_asset[card.source_asset_id] = card
    supplement_paths = tuple(
        sorted(
            library.glob(
                "clips/*/supplements/*/clip-observation-supplement.json"
            )
        )
    )
    supplements_by_asset: dict[str, list[ClipObservationSupplement]] = {}
    supplement_prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "clip_observation_supplement_zh-TW.txt"
    )
    supplement_schema_payload = json.dumps(
        gemini_response_schema(EventObservationSupplement),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_supplement_binding = current_supplement_request_binding(
        model_id=MODEL_ID,
        prompt_sha256=sha256_file(supplement_prompt_path),
        response_schema_sha256=hashlib.sha256(
            supplement_schema_payload.encode("utf-8")
        ).hexdigest(),
    )
    for path in supplement_paths:
        supplement = ClipObservationSupplement.model_validate(read_json(path))
        card = cards_by_asset.get(supplement.source_asset_id)
        if card is None:
            raise DeliveryPipelineBlocked(
                "prepared supplement references an asset outside the catalog"
            )
        validate_supplement(
            card,
            supplement,
            expected_request_binding=expected_supplement_binding,
            require_current_lineage=True,
        )
        supplements_by_asset.setdefault(
            supplement.source_asset_id, []
        ).append(supplement)
    for card in cards_by_asset.values():
        compact_evidence.append(
            compact_retrieval_card(
                card,
                supplements_by_asset.get(card.source_asset_id, ()),
            )
        )
    # Character count deliberately overestimates the token count for the
    # compact multilingual JSON. This is a reserve, not post-hoc billing.
    evidence_characters = len(
        json.dumps(
            compact_evidence,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return library.resolve(), catalog, supplement_paths, evidence_characters


def _reconcile_planning_subprocess_usage(
    *,
    budget_ledger: BudgetLedger,
    reservation_id: str,
    stage_dir: Path,
    estimate_input_tokens: int,
    estimate_output_tokens: int,
    estimate_thought_tokens: int,
) -> dict[str, Any]:
    usage = summarize_usage_and_list_price(stage_dir)
    request_count = int(usage["request_count"])
    if request_count == 0:
        dispatched_request_paths = tuple(stage_dir.rglob("*.request.json"))
        if not dispatched_request_paths:
            budget_ledger.cancel_before_dispatch(reservation_id)
            return usage
        # The request artifact is written immediately before dispatch. If the
        # API failed before returning immutable usage, keep the run fail-closed
        # and charge the reservation's conservative ceiling. A resume must
        # never regain budget merely because a 429/503 lacked token metadata.
        budget_ledger.reconcile(
            reservation_id,
            usage={
                "total_input_tokens": estimate_input_tokens,
                "total_cached_tokens": 0,
                "total_output_tokens": estimate_output_tokens,
                "total_thought_tokens": estimate_thought_tokens,
            },
            model_id=MODEL_ID,
        )
        return {
            **usage,
            "usage_status": "dispatch_recorded_usage_unavailable",
            "conservative_reconciliation": True,
            "total_input_tokens": estimate_input_tokens,
            "total_cached_input_tokens": 0,
            "total_output_tokens": estimate_output_tokens,
            "total_thought_tokens": estimate_thought_tokens,
            "request_artifacts": [
                str(path.resolve()) for path in dispatched_request_paths
            ],
        }
    if request_count != 1:
        raise DeliveryPipelineBlocked(
            "bounded autonomous planning stage produced an unexpected number "
            f"of paid interactions: {request_count}"
        )
    budget_ledger.reconcile(
        reservation_id,
        usage={
            "total_input_tokens": usage["total_input_tokens"],
            "total_cached_tokens": usage["total_cached_input_tokens"],
            "total_output_tokens": usage["total_output_tokens"],
            "total_thought_tokens": usage["total_thought_tokens"],
        },
        model_id=MODEL_ID,
    )
    return usage


def _run_budgeted_planning_stage(
    *,
    command: list[str],
    stage: str,
    stage_dir: Path,
    budget_ledger: BudgetLedger,
    estimated_text_tokens: int,
    media_duration_ms: int = 0,
    media_resolution: str = "low",
    max_output_tokens: int = 12_000,
    thinking_level: str = "low",
) -> dict[str, Any]:
    estimate = estimate_paid_call(
        stage=stage,
        model_id=MODEL_ID,
        media_duration_ms=media_duration_ms,
        media_resolution=media_resolution,
        text_input_tokens=estimated_text_tokens,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
        retry_allowance=0,
    )
    reservation = budget_ledger.reserve(estimate)
    started = monotonic()
    completed: subprocess.CompletedProcess[str] | None = None
    error: BaseException | None = None
    usage: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=_planning_subprocess_environment(
                project_root=Path(__file__).resolve().parents[2]
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as caught:
        # ``subprocess.run(check=True)`` raises with the captured process
        # result attached. Preserve it in the immutable orchestration artifact
        # instead of replacing the most useful failure evidence with nulls.
        completed = subprocess.CompletedProcess(
            args=caught.cmd,
            returncode=caught.returncode,
            stdout=caught.stdout,
            stderr=caught.stderr,
        )
        error = caught
    except BaseException as caught:
        error = caught
    finally:
        usage = _reconcile_planning_subprocess_usage(
            budget_ledger=budget_ledger,
            reservation_id=reservation.reservation_id,
            stage_dir=stage_dir,
            estimate_input_tokens=estimate.estimated_input_tokens,
            estimate_output_tokens=estimate.max_output_tokens,
            estimate_thought_tokens=estimate.reserved_thought_tokens,
        )
        write_json(
            stage_dir / "orchestration.json",
            {
                "contract_version": "autonomous-planning-orchestration-v1",
                "stage": stage,
                "command": command,
                "returncode": (
                    completed.returncode if completed is not None else None
                ),
                "stdout": (
                    completed.stdout if completed is not None else None
                ),
                "stderr": (
                    completed.stderr if completed is not None else None
                ),
                "elapsed_seconds": round(monotonic() - started, 3),
                "usage": usage,
                "error": (
                    None
                    if error is None
                    else {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                ),
                "generated_at": utc_now(),
            },
        )
    try:
        _migrate_completed_planning_dispatches(
            stage_dir=stage_dir,
            stage=stage,
        )
    except BaseException as caught:
        if error is None:
            error = caught
    if error is not None:
        raise DeliveryPipelineBlocked(
            f"{stage} failed; immutable subprocess artifacts were preserved"
        ) from error
    return usage


def _refresh_stale_clip_cards(
    *,
    catalog_path: Path,
    prepared_library_path: Path | None,
    output_dir: Path,
    max_cold_ingest_cost_usd: float | None,
) -> tuple[dict[str, Any], ...]:
    """Refresh stale cards under a graph-sized, cold-only budget ledger.

    Discovery completes before the first paid dispatch. Completed refresh
    records from the same output namespace are adopted into the cold ledger,
    so a process restart cannot regain already-spent money.
    """

    resolved_catalog = catalog_path.expanduser().resolve(strict=True)
    catalog = RushesCatalog.model_validate(read_json(resolved_catalog))
    library = _prepared_clip_card_library_root(
        catalog_path=resolved_catalog,
        prepared_library_path=prepared_library_path,
        create_explicit=prepared_library_path is not None,
    )
    planned_refreshes: list[tuple[Any, Path, str]] = []
    for clip in catalog.clips:
        clip_root = library / "clips" / clip.sha256[:16]
        stale_reason = _clip_card_entry_stale_reason(
            clip=clip,
            clip_root=clip_root,
        )
        if stale_reason is not None:
            planned_refreshes.append((clip, clip_root, stale_reason))

    cold_root = output_dir / "cold-ingest"
    prior_records: list[Mapping[str, Any]] = []
    for record_path in sorted(cold_root.glob("*/refresh-record.json")):
        payload = read_json(record_path)
        if not isinstance(payload, Mapping):
            raise DeliveryPipelineBlocked(
                f"cold-ingest refresh record is not an object: {record_path}"
            )
        usage = payload.get("usage")
        source_asset_id = str(payload.get("source_asset_id") or "")
        if (
            not isinstance(usage, Mapping)
            or int(usage.get("request_count") or 0) != 1
            or not source_asset_id.startswith("sha256:")
        ):
            raise DeliveryPipelineBlocked(
                "cold-ingest refresh record lacks one immutable paid usage "
                f"claim: {record_path}"
            )
        prior_records.append(payload)
    prior_failed_dispatches: list[Mapping[str, Any]] = []
    for record_path in sorted(
        cold_root.glob("*/failed-dispatch-record.json")
    ):
        payload = read_json(record_path)
        if not isinstance(payload, Mapping):
            raise DeliveryPipelineBlocked(
                f"cold-ingest failure record is not an object: {record_path}"
            )
        if payload.get("charged_interaction") is not True:
            continue
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            raise DeliveryPipelineBlocked(
                "charged cold-ingest failure has no immutable usage reserve: "
                f"{record_path}"
            )
        prior_failed_dispatches.append(payload)

    total_cold_interactions = len(prior_records) + max(
        len(planned_refreshes),
        len(prior_failed_dispatches),
    )
    if total_cold_interactions == 0:
        write_json(
            cold_root / "budget-report.json",
            {
                "contract_version": "cold-ingest-budget-report-v1",
                "namespace": "cold_ingest",
                "planned_refresh_count": 0,
                "adopted_refresh_count": 0,
                "budget": None,
                "generated_at": utc_now(),
            },
        )
        return ()
    if max_cold_ingest_cost_usd is None:
        raise DeliveryPipelineBlocked(
            f"{len(planned_refreshes)} stale or missing Clip Cards require "
            "paid refresh, but policy budget.max_cold_ingest_cost_usd is "
            "not authorized"
        )

    cold_ledger = BudgetLedger(
        max_cost_usd=max_cold_ingest_cost_usd,
        max_interactions=total_cold_interactions,
        reserved_recovery_fraction=0.20,
        mandatory_stage_minimums={
            "autonomous_base_clip_card_refresh": total_cold_interactions,
        },
    )
    for record in prior_records:
        cold_ledger.adopt_reconciled_usage(
            stage="autonomous_base_clip_card_refresh",
            model_id=MODEL_ID,
            usage=record["usage"],
        )
    for record in prior_failed_dispatches:
        cold_ledger.adopt_reconciled_usage(
            stage="autonomous_base_clip_card_refresh",
            model_id=MODEL_ID,
            usage=record["usage"],
        )
    write_json(
        cold_root / "budget-report.json",
        {
            "contract_version": "cold-ingest-budget-report-v1",
            "namespace": "cold_ingest",
            "planned_refresh_count": len(planned_refreshes),
            "adopted_refresh_count": len(prior_records),
            "adopted_failed_dispatch_count": len(
                prior_failed_dispatches
            ),
            "budget": cold_ledger.report(),
            "generated_at": utc_now(),
        },
    )

    refresh_records: list[dict[str, Any]] = []
    entrypoint = Path(sys.executable).with_name("jascue-video-lab")
    if not entrypoint.is_file():
        raise DeliveryPipelineBlocked(
            "cannot refresh Clip Cards because the installed CLI entrypoint "
            "is unavailable"
        )
    for clip, clip_root, stale_reason in planned_refreshes:
        refresh_root = (
            library
            / ".refresh"
            / f"{clip.sha256[:16]}-{uuid.uuid4().hex}"
        )
        source_path = Path(clip.path).expanduser().resolve(strict=True)
        command = [
            str(entrypoint),
            "full-clip",
            str(source_path),
            "--output-dir",
            str(refresh_root),
            "--dense-mode",
            "none",
        ]
        run_record_dir = (
            output_dir / "cold-ingest" / clip.sha256[:16]
        )
        run_record_dir.mkdir(parents=True, exist_ok=True)
        try:
            usage = _run_budgeted_planning_stage(
                command=command,
                stage="autonomous_base_clip_card_refresh",
                stage_dir=refresh_root,
                budget_ledger=cold_ledger,
                estimated_text_tokens=12_000,
                media_duration_ms=int(clip.duration_ms),
                media_resolution="low",
                max_output_tokens=4_096,
                thinking_level="low",
            )
        except BaseException as error:
            for raw_path in refresh_root.rglob("*raw_interaction.json"):
                relative = raw_path.relative_to(refresh_root)
                destination = run_record_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_path, destination)
            failed_orchestration = refresh_root / "orchestration.json"
            if failed_orchestration.is_file():
                shutil.copy2(
                    failed_orchestration,
                    run_record_dir / "orchestration.json",
                )
            failed_payload = (
                read_json(failed_orchestration)
                if failed_orchestration.is_file()
                else {}
            )
            failed_usage = (
                failed_payload.get("usage")
                if isinstance(failed_payload, Mapping)
                and isinstance(failed_payload.get("usage"), Mapping)
                else {}
            )
            charged_interaction = bool(
                int(failed_usage.get("request_count") or 0)
                or failed_usage.get("usage_status")
                == "dispatch_recorded_usage_unavailable"
            )
            write_json(
                run_record_dir / "failed-dispatch-record.json",
                {
                    "contract_version": (
                        "base-clip-card-refresh-failure-v1"
                    ),
                    "source_asset_id": f"sha256:{clip.sha256}",
                    "reason_code": stale_reason,
                    "charged_interaction": charged_interaction,
                    "usage": failed_usage,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "generated_at": utc_now(),
                },
            )
            write_json(
                cold_root / "budget-report.json",
                {
                    "contract_version": "cold-ingest-budget-report-v1",
                    "namespace": "cold_ingest",
                    "planned_refresh_count": len(planned_refreshes),
                    "adopted_refresh_count": len(prior_records),
                    "adopted_failed_dispatch_count": len(
                        prior_failed_dispatches
                    ),
                    "completed_refresh_count": len(refresh_records),
                    "budget": cold_ledger.report(),
                    "generated_at": utc_now(),
                },
            )
            raise
        refreshed_reason = _clip_card_entry_stale_reason(
            clip=clip,
            clip_root=refresh_root,
        )
        if refreshed_reason is not None:
            raise DeliveryPipelineBlocked(
                "refreshed Clip Card failed current lineage validation: "
                f"{refreshed_reason}"
            )
        for raw_path in refresh_root.rglob("*raw_interaction.json"):
            relative = raw_path.relative_to(refresh_root)
            destination = run_record_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, destination)
        orchestration_path = refresh_root / "orchestration.json"
        if orchestration_path.is_file():
            shutil.copy2(
                orchestration_path,
                run_record_dir / "orchestration.json",
            )
        archived_path: Path | None = None
        if clip_root.exists():
            archive_root = library / "archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            archived_path = (
                archive_root
                / f"{clip.sha256[:16]}-{uuid.uuid4().hex}"
            )
            clip_root.rename(archived_path)
        clip_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(refresh_root), str(clip_root))
        # Planning journals are created while the artifact lives under the
        # temporary refresh root. Rebind their local provenance after the
        # atomic move so the durable journal never points at a vanished path.
        for journal_path in clip_root.rglob("*.paid_dispatch.json"):
            journal = read_json(journal_path)
            if not isinstance(journal, Mapping):
                raise DeliveryPipelineBlocked(
                    f"invalid cold-ingest dispatch journal: {journal_path}"
                )
            rebound = dict(journal)
            for key in ("raw_artifact_path",):
                raw_value = rebound.get(key)
                if isinstance(raw_value, str):
                    try:
                        relative = Path(raw_value).relative_to(refresh_root)
                    except ValueError:
                        continue
                    rebound[key] = str((clip_root / relative).resolve())
            migration = rebound.get("migration")
            if isinstance(migration, Mapping):
                rebound_migration = dict(migration)
                request_value = rebound_migration.get("request_path")
                if isinstance(request_value, str):
                    try:
                        relative = Path(request_value).relative_to(
                            refresh_root
                        )
                    except ValueError:
                        pass
                    else:
                        rebound_migration["request_path"] = str(
                            (clip_root / relative).resolve()
                        )
                rebound["migration"] = rebound_migration
            write_json(journal_path, rebound)
        record = {
            "contract_version": "base-clip-card-refresh-v1",
            "source_asset_id": f"sha256:{clip.sha256}",
            "reason_code": stale_reason,
            "archived_path": (
                str(archived_path.resolve())
                if archived_path is not None
                else None
            ),
            "refreshed_path": str(clip_root.resolve()),
            "usage": usage,
            "generated_at": utc_now(),
        }
        write_json(run_record_dir / "refresh-record.json", record)
        refresh_records.append(record)
        write_json(
            cold_root / "budget-report.json",
            {
                "contract_version": "cold-ingest-budget-report-v1",
                "namespace": "cold_ingest",
                "planned_refresh_count": len(planned_refreshes),
                "adopted_refresh_count": len(prior_records),
                "adopted_failed_dispatch_count": len(
                    prior_failed_dispatches
                ),
                "completed_refresh_count": len(refresh_records),
                "budget": cold_ledger.report(),
                "generated_at": utc_now(),
            },
        )
    return tuple(refresh_records)


def _archive_stale_clip_card_supplements(
    *,
    catalog_path: Path,
    prepared_library_path: Path | None,
    output_dir: Path,
) -> tuple[dict[str, Any], ...]:
    """Remove stale optional supplements from active lookup without deleting them."""

    catalog = RushesCatalog.model_validate(
        read_json(catalog_path.expanduser().resolve(strict=True))
    )
    library = _prepared_clip_card_library_root(
        catalog_path=catalog_path,
        prepared_library_path=prepared_library_path,
    )
    cards = {
        f"sha256:{clip.sha256}": FullClipCard.model_validate(
            read_json(
                library
                / "clips"
                / clip.sha256[:16]
                / "gemini"
                / "clip-card"
                / "clip_card.json"
            )
        )
        for clip in catalog.clips
    }
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "clip_observation_supplement_zh-TW.txt"
    )
    schema_payload = json.dumps(
        gemini_response_schema(EventObservationSupplement),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_binding = current_supplement_request_binding(
        model_id=MODEL_ID,
        prompt_sha256=sha256_file(prompt_path),
        response_schema_sha256=hashlib.sha256(
            schema_payload.encode("utf-8")
        ).hexdigest(),
    )
    records: list[dict[str, Any]] = []
    for path in tuple(
        sorted(
            library.glob(
                "clips/*/supplements/*/clip-observation-supplement.json"
            )
        )
    ):
        reason: str | None = None
        try:
            supplement = ClipObservationSupplement.model_validate(
                read_json(path)
            )
            card = cards.get(supplement.source_asset_id)
            if card is None:
                raise ValueError("supplement source is outside current catalog")
            validate_supplement(
                card,
                supplement,
                expected_request_binding=expected_binding,
                require_current_lineage=True,
            )
        except Exception as error:
            reason = f"{type(error).__name__}:{error}"
        if reason is None:
            continue
        source_dir = path.parent
        archive_dir = (
            library
            / "archive"
            / "supplements"
            / f"{source_dir.name}-{uuid.uuid4().hex}"
        )
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir.rename(archive_dir)
        records.append(
            {
                "contract_version": "stale-supplement-archive-v1",
                "source_path": str(source_dir),
                "archived_path": str(archive_dir.resolve()),
                "reason": reason,
                "generated_at": utc_now(),
            }
        )
    if records:
        write_json(
            output_dir
            / "cold-ingest"
            / "stale-supplement-archive.json",
            {
                "contract_version": "stale-supplement-archive-set-v1",
                "records": records,
                "generated_at": utc_now(),
            },
        )
    return tuple(records)


def _archive_stale_planning_stage(
    stage_dir: Path,
    *,
    output_dir: Path,
    reason_code: str,
) -> Path | None:
    """Move a stale fixed-path planning stage out of the active namespace."""

    if not stage_dir.exists():
        return None
    archive_dir = (
        output_dir
        / "archive"
        / "planning-lineage"
        / f"{stage_dir.name}-{uuid.uuid4().hex}"
    )
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir.rename(archive_dir)
    write_json(
        archive_dir / "archive-record.json",
        {
            "contract_version": "stale-planning-stage-archive-v1",
            "source_path": str(stage_dir),
            "archived_path": str(archive_dir.resolve()),
            "reason_code": reason_code,
            "generated_at": utc_now(),
        },
    )
    return archive_dir.resolve()


def _direct_plan_binds_current_shortlist(
    *,
    plan_dir: Path,
    shortlist_path: Path,
) -> bool:
    """Verify that a fixed-path direct plan names this exact shortlist."""

    pointer_path = plan_dir / "feature-plan.external-projection.json"
    if not pointer_path.is_file() or not shortlist_path.is_file():
        return False
    try:
        pointer = read_json(pointer_path)
        if not isinstance(pointer, Mapping):
            return False
        record_path = (plan_dir / str(pointer["record_path"])).resolve(
            strict=True
        )
        if plan_dir.resolve() not in record_path.parents:
            return False
        if sha256_file(record_path) != str(pointer["record_sha256"]):
            return False
        record = read_json(record_path)
        if not isinstance(record, Mapping):
            return False
        shortlist_rows = [
            row
            for row in record.get("source_artifacts", [])
            if isinstance(row, Mapping)
            and row.get("role") == "feature_shortlist"
        ]
        if len(shortlist_rows) != 1:
            return False
        row = shortlist_rows[0]
        return (
            Path(str(row["path"])).expanduser().resolve(strict=True)
            == shortlist_path.resolve(strict=True)
            and str(row["sha256"]) == sha256_file(shortlist_path)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _prepare_fresh_autonomous_direct_plan(
    *,
    catalog_path: Path,
    brief_path: Path,
    music_path: Path | None,
    music_duration_ms: int,
    policy_path: Path,
    editorial_contracts_path: Path,
    prepared_library_path: Path | None,
    output_dir: Path,
    budget_ledger: BudgetLedger,
) -> dict[str, Any]:
    """Build the only fresh autonomous plan accepted by feature-cut."""

    policy = AutonomousEditPolicy.model_validate(
        read_json(policy_path.expanduser().resolve(strict=True))
    )
    if (
        policy.media_resolution.base_clip_card != "low"
        or policy.gemini_limits.base_clip_card.thinking_level != "low"
        or policy.gemini_limits.base_clip_card.max_output_tokens != 4_096
    ):
        raise DeliveryPipelineBlocked(
            "Base Clip Card refresh currently supports the signed low / "
            "low-thinking / 4096-token contract only"
        )
    clip_card_refreshes = _refresh_stale_clip_cards(
        catalog_path=catalog_path,
        prepared_library_path=prepared_library_path,
        output_dir=output_dir,
        max_cold_ingest_cost_usd=(
            policy.budget.max_cold_ingest_cost_usd
        ),
    )
    archived_supplements = _archive_stale_clip_card_supplements(
        catalog_path=catalog_path,
        prepared_library_path=prepared_library_path,
        output_dir=output_dir,
    )
    shortlist_dir = output_dir / "retrieval"
    plan_dir = output_dir / "picture" / "gemini-plan"
    if clip_card_refreshes or archived_supplements:
        evidence_change_reason = (
            "base_clip_card_or_supplement_lineage_changed"
        )
        _archive_stale_planning_stage(
            shortlist_dir,
            output_dir=output_dir,
            reason_code=evidence_change_reason,
        )
        _archive_stale_planning_stage(
            plan_dir,
            output_dir=output_dir,
            reason_code=evidence_change_reason,
        )
    library, _catalog, supplements, evidence_characters = (
        _resolve_prepared_clip_card_library(
            catalog_path=catalog_path,
            prepared_library_path=prepared_library_path,
        )
    )
    project_root = Path(__file__).resolve().parents[2]
    shortlist_path = shortlist_dir / "feature-shortlist.json"
    supplement_args = [
        argument
        for path in supplements
        for argument in ("--supplement", str(path))
    ]
    shortlist_command = [
        sys.executable,
        str(project_root / "scripts/shortlist_clip_card_feature_candidates.py"),
        str(catalog_path.expanduser().resolve(strict=True)),
        str(brief_path.expanduser().resolve(strict=True)),
        str(library),
        str(shortlist_dir),
        "--thinking-level",
        "low",
        "--editorial-beat-contracts",
        str(editorial_contracts_path.expanduser().resolve(strict=True)),
        *supplement_args,
    ]
    if shortlist_path.is_file():
        FeatureShortlistPlan.model_validate(read_json(shortlist_path))
        validation = subprocess.run(
            [*shortlist_command, "--reuse-raw-output"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if validation.returncode == 0:
            shortlist_usage = {
                "request_count": 0,
                "reuse_status": "validated_existing_shortlist",
            }
        else:
            _archive_stale_planning_stage(
                shortlist_dir,
                output_dir=output_dir,
                reason_code="shortlist_input_binding_changed",
            )
            _archive_stale_planning_stage(
                plan_dir,
                output_dir=output_dir,
                reason_code="shortlist_input_binding_changed",
            )
            shortlist_usage = _run_budgeted_planning_stage(
                command=shortlist_command,
                stage="autonomous_clip_card_shortlist",
                stage_dir=shortlist_dir,
                budget_ledger=budget_ledger,
                estimated_text_tokens=max(
                    30_000,
                    evidence_characters + 20_000,
                ),
            )
    else:
        if any(shortlist_dir.rglob("*.request.json")):
            raise DeliveryPipelineBlocked(
                "a prior shortlist dispatch has no validated result; "
                "refusing an implicit paid retry"
            )
        shortlist_usage = _run_budgeted_planning_stage(
            command=shortlist_command,
            stage="autonomous_clip_card_shortlist",
            stage_dir=shortlist_dir,
            budget_ledger=budget_ledger,
            estimated_text_tokens=max(
                30_000,
                evidence_characters + 20_000,
            ),
        )
    if not shortlist_path.is_file():
        raise DeliveryPipelineBlocked(
            "autonomous shortlist completed without its typed artifact"
        )
    plan_command = [
        sys.executable,
        str(project_root / "scripts/plan_clip_card_feature_cut.py"),
        str(catalog_path.expanduser().resolve(strict=True)),
        str(brief_path.expanduser().resolve(strict=True)),
        str(library),
        str(plan_dir),
        "--thinking-level",
        "low",
        "--repair-attempts",
        "0",
        "--shortlist",
        str(shortlist_path),
        "--autonomous-policy",
        str(policy_path.expanduser().resolve(strict=True)),
        "--editorial-beat-contracts",
        str(editorial_contracts_path.expanduser().resolve(strict=True)),
        "--candidate-video-evidence",
        "--candidate-video-depth",
        "3",
        "--maximum-candidate-video-seconds",
        "360",
        *supplement_args,
    ]
    if music_path is not None:
        plan_command.extend(
            [
                "--music",
                str(music_path.expanduser().resolve(strict=True)),
            ]
        )
    required = (
        plan_dir / "feature_edit_plan.json",
        plan_dir / "selected-clip-card-evidence.json",
        plan_dir / "feature-plan.external-projection.json",
    )
    if all(path.is_file() for path in required) and (
        _direct_plan_binds_current_shortlist(
            plan_dir=plan_dir,
            shortlist_path=shortlist_path,
        )
    ):
        planning_usage = {
            "request_count": 0,
            "reuse_status": "validated_existing_direct_plan_artifacts",
        }
    else:
        if plan_dir.exists():
            _archive_stale_planning_stage(
                plan_dir,
                output_dir=output_dir,
                reason_code="direct_plan_shortlist_binding_changed",
            )
        if any(plan_dir.rglob("*.request.json")):
            raise DeliveryPipelineBlocked(
                "a prior direct-plan dispatch has no complete typed "
                "projection; refusing an implicit paid retry"
            )
        planning_usage = _run_budgeted_planning_stage(
            command=plan_command,
            stage="autonomous_direct_video_edit_plan",
            stage_dir=plan_dir,
            budget_ledger=budget_ledger,
            estimated_text_tokens=max(
                60_000,
                len(shortlist_path.read_text(encoding="utf-8")) * 3
                + 30_000,
            ),
            media_duration_ms=360_000 + music_duration_ms,
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise DeliveryPipelineBlocked(
            "direct-video planning completed without required projection "
            "artifacts: " + ", ".join(missing)
        )
    return {
        "contract_version": "fresh-autonomous-direct-plan-orchestration-v1",
        "prepared_library": str(library),
        "clip_card_refreshes": list(clip_card_refreshes),
        "cold_ingest_budget_report": str(
            (output_dir / "cold-ingest" / "budget-report.json").resolve()
        ),
        "archived_stale_supplements": list(archived_supplements),
        "supplements": [str(path) for path in supplements],
        "shortlist_path": str(shortlist_path.resolve()),
        "plan_dir": str(plan_dir.resolve()),
        "shortlist_usage": shortlist_usage,
        "planning_usage": planning_usage,
    }


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
    try:
        proposal = validate_music_map_lock_integrity(
            saved_lock,
            music_path=resolved_music,
            lock_path=resolved_lock,
            policy=(
                policy
                if (
                    saved_lock.authority is not None
                    and saved_lock.authority.policy_reference
                    == policy.policy_reference
                )
                else None
            ),
            # Older current-policy locks can be upgraded below.  The newly
            # issued authority always binds both proposal and soundtrack.
            require_authority_source_binding=False,
            validate_authority_policy=(
                saved_lock.authority is not None
                and saved_lock.authority.policy_reference
                == policy.policy_reference
            ),
        )
    except (OSError, ValueError) as exc:
        raise DeliveryPipelineBlocked(
            "cannot refresh MusicMap authority because saved lineage failed "
            f"integrity validation: {exc}"
        ) from exc
    proposal_path = Path(saved_lock.proposal_path).expanduser().resolve(
        strict=True
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
    validate_music_map_lock_integrity(
        refreshed,
        music_path=resolved_music,
        lock_path=refreshed_path,
        policy=policy,
    )
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
    return isinstance(result, AutonomousFinalEditQa) and (
        result.qa_observation_status == "no_blocking_observation"
        and not result.issues
    )


def _validate_autonomous_context_aspect_binding(
    paths: Mapping[str, Path],
    *,
    expected_aspect: str,
    policy: AutonomousEditPolicy | None = None,
) -> None:
    """Reject explicit or selected-window context reuse across aspects."""

    for key, path in paths.items():
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            continue
        declared_aspect = payload.get("aspect")
        if (
            declared_aspect is not None
            and declared_aspect != expected_aspect
        ):
            raise DeliveryPipelineBlocked(
                f"{expected_aspect} autonomous QA cannot consume "
                f"{declared_aspect} context artifact {key}"
            )
        if key != "exact_event_locks":
            continue
        selected_windows = payload.get("selected_windows", [])
        if not isinstance(selected_windows, list):
            continue
        wrong_window_aspects = {
            str(window["aspect"])
            for window in selected_windows
            if isinstance(window, Mapping)
            and window.get("aspect") is not None
            and window.get("aspect") != expected_aspect
        }
        if wrong_window_aspects:
            raise DeliveryPipelineBlocked(
                f"{expected_aspect} autonomous QA cannot consume exact-event "
                f"windows for {sorted(wrong_window_aspects)}"
            )
    if policy is not None:
        cue_plan_path = paths.get("cue_plan")
        if cue_plan_path is None:
            raise DeliveryPipelineBlocked(
                f"{expected_aspect} autonomous context omitted CuePlan"
            )
        try:
            validate_authorized_selected_window_cue_plan(
                cue_plan_path,
                policy=policy,
                expected_aspect=expected_aspect,
            )
        except (OSError, ValueError) as exc:
            raise DeliveryPipelineBlocked(
                f"{expected_aspect} selected-window CuePlan authority failed: "
                f"{exc}"
            ) from exc


def authorize_autonomous_delivery(
    *,
    policy: AutonomousEditPolicy,
    deterministic_qa: DeterministicDeliveryQaReport,
    qa_results: Mapping[str, Any],
    qa_context_hashes_by_aspect: Mapping[str, Mapping[str, str]],
    degradation: AutonomousDegradationManifest,
    input_artifact_hashes: tuple[str, ...],
    final_render_sha256_by_aspect: Mapping[str, str],
    final_manifest_sha256: str,
    brief_sha256: str,
    gemini_interaction_ids: tuple[str, ...] = (),
    music_supplied: bool = True,
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
    if set(final_render_sha256_by_aspect) != set(policy.requested_aspects):
        raise DeliveryPipelineBlocked(
            "final render hashes do not cover every requested aspect"
        )
    if set(qa_context_hashes_by_aspect) != set(policy.requested_aspects):
        raise DeliveryPipelineBlocked(
            "final QA context hashes do not cover every requested aspect"
        )
    for aspect, result in qa_results.items():
        expected_mode = (
            "autonomous_final_16x9"
            if aspect == "16:9"
            else "autonomous_final_9x16"
        )
        if (
            not isinstance(result, AutonomousFinalEditQa)
            or result.mode != expected_mode
        ):
            raise DeliveryPipelineBlocked(
                f"{aspect} final authority received the wrong semantic QA mode"
            )
        if result.render_sha256 != final_render_sha256_by_aspect[aspect]:
            raise DeliveryPipelineBlocked(
                f"{aspect} semantic QA does not bind the current final render"
            )
        if result.manifest_sha256 != final_manifest_sha256:
            raise DeliveryPipelineBlocked(
                f"{aspect} semantic QA does not bind the current render manifest"
            )
        if result.brief_sha256 != brief_sha256:
            raise DeliveryPipelineBlocked(
                f"{aspect} semantic QA does not bind the current brief"
            )
        if result.context_hashes != dict(
            qa_context_hashes_by_aspect[aspect]
        ):
            raise DeliveryPipelineBlocked(
                f"{aspect} semantic QA does not bind the current autonomous "
                "context"
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
    if not omissions_are_policy_authorized(policy, degradation.records):
        raise DeliveryPipelineBlocked(
            "degradation manifest contains an omission or substitution that "
            "the autonomous policy did not authorize"
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
        (
            "music_sync_passed"
            if music_supplied
            else "semantic_visual_cadence_passed"
        ),
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


AutonomousRepairExecutor = Callable[..., Mapping[str, Any]]


def _remaining_run_global_followup_qa_slots(
    *,
    maximum_full_final_qa_calls: int,
    completed_full_final_qa_calls: int,
    requested_initial_aspects: Sequence[str],
    started_initial_aspects: Collection[str],
) -> int:
    """Reserve one initial final-QA observation for every requested aspect."""

    if maximum_full_final_qa_calls < 1:
        raise ValueError("full final QA cap must be positive")
    requested = tuple(dict.fromkeys(requested_initial_aspects))
    started = set(started_initial_aspects)
    if not started <= set(requested):
        raise ValueError("started QA aspects are outside the requested set")
    pending_initial = len(set(requested) - started)
    return (
        maximum_full_final_qa_calls
        - completed_full_final_qa_calls
        - pending_initial
    )


def _write_recovery_execution(
    output_path: Path,
    execution: AutonomousRecoveryExecution,
) -> Path:
    write_json(output_path, execution)
    return output_path.resolve(strict=True)


def _render_manifest_segment_content_hashes(
    *,
    render_manifest_path: Path,
    aspect: str,
) -> dict[str, str]:
    """Resolve the exact segment bytes represented by one aspect manifest."""

    payload = read_json(render_manifest_path.expanduser().resolve(strict=True))
    if not isinstance(payload, Mapping):
        raise ValueError("render manifest must be an object")
    section_name = "vertical" if aspect == "9:16" else "horizontal"
    section = payload.get(section_name)
    if not isinstance(section, Mapping):
        raise ValueError(f"render manifest has no {section_name} section")
    chapters = section.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("render manifest has no segment-bearing chapters")
    hashes: dict[str, str] = {}
    for index, row in enumerate(chapters, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("render manifest chapter must be an object")
        segment_id = str(row.get("segment_id") or f"segment-{index:03d}")
        segment_value = row.get("segment_path")
        if not isinstance(segment_value, str):
            raise ValueError(f"{segment_id} has no immutable segment path")
        segment_path = Path(segment_value).expanduser().resolve(strict=True)
        if segment_id in hashes:
            raise ValueError("render manifest segment IDs must be unique")
        hashes[segment_id] = sha256_file(segment_path)
    return hashes


def _bind_deterministic_evidence_to_delivery(
    *,
    evidence: DeterministicDeliveryEvidence,
    aspect: str,
    policy: AutonomousEditPolicy,
    render_path: Path,
    render_manifest_path: Path,
    delivery_manifest_path: Path,
    music_assembly_manifest_path: Path,
    autonomous_context_paths: Mapping[str, Path],
    output_path: Path,
    changed_segment_ids: tuple[str, ...] = (),
    reused_segment_ids: tuple[str, ...] = (),
) -> tuple[DeterministicDeliveryEvidence, Path]:
    """Persist a delivery-scoped evidence envelope without trusting model IO."""

    context_hashes = {
        key: sha256_file(path.expanduser().resolve(strict=True))
        for key, path in autonomous_context_paths.items()
    }
    segment_hashes = _render_manifest_segment_content_hashes(
        render_manifest_path=render_manifest_path,
        aspect=aspect,
    )
    binding = DeterministicEvidenceCausalBinding(
        render_sha256=sha256_file(render_path),
        render_manifest_sha256=sha256_file(render_manifest_path),
        policy_reference=policy.policy_reference,
        context_hashes=context_hashes,
        delivery_manifest_sha256=sha256_file(delivery_manifest_path),
        music_assembly_manifest_sha256=sha256_file(
            music_assembly_manifest_path
        ),
        segment_content_hashes=segment_hashes,
        changed_segment_ids=changed_segment_ids,
        reused_segment_ids=reused_segment_ids,
    )
    bound = evidence.model_copy(update={"causal_binding": binding})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, bound)
    validate_deterministic_evidence_causal_binding(
        bound,
        render_sha256=sha256_file(render_path),
        render_manifest_sha256=sha256_file(render_manifest_path),
        policy_reference=policy.policy_reference,
        context_hashes=context_hashes,
        delivery_manifest_sha256=sha256_file(delivery_manifest_path),
        music_assembly_manifest_sha256=sha256_file(
            music_assembly_manifest_path
        ),
        segment_content_hashes=segment_hashes,
        changed_segment_ids=changed_segment_ids,
        reused_segment_ids=reused_segment_ids,
    )
    return bound, output_path.resolve(strict=True)


def _feature_manifest_attests_deterministic_evidence(
    *,
    render_manifest_path: Path,
    aspect: str,
    evidence_path: Path,
) -> None:
    """Reject detached evidence before minting final-delivery bindings."""

    manifest = read_json(render_manifest_path.expanduser().resolve(strict=True))
    if not isinstance(manifest, Mapping):
        raise ValueError("render manifest must be an object")
    per_aspect = manifest.get(
        "deterministic_delivery_evidence_by_aspect"
    )
    if isinstance(per_aspect, Mapping):
        reference = per_aspect.get(aspect)
    else:
        reference = manifest.get("deterministic_delivery_evidence")
    if not isinstance(reference, Mapping):
        raise ValueError(
            "render manifest does not attest deterministic evidence"
        )
    resolved_evidence = evidence_path.expanduser().resolve(strict=True)
    if (
        str(reference.get("path")) != str(resolved_evidence)
        or str(reference.get("sha256")) != sha256_file(resolved_evidence)
    ):
        raise ValueError(
            "deterministic evidence is detached from the current render manifest"
        )


def _execute_autonomous_recovery_plan(
    *,
    plan: AutonomousRecoveryPlan,
    policy: AutonomousEditPolicy,
    input_qa_path: Path,
    input_render_path: Path,
    input_render_manifest_path: Path,
    input_delivery_manifest_path: Path,
    input_music_assembly_manifest_path: Path,
    autonomous_context_paths: Mapping[str, Path],
    deterministic_delivery_evidence_path: Path,
    segment_contract: Sequence[Mapping[str, Any]],
    output_dir: Path,
    budget_ledger: BudgetLedger,
    executor: AutonomousRepairExecutor | None,
) -> tuple[AutonomousRecoveryExecution, dict[str, Any] | None]:
    """Execute one bounded recovery or persist why it cannot run safely.

    The delivery layer does not reinterpret a QA suggestion as executable edit
    instructions.  A repair executor must return rebuilt, hash-verifiable final
    media, complete changed/reused segment coverage, updated deterministic
    evidence, and all six autonomous context artifacts.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "recovery-plan.json"
    write_json(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    input_qa = input_qa_path.expanduser().resolve(strict=True)
    input_qa_sha256 = sha256_file(input_qa)
    input_render = input_render_path.expanduser().resolve(strict=True)
    input_render_sha256 = sha256_file(input_render)
    execution_path = output_dir / "recovery-execution.json"

    def nonexecuted(
        status: str,
        reason_code: str,
    ) -> tuple[AutonomousRecoveryExecution, None]:
        execution = AutonomousRecoveryExecution(
            status=status,
            plan_sha256=plan_sha256,
            input_qa_sha256=input_qa_sha256,
            reason_code=reason_code,
            input_render_sha256=input_render_sha256,
            qa_passes_completed=plan.qa_passes_completed,
            semantic_replans_used=plan.semantic_replans_used,
            generated_at=utc_now(),
        )
        _write_recovery_execution(execution_path, execution)
        return execution, None

    if plan.outcome == "complete":
        return nonexecuted("not_required", "semantic_qa_passed")
    if plan.outcome == "blocked":
        return nonexecuted("blocked", "recovery_plan_blocked")
    if executor is None:
        return nonexecuted(
            "unavailable",
            "no_verified_autonomous_repair_executor",
        )

    expected_segment_ids = {
        str(row["segment_id"])
        for row in segment_contract
        if isinstance(row, Mapping)
        and isinstance(row.get("segment_id"), str)
    }
    action_segment_ids = {
        str(action.segment_id)
        for action in plan.actions
        if action.segment_id is not None
    }
    if (
        not expected_segment_ids
        or len(action_segment_ids) != len(plan.actions)
        or not action_segment_ids <= expected_segment_ids
    ):
        return nonexecuted(
            "unavailable",
            "repair_actions_lack_exact_segment_binding",
        )

    semantic_replan_required = any(
        action.requires_semantic_replan for action in plan.actions
    )
    paid_before = int(budget_ledger.report()["committed_interactions"])
    try:
        raw_result = executor(
            plan=plan,
            policy=policy,
            input_render_path=input_render,
            input_render_manifest_path=input_render_manifest_path,
            input_delivery_manifest_path=input_delivery_manifest_path,
            input_music_assembly_manifest_path=(
                input_music_assembly_manifest_path
            ),
            autonomous_context_paths=dict(autonomous_context_paths),
            deterministic_delivery_evidence_path=(
                deterministic_delivery_evidence_path
            ),
            segment_contract=tuple(dict(row) for row in segment_contract),
            output_dir=output_dir / "executor",
            budget_ledger=budget_ledger,
        )
    except Exception as error:
        write_json(
            output_dir / "executor-error.json",
            {
                "contract_version": "autonomous-recovery-executor-error-v1",
                "type": type(error).__name__,
                "message": str(error),
                "generated_at": utc_now(),
            },
        )
        return nonexecuted(
            "unavailable",
            "autonomous_repair_executor_failed",
        )
    paid_after = int(budget_ledger.report()["committed_interactions"])
    paid_delta = paid_after - paid_before
    if paid_delta < 0 or paid_delta > (1 if semantic_replan_required else 0):
        return nonexecuted(
            "unavailable",
            "repair_executor_violated_paid_call_bound",
        )
    if not isinstance(raw_result, Mapping):
        return nonexecuted(
            "unavailable",
            "repair_executor_returned_invalid_contract",
        )

    try:
        output_render = Path(str(raw_result["render_path"])).expanduser().resolve(
            strict=True
        )
        output_render_manifest = Path(
            str(raw_result["render_manifest_path"])
        ).expanduser().resolve(strict=True)
        output_delivery_manifest = Path(
            str(raw_result["delivery_manifest_path"])
        ).expanduser().resolve(strict=True)
        output_music_manifest = Path(
            str(raw_result["music_assembly_manifest_path"])
        ).expanduser().resolve(strict=True)
        output_deterministic_path = Path(
            str(raw_result["deterministic_delivery_evidence_path"])
        ).expanduser().resolve(strict=True)
        output_context_raw = raw_result["autonomous_context_paths"]
        if not isinstance(output_context_raw, Mapping):
            raise ValueError("recovery context paths must be a mapping")
        output_context = {
            str(key): Path(str(path)).expanduser().resolve(strict=True)
            for key, path in output_context_raw.items()
        }
        if set(output_context) != set(autonomous_context_paths):
            raise ValueError("recovery changed autonomous context key set")
        changed_segment_ids = tuple(
            str(value) for value in raw_result["changed_segment_ids"]
        )
        reused_segment_ids = tuple(
            str(value) for value in raw_result["reused_segment_ids"]
        )
        semantic_interaction_ids = tuple(
            str(value)
            for value in raw_result.get(
                "semantic_replan_interaction_ids",
                (),
            )
        )
    except (KeyError, TypeError, ValueError, OSError):
        return nonexecuted(
            "unavailable",
            "repair_executor_returned_invalid_contract",
        )

    changed = set(changed_segment_ids)
    reused = set(reused_segment_ids)
    if (
        not changed
        or changed & reused
        or changed | reused != expected_segment_ids
        or not action_segment_ids <= changed
        or not changed <= action_segment_ids
    ):
        return nonexecuted(
            "unavailable",
            "repair_executor_did_not_prove_changed_segment_only_render",
        )
    if semantic_replan_required:
        if (
            plan.semantic_replans_used != 1
            or len(semantic_interaction_ids) != 1
            or paid_delta not in {0, 1}
        ):
            return nonexecuted(
                "unavailable",
                "semantic_replan_provenance_or_bound_missing",
            )
    elif semantic_interaction_ids or paid_delta:
        return nonexecuted(
            "unavailable",
            "deterministic_repair_used_semantic_paid_work",
        )

    if sha256_file(output_render) == input_render_sha256:
        return nonexecuted(
            "unavailable",
            "repair_executor_did_not_change_final_media",
        )
    try:
        output_deterministic = DeterministicDeliveryEvidence.model_validate(
            read_json(output_deterministic_path)
        )
        output_context_hashes = {
            key: sha256_file(path)
            for key, path in output_context.items()
        }
        output_segment_hashes = _render_manifest_segment_content_hashes(
            render_manifest_path=output_render_manifest,
            aspect=str(output_deterministic.aspect or policy.requested_aspects[0]),
        )
        validate_deterministic_evidence_causal_binding(
            output_deterministic,
            render_sha256=sha256_file(output_render),
            render_manifest_sha256=sha256_file(output_render_manifest),
            policy_reference=policy.policy_reference,
            context_hashes=output_context_hashes,
            delivery_manifest_sha256=sha256_file(output_delivery_manifest),
            music_assembly_manifest_sha256=sha256_file(
                output_music_manifest
            ),
            segment_content_hashes=output_segment_hashes,
            changed_segment_ids=changed_segment_ids,
            reused_segment_ids=reused_segment_ids,
        )
        output_deterministic_report = run_deterministic_delivery_qa(
            output_deterministic,
            policy=policy,
        )
    except Exception:
        return nonexecuted(
            "unavailable",
            "post_repair_deterministic_evidence_invalid",
        )
    if not output_deterministic_report.passed:
        write_json(
            output_dir / "post-repair-deterministic-qa.json",
            output_deterministic_report,
        )
        return nonexecuted(
            "unavailable",
            "post_repair_deterministic_qa_failed",
        )

    execution = AutonomousRecoveryExecution(
        status="executed",
        plan_sha256=plan_sha256,
        input_qa_sha256=input_qa_sha256,
        reason_code="bounded_repair_executed_and_locally_verified",
        input_render_sha256=input_render_sha256,
        output_render_sha256=sha256_file(output_render),
        changed_segment_ids=changed_segment_ids,
        reused_segment_ids=reused_segment_ids,
        deterministic_delivery_evidence_sha256=sha256_file(
            output_deterministic_path
        ),
        semantic_replan_interaction_ids=semantic_interaction_ids,
        qa_passes_completed=plan.qa_passes_completed,
        semantic_replans_used=plan.semantic_replans_used,
        generated_at=utc_now(),
    )
    _write_recovery_execution(execution_path, execution)
    return execution, {
        "render_path": output_render,
        "render_manifest_path": output_render_manifest,
        "delivery_manifest_path": output_delivery_manifest,
        "music_assembly_manifest_path": output_music_manifest,
        "autonomous_context_paths": output_context,
        "deterministic_delivery_evidence_path": output_deterministic_path,
        "deterministic_delivery_evidence": output_deterministic,
        "deterministic_delivery_qa": output_deterministic_report,
        "semantic_replan_interaction_ids": semantic_interaction_ids,
        "recovery_plan_path": plan_path.resolve(strict=True),
        "recovery_execution_path": execution_path.resolve(strict=True),
    }


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
    music_path: Path | None,
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
        "music_sha256": (
            sha256_file(music_path.expanduser().resolve(strict=True))
            if music_path is not None
            else None
        ),
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
    saved_context_by_aspect = manifest.get(
        "autonomous_context_by_aspect"
    )
    result_context_by_aspect = result.get(
        "autonomous_context_paths_by_aspect"
    )
    if (
        saved_context_by_aspect is not None
        or result_context_by_aspect is not None
    ):
        if not isinstance(saved_context_by_aspect, Mapping) or not isinstance(
            result_context_by_aspect,
            Mapping,
        ):
            raise DeliveryPipelineBlocked(
                "picture resume per-aspect autonomous context is malformed"
            )
        if set(saved_context_by_aspect) != set(result_context_by_aspect):
            raise DeliveryPipelineBlocked(
                "picture resume autonomous aspect context keys changed"
            )
        for aspect, raw_paths in result_context_by_aspect.items():
            saved_paths = saved_context_by_aspect.get(aspect)
            if not isinstance(raw_paths, Mapping) or not isinstance(
                saved_paths,
                Mapping,
            ):
                raise DeliveryPipelineBlocked(
                    f"picture resume {aspect} context is malformed"
                )
            if set(raw_paths) != set(saved_paths):
                raise DeliveryPipelineBlocked(
                    f"picture resume {aspect} context keys changed"
                )
            for key, value in raw_paths.items():
                context_path = Path(str(value)).resolve(strict=True)
                manifest_row = saved_paths.get(key)
                if (
                    not isinstance(manifest_row, Mapping)
                    or manifest_row.get("path") != str(context_path)
                    or manifest_row.get("sha256")
                    != sha256_file(context_path)
                ):
                    raise DeliveryPipelineBlocked(
                        f"picture resume rejected because {aspect}/{key} "
                        "context changed"
                    )
    for manifest_key, result_key, label in (
        (
            "deterministic_delivery_evidence_by_aspect",
            "deterministic_delivery_evidence_paths_by_aspect",
            "deterministic evidence",
        ),
        (
            "autonomous_evidence_bundle_by_aspect",
            "autonomous_evidence_bundle_paths_by_aspect",
            "evidence bundle",
        ),
    ):
        saved_by_aspect = manifest.get(manifest_key)
        result_by_aspect = result.get(result_key)
        if saved_by_aspect is None and result_by_aspect is None:
            continue
        if not isinstance(saved_by_aspect, Mapping) or not isinstance(
            result_by_aspect,
            Mapping,
        ):
            raise DeliveryPipelineBlocked(
                f"picture resume per-aspect {label} is malformed"
            )
        if set(saved_by_aspect) != set(result_by_aspect):
            raise DeliveryPipelineBlocked(
                f"picture resume per-aspect {label} keys changed"
            )
        for aspect, raw_path in result_by_aspect.items():
            artifact_path = Path(str(raw_path)).resolve(strict=True)
            manifest_row = saved_by_aspect.get(aspect)
            if (
                not isinstance(manifest_row, Mapping)
                or manifest_row.get("path") != str(artifact_path)
                or manifest_row.get("sha256")
                != sha256_file(artifact_path)
            ):
                raise DeliveryPipelineBlocked(
                    f"picture resume rejected because {aspect} {label} changed"
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


def _migrate_completed_planning_dispatches(
    *,
    stage_dir: Path,
    stage: str,
) -> tuple[Path, ...]:
    """Journal completed child-process calls without dispatching again."""

    if not stage_dir.is_dir():
        return ()
    request_paths = sorted(
        stage_dir.rglob("*.request.json"),
        key=lambda path: (
            0 if ".attempt-" in path.name else 1,
            str(path),
        ),
    )
    migrated: list[Path] = []
    seen_raw_sha256: set[str] = set()
    for request_path in request_paths:
        raw_path = request_path.with_name(
            request_path.name.removesuffix(".request.json")
            + ".raw_interaction.json"
        )
        if not raw_path.is_file():
            continue
        raw_sha256 = sha256_file(raw_path)
        if raw_sha256 in seen_raw_sha256:
            continue
        migrated.append(
            migrate_completed_legacy_paid_dispatch(
                stage=stage,
                request_path=request_path,
                raw_artifact_path=raw_path,
            )
        )
        seen_raw_sha256.add(raw_sha256)
    return tuple(migrated)


def _migrate_completed_warm_dispatches(
    *,
    root: Path,
    allowed_top_level: Collection[str],
) -> tuple[Path, ...]:
    """Migrate only explicitly orchestrated warm-run planning stages."""

    migrated: list[Path] = []
    for orchestration_path in sorted(root.rglob("orchestration.json")):
        relative = orchestration_path.relative_to(root)
        if not relative.parts or relative.parts[0] not in allowed_top_level:
            continue
        payload = read_json(orchestration_path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("contract_version")
            != "autonomous-planning-orchestration-v1"
            or not isinstance(payload.get("stage"), str)
        ):
            continue
        migrated.extend(
            _migrate_completed_planning_dispatches(
                stage_dir=orchestration_path.parent,
                stage=str(payload["stage"]),
            )
        )
    return tuple(migrated)


def _find_unjournaled_warm_paid_artifacts(
    *,
    root: Path,
    allowed_top_level: Collection[str],
    journaled_raw_paths: Collection[Path],
) -> tuple[Path, ...]:
    """Find raw evidence not covered by a journal or exact byte alias."""

    resolved_journaled = {
        path.resolve() for path in journaled_raw_paths
    }
    journaled_sha256 = {
        sha256_file(path)
        for path in resolved_journaled
        if path.is_file()
    }
    return tuple(
        path
        for path in root.rglob("*raw_interaction.json")
        if (
            path.relative_to(root).parts
            and path.relative_to(root).parts[0] in allowed_top_level
            and path.resolve() not in resolved_journaled
            and sha256_file(path) not in journaled_sha256
        )
    )


def run_feature_delivery_pipeline(
    *,
    feature_cut_kwargs: Mapping[str, Any],
    brief_path: Path,
    music_path: Path | None,
    music_lock_path: Path | None,
    output_dir: Path,
    model_id: str = MODEL_ID,
    execution_profile: str = "production_review",
    reuse_picture_result: bool = False,
    autonomous_policy_path: Path | None = None,
    max_gemini_cost_usd: float | None = None,
    autonomous_context_paths: Mapping[str, Path] | None = None,
    deterministic_delivery_evidence_path: Path | None = None,
    autonomous_context_paths_by_aspect: (
        Mapping[str, Mapping[str, Path]] | None
    ) = None,
    deterministic_delivery_evidence_paths_by_aspect: (
        Mapping[str, Path] | None
    ) = None,
    editorial_beat_contracts_path: Path | None = None,
    prepared_clip_card_library_path: Path | None = None,
    autonomous_repair_executor: AutonomousRepairExecutor | None = None,
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
    if (music_path is None) != (music_lock_path is None):
        raise DeliveryPipelineBlocked(
            "music and MusicMap lock must either both be supplied or both be "
            "omitted"
        )
    music_supplied = music_path is not None
    if (
        profile
        not in {
            FeatureCutExecutionProfile.AUTONOMOUS_STRICT,
            FeatureCutExecutionProfile.AUTONOMOUS_BEST_EFFORT,
        }
        and not music_supplied
    ):
        raise DeliveryPipelineBlocked(
            "review feature-delivery requires music and a MusicMap lock; "
            "silent delivery is an autonomous-only extension"
        )
    policy: AutonomousEditPolicy | None = None
    budget_ledger: BudgetLedger | None = None
    deterministic_evidence: DeterministicDeliveryEvidence | None = None
    deterministic_report: DeterministicDeliveryQaReport | None = None
    resolved_autonomous_context: dict[str, Path] = {}
    resolved_autonomous_context_by_aspect: dict[
        str, dict[str, Path]
    ] = {}
    resolved_deterministic_evidence_by_aspect: dict[str, Path] = {}
    deterministic_evidence_by_aspect: dict[
        str, DeterministicDeliveryEvidence
    ] = {}
    deterministic_report_by_aspect: dict[
        str, DeterministicDeliveryQaReport
    ] = {}
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
        if model_id != policy.model_id or MODEL_ID != policy.model_id:
            raise DeliveryPipelineBlocked(
                "autonomous delivery model does not match the signed policy"
            )
        expected_profile = AutonomousExecutionProfile(profile.value)
        if policy.execution_profile != expected_profile:
            raise DeliveryPipelineBlocked(
                "execution profile does not match the autonomous policy"
            )
        # A scoped semantic replan is executed inside the hash-bound
        # selected-window production frontier.  It is not the post-render QA
        # repair executor represented by ``autonomous_repair_executor``.
        # The former is always available through feature-cut, capped by the
        # signed policy, and fails closed when its immutable option frontier
        # is absent; conflating the two used to block a valid run before any
        # paid dispatch.
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
            # This is a circuit-breaker margin, not callable work. It keeps
            # the frozen mandatory frontier below the policy's absolute cap
            # even when a provider dispatch becomes ambiguous.
            interaction_guard=2,
            reserved_recovery_fraction=(
                policy.budget.reserved_recovery_fraction
            ),
            mandatory_stage_minimums=_mandatory_paid_stage_minimums(
                policy=policy,
                editorial_beat_contracts_path=editorial_beat_contracts_path,
            ),
            mandatory_stage_cost_holds=_mandatory_paid_stage_cost_holds(
                policy=policy,
                editorial_beat_contracts_path=editorial_beat_contracts_path,
            ),
        )
        # Cold Clip Card ingest has its own budget namespace. Warm edit resume
        # adopts only stable node journals and never infers a stage from a
        # directory or filename.
        warm_paid_namespaces = {
            "retrieval",
            "picture",
            "aspects",
            "audition",
        }
        _migrate_completed_warm_dispatches(
            root=resolved_output,
            allowed_top_level=warm_paid_namespaces,
        )
        (
            adopted_dispatch_journals,
            journaled_raw_paths,
        ) = adopt_paid_dispatch_journal_state(
            budget_ledger=budget_ledger,
            root=resolved_output,
            allowed_top_level=warm_paid_namespaces,
        )
        conservative_dispatches = [
            node
            for node in adopted_dispatch_journals
            if node["adoption_basis"] == "conservative_worst_case"
        ]
        unjournaled_warm_paid_artifacts = (
            _find_unjournaled_warm_paid_artifacts(
                root=resolved_output,
                allowed_top_level=warm_paid_namespaces,
                journaled_raw_paths=journaled_raw_paths,
            )
        )
        if unjournaled_warm_paid_artifacts:
            raise DeliveryPipelineBlocked(
                "warm-run paid artifacts lack stable dispatch-node journals: "
                + ", ".join(
                    str(path.relative_to(resolved_output))
                    for path in unjournaled_warm_paid_artifacts[:5]
                )
            )
        prior_usage = summarize_usage_and_list_price(resolved_output)
        required_context_keys = {
            "editorial_beat_contracts",
            "music_map",
            "cue_plan",
            "exact_event_locks",
            "sequence_optimization",
            "reuse_degradation",
            "resolved_timeline",
        }
        optional_context_keys: set[str] = set()
        if (
            autonomous_context_paths is not None
            and autonomous_context_paths_by_aspect is not None
        ):
            raise DeliveryPipelineBlocked(
                "autonomous context must use either the flat single-aspect "
                "shape or the per-aspect shape, never both"
            )
        if (
            deterministic_delivery_evidence_path is not None
            and deterministic_delivery_evidence_paths_by_aspect is not None
        ):
            raise DeliveryPipelineBlocked(
                "deterministic evidence must use either the flat single-aspect "
                "shape or the per-aspect shape, never both"
            )
        if len(requested) > 1 and (
            autonomous_context_paths is not None
            or deterministic_delivery_evidence_path is not None
        ):
            raise DeliveryPipelineBlocked(
                "multi-aspect autonomous delivery rejects ambiguous flat "
                "context/evidence; supply independently bound per-aspect maps"
            )
        if autonomous_context_paths_by_aspect is not None:
            if set(autonomous_context_paths_by_aspect) != requested:
                raise DeliveryPipelineBlocked(
                    "per-aspect autonomous context keys do not match the "
                    "requested aspects"
                )
            for aspect_key, supplied_paths in (
                autonomous_context_paths_by_aspect.items()
            ):
                supplied_keys = set(supplied_paths)
                if (
                    not required_context_keys.issubset(supplied_keys)
                    or supplied_keys
                    - required_context_keys
                    - optional_context_keys
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} autonomous final-QA context keys are "
                        "incomplete or unknown"
                    )
                resolved_paths = {
                    key: path.expanduser().resolve(strict=True)
                    for key, path in supplied_paths.items()
                }
                _validate_autonomous_context_aspect_binding(
                    resolved_paths,
                    expected_aspect=aspect_key,
                    policy=policy,
                )
                preflight_degradation = (
                    AutonomousDegradationManifest.model_validate(
                        read_json(resolved_paths["reuse_degradation"])
                    )
                )
                if (
                    preflight_degradation.policy_reference
                    != policy.policy_reference
                    or (
                        preflight_degradation.aspect is not None
                        and preflight_degradation.aspect != aspect_key
                    )
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} degradation manifest does not bind the "
                        "policy and aspect"
                    )
                resolved_autonomous_context_by_aspect[aspect_key] = (
                    resolved_paths
                )
        elif autonomous_context_paths is not None:
            supplied_keys = set(autonomous_context_paths)
            if (
                not required_context_keys.issubset(supplied_keys)
                or supplied_keys
                - required_context_keys
                - optional_context_keys
            ):
                raise DeliveryPipelineBlocked(
                    "autonomous final-QA context keys are incomplete or unknown"
                )
            only_aspect = next(iter(requested))
            resolved_autonomous_context = {
                key: path.expanduser().resolve(strict=True)
                for key, path in autonomous_context_paths.items()
            }
            _validate_autonomous_context_aspect_binding(
                resolved_autonomous_context,
                expected_aspect=only_aspect,
                policy=policy,
            )
            preflight_degradation = (
                AutonomousDegradationManifest.model_validate(
                    read_json(
                        resolved_autonomous_context["reuse_degradation"]
                    )
                )
            )
            if (
                preflight_degradation.policy_reference
                != policy.policy_reference
                or (
                    preflight_degradation.aspect is not None
                    and preflight_degradation.aspect != only_aspect
                )
            ):
                raise DeliveryPipelineBlocked(
                    "degradation manifest does not bind the policy and aspect"
                )
            resolved_autonomous_context_by_aspect[only_aspect] = dict(
                resolved_autonomous_context
            )
        elif editorial_beat_contracts_path is None:
            raise DeliveryPipelineBlocked(
                "autonomous delivery requires editorial beat contracts so "
                "feature-cut can generate selected-window context"
            )
        resolved_deterministic_evidence: Path | None = None
        if deterministic_delivery_evidence_paths_by_aspect is not None:
            if set(
                deterministic_delivery_evidence_paths_by_aspect
            ) != requested:
                raise DeliveryPipelineBlocked(
                    "per-aspect deterministic evidence keys do not match the "
                    "requested aspects"
                )
            for aspect_key, evidence_path in (
                deterministic_delivery_evidence_paths_by_aspect.items()
            ):
                resolved_path = evidence_path.expanduser().resolve(
                    strict=True
                )
                evidence = DeterministicDeliveryEvidence.model_validate(
                    read_json(resolved_path)
                )
                if (
                    evidence.aspect is not None
                    and evidence.aspect != aspect_key
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} autonomous QA cannot consume "
                        f"{evidence.aspect} deterministic evidence"
                    )
                report = run_deterministic_delivery_qa(
                    evidence,
                    policy=policy,
                )
                if not report.passed:
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} deterministic autonomous gates failed "
                        "before paid work: "
                        + ", ".join(report.failure_codes)
                    )
                resolved_deterministic_evidence_by_aspect[aspect_key] = (
                    resolved_path
                )
                deterministic_evidence_by_aspect[aspect_key] = evidence
                deterministic_report_by_aspect[aspect_key] = report
        elif deterministic_delivery_evidence_path is not None:
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
            only_aspect = next(iter(requested))
            if (
                deterministic_evidence.aspect is not None
                and deterministic_evidence.aspect != only_aspect
            ):
                raise DeliveryPipelineBlocked(
                    f"{only_aspect} autonomous QA cannot consume "
                    f"{deterministic_evidence.aspect} deterministic evidence"
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
            resolved_deterministic_evidence_by_aspect[only_aspect] = (
                resolved_deterministic_evidence
            )
            deterministic_evidence_by_aspect[only_aspect] = (
                deterministic_evidence
            )
            deterministic_report_by_aspect[only_aspect] = (
                deterministic_report
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
                            adopted_dispatch_journals
                    ),
                    "included_conservative_dispatches": len(
                        conservative_dispatches
                    ),
                    "included_non_subprocess_dispatch_journals": len(
                        adopted_dispatch_journals
                    ),
                        "excluded_pre_policy_requests": (
                            prior_usage["request_count"]
                            - len(journaled_raw_paths)
                        ),
                        "interpretation": (
                            "warm edit usage is adopted only from stable dispatch "
                            "journals; cold ingest remains a separate budget "
                            "namespace"
                        ),
                },
                "autonomous_context_paths": {
                    key: str(path)
                    for key, path in resolved_autonomous_context.items()
                },
                "autonomous_context_paths_by_aspect": {
                    aspect_key: {
                        key: str(path) for key, path in paths.items()
                    }
                    for aspect_key, paths in (
                        resolved_autonomous_context_by_aspect.items()
                    )
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
                "deterministic_delivery_evidence_paths_by_aspect": {
                    aspect_key: str(path)
                    for aspect_key, path in (
                        resolved_deterministic_evidence_by_aspect.items()
                    )
                },
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
        if policy is not None and music_path is not None:
            assert music_lock_path is not None
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
                    or (
                        next(
                            iter(
                                resolved_autonomous_context_by_aspect.values()
                            )
                        )["editorial_beat_contracts"]
                        if resolved_autonomous_context_by_aspect
                        else None
                    )
                )
            if (
                not reuse_picture_result
                and not bool(kwargs.get("reuse_feature_plan"))
                and not bool(kwargs.get("reuse_feature_plan_raw_output"))
            ):
                assert budget_ledger is not None
                contracts_path = kwargs["editorial_beat_contracts_path"]
                if contracts_path is None:
                    raise DeliveryPipelineBlocked(
                        "fresh autonomous planning requires editorial beat "
                        "contracts"
                    )
                planning_orchestration = (
                    _prepare_fresh_autonomous_direct_plan(
                        catalog_path=Path(kwargs["catalog_path"]),
                        brief_path=brief_path,
                        music_path=music_path,
                        music_duration_ms=(
                            MusicMapLock.model_validate(
                                read_json(effective_music_lock_path)
                            ).duration_ms
                            if effective_music_lock_path is not None
                            else 0
                        ),
                        policy_path=Path(autonomous_policy_path),
                        editorial_contracts_path=Path(contracts_path),
                        prepared_library_path=(
                            prepared_clip_card_library_path
                        ),
                        output_dir=resolved_output,
                        budget_ledger=budget_ledger,
                    )
                )
                kwargs["reuse_feature_plan"] = True
                outputs["fresh_autonomous_planning"] = (
                    planning_orchestration
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
        if policy is not None:
            presentation_authorities = feature_result.get(
                "presentation_authority_paths_by_aspect"
            )
            feature_cut_authority_value = feature_result.get(
                "feature_cut_authority_path"
            )
            if (
                not isinstance(presentation_authorities, Mapping)
                or set(presentation_authorities) != requested
                or not isinstance(feature_cut_authority_value, str)
            ):
                raise DeliveryPipelineBlocked(
                    "autonomous feature-cut omitted policy-bound presentation "
                    "or feature-cut authority"
                )
            feature_manifest_path = Path(
                str(feature_result["manifest_path"])
            ).expanduser().resolve(strict=True)
            feature_manifest = read_json(feature_manifest_path)
            saved_presentation_authorities = (
                feature_manifest.get("presentation_authority_by_aspect")
                if isinstance(feature_manifest, Mapping)
                else None
            )
            if (
                not isinstance(saved_presentation_authorities, Mapping)
                or set(saved_presentation_authorities) != requested
            ):
                raise DeliveryPipelineBlocked(
                    "render manifest presentation authority is incomplete"
                )
            for aspect_key, raw_path in presentation_authorities.items():
                presentation_path = Path(str(raw_path)).expanduser().resolve(
                    strict=True
                )
                saved_row = saved_presentation_authorities.get(aspect_key)
                if (
                    not isinstance(saved_row, Mapping)
                    or saved_row.get("path") != str(presentation_path)
                    or saved_row.get("sha256")
                    != sha256_file(presentation_path)
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} presentation authority differs from "
                        "render manifest"
                    )
                try:
                    presentation_artifact = validate_policy_decision_artifact(
                        presentation_path,
                        policy=policy,
                        expected_scope="reframe",
                        expected_aspect=str(aspect_key),
                    )
                except (OSError, ValueError) as exc:
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} presentation authority failed: {exc}"
                    ) from exc
                presentation_proposal = read_json(
                    Path(
                        str(presentation_artifact["proposal_path"])
                    ).expanduser().resolve(strict=True)
                )
                result_output_key = (
                    "horizontal_output"
                    if aspect_key == "16:9"
                    else "vertical_output"
                )
                result_output = Path(
                    str(feature_result.get(result_output_key) or "")
                ).expanduser().resolve()
                if (
                    not isinstance(presentation_proposal, Mapping)
                    or presentation_proposal.get("final_output_path")
                    != str(result_output)
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} presentation authority does not bind "
                        "the feature-cut output"
                    )
            feature_cut_authority_path = Path(
                feature_cut_authority_value
            ).expanduser().resolve(strict=True)
            saved_feature_authority = (
                feature_manifest.get("feature_cut_authority")
                if isinstance(feature_manifest, Mapping)
                else None
            )
            if (
                not isinstance(saved_feature_authority, Mapping)
                or saved_feature_authority.get("path")
                != str(feature_cut_authority_path)
                or saved_feature_authority.get("sha256")
                != sha256_file(feature_cut_authority_path)
            ):
                raise DeliveryPipelineBlocked(
                    "feature-cut authority differs from render manifest"
                )
            try:
                validate_policy_decision_artifact(
                    feature_cut_authority_path,
                    policy=policy,
                    expected_scope="feature_cut",
                    expected_aspect=None,
                )
            except (OSError, ValueError) as exc:
                raise DeliveryPipelineBlocked(
                    f"feature-cut authority failed: {exc}"
                ) from exc
            generated_context_by_aspect = feature_result.get(
                "autonomous_context_paths_by_aspect"
            )
            generated_evidence_by_aspect = feature_result.get(
                "deterministic_delivery_evidence_paths_by_aspect"
            )
            if isinstance(generated_context_by_aspect, Mapping):
                if set(generated_context_by_aspect) != requested:
                    raise DeliveryPipelineBlocked(
                        "feature-cut per-aspect context does not match the "
                        "requested aspects"
                    )
                resolved_autonomous_context_by_aspect = {}
                for aspect_key, raw_paths in (
                    generated_context_by_aspect.items()
                ):
                    raw_keys = (
                        set(raw_paths)
                        if isinstance(raw_paths, Mapping)
                        else set()
                    )
                    if (
                        not isinstance(raw_paths, Mapping)
                        or not required_context_keys.issubset(raw_keys)
                        or raw_keys
                        - required_context_keys
                        - optional_context_keys
                    ):
                        raise DeliveryPipelineBlocked(
                            f"generated {aspect_key} autonomous context is "
                            "incomplete"
                        )
                    resolved_autonomous_context_by_aspect[str(aspect_key)] = {
                        str(key): Path(str(path))
                        .expanduser()
                        .resolve(strict=True)
                        for key, path in raw_paths.items()
                    }
                    _validate_autonomous_context_aspect_binding(
                        resolved_autonomous_context_by_aspect[
                            str(aspect_key)
                        ],
                        expected_aspect=str(aspect_key),
                        policy=policy,
                    )
            else:
                generated_context = feature_result.get(
                    "autonomous_context_paths"
                )
                if len(requested) != 1 or not isinstance(
                    generated_context,
                    Mapping,
                ):
                    raise DeliveryPipelineBlocked(
                        "multi-aspect feature-cut must persist independently "
                        "bound per-aspect context"
                    )
                generated_keys = set(generated_context)
                if (
                    not required_context_keys.issubset(generated_keys)
                    or generated_keys
                    - required_context_keys
                    - optional_context_keys
                ):
                    raise DeliveryPipelineBlocked(
                        "generated autonomous context is incomplete"
                    )
                only_aspect = next(iter(requested))
                resolved_autonomous_context = {
                    str(key): Path(str(path))
                    .expanduser()
                    .resolve(strict=True)
                    for key, path in generated_context.items()
                }
                _validate_autonomous_context_aspect_binding(
                    resolved_autonomous_context,
                    expected_aspect=only_aspect,
                    policy=policy,
                )
                resolved_autonomous_context_by_aspect[only_aspect] = dict(
                    resolved_autonomous_context
                )
            if isinstance(generated_evidence_by_aspect, Mapping):
                if set(generated_evidence_by_aspect) != requested:
                    raise DeliveryPipelineBlocked(
                        "feature-cut per-aspect deterministic evidence does "
                        "not match the requested aspects"
                    )
                resolved_deterministic_evidence_by_aspect = {
                    str(aspect_key): Path(str(path))
                    .expanduser()
                    .resolve(strict=True)
                    for aspect_key, path in (
                        generated_evidence_by_aspect.items()
                    )
                }
            else:
                generated_evidence = feature_result.get(
                    "deterministic_delivery_evidence_path"
                )
                if len(requested) != 1 or not isinstance(
                    generated_evidence,
                    str,
                ):
                    raise DeliveryPipelineBlocked(
                        "multi-aspect feature-cut must persist independently "
                        "bound per-aspect deterministic evidence"
                    )
                only_aspect = next(iter(requested))
                resolved_deterministic_evidence = Path(
                    generated_evidence
                ).expanduser().resolve(strict=True)
                resolved_deterministic_evidence_by_aspect[only_aspect] = (
                    resolved_deterministic_evidence
                )
            deterministic_evidence_by_aspect = {}
            deterministic_report_by_aspect = {}
            aspect_failure_codes: list[str] = []
            for aspect_key in sorted(requested):
                evidence_path = (
                    resolved_deterministic_evidence_by_aspect[aspect_key]
                )
                evidence = DeterministicDeliveryEvidence.model_validate(
                    read_json(evidence_path)
                )
                if (
                    evidence.aspect is not None
                    and evidence.aspect != aspect_key
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} autonomous QA cannot consume "
                        f"{evidence.aspect} deterministic evidence"
                    )
                report = run_deterministic_delivery_qa(
                    evidence,
                    policy=policy,
                )
                deterministic_evidence_by_aspect[aspect_key] = evidence
                deterministic_report_by_aspect[aspect_key] = report
                aspect_failure_codes.extend(
                    f"{aspect_key}:{code}"
                    for code in report.failure_codes
                )
            deterministic_failure_codes = tuple(aspect_failure_codes)
            if len(requested) == 1:
                only_aspect = next(iter(requested))
                resolved_autonomous_context = dict(
                    resolved_autonomous_context_by_aspect[only_aspect]
                )
                resolved_deterministic_evidence = (
                    resolved_deterministic_evidence_by_aspect[only_aspect]
                )
                deterministic_evidence = (
                    deterministic_evidence_by_aspect[only_aspect]
                )
                deterministic_report = (
                    deterministic_report_by_aspect[only_aspect]
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

        resolved_music = (
            music_path.expanduser().resolve(strict=True)
            if music_path is not None
            else None
        )
        resolved_lock_path = (
            effective_music_lock_path.expanduser().resolve(strict=True)
            if effective_music_lock_path is not None
            else None
        )
        music_lock: MusicMapLock | None = None
        if resolved_music is not None:
            assert resolved_lock_path is not None
            music_lock = MusicMapLock.model_validate(
                read_json(resolved_lock_path)
            )
            if music_lock.music_id != f"sha256:{sha256_file(resolved_music)}":
                raise DeliveryPipelineBlocked(
                    "reviewed MusicMap lock does not bind the supplied soundtrack"
                )

        render_manifest_path = Path(feature_result["manifest_path"]).resolve(
            strict=True
        )
        final_results: dict[str, Any] = {}
        qa_results_by_aspect: dict[str, Any] = {}
        qa_context_hashes_by_aspect: dict[str, dict[str, str]] = {}
        qa_interaction_ids: list[str] = []
        autonomous_final_qa_calls_completed = 0
        autonomous_semantic_replans_used = 0
        autonomous_initial_aspect_order = tuple(
            aspect_ratio
            for aspect_key, aspect_ratio in (
                ("horizontal", "16:9"),
                ("vertical", "9:16"),
            )
            if feature_result.get(f"{aspect_key}_output") is not None
        )
        autonomous_initial_aspects_started: set[str] = set()
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
            resolved_timeline_path = (
                resolved_autonomous_context_by_aspect
                .get(aspect_ratio, {})
                .get("resolved_timeline")
            )
            resolved_timeline_sha256 = (
                sha256_file(resolved_timeline_path)
                if resolved_timeline_path is not None
                and resolved_timeline_path.is_file()
                else None
            )
            if expected_timeline is not None:
                expected_plan = expected_timeline.get(
                    "plan_definition",
                    {},
                )
                sample_rate = int(
                    expected_plan.get("master_sample_rate") or 0
                )
                output_samples = int(
                    expected_plan.get("output_duration_samples") or 0
                )
                planned_music_duration_ms = (
                    round(output_samples * 1000 / sample_rate)
                    if sample_rate > 0 and output_samples > 0
                    else -1
                )
                if (
                    planned_music_duration_ms < 0
                    or abs(
                        planned_music_duration_ms
                        - picture_duration_ms
                    )
                    > 100
                ):
                    # Runtime substitutions changed the authoritative picture
                    # duration. Re-plan locally from the locked MusicMap rather
                    # than reusing a stale planned-duration soundtrack.
                    expected_timeline = None
            aspect_dir = resolved_output / (
                "audition" if deterministic_failure_codes else "aspects"
            ) / aspect_key
            assembly_key = hashlib.sha256(
                (
                    (
                        "feature-delivery-music-assembly-v2:"
                        if music_supplied
                        else "feature-delivery-picture-only-v1:"
                    )
                    + f"{sha256_file(picture)}:"
                    + f"{picture_duration_ms}:"
                    + (
                        f"{sha256_file(resolved_lock_path)}:"
                        if resolved_lock_path is not None
                        else "no-music:"
                    )
                    + (
                        sha256_file(music_timeline_path)
                        if expected_timeline
                        else "unplanned"
                    )
                    + ":"
                    + (
                        resolved_timeline_sha256
                        if resolved_timeline_sha256 is not None
                        else "no-resolved-timeline"
                    )
                ).encode("utf-8")
            ).hexdigest()
            assembly_dir = (
                aspect_dir / "music-assembly" / "runs" / assembly_key
            )
            rendered_music = None
            if not music_supplied:
                delivery = assemble_picture_only_delivery(
                    picture_path=picture,
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
                    aspect_ratio=aspect_ratio,
                    artifact_bindings={
                        "feature_render_manifest_sha256": sha256_file(
                            render_manifest_path
                        ),
                        "audio_policy": "explicitly_absent",
                        **(
                            {
                                "resolved_timeline_sha256": (
                                    resolved_timeline_sha256
                                )
                            }
                            if resolved_timeline_sha256 is not None
                            else {}
                        ),
                    },
                )
            elif expected_timeline is not None:
                assert resolved_music is not None
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
            elif music_lock is not None:
                assert resolved_music is not None
                assert resolved_lock_path is not None
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
            else:
                raise DeliveryPipelineBlocked(
                    "music delivery state is internally inconsistent"
                )
            if rendered_music is not None:
                assert resolved_lock_path is not None
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
                        "music_map_lock_sha256": sha256_file(
                            resolved_lock_path
                        ),
                        **(
                            {
                                "resolved_timeline_sha256": (
                                    resolved_timeline_sha256
                                )
                            }
                            if resolved_timeline_sha256 is not None
                            else {}
                        ),
                    },
                )
            if policy is not None:
                source_evidence_path = (
                    resolved_deterministic_evidence_by_aspect.get(
                        aspect_ratio
                    )
                )
                source_evidence = deterministic_evidence_by_aspect.get(
                    aspect_ratio
                )
                source_context = (
                    resolved_autonomous_context_by_aspect.get(aspect_ratio)
                )
                if (
                    source_evidence_path is None
                    or source_evidence is None
                    or source_context is None
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_ratio} has no deterministic evidence "
                        "available for final delivery binding"
                    )
                final_media_authority_path = (
                    rendered_music.manifest_path
                    if rendered_music is not None
                    else delivery.manifest_path
                )
                try:
                    _feature_manifest_attests_deterministic_evidence(
                        render_manifest_path=render_manifest_path,
                        aspect=aspect_ratio,
                        evidence_path=source_evidence_path,
                    )
                    bound_evidence, bound_evidence_path = (
                        _bind_deterministic_evidence_to_delivery(
                            evidence=source_evidence,
                            aspect=aspect_ratio,
                            policy=policy,
                            render_path=delivery.output_path,
                            render_manifest_path=render_manifest_path,
                            delivery_manifest_path=delivery.manifest_path,
                            music_assembly_manifest_path=(
                                final_media_authority_path
                            ),
                            autonomous_context_paths=source_context,
                            output_path=(
                                aspect_dir
                                / "deterministic-delivery-evidence.bound.json"
                            ),
                        )
                    )
                except (OSError, TypeError, ValueError) as error:
                    raise DeliveryPipelineBlocked(
                        f"{aspect_ratio} deterministic evidence is not "
                        f"causally bound to the current delivery: {error}"
                    ) from error
                resolved_deterministic_evidence_by_aspect[aspect_ratio] = (
                    bound_evidence_path
                )
                deterministic_evidence_by_aspect[aspect_ratio] = (
                    bound_evidence
                )
                deterministic_report_by_aspect[aspect_ratio] = (
                    run_deterministic_delivery_qa(
                        bound_evidence,
                        policy=policy,
                    )
                )
            if deterministic_failure_codes:
                final_results[aspect_key] = {
                    "audition_output": str(delivery.output_path),
                    "audition_output_sha256": sha256_file(
                        delivery.output_path
                    ),
                    "audition_manifest": str(delivery.manifest_path),
                    "media_assembly_manifest": str(
                        rendered_music.manifest_path
                        if rendered_music is not None
                        else delivery.manifest_path
                    ),
                    "delivery_eligible": False,
                    "interpretation": (
                        "playable review artifact preserved after local "
                        "autonomous gates blocked delivery"
                    ),
                }
                continue
            assert client is not None
            qa_dir = aspect_dir / "final-qa"
            if policy is not None:
                qa_mode = (
                    "autonomous_final_16x9"
                    if aspect_ratio == "16:9"
                    else "autonomous_final_9x16"
                )
                aspect_autonomous_context = (
                    resolved_autonomous_context_by_aspect.get(aspect_ratio)
                )
                aspect_deterministic_path = (
                    resolved_deterministic_evidence_by_aspect.get(
                        aspect_ratio
                    )
                )
                aspect_deterministic_evidence = (
                    deterministic_evidence_by_aspect.get(aspect_ratio)
                )
                aspect_deterministic_report = (
                    deterministic_report_by_aspect.get(aspect_ratio)
                )
                if (
                    aspect_autonomous_context is None
                    or aspect_deterministic_path is None
                    or aspect_deterministic_evidence is None
                    or aspect_deterministic_report is None
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_ratio} final QA has no matching autonomous "
                        "context/evidence authority"
                    )
            else:
                aspect_autonomous_context = None
                aspect_deterministic_path = None
                aspect_deterministic_evidence = None
                aspect_deterministic_report = None
            prepared = prepare_final_edit_qa(
                mode=qa_mode,
                render_path=delivery.output_path,
                manifest_path=render_manifest_path,
                output_dir=qa_dir,
                model_id=model_id,
                brief_path=(
                    brief_path
                    if qa_mode
                    in {
                        "canonical_16x9",
                        "autonomous_final_16x9",
                        "autonomous_final_9x16",
                    }
                    else None
                ),
                crop_include_audio=(
                    music_supplied and qa_mode != "crop_only_9x16"
                ),
                music_supplied=(
                    music_supplied if policy is not None else None
                ),
                autonomous_policy=policy,
                autonomous_context_paths=(
                    aspect_autonomous_context
                    if qa_mode
                    in {
                        "autonomous_final_16x9",
                        "autonomous_final_9x16",
                    }
                    else None
                ),
            )
            if policy is not None:
                qa_context_hashes_by_aspect[aspect_ratio] = dict(
                    prepared.autonomous_context_hashes
                )
            uploaded, file_reused = client.ensure_video_upload(
                prepared.proxy_path,
                qa_dir / "file-api" / prepared.input_hashes["proxy_sha256"],
            )
            if (
                policy is not None
                and autonomous_final_qa_calls_completed
                >= _maximum_full_final_qa_calls(policy)
            ):
                raise DeliveryPipelineBlocked(
                    "run-global full final QA pass limit reached before "
                    f"{aspect_ratio} initial QA"
                )
            qa = execute_final_edit_qa(
                prepared=prepared,
                client=client.client,
                uploaded_video=uploaded,
                output_dir=qa_dir,
                budget_ledger=budget_ledger,
                recovery_call=False,
            )
            if policy is not None:
                autonomous_final_qa_calls_completed += 1
                autonomous_initial_aspects_started.add(aspect_ratio)
            qa_passes = [qa]
            raw_path = qa.run_dir / "raw_interaction.json"
            if raw_path.is_file():
                interaction_id = read_json(raw_path).get("id")
                if isinstance(interaction_id, str) and interaction_id:
                    qa_interaction_ids.append(interaction_id)
            final_output_path = delivery.output_path
            final_delivery_manifest_path = delivery.manifest_path
            final_media_manifest_path = (
                rendered_music.manifest_path
                if rendered_music is not None
                else delivery.manifest_path
            )
            final_qa = qa
            recovery_summary: dict[str, Any] | None = None
            if policy is not None and isinstance(
                qa.result,
                AutonomousFinalEditQa,
            ):
                assert budget_ledger is not None
                assert aspect_deterministic_path is not None
                assert aspect_deterministic_evidence is not None
                assert aspect_deterministic_report is not None
                first_plan = plan_autonomous_recovery(
                    qa.result,
                    policy=policy,
                    qa_passes_completed=1,
                    semantic_replans_used=(
                        autonomous_semantic_replans_used
                    ),
                )
                recovery_dir = qa_dir / "recovery" / "pass-01"
                remaining_followup_slots = (
                    _remaining_run_global_followup_qa_slots(
                        maximum_full_final_qa_calls=(
                            _maximum_full_final_qa_calls(policy)
                        ),
                        completed_full_final_qa_calls=(
                            autonomous_final_qa_calls_completed
                        ),
                        requested_initial_aspects=(
                            autonomous_initial_aspect_order
                        ),
                        started_initial_aspects=(
                            autonomous_initial_aspects_started
                        ),
                    )
                )
                if (
                    first_plan.outcome == "repair"
                    and remaining_followup_slots < 1
                ):
                    recovery_dir.mkdir(parents=True, exist_ok=True)
                    write_json(
                        recovery_dir / "recovery-plan.json",
                        first_plan,
                    )
                    write_json(
                        recovery_dir
                        / "run-global-qa-cap-block.json",
                        {
                            "contract_version": (
                                "autonomous-run-global-qa-cap-v1"
                            ),
                            "policy_reference": policy.policy_reference,
                            "completed_full_final_qa_calls": (
                                autonomous_final_qa_calls_completed
                            ),
                            "reserved_unstarted_initial_aspects": sorted(
                                set(autonomous_initial_aspect_order)
                                - autonomous_initial_aspects_started
                            ),
                            "maximum_full_final_qa_calls": (
                                _maximum_full_final_qa_calls(policy)
                            ),
                            "semantic_replans_used": (
                                autonomous_semantic_replans_used
                            ),
                            "maximum_semantic_replans": (
                                policy.budget.max_semantic_replans
                            ),
                            "decision_code": (
                                "repair_requires_unavailable_global_second_qa"
                            ),
                            "generated_at": utc_now(),
                        },
                    )
                    raise DeliveryPipelineBlocked(
                        "autonomous repair requires a follow-up QA, but the "
                        "run-global final QA cap is reserved for requested "
                        "aspect initial reviews"
                    )
                effective_repair_executor = autonomous_repair_executor
                if (
                    effective_repair_executor is None
                    and first_plan.outcome == "repair"
                ):
                    # The production default may replay only options that the
                    # feature-cut persisted from its original hard-gate-passed
                    # presentation frontier. It cannot translate arbitrary QA
                    # prose into a new crop, timestamp, or candidate.
                    def execute_feature_cut_repair(
                        *,
                        plan: AutonomousRecoveryPlan,
                        input_render_path: Path,
                        input_render_manifest_path: Path,
                        autonomous_context_paths: Mapping[str, Path],
                        deterministic_delivery_evidence_path: Path,
                        output_dir: Path,
                        **_: Any,
                    ) -> Mapping[str, Any]:
                        request = compile_repair_request(
                            render_manifest_path=(
                                input_render_manifest_path
                            ),
                            input_picture_path=picture,
                            aspect=aspect_ratio,
                            actions=tuple(
                                action.model_dump(mode="json")
                                for action in plan.actions
                            ),
                            output_dir=output_dir / "compile",
                        )
                        if request["status"] != "compiled":
                            raise ValueError(
                                "feature-cut repair remains fail-closed: "
                                + ",".join(
                                    str(row.get("reason_code"))
                                    for row in request["blockers"]
                                )
                            )
                        repaired_picture = (
                            render_changed_segments_and_concat(
                                compiled_request_path=Path(
                                    request["path"]
                                ),
                                deterministic_delivery_evidence_path=(
                                    deterministic_delivery_evidence_path
                                ),
                                autonomous_context_paths=(
                                    autonomous_context_paths
                                ),
                                output_dir=output_dir / "picture",
                            )
                        )
                        repaired_picture_path = Path(
                            repaired_picture["picture_path"]
                        )
                        repaired_manifest_path = Path(
                            repaired_picture["render_manifest_path"]
                        )
                        if rendered_music is None:
                            repaired_delivery = (
                                assemble_picture_only_delivery(
                                    picture_path=repaired_picture_path,
                                    output_path=(
                                        output_dir / "final-repaired.mp4"
                                    ),
                                    manifest_path=(
                                        output_dir
                                        / "final-repaired-delivery.json"
                                    ),
                                    aspect_ratio=aspect_ratio,
                                    artifact_bindings={
                                        "feature_render_manifest_sha256": (
                                            sha256_file(
                                                repaired_manifest_path
                                            )
                                        ),
                                        "repair_request_sha256": request[
                                            "sha256"
                                        ],
                                        "audio_policy": (
                                            "explicitly_absent"
                                        ),
                                    },
                                )
                            )
                            repaired_music_manifest_path = (
                                repaired_delivery.manifest_path
                            )
                        else:
                            assert resolved_lock_path is not None
                            repaired_delivery = (
                                assemble_music_only_delivery(
                                    picture_path=repaired_picture_path,
                                    music_path=(
                                        rendered_music.output_audio_path
                                    ),
                                    output_path=(
                                        output_dir / "final-repaired.mp4"
                                    ),
                                    manifest_path=(
                                        output_dir
                                        / "final-repaired-delivery.json"
                                    ),
                                    music_assembly_artifact_dir=(
                                        assembly_dir
                                    ),
                                    aspect_ratio=aspect_ratio,
                                    artifact_bindings={
                                        "feature_render_manifest_sha256": (
                                            sha256_file(
                                                repaired_manifest_path
                                            )
                                        ),
                                        "music_map_lock_sha256": (
                                            sha256_file(
                                                resolved_lock_path
                                            )
                                        ),
                                        "repair_request_sha256": request[
                                            "sha256"
                                        ],
                                    },
                                )
                            )
                            repaired_music_manifest_path = (
                                rendered_music.manifest_path
                            )
                        repaired_evidence_path = Path(
                            repaired_picture[
                                "deterministic_delivery_evidence_path"
                            ]
                        )
                        repaired_evidence = (
                            DeterministicDeliveryEvidence.model_validate(
                                read_json(repaired_evidence_path)
                            )
                        )
                        _, repaired_evidence_path = (
                            _bind_deterministic_evidence_to_delivery(
                                evidence=repaired_evidence,
                                aspect=aspect_ratio,
                                policy=policy,
                                render_path=repaired_delivery.output_path,
                                render_manifest_path=(
                                    repaired_manifest_path
                                ),
                                delivery_manifest_path=(
                                    repaired_delivery.manifest_path
                                ),
                                music_assembly_manifest_path=(
                                    repaired_music_manifest_path
                                ),
                                autonomous_context_paths=(
                                    repaired_picture[
                                        "autonomous_context_paths"
                                    ]
                                ),
                                output_path=repaired_evidence_path,
                                changed_segment_ids=tuple(
                                    repaired_picture[
                                        "changed_segment_ids"
                                    ]
                                ),
                                reused_segment_ids=tuple(
                                    repaired_picture[
                                        "reused_segment_ids"
                                    ]
                                ),
                            )
                        )
                        return {
                            "render_path": (
                                repaired_delivery.output_path
                            ),
                            "render_manifest_path": (
                                repaired_manifest_path
                            ),
                            "delivery_manifest_path": (
                                repaired_delivery.manifest_path
                            ),
                            "music_assembly_manifest_path": (
                                repaired_music_manifest_path
                            ),
                            "autonomous_context_paths": repaired_picture[
                                "autonomous_context_paths"
                            ],
                            "deterministic_delivery_evidence_path": (
                                repaired_evidence_path
                            ),
                            "changed_segment_ids": repaired_picture[
                                "changed_segment_ids"
                            ],
                            "reused_segment_ids": repaired_picture[
                                "reused_segment_ids"
                            ],
                            "semantic_replan_interaction_ids": (),
                        }

                    effective_repair_executor = execute_feature_cut_repair
                first_execution, repaired = (
                    _execute_autonomous_recovery_plan(
                        plan=first_plan,
                        policy=policy,
                        input_qa_path=qa.run_dir / "validated.json",
                        input_render_path=final_output_path,
                        input_render_manifest_path=render_manifest_path,
                        input_delivery_manifest_path=(
                            final_delivery_manifest_path
                        ),
                        input_music_assembly_manifest_path=(
                            final_media_manifest_path
                        ),
                        autonomous_context_paths=(
                            aspect_autonomous_context
                        ),
                        deterministic_delivery_evidence_path=(
                            aspect_deterministic_path
                        ),
                        segment_contract=prepared.segment_contract,
                        output_dir=recovery_dir,
                        budget_ledger=budget_ledger,
                        executor=effective_repair_executor,
                    )
                )
                recovery_summary = {
                    "first_plan": str(
                        (recovery_dir / "recovery-plan.json").resolve(
                            strict=True
                        )
                    ),
                    "first_execution": str(
                        (
                            recovery_dir / "recovery-execution.json"
                        ).resolve(strict=True)
                    ),
                    "first_status": first_execution.status,
                }
                if first_plan.outcome == "repair" and repaired is None:
                    final_results[aspect_key] = {
                        "final_output": str(final_output_path),
                        "final_output_sha256": sha256_file(final_output_path),
                        "delivery_manifest": str(
                            final_delivery_manifest_path
                        ),
                        "media_assembly_manifest": str(
                            final_media_manifest_path
                        ),
                        "qa_run_dir": str(qa.run_dir),
                        "qa_disposition": (
                            qa.result.qa_observation_status
                        ),
                        "qa_cache_hit": qa.cache_hit,
                        "file_api_reused": file_reused,
                        "qa_pass_count": 1,
                        "autonomous_recovery": recovery_summary,
                        "delivery_eligible": False,
                    }
                    outputs["aspects"] = final_results
                    outputs["autonomous_recovery"] = recovery_summary
                    raise DeliveryPipelineBlocked(
                        "final semantic QA requested a repair, but no verified "
                        "changed-segment-only recovery executor completed it"
                    )
                if first_plan.outcome == "blocked":
                    outputs["autonomous_recovery"] = recovery_summary
                    raise DeliveryPipelineBlocked(
                        "final semantic QA produced a non-repairable bounded "
                        "recovery plan"
                    )
                if repaired is not None:
                    autonomous_semantic_replans_used = max(
                        autonomous_semantic_replans_used,
                        first_plan.semantic_replans_used,
                    )
                    final_output_path = repaired["render_path"]
                    render_manifest_path = repaired["render_manifest_path"]
                    final_delivery_manifest_path = repaired[
                        "delivery_manifest_path"
                    ]
                    final_media_manifest_path = repaired[
                        "music_assembly_manifest_path"
                    ]
                    aspect_autonomous_context = repaired[
                        "autonomous_context_paths"
                    ]
                    aspect_deterministic_path = repaired[
                        "deterministic_delivery_evidence_path"
                    ]
                    aspect_deterministic_evidence = repaired[
                        "deterministic_delivery_evidence"
                    ]
                    aspect_deterministic_report = repaired[
                        "deterministic_delivery_qa"
                    ]
                    resolved_autonomous_context_by_aspect[aspect_ratio] = (
                        aspect_autonomous_context
                    )
                    resolved_deterministic_evidence_by_aspect[aspect_ratio] = (
                        aspect_deterministic_path
                    )
                    deterministic_evidence_by_aspect[aspect_ratio] = (
                        aspect_deterministic_evidence
                    )
                    deterministic_report_by_aspect[aspect_ratio] = (
                        aspect_deterministic_report
                    )
                    qa_interaction_ids.extend(
                        repaired["semantic_replan_interaction_ids"]
                    )
                    second_qa_dir = qa_dir / "pass-02"
                    second_prepared = prepare_final_edit_qa(
                        mode=qa_mode,
                        render_path=final_output_path,
                        manifest_path=render_manifest_path,
                        output_dir=second_qa_dir,
                        model_id=model_id,
                        brief_path=brief_path,
                        crop_include_audio=music_supplied,
                        music_supplied=music_supplied,
                        autonomous_policy=policy,
                        autonomous_context_paths=(
                            aspect_autonomous_context
                        ),
                    )
                    qa_context_hashes_by_aspect[aspect_ratio] = dict(
                        second_prepared.autonomous_context_hashes
                    )
                    second_uploaded, second_file_reused = (
                        client.ensure_video_upload(
                            second_prepared.proxy_path,
                            second_qa_dir
                            / "file-api"
                            / second_prepared.input_hashes["proxy_sha256"],
                        )
                    )
                    if (
                        autonomous_final_qa_calls_completed
                        >= _maximum_full_final_qa_calls(policy)
                    ):
                        raise DeliveryPipelineBlocked(
                            "run-global full final QA pass limit reached "
                            "before repaired render QA"
                        )
                    second_qa = execute_final_edit_qa(
                        prepared=second_prepared,
                        client=client.client,
                        uploaded_video=second_uploaded,
                        output_dir=second_qa_dir,
                        budget_ledger=budget_ledger,
                        recovery_call=True,
                    )
                    autonomous_final_qa_calls_completed += 1
                    qa_passes.append(second_qa)
                    second_raw_path = (
                        second_qa.run_dir / "raw_interaction.json"
                    )
                    if second_raw_path.is_file():
                        interaction_id = read_json(second_raw_path).get("id")
                        if isinstance(interaction_id, str) and interaction_id:
                            qa_interaction_ids.append(interaction_id)
                    second_plan = plan_autonomous_recovery(
                        second_qa.result,
                        policy=policy,
                        qa_passes_completed=2,
                        semantic_replans_used=(
                            autonomous_semantic_replans_used
                        ),
                    )
                    second_recovery_dir = (
                        qa_dir / "recovery" / "pass-02"
                    )
                    second_execution, _ = (
                        _execute_autonomous_recovery_plan(
                            plan=second_plan,
                            policy=policy,
                            input_qa_path=(
                                second_qa.run_dir / "validated.json"
                            ),
                            input_render_path=final_output_path,
                            input_render_manifest_path=render_manifest_path,
                            input_delivery_manifest_path=(
                                final_delivery_manifest_path
                            ),
                            input_music_assembly_manifest_path=(
                                final_media_manifest_path
                            ),
                            autonomous_context_paths=(
                                aspect_autonomous_context
                            ),
                            deterministic_delivery_evidence_path=(
                                aspect_deterministic_path
                            ),
                            segment_contract=(
                                second_prepared.segment_contract
                            ),
                            output_dir=second_recovery_dir,
                            budget_ledger=budget_ledger,
                            executor=None,
                        )
                    )
                    recovery_summary.update(
                        {
                            "second_plan": str(
                                (
                                    second_recovery_dir
                                    / "recovery-plan.json"
                                ).resolve(strict=True)
                            ),
                            "second_execution": str(
                                (
                                    second_recovery_dir
                                    / "recovery-execution.json"
                                ).resolve(strict=True)
                            ),
                            "second_status": second_execution.status,
                        }
                    )
                    if second_plan.outcome != "complete":
                        final_results[aspect_key] = {
                            "final_output": str(final_output_path),
                            "final_output_sha256": sha256_file(
                                final_output_path
                            ),
                            "delivery_manifest": str(
                                final_delivery_manifest_path
                            ),
                            "media_assembly_manifest": str(
                                final_media_manifest_path
                            ),
                            "qa_run_dir": str(second_qa.run_dir),
                            "qa_disposition": (
                                second_qa.result.qa_observation_status
                            ),
                            "qa_cache_hit": second_qa.cache_hit,
                            "file_api_reused": second_file_reused,
                            "qa_pass_count": 2,
                            "autonomous_recovery": recovery_summary,
                            "delivery_eligible": False,
                        }
                        outputs["aspects"] = final_results
                        outputs["autonomous_recovery"] = recovery_summary
                        raise DeliveryPipelineBlocked(
                            "second and final semantic QA still reported "
                            "blocking observations"
                        )
                    final_qa = second_qa
                    file_reused = second_file_reused

            qa_results_by_aspect[aspect_ratio] = final_qa.result
            qa_disposition = (
                final_qa.result.qa_observation_status
                if isinstance(final_qa.result, AutonomousFinalEditQa)
                else _qa_disposition(final_qa)
            )
            final_results[aspect_key] = {
                "final_output": str(final_output_path),
                "final_output_sha256": sha256_file(final_output_path),
                "delivery_manifest": str(final_delivery_manifest_path),
                "media_assembly_manifest": str(final_media_manifest_path),
                "qa_run_dir": str(final_qa.run_dir),
                "qa_disposition": qa_disposition,
                "qa_cache_hit": final_qa.cache_hit,
                "file_api_reused": file_reused,
                "qa_pass_count": len(qa_passes),
                "autonomous_recovery": recovery_summary,
                **(
                    {
                        "autonomous_context_aspect": aspect_ratio,
                        "autonomous_context_sha256": {
                            key: sha256_file(path)
                            for key, path in (
                                aspect_autonomous_context or {}
                            ).items()
                        },
                        "deterministic_delivery_evidence_sha256": (
                            sha256_file(aspect_deterministic_path)
                            if aspect_deterministic_path is not None
                            else None
                        ),
                    }
                    if policy is not None
                    else {}
                ),
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
                "final QA; a playable audition was preserved: "
                + ", ".join(deterministic_failure_codes)
            )
        if not final_results:
            raise DeliveryPipelineBlocked(
                "feature-cut did not produce any requested picture output"
            )
        delivery_authority: DecisionAuthorityV2 | None = None
        if policy is not None:
            if (
                set(deterministic_report_by_aspect) != requested
                or set(resolved_deterministic_evidence_by_aspect)
                != requested
                or set(resolved_autonomous_context_by_aspect) != requested
            ):
                raise DeliveryPipelineBlocked(
                    "per-aspect autonomous authority artifacts are incomplete"
                )
            combined_gate_results = {
                f"{aspect_key}:{gate}": status
                for aspect_key, report in (
                    deterministic_report_by_aspect.items()
                )
                for gate, status in report.gate_results.items()
            }
            deterministic_report = DeterministicDeliveryQaReport(
                gate_results=combined_gate_results,
                failure_codes=tuple(
                    f"{aspect_key}:{code}"
                    for aspect_key, report in (
                        deterministic_report_by_aspect.items()
                    )
                    for code in report.failure_codes
                ),
                passed=all(
                    report.passed
                    for report in deterministic_report_by_aspect.values()
                ),
            )
            degradation_manifests: list[
                AutonomousDegradationManifest
            ] = []
            for aspect_key, context_paths in (
                resolved_autonomous_context_by_aspect.items()
            ):
                degradation_path = context_paths.get("reuse_degradation")
                if degradation_path is None:
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} autonomous context omitted "
                        "reuse_degradation"
                    )
                aspect_degradation = (
                    AutonomousDegradationManifest.model_validate(
                        read_json(degradation_path)
                    )
                )
                if (
                    aspect_degradation.aspect is not None
                    and aspect_degradation.aspect != aspect_key
                ):
                    raise DeliveryPipelineBlocked(
                        f"{aspect_key} final authority received "
                        f"{aspect_degradation.aspect} degradation evidence"
                    )
                degradation_manifests.append(aspect_degradation)
            degradation = AutonomousDegradationManifest(
                policy_reference=policy.policy_reference,
                records=tuple(
                    record
                    for manifest_value in degradation_manifests
                    for record in manifest_value.records
                ),
                generated_at=utc_now(),
            )
            deterministic_report_path = (
                resolved_output / "deterministic-delivery-qa.json"
            )
            write_json(
                deterministic_report_path,
                {
                    **deterministic_report.model_dump(mode="json"),
                    "aspect_reports": {
                        aspect_key: report.model_dump(mode="json")
                        for aspect_key, report in (
                            deterministic_report_by_aspect.items()
                        )
                    },
                },
            )
            authority_hashes = {
                f"sha256:{policy.definition_sha256()}",
                *(
                    f"sha256:{sha256_file(path)}"
                    for path in (
                        resolved_deterministic_evidence_by_aspect.values()
                    )
                ),
                f"sha256:{sha256_file(deterministic_report_path)}",
                f"sha256:{sha256_file(render_manifest_path)}",
                *(
                    f"sha256:{sha256_file(path)}"
                    for paths in (
                        resolved_autonomous_context_by_aspect.values()
                    )
                    for path in paths.values()
                ),
                *(
                    f"sha256:{row['final_output_sha256']}"
                    for row in final_results.values()
                ),
                *(
                    f"sha256:{sha256_file(Path(row[path_key]))}"
                    for row in final_results.values()
                    for path_key in (
                        "delivery_manifest",
                        "media_assembly_manifest",
                    )
                ),
                *(
                    f"sha256:{sha256_file(Path(row['qa_run_dir']) / qa_artifact)}"
                    for row in final_results.values()
                    for qa_artifact in (
                        "input_hashes.json",
                        "schema_validation.json",
                        "validated.json",
                    )
                ),
                *(
                    f"sha256:{sha256_file(Path(str(path_value)))}"
                    for row in final_results.values()
                    for recovery in (row.get("autonomous_recovery"),)
                    if isinstance(recovery, Mapping)
                    for path_key, path_value in recovery.items()
                    if path_key.endswith(("_plan", "_execution"))
                    and isinstance(path_value, str)
                ),
            }
            state, delivery_authority = authorize_autonomous_delivery(
                policy=policy,
                deterministic_qa=deterministic_report,
                qa_results=qa_results_by_aspect,
                qa_context_hashes_by_aspect=(
                    qa_context_hashes_by_aspect
                ),
                degradation=degradation,
                input_artifact_hashes=tuple(sorted(authority_hashes)),
                final_render_sha256_by_aspect={
                    aspect_ratio: final_results[aspect_key][
                        "final_output_sha256"
                    ]
                    for aspect_key, aspect_ratio in (
                        ("horizontal", "16:9"),
                        ("vertical", "9:16"),
                    )
                    if aspect_key in final_results
                },
                final_manifest_sha256=sha256_file(render_manifest_path),
                brief_sha256=sha256_file(
                    brief_path.expanduser().resolve(strict=True)
                ),
                gemini_interaction_ids=tuple(
                    dict.fromkeys(qa_interaction_ids)
                ),
                music_supplied=music_supplied,
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
            "autonomous_run_limits": (
                {
                    "full_final_qa_calls_completed": (
                        autonomous_final_qa_calls_completed
                    ),
                    "max_full_final_qa_calls": (
                        _maximum_full_final_qa_calls(policy)
                    ),
                    "semantic_replans_used": (
                        autonomous_semantic_replans_used
                    ),
                    "max_semantic_replans": (
                        policy.budget.max_semantic_replans
                    ),
                }
                if policy is not None
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
            "deterministic_delivery_qa_by_aspect": (
                {
                    aspect_key: report.model_dump(mode="json")
                    for aspect_key, report in (
                        deterministic_report_by_aspect.items()
                    )
                }
                if deterministic_report_by_aspect
                else None
            ),
            "autonomous_context_paths_by_aspect": (
                {
                    aspect_key: {
                        key: str(path) for key, path in paths.items()
                    }
                    for aspect_key, paths in (
                        resolved_autonomous_context_by_aspect.items()
                    )
                }
                if resolved_autonomous_context_by_aspect
                else None
            ),
            "deterministic_delivery_evidence_paths_by_aspect": (
                {
                    aspect_key: str(path)
                    for aspect_key, path in (
                        resolved_deterministic_evidence_by_aspect.items()
                    )
                }
                if resolved_deterministic_evidence_by_aspect
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
