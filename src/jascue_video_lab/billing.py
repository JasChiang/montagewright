from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal, Mapping
import uuid

from .storage import read_json, utc_now, write_json


STANDARD_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {
        "input": 1.50,
        "cached_input": 0.15,
        "output_including_thought": 9.00,
    },
    "gemini-3.6-flash": {
        "input": 1.50,
        "cached_input": 0.15,
        "output_including_thought": 7.50,
    },
}


_NUMBERED_ATTEMPT_COMPONENT = re.compile(r"^attempt-[0-9]+$")
_LEGACY_ATTEMPT_INTERACTION = re.compile(
    r"(?:^|\.)attempt-[0-9]+\.raw_interaction\.json$"
)
_IMMUTABLE_ATTEMPT_UUID = re.compile(
    r"\.([0-9a-f]{32})\.raw_interaction\.json$"
)


def _is_paid_attempt_artifact(path: Path) -> bool:
    """Return whether a path represents one immutable paid API attempt.

    Current writers use an ``attempts/`` tree, trim flows use numbered
    ``attempt-N/`` directories, and older content-map runs used numbered
    interaction filenames.  Payload equality must never collapse any of these
    because two identical responses can still come from two billed calls.
    """

    parent_components = path.parts[:-1]
    return (
        "attempts" in parent_components
        or any(
            _NUMBERED_ATTEMPT_COMPONENT.fullmatch(component)
            for component in parent_components
        )
        or _LEGACY_ATTEMPT_INTERACTION.search(path.name) is not None
    )


def summarize_usage_files(paths: list[Path], *, relative_to: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    total_thought = 0
    total_cached = 0
    input_by_modality: dict[str, int] = defaultdict(int)
    usage_by_model: dict[str, dict[str, int | float]] = {}
    seen_canonical_payloads: set[str] = set()
    immutable_attempt_payloads: set[str] = set()
    immutable_attempt_ids: dict[str, str] = {}
    duplicate_paths: list[str] = []
    unpriced_paths: list[str] = []
    ordered_paths = sorted(
        paths,
        key=lambda path: (not _is_paid_attempt_artifact(path), str(path)),
    )
    for path in ordered_paths:
        payload = read_json(path)
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        is_immutable_attempt = _is_paid_attempt_artifact(path)
        attempt_match = (
            _IMMUTABLE_ATTEMPT_UUID.search(path.name)
            if is_immutable_attempt
            else None
        )
        attempt_id = attempt_match.group(1) if attempt_match else None
        if attempt_id is not None and attempt_id in immutable_attempt_ids:
            if immutable_attempt_ids[attempt_id] != fingerprint:
                raise ValueError(
                    "immutable paid-attempt UUID maps to different payloads: "
                    f"{path}"
                )
            duplicate_paths.append(
                str(path.relative_to(relative_to)) if relative_to else str(path)
            )
            continue
        if not is_immutable_attempt and (
            fingerprint in immutable_attempt_payloads
            or fingerprint in seen_canonical_payloads
        ):
            duplicate_paths.append(
                str(path.relative_to(relative_to)) if relative_to else str(path)
            )
            continue
        if is_immutable_attempt:
            # Two identical responses can still represent two paid calls.  The
            # generated attempt UUID, not payload equality, defines cardinality.
            # Cache projection may copy the same immutable attempt directory;
            # those path aliases retain one UUID and must not be billed twice.
            immutable_attempt_payloads.add(fingerprint)
            if attempt_id is not None:
                immutable_attempt_ids[attempt_id] = fingerprint
        else:
            seen_canonical_payloads.add(fingerprint)
        usage = payload.get("usage") or {}
        if not usage:
            unpriced_paths.append(
                str(path.relative_to(relative_to)) if relative_to else str(path)
            )
            continue
        model_id = str(payload.get("model") or "")
        if not model_id:
            raise ValueError(f"usage artifact has no model id: {path}")
        if model_id not in STANDARD_PRICING_USD_PER_MILLION:
            raise ValueError(f"no Standard pricing is registered for {model_id!r}: {path}")
        input_tokens = int(usage.get("total_input_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        thought_tokens = int(usage.get("total_thought_tokens") or 0)
        cached_tokens = int(usage.get("total_cached_tokens") or 0)
        if cached_tokens < 0 or cached_tokens > input_tokens:
            raise ValueError(f"invalid cached token count in usage artifact: {path}")
        total_input += input_tokens
        total_output += output_tokens
        total_thought += thought_tokens
        total_cached += cached_tokens
        model_usage = usage_by_model.setdefault(
            model_id,
            {
                "request_count": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "thought_tokens": 0,
            },
        )
        model_usage["request_count"] = int(model_usage["request_count"]) + 1
        model_usage["input_tokens"] = int(model_usage["input_tokens"]) + input_tokens
        model_usage["cached_input_tokens"] = (
            int(model_usage["cached_input_tokens"]) + cached_tokens
        )
        model_usage["output_tokens"] = int(model_usage["output_tokens"]) + output_tokens
        model_usage["thought_tokens"] = int(model_usage["thought_tokens"]) + thought_tokens
        modalities: dict[str, int] = {}
        for item in usage.get("input_tokens_by_modality") or []:
            modality = str(item.get("modality") or "UNKNOWN")
            tokens = int(item.get("tokens") or 0)
            modalities[modality] = modalities.get(modality, 0) + tokens
            input_by_modality[modality] += tokens
        records.append(
            {
                "path": str(path.relative_to(relative_to)) if relative_to else str(path),
                "model": model_id,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "thought_tokens": thought_tokens,
                "input_tokens_by_modality": modalities,
            }
        )
    billed_output = total_output + total_thought
    input_cost = 0.0
    output_cost = 0.0
    model_breakdown: dict[str, dict[str, int | float]] = {}
    for model_id, model_usage in sorted(usage_by_model.items()):
        rates = STANDARD_PRICING_USD_PER_MILLION[model_id]
        model_input = int(model_usage["input_tokens"])
        model_cached_input = int(model_usage["cached_input_tokens"])
        model_uncached_input = model_input - model_cached_input
        model_output = int(model_usage["output_tokens"])
        model_thought = int(model_usage["thought_tokens"])
        model_billed_output = model_output + model_thought
        model_input_cost = (
            model_uncached_input / 1_000_000 * rates["input"]
            + model_cached_input / 1_000_000 * rates["cached_input"]
        )
        model_output_cost = (
            model_billed_output / 1_000_000 * rates["output_including_thought"]
        )
        input_cost += model_input_cost
        output_cost += model_output_cost
        model_breakdown[model_id] = {
            **model_usage,
            "uncached_input_tokens": model_uncached_input,
            "billed_output_tokens": model_billed_output,
            "input_usd_per_million_tokens": rates["input"],
            "cached_input_usd_per_million_tokens": rates["cached_input"],
            "output_including_thought_usd_per_million_tokens": rates[
                "output_including_thought"
            ],
            "estimated_input_cost_usd": round(model_input_cost, 8),
            "estimated_output_cost_usd": round(model_output_cost, 8),
            "estimated_total_cost_usd": round(model_input_cost + model_output_cost, 8),
        }
    models = sorted(model_breakdown)
    single_model_rates = (
        STANDARD_PRICING_USD_PER_MILLION[models[0]] if len(models) == 1 else None
    )
    return {
        "model": models[0] if len(models) == 1 else ("mixed" if models else None),
        "pricing_basis": "Standard paid-tier public list price; actual invoice may be free-tier or differ",
        "pricing_complete": not unpriced_paths,
        "cost_interpretation": (
            "estimated_total"
            if not unpriced_paths
            else "lower_bound_incomplete_usage_metadata"
        ),
        "input_usd_per_million_tokens": (
            single_model_rates["input"] if single_model_rates else None
        ),
        "cached_input_usd_per_million_tokens": (
            single_model_rates["cached_input"] if single_model_rates else None
        ),
        "output_including_thought_usd_per_million_tokens": (
            single_model_rates["output_including_thought"] if single_model_rates else None
        ),
        "models": model_breakdown,
        "request_count": len(records) + len(unpriced_paths),
        "priced_request_count": len(records),
        "unpriced_request_count": len(unpriced_paths),
        "unpriced_request_paths": unpriced_paths,
        "total_input_tokens": total_input,
        "total_cached_input_tokens": total_cached,
        "total_uncached_input_tokens": total_input - total_cached,
        "total_output_tokens": total_output,
        "total_thought_tokens": total_thought,
        "billed_output_tokens": billed_output,
        "input_tokens_by_modality": dict(sorted(input_by_modality.items())),
        "estimated_input_cost_usd": round(input_cost, 8),
        "estimated_output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(input_cost + output_cost, 8),
        "duplicate_artifact_count": len(duplicate_paths),
        "duplicate_artifact_paths": duplicate_paths,
        "requests": records,
    }


def summarize_usage_and_list_price(root: Path) -> dict[str, Any]:
    return summarize_usage_files(
        list(root.rglob("*raw_interaction.json")), relative_to=root
    )


VIDEO_TOKENS_PER_SECOND: dict[str, int] = {
    "low": 100,
    "medium": 100,
    "high": 400,
}
IMAGE_TOKENS_PER_ITEM: dict[str, int] = {
    "low": 280,
    "medium": 560,
    "high": 1_120,
}
THINKING_TOKEN_RESERVE: dict[str, int] = {
    "minimal": 0,
    "low": 1_024,
    "medium": 4_096,
    "high": 8_192,
}


class BudgetExceeded(RuntimeError):
    """Raised before a paid request when its reserve would exceed policy."""


class PaidDispatchAlreadyRecorded(RuntimeError):
    """The exact paid request may already have reached the provider."""


@dataclass(frozen=True)
class PaidCallEstimate:
    stage: str
    model_id: str
    media_resolution: Literal["low", "medium", "high"]
    estimated_input_tokens: int
    max_output_tokens: int
    reserved_thought_tokens: int
    retry_allowance: int
    worst_case_interactions: int
    worst_case_cost_usd: float


@dataclass
class BudgetReservation:
    reservation_id: str
    estimate: PaidCallEstimate
    recovery_call: bool
    state: Literal["reserved", "reconciled", "cancelled"] = "reserved"
    actual_input_tokens: int | None = None
    actual_cached_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    actual_thought_tokens: int | None = None
    actual_cost_usd: float | None = None
    reserved_at: str = ""
    reconciled_at: str | None = None
    reconciliation_basis: Literal["actual_usage", "conservative_worst_case"] | None = None
    dispatch_id: str | None = None


@dataclass(frozen=True)
class PaidDispatchHandle:
    dispatch_id: str
    request_sha256: str
    journal_path: Path
    reservation_id: str | None


def estimate_paid_call(
    *,
    stage: str,
    model_id: str,
    media_duration_ms: int = 0,
    media_resolution: Literal["low", "medium", "high"] = "low",
    image_count: int = 0,
    text_input_tokens: int = 0,
    max_output_tokens: int,
    thinking_level: Literal["minimal", "low", "medium", "high"] = "low",
    retry_allowance: int = 0,
) -> PaidCallEstimate:
    """Conservatively price a request before any upload or paid interaction."""

    if model_id not in STANDARD_PRICING_USD_PER_MILLION:
        raise ValueError(f"no Standard pricing is registered for {model_id!r}")
    if media_duration_ms < 0 or image_count < 0 or text_input_tokens < 0:
        raise ValueError("paid-call estimate inputs cannot be negative")
    if max_output_tokens <= 0 or retry_allowance < 0:
        raise ValueError("output limit must be positive and retries non-negative")
    seconds = (media_duration_ms + 999) // 1_000
    media_tokens = seconds * VIDEO_TOKENS_PER_SECOND[media_resolution]
    image_tokens = image_count * IMAGE_TOKENS_PER_ITEM[media_resolution]
    input_tokens = text_input_tokens + media_tokens + image_tokens
    thought_tokens = THINKING_TOKEN_RESERVE[thinking_level]
    attempts = 1 + retry_allowance
    rates = STANDARD_PRICING_USD_PER_MILLION[model_id]
    per_attempt = (
        input_tokens / 1_000_000 * rates["input"]
        + (max_output_tokens + thought_tokens)
        / 1_000_000
        * rates["output_including_thought"]
    )
    return PaidCallEstimate(
        stage=stage,
        model_id=model_id,
        media_resolution=media_resolution,
        estimated_input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        reserved_thought_tokens=thought_tokens,
        retry_allowance=retry_allowance,
        worst_case_interactions=attempts,
        worst_case_cost_usd=round(per_attempt * attempts, 8),
    )


def actual_usage_cost(
    *,
    model_id: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    thought_tokens: int,
) -> float:
    if model_id not in STANDARD_PRICING_USD_PER_MILLION:
        raise ValueError(f"no Standard pricing is registered for {model_id!r}")
    if not 0 <= cached_input_tokens <= input_tokens:
        raise ValueError("cached input tokens must remain inside total input")
    values = (input_tokens, output_tokens, thought_tokens)
    if any(value < 0 for value in values):
        raise ValueError("usage tokens cannot be negative")
    rates = STANDARD_PRICING_USD_PER_MILLION[model_id]
    uncached = input_tokens - cached_input_tokens
    return round(
        uncached / 1_000_000 * rates["input"]
        + cached_input_tokens / 1_000_000 * rates["cached_input"]
        + (output_tokens + thought_tokens)
        / 1_000_000
        * rates["output_including_thought"],
        8,
    )


def _canonical_request_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _paid_work_node_id(*, stage: str, request_sha256: str) -> str:
    return "sha256:" + hashlib.sha256(
        (
            "paid-work-node-v1:"
            + stage
            + ":"
            + request_sha256
        ).encode("utf-8")
    ).hexdigest()


def migrate_completed_legacy_paid_dispatch(
    *,
    stage: str,
    request_path: Path,
    raw_artifact_path: Path,
) -> Path:
    """Create a zero-dispatch journal for one proven legacy paid response.

    This is accounting/resume lineage only. Semantic reuse remains governed
    by the caller's independent input/schema bindings. Both immutable request
    and raw usage artifacts must already exist; this function never calls a
    provider and never invents missing usage.
    """

    request = read_json(request_path)
    raw = read_json(raw_artifact_path)
    if not isinstance(request, Mapping) or not isinstance(raw, Mapping):
        raise ValueError("legacy paid migration requires object artifacts")
    usage = raw.get("usage")
    if not isinstance(usage, Mapping) or not usage:
        raise ValueError(
            "legacy paid migration requires durable raw usage"
        )
    model_id = str(request.get("model") or raw.get("model") or "")
    if not model_id:
        raise ValueError("legacy paid migration has no model ID")
    generation_config = request.get("generation_config")
    generation = (
        generation_config
        if isinstance(generation_config, Mapping)
        else {}
    )
    estimate = estimate_paid_call(
        stage=stage,
        model_id=model_id,
        text_input_tokens=max(
            1,
            int(usage.get("total_input_tokens") or 0),
        ),
        max_output_tokens=max(
            1,
            int(
                generation.get("max_output_tokens")
                or usage.get("total_output_tokens")
                or 1
            ),
        ),
        thinking_level=str(
            generation.get("thinking_level") or "low"
        ),
        retry_allowance=0,
    )
    request_sha256 = _canonical_request_sha256(request)
    work_node_id = _paid_work_node_id(
        stage=stage,
        request_sha256=request_sha256,
    )
    safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "-", stage).strip("-")
    journal_path = request_path.parent / (
        f"{safe_stage}.{request_sha256}.paid_dispatch.json"
    )
    request_file_sha256 = hashlib.sha256(
        request_path.read_bytes()
    ).hexdigest()
    raw_file_sha256 = hashlib.sha256(
        raw_artifact_path.read_bytes()
    ).hexdigest()
    dispatch_id = hashlib.sha256(
        (
            "legacy-completed-paid-dispatch-v1:"
            + stage
            + ":"
            + request_sha256
            + ":"
            + raw_file_sha256
        ).encode("utf-8")
    ).hexdigest()
    journal = {
        "contract_version": "paid-dispatch-journal-v2",
        "work_node_id": work_node_id,
        "dispatch_id": dispatch_id,
        "request_sha256": request_sha256,
        "stage": stage,
        "model_id": model_id,
        "estimate": asdict(estimate),
        "status": "raw_usage_persisted",
        "request_artifact_persisted": True,
        "raw_usage_persisted": True,
        "raw_artifact_path": str(raw_artifact_path.resolve()),
        "interaction_id": str(raw.get("id") or ""),
        "migration": {
            "contract_version": (
                "completed-legacy-paid-dispatch-migration-v1"
            ),
            "request_path": str(request_path.resolve()),
            "request_file_sha256": request_file_sha256,
            "raw_file_sha256": raw_file_sha256,
            "provider_dispatch_added": False,
        },
        "completed_at": utc_now(),
    }
    if journal_path.is_file():
        saved = read_json(journal_path)
        if saved != journal:
            # completed_at is intentionally stable after first migration.
            saved_without_time = dict(saved)
            journal_without_time = dict(journal)
            saved_without_time.pop("completed_at", None)
            journal_without_time.pop("completed_at", None)
            if saved_without_time != journal_without_time:
                raise ValueError(
                    "legacy paid dispatch migration journal changed"
                )
        return journal_path
    write_json(journal_path, journal)
    return journal_path


def _estimate_from_dispatch_journal(payload: Mapping[str, Any]) -> PaidCallEstimate:
    raw = payload.get("estimate")
    if not isinstance(raw, Mapping):
        raise ValueError("paid dispatch journal has no estimate")
    estimate = PaidCallEstimate(
        stage=str(raw["stage"]),
        model_id=str(raw["model_id"]),
        media_resolution=str(raw["media_resolution"]),  # type: ignore[arg-type]
        estimated_input_tokens=int(raw["estimated_input_tokens"]),
        max_output_tokens=int(raw["max_output_tokens"]),
        reserved_thought_tokens=int(raw["reserved_thought_tokens"]),
        retry_allowance=int(raw["retry_allowance"]),
        worst_case_interactions=int(raw["worst_case_interactions"]),
        worst_case_cost_usd=float(raw["worst_case_cost_usd"]),
    )
    if estimate.retry_allowance != 0 or estimate.worst_case_interactions != 1:
        raise ValueError(
            "paid dispatch journal may bind only one non-retrying provider call"
        )
    return estimate


def _dispatch_needs_conservative_adoption(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "") in {
        "dispatch_started",
        "dispatch_failed_usage_unavailable",
        "raw_persisted_usage_unavailable",
    }


def dispatch_paid_interaction(
    *,
    client: Any,
    request: Mapping[str, Any],
    request_record: Mapping[str, Any],
    journal_dir: Path,
    estimate: PaidCallEstimate,
    budget_ledger: "BudgetLedger | None",
    recovery_call: bool = False,
) -> tuple[Any, PaidDispatchHandle]:
    """Dispatch one exact request with a crash-safe, hash-bound journal.

    The journal is durable immediately before ``interactions.create``. If the
    provider call raises without usage, the active reserve is reconciled at its
    worst case. A later process encountering the same hash adopts that journal
    and refuses to dispatch the exact request again.
    """

    if estimate.retry_allowance != 0 or estimate.worst_case_interactions != 1:
        raise ValueError(
            "journaled non-subprocess calls must disable provider retries"
        )
    resolved_dir = journal_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    request_sha256 = _canonical_request_sha256(request_record)
    work_node_id = _paid_work_node_id(
        stage=estimate.stage,
        request_sha256=request_sha256,
    )
    safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "-", estimate.stage).strip("-")
    journal_path = (
        resolved_dir / f"{safe_stage}.{request_sha256}.paid_dispatch.json"
    )
    if journal_path.exists():
        payload = read_json(journal_path)
        if str(payload.get("request_sha256") or "") != request_sha256:
            raise ValueError("paid dispatch journal path/hash mismatch")
        if str(payload.get("work_node_id") or "") != work_node_id:
            raise ValueError("paid dispatch journal work-node identity mismatch")
        dispatch_id = str(payload.get("dispatch_id") or "")
        if not dispatch_id:
            raise ValueError("paid dispatch journal has no dispatch ID")
        if budget_ledger is not None and _dispatch_needs_conservative_adoption(
            payload
        ):
            budget_ledger.adopt_conservative_dispatch(
                dispatch_id=dispatch_id,
                estimate=_estimate_from_dispatch_journal(payload),
            )
        raise PaidDispatchAlreadyRecorded(
            "the exact paid request already has a durable dispatch journal; "
            "refusing interactions.create"
        )

    reservation = None
    if budget_ledger is not None:
        reservation = budget_ledger.reserve(
            estimate,
            recovery_call=recovery_call,
        )
    dispatch_id = uuid.uuid4().hex
    if reservation is not None:
        reservation.dispatch_id = dispatch_id
    handle = PaidDispatchHandle(
        dispatch_id=dispatch_id,
        request_sha256=request_sha256,
        journal_path=journal_path,
        reservation_id=(
            reservation.reservation_id if reservation is not None else None
        ),
    )
    journal = {
        "contract_version": "paid-dispatch-journal-v2",
        "work_node_id": work_node_id,
        "dispatch_id": dispatch_id,
        "request_sha256": request_sha256,
        "stage": estimate.stage,
        "model_id": estimate.model_id,
        "estimate": asdict(estimate),
        "status": "dispatch_started",
        "request_artifact_persisted": True,
        "raw_usage_persisted": False,
        "started_at": utc_now(),
    }
    write_json(journal_path, journal)
    try:
        interaction = client.interactions.create(**dict(request))
    except BaseException as error:
        journal.update(
            {
                "status": "dispatch_failed_usage_unavailable",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "failed_at": utc_now(),
            }
        )
        write_json(journal_path, journal)
        if budget_ledger is not None and reservation is not None:
            budget_ledger.reconcile_conservative_dispatch(
                reservation.reservation_id,
                dispatch_id=dispatch_id,
            )
        raise
    return interaction, handle


def complete_paid_dispatch(
    *,
    handle: PaidDispatchHandle,
    raw_interaction: Mapping[str, Any],
    raw_artifact_path: Path,
    budget_ledger: "BudgetLedger | None",
    model_id: str,
) -> None:
    """Bind persisted raw usage to a started dispatch and settle its reserve."""

    if not raw_artifact_path.is_file():
        raise ValueError(
            "raw interaction must be durably persisted before dispatch completion"
        )
    journal = read_json(handle.journal_path)
    if str(journal.get("dispatch_id") or "") != handle.dispatch_id:
        raise ValueError("paid dispatch handle/journal mismatch")
    usage = raw_interaction.get("usage")
    has_usage = isinstance(usage, Mapping) and bool(usage)
    if budget_ledger is not None and handle.reservation_id is not None:
        if has_usage:
            budget_ledger.reconcile(
                handle.reservation_id,
                usage=usage,
                model_id=model_id,
            )
        else:
            budget_ledger.reconcile_conservative_dispatch(
                handle.reservation_id,
                dispatch_id=handle.dispatch_id,
            )
    journal.update(
        {
            "status": (
                "raw_usage_persisted"
                if has_usage
                else "raw_persisted_usage_unavailable"
            ),
            "raw_usage_persisted": has_usage,
            "raw_artifact_path": str(raw_artifact_path.resolve()),
            "interaction_id": str(raw_interaction.get("id") or ""),
            "completed_at": utc_now(),
        }
    )
    write_json(handle.journal_path, journal)


def adopt_paid_dispatch_journals(
    *,
    budget_ledger: "BudgetLedger",
    root: Path,
    allowed_top_level: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Adopt every ambiguous non-subprocess dispatch once on process resume."""

    adopted: list[dict[str, Any]] = []
    seen_dispatch_ids: set[str] = set()
    resolved_root = root.expanduser().resolve()
    if not resolved_root.exists():
        return adopted
    for journal_path in sorted(
        resolved_root.rglob("*.paid_dispatch.json")
    ):
        relative = journal_path.relative_to(resolved_root)
        if (
            allowed_top_level is not None
            and (
                not relative.parts
                or relative.parts[0] not in allowed_top_level
            )
        ):
            continue
        payload = read_json(journal_path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid paid dispatch journal: {journal_path}")
        if not _dispatch_needs_conservative_adoption(payload):
            continue
        dispatch_id = str(payload.get("dispatch_id") or "")
        if not dispatch_id:
            raise ValueError(f"paid dispatch journal has no ID: {journal_path}")
        if dispatch_id in seen_dispatch_ids:
            continue
        estimate = _estimate_from_dispatch_journal(payload)
        request_sha256 = str(payload.get("request_sha256") or "")
        work_node_id = str(payload.get("work_node_id") or "")
        if work_node_id != _paid_work_node_id(
            stage=estimate.stage,
            request_sha256=request_sha256,
        ):
            raise ValueError(
                f"paid dispatch journal work-node identity differs: "
                f"{journal_path}"
            )
        budget_ledger.adopt_conservative_dispatch(
            dispatch_id=dispatch_id,
            estimate=estimate,
        )
        seen_dispatch_ids.add(dispatch_id)
        adopted.append(
            {
                "dispatch_id": dispatch_id,
                "work_node_id": work_node_id,
                "path": str(relative),
                "stage": estimate.stage,
                "model_id": estimate.model_id,
                "status": str(payload.get("status") or ""),
                "worst_case_cost_usd": estimate.worst_case_cost_usd,
            }
        )
    return adopted


def adopt_paid_dispatch_journal_state(
    *,
    budget_ledger: "BudgetLedger",
    root: Path,
    allowed_top_level: set[str] | frozenset[str] | None = None,
    allowed_relative_path: Callable[[Path], bool] | None = None,
) -> tuple[list[dict[str, Any]], frozenset[Path]]:
    """Adopt paid nodes by journal identity, never by artifact path naming."""

    adopted: list[dict[str, Any]] = []
    journaled_raw_paths: set[Path] = set()
    seen_dispatch_ids: set[str] = set()
    resolved_root = root.expanduser().resolve()
    if not resolved_root.exists():
        return adopted, frozenset()
    for journal_path in sorted(
        resolved_root.rglob("*.paid_dispatch.json")
    ):
        relative = journal_path.relative_to(resolved_root)
        if allowed_relative_path is not None:
            allowed = allowed_relative_path(relative)
        elif allowed_top_level is not None:
            allowed = bool(
                relative.parts and relative.parts[0] in allowed_top_level
            )
        else:
            allowed = True
        if not allowed:
            continue
        payload = read_json(journal_path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid paid dispatch journal: {journal_path}")
        dispatch_id = str(payload.get("dispatch_id") or "")
        if not dispatch_id:
            raise ValueError(f"paid dispatch journal has no ID: {journal_path}")
        if dispatch_id in seen_dispatch_ids:
            continue
        estimate = _estimate_from_dispatch_journal(payload)
        request_sha256 = str(payload.get("request_sha256") or "")
        work_node_id = str(payload.get("work_node_id") or "")
        if work_node_id != _paid_work_node_id(
            stage=estimate.stage,
            request_sha256=request_sha256,
        ):
            raise ValueError(
                f"paid dispatch journal work-node identity differs: "
                f"{journal_path}"
            )
        status = str(payload.get("status") or "")
        raw_path_value = payload.get("raw_artifact_path")
        raw_path = (
            Path(str(raw_path_value)).expanduser().resolve()
            if isinstance(raw_path_value, str) and raw_path_value
            else None
        )
        if raw_path is not None:
            try:
                raw_path.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(
                    "paid dispatch raw artifact lies outside the run root"
                ) from error
            journaled_raw_paths.add(raw_path)
        if status == "raw_usage_persisted":
            if raw_path is None or not raw_path.is_file():
                raise ValueError(
                    "completed paid dispatch has no durable raw artifact"
                )
            raw = read_json(raw_path)
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("usage"),
                Mapping,
            ):
                raise ValueError(
                    "completed paid dispatch raw artifact has no usage"
                )
            journal_interaction_id = str(
                payload.get("interaction_id") or ""
            )
            raw_interaction_id = str(raw.get("id") or "")
            if (
                journal_interaction_id
                and raw_interaction_id
                and journal_interaction_id != raw_interaction_id
            ):
                raise ValueError(
                    "paid dispatch journal/raw interaction ID mismatch"
                )
            budget_ledger.adopt_reconciled_usage(
                stage=estimate.stage,
                model_id=estimate.model_id,
                usage=raw["usage"],
            )
            adoption_basis = "actual_usage"
        elif _dispatch_needs_conservative_adoption(payload):
            budget_ledger.adopt_conservative_dispatch(
                dispatch_id=dispatch_id,
                estimate=estimate,
            )
            adoption_basis = "conservative_worst_case"
        else:
            raise ValueError(
                "paid dispatch journal has an unsupported terminal status: "
                f"{status or '<empty>'}"
            )
        seen_dispatch_ids.add(dispatch_id)
        adopted.append(
            {
                "dispatch_id": dispatch_id,
                "work_node_id": work_node_id,
                "path": str(relative),
                "stage": estimate.stage,
                "model_id": estimate.model_id,
                "status": status,
                "adoption_basis": adoption_basis,
                "worst_case_cost_usd": estimate.worst_case_cost_usd,
            }
        )
    return adopted, frozenset(journaled_raw_paths)


class BudgetLedger:
    """Reserve worst-case paid work, then reconcile immutable usage evidence."""

    contract_version = "budget-ledger-v3"

    def __init__(
        self,
        *,
        max_cost_usd: float,
        max_interactions: int,
        reserved_recovery_fraction: float = 0.20,
        mandatory_stage_minimums: Mapping[str, int] | None = None,
        mandatory_stage_cost_holds: Mapping[str, float] | None = None,
        interaction_guard: int = 0,
    ) -> None:
        if max_cost_usd <= 0 or max_interactions <= 0:
            raise ValueError("budget caps must be positive")
        if (
            isinstance(interaction_guard, bool)
            or not isinstance(interaction_guard, int)
            or interaction_guard < 0
        ):
            raise ValueError("interaction guard must be a non-negative integer")
        if not 0.20 <= reserved_recovery_fraction <= 0.50:
            raise ValueError("recovery reserve must be between 20% and 50%")
        resolved_minimums = (
            {"final_qa": 1}
            if mandatory_stage_minimums is None
            else dict(mandatory_stage_minimums)
        )
        normalized_minimums: dict[str, int] = {}
        for raw_stage, raw_minimum in resolved_minimums.items():
            stage = str(raw_stage).strip()
            if not stage:
                raise ValueError("mandatory budget stage cannot be empty")
            if isinstance(raw_minimum, bool) or not isinstance(raw_minimum, int):
                raise ValueError(
                    "mandatory stage interaction minimums must be integers"
                )
            if raw_minimum < 0:
                raise ValueError(
                    "mandatory stage interaction minimums cannot be negative"
                )
            if raw_minimum:
                normalized_minimums[stage] = raw_minimum
        if (
            sum(normalized_minimums.values()) + interaction_guard
            > max_interactions
        ):
            raise ValueError(
                "mandatory stage interaction minimums and guard exceed the "
                "interaction cap"
            )
        normalized_cost_holds: dict[str, float] = {}
        for raw_stage, raw_cost in dict(
            mandatory_stage_cost_holds or {}
        ).items():
            stage = str(raw_stage).strip()
            if not stage:
                raise ValueError("mandatory cost-hold stage cannot be empty")
            if isinstance(raw_cost, bool) or not isinstance(
                raw_cost,
                (int, float),
            ):
                raise ValueError("mandatory stage cost holds must be numeric")
            cost = float(raw_cost)
            if cost < 0:
                raise ValueError("mandatory stage cost holds cannot be negative")
            if cost:
                normalized_cost_holds[stage] = cost
        if sum(normalized_cost_holds.values()) > max_cost_usd + 1e-9:
            raise ValueError(
                "mandatory stage cost holds exceed the USD cap"
            )
        unknown_cost_hold_stages = (
            set(normalized_cost_holds) - set(normalized_minimums)
        )
        if unknown_cost_hold_stages:
            raise ValueError(
                "mandatory cost holds require matching interaction holds: "
                + ", ".join(sorted(unknown_cost_hold_stages))
            )
        self.max_cost_usd = max_cost_usd
        self.max_interactions = max_interactions
        self.interaction_guard = interaction_guard
        self.reserved_recovery_fraction = reserved_recovery_fraction
        self.mandatory_stage_minimums = dict(
            sorted(normalized_minimums.items())
        )
        self.mandatory_stage_cost_holds = dict(
            sorted(normalized_cost_holds.items())
        )
        self._reservations: dict[str, BudgetReservation] = {}
        self.created_at = utc_now()

    @property
    def reserved_cost_usd(self) -> float:
        return round(
            sum(
                item.estimate.worst_case_cost_usd
                for item in self._reservations.values()
                if item.state == "reserved"
            ),
            8,
        )

    @property
    def actual_cost_usd(self) -> float:
        return round(
            sum(
                item.actual_cost_usd or 0.0
                for item in self._reservations.values()
                if item.state == "reconciled"
            ),
            8,
        )

    @property
    def committed_interactions(self) -> int:
        return sum(
            item.estimate.worst_case_interactions
            for item in self._reservations.values()
            if item.state != "cancelled"
        )

    def _mandatory_hold_stage(self, stage: str) -> str | None:
        """Resolve a concrete call stage to its most-specific configured hold.

        A hold named ``final_qa`` intentionally covers typed sub-stages such as
        ``final_qa:autonomous_final_9x16``. Choosing the longest matching name
        prevents one call from satisfying two overlapping holds.
        """

        hold_stages = (
            set(self.mandatory_stage_minimums)
            | set(self.mandatory_stage_cost_holds)
        )
        matches = [
            hold_stage
            for hold_stage in hold_stages
            if stage == hold_stage or stage.startswith(f"{hold_stage}:")
        ]
        return max(matches, key=len) if matches else None

    def _committed_mandatory_interactions(self) -> dict[str, int]:
        committed = {
            stage: 0 for stage in self.mandatory_stage_minimums
        }
        for reservation in self._reservations.values():
            if reservation.state == "cancelled":
                continue
            hold_stage = self._mandatory_hold_stage(
                reservation.estimate.stage
            )
            if hold_stage is not None:
                committed[hold_stage] += (
                    reservation.estimate.worst_case_interactions
                )
        return committed

    def remaining_mandatory_interaction_holds(self) -> dict[str, int]:
        committed = self._committed_mandatory_interactions()
        return {
            stage: max(0, minimum - committed[stage])
            for stage, minimum in self.mandatory_stage_minimums.items()
        }

    def _committed_mandatory_cost_ceiling(self) -> dict[str, float]:
        committed = {
            stage: 0.0 for stage in self.mandatory_stage_cost_holds
        }
        for reservation in self._reservations.values():
            if reservation.state == "cancelled":
                continue
            hold_stage = self._mandatory_hold_stage(
                reservation.estimate.stage
            )
            if hold_stage in committed:
                committed[hold_stage] += (
                    reservation.estimate.worst_case_cost_usd
                )
        return {
            stage: round(cost, 8)
            for stage, cost in committed.items()
        }

    def remaining_mandatory_cost_holds(self) -> dict[str, float]:
        committed = self._committed_mandatory_cost_ceiling()
        return {
            stage: round(max(0.0, minimum - committed[stage]), 8)
            for stage, minimum in self.mandatory_stage_cost_holds.items()
        }

    def _remaining_mandatory_cost_after(
        self,
        *,
        stage: str,
        cost_usd: float,
    ) -> float:
        committed = self._committed_mandatory_cost_ceiling()
        hold_stage = self._mandatory_hold_stage(stage)
        if hold_stage in committed:
            committed[hold_stage] = round(
                committed[hold_stage] + cost_usd,
                8,
            )
        return round(
            sum(
                max(0.0, minimum - committed[configured_stage])
                for configured_stage, minimum
                in self.mandatory_stage_cost_holds.items()
            ),
            8,
        )

    def _interaction_limit_after(
        self,
        *,
        stage: str,
        interactions: int,
    ) -> int:
        committed_by_hold = self._committed_mandatory_interactions()
        mandatory_hold_stage = self._mandatory_hold_stage(stage)
        if mandatory_hold_stage is not None:
            committed_by_hold[mandatory_hold_stage] += interactions
        remaining_holds = sum(
            max(0, minimum - committed_by_hold[hold_stage])
            for hold_stage, minimum in self.mandatory_stage_minimums.items()
        )
        return self.max_interactions - self.interaction_guard - remaining_holds

    def _reserve_failure_messages(
        self,
        estimate: PaidCallEstimate,
        *,
        recovery_call: bool,
    ) -> list[str]:
        """Return the exact pre-dispatch reserve failures without mutation."""

        projected_cost = (
            self.actual_cost_usd
            + self.reserved_cost_usd
            + estimate.worst_case_cost_usd
        )
        projected_interactions = (
            self.committed_interactions + estimate.worst_case_interactions
        )
        mandatory_hold_stage = self._mandatory_hold_stage(estimate.stage)
        if recovery_call or mandatory_hold_stage is not None:
            coarse_cost_limit = self.max_cost_usd
        else:
            usable_fraction = 1.0 - self.reserved_recovery_fraction
            coarse_cost_limit = self.max_cost_usd * usable_fraction
        remaining_mandatory_cost = self._remaining_mandatory_cost_after(
            stage=estimate.stage,
            cost_usd=estimate.worst_case_cost_usd,
        )
        completion_cost_limit = self.max_cost_usd - remaining_mandatory_cost
        cost_limit = min(coarse_cost_limit, completion_cost_limit)
        interaction_limit = self._interaction_limit_after(
            stage=estimate.stage,
            interactions=estimate.worst_case_interactions,
        )
        failures: list[str] = []
        if projected_cost > cost_limit + 1e-9:
            failures.append(
                f"cost reserve {projected_cost:.8f} exceeds {cost_limit:.8f}"
            )
        if projected_interactions > interaction_limit:
            failures.append(
                "interaction reserve "
                f"{projected_interactions} exceeds {interaction_limit}"
            )
        return failures

    def can_reserve(
        self,
        estimate: PaidCallEstimate,
        *,
        recovery_call: bool = False,
    ) -> bool:
        """Check whether a worst-case call fits without creating a reservation."""

        return not self._reserve_failure_messages(
            estimate,
            recovery_call=recovery_call,
        )

    def reserve(
        self,
        estimate: PaidCallEstimate,
        *,
        recovery_call: bool = False,
    ) -> BudgetReservation:
        failures = self._reserve_failure_messages(
            estimate,
            recovery_call=recovery_call,
        )
        if failures:
            raise BudgetExceeded(
                f"paid call blocked before request ({estimate.stage}): "
                + "; ".join(failures)
            )
        reservation = BudgetReservation(
            reservation_id=uuid.uuid4().hex,
            estimate=estimate,
            recovery_call=recovery_call,
            reserved_at=utc_now(),
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def adopt_reconciled_usage(
        self,
        *,
        stage: str,
        model_id: str,
        usage: Mapping[str, Any],
    ) -> BudgetReservation:
        """Count an already persisted paid call when resuming the same run.

        A new process must not regain the full interaction or USD allowance
        merely because earlier successful calls are now cache artifacts.
        """

        input_tokens = int(usage.get("total_input_tokens") or 0)
        cached_tokens = int(usage.get("total_cached_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        thought_tokens = int(usage.get("total_thought_tokens") or 0)
        cost = actual_usage_cost(
            model_id=model_id,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
        )
        estimate = PaidCallEstimate(
            stage=stage,
            model_id=model_id,
            media_resolution="low",
            estimated_input_tokens=input_tokens,
            max_output_tokens=max(1, output_tokens),
            reserved_thought_tokens=thought_tokens,
            retry_allowance=0,
            worst_case_interactions=1,
            worst_case_cost_usd=cost,
        )
        reservation = BudgetReservation(
            reservation_id=uuid.uuid4().hex,
            estimate=estimate,
            recovery_call=False,
            state="reconciled",
            actual_input_tokens=input_tokens,
            actual_cached_input_tokens=cached_tokens,
            actual_output_tokens=output_tokens,
            actual_thought_tokens=thought_tokens,
            actual_cost_usd=cost,
            reserved_at=utc_now(),
            reconciled_at=utc_now(),
            reconciliation_basis="actual_usage",
        )
        projected_interactions = self.committed_interactions + 1
        projected_cost = self.actual_cost_usd + cost
        interaction_limit = self._interaction_limit_after(
            stage=stage,
            interactions=1,
        )
        if projected_interactions > interaction_limit:
            raise BudgetExceeded(
                "persisted paid interactions consume mandatory future "
                "interaction holds or exceed the configured cap"
            )
        if projected_cost > self.max_cost_usd + 1e-9:
            raise BudgetExceeded(
                "persisted paid usage already exceeds the configured cost cap"
            )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def adopt_conservative_dispatch(
        self,
        *,
        dispatch_id: str,
        estimate: PaidCallEstimate,
    ) -> BudgetReservation:
        """Adopt one ambiguous persisted dispatch at its worst-case reserve.

        ``dispatch_id`` is stable across copied artifact aliases and process
        restarts. Re-adopting the same journal into one ledger is therefore a
        no-op instead of double-counting the paid interaction.
        """

        for reservation in self._reservations.values():
            if reservation.dispatch_id == dispatch_id:
                return reservation
        if estimate.retry_allowance != 0 or estimate.worst_case_interactions != 1:
            raise ValueError(
                "a dispatch journal must describe exactly one provider attempt"
            )
        reservation = BudgetReservation(
            reservation_id=uuid.uuid4().hex,
            estimate=estimate,
            recovery_call=False,
            state="reconciled",
            actual_input_tokens=estimate.estimated_input_tokens,
            actual_cached_input_tokens=0,
            actual_output_tokens=estimate.max_output_tokens,
            actual_thought_tokens=estimate.reserved_thought_tokens,
            actual_cost_usd=estimate.worst_case_cost_usd,
            reserved_at=utc_now(),
            reconciled_at=utc_now(),
            reconciliation_basis="conservative_worst_case",
            dispatch_id=dispatch_id,
        )
        projected_interactions = self.committed_interactions + 1
        projected_cost = self.actual_cost_usd + estimate.worst_case_cost_usd
        interaction_limit = self._interaction_limit_after(
            stage=estimate.stage,
            interactions=1,
        )
        if projected_interactions > interaction_limit:
            raise BudgetExceeded(
                "persisted ambiguous dispatches already exceed the configured "
                "interaction headroom"
            )
        if projected_cost > self.max_cost_usd + 1e-9:
            raise BudgetExceeded(
                "persisted ambiguous dispatches already exceed the configured "
                "cost cap"
            )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def reconcile(
        self,
        reservation_id: str,
        *,
        usage: Mapping[str, Any],
        model_id: str | None = None,
    ) -> BudgetReservation:
        reservation = self._reservations[reservation_id]
        if reservation.state != "reserved":
            raise ValueError("only an active budget reservation can reconcile")
        input_tokens = int(usage.get("total_input_tokens") or 0)
        cached_tokens = int(usage.get("total_cached_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        thought_tokens = int(usage.get("total_thought_tokens") or 0)
        resolved_model = model_id or reservation.estimate.model_id
        reservation.actual_input_tokens = input_tokens
        reservation.actual_cached_input_tokens = cached_tokens
        reservation.actual_output_tokens = output_tokens
        reservation.actual_thought_tokens = thought_tokens
        reservation.actual_cost_usd = actual_usage_cost(
            model_id=resolved_model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
        )
        reservation.state = "reconciled"
        reservation.reconciled_at = utc_now()
        reservation.reconciliation_basis = "actual_usage"
        if self.actual_cost_usd > self.max_cost_usd + 1e-9:
            raise BudgetExceeded("actual usage exceeded the configured cost cap")
        return reservation

    def reconcile_conservative_dispatch(
        self,
        reservation_id: str,
        *,
        dispatch_id: str,
    ) -> BudgetReservation:
        """Charge one started request at its reserve when usage is unavailable."""

        reservation = self._reservations[reservation_id]
        if reservation.state != "reserved":
            raise ValueError("only an active budget reservation can reconcile")
        estimate = reservation.estimate
        if estimate.retry_allowance != 0 or estimate.worst_case_interactions != 1:
            raise ValueError(
                "a dispatch journal must describe exactly one provider attempt"
            )
        reservation.actual_input_tokens = estimate.estimated_input_tokens
        reservation.actual_cached_input_tokens = 0
        reservation.actual_output_tokens = estimate.max_output_tokens
        reservation.actual_thought_tokens = estimate.reserved_thought_tokens
        reservation.actual_cost_usd = estimate.worst_case_cost_usd
        reservation.state = "reconciled"
        reservation.reconciled_at = utc_now()
        reservation.reconciliation_basis = "conservative_worst_case"
        reservation.dispatch_id = dispatch_id
        if self.actual_cost_usd > self.max_cost_usd + 1e-9:
            raise BudgetExceeded("conservative usage exceeded the configured cost cap")
        return reservation

    def cancel_before_dispatch(self, reservation_id: str) -> None:
        reservation = self._reservations[reservation_id]
        if reservation.state != "reserved":
            raise ValueError("only an undispatched reservation can be cancelled")
        reservation.state = "cancelled"

    def report(self) -> dict[str, Any]:
        stages: dict[str, dict[str, int | float]] = {}
        for reservation in self._reservations.values():
            if reservation.state == "cancelled":
                continue
            stage = reservation.estimate.stage
            row = stages.setdefault(
                stage,
                {
                    "reserved_interactions": 0,
                    "reserved_cost_usd": 0.0,
                    "actual_input_tokens": 0,
                    "actual_cached_input_tokens": 0,
                    "actual_output_tokens": 0,
                    "actual_thought_tokens": 0,
                    "actual_cost_usd": 0.0,
                },
            )
            row["reserved_interactions"] = int(
                row["reserved_interactions"]
            ) + reservation.estimate.worst_case_interactions
            row["reserved_cost_usd"] = round(
                float(row["reserved_cost_usd"])
                + reservation.estimate.worst_case_cost_usd,
                8,
            )
            for field in (
                "actual_input_tokens",
                "actual_cached_input_tokens",
                "actual_output_tokens",
                "actual_thought_tokens",
            ):
                row[field] = int(row[field]) + int(
                    getattr(reservation, field) or 0
                )
            row["actual_cost_usd"] = round(
                float(row["actual_cost_usd"])
                + (reservation.actual_cost_usd or 0.0),
                8,
            )
        return {
            "contract_version": self.contract_version,
            "max_cost_usd": self.max_cost_usd,
            "max_interactions": self.max_interactions,
            "interaction_guard": self.interaction_guard,
            "reserved_recovery_fraction": self.reserved_recovery_fraction,
            "mandatory_stage_minimums": self.mandatory_stage_minimums,
            "mandatory_stage_cost_holds": self.mandatory_stage_cost_holds,
            "mandatory_interaction_holds": {
                stage: {
                    "minimum_interactions": minimum,
                    "committed_interactions": (
                        self._committed_mandatory_interactions()[stage]
                    ),
                    "remaining_held_interactions": (
                        self.remaining_mandatory_interaction_holds()[stage]
                    ),
                }
                for stage, minimum in self.mandatory_stage_minimums.items()
            },
            "remaining_held_interactions": sum(
                self.remaining_mandatory_interaction_holds().values()
            ),
            "mandatory_cost_holds": {
                stage: {
                    "minimum_cost_usd": minimum,
                    "committed_cost_ceiling_usd": (
                        self._committed_mandatory_cost_ceiling()[stage]
                    ),
                    "remaining_held_cost_usd": (
                        self.remaining_mandatory_cost_holds()[stage]
                    ),
                }
                for stage, minimum
                in self.mandatory_stage_cost_holds.items()
            },
            "remaining_held_cost_usd": round(
                sum(self.remaining_mandatory_cost_holds().values()),
                8,
            ),
            "available_unheld_interactions": max(
                0,
                self.max_interactions
                - self.committed_interactions
                - self.interaction_guard
                - sum(
                    self.remaining_mandatory_interaction_holds().values()
                ),
            ),
            "actual_cost_usd": self.actual_cost_usd,
            "active_reserved_cost_usd": self.reserved_cost_usd,
            "committed_interactions": self.committed_interactions,
            "remaining_interactions": (
                self.max_interactions
                - self.committed_interactions
                - self.interaction_guard
            ),
            "circuit_breaker_remaining_interactions": (
                self.max_interactions - self.committed_interactions
            ),
            "remaining_cost_usd": round(
                self.max_cost_usd
                - self.actual_cost_usd
                - self.reserved_cost_usd,
                8,
            ),
            "stages": dict(sorted(stages.items())),
            "reservations": [
                {
                    **asdict(item),
                    "estimate": asdict(item.estimate),
                }
                for item in self._reservations.values()
            ],
            "created_at": self.created_at,
            "generated_at": utc_now(),
        }

    def write_report(self, path: Path) -> None:
        write_json(path, self.report())
