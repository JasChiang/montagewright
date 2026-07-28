from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping
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
            # immutable attempt path, not payload equality, defines cardinality.
            immutable_attempt_payloads.add(fingerprint)
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


class BudgetLedger:
    """Reserve worst-case paid work, then reconcile immutable usage evidence."""

    contract_version = "budget-ledger-v1"

    def __init__(
        self,
        *,
        max_cost_usd: float,
        max_interactions: int,
        reserved_recovery_fraction: float = 0.20,
    ) -> None:
        if max_cost_usd <= 0 or max_interactions <= 0:
            raise ValueError("budget caps must be positive")
        if not 0.20 <= reserved_recovery_fraction <= 0.50:
            raise ValueError("recovery reserve must be between 20% and 50%")
        self.max_cost_usd = max_cost_usd
        self.max_interactions = max_interactions
        self.reserved_recovery_fraction = reserved_recovery_fraction
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

    def reserve(
        self,
        estimate: PaidCallEstimate,
        *,
        recovery_call: bool = False,
    ) -> BudgetReservation:
        projected_cost = (
            self.actual_cost_usd
            + self.reserved_cost_usd
            + estimate.worst_case_cost_usd
        )
        projected_interactions = (
            self.committed_interactions + estimate.worst_case_interactions
        )
        if recovery_call:
            cost_limit = self.max_cost_usd
            interaction_limit = self.max_interactions
        else:
            usable_fraction = 1.0 - self.reserved_recovery_fraction
            cost_limit = self.max_cost_usd * usable_fraction
            interaction_limit = int(self.max_interactions * usable_fraction)
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
        if self.actual_cost_usd > self.max_cost_usd + 1e-9:
            raise BudgetExceeded("actual usage exceeded the configured cost cap")
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
            "reserved_recovery_fraction": self.reserved_recovery_fraction,
            "actual_cost_usd": self.actual_cost_usd,
            "active_reserved_cost_usd": self.reserved_cost_usd,
            "committed_interactions": self.committed_interactions,
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
