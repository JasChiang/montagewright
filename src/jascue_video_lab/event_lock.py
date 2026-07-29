"""Exact-frame event evidence compiled from the existing dense-frame path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from PIL import Image, ImageChops, ImageStat
from pydantic import Field, model_validator

from .autonomous_policy import (
    AutonomousEditPolicy,
    DecisionAuthorityV2,
    validate_authority_binding,
)
from .models import (
    DenseFrame,
    DenseFrameCatalog,
    FeatureEvidenceProvenance,
    FrozenStrictModel,
    TrimIntentDecision,
)
from .media import sha256_file
from .schema import gemini_response_schema
from .storage import utc_now


VisualEventType = Literal[
    "camera_gesture_apex",
    "generation_result_stable_start",
    "watch_ui_state_change",
    "underwater_lift_apex",
    "group_laugh_reaction_peak",
    "freeze_start",
    "action_onset",
    "action_apex",
    "state_change",
    "result_stable_start",
    "reaction_peak",
    "clean_out",
]
CueRelation = Literal[
    "accent",
    "principal_downbeat",
    "music_emphasis",
    "phrase_ending",
]
EXACT_EVENT_RESOLVER_VERSION = "exact-event-frame-selection-v2"


class ReadabilityDuration(FrozenStrictModel):
    minimum_readable_frames: int = Field(ge=1, le=600)
    preferred_frames: int = Field(ge=1, le=900)
    maximum_frames: int = Field(ge=1, le=1_800)

    @model_validator(mode="after")
    def validate_range(self) -> "ReadabilityDuration":
        if not (
            self.minimum_readable_frames
            <= self.preferred_frames
            <= self.maximum_frames
        ):
            raise ValueError(
                "readability must satisfy minimum <= preferred <= maximum"
            )
        return self


class EditorialVisualEvent(FrozenStrictModel):
    event_type: VisualEventType
    cue_relation: CueRelation
    tolerance_frames: int = Field(ge=0, le=24)


class SyntheticMotionPermission(FrozenStrictModel):
    before_event: Literal["forbidden", "optional", "required"] = "forbidden"
    after_event: Literal[
        "forbidden", "optional", "optional_emphasis", "required"
    ] = "optional"


class EditorialBeatContract(FrozenStrictModel):
    contract_version: Literal["editorial-beat-contract-v1"] = (
        "editorial-beat-contract-v1"
    )
    beat_id: str = Field(min_length=1)
    feature_id: str | None = Field(default=None, min_length=1)
    priority: Literal["hard", "preferred", "optional"]
    evidence_query_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_target_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    allowed_evidence_provenance: tuple[
        FeatureEvidenceProvenance, ...
    ] = (
        "direct_physical_action",
        "direct_ui_interaction",
        "direct_result",
    )
    narrative_function: Literal[
        "opening",
        "setup",
        "feature_evidence",
        "comparison",
        "reaction",
        "global_energy_peak",
        "closing",
    ]
    visual_events: tuple[EditorialVisualEvent, ...] = Field(
        min_length=1,
        max_length=8,
    )
    duration: ReadabilityDuration
    relation_mode: Literal[
        "single_subject",
        "sequential_focus",
        "simultaneous_relation",
        "context_detail",
    ]
    allowed_reconstruction: tuple[
        Literal[
            "continuous",
            "hard_cut_after_result",
            "hard_cut_between_views",
            "two_panel_layout",
            "solid_fit",
            "intentional_freeze",
        ],
        ...,
    ] = Field(min_length=1)
    synthetic_motion: SyntheticMotionPermission = SyntheticMotionPermission()

    @model_validator(mode="after")
    def validate_contract(self) -> "EditorialBeatContract":
        if len(set(self.required_target_ids)) != len(self.required_target_ids):
            raise ValueError("beat target IDs must be unique")
        event_types = [event.event_type for event in self.visual_events]
        if len(set(event_types)) != len(event_types):
            raise ValueError("beat visual event types must be unique")
        if len(set(self.allowed_reconstruction)) != len(
            self.allowed_reconstruction
        ):
            raise ValueError("allowed reconstruction modes must be unique")
        if not self.allowed_evidence_provenance:
            raise ValueError(
                "editorial beats require at least one evidence provenance"
            )
        if len(set(self.allowed_evidence_provenance)) != len(
            self.allowed_evidence_provenance
        ):
            raise ValueError("allowed evidence provenance values must be unique")
        return self


class ExactEventSelection(FrozenStrictModel):
    event_id: str
    event_type: VisualEventType
    selected_frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    support_start_frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    support_end_frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    confidence: float = Field(ge=0.0, le=1.0)


class ExactEventSelectionGroup(FrozenStrictModel):
    source_asset_id: str
    catalog_event_id: str
    selections: tuple[ExactEventSelection, ...] = Field(
        max_length=8,
    )


class ExactEventResolverProvenance(FrozenStrictModel):
    local_bracket_method: Literal["frame_difference"]
    sampling_fps: float = Field(gt=0, le=8)
    gemini_interaction_id: str
    contact_sheet_hashes: tuple[str, ...] = Field(min_length=1)


class ExactEventLockV2(FrozenStrictModel):
    contract_version: Literal["exact-event-lock-v2"] = "exact-event-lock-v2"
    event_id: str
    event_type: VisualEventType
    source_asset_id: str
    source_frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    source_pts: int
    source_time_ms: int = Field(ge=0)
    source_frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_provenance: FeatureEvidenceProvenance = "unknown"
    support_window_start_frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    support_window_end_frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    support_window_start_ms: int = Field(ge=0)
    support_window_end_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    resolver: ExactEventResolverProvenance
    input_artifact_hashes: tuple[str, ...] = Field(min_length=1)
    generated_at: str

    @model_validator(mode="after")
    def validate_window(self) -> "ExactEventLockV2":
        if not (
            self.support_window_start_ms
            <= self.source_time_ms
            <= self.support_window_end_ms
        ):
            raise ValueError("exact event must lie inside its support window")
        if (
            self.support_window_start_frame_id
            > self.source_frame_id
            or self.source_frame_id > self.support_window_end_frame_id
        ):
            raise ValueError("exact event frame ID must lie inside support IDs")
        return self

    def definition_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))


class CueAlignmentEvidenceV2(FrozenStrictModel):
    contract_version: Literal["cue-alignment-evidence-v2"] = (
        "cue-alignment-evidence-v2"
    )
    exact_event_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_id: str
    cue_id: str
    cue_sample_index: int = Field(ge=0)
    music_sample_rate: int = Field(gt=0)
    planned_video_frame: int = Field(ge=0)
    cue_video_frame: int = Field(ge=0)
    delta_frames: int
    tolerance_frames: int = Field(ge=0, le=24)
    passed: bool

    @model_validator(mode="after")
    def validate_delta(self) -> "CueAlignmentEvidenceV2":
        if self.delta_frames != self.planned_video_frame - self.cue_video_frame:
            raise ValueError("cue alignment delta does not match frame evidence")
        if self.passed != (abs(self.delta_frames) <= self.tolerance_frames):
            raise ValueError("cue alignment pass flag does not match tolerance")
        return self


class AuthorizedTrimIntentDecisionV2(FrozenStrictModel):
    contract_version: Literal["trim-intent-decision-v2"] = (
        "trim-intent-decision-v2"
    )
    decision: TrimIntentDecision
    authority: DecisionAuthorityV2
    approval_status: Literal["approved"] = "approved"
    requires_human_review: Literal[False] = False
    exact_event_lock_sha256s: tuple[str, ...] = Field(min_length=1)
    generated_at: str

    @model_validator(mode="after")
    def validate_decision(self) -> "AuthorizedTrimIntentDecisionV2":
        if not self.decision.usable:
            raise ValueError("automatic trim authority requires a usable decision")
        if self.authority.decision_scope != "trim_intent":
            raise ValueError("trim decision requires trim_intent authority")
        if self.decision.source_in_ms is None or self.decision.source_out_ms is None:
            raise ValueError("authorized trim requires immutable source bounds")
        return self


def bracket_dense_frames_by_difference(
    catalog: DenseFrameCatalog,
    *,
    max_frames: int = 12,
) -> tuple[DenseFrame, ...]:
    """Select an 8–12 frame local frontier without creating a second timeline."""

    if not 8 <= max_frames <= 12:
        raise ValueError("exact-event bracket must contain at most 8–12 frames")
    frames = catalog.frames
    if len(frames) <= max_frames:
        return tuple(frames)
    scores: list[tuple[float, int]] = []
    previous: Image.Image | None = None
    for index, frame in enumerate(frames):
        with Image.open(Path(frame.image_path)) as source:
            image = source.convert("L").resize((96, 54))
        if previous is not None:
            difference = ImageChops.difference(previous, image)
            score = float(ImageStat.Stat(difference).mean[0])
            scores.append((score, index))
        previous = image
    selected: set[int] = {0, len(frames) - 1}
    for _score, index in sorted(scores, reverse=True):
        selected.update({max(0, index - 1), index, min(len(frames) - 1, index + 1)})
        if len(selected) >= max_frames:
            break
    if len(selected) < min(8, len(frames)):
        step = (len(frames) - 1) / (min(8, len(frames)) - 1)
        selected.update(round(step * index) for index in range(min(8, len(frames))))
    ordered = sorted(selected)
    if len(ordered) > max_frames:
        ordered = _evenly_limit_indices(ordered, max_frames)
    return tuple(frames[index] for index in ordered)


def resolve_exact_event_locks(
    catalog: DenseFrameCatalog,
    selections: Sequence[ExactEventSelection],
    *,
    gemini_interaction_id: str,
    input_artifact_hashes: tuple[str, ...],
    evidence_provenance: FeatureEvidenceProvenance = "unknown",
) -> tuple[ExactEventLockV2, ...]:
    """Map model-selected immutable IDs to local PTS; arbitrary time is absent."""

    by_id = {frame.frame_id: frame for frame in catalog.frames}
    positions = {frame.frame_id: index for index, frame in enumerate(catalog.frames)}
    locks: list[ExactEventLockV2] = []
    seen_events: set[str] = set()
    for selection in selections:
        if selection.event_id in seen_events:
            raise ValueError("exact event IDs must be unique in one grouped call")
        seen_events.add(selection.event_id)
        unknown = {
            selection.selected_frame_id,
            selection.support_start_frame_id,
            selection.support_end_frame_id,
        } - by_id.keys()
        if unknown:
            raise ValueError(
                "Gemini selected frame IDs outside the dense catalog: "
                + ", ".join(sorted(unknown))
            )
        if not (
            positions[selection.support_start_frame_id]
            <= positions[selection.selected_frame_id]
            <= positions[selection.support_end_frame_id]
        ):
            raise ValueError("exact event support frame order is invalid")
        selected = by_id[selection.selected_frame_id]
        support_start = by_id[selection.support_start_frame_id]
        support_end = by_id[selection.support_end_frame_id]
        locks.append(
            ExactEventLockV2(
                event_id=selection.event_id,
                event_type=selection.event_type,
                source_asset_id=catalog.source_asset_id,
                source_frame_id=selected.frame_id,
                source_pts=selected.frame_pts,
                source_time_ms=selected.frame_time_ms,
                source_frame_hash=selected.frame_hash,
                evidence_provenance=evidence_provenance,
                support_window_start_frame_id=support_start.frame_id,
                support_window_end_frame_id=support_end.frame_id,
                support_window_start_ms=support_start.frame_time_ms,
                support_window_end_ms=support_end.frame_time_ms,
                confidence=selection.confidence,
                resolver=ExactEventResolverProvenance(
                    local_bracket_method="frame_difference",
                    sampling_fps=catalog.sampling_fps,
                    gemini_interaction_id=gemini_interaction_id,
                    contact_sheet_hashes=tuple(catalog.contact_sheet_hashes),
                ),
                input_artifact_hashes=input_artifact_hashes,
                generated_at=utc_now(),
            )
        )
    return tuple(locks)


def validate_exact_event_evidence_provenance(
    evidence_provenance: FeatureEvidenceProvenance,
    contracts: Sequence[EditorialBeatContract],
) -> None:
    """Fail before a paid exact-frame call when the selected shot is ineligible.

    A dense-frame resolver can locate a change inside a nested playback, but it
    cannot prove that the depicted event happened in the captured scene.  The
    editorial contract therefore binds which provenance classes may satisfy
    each requested event.
    """

    incompatible = [
        contract.beat_id
        for contract in contracts
        if evidence_provenance not in contract.allowed_evidence_provenance
    ]
    if incompatible:
        raise ValueError(
            "selected evidence provenance cannot satisfy exact-event contracts "
            f"({evidence_provenance}): "
            + ", ".join(incompatible)
        )


def exact_event_resolver_binding_sha256(
    *,
    catalog: DenseFrameCatalog,
    contracts: Sequence[EditorialBeatContract],
    model_id: str,
) -> str:
    """Bind reusable locks to the exact dense evidence and resolver contract."""

    verified_files: list[dict[str, str]] = []
    for frame in catalog.frames:
        image_path = Path(frame.image_path).expanduser().resolve(strict=True)
        image_hash = sha256_file(image_path)
        if image_hash != frame.frame_hash:
            raise ValueError(
                f"dense source frame integrity mismatch: {frame.frame_id}"
            )
        transport_path = (
            Path(frame.transport_image_path)
            .expanduser()
            .resolve(strict=True)
        )
        transport_hash = sha256_file(transport_path)
        if transport_hash != frame.transport_image_hash:
            raise ValueError(
                f"dense transport frame integrity mismatch: {frame.frame_id}"
            )
        verified_files.extend(
            (
                {
                    "role": "source_frame",
                    "frame_id": frame.frame_id,
                    "sha256": image_hash,
                },
                {
                    "role": "transport_frame",
                    "frame_id": frame.frame_id,
                    "sha256": transport_hash,
                },
            )
        )
    for index, (path_value, declared_hash) in enumerate(
        zip(
            catalog.contact_sheet_paths,
            catalog.contact_sheet_hashes,
            strict=True,
        )
    ):
        path = Path(path_value).expanduser().resolve(strict=True)
        actual_hash = sha256_file(path)
        if actual_hash != declared_hash:
            raise ValueError(
                f"dense contact sheet integrity mismatch: {index}"
            )
        verified_files.append(
            {
                "role": "contact_sheet",
                "index": str(index),
                "sha256": actual_hash,
            }
        )
    return _canonical_sha256(
        {
            "resolver_version": EXACT_EVENT_RESOLVER_VERSION,
            "model_id": model_id,
            "catalog": catalog.model_dump(mode="json"),
            "verified_files": verified_files,
            "contracts": [
                contract.model_dump(mode="json") for contract in contracts
            ],
            "response_schema": gemini_response_schema(
                ExactEventSelectionGroup
            ),
            "media_resolution": "high",
            "local_bracket_method": "frame_difference",
        }
    )


def authorize_trim_intent_decision(
    decision: TrimIntentDecision,
    *,
    exact_event_locks: Sequence[ExactEventLockV2],
    authority: DecisionAuthorityV2,
    policy: AutonomousEditPolicy,
) -> AuthorizedTrimIntentDecisionV2:
    validate_authority_binding(authority, policy)
    if authority.decision_scope != "trim_intent":
        raise ValueError("automatic trim requires trim_intent authority")
    if decision.source_in_ms is None or decision.source_out_ms is None:
        raise ValueError("automatic trim decision is missing locked bounds")
    for event_lock in exact_event_locks:
        if event_lock.source_asset_id != decision.source_asset_id:
            raise ValueError("exact event lock belongs to another source asset")
        if not (
            decision.source_in_ms
            <= event_lock.source_time_ms
            < decision.source_out_ms
        ):
            raise ValueError("exact event lock lies outside immutable trim")
    return AuthorizedTrimIntentDecisionV2(
        decision=decision,
        authority=authority,
        exact_event_lock_sha256s=tuple(
            lock.definition_sha256() for lock in exact_event_locks
        ),
        generated_at=utc_now(),
    )


def build_cue_alignment_evidence(
    event_lock: ExactEventLockV2,
    *,
    cue_id: str,
    cue_sample_index: int,
    music_sample_rate: int,
    project_event_time_ms: int,
    fps_numerator: int,
    fps_denominator: int = 1,
    tolerance_frames: int,
) -> CueAlignmentEvidenceV2:
    if fps_numerator <= 0 or fps_denominator <= 0:
        raise ValueError("video frame rate must be positive")
    cue_time_ms = cue_sample_index * 1_000 / music_sample_rate
    planned_frame = round(
        project_event_time_ms * fps_numerator / (1_000 * fps_denominator)
    )
    cue_frame = round(cue_time_ms * fps_numerator / (1_000 * fps_denominator))
    delta = planned_frame - cue_frame
    return CueAlignmentEvidenceV2(
        exact_event_lock_sha256=event_lock.definition_sha256(),
        event_id=event_lock.event_id,
        cue_id=cue_id,
        cue_sample_index=cue_sample_index,
        music_sample_rate=music_sample_rate,
        planned_video_frame=planned_frame,
        cue_video_frame=cue_frame,
        delta_frames=delta,
        tolerance_frames=tolerance_frames,
        passed=abs(delta) <= tolerance_frames,
    )


def load_editorial_beat_contracts(path: Path) -> tuple[EditorialBeatContract, ...]:
    """Load either a bare list or the checked-in fixture wrapper."""

    payload = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    rows: Any = payload.get("beats") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("editorial beat contracts must be a list or beats wrapper")
    contracts = tuple(EditorialBeatContract.model_validate(row) for row in rows)
    if len({contract.beat_id for contract in contracts}) != len(contracts):
        raise ValueError("editorial beat IDs must be unique")
    return contracts


def bind_editorial_contract_to_selected_evidence(
    contract: EditorialBeatContract,
    *,
    evidence_query_lock_sha256: str,
    required_target_ids: Sequence[str],
) -> EditorialBeatContract:
    """Replace template placeholders with the selected candidate's evidence."""

    targets = tuple(dict.fromkeys(required_target_ids))
    if not targets:
        raise ValueError("selected evidence must expose at least one target ID")
    return contract.model_copy(
        update={
            "evidence_query_lock_sha256": evidence_query_lock_sha256,
            "required_target_ids": targets,
        }
    )


def bind_grouped_event_lock_ids(
    locks: Sequence[ExactEventLockV2],
    contracts: Sequence[EditorialBeatContract],
) -> tuple[ExactEventLockV2, ...]:
    """Bind model-selected events by declared type, independent of return order."""

    expected_by_type: dict[str, list[str]] = {}
    for contract in contracts:
        for event in contract.visual_events:
            expected_by_type.setdefault(event.event_type, []).append(
                f"{contract.beat_id}:{event.event_type}"
            )
    if len(locks) != sum(len(ids) for ids in expected_by_type.values()):
        raise ValueError("grouped ExactEventLocks omitted a requested event")
    bound: list[ExactEventLockV2] = []
    for lock in locks:
        expected_ids = expected_by_type.get(lock.event_type)
        if not expected_ids:
            raise ValueError(
                "grouped ExactEventLocks returned an undeclared event type: "
                f"{lock.event_type}"
            )
        bound.append(
            lock.model_copy(update={"event_id": expected_ids.pop(0)})
        )
    if any(expected_by_type.values()):
        raise ValueError("grouped ExactEventLocks omitted a requested event type")
    return tuple(bound)


def write_exact_event_bundle(
    output_dir: Path,
    *,
    contracts: Sequence[EditorialBeatContract],
    locks: Sequence[ExactEventLockV2],
    selected_windows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Persist the two grouped selected-window artifacts atomically by content."""

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_path = output_dir / "editorial-beat-contracts.json"
    locks_path = output_dir / "exact-event-locks.json"
    contracts_payload = {
        "contract_version": "editorial-beat-contract-bundle-v1",
        "beats": [contract.model_dump(mode="json") for contract in contracts],
    }
    locks_payload = {
        "contract_version": "exact-event-lock-bundle-v2",
        "locks": [lock.model_dump(mode="json") for lock in locks],
        "selected_windows": [dict(window) for window in selected_windows],
    }
    from .storage import write_json

    write_json(contracts_path, contracts_payload)
    write_json(locks_path, locks_payload)
    return {
        "editorial_beat_contracts": contracts_path.resolve(),
        "exact_event_locks": locks_path.resolve(),
    }


def _evenly_limit_indices(indices: list[int], limit: int) -> list[int]:
    if len(indices) <= limit:
        return indices
    return [
        indices[round(position * (len(indices) - 1) / (limit - 1))]
        for position in range(limit)
    ]


def _canonical_sha256(value: Mapping[str, object] | object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
