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
EvidenceFulfillmentLevel = Literal[
    "contextual_identity",
    "visible_state",
    "visible_result",
    "direct_demonstration",
]
ClaimSupportLevel = Literal[
    "illustrative_only",
    "observable_state",
    "observable_result",
    "direct",
]
ExactEventRequirement = Literal["none", "required_when_selected"]
EXACT_EVENT_RESOLVER_VERSION = "exact-event-frame-selection-v3"
_FULFILLMENT_STRENGTH: dict[EvidenceFulfillmentLevel, int] = {
    "contextual_identity": 0,
    "visible_state": 1,
    "visible_result": 2,
    "direct_demonstration": 3,
}


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


class EvidenceFulfillmentAlternative(FrozenStrictModel):
    """One authorized way to fulfill a beat without changing evidence truth.

    The alternative binds claim strength separately from exact-event
    availability.  In particular, contextual footage may keep a chapter in the
    edit, but it may not manufacture a direct demonstration or an exact event.
    """

    fulfillment_level: EvidenceFulfillmentLevel
    accepted_evidence_provenance: tuple[
        FeatureEvidenceProvenance, ...
    ] = Field(min_length=1)
    required_observable_predicates: tuple[str, ...] = ()
    claim_support_level: ClaimSupportLevel
    exact_event_requirement: ExactEventRequirement
    visual_events: tuple[EditorialVisualEvent, ...] = Field(
        default=(),
        max_length=8,
    )
    degradation_codes: tuple[str, ...] = ()
    copy_suppression_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_alternative(self) -> "EvidenceFulfillmentAlternative":
        if len(set(self.accepted_evidence_provenance)) != len(
            self.accepted_evidence_provenance
        ):
            raise ValueError(
                "fulfillment evidence provenance values must be unique"
            )
        if len(set(self.required_observable_predicates)) != len(
            self.required_observable_predicates
        ):
            raise ValueError(
                "fulfillment observable predicates must be unique"
            )
        event_types = [event.event_type for event in self.visual_events]
        if len(set(event_types)) != len(event_types):
            raise ValueError(
                "fulfillment visual event types must be unique"
            )
        if len(set(self.degradation_codes)) != len(self.degradation_codes):
            raise ValueError("fulfillment degradation codes must be unique")
        if len(set(self.copy_suppression_codes)) != len(
            self.copy_suppression_codes
        ):
            raise ValueError(
                "fulfillment copy-suppression codes must be unique"
            )
        if (
            self.exact_event_requirement == "required_when_selected"
            and not self.visual_events
        ):
            raise ValueError(
                "exact-event fulfillment requires at least one visual event"
            )
        if (
            self.exact_event_requirement == "none"
            and self.visual_events
        ):
            raise ValueError(
                "non-exact fulfillment cannot declare exact visual events"
            )
        if self.fulfillment_level == "contextual_identity":
            if self.claim_support_level != "illustrative_only":
                raise ValueError(
                    "contextual identity footage is illustrative only"
                )
            if self.exact_event_requirement != "none":
                raise ValueError(
                    "contextual identity footage cannot require an exact event"
                )
            if (
                "contextual_visual_substitution"
                not in self.degradation_codes
            ):
                raise ValueError(
                    "contextual fallback must record its substitution"
                )
            if (
                "specific_claim_copy_suppressed"
                not in self.copy_suppression_codes
            ):
                raise ValueError(
                    "contextual fallback must suppress specific claim copy"
                )
        return self


class EvidenceFulfillmentObservation(FrozenStrictModel):
    """Immutable evidence facts available for one bounded candidate."""

    candidate_id: str = Field(min_length=1)
    evidence_provenance: FeatureEvidenceProvenance
    observable_predicates: tuple[str, ...] = ()
    available_visual_event_types: tuple[VisualEventType, ...] | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "EvidenceFulfillmentObservation":
        if len(set(self.observable_predicates)) != len(
            self.observable_predicates
        ):
            raise ValueError("observable predicates must be unique")
        if (
            self.available_visual_event_types is not None
            and len(set(self.available_visual_event_types))
            != len(self.available_visual_event_types)
        ):
            raise ValueError("available visual event types must be unique")
        return self


class EditorialBeatFulfillmentSelection(FrozenStrictModel):
    """The strongest eligible, locally verified fulfillment for one beat."""

    beat_id: str
    candidate_id: str
    fulfillment_level: EvidenceFulfillmentLevel
    evidence_provenance: FeatureEvidenceProvenance
    claim_support_level: ClaimSupportLevel
    visual_events: tuple[EditorialVisualEvent, ...] = ()
    exact_event_required: bool
    degradation_codes: tuple[str, ...] = ()
    copy_suppression_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_selection(self) -> "EditorialBeatFulfillmentSelection":
        if self.exact_event_required != bool(self.visual_events):
            raise ValueError(
                "selected exact-event requirement must match visual events"
            )
        if self.fulfillment_level == "contextual_identity":
            if self.exact_event_required:
                raise ValueError(
                    "contextual fulfillment cannot fabricate exact events"
                )
            if (
                "contextual_visual_substitution"
                not in self.degradation_codes
                or "specific_claim_copy_suppressed"
                not in self.copy_suppression_codes
            ):
                raise ValueError(
                    "contextual selection requires degradation and copy "
                    "suppression"
                )
        return self


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
        default=(),
        max_length=8,
    )
    minimum_fulfillment_level: EvidenceFulfillmentLevel | None = None
    fulfillment_alternatives: tuple[
        EvidenceFulfillmentAlternative, ...
    ] = Field(default=(), max_length=4)
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
        if not self.fulfillment_alternatives:
            if self.minimum_fulfillment_level is not None:
                raise ValueError(
                    "minimum fulfillment level requires alternatives"
                )
            if not self.visual_events:
                raise ValueError(
                    "legacy editorial beats require at least one visual event"
                )
            return self
        if self.minimum_fulfillment_level is None:
            raise ValueError(
                "fulfillment alternatives require a minimum level"
            )
        levels = [
            alternative.fulfillment_level
            for alternative in self.fulfillment_alternatives
        ]
        if len(set(levels)) != len(levels):
            raise ValueError("fulfillment alternative levels must be unique")
        minimum_strength = _FULFILLMENT_STRENGTH[
            self.minimum_fulfillment_level
        ]
        if any(
            _FULFILLMENT_STRENGTH[level] < minimum_strength
            for level in levels
        ):
            raise ValueError(
                "fulfillment alternatives cannot fall below the minimum level"
            )
        return self

    @property
    def effective_fulfillment_alternatives(
        self,
    ) -> tuple[EvidenceFulfillmentAlternative, ...]:
        """Expose legacy contracts as one direct, exact-event alternative."""

        if self.fulfillment_alternatives:
            return self.fulfillment_alternatives
        return (
            EvidenceFulfillmentAlternative(
                fulfillment_level="direct_demonstration",
                accepted_evidence_provenance=(
                    self.allowed_evidence_provenance
                ),
                claim_support_level="direct",
                exact_event_requirement="required_when_selected",
                visual_events=self.visual_events,
            ),
        )


def select_strongest_evidence_fulfillment(
    contract: EditorialBeatContract,
    observations: Sequence[EvidenceFulfillmentObservation],
) -> EditorialBeatFulfillmentSelection:
    """Select the strongest contract-authorized evidence without relabeling it.

    A known-empty ``available_visual_event_types`` tuple means exact-event
    inspection found no declared event and makes an exact alternative
    ineligible. ``None`` means the event has not been resolved yet, so the
    selected alternative remains fail-closed and still requires resolution.
    """

    alternatives = sorted(
        contract.effective_fulfillment_alternatives,
        key=lambda alternative: _FULFILLMENT_STRENGTH[
            alternative.fulfillment_level
        ],
        reverse=True,
    )
    minimum_level = (
        contract.minimum_fulfillment_level or "direct_demonstration"
    )
    minimum_strength = _FULFILLMENT_STRENGTH[minimum_level]
    for alternative in alternatives:
        if (
            _FULFILLMENT_STRENGTH[alternative.fulfillment_level]
            < minimum_strength
        ):
            continue
        required_predicates = set(
            alternative.required_observable_predicates
        )
        required_event_types = {
            event.event_type for event in alternative.visual_events
        }
        for observation in observations:
            if (
                observation.evidence_provenance
                not in alternative.accepted_evidence_provenance
            ):
                continue
            if not required_predicates.issubset(
                observation.observable_predicates
            ):
                continue
            if (
                alternative.exact_event_requirement
                == "required_when_selected"
                and observation.available_visual_event_types is not None
                and not required_event_types.issubset(
                    observation.available_visual_event_types
                )
            ):
                continue
            return EditorialBeatFulfillmentSelection(
                beat_id=contract.beat_id,
                candidate_id=observation.candidate_id,
                fulfillment_level=alternative.fulfillment_level,
                evidence_provenance=observation.evidence_provenance,
                claim_support_level=alternative.claim_support_level,
                visual_events=alternative.visual_events,
                exact_event_required=(
                    alternative.exact_event_requirement
                    == "required_when_selected"
                ),
                degradation_codes=alternative.degradation_codes,
                copy_suppression_codes=(
                    alternative.copy_suppression_codes
                ),
            )
    raise ValueError(
        "no evidence candidate satisfies the editorial beat minimum "
        f"fulfillment level: {contract.beat_id}/{minimum_level}"
    )


def bind_selected_fulfillment(
    contract: EditorialBeatContract,
    selection: EditorialBeatFulfillmentSelection,
) -> EditorialBeatContract:
    """Materialize only the selected alternative for exact-event execution."""

    if selection.beat_id != contract.beat_id:
        raise ValueError("fulfillment selection belongs to another beat")
    return contract.model_copy(
        update={
            "allowed_evidence_provenance": (
                selection.evidence_provenance,
            ),
            "visual_events": selection.visual_events,
        }
    )


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
    exact_event_lock_sha256s: tuple[str, ...] = ()
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
    """Bind locks by beat-qualified ID; allow unambiguous legacy type binding.

    A repeated event type in two beats is not interchangeable evidence. New
    resolvers must therefore return ``<beat_id>:<event_type>``. Historical
    arbitrary IDs remain compatible only when one requested beat owns that
    event type.
    """

    expected_by_type: dict[str, list[str]] = {}
    for contract in contracts:
        for event in contract.visual_events:
            expected_by_type.setdefault(event.event_type, []).append(
                f"{contract.beat_id}:{event.event_type}"
            )
    expected_ids = {
        event_id
        for event_ids in expected_by_type.values()
        for event_id in event_ids
    }
    if len(locks) != len(expected_ids):
        raise ValueError("grouped ExactEventLocks omitted a requested event")
    bound: list[ExactEventLockV2] = []
    consumed: set[str] = set()
    for lock in locks:
        event_ids_for_type = expected_by_type.get(lock.event_type)
        if not event_ids_for_type:
            raise ValueError(
                "grouped ExactEventLocks returned an undeclared event type: "
                f"{lock.event_type}"
            )
        if lock.event_id in expected_ids:
            bound_event_id = lock.event_id
            if bound_event_id not in event_ids_for_type:
                raise ValueError(
                    "grouped ExactEventLock beat-qualified ID disagrees with "
                    f"its event type: {bound_event_id}/{lock.event_type}"
                )
        elif len(event_ids_for_type) == 1:
            bound_event_id = event_ids_for_type[0]
        else:
            raise ValueError(
                "ambiguous legacy ExactEventLock ID cannot bind repeated "
                f"event type across beats: {lock.event_type}"
            )
        if bound_event_id in consumed:
            raise ValueError(
                "grouped ExactEventLocks duplicated a beat-qualified event: "
                f"{bound_event_id}"
            )
        consumed.add(bound_event_id)
        bound.append(lock.model_copy(update={"event_id": bound_event_id}))
    if consumed != expected_ids:
        missing = sorted(expected_ids - consumed)
        raise ValueError(
            "grouped ExactEventLocks omitted requested beat events: "
            + ", ".join(missing)
        )
    return tuple(bound)


def hard_exact_event_requirements_satisfied(
    contracts: Sequence[EditorialBeatContract],
    locks: Sequence[ExactEventLockV2],
) -> bool:
    """Check hard exact evidence by beat-qualified ID, never global type."""

    required = {
        f"{contract.beat_id}:{event.event_type}": event.event_type
        for contract in contracts
        if contract.priority == "hard"
        for event in contract.visual_events
    }
    observed: dict[str, str] = {}
    for lock in locks:
        if lock.event_id in observed:
            return False
        observed[lock.event_id] = lock.event_type
    return all(
        observed.get(event_id) == event_type
        for event_id, event_type in required.items()
    )


def write_exact_event_bundle(
    output_dir: Path,
    *,
    contracts: Sequence[EditorialBeatContract],
    locks: Sequence[ExactEventLockV2],
    selected_windows: Sequence[Mapping[str, Any]],
    aspect: str | None = None,
) -> dict[str, Path]:
    """Persist the two grouped selected-window artifacts atomically by content."""

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_path = output_dir / "editorial-beat-contracts.json"
    locks_path = output_dir / "exact-event-locks.json"
    contracts_payload = {
        "contract_version": "editorial-beat-contract-bundle-v1",
        **({"aspect": aspect} if aspect is not None else {}),
        "beats": [contract.model_dump(mode="json") for contract in contracts],
    }
    locks_payload = {
        "contract_version": "exact-event-lock-bundle-v2",
        **({"aspect": aspect} if aspect is not None else {}),
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
