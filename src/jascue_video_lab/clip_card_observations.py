from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .models import (
    EntityKind,
    EvidenceOriginObservation,
    EvidenceRelation,
    EvidenceModality,
    FeatureEvidenceProvenance,
    FrozenStrictModel,
    FullClipCard,
    FullClipEvent,
    ModelProvenance,
    evidence_relation_from_legacy,
)


SUPPLEMENT_CONTRACT_VERSION = "clip-observation-supplement-v3"
SUPPLEMENT_REQUEST_BINDING_VERSION = "clip-observation-request-binding-v1"


class AssessmentStatus(StrEnum):
    NOT_ASSESSED = "not_assessed"
    ASSESSED_ABSENT = "assessed_absent"
    ASSESSED_PRESENT = "assessed_present"
    NOT_APPLICABLE = "not_applicable"


class ObservationBasis(StrEnum):
    FULL_CLIP_VIDEO = "full_clip_video"
    EVENT_PLUS_CONTEXT_VIDEO = "event_plus_context_video"
    EVENT_VIDEO = "event_video"
    SELECTED_FRAMES = "selected_frames"
    MEDIA_METADATA = "media_metadata"


class EditingClaim(StrEnum):
    TOPICAL_RELEVANCE = "topical_relevance"
    COMPLETE_TRIM = "complete_trim"
    SEQUENTIAL_VIRTUAL_CAMERA = "sequential_virtual_camera"
    SIMULTANEOUS_RELATION = "simultaneous_relation"
    READABLE_CONTENT = "readable_content"
    SOURCE_AUDIO = "source_audio"


class ClaimDecision(StrEnum):
    READY = "ready"
    REQUEST_SUPPLEMENT = "request_supplement"
    ABSTAIN = "abstain"


class EventCapabilityManifest(FrozenStrictModel):
    evidence_origin: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    action_structure: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    evidence_roles: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    observable_beats: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    readability: AssessmentStatus = AssessmentStatus.NOT_ASSESSED
    audio_role: AssessmentStatus = AssessmentStatus.NOT_ASSESSED


class ObservableBeat(FrozenStrictModel):
    """Aspect-neutral, ordered observation; never an executable camera move."""

    beat_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: Literal[
        "setup",
        "action_onset",
        "interaction",
        "state_change",
        "result",
        "reaction",
        "readable_hold",
    ]
    entity_ids: list[str] = Field(min_length=1, max_length=6)
    relation_mode: Literal[
        "independent",
        "simultaneous_required",
        "sequentially_reconstructable",
        "unknown",
    ]
    observable_predicate: str = Field(min_length=1, max_length=800)
    transition_condition: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_entities(self) -> "ObservableBeat":
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("observable beat entity IDs must be unique")
        if self.relation_mode == "simultaneous_required" and len(self.entity_ids) < 2:
            raise ValueError("simultaneous_required beat needs at least two entities")
        return self


class EvidenceRoleMap(FrozenStrictModel):
    primary_subject_ids: list[str] = Field(default_factory=list)
    relation_carrier_ids: list[str] = Field(default_factory=list)
    context_anchor_ids: list[str] = Field(default_factory=list)
    result_evidence_ids: list[str] = Field(default_factory=list)
    readable_region_ids: list[str] = Field(default_factory=list)
    shared_context_required: bool = False
    relative_scale_required: bool = False
    observable_reason: str = ""

    @model_validator(mode="after")
    def validate_unique_roles(self) -> "EvidenceRoleMap":
        for name in (
            "primary_subject_ids",
            "relation_carrier_ids",
            "context_anchor_ids",
            "result_evidence_ids",
            "readable_region_ids",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicate entity IDs")
        return self

    def referenced_entity_ids(self) -> set[str]:
        return {
            entity_id
            for name in (
                "primary_subject_ids",
                "relation_carrier_ids",
                "context_anchor_ids",
                "result_evidence_ids",
                "readable_region_ids",
            )
            for entity_id in getattr(self, name)
        }


class ReadabilityObservation(FrozenStrictModel):
    entity_id: str
    necessity: Literal["must_read", "helpful", "decorative"]
    temporal_behavior: Literal["stable", "transient", "changes", "unknown"]
    reading_load: Literal["glance", "short", "sustained", "unknown"]
    observable_reason: str = Field(min_length=1, max_length=800)


class AudioRoleObservation(FrozenStrictModel):
    speech: Literal["none", "single", "speaker_turns", "overlap", "uncertain"]
    intelligibility: Literal["clear", "partial", "poor", "not_applicable"]
    source_audio_value: Literal["essential", "useful", "accent", "ambience", "dispensable"]
    observable_sync_cues: list[str] = Field(default_factory=list, max_length=8)


class EventObservationSupplement(FrozenStrictModel):
    event_id: str
    event_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_basis: ObservationBasis | None = None
    audio_included: bool = False
    evidence_provenance: FeatureEvidenceProvenance = "unknown"
    evidence_origin: EvidenceOriginObservation | None = None
    capabilities: EventCapabilityManifest
    source_action_completeness: Literal[
        "complete",
        "missing_start",
        "missing_result",
        "reaction_only",
        "uncertain",
    ] = "uncertain"
    clean_entry: Literal[
        "stable",
        "already_in_motion",
        "occluded",
        "camera_unsettled",
        "unknown",
    ] = "unknown"
    clean_exit: Literal[
        "stable_result",
        "action_continues",
        "reset",
        "occluded",
        "camera_unsettled",
        "unknown",
    ] = "unknown"
    evidence_roles: EvidenceRoleMap = Field(default_factory=EvidenceRoleMap)
    observable_beats: list[ObservableBeat] = Field(default_factory=list, max_length=10)
    readability: list[ReadabilityObservation] = Field(default_factory=list, max_length=8)
    audio_role: AudioRoleObservation | None = None
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_capability_payloads(self) -> "EventObservationSupplement":
        payload_presence = {
            "evidence_origin": self.evidence_origin is not None,
            "action_structure": self.source_action_completeness != "uncertain"
            or self.clean_entry != "unknown"
            or self.clean_exit != "unknown",
            "evidence_roles": bool(self.evidence_roles.referenced_entity_ids())
            or self.evidence_roles.shared_context_required
            or self.evidence_roles.relative_scale_required,
            "observable_beats": bool(self.observable_beats),
            "readability": bool(self.readability),
            "audio_role": self.audio_role is not None,
        }
        for capability, present in payload_presence.items():
            status = getattr(self.capabilities, capability)
            if status == AssessmentStatus.ASSESSED_PRESENT and not present:
                raise ValueError(
                    f"{capability}=assessed_present requires an observation payload"
                )
            if status in {
                AssessmentStatus.NOT_ASSESSED,
                AssessmentStatus.ASSESSED_ABSENT,
                AssessmentStatus.NOT_APPLICABLE,
            } and present:
                raise ValueError(f"{capability} payload conflicts with status {status}")
        if (
            self.evidence_origin is not None
            and self.evidence_provenance != "unknown"
            and self.evidence_origin.relation
            != evidence_relation_from_legacy(self.evidence_provenance)
        ):
            raise ValueError(
                "legacy evidence_provenance conflicts with generic evidence_origin"
            )
        assessed = {
            name: getattr(self.capabilities, name)
            for name in type(self.capabilities).model_fields
            if getattr(self.capabilities, name) != AssessmentStatus.NOT_ASSESSED
        }
        if assessed and self.observation_basis is None:
            raise ValueError("assessed capabilities require an observation basis")
        if any(
            status == AssessmentStatus.ASSESSED_ABSENT
            for status in assessed.values()
        ) and self.observation_basis not in {
            ObservationBasis.FULL_CLIP_VIDEO,
            ObservationBasis.EVENT_PLUS_CONTEXT_VIDEO,
        }:
            raise ValueError(
                "assessed_absent requires full-clip or event-plus-context video"
            )
        if self.capabilities.audio_role in {
            AssessmentStatus.ASSESSED_PRESENT,
            AssessmentStatus.ASSESSED_ABSENT,
        } and not self.audio_included:
            raise ValueError("audio assessment requires audio in the observation input")
        return self


class SupplementRequestBinding(FrozenStrictModel):
    """Immutable paid-request lineage for one supplement generation contract."""

    contract_version: Literal["clip-observation-request-binding-v1"] = (
        SUPPLEMENT_REQUEST_BINDING_VERSION
    )
    model_id: str = Field(min_length=1)
    system_instruction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_resolution: Literal["low", "medium", "high"]
    thinking_level: Literal["minimal", "low", "medium", "high"]
    max_output_tokens: int = Field(gt=0)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_binding_hash(self) -> "SupplementRequestBinding":
        expected = supplement_request_binding_sha256(
            model_id=self.model_id,
            system_instruction_sha256=self.system_instruction_sha256,
            prompt_sha256=self.prompt_sha256,
            response_schema_sha256=self.response_schema_sha256,
            media_resolution=self.media_resolution,
            thinking_level=self.thinking_level,
            max_output_tokens=self.max_output_tokens,
        )
        if self.binding_sha256 != expected:
            raise ValueError("supplement request binding hash is invalid")
        return self


class ClipObservationSupplement(FrozenStrictModel):
    contract_version: Literal[
        "clip-observation-supplement-v1",
        "clip-observation-supplement-v2",
        "clip-observation-supplement-v3",
    ] = "clip-observation-supplement-v2"
    supplement_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    supersedes: list[str] = Field(default_factory=list)
    source_asset_id: str
    proxy_asset_id: str
    base_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supplement_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_observations: list[EventObservationSupplement] = Field(min_length=1)
    model_provenance: ModelProvenance
    request_binding: SupplementRequestBinding | None = None

    @model_validator(mode="after")
    def validate_event_ids(self) -> "ClipObservationSupplement":
        event_ids = [item.event_id for item in self.event_observations]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("supplement event IDs must be unique")
        if self.supplement_id in self.supersedes:
            raise ValueError("a supplement cannot supersede itself")
        if len(self.supersedes) != len(set(self.supersedes)):
            raise ValueError("supersedes IDs must be unique")
        if self.contract_version == "clip-observation-supplement-v3":
            if self.request_binding is None:
                raise ValueError("v3 supplement requires request_binding")
            if self.request_binding.model_id != self.model_provenance.model_id:
                raise ValueError(
                    "supplement request binding model differs from provenance"
                )
            if self.request_binding.prompt_sha256 != self.supplement_prompt_sha256:
                raise ValueError(
                    "supplement request binding prompt differs from supplement"
                )
            if (
                self.request_binding.response_schema_sha256
                != self.response_schema_sha256
            ):
                raise ValueError(
                    "supplement request binding schema differs from supplement"
                )
        return self


class SupplementNeed(FrozenStrictModel):
    event_id: str
    event_fingerprint: str
    required_capabilities: list[
        Literal[
            "action_structure",
            "evidence_roles",
            "observable_beats",
            "readability",
            "audio_role",
        ]
    ]
    reason_codes: list[str]


class ClaimCapabilityDecision(FrozenStrictModel):
    claim: EditingClaim
    decision: ClaimDecision
    required_capabilities: list[str]
    missing_capabilities: list[str] = Field(default_factory=list)
    unavailable_capabilities: list[str] = Field(default_factory=list)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def supplement_request_binding_sha256(
    *,
    model_id: str,
    system_instruction_sha256: str,
    prompt_sha256: str,
    response_schema_sha256: str,
    media_resolution: Literal["low", "medium", "high"],
    thinking_level: Literal["minimal", "low", "medium", "high"],
    max_output_tokens: int,
) -> str:
    return _canonical_sha256(
        {
            "contract_version": SUPPLEMENT_REQUEST_BINDING_VERSION,
            "model_id": model_id,
            "system_instruction_sha256": system_instruction_sha256,
            "prompt_sha256": prompt_sha256,
            "response_schema_sha256": response_schema_sha256,
            "media_resolution": media_resolution,
            "thinking_level": thinking_level,
            "max_output_tokens": max_output_tokens,
        }
    )


def build_supplement_request_binding(
    *,
    model_id: str,
    system_instruction_sha256: str,
    prompt_sha256: str,
    response_schema_sha256: str,
    media_resolution: Literal["low", "medium", "high"],
    thinking_level: Literal["minimal", "low", "medium", "high"],
    max_output_tokens: int,
) -> SupplementRequestBinding:
    values = {
        "model_id": model_id,
        "system_instruction_sha256": system_instruction_sha256,
        "prompt_sha256": prompt_sha256,
        "response_schema_sha256": response_schema_sha256,
        "media_resolution": media_resolution,
        "thinking_level": thinking_level,
        "max_output_tokens": max_output_tokens,
    }
    return SupplementRequestBinding(
        **values,
        binding_sha256=supplement_request_binding_sha256(**values),
    )


def clip_card_sha256(card: FullClipCard) -> str:
    return _canonical_sha256(card.model_dump(mode="json"))


def event_fingerprint(event: FullClipEvent) -> str:
    return _canonical_sha256(
        {
            "event_id": event.event_id,
            "start_mmss": event.start_mmss,
            "end_mmss": event.end_mmss,
            "label": event.label,
            "observable_evidence": event.observable_evidence,
            "evidence_provenance": event.evidence_provenance,
            "evidence_origin": (
                event.evidence_origin.model_dump(mode="json")
                if event.evidence_origin
                else None
            ),
            "entity_ids": event.entity_ids,
        }
    )


def _legacy_provenance_from_relation(
    relation: EvidenceRelation,
) -> FeatureEvidenceProvenance:
    """Best-effort compatibility mirror for readers that still require v1."""

    return {
        "direct_source_event": "unknown",
        "mediated_depiction": "prerecorded_screen_playback",
        "graphic_or_text_claim": "promotional_graphic",
        "context_only": "context_only",
        "unknown": "unknown",
    }[relation]


def _migrated_evidence_origin(
    provenance: FeatureEvidenceProvenance,
) -> EvidenceOriginObservation | None:
    if provenance == "unknown":
        return None
    return EvidenceOriginObservation(
        relation=evidence_relation_from_legacy(provenance),
        observable_reason=(
            "Migrated from the legacy evidence_provenance classification "
            f"{provenance!r}."
        ),
    )


def legacy_event_observation(event: FullClipEvent) -> EventObservationSupplement:
    """Project useful legacy attention evidence without trusting camera advice."""

    beats = [
        ObservableBeat(
            beat_id=phase.phase_id,
            kind="interaction" if phase.relation_mode == "joint_relation" else "state_change",
            entity_ids=phase.anchor_entity_ids,
            relation_mode=(
                "simultaneous_required"
                if phase.relation_mode == "joint_relation"
                else "sequentially_reconstructable"
            ),
            observable_predicate=phase.observable_predicate,
            transition_condition=phase.transition_condition,
        )
        for phase in event.portrait_attention_sequence
    ]
    evidence_origin = event.evidence_origin or _migrated_evidence_origin(
        event.evidence_provenance
    )
    return EventObservationSupplement(
        event_id=event.event_id,
        event_fingerprint=event_fingerprint(event),
        observation_basis=(
            ObservationBasis.FULL_CLIP_VIDEO
            if beats or evidence_origin is not None
            else None
        ),
        capabilities=EventCapabilityManifest(
            evidence_origin=(
                AssessmentStatus.ASSESSED_PRESENT
                if evidence_origin is not None
                else AssessmentStatus.NOT_ASSESSED
            ),
            observable_beats=(
                AssessmentStatus.ASSESSED_PRESENT
                if beats
                else AssessmentStatus.NOT_ASSESSED
            )
        ),
        evidence_provenance=event.evidence_provenance,
        evidence_origin=evidence_origin,
        observable_beats=beats,
    )


def validate_supplement(
    card: FullClipCard,
    supplement: ClipObservationSupplement,
    *,
    expected_base_card_sha256: str | None = None,
    expected_request_binding: SupplementRequestBinding | None = None,
    require_current_lineage: bool = False,
) -> None:
    expected_hash = expected_base_card_sha256 or clip_card_sha256(card)
    if supplement.source_asset_id != card.source_asset_id:
        raise ValueError("supplement source asset differs from base Clip Card")
    if supplement.proxy_asset_id != card.proxy_asset_id:
        raise ValueError("supplement proxy asset differs from base Clip Card")
    if supplement.base_card_sha256 != expected_hash:
        raise ValueError("supplement is stale for the current base Clip Card")
    if require_current_lineage and supplement.request_binding is None:
        raise ValueError(
            "autonomous supplement requires current request lineage; "
            "legacy v1/v2 artifacts are review-only"
        )
    if expected_request_binding is not None:
        if supplement.request_binding is None:
            raise ValueError(
                "supplement lacks request lineage required by the current operation"
            )
        if (
            supplement.request_binding.binding_sha256
            != expected_request_binding.binding_sha256
        ):
            raise ValueError(
                "supplement is stale for the current model, prompt, schema, "
                "system instruction, media resolution, or generation limits"
            )
    events = {event.event_id: event for event in card.events}
    known_entities = {entity.entity_id for entity in card.entities}
    for observation in supplement.event_observations:
        event = events.get(observation.event_id)
        if event is None:
            raise ValueError(f"supplement references unknown event {observation.event_id}")
        if observation.event_fingerprint != event_fingerprint(event):
            raise ValueError(f"supplement event {observation.event_id} is stale")
        references = observation.evidence_roles.referenced_entity_ids() | {
            entity_id
            for beat in observation.observable_beats
            for entity_id in beat.entity_ids
        } | {item.entity_id for item in observation.readability}
        unknown = references - known_entities
        if unknown:
            raise ValueError(
                f"supplement event {observation.event_id} references unknown entities: "
                f"{sorted(unknown)}"
            )


def effective_event_observations(
    card: FullClipCard,
    supplements: Iterable[ClipObservationSupplement] = (),
) -> dict[str, EventObservationSupplement]:
    """Merge by capability; conflicting assessed payloads fail closed."""

    supplement_list = list(supplements)
    supplement_by_id: dict[str, ClipObservationSupplement] = {}
    for supplement in supplement_list:
        if supplement.supplement_id in supplement_by_id:
            raise ValueError(f"duplicate supplement ID {supplement.supplement_id}")
        supplement_by_id[supplement.supplement_id] = supplement
        validate_supplement(card, supplement)
    for supplement in supplement_list:
        unknown = set(supplement.supersedes) - set(supplement_by_id)
        if unknown:
            raise ValueError(
                f"supplement {supplement.supplement_id} supersedes unknown IDs: "
                f"{sorted(unknown)}"
            )
    superseded = {
        item_id
        for supplement in supplement_list
        for item_id in supplement.supersedes
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(supplement_id: str) -> None:
        if supplement_id in visiting:
            raise ValueError("supplement supersedes graph contains a cycle")
        if supplement_id in visited:
            return
        visiting.add(supplement_id)
        for target in supplement_by_id[supplement_id].supersedes:
            visit(target)
        visiting.remove(supplement_id)
        visited.add(supplement_id)

    for supplement_id in supplement_by_id:
        visit(supplement_id)
    active = [
        supplement
        for supplement in supplement_list
        if supplement.supplement_id not in superseded
    ]
    active.sort(key=lambda item: item.supplement_id)

    merged = {event.event_id: legacy_event_observation(event) for event in card.events}
    explicit_payloads: dict[tuple[str, str], str] = {}
    for supplement in active:
        for incoming in supplement.event_observations:
            current = merged[incoming.event_id]
            values = current.model_dump(mode="json")
            capabilities = current.capabilities.model_dump(mode="json")
            for capability in capabilities:
                incoming_status = getattr(incoming.capabilities, capability)
                incoming_origin = None
                if capability == "evidence_origin":
                    incoming_origin = incoming.evidence_origin
                    if (
                        incoming_status == AssessmentStatus.NOT_ASSESSED
                        and incoming_origin is None
                    ):
                        incoming_origin = _migrated_evidence_origin(
                            incoming.evidence_provenance
                        )
                        if incoming_origin is not None:
                            incoming_status = AssessmentStatus.ASSESSED_PRESENT
                if incoming_status == AssessmentStatus.NOT_ASSESSED:
                    continue
                payload = (
                    incoming_origin.model_dump(mode="json")
                    if incoming_origin is not None
                    else _capability_payload(incoming, capability)
                )
                payload_sha = _canonical_sha256(
                    {
                        "status": incoming_status,
                        "payload": payload,
                    }
                )
                key = (incoming.event_id, capability)
                prior_sha = explicit_payloads.get(key)
                if prior_sha is not None and prior_sha != payload_sha:
                    raise ValueError(
                        f"conflicting {capability} assessments for {incoming.event_id}"
                    )
                if prior_sha == payload_sha:
                    continue
                explicit_payloads[key] = payload_sha
                capabilities[capability] = incoming_status
                if capability == "evidence_origin":
                    values["evidence_origin"] = payload
                    if incoming.evidence_provenance != "unknown":
                        values["evidence_provenance"] = incoming.evidence_provenance
                    elif incoming_origin is not None:
                        values["evidence_provenance"] = (
                            _legacy_provenance_from_relation(
                                incoming_origin.relation
                            )
                        )
                    else:
                        values["evidence_provenance"] = "unknown"
                elif capability == "action_structure":
                    values["source_action_completeness"] = (
                        incoming.source_action_completeness
                    )
                    values["clean_entry"] = incoming.clean_entry
                    values["clean_exit"] = incoming.clean_exit
                elif capability == "evidence_roles":
                    values["evidence_roles"] = incoming.evidence_roles.model_dump(
                        mode="json"
                    )
                elif capability == "observable_beats":
                    values["observable_beats"] = [
                        item.model_dump(mode="json")
                        for item in incoming.observable_beats
                    ]
                elif capability == "readability":
                    values["readability"] = [
                        item.model_dump(mode="json") for item in incoming.readability
                    ]
                elif capability == "audio_role":
                    values["audio_role"] = (
                        incoming.audio_role.model_dump(mode="json")
                        if incoming.audio_role
                        else None
                    )
            if incoming.observation_basis is not None:
                values["observation_basis"] = incoming.observation_basis
            values["audio_included"] = (
                current.audio_included or incoming.audio_included
            )
            values["capabilities"] = capabilities
            values["uncertainties"] = sorted(
                set(current.uncertainties + incoming.uncertainties)
            )
            merged[incoming.event_id] = EventObservationSupplement.model_validate(
                values
            )
    return merged


def _capability_payload(
    observation: EventObservationSupplement,
    capability: str,
) -> object:
    if capability == "action_structure":
        return {
            "source_action_completeness": observation.source_action_completeness,
            "clean_entry": observation.clean_entry,
            "clean_exit": observation.clean_exit,
        }
    value = getattr(observation, capability)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in value
        ]
    return value


def effective_event_observation_sha256(
    observation: EventObservationSupplement,
) -> str:
    """Content hash for one fully merged observation."""

    return _canonical_sha256(observation.model_dump(mode="json"))


def effective_event_observations_sha256(
    card: FullClipCard,
    supplements: Iterable[ClipObservationSupplement] = (),
) -> str:
    """Order-stable content hash for every effective event observation."""

    observations = effective_event_observations(card, supplements)
    return _canonical_sha256(
        {
            event_id: observation.model_dump(mode="json")
            for event_id, observation in sorted(observations.items())
        }
    )


_CLAIM_CAPABILITIES: dict[EditingClaim, tuple[str, ...]] = {
    EditingClaim.TOPICAL_RELEVANCE: (),
    EditingClaim.COMPLETE_TRIM: ("action_structure",),
    EditingClaim.SEQUENTIAL_VIRTUAL_CAMERA: (
        "evidence_roles",
        "observable_beats",
    ),
    EditingClaim.SIMULTANEOUS_RELATION: ("evidence_roles",),
    EditingClaim.READABLE_CONTENT: ("readability",),
    EditingClaim.SOURCE_AUDIO: ("audio_role",),
}


def assess_editing_claim(
    observation: EventObservationSupplement,
    claim: EditingClaim,
) -> ClaimCapabilityDecision:
    required = list(_CLAIM_CAPABILITIES[claim])
    missing: list[str] = []
    unavailable: list[str] = []
    for capability in required:
        status = getattr(observation.capabilities, capability)
        if status == AssessmentStatus.NOT_ASSESSED:
            missing.append(capability)
        elif status in {
            AssessmentStatus.ASSESSED_ABSENT,
            AssessmentStatus.NOT_APPLICABLE,
        }:
            unavailable.append(capability)
    if unavailable:
        decision = ClaimDecision.ABSTAIN
    elif missing:
        decision = ClaimDecision.REQUEST_SUPPLEMENT
    else:
        decision = ClaimDecision.READY
    return ClaimCapabilityDecision(
        claim=claim,
        decision=decision,
        required_capabilities=required,
        missing_capabilities=missing,
        unavailable_capabilities=unavailable,
    )


def plan_supplement_needs(
    card: FullClipCard,
    *,
    frontier_event_ids: set[str] | None = None,
    requested_claims: dict[str, set[EditingClaim]] | None = None,
) -> list[SupplementNeed]:
    """Plan bounded observation only for hard risks or shortlisted edit claims."""

    readable_kinds = {
        EntityKind.DOCUMENT,
        EntityKind.TEXT_REGION,
        EntityKind.UI_ELEMENT,
        EntityKind.PHONE_SCREEN,
        EntityKind.SCREEN,
    }
    entities = {entity.entity_id: entity for entity in card.entities}
    needs: list[SupplementNeed] = []
    requested_claims = requested_claims or {}
    for event in card.events:
        existing = legacy_event_observation(event).capabilities
        required: list[str] = []
        reasons: list[str] = []
        in_frontier = (
            frontier_event_ids is not None and event.event_id in frontier_event_ids
        )
        if event.action_completeness != "complete" or event.dense_refinement != "not_needed":
            required.append("action_structure")
            reasons.append("action_or_boundary_uncertain")
        # Multiple entities alone are common and do not justify a paid supplement.
        # Multiple primaries or multiple independently-groundable targets are a
        # stronger, domain-neutral signal that relation evidence may affect recall.
        if in_frontier and (
            len(event.primary_entity_ids) > 1 or len(event.grounding_targets) > 1
        ):
            required.append("evidence_roles")
            if existing.observable_beats == AssessmentStatus.NOT_ASSESSED:
                required.append("observable_beats")
            reasons.append("multiple_primary_or_groundable_entities")
        if event.dense_refinement != "not_needed":
            if (
                existing.observable_beats == AssessmentStatus.NOT_ASSESSED
                and "observable_beats" not in required
            ):
                required.append("observable_beats")
            reasons.append("dense_temporal_evidence")
        if in_frontier and any(
            entities[entity_id].kind in readable_kinds
            for entity_id in event.entity_ids
            if entity_id in entities
        ):
            required.append("readability")
            reasons.append("readable_or_stateful_region")
        for claim in sorted(
            requested_claims.get(event.event_id, set()),
            key=lambda item: item.value,
        ):
            for capability in _CLAIM_CAPABILITIES[claim]:
                if getattr(existing, capability) == AssessmentStatus.NOT_ASSESSED:
                    required.append(capability)
            reasons.append(f"editing_claim:{claim.value}")
        # Audio role is useful when a supplement is already warranted, but the
        # broad base-card modality is enough for retrieval and must not force
        # every clip with sound into another paid pass.
        if required and event.evidence_modalities in {
            EvidenceModality.AUDIO,
            EvidenceModality.VISUAL_AND_AUDIO,
        }:
            required.append("audio_role")
            reasons.append("audio_contributes_evidence")
        if required:
            needs.append(
                SupplementNeed(
                    event_id=event.event_id,
                    event_fingerprint=event_fingerprint(event),
                    required_capabilities=list(dict.fromkeys(required)),
                    reason_codes=list(dict.fromkeys(reasons)),
                )
            )
    return needs
