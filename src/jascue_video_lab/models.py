from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
MmSs = Annotated[str, Field(pattern=r"^\d{2,}:[0-5]\d$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _mmss_to_ms(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    return (minutes * 60 + seconds) * 1000


def _local_ms_from_pts(pts: int, start_pts: int, time_base: "Rational") -> int:
    return round(
        Fraction(
            (pts - start_pts) * time_base.numerator * 1000,
            time_base.denominator,
        )
    )


def _half_open_ms_matches_pts(
    actual_ms: int,
    expected_rounded_ms: int,
    end_ms: int | None,
) -> bool:
    return actual_ms == expected_rounded_ms or (
        end_ms is not None
        and expected_rounded_ms == end_ms
        and actual_ms == end_ms - 1
    )


def _proper_segments_intersect(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    def orientation(
        first: tuple[int, int], second: tuple[int, int], third: tuple[int, int]
    ) -> int:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    return (
        orientation(a, b, c) * orientation(a, b, d) < 0
        and orientation(c, d, a) * orientation(c, d, b) < 0
    )


def _polygon_has_proper_self_intersection(points: list[tuple[int, int]]) -> bool:
    compact = [
        point
        for index, point in enumerate(points)
        if index == 0 or point != points[index - 1]
    ]
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    count = len(compact)
    for left_index in range(count):
        left_start = compact[left_index]
        left_end = compact[(left_index + 1) % count]
        for right_index in range(left_index + 1, count):
            if right_index in {left_index, (left_index + 1) % count}:
                continue
            if left_index == (right_index + 1) % count:
                continue
            right_start = compact[right_index]
            right_end = compact[(right_index + 1) % count]
            if _proper_segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return False


class BoundaryPrecision(StrEnum):
    COARSE = "coarse"
    SECOND_LEVEL = "second_level"
    UNCERTAIN = "uncertain"


class EvidenceModality(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"
    VISUAL_AND_AUDIO = "visual_and_audio"


class EntityKind(StrEnum):
    PERSON = "person"
    FACE = "face"
    HAND = "hand"
    ANIMAL = "animal"
    OBJECT = "object"
    PRODUCT = "product"
    DEVICE = "device"
    PHONE = "phone"
    PHONE_SCREEN = "phone_screen"
    SCREEN = "screen"
    DOCUMENT = "document"
    LOGO = "logo"
    TEXT_REGION = "text_region"
    UI_ELEMENT = "ui_element"
    VEHICLE = "vehicle"
    OTHER = "other"


class Occlusion(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class MatchStatus(StrEnum):
    """Semantic target match result; intentionally separate from visibility."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NOT_VISIBLE = "not_visible"
    TARGET_MISMATCH = "target_mismatch"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PredicateStatus(StrEnum):
    """Whether an optional observable event predicate is supported by evidence."""

    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class EvidenceClaimSource(StrEnum):
    USER_BRIEF = "user_brief"
    HUMAN_REVIEW = "human_review"
    IMPORTED_METADATA = "imported_metadata"
    MODEL_PROPOSAL = "model_proposal"


class EvidenceQueryTargetRef(StrictModel):
    """Stable, domain-neutral reference to a selected target instance."""

    target_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    target_description: str = Field(min_length=1)
    positive_attributes: list[str] = Field(default_factory=list)
    negative_attributes: list[str] = Field(default_factory=list)
    reference_frame_ids: list[str] = Field(default_factory=list)
    reference_crop_hashes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target_reference(self) -> "EvidenceQueryTargetRef":
        for field_name in (
            "positive_attributes",
            "negative_attributes",
            "reference_frame_ids",
            "reference_crop_hashes",
        ):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        for digest in self.reference_crop_hashes:
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("reference_crop_hashes must be lowercase SHA-256 digests")
        positive = {value.casefold() for value in self.positive_attributes}
        negative = {value.casefold() for value in self.negative_attributes}
        if positive & negative:
            raise ValueError("positive and negative target attributes must not overlap")
        return self


AspectRatio = Annotated[str, Field(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]


class AspectConstraint(StrictModel):
    aspect_ratio: AspectRatio
    required_target_ids: list[str] = Field(default_factory=list)
    constraint: str = Field(min_length=1)


class EvidenceQueryProvenance(StrictModel):
    created_at: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    source_reference: str | None = None
    parent_query_id: str | None = None


class PredicatePhaseConditions(StrictModel):
    """Observable before/apex/after evidence for a locked temporal predicate."""

    precondition: str = Field(min_length=1)
    apex: str = Field(min_length=1)
    postcondition: str = Field(min_length=1)


class EvidenceQueryLock(StrictModel):
    """Immutable-by-convention editorial/evidence contract for downstream stages.

    It intentionally describes neither a media domain nor a tracker. Consumers may
    persist ``definition_sha256()`` with derived artifacts to prove which revision
    governed a result.
    """

    query_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    targets: list[EvidenceQueryTargetRef] = Field(default_factory=list)
    observable_predicate: str | None = None
    predicate_phases: PredicatePhaseConditions | None = None
    required_evidence: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    editing_uses: list[str] = Field(default_factory=list)
    aspect_constraints: list[AspectConstraint] = Field(default_factory=list)
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenance

    @model_validator(mode="after")
    def validate_query_lock(self) -> "EvidenceQueryLock":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("query lock target_id values must be unique")
        known_targets = set(target_ids)
        for aspect in self.aspect_constraints:
            if len(aspect.required_target_ids) != len(set(aspect.required_target_ids)):
                raise ValueError("aspect required_target_ids must be unique")
            unknown = set(aspect.required_target_ids) - known_targets
            if unknown:
                raise ValueError(
                    f"aspect constraint references unknown targets: {sorted(unknown)}"
                )
        for field_name in (
            "required_evidence",
            "negative_constraints",
            "editing_uses",
        ):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        if self.observable_predicate is not None and not self.observable_predicate.strip():
            raise ValueError("observable_predicate must be non-empty when supplied")
        if self.predicate_phases is not None and self.observable_predicate is None:
            raise ValueError("predicate_phases require observable_predicate")
        return self

    def canonical_definition_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def definition_sha256(self) -> str:
        return hashlib.sha256(self.canonical_definition_json().encode("utf-8")).hexdigest()


class TargetIdentityScope(StrEnum):
    """The geometric level at which a persistent target is identified."""

    WHOLE_INSTANCE = "whole_instance"
    SUBPART = "subpart"
    VISIBLE_REGION = "visible_region"


class PredicateRequiredAt(StrEnum):
    """The stage or interval at which an observable predicate must hold."""

    CANDIDATE = "candidate"
    SEED = "seed"
    TRANSITION = "transition"
    INTERVAL = "interval"


class EvidenceApprovalSource(StrEnum):
    """An approval authority. Models are deliberately not an authority here."""

    USER_BRIEF = "user_brief"
    HUMAN_REVIEW = "human_review"
    AUTO_POLICY = "auto_policy"


def _canonical_contract_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _contract_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_contract_json(value).encode("utf-8")).hexdigest()


def _validate_unique_non_empty_strings(
    values: tuple[str, ...], field_name: str
) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} values must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


class EvidenceAnchor(FrozenStrictModel):
    """A content-addressed crop from one immutable evidence frame."""

    frame_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    crop_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceTargetIdentityV2(FrozenStrictModel):
    """Persistent instance identity, separated from temporary event state."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    target_description: str = Field(min_length=1)
    scope: TargetIdentityScope = TargetIdentityScope.WHOLE_INSTANCE
    parent_target_id: str | None = Field(
        default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    identity_cues: tuple[str, ...] = ()
    context_cues: tuple[str, ...] = ()
    positive_anchors: tuple[EvidenceAnchor, ...] = ()
    stable_exclusions: tuple[str, ...] = ()
    negative_anchors: tuple[EvidenceAnchor, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self) -> "EvidenceTargetIdentityV2":
        for field_name in (
            "identity_cues",
            "context_cues",
            "stable_exclusions",
        ):
            _validate_unique_non_empty_strings(getattr(self, field_name), field_name)
        if self.scope == TargetIdentityScope.WHOLE_INSTANCE:
            if self.parent_target_id is not None:
                raise ValueError("whole_instance targets cannot have parent_target_id")
        elif self.parent_target_id is None:
            raise ValueError("subpart and visible_region targets require parent_target_id")
        if self.parent_target_id == self.target_id:
            raise ValueError("target cannot be its own parent")
        if not self.identity_cues and not self.positive_anchors:
            raise ValueError("identity requires identity_cues or positive_anchors")
        identity = {value.casefold() for value in self.identity_cues}
        exclusions = {value.casefold() for value in self.stable_exclusions}
        if identity & exclusions:
            raise ValueError("identity cues and stable exclusions must not overlap")
        positive = {
            (anchor.frame_id, anchor.crop_sha256) for anchor in self.positive_anchors
        }
        negative = {
            (anchor.frame_id, anchor.crop_sha256) for anchor in self.negative_anchors
        }
        if len(positive) != len(self.positive_anchors):
            raise ValueError("positive_anchors must be unique")
        if len(negative) != len(self.negative_anchors):
            raise ValueError("negative_anchors must be unique")
        if positive & negative:
            raise ValueError("positive and negative anchors must not overlap")
        positive_crop_hashes = {
            anchor.crop_sha256 for anchor in self.positive_anchors
        }
        negative_crop_hashes = {
            anchor.crop_sha256 for anchor in self.negative_anchors
        }
        if positive_crop_hashes & negative_crop_hashes:
            raise ValueError(
                "the same crop bytes cannot be both a positive and negative anchor"
            )
        return self


class EvidenceIdentityContractV2(FrozenStrictModel):
    targets: tuple[EvidenceTargetIdentityV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_graph(self) -> "EvidenceIdentityContractV2":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("identity target_id values must be unique")
        known = set(target_ids)
        parents = {
            target.target_id: target.parent_target_id
            for target in self.targets
            if target.parent_target_id is not None
        }
        unknown = set(parents.values()) - known
        if unknown:
            raise ValueError(f"identity parents reference unknown targets: {sorted(unknown)}")
        for target_id in target_ids:
            visited: set[str] = set()
            cursor: str | None = target_id
            while cursor is not None:
                if cursor in visited:
                    raise ValueError("identity parent links must not contain cycles")
                visited.add(cursor)
                cursor = parents.get(cursor)
        return self

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)

    def target(self, target_id: str) -> EvidenceTargetIdentityV2:
        try:
            return next(target for target in self.targets if target.target_id == target_id)
        except StopIteration as error:
            raise ValueError(f"unknown identity target: {target_id}") from error

    def ancestors(self, target_id: str) -> tuple[EvidenceTargetIdentityV2, ...]:
        """Return nearest-to-farthest parent identities for subpart disambiguation."""

        ancestors: list[EvidenceTargetIdentityV2] = []
        cursor = self.target(target_id)
        while cursor.parent_target_id is not None:
            cursor = self.target(cursor.parent_target_id)
            ancestors.append(cursor)
        return tuple(ancestors)


class EvidencePredicatePhasesV2(FrozenStrictModel):
    precondition: str = Field(min_length=1)
    apex: str = Field(min_length=1)
    postcondition: str = Field(min_length=1)


class EvidencePredicateContractV2(FrozenStrictModel):
    """A media-observable eligibility condition, not a persistent identity cue."""

    predicate_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    statement: str = Field(min_length=1)
    participant_target_ids: tuple[str, ...] = Field(min_length=1)
    required_at: PredicateRequiredAt
    phases: EvidencePredicatePhasesV2 | None = None
    required_evidence: tuple[str, ...] = ()
    disqualifying_conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_predicate(self) -> "EvidencePredicateContractV2":
        for field_name in (
            "participant_target_ids",
            "required_evidence",
            "disqualifying_conditions",
        ):
            _validate_unique_non_empty_strings(getattr(self, field_name), field_name)
        if self.required_at == PredicateRequiredAt.TRANSITION and self.phases is None:
            raise ValueError("transition predicates require pre/apex/post phases")
        return self

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


class EvidenceTargetVisibilityConstraintV2(FrozenStrictModel):
    """A domain-neutral visibility floor for one target in one layout."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    minimum_visible_fraction: float = Field(gt=0.0, le=1.0)
    atomic: bool = False


class EvidenceAspectConstraintV2(FrozenStrictModel):
    aspect_ratio: AspectRatio
    required_target_ids: tuple[str, ...] = ()
    constraint: str = Field(min_length=1)
    target_visibility_constraints: tuple[
        EvidenceTargetVisibilityConstraintV2, ...
    ] = ()
    required_target_clipping_policy: Literal[
        "forbid", "allow_controlled"
    ] = "forbid"

    @model_validator(mode="after")
    def validate_aspect(self) -> "EvidenceAspectConstraintV2":
        _validate_unique_non_empty_strings(
            self.required_target_ids, "required_target_ids"
        )
        visibility_ids = [
            item.target_id for item in self.target_visibility_constraints
        ]
        if len(visibility_ids) != len(set(visibility_ids)):
            raise ValueError("target visibility constraints must be unique")
        if any(
            item.atomic and item.minimum_visible_fraction != 1.0
            for item in self.target_visibility_constraints
        ):
            raise ValueError("atomic target visibility must be 1.0")
        return self


class EvidenceFramingObligationsV2(FrozenStrictModel):
    """Semantic framing priorities; contains no generated crop coordinates."""

    required_target_ids: tuple[str, ...] = ()
    preferred_target_ids: tuple[str, ...] = ()
    sacrificable_target_ids: tuple[str, ...] = ()
    overlay_keepout_target_ids: tuple[str, ...] = ()
    framing_intent: str = Field(min_length=1)
    editing_uses: tuple[str, ...] = ()
    aspect_constraints: tuple[EvidenceAspectConstraintV2, ...] = ()

    @model_validator(mode="after")
    def validate_obligations(self) -> "EvidenceFramingObligationsV2":
        for field_name in (
            "required_target_ids",
            "preferred_target_ids",
            "sacrificable_target_ids",
            "overlay_keepout_target_ids",
            "editing_uses",
        ):
            _validate_unique_non_empty_strings(getattr(self, field_name), field_name)
        required = set(self.required_target_ids)
        preferred = set(self.preferred_target_ids)
        sacrificable = set(self.sacrificable_target_ids)
        if required & preferred or required & sacrificable or preferred & sacrificable:
            raise ValueError(
                "required, preferred, and sacrificable target roles must be disjoint"
            )
        return self

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


class EvidenceQueryApprovalProvenance(FrozenStrictModel):
    approved_at: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approval_source: EvidenceApprovalSource
    source_reference: str | None = None
    policy_reference: str | None = None

    @model_validator(mode="after")
    def validate_approval(self) -> "EvidenceQueryApprovalProvenance":
        if self.approval_source == EvidenceApprovalSource.AUTO_POLICY:
            if self.policy_reference is None or not self.policy_reference.strip():
                raise ValueError("auto_policy approval requires policy_reference")
        elif self.policy_reference is not None:
            raise ValueError("policy_reference is only valid for auto_policy approval")
        return self


class EvidenceQueryProvenanceV2(FrozenStrictModel):
    created_at: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    source_reference: str | None = None
    parent_query_id: str | None = None


def _validate_query_v2_cross_references(
    *,
    identity: EvidenceIdentityContractV2,
    predicate: EvidencePredicateContractV2 | None,
    framing: EvidenceFramingObligationsV2,
) -> None:
    known_targets = {target.target_id for target in identity.targets}
    if predicate is not None:
        unknown = set(predicate.participant_target_ids) - known_targets
        if unknown:
            raise ValueError(
                f"predicate references unknown participant targets: {sorted(unknown)}"
            )
    for field_name in (
        "required_target_ids",
        "preferred_target_ids",
        "sacrificable_target_ids",
        "overlay_keepout_target_ids",
    ):
        unknown = set(getattr(framing, field_name)) - known_targets
        if unknown:
            raise ValueError(
                f"framing {field_name} references unknown targets: {sorted(unknown)}"
            )
    for aspect in framing.aspect_constraints:
        if len(aspect.required_target_ids) != len(set(aspect.required_target_ids)):
            raise ValueError("aspect required_target_ids must be unique")
        unknown = set(aspect.required_target_ids) - known_targets
        if unknown:
            raise ValueError(
                f"aspect constraint references unknown targets: {sorted(unknown)}"
            )
        visibility_unknown = {
            item.target_id for item in aspect.target_visibility_constraints
        } - known_targets
        if visibility_unknown:
            raise ValueError(
                "aspect visibility constraints reference unknown targets: "
                f"{sorted(visibility_unknown)}"
            )


def _query_v2_component_hashes(
    *,
    identity: EvidenceIdentityContractV2,
    predicate: EvidencePredicateContractV2 | None,
    framing: EvidenceFramingObligationsV2,
) -> dict[str, str]:
    return {
        "identity_sha256": identity.definition_sha256(),
        "predicate_sha256": _contract_sha256(predicate),
        "framing_sha256": framing.definition_sha256(),
    }


def _query_v2_composite_sha256(
    *,
    editorial_goal: str,
    identity: EvidenceIdentityContractV2,
    predicate: EvidencePredicateContractV2 | None,
    framing: EvidenceFramingObligationsV2,
) -> str:
    return _contract_sha256(
        {
            "contract_version": "evidence-query-v2",
            "editorial_goal": editorial_goal,
            **_query_v2_component_hashes(
                identity=identity,
                predicate=predicate,
                framing=framing,
            ),
        }
    )


class EvidenceQueryProposalV2(StrictModel):
    """Unapproved three-layer query definition suitable for review."""

    contract_version: Literal["evidence-query-proposal-v2"] = (
        "evidence-query-proposal-v2"
    )
    proposal_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    identity: EvidenceIdentityContractV2
    predicate: EvidencePredicateContractV2 | None = None
    framing: EvidenceFramingObligationsV2
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenanceV2

    @model_validator(mode="after")
    def validate_proposal(self) -> "EvidenceQueryProposalV2":
        _validate_query_v2_cross_references(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )
        return self

    def component_hashes(self) -> dict[str, str]:
        return _query_v2_component_hashes(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def composite_sha256(self) -> str:
        return _query_v2_composite_sha256(
            editorial_goal=self.editorial_goal,
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )


class EvidenceQueryLockV2(StrictModel):
    """Approved, frozen query definition with separate claim and approval origins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["evidence-query-lock-v2"] = "evidence-query-lock-v2"
    query_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    identity: EvidenceIdentityContractV2
    predicate: EvidencePredicateContractV2 | None = None
    framing: EvidenceFramingObligationsV2
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenanceV2
    approval: EvidenceQueryApprovalProvenance

    @model_validator(mode="after")
    def validate_lock(self) -> "EvidenceQueryLockV2":
        _validate_query_v2_cross_references(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )
        return self

    def component_hashes(self) -> dict[str, str]:
        return _query_v2_component_hashes(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def composite_sha256(self) -> str:
        return _query_v2_composite_sha256(
            editorial_goal=self.editorial_goal,
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


def approve_evidence_query_proposal_v2(
    proposal: EvidenceQueryProposalV2,
    *,
    query_id: str,
    approval: EvidenceQueryApprovalProvenance,
) -> EvidenceQueryLockV2:
    """Create a new immutable lock without mutating or relabeling the proposal."""

    return EvidenceQueryLockV2(
        query_id=query_id,
        revision=proposal.revision,
        editorial_goal=proposal.editorial_goal,
        identity=proposal.identity,
        predicate=proposal.predicate,
        framing=proposal.framing,
        claim_source=proposal.claim_source,
        provenance=proposal.provenance,
        approval=approval,
    )


def migrate_evidence_query_lock_v1_to_proposal_v2(
    lock: EvidenceQueryLock,
) -> EvidenceQueryProposalV2:
    """Losslessly move a v1 lock definition into the reviewable v2 layers."""

    targets: list[EvidenceTargetIdentityV2] = []
    for target in lock.targets:
        if len(target.reference_frame_ids) != len(target.reference_crop_hashes):
            raise ValueError(
                "v1 reference frame IDs and crop hashes must have equal lengths "
                "for lossless v2 migration"
            )
        targets.append(
            EvidenceTargetIdentityV2(
                target_id=target.target_id,
                target_description=target.target_description,
                scope=TargetIdentityScope.WHOLE_INSTANCE,
                identity_cues=(
                    tuple(target.positive_attributes)
                    or (target.target_description,)
                ),
                positive_anchors=tuple(
                    EvidenceAnchor(frame_id=frame_id, crop_sha256=crop_hash)
                    for frame_id, crop_hash in zip(
                        target.reference_frame_ids,
                        target.reference_crop_hashes,
                        strict=True,
                    )
                ),
                stable_exclusions=tuple(target.negative_attributes),
            )
        )
    if not targets:
        raise ValueError("v1 lock must contain a target for v2 identity migration")

    if (
        (lock.required_evidence or lock.negative_constraints)
        and not lock.observable_predicate
        and lock.predicate_phases is None
    ):
        raise ValueError(
            "v1 evidence constraints without an observable predicate cannot be "
            "losslessly migrated to QueryLock v2"
        )
    has_predicate_evidence = bool(
        lock.observable_predicate or lock.predicate_phases
    )
    predicate = (
        EvidencePredicateContractV2(
            predicate_id=f"{lock.query_id}:predicate",
            statement=(
                lock.observable_predicate
                or (
                    "Observable transition: "
                    f"{lock.predicate_phases.precondition}; "
                    f"{lock.predicate_phases.apex}; "
                    f"{lock.predicate_phases.postcondition}"
                )
            ),
            participant_target_ids=tuple(target.target_id for target in lock.targets),
            required_at=(
                PredicateRequiredAt.TRANSITION
                if lock.predicate_phases is not None
                else PredicateRequiredAt.SEED
            ),
            phases=(
                EvidencePredicatePhasesV2(
                    precondition=lock.predicate_phases.precondition,
                    apex=lock.predicate_phases.apex,
                    postcondition=lock.predicate_phases.postcondition,
                )
                if lock.predicate_phases is not None
                else None
            ),
            required_evidence=tuple(lock.required_evidence),
            disqualifying_conditions=tuple(lock.negative_constraints),
        )
        if has_predicate_evidence
        else None
    )
    required_targets = tuple(
        dict.fromkeys(
            target_id
            for aspect in lock.aspect_constraints
            for target_id in aspect.required_target_ids
        )
    )
    return EvidenceQueryProposalV2(
        proposal_id=f"{lock.query_id}:migrated-v2",
        revision=lock.revision,
        editorial_goal=lock.editorial_goal,
        identity=EvidenceIdentityContractV2(targets=tuple(targets)),
        predicate=predicate,
        framing=EvidenceFramingObligationsV2(
            required_target_ids=required_targets,
            preferred_target_ids=tuple(
                target.target_id
                for target in lock.targets
                if target.target_id not in required_targets
            ),
            framing_intent=(
                "; ".join(aspect.constraint for aspect in lock.aspect_constraints)
                or "Preserve the selected evidence targets for the intended edit."
            ),
            editing_uses=tuple(lock.editing_uses),
            aspect_constraints=tuple(
                EvidenceAspectConstraintV2(
                    aspect_ratio=aspect.aspect_ratio,
                    required_target_ids=tuple(aspect.required_target_ids),
                    constraint=aspect.constraint,
                )
                for aspect in lock.aspect_constraints
            ),
        ),
        claim_source=lock.claim_source,
        provenance=EvidenceQueryProvenanceV2(
            created_at=lock.provenance.created_at,
            created_by=lock.provenance.created_by,
            source_reference=lock.provenance.source_reference,
            parent_query_id=lock.provenance.parent_query_id,
        ),
    )


def migrate_evidence_query_lock_v1_to_v2(
    lock: EvidenceQueryLock,
    *,
    approval: EvidenceQueryApprovalProvenance,
) -> EvidenceQueryLockV2:
    """Migrate v1 through an explicit proposal and truthful approval record."""

    proposal = migrate_evidence_query_lock_v1_to_proposal_v2(lock)
    return approve_evidence_query_proposal_v2(
        proposal,
        query_id=lock.query_id,
        approval=approval,
    )


def _validated_grounding_match_status(
    *, visible: bool, candidate_count: int, match_status: MatchStatus | None
) -> MatchStatus:
    status = match_status
    if status is None:
        if not visible:
            status = MatchStatus.NOT_VISIBLE
        elif candidate_count == 1:
            status = MatchStatus.MATCHED
        else:
            status = MatchStatus.AMBIGUOUS
    if status == MatchStatus.MATCHED:
        if not visible or candidate_count != 1:
            raise ValueError(
                "match_status=matched requires visible=true and exactly one candidate"
            )
    elif status == MatchStatus.AMBIGUOUS:
        if not visible or candidate_count == 0:
            raise ValueError(
                f"match_status={status.value} requires visible=true and candidates"
            )
    elif visible or candidate_count:
        raise ValueError(
            f"match_status={status.value} requires visible=false and no candidates"
        )
    return status


class ModelProvenance(StrictModel):
    model_id: str
    api: Literal["gemini_interactions"]
    sdk: Literal["google-genai"]
    sdk_version: str
    interaction_id: str | None = None
    run_id: str
    generated_at: str


class CardOpportunity(StrictModel):
    kind: Literal["feature_card", "step_card", "object_callout"]
    rationale: str
    entity_ids: list[str] = Field(default_factory=list)


class Entity(StrictModel):
    entity_id: str = Field(min_length=1)
    kind: EntityKind
    label: str
    distinguishing_features: str
    evidence: str


class Event(StrictModel):
    event_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    label: str
    description: str
    evidence_modalities: EvidenceModality
    entity_ids: list[str]
    recommended_keyframe_ms: int | None = Field(default=None, ge=0)
    keyframe_reason: str
    confidence: Confidence
    boundary_precision: BoundaryPrecision
    primary_entity_ids: list[str]
    required_entity_ids: list[str]
    optional_entity_ids: list[str]
    avoid_overlay_entity_ids: list[str]
    framing_intent: str
    card_opportunities: list[CardOpportunity]

    @model_validator(mode="after")
    def validate_interval_and_keyframe(self) -> "Event":
        if self.end_ms <= self.start_ms:
            raise ValueError("event interval must be non-empty and half-open")
        if self.recommended_keyframe_ms is not None and not (
            self.start_ms <= self.recommended_keyframe_ms < self.end_ms
        ):
            raise ValueError("recommended_keyframe_ms must be inside [start_ms, end_ms)")
        return self


class ContentMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    content_type: str
    events: list[Event]
    entities: list[Entity]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_references(self) -> "ContentMap":
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id values must be unique")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique")
        known = set(entity_ids)
        for event in self.events:
            if event.end_ms > self.duration_ms:
                raise ValueError(f"event {event.event_id} exceeds duration_ms")
            refs = (
                event.entity_ids
                + event.primary_entity_ids
                + event.required_entity_ids
                + event.optional_entity_ids
                + event.avoid_overlay_entity_ids
            )
            unknown = set(refs) - known
            if unknown:
                raise ValueError(f"event {event.event_id} references unknown entities: {sorted(unknown)}")
        return self


class TemporalEvent(StrictModel):
    """Small first-pass event contract; intentionally excludes entities and layout advice."""

    event_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    label: str
    observable_evidence: str
    recommended_keyframe_ms: int = Field(ge=0)
    keyframe_reason: str
    confidence: Confidence
    boundary_precision: BoundaryPrecision

    @model_validator(mode="after")
    def validate_interval_and_keyframe(self) -> "TemporalEvent":
        if self.end_ms <= self.start_ms:
            raise ValueError("event interval must be non-empty and half-open")
        if not self.start_ms <= self.recommended_keyframe_ms < self.end_ms:
            raise ValueError("recommended_keyframe_ms must be inside [start_ms, end_ms)")
        return self


class TemporalMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    events: list[TemporalEvent]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timeline(self) -> "TemporalMap":
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id values must be unique")
        previous_end = 0
        for event in self.events:
            if event.end_ms > self.duration_ms:
                raise ValueError(f"event {event.event_id} exceeds duration_ms")
            if event.start_ms < previous_end:
                raise ValueError(f"event {event.event_id} overlaps or is out of order")
            previous_end = event.end_ms
        return self


class IndexedFrameEvent(StrictModel):
    event_id: str = Field(min_length=1)
    first_frame_id: str
    last_frame_id: str
    recommended_frame_id: str
    label: str
    observable_evidence: str
    grounding_target_id: str = Field(min_length=1)
    grounding_target_description: str
    confidence: Confidence
    boundary_precision: BoundaryPrecision


class IndexedStoryboardMap(StrictModel):
    """Model selects supplied IDs; local code owns all timestamp arithmetic."""

    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    events: list[IndexedFrameEvent]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_ids(self) -> "IndexedStoryboardMap":
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id values must be unique")
        return self


class FullClipGroundingTarget(StrictModel):
    entity_id: str = Field(min_length=1)
    target_kind: EntityKind
    target_description: str = Field(min_length=1)
    purpose: Literal["reframe", "callout", "isolation", "identity_check"]


class FullClipAttentionPhase(StrictModel):
    """Ordered visible attention evidence recorded while Gemini watches a clip."""

    phase_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    anchor_entity_ids: list[str] = Field(min_length=1, max_length=4)
    relation_mode: Literal["single_focus", "joint_relation"]
    suggested_camera_behavior: Literal[
        "hold",
        "follow",
        "follow_deadband",
        "push_in",
        "pull_out",
        "punch_in_cut",
    ]
    observable_predicate: str = Field(min_length=1, max_length=800)
    transition_condition: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_attention_phase(self) -> "FullClipAttentionPhase":
        if len(self.anchor_entity_ids) != len(set(self.anchor_entity_ids)):
            raise ValueError("attention phase entity IDs must be unique")
        if self.relation_mode == "single_focus" and len(self.anchor_entity_ids) != 1:
            raise ValueError("single-focus attention phase requires one entity")
        if self.relation_mode == "joint_relation" and len(self.anchor_entity_ids) < 2:
            raise ValueError("joint-relation attention phase requires two entities")
        return self


class FullClipEvent(StrictModel):
    """Gemini semantic event with second-level MM:SS anchors, never model milliseconds."""

    event_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    start_mmss: MmSs
    end_mmss: MmSs
    recommended_keyframe_mmss: MmSs | None
    label: str
    description: str
    observable_evidence: str
    evidence_modalities: EvidenceModality
    entity_ids: list[str]
    primary_entity_ids: list[str]
    required_entity_ids: list[str]
    optional_entity_ids: list[str]
    avoid_overlay_entity_ids: list[str]
    keyframe_reason: str
    boundary_precision: BoundaryPrecision
    confidence: Confidence
    action_completeness: Literal["complete", "partial", "uncertain"]
    editing_uses: list[
        Literal[
            "opening",
            "establishing",
            "hero",
            "detail",
            "demo",
            "reaction",
            "transition",
            "ending",
        ]
    ]
    quality_risks: list[str]
    framing_intent: str
    card_opportunities: list[CardOpportunity]
    dense_refinement: Literal["required", "recommended", "not_needed"]
    dense_refinement_reasons: list[str]
    grounding_targets: list[FullClipGroundingTarget]
    portrait_attention_sequence: list[FullClipAttentionPhase] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Ordered visible attention evidence only; no timestamps, crop "
            "coordinates, or authorization to execute a virtual camera."
        ),
    )

    @model_validator(mode="after")
    def validate_mmss_interval(self) -> "FullClipEvent":
        start_ms = _mmss_to_ms(self.start_mmss)
        end_ms = _mmss_to_ms(self.end_mmss)
        if end_ms <= start_ms:
            raise ValueError("event MM:SS interval must be non-empty and half-open")
        if self.recommended_keyframe_mmss is not None:
            keyframe_ms = _mmss_to_ms(self.recommended_keyframe_mmss)
            if not start_ms <= keyframe_ms < end_ms:
                raise ValueError("recommended MM:SS keyframe must be inside [start, end)")
        return self

    def resolved_end_ms(self, duration_ms: int) -> int:
        """Resolve the only MM:SS interval that can represent a sub-second clip."""
        labeled_end_ms = _mmss_to_ms(self.end_mmss)
        if (
            duration_ms < 1000
            and _mmss_to_ms(self.start_mmss) == 0
            and labeled_end_ms == 1000
        ):
            return duration_ms
        return labeled_end_ms


class FullClipCard(StrictModel):
    """Complete per-clip semantic record produced from a full analysis proxy."""

    source_asset_id: str = Field(min_length=1)
    proxy_asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    content_type: str
    entities: list[Entity]
    events: list[FullClipEvent]
    clip_uses: list[str]
    portrait_reframe_feasibility: Literal["good", "conditional", "poor", "uncertain"]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timeline_and_references(self) -> "FullClipCard":
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id values must be unique")
        event_ids = [event.event_id for event in self.events]
        if not event_ids:
            raise ValueError("a full Clip Card must contain at least one event")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique")
        known_entities = set(entity_ids)
        entity_kinds = {entity.entity_id: entity.kind for entity in self.entities}
        previous_end = 0
        for event in self.events:
            start_ms = _mmss_to_ms(event.start_mmss)
            end_ms = event.resolved_end_ms(self.duration_ms)
            if end_ms > self.duration_ms:
                raise ValueError(f"event {event.event_id} MM:SS exceeds duration")
            if end_ms <= start_ms:
                raise ValueError(
                    f"event {event.event_id} resolved interval must be non-empty"
                )
            if event.recommended_keyframe_mmss is not None:
                keyframe_ms = _mmss_to_ms(event.recommended_keyframe_mmss)
                if not start_ms <= keyframe_ms < end_ms:
                    raise ValueError(
                        f"event {event.event_id} keyframe exceeds resolved interval"
                    )
            if start_ms < previous_end:
                raise ValueError(f"event {event.event_id} overlaps or is out of order")
            previous_end = end_ms
            references = (
                event.entity_ids
                + event.primary_entity_ids
                + event.required_entity_ids
                + event.optional_entity_ids
                + event.avoid_overlay_entity_ids
                + [target.entity_id for target in event.grounding_targets]
                + [
                    entity_id
                    for phase in event.portrait_attention_sequence
                    for entity_id in phase.anchor_entity_ids
                ]
            )
            unknown = sorted(set(references) - known_entities)
            if unknown:
                raise ValueError(
                    f"event {event.event_id} references unknown entities: {unknown}"
                )
            for target in event.grounding_targets:
                if entity_kinds[target.entity_id] != target.target_kind:
                    raise ValueError(
                        f"event {event.event_id} Grounding target kind differs from Entity kind"
                    )
            for opportunity in event.card_opportunities:
                unknown_card_entities = sorted(
                    set(opportunity.entity_ids) - known_entities
                )
                if unknown_card_entities:
                    raise ValueError(
                        f"event {event.event_id} card references unknown entities: "
                        f"{unknown_card_entities}"
                    )
        return self


class DerivedClipEvent(StrictModel):
    """Local conversion of model MM:SS plus FFmpeg shot membership."""

    event_id: str
    start_mmss: MmSs
    end_mmss: MmSs
    recommended_keyframe_mmss: MmSs | None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    recommended_keyframe_ms: int | None = Field(default=None, ge=0)
    shot_ids: list[str]
    boundary_source: Literal[
        "gemini_mmss_local_conversion",
        "gemini_mmss_subsecond_clip_end_conversion",
    ]
    exact_frame_required: bool

    @model_validator(mode="after")
    def validate_derived_interval(self) -> "DerivedClipEvent":
        if self.start_ms != _mmss_to_ms(self.start_mmss):
            raise ValueError("start_ms must be locally derived from start_mmss")
        labeled_end_ms = _mmss_to_ms(self.end_mmss)
        if self.boundary_source == "gemini_mmss_local_conversion":
            if self.end_ms != labeled_end_ms:
                raise ValueError("end_ms must be locally derived from end_mmss")
        elif not (
            self.start_ms == 0
            and labeled_end_ms == 1000
            and 0 < self.end_ms < 1000
        ):
            raise ValueError(
                "sub-second clip-end conversion requires 00:00–00:01 display "
                "labels and an authoritative end below 1000 ms"
            )
        expected_keyframe = (
            _mmss_to_ms(self.recommended_keyframe_mmss)
            if self.recommended_keyframe_mmss is not None
            else None
        )
        if self.recommended_keyframe_ms != expected_keyframe:
            raise ValueError("recommended_keyframe_ms must be locally derived from MM:SS")
        if self.end_ms <= self.start_ms:
            raise ValueError("derived event interval must be non-empty")
        return self


class DerivedClipTimeline(StrictModel):
    source_asset_id: str
    duration_ms: int = Field(gt=0)
    events: list[DerivedClipEvent]
    generated_at: str


class ShotRepresentativeFrame(StrictModel):
    frame_id: str = Field(pattern=r"^CF[0-9]{6}$")
    shot_id: str
    role: Literal["start", "middle", "end"]
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_path: str


class ClipShotCatalog(StrictModel):
    source_asset_id: str
    duration_ms: int = Field(gt=0)
    frames: list[ShotRepresentativeFrame]
    generated_at: str


class DenseFrame(StrictModel):
    frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    event_id: str
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_path: str
    transport_image_path: str
    transport_image_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DenseFrameCatalog(StrictModel):
    source_asset_id: str
    event_id: str
    sampling_fps: float = Field(gt=0, le=8)
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(gt=0)
    frames: list[DenseFrame] = Field(min_length=1, max_length=3600)
    contact_sheet_paths: list[str] = Field(min_length=1)
    contact_sheet_hashes: list[str] = Field(min_length=1)
    generated_at: str

    @model_validator(mode="after")
    def validate_contact_sheets(self) -> "DenseFrameCatalog":
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError("dense source window must be non-empty and half-open")
        if len(self.contact_sheet_paths) != len(self.contact_sheet_hashes):
            raise ValueError("contact sheet paths and hashes must align")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("dense frame IDs must be unique")
        if frame_ids != sorted(frame_ids):
            raise ValueError("dense frame IDs must be ordered")
        requested_times = [frame.requested_time_ms for frame in self.frames]
        if any(
            current >= following
            for current, following in zip(requested_times, requested_times[1:])
        ):
            raise ValueError("dense requested times must be strictly increasing")
        frame_times = [frame.frame_time_ms for frame in self.frames]
        if any(
            current > following
            for current, following in zip(frame_times, frame_times[1:])
        ):
            raise ValueError("dense source frame times must be ordered")
        for frame in self.frames:
            if frame.event_id != self.event_id:
                raise ValueError("dense frame event_id must match its catalog")
            if not self.source_start_ms <= frame.requested_time_ms < self.source_end_ms:
                raise ValueError("dense requested time must be inside the source window")
            if not self.source_start_ms <= frame.frame_time_ms < self.source_end_ms:
                raise ValueError("dense source frame time must be inside the source window")
        return self


class DenseEventSelection(StrictModel):
    source_asset_id: str
    event_id: str
    visible: bool
    first_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    recommended_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    last_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    target_entity_id: str | None = None
    target_description: str | None = None
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    observable_evidence: str
    selection_reason: str
    uncertainties: list[str]
    confidence: Confidence
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "DenseEventSelection":
        frame_ids = [self.first_frame_id, self.recommended_frame_id, self.last_frame_id]
        target_fields = [self.target_entity_id, self.target_description]
        self.match_status = self.match_status or (
            MatchStatus.MATCHED if self.visible else MatchStatus.NOT_VISIBLE
        )
        if self.visible:
            if any(frame_id is None for frame_id in frame_ids):
                raise ValueError("visible dense selections require first/recommended/last IDs")
            if self.match_status not in {MatchStatus.MATCHED, MatchStatus.AMBIGUOUS}:
                raise ValueError("visible dense selections require matched or ambiguous status")
            assert all(frame_id is not None for frame_id in frame_ids)
            if not (
                self.first_frame_id
                <= self.recommended_frame_id
                <= self.last_frame_id
            ):
                raise ValueError("dense selection frame IDs must be ordered")
        else:
            if any(value is not None for value in frame_ids + target_fields):
                raise ValueError(
                    "invisible dense selections cannot reference frame or target fields"
                )
            if self.match_status not in {
                MatchStatus.NOT_VISIBLE,
                MatchStatus.TARGET_MISMATCH,
                MatchStatus.INSUFFICIENT_EVIDENCE,
            }:
                raise ValueError("invisible dense selection has incompatible match_status")
        if bool(self.target_entity_id) != bool(self.target_description):
            raise ValueError("dense selection target ID and description must appear together")
        return self


QualityRiskReason = Literal[
    "black",
    "white_clip",
    "freeze",
    "focus_loss",
    "motion_blur",
    "camera_shake",
    "occlusion",
    "target_not_visible",
    "duplicate_pts",
    "decoder_gap",
    "compression_artifact",
]

QualityRiskSeverity = Literal[
    "hard_block",
    "trim_candidate",
    "review",
    "note",
]

QualityRiskIntent = Literal[
    "accidental",
    "intentional",
    "unknown",
]


class QualityFrameEvidence(StrictModel):
    """One reproducible analysis frame tied back to decoded source PTS."""

    frame_id: str = Field(pattern=r"^QF-[0-9a-f]{16}$")
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    analysis_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QualityRiskWindow(StrictModel):
    """Measured source interval; it is not itself an edit instruction."""

    risk_window_id: str = Field(pattern=r"^QRW-[0-9]{4}$")
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    shot_id: str = Field(min_length=1)
    start_pts: int
    end_pts: int
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    reason_code: QualityRiskReason
    severity: QualityRiskSeverity
    intent: QualityRiskIntent
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frame_ids: list[str] = Field(max_length=16)
    metric_summary: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_quality_window(self) -> "QualityRiskWindow":
        if self.end_pts <= self.start_pts:
            raise ValueError("quality risk PTS interval must be non-empty")
        if self.end_ms <= self.start_ms:
            raise ValueError("quality risk time interval must be non-empty")
        if len(self.evidence_frame_ids) != len(set(self.evidence_frame_ids)):
            raise ValueError("quality risk evidence frame IDs must be unique")
        if self.reason_code != "decoder_gap" and not self.evidence_frame_ids:
            raise ValueError(
                "measured quality risks require at least one evidence frame"
            )
        return self


class ShotQualityMap(StrictModel):
    """Deterministic source-FPS measurements for one immutable source shot."""

    contract_version: Literal["shot-quality-map-v1"] = "shot-quality-map-v1"
    scanner_version: str = Field(min_length=1)
    source_path: str
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    shot_id: str = Field(min_length=1)
    shot_start_pts: int
    shot_end_pts: int
    shot_start_ms: int = Field(ge=0)
    shot_end_ms: int = Field(gt=0)
    source_time_base: Rational
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    decoded_frame_count: int = Field(ge=0)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_frames: list[QualityFrameEvidence]
    risk_windows: list[QualityRiskWindow]
    warnings: list[str]
    generated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_quality_map(self) -> "ShotQualityMap":
        if self.shot_end_pts <= self.shot_start_pts:
            raise ValueError("shot quality PTS interval must be non-empty")
        if self.shot_end_ms <= self.shot_start_ms:
            raise ValueError("shot quality time interval must be non-empty")
        evidence_ids = [frame.frame_id for frame in self.evidence_frames]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("quality evidence frame IDs must be unique")
        known_evidence = set(evidence_ids)
        window_ids = [window.risk_window_id for window in self.risk_windows]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("quality risk window IDs must be unique")
        for window in self.risk_windows:
            if (
                window.source_asset_id != self.source_asset_id
                or window.shot_id != self.shot_id
            ):
                raise ValueError("quality risk window lineage differs from its map")
            if not (
                self.shot_start_ms
                <= window.start_ms
                < window.end_ms
                <= self.shot_end_ms
            ):
                raise ValueError("quality risk window lies outside its shot")
            if not set(window.evidence_frame_ids) <= known_evidence:
                raise ValueError("quality risk references unknown evidence frames")
        return self


class QualitySafeInterval(StrictModel):
    """One continuous interval eligible for deterministic source trimming."""

    interval_id: str = Field(pattern=r"^QSI-[0-9]{4}$")
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    shot_id: str = Field(min_length=1)
    start_pts: int
    end_pts: int
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    excluded_risk_window_ids: list[str]
    review_risk_window_ids: list[str]
    requires_human_review: bool
    quality_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_safe_interval(self) -> "QualitySafeInterval":
        if self.end_pts <= self.start_pts or self.end_ms <= self.start_ms:
            raise ValueError("quality-safe interval must be non-empty")
        if len(self.excluded_risk_window_ids) != len(
            set(self.excluded_risk_window_ids)
        ):
            raise ValueError("excluded risk IDs must be unique")
        if len(self.review_risk_window_ids) != len(
            set(self.review_risk_window_ids)
        ):
            raise ValueError("review risk IDs must be unique")
        if self.requires_human_review != bool(self.review_risk_window_ids):
            raise ValueError(
                "safe interval review flag must reflect unresolved review risks"
            )
        return self


class AspectCandidateCapacity(StrictModel):
    """Quality and geometry capacity for one delivery aspect."""

    aspect: Literal["16:9", "9:16"]
    geometry_status: Literal[
        "not_evaluated",
        "feasible",
        "partial",
        "blocked",
    ]
    safe_intervals: list[QualitySafeInterval]
    maximum_continuous_seconds: float = Field(ge=0.0)
    requires_human_review: bool

    @model_validator(mode="after")
    def validate_capacity(self) -> "AspectCandidateCapacity":
        if self.geometry_status == "blocked" and (
            self.safe_intervals or self.maximum_continuous_seconds != 0
        ):
            raise ValueError(
                "geometry-blocked capacity cannot expose executable intervals"
            )
        measured = max(
            (
                (interval.end_ms - interval.start_ms) / 1000
                for interval in self.safe_intervals
            ),
            default=0.0,
        )
        if abs(measured - self.maximum_continuous_seconds) > 0.001:
            raise ValueError(
                "aspect capacity must equal its longest continuous safe interval"
            )
        if self.requires_human_review != any(
            interval.requires_human_review for interval in self.safe_intervals
        ):
            raise ValueError(
                "aspect review flag must reflect its safe interval evidence"
            )
        return self


class CandidateCapacity(StrictModel):
    """Planning capacity; never inferred from the complete shot duration."""

    contract_version: Literal["candidate-capacity-v1"] = "candidate-capacity-v1"
    candidate_id: str = Field(min_length=1)
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    shot_id: str = Field(min_length=1)
    horizontal: AspectCandidateCapacity
    vertical: AspectCandidateCapacity
    min_editorial_duration: float = Field(ge=0.0)
    preferred_duration: float = Field(ge=0.0)
    max_editorial_duration: float = Field(ge=0.0)
    quality_map_path: str
    quality_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_candidate_capacity(self) -> "CandidateCapacity":
        if self.horizontal.aspect != "16:9" or self.vertical.aspect != "9:16":
            raise ValueError("candidate capacity aspect fields are reversed")
        if not (
            self.min_editorial_duration
            <= self.preferred_duration
            <= self.max_editorial_duration
        ):
            raise ValueError(
                "candidate editorial durations must satisfy min <= preferred <= max"
            )
        available = min(
            self.horizontal.maximum_continuous_seconds,
            self.vertical.maximum_continuous_seconds,
        )
        if self.max_editorial_duration > available + 0.001:
            raise ValueError(
                "candidate maximum editorial duration exceeds safe aspect capacity"
            )
        return self


class AttentionChapterProfile(StrictModel):
    """One chapter's attention evidence and executable dwell envelope."""

    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    evidence_authority: Literal[
        "gemini_attention_observation",
        "gemini_relative_dwell_legacy",
        "brief_fallback",
    ]
    semantic_novelty: float | None = Field(default=None, ge=0.0, le=1.0)
    action_progress: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_motion: float | None = Field(default=None, ge=0.0, le=1.0)
    composition_change: float | None = Field(default=None, ge=0.0, le=1.0)
    reading_load: float | None = Field(default=None, ge=0.0, le=1.0)
    unresolved_tension: float | None = Field(default=None, ge=0.0, le=1.0)
    emotional_hold_value: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    music_transition_opportunity: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    minimum_dwell_seconds: float = Field(gt=0.0)
    preferred_dwell_seconds: float = Field(gt=0.0)
    maximum_dwell_seconds: float = Field(gt=0.0)
    quality_safe_capacity_seconds: float | None = Field(default=None, ge=0.0)
    flow_intent: "ShotFlowIntent | None" = None
    rationale: str = Field(min_length=1)
    uncertainties: list[str]
    requires_human_review: bool

    @model_validator(mode="after")
    def validate_attention_profile(self) -> "AttentionChapterProfile":
        if not (
            self.minimum_dwell_seconds
            <= self.preferred_dwell_seconds
            <= self.maximum_dwell_seconds
        ):
            raise ValueError(
                "attention dwell must satisfy minimum <= preferred <= maximum"
            )
        if (
            self.quality_safe_capacity_seconds is not None
            and self.maximum_dwell_seconds
            > self.quality_safe_capacity_seconds + 0.001
        ):
            raise ValueError("attention maximum exceeds quality-safe capacity")
        metrics = (
            self.semantic_novelty,
            self.action_progress,
            self.visual_motion,
            self.composition_change,
            self.reading_load,
            self.unresolved_tension,
            self.emotional_hold_value,
            self.repetition_pressure,
            self.music_transition_opportunity,
        )
        if self.evidence_authority == "gemini_attention_observation":
            if any(value is None for value in metrics):
                raise ValueError("Gemini attention profiles require the full vector")
        elif any(value is not None for value in metrics):
            raise ValueError("legacy attention profiles cannot invent vector values")
        return self


class AttentionProfile(StrictModel):
    contract_version: Literal["attention-profile-v1"] = "attention-profile-v1"
    project_id: str
    source_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_feature_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chapters: list[AttentionChapterProfile] = Field(min_length=1)
    generated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_attention_chapters(self) -> "AttentionProfile":
        ids = [chapter.feature_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("attention profile chapter IDs must be unique")
        return self


class RhythmChapterPlan(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    minimum_duration_seconds: float = Field(gt=0.0)
    preferred_duration_seconds: float = Field(gt=0.0)
    maximum_duration_seconds: float = Field(gt=0.0)
    cut_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_priority: Literal["low", "normal", "high"]
    boundary_alignment: Literal[
        "content_locked",
        "phrase_preferred",
        "accent_preferred",
        "free",
    ] = "free"
    flow_intent: "ShotFlowIntent | None" = None
    protected_reasons: list[str]
    transition_reasons: list[str]
    evidence_authority: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rhythm_durations(self) -> "RhythmChapterPlan":
        if not (
            self.minimum_duration_seconds
            <= self.preferred_duration_seconds
            <= self.maximum_duration_seconds
        ):
            raise ValueError(
                "rhythm durations must satisfy minimum <= preferred <= maximum"
            )
        return self


class RhythmPlan(StrictModel):
    contract_version: Literal["rhythm-plan-v1"] = "rhythm-plan-v1"
    project_id: str
    style_profile: Literal["calm", "standard", "energetic"]
    target_duration_seconds: float = Field(gt=0.0)
    attention_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chapters: list[RhythmChapterPlan] = Field(min_length=1)
    interpretation: Literal[
        "attention_bounds_and_boundary_pressure_not_frame_accurate_cuts"
    ] = "attention_bounds_and_boundary_pressure_not_frame_accurate_cuts"
    generated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rhythm_chapters(self) -> "RhythmPlan":
        ids = [chapter.feature_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("rhythm plan chapter IDs must be unique")
        if sum(chapter.minimum_duration_seconds for chapter in self.chapters) > (
            self.target_duration_seconds + 0.001
        ):
            raise ValueError("rhythm minimum durations exceed project duration")
        if sum(chapter.maximum_duration_seconds for chapter in self.chapters) + (
            0.001
        ) < self.target_duration_seconds:
            raise ValueError("rhythm maximum durations cannot fill project duration")
        return self


class VirtualCameraKeyframe(StrictModel):
    time_seconds: float = Field(ge=0.0)
    source_pts: int | None = None
    scale: float = Field(ge=1.0)
    center_x_normalized: float = Field(ge=0.0, le=1000.0)
    center_y_normalized: float = Field(ge=0.0, le=1000.0)


class VirtualCameraPlan(StrictModel):
    contract_version: Literal["virtual-camera-plan-v1"] = "virtual-camera-plan-v1"
    requested_intent: VirtualCameraIntent
    applied_intent: VirtualCameraIntent
    anchor_target_ids: list[str]
    keyframes: list[VirtualCameraKeyframe] = Field(min_length=1)
    easing: Literal["hold", "linear", "smoothstep", "cut"]
    geometry_safe_max_scale: float = Field(ge=1.0)
    source_resolution_native_scale_limit: float = Field(ge=1.0)
    source_resolution_upscale_required: bool
    max_velocity: float = Field(ge=0.0)
    max_acceleration: float = Field(ge=0.0)
    max_jerk: float = Field(ge=0.0)
    execution_status: Literal["applied", "fallback", "blocked"]
    fallback_reason: str | None = None
    editorial_reason: str = Field(min_length=1)
    source_track_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_virtual_camera(self) -> "VirtualCameraPlan":
        times = [keyframe.time_seconds for keyframe in self.keyframes]
        if times != sorted(set(times)):
            raise ValueError("virtual-camera keyframe times must be strictly increasing")
        if self.execution_status == "applied" and self.fallback_reason is not None:
            raise ValueError("applied virtual-camera plans cannot have a fallback reason")
        if self.execution_status != "applied" and not self.fallback_reason:
            raise ValueError("non-applied virtual-camera plans require a reason")
        if self.requested_intent == "pan_reveal" and len(self.anchor_target_ids) < 2:
            if self.execution_status == "applied":
                raise ValueError(
                    "pan reveal requires two anchors or a non-applied plan"
                )
        if max(keyframe.scale for keyframe in self.keyframes) > (
            self.geometry_safe_max_scale + 0.001
        ):
            raise ValueError("virtual-camera scale exceeds geometry-safe limit")
        expected_upscale = max(
            keyframe.scale for keyframe in self.keyframes
        ) > (self.source_resolution_native_scale_limit + 0.001)
        if self.source_resolution_upscale_required != expected_upscale:
            raise ValueError(
                "source-resolution upscale flag disagrees with camera scales"
            )
        return self


class VerticalVirtualCameraPhase(StrictModel):
    """One reviewable portrait composition phase over normalized edit progress.

    A phase changes which already-grounded regions are compositionally required.
    It does not create a new identity, bbox, mask, timestamp, or source edit.
    """

    phase_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    start_progress: float = Field(ge=0.0, le=1.0)
    end_progress: float = Field(gt=0.0, le=1.0)
    anchor_region_ids: list[str] = Field(min_length=1, max_length=4)
    camera_behavior: Literal[
        "hold",
        "follow",
        "follow_deadband",
        "push_in",
        "pull_out",
        "punch_in_cut",
    ] = "follow_deadband"
    movement_motivation: Literal[
        "none",
        "maintain_framing",
        "attention_handoff",
        "reveal",
        "emphasis",
    ] = "none"
    traversal_policy: Literal[
        "semantic_order_locked",
        "spatially_optimizable",
        "no_continuous_traversal",
    ] = "semantic_order_locked"
    cut_admissible: bool = False
    transition_in: Literal["cut", "smoothstep"] = "cut"
    transition_duration_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_anchor_visible_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Reviewed visibility floor for the active anchor union during the "
            "steady part of this phase. A value below 1 permits intentional, "
            "auditable clipping rather than an implicit center-crop fallback."
        ),
    )
    editorial_reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_phase(self) -> "VerticalVirtualCameraPhase":
        if self.end_progress <= self.start_progress:
            raise ValueError("vertical camera phase must have positive duration")
        if len(self.anchor_region_ids) != len(set(self.anchor_region_ids)):
            raise ValueError("vertical camera phase anchor IDs must be unique")
        if self.transition_in == "cut" and self.transition_duration_fraction != 0:
            raise ValueError("cut transition cannot have a transition duration")
        if self.transition_in == "smoothstep" and (
            self.transition_duration_fraction <= 0
        ):
            raise ValueError("smoothstep transition requires a positive duration")
        return self


class VerticalVirtualCameraProposalPhase(StrictModel):
    """Gemini-authored editorial phase before geometry is allowed to execute.

    Progress values describe relative editorial order only.  They are not source
    timestamps, frame boundaries, crop coordinates, or proof that the requested
    anchors can coexist inside a portrait crop.
    """

    phase_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    start_progress: float = Field(ge=0.0, le=1.0)
    end_progress: float = Field(gt=0.0, le=1.0)
    anchor_region_ids: list[str] = Field(min_length=1, max_length=4)
    camera_behavior: Literal[
        "hold",
        "follow",
        "follow_deadband",
        "push_in",
        "pull_out",
        "punch_in_cut",
    ] = "follow_deadband"
    movement_motivation: Literal[
        "none",
        "maintain_framing",
        "attention_handoff",
        "reveal",
        "emphasis",
    ] = "none"
    cut_admissible: bool = False
    transition_in: Literal["cut", "smoothstep"] = "cut"
    transition_duration_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    observable_predicate: str = Field(
        min_length=1,
        max_length=800,
        description=(
            "Visible condition that makes this anchor relevant; do not use brand "
            "knowledge, an invented timestamp, or an inferred off-screen state."
        ),
    )
    transition_condition: str = Field(
        min_length=1,
        max_length=800,
        description=(
            "Visible editorial hand-off condition.  A later dense-frame pass may "
            "refine it to immutable frame IDs before any source trim changes."
        ),
    )
    editorial_reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_proposal_phase(self) -> "VerticalVirtualCameraProposalPhase":
        if self.end_progress <= self.start_progress:
            raise ValueError("vertical camera proposal phase must have positive duration")
        if len(self.anchor_region_ids) != len(set(self.anchor_region_ids)):
            raise ValueError("vertical camera proposal anchor IDs must be unique")
        if self.transition_in == "cut" and self.transition_duration_fraction != 0:
            raise ValueError("cut transition cannot have a transition duration")
        if self.transition_in == "smoothstep" and (
            self.transition_duration_fraction <= 0
        ):
            raise ValueError("smoothstep transition requires a positive duration")
        return self


class VerticalVirtualCameraProposal(StrictModel):
    """Evidence-only Gemini proposal; local geometry remains authoritative."""

    contract_version: Literal["vertical-virtual-camera-proposal-v1"] = (
        "vertical-virtual-camera-proposal-v1"
    )
    composition_mode: Literal[
        "single_anchor_hold",
        "single_anchor_follow",
        "sequential_focus",
        "joint_relation",
        "mixed_relation",
    ]
    traversal_policy: Literal[
        "semantic_order_locked",
        "spatially_optimizable",
        "no_continuous_traversal",
    ] = "semantic_order_locked"
    phases: list[VerticalVirtualCameraProposalPhase] = Field(
        min_length=1,
        max_length=8,
    )
    proposal_reason: str = Field(min_length=1, max_length=1200)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_proposal(self) -> "VerticalVirtualCameraProposal":
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("vertical camera proposal phase IDs must be unique")
        if abs(self.phases[0].start_progress) > 1e-6:
            raise ValueError("vertical camera proposal must start at progress zero")
        if abs(self.phases[-1].end_progress - 1.0) > 1e-6:
            raise ValueError("vertical camera proposal must end at progress one")
        for prior, current in zip(self.phases[:-1], self.phases[1:], strict=True):
            if abs(prior.end_progress - current.start_progress) > 1e-6:
                raise ValueError(
                    "vertical camera proposal phases must be contiguous"
                )
        if any(
            phase.transition_in == "cut" and not phase.cut_admissible
            for phase in self.phases[1:]
        ):
            raise ValueError(
                "a proposed hard cut requires an explicit semantic "
                "cut-admissibility decision"
            )
        if self.phases[0].transition_in != "cut":
            raise ValueError(
                "the first vertical camera proposal phase must use a cut"
            )
        unique_anchor_ids = {
            region_id
            for phase in self.phases
            for region_id in phase.anchor_region_ids
        }
        if self.composition_mode == "sequential_focus" and (
            len(self.phases) < 2 or len(unique_anchor_ids) < 2
        ):
            raise ValueError(
                "sequential focus requires at least two phases and two anchors"
            )
        if self.composition_mode in {
            "single_anchor_hold",
            "single_anchor_follow",
        } and len(unique_anchor_ids) != 1:
            raise ValueError("single-anchor modes must reference exactly one anchor")
        if self.composition_mode == "joint_relation" and not any(
            len(phase.anchor_region_ids) >= 2 for phase in self.phases
        ):
            raise ValueError(
                "joint-relation composition requires at least one multi-anchor phase"
            )
        if self.composition_mode == "mixed_relation":
            if not any(
                len(phase.anchor_region_ids) >= 2 for phase in self.phases
            ) or not any(
                len(phase.anchor_region_ids) == 1 for phase in self.phases
            ):
                raise ValueError(
                    "mixed-relation composition requires both joint and "
                    "single-anchor phases"
                )
        if self.traversal_policy == "spatially_optimizable":
            if self.composition_mode != "sequential_focus" or any(
                len(phase.anchor_region_ids) != 1 for phase in self.phases
            ):
                raise ValueError(
                    "spatially optimizable traversal is only valid for "
                    "independent single-anchor sequential phases"
                )
        return self


class VerticalVirtualCameraPlan(StrictModel):
    """Executed phase-based 9:16 crop plan tied to tracked region evidence."""

    contract_version: Literal["vertical-virtual-camera-plan-v1"] = (
        "vertical-virtual-camera-plan-v1"
    )
    phases: list[VerticalVirtualCameraPhase] = Field(min_length=1)
    anchor_region_ids: list[str] = Field(min_length=1)
    keyframes: list[VirtualCameraKeyframe] = Field(min_length=2)
    steady_containment_passed: bool
    transition_minimum_anchor_visible_fraction: float = Field(ge=0.0, le=1.0)
    max_velocity: float = Field(ge=0.0)
    max_acceleration: float = Field(ge=0.0)
    max_jerk: float = Field(ge=0.0)
    execution_status: Literal["applied", "fallback", "blocked"]
    fallback_reason: str | None = None
    requires_human_review: Literal[True] = True
    editorial_reason: str = Field(min_length=1)
    source_track_fingerprints: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_vertical_virtual_camera(self) -> "VerticalVirtualCameraPlan":
        times = [keyframe.time_seconds for keyframe in self.keyframes]
        if times != sorted(set(times)):
            raise ValueError(
                "vertical virtual-camera keyframe times must be strictly increasing"
            )
        if self.execution_status == "applied" and self.fallback_reason is not None:
            raise ValueError(
                "applied vertical virtual-camera plans cannot have a fallback reason"
            )
        if self.execution_status != "applied" and not self.fallback_reason:
            raise ValueError(
                "non-applied vertical virtual-camera plans require a reason"
            )
        expected_ids = list(
            dict.fromkeys(
                region_id
                for phase in self.phases
                for region_id in phase.anchor_region_ids
            )
        )
        if self.anchor_region_ids != expected_ids:
            raise ValueError(
                "vertical virtual-camera anchor IDs must match phase order"
            )
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
            for fingerprint in self.source_track_fingerprints.values()
        ):
            raise ValueError("vertical camera track fingerprints must be SHA-256")
        return self


TrimTailIntent = Literal[
    "none",
    "natural_pause",
    "intentional_hold",
    "title_safe_hold",
    "clean_plate",
    "reset_or_false_end",
    "uncertain",
]


TrimPhase = Literal[
    "setup_start",
    "action_start",
    "result_start",
    "hold_start",
    "hold_end",
    "reset_start",
    "recommended_in",
    "recommended_out",
]


class TrimPhaseSelection(StrictModel):
    """One observable trim phase tied to one supplied dense-frame ID."""

    phase: TrimPhase
    frame_id: str = Field(min_length=8, max_length=8)


class TrimIntentProposal(StrictModel):
    """Evidence-bound trim phases selected from immutable dense frame IDs."""

    source_asset_id: str
    event_id: str
    usable: bool
    selections: list[TrimPhaseSelection] = Field(max_length=8)
    tail_intent: TrimTailIntent
    observed_phase_evidence: str = Field(max_length=800)
    hold_evidence: str = Field(max_length=500)
    trim_reason: str = Field(max_length=500)
    quality_risks: list[str] = Field(max_length=8)
    uncertainties: list[str] = Field(max_length=8)
    requires_human_review: bool
    confidence: Confidence
    model_provenance: ModelProvenance

    def frame_id_for(self, phase: TrimPhase) -> str | None:
        selection = next((item for item in self.selections if item.phase == phase), None)
        return selection.frame_id if selection is not None else None

    @model_validator(mode="after")
    def validate_usable_fields(self) -> "TrimIntentProposal":
        if not self.requires_human_review:
            raise ValueError("Gemini trim proposals always require human review")
        phases = [selection.phase for selection in self.selections]
        if len(phases) != len(set(phases)):
            raise ValueError("trim phases must be unique")
        required = [self.frame_id_for("recommended_in"), self.frame_id_for("recommended_out")]
        if self.usable:
            if any(frame_id is None for frame_id in required):
                raise ValueError("usable trim proposals require recommended in/out frame IDs")
            if required[0] == required[1]:
                raise ValueError("trim proposal must include at least two sampled frames")
        elif self.selections:
            raise ValueError("unusable trim proposals cannot reference frame IDs")
        if ("hold_start" in phases) != ("hold_end" in phases):
            raise ValueError("hold start/end frame IDs must appear together")
        return self


class VideoTrimIntentProposal(StrictModel):
    """Second-level direct-video trim proposal; local code resolves exact frame PTS."""

    source_asset_id: str
    event_id: str
    usable: bool
    recommended_in_mmss: MmSs | None
    recommended_out_mmss: MmSs | None
    hold_start_mmss: MmSs | None = None
    hold_end_mmss: MmSs | None = None
    reset_start_mmss: MmSs | None = None
    tail_intent: TrimTailIntent
    observed_phase_evidence: str = Field(max_length=800)
    hold_evidence: str = Field(max_length=500)
    trim_reason: str = Field(max_length=500)
    quality_risks: list[str] = Field(max_length=8)
    uncertainties: list[str] = Field(max_length=8)
    requires_human_review: bool
    confidence: Confidence
    model_provenance: ModelProvenance

    @model_validator(mode="before")
    @classmethod
    def preserve_incomplete_hold_as_uncertainty(cls, value: object) -> object:
        """Omit an unusable half-interval explicitly instead of inventing its endpoint."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        start = normalized.get("hold_start_mmss")
        end = normalized.get("hold_end_mmss")
        if bool(start) == bool(end):
            return normalized
        warning = (
            "contract_normalization: Gemini returned only one hold endpoint; "
            "the incomplete hold interval was omitted without inferring a missing time"
        )
        uncertainties = list(normalized.get("uncertainties") or [])
        if len(uncertainties) < 8:
            uncertainties.append(warning)
        else:
            uncertainties[-1] = f"{uncertainties[-1]}; {warning}"
        normalized["uncertainties"] = uncertainties
        normalized["hold_start_mmss"] = None
        normalized["hold_end_mmss"] = None
        return normalized

    @model_validator(mode="after")
    def validate_video_trim_fields(self) -> "VideoTrimIntentProposal":
        if not self.requires_human_review:
            raise ValueError("Gemini video trim proposals always require human review")
        boundaries = [self.recommended_in_mmss, self.recommended_out_mmss]
        if self.usable:
            if any(value is None for value in boundaries):
                raise ValueError("usable video trim proposals require in/out MM:SS")
            assert self.recommended_in_mmss is not None
            assert self.recommended_out_mmss is not None
            if _mmss_to_ms(self.recommended_out_mmss) <= _mmss_to_ms(
                self.recommended_in_mmss
            ):
                raise ValueError("video trim proposal must have in < exclusive out")
        elif any(
            value is not None
            for value in [
                *boundaries,
                self.hold_start_mmss,
                self.hold_end_mmss,
                self.reset_start_mmss,
            ]
        ):
            raise ValueError("unusable video trim proposals cannot reference MM:SS")
        if bool(self.hold_start_mmss) != bool(self.hold_end_mmss):
            raise ValueError("video hold start/end MM:SS must appear together")
        if self.hold_start_mmss is not None and self.hold_end_mmss is not None:
            if _mmss_to_ms(self.hold_end_mmss) < _mmss_to_ms(self.hold_start_mmss):
                raise ValueError("video hold interval must be chronological")
        return self


class TrimFrameEvidence(StrictModel):
    frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrimHumanReview(StrictModel):
    reviewer: str = Field(min_length=1)
    reviewed_at: str
    decision: Literal["approved", "rejected"]
    notes: str = ""


class TrimIntentDecision(StrictModel):
    """Local PTS derivation from a model proposal; never a human-approved cut by default."""

    source_asset_id: str
    event_id: str
    shot_id: str
    usable: bool
    first_included_frame: TrimFrameEvidence | None
    last_included_frame: TrimFrameEvidence | None
    exclusive_out_frame: TrimFrameEvidence | None
    hold_start_frame: TrimFrameEvidence | None
    hold_end_frame: TrimFrameEvidence | None
    source_in_ms: int | None = Field(default=None, ge=0)
    source_out_ms: int | None = Field(default=None, gt=0)
    source_in_pts: int | None = None
    source_out_pts: int | None = None
    handle_in_ms: int | None = Field(default=None, ge=0)
    handle_out_ms: int | None = Field(default=None, gt=0)
    tail_intent: TrimTailIntent
    approval_status: Literal["proposed", "approved", "rejected"] = "proposed"
    requires_human_review: bool = True
    human_review: TrimHumanReview | None = None
    proposal_path: str
    catalog_path: str

    @model_validator(mode="after")
    def validate_derived_bounds(self) -> "TrimIntentDecision":
        required = [
            self.first_included_frame,
            self.source_in_ms,
            self.source_out_ms,
            self.source_in_pts,
            self.source_out_pts,
        ]
        if self.usable:
            if any(value is None for value in required):
                raise ValueError("usable trim decisions require derived in/out evidence")
            assert self.source_in_ms is not None and self.source_out_ms is not None
            if self.source_out_ms <= self.source_in_ms:
                raise ValueError("trim decision must be a non-empty half-open interval")
            if self.handle_in_ms is None or self.handle_out_ms is None:
                raise ValueError("usable trim decisions require adjacent handles")
            if not self.handle_in_ms <= self.source_in_ms < self.source_out_ms <= self.handle_out_ms:
                raise ValueError("trim bounds must remain inside saved handles")
        elif any(value is not None for value in required):
            raise ValueError("unusable trim decisions cannot contain derived bounds")
        if self.approval_status == "proposed":
            if not self.requires_human_review or self.human_review is not None:
                raise ValueError("proposed trim decisions must remain unreviewed")
        else:
            if self.requires_human_review or self.human_review is None:
                raise ValueError("reviewed trim decisions require a human review record")
            if self.human_review.decision != self.approval_status:
                raise ValueError("human review decision must match approval status")
        return self


class DirectMoment(StrictModel):
    """A salient screenshot request using Gemini's documented MM:SS notation."""

    moment_id: str = Field(min_length=1)
    timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    label: str
    observable_evidence: str
    grounding_target_id: str = Field(min_length=1)
    grounding_target_description: str
    confidence: Confidence


class DirectMomentMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    moments: list[DirectMoment]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_moments(self) -> "DirectMomentMap":
        ids = [moment.moment_id for moment in self.moments]
        if len(ids) != len(set(ids)):
            raise ValueError("moment_id values must be unique")
        previous_ms = -1
        for moment in self.moments:
            minutes, seconds = (int(part) for part in moment.timestamp_mmss.split(":"))
            timestamp_ms = (minutes * 60 + seconds) * 1000
            if timestamp_ms >= self.duration_ms:
                raise ValueError(
                    f"moment {moment.moment_id} timestamp {moment.timestamp_mmss} exceeds duration"
                )
            if timestamp_ms <= previous_ms:
                raise ValueError("moment timestamps must be strictly increasing")
            previous_ms = timestamp_ms
        return self


class TargetCandidate(StrictModel):
    """A user-selectable object proposal; this stage deliberately has no bbox."""

    candidate_id: str = Field(min_length=1)
    label: str
    entity_kind: EntityKind
    target_description: str
    distinguishing_features: str
    representative_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    selection_reason: str
    confidence: Confidence


class TargetCandidateMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    candidates: list[TargetCandidate]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_candidates(self) -> "TargetCandidateMap":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if not ids:
            raise ValueError("at least one target candidate is required")
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        for candidate in self.candidates:
            minutes, seconds = (
                int(part) for part in candidate.representative_timestamp_mmss.split(":")
            )
            timestamp_ms = (minutes * 60 + seconds) * 1000
            if timestamp_ms >= self.duration_ms:
                raise ValueError(
                    f"candidate {candidate.candidate_id} representative timestamp exceeds duration"
                )
        return self


class GroundingCandidate(StrictModel):
    box_2d: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    label: str
    confidence: Confidence
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_box(self) -> "GroundingCandidate":
        x_min, y_min, x_max, y_max = self.box_2d
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("box_2d must satisfy x_min < x_max and y_min < y_max")
        return self


class GeminiNativeGroundingCandidate(StrictModel):
    """API-boundary bbox using Gemini's documented y-first coordinate order."""

    box_2d_yxyx: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    label: str
    confidence: Confidence
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_box(self) -> "GeminiNativeGroundingCandidate":
        y_min, x_min, y_max, x_max = self.box_2d_yxyx
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("box_2d_yxyx must satisfy ymin < ymax and xmin < xmax")
        return self


class GeminiNativeSegmentationCandidate(StrictModel):
    """Gemini single-image bbox plus polygon mask in documented native orders."""

    box_2d_yxyx: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    mask: list[tuple[NormalizedCoordinate, NormalizedCoordinate]]
    label: str
    confidence: Confidence
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_geometry(self) -> "GeminiNativeSegmentationCandidate":
        y_min, x_min, y_max, x_max = self.box_2d_yxyx
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("box_2d_yxyx must satisfy ymin < ymax and xmin < xmax")
        if len(self.mask) < 3:
            raise ValueError("segmentation mask polygon must contain at least three points")
        twice_area = abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(self.mask, self.mask[1:] + self.mask[:1])
            )
        )
        if _polygon_has_proper_self_intersection(self.mask):
            raise ValueError("segmentation mask polygon must not self-intersect")
        if twice_area == 0:
            raise ValueError("segmentation mask polygon must have non-zero area")
        xs = [point[0] for point in self.mask]
        ys = [point[1] for point in self.mask]
        tolerance = 5
        if (
            min(xs) < x_min - tolerance
            or max(xs) > x_max + tolerance
            or min(ys) < y_min - tolerance
            or max(ys) > y_max + tolerance
        ):
            raise ValueError("segmentation mask polygon must remain inside its bounding box")
        return self


class GroundingProposal(StrictModel):
    asset_id: str
    event_id: str
    entity_id: str
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    visible: bool
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
        return self


class GeminiNativeGroundingProposal(StrictModel):
    """Structured API response converted locally into GroundingProposal."""

    asset_id: str
    event_id: str
    entity_id: str
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    visible: bool
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeGroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeGroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
        return self


class GeminiNativeSegmentationProposal(StrictModel):
    """Structured Gemini single-frame segmentation response; polygons remain x/y ordered."""

    asset_id: str
    event_id: str
    entity_id: str
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    visible: bool
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeSegmentationCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeSegmentationProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
        return self


class DirectVideoGroundingProposal(StrictModel):
    """Experimental video-input bbox whose exact sampled source frame is unknowable locally."""

    asset_id: str
    event_id: str
    entity_id: str
    requested_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    reference_frame_status: Literal["unknown_gemini_video_sample"]
    reference_frame_description: str
    visible: bool
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "DirectVideoGroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
        return self


class GeminiNativeDirectVideoGroundingProposal(StrictModel):
    asset_id: str
    event_id: str
    entity_id: str
    requested_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    reference_frame_status: Literal["unknown_gemini_video_sample"]
    reference_frame_description: str
    visible: bool
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeGroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeDirectVideoGroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
        return self


class Rational(StrictModel):
    numerator: int
    denominator: int = Field(gt=0)


class VideoStreamInfo(StrictModel):
    index: int
    codec_name: str | None
    coded_width: int
    coded_height: int
    display_width: int
    display_height: int
    rotation_degrees: int
    sample_aspect_ratio: Rational = Field(
        default_factory=lambda: Rational(numerator=1, denominator=1)
    )
    display_sample_aspect_ratio: Rational = Field(
        default_factory=lambda: Rational(numerator=1, denominator=1)
    )
    average_frame_rate: Rational | None
    real_frame_rate: Rational | None
    time_base: Rational
    start_pts: int | None
    duration_ts: int | None
    metadata: dict[str, str]


class MediaInfo(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_id: str
    format_name: str | None
    duration_ms: int
    size_bytes: int
    format_metadata: dict[str, str]
    video: VideoStreamInfo


class ExtractedFrame(StrictModel):
    path: str
    requested_time_ms: int
    frame_time_ms: int
    frame_pts: int
    frame_hash: str
    width: int
    height: int


class RunStatus(StrictModel):
    run_id: str
    stage: str
    ok: bool
    errors: list[dict[str, object]] = Field(default_factory=list)


class FeatureCutExecutionProfile(StrEnum):
    """How strictly feature-cut prerequisites are enforced before rendering."""

    REVIEW_PREVIEW = "review_preview"
    PRODUCTION_REVIEW = "production_review"
    AUTONOMOUS_STRICT = "autonomous_strict"
    AUTONOMOUS_BEST_EFFORT = "autonomous_best_effort"


class FeatureCutRunState(StrEnum):
    """Editorial state; independent from whether an MP4 was encoded."""

    FAILED = "failed"
    PARTIAL = "partial"
    REVIEW_PREVIEW = "review_preview"
    READY_FOR_HUMAN_REVIEW = "ready_for_human_review"
    DELIVERY_ELIGIBLE = "delivery_eligible"
    BEST_EFFORT_COMPLETE = "best_effort_complete"


class EligibilityGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    NOT_REQUIRED = "not_required"


class FeatureCutEditorialContract(StrictModel):
    evidence_complete: EligibilityGateStatus
    candidate_recall_complete: EligibilityGateStatus
    candidate_resolution_passed: EligibilityGateStatus
    quality_coverage_complete: EligibilityGateStatus
    geometry_execution_passed: EligibilityGateStatus
    human_intent_execution_verified: EligibilityGateStatus
    technical_quality_passed: EligibilityGateStatus
    final_sequence_qa_passed: EligibilityGateStatus
    human_approval_passed: EligibilityGateStatus


class FeatureCutEligibilityReport(StrictModel):
    """Machine-readable handoff state for one feature-cut review render."""

    contract_version: Literal["feature-cut-delivery-eligibility-v1"] = (
        "feature-cut-delivery-eligibility-v1"
    )
    execution_profile: FeatureCutExecutionProfile
    media_rendered: bool
    run_state: FeatureCutRunState
    ready_for_human_review: bool
    delivery_eligible: bool
    editorial_contract: FeatureCutEditorialContract
    blocking_reasons: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    generated_at: str

    @model_validator(mode="after")
    def validate_state(self) -> "FeatureCutEligibilityReport":
        if self.delivery_eligible:
            if self.run_state != FeatureCutRunState.DELIVERY_ELIGIBLE:
                raise ValueError(
                    "delivery eligibility requires delivery-eligible run state"
                )
            if not self.ready_for_human_review:
                raise ValueError(
                    "delivery-eligible output must also be review-ready"
                )
        if self.ready_for_human_review and self.run_state not in {
            FeatureCutRunState.READY_FOR_HUMAN_REVIEW,
            FeatureCutRunState.DELIVERY_ELIGIBLE,
            FeatureCutRunState.BEST_EFFORT_COMPLETE,
        }:
            raise ValueError(
                "review-ready flag conflicts with the feature-cut run state"
            )
        if not self.media_rendered and self.run_state != FeatureCutRunState.FAILED:
            raise ValueError("a non-rendered run must remain failed")
        return self


class TrackingState(StrEnum):
    """Geometry state; it must not be mistaken for semantic identity confidence."""

    TRACKED = "tracked"
    REACQUIRED = "reacquired"
    OCCLUDED = "occluded"
    LOW_CONFIDENCE = "low_confidence"
    DRIFT_SUSPECTED = "drift_suspected"
    LOST = "lost"


class SemanticIdentityStatus(StrEnum):
    SEED_GROUNDED = "seed_grounded"
    NOT_REVALIDATED = "not_revalidated"
    REVALIDATION_REQUIRED = "revalidation_required"
    REVALIDATION_FAILED = "revalidation_failed"


class SegmentationModelProvenance(StrictModel):
    model_id: str
    implementation: str
    implementation_revision: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: str
    torch_version: str
    generated_at: str


class SegmentationSample(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int | None = None
    timing_basis: Literal[
        "decoded_source_pts",
        "uniform_ffmpeg_analysis_sample",
    ]
    mask_path: str | None
    mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mask_area_pixels: int = Field(ge=0)
    mask_area_ratio: float = Field(ge=0.0, le=1.0)
    connected_components: int = Field(ge=0)
    derived_tracking_box: list[NormalizedCoordinate] | None
    center_2d: list[float] | None
    mean_positive_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    scene_cut_score: float | None = Field(default=None, ge=0.0, le=100.0)
    shot_boundary: bool
    tracking_state: TrackingState
    state_reasons: list[str]
    semantic_identity_status: SemanticIdentityStatus

    @model_validator(mode="after")
    def validate_mask_geometry(self) -> "SegmentationSample":
        if self.derived_tracking_box is not None:
            if len(self.derived_tracking_box) != 4:
                raise ValueError("derived_tracking_box must contain four coordinates")
            x_min, y_min, x_max, y_max = self.derived_tracking_box
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("derived_tracking_box must satisfy xmin < xmax and ymin < ymax")
        if self.mask_area_pixels == 0:
            if self.mask_path is not None or self.mask_sha256 is not None:
                raise ValueError("empty masks cannot reference a mask artifact")
            if self.derived_tracking_box is not None or self.center_2d is not None:
                raise ValueError("empty masks cannot contain geometry")
            if self.tracking_state != TrackingState.LOST:
                raise ValueError("empty masks must use tracking_state=lost")
        elif self.mask_path is None or self.mask_sha256 is None:
            raise ValueError("non-empty masks must reference a hashed mask artifact")
        return self


class SegmentationTrack(StrictModel):
    method: Literal[
        "bbox_seed_sam2_video_mask_propagation",
        "gemini_bbox_seed_sam2_video_mask_propagation",
        "gemini_polygon_seed_sam2_video_mask_propagation",
    ]
    asset_id: str
    video_path: str
    target_description: str
    seed_source: str
    seed_time_ms: int = Field(ge=0)
    seed_sample_index: int = Field(ge=0)
    seed_frame_pts: int | None = None
    seed_frame_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    seed_source_width: int | None = Field(default=None, gt=0)
    seed_source_height: int | None = Field(default=None, gt=0)
    semantic_seed_box: list[NormalizedCoordinate]
    seed_prompt_type: Literal["box", "mask_polygon"] = "box"
    sam_prompt_box: list[NormalizedCoordinate] | None
    sam_prompt_mask_polygon_xy: list[
        tuple[NormalizedCoordinate, NormalizedCoordinate]
    ] | None = None
    seed_box_padding_ratio: float = Field(ge=0.0, le=1.0)
    refined_seed_mask_path: str
    analysis_fps: float = Field(gt=0, le=60)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    analysis_start_ms: int = Field(default=0, ge=0)
    analysis_end_ms: int | None = Field(default=None, gt=0)
    source_start_pts: int | None = None
    source_time_base: Rational | None = None
    timing_warning: str
    semantic_warning: str
    total_samples: int = Field(gt=0)
    state_counts: dict[TrackingState, int]
    elapsed_seconds: float = Field(ge=0)
    effective_fps: float = Field(ge=0)
    model_provenance: SegmentationModelProvenance
    samples: list[SegmentationSample]
    target_id: str | None = Field(
        default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    shared_session_id: str | None = Field(default=None, min_length=1)
    analysis_frames_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_track(self) -> "SegmentationTrack":
        if len(self.semantic_seed_box) != 4:
            raise ValueError("semantic_seed_box must contain four coordinates")
        x_min, y_min, x_max, y_max = self.semantic_seed_box
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("semantic_seed_box must satisfy xmin < xmax and ymin < ymax")
        if self.seed_prompt_type == "box":
            if self.sam_prompt_box is None or len(self.sam_prompt_box) != 4:
                raise ValueError("box prompts require four sam_prompt_box coordinates")
            prompt_x_min, prompt_y_min, prompt_x_max, prompt_y_max = self.sam_prompt_box
            if prompt_x_min >= prompt_x_max or prompt_y_min >= prompt_y_max:
                raise ValueError("sam_prompt_box must satisfy xmin < xmax and ymin < ymax")
            if self.sam_prompt_mask_polygon_xy is not None:
                raise ValueError("box prompts cannot contain a mask polygon")
        else:
            if self.sam_prompt_box is not None:
                raise ValueError("mask polygon prompts cannot contain sam_prompt_box")
            if (
                self.sam_prompt_mask_polygon_xy is None
                or len(self.sam_prompt_mask_polygon_xy) < 3
            ):
                raise ValueError("mask polygon prompts require at least three points")
        if self.total_samples != len(self.samples):
            raise ValueError("total_samples must equal len(samples)")
        if sum(self.state_counts.values()) != self.total_samples:
            raise ValueError("state_counts must cover every sample")
        observed_state_counts: dict[TrackingState, int] = {}
        for sample in self.samples:
            observed_state_counts[sample.tracking_state] = (
                observed_state_counts.get(sample.tracking_state, 0) + 1
            )
        declared_state_counts = {
            state: count for state, count in self.state_counts.items() if count != 0
        }
        if declared_state_counts != observed_state_counts:
            raise ValueError("state_counts must match sample tracking_state values")
        if self.seed_sample_index >= self.total_samples:
            raise ValueError("seed_sample_index is outside sampled frames")
        if [sample.sample_index for sample in self.samples] != list(
            range(len(self.samples))
        ):
            raise ValueError("sample indexes must be contiguous from zero")
        sample_times = [sample.analysis_sample_time_ms for sample in self.samples]
        if sample_times != sorted(set(sample_times)):
            raise ValueError("sample times must be strictly increasing")
        timing_bases = {sample.timing_basis for sample in self.samples}
        if len(timing_bases) != 1:
            raise ValueError("all samples in one track must use the same timing basis")
        decoded_pts_timing = timing_bases == {"decoded_source_pts"}
        if decoded_pts_timing:
            if any(sample.source_pts is None for sample in self.samples):
                raise ValueError("decoded-source-PTS samples require source_pts")
            sample_pts = [sample.source_pts for sample in self.samples]
            if sample_pts != sorted(set(sample_pts)):
                raise ValueError("sample source PTS values must be strictly increasing")
        seed_lineage = (
            self.seed_frame_pts,
            self.seed_frame_sha256,
            self.seed_source_width,
            self.seed_source_height,
        )
        if any(value is not None for value in seed_lineage) and not all(
            value is not None for value in seed_lineage
        ):
            raise ValueError("seed frame lineage fields must be provided together")
        shared_identity = (
            self.target_id,
            self.shared_session_id,
            self.analysis_frames_manifest_sha256,
        )
        if any(value is not None for value in shared_identity) and not all(
            value is not None for value in shared_identity
        ):
            raise ValueError("shared track identity fields must be provided together")
        if self.seed_frame_pts is not None:
            seed_sample = self.samples[self.seed_sample_index]
            if seed_sample.source_pts != self.seed_frame_pts:
                raise ValueError("seed frame source PTS must match the seed sample")
        timing_lineage = (self.source_start_pts, self.source_time_base)
        if any(value is not None for value in timing_lineage) and not all(
            value is not None for value in timing_lineage
        ):
            raise ValueError("source timing lineage fields must be provided together")
        if (
            decoded_pts_timing
            and self.source_start_pts is not None
            and self.source_time_base is not None
        ):
            if any(
                not _half_open_ms_matches_pts(
                    sample.analysis_sample_time_ms,
                    _local_ms_from_pts(
                        sample.source_pts,  # type: ignore[arg-type]
                        self.source_start_pts,
                        self.source_time_base,
                    ),
                    self.analysis_end_ms,
                )
                for sample in self.samples
            ):
                raise ValueError("sample times must be derived from source PTS lineage")
        if self.analysis_end_ms is not None:
            if self.analysis_end_ms <= self.analysis_start_ms:
                raise ValueError("analysis interval must be non-empty and half-open")
            if not self.analysis_start_ms <= self.seed_time_ms < self.analysis_end_ms:
                raise ValueError("seed_time_ms must be inside the analysis interval")
            if any(
                not self.analysis_start_ms
                <= sample.analysis_sample_time_ms
                < self.analysis_end_ms
                for sample in self.samples
            ):
                raise ValueError("tracking samples must remain inside the analysis interval")
        return self


class SharedSam21BBoxSeed(StrictModel):
    """One semantic instance seed for a shared SAM 2.1 video session."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    target_description: str = Field(min_length=1)
    seed_source: str = Field(min_length=1)
    seed_time_ms: int = Field(ge=0)
    seed_frame_pts: int
    seed_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_source_width: int = Field(gt=0)
    seed_source_height: int = Field(gt=0)
    seed_box_2d: list[NormalizedCoordinate]

    @model_validator(mode="after")
    def validate_seed_box(self) -> "SharedSam21BBoxSeed":
        if len(self.seed_box_2d) != 4:
            raise ValueError("seed_box_2d must contain four coordinates")
        x_min, y_min, x_max, y_max = self.seed_box_2d
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("seed_box_2d must satisfy xmin < xmax and ymin < ymax")
        return self


class SharedSam21TrackingRequest(StrictModel):
    """BBox-only request; every target must resolve to one shared shot interval."""

    asset_id: str = Field(min_length=1)
    targets: list[SharedSam21BBoxSeed] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_targets(self) -> "SharedSam21TrackingRequest":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("shared SAM target_id values must be unique")
        return self


class SharedSam21AnalysisFrame(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SharedSam21AnalysisFramesManifest(StrictModel):
    """Immutable decoded-frame lineage shared by every track in one session."""

    timing_basis: Literal["decoded_source_pts"]
    frames: list[SharedSam21AnalysisFrame] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_frames(self) -> "SharedSam21AnalysisFramesManifest":
        expected_indexes = list(range(len(self.frames)))
        if [frame.sample_index for frame in self.frames] != expected_indexes:
            raise ValueError("analysis frame sample indexes must be contiguous from zero")
        times = [frame.analysis_sample_time_ms for frame in self.frames]
        if times != sorted(set(times)):
            raise ValueError("analysis frame times must be strictly increasing")
        source_pts = [frame.source_pts for frame in self.frames]
        if source_pts != sorted(set(source_pts)):
            raise ValueError("analysis frame source PTS values must be strictly increasing")
        return self


class SharedSam21SessionTiming(StrictModel):
    shot_detection_seconds: float = Field(ge=0)
    analysis_frame_extraction_seconds: float = Field(ge=0)
    predictor_initialization_seconds: float = Field(ge=0)
    prompt_seconds: float = Field(ge=0)
    forward_propagation_seconds: float = Field(ge=0)
    reverse_propagation_seconds: float = Field(ge=0)
    target_artifact_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)


class SharedSam21SessionTarget(StrictModel):
    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    target_description: str = Field(min_length=1)
    seed_time_ms: int = Field(ge=0)
    seed_sample_index: int = Field(ge=0)
    seed_frame_pts: int
    seed_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_source_width: int = Field(gt=0)
    seed_source_height: int = Field(gt=0)
    track_path: str = Field(min_length=1)
    track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_counts: dict[TrackingState, int]


class SharedSam21SessionManifest(StrictModel):
    """Auditable batch record for tracks sharing decode, backbone, and state."""

    artifact_type: Literal["shared_sam21_multi_object_tracking_session"]
    method: Literal["bbox_seed_shared_sam2_video_mask_propagation"]
    session_id: str = Field(min_length=1)
    asset_id: str
    video_path: str
    shot_id: str = Field(min_length=1)
    analysis_fps: float = Field(gt=0, le=60)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    analysis_start_ms: int = Field(ge=0)
    analysis_end_ms: int = Field(gt=0)
    source_start_pts: int
    source_time_base: Rational
    analysis_frames_path: str = Field(min_length=1)
    analysis_frames_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_frames: list[SharedSam21AnalysisFrame] = Field(min_length=1)
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    target_count: int = Field(ge=2)
    targets: list[SharedSam21SessionTarget] = Field(min_length=2)
    model_provenance: SegmentationModelProvenance
    timing: SharedSam21SessionTiming
    warning: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_session(self) -> "SharedSam21SessionManifest":
        if self.analysis_start_ms >= self.analysis_end_ms:
            raise ValueError("analysis interval must be non-empty and half-open")
        if self.target_count != len(self.targets):
            raise ValueError("target_count must equal len(targets)")
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("shared session target_id values must be unique")
        expected_indexes = list(range(len(self.analysis_frames)))
        if [frame.sample_index for frame in self.analysis_frames] != expected_indexes:
            raise ValueError("analysis frame sample indexes must be contiguous from zero")
        times = [frame.analysis_sample_time_ms for frame in self.analysis_frames]
        if times != sorted(set(times)):
            raise ValueError("analysis frame times must be strictly increasing")
        source_pts = [frame.source_pts for frame in self.analysis_frames]
        if source_pts != sorted(set(source_pts)):
            raise ValueError("analysis frame source PTS values must be strictly increasing")
        if any(
            not _half_open_ms_matches_pts(
                frame.analysis_sample_time_ms,
                _local_ms_from_pts(
                    frame.source_pts,
                    self.source_start_pts,
                    self.source_time_base,
                ),
                self.analysis_end_ms,
            )
            for frame in self.analysis_frames
        ):
            raise ValueError("analysis frame times must be derived from source PTS lineage")
        if any(
            not self.analysis_start_ms <= time_ms < self.analysis_end_ms
            for time_ms in times
        ):
            raise ValueError("analysis frames must remain inside the shared interval")
        if any(target.seed_sample_index >= len(self.analysis_frames) for target in self.targets):
            raise ValueError("target seed_sample_index is outside analysis frames")
        for target in self.targets:
            seed_frame = self.analysis_frames[target.seed_sample_index]
            if target.seed_frame_pts != seed_frame.source_pts:
                raise ValueError("target seed frame PTS must match analysis frame lineage")
            if target.seed_time_ms != seed_frame.analysis_sample_time_ms:
                raise ValueError("target seed time must match analysis frame lineage")
            if any(count < 0 for count in target.state_counts.values()):
                raise ValueError("target state counts cannot be negative")
            if sum(target.state_counts.values()) != len(self.analysis_frames):
                raise ValueError("target state counts must cover every analysis frame")
        return self


class MultiSegmentationReviewMember(StrictModel):
    """One per-target track shown in a synchronized review video."""

    label: str = Field(min_length=1)
    color_rgb: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
    ]
    target_description: str = Field(min_length=1)
    track_json_path: str
    track_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_time_ms: int = Field(ge=0)


class MultiSegmentationReviewManifest(StrictModel):
    """Provenance for a synchronized multi-track manual-review visualization."""

    artifact_type: Literal["multi_segmentation_track_review"]
    interpretation: Literal["manual_review_visualization_not_accuracy"]
    asset_id: str
    source_video_path: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_fps: float = Field(gt=0, le=60)
    display_fps: float = Field(gt=0, le=60)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    analysis_start_ms: int = Field(ge=0)
    analysis_end_ms: int = Field(gt=0)
    total_samples: int = Field(gt=0)
    analysis_frames_dir: str
    analysis_frames_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    audio_muxed: bool
    output_video_path: str
    output_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_duration_ms: int = Field(gt=0)
    output_video_duration_ms: int = Field(gt=0)
    output_frame_count: int = Field(gt=0)
    output_codec_name: Literal["h264"]
    output_pixel_format: Literal["yuv420p"]
    output_frame_rate: Rational
    warning: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    members: list[MultiSegmentationReviewMember] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_review_manifest(self) -> "MultiSegmentationReviewManifest":
        if self.analysis_start_ms >= self.analysis_end_ms:
            raise ValueError("analysis interval must be non-empty and half-open")
        labels = [member.label for member in self.members]
        if len(labels) != len(set(labels)):
            raise ValueError("multi-track review labels must be unique")
        colors = [member.color_rgb for member in self.members]
        if len(colors) != len(set(colors)):
            raise ValueError("multi-track review colors must be unique")
        return self


class SegmentationTrackAgreementSample(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int
    tracking_state_a: TrackingState
    tracking_state_b: TrackingState
    state_agreement: bool
    mask_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    center_distance_normalized: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_agreement(self) -> "SegmentationTrackAgreementSample":
        if self.state_agreement != (self.tracking_state_a == self.tracking_state_b):
            raise ValueError("state_agreement must reflect the two tracking states")
        if (self.bbox_iou is None) != (self.center_distance_normalized is None):
            raise ValueError("bbox IoU and center distance must have identical coverage")
        return self


class SegmentationTrackAgreementReport(StrictModel):
    """Symmetric agreement metrics for two exactly aligned segmentation tracks."""

    artifact_type: Literal["segmentation_track_agreement_report"]
    interpretation: Literal["peer_agreement_not_accuracy"]
    asset_id: str
    track_a_path: str
    track_a_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_a_target_description: str = Field(min_length=1)
    track_b_path: str
    track_b_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_b_target_description: str = Field(min_length=1)
    total_samples: int = Field(gt=0)
    mask_iou_samples: int = Field(ge=0)
    mean_mask_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_iou_samples: int = Field(ge=0)
    mean_bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    center_distance_samples: int = Field(ge=0)
    mean_center_distance_normalized: float | None = Field(default=None, ge=0.0)
    state_agreement_samples: int = Field(ge=0)
    state_agreement_rate: float = Field(ge=0.0, le=1.0)
    warning: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    samples: list[SegmentationTrackAgreementSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary(self) -> "SegmentationTrackAgreementReport":
        if self.track_a_path == self.track_b_path:
            raise ValueError("track A and track B paths must be different")
        if self.total_samples != len(self.samples):
            raise ValueError("total_samples must equal len(samples)")
        if [sample.sample_index for sample in self.samples] != list(
            range(len(self.samples))
        ):
            raise ValueError("agreement sample indexes must be contiguous from zero")
        sample_times = [sample.analysis_sample_time_ms for sample in self.samples]
        if sample_times != sorted(set(sample_times)):
            raise ValueError("agreement sample times must be strictly increasing")
        sample_pts = [sample.source_pts for sample in self.samples]
        if sample_pts != sorted(set(sample_pts)):
            raise ValueError("agreement sample source PTS values must be strictly increasing")
        mask_count = sum(sample.mask_iou is not None for sample in self.samples)
        bbox_count = sum(sample.bbox_iou is not None for sample in self.samples)
        center_count = sum(
            sample.center_distance_normalized is not None for sample in self.samples
        )
        state_count = sum(sample.state_agreement for sample in self.samples)
        if self.mask_iou_samples != mask_count:
            raise ValueError("mask_iou_samples does not match sample coverage")
        if self.bbox_iou_samples != bbox_count:
            raise ValueError("bbox_iou_samples does not match sample coverage")
        if self.center_distance_samples != center_count:
            raise ValueError("center_distance_samples does not match sample coverage")
        if self.state_agreement_samples != state_count:
            raise ValueError("state_agreement_samples does not match samples")
        if (self.mean_mask_iou is None) != (mask_count == 0):
            raise ValueError("mean_mask_iou must reflect mask metric coverage")
        if (self.mean_bbox_iou is None) != (bbox_count == 0):
            raise ValueError("mean_bbox_iou must reflect bbox metric coverage")
        if (self.mean_center_distance_normalized is None) != (center_count == 0):
            raise ValueError("mean center distance must reflect metric coverage")
        expected_mask_mean = (
            round(
                sum(
                    sample.mask_iou
                    for sample in self.samples
                    if sample.mask_iou is not None
                )
                / mask_count,
                6,
            )
            if mask_count
            else None
        )
        expected_bbox_mean = (
            round(
                sum(
                    sample.bbox_iou
                    for sample in self.samples
                    if sample.bbox_iou is not None
                )
                / bbox_count,
                6,
            )
            if bbox_count
            else None
        )
        expected_center_mean = (
            round(
                sum(
                    sample.center_distance_normalized
                    for sample in self.samples
                    if sample.center_distance_normalized is not None
                )
                / center_count,
                6,
            )
            if center_count
            else None
        )
        expected_state_rate = round(state_count / len(self.samples), 6)
        if self.mean_mask_iou != expected_mask_mean:
            raise ValueError("mean_mask_iou does not match samples")
        if self.mean_bbox_iou != expected_bbox_mean:
            raise ValueError("mean_bbox_iou does not match samples")
        if self.mean_center_distance_normalized != expected_center_mean:
            raise ValueError("mean center distance does not match samples")
        if self.state_agreement_rate != expected_state_rate:
            raise ValueError("state_agreement_rate does not match samples")
        return self


class TrackerAgreementSample(StrictModel):
    analysis_sample_time_ms: int = Field(ge=0)
    reference_time_ms: float = Field(ge=0)
    segmentation_box: list[NormalizedCoordinate]
    reference_box: list[NormalizedCoordinate]
    bbox_iou: float = Field(ge=0.0, le=1.0)
    center_distance_normalized: float = Field(ge=0.0)


class TrackerAgreementReport(StrictModel):
    interpretation: Literal["tracker_agreement_not_accuracy"]
    segmentation_path: str
    reference_path: str
    reference_method: str
    aligned_samples: int = Field(gt=0)
    mean_bbox_iou: float = Field(ge=0.0, le=1.0)
    min_bbox_iou: float = Field(ge=0.0, le=1.0)
    mean_center_distance_normalized: float = Field(ge=0.0)
    max_center_distance_normalized: float = Field(ge=0.0)
    warning: str
    samples: list[TrackerAgreementSample]


class RushClip(StrictModel):
    clip_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: str
    size_bytes: int = Field(gt=0)


class RushFrame(StrictModel):
    frame_id: str = Field(
        pattern=r"^RF[0-9]{6}$",
        min_length=8,
        max_length=8,
    )
    clip_id: str
    requested_time_ms: int = Field(ge=0)
    image_path: str
    source_image_path: str | None = None
    frame_time_ms: int | None = Field(default=None, ge=0)
    frame_pts: int | None = None
    frame_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_exact_frame_lineage(self) -> "RushFrame":
        exact = (
            self.source_image_path,
            self.frame_time_ms,
            self.frame_pts,
            self.frame_hash,
        )
        if any(value is not None for value in exact) and not all(
            value is not None for value in exact
        ):
            raise ValueError(
                "rush frame exact lineage requires time, PTS and hash together"
            )
        return self


class RushesCatalog(StrictModel):
    catalog_id: str
    source_directory: str
    sample_interval_ms: int = Field(ge=500)
    total_duration_ms: int = Field(gt=0)
    clips: list[RushClip]
    frames: list[RushFrame]
    analysis_reel_path: str
    generated_at: str


class RushesSelectShot(StrictModel):
    select_id: str = Field(min_length=1)
    representative_frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    suggested_duration_seconds: float = Field(ge=1.5, le=6.0)
    role: Literal["opening", "establishing", "product", "detail", "movement", "transition", "closing"]
    visual_description: str
    selection_reason: str
    quality_risks: list[str]
    vertical_focus: Literal["left", "center", "right"]
    confidence: Confidence


class RushesTimelinePlan(StrictModel):
    aspect_ratio: Literal["16:9", "9:16"]
    title: str
    editorial_intent: str
    shots: list[RushesSelectShot] = Field(min_length=1, max_length=16)


class RushesEditPlan(StrictModel):
    project_id: str
    catalog_id: str
    summary: str
    timelines: list[RushesTimelinePlan]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timelines(self) -> "RushesEditPlan":
        aspects = [timeline.aspect_ratio for timeline in self.timelines]
        if sorted(aspects) != ["16:9", "9:16"]:
            raise ValueError("timelines must contain exactly one 16:9 and one 9:16 plan")
        for timeline in self.timelines:
            ids = [shot.select_id for shot in timeline.shots]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate select_id in {timeline.aspect_ratio} timeline")
        return self


class FramingRegionIntent(StrictModel):
    """One domain-neutral visual region used to guide a reframe.

    A region may describe a person, animal, product, document, sign, UI area,
    or any other directly visible subject.  The vocabulary intentionally does
    not encode fixture-specific brands or object classes.
    """

    region_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    entity_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$",
        description=(
            "Immutable Clip Card entity reference when this region was resolved "
            "from catalog evidence. Legacy/manual regions may omit it."
        ),
    )
    target_description: str = Field(min_length=1)
    kind: Literal["subject", "text_region", "ui_region", "graphic", "other"] = (
        "subject"
    )
    evidence_role: Literal[
        "primary_subject",
        "relation_participant",
        "relation_carrier",
        "state_evidence",
        "context_reference",
    ] = "primary_subject"
    role: Literal["required", "preferred", "avoid_overlay"] = "required"
    atomic: bool = Field(
        default=False,
        description=(
            "True when partial clipping changes the meaning of the region, for "
            "example a text or UI state. Atomic regions are treated as hard cores."
        ),
    )
    minimum_visible_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    observable_relations: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def upgrade_relation_field(cls, value: Any) -> Any:
        if isinstance(value, dict) and "relation_constraints" in value:
            if "observable_relations" in value:
                raise ValueError("region cannot define both relation field versions")
            value = dict(value)
            value["observable_relations"] = value.pop("relation_constraints")
        return value

    @model_validator(mode="after")
    def validate_region_policy(self) -> "FramingRegionIntent":
        for field_name in ("observable_relations", "exclusions"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        if self.atomic and self.minimum_visible_fraction not in (None, 1.0):
            raise ValueError("atomic regions must be fully visible")
        if self.role == "avoid_overlay" and self.minimum_visible_fraction is not None:
            raise ValueError("avoid_overlay regions do not use a crop visible fraction")
        return self

    @property
    def execution_role(self) -> Literal["hard_core", "soft_extent", "overlay_keepout"]:
        if self.role == "avoid_overlay":
            return "overlay_keepout"
        if self.role == "required" or self.atomic:
            return "hard_core"
        return "soft_extent"

    @property
    def effective_minimum_visible_fraction(self) -> float:
        if self.atomic:
            return 1.0
        if self.execution_role == "overlay_keepout":
            return 0.0
        if self.role == "required":
            return (
                self.minimum_visible_fraction
                if self.minimum_visible_fraction is not None
                else 1.0
            )
        return self.minimum_visible_fraction if self.minimum_visible_fraction is not None else 0.72


VirtualCameraIntent = Literal[
    "hold",
    "follow",
    "punch_in_cut",
    "push_in",
    "pull_out",
    "pan_reveal",
    "recenter",
]


class AttentionObservation(StrictModel):
    """Gemini's reviewable editorial-attention vector, never a cut point."""

    semantic_novelty: float = Field(ge=0.0, le=1.0)
    action_progress: float = Field(ge=0.0, le=1.0)
    visual_motion: float = Field(ge=0.0, le=1.0)
    composition_change: float = Field(ge=0.0, le=1.0)
    reading_load: float = Field(ge=0.0, le=1.0)
    unresolved_tension: float = Field(ge=0.0, le=1.0)
    emotional_hold_value: float = Field(ge=0.0, le=1.0)
    repetition_pressure: float = Field(ge=0.0, le=1.0)
    music_transition_opportunity: float = Field(ge=0.0, le=1.0)
    minimum_dwell_seconds: float = Field(ge=0.5, le=15.0)
    maximum_dwell_seconds: float = Field(ge=0.5, le=15.0)
    rationale: str = Field(min_length=1, max_length=800)
    uncertainties: list[str] = Field(default_factory=list, max_length=8)
    requires_human_review: Literal[True] = True

    @model_validator(mode="after")
    def validate_dwell_bounds(self) -> "AttentionObservation":
        if self.maximum_dwell_seconds < self.minimum_dwell_seconds:
            raise ValueError("attention dwell bounds must satisfy minimum <= maximum")
        return self


class ShotFlowIntent(StrictModel):
    """Gemini's semantic sequence intent; never an EDL or timing curve."""

    narrative_role: Literal[
        "hook",
        "setup",
        "development",
        "contrast",
        "proof",
        "payoff",
        "breath",
        "resolution",
    ]
    energy_role: Literal[
        "low_hold",
        "rise",
        "peak",
        "release",
        "reset",
    ]
    relation_to_previous: Literal[
        "start",
        "new_context",
        "continue_action",
        "answer",
        "reaction",
        "contrast",
        "reveal",
        "reset",
    ]
    boundary_alignment: Literal[
        "content_locked",
        "phrase_preferred",
        "accent_preferred",
        "free",
    ]
    visual_sync_event: Literal[
        "action_apex",
        "reveal",
        "result_state",
        "gesture",
        "ui_change",
        "intentional_hold",
    ] | None = None
    visual_sync_predicate: str | None = Field(default=None, max_length=300)
    music_target: Literal[
        "phrase_start",
        "phrase_end",
        "downbeat",
        "accent",
        "section_change",
    ] | None = None

    @model_validator(mode="after")
    def validate_sync_event(self) -> "ShotFlowIntent":
        values = (
            self.visual_sync_event,
            self.visual_sync_predicate,
            self.music_target,
        )
        if any(value is not None for value in values) and any(
            value is None for value in values
        ):
            raise ValueError(
                "visual sync event, observable predicate, and music target "
                "must be supplied together"
            )
        if self.visual_sync_predicate is not None and not (
            self.visual_sync_predicate.strip()
        ):
            raise ValueError("visual sync predicate must be observable and non-empty")
        return self


class _SelectedFramingRegionBase(StrictModel):
    """Response-only region base with role-specific JSON schema branches."""

    region_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    entity_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$",
    )
    target_description: str = Field(min_length=1)
    kind: Literal["subject", "text_region", "ui_region", "graphic", "other"] = (
        "subject"
    )
    evidence_role: Literal[
        "primary_subject",
        "relation_participant",
        "relation_carrier",
        "state_evidence",
        "context_reference",
    ]
    observable_relations: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_text_lists(self) -> "_SelectedFramingRegionBase":
        for field_name in ("observable_relations", "exclusions"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        return self

    def to_framing_region_intent(self) -> "FramingRegionIntent":
        return FramingRegionIntent.model_validate(self.model_dump(mode="python"))


class SelectedHardCoreFramingRegion(_SelectedFramingRegionBase):
    role: Literal["required"]
    atomic: bool = False
    minimum_visible_fraction: Literal[1.0] | None = None

    @property
    def execution_role(self) -> Literal["hard_core"]:
        return "hard_core"


class SelectedSoftExtentFramingRegion(_SelectedFramingRegionBase):
    role: Literal["preferred"]
    atomic: Literal[False] = False
    minimum_visible_fraction: float | None = Field(default=None, gt=0.0, le=1.0)

    @property
    def execution_role(self) -> Literal["soft_extent"]:
        return "soft_extent"


class SelectedOverlayKeepoutFramingRegion(_SelectedFramingRegionBase):
    role: Literal["avoid_overlay"]
    atomic: Literal[False] = False
    minimum_visible_fraction: Literal[None] = None

    @property
    def execution_role(self) -> Literal["overlay_keepout"]:
        return "overlay_keepout"


SelectedFramingRegion = Annotated[
    SelectedHardCoreFramingRegion
    | SelectedSoftExtentFramingRegion
    | SelectedOverlayKeepoutFramingRegion,
    Field(discriminator="role"),
]


class VerticalPresentationOptionAssessment(StrictModel):
    """One evidence-based alternative considered before portrait fallback."""

    mode: Literal[
        "single_full_bleed_crop",
        "sequential_virtual_camera",
        "controlled_clipping",
        "fit_or_layout",
        "try_next_candidate",
    ]
    verdict: Literal["feasible", "not_feasible", "uncertain", "not_applicable"]
    observable_reason: str = Field(min_length=1, max_length=800)


class SequentialReconstructionContract(StrictModel):
    """Evidence needed to preserve meaning across separate portrait phases."""

    linkage_type: Literal[
        "joint_establishing_phase",
        "shared_tracked_anchor",
        "visible_transition",
        "ordered_state_change",
        "scale_locked_comparison",
    ]
    linkage_region_ids: list[str] = Field(min_length=1, max_length=4)
    preserve_scale: bool = False
    observable_reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_reconstruction(self) -> "SequentialReconstructionContract":
        if len(self.linkage_region_ids) != len(set(self.linkage_region_ids)):
            raise ValueError("sequential linkage region IDs must be unique")
        if (
            self.linkage_type == "scale_locked_comparison"
            and not self.preserve_scale
        ):
            raise ValueError(
                "scale-locked comparison reconstruction must preserve scale"
            )
        return self


class SelectedVerticalFramingProposal(StrictModel):
    """Full-clip semantic framing decision made after editorial selection.

    The proposal may change how one already-selected candidate is presented,
    but it cannot change the source asset, event, evidence frame, or candidate
    identity.  Exact coordinates and motion remain downstream local work.
    """

    contract_version: Literal["selected-vertical-framing-proposal-v3"] = (
        "selected-vertical-framing-proposal-v3"
    )
    candidate_id: str = Field(
        pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64
    )
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    event_id: str = Field(min_length=1)
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    semantic_requirement: Literal[
        "single_primary",
        "group_coverage",
        "sequential_attention",
        "simultaneous_relation",
    ]
    relation_temporal_mode: Literal[
        "not_applicable",
        "simultaneous_required",
        "sequentially_reconstructable",
        "phase_mixed",
        "uncertain",
    ]
    recommended_action: Literal[
        "tracked_crop",
        "fit_or_layout",
        "try_next_candidate",
    ]
    regions: list[SelectedFramingRegion] = Field(default_factory=list, max_length=8)
    virtual_camera_proposal: VerticalVirtualCameraProposal | None = None
    sequential_reconstruction: SequentialReconstructionContract | None = None
    presentation_options: list[VerticalPresentationOptionAssessment] = Field(
        min_length=1,
        max_length=5,
    )
    observed_evidence: list[str] = Field(min_length=1, max_length=12)
    decision_reason: str = Field(min_length=1, max_length=1200)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    confidence: Confidence
    model_provenance: ModelProvenance

    def _has_bound_atomic_relation_core(self, required_count: int = 2) -> bool:
        """Return whether one compound core is backed by bound participants.

        Some observable relations are spatially smaller than either complete
        participant (for example a contact point, an interface state next to
        the hand operating it, or two adjacent edges used for comparison).
        Tracking both complete participants can make a full-bleed portrait
        crop impossible even though one indivisible relation carrier preserves
        the evidence.  The carrier is only valid when it names an observable
        relation and the proposal separately binds enough participant entity
        IDs; prose alone is never sufficient.
        """

        carriers = [
            region
            for region in self.regions
            if (
                region.execution_role == "hard_core"
                and region.atomic
                and region.evidence_role == "relation_carrier"
                and region.observable_relations
            )
        ]
        participant_entity_ids = {
            region.entity_id
            for region in self.regions
            if (
                region.evidence_role == "relation_participant"
                and region.entity_id is not None
            )
        }
        return bool(carriers) and len(participant_entity_ids) >= required_count

    @model_validator(mode="after")
    def validate_selected_framing(self) -> "SelectedVerticalFramingProposal":
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("selected framing region IDs must be unique")
        entity_ids = [
            region.entity_id for region in self.regions if region.entity_id is not None
        ]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("selected framing entity IDs must be unique")
        option_modes = [option.mode for option in self.presentation_options]
        if len(option_modes) != len(set(option_modes)):
            raise ValueError("portrait presentation option modes must be unique")
        option_by_mode = {
            option.mode: option for option in self.presentation_options
        }
        if "single_full_bleed_crop" not in option_by_mode:
            raise ValueError(
                "portrait framing must assess a single full-bleed crop"
            )
        multi_subject_requirement = self.semantic_requirement in {
            "group_coverage",
            "sequential_attention",
            "simultaneous_relation",
        }
        if multi_subject_requirement:
            missing_modes = sorted(
                {
                    "sequential_virtual_camera",
                    "controlled_clipping",
                    "fit_or_layout",
                }
                - set(option_by_mode)
            )
            if missing_modes:
                raise ValueError(
                    "multi-subject portrait framing must assess alternatives: "
                    + ", ".join(missing_modes)
                )
        if (
            self.semantic_requirement == "single_primary"
            and self.relation_temporal_mode != "not_applicable"
        ):
            raise ValueError(
                "single-primary framing cannot claim a temporal relation"
            )
        if (
            self.semantic_requirement == "sequential_attention"
            and self.relation_temporal_mode != "sequentially_reconstructable"
        ):
            raise ValueError(
                "sequential attention must be sequentially reconstructable"
            )
        if (
            self.semantic_requirement == "simultaneous_relation"
            and self.relation_temporal_mode == "not_applicable"
        ):
            raise ValueError(
                "a simultaneous relation must declare a strict, sequential, "
                "mixed, or uncertain temporal mode"
            )
        if (
            self.relation_temporal_mode == "phase_mixed"
            and not multi_subject_requirement
        ):
            raise ValueError(
                "phase-mixed temporal relations require a multi-subject framing"
            )
        if (
            self.relation_temporal_mode == "uncertain"
            and self.recommended_action == "tracked_crop"
        ):
            raise ValueError(
                "an uncertain temporal relation cannot authorize tracked crop"
            )
        if self.recommended_action == "fit_or_layout":
            fit_assessment = option_by_mode.get("fit_or_layout")
            if fit_assessment is None or fit_assessment.verdict != "feasible":
                raise ValueError(
                    "fit-or-layout action requires a feasible fit/layout assessment"
                )
            for mode in (
                "single_full_bleed_crop",
                "sequential_virtual_camera",
                "controlled_clipping",
            ):
                assessment = option_by_mode.get(mode)
                if assessment is not None and assessment.verdict == "feasible":
                    raise ValueError(
                        "fit-or-layout cannot bypass a feasible full-bleed "
                        f"presentation option: {mode}"
                    )
        if self.recommended_action == "tracked_crop" and not self.regions:
            raise ValueError("tracked-crop framing requires explicit regions")
        if self.recommended_action != "tracked_crop":
            # Some Structured Output responses redundantly include a camera
            # idea even after choosing fit/layout or another candidate.  The
            # action is authoritative: consumers preserve that surplus field
            # as evidence but must never execute it.
            return self
        if self.virtual_camera_proposal is None:
            raise ValueError(
                "tracked-crop framing requires an explicit hold, follow, or "
                "multi-phase virtual-camera proposal"
            )
        proposal = self.virtual_camera_proposal
        known_region_ids = set(region_ids)
        referenced = {
            region_id
            for phase in proposal.phases
            for region_id in phase.anchor_region_ids
        }
        unknown = sorted(referenced - known_region_ids)
        if unknown:
            raise ValueError(
                "selected framing proposal references unknown regions: "
                + ", ".join(unknown)
            )
        required = {
            region.region_id
            for region in self.regions
            if region.execution_role == "hard_core"
        }
        missing_required = sorted(required - referenced)
        if missing_required:
            raise ValueError(
                "every hard-core selected framing region must be referenced: "
                + ", ".join(missing_required)
            )
        if (
            self.semantic_requirement == "group_coverage"
            and len(required) < 2
            and not any(
                region.atomic
                for region in self.regions
                if region.execution_role == "hard_core"
            )
        ):
            raise ValueError(
                "group coverage requires at least two hard-core member regions "
                "or one explicitly atomic compound group region"
            )
        if (
            self.semantic_requirement == "group_coverage"
            and proposal.composition_mode
            not in {"sequential_focus", "joint_relation", "mixed_relation"}
            and not (
                len(required) == 1
                and any(
                    region.atomic
                    for region in self.regions
                    if region.execution_role == "hard_core"
                )
                and proposal.composition_mode == "single_anchor_hold"
            )
        ):
            raise ValueError(
                "group coverage requires sequential-focus, joint-relation, or "
                "mixed-relation phases, or one held atomic compound group"
            )
        if (
            self.semantic_requirement == "simultaneous_relation"
            and len(required) < 2
            and not self._has_bound_atomic_relation_core()
        ):
            raise ValueError(
                "a simultaneous relation requires at least two independently "
                "grounded hard-core participant regions, or one atomic relation "
                "carrier backed by at least two bound participant entities"
            )
        if (
            self.semantic_requirement == "sequential_attention"
            and proposal.composition_mode != "sequential_focus"
        ):
            raise ValueError(
                "sequential attention requires sequential-focus camera phases"
            )
        if (
            self.relation_temporal_mode == "sequentially_reconstructable"
            and multi_subject_requirement
            and proposal.composition_mode != "sequential_focus"
        ):
            raise ValueError(
                "a sequentially reconstructable multi-subject relation requires "
                "sequential-focus camera phases"
            )
        if (
            self.relation_temporal_mode == "phase_mixed"
            and proposal.composition_mode != "mixed_relation"
        ):
            raise ValueError(
                "a phase-mixed temporal relation requires mixed-relation camera "
                "phases"
            )
        if (
            self.relation_temporal_mode == "simultaneous_required"
            and proposal.composition_mode == "sequential_focus"
        ):
            raise ValueError(
                "a strictly simultaneous relation cannot use sequential focus"
            )
        if self.relation_temporal_mode in {
            "sequentially_reconstructable",
            "phase_mixed",
        }:
            reconstruction = self.sequential_reconstruction
            if reconstruction is None:
                raise ValueError(
                    "sequential or phase-mixed framing requires an explicit "
                    "reconstruction contract"
                )
            linkage_ids = set(reconstruction.linkage_region_ids)
            unknown_linkage = sorted(linkage_ids - known_region_ids)
            if unknown_linkage:
                raise ValueError(
                    "sequential reconstruction references unknown regions: "
                    + ", ".join(unknown_linkage)
                )
            unreferenced_linkage = sorted(linkage_ids - referenced)
            if unreferenced_linkage:
                raise ValueError(
                    "sequential reconstruction linkage regions must be used by "
                    "camera phases: "
                    + ", ".join(unreferenced_linkage)
                )
            if reconstruction.linkage_type == "joint_establishing_phase":
                if not any(
                    linkage_ids.issubset(set(phase.anchor_region_ids))
                    for phase in proposal.phases
                ):
                    raise ValueError(
                        "joint establishing reconstruction requires one phase "
                        "that contains every linkage region"
                    )
            if reconstruction.linkage_type == "shared_tracked_anchor":
                appearances = {
                    region_id: sum(
                        region_id in phase.anchor_region_ids
                        for phase in proposal.phases
                    )
                    for region_id in linkage_ids
                }
                if not any(count >= 2 for count in appearances.values()):
                    raise ValueError(
                        "shared-anchor reconstruction requires a linkage region "
                        "in at least two phases"
                    )
            if reconstruction.preserve_scale:
                scale_changing = [
                    phase.phase_id
                    for phase in proposal.phases
                    if phase.camera_behavior
                    in {"push_in", "pull_out", "punch_in_cut"}
                ]
                if scale_changing:
                    raise ValueError(
                        "scale-preserving reconstruction cannot use scale-changing "
                        f"phases: {scale_changing}"
                    )
        if (
            self.semantic_requirement == "simultaneous_relation"
            and proposal.composition_mode == "sequential_focus"
        ):
            scale_changing = [
                phase.phase_id
                for phase in proposal.phases
                if phase.camera_behavior
                in {"push_in", "pull_out", "punch_in_cut"}
            ]
            if scale_changing:
                raise ValueError(
                    "a sequential comparison may reconstruct a simultaneous "
                    "relation only with scale-preserving camera behaviors; "
                    f"scale-changing phases: {scale_changing}"
                )
        return self


class FeatureHorizontalCandidate(StrictModel):
    """One evidence-bound 16:9 option retained for local automatic routing."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    rank: int = Field(ge=1, le=4)
    source_asset_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    frame_id: str = Field(
        pattern=r"^RF[0-9]{6}$",
        min_length=8,
        max_length=8,
    )
    observed_visual_evidence: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    strategy: Literal["original", "tracked_reframe"]
    zoom_intent: Literal["none", "subtle", "detail"]
    camera_intent: VirtualCameraIntent = "hold"
    target_description: str | None = None
    quality_risks: list[str] = Field(default_factory=list)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_geometry_intent(self) -> "FeatureHorizontalCandidate":
        if self.strategy == "tracked_reframe":
            if self.zoom_intent == "none" or not self.target_description:
                raise ValueError("tracked_reframe candidate requires zoom intent and target")
        elif self.zoom_intent != "none":
            raise ValueError("original candidate must use zoom intent none")
        if self.strategy == "original" and self.camera_intent != "hold":
            raise ValueError("original candidate must use virtual-camera intent hold")
        return self


class FeatureVerticalCandidate(StrictModel):
    """One evidence-bound 9:16 option retained for geometry-first selection."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    rank: int = Field(ge=1, le=4)
    source_asset_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    strategy: Literal["tracked_crop", "fit_with_background"]
    crop_mode: Literal["strict", "primary_center"] = "strict"
    coverage_mode: Literal[
        "simultaneous",
        "sequential",
        "relation_core",
        "primary_with_context",
        "independent_detail",
    ] = "simultaneous"
    allow_controlled_clip: bool = False
    target_description: str | None = None
    regions: list[FramingRegionIntent] = Field(default_factory=list, max_length=8)
    virtual_camera_proposal: VerticalVirtualCameraProposal | None = None
    quality_risks: list[str] = Field(default_factory=list)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_geometry_intent(self) -> "FeatureVerticalCandidate":
        hard_regions = [
            region for region in self.regions if region.execution_role == "hard_core"
        ]
        if self.strategy == "tracked_crop" and not (
            self.target_description or hard_regions
        ):
            raise ValueError("tracked_crop candidate requires a target or hard-core region")
        if self.allow_controlled_clip and self.crop_mode != "primary_center":
            raise ValueError(
                "controlled clipping requires primary_center crop mode"
            )
        if self.strategy == "fit_with_background" and self.allow_controlled_clip:
            raise ValueError(
                "fit-with-background cannot request controlled clipping"
            )
        if (
            self.strategy == "tracked_crop"
            and self.crop_mode == "strict"
            and any(
                region.execution_role == "hard_core"
                and region.effective_minimum_visible_fraction < 1.0
                for region in self.regions
            )
        ):
            raise ValueError(
                "strict tracked crops require full visibility for every hard-core region"
            )
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate region IDs must be unique")
        entity_ids = [region.entity_id for region in self.regions if region.entity_id]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("candidate entity references must be unique")
        if self.virtual_camera_proposal is not None:
            if self.strategy != "tracked_crop":
                raise ValueError(
                    "vertical camera proposal requires tracked_crop geometry"
                )
            if not self.regions:
                raise ValueError(
                    "vertical camera proposal requires explicit framing regions"
                )
            known_region_ids = set(ids)
            referenced_region_ids = {
                region_id
                for phase in self.virtual_camera_proposal.phases
                for region_id in phase.anchor_region_ids
            }
            unknown = sorted(referenced_region_ids - known_region_ids)
            if unknown:
                raise ValueError(
                    "vertical camera proposal references unknown regions: "
                    + ", ".join(unknown)
                )
            keepout_ids = {
                region.region_id
                for region in self.regions
                if region.execution_role == "overlay_keepout"
            }
            invalid_keepouts = sorted(referenced_region_ids & keepout_ids)
            if invalid_keepouts:
                raise ValueError(
                    "overlay keepout regions cannot be virtual-camera anchors: "
                    + ", ".join(invalid_keepouts)
                )
            required_ids = {
                region.region_id
                for region in self.regions
                if region.execution_role == "hard_core"
            }
            missing_required = sorted(required_ids - referenced_region_ids)
            if missing_required:
                raise ValueError(
                    "every hard-core region must appear in the virtual-camera "
                    "proposal: " + ", ".join(missing_required)
                )
        return self


class ReframePolicyBinding(StrictModel):
    """Immutable human-policy provenance embedded in a revised edit brief.

    The sidecar is content addressed and binds the policy decision to the
    exact source brief, catalog, saved feature plan, and plan binding.  It is
    intentionally domain-neutral: the chapter overrides carry the visible
    region descriptions, while this record only establishes provenance.
    """

    binding_version: Literal["human-reframe-policy-binding-v1"]
    policy_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    sidecar_path: str = Field(min_length=1)
    sidecar_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_brief_path: str = Field(min_length=1)
    source_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_feature_plan_path: str = Field(min_length=1)
    source_feature_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_plan_binding_path: str = Field(min_length=1)
    source_plan_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_path: str = Field(min_length=1)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeatureChapterBrief(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    title: str
    detail_lines: list[str]
    target_duration_seconds: float = Field(ge=3.0, le=10.0)
    vertical_primary_target_description: str | None = None
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"
    vertical_regions: list[FramingRegionIntent] = Field(default_factory=list, max_length=4)
    vertical_camera_phases: list[VerticalVirtualCameraPhase] = Field(
        default_factory=list,
        max_length=8,
    )
    vertical_overflow_policy: Literal["preserve_all", "controlled_clip"] = (
        "preserve_all"
    )
    vertical_edge_priority: Literal[
        "balanced", "preserve_start", "preserve_end"
    ] = "balanced"

    @model_validator(mode="after")
    def validate_vertical_regions(self) -> "FeatureChapterBrief":
        ids = [region.region_id for region in self.vertical_regions]
        if len(ids) != len(set(ids)):
            raise ValueError("vertical region IDs must be unique within a chapter")
        if self.vertical_regions and not any(
            region.role == "required" for region in self.vertical_regions
        ):
            raise ValueError("vertical regions must include at least one required region")
        if (
            self.vertical_edge_priority != "balanced"
            and self.vertical_overflow_policy != "controlled_clip"
        ):
            raise ValueError(
                "edge priority only applies when vertical_overflow_policy is controlled_clip"
            )
        if self.vertical_camera_phases:
            if not self.vertical_regions:
                raise ValueError(
                    "vertical camera phases require explicit framing regions"
                )
            phase_ids = [
                phase.phase_id for phase in self.vertical_camera_phases
            ]
            if len(phase_ids) != len(set(phase_ids)):
                raise ValueError("vertical camera phase IDs must be unique")
            if abs(self.vertical_camera_phases[0].start_progress) > 1e-6:
                raise ValueError("vertical camera phases must start at progress zero")
            if abs(self.vertical_camera_phases[-1].end_progress - 1.0) > 1e-6:
                raise ValueError("vertical camera phases must end at progress one")
            for prior, current in zip(
                self.vertical_camera_phases[:-1],
                self.vertical_camera_phases[1:],
                strict=True,
            ):
                if abs(prior.end_progress - current.start_progress) > 1e-6:
                    raise ValueError(
                        "vertical camera phases must be contiguous and non-overlapping"
                    )
            if self.vertical_camera_phases[0].transition_in != "cut":
                raise ValueError(
                    "the first vertical camera phase cannot transition from an "
                    "unknown prior anchor"
                )
            known_region_ids = set(ids)
            referenced_region_ids = {
                region_id
                for phase in self.vertical_camera_phases
                for region_id in phase.anchor_region_ids
            }
            unknown = sorted(referenced_region_ids - known_region_ids)
            if unknown:
                raise ValueError(
                    "vertical camera phases reference unknown regions: "
                    + ", ".join(unknown)
                )
            keepout_ids = {
                region.region_id
                for region in self.vertical_regions
                if region.execution_role == "overlay_keepout"
            }
            invalid_keepouts = sorted(referenced_region_ids & keepout_ids)
            if invalid_keepouts:
                raise ValueError(
                    "overlay keepout regions cannot be camera anchors: "
                    + ", ".join(invalid_keepouts)
                )
            atomic_ids = {
                region.region_id
                for region in self.vertical_regions
                if region.atomic
            }
            for phase in self.vertical_camera_phases:
                if (
                    phase.minimum_anchor_visible_fraction < 1.0
                    and atomic_ids.intersection(phase.anchor_region_ids)
                ):
                    raise ValueError(
                        "camera phases cannot relax visibility for atomic "
                        "text or UI regions"
                    )
            required_ids = {
                region.region_id
                for region in self.vertical_regions
                if region.execution_role == "hard_core"
            }
            missing_required = sorted(required_ids - referenced_region_ids)
            if missing_required:
                raise ValueError(
                    "every hard-core region must be active in at least one camera "
                    "phase: "
                    + ", ".join(missing_required)
                )
        return self


class FeatureEditBrief(StrictModel):
    project_id: str
    title: str
    target_duration_seconds: float = Field(ge=60.0, le=90.0)
    render_title_overlays: bool = True
    vertical_fallback_strategy: Literal["fit_with_background", "center_crop"] = (
        "center_crop"
    )
    reframe_policy_binding: ReframePolicyBinding | None = None
    chapters: list[FeatureChapterBrief] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_chapters(self) -> "FeatureEditBrief":
        ids = [chapter.feature_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("feature brief chapter IDs must be unique")
        if any(chapter.vertical_camera_phases for chapter in self.chapters) and (
            self.reframe_policy_binding is None
        ):
            raise ValueError(
                "vertical camera phases require an immutable human reframe "
                "policy binding"
            )
        return self


class FeatureChapterSelect(StrictModel):
    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    horizontal_frame_id: str | None = Field(
        default=None,
        pattern=r"^RF[0-9]{6}$",
        min_length=8,
        max_length=8,
    )
    vertical_frame_id: str | None = Field(
        default=None,
        pattern=r"^RF[0-9]{6}$",
        min_length=8,
        max_length=8,
    )
    observed_visual_evidence: str
    selection_reason: str
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_camera_intent: VirtualCameraIntent = "hold"
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_target_description: str | None
    vertical_coverage_intent: Literal[
        "single_primary",
        "group_coverage",
        "sequential_attention",
        "simultaneous_relation",
    ] = "single_primary"
    vertical_coverage_target_descriptions: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Distinct visible subjects or regions whose coverage carries the "
            "chapter meaning. This is editorial identity, not geometry."
        ),
    )
    quality_risks: list[str]
    confidence: Confidence
    recommended_duration_seconds: float | None = Field(
        default=None,
        ge=1.0,
        le=15.0,
        description=(
            "Gemini's editorial dwell recommendation based on the observable "
            "action/information, the brief, and (when supplied) the audible music. "
            "It is a relative planning recommendation, not a source cut point."
        ),
    )
    duration_rationale: str | None = Field(
        default=None,
        description=(
            "Observable editorial reason for the recommended dwell. It must not "
            "claim fixed pacing rules or invent unsupported content."
        ),
    )
    attention_observation: AttentionObservation | None = None
    flow_intent: ShotFlowIntent | None = None
    source_reuse_mode: Literal[
        "none",
        "distinct_interval",
        "alternate_presentation",
        "editorial_reprise",
    ] = Field(
        default="none",
        description=(
            "Typed editorial authority for intentionally selecting a source clip "
            "already used by another chapter. It does not create additional "
            "unique source capacity."
        ),
    )
    source_reuse_justification: str | None = Field(
        default=None,
        description=(
            "Observable editorial reason required when source_reuse_mode is not "
            "none. Reuse solely to fill project duration is forbidden."
        ),
    )
    horizontal_candidates: list[FeatureHorizontalCandidate] = Field(
        default_factory=list, max_length=4
    )
    vertical_candidates: list[FeatureVerticalCandidate] = Field(
        default_factory=list, max_length=4
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> "FeatureChapterSelect":
        if self.source_reuse_mode == "none":
            if self.source_reuse_justification is not None:
                raise ValueError(
                    "source reuse justification requires a non-none reuse mode"
                )
        elif not (
            self.source_reuse_justification
            and self.source_reuse_justification.strip()
        ):
            raise ValueError("intentional source reuse requires a justification")
        atomic_compound_group = (
            self.vertical_coverage_intent == "group_coverage"
            and bool(self.vertical_candidates)
            and all(
                len(candidate.regions) == 1
                and candidate.regions[0].atomic
                and candidate.regions[0].execution_role == "hard_core"
                for candidate in self.vertical_candidates
            )
        )
        if self.vertical_coverage_intent in {
            "group_coverage",
            "sequential_attention",
            "simultaneous_relation",
        } and len(self.vertical_coverage_target_descriptions) < 2 and not (
            atomic_compound_group
        ):
            raise ValueError(
                "multi-subject vertical coverage intent requires at least two "
                "distinct target descriptions unless group coverage is bound "
                "to one explicitly atomic compound group region"
            )
        if (
            self.vertical_coverage_intent == "single_primary"
            and len(self.vertical_coverage_target_descriptions) > 1
        ):
            raise ValueError(
                "single-primary vertical coverage cannot require multiple targets"
            )
        if self.recommended_duration_seconds is not None and not (
            self.duration_rationale and self.duration_rationale.strip()
        ):
            raise ValueError(
                "recommended_duration_seconds requires duration_rationale"
            )
        if self.attention_observation is not None:
            if self.recommended_duration_seconds is None:
                raise ValueError(
                    "attention observation requires a preferred dwell recommendation"
                )
            if not (
                self.attention_observation.minimum_dwell_seconds
                <= self.recommended_duration_seconds
                <= self.attention_observation.maximum_dwell_seconds
            ):
                raise ValueError(
                    "recommended dwell must lie inside the attention observation bounds"
                )
        if self.evidence_status == "not_found":
            if self.horizontal_frame_id is not None or self.vertical_frame_id is not None:
                raise ValueError("not_found feature chapters cannot reference catalog frames")
        elif self.horizontal_frame_id is None or self.vertical_frame_id is None:
            raise ValueError("supported/partial feature chapters require both aspect frame IDs")
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError(
                    "tracked_reframe requires a zoom intent and precise horizontal target"
                )
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original horizontal strategy must use zoom intent none")
        if (
            self.horizontal_strategy == "original"
            and self.horizontal_camera_intent != "hold"
        ):
            raise ValueError("original horizontal strategy must hold the virtual camera")
        if self.vertical_strategy == "tracked_crop" and not self.vertical_target_description:
            primary_candidate = next(
                (candidate for candidate in self.vertical_candidates if candidate.rank == 1),
                None,
            )
            if primary_candidate is None or not primary_candidate.regions:
                raise ValueError(
                    "tracked_crop requires a precise target or rank-1 region contract"
                )
        for field_name in ("horizontal_candidates", "vertical_candidates"):
            candidates = getattr(self, field_name)
            if candidates and not 2 <= len(candidates) <= 4:
                raise ValueError(f"{field_name} must preserve 2-4 options when present")
            ids = [candidate.candidate_id for candidate in candidates]
            ranks = [candidate.rank for candidate in candidates]
            if len(ids) != len(set(ids)) or len(ranks) != len(set(ranks)):
                raise ValueError(f"{field_name} candidate IDs and ranks must be unique")
            references = [
                (candidate.source_asset_id, candidate.event_id, candidate.frame_id)
                for candidate in candidates
            ]
            if len(references) != len(set(references)):
                raise ValueError(
                    f"{field_name} candidates must reference distinct evidence frames"
                )
            if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
                raise ValueError(f"{field_name} ranks must be contiguous from 1")
        if self.evidence_status == "not_found" and (
            self.horizontal_candidates or self.vertical_candidates
        ):
            raise ValueError("not_found chapters cannot retain execution candidates")
        if self.horizontal_candidates:
            primary = min(self.horizontal_candidates, key=lambda item: item.rank)
            if (
                self.horizontal_frame_id != primary.frame_id
                or self.horizontal_strategy != primary.strategy
                or self.horizontal_zoom_intent != primary.zoom_intent
                or self.horizontal_camera_intent != primary.camera_intent
                or self.horizontal_target_description != primary.target_description
            ):
                raise ValueError("rank-1 horizontal candidate must match legacy projection")
        if self.vertical_candidates:
            primary = min(self.vertical_candidates, key=lambda item: item.rank)
            if (
                self.vertical_frame_id != primary.frame_id
                or self.vertical_strategy != primary.strategy
                or self.vertical_target_description != primary.target_description
            ):
                raise ValueError("rank-1 vertical candidate must match legacy projection")
        return self


class FeatureEditPlan(StrictModel):
    project_id: str
    catalog_id: str
    title: str
    chapters: list[FeatureChapterSelect]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_chapters(self) -> "FeatureEditPlan":
        ids = [chapter.feature_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("feature plan chapter IDs must be unique")
        return self


class MusicAssemblySpan(FrozenStrictModel):
    """One half-open source interval mapped onto the output music timeline."""

    span_id: str = Field(pattern=r"^music-span-[0-9]{3}$")
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    output_start_sample: int = Field(ge=0)
    output_end_sample: int = Field(gt=0)
    start_boundary_kind: Literal[
        "track_start",
        "section_boundary",
        "phrase_grid",
    ]
    end_boundary_kind: Literal["phrase_grid", "natural_track_end"]
    start_bar_index: int | None = Field(default=None, ge=0)
    end_bar_index: int | None = Field(default=None, gt=0)
    bar_count: int | None = Field(default=None, gt=0)
    phrase_bar_multiple: int | None = Field(default=None, gt=0)
    start_boundary_cue_id: str | None = Field(
        default=None,
        pattern=r"^locked-cue-[0-9]{5}$",
    )
    end_boundary_cue_id: str | None = Field(
        default=None,
        pattern=r"^locked-cue-[0-9]{5}$",
    )

    @model_validator(mode="after")
    def validate_mapping(self) -> "MusicAssemblySpan":
        source_duration = self.source_end_sample - self.source_start_sample
        output_duration = self.output_end_sample - self.output_start_sample
        if source_duration <= 0:
            raise ValueError("music assembly source span must be non-empty")
        if output_duration != source_duration:
            raise ValueError(
                "music assembly v1 must preserve sample duration without time-stretching"
            )
        if self.start_boundary_kind == "track_start" and self.source_start_sample != 0:
            raise ValueError("track_start boundary must begin at source sample zero")
        if (
            self.start_boundary_kind == "section_boundary"
            and self.source_start_sample == 0
        ):
            raise ValueError("source sample zero must use track_start boundary kind")
        if self.start_boundary_kind == "phrase_grid":
            if self.start_bar_index is None or self.phrase_bar_multiple is None:
                raise ValueError(
                    "phrase-grid start requires bar index and alignment multiple"
                )
            if self.start_bar_index % self.phrase_bar_multiple != 0:
                raise ValueError("music assembly start is not phrase-grid aligned")
        elif self.start_bar_index is not None or self.phrase_bar_multiple is not None:
            raise ValueError(
                "non-grid start cannot claim phrase-grid alignment metadata"
            )

        if self.end_boundary_kind == "phrase_grid":
            if self.start_boundary_kind != "phrase_grid":
                raise ValueError(
                    "phrase-grid end requires a phrase-grid start in assembly v1"
                )
            if (
                self.start_bar_index is None
                or self.end_bar_index is None
                or self.bar_count is None
                or self.phrase_bar_multiple is None
            ):
                raise ValueError("phrase-grid interval requires complete bar metadata")
            if self.end_bar_index <= self.start_bar_index:
                raise ValueError("music assembly bar interval must be non-empty")
            if self.bar_count != self.end_bar_index - self.start_bar_index:
                raise ValueError(
                    "music assembly bar_count does not match its bar interval"
                )
            if self.bar_count % self.phrase_bar_multiple != 0:
                raise ValueError("music assembly duration is not phrase-grid aligned")
        elif self.end_bar_index is not None or self.bar_count is not None:
            raise ValueError(
                "natural track end cannot claim a complete phrase-grid ending"
            )
        return self


class MusicAssemblyCueInstance(FrozenStrictModel):
    """A locked source cue projected onto the assembled output timeline."""

    cue_instance_id: str = Field(pattern=r"^music-cue-instance-[0-9]{5}$")
    source_cue_id: str = Field(pattern=r"^locked-cue-[0-9]{5}$")
    span_id: str = Field(pattern=r"^music-span-[0-9]{3}$")
    kind: Literal[
        "section_boundary",
        "downbeat",
        "beat",
        "accent",
        "ending_hit",
    ]
    priority: Literal["hard", "preferred", "optional"]
    source_sample_index: int = Field(ge=0)
    output_sample_index: int = Field(ge=0)
    strength: float = Field(ge=0.0, le=1.0)


class MusicAssemblyPlan(StrictModel):
    """Immutable v1 plan for one continuous, non-spliced music interval."""

    contract_version: Literal["music-assembly-plan-v1"] = "music-assembly-plan-v1"
    assembly_id: str = Field(pattern=r"^music-assembly:[0-9a-f]{64}$")
    assembly_mode: Literal["single_continuous_interval"] = (
        "single_continuous_interval"
    )
    join_count: Literal[0] = 0
    music_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    music_lock_path: str = Field(min_length=1)
    music_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    master_sample_rate: int = Field(ge=8_000, le=192_000)
    source_duration_samples: int = Field(gt=0)
    target_duration_samples: int = Field(gt=0)
    minimum_duration_samples: int = Field(gt=0)
    maximum_duration_samples: int = Field(gt=0)
    output_duration_samples: int = Field(gt=0)
    target_duration_error_samples: int = Field(ge=0)
    ending_policy: Literal[
        "short_fade_at_phrase_grid_boundary",
        "preserve_natural_track_end_no_fade_out",
    ]
    preferred_phrase_bars: tuple[int, ...] = Field(min_length=1, max_length=16)
    spans: list[MusicAssemblySpan] = Field(min_length=1, max_length=1)
    cue_instances: list[MusicAssemblyCueInstance]
    uncertainties: list[str]
    requires_human_review: Literal[True] = True
    generated_at: str

    @model_validator(mode="after")
    def validate_single_continuous_interval(self) -> "MusicAssemblyPlan":
        if not (
            self.minimum_duration_samples
            <= self.target_duration_samples
            <= self.maximum_duration_samples
        ):
            raise ValueError("target music duration must lie inside the requested range")
        if len(set(self.preferred_phrase_bars)) != len(self.preferred_phrase_bars):
            raise ValueError("preferred phrase-bar values must be unique")
        if any(value <= 0 for value in self.preferred_phrase_bars):
            raise ValueError("preferred phrase-bar values must be positive")
        if len(self.spans) != 1:
            raise ValueError(
                "music assembly v1 permits exactly one continuous source interval"
            )
        span = self.spans[0]
        if span.output_start_sample != 0:
            raise ValueError("music assembly v1 output must begin at sample zero")
        if span.source_end_sample > self.source_duration_samples:
            raise ValueError("music assembly source span exceeds the locked source")
        if span.output_end_sample != self.output_duration_samples:
            raise ValueError("music assembly span does not cover the output timeline")
        if not (
            self.minimum_duration_samples
            <= self.output_duration_samples
            <= self.maximum_duration_samples
        ):
            raise ValueError("assembled music duration lies outside the requested range")
        if self.target_duration_error_samples != abs(
            self.output_duration_samples - self.target_duration_samples
        ):
            raise ValueError("target music duration error is inconsistent")
        if (
            span.phrase_bar_multiple is not None
            and span.phrase_bar_multiple not in self.preferred_phrase_bars
        ):
            raise ValueError("selected phrase grid was not one of the requested grids")
        if span.end_boundary_kind == "natural_track_end":
            if self.ending_policy != "preserve_natural_track_end_no_fade_out":
                raise ValueError(
                    "natural track end requires the no-fade preservation policy"
                )
            if span.source_end_sample != self.source_duration_samples:
                raise ValueError(
                    "natural_track_end must preserve the locked source endpoint"
                )
            if span.end_boundary_cue_id is not None:
                raise ValueError(
                    "natural_track_end is an exclusive endpoint, not a source cue"
                )
        elif self.ending_policy != "short_fade_at_phrase_grid_boundary":
            raise ValueError(
                "phrase-grid ending requires the explicit short-fade policy"
            )

        instance_ids = [item.cue_instance_id for item in self.cue_instances]
        source_ids = [item.source_cue_id for item in self.cue_instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("music cue instance IDs must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("a locked source cue may only be mapped once in v1")
        ordering = [
            (item.output_sample_index, item.cue_instance_id)
            for item in self.cue_instances
        ]
        if ordering != sorted(ordering):
            raise ValueError("music cue instances must be chronological")
        for item in self.cue_instances:
            if item.span_id != span.span_id:
                raise ValueError("music cue instance references an unknown span")
            if not (
                span.source_start_sample
                <= item.source_sample_index
                < span.source_end_sample
            ):
                raise ValueError("music cue instance lies outside its source span")
            expected_output = (
                span.output_start_sample
                + item.source_sample_index
                - span.source_start_sample
            )
            if item.output_sample_index != expected_output:
                raise ValueError("music cue source/output sample mapping is inconsistent")
        return self


class MusicAssemblyArtifactBinding(StrictModel):
    """Hashes that bind a saved assembly plan to its reviewed music lock."""

    contract_version: Literal["music-assembly-artifact-binding-v1"] = (
        "music-assembly-artifact-binding-v1"
    )
    assembly_id: str = Field(pattern=r"^music-assembly:[0-9a-f]{64}$")
    assembly_plan_path: str = Field(min_length=1)
    assembly_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_lock_path: str = Field(min_length=1)
    music_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str


class MusicAssemblyRenderManifest(StrictModel):
    """Auditable QC record for one FFmpeg-rendered music interval."""

    contract_version: Literal["music-assembly-render-v1"] = (
        "music-assembly-render-v1"
    )
    render_id: str = Field(pattern=r"^music-render:[0-9a-f]{64}$")
    assembly_id: str = Field(pattern=r"^music-assembly:[0-9a-f]{64}$")
    assembly_plan_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_path: str = Field(min_length=1)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_audio_path: str = Field(min_length=1)
    output_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_codec: Literal["pcm_s16le"] = "pcm_s16le"
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_master_sample_rate: int = Field(ge=8_000, le=192_000)
    end_boundary_kind: Literal["phrase_grid", "natural_track_end"]
    output_sample_rate: Literal[48_000] = 48_000
    output_channels: Literal[2] = 2
    expected_output_samples: int = Field(gt=0)
    probed_output_samples: int = Field(gt=0)
    duration_delta_samples: int
    duration_tolerance_samples: int = Field(ge=0, le=16)
    fade_in_samples: int = Field(gt=0)
    fade_out_samples: int = Field(ge=0)
    internal_join_count: Literal[0] = 0
    ending_policy: Literal[
        "short_fade_at_phrase_grid_boundary",
        "preserve_natural_track_end_no_fade_out",
    ]
    natural_track_end_preserved: bool
    ffmpeg_filter_graph: str = Field(min_length=1)
    ffmpeg_command: list[str] = Field(min_length=1)
    ffprobe_audio_stream: dict[str, Any]
    qc_passed: bool
    qc_errors: list[str]
    generated_at: str

    @model_validator(mode="after")
    def validate_render_qc(self) -> "MusicAssemblyRenderManifest":
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("render source interval must be non-empty")
        if self.duration_delta_samples != (
            self.probed_output_samples - self.expected_output_samples
        ):
            raise ValueError("render duration delta is inconsistent")
        duration_passed = (
            abs(self.duration_delta_samples) <= self.duration_tolerance_samples
        )
        if self.qc_passed != (duration_passed and not self.qc_errors):
            raise ValueError("render QC status does not match its measurements")
        if self.qc_passed and self.qc_errors:
            raise ValueError("passing render QC cannot retain errors")
        if not self.qc_passed and not self.qc_errors:
            raise ValueError("failed render QC must preserve at least one error")
        expected_natural_end = self.end_boundary_kind == "natural_track_end"
        if self.natural_track_end_preserved != expected_natural_end:
            raise ValueError("natural-ending flag is inconsistent with the plan")
        if expected_natural_end:
            if self.ending_policy != "preserve_natural_track_end_no_fade_out":
                raise ValueError(
                    "natural track end requires the no-fade preservation policy"
                )
            if self.fade_out_samples != 0:
                raise ValueError("natural track end cannot apply a fade-out")
        else:
            if self.ending_policy != "short_fade_at_phrase_grid_boundary":
                raise ValueError(
                    "phrase-grid ending requires the explicit short-fade policy"
                )
            if self.fade_out_samples <= 0:
                raise ValueError("phrase-grid ending requires a non-zero fade-out")
        if "concat" in self.ffmpeg_filter_graph.lower():
            raise ValueError("music assembly v1 render graph cannot contain concat")
        return self


class MusicEditSpanV2(FrozenStrictModel):
    """One reviewed source passage mapped without time-stretching."""

    span_id: str = Field(pattern=r"^music-edit-span-[0-9]{3}$")
    section_id: str = Field(pattern=r"^section-[0-9]{3}$")
    semantic_role: Literal[
        "intro",
        "establish",
        "build",
        "climax",
        "release",
        "outro",
        "neutral",
    ]
    energy_band: Literal["low", "medium", "high", "unknown"]
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    output_start_sample: int = Field(ge=0)
    output_end_sample: int = Field(gt=0)
    start_boundary_kind: Literal[
        "track_start",
        "section_boundary",
        "locked_cue",
    ]
    end_boundary_kind: Literal[
        "section_boundary",
        "locked_cue",
        "natural_track_end",
    ]
    start_boundary_cue_id: str | None = Field(
        default=None,
        pattern=r"^locked-cue-[0-9]{5}$",
    )
    end_boundary_cue_id: str | None = Field(
        default=None,
        pattern=r"^locked-cue-[0-9]{5}$",
    )

    @model_validator(mode="after")
    def validate_music_edit_span(self) -> "MusicEditSpanV2":
        source_duration = self.source_end_sample - self.source_start_sample
        output_duration = self.output_end_sample - self.output_start_sample
        if source_duration != output_duration:
            raise ValueError("music edit spans cannot time-stretch source audio")
        if self.start_boundary_kind == "locked_cue":
            if self.start_boundary_cue_id is None:
                raise ValueError("locked-cue start requires its cue ID")
        elif self.start_boundary_cue_id is not None:
            raise ValueError("non-cue start cannot claim a locked cue ID")
        if self.end_boundary_kind == "locked_cue":
            if self.end_boundary_cue_id is None:
                raise ValueError("locked-cue end requires its cue ID")
        elif self.end_boundary_cue_id is not None:
            raise ValueError("non-cue end cannot claim a locked cue ID")
        if (
            self.start_boundary_kind == "track_start"
            and self.source_start_sample != 0
        ):
            raise ValueError("track-start music span must begin at sample zero")
        return self


class MusicEditJoinV2(FrozenStrictModel):
    """An explicit, reviewable transition between two music passages."""

    join_id: str = Field(pattern=r"^music-edit-join-[0-9]{3}$")
    left_span_id: str = Field(pattern=r"^music-edit-span-[0-9]{3}$")
    right_span_id: str = Field(pattern=r"^music-edit-span-[0-9]{3}$")
    join_type: Literal["cut", "micro_crossfade"]
    duration_samples: int = Field(ge=0)
    alignment: Literal[
        "section_boundary",
        "phrase_grid",
        "downbeat",
        "accent",
        "transient",
    ]
    energy_transition: Literal[
        "matched",
        "rising",
        "falling",
        "intentional_contrast",
        "unknown",
    ]
    editorial_reason: str = Field(min_length=1, max_length=500)
    requires_human_review: Literal[True] = True

    @model_validator(mode="after")
    def validate_join(self) -> "MusicEditJoinV2":
        if self.left_span_id == self.right_span_id:
            raise ValueError("music join must connect two different spans")
        if self.join_type == "cut" and self.duration_samples != 0:
            raise ValueError("hard music cut must have zero overlap")
        if self.join_type == "micro_crossfade" and self.duration_samples <= 0:
            raise ValueError("micro crossfade requires a positive overlap")
        return self


class MusicDuckingRegionV2(FrozenStrictModel):
    """An output-timeline gain reduction with a typed editorial purpose."""

    region_id: str = Field(pattern=r"^music-duck-[0-9]{3}$")
    output_start_sample: int = Field(ge=0)
    output_end_sample: int = Field(gt=0)
    gain_db: float = Field(ge=-30.0, le=0.0)
    reason: Literal[
        "dialogue",
        "narration",
        "ui_focus",
        "editorial_emphasis",
    ]

    @model_validator(mode="after")
    def validate_ducking_region(self) -> "MusicDuckingRegionV2":
        if self.output_end_sample <= self.output_start_sample:
            raise ValueError("music ducking region must be non-empty")
        return self


class MusicEditEndingV2(FrozenStrictModel):
    """The reviewed way a shortened soundtrack resolves."""

    mode: Literal[
        "natural_track_end",
        "phrase_fade_out",
        "reviewed_ending_hit",
    ]
    fade_out_samples: int = Field(ge=0)
    ending_cue_id: str | None = Field(
        default=None,
        pattern=r"^locked-cue-[0-9]{5}$",
    )
    editorial_reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_ending(self) -> "MusicEditEndingV2":
        if self.mode == "natural_track_end":
            if self.fade_out_samples != 0 or self.ending_cue_id is not None:
                raise ValueError("natural music ending cannot add a fade or cue")
        elif self.mode == "phrase_fade_out":
            if self.fade_out_samples <= 0 or self.ending_cue_id is not None:
                raise ValueError("phrase fade requires a fade and no ending cue")
        elif self.ending_cue_id is None:
            raise ValueError("reviewed ending hit requires its locked cue ID")
        return self


class MusicEditPlanV2(StrictModel):
    """Reviewed multi-passage soundtrack edit; exact samples remain local."""

    contract_version: Literal["music-edit-plan-v2"] = "music-edit-plan-v2"
    edit_id: str = Field(pattern=r"^music-edit:[0-9a-f]{64}$")
    music_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    music_lock_path: str = Field(min_length=1)
    music_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    master_sample_rate: int = Field(ge=8_000, le=192_000)
    source_duration_samples: int = Field(gt=0)
    target_duration_samples: int = Field(gt=0)
    minimum_duration_samples: int = Field(gt=0)
    maximum_duration_samples: int = Field(gt=0)
    output_duration_samples: int = Field(gt=0)
    target_duration_error_samples: int = Field(ge=0)
    spans: list[MusicEditSpanV2] = Field(min_length=1, max_length=4)
    joins: list[MusicEditJoinV2] = Field(max_length=3)
    ending: MusicEditEndingV2
    ducking_regions: list[MusicDuckingRegionV2] = Field(max_length=32)
    uncertainties: list[str]
    requires_human_review: Literal[True] = True
    generated_at: str

    @model_validator(mode="after")
    def validate_music_edit_plan(self) -> "MusicEditPlanV2":
        if not (
            self.minimum_duration_samples
            <= self.target_duration_samples
            <= self.maximum_duration_samples
        ):
            raise ValueError("target music duration must lie inside its range")
        if not (
            self.minimum_duration_samples
            <= self.output_duration_samples
            <= self.maximum_duration_samples
        ):
            raise ValueError("music edit output duration lies outside its range")
        if self.target_duration_error_samples != abs(
            self.output_duration_samples - self.target_duration_samples
        ):
            raise ValueError("music edit target duration error is inconsistent")
        if len(self.joins) != len(self.spans) - 1:
            raise ValueError("music edit must have exactly one join between spans")

        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("music edit span IDs must be unique")
        if self.spans[0].output_start_sample != 0:
            raise ValueError("music edit output must begin at sample zero")
        for index, span in enumerate(self.spans):
            if span.source_end_sample > self.source_duration_samples:
                raise ValueError("music edit span exceeds the locked source")
            if index == 0:
                continue
            join = self.joins[index - 1]
            prior = self.spans[index - 1]
            if join.left_span_id != prior.span_id:
                raise ValueError("music join left span is not adjacent")
            if join.right_span_id != span.span_id:
                raise ValueError("music join right span is not adjacent")
            if join.join_type == "micro_crossfade":
                minimum = max(1, round(self.master_sample_rate * 0.005))
                maximum = round(self.master_sample_rate * 0.200)
                if not minimum <= join.duration_samples <= maximum:
                    raise ValueError(
                        "micro crossfade must remain between 5 and 200 ms"
                    )
                if join.duration_samples >= (
                    span.source_end_sample - span.source_start_sample
                ):
                    raise ValueError("crossfade cannot consume the right span")
                if join.duration_samples >= (
                    prior.source_end_sample - prior.source_start_sample
                ):
                    raise ValueError("crossfade cannot consume the left span")
            expected_start = prior.output_end_sample - join.duration_samples
            if span.output_start_sample != expected_start:
                raise ValueError("music span placement disagrees with its join")

        if self.spans[-1].output_end_sample != self.output_duration_samples:
            raise ValueError("music spans do not cover the output timeline")
        source_intervals = sorted(
            (span.source_start_sample, span.source_end_sample)
            for span in self.spans
        )
        for prior, current in zip(
            source_intervals[:-1],
            source_intervals[1:],
            strict=True,
        ):
            if current[0] < prior[1]:
                raise ValueError(
                    "music edit cannot silently replay overlapping source passages"
                )
        if self.ending.mode == "natural_track_end":
            if self.spans[-1].source_end_sample != self.source_duration_samples:
                raise ValueError("natural ending must preserve the source endpoint")
            if self.spans[-1].end_boundary_kind != "natural_track_end":
                raise ValueError("natural ending requires a natural-end span")
        elif self.spans[-1].end_boundary_kind == "natural_track_end":
            raise ValueError("natural-end span cannot claim an artificial ending")
        if self.ending.fade_out_samples >= self.output_duration_samples:
            raise ValueError("music ending fade cannot consume the full edit")
        if self.ending.mode == "reviewed_ending_hit":
            if self.spans[-1].end_boundary_cue_id != self.ending.ending_cue_id:
                raise ValueError("ending hit must match the final locked cue")
        for region in self.ducking_regions:
            if region.output_end_sample > self.output_duration_samples:
                raise ValueError("music ducking region exceeds the output timeline")
        return self


class MusicEditRenderManifestV2(StrictModel):
    """Deterministic FFmpeg render evidence for a reviewed MusicEditPlanV2."""

    contract_version: Literal["music-edit-render-v2"] = "music-edit-render-v2"
    render_id: str = Field(pattern=r"^music-edit-render:[0-9a-f]{64}$")
    edit_id: str = Field(pattern=r"^music-edit:[0-9a-f]{64}$")
    edit_plan_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_path: str = Field(min_length=1)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_audio_path: str = Field(min_length=1)
    output_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_codec: Literal["pcm_s16le"] = "pcm_s16le"
    output_sample_rate: Literal[48_000] = 48_000
    output_channels: Literal[2] = 2
    expected_output_samples: int = Field(gt=0)
    probed_output_samples: int = Field(gt=0)
    duration_delta_samples: int
    duration_tolerance_samples: int = Field(ge=0, le=16)
    internal_join_count: int = Field(ge=0, le=3)
    crossfade_samples: int = Field(ge=0)
    fade_in_samples: int = Field(gt=0)
    fade_out_samples: int = Field(ge=0)
    ducking_region_count: int = Field(ge=0, le=32)
    ffmpeg_filter_graph: str = Field(min_length=1)
    ffmpeg_command: list[str] = Field(min_length=1)
    ffprobe_audio_stream: dict[str, Any]
    qc_passed: bool
    qc_errors: list[str]
    generated_at: str

    @model_validator(mode="after")
    def validate_music_edit_render(self) -> "MusicEditRenderManifestV2":
        if self.duration_delta_samples != (
            self.probed_output_samples - self.expected_output_samples
        ):
            raise ValueError("music edit render duration delta is inconsistent")
        passed = (
            abs(self.duration_delta_samples) <= self.duration_tolerance_samples
            and not self.qc_errors
        )
        if self.qc_passed != passed:
            raise ValueError("music edit render QC status is inconsistent")
        return self
