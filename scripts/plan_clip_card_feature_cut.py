#!/usr/bin/env python3
"""Plan an auditable feature cut from a complete Clip Card library.

The model may only select immutable catalog frame IDs backed by a validated
Clip Card event. Local validation projects the richer audit plan into the
FeatureEditPlan consumed by the existing Grounding and tracking renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.clip_card_retrieval import (
    FeatureShortlistPlan,
    validate_feature_shortlist,
)
from jascue_video_lab.clip_card_observations import (
    ClaimDecision,
    ClipObservationSupplement,
    EditingClaim,
    assess_editing_claim,
    effective_event_observations,
)
from jascue_video_lab.clip_card_supplement_runner import (
    bounded_event_window_ms,
    render_bounded_event_proxy,
)
from jascue_video_lab.feature_cut import write_external_feature_plan_projection
from jascue_video_lab.editing_capabilities import (
    EditingCapabilityCatalog,
    simple_production_capability_catalog,
)
from jascue_video_lab.gemini import (
    GeminiLabClient,
    MODEL_ID,
    _raw_dump,
    canonical_interactions_mime_type,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    AttentionObservation,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FeatureHorizontalCandidate,
    FeatureVerticalCandidate,
    FramingRegionIntent,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
    ShotFlowIntent,
    VerticalVirtualCameraProposal,
    VerticalVirtualCameraProposalPhase,
    VirtualCameraIntent,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Keep these two v1 models byte-for-byte schema compatible with the historical
# source request.  In particular, do not add class docstrings: Pydantic emits
# them as JSON Schema descriptions and provenance validation compares schemas.
class ClipCardFeatureSelect(StrictModel):
    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    horizontal_source_asset_id: str | None = None
    horizontal_event_id: str | None = None
    horizontal_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    vertical_source_asset_id: str | None = None
    vertical_event_id: str | None = None
    vertical_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_target_description: str | None
    quality_risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "ClipCardFeatureSelect":
        ids = (
            self.horizontal_source_asset_id,
            self.horizontal_event_id,
            self.horizontal_frame_id,
            self.vertical_source_asset_id,
            self.vertical_event_id,
            self.vertical_frame_id,
        )
        if self.evidence_status == "not_found":
            if any(value is not None for value in ids):
                raise ValueError("not_found chapters cannot reference source evidence")
        elif any(value is None for value in ids):
            raise ValueError("supported/partial chapters require both source/event/frame triples")
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError("tracked_reframe requires zoom intent and target")
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original horizontal strategy must use zoom intent none")
        if self.vertical_strategy == "tracked_crop" and not self.vertical_target_description:
            raise ValueError("tracked_crop requires a target")
        return self


class ClipCardFeaturePlan(StrictModel):
    project_id: str
    catalog_id: str
    title: str
    strategy_summary: str
    chapters: list[ClipCardFeatureSelect]
    uncertainties: list[str]
    model_provenance: ModelProvenance


class ResolvedEntityRef(StrictModel):
    """An auditable link from a planner region back to one Clip Card event."""

    entity_id: str = Field(min_length=1)
    event_relation: Literal[
        "event_member",
        "primary",
        "required",
        "optional",
        "avoid_overlay",
        "grounding_target",
    ]


class ResolvedFramingRegion(StrictModel):
    """Domain-neutral crop evidence resolved to immutable Clip Card entities.

    ``hard_core`` is content that must remain visible, ``soft_extent`` is useful
    context that may be sacrificed, and ``overlay_keepout`` is content that a
    later layout system should avoid covering.  ``atomic`` regions intentionally
    refer to one entity; a ``union`` makes a multi-entity constraint explicit.
    """

    region_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    target_description: str = Field(min_length=1)
    kind: Literal["subject", "text_region", "ui_region", "graphic", "other"]
    constraint_role: Literal["hard_core", "soft_extent", "overlay_keepout"]
    composition: Literal["atomic", "union"] = "atomic"
    atomic: bool = False
    entity_refs: list[ResolvedEntityRef] = Field(min_length=1, max_length=4)
    observable_relation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entity_refs(self) -> "ResolvedFramingRegion":
        ids = [ref.entity_id for ref in self.entity_refs]
        if len(ids) != len(set(ids)):
            raise ValueError("resolved region entity refs must be unique")
        if self.composition == "atomic" and len(ids) != 1:
            raise ValueError("atomic resolved regions must reference exactly one entity")
        if self.composition == "union" and len(ids) < 2:
            raise ValueError("union resolved regions must reference at least two entities")
        if self.atomic and self.constraint_role != "hard_core":
            raise ValueError("atomic crop content must use hard_core constraint role")
        return self


class ClipCardFeatureCandidate(StrictModel):
    """One ranked take whose source and geometry intent remain auditable."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    source_asset_id: str
    event_id: str
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str = Field(min_length=1, max_length=600)
    selection_reason: str = Field(min_length=1, max_length=500)
    quality_risks: list[str] = Field(max_length=8)
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"
    vertical_target_description: str | None
    resolved_regions: list[ResolvedFramingRegion] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_candidate(self) -> "ClipCardFeatureCandidate":
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError("tracked_reframe candidate requires zoom intent and target")
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original candidate must use zoom intent none")
        hard_regions = [
            region
            for region in self.resolved_regions
            if region.constraint_role == "hard_core"
        ]
        if self.vertical_strategy == "tracked_crop" and not (
            self.vertical_target_description or hard_regions
        ):
            raise ValueError("tracked_crop candidate requires a target or hard-core region")
        if self.resolved_regions and self.vertical_strategy != "tracked_crop":
            raise ValueError("resolved crop regions require tracked_crop")
        region_ids = [region.region_id for region in self.resolved_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("candidate resolved region IDs must be unique")
        return self


class ClipCardFeatureSelectV2(StrictModel):
    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    horizontal_source_asset_id: str | None = None
    horizontal_event_id: str | None = None
    horizontal_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    vertical_source_asset_id: str | None = None
    vertical_event_id: str | None = None
    vertical_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_target_description: str | None
    quality_risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: list[ClipCardFeatureCandidate] = Field(default_factory=list, max_length=4)
    horizontal_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )
    vertical_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "ClipCardFeatureSelectV2":
        ids = (
            self.horizontal_source_asset_id,
            self.horizontal_event_id,
            self.horizontal_frame_id,
            self.vertical_source_asset_id,
            self.vertical_event_id,
            self.vertical_frame_id,
        )
        if self.evidence_status == "not_found":
            if any(value is not None for value in ids):
                raise ValueError("not_found chapters cannot reference source evidence")
        elif any(value is None for value in ids):
            raise ValueError("supported/partial chapters require both source/event/frame triples")
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError("tracked_reframe requires zoom intent and target")
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original horizontal strategy must use zoom intent none")
        if self.vertical_strategy == "tracked_crop" and not self.vertical_target_description:
            selected_vertical = next(
                (
                    candidate
                    for candidate in self.candidates
                    if candidate.candidate_id == self.vertical_candidate_id
                ),
                None,
            )
            selected_hard_regions = (
                [
                    region
                    for region in selected_vertical.resolved_regions
                    if region.constraint_role == "hard_core"
                ]
                if selected_vertical is not None
                else []
            )
            if not selected_hard_regions:
                raise ValueError("tracked_crop requires a target or selected hard-core region")
        if self.candidates:
            candidate_ids = [candidate.candidate_id for candidate in self.candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("candidate IDs must be unique within a chapter")
            references = [
                (candidate.source_asset_id, candidate.event_id, candidate.frame_id)
                for candidate in self.candidates
            ]
            if len(references) != len(set(references)):
                raise ValueError("Top-K candidates must reference distinct evidence frames")
            if self.evidence_status == "not_found":
                raise ValueError("not_found chapters cannot preserve candidates")
            if self.horizontal_candidate_id not in candidate_ids:
                raise ValueError("horizontal candidate must be present in candidates")
            if self.vertical_candidate_id not in candidate_ids:
                raise ValueError("vertical candidate must be present in candidates")
            by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
            horizontal = by_id[self.horizontal_candidate_id]
            vertical = by_id[self.vertical_candidate_id]
            if (
                self.horizontal_source_asset_id,
                self.horizontal_event_id,
                self.horizontal_frame_id,
            ) != (horizontal.source_asset_id, horizontal.event_id, horizontal.frame_id):
                raise ValueError("legacy horizontal selection must match selected candidate")
            if (
                self.vertical_source_asset_id,
                self.vertical_event_id,
                self.vertical_frame_id,
            ) != (vertical.source_asset_id, vertical.event_id, vertical.frame_id):
                raise ValueError("legacy vertical selection must match selected candidate")
            if (
                self.horizontal_strategy,
                self.horizontal_zoom_intent,
                self.horizontal_target_description,
            ) != (
                horizontal.horizontal_strategy,
                horizontal.horizontal_zoom_intent,
                horizontal.horizontal_target_description,
            ):
                raise ValueError("legacy horizontal geometry must match selected candidate")
            if (
                self.vertical_strategy,
                self.vertical_target_description,
            ) != (
                vertical.vertical_strategy,
                vertical.vertical_target_description,
            ):
                raise ValueError("legacy vertical geometry must match selected candidate")
        elif self.horizontal_candidate_id is not None or self.vertical_candidate_id is not None:
            raise ValueError("candidate IDs require a candidate list")
        return self


class ClipCardFeaturePlanV2(StrictModel):
    contract_version: Literal["legacy-v1", "clip-card-feature-cut-v2"]
    project_id: str
    catalog_id: str
    title: str
    strategy_summary: str
    chapters: list[ClipCardFeatureSelectV2]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_contract(cls, value: Any) -> Any:
        # Previously saved planner JSON has no contract_version or alternatives.
        # Keep it readable and deterministically projectable without pretending
        # that a legacy single selection is a genuine Top-K result.
        if isinstance(value, dict) and "contract_version" not in value:
            return {"contract_version": "legacy-v1", **value}
        return value

    @model_validator(mode="after")
    def validate_contract_version(self) -> "ClipCardFeaturePlanV2":
        if self.contract_version == "clip-card-feature-cut-v2":
            for chapter in self.chapters:
                if chapter.evidence_status == "not_found":
                    if chapter.candidates:
                        raise ValueError("v2 not_found chapters cannot contain candidates")
                    continue
                if not 2 <= len(chapter.candidates) <= 4:
                    raise ValueError("v2 chapters must preserve Top-K 2-4 candidates")
        elif any(chapter.candidates for chapter in self.chapters):
            raise ValueError("legacy-v1 plans cannot claim v2 candidate alternatives")
        return self


class ClipCardVirtualCameraPhaseV1(StrictModel):
    """Editorial phase referencing immutable Clip Card entity IDs."""

    phase_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    start_progress: float = Field(ge=0.0, le=1.0)
    end_progress: float = Field(gt=0.0, le=1.0)
    anchor_entity_ids: list[str] = Field(min_length=1, max_length=4)
    camera_behavior: Literal[
        "hold",
        "follow",
        "follow_deadband",
        "push_in",
        "pull_out",
        "punch_in_cut",
    ] = "follow_deadband"
    transition_in: Literal["cut", "smoothstep"] = "cut"
    transition_duration_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    observable_predicate: str = Field(min_length=1, max_length=800)
    transition_condition: str = Field(min_length=1, max_length=800)
    editorial_reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_phase(self) -> "ClipCardVirtualCameraPhaseV1":
        if self.end_progress <= self.start_progress:
            raise ValueError("virtual camera phase must have positive duration")
        if len(self.anchor_entity_ids) != len(set(self.anchor_entity_ids)):
            raise ValueError("virtual camera phase entity IDs must be unique")
        if self.transition_in == "cut" and self.transition_duration_fraction != 0:
            raise ValueError("cut transition cannot have a transition duration")
        if self.transition_in == "smoothstep" and (
            self.transition_duration_fraction <= 0
        ):
            raise ValueError("smoothstep transition requires a positive duration")
        return self


class ClipCardVirtualCameraProposalV1(StrictModel):
    composition_mode: Literal[
        "single_anchor_hold",
        "single_anchor_follow",
        "sequential_focus",
        "joint_relation",
    ]
    phases: list[ClipCardVirtualCameraPhaseV1] = Field(min_length=1, max_length=8)
    proposal_reason: str = Field(min_length=1, max_length=1200)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_proposal(self) -> "ClipCardVirtualCameraProposalV1":
        if abs(self.phases[0].start_progress) > 1e-6:
            raise ValueError("virtual camera proposal must start at progress zero")
        if abs(self.phases[-1].end_progress - 1.0) > 1e-6:
            raise ValueError("virtual camera proposal must end at progress one")
        for prior, current in zip(self.phases[:-1], self.phases[1:], strict=True):
            if abs(prior.end_progress - current.start_progress) > 1e-6:
                raise ValueError("virtual camera proposal phases must be contiguous")
        if self.phases[0].transition_in != "cut":
            raise ValueError("first virtual camera phase must use a cut")
        unique_entity_ids = {
            entity_id
            for phase in self.phases
            for entity_id in phase.anchor_entity_ids
        }
        if self.composition_mode == "sequential_focus" and (
            len(self.phases) < 2 or len(unique_entity_ids) < 2
        ):
            raise ValueError(
                "sequential focus requires at least two phases and entities"
            )
        if self.composition_mode in {
            "single_anchor_hold",
            "single_anchor_follow",
        } and len(unique_entity_ids) != 1:
            raise ValueError("single-anchor mode must reference one entity")
        return self


class ClipCardFeatureCandidateV3(StrictModel):
    """One ranked take; local evidence owns descriptions and crop regions."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    source_asset_id: str
    event_id: str
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    quality_risks: list[str]
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_camera_intent: VirtualCameraIntent = "hold"
    horizontal_focus_entity_id: str | None = None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"
    coverage_mode: Literal[
        "simultaneous",
        "sequential",
        "relation_core",
        "primary_with_context",
        "independent_detail",
    ] = "simultaneous"
    allow_controlled_clip: bool = False
    framing_intent: str = Field(min_length=1, max_length=300)
    required_entity_ids: list[str] = Field(default_factory=list, max_length=4)
    preferred_entity_ids: list[str] = Field(default_factory=list, max_length=4)
    sacrificable_entity_ids: list[str] = Field(default_factory=list, max_length=4)
    virtual_camera_proposal: "ClipCardVirtualCameraProposalV1 | None" = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_candidate(self) -> "ClipCardFeatureCandidateV3":
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_focus_entity_id:
                raise ValueError("tracked_reframe requires a focus entity and zoom intent")
        elif self.horizontal_zoom_intent != "none" or self.horizontal_focus_entity_id:
            raise ValueError("original horizontal strategy cannot declare a focus entity or zoom")
        if (
            self.horizontal_strategy == "original"
            and self.horizontal_camera_intent != "hold"
        ):
            raise ValueError("original horizontal candidate must hold the camera")
        if self.vertical_strategy == "tracked_crop" and not self.required_entity_ids:
            raise ValueError("tracked_crop requires at least one required entity")
        if self.allow_controlled_clip and self.vertical_crop_mode != "primary_center":
            raise ValueError(
                "controlled clipping requires primary_center crop mode"
            )
        if self.vertical_strategy == "fit_with_background" and self.allow_controlled_clip:
            raise ValueError(
                "fit-with-background cannot request controlled clipping"
            )
        classified = (
            self.required_entity_ids
            + self.preferred_entity_ids
            + self.sacrificable_entity_ids
        )
        if len(classified) != len(set(classified)):
            raise ValueError("vertical semantic entity roles must be disjoint and unique")
        if self.virtual_camera_proposal is not None:
            if self.vertical_strategy != "tracked_crop":
                raise ValueError(
                    "virtual camera proposal requires tracked_crop"
                )
            trackable_ids = set(
                self.required_entity_ids + self.preferred_entity_ids
            )
            referenced_ids = {
                entity_id
                for phase in self.virtual_camera_proposal.phases
                for entity_id in phase.anchor_entity_ids
            }
            unknown = sorted(referenced_ids - trackable_ids)
            if unknown:
                raise ValueError(
                    "virtual camera proposal references entities without crop "
                    "regions: " + ", ".join(unknown)
                )
            missing_required = sorted(
                set(self.required_entity_ids) - referenced_ids
            )
            if missing_required:
                raise ValueError(
                    "every required entity must appear in the virtual camera "
                    "proposal: " + ", ".join(missing_required)
                )
        return self


class ClipCardFeatureSelectV3(StrictModel):
    """Selection-only chapter; rank-one mirror fields are projected locally."""

    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    candidates: list[ClipCardFeatureCandidateV3] = Field(default_factory=list, max_length=4)
    horizontal_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )
    vertical_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )
    recommended_duration_seconds: float | None = Field(
        default=None, ge=1.0, le=15.0
    )
    duration_rationale: str | None = None
    attention_observation: AttentionObservation | None = None
    flow_intent: ShotFlowIntent | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "ClipCardFeatureSelectV3":
        if self.recommended_duration_seconds is not None and not (
            self.duration_rationale and self.duration_rationale.strip()
        ):
            raise ValueError("recommended dwell requires a duration rationale")
        if self.attention_observation is not None:
            if self.recommended_duration_seconds is None:
                raise ValueError("attention observation requires preferred dwell")
            if not (
                self.attention_observation.minimum_dwell_seconds
                <= self.recommended_duration_seconds
                <= self.attention_observation.maximum_dwell_seconds
            ):
                raise ValueError("preferred dwell lies outside attention bounds")
        if self.evidence_status == "not_found":
            if self.candidates or self.horizontal_candidate_id or self.vertical_candidate_id:
                raise ValueError("not_found chapters cannot reference candidates")
            return self
        minimum = 2 if self.evidence_status == "supported" else 1
        if not minimum <= len(self.candidates) <= 4:
            raise ValueError(
                f"v3 {self.evidence_status} chapters must preserve "
                f"Top-K {minimum}-4 candidates"
            )
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a chapter")
        references = [
            (candidate.source_asset_id, candidate.event_id, candidate.frame_id)
            for candidate in self.candidates
        ]
        if len(references) != len(set(references)):
            raise ValueError("Top-K candidates must reference distinct evidence frames")
        if self.horizontal_candidate_id not in candidate_ids:
            raise ValueError("horizontal candidate must be present in candidates")
        if self.vertical_candidate_id not in candidate_ids:
            raise ValueError("vertical candidate must be present in candidates")
        return self


class ClipCardFeaturePlanV3(StrictModel):
    """Cost-bounded model output containing editorial choices, not mirrors."""

    contract_version: Literal["clip-card-feature-cut-v3"]
    project_id: str
    catalog_id: str
    title: str
    strategy_summary: str
    chapters: list[ClipCardFeatureSelectV3]
    uncertainties: list[str]
    model_provenance: ModelProvenance


class DirectVideoAttentionStep(StrictModel):
    start_progress: float = Field(ge=0.0, le=1.0)
    end_progress: float = Field(gt=0.0, le=1.0)
    anchor_entity_indices: list[int] = Field(min_length=1, max_length=4)
    camera_behavior: Literal[
        "hold",
        "follow",
        "follow_deadband",
        "push_in",
        "pull_out",
        "punch_in_cut",
    ]
    transition_preference: Literal["auto", "continuous", "cut"] = "auto"

    @model_validator(mode="after")
    def validate_step(self) -> "DirectVideoAttentionStep":
        if self.end_progress <= self.start_progress:
            raise ValueError("attention step must have positive duration")
        if any(index < 1 for index in self.anchor_entity_indices):
            raise ValueError("attention step entity indices must be positive")
        if len(self.anchor_entity_indices) != len(set(self.anchor_entity_indices)):
            raise ValueError("attention step entity indices must be unique")
        return self


class DirectVideoHorizontalDecision(StrictModel):
    candidate_rank: int = Field(ge=1, le=4)
    strategy: Literal["original", "tracked_reframe"]
    zoom_intent: Literal["none", "subtle", "detail"]
    camera_intent: VirtualCameraIntent
    focus_entity_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_horizontal(self) -> "DirectVideoHorizontalDecision":
        if self.strategy == "original":
            if (
                self.zoom_intent != "none"
                or self.camera_intent != "hold"
                or self.focus_entity_index is not None
            ):
                raise ValueError(
                    "original horizontal framing must hold without zoom or focus"
                )
        elif self.zoom_intent == "none" or self.focus_entity_index is None:
            raise ValueError(
                "tracked horizontal framing requires zoom and a focus entity"
            )
        return self


class DirectVideoVerticalDecision(StrictModel):
    candidate_rank: int = Field(ge=1, le=4)
    strategy: Literal["tracked_crop", "fit_with_background"]
    crop_mode: Literal["strict", "primary_center"]
    coverage_mode: Literal[
        "simultaneous",
        "sequential",
        "relation_core",
        "primary_with_context",
        "independent_detail",
    ]
    allow_controlled_clip: bool = False
    framing_intent: str = Field(min_length=1, max_length=300)
    required_entity_indices: list[int] = Field(max_length=4)
    preferred_entity_indices: list[int] = Field(max_length=4)
    sacrificable_entity_indices: list[int] = Field(max_length=4)
    attention_sequence: list[DirectVideoAttentionStep] = Field(
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_vertical(self) -> "DirectVideoVerticalDecision":
        classified = (
            self.required_entity_indices
            + self.preferred_entity_indices
            + self.sacrificable_entity_indices
        )
        if any(index < 1 for index in classified):
            raise ValueError("vertical entity indices must be positive")
        if len(classified) != len(set(classified)):
            raise ValueError("vertical entity roles must be disjoint")
        if self.strategy == "tracked_crop" and not self.required_entity_indices:
            raise ValueError("tracked crop requires a visible required entity")
        if self.strategy == "fit_with_background" and self.attention_sequence:
            raise ValueError("fit-with-background cannot declare camera movement")
        if self.strategy == "fit_with_background" and self.allow_controlled_clip:
            raise ValueError("fit-with-background cannot request controlled clipping")
        if self.coverage_mode == "simultaneous" and len(self.attention_sequence) > 1:
            raise ValueError(
                "simultaneous coverage cannot split evidence across phases"
            )
        if self.coverage_mode == "sequential":
            if len(self.attention_sequence) < 2:
                raise ValueError(
                    "sequential coverage requires at least two attention phases"
                )
            unique_phase_anchors = {
                index
                for step in self.attention_sequence
                for index in step.anchor_entity_indices
            }
            if len(unique_phase_anchors) < 2:
                raise ValueError(
                    "sequential coverage requires at least two distinct anchors"
                )
        if self.coverage_mode in {
            "relation_core",
            "primary_with_context",
            "independent_detail",
        } and self.strategy == "tracked_crop":
            if self.crop_mode != "primary_center":
                raise ValueError(
                    "semantic-core coverage requires primary_center crop mode"
                )
            if not self.allow_controlled_clip:
                raise ValueError(
                    "semantic-core coverage must explicitly allow controlled clipping"
                )
        if self.attention_sequence:
            if abs(self.attention_sequence[0].start_progress) > 1e-6:
                raise ValueError("attention sequence must start at zero")
            if abs(self.attention_sequence[-1].end_progress - 1.0) > 1e-6:
                raise ValueError("attention sequence must end at one")
            for prior, current in zip(
                self.attention_sequence[:-1],
                self.attention_sequence[1:],
                strict=True,
            ):
                if abs(prior.end_progress - current.start_progress) > 1e-6:
                    raise ValueError("attention sequence must be contiguous")
            referenced = {
                entity_id
                for step in self.attention_sequence
                for entity_id in step.anchor_entity_indices
            }
            allowed = set(
                self.required_entity_indices + self.preferred_entity_indices
            )
            if referenced - allowed:
                raise ValueError(
                    "attention sequence may only use required or preferred entities"
                )
            if set(self.required_entity_indices) - referenced:
                raise ValueError(
                    "attention sequence must represent every required entity"
                )
        return self


class DirectVideoChapterDecision(StrictModel):
    chapter_index: int = Field(ge=1)
    evidence_status: Literal["supported", "partial", "not_found"]
    observed_visual_evidence: str
    selection_reason: str
    quality_risks: list[str] = Field(default_factory=list, max_length=8)
    horizontal: DirectVideoHorizontalDecision | None = None
    vertical: DirectVideoVerticalDecision | None = None
    recommended_duration_seconds: float | None = Field(
        default=None,
        ge=1.0,
        le=15.0,
    )
    duration_rationale: str | None = None
    attention_observation: AttentionObservation | None
    flow_intent: ShotFlowIntent | None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chapter(self) -> "DirectVideoChapterDecision":
        if self.evidence_status == "not_found":
            if (
                self.horizontal is not None
                or self.vertical is not None
                or self.recommended_duration_seconds is not None
                or self.attention_observation is not None
                or self.flow_intent is not None
            ):
                raise ValueError("not-found chapter cannot contain edit decisions")
            return self
        if (
            self.horizontal is None
            or self.vertical is None
            or self.recommended_duration_seconds is None
            or not self.duration_rationale
            or self.attention_observation is None
            or self.flow_intent is None
        ):
            raise ValueError(
                "supported or partial chapter requires aspect, flow, and duration decisions"
            )
        if not (
            self.attention_observation.minimum_dwell_seconds
            <= self.recommended_duration_seconds
            <= self.attention_observation.maximum_dwell_seconds
        ):
            raise ValueError("recommended dwell lies outside attention bounds")
        return self


class DirectVideoEditPlan(StrictModel):
    contract_version: Literal["direct-video-edit-plan-v2"]
    capability_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str
    strategy_summary: str
    chapters: list[DirectVideoChapterDecision]
    uncertainties: list[str]


class SelectedEvidenceEntity(StrictModel):
    entity_id: str
    kind: str
    label: str
    distinguishing_features: str


class SelectedEvidenceGroundingTarget(StrictModel):
    entity_id: str
    target_description: str


class SelectedEvidenceEvent(StrictModel):
    source_asset_id: str
    event_id: str
    entity_ids: list[str]
    primary_entity_ids: list[str]
    required_entity_ids: list[str]
    optional_entity_ids: list[str]
    avoid_overlay_entity_ids: list[str]
    entities: list[SelectedEvidenceEntity]
    grounding_targets: list[SelectedEvidenceGroundingTarget]


class SelectedClipCardEvidence(StrictModel):
    """Hash-bound local evidence required to reproduce a v3 projection."""

    contract_version: Literal["clip-card-feature-cut-selected-evidence-v1"]
    events: list[SelectedEvidenceEvent]


FEATURE_PLAN_NORMALIZATION_VERSION = "clip-card-feature-plan-normalization-v2"


def canonicalize_feature_plan_output(
    output_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Canonicalize narrow, deterministic representation errors.

    The function is deliberately narrow and deterministic.  It never changes
    editorial selections.  Short RF identifiers are zero-padded only to the
    contract's fixed width; downstream catalog lineage validation must still
    prove that the resulting identifier exists and belongs to the selected
    event.  Explicit
    ``horizontal_strategy=original`` has conservative precedence: local
    normalization disables contradictory zoom and tracking focus rather than
    promoting a non-tracking choice into executable tracking.
    """

    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise ValueError("feature planner output must be a JSON object")
    changes: list[dict[str, Any]] = []
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), changes
    for chapter_index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        candidates = chapter.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            base = f"$.chapters[{chapter_index}].candidates[{candidate_index}]"
            frame_id = candidate.get("frame_id")
            if (
                isinstance(frame_id, str)
                and frame_id.startswith("RF")
                and frame_id[2:].isdigit()
                and 1 <= len(frame_id[2:]) < 6
            ):
                normalized_frame_id = f"RF{int(frame_id[2:]):06d}"
                candidate["frame_id"] = normalized_frame_id
                changes.append(
                    {
                        "json_path": f"{base}.frame_id",
                        "before": frame_id,
                        "after": normalized_frame_id,
                        "rule": "fixed_width_rf_identifier_zero_padding",
                    }
                )
            strategy = candidate.get("horizontal_strategy")
            zoom = candidate.get("horizontal_zoom_intent")
            focus = candidate.get("horizontal_focus_entity_id")
            if strategy == "original" and zoom in {"subtle", "detail"}:
                candidate["horizontal_zoom_intent"] = "none"
                changes.append(
                    {
                        "json_path": f"{base}.horizontal_zoom_intent",
                        "before": zoom,
                        "after": "none",
                        "rule": "explicit_original_strategy_disables_zoom",
                    }
                )
            if strategy == "original" and focus is not None:
                candidate["horizontal_focus_entity_id"] = None
                changes.append(
                    {
                        "json_path": f"{base}.horizontal_focus_entity_id",
                        "before": focus,
                        "after": None,
                        "rule": "explicit_original_strategy_has_no_focus_entity",
                    }
                )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), changes


def canonicalize_direct_video_edit_plan_output(
    output_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply conservative, domain-neutral precedence to the compact plan.

    The model chooses ranks, entity roles, and attention intent. Local
    canonicalization only removes contradictory duplicate classifications and
    defaults an omitted 16:9 decision to the unmodified source composition.
    It deliberately preserves phase-local anchors: global coverage means that
    every required entity must be observed somewhere, not that every entity
    must remain visible in every phase.
    """

    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise ValueError("direct-video planner output must be a JSON object")
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), []
    changes: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict) or chapter.get("evidence_status") == "not_found":
            continue
        flow = chapter.get("flow_intent")
        if isinstance(flow, dict):
            sync_keys = (
                "visual_sync_event",
                "visual_sync_predicate",
                "music_target",
            )
            sync_values = [flow.get(key) for key in sync_keys]
            if any(value is None for value in sync_values) and any(
                value is not None for value in sync_values
            ):
                before = {key: flow.get(key) for key in sync_keys}
                for key in sync_keys:
                    flow[key] = None
                changes.append(
                    {
                        "json_path": f"chapters[{chapter_index}].flow_intent",
                        "before": before,
                        "after": {key: None for key in sync_keys},
                        "rule": (
                            "incomplete_optional_visual_sync_is_removed_"
                            "rather_than_invented"
                        ),
                    }
                )
        vertical = chapter.get("vertical")
        if not isinstance(vertical, dict):
            continue
        base = f"chapters[{chapter_index}]"
        if (
            vertical.get("strategy") == "tracked_crop"
            and vertical.get("allow_controlled_clip") is True
            and vertical.get("crop_mode") == "strict"
        ):
            changes.append(
                {
                    "json_path": f"{base}.vertical.crop_mode",
                    "before": "strict",
                    "after": "primary_center",
                    "rule": (
                        "explicit_controlled_clip_uses_primary_center_"
                        "representation"
                    ),
                }
            )
            vertical["crop_mode"] = "primary_center"
        if not isinstance(chapter.get("horizontal"), dict):
            fallback_rank = vertical.get("candidate_rank")
            chapter["horizontal"] = {
                "candidate_rank": fallback_rank,
                "strategy": "original",
                "zoom_intent": "none",
                "camera_intent": "hold",
                "focus_entity_index": None,
            }
            changes.append(
                {
                    "json_path": f"{base}.horizontal",
                    "before": None,
                    "after": chapter["horizontal"],
                    "rule": "missing_horizontal_uses_source_hold",
                }
            )
        required_before = list(vertical.get("required_entity_indices") or [])
        preferred_before = list(vertical.get("preferred_entity_indices") or [])
        sacrificable_before = list(
            vertical.get("sacrificable_entity_indices") or []
        )
        required = list(dict.fromkeys(required_before))
        preferred = [
            index
            for index in dict.fromkeys(preferred_before)
            if index not in required
        ]
        sacrificable = [
            index
            for index in dict.fromkeys(sacrificable_before)
            if index not in required and index not in preferred
        ]
        sequence = vertical.get("attention_sequence")
        if not isinstance(sequence, list):
            sequence = []
            vertical["attention_sequence"] = sequence
        if vertical.get("strategy") == "fit_with_background" and sequence:
            changes.append(
                {
                    "json_path": f"{base}.vertical.attention_sequence",
                    "before": sequence,
                    "after": [],
                    "rule": "fit_with_background_has_no_virtual_camera",
                }
            )
            sequence = []
            vertical["attention_sequence"] = []
        for step_index, step in enumerate(sequence):
            if not isinstance(step, dict):
                continue
            anchors_before = list(step.get("anchor_entity_indices") or [])
            anchors = list(dict.fromkeys(anchors_before))
            step["anchor_entity_indices"] = anchors
            for index in anchors:
                if index not in required and index not in preferred:
                    preferred.append(index)
                if index in sacrificable:
                    sacrificable.remove(index)
            if anchors != anchors_before:
                changes.append(
                    {
                        "json_path": (
                            f"{base}.vertical.attention_sequence"
                            f"[{step_index}].anchor_entity_indices"
                        ),
                        "before": anchors_before,
                        "after": anchors,
                        "rule": "phase_anchor_deduplication_without_global_injection",
                    }
                )
        normalized_roles = {
            "required_entity_indices": required,
            "preferred_entity_indices": preferred,
            "sacrificable_entity_indices": sacrificable,
        }
        for field, after in normalized_roles.items():
            before = list(vertical.get(field) or [])
            vertical[field] = after
            if before != after:
                changes.append(
                    {
                        "json_path": f"{base}.vertical.{field}",
                        "before": before,
                        "after": after,
                        "rule": "entity_role_precedence_required_preferred_sacrificable",
                    }
                )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), changes


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_feature_normalization_artifacts(
    *,
    output_dir: Path,
    artifact_stem: str,
    raw_output_path: Path,
    raw_output_text: str,
    direct_video_plan: bool = False,
) -> tuple[str, Path, Path]:
    canonicalizer = (
        canonicalize_direct_video_edit_plan_output
        if direct_video_plan
        else canonicalize_feature_plan_output
    )
    canonical_text, changes = canonicalizer(raw_output_text)
    canonical_path = output_dir / f"{artifact_stem}.canonical_output.json"
    audit_path = output_dir / f"{artifact_stem}.normalization-audit.json"
    write_json(canonical_path, {"output_text": canonical_text})
    write_json(
        audit_path,
        {
            "contract_version": FEATURE_PLAN_NORMALIZATION_VERSION,
            "interpretation": (
                "compact_direct_video_plan_conservative_precedence"
                if direct_video_plan
                else "conditional_schema_contradictions_only"
            ),
            "raw_output_path": str(raw_output_path.resolve()),
            "raw_output_artifact_sha256": sha256_file(raw_output_path),
            "input_output_text_sha256": _text_sha256(raw_output_text),
            "canonical_output_path": str(canonical_path.resolve()),
            "canonical_output_artifact_sha256": sha256_file(canonical_path),
            "canonical_output_text_sha256": _text_sha256(canonical_text),
            "changes": changes,
            "change_count": len(changes),
            "created_at": utc_now(),
        },
    )
    return canonical_text, canonical_path, audit_path


def _resolve_feature_reuse_artifacts(output_dir: Path) -> dict[str, Any]:
    """Resolve one complete, non-mixed paid-response artifact set."""

    sets = (
        {
            "kind": "canonical",
            "request": output_dir / "clip-card-feature-plan.request.json",
            "raw_output": output_dir / "clip-card-feature-plan.raw_output.json",
            "raw_interaction": output_dir / "clip-card-feature-plan.raw_interaction.json",
        },
        {
            "kind": "attempt-01",
            "request": output_dir / "clip-card-feature-plan.attempt-01.request.json",
            "raw_output": output_dir / "clip-card-feature-plan.attempt-01.raw_output.json",
            "raw_interaction": output_dir
            / "clip-card-feature-plan.attempt-01.raw_interaction.json",
        },
    )
    incomplete: list[str] = []
    for artifact_set in sets:
        paths = [artifact_set[key] for key in ("request", "raw_output", "raw_interaction")]
        present = [path.exists() for path in paths]
        if all(present):
            return artifact_set
        if any(present):
            incomplete.append(str(artifact_set["kind"]))
    detail = f"; incomplete sets: {incomplete}" if incomplete else ""
    raise FileNotFoundError(
        "--reuse-raw-output requires one complete canonical or attempt-01 "
        f"request/raw-output/raw-interaction set{detail}"
    )


def _resolve_latest_failed_feature_plan_attempt(
    output_dir: Path,
) -> dict[str, Any]:
    """Resolve the newest fully persisted failed paid attempt for one repair.

    This keeps a schema repair from paying for the original multimodal request
    again. The repair request reuses the exact saved File API URIs and appends
    only the local contract error to the original planning prompt.
    """

    attempts: list[tuple[int, dict[str, Any]]] = []
    for validation_path in output_dir.glob(
        "clip-card-feature-plan.attempt-*.schema-validation.json"
    ):
        stem = validation_path.name.removesuffix(".schema-validation.json")
        try:
            attempt_number = int(stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        request_path = output_dir / f"{stem}.request.json"
        raw_output_path = output_dir / f"{stem}.raw_output.json"
        raw_interaction_path = output_dir / f"{stem}.raw_interaction.json"
        if not all(
            path.exists()
            for path in (request_path, raw_output_path, raw_interaction_path)
        ):
            continue
        validation = read_json(validation_path)
        if validation.get("ok") is not False:
            continue
        attempts.append(
            (
                attempt_number,
                {
                    "attempt_number": attempt_number,
                    "request": request_path,
                    "raw_output": raw_output_path,
                    "raw_interaction": raw_interaction_path,
                    "schema_validation": validation_path,
                    "error": str(validation.get("error") or ""),
                },
            )
        )
    if not attempts:
        raise FileNotFoundError(
            "--resume-failed-plan requires a complete failed paid attempt"
        )
    return max(attempts, key=lambda item: item[0])[1]


def _verified_feature_raw_output_text(
    *, raw_output: dict[str, Any], raw_interaction: dict[str, Any]
) -> str:
    """Return a paid response only when both independently saved copies agree."""

    output_text = raw_output.get("output_text")
    interaction_text = raw_interaction.get("output_text")
    if not isinstance(output_text, str) or not isinstance(interaction_text, str):
        raise ValueError(
            "--reuse-raw-output requires string output_text in both raw artifacts"
        )
    if output_text != interaction_text:
        raise ValueError(
            "--reuse-raw-output artifact mismatch: raw interaction output_text "
            "does not exactly match raw output output_text"
        )
    return output_text


def _assert_fresh_feature_namespace_empty(output_dir: Path) -> None:
    existing = sorted(output_dir.glob("clip-card-feature-plan*"))
    if existing:
        raise FileExistsError(
            "fresh feature planning refuses an existing paid artifact namespace; "
            "use --reuse-raw-output or a new output directory: "
            + ", ".join(path.name for path in existing[:8])
        )


def _assert_projection_request_hash(
    *, pointer_path: Path, plan_dir: Path, expected_request_path: Path
) -> None:
    pointer = read_json(pointer_path)
    record = read_json(plan_dir / str(pointer["record_path"]))
    expected = sha256_file(expected_request_path)
    if record.get("source_request_sha256") != expected:
        raise RuntimeError(
            "external projection source request does not match the original paid request"
        )


def mmss(milliseconds: int) -> str:
    total = max(0, milliseconds // 1000)
    return f"{total // 60:02d}:{total % 60:02d}"


def planning_candidate_slice(
    candidates: list[Any],
    *,
    direct_video_evidence: bool,
    depth: int,
) -> list[Any]:
    """Return exactly the candidates the paid planning call may select.

    Text-only planning can use the complete validated shortlist. Once direct
    bounded videos are enabled, the selectable set must equal the videos
    actually attached to the request.
    """

    if direct_video_evidence:
        return candidates[:depth]
    return candidates


def planning_candidate_id(rank: int) -> str:
    """Return the local, chapter-scoped ID Gemini must copy verbatim."""

    if rank < 1:
        raise ValueError("candidate rank must be positive")
    return f"rank-{rank:02d}"


def validate_candidate_video_budget(
    *,
    total_duration_ms: int,
    maximum_duration_ms: int,
) -> None:
    """Fail before File API upload or a paid planning interaction."""

    if total_duration_ms > maximum_duration_ms:
        raise ValueError(
            "candidate direct-video budget exceeded before upload or paid "
            f"planning: {total_duration_ms / 1000:.1f}s > "
            f"{maximum_duration_ms / 1000:.1f}s"
        )


def project_direct_video_edit_plan(
    plan: DirectVideoEditPlan,
    *,
    shortlist: FeatureShortlistPlan,
    candidate_depth: int,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
    provenance: ModelProvenance,
) -> ClipCardFeaturePlanV3:
    """Resolve integer candidate ranks into the existing local renderer plan.

    Gemini never generates candidate IDs, asset IDs, event IDs, frame IDs, or
    geometry. Those values are copied or selected locally from validated
    shortlist and catalog artifacts.
    """

    capability_catalog = simple_production_capability_catalog()
    if plan.capability_catalog_sha256 != capability_catalog.definition_sha256():
        raise ValueError(
            "direct-video plan capability catalog differs from the local "
            "executable capability set"
        )
    expected_indices = list(range(1, len(brief.chapters) + 1))
    if [chapter.chapter_index for chapter in plan.chapters] != expected_indices:
        raise ValueError(
            "direct-video plan must preserve every brief chapter index in order"
        )
    catalog_clips = {clip.clip_id: clip for clip in catalog.clips}

    def event_frame_id(source_asset_id: str, event_id: str) -> str:
        card = cards[source_asset_id]
        event = next(item for item in card.events if item.event_id == event_id)
        minute, second = (int(part) for part in event.recommended_keyframe_mmss.split(":"))
        requested_ms = (minute * 60 + second) * 1000
        candidates = [
            frame
            for frame in catalog.frames
            if (
                f"sha256:{catalog_clips[frame.clip_id].sha256}" == source_asset_id
                and event.start_mmss
                <= mmss(frame.requested_time_ms)
                < event.end_mmss
            )
        ]
        if not candidates:
            raise ValueError(
                f"no catalog frame inside selected event: {source_asset_id}/{event_id}"
            )
        return min(
            candidates,
            key=lambda frame: (
                abs(frame.requested_time_ms - requested_ms),
                frame.requested_time_ms,
                frame.frame_id,
            ),
        ).frame_id

    def event_entity_ids(source_asset_id: str, event_id: str) -> list[str]:
        card = cards[source_asset_id]
        event = next(item for item in card.events if item.event_id == event_id)
        event_ids = set(
            event.entity_ids
            + event.primary_entity_ids
            + event.required_entity_ids
            + event.optional_entity_ids
            + event.avoid_overlay_entity_ids
            + [target.entity_id for target in event.grounding_targets]
        )
        return [
            entity.entity_id
            for entity in card.entities
            if entity.entity_id in event_ids
        ]

    def resolve_entity_indices(
        *,
        source_asset_id: str,
        event_id: str,
        indices: list[int],
    ) -> list[str]:
        ids = event_entity_ids(source_asset_id, event_id)
        unknown = [index for index in indices if index > len(ids)]
        if unknown:
            raise ValueError(
                f"direct-video plan selected unseen entity indices: {unknown}"
            )
        return [ids[index - 1] for index in indices]

    def camera_proposal(
        decision: DirectVideoVerticalDecision,
        *,
        source_asset_id: str,
        event_id: str,
    ) -> ClipCardVirtualCameraProposalV1 | None:
        if not decision.attention_sequence:
            return None
        unique_entities = {
            entity_id
            for step in decision.attention_sequence
            for entity_id in step.anchor_entity_indices
        }
        if decision.coverage_mode == "simultaneous":
            composition_mode = "joint_relation"
        elif decision.coverage_mode == "sequential":
            composition_mode = "sequential_focus"
        elif len(unique_entities) == 1:
            composition_mode = (
                "single_anchor_hold"
                if all(
                    step.camera_behavior == "hold"
                    for step in decision.attention_sequence
                )
                else "single_anchor_follow"
            )
        elif len(decision.attention_sequence) >= 2:
            composition_mode = "sequential_focus"
        else:
            composition_mode = "joint_relation"
        phases = []
        for index, step in enumerate(decision.attention_sequence, start=1):
            transition_in = (
                "cut"
                if (
                    index == 1
                    or step.camera_behavior == "punch_in_cut"
                    or step.transition_preference == "cut"
                )
                else "smoothstep"
            )
            phases.append(
                ClipCardVirtualCameraPhaseV1(
                    phase_id=f"phase-{index:02d}",
                    start_progress=step.start_progress,
                    end_progress=step.end_progress,
                    anchor_entity_ids=resolve_entity_indices(
                        source_asset_id=source_asset_id,
                        event_id=event_id,
                        indices=step.anchor_entity_indices,
                    ),
                    camera_behavior=step.camera_behavior,
                    transition_in=transition_in,
                    transition_duration_fraction=(
                        0.0 if transition_in == "cut" else 0.15
                    ),
                    observable_predicate=(
                        "The attached bounded video directly shows the listed "
                        "anchor entities in this relative phase."
                    ),
                    transition_condition=(
                        "Advance at the locally resolved boundary between "
                        "contiguous relative attention phases."
                    ),
                    editorial_reason=decision.framing_intent,
                )
            )
        return ClipCardVirtualCameraProposalV1(
            composition_mode=composition_mode,
            phases=phases,
            proposal_reason=decision.framing_intent,
            uncertainties=[],
        )

    chapters: list[ClipCardFeatureSelectV3] = []
    for chapter_offset, direct_chapter in enumerate(plan.chapters):
        brief_chapter = brief.chapters[chapter_offset]
        shortlist_chapter = shortlist.chapters[chapter_offset]
        if shortlist_chapter.feature_id != brief_chapter.feature_id:
            raise ValueError("shortlist and brief chapter order differ")
        allowed = planning_candidate_slice(
            shortlist_chapter.candidates,
            direct_video_evidence=True,
            depth=candidate_depth,
        )
        if direct_chapter.evidence_status == "not_found":
            chapters.append(
                ClipCardFeatureSelectV3(
                    feature_id=brief_chapter.feature_id,
                    evidence_status="not_found",
                    candidates=[],
                    horizontal_candidate_id=None,
                    vertical_candidate_id=None,
                )
            )
            continue
        assert direct_chapter.horizontal is not None
        assert direct_chapter.vertical is not None
        selected_ranks = {
            direct_chapter.horizontal.candidate_rank,
            direct_chapter.vertical.candidate_rank,
        }
        if any(rank > len(allowed) for rank in selected_ranks):
            raise ValueError(
                f"direct-video plan selected unseen rank for "
                f"{brief_chapter.feature_id}: {sorted(selected_ranks)}"
            )
        candidates: list[ClipCardFeatureCandidateV3] = []
        for rank, shortlisted in enumerate(allowed, start=1):
            card = cards[shortlisted.source_asset_id]
            event = next(
                item for item in card.events if item.event_id == shortlisted.event_id
            )
            horizontal_strategy: Literal["original", "tracked_reframe"] = "original"
            horizontal_zoom_intent: Literal["none", "subtle", "detail"] = "none"
            horizontal_camera_intent: VirtualCameraIntent = "hold"
            horizontal_focus_entity_id: str | None = None
            vertical_strategy: Literal[
                "tracked_crop", "fit_with_background"
            ] = "fit_with_background"
            vertical_crop_mode: Literal["strict", "primary_center"] = (
                "primary_center"
            )
            framing_intent = "Preserve the validated source composition."
            required_entity_ids: list[str] = []
            preferred_entity_ids: list[str] = []
            sacrificable_entity_ids: list[str] = []
            virtual_camera_proposal = None
            if rank == direct_chapter.horizontal.candidate_rank:
                horizontal_strategy = direct_chapter.horizontal.strategy
                horizontal_zoom_intent = direct_chapter.horizontal.zoom_intent
                horizontal_camera_intent = direct_chapter.horizontal.camera_intent
                horizontal_focus_entity_id = (
                    resolve_entity_indices(
                        source_asset_id=shortlisted.source_asset_id,
                        event_id=shortlisted.event_id,
                        indices=[
                            direct_chapter.horizontal.focus_entity_index
                        ],
                    )[0]
                    if direct_chapter.horizontal.focus_entity_index is not None
                    else None
                )
            if rank == direct_chapter.vertical.candidate_rank:
                vertical_strategy = direct_chapter.vertical.strategy
                vertical_crop_mode = direct_chapter.vertical.crop_mode
                framing_intent = direct_chapter.vertical.framing_intent
                required_entity_ids = resolve_entity_indices(
                    source_asset_id=shortlisted.source_asset_id,
                    event_id=shortlisted.event_id,
                    indices=direct_chapter.vertical.required_entity_indices,
                )
                preferred_entity_ids = resolve_entity_indices(
                    source_asset_id=shortlisted.source_asset_id,
                    event_id=shortlisted.event_id,
                    indices=direct_chapter.vertical.preferred_entity_indices,
                )
                sacrificable_entity_ids = resolve_entity_indices(
                    source_asset_id=shortlisted.source_asset_id,
                    event_id=shortlisted.event_id,
                    indices=direct_chapter.vertical.sacrificable_entity_indices,
                )
                virtual_camera_proposal = camera_proposal(
                    direct_chapter.vertical,
                    source_asset_id=shortlisted.source_asset_id,
                    event_id=shortlisted.event_id,
                )
            candidates.append(
                ClipCardFeatureCandidateV3(
                    candidate_id=planning_candidate_id(rank),
                    source_asset_id=shortlisted.source_asset_id,
                    event_id=shortlisted.event_id,
                    frame_id=event_frame_id(
                        shortlisted.source_asset_id,
                        shortlisted.event_id,
                    ),
                    observed_visual_evidence=event.observable_evidence,
                    selection_reason=shortlisted.retrieval_reason,
                    quality_risks=event.quality_risks,
                    horizontal_strategy=horizontal_strategy,
                    horizontal_zoom_intent=horizontal_zoom_intent,
                    horizontal_camera_intent=horizontal_camera_intent,
                    horizontal_focus_entity_id=horizontal_focus_entity_id,
                    vertical_strategy=vertical_strategy,
                    vertical_crop_mode=vertical_crop_mode,
                    coverage_mode=(
                        direct_chapter.vertical.coverage_mode
                        if rank == direct_chapter.vertical.candidate_rank
                        else "simultaneous"
                    ),
                    allow_controlled_clip=(
                        direct_chapter.vertical.allow_controlled_clip
                        if rank == direct_chapter.vertical.candidate_rank
                        else False
                    ),
                    framing_intent=framing_intent,
                    required_entity_ids=required_entity_ids,
                    preferred_entity_ids=preferred_entity_ids,
                    sacrificable_entity_ids=sacrificable_entity_ids,
                    virtual_camera_proposal=virtual_camera_proposal,
                    confidence=direct_chapter.confidence,
                )
            )
        chapters.append(
            ClipCardFeatureSelectV3(
                feature_id=brief_chapter.feature_id,
                evidence_status=direct_chapter.evidence_status,
                candidates=candidates,
                horizontal_candidate_id=planning_candidate_id(
                    direct_chapter.horizontal.candidate_rank
                ),
                vertical_candidate_id=planning_candidate_id(
                    direct_chapter.vertical.candidate_rank
                ),
                recommended_duration_seconds=(
                    direct_chapter.recommended_duration_seconds
                ),
                duration_rationale=(
                    f"{direct_chapter.duration_rationale} "
                    f"Narrative role={direct_chapter.flow_intent.narrative_role}; "
                    f"transition={direct_chapter.flow_intent.relation_to_previous}; "
                    f"visible sync={direct_chapter.flow_intent.visual_sync_event}."
                ),
                attention_observation=direct_chapter.attention_observation,
                flow_intent=direct_chapter.flow_intent,
            )
        )
    return ClipCardFeaturePlanV3(
        contract_version="clip-card-feature-cut-v3",
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=plan.title,
        strategy_summary=plan.strategy_summary,
        chapters=chapters,
        uncertainties=plan.uncertainties,
        model_provenance=provenance,
    )


def compact_card(card: FullClipCard) -> dict[str, object]:
    return {
        "source_asset_id": card.source_asset_id,
        "duration_ms": card.duration_ms,
        "summary": card.summary,
        "content_type": card.content_type,
        "clip_uses": card.clip_uses,
        "portrait_reframe_feasibility": card.portrait_reframe_feasibility,
        "uncertainties": card.uncertainties,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "kind": entity.kind,
                "label": entity.label,
                "distinguishing_features": entity.distinguishing_features,
                "evidence": entity.evidence,
            }
            for entity in card.entities
        ],
        "events": [
            {
                "event_id": event.event_id,
                "start_mmss": event.start_mmss,
                "end_mmss": event.end_mmss,
                "recommended_keyframe_mmss": event.recommended_keyframe_mmss,
                "label": event.label,
                "description": event.description,
                "observable_evidence": event.observable_evidence,
                "action_completeness": event.action_completeness,
                "editing_uses": event.editing_uses,
                "quality_risks": event.quality_risks,
                "framing_intent": event.framing_intent,
                "entity_ids": event.entity_ids,
                "primary_entity_ids": event.primary_entity_ids,
                "required_entity_ids": event.required_entity_ids,
                "optional_entity_ids": event.optional_entity_ids,
                "avoid_overlay_entity_ids": event.avoid_overlay_entity_ids,
                "entity_relations": [
                    {
                        "entity_id": entity_id,
                        "relations": [
                            relation
                            for relation, members in (
                                ("event_member", event.entity_ids),
                                ("primary", event.primary_entity_ids),
                                ("required", event.required_entity_ids),
                                ("optional", event.optional_entity_ids),
                                ("avoid_overlay", event.avoid_overlay_entity_ids),
                                (
                                    "grounding_target",
                                    [target.entity_id for target in event.grounding_targets],
                                ),
                            )
                            if entity_id in members
                        ],
                    }
                    for entity_id in sorted(
                        set(
                            event.entity_ids
                            + event.primary_entity_ids
                            + event.required_entity_ids
                            + event.optional_entity_ids
                            + event.avoid_overlay_entity_ids
                            + [target.entity_id for target in event.grounding_targets]
                        )
                    )
                ],
                "card_opportunities": [
                    {
                        "kind": opportunity.kind,
                        "rationale": opportunity.rationale,
                        "entity_ids": opportunity.entity_ids,
                    }
                    for opportunity in event.card_opportunities
                ],
                "grounding_targets": [
                    {
                        "entity_id": target.entity_id,
                        "target_kind": target.target_kind,
                        "target_description": target.target_description,
                        "purpose": target.purpose,
                    }
                    for target in event.grounding_targets
                ],
            }
            for event in card.events
        ],
    }


def compact_card_v3(
    card: FullClipCard,
    supplements: tuple[ClipObservationSupplement, ...] = (),
) -> dict[str, object]:
    """Compact selection evidence without locally derivable relation mirrors.

    The model still sees every event and the entity IDs needed to choose a take,
    but it does not receive duplicated relation expansions, per-entity evidence,
    or card-layout records that are irrelevant to editorial ranking.
    """

    observations = effective_event_observations(card, supplements)
    return {
        "source_asset_id": card.source_asset_id,
        "duration_ms": card.duration_ms,
        "summary": card.summary,
        "content_type": card.content_type,
        "clip_uses": card.clip_uses,
        "portrait_reframe_feasibility": card.portrait_reframe_feasibility,
        "uncertainties": card.uncertainties,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "kind": entity.kind,
                "label": entity.label,
                "distinguishing_features": entity.distinguishing_features,
            }
            for entity in card.entities
        ],
        "events": [
            {
                "event_id": event.event_id,
                "start_mmss": event.start_mmss,
                "end_mmss": event.end_mmss,
                "recommended_keyframe_mmss": event.recommended_keyframe_mmss,
                "label": event.label,
                "description": event.description,
                "observable_evidence": event.observable_evidence,
                "action_completeness": event.action_completeness,
                "editing_uses": event.editing_uses,
                "quality_risks": event.quality_risks,
                "framing_intent": event.framing_intent,
                "entity_ids": event.entity_ids,
                "primary_entity_ids": event.primary_entity_ids,
                "required_entity_ids": event.required_entity_ids,
                "optional_entity_ids": event.optional_entity_ids,
                "avoid_overlay_entity_ids": event.avoid_overlay_entity_ids,
                "observation_capabilities": observations[
                    event.event_id
                ].capabilities.model_dump(mode="json"),
                "observable_beats": [
                    beat.model_dump(mode="json")
                    for beat in observations[event.event_id].observable_beats
                ],
                "evidence_roles": observations[
                    event.event_id
                ].evidence_roles.model_dump(mode="json"),
                "readability": [
                    item.model_dump(mode="json")
                    for item in observations[event.event_id].readability
                ],
                "audio_role": (
                    observations[event.event_id].audio_role.model_dump(mode="json")
                    if observations[event.event_id].audio_role
                    else None
                ),
                "grounding_target_entity_ids": [
                    target.entity_id for target in event.grounding_targets
                ],
            }
            for event in card.events
        ],
    }


def validate_plan_contract(
    plan: ClipCardFeaturePlanV2,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
    require_v2: bool = True,
) -> None:
    if require_v2 and plan.contract_version != "clip-card-feature-cut-v2":
        raise ValueError("new feature planning requests require the v2 Top-K contract")
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("model changed immutable project or catalog ID")
    expected_features = [chapter.feature_id for chapter in brief.chapters]
    if [chapter.feature_id for chapter in plan.chapters] != expected_features:
        raise ValueError("plan must preserve every brief chapter exactly once and in order")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}

    def validate_evidence_reference(
        *,
        asset_id: str,
        event_id: str,
        frame_id: str,
    ) -> tuple[FullClipCard, Any]:
        card = cards.get(asset_id)
        if card is None:
            raise ValueError(f"unknown selected asset: {asset_id}")
        event = next((item for item in card.events if item.event_id == event_id), None)
        if event is None:
            raise ValueError(f"unknown selected event: {asset_id}/{event_id}")
        frame = frames.get(frame_id)
        if frame is None:
            raise ValueError(f"unknown selected frame: {frame_id}")
        selected_clip = clips.get(frame.clip_id)
        if selected_clip is None or f"sha256:{selected_clip.sha256}" != asset_id:
            raise ValueError(f"frame does not belong to selected asset: {frame_id}")
        frame_mmss = mmss(frame.requested_time_ms)
        if not event.start_mmss <= frame_mmss < event.end_mmss:
            raise ValueError(f"frame lies outside selected event: {frame_id}")
        return card, event

    relation_fields = {
        "event_member": "entity_ids",
        "primary": "primary_entity_ids",
        "required": "required_entity_ids",
        "optional": "optional_entity_ids",
        "avoid_overlay": "avoid_overlay_entity_ids",
    }

    def validate_region_lineage(
        *, candidate: ClipCardFeatureCandidate, card: FullClipCard, event: Any
    ) -> None:
        known_entities = {entity.entity_id for entity in card.entities}
        grounding_entities = {target.entity_id for target in event.grounding_targets}
        for region in candidate.resolved_regions:
            for ref in region.entity_refs:
                if ref.entity_id not in known_entities:
                    raise ValueError(
                        f"candidate region references unknown entity: {candidate.candidate_id}/"
                        f"{region.region_id}/{ref.entity_id}"
                    )
                if ref.event_relation == "grounding_target":
                    valid_relation = ref.entity_id in grounding_entities
                else:
                    valid_relation = ref.entity_id in getattr(
                        event, relation_fields[ref.event_relation]
                    )
                if not valid_relation:
                    raise ValueError(
                        f"candidate region relation is not backed by its event: "
                        f"{candidate.candidate_id}/{region.region_id}/{ref.entity_id}/"
                        f"{ref.event_relation}"
                    )

    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        # A brief target states editorial priority, not the geometry algorithm.
        # Local preflight may legitimately choose a stable fit strategy when a
        # moving crop is unnecessary or cannot preserve the required extent.
        triples = (
            (
                chapter.horizontal_source_asset_id,
                chapter.horizontal_event_id,
                chapter.horizontal_frame_id,
            ),
            (
                chapter.vertical_source_asset_id,
                chapter.vertical_event_id,
                chapter.vertical_frame_id,
            ),
        )
        for asset_id, event_id, frame_id in triples:
            assert asset_id is not None and event_id is not None and frame_id is not None
            validate_evidence_reference(
                asset_id=asset_id, event_id=event_id, frame_id=frame_id
            )
        for candidate in chapter.candidates:
            card, event = validate_evidence_reference(
                asset_id=candidate.source_asset_id,
                event_id=candidate.event_id,
                frame_id=candidate.frame_id,
            )
            validate_region_lineage(candidate=candidate, card=card, event=event)


def validate_plan_contract_v3(
    plan: ClipCardFeaturePlanV3,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
    supplements: dict[str, list[ClipObservationSupplement]] | None = None,
    direct_video_observed_events: set[tuple[str, str]] | None = None,
) -> None:
    """Validate model choices while deriving no semantic values from the model."""

    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("model changed immutable project or catalog ID")
    expected_features = [chapter.feature_id for chapter in brief.chapters]
    if [chapter.feature_id for chapter in plan.chapters] != expected_features:
        raise ValueError("plan must preserve every brief chapter exactly once and in order")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    supplements = supplements or {}
    direct_video_observed_events = direct_video_observed_events or set()
    observations_by_asset = {
        asset_id: effective_event_observations(
            card,
            supplements.get(asset_id, ()),
        )
        for asset_id, card in cards.items()
    }
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        for candidate in chapter.candidates:
            card = cards.get(candidate.source_asset_id)
            if card is None:
                raise ValueError(f"unknown selected asset: {candidate.source_asset_id}")
            event = next(
                (item for item in card.events if item.event_id == candidate.event_id), None
            )
            if event is None:
                raise ValueError(
                    f"unknown selected event: {candidate.source_asset_id}/{candidate.event_id}"
                )
            frame = frames.get(candidate.frame_id)
            if frame is None:
                raise ValueError(f"unknown selected frame: {candidate.frame_id}")
            clip = clips.get(frame.clip_id)
            if clip is None or f"sha256:{clip.sha256}" != candidate.source_asset_id:
                raise ValueError(
                    f"frame does not belong to selected asset: {candidate.frame_id}"
                )
            frame_mmss = mmss(frame.requested_time_ms)
            if not event.start_mmss <= frame_mmss < event.end_mmss:
                raise ValueError(f"frame lies outside selected event: {candidate.frame_id}")
            event_entities = set(
                event.entity_ids
                + event.primary_entity_ids
                + event.required_entity_ids
                + event.optional_entity_ids
                + event.avoid_overlay_entity_ids
                + [target.entity_id for target in event.grounding_targets]
            )
            selected_entities = set(
                candidate.required_entity_ids
                + candidate.preferred_entity_ids
                + candidate.sacrificable_entity_ids
            )
            if candidate.horizontal_focus_entity_id:
                selected_entities.add(candidate.horizontal_focus_entity_id)
            unknown = sorted(selected_entities - event_entities)
            if unknown:
                raise ValueError(
                    f"candidate focus entities are not backed by its event: "
                    f"{candidate.candidate_id}/{unknown}"
                )
            proposal = candidate.virtual_camera_proposal
            if proposal is not None and proposal.composition_mode in {
                "sequential_focus",
                "joint_relation",
            }:
                observation = observations_by_asset[candidate.source_asset_id][
                    candidate.event_id
                ]
                claim = (
                    EditingClaim.SEQUENTIAL_VIRTUAL_CAMERA
                    if proposal.composition_mode == "sequential_focus"
                    else EditingClaim.SIMULTANEOUS_RELATION
                )
                decision = assess_editing_claim(observation, claim)
                direct_observed = (
                    candidate.source_asset_id,
                    candidate.event_id,
                ) in direct_video_observed_events
                if (
                    decision.decision != ClaimDecision.READY
                    and not direct_observed
                ):
                    raise ValueError(
                        "virtual camera claim lacks assessed observation evidence: "
                        f"{candidate.candidate_id}/{claim.value}/"
                        f"{decision.decision.value}/"
                        f"missing={decision.missing_capabilities}/"
                        f"unavailable={decision.unavailable_capabilities}"
                    )
                phase_entity_ids = {
                    entity_id
                    for phase in proposal.phases
                    for entity_id in phase.anchor_entity_ids
                }
                if direct_observed:
                    continue
                if proposal.composition_mode == "sequential_focus":
                    observed_entity_ids = {
                        entity_id
                        for beat in observation.observable_beats
                        for entity_id in beat.entity_ids
                    }
                    unobserved = sorted(phase_entity_ids - observed_entity_ids)
                    if unobserved:
                        raise ValueError(
                            "sequential virtual camera references anchors without "
                            f"observable beats: {candidate.candidate_id}/{unobserved}"
                        )
                    simultaneous_conflicts = [
                        beat.beat_id
                        for beat in observation.observable_beats
                        if beat.relation_mode == "simultaneous_required"
                        and len(set(beat.entity_ids) & phase_entity_ids) >= 2
                    ]
                    if simultaneous_conflicts:
                        raise ValueError(
                            "sequential virtual camera would split a simultaneous "
                            "evidence obligation: "
                            f"{candidate.candidate_id}/{simultaneous_conflicts}"
                        )
                else:
                    simultaneous_sets = [
                        set(beat.entity_ids)
                        for beat in observation.observable_beats
                        if beat.relation_mode == "simultaneous_required"
                    ]
                    required = set(candidate.required_entity_ids) or phase_entity_ids
                    if not any(
                        required <= entity_ids for entity_ids in simultaneous_sets
                    ):
                        raise ValueError(
                            "joint relation lacks one observable beat containing "
                            f"all required entities: {candidate.candidate_id}"
                        )
            # Clip Card primary/required roles describe the source event, not
            # an immutable crop contract for every downstream brief.  A
            # brief-specific candidate may intentionally focus on a subset.
            # Only explicitly classified entities become geometry contracts;
            # omitted event entities are neither silently required nor
            # silently marked sacrificable.


def build_selected_clip_card_evidence(
    plan: ClipCardFeaturePlanV3,
    *,
    cards: dict[str, FullClipCard],
) -> SelectedClipCardEvidence:
    """Snapshot only locally validated events referenced by the v3 source plan."""

    keys = sorted(
        {
            (candidate.source_asset_id, candidate.event_id)
            for chapter in plan.chapters
            for candidate in chapter.candidates
        }
    )
    events: list[SelectedEvidenceEvent] = []
    for asset_id, event_id in keys:
        card = cards.get(asset_id)
        if card is None:
            raise ValueError(f"cannot snapshot unknown asset: {asset_id}")
        event = next((item for item in card.events if item.event_id == event_id), None)
        if event is None:
            raise ValueError(f"cannot snapshot unknown event: {asset_id}/{event_id}")
        referenced_ids = set(
            event.entity_ids
            + event.primary_entity_ids
            + event.required_entity_ids
            + event.optional_entity_ids
            + event.avoid_overlay_entity_ids
            + [target.entity_id for target in event.grounding_targets]
        )
        entities_by_id = {entity.entity_id: entity for entity in card.entities}
        events.append(
            SelectedEvidenceEvent(
                source_asset_id=asset_id,
                event_id=event_id,
                entity_ids=list(event.entity_ids),
                primary_entity_ids=list(event.primary_entity_ids),
                required_entity_ids=list(event.required_entity_ids),
                optional_entity_ids=list(event.optional_entity_ids),
                avoid_overlay_entity_ids=list(event.avoid_overlay_entity_ids),
                entities=[
                    SelectedEvidenceEntity(
                        entity_id=entity_id,
                        kind=entities_by_id[entity_id].kind.value,
                        label=entities_by_id[entity_id].label,
                        distinguishing_features=(
                            entities_by_id[entity_id].distinguishing_features
                        ),
                    )
                    for entity_id in sorted(referenced_ids)
                ],
                grounding_targets=[
                    SelectedEvidenceGroundingTarget(
                        entity_id=target.entity_id,
                        target_description=target.target_description,
                    )
                    for target in event.grounding_targets
                ],
            )
        )
    return SelectedClipCardEvidence(
        contract_version="clip-card-feature-cut-selected-evidence-v1",
        events=events,
    )


def _selected_first_candidates_v3(
    chapter: ClipCardFeatureSelectV3, selected_candidate_id: str | None
) -> list[ClipCardFeatureCandidateV3]:
    if not chapter.candidates:
        return []
    if selected_candidate_id is None:
        raise ValueError("candidate contract is missing its selected candidate ID")
    selected = next(
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id == selected_candidate_id
    )
    return [selected] + [
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id != selected_candidate_id
    ]


def _region_kind(entity_kind: str) -> Literal[
    "subject", "text_region", "ui_region", "graphic", "other"
]:
    if entity_kind == "text_region":
        return "text_region"
    if entity_kind in {"phone_screen", "screen", "ui_element"}:
        return "ui_region"
    if entity_kind == "logo":
        return "graphic"
    if entity_kind == "other":
        return "other"
    return "subject"


def _safe_region_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-.") or "region"


def _event_index(
    evidence: SelectedClipCardEvidence,
) -> dict[tuple[str, str], SelectedEvidenceEvent]:
    index: dict[tuple[str, str], SelectedEvidenceEvent] = {}
    for event in evidence.events:
        key = (event.source_asset_id, event.event_id)
        if key in index:
            raise ValueError(f"duplicate selected evidence event: {key}")
        index[key] = event
    return index


def _target_description(event: SelectedEvidenceEvent, entity_id: str) -> str:
    target = next(
        (item for item in event.grounding_targets if item.entity_id == entity_id), None
    )
    if target is not None:
        return target.target_description
    entity = next((item for item in event.entities if item.entity_id == entity_id), None)
    if entity is None:
        raise ValueError(
            f"selected evidence is missing entity {event.source_asset_id}/"
            f"{event.event_id}/{entity_id}"
        )
    details = entity.distinguishing_features.strip()
    return f"{entity.label}; {details}" if details else entity.label


def _observable_entity_relations(
    event: SelectedEvidenceEvent, entity_id: str
) -> list[str]:
    relations = [
        relation
        for relation, members in (
            ("event_member", event.entity_ids),
            ("primary", event.primary_entity_ids),
            ("required", event.required_entity_ids),
            ("optional", event.optional_entity_ids),
            ("avoid_overlay", event.avoid_overlay_entity_ids),
            ("grounding_target", [item.entity_id for item in event.grounding_targets]),
        )
        if entity_id in members
    ]
    return [f"event_relation={relation}" for relation in relations]


def _project_candidate_regions_v3(
    candidate: ClipCardFeatureCandidateV3,
    event: SelectedEvidenceEvent,
) -> list[FramingRegionIntent]:
    if candidate.vertical_strategy != "tracked_crop":
        return []
    hard_ids = list(candidate.required_entity_ids)
    preferred_ids = [
        entity_id
        for entity_id in candidate.preferred_entity_ids
        if entity_id not in hard_ids
    ]
    overlay_ids = [
        entity_id
        for entity_id in event.avoid_overlay_entity_ids
        if entity_id not in hard_ids and entity_id not in preferred_ids
        and entity_id not in candidate.sacrificable_entity_ids
    ]
    roles = [
        *( (entity_id, "required") for entity_id in hard_ids ),
        *( (entity_id, "preferred") for entity_id in preferred_ids ),
        *( (entity_id, "avoid_overlay") for entity_id in overlay_ids ),
    ]
    if len(roles) > 8:
        raise ValueError(
            f"locally derived crop contract exceeds eight regions: {candidate.candidate_id}"
        )
    entities = {entity.entity_id: entity for entity in event.entities}
    projected: list[FramingRegionIntent] = []
    for entity_id, role in roles:
        entity = entities.get(entity_id)
        if entity is None:
            raise ValueError(
                f"selected evidence is missing crop entity: {candidate.candidate_id}/"
                f"{entity_id}"
            )
        kind = _region_kind(entity.kind)
        atomic = role == "required" and kind in {
            "text_region",
            "ui_region",
            "graphic",
        }
        projected.append(
            FramingRegionIntent(
                region_id=(
                    f"{_safe_region_token(candidate.candidate_id)}."
                    f"{_safe_region_token(role)}.{_safe_region_token(entity_id)}"
                ),
                entity_id=entity_id,
                target_description=_target_description(event, entity_id),
                kind=kind,
                role=role,
                atomic=atomic,
                minimum_visible_fraction=1.0 if role == "required" else None,
                observable_relations=list(
                    dict.fromkeys(
                        _observable_entity_relations(event, entity_id)
                        + [f"editorial_framing_intent={candidate.framing_intent}"]
                    )
                ),
                exclusions=[],
            )
        )
    return projected


def _project_candidate_virtual_camera_v3(
    candidate: ClipCardFeatureCandidateV3,
    regions: list[FramingRegionIntent],
) -> VerticalVirtualCameraProposal | None:
    proposal = candidate.virtual_camera_proposal
    if proposal is None:
        return None
    region_by_entity = {
        region.entity_id: region.region_id
        for region in regions
        if region.entity_id is not None
        and region.execution_role != "overlay_keepout"
    }
    return VerticalVirtualCameraProposal(
        composition_mode=proposal.composition_mode,
        phases=[
            VerticalVirtualCameraProposalPhase(
                phase_id=phase.phase_id,
                start_progress=phase.start_progress,
                end_progress=phase.end_progress,
                anchor_region_ids=[
                    region_by_entity[entity_id]
                    for entity_id in phase.anchor_entity_ids
                ],
                camera_behavior=phase.camera_behavior,
                transition_in=phase.transition_in,
                transition_duration_fraction=phase.transition_duration_fraction,
                observable_predicate=phase.observable_predicate,
                transition_condition=phase.transition_condition,
                editorial_reason=phase.editorial_reason,
            )
            for phase in proposal.phases
        ],
        proposal_reason=proposal.proposal_reason,
        uncertainties=proposal.uncertainties,
    )


def project_feature_contracts_v3(
    plan: ClipCardFeaturePlanV3,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    selected_evidence: SelectedClipCardEvidence,
) -> FeatureEditPlan:
    """Project v3 editorial choices using only hash-bound local Clip Cards."""

    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("source plan differs from projection catalog/brief")
    index = _event_index(selected_evidence)
    projected: list[FeatureChapterSelect] = []
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            projected.append(
                FeatureChapterSelect(
                    feature_id=chapter.feature_id,
                    evidence_status="not_found",
                    horizontal_frame_id=None,
                    vertical_frame_id=None,
                    observed_visual_evidence="No supported Clip Card evidence was selected.",
                    selection_reason="The evidence-bound planner returned not_found.",
                    horizontal_strategy="original",
                    horizontal_zoom_intent="none",
                    horizontal_target_description=None,
                    vertical_strategy="fit_with_background",
                    vertical_target_description=None,
                    quality_risks=["No supported source evidence."],
                    confidence=0.0,
                )
            )
            continue
        horizontal_options = _selected_first_candidates_v3(
            chapter, chapter.horizontal_candidate_id
        )
        vertical_options = _selected_first_candidates_v3(
            chapter, chapter.vertical_candidate_id
        )
        horizontal_primary = horizontal_options[0]
        vertical_primary = vertical_options[0]

        def evidence_event(candidate: ClipCardFeatureCandidateV3) -> SelectedEvidenceEvent:
            event = index.get((candidate.source_asset_id, candidate.event_id))
            if event is None:
                raise ValueError(
                    f"selected evidence artifact is missing candidate event: "
                    f"{candidate.candidate_id}"
                )
            return event

        def horizontal_target(candidate: ClipCardFeatureCandidateV3) -> str | None:
            if candidate.horizontal_focus_entity_id is None:
                return None
            return _target_description(
                evidence_event(candidate), candidate.horizontal_focus_entity_id
            )

        def vertical_target(candidate: ClipCardFeatureCandidateV3) -> str | None:
            if not candidate.required_entity_ids:
                return None
            event = evidence_event(candidate)
            descriptions = [
                _target_description(event, entity_id)
                for entity_id in candidate.required_entity_ids
            ]
            return " | ".join(descriptions)

        def projected_vertical_candidate(
            candidate: ClipCardFeatureCandidateV3,
            rank: int,
        ) -> FeatureVerticalCandidate:
            regions = _project_candidate_regions_v3(
                candidate,
                evidence_event(candidate),
            )
            return FeatureVerticalCandidate(
                candidate_id=candidate.candidate_id,
                rank=rank,
                source_asset_id=candidate.source_asset_id,
                event_id=candidate.event_id,
                frame_id=candidate.frame_id,
                observed_visual_evidence=candidate.observed_visual_evidence,
                selection_reason=candidate.selection_reason,
                strategy=candidate.vertical_strategy,
                crop_mode=candidate.vertical_crop_mode,
                coverage_mode=candidate.coverage_mode,
                allow_controlled_clip=candidate.allow_controlled_clip,
                target_description=vertical_target(candidate),
                regions=regions,
                virtual_camera_proposal=_project_candidate_virtual_camera_v3(
                    candidate,
                    regions,
                ),
                quality_risks=candidate.quality_risks,
                confidence=candidate.confidence,
            )

        horizontal_primary_target = horizontal_target(horizontal_primary)
        vertical_primary_target = vertical_target(vertical_primary)
        observed = horizontal_primary.observed_visual_evidence
        reason = horizontal_primary.selection_reason
        if horizontal_primary.candidate_id != vertical_primary.candidate_id:
            observed = (
                f"16:9: {observed} 9:16: {vertical_primary.observed_visual_evidence}"
            )
            reason = f"16:9: {reason} 9:16: {vertical_primary.selection_reason}"
        quality_risks = list(
            dict.fromkeys(horizontal_primary.quality_risks + vertical_primary.quality_risks)
        )
        if vertical_primary.coverage_mode == "sequential":
            vertical_coverage_intent = "sequential_attention"
        elif len(vertical_primary.required_entity_ids) >= 2:
            # A primary-with-context plan can still declare more than one
            # evidence-bearing hard entity. Preserve that obligation rather
            # than silently weakening it to single-primary.
            vertical_coverage_intent = "simultaneous_relation"
        elif vertical_primary.coverage_mode in {
            "primary_with_context",
            "independent_detail",
        }:
            vertical_coverage_intent = "single_primary"
        else:
            vertical_coverage_intent = (
                "simultaneous_relation"
                if len(vertical_primary.required_entity_ids) >= 2
                else "single_primary"
            )
        projected.append(
            FeatureChapterSelect(
                feature_id=chapter.feature_id,
                evidence_status=chapter.evidence_status,
                horizontal_frame_id=horizontal_primary.frame_id,
                vertical_frame_id=vertical_primary.frame_id,
                observed_visual_evidence=observed,
                selection_reason=reason,
                horizontal_strategy=horizontal_primary.horizontal_strategy,
                horizontal_zoom_intent=horizontal_primary.horizontal_zoom_intent,
                horizontal_camera_intent=(
                    horizontal_primary.horizontal_camera_intent
                ),
                horizontal_target_description=horizontal_primary_target,
                vertical_strategy=vertical_primary.vertical_strategy,
                vertical_target_description=vertical_primary_target,
                vertical_coverage_intent=vertical_coverage_intent,
                vertical_coverage_target_descriptions=[
                    _target_description(
                        evidence_event(vertical_primary),
                        entity_id,
                    )
                    for entity_id in vertical_primary.required_entity_ids
                ],
                quality_risks=quality_risks,
                confidence=min(horizontal_primary.confidence, vertical_primary.confidence),
                recommended_duration_seconds=(
                    chapter.recommended_duration_seconds
                ),
                duration_rationale=chapter.duration_rationale,
                attention_observation=chapter.attention_observation,
                flow_intent=chapter.flow_intent,
                horizontal_candidates=[
                    FeatureHorizontalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.horizontal_strategy,
                        zoom_intent=candidate.horizontal_zoom_intent,
                        camera_intent=candidate.horizontal_camera_intent,
                        target_description=horizontal_target(candidate),
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(horizontal_options, start=1)
                ],
                vertical_candidates=[
                    projected_vertical_candidate(candidate, rank)
                    for rank, candidate in enumerate(vertical_options, start=1)
                ],
            )
        )
    return FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=plan.title,
        chapters=projected,
        uncertainties=plan.uncertainties,
        model_provenance=plan.model_provenance,
    )


def _selected_first_candidates(
    chapter: ClipCardFeatureSelectV2, selected_candidate_id: str | None
) -> list[ClipCardFeatureCandidate]:
    if not chapter.candidates:
        return []
    if selected_candidate_id is None:
        raise ValueError("candidate contract is missing its selected candidate ID")
    selected = next(
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id == selected_candidate_id
    )
    return [selected] + [
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id != selected_candidate_id
    ]


def _project_candidate_regions(
    candidate: ClipCardFeatureCandidate,
) -> list[FramingRegionIntent]:
    """Flatten auditable unions into executable single-entity regions."""

    role_map = {
        "hard_core": "required",
        "soft_extent": "preferred",
        "overlay_keepout": "avoid_overlay",
    }
    projected: list[FramingRegionIntent] = []
    for region in candidate.resolved_regions:
        union = len(region.entity_refs) > 1
        for index, ref in enumerate(region.entity_refs, start=1):
            region_id = (
                f"{region.region_id}.member-{index}" if union else region.region_id
            )
            member_description = (
                f"{region.target_description}; exact member entity_id={ref.entity_id}, "
                f"event_relation={ref.event_relation}"
                if union
                else region.target_description
            )
            projected.append(
                FramingRegionIntent(
                    region_id=region_id,
                    entity_id=ref.entity_id,
                    target_description=member_description,
                    kind=region.kind,
                    role=role_map[region.constraint_role],
                    atomic=region.atomic,
                    minimum_visible_fraction=(
                        1.0
                        if region.constraint_role == "hard_core" or region.atomic
                        else None
                    ),
                    observable_relations=[
                        f"event_relation={ref.event_relation}",
                        region.observable_relation,
                    ],
                    exclusions=[],
                )
            )
    return projected


def _upgrade_legacy_feature_plan(
    plan: ClipCardFeaturePlan,
) -> ClipCardFeaturePlanV2:
    return ClipCardFeaturePlanV2.model_validate(
        {"contract_version": "legacy-v1", **plan.model_dump(mode="json")}
    )


def project_feature_contracts(
    plan: ClipCardFeaturePlan | ClipCardFeaturePlanV2,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    preserve_runtime_candidates: bool | None = None,
) -> FeatureEditPlan:
    """Deterministically project the richer Clip Card plan for the renderer."""

    legacy_source = isinstance(plan, ClipCardFeaturePlan)
    if legacy_source:
        plan = _upgrade_legacy_feature_plan(plan)
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("source plan differs from projection catalog/brief")
    if preserve_runtime_candidates is None:
        preserve_runtime_candidates = not legacy_source and (
            plan.contract_version == "clip-card-feature-cut-v2"
        )
    projected: list[FeatureChapterSelect] = []
    for chapter in plan.chapters:
        horizontal_options = _selected_first_candidates(
            chapter, chapter.horizontal_candidate_id
        )
        vertical_options = _selected_first_candidates(
            chapter, chapter.vertical_candidate_id
        )
        projected.append(
            FeatureChapterSelect(
                feature_id=chapter.feature_id,
                evidence_status=chapter.evidence_status,
                horizontal_frame_id=chapter.horizontal_frame_id,
                vertical_frame_id=chapter.vertical_frame_id,
                observed_visual_evidence=chapter.observed_visual_evidence,
                selection_reason=chapter.selection_reason,
                horizontal_strategy=chapter.horizontal_strategy,
                horizontal_zoom_intent=chapter.horizontal_zoom_intent,
                horizontal_target_description=chapter.horizontal_target_description,
                vertical_strategy=chapter.vertical_strategy,
                vertical_target_description=chapter.vertical_target_description,
                quality_risks=chapter.quality_risks,
                confidence=chapter.confidence,
                horizontal_candidates=[
                    FeatureHorizontalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.horizontal_strategy,
                        zoom_intent=candidate.horizontal_zoom_intent,
                        target_description=candidate.horizontal_target_description,
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(horizontal_options, start=1)
                ] if preserve_runtime_candidates else [],
                vertical_candidates=[
                    FeatureVerticalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.vertical_strategy,
                        crop_mode=candidate.vertical_crop_mode,
                        target_description=candidate.vertical_target_description,
                        regions=_project_candidate_regions(candidate),
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(vertical_options, start=1)
                ] if preserve_runtime_candidates else [],
            )
        )
    return FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=plan.title,
        chapters=projected,
        uncertainties=plan.uncertainties,
        model_provenance=plan.model_provenance,
    )


def reproject_external_feature_plan(
    *,
    source_plan: ClipCardFeaturePlan,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Reproduce the legacy v1 candidate-free projection exactly."""

    del source_artifacts
    if not isinstance(source_plan, ClipCardFeaturePlan):
        raise ValueError("clip-card-feature-cut-v1 requires its exact legacy source schema")
    return brief, project_feature_contracts(
        source_plan,
        brief=brief,
        catalog=catalog,
        preserve_runtime_candidates=False,
    )


def reproject_external_feature_plan_v2(
    *,
    source_plan: ClipCardFeaturePlanV2,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Reproduce the v2 Top-K runtime-candidate projection exactly."""

    del source_artifacts
    if (
        not isinstance(source_plan, ClipCardFeaturePlanV2)
        or source_plan.contract_version != "clip-card-feature-cut-v2"
    ):
        raise ValueError(
            "clip-card-feature-cut-v2 requires a clip-card-feature-cut-v2 source plan"
        )
    return brief, project_feature_contracts(
        source_plan,
        brief=brief,
        catalog=catalog,
        preserve_runtime_candidates=True,
    )


def reproject_external_feature_plan_v3(
    *,
    source_plan: ClipCardFeaturePlanV3,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Reproduce the v3 projection from choices plus hash-bound local evidence."""

    if not isinstance(source_plan, ClipCardFeaturePlanV3):
        raise ValueError("clip-card-feature-cut-v3 requires its exact v3 source schema")
    evidence_path = source_artifacts.get("selected_clip_card_evidence")
    if evidence_path is None:
        raise ValueError("clip-card-feature-cut-v3 requires selected_clip_card_evidence")
    selected_evidence = SelectedClipCardEvidence.model_validate(read_json(evidence_path))
    return brief, project_feature_contracts_v3(
        source_plan,
        brief=brief,
        catalog=catalog,
        selected_evidence=selected_evidence,
    )


def reproject_direct_video_edit_plan(
    *,
    source_plan: DirectVideoEditPlan,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Verify rank/index resolution before reusing the v3 local projection."""

    derived_path = source_artifacts.get("derived_clip_card_feature_plan")
    evidence_path = source_artifacts.get("selected_clip_card_evidence")
    shortlist_path = source_artifacts.get("feature_shortlist")
    manifest_path = source_artifacts.get("candidate_video_evidence_manifest")
    capability_path = source_artifacts.get("editing_capability_catalog")
    if any(
        path is None
        for path in (
            derived_path,
            evidence_path,
            shortlist_path,
            manifest_path,
            capability_path,
        )
    ):
        raise ValueError("direct-video projection is missing local resolution artifacts")
    bound_capabilities = EditingCapabilityCatalog.model_validate(
        read_json(capability_path)
    )
    current_capabilities = simple_production_capability_catalog()
    if source_plan.capability_catalog_sha256 != (
        bound_capabilities.definition_sha256()
    ):
        raise ValueError("direct-video plan capability catalog binding changed")
    if bound_capabilities.definition_sha256() != (
        current_capabilities.definition_sha256()
    ):
        raise ValueError(
            "saved direct-video plan targets a stale local capability catalog"
        )
    derived = ClipCardFeaturePlanV3.model_validate(read_json(derived_path))
    selected_evidence = SelectedClipCardEvidence.model_validate(
        read_json(evidence_path)
    )
    shortlist = FeatureShortlistPlan.model_validate(read_json(shortlist_path))
    manifest = read_json(manifest_path)
    depth = int(manifest["selection_policy"]["rank_depth_per_chapter"])
    if len(source_plan.chapters) != len(brief.chapters):
        raise ValueError("direct-video plan chapter count differs from brief")
    if len(derived.chapters) != len(brief.chapters):
        raise ValueError("derived feature plan chapter count differs from brief")
    events = {
        (event.source_asset_id, event.event_id): event
        for event in selected_evidence.events
    }
    catalog_clips = {clip.clip_id: clip for clip in catalog.clips}
    catalog_frames = {frame.frame_id: frame for frame in catalog.frames}

    for offset, (direct, derived_chapter, brief_chapter, shortlisted) in enumerate(
        zip(
            source_plan.chapters,
            derived.chapters,
            brief.chapters,
            shortlist.chapters,
            strict=True,
        ),
        start=1,
    ):
        if direct.chapter_index != offset:
            raise ValueError("direct-video chapter indices are not contiguous")
        if (
            derived_chapter.feature_id != brief_chapter.feature_id
            or shortlisted.feature_id != brief_chapter.feature_id
        ):
            raise ValueError("direct-video chapter mapping changed feature identity")
        if direct.evidence_status == "not_found":
            if derived_chapter.evidence_status != "not_found":
                raise ValueError("derived plan changed a not-found decision")
            continue
        assert direct.horizontal is not None and direct.vertical is not None
        allowed = shortlisted.candidates[:depth]
        if len(derived_chapter.candidates) != len(allowed):
            raise ValueError("derived candidate count differs from bounded shortlist")
        by_rank = {
            rank: candidate
            for rank, candidate in enumerate(
                derived_chapter.candidates,
                start=1,
            )
        }
        for rank, (candidate, shortlist_candidate) in enumerate(
            zip(derived_chapter.candidates, allowed, strict=True),
            start=1,
        ):
            if candidate.candidate_id != planning_candidate_id(rank):
                raise ValueError("derived candidate ID is not locally ranked")
            if (
                candidate.source_asset_id,
                candidate.event_id,
            ) != (
                shortlist_candidate.source_asset_id,
                shortlist_candidate.event_id,
            ):
                raise ValueError("derived candidate escaped the bounded shortlist")
            frame = catalog_frames.get(candidate.frame_id)
            if frame is None:
                raise ValueError("derived candidate references an unknown frame")
            clip = catalog_clips.get(frame.clip_id)
            if (
                clip is None
                or f"sha256:{clip.sha256}" != candidate.source_asset_id
            ):
                raise ValueError("derived frame does not belong to candidate source")
        horizontal = by_rank[direct.horizontal.candidate_rank]
        vertical = by_rank[direct.vertical.candidate_rank]
        if derived_chapter.horizontal_candidate_id != horizontal.candidate_id:
            raise ValueError("derived horizontal rank differs from direct decision")
        if derived_chapter.vertical_candidate_id != vertical.candidate_id:
            raise ValueError("derived vertical rank differs from direct decision")
        expected_horizontal = (
            direct.horizontal.strategy,
            direct.horizontal.zoom_intent,
            direct.horizontal.camera_intent,
        )
        actual_horizontal = (
            horizontal.horizontal_strategy,
            horizontal.horizontal_zoom_intent,
            horizontal.horizontal_camera_intent,
        )
        if actual_horizontal != expected_horizontal:
            raise ValueError("derived horizontal intent differs from direct decision")
        if (
            vertical.vertical_strategy,
            vertical.vertical_crop_mode,
            vertical.coverage_mode,
            vertical.allow_controlled_clip,
            vertical.framing_intent,
        ) != (
            direct.vertical.strategy,
            direct.vertical.crop_mode,
            direct.vertical.coverage_mode,
            direct.vertical.allow_controlled_clip,
            direct.vertical.framing_intent,
        ):
            raise ValueError("derived vertical intent differs from direct decision")
        if (
            derived_chapter.attention_observation
            != direct.attention_observation
            or derived_chapter.flow_intent != direct.flow_intent
        ):
            raise ValueError("derived attention or flow intent differs from direct plan")
        selected_event = events[(vertical.source_asset_id, vertical.event_id)]
        known_entity_ids = {
            entity.entity_id for entity in selected_event.entities
        }
        if (
            set(
                vertical.required_entity_ids
                + vertical.preferred_entity_ids
                + vertical.sacrificable_entity_ids
                + (
                    [horizontal.horizontal_focus_entity_id]
                    if horizontal.horizontal_focus_entity_id is not None
                    else []
                )
            )
            - known_entity_ids
        ):
            raise ValueError("derived entity resolution escaped selected evidence")
        proposal = vertical.virtual_camera_proposal
        if direct.vertical.attention_sequence:
            if proposal is None or len(proposal.phases) != len(
                direct.vertical.attention_sequence
            ):
                raise ValueError("derived attention sequence is missing or changed")
            for step, phase in zip(
                direct.vertical.attention_sequence,
                proposal.phases,
                strict=True,
            ):
                if (
                    phase.start_progress,
                    phase.end_progress,
                    phase.camera_behavior,
                    phase.transition_in,
                ) != (
                    step.start_progress,
                    step.end_progress,
                    step.camera_behavior,
                    (
                        "cut"
                        if (
                            phase.phase_id == "phase-01"
                            or step.camera_behavior == "punch_in_cut"
                            or step.transition_preference == "cut"
                        )
                        else "smoothstep"
                    ),
                ):
                    raise ValueError("derived attention phase differs from direct plan")
                if set(phase.anchor_entity_ids) - known_entity_ids:
                    raise ValueError(
                        "derived attention anchors escaped selected evidence"
                    )
        elif proposal is not None:
            raise ValueError("derived plan invented a virtual camera proposal")
    return brief, project_feature_contracts_v3(
        derived,
        brief=brief,
        catalog=catalog,
        selected_evidence=selected_evidence,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=0,
        help=(
            "Opt-in full paid retries after schema/lineage validation failure. "
            "The default is zero so one planning command makes at most one "
            "Gemini request."
        ),
    )
    parser.add_argument(
        "--thinking-level",
        choices=["low", "high"],
        default="low",
        help=(
            "Use low for the large Top-K Structured Output so reasoning does "
            "not consume the response budget. High remains an explicit "
            "research option for smaller evidence sets."
        ),
    )
    parser.add_argument(
        "--reuse-raw-output",
        action="store_true",
        help=(
            "Canonicalize, revalidate, and project an existing paid response "
            "without creating another API request"
        ),
    )
    parser.add_argument(
        "--resume-failed-plan",
        action="store_true",
        help=(
            "Send exactly one paid schema/contract repair using the newest "
            "saved failed attempt and its original File API URIs. The "
            "original multimodal planning request is not repeated."
        ),
    )
    parser.add_argument(
        "--shortlist",
        type=Path,
        help=(
            "Optional validated high-recall FeatureShortlistPlan. When present, "
            "only shortlisted events and their RF frames are sent to this "
            "geometry-aware planner."
        ),
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        action="append",
        default=[],
        help="Repeatable validated ClipObservationSupplement JSON.",
    )
    parser.add_argument(
        "--music",
        type=Path,
        help=(
            "Optional actual music file. When supplied, Gemini receives the "
            "audio together with the Clip Card evidence and must use audible "
            "flow for selection and relative dwell. The source hash is bound "
            "into the external projection."
        ),
    )
    parser.add_argument(
        "--file-cache-root",
        type=Path,
        help=(
            "Optional shared SHA-256 keyed Gemini File API cache root. "
            "Defaults to OUTPUT_DIR/../file-cache."
        ),
    )
    parser.add_argument(
        "--candidate-video-evidence",
        action="store_true",
        help=(
            "Attach bounded direct-video evidence for shortlisted candidates "
            "to the single music-aware planning call."
        ),
    )
    parser.add_argument(
        "--candidate-video-depth",
        type=int,
        default=2,
        help="Maximum ranked shortlist candidates per chapter sent as video evidence.",
    )
    parser.add_argument(
        "--candidate-video-context-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-candidate-video-seconds",
        type=float,
        default=360.0,
        help="Fail before upload or paid planning when direct video exceeds this total.",
    )
    args = parser.parse_args()
    if args.repair_attempts < 0:
        parser.error("--repair-attempts must be zero or greater")
    if not 1 <= args.candidate_video_depth <= 4:
        parser.error("--candidate-video-depth must be between 1 and 4")
    if args.candidate_video_context_seconds < 0:
        parser.error("--candidate-video-context-seconds must be non-negative")
    if args.maximum_candidate_video_seconds <= 0:
        parser.error("--maximum-candidate-video-seconds must be positive")
    if args.candidate_video_evidence and args.shortlist is None:
        parser.error("--candidate-video-evidence requires --shortlist")
    if args.reuse_raw_output and args.resume_failed_plan:
        parser.error("--reuse-raw-output and --resume-failed-plan are exclusive")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.reuse_raw_output and not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
    music_path = (
        args.music.expanduser().resolve(strict=True)
        if args.music is not None
        else None
    )
    music_sha256 = sha256_file(music_path) if music_path is not None else None
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    asset_to_clip = {f"sha256:{clip.sha256}": clip for clip in catalog.clips}

    cards: dict[str, FullClipCard] = {}
    for clip in catalog.clips:
        path = (
            args.prepared_library
            / "clips"
            / clip.sha256[:16]
            / "gemini"
            / "clip-card"
            / "clip_card.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Clip Card missing for {clip.clip_id}: {path}")
        card = FullClipCard.model_validate(read_json(path))
        expected_asset = f"sha256:{clip.sha256}"
        if card.source_asset_id != expected_asset:
            raise ValueError(f"Clip Card asset mismatch for {clip.clip_id}")
        cards[expected_asset] = card
    supplements: dict[str, list[ClipObservationSupplement]] = {}
    for path in args.supplement:
        supplement = ClipObservationSupplement.model_validate(read_json(path))
        supplements.setdefault(supplement.source_asset_id, []).append(supplement)

    shortlist: FeatureShortlistPlan | None = None
    shortlist_path: Path | None = None
    shortlist_allowed: dict[str, set[tuple[str, str]]] = {}
    shortlist_candidate_ids: dict[str, dict[tuple[str, str], str]] = {}
    if args.shortlist is not None:
        shortlist_path = args.shortlist.expanduser().resolve(strict=True)
        shortlist = FeatureShortlistPlan.model_validate(read_json(shortlist_path))
        validate_feature_shortlist(
            shortlist,
            brief=brief,
            catalog=catalog,
            cards=cards,
        )
        shortlist_allowed = {
            chapter.feature_id: {
                (candidate.source_asset_id, candidate.event_id)
                for candidate in planning_candidate_slice(
                    chapter.candidates,
                    direct_video_evidence=args.candidate_video_evidence,
                    depth=args.candidate_video_depth,
                )
            }
            for chapter in shortlist.chapters
        }
        shortlist_candidate_ids = {
            chapter.feature_id: {
                (candidate.source_asset_id, candidate.event_id): planning_candidate_id(rank)
                for rank, candidate in enumerate(
                    planning_candidate_slice(
                        chapter.candidates,
                        direct_video_evidence=args.candidate_video_evidence,
                        depth=args.candidate_video_depth,
                    ),
                    start=1,
                )
            }
            for chapter in shortlist.chapters
        }

    def validate_shortlist_membership(plan: ClipCardFeaturePlanV3) -> None:
        if shortlist is None:
            return
        for chapter in plan.chapters:
            unknown = sorted(
                {
                    (candidate.source_asset_id, candidate.event_id)
                    for candidate in chapter.candidates
                }
                - shortlist_allowed[chapter.feature_id]
            )
            if unknown:
                raise ValueError(
                    f"plan escaped shortlist for {chapter.feature_id}: {unknown}"
                )
            mismatched_ids = sorted(
                (
                    candidate.candidate_id,
                    shortlist_candidate_ids[chapter.feature_id].get(
                        (candidate.source_asset_id, candidate.event_id)
                    ),
                )
                for candidate in chapter.candidates
                if candidate.candidate_id
                != shortlist_candidate_ids[chapter.feature_id].get(
                    (candidate.source_asset_id, candidate.event_id)
                )
            )
            if mismatched_ids:
                raise ValueError(
                    f"plan changed local candidate IDs for {chapter.feature_id}: "
                    f"{mismatched_ids}"
                )

    frame_map: dict[str, list[dict[str, object]]] = {}
    for frame in catalog.frames:
        clip = clips[frame.clip_id]
        frame_map.setdefault(f"sha256:{clip.sha256}", []).append(
            {
                "frame_id": frame.frame_id,
                "local_mmss": mmss(frame.requested_time_ms),
            }
        )

    run_id = f"clip-card-feature-plan-{uuid.uuid4().hex[:8]}"
    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        run_id=run_id,
        generated_at=utc_now(),
        interaction_id=None,
    )
    if shortlist is None:
        evidence: list[dict[str, object]] = [
            {
                "clip_id": asset_to_clip[asset_id].clip_id,
                "clip_card": compact_card_v3(
                    card, tuple(supplements.get(asset_id, []))
                ),
                "available_catalog_frames": frame_map[asset_id],
            }
            for asset_id, card in cards.items()
        ]
        evidence_heading = "完整 Clip Card evidence 與可選 RF frame IDs"
        evidence_scope_rule = (
            "你可從下方完整 library 選擇任一合法 asset/event/frame。"
        )
    else:
        evidence = []
        for chapter_index, chapter in enumerate(shortlist.chapters, start=1):
            candidate_events: list[dict[str, object]] = []
            planning_candidates = planning_candidate_slice(
                chapter.candidates,
                direct_video_evidence=args.candidate_video_evidence,
                depth=args.candidate_video_depth,
            )
            for rank, candidate in enumerate(planning_candidates, start=1):
                card = cards[candidate.source_asset_id]
                event = next(
                    item
                    for item in card.events
                    if item.event_id == candidate.event_id
                )
                if args.candidate_video_evidence:
                    event_entity_ids = set(
                        event.entity_ids
                        + event.primary_entity_ids
                        + event.required_entity_ids
                        + event.optional_entity_ids
                        + event.avoid_overlay_entity_ids
                        + [target.entity_id for target in event.grounding_targets]
                    )
                    compact: dict[str, object] = {
                        "source_asset_id": card.source_asset_id,
                        "clip_summary": card.summary,
                        "entities": [
                            {
                                "entity_index": entity_index,
                                "entity_id": entity.entity_id,
                                "kind": entity.kind,
                                "label": entity.label,
                                "distinguishing_features": (
                                    entity.distinguishing_features
                                ),
                            }
                            for entity_index, entity in enumerate(
                                (
                                    entity
                                    for entity in card.entities
                                    if entity.entity_id in event_entity_ids
                                ),
                                start=1,
                            )
                        ],
                        "event": {
                            "event_id": event.event_id,
                            "label": event.label,
                            "observable_evidence": event.observable_evidence,
                            "action_completeness": event.action_completeness,
                            "quality_risks": event.quality_risks,
                            "entity_ids": event.entity_ids,
                            "primary_entity_ids": event.primary_entity_ids,
                            "required_entity_ids": event.required_entity_ids,
                            "optional_entity_ids": event.optional_entity_ids,
                            "avoid_overlay_entity_ids": (
                                event.avoid_overlay_entity_ids
                            ),
                            "grounding_target_entity_ids": [
                                target.entity_id
                                for target in event.grounding_targets
                            ],
                        },
                    }
                else:
                    compact = compact_card_v3(
                        card,
                        tuple(supplements.get(candidate.source_asset_id, [])),
                    )
                    compact["events"] = [
                        compact_event
                        for compact_event in compact["events"]  # type: ignore[index]
                        if compact_event["event_id"]  # type: ignore[index]
                        == candidate.event_id
                    ]
                candidate_event: dict[str, object] = {
                    "candidate_id": planning_candidate_id(rank),
                    "candidate_rank": rank,
                    "retrieval_reason": candidate.retrieval_reason,
                    "clip_id": asset_to_clip[candidate.source_asset_id].clip_id,
                    "clip_card": compact,
                }
                if not args.candidate_video_evidence:
                    candidate_event["available_catalog_frames"] = [
                        frame
                        for frame in frame_map[candidate.source_asset_id]
                        if event.start_mmss
                        <= str(frame["local_mmss"])
                        < event.end_mmss
                    ]
                candidate_events.append(candidate_event)
            evidence.append(
                {
                    "chapter_index": chapter_index,
                    "feature_id": chapter.feature_id,
                    "retrieval_status": chapter.evidence_status,
                    "retrieval_uncertainty": chapter.uncertainty,
                    "candidate_events": candidate_events,
                }
            )
        evidence_heading = "已驗證 shortlist 的 Clip Card evidence 與可選 RF frame IDs"
        evidence_scope_rule = (
            "每章只能從該 feature_id 下列出的 candidate_events 選擇；"
            "不得跨章引用未召回的 asset/event。"
        )
    capability_catalog = simple_production_capability_catalog()
    capability_catalog_path = args.output_dir / "editing-capability-catalog.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(capability_catalog_path, capability_catalog)
    capability_catalog_sha256 = capability_catalog.definition_sha256()
    prompt = f"""
你是 evidence-bound 的資深短影音挑帶剪輯師。請使用完整 Clip Card library，為使用者 brief 的每個 chapter 保留有排序的候選 take，再分別選出橫式與直式代表。你只能引用輸入列出的 source_asset_id、event_id、entity_id 與 RF frame_id。

規則：
1. brief 是允許使用的產品 claim，不是畫面證據；observed_visual_evidence 只能寫 Clip Card 直接支持的內容。
2. 每個 brief feature_id 必須依原順序恰好回傳一次。supported chapter 必須保留 2–4 個、partial chapter 保留 1–4 個依品質排序且 evidence frame 不重複的 candidates；優先完整動作、清楚結果、低遮擋、低反光與不同 take。candidate_id 必須從該章 candidate_events 逐字複製，不得自行命名。not_found 不得虛構候選。
2a. {evidence_scope_rule}
3. selected frame 的 local_mmss 必須位於所引用 event 的 [start_mmss,end_mmss)；不得自行創造 frame ID 或 timestamp。RF frame_id 必須從 available_catalog_frames 逐字複製並保留全部六位數與前導零，例如 RF000204 不可縮成 RF00204。
4. 若可見型號、文字、數字或物件身分與 brief 衝突，優先改選沒有衝突的 take；沒有可靠 take 時用 partial 或 not_found 並保存風險。
5. 每個 candidate 都必須保存可直接重試的 16:9 strategy／zoom／horizontal_focus_entity_id，以及 9:16 strategy、framing_intent 和 brief-specific entity priorities。橫式與直式可以從同一候選組選不同來源；horizontal_candidate_id／vertical_candidate_id 必須指向 candidates。不要重複輸出 rank-1 asset/event/frame mirror、target description 或 resolved crop regions；程式會從所選 candidate 與 hash-bound Clip Card evidence 確定性補出。
   - horizontal_strategy=original 時，horizontal_zoom_intent 必須是 none，而且 horizontal_focus_entity_id 必須是 null；原始構圖不需要追蹤焦點。
   - horizontal_strategy=tracked_reframe 時，horizontal_zoom_intent 必須是 subtle 或 detail，而且 horizontal_focus_entity_id 必須引用該 event 中一個可見 entity。
   - horizontal_camera_intent 只能描述剪輯語言：hold、follow、punch_in_cut、push_in、pull_out、recenter，或在確實存在兩個可區分 anchor 時使用 pan_reveal。不要輸出倍率、座標或運鏡時間。
6. 9:16 應把 brief 的 vertical_primary_target_description 視為內容優先序，不是強制演算法。只有需要動態跟隨且存在可靠 target 時才用 tracked_crop；若穩定構圖已可保留內容，或窄裁切無法安全包含必要範圍，可以使用 fit_with_background。不得只因 brief 有 primary target 就強制 tracked_crop。
7. required_entity_ids、preferred_entity_ids、sacrificable_entity_ids 是針對本 brief 與本 aspect 的編輯決定，三組必須互斥，清單順序代表優先序，且只能引用該 event 已列出的 entity。只分類與這次構圖決策直接相關的 entity；未列入者不會被程式偷偷視為 required 或 sacrificable。不得把未觀察到的 entity 加入。tracked_crop 至少要有一個 required entity。
8. framing_intent 只需簡潔描述本候選的構圖取捨；不得輸出座標、bbox、mask、target description 或 verbose region contract。程式會把這些 entity priority ID 與 Clip Card entity/grounding target 資料轉成 domain-neutral hard-core、soft-extent 與 overlay keepout regions。
8a. 若同一個 source event 有兩個以上「依序重要、但不必同時出現在單一 9:16 crop」的可追蹤 entity，可以提出 virtual_camera_proposal；否則必須為 null。{"本次另附帶 candidate_id 標籤的 bounded direct-video evidence；可只依該影片中實際可見的狀態順序提出 proposal。Clip Card 未評估 camera capability 不代表 direct video 中不存在順序。" if args.candidate_video_evidence else "順序只能引用 Clip Card 的 observable_beats 與 evidence_roles；observable_beats capability 若為 not_assessed、assessed_absent 或 not_applicable，不得自行發明注意力順序或 camera behavior。"}方向可以左→右、右→左、人物→結果、整體→細節或完全不動，不得使用固定方向模板。必須同時可見的 entity 應在同一 phase 共同保留；共同上下文或相對尺度是內容意義時，不得用會破壞參照的獨立特寫取代。phase 的 anchor_entity_ids 只能引用同 candidate 的 required_entity_ids 或 preferred_entity_ids；observable_predicate 與 transition_condition 必須描述可直接觀察的條件，不能引用常識、品牌知識或自創 timestamp。start_progress／end_progress 只表達連續覆蓋 0–1 的相對敘事順序。一般跟隨優先使用 follow_deadband，只有每一段移動都承載動作證據時才使用 follow；push_in 用於可見細節／結果逐漸成為重點，pull_out 用於回到整體關係，punch_in_cut 用於明確資訊落點的硬切放大，hold 用於固定構圖。不得輸出倍率、速度、easing 或曲線；本機會依來源解析度、距離、時長、速度、加速度與 jerk 決定安全運鏡，必要時將過短的遠距平移改為 cut。自動 proposal 不授權裁切 active anchor；後續 Grounding、SAM、containment 與 motion gate 失敗時會改試下一個候選或回退。
9. 每個 supported／partial chapter 應依可見資訊、動作完整性、閱讀需求、情緒停留、重複壓力與音樂角色提出 recommended_duration_seconds、duration_rationale 與 attention_observation。minimum／recommended／maximum dwell 必須依序排列；attention 各分量 0–1，只是待審相對判斷，不是 source timestamp 或客觀真值。action_progress 表示到片段結尾時動作／結果已完成、適合轉場的程度。
9a. {"本次另附實際音樂，music_sha256=" + music_sha256 + "。你必須實際聆聽音訊，依可聽見的段落、能量、留白與收尾安排候選及相對停留；不得只依文字猜音樂，也不得輸出自創 beat timestamp。" if music_sha256 is not None else "本次沒有附音樂；不得推測不存在的節拍、段落或能量變化。"}
10. bbox、mask、crop 座標與精確 cut point 均由後續 Grounding／tracker／FFmpeg 處理；本階段不得輸出座標。
11. confidence 是 proposal，不是人工真值；候選排序仍須由可見 evidence 與風險說明支持。

contract_version 必須原樣回傳：clip-card-feature-cut-v3
project_id 必須原樣回傳：{brief.project_id}
catalog_id 必須原樣回傳：{catalog.catalog_id}
model_provenance 必須先原樣回傳：
{provenance.model_dump_json(indent=2)}

## 使用者 brief
{brief.model_dump_json(indent=2)}

## {evidence_heading}
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()
    if args.candidate_video_evidence:
        prompt = f"""
你是 evidence-bound 的資深短影音剪輯師。請同時閱讀 brief、精簡候選索引，
觀看後續每個有 feature_id 與 candidate_rank 標籤的 bounded candidate video，
並實際聆聽最後附上的完整音樂。你的任務是提出一份精簡的剪輯意圖；
不是產生剪點、座標或追蹤結果。你只能使用下方能力目錄列出的剪輯動詞；
能力目錄說明本機確實能執行什麼，不是要求你每一種都使用。

只能回傳：
1. 每個 chapter_index 的橫式與直式各選哪一個 candidate_rank。
2. 選擇理由、直接可見的證據、風險與建議停留秒數。
3. 橫式是否保持原構圖，或對一個可見 entity 做有目的的推近／跟隨。
4. 直式 coverage_mode、是否允許 controlled semantic clip，以及全段必須曾被
   看見、最好保留、可犧牲的 entity_index。
5. 若注意力確實依序轉移，使用 0–1 相對進度列出 attention_sequence。
   每個 phase 的 anchor 只代表該 phase 必須看見的主體；全段 required entity
   不得被本機強塞進每個 phase。若關係必須同時存在，coverage_mode 使用
   simultaneous 且不得拆成多 phase。
6. 每章提供 attention_observation 與 flow_intent，描述資訊量、動作完成度、
   閱讀需求、敘事角色、能量、前後鏡頭關係與 boundary_alignment。
   visual_sync_event 只在影片中真的有可觀察落點時提供，並同時給
   visual_sync_predicate 與 music_target；安靜訪談、空景或純 hold 可為 null。
   這些是相對剪輯意圖，不是精確剪點。

禁止回傳或推測：
- project_id、catalog_id、feature_id、source_asset_id、event_id、frame_id、
  candidate_id、entity_id 或 model_provenance；
- MM:SS 以外的時間，更不得輸出毫秒；
- bbox、mask、crop center、逐幀座標；
- 運鏡倍率、速度、加速度、easing 或 jerk；
- 影片、brief 與候選索引沒有直接支持的品牌、型號、數字或功能。

構圖規則：
- 9:16 優先採可安全滿版的 tracked_crop。只有必要關係或必要範圍無法在
  9:16 同時成立，而且換候選也無法成立時，才使用 fit_with_background；
  它只會產生非交付的人工 review preview。不要因為邊緣略有裁切就退回補邊。
- coverage_mode=sequential：兩個以上主體可以依序看懂，每個 phase 只鎖自己的
  anchor；本機可平移或在距離過遠時切換視角。
- coverage_mode=relation_core：只需保留承載比較、接觸、方向或相對尺度的可見
  核心，非關鍵物件邊緣可裁，必須 allow_controlled_clip=true。
- coverage_mode=primary_with_context：主體為硬限制、上下文為軟限制；非必要
  上下文可部分離開畫面，必須 allow_controlled_clip=true。
- coverage_mode=independent_detail：細節本身足以成立，不要求整個物件始終完整，
  必須 allow_controlled_clip=true。
- attention_sequence 只描述「看誰」與「hold/follow/push/pull/punch」；
  後續本機會以 Gemini 單幀 bbox Grounding、SAM 與 geometry solver 執行。
- transition_preference 可選 auto、continuous 或 cut。兩個 view 各自成立但距離
  太遠時，cut 是正式剪輯語法；即使選 continuous，本機 motion gate 仍可改成 cut。
- 沒有可見注意力轉移時，sequence 可以是空陣列；不得為了看起來有運鏡而發明移動。
- candidate_rank 必須直接複製該章候選索引中的整數，不得引用未附影片的 rank。
- chapter_index 與 entity_index 也只能複製輸入中的整數；本機會解析成不可變 ID。
- brief 章節順序必須保留。音樂影響相對停留、章節能量、鏡頭關係與視覺
  落點意圖；不得自創 beat timestamp。本機 MusicMap 會解析合法影格與音訊點。

contract_version 必須原樣回傳：direct-video-edit-plan-v2
capability_catalog_sha256 必須原樣回傳：{capability_catalog_sha256}

## 本機可執行的版本化剪輯能力
{capability_catalog.model_dump_json(indent=2)}

## 使用者 brief
{brief.model_dump_json(indent=2)}

## 每章可選的 bounded candidate 索引
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_input: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    candidate_video_manifest_path: Path | None = None
    direct_video_observed_events: set[tuple[str, str]] = set()
    if (
        args.candidate_video_evidence
        and not args.reuse_raw_output
        and not args.resume_failed_plan
    ):
        assert shortlist is not None
        context_ms = round(args.candidate_video_context_seconds * 1000)
        selected_direct_evidence: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        for chapter in shortlist.chapters:
            for rank, candidate in enumerate(
                planning_candidate_slice(
                    chapter.candidates,
                    direct_video_evidence=True,
                    depth=args.candidate_video_depth,
                ),
                start=1,
            ):
                key = (candidate.source_asset_id, candidate.event_id)
                row = selected_direct_evidence.setdefault(
                    key,
                    {
                        "source_asset_id": candidate.source_asset_id,
                        "event_id": candidate.event_id,
                        "references": [],
                    },
                )
                row["references"].append(
                    {
                        "feature_id": chapter.feature_id,
                        "rank": rank,
                        "candidate_id": planning_candidate_id(rank),
                        "retrieval_reason": candidate.retrieval_reason,
                    }
                )
        total_evidence_ms = 0
        for row in selected_direct_evidence.values():
            card = cards[row["source_asset_id"]]
            event = next(
                item
                for item in card.events
                if item.event_id == row["event_id"]
            )
            start_ms, end_ms = bounded_event_window_ms(
                card,
                event,
                context_ms=context_ms,
            )
            row["start_ms"] = start_ms
            row["end_ms"] = end_ms
            row["duration_ms"] = end_ms - start_ms
            total_evidence_ms += end_ms - start_ms
        maximum_evidence_ms = round(
            args.maximum_candidate_video_seconds * 1000
        )
        validate_candidate_video_budget(
            total_duration_ms=total_evidence_ms,
            maximum_duration_ms=maximum_evidence_ms,
        )
        file_cache_root = (
            args.file_cache_root.expanduser().resolve()
            if args.file_cache_root is not None
            else args.output_dir.parent / "file-cache"
        )
        direct_root = args.output_dir / "candidate-video-evidence"
        direct_rows: list[dict[str, Any]] = []
        upload_client = GeminiLabClient(api_key=api_key)
        try:
            for row in selected_direct_evidence.values():
                clip = asset_to_clip[row["source_asset_id"]]
                proxy_path = (
                    direct_root
                    / clip.sha256[:16]
                    / row["event_id"]
                    / "bounded.mp4"
                )
                audio_included = render_bounded_event_proxy(
                    Path(clip.path),
                    proxy_path,
                    start_ms=row["start_ms"],
                    end_ms=row["end_ms"],
                )
                proxy_sha256 = sha256_file(proxy_path)
                uploaded, reused = upload_client.ensure_video_upload(
                    proxy_path,
                    file_cache_root
                    / proxy_sha256
                    / "candidate-video-upload",
                )
                candidate_label = " | ".join(
                    (
                        f"{item['feature_id']}#{item['candidate_id']}"
                        f"(rank={item['rank']})"
                    )
                    for item in row["references"]
                )
                request_input.append(
                    {
                        "type": "text",
                        "text": (
                            "DIRECT CANDIDATE VIDEO EVIDENCE\n"
                            f"candidate_scope={candidate_label}\n"
                            f"source_asset_id={row['source_asset_id']}\n"
                            f"event_id={row['event_id']}\n"
                            "Only use what is directly visible/audible here."
                        ),
                    }
                )
                request_input.append(
                    {
                        "type": "video",
                        "uri": uploaded.uri,
                        "mime_type": canonical_interactions_mime_type(
                            str(uploaded.mime_type)
                        ),
                    }
                )
                direct_rows.append(
                    {
                        **row,
                        "proxy_path": str(proxy_path.resolve()),
                        "proxy_sha256": proxy_sha256,
                        "audio_included": audio_included,
                        "file_api_reused": reused,
                    }
                )
        finally:
            upload_client.close()
        candidate_video_manifest_path = (
            args.output_dir / "candidate-video-evidence.manifest.json"
        )
        direct_video_observed_events = {
            (str(row["source_asset_id"]), str(row["event_id"]))
            for row in direct_rows
        }
        write_json(
            candidate_video_manifest_path,
            {
                "contract_version": "candidate-video-evidence-manifest-v1",
                "selection_policy": {
                    "rank_depth_per_chapter": args.candidate_video_depth,
                    "context_ms": context_ms,
                    "maximum_total_ms": maximum_evidence_ms,
                },
                "total_duration_ms": total_evidence_ms,
                "candidate_count": len(direct_rows),
                "candidates": direct_rows,
            },
        )
        request_input.append(
            {
                "type": "text",
                "text": (
                    "DIRECT CANDIDATE VIDEO MANIFEST\n"
                    f"sha256={sha256_file(candidate_video_manifest_path)}\n"
                    "The manifest binds the bounded videos above to immutable "
                    "shortlist candidate IDs."
                ),
            }
        )
    elif args.candidate_video_evidence and (
        args.reuse_raw_output or args.resume_failed_plan
    ):
        candidate_video_manifest_path = (
            args.output_dir / "candidate-video-evidence.manifest.json"
        )
        if not candidate_video_manifest_path.exists():
            raise FileNotFoundError(
                "--reuse-raw-output with direct videos requires the original "
                "candidate-video-evidence.manifest.json"
            )
        candidate_video_manifest = read_json(candidate_video_manifest_path)
        manifest_policy = candidate_video_manifest.get("selection_policy", {})
        if (
            manifest_policy.get("rank_depth_per_chapter")
            != args.candidate_video_depth
        ):
            raise ValueError(
                "direct-video candidate depth differs from the paid request"
            )
        direct_video_observed_events = {
            (str(row["source_asset_id"]), str(row["event_id"]))
            for row in candidate_video_manifest.get("candidates", [])
        }
    if (
        music_path is not None
        and not args.reuse_raw_output
        and not args.resume_failed_plan
    ):
        file_cache_root = (
            args.file_cache_root.expanduser().resolve()
            if args.file_cache_root is not None
            else args.output_dir.parent / "file-cache"
        )
        upload_dir = (
            file_cache_root
            / music_sha256
            / "music-upload"
        )
        upload_client = GeminiLabClient(api_key=api_key)
        try:
            uploaded_music, _ = upload_client.ensure_video_upload(
                music_path,
                upload_dir,
            )
        finally:
            upload_client.close()
        request_input.append(
            {
                "type": "audio",
                "uri": uploaded_music.uri,
                "mime_type": canonical_interactions_mime_type(
                    str(uploaded_music.mime_type)
                ),
            }
        )
    response_model: type[BaseModel] = (
        DirectVideoEditPlan
        if args.candidate_video_evidence
        else ClipCardFeaturePlanV3
    )
    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "Provided Clip Cards, candidate indexes, explicitly labeled direct "
            "candidate videos, and supplied music are the only evidence. "
            "Never replace visible evidence with model memory or likely product knowledge. "
            "Return editorial selection, relative dwell, and attention intent only. "
            "Never return exact time, candidate IDs, asset/event/frame IDs, bounding "
            "boxes, masks, coordinates, or motion curves. A brief target is "
            "editorial intent, not authorization to force a tracked crop."
        ),
        "store": False,
        "input": request_input,
        "generation_config": {
            "thinking_level": args.thinking_level,
            "max_output_tokens": (
                12_000 if args.candidate_video_evidence else 32_000
            ),
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(response_model),
        },
    }
    plan: ClipCardFeaturePlanV3 | None = None
    direct_video_plan: DirectVideoEditPlan | None = None
    interaction_id = ""
    source_request_path: Path
    source_raw_output_path: Path
    source_raw_interaction_path: Path
    canonical_output_path: Path
    normalization_audit_path: Path
    extra_projection_artifacts: dict[str, Path] = {}
    extra_projection_artifacts["editing_capability_catalog"] = (
        capability_catalog_path
    )
    if music_path is not None:
        extra_projection_artifacts["source_music"] = music_path
    if args.reuse_raw_output:
        artifacts = _resolve_feature_reuse_artifacts(args.output_dir)
        source_request_path = artifacts["request"]
        source_raw_output_path = artifacts["raw_output"]
        source_raw_interaction_path = artifacts["raw_interaction"]
        original_request = read_json(source_request_path)
        original_inputs = original_request.get("input")
        original_inputs = (
            original_inputs if isinstance(original_inputs, list) else []
        )
        original_has_audio = any(
            isinstance(item, dict) and item.get("type") == "audio"
            for item in original_inputs
        )
        if original_has_audio != (music_sha256 is not None):
            raise ValueError(
                "--reuse-raw-output music presence differs from the paid request"
            )
        original_text = "\n".join(
            str(item.get("text"))
            for item in original_inputs
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if candidate_video_manifest_path is not None and (
            f"sha256={sha256_file(candidate_video_manifest_path)}"
            not in original_text
        ):
            raise ValueError(
                "--reuse-raw-output candidate video manifest differs from "
                "the paid request"
            )
        if music_sha256 is not None and (
            f"music_sha256={music_sha256}" not in original_text
        ):
            if not args.candidate_video_evidence:
                raise ValueError(
                    "--reuse-raw-output music hash differs from the paid request"
                )
            file_cache_root = (
                args.file_cache_root.expanduser().resolve()
                if args.file_cache_root is not None
                else args.output_dir.parent / "file-cache"
            )
            music_upload_path = (
                file_cache_root
                / music_sha256
                / "music-upload"
                / "file_upload_final.json"
            )
            if not music_upload_path.exists():
                raise FileNotFoundError(
                    "direct-video raw reuse cannot verify the original music "
                    "File API object"
                )
            expected_music_uri = str(
                read_json(music_upload_path).get("uri") or ""
            )
            original_music_uris = {
                str(item.get("uri") or "")
                for item in original_inputs
                if isinstance(item, dict) and item.get("type") == "audio"
            }
            if not expected_music_uri or expected_music_uri not in original_music_uris:
                raise ValueError(
                    "--reuse-raw-output music File API object differs from "
                    "the paid request"
                )
        raw_interaction = read_json(source_raw_interaction_path)
        artifact_models = {
            "original_request": str(original_request.get("model") or ""),
            "raw_interaction": str(raw_interaction.get("model") or ""),
        }
        mismatched_models = {
            source: model
            for source, model in artifact_models.items()
            if model != MODEL_ID
        }
        if mismatched_models:
            raise ValueError(
                "--reuse-raw-output model mismatch: "
                f"expected {MODEL_ID!r}, got {mismatched_models}. "
                "Run with the artifact's original JASCUE_GEMINI_MODEL instead."
            )
        reprojection_request_path = (
            args.output_dir / "clip-card-feature-plan.reprojection-request.json"
        )
        write_json(reprojection_request_path, request)
        raw_output = read_json(source_raw_output_path)
        output_text = _verified_feature_raw_output_text(
            raw_output=raw_output,
            raw_interaction=raw_interaction,
        )
        output_text, canonical_output_path, normalization_audit_path = (
            _write_feature_normalization_artifacts(
                output_dir=args.output_dir,
                artifact_stem="clip-card-feature-plan",
                raw_output_path=source_raw_output_path,
                raw_output_text=output_text,
                direct_video_plan=args.candidate_video_evidence,
            )
        )
        reuse_record_path = args.output_dir / "clip-card-feature-plan.raw-output-reuse.json"
        write_json(
            reuse_record_path,
            {
                "interpretation": (
                    "saved_model_response_canonicalized_revalidated_and_projected_"
                    "with_no_new_model_call"
                ),
                "artifact_set": artifacts["kind"],
                "original_request_path": str(artifacts["request"].resolve()),
                "original_request_sha256": sha256_file(artifacts["request"]),
                "raw_output_path": str(source_raw_output_path.resolve()),
                "raw_output_sha256": sha256_file(source_raw_output_path),
                "raw_interaction_path": str(source_raw_interaction_path.resolve()),
                "raw_interaction_sha256": sha256_file(source_raw_interaction_path),
                "current_reprojection_request_path": str(
                    reprojection_request_path.resolve()
                ),
                "current_reprojection_request_sha256": sha256_file(
                    reprojection_request_path
                ),
                "normalization_audit_path": str(normalization_audit_path.resolve()),
                "normalization_audit_sha256": sha256_file(normalization_audit_path),
                "reused_at": utc_now(),
            },
        )
        interaction_id = str(raw_interaction.get("id") or "")
        if args.candidate_video_evidence:
            assert shortlist is not None
            direct_video_plan = DirectVideoEditPlan.model_validate_json(
                output_text
            )
            plan = project_direct_video_edit_plan(
                direct_video_plan,
                shortlist=shortlist,
                candidate_depth=args.candidate_video_depth,
                brief=brief,
                catalog=catalog,
                cards=cards,
                provenance=provenance,
            )
            write_json(
                args.output_dir / "direct-video-edit-plan.json",
                direct_video_plan,
            )
        else:
            plan = ClipCardFeaturePlanV3.model_validate_json(output_text)
        validate_plan_contract_v3(
            plan,
            brief=brief,
            catalog=catalog,
            cards=cards,
            supplements=supplements,
            direct_video_observed_events=direct_video_observed_events,
        )
        validate_shortlist_membership(plan)
        if plan.model_provenance.model_id != MODEL_ID:
            raise ValueError(
                "--reuse-raw-output model provenance mismatch: "
                f"expected {MODEL_ID!r}, got {plan.model_provenance.model_id!r}"
            )
        extra_projection_artifacts = {
            "original_request": artifacts["request"],
            "current_reprojection_request": reprojection_request_path,
            "raw_output_reuse_record": reuse_record_path,
            "editing_capability_catalog": capability_catalog_path,
        }
    else:
        if args.resume_failed_plan:
            failed_attempt = _resolve_latest_failed_feature_plan_attempt(
                args.output_dir
            )
            request = read_json(failed_attempt["request"])
            if str(request.get("model") or "") != MODEL_ID:
                raise ValueError(
                    "--resume-failed-plan model differs from the current model"
                )
            original_inputs = request.get("input")
            if not isinstance(original_inputs, list) or not original_inputs:
                raise ValueError(
                    "--resume-failed-plan requires the complete original inputs"
                )
            first_input = original_inputs[0]
            if (
                not isinstance(first_input, dict)
                or first_input.get("type") != "text"
                or not isinstance(first_input.get("text"), str)
            ):
                raise ValueError(
                    "--resume-failed-plan requires the original planner prompt"
                )
            previous_error = str(failed_attempt["error"])
            attempt_numbers = [
                int(failed_attempt["attempt_number"]) + 1
            ]
        else:
            _assert_fresh_feature_namespace_empty(args.output_dir)
            previous_error = ""
            attempt_numbers = list(range(1, args.repair_attempts + 2))
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        try:
            for attempt in attempt_numbers:
                attempt_request = request
                if args.resume_failed_plan or attempt > 1:
                    original_inputs = request.get("input")
                    if not isinstance(original_inputs, list) or not original_inputs:
                        raise ValueError(
                            "schema repair requires the complete original inputs"
                        )
                    original_prompt = original_inputs[0]
                    if (
                        not isinstance(original_prompt, dict)
                        or original_prompt.get("type") != "text"
                        or not isinstance(original_prompt.get("text"), str)
                    ):
                        raise ValueError(
                            "schema repair requires the original text prompt first"
                        )
                    repair_prompt = (
                        str(original_prompt["text"])
                        + "\n\n## 前次輸出未通過本機 contract\n"
                        + previous_error[:6000]
                        + "\n請重新產生完整結果，不得只回傳修補片段。"
                        "所有原始候選影片與音樂仍附在本次 request；"
                        "必須重新觀看並修正契約錯誤。"
                    )
                    attempt_request = {
                        **request,
                        "input": [
                            {"type": "text", "text": repair_prompt},
                            *original_inputs[1:],
                        ],
                        "response_format": {
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": gemini_response_schema(response_model),
                        },
                        "generation_config": {
                            "thinking_level": "low",
                            "max_output_tokens": 32_000,
                        },
                    }
                attempt_stem = f"clip-card-feature-plan.attempt-{attempt:02d}"
                attempt_request_path = args.output_dir / f"{attempt_stem}.request.json"
                attempt_raw_interaction_path = (
                    args.output_dir / f"{attempt_stem}.raw_interaction.json"
                )
                attempt_raw_output_path = args.output_dir / f"{attempt_stem}.raw_output.json"
                write_json(attempt_request_path, attempt_request)
                current = client.interactions.create(**attempt_request)
                raw = _raw_dump(current)
                write_json(attempt_raw_interaction_path, raw)
                write_json(attempt_raw_output_path, {"output_text": current.output_text})
                try:
                    canonical_text, attempt_canonical_path, attempt_audit_path = (
                        _write_feature_normalization_artifacts(
                            output_dir=args.output_dir,
                            artifact_stem=attempt_stem,
                            raw_output_path=attempt_raw_output_path,
                            raw_output_text=current.output_text,
                            direct_video_plan=args.candidate_video_evidence,
                        )
                    )
                    if args.candidate_video_evidence:
                        assert shortlist is not None
                        direct_video_plan = DirectVideoEditPlan.model_validate_json(
                            canonical_text
                        )
                        plan = project_direct_video_edit_plan(
                            direct_video_plan,
                            shortlist=shortlist,
                            candidate_depth=args.candidate_video_depth,
                            brief=brief,
                            catalog=catalog,
                            cards=cards,
                            provenance=provenance,
                        )
                    else:
                        plan = ClipCardFeaturePlanV3.model_validate_json(
                            canonical_text
                        )
                    validate_plan_contract_v3(
                        plan,
                        brief=brief,
                        catalog=catalog,
                        cards=cards,
                        supplements=supplements,
                        direct_video_observed_events=(
                            direct_video_observed_events
                        ),
                    )
                    validate_shortlist_membership(plan)
                    interaction_id = getattr(current, "id", None) or ""
                    source_request_path = args.output_dir / "clip-card-feature-plan.request.json"
                    source_raw_interaction_path = (
                        args.output_dir / "clip-card-feature-plan.raw_interaction.json"
                    )
                    source_raw_output_path = (
                        args.output_dir / "clip-card-feature-plan.raw_output.json"
                    )
                    write_json(source_request_path, attempt_request)
                    write_json(source_raw_interaction_path, raw)
                    write_json(source_raw_output_path, {"output_text": current.output_text})
                    canonical_text, canonical_output_path, normalization_audit_path = (
                        _write_feature_normalization_artifacts(
                            output_dir=args.output_dir,
                            artifact_stem="clip-card-feature-plan",
                            raw_output_path=source_raw_output_path,
                            raw_output_text=current.output_text,
                            direct_video_plan=args.candidate_video_evidence,
                        )
                    )
                    if direct_video_plan is not None:
                        write_json(
                            args.output_dir / "direct-video-edit-plan.json",
                            direct_video_plan,
                        )
                    break
                except (ValidationError, ValueError) as error:
                    plan = None
                    previous_error = str(error)
                    write_json(
                        args.output_dir / f"{attempt_stem}.schema-validation.json",
                        {
                            "ok": False,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
            if plan is None:
                raise ValueError(
                    "Clip Card feature plan failed after "
                    f"{len(attempt_numbers)} new attempt(s): {previous_error}"
                )
        finally:
            client.close()
    assert plan is not None
    final_audit = plan.model_copy(
        update={
            "model_provenance": plan.model_provenance.model_copy(
                update={"interaction_id": interaction_id}
            )
        }
    )
    selected_evidence = build_selected_clip_card_evidence(
        final_audit,
        cards=cards,
    )
    final_plan = project_feature_contracts_v3(
        final_audit,
        brief=brief,
        catalog=catalog,
        selected_evidence=selected_evidence,
    )
    write_json(args.output_dir / "clip-card-feature-plan.json", final_audit)
    write_json(
        args.output_dir / "selected-clip-card-evidence.json",
        selected_evidence,
    )
    write_json(args.output_dir / "feature_edit_plan.json", final_plan)
    write_json(
        args.output_dir / "clip-card-feature-plan.schema-validation.json",
        {"ok": True, "clip_card_count": len(cards), "frame_count": len(frames)},
    )
    if shortlist_path is not None:
        extra_projection_artifacts["feature_shortlist"] = shortlist_path
    if music_path is not None:
        extra_projection_artifacts["source_music"] = music_path
    if candidate_video_manifest_path is not None:
        extra_projection_artifacts[
            "candidate_video_evidence_manifest"
        ] = candidate_video_manifest_path
    direct_video_plan_path = args.output_dir / "direct-video-edit-plan.json"
    if direct_video_plan_path.exists():
        extra_projection_artifacts["derived_clip_card_feature_plan"] = (
            args.output_dir / "clip-card-feature-plan.json"
        )
        if music_sha256 is not None:
            file_cache_root = (
                args.file_cache_root.expanduser().resolve()
                if args.file_cache_root is not None
                else args.output_dir.parent / "file-cache"
            )
            extra_projection_artifacts["source_music_upload"] = (
                file_cache_root
                / music_sha256
                / "music-upload"
                / "file_upload_final.json"
            )
    projection_contract_id = (
        "direct-video-edit-plan-v2"
        if direct_video_plan_path.exists()
        else "clip-card-feature-cut-v3"
    )
    source_plan_path = (
        direct_video_plan_path
        if direct_video_plan_path.exists()
        else args.output_dir / "clip-card-feature-plan.json"
    )
    projection_pointer = write_external_feature_plan_projection(
        plan_dir=args.output_dir,
        projection_contract_id=projection_contract_id,
        catalog_path=args.catalog_json,
        brief_path=args.brief_json,
        feature_plan_path=args.output_dir / "feature_edit_plan.json",
        source_plan_path=source_plan_path,
        source_request_path=source_request_path,
        source_artifacts={
            "source_raw_interaction": source_raw_interaction_path,
            # The projection validator must parse the exact canonical source
            # used to build the saved plan.  The immutable paid response stays
            # separately hash-bound for audit and replay.
            "source_raw_output": canonical_output_path,
            "original_raw_output": source_raw_output_path,
            "canonicalized_output": canonical_output_path,
            "normalization_audit": normalization_audit_path,
            "selected_clip_card_evidence": (
                args.output_dir / "selected-clip-card-evidence.json"
            ),
            **extra_projection_artifacts,
        },
    )
    if args.reuse_raw_output:
        _assert_projection_request_hash(
            pointer_path=projection_pointer,
            plan_dir=args.output_dir,
            expected_request_path=artifacts["request"],
        )
    usage_paths = sorted(
        args.output_dir.glob("clip-card-feature-plan.attempt-*.raw_interaction.json")
    )
    if not usage_paths:
        usage_paths = [source_raw_interaction_path]
    pricing = summarize_usage_files(
        usage_paths,
        relative_to=args.output_dir,
    )
    write_json(args.output_dir / "pricing.json", pricing)
    print(
        json.dumps(
            {
                "clip_card_count": len(cards),
                "chapter_count": len(final_plan.chapters),
                "pricing": pricing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
