"""Deterministic presentation compiler for autonomous feature delivery.

This module owns pixel/layout decisions.  Semantic planners may declare a
relation mode, but never panel orientation, crop coordinates, scale, or motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Literal, Mapping, Sequence

from PIL import Image
from pydantic import Field, model_validator

from .autonomous_policy import AutonomousEditPolicy
from .editing_capabilities import autonomous_capability_registry_v2
from .event_lock import ExactEventLockV2
from .media import extract_frames_bounded
from .models import (
    ExtractedFrame,
    FrozenStrictModel,
    SegmentationTrack,
    SharedSam21BBoxSeed,
)
from .sequence_optimizer import (
    ConstraintResult,
    ExecutableOptionV2,
    ExecutableOptionSelectionV2,
    OptionMetrics,
    select_executable_option,
)
from .storage import read_json, write_json


NormalizedBox = tuple[int, int, int, int]
PresentationFeasibilityStatus = Literal[
    "known_feasible",
    "known_infeasible",
    "needs_exact_event",
    "needs_bbox",
    "needs_sam",
]
PresentationCapability = Literal[
    "static_full_bleed_crop",
    "tracked_full_bleed_crop",
    "phase_virtual_camera",
    "hard_cut_between_views",
    "two_panel_layout",
    "solid_matte_fit",
]

_SOURCE_MOTION_ESTIMATOR_V1 = "background-gftt-lk-ransac-affine-v1"
_SOURCE_MOTION_ESTIMATOR_V2 = "background-gftt-lk-ransac-affine-v2"
_SOURCE_MOTION_ESTIMATOR_V3 = "background-gftt-lk-ransac-affine-batch-v3"
_SOURCE_MOTION_DECODER_V1 = "bounded-batch-source-pts-v1"
_SOURCE_MOTION_SAMPLING_V1 = "legacy-uniform-or-sam-v1"
_SOURCE_MOTION_SAMPLING_V2 = "hybrid-edge-dense-bounded-gap-v2"
_SOURCE_MOTION_MAX_SAMPLE_GAP_MS = 250
_SOURCE_MOTION_EDGE_DENSE_WINDOW_MS = 750
_SOURCE_MOTION_EDGE_SAMPLE_STEP_MS = 100
_SOURCE_MOTION_EDGE_SETTLE_PADDING_MS = 200


def source_motion_estimator_binding_sha256() -> str:
    """Fingerprint every implementation constant that changes motion evidence."""

    payload = {
        "estimator": _SOURCE_MOTION_ESTIMATOR_V3,
        "decoder": _SOURCE_MOTION_DECODER_V1,
        "sampling": _SOURCE_MOTION_SAMPLING_V2,
        "maximum_sample_gap_ms": _SOURCE_MOTION_MAX_SAMPLE_GAP_MS,
        "edge_dense_window_ms": _SOURCE_MOTION_EDGE_DENSE_WINDOW_MS,
        "edge_sample_step_ms": _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS,
        "edge_settle_padding_ms": _SOURCE_MOTION_EDGE_SETTLE_PADDING_MS,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class GroundingTargetRequest(FrozenStrictModel):
    target_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$",
    )
    target_description: str = Field(min_length=1)
    exclusions: tuple[str, ...] = ()


class PresentationModeFeasibility(FrozenStrictModel):
    """One honest pre-paid feasibility statement.

    Deferred evidence is never represented as success. This lets the global
    candidate frontier distinguish a known impossible option from one that is
    still allowed to consume its specifically reserved exact/bbox/SAM node.
    """

    capability_id: PresentationCapability
    status: PresentationFeasibilityStatus
    reason_codes: tuple[str, ...] = Field(min_length=1)


class PresentationFeasibilityLatticeV1(FrozenStrictModel):
    contract_version: Literal["presentation-feasibility-lattice-v1"] = (
        "presentation-feasibility-lattice-v1"
    )
    candidate_id: str = Field(min_length=1)
    relation_mode: Literal[
        "single_subject",
        "sequential_focus",
        "simultaneous_relation",
        "context_detail",
    ]
    modes: tuple[PresentationModeFeasibility, ...] = Field(min_length=6)

    @model_validator(mode="after")
    def validate_modes(self) -> "PresentationFeasibilityLatticeV1":
        ids = [mode.capability_id for mode in self.modes]
        if len(ids) != len(set(ids)):
            raise ValueError("presentation feasibility modes must be unique")
        return self

    def assessment(
        self,
        capability_id: PresentationCapability,
    ) -> PresentationModeFeasibility:
        return next(
            mode
            for mode in self.modes
            if mode.capability_id == capability_id
        )


def assess_prepaid_presentation_feasibility(
    *,
    candidate_id: str,
    aspect_suitability: Literal[
        "natural",
        "reconstructable",
        "unsuitable",
    ],
    relation_mode: Literal[
        "single_subject",
        "sequential_focus",
        "simultaneous_relation",
        "context_detail",
    ],
    hard_region_count: int,
    has_virtual_camera_proposal: bool,
    physical_scale_comparison: bool,
    has_atomic_or_text_region: bool,
    policy: AutonomousEditPolicy,
) -> PresentationFeasibilityLatticeV1:
    """Classify every local presentation family before a paid refinement.

    This is deliberately conservative. Geometry-dependent modes remain
    ``needs_bbox``/``needs_sam`` instead of being optimistically declared
    feasible, while policy and topology failures are rejected immediately.
    """

    if hard_region_count < 0:
        raise ValueError("hard region count cannot be negative")
    if aspect_suitability == "unsuitable":
        return PresentationFeasibilityLatticeV1(
            candidate_id=candidate_id,
            relation_mode=relation_mode,
            modes=tuple(
                PresentationModeFeasibility(
                    capability_id=capability_id,
                    status="known_infeasible",
                    reason_codes=("aspect_declared_unsuitable",),
                )
                for capability_id in (
                    "static_full_bleed_crop",
                    "tracked_full_bleed_crop",
                    "phase_virtual_camera",
                    "hard_cut_between_views",
                    "two_panel_layout",
                    "solid_matte_fit",
                )
            ),
        )

    target_gate: PresentationFeasibilityStatus = (
        "needs_sam" if hard_region_count else "needs_bbox"
    )
    modes: list[PresentationModeFeasibility] = [
        PresentationModeFeasibility(
            capability_id="static_full_bleed_crop",
            status="needs_bbox",
            reason_codes=("required_scope_geometry_unresolved",),
        ),
        PresentationModeFeasibility(
            capability_id="tracked_full_bleed_crop",
            status=target_gate,
            reason_codes=(
                "whole_window_tracking_unresolved"
                if hard_region_count
                else "grounding_seed_unresolved",
            ),
        ),
        PresentationModeFeasibility(
            capability_id="phase_virtual_camera",
            status=(
                target_gate
                if has_virtual_camera_proposal
                else "known_infeasible"
            ),
            reason_codes=(
                ("phase_geometry_and_tracking_unresolved",)
                if has_virtual_camera_proposal
                else ("no_semantic_phase_proposal",)
            ),
        ),
        PresentationModeFeasibility(
            capability_id="hard_cut_between_views",
            status=(
                "needs_exact_event"
                if relation_mode == "sequential_focus"
                else "known_infeasible"
            ),
            reason_codes=(
                ("view_transition_event_unresolved",)
                if relation_mode == "sequential_focus"
                else ("relation_does_not_authorize_sequential_reconstruction",)
            ),
        ),
    ]

    panel_relation_allowed = relation_mode in {
        "simultaneous_relation",
        "context_detail",
    }
    panel_allowed = (
        policy.presentation.allow_two_panel_layout
        and panel_relation_allowed
        and (
            hard_region_count >= 2
            or relation_mode == "context_detail"
        )
    )
    panel_reason = (
        "panel_geometry_and_readability_unresolved"
        if panel_allowed
        else "two_panel_not_authorized_for_candidate_topology"
    )
    if physical_scale_comparison and panel_allowed:
        panel_reason = "relative_scale_and_same_pts_unresolved"
    modes.append(
        PresentationModeFeasibility(
            capability_id="two_panel_layout",
            status=(
                "needs_exact_event"
                if panel_allowed and physical_scale_comparison
                else target_gate
                if panel_allowed
                else "known_infeasible"
            ),
            reason_codes=(panel_reason,),
        )
    )

    fit_allowed = policy.presentation.allow_solid_matte_fit
    modes.append(
        PresentationModeFeasibility(
            capability_id="solid_matte_fit",
            status=(
                "needs_bbox"
                if fit_allowed and has_atomic_or_text_region
                else "known_feasible"
                if fit_allowed
                else "known_infeasible"
            ),
            reason_codes=(
                ("atomic_readability_unresolved",)
                if fit_allowed and has_atomic_or_text_region
                else ("whole_source_scope_preserved",)
                if fit_allowed
                else ("solid_matte_fit_forbidden_by_policy",)
            ),
        )
    )
    return PresentationFeasibilityLatticeV1(
        candidate_id=candidate_id,
        relation_mode=relation_mode,
        modes=tuple(modes),
    )


class MultiTargetGroundingCandidate(FrozenStrictModel):
    box_2d_yxyx: NormalizedBox
    confidence: float = Field(ge=0.0, le=1.0)
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_box(self) -> "MultiTargetGroundingCandidate":
        y_min, x_min, y_max, x_max = self.box_2d_yxyx
        if not (0 <= y_min < y_max <= 1000):
            raise ValueError("grounding y coordinates are invalid")
        if not (0 <= x_min < x_max <= 1000):
            raise ValueError("grounding x coordinates are invalid")
        return self


class MultiTargetGroundingResult(FrozenStrictModel):
    target_id: str
    visible: bool
    candidates: tuple[MultiTargetGroundingCandidate, ...] = Field(
        max_length=5
    )
    ambiguity_reason: str | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> "MultiTargetGroundingResult":
        if self.visible and not self.candidates:
            raise ValueError("visible target requires a grounding candidate")
        if not self.visible and self.candidates:
            raise ValueError("invisible target cannot claim candidate boxes")
        return self


class MultiTargetGroundingGroup(FrozenStrictModel):
    contract_version: Literal["multi-target-grounding-v1"] = (
        "multi-target-grounding-v1"
    )
    source_asset_id: str
    event_lock_id: str
    source_frame_id: str
    source_frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    targets: tuple[MultiTargetGroundingResult, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_targets(self) -> "MultiTargetGroundingGroup":
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("multi-target grounding IDs must be unique")
        return self

    @property
    def ambiguous_target_ids(self) -> tuple[str, ...]:
        return tuple(
            target.target_id
            for target in self.targets
            if target.visible
            and (
                len(target.candidates) != 1
                or target.ambiguity_reason is not None
            )
        )


class PresentationTarget(FrozenStrictModel):
    target_id: str
    source_asset_id: str
    source_pts: int
    box_2d: NormalizedBox
    track_boxes_by_pts: tuple[tuple[int, NormalizedBox], ...] = ()
    minimum_visible_fraction: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_box(self) -> "PresentationTarget":
        x_min, y_min, x_max, y_max = self.box_2d
        if not (0 <= x_min < x_max <= 1000):
            raise ValueError("target x coordinates are invalid")
        if not (0 <= y_min < y_max <= 1000):
            raise ValueError("target y coordinates are invalid")
        pts_values: list[int] = []
        for source_pts, box in self.track_boxes_by_pts:
            track_x_min, track_y_min, track_x_max, track_y_max = box
            if not (0 <= track_x_min < track_x_max <= 1000):
                raise ValueError("tracked target x coordinates are invalid")
            if not (0 <= track_y_min < track_y_max <= 1000):
                raise ValueError("tracked target y coordinates are invalid")
            pts_values.append(source_pts)
        if pts_values != sorted(set(pts_values)):
            raise ValueError("tracked target PTS values must be strictly increasing")
        return self


def _aggregate_presentation_group(
    *,
    targets: Sequence[PresentationTarget],
    target_ids: Sequence[str],
    group_index: int,
) -> PresentationTarget:
    """Collapse one semantic pane group without losing member provenance."""

    target_by_id = {target.target_id: target for target in targets}
    try:
        members = [target_by_id[target_id] for target_id in target_ids]
    except KeyError as error:
        raise ValueError(
            f"panel group references unknown target: {error.args[0]}"
        ) from error
    source_bindings = {
        (member.source_asset_id, member.source_pts) for member in members
    }
    if len(source_bindings) != 1:
        raise ValueError(
            "targets inside one panel group must share source and PTS"
        )
    source_asset_id, source_pts = next(iter(source_bindings))
    envelope = (
        min(member.box_2d[0] for member in members),
        min(member.box_2d[1] for member in members),
        max(member.box_2d[2] for member in members),
        max(member.box_2d[3] for member in members),
    )
    samples_by_member = [
        dict(member.track_boxes_by_pts) for member in members
    ]
    common_pts = (
        set.intersection(
            *(set(samples) for samples in samples_by_member)
        )
        if samples_by_member and all(samples_by_member)
        else set()
    )
    group_samples = tuple(
        (
            sample_pts,
            (
                min(samples[sample_pts][0] for samples in samples_by_member),
                min(samples[sample_pts][1] for samples in samples_by_member),
                max(samples[sample_pts][2] for samples in samples_by_member),
                max(samples[sample_pts][3] for samples in samples_by_member),
            ),
        )
        for sample_pts in sorted(common_pts)
    )
    return PresentationTarget(
        target_id=f"semantic_panel_group_{group_index + 1}",
        source_asset_id=source_asset_id,
        source_pts=source_pts,
        box_2d=envelope,
        track_boxes_by_pts=group_samples,
        minimum_visible_fraction=max(
            member.minimum_visible_fraction for member in members
        ),
    )


class PanelRect(FrozenStrictModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    width: int = Field(gt=0, le=1000)
    height: int = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def validate_rect(self) -> "PanelRect":
        if self.x + self.width > 1000 or self.y + self.height > 1000:
            raise ValueError("panel rectangle lies outside normalized canvas")
        return self


class PanelSpec(FrozenStrictModel):
    panel_id: str
    source_asset_id: str
    source_pts: int
    target_ids: tuple[str, ...] = Field(min_length=1)
    crop_box_2d: NormalizedBox
    output_rect: PanelRect


class PanelLayoutSpec(FrozenStrictModel):
    contract_version: Literal["panel-layout-spec-v1"] = "panel-layout-spec-v1"
    capability_id: Literal["two_panel_layout"] = "two_panel_layout"
    layout_mode: Literal["top_bottom", "side_by_side", "context_detail"]
    relation_mode: Literal[
        "simultaneous_comparison",
        "simultaneous_relation",
        "context_detail",
        "conceptual_comparison",
    ]
    temporal_relation: Literal[
        "same_source_same_pts",
        "different_source_conceptual",
    ]
    relative_scale_policy: Literal["locked", "independent_nonphysical"]
    panels: tuple[PanelSpec, PanelSpec]
    gutter_normalized: float = Field(default=0.018, ge=0.0, le=0.1)
    background: Literal["solid"] = "solid"
    transition_policy: Literal["static"] = "static"
    local_layout_score: float

    @model_validator(mode="after")
    def validate_layout(self) -> "PanelLayoutSpec":
        first, second = self.panels
        same_source = first.source_asset_id == second.source_asset_id
        if self.temporal_relation == "same_source_same_pts":
            if not same_source or first.source_pts != second.source_pts:
                raise ValueError("same-source panels must bind the same PTS")
        elif same_source:
            raise ValueError("same-source panels cannot claim different-source mode")
        if (
            self.relative_scale_policy == "locked"
            and self.temporal_relation != "same_source_same_pts"
        ):
            raise ValueError("physical relative scale requires same-source same-PTS")
        return self


class IntentionalFreezeSpec(FrozenStrictModel):
    contract_version: Literal["intentional-freeze-spec-v1"] = (
        "intentional-freeze-spec-v1"
    )
    capability_id: Literal["intentional_freeze"] = "intentional_freeze"
    exact_event_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asset_id: str
    source_pts: int
    cue_id: str
    duration_ms: int = Field(gt=0, le=5_000)
    motivation: Literal["brief_authorized_phrase_ending"]


class SceneFacts(FrozenStrictModel):
    """Continuous local measurements; never a product/person case label."""

    contract_version: Literal[
        "presentation-scene-facts-v1",
        "presentation-scene-facts-v2",
    ] = (
        "presentation-scene-facts-v2"
    )
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    target_ids: tuple[str, ...] = Field(min_length=1)
    target_center_x: tuple[float, ...] = Field(min_length=1)
    target_center_y: tuple[float, ...] = ()
    target_width_fractions: tuple[float, ...] = Field(min_length=1)
    target_height_fractions: tuple[float, ...] = ()
    normalized_distance_matrix: tuple[tuple[float, ...], ...]
    normalized_xy_distance_matrix: tuple[tuple[float, ...], ...] = ()
    intersection_over_union_matrix: tuple[tuple[float, ...], ...] = ()
    containment_fraction_matrix: tuple[tuple[float, ...], ...] = ()
    nested_target_pairs: tuple[tuple[str, str], ...] = ()
    aligned_track_sample_count_matrix: tuple[tuple[int, ...], ...] = ()
    co_visibility_fraction_matrix: tuple[tuple[float, ...], ...] = ()
    common_motion_residual_matrix: tuple[tuple[float | None, ...], ...] = ()
    common_motion_pairs: tuple[tuple[str, str], ...] = ()
    shared_static_crop_feasible: bool
    shared_tracked_crop_feasible: bool | None = None
    source_camera_motion: SourceCameraMotionEstimate
    source_camera_motion_measured: bool = False
    source_camera_motion_reliable: bool = False
    source_camera_motion_evidence_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    target_readability_by_canvas: Mapping[str, float]
    target_short_edge_pixels_by_mode: Mapping[
        str,
        tuple[int, ...],
    ] = Field(default_factory=dict)
    same_source_same_pts: bool
    physical_scale_evidence_available: bool

    @model_validator(mode="after")
    def validate_facts(self) -> "SceneFacts":
        count = len(self.target_ids)
        if len(set(self.target_ids)) != count:
            raise ValueError("scene fact target IDs must be unique")
        if len(self.target_center_x) != count:
            raise ValueError("scene fact target centers are incomplete")
        if len(self.target_width_fractions) != count:
            raise ValueError("scene fact target widths are incomplete")
        if self.contract_version == "presentation-scene-facts-v1":
            if len(self.normalized_distance_matrix) != count or any(
                len(row) != count
                for row in self.normalized_distance_matrix
            ):
                raise ValueError(
                    "scene fact distance matrix dimensions are invalid"
                )
            return self
        if len(self.target_center_y) != count:
            raise ValueError("scene fact target vertical centers are incomplete")
        if len(self.target_height_fractions) != count:
            raise ValueError("scene fact target heights are incomplete")
        matrices = (
            self.normalized_distance_matrix,
            self.normalized_xy_distance_matrix,
            self.intersection_over_union_matrix,
            self.containment_fraction_matrix,
            self.aligned_track_sample_count_matrix,
            self.co_visibility_fraction_matrix,
            self.common_motion_residual_matrix,
        )
        if any(
            len(matrix) != count
            or any(len(row) != count for row in matrix)
            for matrix in matrices
        ):
            raise ValueError("scene fact matrix dimensions are invalid")
        if any(
            len(values) != count
            for values in self.target_short_edge_pixels_by_mode.values()
        ):
            raise ValueError("scene fact pixel measurements are incomplete")
        return self

    @property
    def definition_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        if self.contract_version == "presentation-scene-facts-v1":
            v1_fields = (
                "contract_version",
                "source_width",
                "source_height",
                "target_ids",
                "target_center_x",
                "target_width_fractions",
                "normalized_distance_matrix",
                "shared_static_crop_feasible",
                "source_camera_motion",
                "source_camera_motion_measured",
                "source_camera_motion_reliable",
                "source_camera_motion_evidence_sha256",
                "target_readability_by_canvas",
                "same_source_same_pts",
                "physical_scale_evidence_available",
            )
            payload = {field: payload[field] for field in v1_fields}
        return _canonical_payload_sha(payload)


class PresentationOption(FrozenStrictModel):
    option: ExecutableOptionV2
    mode: Literal[
        "static_full_bleed_crop",
        "tracked_full_bleed_crop",
        "phase_virtual_camera",
        "hard_cut_between_views",
        "two_panel_layout",
        "solid_matte_fit",
    ]
    static_crop_box_2d: NormalizedBox | None = None
    panel_layout: PanelLayoutSpec | None = None
    filter_graph: str | None = None
    camera_motion: CameraMotionDecision | None = None


class PresentationOptionSet(FrozenStrictModel):
    contract_version: Literal["presentation-option-set-v2"] = (
        "presentation-option-set-v2"
    )
    scene_facts: SceneFacts
    options: tuple[PresentationOption, ...] = Field(min_length=1)
    paid_model_calls_added: Literal[0] = 0

    @model_validator(mode="after")
    def validate_options(self) -> "PresentationOptionSet":
        ids = [item.option.option_id for item in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("presentation option IDs must be unique")
        return self


class PresentationCompilation(FrozenStrictModel):
    contract_version: Literal[
        "presentation-compilation-v1",
        "presentation-compilation-v2",
    ] = (
        "presentation-compilation-v2"
    )
    mode: Literal[
        "static_full_bleed_crop",
        "tracked_full_bleed_crop",
        "phase_virtual_camera",
        "hard_cut_between_views",
        "two_panel_layout",
        "solid_matte_fit",
        "blocked",
    ]
    static_crop_box_2d: NormalizedBox | None = None
    panel_layout: PanelLayoutSpec | None = None
    filter_graph: str | None = None
    camera_motion: CameraMotionDecision | None = None
    scene_facts: SceneFacts | None = None
    # Persist the complete bounded frontier that produced ``selection``.
    # Recovery may replay only an option in this immutable set whose hard
    # constraints passed; it must never reconstruct a new crop from a QA
    # observation after render time.
    option_set: PresentationOptionSet | None = None
    selection: ExecutableOptionSelectionV2 | None = None
    decision_codes: tuple[str, ...]
    paid_model_calls_added: Literal[0] = 0

    @model_validator(mode="after")
    def validate_payload(self) -> "PresentationCompilation":
        if self.mode == "static_full_bleed_crop" and self.static_crop_box_2d is None:
            raise ValueError("static crop mode requires a crop box")
        if self.mode == "two_panel_layout" and self.panel_layout is None:
            raise ValueError("two-panel mode requires a layout spec")
        if self.mode == "solid_matte_fit" and self.filter_graph is None:
            raise ValueError("solid fit requires an executable filter graph")
        return self


class SourceCameraMotionEstimate(FrozenStrictModel):
    direction: Literal[
        "static",
        "left",
        "right",
        "up",
        "down",
        "zoom_in",
        "zoom_out",
        "mixed",
        "unreliable",
    ]
    normalized_travel: float = Field(ge=0.0)
    reversal_count: int = Field(ge=0)


class SourceCameraMotionPairEvidence(FrozenStrictModel):
    before_frame_pts: int
    after_frame_pts: int
    delta_ms: int = Field(gt=0)
    before_time_ms: int | None = Field(default=None, ge=0)
    after_time_ms: int | None = Field(default=None, ge=0)
    detected_features: int = Field(ge=0)
    tracked_background_features: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0.0, le=1.0)
    median_residual_pixels: float = Field(ge=0.0)
    camera_translation_x_normalized: float
    camera_translation_y_normalized: float
    camera_scale_delta: float
    camera_rotation_degrees: float
    normalized_translation_speed_per_second: float = Field(
        default=0.0,
        ge=0.0,
    )
    isolated_jolt: bool = False
    dirty_edge: Literal["head", "tail"] | None = None
    reliable: bool

    @model_validator(mode="after")
    def validate_pair_times(self) -> "SourceCameraMotionPairEvidence":
        if (self.before_time_ms is None) != (self.after_time_ms is None):
            raise ValueError("source motion pair times must be both present or absent")
        if (
            self.before_time_ms is not None
            and self.after_time_ms is not None
            and self.after_time_ms <= self.before_time_ms
        ):
            raise ValueError("source motion pair times must be strictly increasing")
        return self


class SourceCameraMotionEvidence(FrozenStrictModel):
    """Auditable background-geometry evidence for one immutable trim.

    The estimator never decides the editorial purpose of motion. It measures
    the source camera so the presentation compiler can avoid canceling or
    stacking synthetic motion on top of the photographed move.
    """

    contract_version: Literal[
        "source-camera-motion-evidence-v1",
        "source-camera-motion-evidence-v2",
    ] = "source-camera-motion-evidence-v1"
    estimator_version: Literal[
        "background-gftt-lk-ransac-affine-v1",
        "background-gftt-lk-ransac-affine-v2",
        "background-gftt-lk-ransac-affine-batch-v3",
    ] = _SOURCE_MOTION_ESTIMATOR_V1
    sampling_version: Literal[
        "legacy-uniform-or-sam-v1",
        "hybrid-edge-dense-bounded-gap-v2",
    ] = _SOURCE_MOTION_SAMPLING_V1
    requested_max_sample_gap_ms: int | None = Field(default=None, gt=0)
    actual_max_sample_gap_ms: int | None = Field(default=None, ge=0)
    head_sample_coverage_ms: int | None = Field(default=None, ge=0)
    tail_sample_coverage_ms: int | None = Field(default=None, ge=0)
    sampling_complete: bool = False
    source_asset_id: str
    window_start_ms: int = Field(ge=0)
    window_end_ms: int = Field(gt=0)
    sample_times_ms: tuple[int, ...]
    sample_frame_pts: tuple[int, ...]
    sample_frame_hashes: tuple[str, ...]
    subject_exclusion_mode: Literal[
        "sam_track_boxes",
        "none",
    ]
    mean_excluded_area_fraction: float = Field(ge=0.0, le=1.0)
    pairs: tuple[SourceCameraMotionPairEvidence, ...]
    classification: Literal[
        "static",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "zoom_in",
        "zoom_out",
        "mixed",
        "unreliable",
    ]
    reliable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    normalized_translation_x_per_second: float
    normalized_translation_y_per_second: float
    scale_rate_per_second: float
    rotation_degrees_per_second: float
    normalized_travel: float = Field(ge=0.0)
    reversal_count: int = Field(ge=0)
    p95_translation_speed_per_second: float = Field(default=0.0, ge=0.0)
    max_translation_speed_per_second: float = Field(default=0.0, ge=0.0)
    max_translation_acceleration_per_second_squared: float = Field(
        default=0.0,
        ge=0.0,
    )
    max_translation_jerk_per_second_cubed: float = Field(
        default=0.0,
        ge=0.0,
    )
    isolated_jolt_count: int = Field(default=0, ge=0)
    jolt_pair_indexes: tuple[int, ...] = ()
    dirty_head: bool = False
    dirty_tail: bool = False
    clean_head_start_ms: int | None = Field(default=None, ge=0)
    clean_tail_end_ms: int | None = Field(default=None, ge=0)
    reason_codes: tuple[str, ...]
    cache_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_window_and_samples(self) -> "SourceCameraMotionEvidence":
        if self.window_end_ms <= self.window_start_ms:
            raise ValueError("source camera motion window must be non-empty")
        if len(self.sample_times_ms) != len(self.sample_frame_pts):
            raise ValueError("source camera sample PTS count is incomplete")
        if len(self.sample_times_ms) != len(self.sample_frame_hashes):
            raise ValueError("source camera sample hashes are incomplete")
        if self.reliable and len(self.sample_times_ms) < 2:
            raise ValueError("reliable source motion needs at least two frames")
        if tuple(sorted(self.sample_times_ms)) != self.sample_times_ms:
            raise ValueError("source camera sample times must be monotonic")
        if self.reliable == (self.classification == "unreliable"):
            raise ValueError(
                "source camera reliability and classification disagree"
            )
        if len(set(self.jolt_pair_indexes)) != len(self.jolt_pair_indexes):
            raise ValueError("source camera jolt pair indexes must be unique")
        if any(
            index < 0 or index >= len(self.pairs)
            for index in self.jolt_pair_indexes
        ):
            raise ValueError("source camera jolt pair index is out of range")
        if self.isolated_jolt_count != len(self.jolt_pair_indexes):
            raise ValueError("source camera jolt count disagrees with pair indexes")
        if self.contract_version == "source-camera-motion-evidence-v2":
            if self.sampling_version != _SOURCE_MOTION_SAMPLING_V2:
                raise ValueError("V2 source motion requires V2 sampling evidence")
            if self.pairs and (
                len(self.pairs) != max(0, len(self.sample_times_ms) - 1)
            ):
                raise ValueError("V2 source motion pairs must cover adjacent samples")
            if self.sample_times_ms and (
                self.head_sample_coverage_ms is None
                or self.tail_sample_coverage_ms is None
                or self.actual_max_sample_gap_ms is None
                or self.requested_max_sample_gap_ms is None
            ):
                raise ValueError("V2 source motion sampling coverage is incomplete")
            if self.sampling_complete and (
                self.actual_max_sample_gap_ms is None
                or self.requested_max_sample_gap_ms is None
                or self.head_sample_coverage_ms is None
                or self.tail_sample_coverage_ms is None
                or self.actual_max_sample_gap_ms
                > self.requested_max_sample_gap_ms + 100
                or self.head_sample_coverage_ms
                > _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS + 100
                or self.tail_sample_coverage_ms
                > _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS + 100
            ):
                raise ValueError(
                    "complete V2 sampling exceeds its bounded coverage"
                )
            if self.clean_head_start_ms is not None and not (
                self.window_start_ms
                <= self.clean_head_start_ms
                <= self.window_end_ms
            ):
                raise ValueError("clean source-motion head is outside the window")
            if self.clean_tail_end_ms is not None and not (
                self.window_start_ms
                <= self.clean_tail_end_ms
                <= self.window_end_ms
            ):
                raise ValueError("clean source-motion tail is outside the window")
            if (
                self.clean_head_start_ms is not None
                and self.clean_tail_end_ms is not None
                and self.clean_head_start_ms > self.clean_tail_end_ms
            ):
                raise ValueError("source-motion clean interval is empty")
        return self

    @property
    def definition_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        if self.contract_version == "source-camera-motion-evidence-v1":
            v1_pair_fields = (
                "before_frame_pts",
                "after_frame_pts",
                "delta_ms",
                "detected_features",
                "tracked_background_features",
                "inlier_ratio",
                "median_residual_pixels",
                "camera_translation_x_normalized",
                "camera_translation_y_normalized",
                "camera_scale_delta",
                "camera_rotation_degrees",
                "reliable",
            )
            payload["pairs"] = [
                {field: pair[field] for field in v1_pair_fields}
                for pair in payload["pairs"]
            ]
            v1_fields = (
                "contract_version",
                "estimator_version",
                "source_asset_id",
                "window_start_ms",
                "window_end_ms",
                "sample_times_ms",
                "sample_frame_pts",
                "sample_frame_hashes",
                "subject_exclusion_mode",
                "mean_excluded_area_fraction",
                "pairs",
                "classification",
                "reliable",
                "confidence",
                "normalized_translation_x_per_second",
                "normalized_translation_y_per_second",
                "scale_rate_per_second",
                "rotation_degrees_per_second",
                "normalized_travel",
                "reversal_count",
                "reason_codes",
                "cache_key_sha256",
            )
            payload = {field: payload[field] for field in v1_fields}
        return _canonical_payload_sha(payload)

    def as_motion_estimate(self) -> SourceCameraMotionEstimate:
        direction_by_classification = {
            "static": "static",
            "pan_left": "left",
            "pan_right": "right",
            "tilt_up": "up",
            "tilt_down": "down",
            "zoom_in": "zoom_in",
            "zoom_out": "zoom_out",
            "mixed": "mixed",
            "unreliable": "unreliable",
        }
        return SourceCameraMotionEstimate(
            direction=direction_by_classification[self.classification],
            normalized_travel=self.normalized_travel,
            reversal_count=self.reversal_count,
        )


class CameraMotionDecision(FrozenStrictModel):
    mode: Literal["hold", "minimal_monotonic_move", "hard_cut"]
    normalized_x_values: tuple[float, ...] = Field(min_length=1)
    movement_motivation: Literal[
        "none",
        "attention_transfer",
        "maintain_required_containment",
    ]
    source_motion: SourceCameraMotionEstimate
    synthetic_reversal_count: int = Field(ge=0)
    settle_required: bool


def compile_presentation(
    *,
    targets: Sequence[PresentationTarget],
    source_width: int,
    source_height: int,
    relation_mode: Literal[
        "single_subject",
        "sequential_focus",
        "simultaneous_relation",
        "context_detail",
    ],
    policy: AutonomousEditPolicy,
    physical_scale_comparison: bool = False,
    allow_conceptual_different_source: bool = False,
    allow_static_full_bleed: bool = True,
    tracking_available: bool = True,
    required_x_values: Sequence[float] = (),
    source_camera_x_values: Sequence[float] = (),
    source_camera_motion_evidence: SourceCameraMotionEvidence | None = None,
    movement_motivated: bool = False,
    source_motion_motivated: bool | None = None,
    preferred_capability_ids: Sequence[str] = (),
    acceptable_capability_ids: Sequence[str] = (),
    forbidden_capability_ids: Sequence[str] = (),
    required_readability_by_target: Mapping[str, float] | None = None,
    panel_semantically_admissible: bool = False,
    panel_target_groups: Sequence[Sequence[str]] = (),
) -> PresentationCompilation:
    """Compatibility wrapper over the v2 option generator and optimizer."""

    if not targets:
        return PresentationCompilation(
            mode="blocked",
            decision_codes=("no_required_targets",),
        )
    option_set = generate_presentation_options(
        targets=targets,
        source_width=source_width,
        source_height=source_height,
        relation_mode=relation_mode,
        policy=policy,
        physical_scale_comparison=physical_scale_comparison,
        allow_conceptual_different_source=allow_conceptual_different_source,
        allow_static_full_bleed=allow_static_full_bleed,
        tracking_available=tracking_available,
        required_x_values=required_x_values,
        source_camera_x_values=source_camera_x_values,
        source_camera_motion_evidence=source_camera_motion_evidence,
        movement_motivated=movement_motivated,
        source_motion_motivated=source_motion_motivated,
        acceptable_capability_ids=acceptable_capability_ids,
        forbidden_capability_ids=forbidden_capability_ids,
        required_readability_by_target=required_readability_by_target,
        panel_semantically_admissible=panel_semantically_admissible,
        panel_target_groups=panel_target_groups,
    )
    selection = select_executable_option(
        [item.option for item in option_set.options],
        preferred_capability_ids=preferred_capability_ids,
    )
    selected = next(
        (
            item
            for item in option_set.options
            if item.option.option_id == selection.selected_option_id
        ),
        None,
    )
    if selected is None:
        return PresentationCompilation(
            mode="blocked",
            scene_facts=option_set.scene_facts,
            option_set=option_set,
            selection=selection,
            decision_codes=selection.decision_codes,
        )
    return PresentationCompilation(
        mode=selected.mode,
        static_crop_box_2d=selected.static_crop_box_2d,
        panel_layout=selected.panel_layout,
        filter_graph=selected.filter_graph,
        camera_motion=selected.camera_motion,
        scene_facts=option_set.scene_facts,
        option_set=option_set,
        selection=selection,
        decision_codes=(
            *selected.option.decision_codes,
            *selection.decision_codes,
        ),
    )


def generate_presentation_options(
    *,
    targets: Sequence[PresentationTarget],
    source_width: int,
    source_height: int,
    relation_mode: Literal[
        "single_subject",
        "sequential_focus",
        "simultaneous_relation",
        "context_detail",
    ],
    policy: AutonomousEditPolicy,
    physical_scale_comparison: bool = False,
    allow_conceptual_different_source: bool = False,
    allow_static_full_bleed: bool = True,
    tracking_available: bool = True,
    required_x_values: Sequence[float] = (),
    source_camera_x_values: Sequence[float] = (),
    source_camera_motion_evidence: SourceCameraMotionEvidence | None = None,
    movement_motivated: bool = False,
    source_motion_motivated: bool | None = None,
    acceptable_capability_ids: Sequence[str] = (),
    forbidden_capability_ids: Sequence[str] = (),
    required_readability_by_target: Mapping[str, float] | None = None,
    panel_semantically_admissible: bool = False,
    panel_target_groups: Sequence[Sequence[str]] = (),
) -> PresentationOptionSet:
    """Enumerate all locally feasible operations before selecting one.

    This function has no content-category branches. It consumes target
    topology, relation intent, continuous motion facts and policy gates.
    """

    if not targets:
        raise ValueError("presentation option generation requires targets")
    # Synthetic framing motion and source-camera motion are different
    # decisions.  Preserve the legacy default for callers that have no
    # candidate-scoped source-motion semantics, while allowing a planner to
    # retain a measured, editorially useful source reveal without authorizing
    # synthetic movement.
    resolved_source_motion_motivated = (
        movement_motivated
        if source_motion_motivated is None
        else source_motion_motivated
    )
    registry = autonomous_capability_registry_v2(
        allow_two_panel_layout=policy.presentation.allow_two_panel_layout,
        allow_solid_matte_fit=policy.presentation.allow_solid_matte_fit,
        allow_intentional_freeze=policy.presentation.allow_intentional_freeze,
    )
    capability_specs = registry.by_id()
    shared_crop = (
        _largest_shared_vertical_crop(
            targets,
            source_width=source_width,
            source_height=source_height,
        )
        if allow_static_full_bleed
        else None
    )
    source_motion = (
        source_camera_motion_evidence.as_motion_estimate()
        if source_camera_motion_evidence is not None
        else estimate_source_camera_motion(
            source_camera_x_values or tuple(0.5 for _ in targets)
        )
    )
    scene_facts = _build_scene_facts(
        targets=targets,
        source_width=source_width,
        source_height=source_height,
        shared_crop=shared_crop,
        source_motion=source_motion,
        source_camera_motion_measured=(
            source_camera_motion_evidence is not None
            or bool(source_camera_x_values)
        ),
        source_camera_motion_reliable=(
            source_camera_motion_evidence.reliable
            if source_camera_motion_evidence is not None
            else bool(source_camera_x_values)
        ),
        source_camera_motion_evidence_sha256=(
            source_camera_motion_evidence.definition_sha256
            if source_camera_motion_evidence is not None
            else None
        ),
    )
    options: list[PresentationOption] = []
    acceptable = frozenset(acceptable_capability_ids)
    forbidden = frozenset(forbidden_capability_ids)
    required_readability = required_readability_by_target or {}
    panel_admissible = (
        panel_semantically_admissible or physical_scale_comparison
    )
    resolved_panel_groups = tuple(
        tuple(dict.fromkeys(group))
        for group in panel_target_groups
        if group
    )
    if not resolved_panel_groups and panel_admissible and len(targets) == 2:
        resolved_panel_groups = tuple(
            (target.target_id,) for target in targets
        )
    flattened_panel_ids = tuple(
        target_id
        for group in resolved_panel_groups
        for target_id in group
    )
    panel_groups_valid = (
        len(resolved_panel_groups) == 2
        and len(flattened_panel_ids) == len(set(flattened_panel_ids))
        and set(flattened_panel_ids) == set(scene_facts.target_ids)
    )
    grouped_panel_targets = (
        tuple(
            _aggregate_presentation_group(
                targets=targets,
                target_ids=group,
                group_index=index,
            )
            for index, group in enumerate(resolved_panel_groups)
        )
        if panel_groups_valid
        else ()
    )
    panel_group_index_by_target = {
        target_id: group_index
        for group_index, group in enumerate(resolved_panel_groups)
        for target_id in group
    }
    split_coupled_pairs = tuple(
        pair
        for pair in scene_facts.nested_target_pairs
        if panel_group_index_by_target.get(pair[0])
        != panel_group_index_by_target.get(pair[1])
    )
    full_bleed_readability = min(
        1.0,
        min(
            scene_facts.target_short_edge_pixels_by_mode["full_bleed"]
        )
        / 240,
    )

    def capability_boundary(capability_id: str) -> ConstraintResult:
        allowed = (
            capability_id not in forbidden
            and (not acceptable or capability_id in acceptable)
        )
        return _constraint(
            "semantic_capability_boundary",
            passed=allowed,
            reason_code=(
                "capability_semantically_acceptable"
                if allowed
                else "capability_outside_semantic_boundary"
            ),
            evidence_refs=(scene_facts.definition_sha256,),
        )

    def readability_passed(score: float) -> bool:
        return all(
            score >= float(required_readability.get(target_id, 0.0))
            for target_id in scene_facts.target_ids
        )

    def source_motion_acceptance(
        *,
        motion_dependent: bool,
    ) -> ConstraintResult:
        """Apply estimator reliability only where an operation depends on it."""

        if not scene_facts.source_camera_motion_measured:
            return (
                _constraint(
                    "source_camera_motion_quality",
                    passed=False,
                    reason_code=(
                        "source_camera_motion_not_measured_"
                        "motion_dependent_presentation_forbidden"
                    ),
                    evidence_refs=(scene_facts.definition_sha256,),
                )
                if motion_dependent
                else _preference_constraint(
                    "source_camera_motion_quality",
                    status="unknown",
                    reason_code="source_camera_motion_not_measured",
                    evidence_refs=(scene_facts.definition_sha256,),
                )
            )
        if (
            source_camera_motion_evidence is not None
            and source_camera_motion_evidence.reliable
            and source_camera_motion_evidence.isolated_jolt_count > 0
        ):
            return _constraint(
                "source_camera_motion_quality",
                passed=False,
                reason_code="unresolved_source_camera_jolt",
                evidence_refs=(
                    scene_facts.definition_sha256,
                    source_camera_motion_evidence.definition_sha256,
                ),
            )
        if not scene_facts.source_camera_motion_reliable:
            return (
                _constraint(
                    "source_camera_motion_quality",
                    passed=False,
                    reason_code=(
                        "source_camera_motion_unreliable_"
                        "motion_dependent_presentation_forbidden"
                    ),
                    evidence_refs=(scene_facts.definition_sha256,),
                )
                if motion_dependent
                else _preference_constraint(
                    "source_camera_motion_quality",
                    status="unknown",
                    reason_code="source_camera_motion_unreliable",
                    evidence_refs=(scene_facts.definition_sha256,),
                )
            )
        if (
            source_camera_motion_evidence is not None
            and source_camera_motion_evidence.contract_version
            == "source-camera-motion-evidence-v2"
            and not source_camera_motion_evidence.sampling_complete
        ):
            return (
                _constraint(
                    "source_camera_motion_quality",
                    passed=False,
                    reason_code=(
                        "source_camera_motion_sampling_incomplete_"
                        "motion_dependent_presentation_forbidden"
                    ),
                    evidence_refs=(
                        scene_facts.definition_sha256,
                        source_camera_motion_evidence.definition_sha256,
                    ),
                )
                if motion_dependent
                else _preference_constraint(
                    "source_camera_motion_quality",
                    status="unknown",
                    reason_code="source_camera_motion_sampling_incomplete",
                    evidence_refs=(
                        scene_facts.definition_sha256,
                        source_camera_motion_evidence.definition_sha256,
                    ),
                )
            )
        accepted = (
            scene_facts.source_camera_motion.direction == "static"
            or (
                resolved_source_motion_motivated
                and scene_facts.source_camera_motion.reversal_count <= 1
            )
        )
        return _constraint(
            "source_camera_motion_quality",
            passed=accepted,
            reason_code=(
                "source_camera_static"
                if scene_facts.source_camera_motion.direction == "static"
                else "source_camera_motion_semantically_motivated"
                if accepted and resolved_source_motion_motivated
                else "source_camera_motion_reversal_exceeds_policy"
                if resolved_source_motion_motivated
                else "unmotivated_source_camera_motion"
            ),
            evidence_refs=(
                scene_facts.definition_sha256,
                *(
                    (scene_facts.source_camera_motion_evidence_sha256,)
                    if scene_facts.source_camera_motion_evidence_sha256
                    else ()
                ),
            ),
        )

    if shared_crop is not None:
        options.append(
            _presentation_option(
                capability_id="static_full_bleed_crop",
                mode="static_full_bleed_crop",
                scene_facts=scene_facts,
                payload={
                    "static_crop_box_2d": shared_crop,
                },
                constraints=(
                    capability_boundary("static_full_bleed_crop"),
                    source_motion_acceptance(motion_dependent=False),
                    _constraint(
                        "shared_static_crop",
                        passed=True,
                        reason_code="required_targets_fit_static_full_bleed",
                        evidence_refs=(scene_facts.definition_sha256,),
                    ),
                    _constraint(
                        "minimum_readability",
                        passed=readability_passed(full_bleed_readability),
                        reason_code=(
                            "minimum_readability_passed"
                            if readability_passed(full_bleed_readability)
                            else "minimum_readability_failed"
                        ),
                        evidence_refs=(scene_facts.definition_sha256,),
                    ),
                ),
                semantic_fit=(
                    0.96
                    if relation_mode != "sequential_focus"
                    else 0.78
                ),
                readability=full_bleed_readability,
                technical_quality=1.0,
                intrusion_rank=capability_specs[
                    "static_full_bleed_crop"
                ].intrusion_rank,
                decision_codes=(
                    "required_targets_fit_static_full_bleed",
                    "stable_hold_selected",
                ),
                static_crop_box_2d=shared_crop,
            )
        )

    if tracking_available:
        tracked_constraint = (
            capability_boundary("tracked_full_bleed_crop"),
            source_motion_acceptance(motion_dependent=True),
            _constraint(
                "tracking_available",
                passed=True,
                reason_code="target_tracks_available",
                evidence_refs=(scene_facts.definition_sha256,),
            ),
            _constraint(
                "required_relation",
                passed=(
                    relation_mode
                    not in {"simultaneous_relation", "context_detail"}
                    or shared_crop is not None
                    or scene_facts.shared_tracked_crop_feasible is True
                ),
                reason_code=(
                    "required_relation_preserved"
                    if (
                        relation_mode
                        not in {"simultaneous_relation", "context_detail"}
                        or shared_crop is not None
                        or scene_facts.shared_tracked_crop_feasible is True
                    )
                    else "tracked_single_view_cannot_preserve_relation"
                ),
                evidence_refs=(scene_facts.definition_sha256,),
            ),
            _constraint(
                "minimum_readability",
                passed=readability_passed(full_bleed_readability),
                reason_code=(
                    "minimum_readability_passed"
                    if readability_passed(full_bleed_readability)
                    else "minimum_readability_failed"
                ),
                evidence_refs=(scene_facts.definition_sha256,),
            ),
        )
        options.append(
            _presentation_option(
                capability_id="tracked_full_bleed_crop",
                mode="tracked_full_bleed_crop",
                scene_facts=scene_facts,
                payload={"tracking_required": True},
                constraints=tracked_constraint,
                semantic_fit=(
                    0.90 if relation_mode == "single_subject" else 0.72
                ),
                readability=full_bleed_readability,
                technical_quality=0.90,
                intrusion_rank=capability_specs[
                    "tracked_full_bleed_crop"
                ].intrusion_rank,
                local_cost_rank=2,
                decision_codes=("tracked_full_bleed_candidate_generated",),
            )
        )

    motion_positions = tuple(required_x_values)
    if (
        tracking_available
        and relation_mode == "sequential_focus"
        and len(motion_positions) >= 2
    ):
        camera_motion = compile_minimal_camera_motion(
            motion_positions,
            source_camera_x_values=source_camera_x_values,
            source_camera_motion=source_motion,
            movement_motivated=movement_motivated,
        )
        motion_mode = (
            "hard_cut_between_views"
            if camera_motion.mode == "hard_cut"
            else "phase_virtual_camera"
            if camera_motion.mode == "minimal_monotonic_move"
            else None
        )
        if motion_mode is not None:
            motion_spec = capability_specs[motion_mode]
            options.append(
                _presentation_option(
                    capability_id=motion_mode,
                    mode=motion_mode,
                    scene_facts=scene_facts,
                    payload={
                        "camera_motion": camera_motion.model_dump(mode="json"),
                    },
                    constraints=(
                        capability_boundary(motion_mode),
                        source_motion_acceptance(
                            motion_dependent=(
                                motion_mode == "phase_virtual_camera"
                            )
                        ),
                        _constraint(
                            "motion_motivated",
                            passed=movement_motivated,
                            reason_code=(
                                "semantic_attention_transfer"
                                if movement_motivated
                                else "unmotivated_synthetic_motion"
                            ),
                            evidence_refs=(scene_facts.definition_sha256,),
                        ),
                        (
                            _constraint(
                                "source_motion_compatible",
                                passed=(
                                    motion_mode == "hard_cut_between_views"
                                    or
                                    camera_motion.source_motion.direction
                                    == "static"
                                ),
                                reason_code=(
                                    "hard_cut_does_not_stack_source_motion"
                                    if motion_mode == "hard_cut_between_views"
                                    else "source_camera_static"
                                    if (
                                        camera_motion.source_motion.direction
                                        == "static"
                                    )
                                    else "source_motion_would_be_counteracted"
                                ),
                                evidence_refs=(scene_facts.definition_sha256,),
                            )
                            if (
                                scene_facts.source_camera_motion_measured
                                and scene_facts.source_camera_motion_reliable
                            )
                            else _constraint(
                                "source_motion_compatible",
                                passed=(
                                    motion_mode
                                    == "hard_cut_between_views"
                                ),
                                reason_code=(
                                    "hard_cut_does_not_require_source_"
                                    "motion_measurement"
                                    if (
                                        motion_mode
                                        == "hard_cut_between_views"
                                    )
                                    else
                                    "source_camera_motion_unreliable_"
                                    "synthetic_motion_forbidden"
                                    if (
                                        scene_facts
                                        .source_camera_motion_measured
                                    )
                                    else
                                    "source_camera_motion_not_measured_"
                                    "synthetic_motion_forbidden"
                                ),
                                evidence_refs=(
                                    scene_facts.definition_sha256,
                                ),
                            )
                        ),
                    ),
                    semantic_fit=0.98,
                    readability=0.94,
                    technical_quality=0.88,
                    synthetic_motion_distance=(
                        max(camera_motion.normalized_x_values)
                        - min(camera_motion.normalized_x_values)
                    ),
                    intrusion_rank=motion_spec.intrusion_rank,
                    local_cost_rank=2,
                    decision_codes=(
                        "semantic_attention_order_preserved",
                        "minimal_camera_motion_compiled",
                    ),
                    camera_motion=camera_motion,
                )
            )

    if (
        relation_mode in {"simultaneous_relation", "context_detail"}
        and len(grouped_panel_targets) == 2
        and policy.presentation.allow_two_panel_layout
        and panel_admissible
    ):
        panel = choose_two_panel_layout(
            grouped_panel_targets[0],
            grouped_panel_targets[1],
            source_width=source_width,
            source_height=source_height,
            relation_mode=relation_mode,
            physical_scale_comparison=physical_scale_comparison,
            allow_conceptual_different_source=(
                allow_conceptual_different_source
            ),
            allowed_modes=policy.presentation.allowed_panel_modes,
        )
        if panel is not None:
            panel = panel.model_copy(
                update={
                    "panels": tuple(
                        panel_spec.model_copy(
                            update={
                                "target_ids": resolved_panel_groups[index]
                            }
                        )
                        for index, panel_spec in enumerate(panel.panels)
                    )
                }
            )
            target_by_id = {
                target.target_id: target for target in targets
            }
            panel_readability_by_target = {
                target_id: _panel_readability(
                    target_by_id[target_id].box_2d,
                    panel_spec.output_rect,
                    panel_spec.crop_box_2d,
                )
                for panel_spec in panel.panels
                for target_id in panel_spec.target_ids
            }
            panel_readability_passed = all(
                score
                >= float(required_readability.get(target_id, 0.0))
                for target_id, score in panel_readability_by_target.items()
            )
            panel_min_readability = min(
                panel_readability_by_target.values()
            )
            options.append(
                _presentation_option(
                    capability_id="two_panel_layout",
                    mode="two_panel_layout",
                    scene_facts=scene_facts,
                    payload=panel.model_dump(mode="json"),
                    constraints=(
                        capability_boundary("two_panel_layout"),
                        source_motion_acceptance(motion_dependent=False),
                        _constraint(
                            "panel_semantic_admissibility",
                            passed=panel_admissible,
                            reason_code="panel_semantically_admissible",
                            evidence_refs=(scene_facts.definition_sha256,),
                        ),
                        _constraint(
                            "coupled_target_separation",
                            passed=(
                                not split_coupled_pairs
                                or relation_mode == "context_detail"
                            ),
                            reason_code=(
                                "targets_are_independent_panel_groups"
                                if not split_coupled_pairs
                                else "context_detail_explicitly_authorizes_coupled_views"
                                if relation_mode == "context_detail"
                                else "coupled_targets_must_remain_on_one_canvas"
                            ),
                            evidence_refs=(scene_facts.definition_sha256,),
                        ),
                        _constraint(
                            "two_panel_count",
                            passed=panel_groups_valid,
                            reason_code=(
                                "exactly_two_semantic_panel_groups"
                                if panel_groups_valid
                                else "semantic_panel_groups_invalid"
                            ),
                        ),
                        _constraint(
                            "same_pts_or_conceptual",
                            passed=(
                                panel.temporal_relation
                                in {
                                    "same_source_same_pts",
                                    "different_source_conceptual",
                                }
                            ),
                            reason_code="panel_temporal_relation_bound",
                            evidence_refs=(scene_facts.definition_sha256,),
                        ),
                        _constraint(
                            "relative_scale",
                            passed=(
                                not physical_scale_comparison
                                or panel.relative_scale_policy == "locked"
                            ),
                            reason_code=(
                                "relative_scale_locked"
                                if physical_scale_comparison
                                else "physical_scale_not_claimed"
                            ),
                            evidence_refs=(scene_facts.definition_sha256,),
                        ),
                        _constraint(
                            "minimum_readability",
                            passed=panel_readability_passed,
                            reason_code=(
                                "minimum_readability_passed"
                                if panel_readability_passed
                                else "minimum_readability_failed"
                            ),
                            evidence_refs=(scene_facts.definition_sha256,),
                        ),
                    ),
                    semantic_fit=(
                        0.94 if shared_crop is None else 0.68
                    ),
                    readability=panel_min_readability,
                    technical_quality=0.88,
                    intrusion_rank=capability_specs[
                        "two_panel_layout"
                    ].intrusion_rank,
                    local_cost_rank=2,
                    decision_codes=(
                        "two_panel_geometry_passed",
                        "panel_does_not_add_model_calls",
                    ),
                    panel_layout=panel,
                )
            )

    if policy.presentation.allow_solid_matte_fit:
        options.append(
            _presentation_option(
                capability_id="solid_matte_fit",
                mode="solid_matte_fit",
                scene_facts=scene_facts,
                payload={"filter_graph": _vertical_fit_filter()},
                constraints=(
                    capability_boundary("solid_matte_fit"),
                    source_motion_acceptance(motion_dependent=False),
                    _constraint(
                        "policy_authorized",
                        passed=True,
                        reason_code="solid_matte_fit_policy_authorized",
                    ),
                    _constraint(
                        "minimum_readability",
                        passed=readability_passed(
                            scene_facts.target_readability_by_canvas["9:16"]
                        ),
                        reason_code=(
                            "minimum_readability_passed"
                            if readability_passed(
                                scene_facts.target_readability_by_canvas["9:16"]
                            )
                            else "minimum_readability_failed"
                        ),
                        evidence_refs=(scene_facts.definition_sha256,),
                    ),
                ),
                semantic_fit=0.58,
                readability=0.72,
                technical_quality=0.82,
                intrusion_rank=capability_specs[
                    "solid_matte_fit"
                ].intrusion_rank,
                decision_codes=(
                    "whole_source_scope_preserved",
                    "solid_matte_fit_policy_authorized",
                ),
                filter_graph=_vertical_fit_filter(),
            )
        )

    if not options:
        # Preserve an auditable option set even when policy rejects every
        # operation. The selector will fail closed on its unknown hard fact.
        options.append(
            _presentation_option(
                capability_id="static_full_bleed_crop",
                mode="static_full_bleed_crop",
                scene_facts=scene_facts,
                payload={"unavailable": True},
                constraints=(
                    _constraint(
                        "shared_static_crop",
                        passed=None,
                        reason_code="no_policy_authorized_presentation",
                        evidence_refs=(scene_facts.definition_sha256,),
                    ),
                ),
                semantic_fit=0.0,
                readability=0.0,
                technical_quality=0.0,
                intrusion_rank=0,
                decision_codes=("no_policy_authorized_presentation",),
            )
        )
    return PresentationOptionSet(
        scene_facts=scene_facts,
        options=tuple(options),
    )


def _build_scene_facts(
    *,
    targets: Sequence[PresentationTarget],
    source_width: int,
    source_height: int,
    shared_crop: NormalizedBox | None,
    source_motion: SourceCameraMotionEstimate,
    source_camera_motion_measured: bool,
    source_camera_motion_reliable: bool,
    source_camera_motion_evidence_sha256: str | None,
) -> SceneFacts:
    center_x = tuple(
        ((target.box_2d[0] + target.box_2d[2]) / 2) / 1000
        for target in targets
    )
    center_y = tuple(
        ((target.box_2d[1] + target.box_2d[3]) / 2) / 1000
        for target in targets
    )
    widths = tuple(
        (target.box_2d[2] - target.box_2d[0]) / 1000
        for target in targets
    )
    heights = tuple(
        (target.box_2d[3] - target.box_2d[1]) / 1000
        for target in targets
    )
    x_distances = tuple(
        tuple(abs(first - second) for second in center_x)
        for first in center_x
    )
    xy_distances = tuple(
        tuple(
            math.hypot(first_x - second_x, first_y - second_y)
            for second_x, second_y in zip(center_x, center_y, strict=True)
        )
        for first_x, first_y in zip(center_x, center_y, strict=True)
    )

    def intersection_area(first: NormalizedBox, second: NormalizedBox) -> int:
        return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0,
            min(first[3], second[3]) - max(first[1], second[1]),
        )

    areas = tuple(
        (target.box_2d[2] - target.box_2d[0])
        * (target.box_2d[3] - target.box_2d[1])
        for target in targets
    )
    intersections = tuple(
        tuple(
            intersection_area(first.box_2d, second.box_2d)
            for second in targets
        )
        for first in targets
    )
    iou = tuple(
        tuple(
            round(
                intersections[first_index][second_index]
                / max(
                    1,
                    areas[first_index]
                    + areas[second_index]
                    - intersections[first_index][second_index],
                ),
                6,
            )
            for second_index in range(len(targets))
        )
        for first_index in range(len(targets))
    )
    containment = tuple(
        tuple(
            round(
                intersections[first_index][second_index]
                / max(1, areas[second_index]),
                6,
            )
            for second_index in range(len(targets))
        )
        for first_index in range(len(targets))
    )
    nested_pairs = tuple(
        (
            targets[outer_index].target_id,
            targets[inner_index].target_id,
        )
        for outer_index in range(len(targets))
        for inner_index in range(len(targets))
        if outer_index != inner_index
        and containment[outer_index][inner_index] >= 0.9
    )
    samples_by_target = tuple(
        {
            source_pts: box
            for source_pts, box in target.track_boxes_by_pts
        }
        for target in targets
    )
    sample_pts_by_target = tuple(
        set(samples) for samples in samples_by_target
    )
    aligned_counts = tuple(
        tuple(
            len(first_pts & second_pts)
            for second_pts in sample_pts_by_target
        )
        for first_pts in sample_pts_by_target
    )
    co_visibility = tuple(
        tuple(
            round(
                len(first_pts & second_pts)
                / max(1, len(first_pts | second_pts)),
                6,
            )
            for second_pts in sample_pts_by_target
        )
        for first_pts in sample_pts_by_target
    )

    def center(box: NormalizedBox) -> tuple[float, float]:
        return (
            (box[0] + box[2]) / 2000,
            (box[1] + box[3]) / 2000,
        )

    def common_motion_residual(
        first_index: int,
        second_index: int,
    ) -> float | None:
        common_pts = sorted(
            sample_pts_by_target[first_index]
            & sample_pts_by_target[second_index]
        )
        if len(common_pts) < 2:
            return None
        residuals: list[float] = []
        for before_pts, after_pts in zip(
            common_pts[:-1],
            common_pts[1:],
            strict=True,
        ):
            first_before = center(
                samples_by_target[first_index][before_pts]
            )
            first_after = center(
                samples_by_target[first_index][after_pts]
            )
            second_before = center(
                samples_by_target[second_index][before_pts]
            )
            second_after = center(
                samples_by_target[second_index][after_pts]
            )
            residuals.append(
                math.hypot(
                    (first_after[0] - first_before[0])
                    - (second_after[0] - second_before[0]),
                    (first_after[1] - first_before[1])
                    - (second_after[1] - second_before[1]),
                )
            )
        return round(sum(residuals) / len(residuals), 6)

    motion_residuals = tuple(
        tuple(
            common_motion_residual(first_index, second_index)
            for second_index in range(len(targets))
        )
        for first_index in range(len(targets))
    )
    common_motion_pairs = tuple(
        (targets[first_index].target_id, targets[second_index].target_id)
        for first_index in range(len(targets))
        for second_index in range(first_index + 1, len(targets))
        if motion_residuals[first_index][second_index] is not None
        and motion_residuals[first_index][second_index] <= 0.03
        and co_visibility[first_index][second_index] >= 0.8
    )
    common_track_pts = (
        set.intersection(*sample_pts_by_target)
        if sample_pts_by_target
        and all(sample_pts_by_target)
        else set()
    )
    shared_tracked_crop_feasible = (
        all(
            _largest_shared_vertical_crop(
                [
                    target.model_copy(
                        update={
                            "box_2d": samples_by_target[target_index][
                                source_pts
                            ],
                            "track_boxes_by_pts": (),
                        }
                    )
                    for target_index, target in enumerate(targets)
                ],
                source_width=source_width,
                source_height=source_height,
            )
            is not None
            for source_pts in sorted(common_track_pts)
        )
        if common_track_pts
        else None
    )
    same_source_same_pts = len(
        {(target.source_asset_id, target.source_pts) for target in targets}
    ) == 1
    solid_fit_scale = min(
        1080 / source_width,
        1920 / source_height,
    )
    full_bleed_scale = max(
        1080 / source_width,
        1920 / source_height,
    )

    def target_short_edges(scale: float) -> tuple[int, ...]:
        return tuple(
            round(
                min(
                    width * source_width * scale,
                    height * source_height * scale,
                )
            )
            for width, height in zip(widths, heights, strict=True)
        )

    solid_fit_short_edges = target_short_edges(solid_fit_scale)
    full_bleed_short_edges = target_short_edges(full_bleed_scale)
    # 240 px is the executor's conservative short-edge normalization for a
    # legible atomic UI/detail target. Final semantic QA remains independent.
    portrait_readability = min(
        1.0,
        max(
            0.0,
            min(solid_fit_short_edges) / 240,
        ),
    )
    return SceneFacts(
        source_width=source_width,
        source_height=source_height,
        target_ids=tuple(target.target_id for target in targets),
        target_center_x=center_x,
        target_center_y=center_y,
        target_width_fractions=widths,
        target_height_fractions=heights,
        normalized_distance_matrix=x_distances,
        normalized_xy_distance_matrix=xy_distances,
        intersection_over_union_matrix=iou,
        containment_fraction_matrix=containment,
        nested_target_pairs=nested_pairs,
        aligned_track_sample_count_matrix=aligned_counts,
        co_visibility_fraction_matrix=co_visibility,
        common_motion_residual_matrix=motion_residuals,
        common_motion_pairs=common_motion_pairs,
        shared_static_crop_feasible=shared_crop is not None,
        shared_tracked_crop_feasible=shared_tracked_crop_feasible,
        source_camera_motion=source_motion,
        source_camera_motion_measured=source_camera_motion_measured,
        source_camera_motion_reliable=source_camera_motion_reliable,
        source_camera_motion_evidence_sha256=(
            source_camera_motion_evidence_sha256
        ),
        target_readability_by_canvas={"9:16": round(portrait_readability, 6)},
        target_short_edge_pixels_by_mode={
            "solid_matte_fit": solid_fit_short_edges,
            "full_bleed": full_bleed_short_edges,
        },
        same_source_same_pts=same_source_same_pts,
        physical_scale_evidence_available=same_source_same_pts,
    )


def _constraint(
    constraint_id: str,
    *,
    passed: bool | None,
    reason_code: str,
    evidence_refs: tuple[str, ...] = (),
) -> ConstraintResult:
    return ConstraintResult(
        constraint_id=constraint_id,
        level="hard",
        status="pass" if passed is True else "fail" if passed is False else "unknown",
        evidence_refs=evidence_refs,
        measured_value=passed,
        threshold=True,
        reason_code=reason_code,
    )


def _preference_constraint(
    constraint_id: str,
    *,
    status: Literal["pass", "fail", "unknown"],
    reason_code: str,
    evidence_refs: tuple[str, ...] = (),
) -> ConstraintResult:
    return ConstraintResult(
        constraint_id=constraint_id,
        level="preference",
        status=status,
        evidence_refs=evidence_refs,
        reason_code=reason_code,
    )


def _presentation_option(
    *,
    capability_id: str,
    mode: Literal[
        "static_full_bleed_crop",
        "tracked_full_bleed_crop",
        "phase_virtual_camera",
        "hard_cut_between_views",
        "two_panel_layout",
        "solid_matte_fit",
    ],
    scene_facts: SceneFacts,
    payload: Mapping[str, Any],
    constraints: tuple[ConstraintResult, ...],
    semantic_fit: float,
    readability: float,
    technical_quality: float,
    intrusion_rank: int,
    decision_codes: tuple[str, ...],
    local_cost_rank: int = 0,
    synthetic_motion_distance: float = 0.0,
    static_crop_box_2d: NormalizedBox | None = None,
    panel_layout: PanelLayoutSpec | None = None,
    filter_graph: str | None = None,
    camera_motion: CameraMotionDecision | None = None,
) -> PresentationOption:
    payload_sha = _canonical_payload_sha(
        {
            "capability_id": capability_id,
            "scene_facts_sha256": scene_facts.definition_sha256,
            "payload": dict(payload),
        }
    )
    option = ExecutableOptionV2(
        option_id=f"{capability_id}:{payload_sha[:16]}",
        capability_id=capability_id,
        payload_sha256=payload_sha,
        dependency_hashes=(scene_facts.definition_sha256,),
        constraints=constraints,
        metrics=OptionMetrics(
            semantic_fit=semantic_fit,
            readability=readability,
            technical_quality=technical_quality,
            music_flow=0.5,
            synthetic_motion_distance=synthetic_motion_distance,
            intrusion_rank=intrusion_rank,
            local_cost_rank=local_cost_rank,
        ),
        decision_codes=decision_codes,
    )
    return PresentationOption(
        option=option,
        mode=mode,
        static_crop_box_2d=static_crop_box_2d,
        panel_layout=panel_layout,
        filter_graph=filter_graph,
        camera_motion=camera_motion,
    )


def _canonical_payload_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compile_minimal_camera_motion(
    required_x_values: Sequence[float],
    *,
    source_camera_x_values: Sequence[float] = (),
    source_camera_motion: SourceCameraMotionEstimate | None = None,
    movement_motivated: bool,
    initial_position_optimizable: bool = True,
    deadband: float = 0.05,
) -> CameraMotionDecision:
    """Suppress drift/reversals and preserve source camera motion by default."""

    if not required_x_values:
        raise ValueError("camera motion compiler requires positions")
    if any(not 0.0 <= value <= 1.0 for value in required_x_values):
        raise ValueError("camera positions must be normalized")
    source = source_camera_motion or estimate_source_camera_motion(
        source_camera_x_values
        or tuple(0.5 for _ in required_x_values),
        threshold=deadband,
    )
    required_span = max(required_x_values) - min(required_x_values)
    hold_value = sum(required_x_values) / len(required_x_values)
    if required_span <= deadband or not movement_motivated:
        return CameraMotionDecision(
            mode="hold",
            normalized_x_values=tuple(hold_value for _ in required_x_values),
            movement_motivation="none",
            source_motion=source,
            synthetic_reversal_count=0,
            settle_required=False,
        )
    if source.direction == "unreliable":
        # When source motion cannot be measured, a hard cut is the only
        # attention-transfer operation that cannot accidentally counteract or
        # amplify photographed motion.
        return CameraMotionDecision(
            mode="hard_cut",
            normalized_x_values=tuple(required_x_values),
            movement_motivation="attention_transfer",
            source_motion=source,
            synthetic_reversal_count=0,
            settle_required=True,
        )
    if source.direction != "static":
        # Source pan already supplies motion. Without a separately proven need
        # to maintain containment, synthetic travel must not cancel or stack it.
        return CameraMotionDecision(
            mode="hold",
            normalized_x_values=tuple(hold_value for _ in required_x_values),
            movement_motivation="none",
            source_motion=source,
            synthetic_reversal_count=0,
            settle_required=False,
        )
    values = list(required_x_values)
    reversals = _signed_motion_reversal_count(
        values,
        perceptual_threshold=deadband,
    )
    if reversals and initial_position_optimizable and len(values) >= 3:
        next_value = values[1]
        if min(values[1:]) <= values[0] <= max(values[1:]):
            values[0] = next_value
            reversals = _signed_motion_reversal_count(
                values,
                perceptual_threshold=deadband,
            )
    if reversals:
        return CameraMotionDecision(
            mode="hard_cut",
            normalized_x_values=tuple(values),
            movement_motivation="attention_transfer",
            source_motion=source,
            synthetic_reversal_count=reversals,
            settle_required=True,
        )
    return CameraMotionDecision(
        mode="minimal_monotonic_move",
        normalized_x_values=tuple(values),
        movement_motivation="attention_transfer",
        source_motion=source,
        synthetic_reversal_count=0,
        settle_required=True,
    )


def estimate_source_camera_motion(
    x_values: Sequence[float],
    *,
    threshold: float = 0.05,
) -> SourceCameraMotionEstimate:
    if not x_values:
        return SourceCameraMotionEstimate(
            direction="static",
            normalized_travel=0.0,
            reversal_count=0,
        )
    travel = sum(
        abs(after - before)
        for before, after in zip(x_values[:-1], x_values[1:], strict=True)
    )
    reversals = _signed_motion_reversal_count(
        x_values,
        perceptual_threshold=threshold,
    )
    net = x_values[-1] - x_values[0]
    direction: Literal["static", "left", "right", "mixed"]
    if travel < threshold:
        direction = "static"
    elif reversals:
        direction = "mixed"
    else:
        direction = "right" if net > 0 else "left"
    return SourceCameraMotionEstimate(
        direction=direction,
        normalized_travel=round(travel, 6),
        reversal_count=reversals,
    )


def measure_source_camera_motion(
    *,
    source_path: Path,
    source_asset_id: str,
    window_start_ms: int,
    window_end_ms: int,
    subject_tracks: Sequence[SegmentationTrack],
    output_dir: Path,
    sample_count: int = 8,
    max_width: int = 640,
) -> SourceCameraMotionEvidence:
    """Measure source-camera motion from background features, without Gemini.

    Frames are decoded through FFmpeg's orientation-corrected path. Foreground
    regions come from the nearest SAM track boxes and are excluded before
    forward/backward optical-flow matching and RANSAC affine fitting.
    """

    if window_start_ms < 0 or window_end_ms <= window_start_ms:
        raise ValueError("source camera motion window must be non-empty")
    if sample_count < 3 or sample_count > 16:
        raise ValueError("source camera sample_count must be in [3, 16]")
    if max_width < 320 or max_width > 1_280:
        raise ValueError("source camera max_width must be in [320, 1280]")
    resolved_source = source_path.expanduser().resolve()
    track_facts = _source_motion_track_facts(subject_tracks)
    cache_key = _canonical_payload_sha(
        {
            "estimator_version": _SOURCE_MOTION_ESTIMATOR_V3,
            "decoder_version": _SOURCE_MOTION_DECODER_V1,
            "sampling_version": _SOURCE_MOTION_SAMPLING_V2,
            "requested_max_sample_gap_ms": (
                _SOURCE_MOTION_MAX_SAMPLE_GAP_MS
            ),
            "edge_dense_window_ms": _SOURCE_MOTION_EDGE_DENSE_WINDOW_MS,
            "edge_sample_step_ms": _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS,
            "source_asset_id": source_asset_id,
            "source_size": (
                resolved_source.stat().st_size
                if resolved_source.exists()
                else None
            ),
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "sample_count": sample_count,
            "max_width": max_width,
            "subject_tracks": track_facts,
        }
    )
    artifact_dir = output_dir / f"source-camera-motion-{cache_key[:16]}"
    artifact_path = artifact_dir / "evidence.json"
    if artifact_path.exists():
        cached = SourceCameraMotionEvidence.model_validate(
            read_json(artifact_path)
        )
        if cached.cache_key_sha256 == cache_key:
            return cached
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not resolved_source.is_file():
        evidence = _unreliable_source_motion_evidence(
            source_asset_id=source_asset_id,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            cache_key=cache_key,
            reason_codes=(
                "source_motion_media_unavailable",
                "synthetic_motion_fail_closed",
            ),
            subject_exclusion_mode=(
                "sam_track_boxes" if subject_tracks else "none"
            ),
        )
        write_json(artifact_path, evidence.model_dump(mode="json"))
        return evidence

    try:
        import cv2
        import numpy as np
    except ImportError:
        evidence = _unreliable_source_motion_evidence(
            source_asset_id=source_asset_id,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            cache_key=cache_key,
            reason_codes=("opencv_tracking_dependency_unavailable",),
        )
        write_json(artifact_path, evidence.model_dump(mode="json"))
        return evidence

    requested_times = _source_motion_sample_times(
        window_start_ms,
        window_end_ms,
        sample_count=sample_count,
    )
    extracted_frames = _reuse_tracking_analysis_frames(
        subject_tracks,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        sample_count=len(requested_times),
    )
    try:
        reusable_times = {
            frame.frame_time_ms
            for frame in extracted_frames
        }
        missing_requested_times = []
        for requested_ms in requested_times:
            if any(
                abs(requested_ms - reusable_ms)
                <= _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS // 2
                for reusable_ms in reusable_times
            ):
                continue
            missing_requested_times.append(requested_ms)
        if missing_requested_times:
            decoded = extract_frames_bounded(
                resolved_source,
                missing_requested_times,
                artifact_dir / "frames",
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                max_width=max_width,
            )
            # Preserve the legacy sequential alias collapse exactly. A prior
            # request can resolve to a decoded frame close enough to satisfy a
            # later semantic request even though their requested times were
            # farther apart. Batch decoding must not turn that later request
            # into an additional motion pair.
            for extracted in decoded:
                if any(
                    abs(extracted.requested_time_ms - reusable_ms)
                    <= _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS // 2
                    for reusable_ms in reusable_times
                ):
                    continue
                extracted_frames.append(extracted)
                reusable_times.add(extracted.frame_time_ms)
    except Exception as error:
        evidence = _unreliable_source_motion_evidence(
            source_asset_id=source_asset_id,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            cache_key=cache_key,
            reason_codes=(
                "source_motion_frame_decode_failed",
                f"{type(error).__name__}",
            ),
            extracted_frames=extracted_frames,
        )
        write_json(artifact_path, evidence.model_dump(mode="json"))
        return evidence

    # Reused SAM frames and newly decoded samples share the same immutable
    # source timeline.  Sort and collapse seek aliases before measuring pairs.
    extracted_frames = sorted(
        extracted_frames,
        key=lambda frame: (frame.frame_time_ms, frame.frame_pts),
    )
    distinct_frames: list[ExtractedFrame] = []
    seen_pts: set[int] = set()
    for extracted in extracted_frames:
        if extracted.frame_pts in seen_pts:
            continue
        distinct_frames.append(extracted)
        seen_pts.add(extracted.frame_pts)
    extracted_frames = distinct_frames

    if len(extracted_frames) < 2:
        evidence = _unreliable_source_motion_evidence(
            source_asset_id=source_asset_id,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            cache_key=cache_key,
            reason_codes=("insufficient_distinct_decoded_frames",),
            extracted_frames=extracted_frames,
        )
        write_json(artifact_path, evidence.model_dump(mode="json"))
        return evidence

    decoded_frames = []
    for extracted in extracted_frames:
        frame = cv2.imread(extracted.path)
        if frame is None:
            evidence = _unreliable_source_motion_evidence(
                source_asset_id=source_asset_id,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                cache_key=cache_key,
                reason_codes=("source_motion_frame_read_failed",),
                extracted_frames=extracted_frames,
            )
            write_json(artifact_path, evidence.model_dump(mode="json"))
            return evidence
        decoded_frames.append(frame)
    common_width = min(frame.shape[1] for frame in decoded_frames)
    common_height = min(frame.shape[0] for frame in decoded_frames)

    gray_frames = []
    background_masks = []
    excluded_fractions = []
    for extracted, frame in zip(
        extracted_frames,
        decoded_frames,
        strict=True,
    ):
        if (
            frame.shape[1] != common_width
            or frame.shape[0] != common_height
        ):
            frame = cv2.resize(
                frame,
                (common_width, common_height),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_mask, excluded_fraction = _background_feature_mask(
            width=gray.shape[1],
            height=gray.shape[0],
            time_ms=extracted.frame_time_ms,
            subject_tracks=subject_tracks,
            cv2=cv2,
            np=np,
        )
        gray_frames.append(gray)
        background_masks.append(background_mask)
        excluded_fractions.append(excluded_fraction)

    raw_pairs = tuple(
        _measure_background_motion_pair(
            before_gray=gray_frames[index],
            after_gray=gray_frames[index + 1],
            before_mask=background_masks[index],
            after_mask=background_masks[index + 1],
            before_frame_pts=extracted_frames[index].frame_pts,
            after_frame_pts=extracted_frames[index + 1].frame_pts,
            delta_ms=max(
                1,
                extracted_frames[index + 1].frame_time_ms
                - extracted_frames[index].frame_time_ms,
            ),
            cv2=cv2,
            np=np,
        )
        for index in range(len(extracted_frames) - 1)
    )
    pairs = _annotate_source_motion_pairs(
        raw_pairs,
        extracted_frames=extracted_frames,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )
    reliable_pairs = tuple(pair for pair in pairs if pair.reliable)
    required_pair_count = max(2, math.ceil(len(pairs) * 0.5))
    if len(reliable_pairs) < required_pair_count:
        evidence = _unreliable_source_motion_evidence(
            source_asset_id=source_asset_id,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            cache_key=cache_key,
            reason_codes=(
                "insufficient_reliable_background_pairs",
                "synthetic_motion_fail_closed",
            ),
            extracted_frames=extracted_frames,
            pairs=pairs,
            mean_excluded_area_fraction=(
                sum(excluded_fractions) / len(excluded_fractions)
            ),
            subject_exclusion_mode=(
                "sam_track_boxes" if subject_tracks else "none"
            ),
        )
        write_json(artifact_path, evidence.model_dump(mode="json"))
        return evidence

    x_rates = [
        pair.camera_translation_x_normalized * 1000 / pair.delta_ms
        for pair in reliable_pairs
    ]
    y_rates = [
        pair.camera_translation_y_normalized * 1000 / pair.delta_ms
        for pair in reliable_pairs
    ]
    scale_rates = [
        pair.camera_scale_delta * 1000 / pair.delta_ms
        for pair in reliable_pairs
    ]
    rotation_rates = [
        pair.camera_rotation_degrees * 1000 / pair.delta_ms
        for pair in reliable_pairs
    ]
    x_rate = float(median(x_rates))
    y_rate = float(median(y_rates))
    scale_rate = float(median(scale_rates))
    rotation_rate = float(median(rotation_rates))
    translation_speeds = [
        pair.normalized_translation_speed_per_second
        for pair in reliable_pairs
    ]
    p95_translation_speed = _percentile(translation_speeds, 0.95)
    max_translation_speed = max(translation_speeds, default=0.0)
    (
        max_translation_acceleration,
        max_translation_jerk,
    ) = _source_motion_temporal_extrema(reliable_pairs)
    jolt_pair_indexes = tuple(
        index
        for index, pair in enumerate(pairs)
        if pair.isolated_jolt
    )
    dirty_head = any(
        pairs[index].dirty_edge == "head"
        for index in jolt_pair_indexes
    )
    dirty_tail = any(
        pairs[index].dirty_edge == "tail"
        for index in jolt_pair_indexes
    )
    clean_head_start_ms = window_start_ms
    if dirty_head:
        clean_head_start_ms = min(
            window_end_ms,
            max(
                pair.after_time_ms or window_start_ms
                for pair in pairs
                if pair.dirty_edge == "head"
            )
            + _SOURCE_MOTION_EDGE_SETTLE_PADDING_MS,
        )
    clean_tail_end_ms = window_end_ms
    if dirty_tail:
        clean_tail_end_ms = max(
            window_start_ms,
            min(
                pair.before_time_ms or window_end_ms
                for pair in pairs
                if pair.dirty_edge == "tail"
            )
            - _SOURCE_MOTION_EDGE_SETTLE_PADDING_MS,
        )
    if clean_head_start_ms > clean_tail_end_ms:
        clean_head_start_ms = clean_tail_end_ms
    normalized_travel = sum(
        math.hypot(
            pair.camera_translation_x_normalized,
            pair.camera_translation_y_normalized,
        )
        for pair in reliable_pairs
    )
    dominant_deltas = (
        [
            pair.camera_translation_x_normalized
            for pair in reliable_pairs
        ]
        if abs(x_rate) >= abs(y_rate)
        else [
            pair.camera_translation_y_normalized
            for pair in reliable_pairs
        ]
    )
    cumulative_positions = [0.5]
    for delta in dominant_deltas:
        cumulative_positions.append(cumulative_positions[-1] + delta)
    reversals = _signed_motion_reversal_count(
        cumulative_positions,
        perceptual_threshold=0.006,
    )
    classification = _classify_measured_source_motion(
        x_rate=x_rate,
        y_rate=y_rate,
        scale_rate=scale_rate,
        rotation_rate=rotation_rate,
        reversal_count=reversals,
    )
    median_inlier_ratio = float(
        median(pair.inlier_ratio for pair in reliable_pairs)
    )
    median_residual = float(
        median(pair.median_residual_pixels for pair in reliable_pairs)
    )
    valid_fraction = len(reliable_pairs) / len(pairs)
    confidence = max(
        0.0,
        min(
            1.0,
            valid_fraction
            * median_inlier_ratio
            * max(0.25, 1.0 - median_residual / 5.0),
        ),
    )
    actual_max_sample_gap_ms = _maximum_sample_gap_ms(extracted_frames)
    head_sample_coverage_ms = max(
        0,
        extracted_frames[0].frame_time_ms - window_start_ms,
    )
    tail_sample_coverage_ms = max(
        0,
        window_end_ms - extracted_frames[-1].frame_time_ms,
    )
    sampling_complete = (
        actual_max_sample_gap_ms
        <= _SOURCE_MOTION_MAX_SAMPLE_GAP_MS + 100
        and head_sample_coverage_ms
        <= _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS + 100
        and tail_sample_coverage_ms
        <= _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS + 100
    )
    evidence = SourceCameraMotionEvidence(
        contract_version="source-camera-motion-evidence-v2",
        estimator_version=_SOURCE_MOTION_ESTIMATOR_V3,
        sampling_version=_SOURCE_MOTION_SAMPLING_V2,
        requested_max_sample_gap_ms=_SOURCE_MOTION_MAX_SAMPLE_GAP_MS,
        actual_max_sample_gap_ms=actual_max_sample_gap_ms,
        head_sample_coverage_ms=head_sample_coverage_ms,
        tail_sample_coverage_ms=tail_sample_coverage_ms,
        sampling_complete=sampling_complete,
        source_asset_id=source_asset_id,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        sample_times_ms=tuple(
            frame.frame_time_ms for frame in extracted_frames
        ),
        sample_frame_pts=tuple(
            frame.frame_pts for frame in extracted_frames
        ),
        sample_frame_hashes=tuple(
            frame.frame_hash for frame in extracted_frames
        ),
        subject_exclusion_mode=(
            "sam_track_boxes" if subject_tracks else "none"
        ),
        mean_excluded_area_fraction=round(
            sum(excluded_fractions) / len(excluded_fractions),
            6,
        ),
        pairs=pairs,
        classification=classification,
        reliable=True,
        confidence=round(confidence, 6),
        normalized_translation_x_per_second=round(x_rate, 6),
        normalized_translation_y_per_second=round(y_rate, 6),
        scale_rate_per_second=round(scale_rate, 6),
        rotation_degrees_per_second=round(rotation_rate, 6),
        normalized_travel=round(normalized_travel, 6),
        reversal_count=reversals,
        p95_translation_speed_per_second=round(
            p95_translation_speed,
            6,
        ),
        max_translation_speed_per_second=round(
            max_translation_speed,
            6,
        ),
        max_translation_acceleration_per_second_squared=round(
            max_translation_acceleration,
            6,
        ),
        max_translation_jerk_per_second_cubed=round(
            max_translation_jerk,
            6,
        ),
        isolated_jolt_count=len(jolt_pair_indexes),
        jolt_pair_indexes=jolt_pair_indexes,
        dirty_head=dirty_head,
        dirty_tail=dirty_tail,
        clean_head_start_ms=clean_head_start_ms,
        clean_tail_end_ms=clean_tail_end_ms,
        reason_codes=(
            "background_features_exclude_tracked_subjects",
            "forward_backward_flow_validated",
            "ransac_affine_motion_measured",
            "hybrid_edge_dense_sampling",
            *(
                ()
                if sampling_complete
                else ("source_motion_sampling_incomplete",)
            ),
            *(
                ("isolated_source_camera_jolt_detected",)
                if jolt_pair_indexes
                else ()
            ),
        ),
        cache_key_sha256=cache_key,
    )
    write_json(artifact_path, evidence.model_dump(mode="json"))
    return evidence


def _source_motion_sample_times(
    start_ms: int,
    end_ms: int,
    *,
    sample_count: int,
) -> tuple[int, ...]:
    duration_ms = end_ms - start_ms
    # Keep the final request inside the half-open trim even for low-FPS media;
    # asking at end-1ms can legitimately have no following decoded frame.
    end_guard_ms = min(100, max(1, duration_ms // 4))
    last_ms = max(start_ms, end_ms - end_guard_ms)
    if last_ms == start_ms:
        return (start_ms, start_ms + 1)
    requested = {
        round(start_ms + (last_ms - start_ms) * index / (sample_count - 1))
        for index in range(sample_count)
    }
    requested.update(
        range(
            start_ms,
            last_ms + 1,
            _SOURCE_MOTION_MAX_SAMPLE_GAP_MS,
        )
    )
    requested.add(last_ms)
    leading_end = min(
        last_ms,
        start_ms + _SOURCE_MOTION_EDGE_DENSE_WINDOW_MS,
    )
    requested.update(
        range(
            start_ms,
            leading_end + 1,
            _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS,
        )
    )
    trailing_start = max(
        start_ms,
        last_ms - _SOURCE_MOTION_EDGE_DENSE_WINDOW_MS,
    )
    requested.update(
        range(
            trailing_start,
            last_ms + 1,
            _SOURCE_MOTION_EDGE_SAMPLE_STEP_MS,
        )
    )
    return tuple(sorted(requested))


def _maximum_sample_gap_ms(
    frames: Sequence[ExtractedFrame],
) -> int:
    return max(
        (
            after.frame_time_ms - before.frame_time_ms
            for before, after in zip(frames, frames[1:])
        ),
        default=0,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _annotate_source_motion_pairs(
    pairs: Sequence[SourceCameraMotionPairEvidence],
    *,
    extracted_frames: Sequence[ExtractedFrame],
    window_start_ms: int,
    window_end_ms: int,
) -> tuple[SourceCameraMotionPairEvidence, ...]:
    speeds = [
        (
            math.hypot(
                pair.camera_translation_x_normalized,
                pair.camera_translation_y_normalized,
            )
            * 1000
            / pair.delta_ms
            if pair.reliable
            else 0.0
        )
        for pair in pairs
    ]
    reliable_speeds = [
        speed
        for speed, pair in zip(speeds, pairs, strict=True)
        if pair.reliable
    ]
    central_speed = float(median(reliable_speeds)) if reliable_speeds else 0.0
    speed_mad = (
        float(
            median(
                abs(speed - central_speed)
                for speed in reliable_speeds
            )
        )
        if reliable_speeds
        else 0.0
    )
    # The central statistic still describes a steady photographed move, while
    # the independent peak threshold preserves a short move-and-return jolt.
    jolt_threshold = max(
        0.08,
        central_speed * 3.0,
        central_speed + speed_mad * 6.0 + 0.02,
    )
    annotated: list[SourceCameraMotionPairEvidence] = []
    for index, (pair, speed) in enumerate(
        zip(pairs, speeds, strict=True)
    ):
        before_time_ms = extracted_frames[index].frame_time_ms
        after_time_ms = extracted_frames[index + 1].frame_time_ms
        prior_speed = speeds[index - 1] if index > 0 else central_speed
        next_speed = (
            speeds[index + 1]
            if index + 1 < len(speeds)
            else central_speed
        )
        isolated = bool(
            pair.reliable
            and speed >= jolt_threshold
            and (
                prior_speed < speed * 0.55
                or next_speed < speed * 0.55
                or _pair_direction_reverses(
                    pair,
                    pairs[index + 1]
                    if index + 1 < len(pairs)
                    else None,
                )
            )
        )
        dirty_edge: Literal["head", "tail"] | None = None
        if isolated and (
            before_time_ms
            < window_start_ms + _SOURCE_MOTION_EDGE_DENSE_WINDOW_MS
        ):
            dirty_edge = "head"
        elif isolated and (
            after_time_ms
            > window_end_ms - _SOURCE_MOTION_EDGE_DENSE_WINDOW_MS
        ):
            dirty_edge = "tail"
        annotated.append(
            pair.model_copy(
                update={
                    "before_time_ms": before_time_ms,
                    "after_time_ms": after_time_ms,
                    "normalized_translation_speed_per_second": round(
                        speed,
                        8,
                    ),
                    "isolated_jolt": isolated,
                    "dirty_edge": dirty_edge,
                }
            )
        )
    return tuple(annotated)


def _pair_direction_reverses(
    first: SourceCameraMotionPairEvidence,
    second: SourceCameraMotionPairEvidence | None,
) -> bool:
    if second is None or not second.reliable:
        return False
    first_x = first.camera_translation_x_normalized
    first_y = first.camera_translation_y_normalized
    second_x = second.camera_translation_x_normalized
    second_y = second.camera_translation_y_normalized
    first_magnitude = math.hypot(first_x, first_y)
    second_magnitude = math.hypot(second_x, second_y)
    if first_magnitude < 0.002 or second_magnitude < 0.002:
        return False
    return first_x * second_x + first_y * second_y < 0


def _source_motion_temporal_extrema(
    pairs: Sequence[SourceCameraMotionPairEvidence],
) -> tuple[float, float]:
    if len(pairs) < 2:
        return 0.0, 0.0
    accelerations: list[tuple[float, float]] = []
    for before, after in zip(pairs, pairs[1:]):
        dt_seconds = max(
            0.001,
            (before.delta_ms + after.delta_ms) / 2000,
        )
        acceleration = abs(
            after.normalized_translation_speed_per_second
            - before.normalized_translation_speed_per_second
        ) / dt_seconds
        accelerations.append((acceleration, dt_seconds))
    jerks = [
        abs(after[0] - before[0])
        / max(0.001, (before[1] + after[1]) / 2)
        for before, after in zip(accelerations, accelerations[1:])
    ]
    return (
        max((item[0] for item in accelerations), default=0.0),
        max(jerks, default=0.0),
    )


def _reuse_tracking_analysis_frames(
    tracks: Sequence[SegmentationTrack],
    *,
    window_start_ms: int,
    window_end_ms: int,
    sample_count: int,
) -> list[ExtractedFrame]:
    """Reuse the SAM decoder frontier instead of decoding the trim again."""

    for track in tracks:
        seed_source = Path(track.seed_source).expanduser()
        roots = (
            seed_source.parent,
            seed_source.parent.parent,
            seed_source.parent.parent.parent,
        )
        frames_dir = next(
            (
                root / "analysis-frames"
                for root in roots
                if (root / "analysis-frames").is_dir()
            ),
            None,
        )
        if frames_dir is None:
            continue
        available: list[ExtractedFrame] = []
        for sample in track.samples:
            if (
                sample.source_pts is None
                or not (
                    window_start_ms
                    <= sample.analysis_sample_time_ms
                    < window_end_ms
                )
            ):
                continue
            frame_path = frames_dir / f"{sample.sample_index:06d}.jpg"
            if not frame_path.is_file():
                continue
            with Image.open(frame_path) as image:
                width, height = image.size
            available.append(
                ExtractedFrame(
                    path=str(frame_path.resolve()),
                    requested_time_ms=sample.analysis_sample_time_ms,
                    frame_time_ms=sample.analysis_sample_time_ms,
                    frame_pts=sample.source_pts,
                    frame_hash=_sha256_path(frame_path),
                    width=width,
                    height=height,
                )
            )
        if len(available) < 2:
            continue
        if len(available) <= sample_count:
            return available
        selected_indexes = sorted(
            {
                round(
                    index
                    * (len(available) - 1)
                    / (sample_count - 1)
                )
                for index in range(sample_count)
            }
        )
        return [available[index] for index in selected_indexes]
    return []


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_motion_track_facts(
    tracks: Sequence[SegmentationTrack],
) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": track.asset_id,
            "target_id": track.target_id,
            "analysis_frames_manifest_sha256": (
                track.analysis_frames_manifest_sha256
            ),
            "analysis_start_ms": track.analysis_start_ms,
            "analysis_end_ms": track.analysis_end_ms,
            "samples": [
                {
                    "time_ms": sample.analysis_sample_time_ms,
                    "box": sample.derived_tracking_box,
                    "mask_sha256": sample.mask_sha256,
                    "tracking_state": str(sample.tracking_state),
                }
                for sample in track.samples
            ],
        }
        for track in tracks
    ]


def _nearest_track_box(
    track: SegmentationTrack,
    time_ms: int,
) -> tuple[int, int, int, int] | None:
    candidates = [
        sample
        for sample in track.samples
        if sample.derived_tracking_box is not None
    ]
    if not candidates:
        return None
    sample = min(
        candidates,
        key=lambda item: abs(item.analysis_sample_time_ms - time_ms),
    )
    max_gap_ms = max(750, round(2_500 / track.analysis_fps))
    if abs(sample.analysis_sample_time_ms - time_ms) > max_gap_ms:
        return None
    assert sample.derived_tracking_box is not None
    return tuple(int(value) for value in sample.derived_tracking_box)


def _background_feature_mask(
    *,
    width: int,
    height: int,
    time_ms: int,
    subject_tracks: Sequence[SegmentationTrack],
    cv2: Any,
    np: Any,
) -> tuple[Any, float]:
    mask = np.full((height, width), 255, dtype=np.uint8)
    for track in subject_tracks:
        box = _nearest_track_box(track, time_ms)
        if box is None:
            continue
        x_min, y_min, x_max, y_max = box
        padding_x = round((x_max - x_min) * 0.12)
        padding_y = round((y_max - y_min) * 0.12)
        left = max(0, math.floor((x_min - padding_x) * width / 1000))
        top = max(0, math.floor((y_min - padding_y) * height / 1000))
        right = min(
            width,
            math.ceil((x_max + padding_x) * width / 1000),
        )
        bottom = min(
            height,
            math.ceil((y_max + padding_y) * height / 1000),
        )
        cv2.rectangle(mask, (left, top), (right, bottom), 0, thickness=-1)
    excluded_fraction = 1.0 - float(cv2.countNonZero(mask)) / mask.size
    return mask, max(0.0, min(1.0, excluded_fraction))


def _measure_background_motion_pair(
    *,
    before_gray: Any,
    after_gray: Any,
    before_mask: Any,
    after_mask: Any,
    before_frame_pts: int,
    after_frame_pts: int,
    delta_ms: int,
    cv2: Any,
    np: Any,
) -> SourceCameraMotionPairEvidence:
    features = cv2.goodFeaturesToTrack(
        before_gray,
        mask=before_mask,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    detected_count = 0 if features is None else len(features)
    if features is None or detected_count < 16:
        return _unreliable_motion_pair(
            before_frame_pts,
            after_frame_pts,
            delta_ms,
            detected_count,
        )
    after_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        before_gray,
        after_gray,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if after_points is None or forward_status is None:
        return _unreliable_motion_pair(
            before_frame_pts,
            after_frame_pts,
            delta_ms,
            detected_count,
        )
    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        after_gray,
        before_gray,
        after_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if back_points is None or backward_status is None:
        return _unreliable_motion_pair(
            before_frame_pts,
            after_frame_pts,
            delta_ms,
            detected_count,
        )
    before_points = features.reshape(-1, 2)
    after_points_2d = after_points.reshape(-1, 2)
    back_points_2d = back_points.reshape(-1, 2)
    forward_ok = forward_status.reshape(-1).astype(bool)
    backward_ok = backward_status.reshape(-1).astype(bool)
    round_trip_error = np.linalg.norm(
        before_points - back_points_2d,
        axis=1,
    )
    after_x = np.clip(
        np.rint(after_points_2d[:, 0]).astype(int),
        0,
        after_gray.shape[1] - 1,
    )
    after_y = np.clip(
        np.rint(after_points_2d[:, 1]).astype(int),
        0,
        after_gray.shape[0] - 1,
    )
    stays_on_background = after_mask[after_y, after_x] > 0
    valid = (
        forward_ok
        & backward_ok
        & (round_trip_error <= 1.5)
        & stays_on_background
    )
    before_valid = before_points[valid]
    after_valid = after_points_2d[valid]
    tracked_count = len(before_valid)
    if tracked_count < 12:
        return _unreliable_motion_pair(
            before_frame_pts,
            after_frame_pts,
            delta_ms,
            detected_count,
            tracked_count,
        )
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        before_valid,
        after_valid,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2_000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        return _unreliable_motion_pair(
            before_frame_pts,
            after_frame_pts,
            delta_ms,
            detected_count,
            tracked_count,
        )
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_ratio = float(inliers.mean())
    predicted = cv2.transform(
        before_valid.reshape(-1, 1, 2),
        affine,
    ).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - after_valid, axis=1)
    median_residual = float(np.median(residuals[inliers]))
    a = float(affine[0, 0])
    b = float(affine[0, 1])
    scale = math.hypot(a, b)
    image_dx = float(affine[0, 2])
    image_dy = float(affine[1, 2])
    reliable = (
        inlier_ratio >= 0.45
        and median_residual <= 3.5
        and 0.85 <= scale <= 1.15
    )
    return SourceCameraMotionPairEvidence(
        before_frame_pts=before_frame_pts,
        after_frame_pts=after_frame_pts,
        delta_ms=delta_ms,
        detected_features=detected_count,
        tracked_background_features=tracked_count,
        inlier_ratio=round(inlier_ratio, 6),
        median_residual_pixels=round(median_residual, 6),
        # Background displacement is inverse source-camera movement.
        camera_translation_x_normalized=round(
            -image_dx / before_gray.shape[1],
            8,
        ),
        camera_translation_y_normalized=round(
            -image_dy / before_gray.shape[0],
            8,
        ),
        camera_scale_delta=round(scale - 1.0, 8),
        camera_rotation_degrees=round(
            -math.degrees(math.atan2(b, a)),
            6,
        ),
        reliable=reliable,
    )


def _unreliable_motion_pair(
    before_frame_pts: int,
    after_frame_pts: int,
    delta_ms: int,
    detected_features: int,
    tracked_features: int = 0,
) -> SourceCameraMotionPairEvidence:
    return SourceCameraMotionPairEvidence(
        before_frame_pts=before_frame_pts,
        after_frame_pts=after_frame_pts,
        delta_ms=delta_ms,
        detected_features=detected_features,
        tracked_background_features=tracked_features,
        inlier_ratio=0.0,
        median_residual_pixels=0.0,
        camera_translation_x_normalized=0.0,
        camera_translation_y_normalized=0.0,
        camera_scale_delta=0.0,
        camera_rotation_degrees=0.0,
        reliable=False,
    )


def _classify_measured_source_motion(
    *,
    x_rate: float,
    y_rate: float,
    scale_rate: float,
    rotation_rate: float,
    reversal_count: int,
) -> Literal[
    "static",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "zoom_in",
    "zoom_out",
    "mixed",
]:
    x_strength = abs(x_rate) / 0.010
    y_strength = abs(y_rate) / 0.010
    scale_strength = abs(scale_rate) / 0.010
    rotation_strength = abs(rotation_rate) / 0.6
    strengths = {
        "x": x_strength,
        "y": y_strength,
        "scale": scale_strength,
        "rotation": rotation_strength,
    }
    dominant_axis, dominant_strength = max(
        strengths.items(),
        key=lambda item: item[1],
    )
    if dominant_strength < 1.0:
        return "static"
    ordered_strengths = sorted(strengths.values(), reverse=True)
    if (
        reversal_count > 0
        or dominant_axis == "rotation"
        or (
            len(ordered_strengths) > 1
            and ordered_strengths[1] >= dominant_strength * 0.67
        )
    ):
        return "mixed"
    if dominant_axis == "x":
        return "pan_right" if x_rate > 0 else "pan_left"
    if dominant_axis == "y":
        return "tilt_down" if y_rate > 0 else "tilt_up"
    return "zoom_in" if scale_rate > 0 else "zoom_out"


def _unreliable_source_motion_evidence(
    *,
    source_asset_id: str,
    window_start_ms: int,
    window_end_ms: int,
    cache_key: str,
    reason_codes: tuple[str, ...],
    extracted_frames: Sequence[Any] = (),
    pairs: Sequence[SourceCameraMotionPairEvidence] = (),
    mean_excluded_area_fraction: float = 0.0,
    subject_exclusion_mode: Literal[
        "sam_track_boxes",
        "none",
    ] = "none",
) -> SourceCameraMotionEvidence:
    ordered_frames = tuple(
        sorted(
            extracted_frames,
            key=lambda frame: (frame.frame_time_ms, frame.frame_pts),
        )
    )
    sample_times_ms = tuple(
        frame.frame_time_ms for frame in ordered_frames
    )
    return SourceCameraMotionEvidence(
        contract_version="source-camera-motion-evidence-v2",
        estimator_version=_SOURCE_MOTION_ESTIMATOR_V3,
        sampling_version=_SOURCE_MOTION_SAMPLING_V2,
        requested_max_sample_gap_ms=_SOURCE_MOTION_MAX_SAMPLE_GAP_MS,
        actual_max_sample_gap_ms=(
            max(
                (
                    after - before
                    for before, after in zip(
                        sample_times_ms,
                        sample_times_ms[1:],
                    )
                ),
                default=0,
            )
            if sample_times_ms
            else None
        ),
        head_sample_coverage_ms=(
            max(0, sample_times_ms[0] - window_start_ms)
            if sample_times_ms
            else None
        ),
        tail_sample_coverage_ms=(
            max(0, window_end_ms - sample_times_ms[-1])
            if sample_times_ms
            else None
        ),
        source_asset_id=source_asset_id,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        sample_times_ms=sample_times_ms,
        sample_frame_pts=tuple(frame.frame_pts for frame in ordered_frames),
        sample_frame_hashes=tuple(
            frame.frame_hash for frame in ordered_frames
        ),
        subject_exclusion_mode=subject_exclusion_mode,
        mean_excluded_area_fraction=round(
            mean_excluded_area_fraction,
            6,
        ),
        pairs=tuple(pairs),
        classification="unreliable",
        reliable=False,
        confidence=0.0,
        normalized_translation_x_per_second=0.0,
        normalized_translation_y_per_second=0.0,
        scale_rate_per_second=0.0,
        rotation_degrees_per_second=0.0,
        normalized_travel=0.0,
        reversal_count=0,
        reason_codes=reason_codes,
        cache_key_sha256=cache_key,
    )


def choose_two_panel_layout(
    first: PresentationTarget,
    second: PresentationTarget,
    *,
    source_width: int,
    source_height: int,
    relation_mode: Literal["simultaneous_relation", "context_detail"],
    physical_scale_comparison: bool,
    allow_conceptual_different_source: bool,
    allowed_modes: Sequence[
        Literal["top_bottom", "side_by_side", "context_detail"]
    ],
) -> PanelLayoutSpec | None:
    same_source = first.source_asset_id == second.source_asset_id
    if same_source and first.source_pts != second.source_pts:
        return None
    if not same_source:
        if physical_scale_comparison or not allow_conceptual_different_source:
            return None
    candidates: list[PanelLayoutSpec] = []
    for mode in allowed_modes:
        if mode == "context_detail" and relation_mode != "context_detail":
            continue
        rects = _panel_rects(mode)
        crop_boxes = [
            _panel_crop_around(
                target.box_2d,
                rect=rect,
                source_width=source_width,
                source_height=source_height,
            )
            for target, rect in zip((first, second), rects, strict=True)
        ]
        if any(crop is None for crop in crop_boxes):
            continue
        first_crop, second_crop = crop_boxes
        assert first_crop is not None and second_crop is not None
        if physical_scale_comparison:
            if rects[0].width != rects[1].width or (
                rects[0].height != rects[1].height
            ):
                continue
            shared_height = max(
                first_crop[3] - first_crop[1],
                second_crop[3] - second_crop[1],
            )
            first_crop = _panel_crop_around(
                first.box_2d,
                rect=rects[0],
                source_width=source_width,
                source_height=source_height,
                fixed_crop_height=shared_height,
            )
            second_crop = _panel_crop_around(
                second.box_2d,
                rect=rects[1],
                source_width=source_width,
                source_height=source_height,
                fixed_crop_height=shared_height,
            )
            if first_crop is None or second_crop is None:
                continue
        score = min(
            _panel_readability(target.box_2d, rect, crop)
            for target, rect, crop in zip(
                (first, second),
                rects,
                (first_crop, second_crop),
                strict=True,
            )
        )
        candidates.append(
            PanelLayoutSpec(
                layout_mode=mode,
                relation_mode=(
                    "context_detail"
                    if relation_mode == "context_detail"
                    else (
                        "simultaneous_comparison"
                        if physical_scale_comparison
                        else (
                            "simultaneous_relation"
                            if same_source
                            else "conceptual_comparison"
                        )
                    )
                ),
                temporal_relation=(
                    "same_source_same_pts"
                    if same_source
                    else "different_source_conceptual"
                ),
                relative_scale_policy=(
                    "locked"
                    if physical_scale_comparison
                    else "independent_nonphysical"
                ),
                panels=(
                    PanelSpec(
                        panel_id="first",
                        source_asset_id=first.source_asset_id,
                        source_pts=first.source_pts,
                        target_ids=(first.target_id,),
                        crop_box_2d=first_crop,
                        output_rect=rects[0],
                    ),
                    PanelSpec(
                        panel_id="second",
                        source_asset_id=second.source_asset_id,
                        source_pts=second.source_pts,
                        target_ids=(second.target_id,),
                        crop_box_2d=second_crop,
                        output_rect=rects[1],
                    ),
                ),
                local_layout_score=round(score, 6),
            )
        )
    return max(
        candidates,
        key=lambda item: (
            item.local_layout_score,
            item.layout_mode == "top_bottom",
        ),
        default=None,
    )


def compile_intentional_freeze(
    event_lock: ExactEventLockV2,
    *,
    cue_id: str,
    duration_ms: int,
    policy: AutonomousEditPolicy,
) -> IntentionalFreezeSpec:
    if not policy.presentation.allow_intentional_freeze:
        raise ValueError("policy forbids intentional freeze")
    if event_lock.event_type not in {
        "reaction_peak",
        "group_laugh_reaction_peak",
        "action_apex",
        "freeze_start",
    }:
        raise ValueError("intentional freeze requires an exact reaction/action frame")
    if duration_ms > policy.presentation.max_intentional_freeze_ms:
        raise ValueError("intentional freeze exceeds policy duration")
    return IntentionalFreezeSpec(
        exact_event_lock_sha256=event_lock.definition_sha256(),
        source_asset_id=event_lock.source_asset_id,
        source_pts=event_lock.source_pts,
        cue_id=cue_id,
        duration_ms=duration_ms,
        motivation="brief_authorized_phrase_ending",
    )


def shared_sam_seeds_from_grounding(
    grounding: MultiTargetGroundingGroup,
    *,
    target_requests: Sequence[GroundingTargetRequest],
    seed_time_ms: int,
    seed_frame_pts: int,
) -> tuple[SharedSam21BBoxSeed, ...]:
    """Compile one multi-object SAM session; ambiguous targets fail closed."""

    if grounding.ambiguous_target_ids:
        raise ValueError(
            "ambiguous grounding targets require a scoped retry: "
            + ", ".join(grounding.ambiguous_target_ids)
        )
    requests = {target.target_id: target for target in target_requests}
    seeds: list[SharedSam21BBoxSeed] = []
    for result in grounding.targets:
        request = requests.get(result.target_id)
        if request is None or not result.visible or len(result.candidates) != 1:
            raise ValueError("shared SAM seed requires one visible bound target")
        native = result.candidates[0].box_2d_yxyx
        y_min, x_min, y_max, x_max = native
        seeds.append(
            SharedSam21BBoxSeed(
                target_id=result.target_id,
                target_description=request.target_description,
                seed_source="gemini_multi_target_exact_frame",
                seed_time_ms=seed_time_ms,
                seed_frame_pts=seed_frame_pts,
                seed_frame_sha256=grounding.source_frame_hash,
                seed_source_width=grounding.source_width,
                seed_source_height=grounding.source_height,
                seed_box_2d=[x_min, y_min, x_max, y_max],
            )
        )
    return tuple(seeds)


def two_panel_ffmpeg_filter(
    spec: PanelLayoutSpec,
    *,
    canvas_width: int = 1080,
    canvas_height: int = 1920,
    background_color: str = "0x0b0e12",
) -> str:
    """Use one input decode for same-source panels and a static stack."""

    if spec.temporal_relation != "same_source_same_pts":
        raise ValueError(
            "one-decode two-panel filter requires same-source same-PTS panels"
        )
    first, second = spec.panels
    filter_parts = ["[0:v]split=2[p0][p1]"]
    labels: list[str] = []
    for index, panel in enumerate((first, second)):
        x0, y0, x1, y1 = panel.crop_box_2d
        crop_x = f"iw*{x0}/1000"
        crop_y = f"ih*{y0}/1000"
        crop_w = f"iw*{x1 - x0}/1000"
        crop_h = f"ih*{y1 - y0}/1000"
        width = round(canvas_width * panel.output_rect.width / 1000)
        height = round(canvas_height * panel.output_rect.height / 1000)
        label = f"panel{index}"
        labels.append(label)
        filter_parts.append(
            f"[p{index}]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={width}:{height}:flags=lanczos[{label}]"
        )
    x0 = round(canvas_width * first.output_rect.x / 1000)
    y0 = round(canvas_height * first.output_rect.y / 1000)
    x1 = round(canvas_width * second.output_rect.x / 1000)
    y1 = round(canvas_height * second.output_rect.y / 1000)
    filter_parts.append(
        f"color={background_color}:s={canvas_width}x{canvas_height}[bg]"
    )
    filter_parts.append(
        f"[bg][{labels[0]}]overlay={x0}:{y0}[with0]"
    )
    filter_parts.append(
        f"[with0][{labels[1]}]overlay={x1}:{y1}[base]"
    )
    return ";".join(filter_parts)


# Extracted from feature_cut.py.  Legacy callers import these names back from
# feature_cut, while autonomous compilation uses the same implementations.
def _signed_motion_reversal_count(
    values: Sequence[float],
    *,
    perceptual_threshold: float,
) -> int:
    """Count meaningful direction reversals while ignoring tracking jitter."""

    signs: list[int] = []
    for before, after in zip(values[:-1], values[1:], strict=True):
        delta = after - before
        if abs(delta) < perceptual_threshold:
            continue
        sign = 1 if delta > 0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return max(0, len(signs) - 1)


def _vertical_fit_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x0b0e12,setsar=1[base]"
    )


def _vertical_intentional_freeze_fit_filter(
    *,
    freeze_start_seconds: float,
    total_duration_seconds: float,
) -> str:
    return _vertical_intentional_freeze_filter(
        _vertical_fit_filter(),
        freeze_start_seconds=freeze_start_seconds,
        total_duration_seconds=total_duration_seconds,
    )


def _vertical_intentional_freeze_filter(
    presentation_filter_graph: str,
    *,
    freeze_start_seconds: float,
    total_duration_seconds: float,
) -> str:
    """Freeze the already-compiled presentation, not the uncropped source."""

    if not 0 < freeze_start_seconds < total_duration_seconds:
        raise ValueError("intentional freeze must start inside the segment")
    if not presentation_filter_graph.endswith("[base]"):
        raise ValueError(
            "intentional freeze requires a presentation graph ending in [base]"
        )
    freeze_duration = total_duration_seconds - freeze_start_seconds
    return (
        presentation_filter_graph[:-6]
        + "[presented];"
        + "[presented]"
        + f"trim=end={freeze_start_seconds:.6f},setpts=PTS-STARTPTS,"
        + f"tpad=stop_mode=clone:stop_duration={freeze_duration:.6f}"
        + "[base]"
    )


def _vertical_required_scope_fit_filter(
    geometry: Mapping[str, Any],
    *,
    margin_normalized: float = 45.0,
    autonomous_policy_reference: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Build a static solid-matte fit around the tracked required envelope."""

    keyframes = geometry.get("crop_keyframes")
    source_width = geometry.get("source_display_width")
    source_height = geometry.get("source_display_height")
    if (
        not isinstance(keyframes, list)
        or not keyframes
        or not isinstance(source_width, int)
        or not isinstance(source_height, int)
        or source_width <= 0
        or source_height <= 0
    ):
        return None
    boxes: list[list[float]] = []
    for keyframe in keyframes:
        box = (
            keyframe.get("required_union_box")
            if isinstance(keyframe, dict)
            else None
        )
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) for value in box)
            or not (0 <= box[0] < box[2] <= 1000)
            or not (0 <= box[1] < box[3] <= 1000)
        ):
            return None
        boxes.append([float(value) for value in box])
    envelope = [
        max(0.0, min(box[0] for box in boxes) - margin_normalized),
        max(0.0, min(box[1] for box in boxes) - margin_normalized),
        min(1000.0, max(box[2] for box in boxes) + margin_normalized),
        min(1000.0, max(box[3] for box in boxes) + margin_normalized),
    ]
    if envelope[2] <= envelope[0] or envelope[3] <= envelope[1]:
        return None
    crop_x = max(0, round(source_width * envelope[0] / 1000))
    crop_y = max(0, round(source_height * envelope[1] / 1000))
    crop_width = max(
        2,
        int(source_width * (envelope[2] - envelope[0]) / 1000) // 2 * 2,
    )
    crop_height = max(
        2,
        int(source_height * (envelope[3] - envelope[1]) / 1000) // 2 * 2,
    )
    crop_width = min(crop_width, source_width - crop_x)
    crop_height = min(crop_height, source_height - crop_y)
    crop_width -= crop_width % 2
    crop_height -= crop_height % 2
    if crop_width < 2 or crop_height < 2:
        return None
    filter_graph = (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x0b0e12,setsar=1[base]"
    )
    policy_authorized = autonomous_policy_reference is not None
    return filter_graph, {
        "applied_strategy": "required_scope_solid_fit",
        "fallback_reason": "required_region_union_too_large_for_safe_9x16_crop",
        "scope_fit_source": "tracked_required_union_all_frame_envelope",
        "scope_envelope_box_2d": [round(value, 3) for value in envelope],
        "scope_margin_normalized": margin_normalized,
        "scope_crop_pixels": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_width,
            "height": crop_height,
        },
        "required_sample_count": len(boxes),
        "required_envelope_contained": True,
        "risk_codes": [
            "scope_preserving_solid_fit",
            (
                "auto_policy_authorized"
                if policy_authorized
                else "human_review_required"
            ),
        ],
        "requires_gemini_review": not policy_authorized,
        "autonomous_policy_reference": autonomous_policy_reference,
        "source_geometry_lineage_passed": bool(
            geometry.get("source_geometry_lineage_passed")
        ),
        "tracking_confidence_gate_passed": bool(
            geometry.get("tracking_confidence_gate_passed")
        ),
    }


def _vertical_center_crop_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:x=(iw-ow)/2:y=(ih-oh)/2,setsar=1[base]"
    )


def static_full_bleed_crop_filter(crop_box_2d: NormalizedBox) -> str:
    """Render one normalized, geometry-proven 9:16 crop without tracking."""

    x_min, y_min, x_max, y_max = crop_box_2d
    if not (
        0 <= x_min < x_max <= 1000
        and 0 <= y_min < y_max <= 1000
    ):
        raise ValueError("static full-bleed crop is outside normalized source")
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        f"crop=w='max(2,trunc(iw*{crop_width}/1000/2)*2)':"
        f"h='max(2,trunc(ih*{crop_height}/1000/2)*2)':"
        f"x='trunc(iw*{x_min}/1000/2)*2':"
        f"y='trunc(ih*{y_min}/1000/2)*2',"
        "scale=1080:1920,setsar=1[base]"
    )


def _vertical_delivery_fallback(
    strategy: Literal["fit_with_background", "center_crop"],
    *,
    reason: str,
) -> tuple[str, dict[str, Any]]:
    """Preserve legacy review lineage while sharing the fit implementation."""

    if strategy == "center_crop":
        return _vertical_center_crop_filter(), {
            "applied_strategy": "full_bleed_center_crop_review",
            "fallback_reason": reason,
            "risk_codes": [
                "explicit_full_bleed_delivery_preference",
                "unverified_center_crop",
                "human_review_required",
            ],
            "requires_gemini_review": True,
            "full_bleed": True,
            "semantic_review_reasons": [
                "fallback_crop_requires_sequence_review",
            ],
        }
    return _vertical_fit_filter(), {
        "applied_strategy": "fit_with_solid_matte",
        "fallback_reason": reason,
        "risk_codes": [
            "scope_preserving_solid_fit",
            "human_review_required",
        ],
        "requires_gemini_review": True,
        "full_bleed": False,
        "semantic_review_reasons": [
            "scope_preserving_fit_is_review_only",
        ],
    }


def _largest_shared_vertical_crop(
    targets: Sequence[PresentationTarget],
    *,
    source_width: int,
    source_height: int,
) -> NormalizedBox | None:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    union = (
        min(target.box_2d[0] for target in targets),
        min(target.box_2d[1] for target in targets),
        max(target.box_2d[2] for target in targets),
        max(target.box_2d[3] for target in targets),
    )
    if source_width / source_height >= 9 / 16:
        crop_h = 1000
        crop_w = round(1000 * source_height * 9 / (source_width * 16))
    else:
        crop_w = 1000
        crop_h = round(1000 * source_width * 16 / (source_height * 9))
    union_w = union[2] - union[0]
    union_h = union[3] - union[1]
    if union_w > crop_w or union_h > crop_h:
        return None
    center_x = (union[0] + union[2]) / 2
    center_y = (union[1] + union[3]) / 2
    x0 = max(0, min(1000 - crop_w, round(center_x - crop_w / 2)))
    y0 = max(0, min(1000 - crop_h, round(center_y - crop_h / 2)))
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _panel_rects(
    mode: Literal["top_bottom", "side_by_side", "context_detail"],
) -> tuple[PanelRect, PanelRect]:
    gutter = 18
    if mode == "side_by_side":
        width = (1000 - gutter) // 2
        return (
            PanelRect(x=0, y=0, width=width, height=1000),
            PanelRect(x=width + gutter, y=0, width=width, height=1000),
        )
    if mode == "context_detail":
        first_height = 589
        return (
            PanelRect(x=0, y=0, width=1000, height=first_height),
            PanelRect(
                x=0,
                y=first_height + gutter,
                width=1000,
                height=1000 - first_height - gutter,
            ),
        )
    height = (1000 - gutter) // 2
    return (
        PanelRect(x=0, y=0, width=1000, height=height),
        PanelRect(x=0, y=height + gutter, width=1000, height=height),
    )


def _panel_readability(
    box: NormalizedBox,
    rect: PanelRect,
    crop: NormalizedBox,
) -> float:
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    projected_width_pixels = (
        box_width / crop_width * rect.width / 1000 * 1080
    )
    projected_height_pixels = (
        box_height / crop_height * rect.height / 1000 * 1920
    )
    return min(
        1.0,
        min(projected_width_pixels, projected_height_pixels) / 240,
    )


def _panel_crop_around(
    box: NormalizedBox,
    *,
    rect: PanelRect,
    source_width: int,
    source_height: int,
    canvas_width: int = 1080,
    canvas_height: int = 1920,
    fixed_crop_height: int | None = None,
) -> NormalizedBox | None:
    """Return a contained crop whose pixel aspect exactly fills one panel."""

    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    panel_pixel_aspect = (
        canvas_width * rect.width / 1000
    ) / (canvas_height * rect.height / 1000)
    normalized_width_per_height = (
        panel_pixel_aspect * source_height / source_width
    )
    minimum_height = max(
        float(box_height),
        float(box_width) / normalized_width_per_height,
    )
    crop_height = float(fixed_crop_height or minimum_height)
    if fixed_crop_height is None:
        maximum_scale = min(
            1000 / crop_height,
            1000 / (crop_height * normalized_width_per_height),
        )
        crop_height *= min(1.12, maximum_scale)
    crop_width = crop_height * normalized_width_per_height
    if (
        crop_height > 1000.5
        or crop_width > 1000.5
        or crop_height + 0.5 < box_height
        or crop_width + 0.5 < box_width
    ):
        return None
    crop_height_i = min(1000, max(1, round(crop_height)))
    crop_width_i = min(1000, max(1, round(crop_width)))
    if crop_height_i < box_height or crop_width_i < box_width:
        return None
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    left = max(
        0,
        min(1000 - crop_width_i, round(center_x - crop_width_i / 2)),
    )
    top = max(
        0,
        min(1000 - crop_height_i, round(center_y - crop_height_i / 2)),
    )
    if left > x0 or top > y0:
        return None
    if left + crop_width_i < x1 or top + crop_height_i < y1:
        return None
    return (left, top, left + crop_width_i, top + crop_height_i)


def presentation_cache_fragment(compilation: PresentationCompilation) -> str:
    return hashlib.sha256(
        json.dumps(
            compilation.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
