"""Deterministic presentation compiler for autonomous feature delivery.

This module owns pixel/layout decisions.  Semantic planners may declare a
relation mode, but never panel orientation, crop coordinates, scale, or motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from .autonomous_policy import AutonomousEditPolicy
from .editing_capabilities import autonomous_capability_registry_v2
from .event_lock import ExactEventLockV2
from .models import (
    FrozenStrictModel,
    SharedSam21BBoxSeed,
)
from .sequence_optimizer import (
    ConstraintResult,
    ExecutableOptionV2,
    ExecutableOptionSelectionV2,
    OptionMetrics,
    select_executable_option,
)


NormalizedBox = tuple[int, int, int, int]


class GroundingTargetRequest(FrozenStrictModel):
    target_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$",
    )
    target_description: str = Field(min_length=1)
    exclusions: tuple[str, ...] = ()


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
    minimum_visible_fraction: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_box(self) -> "PresentationTarget":
        x_min, y_min, x_max, y_max = self.box_2d
        if not (0 <= x_min < x_max <= 1000):
            raise ValueError("target x coordinates are invalid")
        if not (0 <= y_min < y_max <= 1000):
            raise ValueError("target y coordinates are invalid")
        return self


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

    contract_version: Literal["presentation-scene-facts-v1"] = (
        "presentation-scene-facts-v1"
    )
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    target_ids: tuple[str, ...] = Field(min_length=1)
    target_center_x: tuple[float, ...] = Field(min_length=1)
    target_width_fractions: tuple[float, ...] = Field(min_length=1)
    normalized_distance_matrix: tuple[tuple[float, ...], ...]
    shared_static_crop_feasible: bool
    source_camera_motion: SourceCameraMotionEstimate
    source_camera_motion_measured: bool = False
    target_readability_by_canvas: Mapping[str, float]
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
        if len(self.normalized_distance_matrix) != count or any(
            len(row) != count for row in self.normalized_distance_matrix
        ):
            raise ValueError("scene fact distance matrix dimensions are invalid")
        return self

    @property
    def definition_sha256(self) -> str:
        return _canonical_payload_sha(self.model_dump(mode="json"))


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
    direction: Literal["static", "left", "right", "mixed"]
    normalized_travel: float = Field(ge=0.0)
    reversal_count: int = Field(ge=0)


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
    movement_motivated: bool = False,
    preferred_capability_ids: Sequence[str] = (),
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
        movement_motivated=movement_motivated,
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
    movement_motivated: bool = False,
) -> PresentationOptionSet:
    """Enumerate all locally feasible operations before selecting one.

    This function has no content-category branches. It consumes target
    topology, relation intent, continuous motion facts and policy gates.
    """

    if not targets:
        raise ValueError("presentation option generation requires targets")
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
    source_motion = estimate_source_camera_motion(
        source_camera_x_values
        or tuple(0.5 for _ in targets)
    )
    scene_facts = _build_scene_facts(
        targets=targets,
        source_width=source_width,
        source_height=source_height,
        shared_crop=shared_crop,
        source_motion=source_motion,
        source_camera_motion_measured=bool(source_camera_x_values),
    )
    options: list[PresentationOption] = []

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
                    _constraint(
                        "shared_static_crop",
                        passed=True,
                        reason_code="required_targets_fit_static_full_bleed",
                        evidence_refs=(scene_facts.definition_sha256,),
                    ),
                ),
                semantic_fit=(
                    0.96
                    if relation_mode != "sequential_focus"
                    else 0.78
                ),
                readability=1.0,
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
                ),
                reason_code=(
                    "required_relation_preserved"
                    if (
                        relation_mode
                        not in {"simultaneous_relation", "context_detail"}
                        or shared_crop is not None
                    )
                    else "tracked_single_view_cannot_preserve_relation"
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
                readability=0.92,
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
                                    camera_motion.source_motion.direction
                                    == "static"
                                ),
                                reason_code=(
                                    "source_camera_static"
                                    if camera_motion.source_motion.direction
                                    == "static"
                                    else "source_motion_would_be_counteracted"
                                ),
                                evidence_refs=(scene_facts.definition_sha256,),
                            )
                            if scene_facts.source_camera_motion_measured
                            else _preference_constraint(
                                "source_motion_measurement",
                                status="unknown",
                                reason_code="source_camera_motion_not_measured",
                                evidence_refs=(scene_facts.definition_sha256,),
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
        and len(targets) == 2
        and policy.presentation.allow_two_panel_layout
    ):
        panel = choose_two_panel_layout(
            targets[0],
            targets[1],
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
            options.append(
                _presentation_option(
                    capability_id="two_panel_layout",
                    mode="two_panel_layout",
                    scene_facts=scene_facts,
                    payload=panel.model_dump(mode="json"),
                    constraints=(
                        _constraint(
                            "two_panel_count",
                            passed=True,
                            reason_code="exactly_two_panels",
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
                    ),
                    semantic_fit=(
                        0.94 if shared_crop is None else 0.68
                    ),
                    readability=max(0.5, panel.local_layout_score),
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
                    _constraint(
                        "policy_authorized",
                        passed=True,
                        reason_code="solid_matte_fit_policy_authorized",
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
) -> SceneFacts:
    centers = tuple(
        ((target.box_2d[0] + target.box_2d[2]) / 2) / 1000
        for target in targets
    )
    widths = tuple(
        (target.box_2d[2] - target.box_2d[0]) / 1000
        for target in targets
    )
    distances = tuple(
        tuple(abs(first - second) for second in centers)
        for first in centers
    )
    same_source_same_pts = len(
        {(target.source_asset_id, target.source_pts) for target in targets}
    ) == 1
    # This is a geometry proxy, not a semantic claim. Exact text/UI
    # readability remains a separate evidence provider.
    portrait_readability = min(1.0, max(0.0, min(widths) * 4.0))
    return SceneFacts(
        source_width=source_width,
        source_height=source_height,
        target_ids=tuple(target.target_id for target in targets),
        target_center_x=centers,
        target_width_fractions=widths,
        normalized_distance_matrix=distances,
        shared_static_crop_feasible=shared_crop is not None,
        source_camera_motion=source_motion,
        source_camera_motion_measured=source_camera_motion_measured,
        target_readability_by_canvas={"9:16": round(portrait_readability, 6)},
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
    movement_motivated: bool,
    initial_position_optimizable: bool = True,
    deadband: float = 0.05,
) -> CameraMotionDecision:
    """Suppress drift/reversals and preserve source camera motion by default."""

    if not required_x_values:
        raise ValueError("camera motion compiler requires positions")
    if any(not 0.0 <= value <= 1.0 for value in required_x_values):
        raise ValueError("camera positions must be normalized")
    source = estimate_source_camera_motion(
        source_camera_x_values or tuple(0.5 for _ in required_x_values),
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
        score = sum(
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
    if not 0 < freeze_start_seconds < total_duration_seconds:
        raise ValueError("intentional freeze must start inside the segment")
    freeze_duration = total_duration_seconds - freeze_start_seconds
    return (
        "[0:v]fps=30,"
        f"trim=end={freeze_start_seconds:.6f},setpts=PTS-STARTPTS,"
        f"tpad=stop_mode=clone:stop_duration={freeze_duration:.6f},"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x0b0e12,setsar=1[base]"
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
    target_occupancy = min(
        1.0,
        (box_width * box_height) / (crop_width * crop_height),
    )
    area = rect.width * rect.height / 1_000_000
    return area * target_occupancy**0.5


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
