"""Versioned editing capabilities exposed to semantic planners.

The catalog is intentionally small.  Gemini chooses editorial intent from
these verbs; deterministic code still owns exact PTS, Grounding, tracking,
geometry, motion limits, candidate routing, and delivery eligibility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EditorialObjectiveProfile(_StrictModel):
    """Continuous editorial goals shared by pure and mixed content types."""

    dialogue_dependency: float = Field(default=0.0, ge=0.0, le=1.0)
    music_dependency: float = Field(default=0.0, ge=0.0, le=1.0)
    instructional_order_strictness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    chronology_strictness: float = Field(default=0.0, ge=0.0, le=1.0)
    hook_priority: float = Field(default=0.5, ge=0.0, le=1.0)
    source_audio_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    text_legibility_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    physical_relation_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    reaction_payoff_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class CanvasSpec(_StrictModel):
    canvas_id: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_aspect_ratio: float = Field(default=1.0, gt=0.0)
    safe_area_top: float = Field(default=0.0, ge=0.0, lt=0.5)
    safe_area_right: float = Field(default=0.0, ge=0.0, lt=0.5)
    safe_area_bottom: float = Field(default=0.0, ge=0.0, lt=0.5)
    safe_area_left: float = Field(default=0.0, ge=0.0, lt=0.5)
    background_policy: Literal["none", "solid"] = "none"


class VisibilityTarget(_StrictModel):
    target_id: str = Field(min_length=1)
    minimum_visible_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    minimum_readability: float = Field(default=0.5, ge=0.0, le=1.0)
    atomic: bool = False


class VisibilityContract(_StrictModel):
    targets: tuple[VisibilityTarget, ...] = Field(min_length=1)
    temporal_visibility: Literal["one", "ordered", "simultaneous"]
    preserve_spatial_relation: bool = False
    preserve_relative_scale: bool = False

    @model_validator(mode="after")
    def validate_visibility(self) -> "VisibilityContract":
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("visibility target IDs must be unique")
        if self.temporal_visibility == "one" and len(self.targets) != 1:
            raise ValueError("one-target visibility requires exactly one target")
        if self.preserve_relative_scale and len(self.targets) < 2:
            raise ValueError("relative scale preservation requires two targets")
        return self


class AttentionIntent(_StrictModel):
    ordered_target_ids: tuple[str, ...] = ()
    goal: Literal[
        "hold",
        "follow",
        "reveal",
        "compare",
        "emphasize",
        "transfer",
    ]
    motion_motivation: Literal[
        "none",
        "containment",
        "attention_transfer",
        "reveal",
        "emphasis",
    ] = "none"


class SemanticBeat(_StrictModel):
    beat_id: str = Field(min_length=1)
    priority: Literal["hard", "preferred", "optional"]
    narrative_function: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    candidate_refs: tuple[str, ...] = ()
    visibility_contract: VisibilityContract
    attention_intent: AttentionIntent
    sync_intent_ids: tuple[str, ...] = ()
    minimum_duration_ms: int = Field(gt=0)
    preferred_duration_ms: int = Field(gt=0)
    maximum_duration_ms: int = Field(gt=0)
    acceptable_capability_ids: tuple[str, ...] = Field(min_length=1)
    forbidden_capability_ids: tuple[str, ...] = ()
    panel_target_groups: tuple[tuple[str, ...], ...] = ()

    @model_validator(mode="after")
    def validate_beat(self) -> "SemanticBeat":
        if not (
            self.minimum_duration_ms
            <= self.preferred_duration_ms
            <= self.maximum_duration_ms
        ):
            raise ValueError("semantic beat duration bounds must be ordered")
        overlap = set(self.acceptable_capability_ids) & set(
            self.forbidden_capability_ids
        )
        if overlap:
            raise ValueError(
                "capabilities cannot be both acceptable and forbidden: "
                + ", ".join(sorted(overlap))
            )
        known_targets = {
            target.target_id for target in self.visibility_contract.targets
        }
        unknown_attention = set(
            self.attention_intent.ordered_target_ids
        ) - known_targets
        if unknown_attention:
            raise ValueError(
                "attention intent references unknown targets: "
                + ", ".join(sorted(unknown_attention))
            )
        if self.panel_target_groups:
            if len(self.panel_target_groups) != 2:
                raise ValueError(
                    "panel semantics must declare exactly two target groups"
                )
            flattened = [
                target_id
                for group in self.panel_target_groups
                for target_id in group
            ]
            if any(not group for group in self.panel_target_groups):
                raise ValueError("panel target groups cannot be empty")
            if len(flattened) != len(set(flattened)):
                raise ValueError(
                    "panel target groups cannot repeat target IDs"
                )
            unknown_panel_targets = set(flattened) - known_targets
            if unknown_panel_targets:
                raise ValueError(
                    "panel groups reference unknown targets: "
                    + ", ".join(sorted(unknown_panel_targets))
                )
        return self


class EditProjectIR(_StrictModel):
    """Aspect-neutral semantic edit contract; it contains no render geometry."""

    contract_version: Literal["semantic-edit-ir-v1"] = "semantic-edit-ir-v1"
    project_id: str = Field(min_length=1)
    source_catalog_ref: str = Field(min_length=1)
    policy_ref: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    objective_profile: EditorialObjectiveProfile
    outputs: tuple[CanvasSpec, ...] = Field(min_length=1)
    beats: tuple[SemanticBeat, ...] = Field(min_length=1)
    global_constraint_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_project(self) -> "EditProjectIR":
        canvas_ids = [canvas.canvas_id for canvas in self.outputs]
        beat_ids = [beat.beat_id for beat in self.beats]
        if len(canvas_ids) != len(set(canvas_ids)):
            raise ValueError("semantic edit canvas IDs must be unique")
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("semantic edit beat IDs must be unique")
        return self

    def definition_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))


class EditingCapability(_StrictModel):
    capability_id: Literal[
        "source_hold",
        "static_full_bleed_crop",
        "tracked_full_bleed_crop",
        "phase_virtual_camera",
        "hard_cut_between_views",
        "controlled_semantic_clip",
        "alternate_candidate",
        "solid_fit_review_fallback",
        "two_panel_layout",
        "solid_matte_fit",
        "intentional_freeze",
        "music_phrase_alignment",
        "bounded_final_qa_replan",
    ]
    planner_use: str = Field(min_length=1, max_length=300)
    local_executor: str = Field(min_length=1, max_length=120)
    delivery_scope: Literal[
        "delivery_candidate",
        "review_preview_only",
        "planning_only",
    ]


class EditingCapabilityCatalog(_StrictModel):
    """Hashable capability inventory supplied to the Gemini edit planner."""

    contract_version: Literal["editing-capability-catalog-v1"] = (
        "editing-capability-catalog-v1"
    )
    planner_boundary: Literal[
        "semantic_intent_only_no_exact_time_or_geometry"
    ] = "semantic_intent_only_no_exact_time_or_geometry"
    vertical_delivery_preference: Literal["full_bleed_first"] = "full_bleed_first"
    capabilities: list[EditingCapability] = Field(min_length=1)
    automatic_fallback_order: list[str] = Field(min_length=1)
    prohibited_automatic_delivery: list[
        Literal["solid_fit", "blurred_background"]
    ]

    @model_validator(mode="after")
    def validate_catalog(self) -> "EditingCapabilityCatalog":
        ids = [item.capability_id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("editing capability IDs must be unique")
        if "solid_fit_review_fallback" not in ids:
            raise ValueError("catalog must expose the typed review fallback")
        return self

    def definition_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def execution_compatible_with(
        self,
        current: "EditingCapabilityCatalog",
    ) -> bool:
        """Allow hash-bound planner prose to migrate into stricter executors.

        The semantic output never contains executable geometry. A saved plan is
        reusable only when every capability ID still exists with the same
        delivery scope and the planner boundary/aspect preference are unchanged.
        Local executor descriptions and fallback ordering may become stricter.
        """

        if (
            self.planner_boundary != current.planner_boundary
            or self.vertical_delivery_preference
            != current.vertical_delivery_preference
        ):
            return False
        saved = {
            capability.capability_id: capability.delivery_scope
            for capability in self.capabilities
        }
        active = {
            capability.capability_id: capability.delivery_scope
            for capability in current.capabilities
        }
        return all(active.get(key) == scope for key, scope in saved.items())


class CapabilitySpecV2(_StrictModel):
    """Declarative planner/runtime contract for one local editing operation."""

    capability_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    version: str = Field(min_length=1)
    planner_semantics: tuple[str, ...] = Field(min_length=1)
    required_artifact_types: tuple[str, ...] = ()
    precondition_ids: tuple[str, ...] = ()
    guarantee_ids: tuple[str, ...] = Field(min_length=1)
    verifier_ids: tuple[str, ...] = Field(min_length=1)
    policy_gate: str | None = None
    local_cost_class: Literal["cheap", "tracking", "render_proxy"] = "cheap"
    intrusion_rank: int = Field(default=0, ge=0, le=10)
    max_options_per_candidate: int = Field(default=1, ge=1, le=8)
    executor_id: str = Field(min_length=1)
    paid_model_calls: Literal[0] = 0


class CapabilityRegistryV2(_StrictModel):
    contract_version: Literal["editing-capability-registry-v2"] = (
        "editing-capability-registry-v2"
    )
    planner_boundary: Literal[
        "semantic_intent_and_immutable_option_ids_only"
    ] = "semantic_intent_and_immutable_option_ids_only"
    capabilities: tuple[CapabilitySpecV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> "CapabilityRegistryV2":
        ids = [capability.capability_id for capability in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("capability registry IDs must be unique")
        return self

    def definition_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))

    def by_id(self) -> dict[str, CapabilitySpecV2]:
        return {
            capability.capability_id: capability
            for capability in self.capabilities
        }


@runtime_checkable
class CapabilityExecutor(Protocol):
    spec: CapabilitySpecV2

    def enumerate(self, context: Mapping[str, Any]) -> Sequence[Any]: ...

    def audit(self, option: Any) -> Sequence[Any]: ...

    def compile(self, option: Any) -> Any: ...


class RuntimeCapabilityRegistry:
    """Runtime counterpart of the signed planner catalog.

    Executors only enumerate and compile. Selection remains with the common
    constraint optimizer, so adding an operation does not extend a fallback
    decision tree.
    """

    def __init__(self, manifest: CapabilityRegistryV2) -> None:
        self.manifest = manifest
        self._executors: dict[str, CapabilityExecutor] = {}

    def register(self, executor: CapabilityExecutor) -> None:
        capability_id = executor.spec.capability_id
        declared = self.manifest.by_id().get(capability_id)
        if declared is None:
            raise ValueError(
                f"executor capability is not declared: {capability_id}"
            )
        if executor.spec != declared:
            raise ValueError(
                f"executor spec does not match manifest: {capability_id}"
            )
        if capability_id in self._executors:
            raise ValueError(f"executor already registered: {capability_id}")
        self._executors[capability_id] = executor

    def require_executor(self, capability_id: str) -> CapabilityExecutor:
        try:
            return self._executors[capability_id]
        except KeyError as error:
            raise ValueError(
                f"capability has no runtime executor: {capability_id}"
            ) from error

    @property
    def registered_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))


def simple_production_capability_catalog() -> EditingCapabilityCatalog:
    """Return the production planner's executable, domain-neutral verbs."""

    return EditingCapabilityCatalog(
        capabilities=[
            EditingCapability(
                capability_id="source_hold",
                planner_use=(
                    "Keep an already useful source composition stable when no "
                    "attention transfer or corrective crop is needed."
                ),
                local_executor="FFmpeg deterministic source composition",
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="static_full_bleed_crop",
                planner_use=(
                    "Fill the requested aspect with one stable composition when "
                    "the semantic core fits without camera motion."
                ),
                local_executor="local aspect-preserving crop solver",
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="tracked_full_bleed_crop",
                planner_use=(
                    "Follow one active semantic anchor while keeping a full-bleed "
                    "canvas; movement is generated locally from tracked geometry."
                ),
                local_executor="Gemini bbox seed + SAM + crop solver",
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="phase_virtual_camera",
                planner_use=(
                    "Show evidence-bound anchors in temporal phases. Declare the "
                    "motion reason and whether positions are optimizable; local "
                    "geometry preserves source-time order, reuses a valid static "
                    "crop, and otherwise compiles a low-travel path."
                ),
                local_executor=(
                    "phase compiler + SAM tracks + motivated monotonic motion solver"
                ),
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="hard_cut_between_views",
                planner_use=(
                    "Use a cut rather than an unsafe or excessively fast pan "
                    "between independently understandable views."
                ),
                local_executor="phase compiler + deterministic cut boundary",
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="controlled_semantic_clip",
                planner_use=(
                    "Allow nonessential context or noncritical object extent to "
                    "leave frame while the explicitly selected semantic core stays."
                ),
                local_executor="visible-fraction and containment policy",
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="alternate_candidate",
                planner_use=(
                    "Try the next evidence-supported take when the selected take "
                    "cannot satisfy semantic or geometric constraints."
                ),
                local_executor="Top-K candidate router",
                delivery_scope="delivery_candidate",
            ),
            EditingCapability(
                capability_id="solid_fit_review_fallback",
                planner_use=(
                    "Preserve an inseparable wide composition only as an explicit "
                    "non-deliverable review preview."
                ),
                local_executor="solid-matte review renderer",
                delivery_scope="review_preview_only",
            ),
            EditingCapability(
                capability_id="music_phrase_alignment",
                planner_use=(
                    "Describe relative narrative energy and visible sync intent; "
                    "local MusicMap resolves legal sample/frame boundaries."
                ),
                local_executor="MusicMap and exact boundary reconciler",
                delivery_scope="planning_only",
            ),
            EditingCapability(
                capability_id="bounded_final_qa_replan",
                planner_use=(
                    "Permit at most one evidence-bound correction proposal after "
                    "QA finds repetition, missing results, crop, flow, or ending issues."
                ),
                local_executor="typed QA correction state machine",
                delivery_scope="planning_only",
            ),
        ],
        automatic_fallback_order=[
            "static_or_tracked_full_bleed",
            "phase_virtual_camera_or_hard_cut",
            "controlled_semantic_clip",
            "alternate_candidate",
            "full_bleed_review_preview",
            "solid_fit_review_preview_only_when_explicitly_requested",
        ],
        prohibited_automatic_delivery=[
            "solid_fit",
            "blurred_background",
        ],
    )


def autonomous_production_capability_catalog(
    *,
    allow_two_panel_layout: bool = True,
    allow_solid_matte_fit: bool = True,
    allow_intentional_freeze: bool = True,
) -> EditingCapabilityCatalog:
    """Extend the same catalog with only the presentations policy authorizes."""

    legacy = simple_production_capability_catalog()
    gated_capabilities = [
        EditingCapability(
            capability_id="two_panel_layout",
            planner_use=(
                "Declare that simultaneous comparison or context-detail "
                "presentation is semantically acceptable. Local code "
                "chooses top/bottom, side-by-side, rectangles and scale."
            ),
            local_executor="two-panel geometry compiler + FFmpeg",
            delivery_scope="delivery_candidate",
        ),
        EditingCapability(
            capability_id="solid_matte_fit",
            planner_use=(
                "Preserve an inseparable required scope when the signed "
                "AutonomousEditPolicy authorizes solid matte delivery."
            ),
            local_executor="scope-preserving solid-matte renderer",
            delivery_scope="delivery_candidate",
        ),
        EditingCapability(
            capability_id="intentional_freeze",
            planner_use=(
                "Hold an exact action/reaction frame only when brief and "
                "policy authorize it and a music cue binds the start."
            ),
            local_executor="exact-event PTS freeze renderer",
            delivery_scope="delivery_candidate",
        ),
    ]
    allowed = {
        "two_panel_layout": allow_two_panel_layout,
        "solid_matte_fit": allow_solid_matte_fit,
        "intentional_freeze": allow_intentional_freeze,
    }
    fallback_order = [
        "static_or_tracked_full_bleed",
        "phase_virtual_camera_or_hard_cut",
        "alternate_candidate",
    ]
    if allow_two_panel_layout:
        fallback_order.append("two_panel_layout_when_relation_requires")
    if allow_solid_matte_fit:
        fallback_order.append("solid_matte_fit_when_policy_authorized")
    fallback_order.append("optional_beat_omission_when_policy_authorized")
    return legacy.model_copy(
        update={
            "capabilities": [
                *legacy.capabilities,
                *[
                    capability
                    for capability in gated_capabilities
                    if allowed[capability.capability_id]
                ],
            ],
            "automatic_fallback_order": fallback_order,
            "prohibited_automatic_delivery": [
                *([] if allow_solid_matte_fit else ["solid_fit"]),
                "blurred_background",
            ],
        }
    )


def autonomous_capability_registry_v2(
    *,
    allow_two_panel_layout: bool = True,
    allow_solid_matte_fit: bool = True,
    allow_intentional_freeze: bool = True,
) -> CapabilityRegistryV2:
    """Return executable capability contracts without content-specific cases."""

    specs = [
        CapabilitySpecV2(
            capability_id="source_hold",
            version="v1",
            planner_semantics=("hold", "preserve_source_composition"),
            guarantee_ids=("exact_pts_preserved", "source_motion_preserved"),
            verifier_ids=("media_valid", "evidence_contained"),
            executor_id="presentation.source_hold",
        ),
        CapabilitySpecV2(
            capability_id="static_full_bleed_crop",
            version="v1",
            planner_semantics=("hold", "full_bleed", "single_composition"),
            required_artifact_types=("scene_facts",),
            precondition_ids=("shared_static_crop_feasible",),
            guarantee_ids=("exact_pts_preserved", "full_bleed"),
            verifier_ids=("target_containment", "minimum_readability"),
            executor_id="presentation.static_full_bleed",
        ),
        CapabilitySpecV2(
            capability_id="tracked_full_bleed_crop",
            version="v1",
            planner_semantics=("follow", "full_bleed", "containment"),
            required_artifact_types=("scene_facts", "target_tracks"),
            precondition_ids=("valid_tracks", "tracked_crop_feasible"),
            guarantee_ids=("exact_pts_preserved", "full_bleed"),
            verifier_ids=(
                "target_containment",
                "identity_preserved",
                "deadband_enforced",
            ),
            local_cost_class="tracking",
            intrusion_rank=1,
            executor_id="presentation.tracked_full_bleed",
        ),
        CapabilitySpecV2(
            capability_id="phase_virtual_camera",
            version="v1",
            planner_semantics=(
                "sequential_attention",
                "reveal",
                "controlled_emphasis",
            ),
            required_artifact_types=("scene_facts", "target_tracks"),
            precondition_ids=(
                "valid_tracks",
                "ordered_attention_targets",
                "motivated_path_feasible",
            ),
            guarantee_ids=("exact_pts_preserved", "semantic_order_preserved"),
            verifier_ids=(
                "target_containment",
                "no_unmotivated_reversal",
                "source_motion_compatible",
                "settle_present",
            ),
            local_cost_class="tracking",
            intrusion_rank=2,
            executor_id="presentation.minimal_virtual_camera",
        ),
        CapabilitySpecV2(
            capability_id="hard_cut_between_views",
            version="v1",
            planner_semantics=("sequential_attention", "unsafe_pan_replacement"),
            required_artifact_types=("scene_facts", "target_tracks"),
            precondition_ids=("cut_boundary_admissible",),
            guarantee_ids=("semantic_order_preserved",),
            verifier_ids=("action_complete", "cut_boundary_clean"),
            local_cost_class="tracking",
            intrusion_rank=2,
            executor_id="presentation.hard_cut_between_views",
        ),
        CapabilitySpecV2(
            capability_id="controlled_semantic_clip",
            version="v1",
            planner_semantics=("full_bleed", "nonessential_extent_clip"),
            required_artifact_types=("scene_facts",),
            precondition_ids=("semantic_core_contained",),
            guarantee_ids=("hard_core_preserved",),
            verifier_ids=("controlled_clip_authorized",),
            policy_gate="controlled_clip_authorized",
            intrusion_rank=3,
            executor_id="presentation.controlled_semantic_clip",
        ),
    ]
    if allow_two_panel_layout:
        specs.append(
            CapabilitySpecV2(
                capability_id="two_panel_layout",
                version="v1",
                planner_semantics=(
                    "simultaneous_comparison",
                    "simultaneous_relation",
                    "context_detail",
                ),
                required_artifact_types=("scene_facts", "target_tracks"),
                precondition_ids=("exactly_two_panels", "panel_geometry_feasible"),
                guarantee_ids=("exact_pts_preserved", "required_scope_preserved"),
                verifier_ids=(
                    "same_source_same_pts",
                    "relative_scale_policy",
                    "minimum_readability",
                ),
                policy_gate="allow_two_panel_layout",
                local_cost_class="tracking",
                intrusion_rank=5,
                executor_id="presentation.two_panel_layout",
            )
        )
    if allow_solid_matte_fit:
        specs.append(
            CapabilitySpecV2(
                capability_id="solid_matte_fit",
                version="v1",
                planner_semantics=("preserve_inseparable_scope",),
                required_artifact_types=("scene_facts",),
                guarantee_ids=("whole_source_scope_preserved",),
                verifier_ids=("minimum_readability", "solid_background_only"),
                policy_gate="allow_solid_matte_fit",
                intrusion_rank=8,
                executor_id="presentation.solid_matte_fit",
            )
        )
    if allow_intentional_freeze:
        specs.append(
            CapabilitySpecV2(
                capability_id="intentional_freeze",
                version="v1",
                planner_semantics=("phrase_ending", "exact_event_emphasis"),
                required_artifact_types=("exact_event_lock", "music_cue_lock"),
                precondition_ids=("brief_authorized", "cue_bound"),
                guarantee_ids=("exact_event_pts_preserved",),
                verifier_ids=("maximum_freeze_duration", "freeze_motivated"),
                policy_gate="allow_intentional_freeze",
                intrusion_rank=4,
                executor_id="presentation.intentional_freeze",
            )
        )
    return CapabilityRegistryV2(capabilities=tuple(specs))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
