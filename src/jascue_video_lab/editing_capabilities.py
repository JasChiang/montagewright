"""Versioned editing capabilities exposed to semantic planners.

The catalog is intentionally small.  Gemini chooses editorial intent from
these verbs; deterministic code still owns exact PTS, Grounding, tracking,
geometry, motion limits, candidate routing, and delivery eligibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
