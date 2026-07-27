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
                    "Show different anchors in contiguous relative phases when "
                    "the meaning can be reconstructed sequentially."
                ),
                local_executor="phase compiler + SAM tracks + motion solver",
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
