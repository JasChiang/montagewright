"""Evidence-bound attention and rhythm planning.

Gemini may describe relative attention needs in the existing selection call.
This module never invents frame-accurate cuts: it converts that reviewable
vector into explicit dwell bounds and boundary pressure, then clamps every
maximum to local quality-safe source capacity.
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import (
    AttentionChapterProfile,
    AttentionProfile,
    FeatureEditBrief,
    FeatureEditPlan,
    RhythmChapterPlan,
    RhythmPlan,
)
from .storage import utc_now


def _bounded_legacy_dwell_envelope(
    preferred_dwell_seconds: float,
) -> tuple[float, float]:
    """Migrate a legacy single dwell value without inventing a wide window.

    Legacy plans predate attention observations.  Their one chapter-local
    duration remains the preferred value, while the migration permits at most
    two seconds (or half the preferred dwell for short shots) of extra hold.
    The lower bound intentionally preserves the historical one-second floor
    for compatibility.  Quality-safe capacity is applied separately below.
    """

    minimum = min(1.0, preferred_dwell_seconds)
    extension = min(2.0, preferred_dwell_seconds * 0.5)
    maximum = preferred_dwell_seconds + extension
    return round(minimum, 3), round(maximum, 3)


def build_attention_profile(
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    *,
    source_brief_sha256: str,
    source_feature_plan_sha256: str,
    quality_safe_capacity_seconds: Mapping[str, float] | None = None,
) -> AttentionProfile:
    """Resolve Gemini attention observations against executable capacity."""

    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    capacities = dict(quality_safe_capacity_seconds or {})
    chapters: list[AttentionChapterProfile] = []
    for selected in plan.chapters:
        chapter_brief = brief_by_id[selected.feature_id]
        capacity = capacities.get(selected.feature_id)
        preferred = (
            selected.recommended_duration_seconds
            if selected.recommended_duration_seconds is not None
            else chapter_brief.target_duration_seconds
        )
        observation = selected.attention_observation
        if observation is not None:
            minimum = observation.minimum_dwell_seconds
            maximum = observation.maximum_dwell_seconds
            authority = "gemini_attention_observation"
            vector = {
                "semantic_novelty": observation.semantic_novelty,
                "action_progress": observation.action_progress,
                "visual_motion": observation.visual_motion,
                "composition_change": observation.composition_change,
                "reading_load": observation.reading_load,
                "unresolved_tension": observation.unresolved_tension,
                "emotional_hold_value": observation.emotional_hold_value,
                "repetition_pressure": observation.repetition_pressure,
                "music_transition_opportunity": (
                    observation.music_transition_opportunity
                ),
            }
            rationale = observation.rationale
            uncertainties = list(observation.uncertainties)
            requires_review = observation.requires_human_review
        else:
            # Legacy plans contain a preferred relative dwell but no diagnostic
            # vector. Preserve that chapter-local preferred value and use only
            # a narrow deterministic migration envelope. Never let one missing
            # observation expand a chapter to the duration of the whole brief.
            minimum, maximum = _bounded_legacy_dwell_envelope(preferred)
            authority = (
                "gemini_relative_dwell_legacy"
                if selected.recommended_duration_seconds is not None
                else "brief_fallback"
            )
            vector = {
                "semantic_novelty": None,
                "action_progress": None,
                "visual_motion": None,
                "composition_change": None,
                "reading_load": None,
                "unresolved_tension": None,
                "emotional_hold_value": None,
                "repetition_pressure": None,
                "music_transition_opportunity": None,
            }
            rationale = (
                selected.duration_rationale
                or "Legacy brief dwell only; no attention vector was observed."
            )
            uncertainties = [
                "attention_vector_unavailable_for_legacy_plan",
            ]
            requires_review = True
        if capacity is not None:
            maximum = min(maximum, capacity)
        if maximum + 0.001 < minimum:
            raise ValueError(
                f"quality-safe capacity for {selected.feature_id} is shorter "
                "than its minimum attention dwell"
            )
        preferred = max(minimum, min(preferred, maximum))
        chapters.append(
            AttentionChapterProfile(
                feature_id=selected.feature_id,
                evidence_authority=authority,
                **vector,
                minimum_dwell_seconds=round(minimum, 3),
                preferred_dwell_seconds=round(preferred, 3),
                maximum_dwell_seconds=round(maximum, 3),
                quality_safe_capacity_seconds=(
                    round(capacity, 3) if capacity is not None else None
                ),
                flow_intent=selected.flow_intent,
                rationale=rationale,
                uncertainties=uncertainties,
                requires_human_review=requires_review,
            )
        )
    return AttentionProfile(
        project_id=brief.project_id,
        source_brief_sha256=source_brief_sha256,
        source_feature_plan_sha256=source_feature_plan_sha256,
        chapters=chapters,
        generated_at=utc_now(),
    )


def reconcile_attention_delivery_floor(
    attention: AttentionProfile,
    *,
    delivery_floor_seconds: float,
    maximum_shortfall_tolerance_seconds: float = 1.0,
) -> tuple[AttentionProfile, dict[str, object]]:
    """Resolve a tiny cross-chapter rounding conflict without hiding it.

    Gemini proposes subjective per-chapter maximum dwell values independently.
    Their sum can miss an explicit delivery floor by a fraction even when local
    QualitySafeInterval capacity is ample.  This function may distribute only
    a small, configured shortfall to the strongest hold/read chapters.  It
    never exceeds quality-safe capacity and records every adjustment.
    """

    if delivery_floor_seconds <= 0:
        raise ValueError("delivery floor must be positive")
    if maximum_shortfall_tolerance_seconds < 0:
        raise ValueError("maximum shortfall tolerance cannot be negative")
    original_total = round(
        sum(chapter.maximum_dwell_seconds for chapter in attention.chapters),
        3,
    )
    shortfall = round(max(0.0, delivery_floor_seconds - original_total), 3)
    audit: dict[str, object] = {
        "contract_version": "attention-delivery-floor-reconciliation-v1",
        "delivery_floor_seconds": delivery_floor_seconds,
        "original_attention_maximum_seconds": original_total,
        "maximum_shortfall_tolerance_seconds": (
            maximum_shortfall_tolerance_seconds
        ),
        "shortfall_seconds": shortfall,
        "applied": False,
        "adjustments": [],
        "interpretation": (
            "local_reconciliation_of_small_subjective_dwell_rounding_only"
        ),
    }
    if shortfall <= 0.001:
        audit["resolved_attention_maximum_seconds"] = original_total
        return attention, audit
    if shortfall > maximum_shortfall_tolerance_seconds + 0.001:
        raise ValueError(
            "attention and QualitySafeInterval capacity cannot reach the "
            f"{delivery_floor_seconds:g}-second delivery floor; "
            f"shortfall={shortfall:.3f}s exceeds local reconciliation tolerance"
        )

    def hold_value(chapter: AttentionChapterProfile) -> float:
        values = (
            (0.30, chapter.emotional_hold_value),
            (0.25, chapter.reading_load),
            (0.20, chapter.semantic_novelty),
            (0.15, chapter.unresolved_tension),
            (-0.10, chapter.repetition_pressure),
        )
        return sum(weight * float(value or 0.0) for weight, value in values)

    ordered = sorted(
        enumerate(attention.chapters),
        key=lambda item: (
            -hold_value(item[1]),
            -(
                float(item[1].quality_safe_capacity_seconds or 0.0)
                - item[1].maximum_dwell_seconds
            ),
            item[0],
        ),
    )
    remaining = shortfall
    replacements: dict[int, AttentionChapterProfile] = {}
    adjustments: list[dict[str, object]] = []
    for index, chapter in ordered:
        if remaining <= 0.001:
            break
        capacity = chapter.quality_safe_capacity_seconds
        if capacity is None:
            continue
        headroom = max(0.0, capacity - chapter.maximum_dwell_seconds)
        if headroom <= 0.001:
            continue
        added = round(min(headroom, remaining), 3)
        if added <= 0:
            continue
        resolved_maximum = round(chapter.maximum_dwell_seconds + added, 3)
        replacements[index] = chapter.model_copy(
            update={
                "maximum_dwell_seconds": resolved_maximum,
                "uncertainties": [
                    *chapter.uncertainties,
                    "maximum_dwell_extended_by_local_delivery_floor_reconciliation",
                ],
                "requires_human_review": True,
            }
        )
        adjustments.append(
            {
                "feature_id": chapter.feature_id,
                "from_maximum_dwell_seconds": chapter.maximum_dwell_seconds,
                "to_maximum_dwell_seconds": resolved_maximum,
                "added_seconds": added,
                "quality_safe_capacity_seconds": capacity,
                "selection_reason": (
                    "highest_generalized_hold_read_value_with_safe_capacity"
                ),
            }
        )
        remaining = round(max(0.0, remaining - added), 3)
    if remaining > 0.001:
        raise ValueError(
            "attention delivery-floor reconciliation has insufficient "
            f"QualitySafeInterval headroom; remaining={remaining:.3f}s"
        )
    chapters = [
        replacements.get(index, chapter)
        for index, chapter in enumerate(attention.chapters)
    ]
    resolved = attention.model_copy(update={"chapters": chapters})
    audit.update(
        {
            "applied": True,
            "adjustments": adjustments,
            "resolved_attention_maximum_seconds": round(
                sum(chapter.maximum_dwell_seconds for chapter in chapters), 3
            ),
        }
    )
    return resolved, audit


def _cut_pressure(chapter: AttentionChapterProfile) -> float | None:
    values = (
        chapter.semantic_novelty,
        chapter.action_progress,
        chapter.visual_motion,
        chapter.composition_change,
        chapter.reading_load,
        chapter.unresolved_tension,
        chapter.emotional_hold_value,
        chapter.repetition_pressure,
        chapter.music_transition_opportunity,
    )
    if any(value is None for value in values):
        return None
    assert all(value is not None for value in values)
    transition = (
        0.30 * float(chapter.repetition_pressure)
        + 0.25 * float(chapter.music_transition_opportunity)
        + 0.20 * float(chapter.action_progress)
        + 0.15 * float(chapter.composition_change)
        + 0.10 * float(chapter.visual_motion)
    )
    protection = (
        0.35 * float(chapter.reading_load)
        + 0.25 * float(chapter.unresolved_tension)
        + 0.25 * float(chapter.emotional_hold_value)
        + 0.15 * float(chapter.semantic_novelty)
    )
    return round(max(0.0, min(1.0, 0.5 + transition - protection)), 6)


def build_rhythm_plan(
    attention: AttentionProfile,
    *,
    target_duration_seconds: float,
    attention_profile_sha256: str,
    style_profile: str = "standard",
) -> RhythmPlan:
    """Create transparent boundary priorities without snapping to frame time."""

    if style_profile not in {"calm", "standard", "energetic"}:
        raise ValueError("rhythm style must be calm, standard, or energetic")
    thresholds = {
        "calm": (0.35, 0.75),
        "standard": (0.40, 0.65),
        "energetic": (0.30, 0.55),
    }
    low_threshold, high_threshold = thresholds[style_profile]
    chapters: list[RhythmChapterPlan] = []
    for chapter in attention.chapters:
        pressure = _cut_pressure(chapter)
        if pressure is None:
            priority = "normal"
        elif pressure >= high_threshold:
            priority = "high"
        elif pressure <= low_threshold:
            priority = "low"
        else:
            priority = "normal"
        boundary_alignment = (
            chapter.flow_intent.boundary_alignment
            if chapter.flow_intent is not None
            else "free"
        )
        if boundary_alignment == "content_locked":
            priority = "low"
        elif boundary_alignment == "accent_preferred":
            priority = "high"
        protected: list[str] = []
        transition: list[str] = []
        for value, reason in (
            (chapter.reading_load, "reading_load"),
            (chapter.unresolved_tension, "unresolved_action_or_tension"),
            (chapter.emotional_hold_value, "intentional_emotional_hold"),
        ):
            if value is not None and value >= 0.6:
                protected.append(reason)
        for value, reason in (
            (chapter.repetition_pressure, "repetition_pressure"),
            (chapter.music_transition_opportunity, "music_transition_opportunity"),
            (chapter.action_progress, "action_or_result_complete"),
        ):
            if value is not None and value >= 0.6:
                transition.append(reason)
        chapters.append(
            RhythmChapterPlan(
                feature_id=chapter.feature_id,
                minimum_duration_seconds=chapter.minimum_dwell_seconds,
                preferred_duration_seconds=chapter.preferred_dwell_seconds,
                maximum_duration_seconds=chapter.maximum_dwell_seconds,
                cut_pressure=pressure,
                boundary_priority=priority,
                boundary_alignment=boundary_alignment,
                flow_intent=chapter.flow_intent,
                protected_reasons=protected,
                transition_reasons=transition,
                evidence_authority=chapter.evidence_authority,
            )
        )
    resolved_target_duration_seconds = target_duration_seconds
    if any(
        chapter.evidence_authority
        in {"gemini_relative_dwell_legacy", "brief_fallback"}
        for chapter in attention.chapters
    ):
        # A legacy brief may carry a historical project target that cannot be
        # filled by its chapter-local evidence.  Do not re-expand one shot to
        # that target; bound the review plan to the aggregate migrated envelope.
        resolved_target_duration_seconds = max(
            sum(chapter.minimum_dwell_seconds for chapter in attention.chapters),
            min(
                target_duration_seconds,
                sum(
                    chapter.maximum_dwell_seconds
                    for chapter in attention.chapters
                ),
            ),
        )
    return RhythmPlan(
        project_id=attention.project_id,
        style_profile=style_profile,
        target_duration_seconds=resolved_target_duration_seconds,
        attention_profile_sha256=attention_profile_sha256,
        chapters=chapters,
        generated_at=utc_now(),
    )
