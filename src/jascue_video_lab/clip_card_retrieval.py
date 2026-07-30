from __future__ import annotations

import re
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .models import (
    FeatureEditBrief,
    FrozenStrictModel,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
    StrictModel,
)
from .clip_card_observations import (
    ClipObservationSupplement,
    effective_event_observations,
)


SHORTLIST_CONTRACT_VERSION = "clip-card-feature-shortlist-v1"


def _numeric_padding_key(identifier: str) -> tuple[tuple[str, object], ...]:
    """Preserve text exactly while making decimal zero padding comparable."""

    return tuple(
        ("number", int(part)) if part.isdigit() else ("text", part)
        for part in re.split(r"(\d+)", identifier)
        if part
    )


def normalize_shortlist_event_ids(
    payload: object,
    *,
    cards: dict[str, FullClipCard],
) -> list[dict[str, object]]:
    """Repair only a unique zero-padding variant inside the referenced asset.

    Gemini sometimes copies ``evt_001`` as ``evt_01`` even though the prompt
    requires verbatim immutable IDs.  Treating arbitrary near-matches as
    equivalent would weaken evidence binding, so this normalizer is narrower:
    the source asset must already be exact, the textual ID components must be
    byte-for-byte equal, and numeric-padding equivalence must identify exactly
    one event in that card.  Ambiguous or otherwise unknown IDs remain
    unchanged and fail the existing validator.
    """

    if not isinstance(payload, dict):
        return []
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        return []
    changes: list[dict[str, object]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        candidates = chapter.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            source_asset_id = candidate.get("source_asset_id")
            event_id = candidate.get("event_id")
            if not isinstance(source_asset_id, str) or not isinstance(
                event_id, str
            ):
                continue
            card = cards.get(source_asset_id)
            if card is None:
                continue
            canonical_ids = {event.event_id for event in card.events}
            if event_id in canonical_ids:
                continue
            key = _numeric_padding_key(event_id)
            matches = sorted(
                canonical_id
                for canonical_id in canonical_ids
                if _numeric_padding_key(canonical_id) == key
            )
            if len(matches) != 1:
                continue
            canonical_id = matches[0]
            candidate["event_id"] = canonical_id
            changes.append(
                {
                    "feature_id": chapter.get("feature_id"),
                    "source_asset_id": source_asset_id,
                    "field": "event_id",
                    "from": event_id,
                    "to": canonical_id,
                    "reason": "unique_numeric_padding_equivalence",
                }
            )
    return changes


class FeatureShortlistCandidate(FrozenStrictModel):
    source_asset_id: str
    event_id: str
    retrieval_reason: str = Field(min_length=1, max_length=300)


class FeatureChapterShortlist(StrictModel):
    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    candidates: list[FeatureShortlistCandidate] = Field(
        default_factory=list, max_length=8
    )
    uncertainty: str = ""

    @model_validator(mode="after")
    def validate_candidate_count(self) -> "FeatureChapterShortlist":
        if self.evidence_status == "not_found":
            if self.candidates:
                raise ValueError("not_found shortlist chapters cannot contain candidates")
            return self
        minimum = 2 if self.evidence_status == "supported" else 1
        if not minimum <= len(self.candidates) <= 8:
            raise ValueError(
                f"{self.evidence_status} shortlist requires {minimum}-8 candidates"
            )
        references = [
            (candidate.source_asset_id, candidate.event_id)
            for candidate in self.candidates
        ]
        if len(references) != len(set(references)):
            raise ValueError("shortlist candidates must reference distinct events")
        return self


class FeatureShortlistPlan(StrictModel):
    contract_version: Literal["clip-card-feature-shortlist-v1"] = (
        SHORTLIST_CONTRACT_VERSION
    )
    project_id: str
    catalog_id: str
    chapters: list[FeatureChapterShortlist]
    uncertainties: list[str]
    model_provenance: ModelProvenance


def compact_retrieval_card(
    card: FullClipCard,
    supplements: Iterable[ClipObservationSupplement] = (),
) -> dict[str, object]:
    """High-recall text evidence only; exact geometry remains a later stage."""

    observations = effective_event_observations(card, supplements)
    return {
        "source_asset_id": card.source_asset_id,
        "duration_ms": card.duration_ms,
        "summary": card.summary,
        "content_type": card.content_type,
        "clip_uses": card.clip_uses,
        "uncertainties": card.uncertainties,
        "entities": {
            entity.entity_id: {
                "kind": entity.kind.value,
                "label": entity.label,
                "distinguishing_features": entity.distinguishing_features,
            }
            for entity in card.entities
        },
        "events": [
            {
                "event_id": event.event_id,
                "label": event.label,
                "observable_evidence": event.observable_evidence,
                "action_completeness": event.action_completeness,
                "editing_uses": event.editing_uses,
                "quality_risks": event.quality_risks,
                "entity_ids": event.entity_ids,
                "evidence_origin": (
                    observations[event.event_id].evidence_origin.model_dump(
                        mode="json"
                    )
                    if observations[event.event_id].evidence_origin
                    else None
                ),
                "observation_basis": (
                    observations[event.event_id].observation_basis.value
                    if observations[event.event_id].observation_basis
                    else None
                ),
                "capabilities": observations[
                    event.event_id
                ].capabilities.model_dump(mode="json"),
                "source_action_completeness": observations[
                    event.event_id
                ].source_action_completeness,
                "clean_entry": observations[event.event_id].clean_entry,
                "clean_exit": observations[event.event_id].clean_exit,
                "evidence_roles": observations[
                    event.event_id
                ].evidence_roles.model_dump(mode="json"),
                "observable_beats": [
                    {
                        "kind": beat.kind,
                        "entity_ids": beat.entity_ids,
                        "relation_mode": beat.relation_mode,
                        "observable_predicate": beat.observable_predicate,
                    }
                    for beat in observations[event.event_id].observable_beats
                ],
                "readability": [
                    item.model_dump(mode="json")
                    for item in observations[event.event_id].readability
                ],
                "audio_role": (
                    observations[event.event_id].audio_role.model_dump(mode="json")
                    if observations[event.event_id].audio_role
                    else None
                ),
                "observation_uncertainties": observations[
                    event.event_id
                ].uncertainties,
            }
            for event in card.events
        ],
    }


def validate_feature_shortlist(
    plan: FeatureShortlistPlan,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
) -> None:
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("shortlist changed immutable project or catalog identity")
    expected = [chapter.feature_id for chapter in brief.chapters]
    actual = [chapter.feature_id for chapter in plan.chapters]
    if actual != expected:
        raise ValueError("shortlist must preserve every brief chapter in order")
    catalog_assets = {f"sha256:{clip.sha256}" for clip in catalog.clips}
    for chapter in plan.chapters:
        for candidate in chapter.candidates:
            if (
                candidate.source_asset_id not in catalog_assets
                or candidate.source_asset_id not in cards
            ):
                raise ValueError(
                    f"shortlist references unknown asset: {candidate.source_asset_id}"
                )
            card = cards[candidate.source_asset_id]
            if not any(
                event.event_id == candidate.event_id for event in card.events
            ):
                raise ValueError(
                    "shortlist references unknown event: "
                    f"{candidate.source_asset_id}/{candidate.event_id}"
                )
