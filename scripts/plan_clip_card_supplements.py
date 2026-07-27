#!/usr/bin/env python3
"""Plan capability-scoped Clip Card supplements without making paid calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from jascue_video_lab.clip_card_observations import (
    clip_card_sha256,
    plan_supplement_needs,
)
from jascue_video_lab.clip_card_retrieval import FeatureShortlistPlan
from jascue_video_lab.models import FullClipCard, RushesCatalog
from jascue_video_lab.storage import read_json, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--frontier",
        type=Path,
        help=(
            "Optional validated FeatureShortlistPlan. Soft capability triggers "
            "are evaluated only for events in this Top-K frontier; without it, "
            "only hard action/boundary risks are planned."
        ),
    )
    args = parser.parse_args()

    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    frontier_by_asset: dict[str, set[str]] = {}
    if args.frontier is not None:
        frontier = FeatureShortlistPlan.model_validate(read_json(args.frontier))
        if frontier.catalog_id != catalog.catalog_id:
            raise ValueError("frontier catalog does not match supplement catalog")
        for chapter in frontier.chapters:
            for candidate in chapter.candidates:
                frontier_by_asset.setdefault(candidate.source_asset_id, set()).add(
                    candidate.event_id
                )
    clips: list[dict[str, object]] = []
    for clip in catalog.clips:
        card_path = (
            args.prepared_library
            / "clips"
            / clip.sha256[:16]
            / "gemini"
            / "clip-card"
            / "clip_card.json"
        )
        card = FullClipCard.model_validate(read_json(card_path))
        expected_asset = f"sha256:{clip.sha256}"
        if card.source_asset_id != expected_asset:
            raise ValueError(f"Clip Card asset mismatch for {clip.clip_id}")
        needs = plan_supplement_needs(
            card,
            frontier_event_ids=(
                frontier_by_asset.get(card.source_asset_id, set())
                if args.frontier is not None
                else None
            ),
        )
        if needs:
            clips.append(
                {
                    "clip_id": clip.clip_id,
                    "source_asset_id": card.source_asset_id,
                    "proxy_asset_id": card.proxy_asset_id,
                    "base_card_path": str(card_path.resolve()),
                    "base_card_sha256": clip_card_sha256(card),
                    "events": [item.model_dump(mode="json") for item in needs],
                }
            )

    write_json(
        args.output_json,
        {
            "contract_version": "clip-observation-supplement-plan-v1",
            "catalog_id": catalog.catalog_id,
            "generated_at": utc_now(),
            "clip_count": len(clips),
            "event_count": sum(len(item["events"]) for item in clips),
            "clips": clips,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
