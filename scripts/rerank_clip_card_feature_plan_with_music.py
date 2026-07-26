#!/usr/bin/env python3
"""Rerank an existing evidence-bound Top-K feature plan with actual music.

This stage deliberately cannot invent sources, frames, entities, boxes, masks,
or framing regions. Gemini hears the supplied music and selects only candidate
IDs already present in a validated ClipCardFeaturePlanV3. Local code then
reprojects the immutable upstream evidence into the executable FeatureEditPlan.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.feature_cut import write_external_feature_plan_projection
from jascue_video_lab.gemini import GeminiLabClient, MODEL_ID, _raw_dump
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import FeatureEditBrief, ModelProvenance, RushesCatalog
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json
from scripts.plan_clip_card_feature_cut import (
    ClipCardFeaturePlanV3,
    SelectedClipCardEvidence,
    project_feature_contracts_v3,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MusicAwareChapterSelection(StrictModel):
    feature_id: str
    horizontal_candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    vertical_candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    music_role: Literal["opening", "build", "peak", "breath", "closing", "neutral"]
    selection_reason: str = Field(min_length=1, max_length=500)


class MusicAwareTopKSelection(StrictModel):
    contract_version: Literal["clip-card-feature-music-rerank-v1"]
    project_id: str
    catalog_id: str
    upstream_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chapters: list[MusicAwareChapterSelection]
    uncertainties: list[str]
    model_provenance: ModelProvenance


def _validate_selection(
    selection: MusicAwareTopKSelection,
    *,
    upstream: ClipCardFeaturePlanV3,
    upstream_sha256: str,
    music_sha256: str,
) -> None:
    if selection.project_id != upstream.project_id:
        raise ValueError("music reranker changed immutable project ID")
    if selection.catalog_id != upstream.catalog_id:
        raise ValueError("music reranker changed immutable catalog ID")
    if selection.upstream_plan_sha256 != upstream_sha256:
        raise ValueError("music reranker changed immutable upstream plan hash")
    if selection.music_sha256 != music_sha256:
        raise ValueError("music reranker changed immutable music hash")
    expected = [chapter.feature_id for chapter in upstream.chapters]
    actual = [chapter.feature_id for chapter in selection.chapters]
    if actual != expected:
        raise ValueError("music reranker must preserve every chapter once and in order")
    for choice, chapter in zip(selection.chapters, upstream.chapters, strict=True):
        candidate_ids = {candidate.candidate_id for candidate in chapter.candidates}
        if chapter.evidence_status == "not_found":
            raise ValueError(
                "music reranking requires an upstream candidate for every chapter"
            )
        if choice.horizontal_candidate_id not in candidate_ids:
            raise ValueError(
                f"unknown horizontal candidate for {choice.feature_id}: "
                f"{choice.horizontal_candidate_id}"
            )
        if choice.vertical_candidate_id not in candidate_ids:
            raise ValueError(
                f"unknown vertical candidate for {choice.feature_id}: "
                f"{choice.vertical_candidate_id}"
            )


def _apply_selection(
    selection: MusicAwareTopKSelection,
    *,
    upstream: ClipCardFeaturePlanV3,
) -> ClipCardFeaturePlanV3:
    choices = {chapter.feature_id: chapter for chapter in selection.chapters}
    return upstream.model_copy(
        update={
            "strategy_summary": (
                upstream.strategy_summary
                + " Candidate rank-one choices were reranked by a separate "
                "actual-audio listening pass; candidate evidence and geometry "
                "contracts remain unchanged."
            ),
            "chapters": [
                chapter.model_copy(
                    update={
                        "horizontal_candidate_id": choices[
                            chapter.feature_id
                        ].horizontal_candidate_id,
                        "vertical_candidate_id": choices[
                            chapter.feature_id
                        ].vertical_candidate_id,
                    }
                )
                for chapter in upstream.chapters
            ],
            "uncertainties": list(
                dict.fromkeys(upstream.uncertainties + selection.uncertainties)
            ),
            "model_provenance": selection.model_provenance,
        }
    )


def reproject_music_aware_topk_selection(
    *,
    source_plan: MusicAwareTopKSelection,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, Any]:
    upstream_path = source_artifacts["upstream_source_plan"]
    music_path = source_artifacts["source_music"]
    evidence_path = source_artifacts["selected_clip_card_evidence"]
    upstream = ClipCardFeaturePlanV3.model_validate(read_json(upstream_path))
    evidence = SelectedClipCardEvidence.model_validate(read_json(evidence_path))
    _validate_selection(
        source_plan,
        upstream=upstream,
        upstream_sha256=sha256_file(upstream_path),
        music_sha256=sha256_file(music_path),
    )
    reranked = _apply_selection(source_plan, upstream=upstream)
    return brief, project_feature_contracts_v3(
        reranked,
        brief=brief,
        catalog=catalog,
        selected_evidence=evidence,
    )


def _candidate_payload(plan: ClipCardFeaturePlanV3) -> list[dict[str, Any]]:
    return [
        {
            "feature_id": chapter.feature_id,
            "current_horizontal_candidate_id": chapter.horizontal_candidate_id,
            "current_vertical_candidate_id": chapter.vertical_candidate_id,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "observed_visual_evidence": candidate.observed_visual_evidence,
                    "selection_reason": candidate.selection_reason,
                    "quality_risks": candidate.quality_risks,
                    "horizontal": {
                        "strategy": candidate.horizontal_strategy,
                        "camera_intent": candidate.horizontal_camera_intent,
                    },
                    "vertical": {
                        "strategy": candidate.vertical_strategy,
                        "crop_mode": candidate.vertical_crop_mode,
                        "framing_intent": candidate.framing_intent,
                        "required_entity_ids": candidate.required_entity_ids,
                        "preferred_entity_ids": candidate.preferred_entity_ids,
                        "sacrificable_entity_ids": candidate.sacrificable_entity_ids,
                    },
                }
                for candidate in chapter.candidates
            ],
        }
        for chapter in plan.chapters
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("upstream_plan_json", type=Path)
    parser.add_argument("selected_evidence_json", type=Path)
    parser.add_argument("music", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--file-cache-root", type=Path)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog_path = args.catalog_json.expanduser().resolve(strict=True)
    brief_path = args.brief_json.expanduser().resolve(strict=True)
    upstream_path = args.upstream_plan_json.expanduser().resolve(strict=True)
    evidence_path = args.selected_evidence_json.expanduser().resolve(strict=True)
    music_path = args.music.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output directory must be empty for a paid rerank")

    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    upstream = ClipCardFeaturePlanV3.model_validate(read_json(upstream_path))
    evidence = SelectedClipCardEvidence.model_validate(read_json(evidence_path))
    upstream_sha256 = sha256_file(upstream_path)
    music_sha256 = sha256_file(music_path)
    if upstream.project_id != brief.project_id or upstream.catalog_id != catalog.catalog_id:
        raise ValueError("upstream plan differs from catalog/brief")

    run_id = f"music-topk-rerank-{uuid.uuid4().hex[:8]}"
    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        interaction_id=None,
        run_id=run_id,
        generated_at=utc_now(),
    )
    prompt = f"""
你是短影音的音樂感知剪輯師。請實際聆聽附上的音樂，並只在每章既有 Top-K 候選中，分別選擇最適合 16:9 與 9:16 的 candidate_id。

你不能新增、刪除或改寫候選；不能產生 timestamp、bbox、mask、crop 座標或新的 entity。影像內容證據只來自 candidate 的 observed_visual_evidence。音樂角色和理由可以描述可聽見的開場、建立、峰值、呼吸、收尾與整體 flow，但不得捏造精確 beat 時間。

9:16 的選擇規則：
- required_entity_ids 是必須完整保留的 hard core。
- preferred_entity_ids 是盡量保留、但滿版直式構圖必要時可局部裁切的 soft context。
- sacrificable_entity_ids 可為了主體清楚與滿版而犧牲。
- 不要因為來源畫面有三個人，就自動把三個人都當成 hard core。
- 若 brief 與候選 evidence 只需要其中一人、產品或操作，優先選有明確 required target 且可 tracked_crop 的候選。
- 若三人關係本身就是必要內容，才選 fit_with_background 或能完整包含三人的候選。

contract_version 必須是 clip-card-feature-music-rerank-v1
project_id 必須原樣回傳：{upstream.project_id}
catalog_id 必須原樣回傳：{upstream.catalog_id}
upstream_plan_sha256 必須原樣回傳：{upstream_sha256}
music_sha256={music_sha256}，必須原樣回傳
model_provenance 必須先原樣回傳：
{provenance.model_dump_json(indent=2)}

## Brief
{brief.model_dump_json(indent=2)}

## 既有 Top-K 候選
{json.dumps(_candidate_payload(upstream), ensure_ascii=False, indent=2)}
""".strip()

    cache_root = (
        args.file_cache_root.expanduser().resolve()
        if args.file_cache_root is not None
        else output_dir.parent / "file-cache"
    )
    upload_client = GeminiLabClient(api_key=api_key)
    try:
        uploaded, reused = upload_client.ensure_video_upload(
            music_path,
            cache_root / music_sha256 / "music-upload",
        )
    finally:
        upload_client.close()

    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "Choose only candidate IDs from the supplied evidence-bound Top-K plan. "
            "Listen to the actual audio. Never invent visual evidence, timestamps, "
            "geometry, entities, or candidates. Return the structured contract only."
        ),
        "store": False,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "audio",
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type,
            },
        ],
        "generation_config": {
            "thinking_level": "low",
            "max_output_tokens": 8192,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(MusicAwareTopKSelection),
        },
    }
    request_path = output_dir / "music-rerank.request.json"
    raw_interaction_path = output_dir / "music-rerank.raw_interaction.json"
    raw_output_path = output_dir / "music-rerank.raw_output.json"
    write_json(request_path, request)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    try:
        interaction = client.interactions.create(**request)
    finally:
        client.close()
    write_json(raw_interaction_path, _raw_dump(interaction))
    write_json(raw_output_path, {"output_text": interaction.output_text})
    selection = MusicAwareTopKSelection.model_validate_json(interaction.output_text)
    interaction_id = str(getattr(interaction, "id", None) or "")
    selection = selection.model_copy(
        update={
            "model_provenance": selection.model_provenance.model_copy(
                update={"interaction_id": interaction_id}
            )
        }
    )
    _validate_selection(
        selection,
        upstream=upstream,
        upstream_sha256=upstream_sha256,
        music_sha256=music_sha256,
    )
    selection_path = output_dir / "music-rerank.selection.json"
    write_json(selection_path, selection)
    reranked = _apply_selection(selection, upstream=upstream)
    feature_plan = project_feature_contracts_v3(
        reranked,
        brief=brief,
        catalog=catalog,
        selected_evidence=evidence,
    )
    feature_plan_path = output_dir / "feature_edit_plan.json"
    write_json(output_dir / "clip-card-feature-plan.reranked.json", reranked)
    write_json(feature_plan_path, feature_plan)
    write_json(
        output_dir / "music-rerank.schema-validation.json",
        {
            "ok": True,
            "chapter_count": len(selection.chapters),
            "file_api_cache_reused": reused,
        },
    )
    write_external_feature_plan_projection(
        plan_dir=output_dir,
        projection_contract_id="clip-card-feature-music-rerank-v1",
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=selection_path,
        source_request_path=request_path,
        source_artifacts={
            "source_raw_interaction": raw_interaction_path,
            "source_raw_output": raw_output_path,
            "source_music": music_path,
            "upstream_source_plan": upstream_path,
            "selected_clip_card_evidence": evidence_path,
        },
    )
    pricing = summarize_usage_files([raw_interaction_path], relative_to=output_dir)
    write_json(output_dir / "pricing.json", pricing)
    print(
        json.dumps(
            {
                "chapter_count": len(selection.chapters),
                "file_api_cache_reused": reused,
                "pricing": pricing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
