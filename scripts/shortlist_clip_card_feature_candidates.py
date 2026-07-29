#!/usr/bin/env python3
"""High-recall Gemini retrieval before the geometry-aware feature planner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.clip_card_retrieval import (
    FeatureShortlistPlan,
    compact_retrieval_card,
    validate_feature_shortlist,
)
from jascue_video_lab.clip_card_observations import ClipObservationSupplement
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.event_lock import (
    EditorialBeatContract,
    load_editorial_beat_contracts,
)
from jascue_video_lab.models import (
    FeatureEditBrief,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import append_error, read_json, utc_now, write_json


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shortlist_input_binding(
    *,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    editorial_contracts: tuple[EditorialBeatContract, ...],
    evidence: list[dict[str, object]],
    thinking_level: str,
) -> dict[str, object]:
    """Stable inputs that authorize reuse of a saved paid shortlist response."""

    schema = gemini_response_schema(FeatureShortlistPlan)
    return {
        "contract_version": "feature-shortlist-input-binding-v1",
        "model": MODEL_ID,
        "catalog_sha256": _canonical_sha256(catalog.model_dump(mode="json")),
        "brief_sha256": _canonical_sha256(brief.model_dump(mode="json")),
        "editorial_contracts_sha256": _canonical_sha256(
            [
                contract.model_dump(mode="json")
                for contract in editorial_contracts
            ]
        ),
        "effective_evidence_sha256": _canonical_sha256(evidence),
        "response_schema_sha256": _canonical_sha256(schema),
        "thinking_level": thinking_level,
        "max_output_tokens": 12_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--thinking-level", choices=["low", "high"], default="low")
    parser.add_argument(
        "--reuse-raw-output",
        action="store_true",
        help=(
            "Normalize and validate the saved paid response without another "
            "Gemini request."
        ),
    )
    parser.add_argument(
        "--editorial-beat-contracts",
        type=Path,
        help=(
            "Optional selected-window contracts that constrain event recall "
            "before final semantic planning."
        ),
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        action="append",
        default=[],
        help="Repeatable validated ClipObservationSupplement JSON.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.reuse_raw_output and not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
    editorial_contracts = (
        load_editorial_beat_contracts(
            args.editorial_beat_contracts.expanduser().resolve(strict=True)
        )
        if args.editorial_beat_contracts is not None
        else ()
    )
    cards: dict[str, FullClipCard] = {}
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
        cards[card.source_asset_id] = card
    supplements: dict[str, list[ClipObservationSupplement]] = {}
    for path in args.supplement:
        supplement = ClipObservationSupplement.model_validate(read_json(path))
        supplements.setdefault(supplement.source_asset_id, []).append(supplement)

    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        run_id=f"feature-shortlist-{uuid.uuid4().hex[:12]}",
        generated_at=utc_now(),
        interaction_id=None,
    )
    evidence = [
        compact_retrieval_card(card, supplements.get(card.source_asset_id, []))
        for card in cards.values()
    ]
    prompt = f"""
你是 evidence-bound 的影片素材召回器。請先為每個 brief chapter 從完整 Clip Card
library 找出值得進入精細選片的 event 候選。本階段只做高召回 retrieval，不決定
frame、bbox、crop、剪點或最終排名。

規則：
1. brief 是使用者允許的敘事 claim，不是畫面證據。只能依 observable_evidence
   與 generic evidence_origin 判斷候選；不得使用品牌、人物、活動或物件常識
   補足畫面。direct_source_event、mediated_depiction、
   graphic_or_text_claim、context_only 與 unknown 描述的是證據來源關係，
   不是內容類型，也不是 claim 已成立的結論。
2. chapters 必須依 brief 順序恰好回傳一次。
3. supported 回傳 2–8 個不同 asset/event；partial 回傳 1–8 個；
   not_found 回傳空 candidates。真實證據不足時不得為了數量虛構候選。
4. source_asset_id 與 event_id 只能逐字引用下方 library。
5. retrieval_reason 簡要說明畫面為何可能符合 brief，並保留衝突與風險。
6. 不輸出 frame ID、時間、座標、模型規格或未觀察到的功能。
7. capability 為 not_assessed 時只代表尚未補件，不能據此否定候選；
   也不得自行補出 action、result、relation、readability 或 audio role。
8. 這是粗召回，不要求 Base Clip Card 已有 exact event type 或 exact frame。
   當 event 的 observable_evidence 直接描述 contracted 動作、狀態轉換或結果，
   即使整段仍需下游 ExactEventLock 定位，也必須保留為候選。反之，只顯示
   相同物件的靜態 setup 不能取代 UI state change、action apex、reaction peak
   或 result stable start；完整 library 都沒有可觀察描述時，hard event 才是
   not_found。
9. mediated depiction、圖文 claim 或 context-only 畫面可以保留為 contextual
   候選，但不得用它們冒充 hard predicate 的 direct source event。若 brief
   允許替代表達，retrieval_reason 必須明確說明它只支持哪一部分、缺少哪個
   observable predicate。

contract_version 必須原樣回傳：clip-card-feature-shortlist-v1
project_id 必須原樣回傳：{brief.project_id}
catalog_id 必須原樣回傳：{catalog.catalog_id}
model_provenance 必須原樣回傳：
{provenance.model_dump_json(indent=2)}

## 使用者 brief
{brief.model_dump_json(indent=2)}

## Selected-window EditorialBeatContracts
{json.dumps([contract.model_dump(mode="json") for contract in editorial_contracts], ensure_ascii=False, indent=2)}

## 完整精簡 Clip Card library
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}
""".strip()
    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "Use only supplied Clip Card evidence. This is retrieval, not final "
            "selection or geometry. Never replace visible evidence with model memory."
        ),
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {
            "thinking_level": args.thinking_level,
            "max_output_tokens": 12_000,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(FeatureShortlistPlan),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "feature-shortlist.request.json"
    binding_path = args.output_dir / "feature-shortlist.input-binding.json"
    current_binding = _shortlist_input_binding(
        catalog=catalog,
        brief=brief,
        editorial_contracts=tuple(editorial_contracts),
        evidence=evidence,
        thinking_level=args.thinking_level,
    )
    raw_path = args.output_dir / "feature-shortlist.raw_interaction.json"
    raw_output_path = args.output_dir / "feature-shortlist.raw_output.json"
    if args.reuse_raw_output:
        if not raw_path.is_file() or not raw_output_path.is_file():
            raise FileNotFoundError(
                "shortlist raw reuse requires saved interaction and output"
            )
        if not binding_path.is_file():
            raise FileNotFoundError(
                "shortlist raw reuse requires a saved input binding"
            )
        saved_binding = read_json(binding_path)
        if saved_binding != current_binding:
            raise ValueError(
                "shortlist raw output is stale for the current brief, catalog, "
                "contracts, effective Clip Card observations, schema, or model"
            )
        write_json(
            args.output_dir / "feature-shortlist.reprojection-request.json",
            request,
        )
        raw_interaction = read_json(raw_path)
        output_text = str(read_json(raw_output_path)["output_text"])
        interaction_id = str(raw_interaction.get("id") or "")
    else:
        write_json(request_path, request)
        write_json(binding_path, current_binding)
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        try:
            try:
                interaction = client.interactions.create(**request)
            except Exception as error:
                append_error(args.output_dir, "feature_shortlist", error)
                write_json(
                    args.output_dir / "feature-shortlist.schema-validation.json",
                    {
                        "ok": False,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "request_sent": True,
                        "raw_interaction_saved": False,
                    },
                )
                raise
        finally:
            client.close()
        raw_interaction = _raw_dump(interaction)
        output_text = interaction.output_text
        interaction_id = getattr(interaction, "id", None) or ""
        write_json(raw_path, raw_interaction)
        write_json(raw_output_path, {"output_text": output_text})
    payload = json.loads(output_text)
    normalization_changes: list[dict[str, object]] = []
    for chapter in payload.get("chapters", []):
        candidates = chapter.get("candidates")
        if (
            chapter.get("evidence_status") == "supported"
            and isinstance(candidates, list)
            and len(candidates) == 1
        ):
            chapter["evidence_status"] = "partial"
            normalization_changes.append(
                {
                    "feature_id": chapter.get("feature_id"),
                    "from": "supported",
                    "to": "partial",
                    "reason": "exhaustive_library_returned_one_candidate",
                }
            )
    write_json(
        args.output_dir / "feature-shortlist.normalization-audit.json",
        {
            "contract_version": "feature-shortlist-normalization-v1",
            "changes": normalization_changes,
            "paid_media_replayed": False,
            "reused_raw_output": args.reuse_raw_output,
        },
    )
    plan = FeatureShortlistPlan.model_validate(payload)
    validate_feature_shortlist(
        plan,
        brief=brief,
        catalog=catalog,
        cards=cards,
    )
    final = plan.model_copy(
        update={
            "model_provenance": plan.model_provenance.model_copy(
                update={"interaction_id": interaction_id}
            )
        }
    )
    plan_path = args.output_dir / "feature-shortlist.json"
    write_json(plan_path, final)
    write_json(
        args.output_dir / "feature-shortlist.schema-validation.json",
        {"ok": True, "chapter_count": len(final.chapters)},
    )
    pricing = summarize_usage_files([raw_path], relative_to=args.output_dir)
    write_json(args.output_dir / "pricing.json", pricing)
    print(
        json.dumps(
            {
                "shortlist_path": str(plan_path.resolve()),
                "chapter_count": len(final.chapters),
                "candidate_count": sum(
                    len(chapter.candidates) for chapter in final.chapters
                ),
                "pricing": pricing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
