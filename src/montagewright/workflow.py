"""User-facing artifacts shared by CLI, Web and resumed jobs."""
from pathlib import Path

from montagewright.checkpoints import write_json


def publish_proposal(output: Path, plan: dict, *, aspect: str, brief: str) -> None:
    shots = plan.get("shots") or []
    payload = {"status": "proposal_ready", "aspect": aspect,
               "brief": brief, "plan": plan}
    write_json(output / "proposal.json", payload)
    lines = ["# 剪輯方案", "", str(plan.get("direction", "")), "",
             f"比例：{aspect}；建議片長：{plan.get('target_seconds', '依素材')} 秒", "",
             "這是素材理解與剪輯提案；尚未完成裁切、追蹤與成片驗收。", ""]
    contract = plan.get("duration_contract") or {}
    if contract.get("mode") == "range":
        minimum, maximum = contract.get("minimum_seconds"), contract.get("maximum_seconds")
        lines.extend([f"交付片長範圍：{minimum}–{maximum} 秒。素材不足保留短版並說明缺口，短版不算符合交付。", ""])
    for i, shot in enumerate(shots, 1):
        source = shot.get("source_id") or str(shot.get("span_id", "")).rsplit(":", 1)[0]
        lines.extend([f"## {i}. {source}", "",
                      f"來源區段：{shot.get('span_id', '')}；區段內進點：{shot.get('start_offset_seconds', shot.get('start_seconds', '0:00'))}；"
                      f"長度：{shot.get('seconds_needed', '')} 秒", "",
                      str(shot.get("why", "")), ""])
    (output / "proposal.md").write_text("\n".join(lines), encoding="utf-8")


def publish_coverage(output: Path, proxies: dict, cards: dict) -> dict:
    from montagewright.clipcard import load_card, clip_seconds
    rows = []
    for source_id, path in proxies.items():
        card = load_card(cards[source_id]) if source_id in cards else None
        rows.append({"source_id": source_id, "duration_seconds": clip_seconds(path),
                     "status": "analyzed" if card else "missing",
                     "evidence": (card or {}).get("inspection", {}),
                     "card": str(cards[source_id]) if card else None})
    payload = {"version": 1, "sources": rows,
               "complete": all(r["status"] == "analyzed" for r in rows),
               "scope": "full source sampled video and audio; not frame-perfect semantic recall"}
    write_json(output / "work" / "material-coverage.json", payload)
    return payload
