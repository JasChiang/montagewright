"""Durable boundaries between paid inference and fallible local processing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse
from uuid import uuid4


def key_for(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str,
    ).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path):
    if not path.exists():
        return None
    # Corrupt paid evidence is not permission to silently pay again.
    return json.loads(path.read_text(encoding="utf-8"))


def response_path(ledger, stage: str, request: dict, upload_cache=None,
                  *, provider: str = "gemini_interactions") -> Path | None:
    journal = getattr(ledger, "journal_path", None)
    if journal is None:
        return None
    identities = {str(entry.get("uri")): digest
                  for digest, entry in getattr(upload_cache, "entries", {}).items()}
    def canonical(value):
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k == "uri" and isinstance(v, str):
                    if provider == "fal_openrouter" and v.startswith("file://"):
                        source = Path(unquote(urlparse(v).path))
                        digest = hashlib.sha256()
                        with source.open("rb") as handle:
                            for block in iter(lambda: handle.read(1 << 20), b""):
                                digest.update(block)
                        result[k] = "sha256:" + digest.hexdigest()
                        continue
                    if v in identities:
                        result[k] = "sha256:" + identities[v]
                        continue
                result[k] = canonical(v)
            return result
        if isinstance(value, list):
            return [canonical(v) for v in value]
        return value
    stable = canonical({k: v for k, v in request.items() if k != "timeout"})
    if provider != "gemini_interactions":
        stable = {"provider": provider, "request": stable}
    return Path(journal).parent / "work" / "responses" / (
        key_for({"stage": stage, "request": stable}) + ".json"
    )


def capture(interaction, stage: str, model: str) -> dict:
    from montagewright.planner import Usage
    usage = Usage.from_interaction(interaction)
    raw = getattr(interaction, "usage", None) or {}
    if not isinstance(raw, dict):
        raw = getattr(raw, "__dict__", {}) or {}
    status = getattr(interaction, "status", "completed")
    return {
        "status": getattr(status, "value", status),
        "output_text": getattr(interaction, "output_text", None),
        "provider_id": getattr(interaction, "id", None),
        "provider": getattr(interaction, "provider", "gemini_interactions"),
        "stage": stage, "model": model,
        "steps": [
            step.model_dump(mode="json", exclude_none=True) if hasattr(step, "model_dump")
            else json.loads(json.dumps(step if isinstance(step, dict) else vars(step), default=str))
            for step in getattr(interaction, "steps", None) or []
        ],
        "reasoning_details": getattr(interaction, "reasoning_details", None) or [],
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens + usage.thought_tokens,
            "cached_tokens": int(raw.get("total_cached_tokens") or 0),
            "tool_use_tokens": usage.tool_use_tokens,
            "processing_calls": usage.processing_calls,
            "processing_results": usage.processing_results,
            "provider_cost_usd": raw.get("provider_cost_usd"),
        },
    }


def settle_saved(ledger, path: Path, saved: dict) -> None:
    receipt = path.stem
    journal = ledger.journal_path
    if journal and Path(journal).exists():
        for line in Path(journal).read_text(encoding="utf-8").splitlines():
            if json.loads(line).get("response_id") == receipt:
                return
    ledger.record(saved["stage"], model_id=saved["model"],
                  response_id=receipt, **saved["usage"])


def replay(saved: dict):
    # Historical cost remains in the ledger, not in this invocation's usage.
    return SimpleNamespace(status=saved["status"], output_text=saved["output_text"],
                           id=saved.get("provider_id"), usage={}, steps=[],
                           reasoning_details=saved.get("reasoning_details", []),
                           saved_processing_steps=saved.get("steps", []),
                           checkpoint_reused=True)
