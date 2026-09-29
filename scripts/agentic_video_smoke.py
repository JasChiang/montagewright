"""Run exactly one paid Agentic Video interaction and retain its proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from montagewright.cost import Ledger
from montagewright.environment import load_project_env
from montagewright.gemini import video_content
from montagewright.planner import MODEL_ID, Usage, _http_options, ask


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    dumped = getattr(value, "model_dump", None)
    if callable(dumped):
        return _plain(dumped(mode="json", exclude_none=True))
    return str(value)


def _step_type(step: Any) -> str:
    kind = (
        step.get("type")
        if isinstance(step, dict)
        else getattr(step, "type", "")
    )
    return str(getattr(kind, "value", kind) or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/agentic-smoke"),
    )
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        parser.error(f"video does not exist: {video}")
    if args.model != "gemini-3.8-flash":
        parser.error("the reproducible smoke test requires gemini-3.8-flash")

    load_project_env()
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    client = genai.Client(api_key=key, http_options=_http_options(types))

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = args.output / f"{stamp}-{video.stem}.json"
    ledger = Ledger(
        cap_usd=args.budget,
        model_id=args.model,
        journal_path=args.output / "spend-events.jsonl",
    )

    uploaded = client.files.upload(file=video)
    while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    state = getattr(uploaded.state, "name", str(uploaded.state))
    if state != "ACTIVE":
        raise RuntimeError(f"video upload ended in state {state}")

    interaction = ask(
        client,
        model=args.model,
        store=False,
        input=[
            video_content(
                str(uploaded.uri),
                mime_type=str(uploaded.mime_type or "video/mp4"),
                resolution="low",
                processing="agentic",
            ),
            {
                "type": "text",
                "text": (
                    "Inspect the full clip and report the exact sequence of visible "
                    "events, including any brief UI transition. Cite approximate "
                    "timestamps and say what evidence you inspected."
                ),
            },
        ],
        generation_config={"thinking_level": "high", "max_output_tokens": 2048},
        ledger=ledger,
        budget_stage="agentic_smoke",
        max_attempts=1,
    )
    usage = Usage.from_interaction(interaction)
    steps = tuple(getattr(interaction, "steps", None) or ())
    step_types = [_step_type(step) for step in steps]
    verified = usage.processing_calls > 0 and usage.processing_results > 0
    payload = {
        "schema_version": "agentic-video-smoke-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "video": {
            "name": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "bytes": video.stat().st_size,
        },
        "agentic_verified": verified,
        "step_types": step_types,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "thought_tokens": usage.thought_tokens,
            "tool_use_tokens": usage.tool_use_tokens,
            "processing_calls": usage.processing_calls,
            "processing_results": usage.processing_results,
        },
        "known_spend": ledger.summary(),
        "output_text": str(getattr(interaction, "output_text", "") or ""),
        "steps": [_plain(step) for step in steps],
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": str(result_path),
        "agentic_verified": verified,
        "step_types": step_types,
        "usage": payload["usage"],
        "known_spend": payload["known_spend"],
    }, ensure_ascii=False, indent=2))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
