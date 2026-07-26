#!/usr/bin/env python3
"""Run one cached, observation-only Gemini review of a final rendered edit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from jascue_video_lab.final_edit_qa import (
    execute_final_edit_qa,
    load_cached_final_edit_qa,
    prepare_final_edit_qa,
    upload_video_and_wait,
)
from jascue_video_lab.storage import utc_now, write_json


def _raw_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Gemini final-edit QA proposal. The command never modifies "
            "the supplied render."
        )
    )
    parser.add_argument(
        "mode",
        choices=("canonical_16x9", "crop_only_9x16"),
    )
    parser.add_argument("render", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--brief",
        type=Path,
        help="Required for canonical_16x9 and forbidden for crop_only_9x16.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("JASCUE_GEMINI_MODEL", "gemini-3.6-flash"),
    )
    parser.add_argument("--prompt", type=Path)
    parser.add_argument(
        "--crop-include-audio",
        action="store_true",
        help=(
            "Keep audio in the crop-only transport proxy. Crop QA still ignores "
            "music, pacing, and sound."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_final_edit_qa(
        mode=args.mode,
        render_path=args.render,
        manifest_path=args.manifest,
        brief_path=args.brief,
        output_dir=output_dir,
        model_id=args.model,
        prompt_override=args.prompt,
        crop_include_audio=args.crop_include_audio,
    )
    cached = load_cached_final_edit_qa(prepared, output_dir=output_dir)
    if cached is not None:
        print(
            json.dumps(
                {
                    "cache_hit": True,
                    "run_dir": str(cached.run_dir),
                    "validated_path": str(cached.run_dir / "validated.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required on a cache miss"
        )
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    try:
        uploaded = upload_video_and_wait(client, prepared.proxy_path)
        run_dir = output_dir / "runs" / prepared.cache_key
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / "file_upload.json",
            {
                "proxy_path": str(prepared.proxy_path),
                "proxy_sha256": prepared.input_hashes["proxy_sha256"],
                "uploaded_file": _raw_dump(uploaded),
                "recorded_at": utc_now(),
            },
        )
        execution = execute_final_edit_qa(
            prepared=prepared,
            client=client,
            uploaded_video=uploaded,
            output_dir=output_dir,
        )
    finally:
        client.close()

    print(
        json.dumps(
            {
                "cache_hit": False,
                "run_dir": str(execution.run_dir),
                "attempt_dir": str(execution.attempt_dir),
                "validated_path": str(execution.run_dir / "validated.json"),
                "pricing_path": str(execution.run_dir / "pricing.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
