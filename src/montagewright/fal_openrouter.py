"""OpenRouter Chat Completions through fal, behind the existing ask boundary.

OpenRouter forwards Gemini's agentic/static video mode, but reports agentic
navigation as encrypted reasoning rather than Interactions processing steps.
It does not expose the Interactions fixed-FPS/offset contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


BASE_URL = "https://fal.run/openrouter/router/openai/v1/chat/completions"


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("fal media must be a local file URI")
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise FileNotFoundError(f"fal media is no longer present: {path}")
    return path


def _media_data(uri: str, mime_type: str) -> str:
    if uri.startswith("data:"):
        return uri
    if uri.startswith(("https://", "http://")):
        return uri
    path = _local_path(uri)
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _content(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("fal OpenRouter input must be text or a list of parts")
    parts = []
    for part in value:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in {"video", "image", "audio"}:
            uri = str(part.get("uri") or "")
            mime = str(part.get("mime_type") or mimetypes.guess_type(uri)[0] or "application/octet-stream")
            if not uri:
                raise ValueError(f"{kind} part has no URI")
            if kind == "video":
                source_processing = part.get("processing")
                if isinstance(source_processing, dict):
                    if source_processing.get("type") != "static":
                        raise ValueError("fal OpenRouter video processing must be agentic or static")
                    if source_processing.get("start_offset") or source_processing.get("end_offset"):
                        raise ValueError("fal OpenRouter cannot preserve video start/end offsets")
                    mode = "static"
                else:
                    mode = source_processing
                if mode not in {None, "agentic", "static"}:
                    raise ValueError(f"unsupported fal OpenRouter video processing {mode!r}")
                video_url = {"url": _media_data(uri, mime)}
                if mode is not None:
                    video_url["processing"] = mode
                parts.append({"type": "video_url", "video_url": video_url})
            elif kind == "image":
                parts.append({"type": "image_url", "image_url": {"url": _media_data(uri, mime)}})
            else:
                encoded = _media_data(uri, mime)
                if not encoded.startswith("data:"):
                    raise ValueError("fal OpenRouter audio requires local bytes or a data URI")
                fmt = {
                    "audio/mpeg": "mp3", "audio/mp3": "mp3",
                    "audio/mp4": "m4a", "audio/x-m4a": "m4a",
                    "audio/wav": "wav", "audio/x-wav": "wav",
                    "audio/flac": "flac", "audio/ogg": "ogg",
                    "audio/webm": "webm", "audio/aac": "aac",
                }.get(mime)
                if fmt is None:
                    raise ValueError(f"fal OpenRouter does not support audio MIME {mime!r}")
                parts.append({"type": "input_audio", "input_audio": {
                    "data": encoded.split(",", 1)[1], "format": fmt,
                }})
        else:
            raise TypeError(f"fal OpenRouter cannot translate content type {kind!r}")
    return parts


class _Files:
    """Keep local media addressable by the existing content-hash cache.

    Bytes are placed in the fal request only at dispatch, after budget checks.
    No Google Files URI or Google credential is used.
    """

    def upload(self, *, file: str):
        return self.get(name=Path(file).resolve().as_uri())

    def get(self, *, name: str):
        path = _local_path(name)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return SimpleNamespace(
            name=name, uri=name, state=SimpleNamespace(name="ACTIVE"),
            size_bytes=path.stat().st_size,
            sha256_hash=base64.b64encode(digest.digest()).decode("ascii"),
        )


class _Interactions:
    def __init__(self, key: str):
        self.key = key

    def create(self, **request):
        if request.get("store") not in {None, False}:
            raise ValueError("fal OpenRouter does not support the Interactions store setting")
        model = str(request["model"])
        if model.startswith("gemini-"):
            model = "google/" + model
        if not model.startswith("google/gemini-"):
            raise ValueError(f"fal backend expects a Gemini model, got {model!r}")
        response_format = request.get("response_format")
        body = {
            "model": model,
            "messages": [{"role": "user", "content": _content(request["input"])}],
        }
        generation = request.get("generation_config") or {}
        if generation.get("max_output_tokens"):
            body["max_tokens"] = int(generation["max_output_tokens"])
        thinking = generation.get("thinking_level")
        if thinking is not None:
            if thinking not in {"minimal", "low", "medium", "high"}:
                raise ValueError(f"unsupported OpenRouter reasoning effort {thinking!r}")
            body["reasoning"] = {"effort": thinking}
        if response_format is not None:
            if response_format.get("type") != "text" or response_format.get("mime_type") != "application/json":
                raise ValueError("fal OpenRouter requires a JSON text response schema")
            body["response_format"] = {"type": "json_schema", "json_schema": {
                # Existing Gemini schemas contain optional properties. Keep
                # them optional; local parsers validate the returned JSON.
                "name": "montagewright_response", "strict": False,
                "schema": response_format["schema"],
            }}
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 19_000_000:
            raise ValueError(
                f"fal OpenRouter inline request is {len(encoded)} bytes; "
                "Google AI Studio routes accept at most 20 MB. "
                "Use shorter or smaller planning media before dispatch."
            )
        http_request = Request(BASE_URL, data=encoded, headers={
            "Authorization": f"Key {self.key}",
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(http_request, timeout=float(request.get("timeout") or 1500)) as response:
                answer = json.load(response)
        except HTTPError as error:
            detail = error.read(2048).decode("utf-8", "replace")
            raise RuntimeError(f"fal OpenRouter HTTP {error.code}: {detail}") from error
        choices = answer.get("choices") or []
        if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
            raise RuntimeError("fal OpenRouter returned no text choice")
        usage = answer.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        input_details = usage.get("prompt_tokens_details") or {}
        return SimpleNamespace(
            id=answer.get("id"),
            status="incomplete" if choices[0].get("finish_reason") == "length" else "completed",
            provider="fal_openrouter",
            output_text=choices[0]["message"]["content"], steps=[],
            reasoning_details=choices[0]["message"].get("reasoning_details") or [],
            usage={
                "total_input_tokens": int(usage.get("prompt_tokens") or 0),
                "total_output_tokens": int(usage.get("completion_tokens") or 0),
                "total_thought_tokens": 0,
                "total_cached_tokens": int(input_details.get("cached_tokens") or 0),
                "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                "provider_cost_usd": usage.get("cost"),
            },
        )


class FalOpenRouterClient:
    provider = "fal_openrouter"

    def __init__(self, key: str):
        if not key:
            raise ValueError("FAL_KEY is required")
        self.files = _Files()
        self.interactions = _Interactions(key)

    def estimate_request_tokens(self, *, model: str, input_value,
                                response_format=None) -> int:
        """Reserve before dispatch; fal has no countTokens endpoint.

        Media estimates deliberately overstate a one-FPS video read. Provider
        reported usage/cost replaces this estimate after a successful call.
        """
        parts = input_value if isinstance(input_value, list) else [input_value]
        tokens = 0
        for part in parts:
            if isinstance(part, str):
                tokens += math.ceil(len(part.encode("utf-8")) / 2)
                continue
            kind = part.get("type")
            if kind == "text":
                tokens += math.ceil(len(str(part.get("text", "")).encode("utf-8")) / 2)
                continue
            uri = str(part.get("uri") or "")
            path = _local_path(uri)
            if kind == "image":
                tokens += 2048
                continue
            if kind not in {"video", "audio"}:
                raise TypeError(f"cannot estimate fal input type {kind!r}")
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, check=True,
            )
            seconds = float(probe.stdout.strip())
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"cannot estimate media duration: {path}")
            if kind == "audio":
                tokens += math.ceil(seconds * 100)
            else:
                processing = part.get("processing")
                fps = float(processing.get("fps", 1)) if isinstance(processing, dict) else 4.0
                tokens += math.ceil(seconds * (300 * max(1.0, fps) + 100))
        if response_format is not None:
            tokens += math.ceil(len(json.dumps(response_format).encode("utf-8")) / 2)
        return math.ceil(tokens * 1.1) + 256


def client_from_env():
    from montagewright.environment import load_project_env

    load_project_env()
    key = os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError("FAL_KEY is required for the fal OpenRouter backend")
    return FalOpenRouterClient(key)
