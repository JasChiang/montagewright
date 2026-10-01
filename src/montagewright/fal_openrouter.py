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
import tempfile
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


# Measured 2026-09-30 through this route: a 14 MB video is read correctly, a
# 26 MB one is refused with INVALID_ARGUMENT -- fal inlines each file, so
# Google's inline ceiling applies per file. Two 11 MB files in one request
# were both read; six (68 MB) timed out. Stay well inside the per-file line.
FAL_VIDEO_MAX_BYTES = 14_000_000
# Inline bytes count against the request body: OpenRouter refuses a body
# over 20,000,000 bytes for Google AI Studio with HTTP 413 (measured
# 2026-09-30 on a stringout plus music). Media is base64 here, so it is the
# encoded size that has to fit.
FAL_REQUEST_MAX_BYTES = 19_500_000
# Gemini reads audio at a fixed 32 tokens a second whatever the bitrate, so
# a music bed re-encoded as small mono costs the model nothing and frees
# room in the body for pictures.
FAL_AUDIO_MAX_BYTES = 1_500_000


def _duration_seconds(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def _fal_sized(path: Path, max_bytes: int = FAL_VIDEO_MAX_BYTES) -> Path:
    """The same timeline in a file this route will accept.

    Only the encoding changes: resolution, frame rate and bitrate. Every
    timestamp the model reads is still the source's, so MM:SS answers mean
    the same thing on either file. Content-addressed and kept, so a stringout
    is shrunk once however many calls carry it.

    `max_bytes` is lowered by the request when many videos travel together
    and their sum, not any one of them, is what does not fit.
    """

    max_bytes = min(int(max_bytes), FAL_VIDEO_MAX_BYTES)
    if path.stat().st_size <= max_bytes:
        return path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    kept_dir = Path(tempfile.gettempdir()) / "montagewright-fal-media"
    kept_dir.mkdir(parents=True, exist_ok=True)
    kept = kept_dir / f"sized-{digest.hexdigest()}-{max_bytes}.mp4"
    if kept.exists() and kept.stat().st_size <= max_bytes:
        return kept
    seconds = max(1.0, _duration_seconds(path))
    budget_kbps = max_bytes * 8 * 0.92 / seconds / 1000
    # Static reads one frame a second. A long reel spends its few bits on
    # sharper frames rather than frames nobody samples -- the burned source
    # ids and clock on a stringout have to stay legible.
    fps = 10 if seconds <= 120 else 2
    audio_kbps = 32 if budget_kbps >= 160 else 16
    for share in (1.0, 0.6, 0.35):
        video_kbps = max(12, int(budget_kbps * share) - audio_kbps)
        staged = kept_dir / f".sized-{digest.hexdigest()}.{os.getpid()}.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(path),
             "-vf", f"scale='min(640,iw)':-2,fps={fps}",
             "-c:v", "libx264", "-preset", "veryfast",
             "-b:v", f"{video_kbps}k", "-maxrate", f"{video_kbps}k",
             "-bufsize", f"{video_kbps * 2}k", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ac", "1", "-b:a", f"{audio_kbps}k",
             "-movflags", "+faststart", str(staged)],
            check=True,
        )
        if staged.stat().st_size <= max_bytes:
            staged.replace(kept)
            return kept
        staged.unlink(missing_ok=True)
    raise ValueError(
        f"{path.name} ({seconds:.0f}s) cannot be encoded under "
        f"{max_bytes / 1_000_000:.2f} MB for fal; split it before dispatch"
    )


def _fal_sized_audio(path: Path) -> Path:
    """The same recording as small mono MP3, same clock, kept by hash."""

    if path.stat().st_size <= FAL_AUDIO_MAX_BYTES:
        return path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    kept_dir = Path(tempfile.gettempdir()) / "montagewright-fal-media"
    kept_dir.mkdir(parents=True, exist_ok=True)
    kept = kept_dir / f"audio-{digest.hexdigest()}.mp3"
    if kept.exists() and kept.stat().st_size <= FAL_AUDIO_MAX_BYTES:
        return kept
    seconds = max(1.0, _duration_seconds(path))
    kbps = max(24, min(64, int(FAL_AUDIO_MAX_BYTES * 8 * 0.9 / seconds / 1000)))
    staged = kept_dir / f".audio-{digest.hexdigest()}.{os.getpid()}.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
         "-vn", "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{kbps}k",
         str(staged)],
        check=True,
    )
    staged.replace(kept)
    return kept


def _media_data(uri: str, mime_type: str) -> str:
    if uri.startswith("data:"):
        return uri
    if uri.startswith(("https://", "http://")):
        return uri
    path = _local_path(uri)
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _content(value, video_share: float = 1.0):
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
                if mode == "agentic":
                    # Measured on this route: agentic either returned 503
                    # "high demand" after 82 s or answered HTTP 200 with zero
                    # input tokens and an empty result -- still billed. Static
                    # reads the same file reliably.
                    mode = "static"
                if uri.startswith("file:"):
                    local = _local_path(uri)
                    cap = (
                        FAL_VIDEO_MAX_BYTES if video_share >= 1.0
                        else max(40_000, int(local.stat().st_size * video_share))
                    )
                    uri = _fal_sized(local, cap).as_uri()
                video_url = {"url": _media_data(uri, mime)}
                if mode is not None:
                    video_url["processing"] = mode
                parts.append({"type": "video_url", "video_url": video_url})
            elif kind == "image":
                parts.append({"type": "image_url", "image_url": {"url": _media_data(uri, mime)}})
            else:
                if uri.startswith("file:"):
                    smaller = _fal_sized_audio(_local_path(uri))
                    if smaller.suffix == ".mp3" and smaller.name.startswith("audio-"):
                        mime = "audio/mpeg"
                    uri = smaller.as_uri()
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
        # Many videos in one request (a selection over dozens of proxies) can
        # each be small and still sum past the body limit. Re-encode all of
        # them by one common share so every source stays in the request on
        # its own clock, rather than dropping any.
        share = 1.0
        for _ in range(3):
            if len(encoded) <= FAL_REQUEST_MAX_BYTES:
                break
            video_bytes = sum(
                len(part["video_url"]["url"])
                for part in body["messages"][0]["content"]
                if isinstance(part, dict) and part.get("type") == "video_url"
            ) if isinstance(body["messages"][0]["content"], list) else 0
            if not video_bytes:
                break
            other = len(encoded) - video_bytes
            room = FAL_REQUEST_MAX_BYTES * 0.95 - other
            if room <= 0:
                break
            share = min(share, share * room / video_bytes * 0.9)
            body["messages"][0]["content"] = _content(
                request["input"], video_share=share
            )
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(encoded) > FAL_REQUEST_MAX_BYTES:
            raise ValueError(
                f"fal OpenRouter inline request is {len(encoded)} bytes; "
                f"above the {FAL_REQUEST_MAX_BYTES // 1_000_000} MB measured "
                "to be reliable on this route. Use shorter or smaller media "
                "before dispatch."
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
        carries_video = any(
            part.get("type") == "video_url"
            for part in body["messages"][0]["content"]
            if isinstance(part, dict)
        ) if isinstance(body["messages"][0]["content"], list) else False
        if carries_video and not int(usage.get("prompt_tokens") or 0):
            # Seen on this route: HTTP 200, a well-formed empty answer, zero
            # input tokens -- the video was never read. Parsed as-is it says
            # "nothing there", which is a verdict nobody made.
            raise RuntimeError(
                "fal OpenRouter answered a video request with zero input "
                "tokens; the media was not read, so the answer is not used"
            )
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
