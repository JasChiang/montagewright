from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from bisect import bisect_left
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image

from .models import ExtractedFrame, MediaInfo, Rational, VideoStreamInfo


class MediaCommandError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise MediaCommandError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr.strip()}"
        )
    return completed


def _rational(value: str | None) -> Rational | None:
    if not value or value in {"0/0", "0:0", "N/A"}:
        return None
    try:
        fraction = Fraction(value.replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        return None
    if fraction <= 0:
        return None
    return Rational(numerator=fraction.numerator, denominator=fraction.denominator)


def _rotation(stream: dict[str, object]) -> int:
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and "rotate" in tags:
        try:
            return int(float(str(tags["rotate"]))) % 360
        except ValueError:
            pass
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and "rotation" in side_data:
            return int(float(str(side_data["rotation"]))) % 360
    return 0


def probe_video(path: Path) -> MediaInfo:
    source = path.expanduser().resolve(strict=True)
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ]
    )
    payload = json.loads(result.stdout)
    streams = [stream for stream in payload["streams"] if stream.get("codec_type") == "video"]
    if not streams:
        raise MediaCommandError(f"no video stream found: {source}")
    stream = streams[0]
    rotation = _rotation(stream)
    coded_width = int(stream["width"])
    coded_height = int(stream["height"])
    sample_aspect_ratio = _rational(stream.get("sample_aspect_ratio")) or Rational(
        numerator=1,
        denominator=1,
    )
    display_sample_aspect_ratio = (
        Rational(
            numerator=sample_aspect_ratio.denominator,
            denominator=sample_aspect_ratio.numerator,
        )
        if rotation in {90, 270}
        else sample_aspect_ratio
    )
    display_width, display_height = (
        (coded_height, coded_width) if rotation in {90, 270} else (coded_width, coded_height)
    )
    time_base = _rational(stream.get("time_base"))
    if time_base is None:
        raise MediaCommandError("video stream has no usable time_base")
    format_info = payload.get("format", {})
    duration_s = format_info.get("duration") or stream.get("duration")
    if duration_s is None:
        raise MediaCommandError("video has no duration")
    file_hash = sha256_file(source)
    return MediaInfo(
        path=str(source),
        sha256=file_hash,
        asset_id=f"sha256:{file_hash}",
        format_name=format_info.get("format_name"),
        duration_ms=round(float(duration_s) * 1000),
        size_bytes=int(format_info.get("size") or source.stat().st_size),
        format_metadata={str(k): str(v) for k, v in (format_info.get("tags") or {}).items()},
        video=VideoStreamInfo(
            index=int(stream["index"]),
            codec_name=stream.get("codec_name"),
            coded_width=coded_width,
            coded_height=coded_height,
            display_width=display_width,
            display_height=display_height,
            rotation_degrees=rotation,
            sample_aspect_ratio=sample_aspect_ratio,
            display_sample_aspect_ratio=display_sample_aspect_ratio,
            average_frame_rate=_rational(stream.get("avg_frame_rate")),
            real_frame_rate=_rational(stream.get("r_frame_rate")),
            time_base=time_base,
            start_pts=int(stream["start_pts"]) if stream.get("start_pts") is not None else None,
            duration_ts=int(stream["duration_ts"]) if stream.get("duration_ts") is not None else None,
            metadata={str(k): str(v) for k, v in (stream.get("tags") or {}).items()},
        ),
    )


def has_audio_stream(path: Path) -> bool:
    """Return whether the media contains at least one audio stream."""
    source = path.expanduser().resolve(strict=True)
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(source),
        ]
    )
    return bool(json.loads(result.stdout).get("streams"))


_SHOWINFO_RE = re.compile(r"pts:\s*(?P<pts>-?\d+)\s+pts_time:(?P<time>-?[0-9.]+)")


@lru_cache(maxsize=128)
def _cached_video_stream_timing(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
) -> tuple[int, Fraction]:
    """Read stream origin/time base without re-probing unchanged files per frame."""
    del size_bytes, mtime_ns
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_pts,time_base",
            "-of",
            "json",
            resolved_path,
        ]
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise MediaCommandError(f"no video stream found: {resolved_path}")
    stream = streams[0]
    return int(stream.get("start_pts") or 0), Fraction(stream["time_base"])


def extract_frame(
    source: Path,
    requested_time_ms: int,
    output: Path,
    *,
    max_width: int | None = None,
) -> ExtractedFrame:
    if requested_time_ms < 0:
        raise ValueError("requested_time_ms must be non-negative")
    resolved_source = source.expanduser().resolve(strict=True)
    stat = resolved_source.stat()
    source_start_pts, source_time_base = _cached_video_stream_timing(
        str(resolved_source),
        stat.st_size,
        stat.st_mtime_ns,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    absolute_seconds = (
        Fraction(source_start_pts) * source_time_base
        + Fraction(requested_time_ms, 1000)
    )
    # select runs after FFmpeg's default orientation correction. showinfo records
    # the exact chosen source PTS rather than pretending the semantic time is exact.
    if max_width is not None and max_width < 64:
        raise ValueError("max_width must be at least 64 when provided")
    filters = [f"select=gte(t\\,{float(absolute_seconds):.9f})"]
    if max_width is not None:
        filters.append(f"scale='min({max_width},iw)':-2")
    filters.append("showinfo")
    filter_graph = ",".join(filters)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-copyts",
        "-i",
        str(resolved_source),
        "-map",
        "0:v:0",
        "-vf",
        filter_graph,
        "-fps_mode",
        "vfr",
        "-frames:v",
        "1",
        "-y",
        str(output),
    ]
    try:
        completed = _run(command)
        match = _SHOWINFO_RE.search(completed.stderr)
        if not match:
            raise MediaCommandError(
                "could not parse selected frame PTS from ffmpeg showinfo"
            )
    except MediaCommandError as original_error:
        # Container duration and edit-list endpoints are frequently a few
        # milliseconds later than the final decodable frame.  A semantic
        # request in that final sub-frame interval has no later frame for
        # select=gte(t), so use the authoritative last decoded PTS.  Keep the
        # tolerance narrow: corruption or a materially out-of-range request
        # must still fail instead of being disguised as a valid selection.
        last_pts = last_decoded_video_frame_pts(resolved_source)
        last_time_ms = round(
            Fraction(last_pts - source_start_pts)
            * source_time_base
            * 1000
        )
        eof_delta_ms = requested_time_ms - last_time_ms
        if not 0 <= eof_delta_ms <= 250:
            raise original_error
        fallback = extract_frame_at_pts(
            resolved_source,
            last_pts,
            output,
            max_width=max_width,
        )
        return fallback.model_copy(
            update={"requested_time_ms": requested_time_ms}
        )
    frame_pts = int(match.group("pts"))
    frame_time_ms = round(
        Fraction(frame_pts - source_start_pts) * source_time_base * 1000
    )
    if frame_time_ms < 0:
        raise MediaCommandError("selected frame precedes the video stream start PTS")
    with Image.open(output) as image:
        width, height = image.size
    frame_hash = sha256_file(output)
    return ExtractedFrame(
        path=str(output.resolve()),
        requested_time_ms=requested_time_ms,
        frame_time_ms=frame_time_ms,
        frame_pts=frame_pts,
        frame_hash=frame_hash,
        width=width,
        height=height,
    )


def extract_frames_bounded(
    source: Path,
    requested_times_ms: Sequence[int],
    output_dir: Path,
    *,
    window_start_ms: int,
    window_end_ms: int,
    max_width: int | None = None,
    filename_prefix: str = "frame",
) -> tuple[ExtractedFrame, ...]:
    """Decode one bounded source interval and resolve many semantic times.

    Every returned frame is the first decoded source frame at or after its
    requested local playback time.  FFmpeg decodes the interval once, keeps
    the source stream PTS through ``-copyts``, and applies its normal display
    orientation before optional scaling.  Requests in the final sub-frame
    interval retain :func:`extract_frame`'s narrowly bounded EOF behavior.

    The temporary sequential decode is intentionally separate from the final
    per-request files.  Multiple requests may resolve to one immutable PTS,
    but each returned artifact still records its own semantic request.
    """

    if window_start_ms < 0 or window_end_ms <= window_start_ms:
        raise ValueError("bounded frame extraction window must be non-empty")
    requests = tuple(requested_times_ms)
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in requests
    ):
        raise ValueError("bounded frame requests must be integer milliseconds")
    if any(
        value < window_start_ms or value >= window_end_ms
        for value in requests
    ):
        raise ValueError("bounded frame request is outside the decode window")
    if max_width is not None and max_width < 64:
        raise ValueError("max_width must be at least 64 when provided")
    if not filename_prefix or any(
        character in filename_prefix for character in ("/", "\\", "\0")
    ):
        raise ValueError("filename_prefix must be one safe path component")
    if not requests:
        return ()

    resolved_source = source.expanduser().resolve(strict=True)
    stat = resolved_source.stat()
    source_start_pts, source_time_base = _cached_video_stream_timing(
        str(resolved_source),
        stat.st_size,
        stat.st_mtime_ns,
    )
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    absolute_start = (
        Fraction(source_start_pts) * source_time_base
        + Fraction(window_start_ms, 1000)
    )
    absolute_end = (
        Fraction(source_start_pts) * source_time_base
        + Fraction(window_end_ms, 1000)
    )
    filters = [
        "select="
        f"gte(t\\,{float(absolute_start):.9f})*"
        # Keep a frame exactly on the bounded endpoint available as the
        # first-frame-at-or-after answer for a near-tail request.  The legacy
        # single-frame extractor can select that frame too; excluding it
        # changes sparse/VFR evidence to the previous frame.
        f"lte(t\\,{float(absolute_end):.9f})"
    ]
    if max_width is not None:
        filters.append(f"scale='min({max_width},iw)':-2")
    filters.append("showinfo")

    with tempfile.TemporaryDirectory(
        prefix=".bounded-frame-decode-",
        dir=resolved_output_dir,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        output_pattern = temporary_dir / "decoded-%06d.jpg"
        completed = _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "info",
                "-copyts",
                # Input -ss/-to are local playback offsets. Accurate seeking
                # decodes from the preceding keyframe while -copyts keeps the
                # original source PTS visible to select/showinfo.
                "-ss",
                f"{window_start_ms / 1000:.9f}",
                "-to",
                # Input -to is exclusive on some containers. Decode only a
                # narrow extra tail so a source frame exactly at the logical
                # endpoint reaches the inclusive select predicate above.
                f"{(window_end_ms + 250) / 1000:.9f}",
                "-i",
                str(resolved_source),
                "-map",
                "0:v:0",
                "-vf",
                ",".join(filters),
                "-fps_mode",
                "vfr",
                "-start_number",
                "0",
                "-y",
                str(output_pattern),
            ]
        )
        decoded_matches = tuple(_SHOWINFO_RE.finditer(completed.stderr))
        decoded_paths = tuple(sorted(temporary_dir.glob("decoded-*.jpg")))
        if not decoded_matches or len(decoded_matches) != len(decoded_paths):
            raise MediaCommandError(
                "bounded frame decode did not produce a complete PTS/file mapping"
            )

        decoded: list[tuple[int, Fraction, int, Path]] = []
        for match, decoded_path in zip(
            decoded_matches,
            decoded_paths,
            strict=True,
        ):
            frame_pts = int(match.group("pts"))
            local_time = (
                Fraction(frame_pts - source_start_pts) * source_time_base * 1000
            )
            if local_time < 0:
                raise MediaCommandError(
                    "bounded frame decode produced a frame before stream start"
                )
            decoded.append(
                (frame_pts, local_time, round(local_time), decoded_path)
            )
        if any(
            after[1] <= before[1]
            for before, after in zip(decoded, decoded[1:])
        ):
            raise MediaCommandError(
                "bounded frame decode produced non-monotonic source PTS"
            )

        exact_times = [item[1] for item in decoded]
        extracted: list[ExtractedFrame] = []
        for request_index, requested_time_ms in enumerate(requests):
            decoded_index = bisect_left(
                exact_times,
                Fraction(requested_time_ms, 1),
            )
            if decoded_index >= len(decoded):
                last = decoded[-1]
                eof_delta_ms = requested_time_ms - last[2]
                if not 0 <= eof_delta_ms <= 250:
                    raise MediaCommandError(
                        "bounded frame decode has no frame at or after request"
                    )
                chosen = last
            else:
                chosen = decoded[decoded_index]
            frame_pts, _exact_time, frame_time_ms, decoded_path = chosen
            output_path = (
                resolved_output_dir
                / f"{filename_prefix}-{request_index:03d}.jpg"
            )
            shutil.copyfile(decoded_path, output_path)
            with Image.open(output_path) as image:
                width, height = image.size
            extracted.append(
                ExtractedFrame(
                    path=str(output_path.resolve()),
                    requested_time_ms=requested_time_ms,
                    frame_time_ms=frame_time_ms,
                    frame_pts=frame_pts,
                    frame_hash=sha256_file(output_path),
                    width=width,
                    height=height,
                )
            )
    return tuple(extracted)


def extract_frame_at_pts(
    source: Path,
    frame_pts: int,
    output: Path,
    *,
    max_width: int | None = None,
) -> ExtractedFrame:
    """Extract one exact decoded source frame by immutable stream PTS.

    This is the authoritative path for semantic checkpoints and render-boundary
    evidence.  Milliseconds remain a derived display value and are never used to
    re-select the requested frame.
    """

    resolved_source = source.expanduser().resolve(strict=True)
    stat = resolved_source.stat()
    source_start_pts, source_time_base = _cached_video_stream_timing(
        str(resolved_source),
        stat.st_size,
        stat.st_mtime_ns,
    )
    if frame_pts < source_start_pts:
        raise ValueError("frame_pts precedes the video stream start PTS")
    if max_width is not None and max_width < 64:
        raise ValueError("max_width must be at least 64 when provided")
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"select=eq(pts\\,{frame_pts})"]
    if max_width is not None:
        filters.append(f"scale='min({max_width},iw)':-2")
    filters.append("showinfo")
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-copyts",
            "-i",
            str(resolved_source),
            "-map",
            "0:v:0",
            "-vf",
            ",".join(filters),
            "-fps_mode",
            "vfr",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ]
    )
    match = _SHOWINFO_RE.search(completed.stderr)
    if not match:
        raise MediaCommandError(
            f"could not decode requested source PTS {frame_pts}"
        )
    decoded_pts = int(match.group("pts"))
    if decoded_pts != frame_pts:
        raise MediaCommandError(
            f"decoded frame PTS {decoded_pts} differs from requested {frame_pts}"
        )
    frame_time_ms = round(
        Fraction(decoded_pts - source_start_pts) * source_time_base * 1000
    )
    with Image.open(output) as image:
        width, height = image.size
    return ExtractedFrame(
        path=str(output.resolve()),
        requested_time_ms=frame_time_ms,
        frame_time_ms=frame_time_ms,
        frame_pts=decoded_pts,
        frame_hash=sha256_file(output),
        width=width,
        height=height,
    )


def last_decoded_video_frame_pts(source: Path) -> int:
    """Return the last decoded video-frame PTS without guessing from duration.

    Container duration is often rounded and may point just beyond the final
    frame, especially after concat or with variable frame rates. Enumerating
    best-effort timestamps is slower than a duration seek but authoritative for
    the short rendered segments whose lineage we verify.
    """

    resolved_source = source.expanduser().resolve(strict=True)
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp",
            "-of",
            "json",
            str(resolved_source),
        ]
    )
    payload = json.loads(completed.stdout)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise MediaCommandError("ffprobe omitted decoded video frames")
    timestamps: list[int] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        value = frame.get("best_effort_timestamp")
        if value in {None, "N/A"}:
            continue
        try:
            timestamps.append(int(str(value)))
        except ValueError:
            continue
    if not timestamps:
        raise MediaCommandError("ffprobe omitted decoded video-frame PTS values")
    return timestamps[-1]


def create_analysis_proxy(
    source: Path,
    output: Path,
    *,
    max_side: int = 1920,
    fps: int = 30,
    preserve_audio: bool = False,
    max_duration_delta_ms: int = 100,
) -> tuple[MediaInfo, dict[str, object]]:
    """Create a small orientation-corrected semantic-analysis proxy; geometry stays on source."""
    if max_side < 320 or fps < 1:
        raise ValueError("analysis proxy max_side and fps must be positive practical values")
    source_media = probe_video(source)
    source_has_audio = has_audio_stream(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source.expanduser().resolve(strict=True)),
            "-map",
            "0:v:0",
    ]
    if preserve_audio:
        command.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.extend(
        [
            "-vf",
            f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-t",
            f"{source_media.duration_ms / 1000:.3f}",
            "-y",
            str(output),
        ]
    )
    _run(command)
    proxy_media = probe_video(output)
    proxy_has_audio = has_audio_stream(output)
    duration_delta_ms = abs(proxy_media.duration_ms - source_media.duration_ms)
    if duration_delta_ms > max_duration_delta_ms:
        raise MediaCommandError(
            f"analysis proxy duration differs by {duration_delta_ms} ms; "
            f"maximum is {max_duration_delta_ms} ms"
        )
    record = {
        "purpose": "Gemini semantic analysis only; original source remains geometry authority",
        "source_asset_id": source_media.asset_id,
        "proxy_asset_id": proxy_media.asset_id,
        "duration_delta_ms": duration_delta_ms,
        "max_side": max_side,
        "fps": fps,
        "preserve_audio": preserve_audio,
        "source_has_audio": source_has_audio,
        "proxy_has_audio": proxy_has_audio,
        "original_bytes": source_media.size_bytes,
        "proxy_bytes": proxy_media.size_bytes,
        "byte_reduction_ratio": round(1 - proxy_media.size_bytes / source_media.size_bytes, 8),
    }
    return proxy_media, record
