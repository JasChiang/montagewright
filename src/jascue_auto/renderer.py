"""Render a compiled plan with ffmpeg.

Segments are cut individually and then concatenated, rather than assembled in
one filter graph. A graph is marginally faster and much harder to debug: when
one shot is wrong you want to open that shot, not bisect a filter chain. The
segment files are also the natural cache boundary once only part of a cut
changes between review rounds.

Every render produces two files. The deliverable is full resolution; the
preview is small enough to send somewhere. The review loop reads the preview
by design -- judging a cut from a low-resolution copy is what a director does
with a viewing link, and it keeps the reviewer's attention on the edit rather
than on grain.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jascue_auto.executor import RenderPlan, Segment

# Short-form platforms normalise to roughly this; matching it here means the
# cut sounds the same locally as it will after upload.
TARGET_LUFS = -14.0
# Headroom below full scale. Lossy re-encoding on the way to a platform adds
# its own overshoot, so a master that already touches 0 dBFS clips there.
TRUE_PEAK_CEILING_DB = -1.5
# alimiter takes a linear ceiling, not decibels. Passing "-1.5dB" is accepted
# and silently clamped, which is how a cut measured at +0.017 dBFS came out of
# a filter chain that asked for -1.5.
TRUE_PEAK_CEILING_LINEAR = 10 ** (TRUE_PEAK_CEILING_DB / 20)
PREVIEW_HEIGHT = 640


class RenderError(RuntimeError):
    """ffmpeg refused, and the command plus its stderr are attached."""


@dataclass(frozen=True)
class RenderResult:
    deliverable: Path
    preview: Path
    segment_paths: tuple[Path, ...]
    duration_seconds: float

    def sizes_mb(self) -> dict[str, float]:
        return {
            "deliverable": self.deliverable.stat().st_size / 1_048_576,
            "preview": self.preview.stat().st_size / 1_048_576,
        }


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
        raise RenderError(
            f"ffmpeg failed ({completed.returncode})\n"
            f"  {' '.join(command[:12])} ...\n{tail}"
        )
    return completed


def _encoder(preferred: str, fallback: str) -> str:
    """Prefer the hardware encoder, but do not fail without it.

    VideoToolbox is the fast path on this hardware and absent everywhere else.
    A renderer that only works on one laptop is not much of a renderer.
    """

    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return preferred if preferred in probe.stdout else fallback


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RenderError(f"ffprobe could not read {path}")
    return float(json.loads(completed.stdout)["format"]["duration"])


def _render_segment(
    segment: Segment, destination: Path, *, video_encoder: str
) -> Path:
    """Cut one shot, cropping if the plan asked for it."""

    filters: list[str] = []
    if segment.crop is not None:
        source = segment.source
        x, y, width, height = segment.crop.to_pixels(source.width, source.height)
        filters.append(f"crop={width}:{height}:{x}:{y}")

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        # Seeking before -i decodes from the preceding keyframe, which is both
        # faster and frame-accurate here because the output is re-encoded.
        "-ss", f"{segment.in_seconds:.6f}",
        "-to", f"{segment.out_seconds:.6f}",
        "-i", str(segment.source.path),
    ]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += [
        "-c:v", video_encoder, "-b:v", "12M",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(destination),
    ]
    _run(command)
    return destination


def _concat(segment_paths: list[Path], destination: Path, work_dir: Path) -> Path:
    """Join the segments without touching the streams again.

    The segments already share a codec, pixel format, and sample rate because
    this module wrote all of them, so a stream copy is exact. Re-encoding here
    would be a second generation loss for no gain.
    """

    listing = work_dir / "concat.txt"
    listing.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(destination),
        ]
    )
    return destination


def _mux_music(
    picture: Path, music: Path, destination: Path, *, video_encoder: str
) -> Path:
    """Lay a music bed under the cut and normalise the result.

    The music is trimmed to the picture, never the other way round: stretching
    a track to fit a cut is audible, and holding the picture to fit the track
    means padding it with something nobody chose.
    """

    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(picture), "-i", str(music),
            "-filter_complex",
            f"[1:a]atrim=0:{probe_duration(picture):.6f},"
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_CEILING_DB}:LRA=11,"
            # loudnorm in one pass predicts its true peak rather than
            # measuring it, and overshoots often enough to matter: this cut
            # came back at +0.017 dBFS against a -1.5 request. A limiter after
            # it holds the ceiling for real, which is what stops a platform's
            # own re-encode from clipping what we sent.
            f"alimiter=limit={TRUE_PEAK_CEILING_LINEAR:.6f}:level=disabled"
            "[music]",
            "-map", "0:v:0", "-map", "[music]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(destination),
        ]
    )
    return destination


def _preview(source: Path, destination: Path, *, video_encoder: str) -> Path:
    """A copy small enough to send over a slow link."""

    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", f"scale=-2:{PREVIEW_HEIGHT}",
            "-c:v", video_encoder, "-b:v", "1200k",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(destination),
        ]
    )
    return destination


def render(
    plan: RenderPlan,
    output_dir: Path,
    *,
    music: Path | None = None,
    keep_segments: bool = True,
) -> RenderResult:
    """Render a plan to a deliverable and a preview.

    Raises only when ffmpeg itself fails. Whether the cut is any good is the
    review loop's question, and it needs a finished file to answer it.
    """

    if not plan.segments:
        raise RenderError("nothing to render: the plan has no segments")

    output_dir = output_dir.expanduser().resolve()
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    video_encoder = _encoder("h264_videotoolbox", "libx264")

    segment_paths: list[Path] = []
    for index, segment in enumerate(plan.segments):
        destination = segment_dir / f"{index:03d}-{segment.clip_id}.mp4"
        segment_paths.append(
            _render_segment(segment, destination, video_encoder=video_encoder)
        )

    picture = _concat(segment_paths, output_dir / "picture.mp4", output_dir)

    deliverable = output_dir / "deliverable.mp4"
    if music is not None:
        _mux_music(
            picture, music, deliverable, video_encoder=video_encoder
        )
    else:
        shutil.copyfile(picture, deliverable)

    preview = _preview(
        deliverable, output_dir / "preview.mp4", video_encoder=video_encoder
    )

    if not keep_segments:
        shutil.rmtree(segment_dir, ignore_errors=True)
        segment_paths = []

    return RenderResult(
        deliverable=deliverable,
        preview=preview,
        segment_paths=tuple(segment_paths),
        duration_seconds=probe_duration(deliverable),
    )
