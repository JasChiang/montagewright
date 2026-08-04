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

# Extra material kept either side of each cut, never shown. A transition needs
# two shots to overlap, and an exact cut leaves nothing to overlap with; a
# nudge of a quarter of a second later needs frames that were never rendered.
# Editors call these handles and always cut them.
HANDLE_SECONDS = 0.5


class RenderError(RuntimeError):
    """ffmpeg refused, and the command plus its stderr are attached."""


@dataclass(frozen=True)
class Handles:
    """How much spare material a segment carries, and where."""

    head_seconds: float
    tail_seconds: float


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
) -> tuple[Path, "Handles"]:
    """Cut one shot, cropping if the plan asked for it."""

    filters: list[str] = []
    source = segment.source
    if segment.crop_path is not None and not segment.crop_path.is_static:
        # A following camera. The x expression is evaluated per frame, so the
        # motion lives in the same filter as the crop rather than in a
        # separate command stream.
        from jascue_auto.reframe import ffmpeg_crop_expression

        w_expr, h_expr, x_expr, y_expr = ffmpeg_crop_expression(
            segment.crop_path, source.width, source.height
        )
        filters.append(
            f"crop=w='{w_expr}':h='{h_expr}':x='{x_expr}':y='{y_expr}'"
        )
        # A zoom changes the crop size per frame, so the output has to be
        # pinned to one resolution or the encoder sees a stream that changes
        # shape mid-shot.
        out_w, out_h = segment.crop_path.keyframes[0].crop.to_pixels(
            source.width, source.height
        )[2:]
        filters.append(f"scale={out_w}:{out_h}")
    elif segment.crop is not None:
        x, y, width, height = segment.crop.to_pixels(source.width, source.height)
        filters.append(f"crop={width}:{height}:{x}:{y}")

    # The delivered segment is cut exactly. Handles are written alongside it
    # as their own file, so a transition or a nudge has material without the
    # timeline paying for it -- the concat stays frame-exact because every
    # segment is already the length it is meant to be.
    head = min(HANDLE_SECONDS, segment.in_seconds)
    tail = min(
        HANDLE_SECONDS,
        max(0.0, source.duration_seconds - segment.out_seconds),
    )
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

    if head > 0.0 or tail > 0.0:
        spare = destination.with_name(f"{destination.stem}.handles.mp4")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{segment.in_seconds - head:.6f}",
                "-to", f"{segment.out_seconds + tail:.6f}",
                "-i", str(segment.source.path),
            ]
            + (["-vf", ",".join(filters)] if filters else [])
            + [
                "-c:v", video_encoder, "-b:v", "12M",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p", str(spare),
            ]
        )
    return destination, Handles(head_seconds=head, tail_seconds=tail)


def _concat(
    segment_paths: list[tuple[Path, Handles, float]],
    destination: Path,
    work_dir: Path,
) -> Path:
    """Join the segments without touching the streams again.

    The segments already share a codec, pixel format, and sample rate because
    this module wrote all of them, so a stream copy is exact. Re-encoding here
    would be a second generation loss for no gain.

    Handles are trimmed at render time rather than here. Asking the concat
    demuxer for inpoint/outpoint on a copied stream lands on the nearest
    keyframe, which drifted a two-shot test by 124ms -- and every cut in this
    pipeline is placed against a measured beat, so drift that accumulates
    across a timeline is not a rounding detail, it is the alignment gone.
    """

    listing = work_dir / "concat.txt"
    listing.write_text(
        "".join(f"file '{path.resolve()}'\n" for path, _, _ in segment_paths),
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
        rendered, handles = _render_segment(
            segment, destination, video_encoder=video_encoder
        )
        segment_paths.append((rendered, handles, segment.duration_seconds))

    picture = _concat(segment_paths, output_dir / "picture.mp4", output_dir)
    kept_paths = [path for path, _, _ in segment_paths]

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
        kept_paths = []

    return RenderResult(
        deliverable=deliverable,
        preview=preview,
        segment_paths=tuple(kept_paths),
        duration_seconds=probe_duration(deliverable),
    )
