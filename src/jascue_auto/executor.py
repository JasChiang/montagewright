"""Turn an EDL into concrete render instructions.

This layer has no veto. Every clip in the plan produces a segment, always. When
something cannot be done as written it is done a rung lower and the reason is
recorded with the measurement that forced it -- but the segment still exists,
and the film is still deliverable.

That is a deliberate inversion. The previous system could decide a piece of
material was not worth using and abandon it, or abandon a whole aspect, and it
did: a run during verification dropped every 9:16 candidate because one
declared region wanted to be fully visible, and another dropped the 16:9
aspect because a frame held two similar handsets. Both times the semantic layer
had already said what it wanted and the execution layer said no. Only the
review loop gets to say no here, and it says it about a finished cut.

Degrading is not free either. Each rung down needs its own evidence that the
rung above was attempted and failed, because a layer permitted to skip to the
safe option takes it every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jascue_auto.schema import EDL, Clip, DegradationStep

# One safety margin, applied once, at the end. The old code added a little
# padding at detection, a little more at tracking, and more again at
# smoothing, so the crop that reached ffmpeg was tighter than any single layer
# intended and no one layer looked wrong.
CROP_MARGIN = 0.05


@dataclass(frozen=True)
class Source:
    """A resolved input file and the facts needed to place cuts in it."""

    source_id: str
    path: Path
    duration_seconds: float
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class CropBox:
    """A crop in normalised coordinates. Pixels happen at the ffmpeg edge."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not (0.0 < self.width <= 1.0 and 0.0 < self.height <= 1.0):
            raise ValueError("crop extent must sit within the frame")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("crop origin must sit within the frame")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Project onto a real frame.

        Chroma-subsampled encoders want even numbers. An origin of zero is a
        legitimate answer and stays zero; only the extent is held above zero,
        since a crop of no width is not a crop.
        """

        def even_origin(value: float) -> int:
            return max(0, int(value) // 2 * 2)

        def even_extent(value: float, limit: int) -> int:
            return max(2, min(limit, int(value) // 2 * 2))

        x = even_origin(self.x * width)
        y = even_origin(self.y * height)
        return (
            x,
            y,
            even_extent(self.width * width, width - x),
            even_extent(self.height * height, height - y),
        )


@dataclass
class Segment:
    """One rendered piece of the timeline."""

    clip_id: str
    source: Source
    in_seconds: float
    out_seconds: float
    crop: CropBox | None = None

    @property
    def duration_seconds(self) -> float:
        return self.out_seconds - self.in_seconds


@dataclass
class RenderPlan:
    """Everything the renderer needs, plus an honest account of the cost."""

    project_id: str
    segments: list[Segment]
    degradations: list[DegradationStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return sum(segment.duration_seconds for segment in self.segments)

    @property
    def degraded_clip_ids(self) -> set[str]:
        return {step.clip_id for step in self.degradations}


class MissingSource(KeyError):
    """Raised when an EDL names a source nobody supplied.

    This is the one refusal the executor is allowed. It is not a judgement
    about whether the material is good enough -- it is the plan referring to a
    file that does not exist, which no amount of degrading can render.
    """


def _centred_crop(source_aspect: float, target_aspect: float) -> CropBox:
    """The widest crop of `source_aspect` that fills `target_aspect`."""

    if target_aspect < source_aspect:
        width = target_aspect / source_aspect
        return CropBox(x=(1.0 - width) / 2.0, y=0.0, width=width, height=1.0)
    height = source_aspect / target_aspect
    return CropBox(x=0.0, y=(1.0 - height) / 2.0, width=1.0, height=height)


def _subject_crop(
    source_aspect: float, target_aspect: float, clip: Clip
) -> tuple[CropBox, str | None]:
    """Frame the stated subject, or say why the centre had to do.

    Only the subject's coarse position is available at this stage; tracking
    refines it later. A coarse anchor still beats a centred guess, because a
    subject the planner put on the left is not at the centre and cropping to
    the middle is how it leaves the frame.
    """

    base = _centred_crop(source_aspect, target_aspect)
    reframe = clip.reframe
    if reframe is None or reframe.subject is None:
        return base, "no subject stated; centred"

    horizontal, _, vertical = reframe.subject.coarse_position.partition("_")
    # Nine-box names read vertical-first: "mid_left" is mid vertically.
    bias = {"left": 0.0, "center": 0.5, "right": 1.0}.get(vertical, 0.5)
    top_bias = {"top": 0.0, "mid": 0.5, "bottom": 1.0}.get(horizontal, 0.5)

    free_x = 1.0 - base.width
    free_y = 1.0 - base.height
    # Pull toward the subject but keep a margin of the opposite side, so a
    # subject at the edge does not put the frame edge through it.
    x = min(max(bias * free_x, 0.0), free_x)
    y = min(max(top_bias * free_y, 0.0), free_y)
    if free_x > 0.0:
        x = min(max(x, free_x * CROP_MARGIN), free_x * (1.0 - CROP_MARGIN))
    if free_y > 0.0:
        y = min(max(y, free_y * CROP_MARGIN), free_y * (1.0 - CROP_MARGIN))
    return CropBox(x=x, y=y, width=base.width, height=base.height), None


def plan_render(
    edl: EDL,
    sources: dict[str, Source],
    *,
    target_aspect: float | None = None,
) -> RenderPlan:
    """Compile an EDL into segments. Never returns fewer than it was given.

    `target_aspect` is width over height; leave it out to keep each source as
    shot. The EDL's origin is irrelevant here -- a hand-written plan and a
    generated one compile identically, which is what makes a hand-written one
    useful for telling an execution bug apart from a planning one.
    """

    segments: list[Segment] = []
    degradations: list[DegradationStep] = []
    notes: list[str] = []

    for clip in edl.clips:
        source = sources.get(clip.source_id)
        if source is None:
            raise MissingSource(
                f"clip {clip.clip_id} names source {clip.source_id!r}, which "
                f"was not supplied. Known: {sorted(sources)}"
            )

        in_seconds, out_seconds = _resolve_times(
            clip, source, degradations, notes
        )
        crop = None
        if target_aspect is not None:
            crop = _resolve_crop(
                clip, source, target_aspect, degradations
            )
        segments.append(
            Segment(
                clip_id=clip.clip_id,
                source=source,
                in_seconds=in_seconds,
                out_seconds=out_seconds,
                crop=crop,
            )
        )

    assert len(segments) == len(edl.clips), "the executor never drops a clip"
    return RenderPlan(
        project_id=edl.project_id,
        segments=segments,
        degradations=degradations,
        notes=notes,
    )


def _resolve_times(
    clip: Clip,
    source: Source,
    degradations: list[DegradationStep],
    notes: list[str],
) -> tuple[float, float]:
    """Clamp a clip into its source without ever discarding it."""

    in_seconds = max(0.0, clip.approx_in_seconds)
    out_seconds = min(source.duration_seconds, clip.approx_out_seconds)

    if out_seconds <= in_seconds:
        # The window fell outside the material. Keep whatever tail exists
        # rather than dropping the beat: a short segment can be reviewed and
        # replanned, a missing one just leaves a hole nobody can see.
        in_seconds = max(0.0, min(in_seconds, source.duration_seconds - 0.1))
        out_seconds = source.duration_seconds
        degradations.append(
            DegradationStep(
                clip_id=clip.clip_id,
                ladder="other",
                ladder_other="trim_window_clamped_to_source",
                trigger=(
                    f"requested {clip.approx_in_seconds:.3f}.."
                    f"{clip.approx_out_seconds:.3f}s from a "
                    f"{source.duration_seconds:.3f}s source"
                ),
                measured={
                    "requested_in": clip.approx_in_seconds,
                    "requested_out": clip.approx_out_seconds,
                    "source_duration": source.duration_seconds,
                    "resolved_duration": out_seconds - in_seconds,
                },
            )
        )
    elif clip.approx_out_seconds > source.duration_seconds:
        notes.append(
            f"{clip.clip_id}: out-point trimmed to the end of "
            f"{clip.source_id} ({source.duration_seconds:.3f}s)"
        )
    return in_seconds, out_seconds


def _resolve_crop(
    clip: Clip,
    source: Source,
    target_aspect: float,
    degradations: list[DegradationStep],
) -> CropBox | None:
    """Frame for the target aspect, recording any fall back to centre."""

    if abs(source.aspect_ratio - target_aspect) < 1e-3:
        return None

    crop, fallback_reason = _subject_crop(
        source.aspect_ratio, target_aspect, clip
    )
    if fallback_reason is not None:
        degradations.append(
            DegradationStep(
                clip_id=clip.clip_id,
                ladder="center_crop",
                trigger=fallback_reason,
                measured={
                    "source_aspect": source.aspect_ratio,
                    "target_aspect": target_aspect,
                },
            )
        )
    return crop
