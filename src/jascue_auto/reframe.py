"""Follow a subject with the crop instead of holding it still.

The semantic layer names the subject; grounding finds it at a handful of
sampled frames; this module turns those observations into a crop that moves.

Three choices here come straight from what the previous system got wrong.

Failure is judged on the raw track, before smoothing. Filtering a trajectory
and then testing the filtered result lets the filter manufacture the problem
it is then blamed for.

The follow uses the ninetieth percentile of where the subject actually went,
not its extremes. One frame of a hand crossing the lens should not define the
whole move; the previous system planned for the worst frame and so planned a
move nobody asked for.

Speed limits live in viewport widths per second, not pixels. A pixel limit is
only meaningful beside the resolution it was written against, which is how the
same constant meant two different things on 1080 and 4K sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jascue_auto.executor import CROP_MARGIN, CropBox
from jascue_auto.schema import CameraEnergy, DegradationStep

# Per camera_energy, in viewport widths per second and per second squared.
# 720 px/s on a 1080-wide portrait viewport, the old constant, is 0.667 vw/s --
# which is roughly what "active" means here.
ENERGY_LIMITS: dict[CameraEnergy, dict[str, float]] = {
    "calm": {"max_speed": 0.25, "max_accel": 0.60, "lead": 0.10},
    "active": {"max_speed": 0.67, "max_accel": 1.67, "lead": 0.20},
    "dynamic": {"max_speed": 1.20, "max_accel": 3.00, "lead": 0.30},
}

# Movement below this is invisible and only adds jitter, so the camera holds.
DEADBAND = 0.02

# How much of a subject's travel has to end up as net displacement before a
# follow is worth executing. A subject that steps out and comes back inside
# one short shot has gone nowhere, and chasing each swing reads as a wobble.
#
# Crossing this threshold is not a veto. The planner asked for a follow and a
# follow is what it gets when there is motion to follow; below the threshold
# the shot is framed on where the subject spent its time AND the substitution
# is written into the degradation record. Silently returning a hold, which is
# what this did first, tells the planner its instruction was carried out.
MIN_DIRECTNESS = 0.6


class OutOfFrame(ValueError):
    """An observation that is not in normalised coordinates.

    Worth its own type because the failure it prevents is silent. A model that
    answers in pixels produces values like 381 where 0..1 was asked for, and
    clamping those into range yields a crop that sits at the edge for every
    frame -- which reads downstream as "the subject never moved" and renders
    as a considered hold. A wrong answer wearing the shape of a decision is
    worse than a loud one.
    """


@dataclass(frozen=True)
class Observation:
    """Where the subject was at one sampled moment, normalised."""

    seconds: float
    centre_x: float
    centre_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in (
            ("centre_x", self.centre_x),
            ("centre_y", self.centre_y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not 0.0 <= value <= 1.0:
                raise OutOfFrame(
                    f"{name}={value:g} is outside 0..1 at t={self.seconds:g}s. "
                    "Observations are frame fractions; a value above 1 is "
                    "usually pixels, which has to be converted by whoever "
                    "knows the frame size rather than clamped here."
                )


@dataclass(frozen=True)
class Keyframe:
    seconds: float
    crop: CropBox


@dataclass
class CropPath:
    """A crop over time. One keyframe means a static crop."""

    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def is_static(self) -> bool:
        if len(self.keyframes) < 2:
            return True
        first = self.keyframes[0].crop
        return all(
            abs(frame.crop.x - first.x) < 1e-6
            and abs(frame.crop.y - first.y) < 1e-6
            and abs(frame.crop.width - first.width) < 1e-6
            and abs(frame.crop.height - first.height) < 1e-6
            for frame in self.keyframes
        )

    def travel(self) -> float:
        """Total movement across every axis, in viewport widths."""

        return sum(
            max(
                abs(later.crop.x - earlier.crop.x),
                abs(later.crop.y - earlier.crop.y),
                abs(later.crop.width - earlier.crop.width),
            )
            for earlier, later in zip(self.keyframes, self.keyframes[1:])
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _limit_speed(
    keyframes: list[Keyframe], limits: dict[str, float]
) -> tuple[list[Keyframe], float]:
    """Hold the camera inside its energy budget, reporting what it hit."""

    if len(keyframes) < 2:
        return keyframes, 0.0

    limited = [keyframes[0]]
    peak = 0.0
    for previous, current in zip(keyframes, keyframes[1:]):
        span = max(current.seconds - previous.seconds, 1e-6)
        anchor = limited[-1].crop
        desired = current.crop.x - anchor.x
        speed = abs(desired) / span
        peak = max(peak, speed)
        allowed = limits["max_speed"] * span
        if abs(desired) > allowed:
            desired = allowed if desired > 0 else -allowed
        limited.append(
            Keyframe(
                seconds=current.seconds,
                crop=CropBox(
                    x=min(max(anchor.x + desired, 0.0), 1.0 - current.crop.width),
                    y=current.crop.y,
                    width=current.crop.width,
                    height=current.crop.height,
                ),
            )
        )
    return limited, peak


def _smooth(keyframes: list[Keyframe], strength: float = 0.5) -> list[Keyframe]:
    """Take the corners off. Endpoints stay put so the shot starts where it starts."""

    if len(keyframes) < 3:
        return keyframes
    smoothed = [keyframes[0]]
    for previous, current, following in zip(
        keyframes, keyframes[1:], keyframes[2:]
    ):
        blended = (
            previous.crop.x + following.crop.x
        ) / 2.0 * strength + current.crop.x * (1.0 - strength)
        smoothed.append(
            Keyframe(
                seconds=current.seconds,
                crop=CropBox(
                    x=min(max(blended, 0.0), 1.0 - current.crop.width),
                    y=current.crop.y,
                    width=current.crop.width,
                    height=current.crop.height,
                ),
            )
        )
    smoothed.append(keyframes[-1])
    return smoothed


def build_sweep_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    direction: str,
    energy: CameraEnergy = "calm",
) -> CropPath:
    """A designed move across a static arrangement.

    No subject is followed here because nothing is moving. A row of handsets
    wants the eye carried across it, and that is a decision about how to
    present the frame rather than a reaction to something in it. Treating this
    as a follow of the group's centre yields a hold, which is the answer to a
    question nobody asked.
    """

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)
    if free_x <= 0.0:
        return CropPath(
            [Keyframe(0.0, CropBox(0.0, (1.0 - crop_height) / 2.0, crop_width, crop_height))]
        )

    # Sweep the width the energy allows in the time available, never more of
    # the frame than exists.
    limits = ENERGY_LIMITS[energy]
    reach = min(free_x, limits["max_speed"] * duration_seconds)
    inset = free_x * CROP_MARGIN
    if direction == "sweep_left":
        start, end = min(free_x - inset, inset + reach), inset
    else:
        start, end = inset, min(free_x - inset, inset + reach)

    y = (1.0 - crop_height) / 2.0
    return CropPath(
        [
            Keyframe(0.0, CropBox(start, y, crop_width, crop_height)),
            Keyframe(
                duration_seconds,
                CropBox(end, y, crop_width, crop_height),
            ),
        ]
    )


def build_handoff_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    from_position: str,
    to_position: str,
) -> CropPath:
    """Carry the eye from one subject to another inside one shot.

    Splitting the shot instead, which this did first, cuts a continuous take
    to itself: same background, same light, same moment, and the frame jumps
    2362 pixels sideways. That is not an edit, it is a jump cut with nothing
    motivating it, and it is what a viewer notices at the two-second mark.

    An editor moving attention across a static frame either pans across it or
    cuts away and back. Panning is what a handoff means, so panning is what it
    does.
    """

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)
    free_y = max(0.0, 1.0 - crop_height)

    def anchor(position: str) -> tuple[float, float]:
        vertical, _, horizontal = position.partition("_")
        across = {"left": 0.0, "center": 0.5, "right": 1.0}.get(horizontal, 0.5)
        down = {"top": 0.0, "mid": 0.5, "bottom": 1.0}.get(vertical, 0.5)
        x = across * free_x
        y = down * free_y
        if free_x > 0.0:
            x = min(max(x, free_x * CROP_MARGIN), free_x * (1.0 - CROP_MARGIN))
        if free_y > 0.0:
            y = min(max(y, free_y * CROP_MARGIN), free_y * (1.0 - CROP_MARGIN))
        return x, y

    start_x, start_y = anchor(from_position)
    end_x, end_y = anchor(to_position)
    return CropPath(
        [
            Keyframe(0.0, CropBox(start_x, start_y, crop_width, crop_height)),
            Keyframe(
                duration_seconds,
                CropBox(end_x, end_y, crop_width, crop_height),
            ),
        ]
    )


def build_zoom_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    direction: str,
    centre_x: float = 0.5,
    centre_y: float = 0.5,
    energy: CameraEnergy = "calm",
) -> CropPath:
    """Close in on something, or open out from it.

    The most-asked-for move on this material by some distance -- four of
    eleven shots in one selection -- because a product film is mostly about
    looking closer at things rather than chasing them.

    The crop shrinks toward the subject for a push in and grows away for a
    pull out. How far it travels is bounded by how much frame there is: a crop
    that is already most of the source has nowhere to go, and forcing one
    would soften the picture rather than move it.
    """

    if target_aspect < source_aspect:
        base_width = target_aspect / source_aspect
        base_height = 1.0
    else:
        base_width = 1.0
        base_height = source_aspect / target_aspect

    # Zoom range scales with energy: a calm push is barely felt, a dynamic one
    # is a move in its own right. Capped so the tightest crop still has real
    # pixels behind it.
    span = {"calm": 0.10, "active": 0.18, "dynamic": 0.28}[energy]
    tight = max(0.35, 1.0 - span)

    wide = (base_width, base_height)
    close = (base_width * tight, base_height * tight)
    first, last = (wide, close) if direction == "push_in" else (close, wide)

    def box(size: tuple[float, float]) -> CropBox:
        width, height = size
        x = min(max(centre_x - width / 2.0, 0.0), max(0.0, 1.0 - width))
        y = min(max(centre_y - height / 2.0, 0.0), max(0.0, 1.0 - height))
        return CropBox(x, y, width, height)

    return CropPath(
        [Keyframe(0.0, box(first)), Keyframe(duration_seconds, box(last))]
    )


def build_crop_path(
    observations: list[Observation],
    *,
    source_aspect: float,
    target_aspect: float,
    energy: CameraEnergy = "calm",
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
) -> CropPath:
    """Turn subject observations into a crop that follows them.

    Returns a single-keyframe path when the subject barely moves. That is a
    hold, and when a follow was asked for it is also a substitution, so it is
    recorded as one.
    """

    if not observations:
        raise ValueError("a crop path needs at least one observation")

    limits = ENERGY_LIMITS[energy]
    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)

    # Judge the movement on the raw observations, before anything is filtered.
    centres = [observation.centre_x for observation in observations]
    spread = _percentile(centres, 0.95) - _percentile(centres, 0.05)

    def crop_at(centre_x: float, lead: float = 0.0) -> CropBox:
        x = centre_x + lead - crop_width / 2.0
        if free_x > 0.0:
            x = min(max(x, free_x * CROP_MARGIN), free_x * (1.0 - CROP_MARGIN))
        else:
            x = 0.0
        return CropBox(x=x, y=(1.0 - crop_height) / 2.0, width=crop_width, height=crop_height)

    if spread < DEADBAND or free_x <= 0.0:
        if degradations is not None:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="static_on_subject",
                    trigger=(
                        "a follow was planned but the subject does not move "
                        "in this shot"
                        if free_x > 0.0
                        else "a follow was planned but the crop already fills "
                        "the frame on this axis, leaving nowhere to move"
                    ),
                    measured={
                        "subject_spread_vw": round(spread, 4),
                        "deadband_vw": DEADBAND,
                        "free_travel_vw": round(free_x, 4),
                    },
                )
            )
        return CropPath([Keyframe(observations[0].seconds, crop_at(_percentile(centres, 0.5)))])

    wandered = sum(
        abs(later.centre_x - earlier.centre_x)
        for earlier, later in zip(observations, observations[1:])
    )
    net = abs(centres[-1] - centres[0])
    directness = net / wandered if wandered > 1e-9 else 0.0
    if directness < MIN_DIRECTNESS:
        # The subject moved but did not go anywhere. Framing on where it spent
        # its time beats retracing each swing -- and this is a substitution
        # for what was asked, so it is recorded rather than passed off as the
        # plan being carried out.
        if degradations is not None:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="static_on_subject",
                    trigger=(
                        "a follow was planned but the subject returned to "
                        "where it started, so following it would read as a "
                        "wobble rather than a move"
                    ),
                    measured={
                        "directness": round(directness, 4),
                        "threshold": MIN_DIRECTNESS,
                        "net_displacement_vw": round(net, 4),
                        "total_wander_vw": round(wandered, 4),
                    },
                )
            )
        return CropPath(
            [
                Keyframe(
                    observations[0].seconds,
                    crop_at(_percentile(centres, 0.5)),
                )
            ]
        )

    raw: list[Keyframe] = []
    for index, observation in enumerate(observations):
        # Lead the subject in its direction of travel so it is not pinned to
        # the trailing edge of the frame. The size of that lead scales with
        # how fast the subject is actually going.
        #
        # Taking only the sign of the drift, as this did first, gives a
        # subject creeping by a thousandth of a frame the same full-magnitude
        # lead as one crossing it -- and then swings the crop by twice that
        # the moment the subject pauses or drifts back. A shot of a nearly
        # still product measured a subject travel of 0.100 and a crop travel
        # of 0.158, reversing once on the way. It reads as the camera
        # wobbling, because that is what it is.
        if index + 1 < len(observations):
            neighbour = observations[index + 1]
        else:
            neighbour = observations[index - 1]
        span = max(abs(neighbour.seconds - observation.seconds), 1e-6)
        drift = neighbour.centre_x - observation.centre_x
        if index + 1 >= len(observations):
            drift = -drift  # looking backwards; the direction of travel flips
        speed = drift / span
        # Full lead at the energy's top speed, proportionally less below it,
        # and nothing at all when the subject is effectively parked.
        share = max(-1.0, min(1.0, speed / limits["max_speed"]))
        lead = limits["lead"] * crop_width * share
        raw.append(Keyframe(observation.seconds, crop_at(observation.centre_x, lead)))

    limited, peak_speed = _limit_speed(raw, limits)
    path = CropPath(_smooth(limited))

    if degradations is not None and peak_speed > limits["max_speed"]:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="slower_follow",
                trigger=(
                    "the subject outran the camera budget for this energy, so "
                    "the follow was held to its limit"
                ),
                measured={
                    "observed_speed_vw_per_s": round(peak_speed, 4),
                    "limit_vw_per_s": limits["max_speed"],
                    "subject_spread_vw": round(spread, 4),
                },
            )
        )
    return path


def _eased(start: float, end: float, at: float, span: float) -> str:
    """A smoothstep ramp between two values, as an ffmpeg expression.

    Linear interpolation, which this used first, gives constant velocity with
    an instant start and an instant stop. Nothing physical moves that way and
    nothing shot by hand looks that way, which is most of what reads as
    mechanical in a generated move. Smoothstep eases both ends, so the camera
    takes up the move and sets it down.

    This is also where the acceleration budget finally does something:
    ENERGY_LIMITS carried a max_accel that nothing referenced, because a
    constant-velocity ramp has no acceleration to bound.
    """

    delta = end - start
    # 3u^2 - 2u^3 over the segment's own normalised time.
    unit = f"clip((t-{at:.3f})/{span:.6f},0,1)"
    return f"{start:.3f}+({delta:.3f})*({unit}*{unit}*(3-2*{unit}))"


def _axis_expression(
    path: CropPath, pick, scale: int, *, ease: bool = True
) -> str:
    """Piecewise expression for one axis over the whole shot."""

    first = pick(path.keyframes[0].crop) * scale
    expression = f"{first:.3f}"
    for earlier, later in zip(path.keyframes, path.keyframes[1:]):
        start = pick(earlier.crop) * scale
        end = pick(later.crop) * scale
        span = max(later.seconds - earlier.seconds, 1e-6)
        ramp = (
            _eased(start, end, earlier.seconds, span)
            if ease
            else f"{start:.3f}+({end - start:.3f})*(t-{earlier.seconds:.3f})/{span:.6f}"
        )
        expression = (
            f"if(between(t,{earlier.seconds:.3f},{later.seconds:.3f}),"
            f"{ramp},{expression})"
        )
    last = path.keyframes[-1]
    return (
        f"if(gte(t,{last.seconds:.3f}),{pick(last.crop) * scale:.3f},"
        f"{expression})"
    )


def ffmpeg_crop_expression(
    path: CropPath, width: int, height: int
) -> tuple[str, str, str, str]:
    """Render a path as a crop filter's four arguments.

    Returns expressions for width, height, x and y. A static path yields
    plain numbers. Anything that moves -- across, down, or in -- yields eased
    expressions in `t`, so one filter carries the whole move.
    """

    first = path.keyframes[0].crop
    if path.is_static:
        x, y, crop_w, crop_h = first.to_pixels(width, height)
        return str(crop_w), str(crop_h), str(x), str(y)

    # Crop extents must stay even for chroma subsampling, and must not run off
    # the frame at any point in the ramp.
    w_expr = f"floor(min({_axis_expression(path, lambda c: c.width, width)},{width})/2)*2"
    h_expr = f"floor(min({_axis_expression(path, lambda c: c.height, height)},{height})/2)*2"
    x_expr = f"floor(max(0,min({_axis_expression(path, lambda c: c.x, width)},{width}-out_w))/2)*2"
    y_expr = f"floor(max(0,min({_axis_expression(path, lambda c: c.y, height)},{height}-out_h))/2)*2"
    return w_expr, h_expr, x_expr, y_expr
