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

# How much of a subject's travel has to end up as net displacement before the
# camera commits to following it. A subject that steps out and comes back
# inside one short shot has gone nowhere; an operator watching that holds the
# frame, and a camera that chases each swing reads as a wobble rather than a
# move. Below this the shot is framed on where the subject spent its time.
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
            for frame in self.keyframes
        )

    def travel(self) -> float:
        """Total horizontal movement, in viewport widths."""

        return sum(
            abs(later.crop.x - earlier.crop.x)
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

    Returns a single-keyframe path when the subject barely moves; that is a
    hold, not a failure, and it is recorded as neither.
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
        # Nothing worth following. A still camera on a still subject is the
        # right answer, so no degradation is recorded.
        return CropPath([Keyframe(observations[0].seconds, crop_at(_percentile(centres, 0.5)))])

    wandered = sum(
        abs(later.centre_x - earlier.centre_x)
        for earlier, later in zip(observations, observations[1:])
    )
    net = abs(centres[-1] - centres[0])
    directness = net / wandered if wandered > 1e-9 else 0.0
    if directness < MIN_DIRECTNESS:
        # The subject moved but did not go anywhere. Hold on where it spent
        # its time rather than retracing each swing. This is a framing
        # decision, not a fallback, so nothing is recorded as degraded.
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


def ffmpeg_crop_expression(
    path: CropPath, width: int, height: int
) -> tuple[str, str, int, int]:
    """Render a path as an ffmpeg crop filter's arguments.

    Returns (x expression, y expression, crop width, crop height). A static
    path yields plain numbers; a moving one yields a piecewise-linear
    expression in `t`, which keeps the motion in one filter rather than
    driving it from a command file.
    """

    first = path.keyframes[0].crop
    crop_w, crop_h = first.to_pixels(width, height)[2:]

    if path.is_static:
        x, y = first.to_pixels(width, height)[:2]
        return str(x), str(y), crop_w, crop_h

    # Build from the last segment backwards so each `if` falls through to the
    # earlier one, ending at the first keyframe's value.
    expression = f"{path.keyframes[0].crop.x * width:.3f}"
    for earlier, later in zip(path.keyframes, path.keyframes[1:]):
        start_x = earlier.crop.x * width
        end_x = later.crop.x * width
        span = max(later.seconds - earlier.seconds, 1e-6)
        ramp = (
            f"{start_x:.3f}+({end_x - start_x:.3f})*"
            f"(t-{earlier.seconds:.3f})/{span:.3f}"
        )
        expression = (
            f"if(between(t,{earlier.seconds:.3f},{later.seconds:.3f}),"
            f"{ramp},{expression})"
        )
    # Past the final keyframe, hold the last position.
    last = path.keyframes[-1]
    expression = (
        f"if(gte(t,{last.seconds:.3f}),{last.crop.x * width:.3f},{expression})"
    )
    y = first.to_pixels(width, height)[1]
    return expression, str(y), crop_w, crop_h
