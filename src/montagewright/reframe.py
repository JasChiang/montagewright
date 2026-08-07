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

from montagewright.executor import CROP_MARGIN, CropBox
from montagewright.schema import CameraEnergy, DegradationStep

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
    from_centre: float,
    to_centre: float,
    from_width: float = 0.0,
    to_width: float = 0.0,
    energy: CameraEnergy = "calm",
) -> CropPath:
    """Carry the eye from one subject to another inside one shot.

    Splitting the shot instead, which this did first, cuts a continuous take
    to itself: same background, same light, same moment, and the frame jumps
    sideways. That is a jump cut with nothing motivating it.

    The endpoints are measured subject centres, not nine-box guesses. Mapping
    "mid_left" and "mid_right" onto the edges of the frame sent the crop to
    0.192 and 0.808 for two handsets that actually sit at 0.359 and 0.635 --
    overshooting each subject by about a sixth of the frame, so the pan began
    with the first phone half out of frame on the right, crossed empty
    background in the middle, and ended with the second one clipped on the
    left. It also made the move twice as long as it needed to be, which is the
    same bug showing up as speed.
    """

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)
    y = (1.0 - crop_height) / 2.0

    def centred_on(centre: float, subject_width: float = 0.0) -> float:
        """Frame the subject without travelling past it into background.

        Centring alone is not framing. A subject sitting at 0.72 of the frame
        is centred by pushing the crop's right edge to 0.88 -- and if the
        subject itself ends at 0.83, the last twentieth of the shot is wall.
        The pan stops where the subject's far edge does, with a margin, so it
        arrives on the subject rather than beside it.
        """

        wanted = centre - crop_width / 2.0
        if subject_width > 0.0:
            far = centre + subject_width / 2.0
            # Keep a little air past the subject, no more.
            limit = min(free_x, max(0.0, far + crop_width * CROP_MARGIN - crop_width))
            near = centre - subject_width / 2.0
            floor = max(0.0, near - crop_width * CROP_MARGIN)
            # Both bounds exist to stop the crop drifting off a subject that
            # fills it. A subject narrower than the crop cannot touch both
            # edges, so they contradict -- and resolving that by taking one of
            # them hugs the subject to that edge: a folded Flip ending a pan
            # sat at a tenth of the frame with a third of it wall, which reads
            # as the pan having gone too far. Slack means centred.
            if floor <= limit:
                wanted = min(max(wanted, floor), limit)
        return min(max(wanted, 0.0), free_x)

    start_x = centred_on(from_centre, from_width)
    end_x = centred_on(to_centre, to_width)

    path = CropPath(
        [
            Keyframe(0.0, CropBox(start_x, y, crop_width, crop_height)),
            Keyframe(duration_seconds, CropBox(end_x, y, crop_width, crop_height)),
        ]
    )
    limited, _ = _limit_speed(path.keyframes, ENERGY_LIMITS[energy])
    return CropPath(limited)


# How far the delivered frame may be enlarged past what the source supplies.
#
# These are the cost of each editorial answer, not a rule about which to pick.
# Holding every shot to the sharp figure is a local veto on a question only the
# edit can settle: a coin held against a hinge to show 4.1mm is worth showing
# soft, and a wide establisher is not. The planner says which this is; these
# numbers say what it costs.
# One ceiling, because softening the picture was never the thing worth
# trading. Empty frame is an acceptable outcome; a soft frame is not.
MAX_UPSCALE = 1.35

# Where the subject centre lands vertically within the crop, per intent. The
# thirds figure leaves the air above a subject that a centred frame throws
# away evenly on both sides -- which is why a coin dead-centre in a tall frame
# reads as untouched rather than composed.
PLACEMENT: dict[str, float] = {
    "thirds": 0.618,
    "centre": 0.5,
    "fill": 0.5,
}


def travel_room(
    *,
    source_width: int,
    source_height: int,
    target_aspect: float,
    output_width: int,
    output_height: int,
) -> tuple[float, float]:
    """How far a crop can move across this source, per axis, for free.

    Fractions of the source frame, horizontal first. Zero means a move along
    that axis has nowhere to go.

    The first version of this asked only about shape, and so assumed the crop
    is always the largest one that fits -- full height for a wide source at a
    tall target. That is only forced when the source has no resolution to
    spare. A 4K take delivering 1080x1920 needs a crop just 1920 pixels tall,
    not the 2160 it has, and the 240 left over are room to tilt: 0.111 of the
    frame at no cost at all, and 0.342 if the whole zoom budget goes on it.
    Reported as zero, the one move that take could carry looked impossible.

    What is returned is the free room -- the crop that needs no enlarging --
    because travel and tightening are drawn from the same spare pixels, and a
    number that has already spent the zoom budget is not one the planner can
    combine with `push_room`. A source with nothing spare falls back to the
    largest crop that fits, which is what it was doing before.
    """

    if min(source_width, source_height, output_width, output_height) <= 0:
        return 0.0, 0.0
    if target_aspect <= 0:
        return 0.0, 0.0

    # As small as the delivery, never larger than the frame, and never wider
    # than the frame is either.
    tall = min(float(source_height), float(output_height), source_width / target_aspect)
    wide = tall * target_aspect
    return (
        max(0.0, (source_width - wide) / source_width),
        max(0.0, (source_height - tall) / source_height),
    )


def zoom_budget(
    *,
    source_width: int,
    source_height: int,
    source_aspect: float,
    target_aspect: float,
    output_width: int,
    output_height: int,
    max_upscale: float = MAX_UPSCALE,
) -> float:
    """The tightest crop this source can supply at this output size.

    Read before choosing a zoom rather than after. A fixed percentage per
    energy level is blind to what it is cropping: the same 28% is nothing on
    a 4K source and unwatchable on a 1080 one.

    Returned as a scale factor -- 1.0 means no room to push at all, 0.4 means
    the crop may shrink to 40% of its base size before the delivered frame is
    being enlarged past the budget.
    """

    if target_aspect < source_aspect:
        base_w = (target_aspect / source_aspect) * source_width
        base_h = float(source_height)
    else:
        base_w = float(source_width)
        base_h = (source_aspect / target_aspect) * source_height

    # The crop may shrink until delivering it would enlarge past the budget.
    limit_w = (output_width / max_upscale) / max(base_w, 1e-6)
    limit_h = (output_height / max_upscale) / max(base_h, 1e-6)
    return min(1.0, max(limit_w, limit_h, 0.05))


def achieved_upscale(
    crop: CropBox,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> float:
    """How much this crop has to be enlarged to fill the output.

    A number, reported alongside the cut. Sharpness is measurable, so nobody
    should be asked to judge it from a preview -- least of all to tell a soft
    proxy apart from a genuinely over-enlarged shot, which at preview
    resolution look identical.
    """

    pixels_w = max(crop.width * source_width, 1e-6)
    pixels_h = max(crop.height * source_height, 1e-6)
    return max(output_width / pixels_w, output_height / pixels_h)


def build_tilt_path(
    observations: list[Observation],
    *,
    source_aspect: float,
    target_aspect: float,
    energy: CameraEnergy = "calm",
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
) -> CropPath:
    """Follow a subject up or down the frame.

    A watch lowered into a tank, a handset lifted off a table: the motion that
    carries the shot is vertical, and a crop that only ever moves sideways
    cannot follow it. Worse, it reports the subject as having no horizontal
    spread and holds -- the shot is described as needing no camera work when
    what actually happened is that its movement was on an axis nothing looked
    at.

    Converting a wide source to a tall one leaves the crop at full height with
    nowhere to go vertically. That is a real limit rather than a failure to
    try, so it is stated as one.
    """

    if not observations:
        raise ValueError("a tilt needs at least one observation")

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_y = max(0.0, 1.0 - crop_height)
    centres_y = [observation.centre_y for observation in observations]
    spread = _percentile(centres_y, 0.95) - _percentile(centres_y, 0.05)
    x = min(
        max(
            sum(o.centre_x for o in observations) / len(observations)
            - crop_width / 2.0,
            0.0,
        ),
        max(0.0, 1.0 - crop_width),
    )

    def at(centre_y: float) -> CropBox:
        y = min(max(centre_y - crop_height / 2.0, 0.0), free_y)
        return CropBox(x=x, y=y, width=crop_width, height=crop_height)

    if free_y <= 0.0 or spread < DEADBAND:
        if degradations is not None:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="static_on_subject",
                    trigger=(
                        "a tilt was planned but the crop already fills the "
                        "frame vertically, leaving nowhere to move"
                        if free_y <= 0.0
                        else "a tilt was planned but the subject does not "
                        "move vertically in this shot"
                    ),
                    measured={
                        "vertical_spread_vw": round(spread, 4),
                        "free_travel_vw": round(free_y, 4),
                    },
                )
            )
        return CropPath(
            [Keyframe(observations[0].seconds, at(_percentile(centres_y, 0.5)))]
        )

    raw = [
        Keyframe(observation.seconds, at(observation.centre_y))
        for observation in observations
    ]
    limited, _ = _limit_speed(raw, ENERGY_LIMITS[energy])
    return CropPath(limited)


def build_zoom_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    direction: str,
    centre_x: float = 0.5,
    centre_y: float = 0.5,
    energy: CameraEnergy = "calm",
    framing: str = "thirds",
    budget: float = 0.0,
    subject_height: float | None = None,
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
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

    # What the shot wants: enough of a push that the subject reads, with room
    # around it. What the source allows: whatever keeps the delivered frame
    # inside its enlargement budget. Take the more conservative.
    wanted = {"calm": 0.90, "active": 0.82, "dynamic": 0.72}[energy]
    if subject_height is not None and subject_height > 0.0 and framing == "fill":
        # Only a fill intent chases the subject's size. The others accept the
        # frame the source gives and place the subject inside it.
        wanted = min(wanted, max(0.2, subject_height / 0.66))
    allowed = budget if budget > 0.0 else 0.35
    tight = max(wanted, allowed)

    if degradations is not None and wanted < allowed - 1e-6:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="reduced_zoom",
                trigger=(
                    "the source cannot supply the push this shot wants without "
                    "enlarging the delivered frame past its budget"
                ),
                measured={
                    "wanted_scale": round(wanted, 4),
                    "allowed_scale": round(allowed, 4),
                    "subject_height_vw": round(subject_height or 0.0, 4),
                },
            )
        )

    wide = (base_width, base_height)
    close = (base_width * tight, base_height * tight)
    first, last = (wide, close) if direction == "push_in" else (close, wide)

    def box(size: tuple[float, float]) -> CropBox:
        width, height = size
        # Put the subject on the intended line of the crop rather than in the
        # middle of it. A centred subject splits its negative space evenly
        # above and below, which reads as an untouched frame; placing it on
        # the lower third gathers that space into one piece of air above.
        share = PLACEMENT.get(framing, 0.5)
        x = min(max(centre_x - width / 2.0, 0.0), max(0.0, 1.0 - width))
        y = min(max(centre_y - height * share, 0.0), max(0.0, 1.0 - height))
        return CropBox(x, y, width, height)

    return CropPath(
        [Keyframe(0.0, box(first)), Keyframe(duration_seconds, box(last))]
    )


def visible_fraction(crop: CropBox, observation: Observation) -> float:
    """How much of the subject the crop actually contains.

    Centring on a subject says nothing about whether it fits. A subject 0.55
    of the frame wide inside a 0.316 crop is 57% visible however perfectly it
    is centred, and the path builder reported that shot as executed with no
    degradation -- the crop was exactly where it was asked to be, and half the
    product was outside it.
    """

    left = max(crop.x, observation.centre_x - observation.width / 2.0)
    right = min(crop.x + crop.width, observation.centre_x + observation.width / 2.0)
    across = max(0.0, right - left) / max(observation.width, 1e-9)

    top = max(crop.y, observation.centre_y - observation.height / 2.0)
    bottom = min(
        crop.y + crop.height, observation.centre_y + observation.height / 2.0
    )
    down = max(0.0, bottom - top) / max(observation.height, 1e-9)
    return across * down


def _report_fit(
    path: CropPath,
    observations: list[Observation],
    *,
    clip_id: str,
    min_visible: float,
    degradations: list[DegradationStep] | None,
) -> CropPath:
    """Record how much of the subject the finished path actually holds.

    Every exit from the path builder passes through here. Placing this only on
    the moving branch, as it was first, skipped exactly the shots that need it
    -- a subject too big for the crop usually is not moving, so the static
    early return carried it straight past the check.
    """

    if degradations is None or not observations:
        return path

    worst = min(
        visible_fraction(
            path.keyframes[
                min(index, len(path.keyframes) - 1)
            ].crop,
            observation,
        )
        for index, observation in enumerate(observations)
    )
    if worst < min_visible:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="other",
                ladder_other="subject_larger_than_crop",
                trigger=(
                    "the subject does not fit the target aspect at any point "
                    "in this shot, so it is framed as fully as the crop allows"
                ),
                measured={
                    "worst_visible_fraction": round(worst, 4),
                    "requested_min_visible": min_visible,
                    "subject_width_vw": round(
                        max(o.width for o in observations), 4
                    ),
                    "crop_width_vw": round(path.keyframes[0].crop.width, 4),
                },
            )
        )
    return path


def build_crop_path(
    observations: list[Observation],
    *,
    source_aspect: float,
    target_aspect: float,
    energy: CameraEnergy = "calm",
    framing: str = "thirds",
    clip_id: str = "",
    min_visible: float = 0.85,
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
        share = PLACEMENT.get(framing, 0.5)
        subject_y = (
            sum(o.centre_y for o in observations) / len(observations)
            if observations
            else 0.5
        )
        y = min(max(subject_y - crop_height * share, 0.0), max(0.0, 1.0 - crop_height))
        return CropBox(x=x, y=y, width=crop_width, height=crop_height)

    def hold(trigger: str, measured: dict[str, float]) -> CropPath:
        """Frame where the subject spent its time, and say why not to follow.

        Both reasons to hold end here: the subject barely moved, or it moved
        and came back. Either way a follow was planned and something else was
        delivered, so the substitution is recorded rather than passed off as
        the plan being carried out.
        """

        if degradations is not None:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="static_on_subject",
                    trigger=trigger,
                    measured=measured,
                )
            )
        return _report_fit(
            CropPath(
                [
                    Keyframe(
                        observations[0].seconds,
                        crop_at(_percentile(centres, 0.5)),
                    )
                ]
            ),
            observations,
            clip_id=clip_id,
            min_visible=min_visible,
            degradations=degradations,
        )

    if spread < DEADBAND or free_x <= 0.0:
        return hold(
            "a follow was planned but the subject does not move in this shot"
            if free_x > 0.0
            else "a follow was planned but the crop already fills the frame "
            "on this axis, leaving nowhere to move",
            {
                "subject_spread_vw": round(spread, 4),
                "deadband_vw": DEADBAND,
                "free_travel_vw": round(free_x, 4),
            },
        )

    wandered = sum(
        abs(later.centre_x - earlier.centre_x)
        for earlier, later in zip(observations, observations[1:])
    )
    net = abs(centres[-1] - centres[0])
    directness = net / wandered if wandered > 1e-9 else 0.0
    if directness < MIN_DIRECTNESS:
        return hold(
            "a follow was planned but the subject returned to where it "
            "started, so following it would read as a wobble rather than a "
            "move",
            {
                "directness": round(directness, 4),
                "threshold": MIN_DIRECTNESS,
                "net_displacement_vw": round(net, 4),
                "total_wander_vw": round(wandered, 4),
            },
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


def observations_from_sam(
    track,
    *,
    clip_start_seconds: float,
    accept_states: frozenset[str] = frozenset({"tracked"}),
) -> tuple[list[Observation], dict[str, int]]:
    """Read a propagated track as subject observations.

    Sampling five frames and interpolating, which this replaces, describes a
    three-second shot at one point every 0.6 seconds and guesses the rest. A
    subject that crosses the frame and returns inside one sample interval is
    invisible to that.

    Samples the tracker itself flagged as lost are dropped rather than
    averaged in. A mask that has drifted onto the background still reports a
    box, and feeding that to the crop is how a camera ends up following the
    wrong thing confidently. The counts come back so the caller can say how
    much of the shot was actually tracked.

    The seed still comes from Gemini: which of two similar handsets is meant
    is not a question a tracker can answer.
    """

    observations: list[Observation] = []
    states: dict[str, int] = {}
    for sample in getattr(track, "samples", []) or []:
        state = getattr(sample.tracking_state, "value", str(sample.tracking_state))
        states[state] = states.get(state, 0) + 1
        if state not in accept_states:
            continue
        box = sample.derived_tracking_box
        if not box or len(box) != 4:
            continue
        # The old package reports boxes as x-first 0..1000.
        x0, y0, x1, y1 = (value / 1000.0 for value in box)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width <= 0 or height <= 0:
            continue
        at = sample.analysis_sample_time_ms / 1000.0 - clip_start_seconds
        if at < 0:
            continue
        try:
            observations.append(
                Observation(
                    seconds=at,
                    centre_x=(x0 + x1) / 2.0,
                    centre_y=(y0 + y1) / 2.0,
                    width=width,
                    height=height,
                )
            )
        except OutOfFrame:
            continue
    return observations, states
