"""Where the camera moved, measured rather than watched.

Sparse video browsing can miss camera shake between sampled frames, so a
handheld take and a locked-off one may look like the same series of sharp
stills. Asked whether a clip was stable, the model once answered
"固定鏡頭…畫面穩定清晰" about a take whose first three seconds were the operator
finding the frame. Static 4/8 FPS improves what it can inspect; local geometry
still supplies the reproducible measurement.

That is a measurement, and this measures it. Nothing here decides whether a
stretch is worth using: it says the camera moved, roughly how much, and when
it settled. Whether that movement was a reveal, a follow, handheld texture or
somebody finding their shot is a question about meaning, and it goes to the
model with these intervals attached.

What is deliberately not done here is a threshold that rejects footage. A
number saying "0.42" cannot tell an intentional whip pan from a knock, and a
rule that treats both as failure throws away the shot the film wanted.

Frame differencing alone would not do. A phone on a stand playing a video
changes most of its pixels while the camera never moves, which is exactly the
clip that started this. What separates the two is whether the whole frame
moved together, so this estimates a global shift and reports how much of the
frame agreed with it.

Not everything that changes a picture is a shift. A lens being uncovered, a
push in, a hand passing close by: no offset in any direction explains those,
and the first version of this called them `unobservable` -- "nothing in the
frame is stable enough to compare". That reading was wrong twice over. It
searched horizontally only, so a tilt was also unexplainable and five frames
of one take were called unreadable for moving up and down. And the name
claimed ignorance about the one case the model is best placed to judge:
occlusions and zooms take seconds, so they are plainly visible at a frame a
second, unlike the between-frame shake this exists for. Saying "cannot be
measured" pushed the model towards `setup_reframe`, which earns no span, and
three layers cooperated to delete a clean opening shot.

So the state is now `not_a_shift`, which is what is actually known, and the
model is told it can see this one for itself.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

# Coarse enough to be cheap on seventy-four clips, fine enough to see a
# camera settle: shake lives well above this and a pan is visible at any
# rate.
COARSE_FPS = 4.0

# The deterministic fallback stays deliberately tiny. Affine tracking below
# uses a denser image so background features remain distinct.
WIDTH = 192
HEIGHT = 108
AFFINE_WIDTH = 384
AFFINE_HEIGHT = 216

# Movement below this, as a fraction of frame width per second, is a still
# camera. Sensor noise and compression alone put a floor under any measure.
STILL = 0.004

# Above this the frame is moving enough that a viewer would see it.
MOVING = 0.02

# Mean grey difference, 0..255, left over after the best offset has been
# taken out. Below this a shift explains the change and the reading stands;
# above it nothing does. Calibrated rather than picked: across this library
# the median residual is 0.1 to 0.7 and the worst single frame is 12.2, while
# frames of pure noise -- where by construction no shift explains anything --
# sit at 22.7. The first number chosen was 26, which noise passed, so the
# state it was added for could never occur.
NOT_A_SHIFT = 15.0

# Shortest stretch worth calling a state. Below it the reading is a flicker
# in the measurement rather than a thing the camera did. Interactions static
# 8 FPS can now inspect a sub-second event, so the old one-second visibility
# floor is deliberately gone; one 4 FPS interval is the local evidence floor.
LEAST_SECONDS = 0.25

# What the reading means, in the cache key. A measurement is a fact about the
# bytes, so it is content addressed and never recomputed -- which is right
# until the measurement itself starts meaning something else, and then every
# clip keeps answering the old question forever. Bump this when it does.
READING = "v4-affine-background"


@dataclass(frozen=True)
class MotionInterval:
    """One stretch of a clip during which the camera did one thing."""

    event_id: str
    starts_seconds: float
    ends_seconds: float
    state: str          # still | moving | not_a_shift
    peak_vw_s: float    # fastest global shift, frame widths per second
    travel_vw: float    # total global shift across the interval
    settles: bool       # ends still, having been moving
    motion_kind: str = "translation"
    direction: str = ""
    pan_vw_s: float = 0.0
    tilt_vh_s: float = 0.0
    zoom_rate_s: float = 0.0
    rotation_deg_s: float = 0.0
    confidence: float = 0.0
    phase: str = ""

    @property
    def seconds(self) -> float:
        return max(0.0, self.ends_seconds - self.starts_seconds)


def measure(source: Path, duration: float) -> list[MotionInterval]:
    """Split a clip into stretches of still, moving, and unreadable.

    The whole point is the boundaries. A take that begins with the operator
    finding the frame has a real edge where that stops, and it is the edge a
    cut should respect -- not a second either side of it chosen because the
    seconds happened to fall there.
    """

    affine = _global_affine(source, COARSE_FPS)
    legacy = _global_shift(source, COARSE_FPS) if affine else None
    # At very high translation speeds, adjacent frames share too few tracked
    # corners for RANSAC consensus even though the coarse pyramid can still
    # measure the global shift cleanly. Prefer that measured translation when
    # at least a third of affine samples fail consensus and every pyramid
    # sample remains explainable. This is a fallback between two local
    # measurements, not a semantic guess.
    low_consensus = sum(
        one.residual >= NOT_A_SHIFT for one in (affine or ())
    )
    if (
        affine and legacy
        and low_consensus * 3 >= len(affine)
        and all(residual < NOT_A_SHIFT for _, _, residual in legacy)
    ):
        affine = None
    if affine:
        shifts = [
            (sample.at_seconds, sample.composite_speed, sample.residual)
            for sample in affine
        ]
    else:
        shifts = legacy or _global_shift(source, COARSE_FPS)
    if not shifts:
        return [
            MotionInterval(
                event_id="m00", starts_seconds=0.0, ends_seconds=duration,
                state="not_a_shift", peak_vw_s=0.0, travel_vw=0.0,
                settles=False,
            )
        ]
    intervals = _into_intervals(shifts, duration)
    return _enrich_intervals(intervals, affine or ())


@dataclass(frozen=True)
class AffineSample:
    """One background-dominant camera-motion estimate between two frames."""

    at_seconds: float
    pan_vw_s: float
    tilt_vh_s: float
    zoom_rate_s: float
    rotation_deg_s: float
    inlier_ratio: float
    residual: float

    @property
    def composite_speed(self) -> float:
        # Express dissimilar transformations on one conservative movement
        # scale used only for still/moving hysteresis. Components stay
        # separate on the persisted interval and are never reconstructed
        # from this score.
        return max(
            math.hypot(self.pan_vw_s, self.tilt_vh_s),
            abs(self.zoom_rate_s) * 2.0,
            abs(self.rotation_deg_s) / 45.0,
        )


def _global_affine(source: Path, fps: float) -> list[AffineSample] | None:
    """Estimate translation, zoom and rotation from robust background tracks.

    Features are spread across the whole frame and an affine transform is
    fitted with RANSAC, so a walking foreground subject is normally rejected
    as an outlier instead of being mistaken for a camera pan. OpenCV remains
    an optional tracking dependency; installations without it retain the
    deterministic pyramid-shift fallback below.
    """

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None

    raw = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", (
                f"fps={fps},scale={AFFINE_WIDTH}:{AFFINE_HEIGHT},format=gray"
            ),
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=False,
    ).stdout
    size = AFFINE_WIDTH * AFFINE_HEIGHT
    frames = [
        np.frombuffer(raw[i * size:(i + 1) * size], dtype=np.uint8).reshape(
            AFFINE_HEIGHT, AFFINE_WIDTH
        )
        for i in range(len(raw) // size)
    ]
    if len(frames) < 2:
        return None

    out: list[AffineSample] = []
    for index, (before, after) in enumerate(zip(frames, frames[1:]), start=1):
        points = cv2.goodFeaturesToTrack(
            before, maxCorners=400, qualityLevel=0.01, minDistance=7,
            blockSize=7,
        )
        if points is None or len(points) < 12:
            return None
        moved, status, _errors = cv2.calcOpticalFlowPyrLK(
            before, after, points, None,
            winSize=(21, 21), maxLevel=3,
        )
        if moved is None or status is None:
            return None
        good = status.reshape(-1).astype(bool)
        starts = points.reshape(-1, 2)[good]
        ends = moved.reshape(-1, 2)[good]
        if len(starts) < 10:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            starts, ends, method=cv2.RANSAC,
            ransacReprojThreshold=2.5, maxIters=2000, confidence=0.995,
        )
        if matrix is None or inliers is None:
            return None
        ratio = float(inliers.reshape(-1).mean())
        a, b, dx = (float(one) for one in matrix[0])
        c, d, dy = (float(one) for one in matrix[1])
        scale = math.sqrt(max(1e-12, a * a + c * c))
        rotation = math.degrees(math.atan2(c, a))
        agreed = inliers.reshape(-1).astype(bool)
        homogeneous = np.column_stack(
            [starts, np.ones(len(starts), dtype=np.float32)]
        )
        projected = homogeneous @ matrix.T
        errors = np.linalg.norm(projected - ends, axis=1)
        residual = float(np.median(errors[agreed])) if agreed.any() else 99.0
        # A low-consensus transform is the least-bad explanation of local
        # object motion, not camera geometry. Mark it as not-a-shift through
        # the existing residual boundary.
        if ratio < 0.35:
            residual = max(residual, NOT_A_SHIFT + 1.0)
        out.append(
            AffineSample(
                at_seconds=index / fps,
                pan_vw_s=dx / AFFINE_WIDTH * fps,
                tilt_vh_s=dy / AFFINE_HEIGHT * fps,
                zoom_rate_s=(scale - 1.0) * fps,
                rotation_deg_s=rotation * fps,
                inlier_ratio=ratio,
                residual=residual,
            )
        )
    return out


def _enrich_intervals(
    intervals: list[MotionInterval], samples: "tuple[AffineSample, ...] | list[AffineSample]",
) -> list[MotionInterval]:
    """Attach decomposed affine facts to the existing temporal intervals."""

    if not samples:
        return intervals
    enriched: list[MotionInterval] = []
    previous_moving = False
    for interval in intervals:
        within = [
            one for one in samples
            if interval.starts_seconds <= one.at_seconds < interval.ends_seconds
        ]
        if not within:
            enriched.append(interval)
            previous_moving = interval.state == "moving"
            continue
        pan = max(within, key=lambda one: abs(one.pan_vw_s)).pan_vw_s
        tilt = max(within, key=lambda one: abs(one.tilt_vh_s)).tilt_vh_s
        zoom = max(within, key=lambda one: abs(one.zoom_rate_s)).zoom_rate_s
        rotation = max(
            within, key=lambda one: abs(one.rotation_deg_s)
        ).rotation_deg_s
        translation_speeds = [
            math.hypot(one.pan_vw_s, one.tilt_vh_s) for one in within
        ]
        magnitudes = {
            "translation": math.hypot(pan, tilt),
            "zoom": abs(zoom) * 2.0,
            "rotation": abs(rotation) / 45.0,
        }
        kind = max(magnitudes, key=magnitudes.get)
        direction = ""
        if kind == "translation":
            if abs(pan) >= abs(tilt):
                direction = "right" if pan > 0 else "left"
            else:
                direction = "down" if tilt > 0 else "up"
        elif kind == "zoom":
            direction = "push_in" if zoom > 0 else "pull_out"
        elif kind == "rotation":
            direction = "clockwise" if rotation > 0 else "counterclockwise"
        phase = (
            "movement_to_settle" if interval.state == "moving" and interval.settles
            else "movement" if interval.state == "moving"
            else "settled" if previous_moving and interval.state == "still"
            else "hold" if interval.state == "still"
            else "complex_change"
        )
        enriched.append(
            MotionInterval(
                **{
                    **asdict(interval),
                    # Preserve the historic meaning of these compatibility
                    # fields: translation in frame widths, never the affine
                    # classifier's mixed movement score.
                    "peak_vw_s": round(max(translation_speeds), 4),
                    "travel_vw": round(
                        sum(translation_speeds) / max(COARSE_FPS, 1e-6), 4
                    ),
                    "motion_kind": kind,
                    "direction": direction,
                    "pan_vw_s": round(pan, 4),
                    "tilt_vh_s": round(tilt, 4),
                    "zoom_rate_s": round(zoom, 4),
                    "rotation_deg_s": round(rotation, 3),
                    "confidence": round(
                        sum(one.inlier_ratio for one in within) / len(within), 3
                    ),
                    "phase": phase,
                }
            )
        )
        previous_moving = interval.state == "moving"
    return enriched


# How much of the two frames must still overlap for a comparison between
# them to mean anything. This is the only free number in the search, and it
# also decides how far an offset can be looked for: past half a frame there
# is not enough picture in common for any offset to be evidence of anything,
# so the reach is not a second thing to choose -- it falls out of this.
#
# It replaces a pixel count. That count was 8, then 24, and both were the
# same mistake: an offset outside the search is not found, so the residual
# stays high and a real movement is reported as something other than a
# shift. Three clips came back at exactly 0.17 frame widths per second,
# which is 8/192*4 -- the edge of the search, not a property of any of them.
# Raising it to 24 moved the wall. Reading coarsely first removes it: the
# ceiling is now what physics allows rather than what was affordable.
SHARED = 0.5

# How far down to look first. A pan too fast to find at full size is a small
# offset once the picture is a quarter as wide, and the answer there says
# which handful of full-size offsets are worth trying. Sixteen times fewer
# pixels per comparison pays for the whole coarse sweep several times over.
SHRINK = 4


def _shrink(frame: bytes, width: int, height: int, by: int) -> bytes:
    """The same picture, box-averaged down by an integer factor."""

    across, down = width // by, height // by
    out = bytearray(across * down)
    area = by * by
    for row in range(down):
        for column in range(across):
            total = 0
            for inner in range(by):
                base = (row * by + inner) * width + column * by
                total += sum(frame[base:base + by])
            out[row * across + column] = total // area
    return bytes(out)


def _agreement(
    before: bytes, after: bytes, width: int, height: int, across: int, down: int
) -> float:
    """Mean grey difference once the later frame is slid back by this much.

    Infinite when too little of the two frames still overlaps. That is not a
    guard against bad input -- it is what stops the search reaching offsets
    where a perfect score means nothing, and it is why the range of the
    search never had to be picked.
    """

    if (width - abs(across)) * (height - abs(down)) < SHARED * width * height:
        return float("inf")
    step = max(1, width // 64)
    rows = max(1, height // 9)
    total = count = 0
    for row in range(rows // 2, height, rows):
        if not 0 <= row + down < height:
            continue
        base = row * width
        moved = (row + down) * width
        edge = max(2, width // 48)
        for column in range(
            max(0, -across) + edge, min(width, width - across) - edge, step
        ):
            total += abs(before[base + column] - after[moved + column + across])
            count += 1
    return total / max(count, 1) if count else float("inf")


def _slide(
    before: bytes, after: bytes, width: int, height: int,
    other: int, reach: int, *, sideways: bool,
) -> int:
    """The offset on one axis that best explains the change."""

    def score(value: int) -> float:
        if sideways:
            return _agreement(before, after, width, height, value, other)
        return _agreement(before, after, width, height, other, value)

    return min(range(-reach, reach + 1), key=score)


def _settle(
    before: bytes, after: bytes, width: int, height: int,
    across: int, down: int, reach: int,
) -> tuple[int, int]:
    """Fix one axis, then the other, then revisit the first.

    A camera tilts as readily as it pans, and the first version of this
    searched sideways only -- so an operator lowering the frame left a
    residual nothing could account for and two clips of a twelve-clip
    sample were reported locked-off throughout. Searching the square costs
    the square and buys nothing: over 680 frames of this library, one axis
    at a time found the identical offset 678 times, and the two it missed
    were worse by 0.19 grey levels against a threshold of fifteen.
    """

    across = _slide(before, after, width, height, down, reach, sideways=True)
    down = _slide(before, after, width, height, across, reach, sideways=False)
    across = _slide(before, after, width, height, down, reach, sideways=True)
    return across, down


def _global_shift(source: Path, fps: float) -> list[tuple[float, float, float]]:
    """(seconds, shift, residual) per frame, shift in frame widths per second.

    Estimated by matching each frame against the one before it at a handful
    of offsets and taking the one that agrees best across the frame. Crude
    next to optical flow and enough to separate a camera that moved from a
    screen that changed: a screen playing video has no offset that improves
    the match, so its best score stays poor and its shift reads as zero.

    Coarsely first, then finely. A movement too big to find at full size is
    a small one on a picture a quarter as wide, so the ceiling on what can
    be measured stops being the size of the search and becomes the point
    where two frames no longer share enough picture to compare -- which is a
    fact about pictures rather than a number anyone chose.
    """

    raw = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=False,
    ).stdout
    size = WIDTH * HEIGHT
    frames = [raw[i * size:(i + 1) * size] for i in range(len(raw) // size)]
    if len(frames) < 2:
        return []

    small = [_shrink(one, WIDTH, HEIGHT, SHRINK) for one in frames]
    narrow, short = WIDTH // SHRINK, HEIGHT // SHRINK
    # As far as either picture can be slid while still sharing SHARED of
    # itself with the other. Derived, not chosen -- see `_agreement`.
    far = int(narrow * (1.0 - SHARED))
    near = SHRINK

    out: list[tuple[float, float, float]] = []
    for index, (before, after) in enumerate(zip(frames, frames[1:]), start=1):
        rough_x, rough_y = _settle(
            small[index - 1], small[index], narrow, short, 0, 0, far
        )
        across, down = rough_x * SHRINK, rough_y * SHRINK
        # Refine around what the small picture pointed at, within one coarse
        # pixel of it -- which is all the coarse answer can be wrong by.
        for _ in range(2):
            across = min(
                range(across - near, across + near + 1),
                key=lambda x: _agreement(before, after, WIDTH, HEIGHT, x, down),
            )
            down = min(
                range(down - near, down + near + 1),
                key=lambda y: _agreement(before, after, WIDTH, HEIGHT, across, y),
            )
        # How well the best offset actually explained the change. A camera
        # that moved leaves a low residual: slide the frame back and it
        # matches. A lens being uncovered leaves a high one at every offset
        # in both directions, because nothing about that change is a shift --
        # and that is the difference between "the camera was still" and
        # "this was not the camera moving", which were the same answer once.
        residual = _agreement(before, after, WIDTH, HEIGHT, across, down)
        travelled = (across * across + down * down) ** 0.5
        out.append((index / fps, travelled / WIDTH * fps, residual))
    return out


def _into_intervals(
    shifts: list[tuple[float, float, float]], duration: float
) -> list[MotionInterval]:
    """Group a per-frame reading into stretches, with hysteresis.

    Two thresholds rather than one, because a single one turns a camera
    hovering at the edge of it into a rapid alternation of states that
    describes the measurement rather than the camera.
    """

    states: list[tuple[float, str, float]] = []
    now = "still"
    for at, speed, residual in shifts:
        if residual > NOT_A_SHIFT:
            # No offset in either direction explains what changed, so there
            # is no shift to report and "the camera was still" would be a
            # claim rather than a reading. The offset that scored best is
            # dropped with it: it is the least bad of a bad set, not a
            # measurement of anything.
            states.append((at, "not_a_shift", 0.0))
            continue
        if now == "still" and speed > MOVING:
            now = "moving"
        elif now == "moving" and speed < STILL:
            now = "still"
        states.append((at, now, speed))

    grouped: list[list[tuple[float, str, float]]] = []
    for entry in states:
        if grouped and grouped[-1][-1][1] == entry[1]:
            grouped[-1].append(entry)
        else:
            grouped.append([entry])

    # Anything too short to be a state belongs to its neighbour.
    merged: list[list[tuple[float, str, float]]] = []
    for run in grouped:
        span = run[-1][0] - run[0][0]
        if merged and span < LEAST_SECONDS:
            merged[-1].extend(run)
        else:
            merged.append(run)

    # Adjacent runs of the same state are one thing the camera did. They can
    # end up split when a brief blip between them is absorbed into the run
    # before it, and two neighbouring intervals both saying "the frame is
    # moving" reads as a measurement artefact, which is what it is.
    joined: list[list[tuple[float, str, float]]] = []
    for run in merged:
        state = max(set(one[1] for one in run), key=[o[1] for o in run].count)
        if joined and joined[-1][1] == state:
            joined[-1][0].extend(run)
        else:
            joined.append(([*run], state))
    merged = [run for run, _ in joined]

    # Joining may have left neighbours agreeing again.
    joined = []
    for run in merged:
        state = max(set(one[1] for one in run), key=[o[1] for o in run].count)
        if joined and joined[-1][1] == state:
            joined[-1][0].extend(run)
        else:
            joined.append(([*run], state))
    merged = [run for run, _ in joined]

    out: list[MotionInterval] = []
    for index, run in enumerate(merged):
        starts = 0.0 if index == 0 else run[0][0]
        ends = duration if index == len(merged) - 1 else merged[index + 1][0][0]
        speeds = [one[2] for one in run]
        state = max(set(one[1] for one in run), key=[o[1] for o in run].count)
        out.append(
            MotionInterval(
                event_id=f"m{index:02d}",
                starts_seconds=round(starts, 3),
                ends_seconds=round(ends, 3),
                state=state,
                peak_vw_s=round(max(speeds), 4),
                travel_vw=round(sum(speeds) / max(COARSE_FPS, 1e-6), 4),
                settles=(
                    state == "moving"
                    and index + 1 < len(merged)
                    and merged[index + 1][0][1] == "still"
                ),
            )
        )
    return out


def travelled_between(
    intervals: "list[MotionInterval] | tuple[MotionInterval, ...]",
    first: float,
    second: float,
) -> float | None:
    """How far the frame moved between two moments, in frame widths.

    What makes a position measured at one second usable at another. On a
    locked-off take the answer is zero and a coordinate keeps forever; on a
    take whose camera travels it is the distance the thing has drifted, and
    past a point the thing is not in the picture at all.

    `None` when a stretch in between is not a shift at all. Counting those
    as zero was the same error this module was rewritten to stop making: a
    subject seen before a lens is uncovered and used after it would have
    been certified as never having moved, because the one kind of change no
    offset describes contributed nothing to a sum of offsets. Nothing is
    known about that distance, and "nothing is known" is not "nothing".
    """

    low, high = sorted((first, second))
    gone = 0.0
    for one in intervals:
        if one.seconds <= 0.0:
            continue
        shared = max(0.0, min(high, one.ends_seconds) - max(low, one.starts_seconds))
        if shared <= 0.0:
            continue
        if one.state == "not_a_shift":
            return None
        if one.state == "moving":
            if getattr(one, "motion_kind", "translation") != "translation":
                return None
            gone += one.travel_vw * shared / one.seconds
    return gone


def describe(intervals: list[MotionInterval]) -> str:
    """The measurement, as the card pass is shown it.

    Written as intervals with ids so the answer can point at one rather than
    inventing a second. The model is asked what each stretch means, never
    whether it happened.
    """

    if not intervals:
        return ""
    lines = ["本機量到的攝影機運動（用來補足模型動態瀏覽可能跳過的影格）："]
    for one in intervals:
        clock = f"{int(one.starts_seconds) // 60}:{one.starts_seconds % 60:04.1f}"
        until = f"{int(one.ends_seconds) // 60}:{one.ends_seconds % 60:04.1f}"
        if one.state == "moving":
            # How hard, because "the frame moved" covers a slow deliberate
            # push and a knock, and telling those apart is the whole job the
            # model is being given here.
            how = (
                "很輕微" if one.peak_vw_s < 0.06
                else "平穩" if one.peak_vw_s < 0.18
                else "快" if one.peak_vw_s < 0.4
                else "很急"
            )
            said = (
                f"整個畫面在位移（{how}，最快每秒 {one.peak_vw_s:.2f} 個畫面寬，"
                f"總共移動約 {one.travel_vw:.2f} 個畫面寬）"
            )
            if one.confidence > 0.0:
                components = [
                    f"類型={one.motion_kind}",
                    f"方向={one.direction or 'none'}",
                    f"pan={one.pan_vw_s:+.3f} 畫面寬/秒",
                    f"tilt={one.tilt_vh_s:+.3f} 畫面高/秒",
                    f"zoom={one.zoom_rate_s:+.3f}/秒",
                    f"rotation={one.rotation_deg_s:+.2f}°/秒",
                    f"背景共識={one.confidence:.0%}",
                    f"階段={one.phase}",
                ]
                said += "；" + "，".join(components)
        else:
            said = {
                "still": "整段畫面沒有位移",
                # Not "cannot be measured". What is known is narrower and
                # more useful: the picture changed and no shift accounts for
                # it. Every cause of that -- a lens uncovered, a push in, a
                # subject filling the frame -- unfolds over seconds, so it
                # is the one kind of movement the model can see perfectly
                # well without help. Told it was unmeasurable, it reached
                # for `setup_reframe`, and that costs the stretch its span.
                "not_a_shift": (
                    "畫面大幅改變，但不是整體平移"
                    "（推拉變焦、有東西靠近或離開鏡頭、主體佔滿畫面都會這樣）。"
                    "這種變化橫跨好幾秒，你在畫面上看得到——照你看到的判斷，"
                    "不要因為這裡沒有數字就當成拍壞了"
                ),
            }[one.state]
        tail = "，最後停下來落定" if one.settles else ""
        lines.append(f"  {one.event_id}  {clock}–{until}  {said}{tail}")
    return "\n".join(lines)


def cached(source: Path, duration: float, library: Path) -> list[MotionInterval]:
    """Measure once per file, ever.

    Content addressed like the proxy and the card it feeds, because this is
    a fact about the bytes and nothing about the edit.
    """

    from montagewright.uploads import content_hash

    where = library / "motion" / f"{content_hash(source)[:20]}.{READING}.json"
    if where.exists():
        try:
            stored = json.loads(where.read_text(encoding="utf-8"))
            return [MotionInterval(**one) for one in stored]
        except (OSError, ValueError, TypeError):
            pass
    found = measure(source, duration)
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(
        json.dumps([asdict(one) for one in found], indent=2), encoding="utf-8"
    )
    return found
