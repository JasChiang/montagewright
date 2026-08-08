"""Where the camera moved, measured rather than watched.

Gemini reads video at a frame a second. Camera shake happens between frames,
so at that rate a handheld take and a locked-off one are the same thing: a
series of individually sharp stills that differ a little. Asked whether a
clip is stable, it answered "固定鏡頭…畫面穩定清晰" about a take whose first
three seconds are the operator still finding the frame -- not carelessly, but
because the question cannot be answered from what it was given.

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
frame agreed with it -- and says `unobservable` when too little of the frame
holds still enough to tell.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

# Coarse enough to be cheap on seventy-four clips, fine enough to see a
# camera settle: shake lives well above this and a pan is visible at any
# rate.
COARSE_FPS = 4.0

# Width to analyse at. Global motion is a whole-frame property and survives
# scaling; the cost does not.
WIDTH = 192

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
UNREADABLE = 15.0

# Shortest stretch worth calling a state. Below it the reading is a flicker
# in the measurement rather than a thing the camera did.
LEAST_SECONDS = 0.5


@dataclass(frozen=True)
class MotionInterval:
    """One stretch of a clip during which the camera did one thing."""

    event_id: str
    starts_seconds: float
    ends_seconds: float
    state: str          # still | moving | unobservable
    peak_vw_s: float    # fastest global shift, frame widths per second
    travel_vw: float    # total global shift across the interval
    settles: bool       # ends still, having been moving

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

    shifts = _global_shift(source, COARSE_FPS)
    if not shifts:
        return [
            MotionInterval(
                event_id="m00", starts_seconds=0.0, ends_seconds=duration,
                state="unobservable", peak_vw_s=0.0, travel_vw=0.0,
                settles=False,
            )
        ]
    return _into_intervals(shifts, duration)


def _global_shift(source: Path, fps: float) -> list[tuple[float, float, float]]:
    """(seconds, shift) per sampled frame, shift in frame widths per second.

    Estimated by matching each frame against the one before it at a handful
    of horizontal offsets and taking the offset that agrees best across the
    frame. Crude next to optical flow and enough to separate a camera that
    moved from a screen that changed: a screen playing video has no offset
    that improves the match, so its best score stays poor and its shift
    reads as zero.
    """

    raw = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={fps},scale={WIDTH}:108,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=False,
    ).stdout
    size = WIDTH * 108
    frames = [raw[i * size:(i + 1) * size] for i in range(len(raw) // size)]
    if len(frames) < 2:
        return []

    out: list[tuple[float, float, float]] = []
    reach = max(2, WIDTH // 24)
    for index, (before, after) in enumerate(zip(frames, frames[1:]), start=1):
        best, offset = None, 0
        for shift in range(-reach, reach + 1):
            total = count = 0
            for row in range(6, 108, 12):
                base = row * WIDTH
                lo = max(0, -shift) + 4
                hi = min(WIDTH, WIDTH - shift) - 4
                for column in range(lo, hi, 3):
                    total += abs(
                        before[base + column] - after[base + column + shift]
                    )
                    count += 1
            score = total / max(count, 1)
            if best is None or score < best:
                best, offset = score, shift
        # How well the best offset actually explained the change. A camera
        # that moved leaves a low residual: shift the frame back and it
        # matches. A screen playing video leaves a high one at every offset,
        # because nothing about the change is a shift -- and that is the
        # difference between "the camera was still" and "this cannot be
        # read from the picture", which were the same answer before.
        out.append((index / fps, abs(offset) / WIDTH * fps, best or 0.0))
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
        if residual > UNREADABLE:
            # No offset explains what changed, so there is no shift to
            # report and "the camera was still" would be a claim rather
            # than a reading.
            states.append((at, "unobservable", speed))
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


def describe(intervals: list[MotionInterval]) -> str:
    """The measurement, as the card pass is shown it.

    Written as intervals with ids so the answer can point at one rather than
    inventing a second. The model is asked what each stretch means, never
    whether it happened.
    """

    if not intervals:
        return ""
    lines = ["本機量到的攝影機運動（你看不到這個，因為影片是一秒一格給你的）："]
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
        else:
            said = {
                "still": "整段畫面沒有位移",
                "unobservable": "量不出來（畫面裡沒有夠穩定的東西可以比對）",
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

    where = library / "motion" / f"{content_hash(source)[:20]}.json"
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
