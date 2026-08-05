"""Resolve symbolic intent into exact times.

The planner says "eight beats, cut on the beat, land on the chorus". It never
says 12.484 seconds. Names and counts resolve here, against a beat grid the
local analyser measured, so the grid owns timing and the planner owns meaning.

The previous system worked the other way: the model sent seconds, and a solver
tried afterwards to slide each cut onto a nearby beat. Measured on real
material, that produced zero aligned boundaries out of eleven -- not because
beats were scarce (the nearest one was three milliseconds away in one case)
but because durations had been stretched to their maximum to hit a target
length, leaving no room to move. Timing derived from the grid cannot get into
that position, because it never has to move anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jascue_auto.schema import EDL, Clip

# Cue kinds worth cutting on. A plain beat is included because an eight-beat
# clip has to land somewhere even when no accent falls there.
CUTTABLE = frozenset({"section_boundary", "downbeat", "accent", "beat"})


@dataclass(frozen=True)
class Cue:
    cue_id: str
    time_seconds: float
    kind: str
    strength: float = 0.0


@dataclass(frozen=True)
class BeatGrid:
    """Measured musical time. Everything symbolic resolves against this."""

    bpm: float
    meter: int
    cues: tuple[Cue, ...]
    duration_seconds: float

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def named_points(self) -> dict[str, float]:
        """Section names the planner may target by name."""

        return {
            cue.cue_id: cue.time_seconds
            for cue in self.cues
            if cue.kind == "section_boundary"
        }

    def cuttable(self) -> tuple[Cue, ...]:
        return tuple(cue for cue in self.cues if cue.kind in CUTTABLE)

    def nearest_cue(self, seconds: float) -> Cue | None:
        candidates = self.cuttable()
        if not candidates:
            return None
        return min(
            candidates, key=lambda cue: abs(cue.time_seconds - seconds)
        )

    def cue_after(self, seconds: float) -> Cue | None:
        later = [
            cue for cue in self.cuttable() if cue.time_seconds > seconds + 1e-6
        ]
        return later[0] if later else None


def analyse_track(path: Path) -> BeatGrid:
    """Measure a track straight into a grid.

    The reviewed lock exists so a delivery can prove which analysis it was
    cut against. That provenance is worth its ceremony for a delivery and is
    pure friction for someone dropping a folder on a page to see what comes
    out, who would otherwise have to produce a lock file before the tool
    would run at all. The measurement underneath is the same one.
    """

    from jascue_video_lab.music import analyze_music

    proposal = analyze_music(Path(path).expanduser().resolve())
    rate = float(proposal.master_sample_rate)
    meter = int(proposal.meter_suggestion or 4)

    # The proposal calls everything a candidate because a human was meant to
    # settle it. Nothing about the measurement changes when they do -- the
    # review decides which to keep, not when they are.
    cues = [
        Cue(
            cue_id=str(cue.cue_id),
            time_seconds=float(cue.time_ms) / 1000.0,
            kind=str(cue.kind).removesuffix("_candidate"),
            strength=float(cue.strength or 0.0),
        )
        for cue in proposal.cues
    ]
    # Downbeats and section boundaries are derived while locking rather than
    # measured, so they have to be derived here too: without them the grid
    # has nothing a planner can name, and every "land on the chorus" resolves
    # to nothing.
    beats = sorted(
        (cue for cue in cues if cue.kind == "beat"),
        key=lambda cue: cue.time_seconds,
    )
    for index, beat in enumerate(beats):
        if index % meter == 0:
            cues.append(
                Cue(
                    cue_id=f"downbeat-{index // meter:05d}",
                    time_seconds=beat.time_seconds,
                    kind="downbeat",
                    strength=beat.strength,
                )
            )
    for section in proposal.sections:
        cues.append(
            Cue(
                cue_id=str(section.label or section.section_id),
                time_seconds=float(section.start_sample) / rate,
                kind="section_boundary",
                strength=float(section.confidence or 0.0),
            )
        )
    return BeatGrid(
        bpm=float(proposal.estimated_bpm or 120.0),
        meter=meter,
        cues=tuple(sorted(cues, key=lambda cue: cue.time_seconds)),
        duration_seconds=float(proposal.duration_ms) / 1000.0,
    )


def load_beat_grid(lock_path: Path) -> BeatGrid:
    """Read a reviewed music map lock into a grid.

    The lock is the old package's format and is genuinely a measurement, so it
    is read directly rather than re-derived.
    """

    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    cues = tuple(
        Cue(
            cue_id=str(entry["cue_id"]),
            time_seconds=float(entry["time_ms"]) / 1000.0,
            kind=str(entry["kind"]),
            strength=float(entry.get("strength") or 0.0),
        )
        for entry in payload.get("cues", [])
    )
    return BeatGrid(
        bpm=float(payload["bpm"]),
        meter=int(payload.get("meter") or 4),
        cues=cues,
        duration_seconds=float(payload.get("duration_ms", 0)) / 1000.0,
    )


@dataclass
class GroundedClip:
    """A clip with its place on the timeline decided."""

    clip: Clip
    timeline_in_seconds: float
    timeline_out_seconds: float
    landed_on: str | None = None
    note: str | None = None
    # Set when a camera move lengthened this shot past what the rhythm asked
    # for, so the report can say why the cut runs where it does.
    move_too_short: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.timeline_out_seconds - self.timeline_in_seconds


@dataclass
class GroundedTimeline:
    clips: list[GroundedClip]
    grid: BeatGrid | None

    @property
    def duration_seconds(self) -> float:
        return self.clips[-1].timeline_out_seconds if self.clips else 0.0

    @property
    def aligned_count(self) -> int:
        """Cuts that landed on a musical event, for the delivery report."""

        return sum(1 for clip in self.clips if clip.landed_on is not None)


def _requested_duration(clip: Clip, grid: BeatGrid | None) -> float:
    """How long this clip wants to be, in seconds.

    The length is the planner's, whole. A flat per-move floor used to raise it
    here, which reads as safety and is a length decision made by a constant:
    how long a sweep needs depends on how far it travels and how much is on
    the way, and the planner is the one who watched the shot. What the floor
    is good for is saying afterwards that the move could not happen in the
    time it was given -- reported, not corrected.
    """

    if grid is not None and clip.music_sync.beats:
        return clip.music_sync.beats * grid.seconds_per_beat
    return clip.approx_out_seconds - clip.approx_in_seconds


def ground_timeline(edl: EDL, grid: BeatGrid | None) -> GroundedTimeline:
    """Lay the clips out in time.

    Each clip starts where the previous one ended, so the cut is continuous by
    construction. Its end is the next musical event at or after its requested
    length -- so a beat count is honoured exactly when the grid agrees, and
    rounded up to the next event when it does not, rather than being nudged
    off the grid to hit an arithmetic total.
    """

    grounded: list[GroundedClip] = []
    cursor = 0.0

    from jascue_auto.capabilities import MOVE_FLOORS

    for clip in edl.clips:
        wanted = _requested_duration(clip, grid)
        move = clip.reframe.camera_move if clip.reframe else "hold"
        floor = MOVE_FLOORS.get(move, 0.0)
        # Not a correction. The move stays as asked and runs in the time it
        # was given; this says it will not read, so the report and the review
        # round see it instead of a shot that quietly arrives too fast.
        too_short = (
            f"{move} needs about {floor:g}s to register and has "
            f"{wanted:.2f}s"
            if floor > 0.0 and wanted < floor - 1e-6
            else None
        )
        end = cursor + wanted
        landed: str | None = None
        note: str | None = None

        if grid is not None and clip.music_sync.cut_on_beat:
            cue = grid.nearest_cue(end)
            if cue is not None and cue.time_seconds > cursor + 1e-6:
                drift = abs(cue.time_seconds - end)
                end = cue.time_seconds
                landed = cue.cue_id
                if drift > grid.seconds_per_beat / 2:
                    note = (
                        f"nearest cue sat {drift:.3f}s from the requested "
                        f"length; took it anyway to stay on the grid"
                    )
            else:
                # Every cue lies behind the cursor, i.e. the music has run
                # out. Keeping the requested length is honest; silence at the
                # tail is the mix's problem, not the edit's.
                note = "past the end of the analysed music; kept as planned"

        grounded.append(
            GroundedClip(
                clip=clip,
                timeline_in_seconds=cursor,
                timeline_out_seconds=end,
                landed_on=landed,
                note=note,
                move_too_short=too_short,
            )
        )
        cursor = end

    return GroundedTimeline(clips=grounded, grid=grid)


def apply_to_edl(edl: EDL, timeline: GroundedTimeline) -> EDL:
    """Rewrite each clip's source window to the length the grid decided.

    The in-point is what the planner chose -- that is a statement about the
    material. The out-point follows from the grounded duration, because a cut
    is a straight copy: one second of timeline is one second of source.

    A source that cannot supply the full grounded length is left as it is. The
    executor clamps it and records the shortfall, which keeps the shortening
    visible instead of silently reopening it here.
    """

    grounded_by_id = {
        entry.clip.clip_id: entry.duration_seconds for entry in timeline.clips
    }
    rewritten = []
    for clip in edl.clips:
        wanted = grounded_by_id.get(clip.clip_id)
        if wanted is None:
            rewritten.append(clip)
            continue
        rewritten.append(
            clip.model_copy(
                update={
                    "approx_out_seconds": clip.approx_in_seconds + wanted
                }
            )
        )
    return edl.model_copy(update={"clips": rewritten})


def resolve_sync_point(grid: BeatGrid, name: str) -> float | None:
    """Look up a named point such as 'chorus_1_start'.

    Unknown names return None rather than raising: a planner naming a section
    this track does not have is describing intent the material cannot serve,
    which is a note for the report, not a crash.
    """

    points = grid.named_points
    if name in points:
        return points[name]
    lowered = name.lower()
    for cue_id, seconds in points.items():
        if lowered in cue_id.lower():
            return seconds
    return None
