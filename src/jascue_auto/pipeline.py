"""Run the stages in order and report what happened.

The order is fixed and each stage answers one question:

    rhythm    how long should each shot be, given the music and the material
    subject   where is the thing this shot is about, at a few sampled moments
    reframe   what crop follows that, inside this shot's energy budget
    ground    which musical event does each cut actually land on
    render    the file

Every semantic question goes to the model; every measurement stays local. The
report at the end says which cuts landed on music and which shots had to
settle for less than the plan asked, because a cut that quietly did neither is
indistinguishable from one that did both.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jascue_auto.executor import RenderPlan, Source, plan_render
from jascue_auto.grounding import BeatGrid, apply_to_edl, ground_timeline
from jascue_auto.planner import Usage, decide_rhythm, locate_subject
from jascue_auto.reframe import (
    CropPath,
    Observation,
    OutOfFrame,
    build_crop_path,
)
from jascue_auto.renderer import RenderResult, render
from jascue_auto.schema import EDL, DegradationStep

# Enough samples to see a subject change direction, few enough that one shot
# costs a fraction of a cent. Interpolation covers the gaps; SAM propagation
# replaces this when per-frame accuracy starts to matter.
SUBJECT_SAMPLES = 5


@dataclass
class Report:
    """What the run did, in the terms someone would ask about it."""

    aligned_cuts: int = 0
    total_cuts: int = 0
    following_shots: int = 0
    static_shots: int = 0
    degradations: list[DegradationStep] = field(default_factory=list)
    subject_notes: dict[str, str] = field(default_factory=dict)
    usages: list[Usage] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.usages)

    @property
    def output_tokens(self) -> int:
        return sum(
            usage.output_tokens + usage.thought_tokens for usage in self.usages
        )

    def summary(self) -> str:
        return (
            f"{self.aligned_cuts}/{self.total_cuts} cuts on a musical event, "
            f"{self.following_shots} shots following a subject, "
            f"{self.static_shots} held, "
            f"{len(self.degradations)} degradations, "
            f"{self.input_tokens} in / {self.output_tokens} out tokens"
        )


def probe(source_id: str, path: Path) -> Source:
    import json

    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    return Source(
        source_id=source_id,
        path=path,
        duration_seconds=float(payload["format"]["duration"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def _sample_frames(
    source: Source, start: float, end: float, work: Path
) -> tuple[list[Path], list[float]]:
    """Pull evenly spaced stills across the shot's own window."""

    span = max(end - start, 0.1)
    times = [
        start + span * (index + 0.5) / SUBJECT_SAMPLES
        for index in range(SUBJECT_SAMPLES)
    ]
    frames = []
    for index, at in enumerate(times):
        destination = work / f"{source.source_id}-{index}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{at:.3f}", "-i", str(source.path),
                "-frames:v", "1", "-vf", "scale=960:-2", str(destination),
            ],
            check=True,
        )
        frames.append(destination)
    return frames, times


def follow_subjects(
    edl: EDL,
    sources: dict[str, Source],
    *,
    target_aspect: float,
    report: Report,
    client: Any | None = None,
) -> dict[str, CropPath]:
    """Build a crop path per shot that names a subject.

    A shot whose subject turns out to be still gets a held frame, which is the
    right answer rather than a failure -- the deadband decides that, not a
    threshold anyone has to tune.
    """

    paths: dict[str, CropPath] = {}
    with tempfile.TemporaryDirectory() as raw_work:
        work = Path(raw_work)
        for clip in edl.clips:
            reframe = clip.reframe
            if reframe is None or reframe.subject is None:
                continue
            source = sources[clip.source_id]
            frames, times = _sample_frames(
                source, clip.approx_in_seconds, clip.approx_out_seconds, work
            )
            boxes, usage = locate_subject(
                frames, reframe.subject.description, client=client
            )
            report.usages.append(usage)

            observations = []
            for box in boxes:
                if not box.get("present"):
                    continue
                index = int(box.get("frame_index", -1))
                if not 0 <= index < len(times):
                    continue
                try:
                    observations.append(
                        Observation(
                            seconds=times[index] - clip.approx_in_seconds,
                            centre_x=float(box["centre_x"]),
                            centre_y=float(box["centre_y"]),
                            width=float(box["width"]),
                            height=float(box["height"]),
                        )
                    )
                except (OutOfFrame, KeyError, TypeError, ValueError) as error:
                    # One unusable observation is not a reason to abandon the
                    # shot; the remaining samples still describe the motion.
                    report.subject_notes[clip.clip_id] = str(error)[:160]

            if not observations:
                report.degradations.append(
                    DegradationStep(
                        clip_id=clip.clip_id,
                        ladder="center_crop",
                        trigger=(
                            "the subject was not located in any sampled "
                            "frame, so the shot is framed centrally"
                        ),
                        measured={"samples": float(len(times))},
                    )
                )
                continue

            note = next(
                (
                    str(box["disambiguation"])
                    for box in boxes
                    if box.get("disambiguation")
                ),
                "",
            )
            if note:
                report.subject_notes[clip.clip_id] = note

            path = build_crop_path(
                observations,
                source_aspect=source.aspect_ratio,
                target_aspect=target_aspect,
                energy=reframe.camera_energy,
                clip_id=clip.clip_id,
                degradations=report.degradations,
            )
            paths[clip.clip_id] = path
            if path.is_static:
                report.static_shots += 1
            else:
                report.following_shots += 1
    return paths


def run(
    edl: EDL,
    sources: dict[str, Source],
    grid: BeatGrid,
    output_dir: Path,
    *,
    target_aspect: float,
    intent: str,
    music: Path | None = None,
    decide_rhythm_first: bool = True,
    client: Any | None = None,
) -> tuple[RenderResult, RenderPlan, Report, EDL]:
    """Take an EDL to a finished file.

    The resolved EDL comes back with it. The rhythm pass rewrites durations
    inside this call, and a caller that renders a second aspect from the EDL
    it passed in gets the placeholder lengths instead of the decided ones --
    which looks like a render that worked and sounds like one that ignored the
    music.
    """

    report = Report()

    if decide_rhythm_first:
        edl, usage = decide_rhythm(
            edl, grid, intent=intent, music=music, client=client
        )
        report.usages.append(usage)

    timeline = ground_timeline(edl, grid)
    report.aligned_cuts = timeline.aligned_count
    report.total_cuts = len(timeline.clips)
    edl = apply_to_edl(edl, timeline)

    paths = follow_subjects(
        edl,
        sources,
        target_aspect=target_aspect,
        report=report,
        client=client,
    )

    plan = plan_render(
        edl, sources, target_aspect=target_aspect, crop_paths=paths
    )
    report.degradations.extend(plan.degradations)

    result = render(plan, output_dir, music=music, keep_segments=False)
    return result, plan, report, edl


def split_handoffs(edl: EDL) -> EDL:
    """Turn a two-subject shot into two shots, each with one subject.

    A camera cannot follow one thing and then another inside a single move
    without losing both: the first subject is abandoned mid-gesture and the
    second is arrived at late. Splitting at the plan layer gives each half its
    own frame and its own follow, and the join between them is a cut, which is
    how an editor carries the eye from one thing to the next anyway.

    Each half keeps the parent's rhythm decision, halved, so a handoff costs
    the same screen time it was given.
    """

    rewritten: list[Any] = []
    for clip in edl.clips:
        reframe = clip.reframe
        second = getattr(reframe, "then_subject", None) if reframe else None
        if reframe is None or second is None:
            rewritten.append(clip)
            continue

        midpoint = (clip.approx_in_seconds + clip.approx_out_seconds) / 2.0
        first_half = clip.model_copy(
            update={
                "clip_id": f"{clip.clip_id}a",
                "approx_out_seconds": midpoint,
            }
        )
        second_half = clip.model_copy(
            update={
                "clip_id": f"{clip.clip_id}b",
                "approx_in_seconds": midpoint,
                "reframe": reframe.model_copy(update={"subject": second}),
            }
        )
        rewritten += [first_half, second_half]
    return edl.model_copy(update={"clips": rewritten})
