from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Sequence

from .media import sha256_file
from .models import (
    MusicAssemblyArtifactBinding,
    MusicAssemblyCueInstance,
    MusicAssemblyPlan,
    MusicAssemblyRenderManifest,
    MusicAssemblySpan,
    MusicDuckingRegionV2,
    MusicEditEndingV2,
    MusicEditJoinV2,
    MusicEditPlanV2,
    MusicEditRenderManifestV2,
    MusicEditSpanV2,
)
from .music import MusicMapLock
from .storage import read_json, utc_now, write_json


MUSIC_ASSEMBLY_PLAN_FILENAME = "music-assembly-plan.json"
MUSIC_ASSEMBLY_BINDING_FILENAME = "music-assembly-plan.binding.json"
MUSIC_ASSEMBLY_RENDER_FILENAME = "music-assembly-render.json"
MUSIC_EDIT_PLAN_V2_FILENAME = "music-edit-plan.v2.json"
MUSIC_EDIT_RENDER_V2_FILENAME = "music-edit-render.v2.json"


class MusicAssemblyError(RuntimeError):
    pass


@dataclass(frozen=True)
class MusicAssemblyArtifactPaths:
    plan_path: Path
    binding_path: Path


@dataclass(frozen=True)
class MusicAssemblyRenderResult:
    output_audio_path: Path
    manifest_path: Path
    manifest: MusicAssemblyRenderManifest


@dataclass(frozen=True)
class MusicEditSegmentRequestV2:
    """A semantic passage request resolved to reviewed local boundaries."""

    section_id: str
    semantic_role: Literal[
        "intro",
        "establish",
        "build",
        "climax",
        "release",
        "outro",
        "neutral",
    ] = "neutral"
    energy_band: Literal["low", "medium", "high", "unknown"] = "unknown"
    start_cue_id: str | None = None
    end_cue_id: str | None = None


@dataclass(frozen=True)
class MusicEditJoinRequestV2:
    join_type: Literal["cut", "micro_crossfade"]
    alignment: Literal[
        "section_boundary",
        "phrase_grid",
        "downbeat",
        "accent",
        "transient",
    ]
    energy_transition: Literal[
        "matched",
        "rising",
        "falling",
        "intentional_contrast",
        "unknown",
    ]
    editorial_reason: str
    crossfade_ms: int = 0


@dataclass(frozen=True)
class MusicEditRenderResultV2:
    output_audio_path: Path
    manifest_path: Path
    manifest: MusicEditRenderManifestV2


@dataclass(frozen=True)
class _IntervalCandidate:
    score: tuple[int, ...]
    source_start: int
    source_end: int
    start_boundary_kind: Literal[
        "track_start",
        "section_boundary",
        "phrase_grid",
    ]
    end_boundary_kind: Literal["phrase_grid", "natural_track_end"]
    start_bar_index: int | None = None
    end_bar_index: int | None = None
    bar_count: int | None = None
    phrase_bar_multiple: int | None = None


def _canonical_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def music_assembly_plan_sha256(plan: MusicAssemblyPlan) -> str:
    """Return the canonical plan hash used by render and delivery evidence."""

    validated = MusicAssemblyPlan.model_validate(plan.model_dump(mode="json"))
    return _canonical_hash(validated)


def _milliseconds_to_samples(milliseconds: int, sample_rate: int) -> int:
    if milliseconds <= 0:
        raise ValueError("music assembly durations must be positive")
    return round(Fraction(milliseconds * sample_rate, 1000))


def _load_bound_music_lock(
    music_lock: MusicMapLock,
    music_lock_path: Path,
) -> tuple[Path, str]:
    resolved = music_lock_path.expanduser().resolve(strict=True)
    saved = MusicMapLock.model_validate(read_json(resolved))
    if saved != music_lock:
        raise ValueError("in-memory music lock differs from the saved artifact")
    return resolved, sha256_file(resolved)


def _bar_boundaries(music_lock: MusicMapLock) -> list[tuple[int, int]]:
    """Return reviewed bar-grid boundaries as (bar_index, source_sample)."""

    beat_period = Fraction(
        music_lock.master_sample_rate * 60_000,
        round(music_lock.bpm * 1000),
    )
    bar_period = beat_period * music_lock.meter
    boundaries: list[tuple[int, int]] = []
    bar_index = 0
    while True:
        sample = music_lock.first_downbeat_sample + round(bar_period * bar_index)
        if sample > music_lock.duration_samples:
            break
        boundaries.append((bar_index, sample))
        if sample == music_lock.duration_samples:
            break
        bar_index += 1
    return boundaries


def _phrase_preference(
    *,
    start_bar_index: int,
    bar_count: int,
    preferred_phrase_bars: tuple[int, ...],
) -> tuple[int, int] | None:
    for rank, phrase_bars in enumerate(preferred_phrase_bars):
        if start_bar_index % phrase_bars == 0 and bar_count % phrase_bars == 0:
            return rank, phrase_bars
    return None


def _start_phrase_preference(
    *,
    start_bar_index: int,
    preferred_phrase_bars: tuple[int, ...],
) -> tuple[int, int] | None:
    for rank, phrase_bars in enumerate(preferred_phrase_bars):
        if start_bar_index % phrase_bars == 0:
            return rank, phrase_bars
    return None


def plan_single_interval_music_assembly(
    music_lock: MusicMapLock,
    *,
    music_lock_path: Path,
    target_duration_ms: int,
    minimum_duration_ms: int,
    maximum_duration_ms: int,
    preferred_phrase_bars: tuple[int, ...] = (8, 4, 2, 1),
) -> MusicAssemblyPlan:
    """Choose one phrase-aligned source interval without creating music joins.

    Version 1 deliberately supports exactly one half-open interval from one
    reviewed music timeline. It can shorten a track, but it cannot concatenate
    distant passages or silently time-stretch the source.
    """

    if not preferred_phrase_bars:
        raise ValueError("at least one preferred phrase-bar value is required")
    if len(set(preferred_phrase_bars)) != len(preferred_phrase_bars):
        raise ValueError("preferred phrase-bar values must be unique")
    if any(value <= 0 for value in preferred_phrase_bars):
        raise ValueError("preferred phrase-bar values must be positive")
    if not minimum_duration_ms <= target_duration_ms <= maximum_duration_ms:
        raise ValueError("target duration must lie inside the requested range")

    resolved_lock, lock_sha256 = _load_bound_music_lock(
        music_lock,
        music_lock_path,
    )
    sample_rate = music_lock.master_sample_rate
    target_samples = _milliseconds_to_samples(target_duration_ms, sample_rate)
    minimum_samples = _milliseconds_to_samples(minimum_duration_ms, sample_rate)
    maximum_samples = _milliseconds_to_samples(maximum_duration_ms, sample_rate)
    boundaries = _bar_boundaries(music_lock)
    if len(boundaries) < 2:
        raise MusicAssemblyError(
            "reviewed music grid does not contain two complete bar boundaries"
        )

    candidates: list[_IntervalCandidate] = []

    # Preserve a real source ending whenever a reviewed semantic section or a
    # phrase-grid start yields an acceptable continuous duration. The final
    # tail may extend beyond the last complete bar; it remains source audio,
    # not a generated or spliced ending.
    natural_start_samples: set[int] = set()
    for section in music_lock.sections:
        source_start = section.start_sample
        natural_start_samples.add(source_start)
        duration = music_lock.duration_samples - source_start
        if not minimum_samples <= duration <= maximum_samples:
            continue
        start_kind = "track_start" if source_start == 0 else "section_boundary"
        start_kind_rank = 1 if source_start == 0 else 0
        candidates.append(
            _IntervalCandidate(
                score=(
                    0,
                    start_kind_rank,
                    abs(duration - target_samples),
                    source_start,
                ),
                source_start=source_start,
                source_end=music_lock.duration_samples,
                start_boundary_kind=start_kind,
                end_boundary_kind="natural_track_end",
            )
        )
    for start_bar, source_start in boundaries[:-1]:
        if source_start in natural_start_samples:
            continue
        duration = music_lock.duration_samples - source_start
        if not minimum_samples <= duration <= maximum_samples:
            continue
        preference = _start_phrase_preference(
            start_bar_index=start_bar,
            preferred_phrase_bars=preferred_phrase_bars,
        )
        if preference is None:
            continue
        preference_rank, phrase_bars = preference
        candidates.append(
            _IntervalCandidate(
                score=(
                    0,
                    2 + preference_rank,
                    abs(duration - target_samples),
                    source_start,
                ),
                source_start=source_start,
                source_end=music_lock.duration_samples,
                start_boundary_kind="phrase_grid",
                end_boundary_kind="natural_track_end",
                start_bar_index=start_bar,
                phrase_bar_multiple=phrase_bars,
            )
        )

    for start_position, (start_bar, source_start) in enumerate(boundaries[:-1]):
        for end_bar, source_end in boundaries[start_position + 1 :]:
            duration = source_end - source_start
            if duration < minimum_samples:
                continue
            if duration > maximum_samples:
                break
            bar_count = end_bar - start_bar
            preference = _phrase_preference(
                start_bar_index=start_bar,
                bar_count=bar_count,
                preferred_phrase_bars=preferred_phrase_bars,
            )
            if preference is None:
                continue
            preference_rank, phrase_bars = preference
            candidates.append(
                _IntervalCandidate(
                    score=(
                        1,
                        preference_rank,
                        abs(duration - target_samples),
                        start_bar,
                        bar_count,
                    ),
                    source_start=source_start,
                    source_end=source_end,
                    start_boundary_kind="phrase_grid",
                    end_boundary_kind="phrase_grid",
                    start_bar_index=start_bar,
                    end_bar_index=end_bar,
                    bar_count=bar_count,
                    phrase_bar_multiple=phrase_bars,
                )
            )

    if not candidates:
        raise MusicAssemblyError(
            "no single reviewed-boundary source interval satisfies the requested duration"
        )

    selected = min(candidates, key=lambda item: item.score)
    source_start = selected.source_start
    source_end = selected.source_end
    output_duration = source_end - source_start
    cue_by_sample = {cue.sample_index: cue.cue_id for cue in music_lock.cues}
    span = MusicAssemblySpan(
        span_id="music-span-001",
        source_start_sample=source_start,
        source_end_sample=source_end,
        output_start_sample=0,
        output_end_sample=output_duration,
        start_boundary_kind=selected.start_boundary_kind,
        end_boundary_kind=selected.end_boundary_kind,
        start_bar_index=selected.start_bar_index,
        end_bar_index=selected.end_bar_index,
        bar_count=selected.bar_count,
        phrase_bar_multiple=selected.phrase_bar_multiple,
        start_boundary_cue_id=cue_by_sample.get(source_start),
        end_boundary_cue_id=(
            cue_by_sample.get(source_end)
            if selected.end_boundary_kind == "phrase_grid"
            else None
        ),
    )
    selected_cues = [
        cue
        for cue in music_lock.cues
        if source_start <= cue.sample_index < source_end
    ]
    cue_instances = [
        MusicAssemblyCueInstance(
            cue_instance_id=f"music-cue-instance-{index:05d}",
            source_cue_id=cue.cue_id,
            span_id=span.span_id,
            kind=cue.kind,
            priority=cue.priority,
            source_sample_index=cue.sample_index,
            output_sample_index=cue.sample_index - source_start,
            strength=cue.strength,
        )
        for index, cue in enumerate(selected_cues, start=1)
    ]
    assembly_definition = {
        "music_lock_sha256": lock_sha256,
        "music_definition_sha256": music_lock.definition_sha256,
        "target_duration_samples": target_samples,
        "minimum_duration_samples": minimum_samples,
        "maximum_duration_samples": maximum_samples,
        "preferred_phrase_bars": preferred_phrase_bars,
        "source_start_sample": source_start,
        "source_end_sample": source_end,
        "start_boundary_kind": selected.start_boundary_kind,
        "end_boundary_kind": selected.end_boundary_kind,
    }
    return MusicAssemblyPlan(
        assembly_id=f"music-assembly:{_canonical_hash(assembly_definition)}",
        music_id=music_lock.music_id,
        music_lock_path=str(resolved_lock),
        music_lock_sha256=lock_sha256,
        music_definition_sha256=music_lock.definition_sha256,
        master_sample_rate=sample_rate,
        source_duration_samples=music_lock.duration_samples,
        target_duration_samples=target_samples,
        minimum_duration_samples=minimum_samples,
        maximum_duration_samples=maximum_samples,
        output_duration_samples=output_duration,
        target_duration_error_samples=abs(output_duration - target_samples),
        ending_policy=(
            "preserve_natural_track_end_no_fade_out"
            if selected.end_boundary_kind == "natural_track_end"
            else "short_fade_at_phrase_grid_boundary"
        ),
        preferred_phrase_bars=preferred_phrase_bars,
        spans=[span],
        cue_instances=cue_instances,
        uncertainties=[
            "Entrances use locked section boundaries or the human-approved tempo, meter, and first-downbeat grid; musical phrase semantics are not inferred.",
            "MusicAssemblyPlan v1 uses one continuous source interval and cannot splice distant passages or time-stretch audio.",
            "A human editor must review the selected opening, ending, and musical fit before rendering.",
        ],
        generated_at=utc_now(),
    )


def music_edit_plan_v2_sha256(plan: MusicEditPlanV2) -> str:
    """Return the canonical hash for the reviewed multi-passage plan."""

    validated = MusicEditPlanV2.model_validate(plan.model_dump(mode="json"))
    return _canonical_hash(validated)


def plan_contiguous_reviewed_music_edit_v2(
    music_lock: MusicMapLock,
    *,
    music_lock_path: Path,
    target_duration_ms: int,
    minimum_duration_ms: int,
    maximum_duration_ms: int,
    ending_fade_out_ms: int = 120,
) -> MusicEditPlanV2:
    """Fit a continuous reviewed section run with bounded micro-crossfades.

    This is the deterministic fallback when no single phrase-grid interval can
    match picture duration closely enough. It never reorders or repeats source
    audio. Up to four adjacent reviewed sections are retained in source order;
    a locked cue may close the final section and tiny crossfades distributed
    across section boundaries absorb only the residual duration mismatch.
    """

    if not minimum_duration_ms <= target_duration_ms <= maximum_duration_ms:
        raise ValueError("target duration must lie inside the requested range")
    sample_rate = music_lock.master_sample_rate
    target_samples = _milliseconds_to_samples(target_duration_ms, sample_rate)
    minimum_samples = _milliseconds_to_samples(minimum_duration_ms, sample_rate)
    maximum_samples = _milliseconds_to_samples(maximum_duration_ms, sample_rate)
    cue_candidates_by_section: dict[str, list[tuple[int, str | None]]] = {}
    for section in music_lock.sections:
        candidates_by_sample = {
            cue.sample_index: cue.cue_id
            for cue in music_lock.cues
            if section.start_sample < cue.sample_index <= section.end_sample
        }
        candidates_by_sample.setdefault(section.end_sample, None)
        cue_candidates_by_section[section.section_id] = sorted(
            candidates_by_sample.items()
        )

    candidates: list[
        tuple[
            tuple[int, int, int, int],
            list[MusicEditSegmentRequestV2],
            list[MusicEditJoinRequestV2],
        ]
    ] = []
    sections = list(music_lock.sections)
    for start_index, start_section in enumerate(sections):
        for end_index in range(
            start_index + 1,
            min(len(sections), start_index + 4),
        ):
            end_section = sections[end_index]
            join_count = end_index - start_index
            source_start = start_section.start_sample
            for source_end, end_cue_id in cue_candidates_by_section[
                end_section.section_id
            ]:
                raw_duration = source_end - source_start
                if raw_duration < minimum_samples:
                    continue
                desired_reduction_samples = max(
                    0,
                    raw_duration - target_samples,
                )
                desired_reduction_ms = round(
                    desired_reduction_samples * 1000 / sample_rate
                )
                if desired_reduction_ms > join_count * 200:
                    continue
                if 0 < desired_reduction_ms < 5:
                    desired_reduction_ms = 0
                crossfade_ms = [0] * join_count
                if desired_reduction_ms:
                    base, remainder = divmod(
                        desired_reduction_ms,
                        join_count,
                    )
                    crossfade_ms = [
                        base + (1 if index < remainder else 0)
                        for index in range(join_count)
                    ]
                    if any(
                        value != 0 and not 5 <= value <= 200
                        for value in crossfade_ms
                    ):
                        continue
                reduction_samples = sum(
                    round(Fraction(value * sample_rate, 1000))
                    for value in crossfade_ms
                )
                output_duration = raw_duration - reduction_samples
                if not minimum_samples <= output_duration <= maximum_samples:
                    continue
                selected_sections = sections[start_index : end_index + 1]
                segment_requests = [
                    MusicEditSegmentRequestV2(
                        section_id=section.section_id,
                        semantic_role=(
                            "intro"
                            if index == 0
                            else "release"
                            if index == len(selected_sections) - 1
                            else "build"
                        ),
                        energy_band="unknown",
                        end_cue_id=(
                            end_cue_id
                            if index == len(selected_sections) - 1
                            and source_end != section.end_sample
                            else None
                        ),
                    )
                    for index, section in enumerate(selected_sections)
                ]
                join_requests = [
                    MusicEditJoinRequestV2(
                        join_type=(
                            "micro_crossfade" if duration_ms else "cut"
                        ),
                        alignment="section_boundary",
                        energy_transition="unknown",
                        editorial_reason=(
                            "Preserve adjacent reviewed source sections while "
                            "absorbing only the picture-duration residual."
                        ),
                        crossfade_ms=duration_ms,
                    )
                    for duration_ms in crossfade_ms
                ]
                candidates.append(
                    (
                        (
                            start_index,
                            abs(output_duration - target_samples),
                            join_count,
                            source_end,
                        ),
                        segment_requests,
                        join_requests,
                    )
                )
    if not candidates:
        raise MusicAssemblyError(
            "no contiguous reviewed-section music edit satisfies the requested duration"
        )
    _, segments, joins = min(candidates, key=lambda item: item[0])
    return plan_reviewed_music_edit_v2(
        music_lock,
        music_lock_path=music_lock_path,
        segments=segments,
        joins=joins,
        target_duration_ms=target_duration_ms,
        minimum_duration_ms=minimum_duration_ms,
        maximum_duration_ms=maximum_duration_ms,
        ending_mode="phrase_fade_out",
        ending_fade_out_ms=ending_fade_out_ms,
    )


def plan_reviewed_music_edit_v2(
    music_lock: MusicMapLock,
    *,
    music_lock_path: Path,
    segments: Sequence[MusicEditSegmentRequestV2],
    joins: Sequence[MusicEditJoinRequestV2],
    target_duration_ms: int,
    minimum_duration_ms: int,
    maximum_duration_ms: int,
    ending_mode: Literal[
        "natural_track_end",
        "phrase_fade_out",
        "reviewed_ending_hit",
    ]
    | None = None,
    ending_fade_out_ms: int = 80,
    ducking_regions: Sequence[MusicDuckingRegionV2] = (),
) -> MusicEditPlanV2:
    """Resolve reviewed section/cue IDs into an auditable multi-span edit.

    A semantic planner may propose section and cue IDs, but exact sample
    boundaries are always looked up in the reviewed MusicMapLock. This function
    never invents a sample position, repeats source passages, or time-stretches
    music.
    """

    if not 1 <= len(segments) <= 4:
        raise ValueError("music edit v2 requires between one and four passages")
    if len(joins) != len(segments) - 1:
        raise ValueError("music edit v2 requires one join between passages")
    if not minimum_duration_ms <= target_duration_ms <= maximum_duration_ms:
        raise ValueError("target duration must lie inside the requested range")

    resolved_lock, lock_sha256 = _load_bound_music_lock(
        music_lock,
        music_lock_path,
    )
    section_by_id = {section.section_id: section for section in music_lock.sections}
    cue_by_id = {cue.cue_id: cue for cue in music_lock.cues}
    sample_rate = music_lock.master_sample_rate
    target_samples = _milliseconds_to_samples(target_duration_ms, sample_rate)
    minimum_samples = _milliseconds_to_samples(minimum_duration_ms, sample_rate)
    maximum_samples = _milliseconds_to_samples(maximum_duration_ms, sample_rate)

    resolved_passages: list[dict[str, object]] = []
    for request in segments:
        try:
            section = section_by_id[request.section_id]
        except KeyError as error:
            raise MusicAssemblyError(
                f"unknown reviewed music section: {request.section_id}"
            ) from error
        if request.start_cue_id is None:
            source_start = section.start_sample
            start_kind = (
                "track_start" if source_start == 0 else "section_boundary"
            )
        else:
            try:
                start_cue = cue_by_id[request.start_cue_id]
            except KeyError as error:
                raise MusicAssemblyError(
                    f"unknown reviewed start cue: {request.start_cue_id}"
                ) from error
            source_start = start_cue.sample_index
            start_kind = "locked_cue"
        if request.end_cue_id is None:
            source_end = section.end_sample
            end_kind = (
                "natural_track_end"
                if source_end == music_lock.duration_samples
                else "section_boundary"
            )
        else:
            try:
                end_cue = cue_by_id[request.end_cue_id]
            except KeyError as error:
                raise MusicAssemblyError(
                    f"unknown reviewed end cue: {request.end_cue_id}"
                ) from error
            source_end = end_cue.sample_index
            end_kind = "locked_cue"
        if not (
            section.start_sample
            <= source_start
            < source_end
            <= section.end_sample
        ):
            raise MusicAssemblyError(
                f"passage boundaries lie outside section {request.section_id}"
            )
        resolved_passages.append(
            {
                "request": request,
                "source_start": source_start,
                "source_end": source_end,
                "start_kind": start_kind,
                "end_kind": end_kind,
            }
        )

    output_cursor = 0
    edit_spans: list[MusicEditSpanV2] = []
    edit_joins: list[MusicEditJoinV2] = []
    for index, passage in enumerate(resolved_passages):
        request = passage["request"]
        assert isinstance(request, MusicEditSegmentRequestV2)
        source_start = int(passage["source_start"])
        source_end = int(passage["source_end"])
        if index:
            join_request = joins[index - 1]
            if join_request.alignment == "section_boundary":
                prior_end_kind = str(
                    resolved_passages[index - 1]["end_kind"]
                )
                current_start_kind = str(passage["start_kind"])
                if (
                    prior_end_kind not in {
                        "section_boundary",
                        "natural_track_end",
                    }
                    and current_start_kind
                    not in {"section_boundary", "track_start"}
                ):
                    raise MusicAssemblyError(
                        "section-boundary join is not bound to a reviewed section edge"
                    )
            else:
                if request.start_cue_id is None:
                    raise MusicAssemblyError(
                        f"{join_request.alignment} join requires a locked entrance cue"
                    )
                entrance_cue = cue_by_id[request.start_cue_id]
                allowed_cue_kinds = {
                    "phrase_grid": {"downbeat"},
                    "downbeat": {"downbeat"},
                    "accent": {"accent"},
                    "transient": {"accent"},
                }[join_request.alignment]
                if entrance_cue.kind not in allowed_cue_kinds:
                    raise MusicAssemblyError(
                        f"{join_request.alignment} join is inconsistent with "
                        f"entrance cue kind {entrance_cue.kind}"
                    )
            duration_samples = (
                0
                if join_request.join_type == "cut"
                else round(
                    Fraction(
                        join_request.crossfade_ms * sample_rate,
                        1000,
                    )
                )
            )
            edit_joins.append(
                MusicEditJoinV2(
                    join_id=f"music-edit-join-{index:03d}",
                    left_span_id=f"music-edit-span-{index:03d}",
                    right_span_id=f"music-edit-span-{index + 1:03d}",
                    join_type=join_request.join_type,
                    duration_samples=duration_samples,
                    alignment=join_request.alignment,
                    energy_transition=join_request.energy_transition,
                    editorial_reason=join_request.editorial_reason,
                )
            )
            output_cursor -= duration_samples
        output_end = output_cursor + source_end - source_start
        edit_spans.append(
            MusicEditSpanV2(
                span_id=f"music-edit-span-{index + 1:03d}",
                section_id=request.section_id,
                semantic_role=request.semantic_role,
                energy_band=request.energy_band,
                source_start_sample=source_start,
                source_end_sample=source_end,
                output_start_sample=output_cursor,
                output_end_sample=output_end,
                start_boundary_kind=str(passage["start_kind"]),
                end_boundary_kind=str(passage["end_kind"]),
                start_boundary_cue_id=request.start_cue_id,
                end_boundary_cue_id=request.end_cue_id,
            )
        )
        output_cursor = output_end

    output_duration = output_cursor
    final_span = edit_spans[-1]
    resolved_ending_mode = ending_mode
    if resolved_ending_mode is None:
        resolved_ending_mode = (
            "natural_track_end"
            if final_span.end_boundary_kind == "natural_track_end"
            else "phrase_fade_out"
        )
    if resolved_ending_mode == "natural_track_end":
        fade_out_samples = 0
        ending_cue_id = None
        ending_reason = "Preserve the reviewed source track ending."
    elif resolved_ending_mode == "phrase_fade_out":
        if not 5 <= ending_fade_out_ms <= 2_000:
            raise ValueError("music ending fade must remain between 5 and 2000 ms")
        fade_out_samples = round(
            Fraction(ending_fade_out_ms * sample_rate, 1000)
        )
        ending_cue_id = None
        ending_reason = "Resolve at a reviewed passage boundary with a short fade."
    else:
        fade_out_samples = 0
        ending_cue_id = final_span.end_boundary_cue_id
        if ending_cue_id is None:
            raise MusicAssemblyError(
                "reviewed ending hit requires the final passage to end on a cue"
            )
        ending_cue = cue_by_id[ending_cue_id]
        if ending_cue.kind != "ending_hit":
            raise MusicAssemblyError(
                "reviewed ending-hit mode requires an ending_hit cue"
            )
        ending_reason = "End on the human-reviewed ending hit."

    definition = {
        "music_lock_sha256": lock_sha256,
        "music_definition_sha256": music_lock.definition_sha256,
        "target_duration_samples": target_samples,
        "minimum_duration_samples": minimum_samples,
        "maximum_duration_samples": maximum_samples,
        "spans": [span.model_dump(mode="json") for span in edit_spans],
        "joins": [join.model_dump(mode="json") for join in edit_joins],
        "ending_mode": resolved_ending_mode,
        "ending_fade_out_samples": fade_out_samples,
        "ducking_regions": [
            region.model_dump(mode="json") for region in ducking_regions
        ],
    }
    return MusicEditPlanV2(
        edit_id=f"music-edit:{_canonical_hash(definition)}",
        music_id=music_lock.music_id,
        music_lock_path=str(resolved_lock),
        music_lock_sha256=lock_sha256,
        music_definition_sha256=music_lock.definition_sha256,
        master_sample_rate=sample_rate,
        source_duration_samples=music_lock.duration_samples,
        target_duration_samples=target_samples,
        minimum_duration_samples=minimum_samples,
        maximum_duration_samples=maximum_samples,
        output_duration_samples=output_duration,
        target_duration_error_samples=abs(output_duration - target_samples),
        spans=edit_spans,
        joins=edit_joins,
        ending=MusicEditEndingV2(
            mode=resolved_ending_mode,
            fade_out_samples=fade_out_samples,
            ending_cue_id=ending_cue_id,
            editorial_reason=ending_reason,
        ),
        ducking_regions=list(ducking_regions),
        uncertainties=[
            "Music passage order and semantic roles remain human-reviewable editorial proposals.",
            "All executable boundaries resolve from reviewed section or locked cue IDs; no model-generated samples are accepted.",
            "The renderer does not time-stretch, loop, or fabricate a new musical ending.",
        ],
        generated_at=utc_now(),
    )


def _validate_music_edit_plan_v2_lock(plan: MusicEditPlanV2) -> MusicMapLock:
    lock_path = Path(plan.music_lock_path).expanduser().resolve(strict=True)
    if sha256_file(lock_path) != plan.music_lock_sha256:
        raise MusicAssemblyError("music lock hash no longer matches the edit plan")
    music_lock = MusicMapLock.model_validate(read_json(lock_path))
    if music_lock.music_id != plan.music_id:
        raise MusicAssemblyError("music lock asset does not match the edit plan")
    if music_lock.definition_sha256 != plan.music_definition_sha256:
        raise MusicAssemblyError(
            "music lock definition does not match the edit plan"
        )
    if music_lock.master_sample_rate != plan.master_sample_rate:
        raise MusicAssemblyError("music lock sample rate does not match the edit")
    if music_lock.duration_samples != plan.source_duration_samples:
        raise MusicAssemblyError("music lock duration does not match the edit")
    return music_lock


def _validate_plan_lock_binding(plan: MusicAssemblyPlan) -> MusicMapLock:
    lock_path = Path(plan.music_lock_path).expanduser().resolve(strict=True)
    if sha256_file(lock_path) != plan.music_lock_sha256:
        raise MusicAssemblyError("music lock hash no longer matches the assembly plan")
    music_lock = MusicMapLock.model_validate(read_json(lock_path))
    if music_lock.music_id != plan.music_id:
        raise MusicAssemblyError("music lock asset does not match the assembly plan")
    if music_lock.definition_sha256 != plan.music_definition_sha256:
        raise MusicAssemblyError(
            "music lock definition does not match the assembly plan"
        )
    if music_lock.master_sample_rate != plan.master_sample_rate:
        raise MusicAssemblyError("music lock sample rate does not match the plan")
    if music_lock.duration_samples != plan.source_duration_samples:
        raise MusicAssemblyError("music lock duration does not match the plan")
    return music_lock


def write_music_assembly_artifacts(
    plan: MusicAssemblyPlan,
    *,
    output_dir: Path,
) -> MusicAssemblyArtifactPaths:
    """Persist an immutable plan and a hash binding to its reviewed music lock."""

    validated = MusicAssemblyPlan.model_validate(plan.model_dump(mode="json"))
    _validate_plan_lock_binding(validated)
    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    plan_path = resolved_output / MUSIC_ASSEMBLY_PLAN_FILENAME
    binding_path = resolved_output / MUSIC_ASSEMBLY_BINDING_FILENAME

    if plan_path.exists():
        existing = MusicAssemblyPlan.model_validate(read_json(plan_path))
        if existing != validated:
            raise FileExistsError(
                "refusing to overwrite a different music assembly plan artifact"
            )
    else:
        write_json(plan_path, validated)

    binding = MusicAssemblyArtifactBinding(
        assembly_id=validated.assembly_id,
        assembly_plan_path=str(plan_path),
        assembly_plan_sha256=sha256_file(plan_path),
        music_lock_path=validated.music_lock_path,
        music_lock_sha256=validated.music_lock_sha256,
        music_definition_sha256=validated.music_definition_sha256,
        generated_at=validated.generated_at,
    )
    if binding_path.exists():
        existing_binding = MusicAssemblyArtifactBinding.model_validate(
            read_json(binding_path)
        )
        if existing_binding != binding:
            raise FileExistsError(
                "refusing to overwrite a different music assembly binding artifact"
            )
    else:
        write_json(binding_path, binding)

    loaded_plan, loaded_binding = load_music_assembly_artifacts(resolved_output)
    if loaded_plan != validated or loaded_binding != binding:
        raise MusicAssemblyError("music assembly artifact round-trip validation failed")
    return MusicAssemblyArtifactPaths(
        plan_path=plan_path,
        binding_path=binding_path,
    )


def load_music_assembly_artifacts(
    output_dir: Path,
) -> tuple[MusicAssemblyPlan, MusicAssemblyArtifactBinding]:
    """Load and verify the plan, binding, and current reviewed music lock."""

    resolved_output = output_dir.expanduser().resolve(strict=True)
    expected_plan_path = resolved_output / MUSIC_ASSEMBLY_PLAN_FILENAME
    expected_binding_path = resolved_output / MUSIC_ASSEMBLY_BINDING_FILENAME
    plan = MusicAssemblyPlan.model_validate(read_json(expected_plan_path))
    binding = MusicAssemblyArtifactBinding.model_validate(
        read_json(expected_binding_path)
    )
    bound_plan_path = Path(binding.assembly_plan_path).expanduser().resolve(strict=True)
    bound_lock_path = Path(binding.music_lock_path).expanduser().resolve(strict=True)
    plan_lock_path = Path(plan.music_lock_path).expanduser().resolve(strict=True)
    if bound_plan_path != expected_plan_path:
        raise MusicAssemblyError("binding references an unexpected assembly plan path")
    if binding.assembly_plan_sha256 != sha256_file(expected_plan_path):
        raise MusicAssemblyError("assembly plan hash does not match its binding")
    if binding.assembly_id != plan.assembly_id:
        raise MusicAssemblyError("assembly ID does not match its binding")
    if bound_lock_path != plan_lock_path:
        raise MusicAssemblyError("plan and binding reference different music locks")
    if binding.music_lock_sha256 != plan.music_lock_sha256:
        raise MusicAssemblyError("plan and binding contain different music lock hashes")
    if binding.music_definition_sha256 != plan.music_definition_sha256:
        raise MusicAssemblyError(
            "plan and binding contain different music definitions"
        )
    _validate_plan_lock_binding(plan)
    return plan, binding


def _probe_rendered_audio(output_audio: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            (
                "stream=codec_name,sample_rate,channels,channel_layout,"
                "duration,duration_ts,time_base"
            ),
            "-of",
            "json",
            str(output_audio),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MusicAssemblyError(
            f"ffprobe could not inspect rendered audio: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        stream = streams[0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise MusicAssemblyError(
            "ffprobe did not return a usable rendered audio stream"
        ) from error
    if not isinstance(stream, dict):
        raise MusicAssemblyError("ffprobe audio stream payload is not an object")
    return stream


def _probed_duration_samples(stream: dict[str, object]) -> int:
    try:
        sample_rate = int(str(stream["sample_rate"]))
    except (KeyError, TypeError, ValueError) as error:
        raise MusicAssemblyError("ffprobe omitted the rendered sample rate") from error
    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts is not None and time_base is not None:
        try:
            return round(
                int(str(duration_ts))
                * Fraction(str(time_base))
                * sample_rate
            )
        except (ValueError, ZeroDivisionError):
            pass
    duration = stream.get("duration")
    if duration is None:
        raise MusicAssemblyError("ffprobe omitted the rendered audio duration")
    try:
        return round(Fraction(str(duration)) * sample_rate)
    except (ValueError, ZeroDivisionError) as error:
        raise MusicAssemblyError(
            "ffprobe returned an invalid rendered audio duration"
        ) from error


def render_single_interval_music_assembly(
    source_audio: Path,
    plan: MusicAssemblyPlan,
    output_audio: Path,
    output_dir: Path,
    *,
    fade_in_ms: int = 10,
    phrase_grid_fade_out_ms: int = 10,
    duration_tolerance_samples: int = 2,
) -> MusicAssemblyRenderResult:
    """Render one continuous plan span as 48 kHz stereo PCM.

    The graph contains one `atrim` and no concat operation. A short fade-in
    suppresses a possible click at the selected entrance. A phrase-grid cut
    must use a short fade-out; a natural source ending remains unmodified.
    """

    validated = MusicAssemblyPlan.model_validate(plan.model_dump(mode="json"))
    _validate_plan_lock_binding(validated)
    if len(validated.spans) != 1 or validated.join_count != 0:
        raise MusicAssemblyError(
            "renderer only accepts a zero-join single-interval assembly plan"
        )
    if not 1 <= fade_in_ms <= 100:
        raise ValueError("fade_in_ms must remain between 1 and 100 ms")
    if not 1 <= phrase_grid_fade_out_ms <= 100:
        raise ValueError(
            "phrase_grid_fade_out_ms must remain between 1 and 100 ms"
        )
    if not 0 <= duration_tolerance_samples <= 16:
        raise ValueError("duration tolerance must remain between 0 and 16 samples")

    resolved_source = source_audio.expanduser().resolve(strict=True)
    source_sha256 = sha256_file(resolved_source)
    if validated.music_id != f"sha256:{source_sha256}":
        raise MusicAssemblyError(
            "source audio hash does not match the music asset locked by the plan"
        )
    resolved_output = output_audio.expanduser().resolve()
    if resolved_output.suffix.lower() != ".wav":
        raise ValueError("music assembly v1 renders auditable PCM to a .wav file")
    if resolved_output == resolved_source:
        raise ValueError("source and rendered music paths must differ")
    resolved_artifact_dir = output_dir.expanduser().resolve()
    manifest_path = resolved_artifact_dir / MUSIC_ASSEMBLY_RENDER_FILENAME
    if resolved_output.exists() or manifest_path.exists():
        if not resolved_output.is_file() or not manifest_path.is_file():
            raise FileExistsError(
                "incomplete existing music assembly cannot be resumed safely"
            )
        existing = MusicAssemblyRenderManifest.model_validate(
            read_json(manifest_path)
        )
        expected_plan_hash = music_assembly_plan_sha256(validated)
        if (
            existing.assembly_id != validated.assembly_id
            or existing.assembly_plan_canonical_sha256 != expected_plan_hash
            or existing.source_audio_sha256 != source_sha256
            or Path(existing.output_audio_path).resolve() != resolved_output
            or existing.output_audio_sha256 != sha256_file(resolved_output)
            or not existing.qc_passed
        ):
            raise FileExistsError(
                "existing music assembly is not bound to the requested "
                "source, plan, output, and passing QC"
            )
        return MusicAssemblyRenderResult(
            output_audio_path=resolved_output,
            manifest_path=manifest_path,
            manifest=existing,
        )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)

    span = validated.spans[0]
    output_sample_rate = 48_000
    expected_output_samples = round(
        Fraction(
            validated.output_duration_samples * output_sample_rate,
            validated.master_sample_rate,
        )
    )
    fade_in_samples = min(
        expected_output_samples,
        round(Fraction(fade_in_ms * output_sample_rate, 1000)),
    )
    if fade_in_samples <= 0:
        raise MusicAssemblyError("selected interval is too short for a fade-in")
    fade_out_samples = 0
    ending_filter = ""
    if span.end_boundary_kind == "phrase_grid":
        fade_out_samples = min(
            expected_output_samples,
            round(
                Fraction(
                    phrase_grid_fade_out_ms * output_sample_rate,
                    1000,
                )
            ),
        )
        if fade_out_samples <= 0:
            raise MusicAssemblyError(
                "phrase-grid ending is too short for the required fade-out"
            )
        fade_out_start = expected_output_samples - fade_out_samples
        ending_filter = (
            f",afade=t=out:start_sample={fade_out_start}:"
            f"nb_samples={fade_out_samples}"
        )
    filter_graph = (
        f"[0:a:0]aresample={validated.master_sample_rate},"
        f"atrim=start_sample={span.source_start_sample}:"
        f"end_sample={span.source_end_sample},"
        "asetpts=N/SR/TB,"
        f"aresample={output_sample_rate},"
        f"afade=t=in:start_sample=0:nb_samples={fade_in_samples}"
        f"{ending_filter}"
        "[music]"
    )
    if "concat" in filter_graph.lower():
        raise MusicAssemblyError("internal concat is forbidden for music assembly v1")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(resolved_source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[music]",
        "-vn",
        "-sn",
        "-dn",
        "-ar",
        str(output_sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "-1",
        str(resolved_output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        resolved_output.unlink(missing_ok=True)
        raise MusicAssemblyError(
            f"FFmpeg could not render the music interval: {completed.stderr.strip()}"
        )
    if not resolved_output.exists() or resolved_output.stat().st_size <= 0:
        raise MusicAssemblyError("FFmpeg did not create a non-empty music render")

    stream = _probe_rendered_audio(resolved_output)
    probed_samples = _probed_duration_samples(stream)
    duration_delta = probed_samples - expected_output_samples
    qc_errors: list[str] = []
    if str(stream.get("codec_name")) != "pcm_s16le":
        qc_errors.append(
            f"unexpected output codec: {stream.get('codec_name', 'missing')}"
        )
    if str(stream.get("sample_rate")) != str(output_sample_rate):
        qc_errors.append(
            f"unexpected output sample rate: {stream.get('sample_rate', 'missing')}"
        )
    if str(stream.get("channels")) != "2":
        qc_errors.append(
            f"unexpected output channel count: {stream.get('channels', 'missing')}"
        )
    if abs(duration_delta) > duration_tolerance_samples:
        qc_errors.append(
            "rendered sample duration differs from the assembly plan "
            f"by {duration_delta} samples"
        )
    output_sha256 = sha256_file(resolved_output)
    plan_sha256 = music_assembly_plan_sha256(validated)
    render_definition = {
        "assembly_id": validated.assembly_id,
        "assembly_plan_canonical_sha256": plan_sha256,
        "source_audio_sha256": source_sha256,
        "output_audio_sha256": output_sha256,
        "source_start_sample": span.source_start_sample,
        "source_end_sample": span.source_end_sample,
        "fade_in_samples": fade_in_samples,
        "fade_out_samples": fade_out_samples,
        "ending_policy": validated.ending_policy,
        "internal_join_count": 0,
    }
    manifest = MusicAssemblyRenderManifest(
        render_id=f"music-render:{_canonical_hash(render_definition)}",
        assembly_id=validated.assembly_id,
        assembly_plan_canonical_sha256=plan_sha256,
        source_audio_path=str(resolved_source),
        source_audio_sha256=source_sha256,
        output_audio_path=str(resolved_output),
        output_audio_sha256=output_sha256,
        source_start_sample=span.source_start_sample,
        source_end_sample=span.source_end_sample,
        source_master_sample_rate=validated.master_sample_rate,
        end_boundary_kind=span.end_boundary_kind,
        expected_output_samples=expected_output_samples,
        probed_output_samples=probed_samples,
        duration_delta_samples=duration_delta,
        duration_tolerance_samples=duration_tolerance_samples,
        fade_in_samples=fade_in_samples,
        fade_out_samples=fade_out_samples,
        ending_policy=validated.ending_policy,
        natural_track_end_preserved=(
            span.end_boundary_kind == "natural_track_end"
        ),
        ffmpeg_filter_graph=filter_graph,
        ffmpeg_command=command,
        ffprobe_audio_stream=stream,
        qc_passed=not qc_errors,
        qc_errors=qc_errors,
        generated_at=utc_now(),
    )
    write_json(manifest_path, manifest)
    saved = MusicAssemblyRenderManifest.model_validate(read_json(manifest_path))
    if saved != manifest:
        raise MusicAssemblyError("music render manifest round-trip validation failed")
    if not manifest.qc_passed:
        raise MusicAssemblyError(
            f"music render QC failed; inspect {manifest_path}"
        )
    return MusicAssemblyRenderResult(
        output_audio_path=resolved_output,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def render_reviewed_music_edit_v2(
    source_audio: Path,
    plan: MusicEditPlanV2,
    output_audio: Path,
    output_dir: Path,
    *,
    fade_in_ms: int = 10,
    duration_tolerance_samples: int = 8,
) -> MusicEditRenderResultV2:
    """Render a reviewed multi-passage plan as deterministic 48 kHz PCM."""

    validated = MusicEditPlanV2.model_validate(plan.model_dump(mode="json"))
    _validate_music_edit_plan_v2_lock(validated)
    if validated.master_sample_rate != 48_000:
        raise MusicAssemblyError(
            "music edit v2 currently requires a 48 kHz reviewed master timeline"
        )
    if not 1 <= fade_in_ms <= 100:
        raise ValueError("fade_in_ms must remain between 1 and 100 ms")
    if not 0 <= duration_tolerance_samples <= 16:
        raise ValueError("duration tolerance must remain between 0 and 16 samples")

    resolved_source = source_audio.expanduser().resolve(strict=True)
    source_sha256 = sha256_file(resolved_source)
    if validated.music_id != f"sha256:{source_sha256}":
        raise MusicAssemblyError(
            "source audio hash does not match the music asset locked by the edit"
        )
    resolved_output = output_audio.expanduser().resolve()
    if resolved_output.suffix.lower() != ".wav":
        raise ValueError("music edit v2 renders auditable PCM to a .wav file")
    if resolved_output == resolved_source:
        raise ValueError("source and rendered music paths must differ")
    artifact_dir = output_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_dir / MUSIC_EDIT_PLAN_V2_FILENAME
    manifest_path = artifact_dir / MUSIC_EDIT_RENDER_V2_FILENAME
    if resolved_output.exists() or manifest_path.exists():
        if not resolved_output.is_file() or not manifest_path.is_file():
            raise FileExistsError(
                "incomplete existing music edit cannot be resumed safely"
            )
        existing = MusicEditRenderManifestV2.model_validate(
            read_json(manifest_path)
        )
        expected_plan_hash = music_edit_plan_v2_sha256(validated)
        if (
            existing.edit_id != validated.edit_id
            or existing.edit_plan_canonical_sha256 != expected_plan_hash
            or existing.source_audio_sha256 != source_sha256
            or Path(existing.output_audio_path).resolve() != resolved_output
            or existing.output_audio_sha256 != sha256_file(resolved_output)
            or not existing.qc_passed
        ):
            raise FileExistsError(
                "existing music edit is not bound to the requested source, "
                "plan, output, and passing QC"
            )
        if plan_path.is_file():
            saved_plan = MusicEditPlanV2.model_validate(read_json(plan_path))
            if saved_plan != validated:
                raise FileExistsError(
                    "existing music edit plan differs from the requested plan"
                )
        else:
            raise FileExistsError(
                "existing music edit is missing its immutable plan artifact"
            )
        return MusicEditRenderResultV2(
            output_audio_path=resolved_output,
            manifest_path=manifest_path,
            manifest=existing,
        )
    if plan_path.exists():
        saved_plan = MusicEditPlanV2.model_validate(read_json(plan_path))
        if saved_plan != validated:
            raise FileExistsError(
                "refusing to overwrite a different music edit plan"
            )
    else:
        write_json(plan_path, validated)

    filter_parts: list[str] = []
    for index, span in enumerate(validated.spans):
        filter_parts.append(
            (
                f"[0:a]aresample=48000,"
                f"atrim=start_sample={span.source_start_sample}:"
                f"end_sample={span.source_end_sample},"
                "asetpts=PTS-STARTPTS,"
                "aformat=sample_fmts=s16:sample_rates=48000:"
                f"channel_layouts=stereo[editspan{index}]"
            )
        )

    current_label = "editspan0"
    for index, join in enumerate(validated.joins, start=1):
        next_label = f"editspan{index}"
        joined_label = f"editjoin{index}"
        if join.join_type == "cut":
            filter_parts.append(
                f"[{current_label}][{next_label}]"
                f"concat=n=2:v=0:a=1[{joined_label}]"
            )
        else:
            duration_seconds = join.duration_samples / 48_000
            filter_parts.append(
                f"[{current_label}][{next_label}]"
                f"acrossfade=d={duration_seconds:.9f}:c1=tri:c2=tri"
                f"[{joined_label}]"
            )
        current_label = joined_label

    tail_filters = [f"afade=t=in:st=0:d={fade_in_ms / 1000:.9f}"]
    for region in validated.ducking_regions:
        gain = 10 ** (region.gain_db / 20)
        start_seconds = region.output_start_sample / 48_000
        end_seconds = region.output_end_sample / 48_000
        tail_filters.append(
            "volume="
            f"{gain:.9f}:enable='between(t,{start_seconds:.9f},"
            f"{end_seconds:.9f})'"
        )
    if validated.ending.fade_out_samples:
        fade_duration = validated.ending.fade_out_samples / 48_000
        fade_start = (
            validated.output_duration_samples
            - validated.ending.fade_out_samples
        ) / 48_000
        tail_filters.append(
            f"afade=t=out:st={fade_start:.9f}:d={fade_duration:.9f}"
        )
    filter_parts.append(
        f"[{current_label}]{','.join(tail_filters)}[musicout]"
    )
    filter_graph = ";".join(filter_parts)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(resolved_source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[musicout]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(resolved_output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MusicAssemblyError(
            f"FFmpeg could not render music edit v2: {completed.stderr.strip()}"
        )
    if not resolved_output.exists() or resolved_output.stat().st_size <= 0:
        raise MusicAssemblyError("FFmpeg did not create a non-empty music edit")

    stream = _probe_rendered_audio(resolved_output)
    probed_samples = _probed_duration_samples(stream)
    expected_samples = validated.output_duration_samples
    duration_delta = probed_samples - expected_samples
    qc_errors: list[str] = []
    if str(stream.get("codec_name")) != "pcm_s16le":
        qc_errors.append(
            f"unexpected output codec: {stream.get('codec_name', 'missing')}"
        )
    if str(stream.get("sample_rate")) != "48000":
        qc_errors.append(
            f"unexpected output sample rate: {stream.get('sample_rate', 'missing')}"
        )
    if str(stream.get("channels")) != "2":
        qc_errors.append(
            f"unexpected output channel count: {stream.get('channels', 'missing')}"
        )
    if abs(duration_delta) > duration_tolerance_samples:
        qc_errors.append(
            "rendered sample duration differs from the music edit plan "
            f"by {duration_delta} samples"
        )
    output_sha256 = sha256_file(resolved_output)
    plan_sha256 = music_edit_plan_v2_sha256(validated)
    render_definition = {
        "edit_id": validated.edit_id,
        "edit_plan_canonical_sha256": plan_sha256,
        "source_audio_sha256": source_sha256,
        "output_audio_sha256": output_sha256,
        "join_count": len(validated.joins),
        "fade_in_ms": fade_in_ms,
        "ending": validated.ending.model_dump(mode="json"),
        "ducking_region_count": len(validated.ducking_regions),
    }
    manifest = MusicEditRenderManifestV2(
        render_id=f"music-edit-render:{_canonical_hash(render_definition)}",
        edit_id=validated.edit_id,
        edit_plan_canonical_sha256=plan_sha256,
        source_audio_path=str(resolved_source),
        source_audio_sha256=source_sha256,
        output_audio_path=str(resolved_output),
        output_audio_sha256=output_sha256,
        expected_output_samples=expected_samples,
        probed_output_samples=probed_samples,
        duration_delta_samples=duration_delta,
        duration_tolerance_samples=duration_tolerance_samples,
        internal_join_count=len(validated.joins),
        crossfade_samples=sum(
            join.duration_samples
            for join in validated.joins
            if join.join_type == "micro_crossfade"
        ),
        fade_in_samples=round(48_000 * fade_in_ms / 1000),
        fade_out_samples=validated.ending.fade_out_samples,
        ducking_region_count=len(validated.ducking_regions),
        ffmpeg_filter_graph=filter_graph,
        ffmpeg_command=command,
        ffprobe_audio_stream=stream,
        qc_passed=not qc_errors,
        qc_errors=qc_errors,
        generated_at=utc_now(),
    )
    write_json(manifest_path, manifest)
    saved = MusicEditRenderManifestV2.model_validate(read_json(manifest_path))
    if saved != manifest:
        raise MusicAssemblyError("music edit render manifest round-trip failed")
    if not manifest.qc_passed:
        raise MusicAssemblyError(
            f"music edit render QC failed; inspect {manifest_path}"
        )
    return MusicEditRenderResultV2(
        output_audio_path=resolved_output,
        manifest_path=manifest_path,
        manifest=manifest,
    )
