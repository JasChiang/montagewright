from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from .media import sha256_file
from .models import (
    MusicAssemblyArtifactBinding,
    MusicAssemblyCueInstance,
    MusicAssemblyPlan,
    MusicAssemblyRenderManifest,
    MusicAssemblySpan,
)
from .music import MusicMapLock
from .storage import read_json, utc_now, write_json


MUSIC_ASSEMBLY_PLAN_FILENAME = "music-assembly-plan.json"
MUSIC_ASSEMBLY_BINDING_FILENAME = "music-assembly-plan.binding.json"
MUSIC_ASSEMBLY_RENDER_FILENAME = "music-assembly-render.json"


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
    if resolved_output.exists():
        raise FileExistsError("refusing to overwrite an existing rendered audio file")
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing render manifest")
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
