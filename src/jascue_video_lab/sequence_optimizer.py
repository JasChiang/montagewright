from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .autonomous_policy import AutonomousEditPolicy
from .models import FrozenStrictModel


BeatPriority = Literal["hard", "preferred", "optional"]
PresentationMode = Literal[
    "source_hold",
    "static_full_bleed_crop",
    "tracked_full_bleed_crop",
    "phase_virtual_camera",
    "hard_cut_between_views",
    "controlled_semantic_clip",
    "two_panel_layout",
    "solid_matte_fit",
    "intentional_freeze",
]


class ConstraintResult(FrozenStrictModel):
    """Typed, evidence-carrying result shared by every editing capability."""

    constraint_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_.:-]*$",
    )
    level: Literal["hard", "preference"]
    status: Literal["pass", "fail", "unknown"]
    evidence_refs: tuple[str, ...] = ()
    measured_value: float | int | bool | str | None = None
    threshold: float | int | bool | str | None = None
    reason_code: str = Field(min_length=1)


class OptionMetrics(FrozenStrictModel):
    semantic_fit: float = Field(default=0.5, ge=0.0, le=1.0)
    readability: float = Field(default=0.5, ge=0.0, le=1.0)
    technical_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    music_flow: float = Field(default=0.5, ge=0.0, le=1.0)
    synthetic_motion_distance: float = Field(default=0.0, ge=0.0)
    intrusion_rank: int = Field(default=0, ge=0, le=10)
    local_cost_rank: int = Field(default=0, ge=0, le=10)


class ExecutableOptionV2(FrozenStrictModel):
    """Capability-neutral option. Payload stays in the owning executor."""

    option_id: str = Field(min_length=1)
    capability_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_hashes: tuple[str, ...] = ()
    constraints: tuple[ConstraintResult, ...] = Field(min_length=1)
    metrics: OptionMetrics = OptionMetrics()
    decision_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_constraints(self) -> "ExecutableOptionV2":
        ids = [item.constraint_id for item in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("option constraint IDs must be unique")
        return self

    @property
    def hard_failure_codes(self) -> tuple[str, ...]:
        # Unknown hard facts fail closed. A preference may remain unknown and
        # merely lower confidence; it can never authorize delivery.
        return tuple(
            result.reason_code
            for result in self.constraints
            if result.level == "hard" and result.status != "pass"
        )


class ExecutableOptionSelectionV2(FrozenStrictModel):
    contract_version: Literal["executable-option-selection-v2"] = (
        "executable-option-selection-v2"
    )
    selected_option_id: str | None
    generated_option_ids: tuple[str, ...]
    generated_capabilities: Mapping[str, str] = Field(default_factory=dict)
    rejected_options: Mapping[str, tuple[str, ...]]
    score: float | None
    runner_up_option_ids: tuple[str, ...] = ()
    score_gap: float | None = None
    semantic_negotiation_recommended: bool = False
    semantic_ambiguity_codes: tuple[str, ...] = ()
    decision_codes: tuple[str, ...]


def select_executable_option(
    options: Sequence[ExecutableOptionV2],
    *,
    preferred_capability_ids: Sequence[str] = (),
) -> ExecutableOptionSelectionV2:
    """Select one legal operation without content-specific fallback branches."""

    generated = tuple(option.option_id for option in options)
    if len(generated) != len(set(generated)):
        raise ValueError("executable option IDs must be unique")
    rejected = {
        option.option_id: option.hard_failure_codes
        for option in options
        if option.hard_failure_codes
    }
    generated_capabilities = {
        option.option_id: option.capability_id for option in options
    }
    legal = [option for option in options if not option.hard_failure_codes]
    if not legal:
        return ExecutableOptionSelectionV2(
            selected_option_id=None,
            generated_option_ids=generated,
            generated_capabilities=generated_capabilities,
            rejected_options=rejected,
            score=None,
            decision_codes=("no_hard_constraint_safe_option",),
        )
    preference_rank = {
        capability_id: index
        for index, capability_id in enumerate(preferred_capability_ids)
    }

    def rank(option: ExecutableOptionV2) -> tuple[float, float, str]:
        metrics = option.metrics
        preference_bonus = 0.0
        if option.capability_id in preference_rank:
            preference_bonus = max(
                0.02,
                0.12 - preference_rank[option.capability_id] * 0.02,
            )
        score = (
            metrics.semantic_fit * 0.34
            + metrics.readability * 0.28
            + metrics.technical_quality * 0.22
            + metrics.music_flow * 0.10
            - metrics.synthetic_motion_distance * 0.04
            - metrics.intrusion_rank * 0.012
            - metrics.local_cost_rank * 0.006
            + preference_bonus
        )
        # Prefer less intrusive and cheaper options on a semantic tie.
        tie_break = -(metrics.intrusion_rank + metrics.local_cost_rank * 0.1)
        return score, tie_break, option.option_id

    ranked = sorted(legal, key=rank, reverse=True)
    selected = ranked[0]
    selected_score = rank(selected)[0]
    runners = tuple(option.option_id for option in ranked[1:3])
    score_gap = (
        selected_score - rank(ranked[1])[0]
        if len(ranked) >= 2
        else None
    )
    semantically_ambiguous = bool(
        len(ranked) >= 2
        and score_gap is not None
        and score_gap <= 0.04
        and selected.capability_id != ranked[1].capability_id
    )
    return ExecutableOptionSelectionV2(
        selected_option_id=selected.option_id,
        generated_option_ids=generated,
        generated_capabilities=generated_capabilities,
        rejected_options=rejected,
        score=round(selected_score, 6),
        runner_up_option_ids=runners,
        score_gap=round(score_gap, 6) if score_gap is not None else None,
        semantic_negotiation_recommended=semantically_ambiguous,
        semantic_ambiguity_codes=(
            ("different_capabilities_have_near_equal_local_scores",)
            if semantically_ambiguous
            else ()
        ),
        decision_codes=(
            "hard_constraints_passed",
            "lexicographic_preference_selected",
        ),
    )


class SequenceOption(FrozenStrictModel):
    """One executable candidate/trim/cue/presentation choice for a beat."""

    option_id: str = Field(min_length=1)
    beat_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_in_pts: int = Field(ge=0)
    source_out_pts: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    minimum_readable_ms: int = Field(gt=0)
    preferred_readable_ms: int = Field(gt=0)
    maximum_readable_ms: int = Field(gt=0)
    cue_id: str = Field(min_length=1)
    cue_delta_frames: int
    cue_tolerance_frames: int = Field(ge=0)
    presentation_mode: PresentationMode
    presentation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_x: float | None = Field(default=None, ge=0.0, le=1.0)
    exit_x: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_evidence_satisfied: bool
    identity_satisfied: bool
    action_complete: bool
    required_relation_satisfied: bool
    relative_scale_satisfied: bool
    quality_safe: bool
    reuse_authorized: bool
    geometry_executable: bool
    legal_musical_exit: bool
    semantic_fit: float = Field(ge=0.0, le=1.0)
    readability: float = Field(ge=0.0, le=1.0)
    technical_quality: float = Field(ge=0.0, le=1.0)
    music_flow: float = Field(ge=0.0, le=1.0)
    synthetic_motion_distance: float = Field(default=0.0, ge=0.0)
    authorized_reprise: bool = False
    freeze_event_lock_id: str | None = None

    @model_validator(mode="after")
    def validate_option(self) -> "SequenceOption":
        if self.source_out_pts <= self.source_in_pts:
            raise ValueError("source interval must be non-empty")
        if not (
            self.minimum_readable_ms
            <= self.preferred_readable_ms
            <= self.maximum_readable_ms
        ):
            raise ValueError("readability bounds must be ordered")
        if not (
            self.minimum_readable_ms
            <= self.duration_ms
            <= self.maximum_readable_ms
        ):
            raise ValueError("duration must remain inside readability bounds")
        if (
            self.presentation_mode == "intentional_freeze"
            and self.freeze_event_lock_id is None
        ):
            raise ValueError("intentional freeze requires an exact event lock")
        return self

    def hard_constraint_failures(self, priority: BeatPriority) -> tuple[str, ...]:
        failures: list[str] = []
        checks = (
            ("identity_failed", self.identity_satisfied),
            ("action_incomplete", self.action_complete),
            ("required_relation_failed", self.required_relation_satisfied),
            ("relative_scale_failed", self.relative_scale_satisfied),
            ("quality_unsafe", self.quality_safe),
            (
                "music_hard_sync_failed",
                abs(self.cue_delta_frames) <= self.cue_tolerance_frames,
            ),
            ("reuse_unauthorized", self.reuse_authorized),
            ("geometry_infeasible", self.geometry_executable),
            ("musical_exit_illegal", self.legal_musical_exit),
        )
        if priority == "hard" and not self.hard_evidence_satisfied:
            failures.append("hard_evidence_missing")
        failures.extend(code for code, passed in checks if not passed)
        return tuple(failures)


class BeatOptionSet(FrozenStrictModel):
    beat_id: str = Field(min_length=1)
    priority: BeatPriority
    options: tuple[SequenceOption, ...]

    @model_validator(mode="after")
    def validate_options(self) -> "BeatOptionSet":
        if any(option.beat_id != self.beat_id for option in self.options):
            raise ValueError("all options must belong to the declared beat")
        return self


class SequenceSelection(FrozenStrictModel):
    beat_id: str
    priority: BeatPriority
    option: SequenceOption | None
    decision_codes: tuple[str, ...]


class SequenceOptimizationResult(FrozenStrictModel):
    contract_version: Literal["sequence-optimization-v1"] = (
        "sequence-optimization-v1"
    )
    selections: tuple[SequenceSelection, ...]
    total_duration_ms: int = Field(ge=0)
    target_duration_ms: int = Field(gt=0)
    duration_delta_ms: int
    score: float
    outcome: Literal["complete", "best_effort_shortened", "blocked"]
    omitted_beat_ids: tuple[str, ...] = ()
    hard_failure_codes: tuple[str, ...] = ()


class _BeamState(FrozenStrictModel):
    selections: tuple[SequenceSelection, ...] = ()
    duration_ms: int = Field(default=0, ge=0)
    panel_duration_ms: int = Field(default=0, ge=0)
    score: float = 0.0
    source_intervals: tuple[tuple[str, int, int], ...] = ()


def optimize_sequence(
    beat_sets: Sequence[BeatOptionSet],
    *,
    policy: AutonomousEditPolicy,
    beam_width: int = 24,
) -> SequenceOptimizationResult:
    """Bounded evidence-first beam search; never lets quality buy a hard pass."""

    if beam_width < 1:
        raise ValueError("beam width must be positive")
    states: list[_BeamState] = [_BeamState()]
    hard_failures: list[str] = []
    total_beats = max(1, len(beat_sets))
    for beat_index, beat in enumerate(beat_sets):
        valid = [
            option
            for option in beat.options
            if not option.hard_constraint_failures(beat.priority)
        ]
        if not valid and beat.priority != "optional":
            failures = sorted(
                {
                    code
                    for option in beat.options
                    for code in option.hard_constraint_failures(beat.priority)
                }
                or {"no_candidate_options"}
            )
            hard_failures.extend(
                f"{beat.beat_id}:{failure}" for failure in failures
            )
            return _blocked_result(
                policy=policy,
                selections=states[0].selections,
                hard_failures=hard_failures,
            )

        choices: list[SequenceOption | None] = list(valid)
        if (
            beat.priority == "optional"
            and policy.editorial.allow_optional_beat_omission
        ):
            choices.append(None)
        if not choices:
            return _blocked_result(
                policy=policy,
                selections=states[0].selections,
                hard_failures=(f"{beat.beat_id}:optional_omission_forbidden",),
            )

        expanded: list[_BeamState] = []
        for state in states:
            for option in choices:
                expanded.append(_extend_state(state, beat, option))
        states = sorted(
            expanded,
            key=lambda state: _beam_rank(
                state,
                policy,
                completion_fraction=(beat_index + 1) / total_beats,
            ),
            reverse=True,
        )[:beam_width]

    panel_safe_states = [
        state
        for state in states
        if state.duration_ms == 0
        or (
            state.panel_duration_ms / state.duration_ms
            <= policy.presentation.max_panel_runtime_fraction + 1e-9
        )
    ]
    acceptable = [
        state
        for state in panel_safe_states
        if policy.duration.min_ms
        <= state.duration_ms
        <= policy.duration.max_ms
    ]
    outcome: Literal["complete", "best_effort_shortened", "blocked"]
    if acceptable:
        best = max(acceptable, key=lambda state: _final_rank(state, policy))
        outcome = "complete"
    elif (
        policy.execution_profile == "autonomous_best_effort"
        and panel_safe_states
        and max(
            state.duration_ms for state in panel_safe_states
        ) < policy.duration.min_ms
    ):
        best = max(
            panel_safe_states,
            key=lambda state: _final_rank(state, policy),
        )
        outcome = "best_effort_shortened"
    else:
        closest = max(states, key=lambda state: _final_rank(state, policy))
        failure = (
            "panel_runtime_fraction_exceeded"
            if states and not panel_safe_states
            else "duration_outside_policy_bounds"
        )
        return _blocked_result(
            policy=policy,
            selections=closest.selections,
            hard_failures=(failure,),
        )

    omitted = tuple(
        selection.beat_id
        for selection in best.selections
        if selection.option is None
    )
    return SequenceOptimizationResult(
        selections=best.selections,
        total_duration_ms=best.duration_ms,
        target_duration_ms=policy.duration.target_ms,
        duration_delta_ms=best.duration_ms - policy.duration.target_ms,
        score=round(_final_rank(best, policy), 6),
        outcome=outcome,
        omitted_beat_ids=omitted,
    )


def _extend_state(
    state: _BeamState,
    beat: BeatOptionSet,
    option: SequenceOption | None,
) -> _BeamState:
    if option is None:
        return _BeamState(
            selections=state.selections
            + (
                SequenceSelection(
                    beat_id=beat.beat_id,
                    priority=beat.priority,
                    option=None,
                    decision_codes=("policy_authorized_optional_omission",),
                ),
            ),
            duration_ms=state.duration_ms,
            panel_duration_ms=state.panel_duration_ms,
            score=state.score - 0.04,
            source_intervals=state.source_intervals,
        )

    interval = (
        option.source_asset_id,
        option.source_in_pts,
        option.source_out_pts,
    )
    reuse_penalty = 0.0
    if interval in state.source_intervals:
        reuse_penalty = 0.20 if option.authorized_reprise else 1.0
    continuity_penalty = _continuity_penalty(state, option)
    intrusion_penalty = {
        "two_panel_layout": 0.04,
        "solid_matte_fit": 0.08,
        "intentional_freeze": 0.05,
    }.get(option.presentation_mode, 0.0)
    preference = (
        option.semantic_fit * 0.32
        + option.readability * 0.26
        + option.technical_quality * 0.20
        + option.music_flow * 0.16
        - option.synthetic_motion_distance * 0.06
        - continuity_penalty
        - intrusion_penalty
        - reuse_penalty
    )
    decision_codes = ["hard_constraints_passed"]
    if option.duration_ms >= option.preferred_readable_ms:
        decision_codes.append("preferred_readability_reached")
    if option.authorized_reprise:
        decision_codes.append("editorial_reprise_authorized")
    return _BeamState(
        selections=state.selections
        + (
            SequenceSelection(
                beat_id=beat.beat_id,
                priority=beat.priority,
                option=option,
                decision_codes=tuple(decision_codes),
            ),
        ),
        duration_ms=state.duration_ms + option.duration_ms,
        panel_duration_ms=(
            state.panel_duration_ms
            + (
                option.duration_ms
                if option.presentation_mode == "two_panel_layout"
                else 0
            )
        ),
        score=state.score + preference,
        source_intervals=state.source_intervals + (interval,),
    )


def _continuity_penalty(
    state: _BeamState,
    option: SequenceOption,
) -> float:
    previous = next(
        (
            selection.option
            for selection in reversed(state.selections)
            if selection.option is not None
        ),
        None,
    )
    if (
        previous is None
        or previous.exit_x is None
        or option.entry_x is None
    ):
        return 0.0
    return abs(previous.exit_x - option.entry_x) * 0.08


def _beam_rank(
    state: _BeamState,
    policy: AutonomousEditPolicy,
    *,
    completion_fraction: float,
) -> float:
    # A bounded intermediate duration pressure prevents the beam from filling
    # with only long or only short variants before later beats are considered.
    expected = policy.duration.target_ms * completion_fraction
    duration_pressure = abs(state.duration_ms - expected) / max(
        policy.duration.target_ms,
        1,
    )
    return state.score - duration_pressure * 0.02


def _final_rank(
    state: _BeamState,
    policy: AutonomousEditPolicy,
) -> float:
    duration_error = abs(state.duration_ms - policy.duration.target_ms)
    duration_penalty = duration_error / max(policy.duration.target_ms, 1)
    return state.score - duration_penalty * 0.35


def _blocked_result(
    *,
    policy: AutonomousEditPolicy,
    selections: tuple[SequenceSelection, ...],
    hard_failures: Sequence[str],
) -> SequenceOptimizationResult:
    total = sum(
        selection.option.duration_ms
        for selection in selections
        if selection.option is not None
    )
    return SequenceOptimizationResult(
        selections=selections,
        total_duration_ms=total,
        target_duration_ms=policy.duration.target_ms,
        duration_delta_ms=total - policy.duration.target_ms,
        score=0.0,
        outcome="blocked",
        hard_failure_codes=tuple(hard_failures),
    )


class SegmentRenderCacheKey(FrozenStrictModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_in_pts: int = Field(ge=0)
    source_out_pts: int = Field(gt=0)
    presentation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracking_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filter_graph_version: str = Field(min_length=1)
    aspect: Literal["16:9", "9:16"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps_numerator: int = Field(gt=0)
    fps_denominator: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pts(self) -> "SegmentRenderCacheKey":
        if self.source_out_pts <= self.source_in_pts:
            raise ValueError("segment cache interval must be non-empty")
        return self

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class SegmentRenderRequest(FrozenStrictModel):
    segment_id: str = Field(min_length=1)
    cache_key: SegmentRenderCacheKey


class SegmentRenderRecord(FrozenStrictModel):
    segment_id: str
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str
    cache_hit: bool


class IncrementalRenderResult(FrozenStrictModel):
    records: tuple[SegmentRenderRecord, ...]
    rendered_segment_ids: tuple[str, ...]
    reused_segment_ids: tuple[str, ...]


def render_segments_incrementally(
    requests: Sequence[SegmentRenderRequest],
    *,
    cache_dir: Path,
    renderer: Callable[[SegmentRenderRequest, Path], None],
    suffix: str = ".mp4",
) -> IncrementalRenderResult:
    """Render cache misses only; unchanged segments are never re-rendered."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[SegmentRenderRecord] = []
    rendered: list[str] = []
    reused: list[str] = []
    for request in requests:
        output = cache_dir / f"{request.cache_key.digest}{suffix}"
        hit = output.is_file() and output.stat().st_size > 0
        if hit:
            reused.append(request.segment_id)
        else:
            temporary = cache_dir / (
                f".{request.cache_key.digest}.{os.getpid()}.partial{suffix}"
            )
            renderer(request, temporary)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(
                    f"renderer did not create segment {request.segment_id}"
                )
            temporary.replace(output)
            rendered.append(request.segment_id)
        records.append(
            SegmentRenderRecord(
                segment_id=request.segment_id,
                cache_key=request.cache_key.digest,
                output_path=str(output),
                cache_hit=hit,
            )
        )
    return IncrementalRenderResult(
        records=tuple(records),
        rendered_segment_ids=tuple(rendered),
        reused_segment_ids=tuple(reused),
    )


def concat_manifest_lines(
    result: IncrementalRenderResult,
) -> tuple[str, ...]:
    """Return deterministic ffconcat entries in requested segment order."""

    return tuple(
        f"file {json.dumps(record.output_path)}" for record in result.records
    )
