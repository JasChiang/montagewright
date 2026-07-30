from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
    EditorialPolicy,
)
from jascue_video_lab.sequence_optimizer import (
    BeatOptionSet,
    CandidateRouteBeat,
    CandidateRouteOption,
    MusicBoundaryCue,
    MusicBoundarySpec,
    RuntimeCueTimingBinding,
    RuntimeSegmentTiming,
    SegmentRenderCacheKey,
    SegmentRenderRequest,
    SemanticRhythmSpec,
    SequenceOption,
    concat_manifest_lines,
    optimize_sequence,
    optimize_pre_render_candidate_route,
    reconcile_runtime_sequence_timing,
    render_segments_incrementally,
    solve_semantic_rhythm_durations,
    solve_music_aligned_boundaries,
)


def _route_option(
    beat_id: str,
    candidate_id: str,
    source_marker: str,
    *,
    rank: int,
    confidence: float,
) -> CandidateRouteOption:
    return CandidateRouteOption(
        beat_id=beat_id,
        candidate_id=candidate_id,
        source_asset_id="sha256:" + source_marker * 64,
        event_id=f"event-{candidate_id}",
        planner_rank=rank,
        semantic_confidence=confidence,
    )


def test_pre_render_route_uses_global_variety_without_discarding_semantics() -> None:
    route = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="a",
                options=(
                    _route_option("a", "a1", "a", rank=1, confidence=0.9),
                ),
            ),
            CandidateRouteBeat(
                beat_id="b",
                options=(
                    _route_option("b", "b1", "a", rank=1, confidence=0.90),
                    _route_option("b", "b2", "b", rank=2, confidence=0.88),
                ),
            ),
        )
    )

    assert [row.candidate_id for row in route.selections] == ["a1", "b2"]
    assert "adjacent_source_variety_preferred" in (
        route.selections[1].decision_codes
    )


def test_pre_render_route_does_not_replace_clear_semantic_winner() -> None:
    route = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="a",
                options=(
                    _route_option("a", "a1", "a", rank=1, confidence=0.9),
                ),
            ),
            CandidateRouteBeat(
                beat_id="b",
                options=(
                    _route_option("b", "b1", "a", rank=1, confidence=0.98),
                    _route_option("b", "b2", "b", rank=2, confidence=0.30),
                ),
            ),
        )
    )

    assert [row.candidate_id for row in route.selections] == ["a1", "b1"]


def test_pre_render_frontier_rejects_known_hard_failure_before_score() -> None:
    unsafe = _route_option(
        "payoff",
        "pretty-but-unsuitable",
        "a",
        rank=1,
        confidence=1.0,
    ).model_copy(
        update={
            "preflight_hard_failures": (
                "aspect_declared_unsuitable",
            ),
            "trim_duration_ms": 12_000,
            "minimum_readable_ms": 8_000,
            "preferred_readable_ms": 12_000,
            "maximum_readable_ms": 15_000,
            "cue_id": "accent-01",
            "presentation_mode": "two_panel_layout",
            "entry_composition": "comparison:a+b",
            "exit_composition": "comparison:a+b",
        }
    )
    safe = _route_option(
        "payoff",
        "proven",
        "b",
        rank=2,
        confidence=0.2,
    ).model_copy(
        update={
            "trim_duration_ms": 12_000,
            "minimum_readable_ms": 8_000,
            "preferred_readable_ms": 12_000,
            "maximum_readable_ms": 15_000,
            "cue_id": "accent-01",
            "presentation_mode": "static_full_bleed_crop",
            "entry_composition": "single:device",
            "exit_composition": "single:device",
        }
    )

    route = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="payoff",
                options=(unsafe, safe),
            ),
        )
    )

    assert route.contract_version == "pre-render-sequence-frontier-v2"
    assert route.selections[0].candidate_id == "proven"
    assert route.selections[0].trim_duration_ms == 12_000
    assert route.selections[0].cue_id == "accent-01"
    assert route.selections[0].presentation_mode == (
        "static_full_bleed_crop"
    )
    assert route.fallback_candidate_ids_by_beat["payoff"] == ("proven",)


def test_pre_render_frontier_fails_closed_when_every_option_is_known_bad() -> None:
    bad = _route_option(
        "payoff",
        "bad",
        "a",
        rank=1,
        confidence=1.0,
    ).model_copy(
        update={
            "preflight_hard_failures": (
                "aspect_declared_unsuitable",
            )
        }
    )

    with pytest.raises(ValueError, match="no hard-safe option"):
        optimize_pre_render_candidate_route(
            (CandidateRouteBeat(beat_id="payoff", options=(bad,)),)
        )


def test_pre_render_frontier_applies_panel_runtime_policy_before_render() -> None:
    panel = _route_option(
        "comparison",
        "panel",
        "a",
        rank=1,
        confidence=1.0,
    ).model_copy(
        update={
            "trim_duration_ms": 10_000,
            "minimum_readable_ms": 8_000,
            "preferred_readable_ms": 10_000,
            "maximum_readable_ms": 12_000,
            "presentation_mode": "two_panel_layout",
        }
    )
    full_bleed = _route_option(
        "comparison",
        "full-bleed",
        "b",
        rank=2,
        confidence=0.2,
    ).model_copy(
        update={
            "trim_duration_ms": 10_000,
            "minimum_readable_ms": 8_000,
            "preferred_readable_ms": 10_000,
            "maximum_readable_ms": 12_000,
            "presentation_mode": "static_full_bleed_crop",
        }
    )

    route = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="comparison",
                options=(panel, full_bleed),
            ),
        ),
        minimum_duration_ms=10_000,
        maximum_duration_ms=10_000,
        max_panel_runtime_fraction=0.25,
    )

    assert route.selections[0].candidate_id == "full-bleed"
    assert route.fallback_candidate_ids_by_beat["comparison"] == (
        "full-bleed",
    )


def _runtime_timing(
    beat_id: str,
    *,
    planned_duration_ms: int,
    actual_duration_ms: int,
    planned_candidate_id: str | None = None,
    runtime_candidate_id: str | None = None,
    cue_bindings: tuple[RuntimeCueTimingBinding, ...] = (),
) -> RuntimeSegmentTiming:
    return RuntimeSegmentTiming(
        beat_id=beat_id,
        planned_candidate_id=planned_candidate_id or f"{beat_id}-primary",
        runtime_candidate_id=runtime_candidate_id or f"{beat_id}-primary",
        source_asset_id="sha256:" + "d" * 64,
        planned_duration_ms=planned_duration_ms,
        actual_source_capacity_ms=actual_duration_ms,
        actual_duration_ms=actual_duration_ms,
        minimum_readable_ms=min(actual_duration_ms, 5_000),
        cue_bindings=cue_bindings,
        input_artifact_hashes=("sha256:" + "e" * 64,),
    )


def test_runtime_substitute_rebases_downstream_boundaries_and_cues() -> None:
    result = reconcile_runtime_sequence_timing(
        (
            _runtime_timing(
                "opening",
                planned_duration_ms=6_000,
                actual_duration_ms=6_000,
            ),
            _runtime_timing(
                "comparison",
                planned_duration_ms=7_587,
                actual_duration_ms=6_973,
                planned_candidate_id="comparison-primary",
                runtime_candidate_id="comparison-substitute",
            ),
            _runtime_timing(
                "payoff",
                planned_duration_ms=7_000,
                actual_duration_ms=7_000,
                cue_bindings=(
                    RuntimeCueTimingBinding(
                        event_id="payoff:stable-result",
                        cue_id="downbeat-04",
                        event_offset_ms=1_000,
                        cue_time_ms=14_587,
                        tolerance_frames=2,
                    ),
                ),
            ),
        ),
        minimum_total_duration_ms=19_000,
        maximum_total_duration_ms=22_000,
    )

    substitute = result.segments[1]
    downstream = result.segments[2]
    cue = downstream.cue_timings[0]
    assert result.planned_total_duration_ms == 20_587
    assert result.resolved_total_duration_ms == 19_973
    assert result.total_duration_delta_ms == -614
    assert substitute.duration_delta_ms == -614
    assert substitute.runtime_substitute_selected is True
    assert substitute.source_capacity_limited is True
    assert downstream.planned_project_start_ms == 13_587
    assert downstream.resolved_project_start_ms == 12_973
    assert downstream.project_shift_before_ms == -614
    assert cue.planned_delta_frames == 0
    assert cue.resolved_delta_frames == -19
    assert cue.delta_change_frames == -19
    assert result.outcome == "blocked"
    assert result.failure_codes == (
        "payoff:stable-result:runtime_cue_alignment_failed",
    )
    assert result.freeze_inserted is False
    assert result.time_stretch_applied is False
    assert all(segment.synthetic_fill_ms == 0 for segment in result.segments)
    assert all(
        segment.time_stretch_ratio == 1.0 for segment in result.segments
    )


def test_runtime_reconciliation_accepts_shorter_sequence_when_gates_pass() -> None:
    result = reconcile_runtime_sequence_timing(
        (
            _runtime_timing(
                "setup",
                planned_duration_ms=5_000,
                actual_duration_ms=4_500,
                planned_candidate_id="setup-primary",
                runtime_candidate_id="setup-substitute",
            ),
            _runtime_timing(
                "result",
                planned_duration_ms=5_000,
                actual_duration_ms=5_000,
                cue_bindings=(
                    RuntimeCueTimingBinding(
                        event_id="result:visible",
                        cue_id="accent-02",
                        event_offset_ms=500,
                        cue_time_ms=5_000,
                        tolerance_frames=0,
                    ),
                ),
            ),
        ),
        minimum_total_duration_ms=9_000,
        maximum_total_duration_ms=11_000,
    )

    assert result.outcome == "reconciled"
    assert result.resolved_total_duration_ms == 9_500
    assert result.segments[1].cue_timings[0].resolved_delta_frames == 0
    assert result.failure_codes == ()
    assert len(result.input_sha256) == 64


def test_runtime_reconciliation_blocks_unreadable_shortfall_without_fill() -> None:
    segment = _runtime_timing(
        "result",
        planned_duration_ms=7_000,
        actual_duration_ms=3_000,
    ).model_copy(update={"minimum_readable_ms": 4_000})

    result = reconcile_runtime_sequence_timing((segment,))

    assert result.outcome == "blocked"
    assert result.failure_codes == (
        "result:runtime_duration_below_minimum_readable",
    )
    assert result.segments[0].resolved_duration_ms == 3_000
    assert result.segments[0].synthetic_fill_ms == 0
    assert result.freeze_inserted is False


def test_runtime_timing_rejects_duration_beyond_source_capacity() -> None:
    with pytest.raises(ValidationError, match="measured source capacity"):
        RuntimeSegmentTiming(
            beat_id="comparison",
            planned_candidate_id="primary",
            runtime_candidate_id="substitute",
            source_asset_id="sha256:" + "d" * 64,
            planned_duration_ms=7_000,
            actual_source_capacity_ms=6_000,
            actual_duration_ms=6_500,
            minimum_readable_ms=5_000,
        )


def _policy(
    *,
    target_ms: int = 30_000,
    min_ms: int = 30_000,
    max_ms: int = 35_000,
    profile: str = "autonomous_strict",
    allow_optional_omission: bool = True,
) -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile=profile,
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=target_ms,
            min_ms=min_ms,
            max_ms=max_ms,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
        editorial=EditorialPolicy(
            allow_optional_beat_omission=allow_optional_omission,
        ),
    )


def _option(
    option_id: str,
    beat_id: str,
    *,
    duration_ms: int = 10_000,
    preferred_ms: int | None = None,
    score: float = 0.8,
    evidence: bool = True,
    cue_delta: int = 0,
    presentation: str = "static_full_bleed_crop",
    source_in_pts: int = 0,
    authorized_reprise: bool = False,
    freeze_event_lock_id: str | None = None,
) -> SequenceOption:
    preferred = preferred_ms or duration_ms
    return SequenceOption(
        option_id=option_id,
        beat_id=beat_id,
        candidate_id=f"candidate-{option_id}",
        source_asset_id="sha256:" + "a" * 64,
        source_in_pts=source_in_pts,
        source_out_pts=source_in_pts + duration_ms,
        duration_ms=duration_ms,
        minimum_readable_ms=min(5_000, duration_ms),
        preferred_readable_ms=preferred,
        maximum_readable_ms=max(duration_ms, preferred),
        cue_id=f"cue-{option_id}",
        cue_delta_frames=cue_delta,
        cue_tolerance_frames=2,
        presentation_mode=presentation,
        presentation_sha256="b" * 64,
        tracking_sha256="c" * 64,
        entry_x=0.5,
        exit_x=0.5,
        hard_evidence_satisfied=evidence,
        identity_satisfied=True,
        action_complete=True,
        required_relation_satisfied=True,
        relative_scale_satisfied=True,
        quality_safe=True,
        reuse_authorized=True,
        geometry_executable=True,
        legal_musical_exit=True,
        semantic_fit=score,
        readability=score,
        technical_quality=score,
        music_flow=score,
        authorized_reprise=authorized_reprise,
        freeze_event_lock_id=freeze_event_lock_id,
    )


def test_high_score_cannot_compensate_for_missing_hard_evidence() -> None:
    result = optimize_sequence(
        [
            BeatOptionSet(
                beat_id="payoff",
                priority="hard",
                options=(
                    _option(
                        "pretty",
                        "payoff",
                        duration_ms=30_000,
                        score=1.0,
                        evidence=False,
                    ),
                    _option(
                        "proven",
                        "payoff",
                        duration_ms=30_000,
                        score=0.2,
                        evidence=True,
                    ),
                ),
            )
        ],
        policy=_policy(target_ms=30_000, min_ms=30_000, max_ms=31_000),
    )

    assert result.outcome == "complete"
    assert result.selections[0].option is not None
    assert result.selections[0].option.option_id == "proven"


def test_missing_hard_evidence_fails_closed() -> None:
    result = optimize_sequence(
        [
            BeatOptionSet(
                beat_id="payoff",
                priority="hard",
                options=(
                    _option(
                        "pretty",
                        "payoff",
                        duration_ms=30_000,
                        evidence=False,
                    ),
                ),
            )
        ],
        policy=_policy(target_ms=30_000, min_ms=30_000, max_ms=31_000),
    )

    assert result.outcome == "blocked"
    assert "payoff:hard_evidence_missing" in result.hard_failure_codes


def test_duration_reconciliation_omits_optional_before_exceeding_max() -> None:
    result = optimize_sequence(
        [
            BeatOptionSet(
                beat_id="required",
                priority="hard",
                options=(_option("required", "required", duration_ms=30_000),),
            ),
            BeatOptionSet(
                beat_id="optional",
                priority="optional",
                options=(_option("optional", "optional", duration_ms=15_000),),
            ),
        ],
        policy=_policy(target_ms=30_000, min_ms=30_000, max_ms=35_000),
    )

    assert result.outcome == "complete"
    assert result.total_duration_ms == 30_000
    assert result.omitted_beat_ids == ("optional",)


def test_optimizer_extends_readability_instead_of_fixed_shot_length() -> None:
    first = BeatOptionSet(
        beat_id="result",
        priority="hard",
        options=(
            _option("brief", "result", duration_ms=10_000, score=0.9),
            _option("readable", "result", duration_ms=20_000, score=0.8),
        ),
    )
    second = BeatOptionSet(
        beat_id="reaction",
        priority="preferred",
        options=(
            _option(
                "reaction",
                "reaction",
                duration_ms=20_000,
                source_in_pts=30_000,
            ),
        ),
    )

    result = optimize_sequence(
        [first, second],
        policy=_policy(target_ms=40_000, min_ms=38_000, max_ms=42_000),
    )

    assert result.outcome == "complete"
    assert result.total_duration_ms == 40_000
    assert result.selections[0].option is not None
    assert result.selections[0].option.option_id == "readable"


def test_best_effort_shortens_total_without_accidental_freeze() -> None:
    result = optimize_sequence(
        [
            BeatOptionSet(
                beat_id="only",
                priority="hard",
                options=(_option("only", "only", duration_ms=10_000),),
            )
        ],
        policy=_policy(
            target_ms=30_000,
            min_ms=30_000,
            max_ms=35_000,
            profile="autonomous_best_effort",
        ),
    )

    assert result.outcome == "best_effort_shortened"
    assert result.total_duration_ms == 10_000
    assert result.selections[0].option is not None
    assert result.selections[0].option.presentation_mode != "intentional_freeze"


def test_panel_runtime_fraction_is_a_sequence_hard_gate() -> None:
    result = optimize_sequence(
        [
            BeatOptionSet(
                beat_id="comparison",
                priority="hard",
                options=(
                    _option(
                        "panel",
                        "comparison",
                        duration_ms=30_000,
                        presentation="two_panel_layout",
                    ),
                ),
            )
        ],
        policy=_policy(
            target_ms=30_000,
            min_ms=30_000,
            max_ms=31_000,
        ),
    )

    assert result.outcome == "blocked"
    assert result.hard_failure_codes == (
        "panel_runtime_fraction_exceeded",
    )


def test_intentional_freeze_requires_exact_event_lock() -> None:
    with pytest.raises(ValidationError, match="exact event lock"):
        _option(
            "freeze",
            "ending",
            presentation="intentional_freeze",
        )


def test_music_boundary_solver_treats_source_capacity_as_a_bound() -> None:
    result = solve_music_aligned_boundaries(
        [
            MusicBoundarySpec(
                beat_id="opening",
                preferred_duration_ms=9_800,
                minimum_duration_ms=5_000,
                maximum_duration_ms=10_000,
                boundary_priority="high",
            ),
            MusicBoundarySpec(
                beat_id="payoff",
                preferred_duration_ms=10_200,
                minimum_duration_ms=8_000,
                maximum_duration_ms=15_000,
            ),
        ],
        [
            MusicBoundaryCue(
                cue_id="cue-capacity-edge",
                time_ms=10_000,
                kind="downbeat",
                strength=0.95,
            )
        ],
        total_duration_ms=20_000,
    )

    assert result.chapter_durations_ms == (10_000, 10_000)
    assert result.selections[0].cue_id == "cue-capacity-edge"
    assert result.cue_aligned_boundary_count == 1


def test_semantic_rhythm_solver_drives_cadence_without_music() -> None:
    result = solve_semantic_rhythm_durations(
        [
            SemanticRhythmSpec(
                beat_id="peak",
                minimum_duration_ms=5_000,
                preferred_duration_ms=10_000,
                maximum_duration_ms=15_000,
                cut_pressure=0.9,
                energy_role="peak",
            ),
            SemanticRhythmSpec(
                beat_id="hold",
                minimum_duration_ms=5_000,
                preferred_duration_ms=10_000,
                maximum_duration_ms=15_000,
                cut_pressure=0.2,
                energy_role="low_hold",
            ),
            SemanticRhythmSpec(
                beat_id="release",
                minimum_duration_ms=5_000,
                preferred_duration_ms=10_000,
                maximum_duration_ms=15_000,
                cut_pressure=0.3,
                energy_role="release",
            ),
        ],
        total_duration_ms=30_000,
    )

    durations = {
        selection.beat_id: selection.duration_ms
        for selection in result.selections
    }
    assert result.cadence_source == "semantic_attention_and_energy"
    assert sum(durations.values()) == 30_000
    assert durations["peak"] < durations["hold"] < durations["release"]


def test_music_boundary_solver_preserves_global_readability_bounds() -> None:
    result = solve_music_aligned_boundaries(
        [
            MusicBoundarySpec(
                beat_id="first",
                preferred_duration_ms=10_000,
                minimum_duration_ms=9_500,
                maximum_duration_ms=10_500,
                boundary_priority="high",
            ),
            MusicBoundarySpec(
                beat_id="second",
                preferred_duration_ms=10_000,
                minimum_duration_ms=9_500,
                maximum_duration_ms=10_500,
                boundary_priority="high",
            ),
            MusicBoundarySpec(
                beat_id="third",
                preferred_duration_ms=10_000,
                minimum_duration_ms=9_500,
                maximum_duration_ms=10_500,
            ),
        ],
        [
            MusicBoundaryCue(
                cue_id="cue-first-too-late-for-pair",
                time_ms=10_500,
                kind="accent",
                strength=1.0,
            ),
            MusicBoundaryCue(
                cue_id="cue-second",
                time_ms=19_600,
                kind="downbeat",
                strength=0.9,
            ),
        ],
        total_duration_ms=30_000,
    )

    assert all(
        9_500 <= duration <= 10_500
        for duration in result.chapter_durations_ms
    )
    assert result.selections[1].cue_id == "cue-second"
    assert result.selections[0].cue_id is None


def _cache_key(
    *,
    source_in_pts: int,
    presentation_sha256: str = "b" * 64,
) -> SegmentRenderCacheKey:
    return SegmentRenderCacheKey(
        source_sha256="a" * 64,
        source_in_pts=source_in_pts,
        source_out_pts=source_in_pts + 30,
        presentation_sha256=presentation_sha256,
        tracking_sha256="c" * 64,
        filter_graph_version="vertical-v1",
        aspect="9:16",
        width=1080,
        height=1920,
        fps_numerator=30,
        fps_denominator=1,
    )


def test_segment_cache_key_covers_presentation_and_timeline() -> None:
    baseline = _cache_key(source_in_pts=0)
    changed_presentation = _cache_key(
        source_in_pts=0,
        presentation_sha256="d" * 64,
    )
    changed_timeline = _cache_key(source_in_pts=30)

    assert baseline.digest != changed_presentation.digest
    assert baseline.digest != changed_timeline.digest


def test_local_repair_rerenders_only_changed_segment(tmp_path: Path) -> None:
    initial = [
        SegmentRenderRequest(segment_id="s1", cache_key=_cache_key(source_in_pts=0)),
        SegmentRenderRequest(segment_id="s2", cache_key=_cache_key(source_in_pts=30)),
        SegmentRenderRequest(segment_id="s3", cache_key=_cache_key(source_in_pts=60)),
    ]
    calls: list[str] = []

    def renderer(request: SegmentRenderRequest, output: Path) -> None:
        calls.append(request.segment_id)
        output.write_bytes(request.segment_id.encode("utf-8"))

    first = render_segments_incrementally(
        initial,
        cache_dir=tmp_path / "segments",
        renderer=renderer,
    )
    assert first.rendered_segment_ids == ("s1", "s2", "s3")

    revised = [
        initial[0],
        SegmentRenderRequest(
            segment_id="s2",
            cache_key=_cache_key(
                source_in_pts=30,
                presentation_sha256="d" * 64,
            ),
        ),
        initial[2],
    ]
    calls.clear()
    second = render_segments_incrementally(
        revised,
        cache_dir=tmp_path / "segments",
        renderer=renderer,
    )

    assert calls == ["s2"]
    assert second.rendered_segment_ids == ("s2",)
    assert second.reused_segment_ids == ("s1", "s3")
    assert len(concat_manifest_lines(second)) == 3
