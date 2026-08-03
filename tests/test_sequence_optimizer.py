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
    CandidateRouteSelection,
    MusicBoundaryCue,
    MusicBoundarySpec,
    RoundRobinFrontierAttemptResult,
    RoundRobinFrontierBeat,
    RoundRobinFrontierCandidate,
    RuntimeCueTimingBinding,
    RuntimeSegmentTiming,
    SegmentRenderCacheKey,
    SegmentRenderRequest,
    SemanticReplanReuseAuthority,
    SemanticRhythmSpec,
    SequenceOption,
    ResolvedTimelineV1,
    build_resolved_timeline,
    candidate_route_execution_sha256,
    concat_manifest_lines,
    initialize_round_robin_frontier,
    next_round_robin_frontier_attempt,
    optimize_sequence,
    optimize_pre_render_candidate_route,
    record_round_robin_frontier_attempt,
    rebuild_route_with_semantic_authorities,
    reconcile_runtime_sequence_timing,
    render_segments_incrementally,
    select_next_compatible_route,
    solve_semantic_rhythm_durations,
    solve_music_aligned_boundaries,
    validate_cumulative_music_boundary_bindings,
)


def _frontier_beat(
    beat_id: str,
    story_order: int,
    *candidates: tuple[str, bool],
    priority: str = "preferred",
) -> RoundRobinFrontierBeat:
    return RoundRobinFrontierBeat(
        beat_id=beat_id,
        story_order=story_order,
        priority=priority,
        candidates=tuple(
            RoundRobinFrontierCandidate(
                beat_id=beat_id,
                candidate_id=candidate_id,
                candidate_order=index,
                requires_exact_event=requires_exact_event,
            )
            for index, (candidate_id, requires_exact_event) in enumerate(
                candidates
            )
        ),
    )


def _record_frontier_outcome(state, outcome: str):
    attempt = next_round_robin_frontier_attempt(state)
    assert attempt is not None
    return record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=attempt,
            outcome=outcome,
        ),
    )


def _run_frontier_with_candidate_outcomes(beats):
    state = initialize_round_robin_frontier(beats)
    while (attempt := next_round_robin_frontier_attempt(state)) is not None:
        if attempt.stage == "local_preflight":
            outcome = "local_preflight_passed"
        elif attempt.stage == "exact_event":
            outcome = "exact_event_passed"
        elif attempt.candidate_id in {"a1", "c1"}:
            outcome = "grounding_failed"
        else:
            outcome = "grounding_accepted"
        state = record_round_robin_frontier_attempt(
            state,
            RoundRobinFrontierAttemptResult(
                attempt=attempt,
                outcome=outcome,
            ),
        )
    return state


def test_round_robin_frontier_finishes_round_one_before_round_two() -> None:
    beats = (
        _frontier_beat("c", 2, ("c1", False), ("c2", False)),
        _frontier_beat("a", 0, ("a1", False), ("a2", False)),
        _frontier_beat("b", 1, ("b1", True)),
    )

    state = _run_frontier_with_candidate_outcomes(beats)
    paid_attempts = [
        (
            row.attempt.beat_id,
            row.attempt.candidate_id,
            row.attempt.round_index,
            row.attempt.stage,
        )
        for row in state.attempt_history
        if row.attempt.paid
    ]

    assert paid_attempts == [
        ("a", "a1", 1, "grounding"),
        ("b", "b1", 1, "exact_event"),
        ("b", "b1", 1, "grounding"),
        ("c", "c1", 1, "grounding"),
        ("a", "a2", 2, "grounding"),
        ("c", "c2", 2, "grounding"),
    ]
    assert [row.status for row in state.beats] == [
        "accepted",
        "accepted",
        "accepted",
    ]


def test_round_robin_frontier_order_uses_explicit_story_order() -> None:
    beats = (
        _frontier_beat("a", 0, ("a1", False), ("a2", False)),
        _frontier_beat("b", 1, ("b1", True)),
        _frontier_beat("c", 2, ("c1", False), ("c2", False)),
    )

    forward = _run_frontier_with_candidate_outcomes(beats)
    reversed_input = _run_frontier_with_candidate_outcomes(
        tuple(reversed(beats))
    )

    forward_order = [
        (
            row.attempt.beat_id,
            row.attempt.candidate_id,
            row.attempt.stage,
            row.outcome,
        )
        for row in forward.attempt_history
    ]
    reversed_order = [
        (
            row.attempt.beat_id,
            row.attempt.candidate_id,
            row.attempt.stage,
            row.outcome,
        )
        for row in reversed_input.attempt_history
    ]
    assert reversed_order == forward_order


def test_round_robin_frontier_resolves_hard_before_earlier_preferred() -> None:
    state = _run_frontier_with_candidate_outcomes(
        (
            _frontier_beat(
                "preferred-opening",
                0,
                ("preferred-1", False),
                priority="preferred",
            ),
            _frontier_beat(
                "hard-payoff",
                1,
                ("hard-1", True),
                priority="hard",
            ),
        )
    )

    paid_attempts = [
        (row.attempt.beat_id, row.attempt.stage, row.attempt.priority)
        for row in state.attempt_history
        if row.attempt.paid
    ]

    assert paid_attempts == [
        ("hard-payoff", "exact_event", "hard"),
        ("hard-payoff", "grounding", "hard"),
        ("preferred-opening", "grounding", "preferred"),
    ]


def test_round_robin_frontier_local_failure_is_zero_paid() -> None:
    state = initialize_round_robin_frontier(
        (
            _frontier_beat(
                "beat",
                0,
                ("locally-bad", False),
                ("usable", False),
            ),
        )
    )

    state = _record_frontier_outcome(
        state,
        "local_preflight_failed",
    )
    next_attempt = next_round_robin_frontier_attempt(state)

    assert state.paid_calls_consumed == 0
    assert next_attempt is not None
    assert next_attempt.candidate_id == "usable"
    assert next_attempt.round_index == 1
    assert next_attempt.stage == "local_preflight"
    assert not next_attempt.paid


def test_round_robin_frontier_requires_exact_before_grounding() -> None:
    state = initialize_round_robin_frontier(
        (_frontier_beat("beat", 0, ("candidate", True)),)
    )
    state = _record_frontier_outcome(
        state,
        "local_preflight_passed",
    )
    exact_attempt = next_round_robin_frontier_attempt(state)
    assert exact_attempt is not None
    assert exact_attempt.stage == "exact_event"

    out_of_order = exact_attempt.model_copy(update={"stage": "grounding"})
    with pytest.raises(ValueError, match="stale or out-of-order"):
        record_round_robin_frontier_attempt(
            state,
            RoundRobinFrontierAttemptResult(
                attempt=out_of_order,
                outcome="grounding_accepted",
            ),
        )

    state = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=exact_attempt,
            outcome="exact_event_passed",
        ),
    )
    grounding_attempt = next_round_robin_frontier_attempt(state)
    assert grounding_attempt is not None
    assert grounding_attempt.stage == "grounding"
    assert grounding_attempt.candidate_id == "candidate"


def test_local_exhaustion_can_audit_one_scoped_semantic_replan_charge() -> None:
    """The decision is paid, even when exhaustion is discovered locally."""

    state = initialize_round_robin_frontier(
        (_frontier_beat("beat", 0, ("candidate", False)),)
    )
    attempt = next_round_robin_frontier_attempt(state)
    assert attempt is not None and attempt.stage == "local_preflight"

    recorded = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=attempt,
            outcome="local_preflight_failed",
            accepted_candidate_id="candidate",
            decision_codes=(
                "local_preflight_candidate_known_infeasible",
                "earlier_deferred_fallback_accepted_before_next_priority_tier",
            ),
            paid_calls_added=1,
        ),
    )

    assert recorded.paid_calls_consumed == 1
    assert recorded.beats[0].status == "accepted"


def test_round_robin_frontier_exact_failure_can_accept_earlier_deferred() -> None:
    state = initialize_round_robin_frontier(
        (
            _frontier_beat(
                "hard",
                0,
                ("deferred-panel", False),
                ("last-exact", True),
                priority="hard",
            ),
            _frontier_beat(
                "preferred",
                1,
                ("preferred-candidate", False),
                priority="preferred",
            ),
        )
    )
    state = _record_frontier_outcome(state, "local_preflight_passed")
    state = _record_frontier_outcome(state, "grounding_failed")
    state = _record_frontier_outcome(state, "local_preflight_passed")
    exact_attempt = next_round_robin_frontier_attempt(state)
    assert exact_attempt is not None
    assert exact_attempt.stage == "exact_event"

    state = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=exact_attempt,
            outcome="exact_event_failed",
            accepted_candidate_id="deferred-panel",
            decision_codes=("deferred_fallback_accepted",),
        ),
    )

    assert state.beats[0].status == "accepted"
    assert state.beats[0].accepted_candidate_id == "deferred-panel"
    next_attempt = next_round_robin_frontier_attempt(state)
    assert next_attempt is not None
    assert next_attempt.beat_id == "preferred"


def test_frontier_distinguishes_same_candidate_across_execution_hashes() -> None:
    first_execution_sha256 = "a" * 64
    second_execution_sha256 = "b" * 64
    state = initialize_round_robin_frontier(
        (
            RoundRobinFrontierBeat(
                beat_id="fold_hero",
                story_order=0,
                candidates=(
                    RoundRobinFrontierCandidate(
                        beat_id="fold_hero",
                        candidate_id="c8361",
                        candidate_execution_sha256=(
                            first_execution_sha256
                        ),
                        candidate_order=0,
                    ),
                    RoundRobinFrontierCandidate(
                        beat_id="fold_hero",
                        candidate_id="c8361",
                        candidate_execution_sha256=(
                            second_execution_sha256
                        ),
                        candidate_order=1,
                    ),
                ),
            ),
        )
    )

    first_local = next_round_robin_frontier_attempt(state)
    assert first_local is not None
    assert (
        first_local.candidate_execution_sha256
        == first_execution_sha256
    )
    state = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=first_local,
            outcome="local_preflight_passed",
        ),
    )
    first_grounding = next_round_robin_frontier_attempt(state)
    assert first_grounding is not None
    assert (
        first_grounding.candidate_execution_sha256
        == first_execution_sha256
    )
    state = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=first_grounding,
            outcome="grounding_failed",
        ),
    )
    second_local = next_round_robin_frontier_attempt(state)
    assert second_local is not None
    assert second_local.candidate_id == "c8361"
    assert (
        second_local.candidate_execution_sha256
        == second_execution_sha256
    )
    state = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=second_local,
            outcome="local_preflight_passed",
        ),
    )
    second_grounding = next_round_robin_frontier_attempt(state)
    assert second_grounding is not None
    state = record_round_robin_frontier_attempt(
        state,
        RoundRobinFrontierAttemptResult(
            attempt=second_grounding,
            outcome="grounding_accepted",
        ),
    )

    assert state.beats[0].accepted_candidate_id == "c8361"
    assert (
        state.beats[0].accepted_candidate_execution_sha256
        == second_execution_sha256
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


def _timed_route_option(
    beat_id: str,
    candidate_id: str,
    source_marker: str,
    *,
    source_in_ms: int,
    source_out_ms: int,
    rank: int = 1,
    confidence: float = 0.9,
    reuse_mode: str = "none",
) -> CandidateRouteOption:
    duration_ms = source_out_ms - source_in_ms
    return CandidateRouteOption(
        beat_id=beat_id,
        candidate_id=candidate_id,
        source_asset_id="sha256:" + source_marker * 64,
        source_clip_id=f"clip-{source_marker}",
        event_id=f"event-{candidate_id}",
        planner_rank=rank,
        semantic_confidence=confidence,
        trim_duration_ms=duration_ms,
        minimum_readable_ms=duration_ms,
        preferred_readable_ms=duration_ms,
        maximum_readable_ms=duration_ms,
        fixed_source_in_ms=source_in_ms,
        fixed_source_out_ms=source_out_ms,
        reuse_mode=reuse_mode,
        reuse_justification=(
            "explicit test reuse authority"
            if reuse_mode != "none"
            else None
        ),
        candidate_timing_sha256=source_marker * 64,
    )


def _movable_route_option(
    beat_id: str,
    candidate_id: str,
    source_marker: str,
    *,
    duration_ms: int,
    safe_window_start_ms: int,
    safe_window_end_ms: int,
    source_anchor_ms: int,
    reuse_mode: str = "none",
) -> CandidateRouteOption:
    return CandidateRouteOption(
        beat_id=beat_id,
        candidate_id=candidate_id,
        source_asset_id="sha256:" + source_marker * 64,
        source_clip_id=f"clip-{source_marker}",
        event_id=f"event-{candidate_id}",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=duration_ms,
        minimum_readable_ms=duration_ms,
        preferred_readable_ms=duration_ms,
        maximum_readable_ms=duration_ms,
        safe_capacity_ms=safe_window_end_ms - safe_window_start_ms,
        safe_window_start_ms=safe_window_start_ms,
        safe_window_end_ms=safe_window_end_ms,
        source_anchor_ms=source_anchor_ms,
        reuse_mode=reuse_mode,
        reuse_justification=(
            "explicit test reuse authority"
            if reuse_mode != "none"
            else None
        ),
        candidate_timing_sha256=source_marker * 64,
    )


def test_pre_render_route_preserves_gemini_rank_without_measured_hard_conflict() -> None:
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

    # The lightweight pre-render facts do not prove an interval collision, so
    # local variety scoring must not overrule Gemini's rank-1 selection. A
    # later measured reuse conflict is a hard gate and reaches scoped replan.
    assert [row.candidate_id for row in route.selections] == ["a1", "b1"]
    assert "gemini_candidate_rank_preserved" in (
        route.selections[1].decision_codes
    )
    # The alternate complete route remains a valid strict-frontier attempt,
    # so its candidate needs the same immutable timing binding as the winner.
    for complete_route in route.ranked_routes:
        for selection in complete_route.selections:
            assert selection.candidate_id in route.option_bindings_by_beat[
                selection.beat_id
            ]


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


def test_overlapping_source_without_authority_is_excluded_before_route() -> None:
    first = _timed_route_option(
        "opening",
        "opening-primary",
        "a",
        source_in_ms=0,
        source_out_ms=3_000,
    )
    unauthorized_overlap = _timed_route_option(
        "payoff",
        "payoff-overlap",
        "a",
        source_in_ms=1_000,
        source_out_ms=4_000,
        confidence=1.0,
    )
    alternate = _timed_route_option(
        "payoff",
        "payoff-alternate",
        "b",
        source_in_ms=0,
        source_out_ms=3_000,
        rank=2,
        confidence=0.2,
    )

    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="opening", options=(first,)),
            CandidateRouteBeat(
                beat_id="payoff",
                options=(unauthorized_overlap, alternate),
            ),
        )
    )

    assert [row.candidate_id for row in result.selections] == [
        "opening-primary",
        "payoff-alternate",
    ]
    assert all(
        "payoff-overlap"
        not in {selection.candidate_id for selection in route.selections}
        for route in result.ranked_routes
    )


def test_non_overlapping_distinct_interval_reuse_is_hard_safe() -> None:
    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="opening",
                options=(
                    _timed_route_option(
                        "opening",
                        "opening-a",
                        "a",
                        source_in_ms=0,
                        source_out_ms=3_000,
                    ),
                ),
            ),
            CandidateRouteBeat(
                beat_id="closing",
                options=(
                    _timed_route_option(
                        "closing",
                        "closing-a",
                        "a",
                        source_in_ms=5_000,
                        source_out_ms=8_000,
                        reuse_mode="distinct_interval",
                    ),
                ),
            ),
        )
    )

    assert [row.candidate_id for row in result.selections] == [
        "opening-a",
        "closing-a",
    ]
    assert result.selections[1].reuse_mode == "distinct_interval"


def test_three_distinct_intervals_from_one_source_are_hard_safe() -> None:
    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="opening",
                options=(
                    _timed_route_option(
                        "opening",
                        "opening-a",
                        "a",
                        source_in_ms=0,
                        source_out_ms=3_000,
                    ),
                ),
            ),
            CandidateRouteBeat(
                beat_id="detail",
                options=(
                    _timed_route_option(
                        "detail",
                        "detail-a",
                        "a",
                        source_in_ms=3_000,
                        source_out_ms=6_000,
                        reuse_mode="distinct_interval",
                    ),
                ),
            ),
            CandidateRouteBeat(
                beat_id="closing",
                options=(
                    _timed_route_option(
                        "closing",
                        "closing-a",
                        "a",
                        source_in_ms=6_000,
                        source_out_ms=9_000,
                        reuse_mode="distinct_interval",
                    ),
                ),
            ),
        )
    )

    assert [selection.source_in_ms for selection in result.selections] == [
        0,
        3_000,
        6_000,
    ]


def test_mixed_alternate_presentation_and_reprise_support_three_source_uses() -> None:
    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="opening",
                options=(
                    _movable_route_option(
                        "opening",
                        "opening-a",
                        "a",
                        duration_ms=4_000,
                        safe_window_start_ms=0,
                        safe_window_end_ms=9_643,
                        source_anchor_ms=7_000,
                    ),
                ),
            ),
            CandidateRouteBeat(
                beat_id="detail",
                options=(
                    _movable_route_option(
                        "detail",
                        "detail-a",
                        "a",
                        duration_ms=4_000,
                        safe_window_start_ms=0,
                        safe_window_end_ms=9_643,
                        source_anchor_ms=7_000,
                        reuse_mode="alternate_presentation",
                    ),
                ),
            ),
            CandidateRouteBeat(
                beat_id="closing",
                options=(
                    _movable_route_option(
                        "closing",
                        "closing-a",
                        "a",
                        duration_ms=4_000,
                        safe_window_start_ms=0,
                        safe_window_end_ms=9_643,
                        source_anchor_ms=7_000,
                        reuse_mode="editorial_reprise",
                    ),
                ),
            ),
        ),
        max_editorial_reprise_overlap_fraction=0.5,
    )

    intervals = [
        (selection.source_in_ms, selection.source_out_ms)
        for selection in result.selections
    ]
    assert all(interval[0] is not None and interval[1] is not None for interval in intervals)
    closing_interval = intervals[-1]
    for interval in intervals[:-1]:
        overlap = max(
            0,
            min(interval[1], closing_interval[1])
            - max(interval[0], closing_interval[0]),
        )
        assert overlap <= 2_000


def test_semantic_replan_rebuilds_full_route_after_gemini_reuse_authority() -> None:
    """A reuse decision rebuilds every hard route constraint, never a splice."""

    opening = _timed_route_option(
        "opening",
        "opening-a",
        "a",
        source_in_ms=0,
        source_out_ms=3_000,
    )
    fold_primary = _timed_route_option(
        "fold",
        "fold-b",
        "b",
        source_in_ms=0,
        source_out_ms=3_000,
        rank=1,
    )
    fold_reuse_candidate = _timed_route_option(
        "fold",
        "fold-a-detail",
        "a",
        source_in_ms=3_000,
        source_out_ms=6_000,
        rank=2,
    )
    closing = _timed_route_option(
        "closing",
        "closing-c",
        "c",
        source_in_ms=0,
        source_out_ms=3_000,
    )
    frontier = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="opening", options=(opening,)),
            CandidateRouteBeat(
                beat_id="fold",
                options=(fold_primary, fold_reuse_candidate),
            ),
            CandidateRouteBeat(beat_id="closing", options=(closing,)),
        )
    )

    assert [selection.candidate_id for selection in frontier.selections] == [
        "opening-a",
        "fold-b",
        "closing-c",
    ]
    replan_binding = next(
        binding
        for binding in frontier.semantic_replan_candidate_bindings_by_beat["fold"]
        if binding.option.candidate_id == "fold-a-detail"
    )
    assert replan_binding.replan_required_codes == (
        "source_reuse_authority_missing",
    )
    assert "fold-a-detail" not in frontier.option_bindings_by_beat["fold"]

    rebuilt = rebuild_route_with_semantic_authorities(
        frontier,
        selected_candidate_ids_by_beat={"fold": "fold-a-detail"},
        reuse_authorities_by_beat={
            "fold": SemanticReplanReuseAuthority(
                beat_id="fold",
                candidate_id="fold-a-detail",
                reuse_mode="distinct_interval",
                reuse_justification="The later detail exposes a different state.",
                reuse_of_beat_ids=("opening",),
            )
        },
    )

    assert [selection.candidate_id for selection in rebuilt.selections] == [
        "opening-a",
        "fold-a-detail",
        "closing-c",
    ]
    assert rebuilt.selections[1].reuse_mode == "distinct_interval"
    assert rebuilt.selections[1].candidate_execution_sha256 != (
        frontier.selections[1].candidate_execution_sha256
    )


def test_local_preflight_replan_freezes_prepared_execution_vector() -> None:
    """A grouped choice cannot stretch a clean execution on a later resume."""

    def movable(beat_id: str, candidate_id: str, marker: str) -> CandidateRouteOption:
        return CandidateRouteOption(
            beat_id=beat_id,
            candidate_id=candidate_id,
            source_asset_id="sha256:" + marker * 64,
            source_clip_id=f"clip-{marker}",
            event_id=f"event-{candidate_id}",
            planner_rank=1,
            semantic_confidence=0.9,
            trim_duration_ms=6_000,
            minimum_readable_ms=3_500,
            preferred_readable_ms=6_000,
            maximum_readable_ms=6_500,
            safe_capacity_ms=11_045,
            safe_window_start_ms=0,
            safe_window_end_ms=11_045,
            source_anchor_ms=4_000,
            candidate_timing_sha256=marker * 64,
        )

    unchanged = movable("unchanged", "unchanged-a", "a")
    affected = movable("affected", "affected-a", "b")
    frontier = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="unchanged", options=(unchanged,)),
            CandidateRouteBeat(beat_id="affected", options=(affected,)),
        ),
        target_duration_ms=12_000,
    )
    assert [selection.trim_duration_ms for selection in frontier.selections] == [
        6_000,
        6_000,
    ]

    def execution(
        option: CandidateRouteOption,
        *,
        start_ms: int,
        duration_ms: int,
    ) -> CandidateRouteSelection:
        return CandidateRouteSelection(
            beat_id=option.beat_id,
            candidate_id=option.candidate_id,
            source_asset_id=option.source_asset_id,
            event_id=option.event_id,
            trim_duration_ms=duration_ms,
            cue_id=option.cue_id,
            cue_aligned=option.cue_aligned,
            presentation_mode=option.presentation_mode,
            entry_composition=option.entry_composition,
            exit_composition=option.exit_composition,
            decision_codes=("measured",),
            source_clip_id=option.source_clip_id,
            source_in_ms=start_ms,
            source_out_ms=start_ms + duration_ms,
            candidate_execution_sha256=candidate_route_execution_sha256(
                option,
                duration_ms=duration_ms,
                source_in_ms=start_ms,
                source_out_ms=start_ms + duration_ms,
            ),
        )

    prepared_clean = execution(affected, start_ms=2_250, duration_ms=3_500)
    preserved_unaffected = execution(unchanged, start_ms=1_000, duration_ms=6_000)
    rebuilt = rebuild_route_with_semantic_authorities(
        frontier,
        # The selected candidate is unchanged; the new authority is the
        # prepared execution identity, not another optimizer duration pass.
        selected_candidate_ids_by_beat={"affected": "affected-a"},
        reuse_authorities_by_beat={},
        frozen_execution_bindings_by_beat={
            "unchanged": preserved_unaffected,
            "affected": prepared_clean,
        },
        minimum_duration_ms=9_000,
        maximum_duration_ms=13_000,
        target_duration_ms=12_000,
    )

    by_beat = {selection.beat_id: selection for selection in rebuilt.selections}
    assert by_beat["affected"].trim_duration_ms == 3_500
    assert by_beat["affected"].source_in_ms == 2_250
    assert by_beat["affected"].source_out_ms == 5_750
    assert by_beat["affected"].candidate_execution_sha256 == (
        prepared_clean.candidate_execution_sha256
    )
    assert by_beat["unchanged"].candidate_execution_sha256 == (
        preserved_unaffected.candidate_execution_sha256
    )


def test_semantic_replan_cannot_attach_reuse_authority_to_safe_candidate() -> None:
    """A replan may authorize only the measured missing-reuse frontier."""

    opening = _timed_route_option(
        "opening", "opening-a", "a", source_in_ms=0, source_out_ms=3_000
    )
    fold = _timed_route_option(
        "fold", "fold-b", "b", source_in_ms=0, source_out_ms=3_000
    )
    frontier = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="opening", options=(opening,)),
            CandidateRouteBeat(beat_id="fold", options=(fold,)),
        )
    )

    with pytest.raises(ValueError, match="without a measured reuse-authority"):
        rebuild_route_with_semantic_authorities(
            frontier,
            selected_candidate_ids_by_beat={"fold": "fold-b"},
            reuse_authorities_by_beat={
                "fold": SemanticReplanReuseAuthority(
                    beat_id="fold",
                    candidate_id="fold-b",
                    reuse_mode="distinct_interval",
                    reuse_justification="Not needed, so this must fail.",
                    reuse_of_beat_ids=("opening",),
                )
            },
        )


def test_center_overlap_repositions_safe_window_for_distinct_interval() -> None:
    fold_hero = _movable_route_option(
        "fold_hero",
        "fold-c8361",
        "a",
        duration_ms=5_500,
        safe_window_start_ms=0,
        safe_window_end_ms=9_009,
        source_anchor_ms=6_000,
    )
    closing = _timed_route_option(
        "closing",
        "closing-c8361",
        "a",
        source_in_ms=0,
        source_out_ms=3_500,
        reuse_mode="distinct_interval",
    )

    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="fold_hero",
                options=(fold_hero,),
            ),
            CandidateRouteBeat(
                beat_id="closing",
                options=(closing,),
            ),
        )
    )

    assert (
        result.selections[0].source_in_ms,
        result.selections[0].source_out_ms,
    ) == (3_500, 9_000)
    assert (
        result.selections[1].source_in_ms,
        result.selections[1].source_out_ms,
    ) == (0, 3_500)
    assert result.selections[0].source_in_ms != 3_250
    assert "source_interval_jointly_repositioned" in (
        result.selections[0].decision_codes
    )
    assert result.selections[0].candidate_execution_sha256 == (
        candidate_route_execution_sha256(
            fold_hero,
            duration_ms=5_500,
            source_in_ms=3_500,
            source_out_ms=9_000,
        )
    )


def test_center_overlap_still_rejects_when_safe_window_cannot_move() -> None:
    immovable_fold = _movable_route_option(
        "fold_hero",
        "fold-c8361",
        "a",
        duration_ms=5_500,
        safe_window_start_ms=0,
        safe_window_end_ms=8_750,
        source_anchor_ms=6_000,
    )
    closing = _timed_route_option(
        "closing",
        "closing-c8361",
        "a",
        source_in_ms=0,
        source_out_ms=3_500,
        reuse_mode="distinct_interval",
    )

    with pytest.raises(ValueError, match="no options for beat closing"):
        optimize_pre_render_candidate_route(
            (
                CandidateRouteBeat(
                    beat_id="fold_hero",
                    options=(immovable_fold,),
                ),
                CandidateRouteBeat(
                    beat_id="closing",
                    options=(closing,),
                ),
            )
        )


def test_source_anchor_at_half_open_out_boundary_is_not_a_valid_execution() -> None:
    first = _movable_route_option(
        "watch_lineup",
        "lineup",
        "a",
        duration_ms=4_000,
        safe_window_start_ms=0,
        safe_window_end_ms=23_524,
        source_anchor_ms=6_000,
    )
    second = _movable_route_option(
        "watch_ui",
        "ui",
        "a",
        duration_ms=5_000,
        safe_window_start_ms=0,
        safe_window_end_ms=23_524,
        source_anchor_ms=6_000,
        reuse_mode="distinct_interval",
    )

    with pytest.raises(ValueError, match="no options for beat watch_ui"):
        optimize_pre_render_candidate_route(
            (
                CandidateRouteBeat(
                    beat_id="watch_lineup",
                    options=(first,),
                ),
                CandidateRouteBeat(
                    beat_id="watch_ui",
                    options=(second,),
                ),
            )
        )


def test_independent_fallback_conflict_never_becomes_complete_route() -> None:
    beats = (
        CandidateRouteBeat(
            beat_id="a",
            options=(
                _timed_route_option(
                    "a",
                    "a-primary",
                    "a",
                    source_in_ms=0,
                    source_out_ms=3_000,
                    confidence=0.95,
                ),
                _timed_route_option(
                    "a",
                    "a-fallback",
                    "c",
                    source_in_ms=0,
                    source_out_ms=3_000,
                    rank=2,
                    confidence=0.8,
                ),
            ),
        ),
        CandidateRouteBeat(
            beat_id="b",
            options=(
                _timed_route_option(
                    "b",
                    "b-primary",
                    "b",
                    source_in_ms=0,
                    source_out_ms=3_000,
                    confidence=0.95,
                ),
                _timed_route_option(
                    "b",
                    "b-fallback",
                    "c",
                    source_in_ms=1_000,
                    source_out_ms=4_000,
                    rank=2,
                    confidence=0.8,
                ),
            ),
        ),
    )

    result = optimize_pre_render_candidate_route(beats)

    assert "a-fallback" in result.fallback_candidate_ids_by_beat["a"]
    assert "b-fallback" in result.fallback_candidate_ids_by_beat["b"]
    assert all(
        {
            "a-fallback",
            "b-fallback",
        }
        - {selection.candidate_id for selection in route.selections}
        for route in result.ranked_routes
    )


def test_allocated_duration_changes_candidate_execution_hash() -> None:
    option = CandidateRouteOption(
        beat_id="detail",
        candidate_id="detail-a",
        source_asset_id="sha256:" + "a" * 64,
        source_clip_id="clip-a",
        event_id="detail-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=4_000,
        minimum_readable_ms=2_000,
        preferred_readable_ms=4_000,
        maximum_readable_ms=8_000,
        safe_capacity_ms=10_000,
        safe_window_start_ms=0,
        safe_window_end_ms=10_000,
        source_anchor_ms=5_000,
        candidate_timing_sha256="a" * 64,
    )
    beat = CandidateRouteBeat(beat_id="detail", options=(option,))

    short = optimize_pre_render_candidate_route(
        (beat,),
        target_duration_ms=4_000,
    )
    long = optimize_pre_render_candidate_route(
        (beat,),
        target_duration_ms=6_000,
    )

    assert short.total_duration_ms == 4_000
    assert long.total_duration_ms == 6_000
    assert (
        short.selections[0].candidate_execution_sha256
        != long.selections[0].candidate_execution_sha256
    )
    assert short.selections[0].candidate_execution_sha256 == (
        candidate_route_execution_sha256(
            option,
            duration_ms=4_000,
            source_in_ms=3_000,
            source_out_ms=7_000,
        )
    )


def test_ranked_routes_include_bounded_duration_recovery_execution() -> None:
    opening = CandidateRouteOption(
        beat_id="opening",
        candidate_id="opening-a",
        source_asset_id="sha256:" + "a" * 64,
        source_clip_id="clip-a",
        event_id="opening-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=6_500,
        minimum_readable_ms=5_000,
        preferred_readable_ms=6_500,
        maximum_readable_ms=8_000,
        safe_capacity_ms=10_000,
        safe_window_start_ms=0,
        safe_window_end_ms=10_000,
        source_anchor_ms=5_000,
        candidate_timing_sha256="a" * 64,
    )
    closing = CandidateRouteOption(
        beat_id="closing",
        candidate_id="closing-a",
        source_asset_id="sha256:" + "b" * 64,
        source_clip_id="clip-b",
        event_id="closing-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=3_500,
        minimum_readable_ms=2_000,
        preferred_readable_ms=3_500,
        maximum_readable_ms=3_500,
        safe_capacity_ms=9_000,
        safe_window_start_ms=0,
        safe_window_end_ms=9_000,
        source_anchor_ms=0,
        candidate_timing_sha256="b" * 64,
    )

    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="opening", options=(opening,)),
            CandidateRouteBeat(beat_id="closing", options=(closing,)),
        ),
        target_duration_ms=10_000,
    )

    duration_vectors = {
        tuple(
            selection.trim_duration_ms
            for selection in route.selections
        )
        for route in result.ranked_routes
    }
    assert (6_500, 3_500) in duration_vectors
    assert (8_000, 2_000) in duration_vectors
    assert all(route.total_duration_ms == 10_000 for route in result.ranked_routes)
    recovery = next(
        route
        for route in result.ranked_routes
        if tuple(
            selection.trim_duration_ms
            for selection in route.selections
        )
        == (8_000, 2_000)
    )
    assert "bounded_duration_recovery_variant" in recovery.decision_codes
    assert (
        recovery.selections[1].candidate_execution_sha256
        != result.selections[1].candidate_execution_sha256
    )
    assert (
        recovery.selections[1].source_in_ms,
        recovery.selections[1].source_out_ms,
    ) == (0, 2_000)
    # Every execution exposed to the production frontier, including a bounded
    # duration variant or a globally different candidate route, must retain
    # its candidate-local timing fact. Otherwise strict preflight would know
    # the selected source interval but not the safe window that authorizes it.
    for route in result.ranked_routes:
        for selection in route.selections:
            assert selection.candidate_id in result.option_bindings_by_beat[
                selection.beat_id
            ]


def test_pre_render_duration_variant_rejects_internal_music_boundary_drift() -> None:
    """Equal total runtime is insufficient when an internal cue is bound."""

    opening = CandidateRouteOption(
        beat_id="opening",
        candidate_id="opening-a",
        source_asset_id="sha256:" + "a" * 64,
        source_clip_id="clip-a",
        event_id="opening-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=6_500,
        minimum_readable_ms=5_000,
        preferred_readable_ms=6_500,
        maximum_readable_ms=8_000,
        safe_capacity_ms=10_000,
        safe_window_start_ms=0,
        safe_window_end_ms=10_000,
        source_anchor_ms=5_000,
        candidate_timing_sha256="a" * 64,
        cue_id="downbeat-06500",
        cue_aligned=True,
        exit_cue_time_ms=6_500,
    )
    closing = CandidateRouteOption(
        beat_id="closing",
        candidate_id="closing-a",
        source_asset_id="sha256:" + "b" * 64,
        source_clip_id="clip-b",
        event_id="closing-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=3_500,
        minimum_readable_ms=2_000,
        preferred_readable_ms=3_500,
        maximum_readable_ms=3_500,
        safe_capacity_ms=9_000,
        safe_window_start_ms=0,
        safe_window_end_ms=9_000,
        source_anchor_ms=0,
        candidate_timing_sha256="b" * 64,
        cue_id="ending-10000",
        cue_aligned=True,
        exit_cue_time_ms=10_000,
    )

    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="opening", options=(opening,)),
            CandidateRouteBeat(beat_id="closing", options=(closing,)),
        ),
        target_duration_ms=10_000,
        cue_tolerance_ms=0,
    )

    assert {
        tuple(selection.trim_duration_ms for selection in route.selections)
        for route in result.ranked_routes
    } == {(6_500, 3_500)}


def test_pre_render_cumulative_music_boundary_binding_accepts_exact_vector() -> None:
    option = CandidateRouteOption(
        beat_id="payoff",
        candidate_id="payoff-a",
        source_asset_id="sha256:" + "c" * 64,
        source_clip_id="clip-c",
        event_id="payoff-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=4_000,
        minimum_readable_ms=4_000,
        preferred_readable_ms=4_000,
        maximum_readable_ms=4_000,
        fixed_source_in_ms=0,
        fixed_source_out_ms=4_000,
        candidate_timing_sha256="c" * 64,
        cue_id="ending-04000",
        cue_aligned=True,
        exit_cue_time_ms=4_000,
    )

    result = optimize_pre_render_candidate_route(
        (CandidateRouteBeat(beat_id="payoff", options=(option,)),),
        target_duration_ms=4_000,
        cue_tolerance_ms=0,
    )

    validate_cumulative_music_boundary_bindings(
        result.selections,
        cue_tolerance_ms=0,
    )
    assert result.selections[0].exit_cue_time_ms == 4_000


def test_duration_recovery_frontier_is_locally_bounded() -> None:
    opening = CandidateRouteOption(
        beat_id="opening",
        candidate_id="opening-a",
        source_asset_id="sha256:" + "a" * 64,
        event_id="opening-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=6_500,
        minimum_readable_ms=5_000,
        preferred_readable_ms=6_500,
        maximum_readable_ms=8_000,
    )
    closing = CandidateRouteOption(
        beat_id="closing",
        candidate_id="closing-a",
        source_asset_id="sha256:" + "b" * 64,
        event_id="closing-event",
        planner_rank=1,
        semantic_confidence=0.9,
        trim_duration_ms=3_500,
        minimum_readable_ms=2_000,
        preferred_readable_ms=3_500,
        maximum_readable_ms=3_500,
    )

    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(beat_id="opening", options=(opening,)),
            CandidateRouteBeat(beat_id="closing", options=(closing,)),
        ),
        target_duration_ms=10_000,
        max_duration_variants_per_route=1,
    )

    assert len(result.ranked_routes) == 1
    assert [
        selection.trim_duration_ms
        for selection in result.ranked_routes[0].selections
    ] == [6_500, 3_500]


def test_next_route_preserves_accepted_execution_and_skips_failed_one() -> None:
    result = optimize_pre_render_candidate_route(
        (
            CandidateRouteBeat(
                beat_id="a",
                options=(
                    _timed_route_option(
                        "a",
                        "a-primary",
                        "a",
                        source_in_ms=0,
                        source_out_ms=3_000,
                    ),
                ),
            ),
            CandidateRouteBeat(
                beat_id="b",
                options=(
                    _timed_route_option(
                        "b",
                        "b-primary",
                        "b",
                        source_in_ms=0,
                        source_out_ms=3_000,
                    ),
                    _timed_route_option(
                        "b",
                        "b-fallback",
                        "c",
                        source_in_ms=0,
                        source_out_ms=3_000,
                        rank=2,
                        confidence=0.7,
                    ),
                ),
            ),
        )
    )
    first = result.ranked_routes[0]
    accepted_a = first.selections[0].candidate_execution_sha256
    failed_b = first.selections[1].candidate_execution_sha256
    assert accepted_a is not None
    assert failed_b is not None

    next_route = select_next_compatible_route(
        result.ranked_routes,
        accepted_execution_sha256_by_beat={"a": accepted_a},
        failed_execution_sha256s=(failed_b,),
        after_route_id=first.route_id,
    )

    assert next_route is not None
    assert next_route.selections[0].candidate_execution_sha256 == accepted_a
    assert next_route.selections[1].candidate_id == "b-fallback"


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


def test_runtime_reconciliation_allows_only_bounded_pts_total_quantization() -> None:
    result = reconcile_runtime_sequence_timing(
        (
            _runtime_timing(
                "opening",
                planned_duration_ms=10_000,
                actual_duration_ms=9_964,
            ),
        ),
        minimum_total_duration_ms=10_000,
        maximum_total_duration_ms=11_000,
        pts_duration_quantization_tolerance_ms=38,
    )

    assert result.outcome == "reconciled"
    blocked = reconcile_runtime_sequence_timing(
        (
            _runtime_timing(
                "opening",
                planned_duration_ms=10_000,
                actual_duration_ms=9_961,
            ),
        ),
        minimum_total_duration_ms=10_000,
        maximum_total_duration_ms=11_000,
        pts_duration_quantization_tolerance_ms=38,
    )
    assert blocked.outcome == "blocked"
    assert blocked.failure_codes == ("resolved_total_duration_below_minimum",)


def test_resolved_timeline_is_hash_bound_per_aspect() -> None:
    reconciliation = reconcile_runtime_sequence_timing(
        (
            _runtime_timing(
                "setup",
                planned_duration_ms=7_587,
                actual_duration_ms=6_973,
            ),
            _runtime_timing(
                "closing",
                planned_duration_ms=5_000,
                actual_duration_ms=5_000,
            ),
        )
    )

    vertical = build_resolved_timeline(
        aspect="9:16",
        reconciliation=reconciliation,
        music_output_timeline_sha256="a" * 64,
    )
    horizontal = build_resolved_timeline(
        aspect="16:9",
        reconciliation=reconciliation,
        music_output_timeline_sha256="a" * 64,
    )

    assert vertical.resolved_total_duration_ms == 11_973
    assert vertical.definition_sha256 != horizontal.definition_sha256
    with pytest.raises(ValidationError, match="definition hash"):
        ResolvedTimelineV1.model_validate(
            {
                **vertical.model_dump(mode="json"),
                "definition_sha256": "f" * 64,
            }
        )
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


def test_runtime_timing_allows_only_one_frame_of_pts_boundary_quantization() -> None:
    timing = RuntimeSegmentTiming(
        beat_id="opening",
        planned_candidate_id="primary",
        runtime_candidate_id="primary",
        source_asset_id="sha256:" + "d" * 64,
        planned_duration_ms=6_000,
        actual_source_capacity_ms=6_006,
        actual_duration_ms=6_006,
        minimum_readable_ms=5_000,
    )

    assert timing.actual_duration_ms == 6_006
    with pytest.raises(ValidationError, match="PTS quantization tolerance"):
        RuntimeSegmentTiming(
            beat_id="opening",
            planned_candidate_id="primary",
            runtime_candidate_id="primary",
            source_asset_id="sha256:" + "d" * 64,
            planned_duration_ms=6_000,
            actual_source_capacity_ms=6_100,
            actual_duration_ms=6_035,
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
