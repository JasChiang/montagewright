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
    SegmentRenderCacheKey,
    SegmentRenderRequest,
    SequenceOption,
    concat_manifest_lines,
    optimize_sequence,
    render_segments_incrementally,
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
