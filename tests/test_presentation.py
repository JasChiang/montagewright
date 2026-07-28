from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
)
from jascue_video_lab.editing_capabilities import (
    autonomous_production_capability_catalog,
)
from jascue_video_lab.event_lock import (
    ExactEventLockV2,
    ExactEventResolverProvenance,
)
from jascue_video_lab.gemini import GeminiLabClient, MODEL_ID
from jascue_video_lab.presentation import (
    GroundingTargetRequest,
    PresentationTarget,
    choose_two_panel_layout,
    compile_intentional_freeze,
    compile_minimal_camera_motion,
    compile_presentation,
    shared_sam_seeds_from_grounding,
    two_panel_ffmpeg_filter,
)


def _policy() -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=85_000,
            min_ms=75_000,
            max_ms=90_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
    )


def _target(
    target_id: str,
    box: tuple[int, int, int, int],
    *,
    asset: str = "sha256:" + "a" * 64,
    pts: int = 30,
) -> PresentationTarget:
    return PresentationTarget(
        target_id=target_id,
        source_asset_id=asset,
        source_pts=pts,
        box_2d=box,
    )


def _event_lock(
    *,
    frame_hash: str = "b" * 64,
    event_type: str = "group_laugh_reaction_peak",
) -> ExactEventLockV2:
    return ExactEventLockV2(
        event_id="event-1",
        event_type=event_type,
        source_asset_id="sha256:" + "a" * 64,
        source_frame_id="DF000004",
        source_pts=120,
        source_time_ms=4_000,
        source_frame_hash=frame_hash,
        support_window_start_frame_id="DF000003",
        support_window_end_frame_id="DF000005",
        support_window_start_ms=3_900,
        support_window_end_ms=4_100,
        confidence=0.9,
        resolver=ExactEventResolverProvenance(
            local_bracket_method="frame_difference",
            sampling_fps=8,
            gemini_interaction_id="exact-1",
            contact_sheet_hashes=("c" * 64,),
        ),
        input_artifact_hashes=("sha256:" + "d" * 64,),
        generated_at="now",
    )


def test_static_subject_compiles_without_synthetic_motion() -> None:
    compilation = compile_presentation(
        targets=[_target("device", (450, 300, 550, 700))],
        source_width=1920,
        source_height=1080,
        relation_mode="single_subject",
        policy=_policy(),
    )

    assert compilation.mode == "static_full_bleed_crop"
    assert compilation.static_crop_box_2d is not None
    assert not any(
        "motion" in code for code in compilation.decision_codes
    )
    assert compilation.paid_model_calls_added == 0


def test_tracker_jitter_stays_inside_deadband_without_drift() -> None:
    decision = compile_minimal_camera_motion(
        (0.50, 0.505, 0.498, 0.502),
        movement_motivated=True,
    )

    assert decision.mode == "hold"
    assert decision.synthetic_reversal_count == 0
    assert decision.settle_required is False


def test_optimizable_center_left_right_avoids_initial_reversal() -> None:
    decision = compile_minimal_camera_motion(
        (0.50, 0.20, 0.80),
        movement_motivated=True,
        initial_position_optimizable=True,
    )

    assert decision.mode == "minimal_monotonic_move"
    assert decision.normalized_x_values == (0.20, 0.20, 0.80)
    assert decision.synthetic_reversal_count == 0
    assert decision.settle_required is True


def test_semantic_order_is_not_reordered_to_hide_a_reversal() -> None:
    decision = compile_minimal_camera_motion(
        (0.50, 0.20, 0.80),
        movement_motivated=True,
        initial_position_optimizable=False,
    )

    assert decision.mode == "hard_cut"
    assert decision.normalized_x_values == (0.50, 0.20, 0.80)
    assert decision.synthetic_reversal_count == 1


def test_existing_source_pan_suppresses_synthetic_camera_motion() -> None:
    decision = compile_minimal_camera_motion(
        (0.20, 0.80),
        source_camera_x_values=(0.15, 0.85),
        movement_motivated=True,
    )

    assert decision.mode == "hold"
    assert decision.source_motion.direction == "right"
    assert decision.movement_motivation == "none"


def test_impossible_two_device_crop_uses_scale_locked_two_panel() -> None:
    compilation = compile_presentation(
        targets=[
            _target("device-a", (20, 250, 270, 750)),
            _target("device-b", (730, 250, 980, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
        physical_scale_comparison=True,
    )

    assert compilation.mode == "two_panel_layout"
    assert compilation.panel_layout is not None
    assert compilation.panel_layout.relative_scale_policy == "locked"
    first, second = compilation.panel_layout.panels
    assert first.source_pts == second.source_pts
    assert first.source_asset_id == second.source_asset_id
    assert first.crop_box_2d[2] - first.crop_box_2d[0] == (
        second.crop_box_2d[2] - second.crop_box_2d[0]
    )
    assert compilation.paid_model_calls_added == 0
    filter_graph = two_panel_ffmpeg_filter(compilation.panel_layout)
    assert "[0:v]split=2" in filter_graph
    assert filter_graph.count("[0:v]") == 1


def test_different_sources_cannot_claim_physical_simultaneity() -> None:
    result = choose_two_panel_layout(
        _target("a", (100, 200, 300, 800), asset="sha256:" + "a" * 64),
        _target("b", (700, 200, 900, 800), asset="sha256:" + "b" * 64),
        relation_mode="simultaneous_relation",
        physical_scale_comparison=True,
        allow_conceptual_different_source=True,
        allowed_modes=("top_bottom", "side_by_side"),
    )

    assert result is None


def test_three_person_group_never_creates_three_panels() -> None:
    compilation = compile_presentation(
        targets=[
            _target("first", (10, 200, 210, 800)),
            _target("second", (400, 200, 600, 800)),
            _target("third", (790, 200, 990, 800)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="simultaneous_relation",
        policy=_policy(),
    )

    assert compilation.mode == "solid_matte_fit"
    assert compilation.panel_layout is None


def test_intentional_freeze_requires_exact_event_and_policy_duration() -> None:
    freeze = compile_intentional_freeze(
        _event_lock(),
        cue_id="locked-cue-00012",
        duration_ms=1_000,
        policy=_policy(),
    )

    assert freeze.source_pts == 120
    assert freeze.motivation == "brief_authorized_phrase_ending"


def test_autonomous_catalog_adds_only_policy_gated_presentations() -> None:
    catalog = autonomous_production_capability_catalog()
    ids = {item.capability_id for item in catalog.capabilities}

    assert {
        "two_panel_layout",
        "solid_matte_fit",
        "intentional_freeze",
    } <= ids
    assert catalog.prohibited_automatic_delivery == [
        "blurred_background"
    ]


def test_multi_target_grounding_is_one_call_and_builds_one_sam_session(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "exact.png"
    Image.new("RGB", (640, 360), color=(20, 30, 40)).save(frame)
    frame_hash = hashlib.sha256(frame.read_bytes()).hexdigest()
    lock = _event_lock(frame_hash=frame_hash)
    target_requests = (
        GroundingTargetRequest(
            target_id="device-a",
            target_description="left device",
        ),
        GroundingTargetRequest(
            target_id="device-b",
            target_description="right device",
        ),
    )
    output = json.dumps(
        {
            "source_asset_id": lock.source_asset_id,
            "event_lock_id": lock.event_id,
            "source_frame_id": lock.source_frame_id,
            "source_frame_hash": frame_hash,
            "source_width": 640,
            "source_height": 360,
            "targets": [
                {
                    "target_id": "device-a",
                    "visible": True,
                    "candidates": [
                        {
                            "box_2d_yxyx": [100, 50, 900, 400],
                            "confidence": 0.9,
                            "disambiguation_reason": "left instance",
                        }
                    ],
                },
                {
                    "target_id": "device-b",
                    "visible": True,
                    "candidates": [
                        {
                            "box_2d_yxyx": [100, 600, 900, 950],
                            "confidence": 0.9,
                            "disambiguation_reason": "right instance",
                        }
                    ],
                },
            ],
        }
    )
    requests: list[dict[str, Any]] = []

    class Interaction:
        id = "ground-1"
        output_text = output

        def model_dump(
            self,
            *,
            mode: str,
            exclude_none: bool,
        ) -> dict[str, object]:
            return {
                "id": self.id,
                "model": MODEL_ID,
                "output_text": self.output_text,
                "usage": {},
            }

    def create(**request: Any) -> Interaction:
        requests.append(request)
        return Interaction()

    client = object.__new__(GeminiLabClient)
    client.model_id = MODEL_ID
    client.client = SimpleNamespace(
        interactions=SimpleNamespace(create=create)
    )

    group = client.ground_multi_target_exact_frame(
        event_lock=lock,
        frame_path=frame,
        targets=target_requests,
        output_dir=tmp_path / "grounding",
    )
    seeds = shared_sam_seeds_from_grounding(
        group,
        target_requests=target_requests,
        seed_time_ms=4_000,
        seed_frame_pts=120,
    )

    assert len(requests) == 1
    assert requests[0]["input"][1]["media_resolution"] == "high"
    assert len(seeds) == 2
    assert {seed.target_id for seed in seeds} == {
        "device-a",
        "device-b",
    }
