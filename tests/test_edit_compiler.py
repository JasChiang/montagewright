from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
    PresentationPolicy,
)
from jascue_video_lab.billing import BudgetLedger
from jascue_video_lab.editing_capabilities import (
    AttentionIntent,
    CanvasSpec,
    EditProjectIR,
    EditorialObjectiveProfile,
    RuntimeCapabilityRegistry,
    SemanticBeat,
    VisibilityContract,
    VisibilityTarget,
    autonomous_capability_registry_v2,
)
from jascue_video_lab.gemini import (
    FunctionToolDeclaration,
    GeminiLabClient,
    MODEL_ID,
)
from jascue_video_lab.feature_cut import _late_cue_shift_disposition
from jascue_video_lab.presentation import (
    PresentationTarget,
    SourceCameraMotionEvidence,
    compile_presentation,
    generate_presentation_options,
)
from jascue_video_lab.sequence_optimizer import (
    ConstraintResult,
    ExecutableOptionV2,
    OptionMetrics,
    select_executable_option,
)


def _policy(
    *,
    allow_solid_matte_fit: bool = True,
) -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=60_000,
            min_ms=55_000,
            max_ms=70_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=1.25,
            max_paid_interactions=25,
        ),
        presentation=PresentationPolicy(
            allow_solid_matte_fit=allow_solid_matte_fit,
        ),
    )


def _target(
    target_id: str,
    box: tuple[int, int, int, int],
) -> PresentationTarget:
    return PresentationTarget(
        target_id=target_id,
        source_asset_id="sha256:" + "a" * 64,
        source_pts=30,
        box_2d=box,
    )


def _static_source_motion() -> SourceCameraMotionEvidence:
    return SourceCameraMotionEvidence(
        source_asset_id="sha256:" + "a" * 64,
        window_start_ms=0,
        window_end_ms=2_000,
        sample_times_ms=(0, 2_000),
        sample_frame_pts=(0, 60),
        sample_frame_hashes=("b" * 64, "c" * 64),
        subject_exclusion_mode="none",
        mean_excluded_area_fraction=0.0,
        pairs=(),
        classification="static",
        reliable=True,
        confidence=0.95,
        normalized_translation_x_per_second=0.0,
        normalized_translation_y_per_second=0.0,
        scale_rate_per_second=0.0,
        rotation_degrees_per_second=0.0,
        normalized_travel=0.0,
        reversal_count=0,
        reason_codes=("source_camera_static",),
        cache_key_sha256="d" * 64,
    )


def test_semantic_edit_ir_represents_mixed_content_without_render_geometry() -> None:
    project = EditProjectIR(
        project_id="mixed-event-demo",
        source_catalog_ref="sha256:" + "b" * 64,
        policy_ref="sha256:" + "c" * 64,
        objective_profile=EditorialObjectiveProfile(
            music_dependency=0.8,
            instructional_order_strictness=0.6,
            chronology_strictness=0.4,
            text_legibility_importance=0.9,
        ),
        outputs=(
            CanvasSpec(
                canvas_id="vertical",
                width=1080,
                height=1920,
                background_policy="solid",
            ),
        ),
        beats=(
            SemanticBeat(
                beat_id="ui-result",
                priority="hard",
                narrative_function="demonstrate_result",
                evidence_refs=("event-lock:result",),
                candidate_refs=("candidate:1", "candidate:2"),
                visibility_contract=VisibilityContract(
                    targets=(
                        VisibilityTarget(
                            target_id="ui",
                            minimum_readability=0.9,
                            atomic=True,
                        ),
                    ),
                    temporal_visibility="one",
                ),
                attention_intent=AttentionIntent(
                    ordered_target_ids=("ui",),
                    goal="emphasize",
                    motion_motivation="emphasis",
                ),
                minimum_duration_ms=1_000,
                preferred_duration_ms=2_000,
                maximum_duration_ms=4_000,
                acceptable_capability_ids=(
                    "static_full_bleed_crop",
                    "tracked_full_bleed_crop",
                ),
            ),
        ),
    )

    payload = project.model_dump(mode="json")
    assert "crop" not in payload["beats"][0]
    assert "source_pts" not in payload["beats"][0]
    assert project.definition_sha256() == project.definition_sha256()


def test_capability_registry_is_runtime_typed_not_only_prompt_text() -> None:
    manifest = autonomous_capability_registry_v2()
    runtime = RuntimeCapabilityRegistry(manifest)

    assert "phase_virtual_camera" in manifest.by_id()
    assert manifest.by_id()["phase_virtual_camera"].paid_model_calls == 0
    with pytest.raises(ValueError, match="no runtime executor"):
        runtime.require_executor("phase_virtual_camera")


def test_option_generator_exposes_static_and_motivated_virtual_camera() -> None:
    options = generate_presentation_options(
        targets=[
            _target("phone-a", (100, 250, 300, 750)),
            _target("phone-b", (650, 250, 850, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="sequential_focus",
        policy=_policy(),
        required_x_values=(0.2, 0.75),
        movement_motivated=True,
        source_camera_motion_evidence=_static_source_motion(),
    )
    modes = {option.mode for option in options.options}

    assert "phase_virtual_camera" in modes
    assert "tracked_full_bleed_crop" in modes
    compilation = compile_presentation(
        targets=[
            _target("phone-a", (100, 250, 300, 750)),
            _target("phone-b", (650, 250, 850, 750)),
        ],
        source_width=1920,
        source_height=1080,
        relation_mode="sequential_focus",
        policy=_policy(),
        required_x_values=(0.2, 0.75),
        movement_motivated=True,
        preferred_capability_ids=("phase_virtual_camera",),
        source_camera_motion_evidence=_static_source_motion(),
    )
    assert compilation.mode == "phase_virtual_camera"
    assert compilation.camera_motion is not None
    assert compilation.camera_motion.mode == "minimal_monotonic_move"
    assert compilation.selection is not None
    assert compilation.selection.rejected_options == {}
    assert compilation.selection.semantic_negotiation_recommended is False


def test_semantic_capability_boundary_is_hard_not_a_saved_annotation() -> None:
    compilation = compile_presentation(
        targets=[_target("phone", (350, 200, 650, 800))],
        source_width=1920,
        source_height=1080,
        relation_mode="single_subject",
        policy=_policy(),
        preferred_capability_ids=("solid_matte_fit",),
        acceptable_capability_ids=("static_full_bleed_crop",),
        source_camera_motion_evidence=_static_source_motion(),
    )

    assert compilation.mode == "static_full_bleed_crop"
    assert compilation.selection is not None
    rejected_capabilities = {
        compilation.selection.generated_capabilities[option_id]: reasons
        for option_id, reasons in compilation.selection.rejected_options.items()
    }
    assert rejected_capabilities["tracked_full_bleed_crop"] == (
        "capability_outside_semantic_boundary",
    )
    assert rejected_capabilities["solid_matte_fit"] == (
        "capability_outside_semantic_boundary",
    )


def test_constructed_presentation_cannot_outscore_safe_single_canvas() -> None:
    def option(
        option_id: str,
        capability_id: str,
        *,
        semantic_fit: float,
        intrusion_rank: int,
    ) -> ExecutableOptionV2:
        return ExecutableOptionV2(
            option_id=option_id,
            capability_id=capability_id,
            payload_sha256="e" * 64,
            constraints=(
                ConstraintResult(
                    constraint_id="safe",
                    level="hard",
                    status="pass",
                    reason_code="safe",
                ),
            ),
            metrics=OptionMetrics(
                semantic_fit=semantic_fit,
                readability=semantic_fit,
                technical_quality=semantic_fit,
                intrusion_rank=intrusion_rank,
            ),
        )

    selection = select_executable_option(
        (
            option(
                "natural",
                "tracked_full_bleed_crop",
                semantic_fit=0.7,
                intrusion_rank=1,
            ),
            option(
                "constructed",
                "two_panel_layout",
                semantic_fit=1.0,
                intrusion_rank=5,
            ),
        ),
        preferred_capability_ids=("two_panel_layout",),
    )

    assert selection.selected_option_id == "natural"
    assert "single_canvas_minimality_applied" in selection.decision_codes


def test_unknown_hard_constraint_fails_closed_before_scoring() -> None:
    option = ExecutableOptionV2(
        option_id="unknown-option",
        capability_id="static_full_bleed_crop",
        payload_sha256="d" * 64,
        constraints=(
            ConstraintResult(
                constraint_id="identity",
                level="hard",
                status="unknown",
                reason_code="identity_not_measured",
            ),
        ),
        metrics=OptionMetrics(
            semantic_fit=1.0,
            readability=1.0,
            technical_quality=1.0,
        ),
    )

    selection = select_executable_option([option])

    assert selection.selected_option_id is None
    assert selection.rejected_options == {
        "unknown-option": ("identity_not_measured",)
    }


def test_late_cue_shift_cannot_force_policy_forbidden_matte_fit() -> None:
    assert (
        _late_cue_shift_disposition(
            _policy(allow_solid_matte_fit=False)
        )
        == "recompile_required"
    )
    assert (
        _late_cue_shift_disposition(
            _policy(allow_solid_matte_fit=True)
        )
        == "recompile_required"
    )


class _FunctionInteraction:
    def __init__(
        self,
        interaction_id: str,
        steps: list[dict[str, Any]],
    ) -> None:
        self.id = interaction_id
        self.steps = steps
        self.output_text = ""

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": MODEL_ID,
            "steps": self.steps,
            "usage": {
                "total_input_tokens": 200,
                "total_cached_tokens": 0,
                "total_output_tokens": 20,
                "total_thought_tokens": 10,
            },
        }


class _FunctionInteractions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses = [
            _FunctionInteraction(
                "negotiation-1",
                [
                    {
                        "type": "function_call",
                        "id": "call-inspect",
                        "name": "enumerate_presentation_options",
                        "arguments": {"beat_id": "beat-1"},
                    }
                ],
            ),
            _FunctionInteraction(
                "negotiation-2",
                [
                    {
                        "type": "function_call",
                        "id": "call-propose",
                        "name": "propose_edit_decision",
                        "arguments": {
                            "beat_id": "beat-1",
                            "selected_option_id": "option-camera",
                            "fallback_option_ids": ["option-static"],
                            "semantic_reason": "sequential_attention",
                            "unresolved_concern_codes": [],
                        },
                    }
                ],
            ),
        ]

    def create(self, **request: Any) -> _FunctionInteraction:
        self.requests.append(request)
        return self.responses.pop(0)


def test_function_negotiation_is_manual_bounded_and_budgeted(tmp_path) -> None:
    interactions = _FunctionInteractions()
    ledger = BudgetLedger(max_cost_usd=1.25, max_interactions=25)
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = MODEL_ID
    client.budget_ledger = ledger
    handler_calls: list[dict[str, Any]] = []

    result = client.negotiate_edit_decision(
        beat_id="beat-1",
        option_ids=("option-static", "option-camera"),
        prompt="Choose the semantically appropriate executable option.",
        tool_declarations=(
            FunctionToolDeclaration(
                name="enumerate_presentation_options",
                description="Return already compiled local presentation facts.",
                parameters={
                    "type": "object",
                    "properties": {"beat_id": {"type": "string"}},
                    "required": ["beat_id"],
                },
            ),
        ),
        tool_handlers={
            "enumerate_presentation_options": lambda arguments: (
                handler_calls.append(dict(arguments))
                or {
                    "options": [
                        {"option_id": "option-static", "feasible": True},
                        {"option_id": "option-camera", "feasible": True},
                    ]
                }
            )
        },
        policy=_policy(),
        run_dir=tmp_path,
    )

    assert result.rounds_used == 2
    assert result.decision.selected_option_id == "option-camera"
    assert result.automatic_function_calling is False
    assert handler_calls == [{"beat_id": "beat-1"}]
    assert len(interactions.requests) == 2
    assert "previous_interaction_id" not in interactions.requests[0]
    assert interactions.requests[1]["previous_interaction_id"] == "negotiation-1"
    assert interactions.requests[0]["generation_config"]["tool_choice"] == {
        "allowed_tools": {
            "mode": "any",
            "tools": [
                "enumerate_presentation_options",
                "propose_edit_decision",
            ],
        }
    }
    assert ledger.committed_interactions == 2


def test_function_negotiation_canonicalizes_explicit_immutable_alias(
    tmp_path,
) -> None:
    """A renderer payload ID is an alias, never a fresh creative option."""

    interactions = _FunctionInteractions()
    interactions.responses = [
        _FunctionInteraction(
            "negotiation-alias",
            [
                {
                    "type": "function_call",
                    "id": "call-propose-alias",
                    "name": "propose_edit_decision",
                    "arguments": {
                        "beat_id": "beat-1",
                        "selected_option_id": "static_full_bleed_crop:abc123",
                        "fallback_option_ids": [],
                        "semantic_reason": "avoid_unmotivated_motion",
                        "unresolved_concern_codes": [],
                    },
                }
            ],
        )
    ]
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = MODEL_ID
    client.budget_ledger = BudgetLedger(max_cost_usd=1.25, max_interactions=25)

    result = client.negotiate_edit_decision(
        beat_id="beat-1",
        option_ids=("replan-01-rank-01-static_full_bleed_crop",),
        option_id_aliases={
            "static_full_bleed_crop:abc123": (
                "replan-01-rank-01-static_full_bleed_crop"
            )
        },
        prompt="Choose one immutable option.",
        tool_declarations=(),
        tool_handlers={},
        policy=_policy(),
        run_dir=tmp_path,
    )

    assert result.decision.selected_option_id == (
        "replan-01-rank-01-static_full_bleed_crop"
    )
    assert (
        tmp_path / "semantic_negotiation.round-1.option-id-normalization.json"
    ).is_file()


class _FailingInteractions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_request: Any) -> None:
        self.calls += 1
        raise RuntimeError("503 upstream unavailable")


def test_function_negotiation_503_has_no_hidden_retry_and_keeps_reserve(
    tmp_path,
) -> None:
    interactions = _FailingInteractions()
    ledger = BudgetLedger(max_cost_usd=1.25, max_interactions=25)
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = MODEL_ID
    client.budget_ledger = ledger

    with pytest.raises(RuntimeError, match="503"):
        client.negotiate_edit_decision(
            beat_id="beat-1",
            option_ids=("option-static", "option-camera"),
            prompt="Choose one option.",
            tool_declarations=(),
            tool_handlers={},
            policy=_policy(),
            run_dir=tmp_path,
        )

    assert interactions.calls == 1
    assert ledger.committed_interactions == 1
