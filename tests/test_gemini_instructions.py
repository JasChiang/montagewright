from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import jascue_video_lab.gemini as gemini_module
from jascue_video_lab.autonomous_policy import (
    AutonomousEditPolicy,
    BudgetPolicy,
    DurationPolicy,
)
from jascue_video_lab.billing import (
    BudgetExceeded,
    BudgetLedger,
    PaidDispatchAlreadyRecorded,
    summarize_usage_and_list_price,
)

from jascue_video_lab.gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    GroundingIdentityReference,
    GroupedEditDecisionProposal,
    GroupedSemanticReplanDecision,
    MODEL_ID,
    SEMANTIC_IDENTITY_GENERATION_CONFIG,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    GeminiLabClient,
    canonical_interactions_mime_type,
    canonicalize_feature_edit_plan_output,
    canonicalize_selected_vertical_framing_output,
    normalize_full_clip_card_output_text,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.models import (
    ContentMap,
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    MediaInfo,
    ModelProvenance,
    Rational,
    RushClip,
    RushFrame,
    RushesCatalog,
    VideoStreamInfo,
)


class _StopRequest(RuntimeError):
    pass


class _FakeInteraction:
    def __init__(self, *, interaction_id: str, output_text: str) -> None:
        self.id = interaction_id
        self.output_text = output_text

    def model_dump(
        self,
        *,
        mode: str,
        exclude_none: bool,
    ) -> dict[str, object]:
        assert mode == "json"
        assert exclude_none is False
        return {
            "id": self.id,
            "model": MODEL_ID,
            "output_text": self.output_text,
            "usage": {
                "total_input_tokens": 1,
                "total_output_tokens": 1,
            },
        }


def test_portrait_prompts_require_relation_timing_and_presentation_alternatives() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts"
    selected = (
        prompt_root / "selected_vertical_framing_zh-TW.txt"
    ).read_text(encoding="utf-8")
    coarse = (
        prompt_root / "feature_cut_selects_zh-TW.txt"
    ).read_text(encoding="utf-8")

    assert "relation_temporal_mode" in selected
    assert "sequentially_reconstructable" in selected
    assert "phase_mixed" in selected
    assert "uncertain" in selected
    assert "sequential_reconstruction" in selected
    assert "relation_carrier" in selected
    assert "state_evidence" in selected
    assert "context_reference" in selected
    assert "presentation_options" in selected
    assert "sequential_virtual_camera" in selected
    assert "不得只說「太寬」就跳到固定中央裁切" in selected
    assert "相對大小本身不會自動要求全程同框" in coarse
    assert "不得固定左→右" in coarse
    assert "剪輯呈現的決策權" in coarse
    assert "互不等價" in coarse
    assert "required region 不只可以是人物或主要物件" in coarse


def test_grouped_semantic_replan_reuse_authority_is_typed() -> None:
    with pytest.raises(ValueError, match="requires a justification"):
        GroupedSemanticReplanDecision(
            beat_id="fold",
            selected_option_id="fold--candidate-a",
            fallback_option_ids=(),
            semantic_reason="preserve_readability",
            unresolved_concern_codes=(),
            source_reuse_mode="distinct_interval",
            source_reuse_justification=None,
            reuse_of_beat_ids=("opening",),
        )
    with pytest.raises(ValueError, match="none source reuse"):
        GroupedSemanticReplanDecision(
            beat_id="fold",
            selected_option_id="fold--candidate-a",
            fallback_option_ids=(),
            semantic_reason="preserve_readability",
            unresolved_concern_codes=(),
            source_reuse_mode="none",
            source_reuse_justification="not allowed",
            reuse_of_beat_ids=(),
        )
    accepted = GroupedSemanticReplanDecision(
        beat_id="fold",
        selected_option_id="fold--candidate-a",
        fallback_option_ids=(),
        semantic_reason="preserve_readability",
        unresolved_concern_codes=(),
        source_reuse_mode="distinct_interval",
        source_reuse_justification="The later interval shows a different state.",
        reuse_of_beat_ids=("opening",),
    )
    assert accepted.reuse_of_beat_ids == ("opening",)


def _grouped_semantic_policy() -> AutonomousEditPolicy:
    return AutonomousEditPolicy(
        execution_profile="autonomous_strict",
        content_mode="music_led_feature",
        requested_aspects=("9:16",),
        duration=DurationPolicy(
            target_ms=30_000,
            min_ms=30_000,
            max_ms=30_000,
        ),
        budget=BudgetPolicy(
            max_gemini_cost_usd=5.0,
            max_paid_interactions=10,
            max_semantic_replans=1,
        ),
    )


def _grouped_interaction(
    interaction_id: str,
    *,
    function_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": interaction_id,
        "model": MODEL_ID,
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 10,
            "total_thought_tokens": 0,
        },
        "steps": [
            {
                "type": "function_call",
                "id": f"call-{interaction_id}",
                "name": function_name,
                "arguments": arguments,
            }
        ],
    }


def _complete_grouped_no_reuse_arguments() -> dict[str, Any]:
    return {
        "decisions": [
            {
                "beat_id": "fold",
                "selected_option_id": "fold--option-a",
                "fallback_option_ids": [],
                "semantic_reason": "preserve_readability",
                "unresolved_concern_codes": [],
                "source_reuse_mode": "none",
                "source_reuse_justification": None,
                "reuse_of_beat_ids": [],
            }
        ]
    }


def _retained_grouped_reuse_authority() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "fold": {
            "fold--option-a": {
                "mode": "editorial_reprise",
                "justification": "The immutable option already carries this reprise.",
            }
        }
    }


def _legacy_execution_binding_arguments(
    option_id: str,
) -> dict[str, Any]:
    return {
        "decisions": [
            {
                "beat_id": "fold",
                "selected_execution_option_id": option_id,
            }
        ]
    }


def _grouped_client_with_responses(
    responses: list[dict[str, Any]] | None = None,
) -> tuple[GeminiLabClient, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    def create(**request: Any) -> dict[str, Any]:
        requests.append(request)
        if not responses:
            raise RuntimeError("503 service unavailable")
        return responses.pop(0)

    client = object.__new__(GeminiLabClient)
    client.model_id = MODEL_ID
    client.budget_ledger = BudgetLedger(
        max_cost_usd=5.0,
        max_interactions=10,
    )
    client.client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    return client, requests


def test_legacy_execution_binding_repair_is_text_only_and_reuses_raw(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy-execution-binding-repair"
    option_a = "fold--fold-2--execution-" + "a" * 64
    option_b = "fold--fold-2--execution-" + "b" * 64
    client, requests = _grouped_client_with_responses(
        [
            _grouped_interaction(
                "legacy-binding",
                function_name="bind_legacy_execution_options",
                arguments=_legacy_execution_binding_arguments(option_b),
            )
        ]
    )
    kwargs = {
        "option_ids_by_beat": {"fold": (option_a, option_b)},
        "prompt": (
            "TEXT-ONLY-CONTEXT: source timestamps and local preflight facts; "
            "do not request visual inputs."
        ),
        "policy": _grouped_semantic_policy(),
        "run_dir": run_dir,
    }

    result = client.repair_legacy_execution_bindings(**kwargs)
    resumed = client.repair_legacy_execution_bindings(**kwargs)

    assert result.interaction_ids == ("legacy-binding",)
    assert resumed.decision.decisions[0].selected_execution_option_id == option_b
    assert len(requests) == 1
    request = requests[0]
    assert [item["type"] for item in request["input"]] == ["text"]
    assert request["tools"][0]["name"] == "bind_legacy_execution_options"
    assert request["input"][0]["text"].count(option_a) == 1
    assert request["input"][0]["text"].count(option_b) == 1
    assert "image" not in request["input"][0]
    assert "video" not in request["input"][0]
    accounting = client.budget_ledger.report()["stages"][
        "legacy_execution_binding_repair"
    ]
    assert accounting["reserved_interactions"] == 1
    assert accounting["actual_cost_usd"] > 0


def test_legacy_execution_binding_repair_rejects_unknown_without_retry(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "unknown-binding"
    option = "fold--fold-2--execution-" + "a" * 64
    client, requests = _grouped_client_with_responses(
        [
            _grouped_interaction(
                "unknown-binding",
                function_name="bind_legacy_execution_options",
                arguments=_legacy_execution_binding_arguments(
                    "fold--fold-2--execution-" + "f" * 64
                ),
            )
        ]
    )
    kwargs = {
        "option_ids_by_beat": {"fold": (option,)},
        "prompt": "text only",
        "policy": _grouped_semantic_policy(),
        "run_dir": run_dir,
    }

    with pytest.raises(ValueError, match="refusing a second provider call"):
        client.repair_legacy_execution_bindings(**kwargs)
    with pytest.raises(ValueError, match="refusing a second provider call"):
        client.repair_legacy_execution_bindings(**kwargs)

    assert len(requests) == 1
    invalid = json.loads(
        (run_dir / "legacy_execution_binding_repair.invalid.json").read_text(
            encoding="utf-8"
        )
    )
    assert invalid["provider_follow_up_dispatched"] is False


def test_legacy_execution_binding_repair_has_no_fixed_beat_cap(
    tmp_path: Path,
) -> None:
    options = {
        f"beat-{index}": (f"beat-{index}--candidate--execution-{index:064x}",)
        for index in range(9)
    }
    arguments = {
        "decisions": [
            {
                "beat_id": beat_id,
                "selected_execution_option_id": option_ids[0],
            }
            for beat_id, option_ids in options.items()
        ]
    }
    client, requests = _grouped_client_with_responses(
        [
            _grouped_interaction(
                "nine-legacy-bindings",
                function_name="bind_legacy_execution_options",
                arguments=arguments,
            )
        ]
    )

    result = client.repair_legacy_execution_bindings(
        option_ids_by_beat=options,
        prompt="text-only exact execution bindings",
        policy=_grouped_semantic_policy(),
        run_dir=tmp_path / "many-bindings",
    )

    assert len(result.decision.decisions) == 9
    assert len(requests) == 1


def test_legacy_execution_binding_repair_is_bounded_by_budget_and_dispatch_journal(
    tmp_path: Path,
) -> None:
    option = "fold--fold-2--execution-" + "a" * 64
    kwargs = {
        "option_ids_by_beat": {"fold": (option,)},
        "prompt": "text only",
        "policy": _grouped_semantic_policy(),
    }

    budget_client, budget_requests = _grouped_client_with_responses()
    budget_client.budget_ledger = BudgetLedger(
        max_cost_usd=0.00000001,
        max_interactions=10,
    )
    with pytest.raises(BudgetExceeded, match="blocked before request"):
        budget_client.repair_legacy_execution_bindings(
            **kwargs,
            run_dir=tmp_path / "budget-blocked",
        )
    assert budget_requests == []

    unavailable_client, unavailable_requests = _grouped_client_with_responses()
    unavailable_kwargs = {
        **kwargs,
        "run_dir": tmp_path / "service-unavailable",
    }
    with pytest.raises(RuntimeError, match="503"):
        unavailable_client.repair_legacy_execution_bindings(**unavailable_kwargs)
    with pytest.raises(PaidDispatchAlreadyRecorded):
        unavailable_client.repair_legacy_execution_bindings(**unavailable_kwargs)
    assert len(unavailable_requests) == 1


def test_grouped_semantic_tool_schema_requires_explicit_reuse_fields() -> None:
    schema = gemini_response_schema(GroupedEditDecisionProposal)
    required = set(schema["$defs"]["GroupedSemanticReplanDecision"]["required"])

    assert {
        "fallback_option_ids",
        "unresolved_concern_codes",
        "source_reuse_mode",
        "source_reuse_justification",
        "reuse_of_beat_ids",
    }.issubset(required)
    assert GroupedSemanticReplanDecision(
        beat_id="fold",
        selected_option_id="fold--option-a",
        fallback_option_ids=(),
        semantic_reason="preserve_readability",
        unresolved_concern_codes=(),
        source_reuse_mode="none",
        source_reuse_justification=None,
        reuse_of_beat_ids=(),
    ).source_reuse_mode == "none"


def test_grouped_semantic_reuse_ids_must_exactly_match_supplied_authority() -> None:
    policy = _grouped_semantic_policy()
    options = {"fold": ("fold--option-a",)}
    allowed = {"fold": {"fold--option-a": ("opening",)}}
    valid = _complete_grouped_no_reuse_arguments()
    valid["decisions"][0].update(
        {
            "source_reuse_mode": "distinct_interval",
            "source_reuse_justification": "A later independent interval.",
            "reuse_of_beat_ids": ["opening"],
        }
    )
    assert gemini_module._validate_grouped_decision_arguments(
        valid,
        option_ids_by_beat=options,
        allowed_reuse_of_beat_ids_by_option=allowed,
        policy=policy,
    ).decisions[0].reuse_of_beat_ids == ("opening",)

    for invalid_ids in ([], ["invented"], ["fold"]):
        invalid = _complete_grouped_no_reuse_arguments()
        invalid["decisions"][0].update(
            {
                "source_reuse_mode": "distinct_interval",
                "source_reuse_justification": "A later independent interval.",
                "reuse_of_beat_ids": invalid_ids,
            }
        )
        with pytest.raises(ValueError):
            gemini_module._validate_grouped_decision_arguments(
                invalid,
                option_ids_by_beat=options,
                allowed_reuse_of_beat_ids_by_option=allowed,
                policy=policy,
            )


def test_grouped_semantic_resume_repairs_invalid_raw_once_without_full_prompt(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "grouped"
    run_dir.mkdir()
    original = _complete_grouped_no_reuse_arguments()
    for key in (
        "fallback_option_ids",
        "unresolved_concern_codes",
        "reuse_of_beat_ids",
    ):
        original["decisions"][0].pop(key)
    (run_dir / "grouped_semantic_negotiation.raw_interaction.json").write_text(
        json.dumps(
            _grouped_interaction(
                "original",
                function_name="propose_grouped_edit_decisions",
                arguments=original,
            )
        ),
        encoding="utf-8",
    )
    client, requests = _grouped_client_with_responses(
        [
            _grouped_interaction(
                "repair",
                function_name="repair_grouped_edit_decisions",
                arguments=_complete_grouped_no_reuse_arguments(),
            )
        ]
    )

    result = client.negotiate_grouped_edit_decisions(
        option_ids_by_beat={"fold": ("fold--option-a",)},
        allowed_reuse_of_beat_ids_by_option={
            "fold": {"fold--option-a": ()}
        },
        prompt="FULL-CONTEXT-SENTINEL-MUST-NOT-BE-RESENT",
        policy=_grouped_semantic_policy(),
        run_dir=run_dir,
        recovery_call=True,
        retained_reuse_authority_by_option=_retained_grouped_reuse_authority(),
    )

    assert result.interaction_ids == ("original", "repair")
    assert result.schema_repair_interaction_ids == ("repair",)
    assert len(requests) == 1
    repair_request = requests[0]
    assert repair_request["tools"][0]["name"] == "repair_grouped_edit_decisions"
    assert [item["type"] for item in repair_request["input"]] == ["text"]
    repair_text = repair_request["input"][0]["text"]
    assert "FULL-CONTEXT-SENTINEL-MUST-NOT-BE-RESENT" not in repair_text
    assert "whole_resolved_timeline" not in repair_text
    repair_context = json.loads(repair_text.split("\n\n", 1)[1])
    action = repair_context["per_option_authority_constraints"]["fold"][
        "fold--option-a"
    ]
    assert action["required_authority_action"] == (
        "retain_existing_no_new_authority"
    )
    assert action["decision_reuse_fields_must_be"] == {
        "source_reuse_mode": "none",
        "source_reuse_justification": None,
        "reuse_of_beat_ids": [],
    }
    audit = json.loads(
        (
            run_dir / "grouped_semantic_negotiation.schema_repair.audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["repair_limit"] == 1
    assert audit["original_request_redispatched"] is False


def test_grouped_semantic_resume_canonicalizes_saved_retained_reuse_restatement(
    tmp_path: Path,
) -> None:
    """A v31-shaped saved repair is reused without a second provider call."""

    run_dir = tmp_path / "grouped"
    run_dir.mkdir()
    retained = _retained_grouped_reuse_authority()
    original = _complete_grouped_no_reuse_arguments()
    for key in (
        "fallback_option_ids",
        "unresolved_concern_codes",
        "reuse_of_beat_ids",
    ):
        original["decisions"][0].pop(key)
    raw_path = run_dir / "grouped_semantic_negotiation.raw_interaction.json"
    raw_path.write_text(
        json.dumps(
            _grouped_interaction(
                "original",
                function_name="propose_grouped_edit_decisions",
                arguments=original,
            )
        ),
        encoding="utf-8",
    )
    repair = _complete_grouped_no_reuse_arguments()
    repair["decisions"][0].update(
        {
            "source_reuse_mode": retained["fold"]["fold--option-a"]["mode"],
            "source_reuse_justification": retained["fold"]["fold--option-a"][
                "justification"
            ],
            "reuse_of_beat_ids": [],
        }
    )
    repair_raw_path = (
        run_dir / "grouped_semantic_negotiation.schema_repair.raw_interaction.json"
    )
    repair_raw_path.write_text(
        json.dumps(
            _grouped_interaction(
                "repair",
                function_name="repair_grouped_edit_decisions",
                arguments=repair,
            )
        ),
        encoding="utf-8",
    )
    (run_dir / "grouped_semantic_negotiation.schema_repair.audit.json").write_text(
        json.dumps(
            {
                "original_interaction_id": "original",
                "original_raw_interaction_sha256": hashlib.sha256(
                    raw_path.read_bytes()
                ).hexdigest(),
                "repair_limit": 1,
            }
        ),
        encoding="utf-8",
    )
    client, requests = _grouped_client_with_responses()
    kwargs = {
        "option_ids_by_beat": {"fold": ("fold--option-a",)},
        "allowed_reuse_of_beat_ids_by_option": {
            "fold": {"fold--option-a": ()}
        },
        "retained_reuse_authority_by_option": retained,
        "prompt": "must never be resent",
        "policy": _grouped_semantic_policy(),
        "run_dir": run_dir,
        "recovery_call": True,
    }

    result = client.negotiate_grouped_edit_decisions(**kwargs)
    resumed = client.negotiate_grouped_edit_decisions(**kwargs)

    assert requests == []
    assert result.interaction_ids == ("original", "repair")
    assert resumed.decision.decisions[0].source_reuse_mode == "none"
    assert resumed.decision.decisions[0].source_reuse_justification is None
    assert resumed.decision.decisions[0].reuse_of_beat_ids == ()
    audit = json.loads(
        (
            run_dir / "grouped_semantic_negotiation.schema_repair.audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["original_raw_interaction_sha256"] == hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    assert audit["repair_canonicalized_redundant_retained_authority"] == [
        {
            "beat_id": "fold",
            "selected_option_id": "fold--option-a",
            "retained_reuse_mode": "editorial_reprise",
            "normalization": (
                "redundant_retained_authority_to_no_new_authority"
            ),
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "retained"),
    (
        ("source_reuse_mode", "distinct_interval", True),
        ("source_reuse_justification", "near-match", True),
        ("reuse_of_beat_ids", ["opening"], True),
        ("source_reuse_mode", "editorial_reprise", False),
    ),
)
def test_grouped_semantic_rejects_near_match_retained_reuse_restatement(
    tmp_path: Path,
    field: str,
    value: Any,
    retained: bool,
) -> None:
    run_dir = tmp_path / field
    run_dir.mkdir()
    original = {"decisions": [{"beat_id": "fold"}]}
    raw_path = run_dir / "grouped_semantic_negotiation.raw_interaction.json"
    raw_path.write_text(
        json.dumps(
            _grouped_interaction(
                "original",
                function_name="propose_grouped_edit_decisions",
                arguments=original,
            )
        ),
        encoding="utf-8",
    )
    repair = _complete_grouped_no_reuse_arguments()
    retained_authority = _retained_grouped_reuse_authority()
    repair["decisions"][0].update(
        {
            "source_reuse_mode": "editorial_reprise",
            "source_reuse_justification": retained_authority["fold"][
                "fold--option-a"
            ]["justification"],
            "reuse_of_beat_ids": [],
            field: value,
        }
    )
    repair_raw_path = (
        run_dir / "grouped_semantic_negotiation.schema_repair.raw_interaction.json"
    )
    repair_raw_path.write_text(
        json.dumps(
            _grouped_interaction(
                "repair",
                function_name="repair_grouped_edit_decisions",
                arguments=repair,
            )
        ),
        encoding="utf-8",
    )
    (run_dir / "grouped_semantic_negotiation.schema_repair.audit.json").write_text(
        json.dumps(
            {
                "original_interaction_id": "original",
                "original_raw_interaction_sha256": hashlib.sha256(
                    raw_path.read_bytes()
                ).hexdigest(),
                "repair_limit": 1,
            }
        ),
        encoding="utf-8",
    )
    client, requests = _grouped_client_with_responses()

    with pytest.raises(ValueError, match="refusing another repair"):
        client.negotiate_grouped_edit_decisions(
            option_ids_by_beat={"fold": ("fold--option-a",)},
            allowed_reuse_of_beat_ids_by_option={
                "fold": {"fold--option-a": ()}
            },
            retained_reuse_authority_by_option=(
                retained_authority if retained else None
            ),
            prompt="must never be resent",
            policy=_grouped_semantic_policy(),
            run_dir=run_dir,
            recovery_call=True,
        )

    assert requests == []


def test_grouped_semantic_schema_repair_is_single_shot_on_invalid_or_503(
    tmp_path: Path,
) -> None:
    def write_invalid_original(run_dir: Path) -> None:
        run_dir.mkdir()
        (run_dir / "grouped_semantic_negotiation.raw_interaction.json").write_text(
            json.dumps(
                _grouped_interaction(
                    "original",
                    function_name="propose_grouped_edit_decisions",
                    arguments={"decisions": [{"beat_id": "fold"}]},
                )
            ),
            encoding="utf-8",
        )

    invalid_repair_dir = tmp_path / "invalid-repair"
    write_invalid_original(invalid_repair_dir)
    client, requests = _grouped_client_with_responses(
        [
            _grouped_interaction(
                "repair-invalid",
                function_name="repair_grouped_edit_decisions",
                arguments={"decisions": [{"beat_id": "fold"}]},
            )
        ]
    )
    kwargs = {
        "option_ids_by_beat": {"fold": ("fold--option-a",)},
        "allowed_reuse_of_beat_ids_by_option": {
            "fold": {"fold--option-a": ()}
        },
        "prompt": "not sent on resume",
        "policy": _grouped_semantic_policy(),
        "run_dir": invalid_repair_dir,
        "recovery_call": True,
    }
    with pytest.raises(ValueError, match="refusing another repair"):
        client.negotiate_grouped_edit_decisions(**kwargs)
    with pytest.raises(ValueError, match="refusing another repair"):
        client.negotiate_grouped_edit_decisions(**kwargs)
    assert len(requests) == 1

    unavailable_dir = tmp_path / "repair-503"
    write_invalid_original(unavailable_dir)
    unavailable_client, unavailable_requests = _grouped_client_with_responses()
    unavailable_kwargs = {**kwargs, "run_dir": unavailable_dir}
    with pytest.raises(RuntimeError, match="503"):
        unavailable_client.negotiate_grouped_edit_decisions(**unavailable_kwargs)
    with pytest.raises(PaidDispatchAlreadyRecorded):
        unavailable_client.negotiate_grouped_edit_decisions(**unavailable_kwargs)
    assert len(unavailable_requests) == 1


def test_interactions_mime_type_normalizes_common_audio_aliases() -> None:
    assert canonical_interactions_mime_type("audio/x-wav") == "audio/wav"
    assert canonical_interactions_mime_type("audio/vnd.wave") == "audio/wav"
    assert canonical_interactions_mime_type("audio/x-m4a") == "audio/m4a"
    assert canonical_interactions_mime_type("audio/mpeg") == "audio/mpeg"


def test_selected_vertical_framing_removes_non_executable_camera() -> None:
    canonical, changes = canonicalize_selected_vertical_framing_output(
        json.dumps(
            {
                "recommended_action": "fit_or_layout",
                "virtual_camera_proposal": {
                    "composition_mode": "joint_relation",
                    "phases": [],
                },
            }
        )
    )
    payload = json.loads(canonical)
    assert payload["recommended_action"] == "fit_or_layout"
    assert payload["virtual_camera_proposal"] is None
    assert [change["reason"] for change in changes] == [
        "non_executable_surplus_removed_for_non_tracked_action"
    ]


def test_selected_vertical_framing_repairs_camera_phase_representation() -> None:
    canonical, changes = canonicalize_selected_vertical_framing_output(
        json.dumps(
            {
                "recommended_action": "tracked_crop",
                "regions": [
                    {"region_id": "subject", "role": "required"},
                    {"region_id": "context", "role": "preferred"},
                ],
                "virtual_camera_proposal": {
                    "composition_mode": "single_anchor_follow",
                    "phases": [
                        {
                            "start_progress": 0.1,
                            "end_progress": 0.8,
                            "anchor_region_ids": ["subject", "context"],
                            "transition_in": "smoothstep",
                            "transition_duration_fraction": 0.0,
                        }
                    ],
                },
            }
        )
    )
    payload = json.loads(canonical)
    phase = payload["virtual_camera_proposal"]["phases"][0]
    assert phase["start_progress"] == 0.0
    assert phase["end_progress"] == 1.0
    assert phase["anchor_region_ids"] == ["subject"]
    assert phase["transition_in"] == "cut"
    assert phase["transition_duration_fraction"] == 0.0
    assert changes


def test_selected_vertical_framing_repairs_joint_hold_mode() -> None:
    canonical, changes = canonicalize_selected_vertical_framing_output(
        json.dumps(
            {
                "semantic_requirement": "simultaneous_relation",
                "recommended_action": "tracked_crop",
                "regions": [
                    {"region_id": "reference", "role": "required"},
                    {"region_id": "subject", "role": "required"},
                ],
                "virtual_camera_proposal": {
                    "composition_mode": "single_anchor_hold",
                    "phases": [
                        {
                            "phase_id": "comparison",
                            "anchor_region_ids": ["reference", "subject"],
                        }
                    ],
                },
            }
        )
    )
    payload = json.loads(canonical)
    assert (
        payload["virtual_camera_proposal"]["composition_mode"]
        == "joint_relation"
    )
    assert changes[0]["reason"].startswith(
        "multiple_simultaneous_evidence_anchors"
    )


def test_selected_vertical_framing_normalizes_clippable_relation_participants() -> None:
    canonical, changes = canonicalize_selected_vertical_framing_output(
        json.dumps(
            {
                "semantic_requirement": "simultaneous_relation",
                "recommended_action": "tracked_crop",
                "presentation_options": [
                    {
                        "mode": "controlled_clipping",
                        "verdict": "feasible",
                        "observable_reason": "Only non-semantic outer extent is clipped.",
                    }
                ],
                "regions": [
                    {
                        "region_id": "subject",
                        "entity_id": "entity-subject",
                        "kind": "subject",
                        "evidence_role": "primary_subject",
                        "role": "required",
                        "atomic": True,
                    },
                    {
                        "region_id": "reference",
                        "entity_id": "entity-reference",
                        "kind": "subject",
                        "evidence_role": "context_reference",
                        "role": "required",
                        "atomic": True,
                    },
                ],
                "virtual_camera_proposal": {
                    "composition_mode": "joint_relation",
                    "phases": [],
                },
            }
        )
    )
    payload = json.loads(canonical)

    assert all(
        region["evidence_role"] == "relation_participant"
        for region in payload["regions"]
    )
    assert all(region["atomic"] is False for region in payload["regions"])
    assert {
        change["reason"] for change in changes
    } >= {
        "bound_required_regions_in_an_explicit_simultaneous_relation_are_relation_participants",
        "controlled_clipping_cannot_treat_an_ordinary_relation_participant_as_an_indivisible_region",
    }


def test_selected_vertical_framing_keeps_atomic_relational_core_single_anchor() -> None:
    canonical, changes = canonicalize_selected_vertical_framing_output(
        json.dumps(
            {
                "semantic_requirement": "simultaneous_relation",
                "recommended_action": "tracked_crop",
                "regions": [
                    {
                        "region_id": "contact-core",
                        "role": "required",
                        "atomic": True,
                        "observable_relations": ["reference touches subject edge"],
                    },
                    {
                        "region_id": "outer-context",
                        "role": "preferred",
                        "atomic": False,
                        "minimum_visible_fraction": 0.5,
                    },
                ],
                "virtual_camera_proposal": {
                    "composition_mode": "single_anchor_hold",
                    "phases": [
                        {
                            "phase_id": "hold",
                            "start_progress": 0,
                            "end_progress": 1,
                            "anchor_region_ids": [
                                "contact-core",
                                "outer-context",
                            ],
                            "transition_in": "cut",
                            "transition_duration_fraction": 0,
                        }
                    ],
                },
            }
        )
    )
    payload = json.loads(canonical)

    assert (
        payload["virtual_camera_proposal"]["composition_mode"]
        == "single_anchor_hold"
    )
    assert payload["virtual_camera_proposal"]["phases"][0][
        "anchor_region_ids"
    ] == ["contact-core"]
    assert any(
        "single_anchor_mode_uses_the_only_hard_core" in change["reason"]
        for change in changes
    )


def test_selected_vertical_framing_removes_zero_fraction_soft_non_constraint() -> None:
    canonical, changes = canonicalize_selected_vertical_framing_output(
        json.dumps(
            {
                "recommended_action": "tracked_crop",
                "regions": [
                    {
                        "region_id": "subject",
                        "role": "required",
                        "minimum_visible_fraction": 1,
                    },
                    {
                        "region_id": "unused-context",
                        "role": "preferred",
                        "minimum_visible_fraction": 0,
                    },
                ],
                "virtual_camera_proposal": {
                    "composition_mode": "single_anchor_hold",
                    "phases": [
                        {
                            "phase_id": "hold",
                            "start_progress": 0,
                            "end_progress": 1,
                            "anchor_region_ids": ["subject"],
                            "transition_in": "cut",
                            "transition_duration_fraction": 0,
                        }
                    ],
                },
            }
        )
    )
    payload = json.loads(canonical)

    assert [region["region_id"] for region in payload["regions"]] == ["subject"]
    assert any(
        change["reason"].startswith(
            "unreferenced_zero_fraction_preferred_region"
        )
        for change in changes
    )


def test_feature_plan_normalizes_strict_hard_core_and_atomic_compound_group() -> None:
    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "evidence_status": "supported",
                        "horizontal_frame_id": "RF000001",
                        "vertical_frame_id": "RF000001",
                        "vertical_coverage_intent": "simultaneous_relation",
                        "vertical_coverage_target_descriptions": [
                            "The complete visible group"
                        ],
                        "vertical_candidates": [
                            {
                                "candidate_id": "strict-candidate",
                                "rank": 1,
                                "strategy": "tracked_crop",
                                "crop_mode": "strict",
                                "regions": [
                                    {
                                        "region_id": "strict-group",
                                        "role": "required",
                                        "atomic": False,
                                        "minimum_visible_fraction": 0.8,
                                    }
                                ],
                            },
                            {
                                "candidate_id": "compound-candidate",
                                "rank": 2,
                                "strategy": "fit_with_background",
                                "crop_mode": "primary_center",
                                "regions": [
                                    {
                                        "region_id": "compound-group",
                                        "role": "required",
                                        "atomic": True,
                                        "minimum_visible_fraction": 1.0,
                                        "observable_relations": [],
                                    }
                                ],
                            },
                            {
                                "candidate_id": "strict-atomic-candidate",
                                "rank": 3,
                                "strategy": "tracked_crop",
                                "crop_mode": "strict",
                                "regions": [
                                    {
                                        "region_id": "strict-atomic-screen",
                                        "role": "required",
                                        "atomic": True,
                                        "minimum_visible_fraction": 0.8,
                                        "observable_relations": [],
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }
        )
    )
    payload = json.loads(canonical)

    assert payload["chapters"][0]["vertical_candidates"][0]["regions"][0][
        "minimum_visible_fraction"
    ] == 1.0
    assert payload["chapters"][0]["vertical_candidates"][2]["regions"][0][
        "minimum_visible_fraction"
    ] == 1.0
    # The candidates are not uniformly atomic compounds, so a claimed
    # simultaneous relation is not silently weakened.
    assert (
        payload["chapters"][0]["vertical_coverage_intent"]
        == "simultaneous_relation"
    )
    assert any(
        change["rule"] == "strict_crop_system_policy_requires_full_hard_core"
        for change in changes
    )


def test_feature_plan_reclassifies_relationless_atomic_compound_as_group() -> None:
    candidate = {
        "strategy": "fit_with_background",
        "regions": [
            {
                "region_id": "compound-group",
                "role": "required",
                "atomic": True,
                "minimum_visible_fraction": 1.0,
                "observable_relations": [],
            }
        ],
    }
    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "evidence_status": "supported",
                        "horizontal_frame_id": "RF000001",
                        "vertical_frame_id": "RF000001",
                        "vertical_coverage_intent": "simultaneous_relation",
                        "vertical_coverage_target_descriptions": [
                            "The complete visible group"
                        ],
                        "vertical_candidates": [
                            {"candidate_id": "a", "rank": 1, **candidate},
                            {"candidate_id": "b", "rank": 2, **candidate},
                        ],
                    }
                ]
            }
        )
    )
    payload = json.loads(canonical)

    assert payload["chapters"][0]["vertical_coverage_intent"] == "group_coverage"
    assert any(
        change["rule"]
        == "atomic_compound_without_relation_is_group_coverage"
        for change in changes
    )


def test_feature_plan_single_candidate_lists_use_legacy_projection() -> None:
    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "horizontal_candidates": [{"candidate_id": "one"}],
                        "vertical_candidates": [{"candidate_id": "one"}],
                    },
                    {
                        "horizontal_candidates": [
                            {"candidate_id": "one"},
                            {"candidate_id": "two"},
                        ],
                        "vertical_candidates": [],
                    },
                ]
            }
        )
    )
    payload = json.loads(canonical)
    assert payload["chapters"][0]["horizontal_candidates"] == []
    assert payload["chapters"][0]["vertical_candidates"] == []
    assert len(payload["chapters"][1]["horizontal_candidates"]) == 2
    assert len(changes) == 2


def test_feature_plan_rank_one_candidate_repairs_redundant_legacy_projection() -> None:
    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "evidence_status": "supported",
                        "horizontal_frame_id": "RF000001",
                        "vertical_frame_id": "RF000002",
                        "horizontal_strategy": "original",
                        "horizontal_zoom_intent": "none",
                        "horizontal_camera_intent": "hold",
                        "horizontal_target_description": None,
                        "vertical_strategy": "fit_with_background",
                        "vertical_target_description": "old target",
                        "horizontal_candidates": [
                            {
                                "rank": 1,
                                "frame_id": "RF000101",
                                "strategy": "tracked_reframe",
                                "zoom_intent": "detail",
                                "camera_intent": "push_in",
                                "target_description": "rank-one focus",
                            },
                            {"rank": 2},
                        ],
                        "vertical_candidates": [
                            {
                                "rank": 1,
                                "frame_id": "RF000202",
                                "strategy": "tracked_crop",
                                "target_description": "rank-one portrait target",
                            },
                            {"rank": 2},
                        ],
                    }
                ]
            }
        )
    )

    chapter = json.loads(canonical)["chapters"][0]
    assert chapter["horizontal_frame_id"] == "RF000101"
    assert chapter["horizontal_strategy"] == "tracked_reframe"
    assert chapter["horizontal_zoom_intent"] == "detail"
    assert chapter["horizontal_camera_intent"] == "push_in"
    assert chapter["horizontal_target_description"] == "rank-one focus"
    assert chapter["vertical_frame_id"] == "RF000202"
    assert chapter["vertical_strategy"] == "tracked_crop"
    assert chapter["vertical_target_description"] == "rank-one portrait target"
    assert len(changes) == 8
    assert {
        change["rule"] for change in changes
    } == {"rank_one_candidate_is_authoritative_legacy_projection"}


class _RejectingInteractions:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **request: Any) -> None:
        self.request = request
        raise _StopRequest("request captured")


def _client() -> tuple[GeminiLabClient, _RejectingInteractions]:
    interactions = _RejectingInteractions()
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = MODEL_ID
    return client, interactions


class _CompletedFeatureInteraction:
    id = "paid-feature-interaction"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": MODEL_ID,
            "output_text": self.output_text,
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 10,
                "total_thought_tokens": 0,
            },
        }


class _CompletedFeatureInteractions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls = 0

    def create(self, **_request: Any) -> _CompletedFeatureInteraction:
        self.calls += 1
        return _CompletedFeatureInteraction(self.output_text)


class _ForbiddenInteractions:
    def create(self, **_request: Any) -> None:
        pytest.fail("raw output reuse must not issue a new Gemini request")


def _feature_plan_fixture(
    tmp_path: Path,
) -> tuple[RushesCatalog, FeatureEditBrief, FeatureEditPlan]:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    reel_path = tmp_path / "catalog-reel.mp4"
    reel_path.write_bytes(b"catalog reel")
    clip = RushClip(
        clip_id="clip-generic",
        path=str(source_path),
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=source_path.stat().st_size,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=2_000,
        image_path=str(tmp_path / "frame.jpg"),
    )
    catalog = RushesCatalog(
        catalog_id="catalog-generic",
        source_directory=str(tmp_path),
        sample_interval_ms=2_000,
        total_duration_ms=clip.duration_ms,
        clips=[clip],
        frames=[frame],
        analysis_reel_path=str(reel_path),
        generated_at="2026-07-23T00:00:00+00:00",
    )
    brief = FeatureEditBrief(
        project_id="project-generic",
        title="Generic edit",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="opening",
                title="Opening",
                detail_lines=[],
                target_duration_seconds=3,
            )
        ],
    )
    plan = FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=brief.title,
        chapters=[
            FeatureChapterSelect(
                feature_id="opening",
                evidence_status="supported",
                horizontal_frame_id=frame.frame_id,
                vertical_frame_id=frame.frame_id,
                observed_visual_evidence="A directly visible subject.",
                selection_reason="Representative visible evidence.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=[],
                confidence=0.9,
                recommended_duration_seconds=3,
                duration_rationale="One concise observable state.",
            )
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id=MODEL_ID,
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="paid-run",
            generated_at="2026-07-23T00:00:00+00:00",
        ),
    )
    return catalog, brief, plan


def _run_paid_feature_plan(
    *,
    tmp_path: Path,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    prompt_template: str = "Select observable evidence.",
    music_sha256: str | None = None,
) -> Path:
    run_dir = tmp_path / "feature-plan"
    run_dir.mkdir()
    interactions = _CompletedFeatureInteractions(plan.model_dump_json())
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = MODEL_ID
    uploaded_audio = (
        SimpleNamespace(
            uri="https://example.invalid/music",
            mime_type="audio/wav",
        )
        if music_sha256 is not None
        else None
    )
    client.plan_feature_edit(
        catalog=catalog,
        brief=brief,
        uploaded=SimpleNamespace(
            uri="https://example.invalid/reel",
            mime_type="video/mp4",
        ),
        uploaded_audio=uploaded_audio,
        music_sha256=music_sha256,
        prompt_template=prompt_template,
        run_id="fresh-run",
        run_dir=run_dir,
    )
    assert interactions.calls == 1
    return run_dir


def test_feature_plan_missing_aspect_fails_closed_without_projection(
    tmp_path: Path,
) -> None:
    _, _, plan = _feature_plan_fixture(tmp_path)
    payload = plan.model_dump(mode="json")
    payload["chapters"][0]["horizontal_frame_id"] = None

    with pytest.raises(
        ValueError,
        match="cross-aspect projection is an editorial decision",
    ):
        canonicalize_feature_edit_plan_output(json.dumps(payload))

    assert payload["chapters"][0]["horizontal_frame_id"] is None
    assert payload["chapters"][0]["vertical_frame_id"] == "RF000001"


def test_feature_plan_derives_missing_preferred_dwell_from_model_envelope(
    tmp_path: Path,
) -> None:
    _, _, plan = _feature_plan_fixture(tmp_path)
    payload = plan.model_dump(mode="json")
    chapter = payload["chapters"][0]
    chapter["recommended_duration_seconds"] = None
    chapter["attention_observation"] = {
        "semantic_novelty": 0.7,
        "action_progress": 0.5,
        "visual_motion": 0.2,
        "composition_change": 0.2,
        "reading_load": 0.4,
        "unresolved_tension": 0.1,
        "emotional_hold_value": 0.5,
        "repetition_pressure": 0.1,
        "music_transition_opportunity": 0.6,
        "minimum_dwell_seconds": 3.0,
        "maximum_dwell_seconds": 7.0,
        "rationale": "Bounded model observation.",
        "uncertainties": [],
        "requires_human_review": True,
    }

    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(payload)
    )

    parsed = FeatureEditPlan.model_validate_json(canonical)
    assert parsed.chapters[0].recommended_duration_seconds == 5.0
    assert any(
        change["reason"]
        == "deterministic_midpoint_of_model_dwell_envelope"
        for change in changes
    )


def test_feature_plan_rf_ids_are_catalog_bound_in_response_schema() -> None:
    legal_ids = ["RF000001", "RF000002"]
    schema = gemini_module._feature_edit_plan_response_schema(legal_ids)
    chapter_schema = schema["$defs"]["FeatureChapterSelect"]
    properties = chapter_schema["properties"]
    for field_name in ("horizontal_frame_id", "vertical_frame_id"):
        assert properties[field_name] == {
            "type": "string",
            "enum": [*legal_ids, "RF_NONE"],
        }
        assert field_name in chapter_schema["required"]

    for definition in ("FeatureHorizontalCandidate", "FeatureVerticalCandidate"):
        frame_schema = schema["$defs"][definition]["properties"]["frame_id"]
        assert frame_schema["enum"] == legal_ids

    for field_name in (
        "horizontal_camera_intent",
        "duration_rationale",
        "attention_observation",
    ):
        assert field_name in chapter_schema["required"]


def test_feature_plan_not_found_transport_sentinel_becomes_local_null() -> None:
    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "evidence_status": "not_found",
                        "horizontal_frame_id": "RF_NONE",
                        "vertical_frame_id": "RF_NONE",
                    }
                ]
            }
        )
    )
    chapter = json.loads(canonical)["chapters"][0]
    assert chapter["horizontal_frame_id"] is None
    assert chapter["vertical_frame_id"] is None
    assert changes[0]["rule"] == "not_found_transport_sentinel_to_local_null"


def test_feature_plan_attention_review_gate_is_system_owned() -> None:
    canonical, changes = canonicalize_feature_edit_plan_output(
        json.dumps(
            {
                "chapters": [
                    {
                        "evidence_status": "supported",
                        "horizontal_frame_id": "RF000001",
                        "vertical_frame_id": "RF000002",
                        "attention_observation": {
                            "requires_human_review": False,
                        },
                    }
                ]
            }
        )
    )
    attention = json.loads(canonical)["chapters"][0]["attention_observation"]
    assert attention["requires_human_review"] is True
    assert changes[0]["rule"] == (
        "system_owned_attention_review_gate_is_always_true"
    )


def test_feature_plan_raw_reuse_accepts_exact_causal_binding_without_api(
    tmp_path: Path,
) -> None:
    catalog, brief, plan = _feature_plan_fixture(tmp_path)
    music_sha256 = "a" * 64
    run_dir = _run_paid_feature_plan(
        tmp_path=tmp_path,
        catalog=catalog,
        brief=brief,
        plan=plan,
        music_sha256=music_sha256,
    )
    binding_path = run_dir / "feature_edit_plan.raw_output_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))

    assert binding["contract_version"] == (
        "feature-edit-plan-raw-reuse-binding-v1"
    )
    assert binding["music_sha256"] == music_sha256
    assert binding["catalog_reel_sha256"] == hashlib.sha256(
        Path(catalog.analysis_reel_path).read_bytes()
    ).hexdigest()
    assert {
        "catalog_definition_sha256",
        "brief_definition_sha256",
        "prompt_template_sha256",
        "causal_prompt_sha256",
        "system_instruction_sha256",
        "response_schema_sha256",
        "model_id_sha256",
        "request_definition_sha256",
        "definition_sha256",
    }.issubset(binding)

    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=_ForbiddenInteractions())
    client.model_id = MODEL_ID
    reused = client.plan_feature_edit(
        catalog=catalog,
        brief=brief,
        uploaded=None,
        music_sha256=music_sha256,
        prompt_template="Select observable evidence.",
        run_id="reuse-run",
        run_dir=run_dir,
        reuse_raw_output=True,
    )

    assert reused.model_provenance.interaction_id == "paid-feature-interaction"
    reuse_record = json.loads(
        (run_dir / "feature_edit_plan.raw_output_reuse.json").read_text(
            encoding="utf-8"
        )
    )
    assert reuse_record["causal_input_binding_sha256"] == hashlib.sha256(
        binding_path.read_bytes()
    ).hexdigest()
    assert reuse_record["causal_input_definition_sha256"] == binding[
        "definition_sha256"
    ]


def test_feature_plan_raw_reuse_fails_closed_without_binding(
    tmp_path: Path,
) -> None:
    catalog, brief, plan = _feature_plan_fixture(tmp_path)
    run_dir = _run_paid_feature_plan(
        tmp_path=tmp_path,
        catalog=catalog,
        brief=brief,
        plan=plan,
    )
    (run_dir / "feature_edit_plan.raw_output_binding.json").unlink()
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=_ForbiddenInteractions())
    client.model_id = MODEL_ID

    with pytest.raises(
        FileNotFoundError,
        match="causal input binding",
    ):
        client.plan_feature_edit(
            catalog=catalog,
            brief=brief,
            uploaded=None,
            prompt_template="Select observable evidence.",
            run_id="reuse-run",
            run_dir=run_dir,
            reuse_raw_output=True,
        )


def test_fresh_feature_plan_refuses_to_resend_existing_paid_evidence(
    tmp_path: Path,
) -> None:
    catalog, brief, plan = _feature_plan_fixture(tmp_path)
    run_dir = _run_paid_feature_plan(
        tmp_path=tmp_path,
        catalog=catalog,
        brief=brief,
        plan=plan,
    )
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=_ForbiddenInteractions())
    client.model_id = MODEL_ID

    with pytest.raises(
        FileExistsError,
        match="explicit --reuse-feature-plan-raw-output",
    ):
        client.plan_feature_edit(
            catalog=catalog,
            brief=brief,
            uploaded=SimpleNamespace(
                uri="https://example.invalid/reel",
                mime_type="video/mp4",
            ),
            prompt_template="Select observable evidence.",
            run_id="accidental-fresh-replay",
            run_dir=run_dir,
            reuse_raw_output=False,
        )


@pytest.mark.parametrize(
    "changed_input",
    [
        "brief",
        "catalog",
        "catalog_reel",
        "prompt",
        "system_instruction",
        "response_schema",
        "model",
        "music",
    ],
)
def test_feature_plan_raw_reuse_rejects_changed_causal_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    catalog, brief, plan = _feature_plan_fixture(tmp_path)
    original_prompt = "Select observable evidence."
    original_music_sha256 = "a" * 64
    run_dir = _run_paid_feature_plan(
        tmp_path=tmp_path,
        catalog=catalog,
        brief=brief,
        plan=plan,
        prompt_template=original_prompt,
        music_sha256=original_music_sha256,
    )
    current_catalog = catalog
    current_brief = brief
    current_prompt = original_prompt
    current_music_sha256 = original_music_sha256
    current_model = MODEL_ID
    if changed_input == "brief":
        current_brief = brief.model_copy(update={"title": "Changed intent"})
    elif changed_input == "catalog":
        current_catalog = catalog.model_copy(
            update={"sample_interval_ms": 3_000}
        )
    elif changed_input == "catalog_reel":
        Path(catalog.analysis_reel_path).write_bytes(b"changed catalog reel")
    elif changed_input == "prompt":
        current_prompt = "Use a changed editorial prompt."
    elif changed_input == "system_instruction":
        monkeypatch.setattr(
            gemini_module,
            "EDITORIAL_SYSTEM_INSTRUCTION",
            EDITORIAL_SYSTEM_INSTRUCTION + "\nChanged binding test.",
        )
    elif changed_input == "response_schema":
        original_schema_builder = gemini_module.gemini_response_schema

        def changed_schema(model: type[Any]) -> dict[str, Any]:
            schema = original_schema_builder(model)
            return {**schema, "x-binding-test": True}

        monkeypatch.setattr(
            gemini_module,
            "gemini_response_schema",
            changed_schema,
        )
    elif changed_input == "model":
        current_model = "gemini-3.5-flash"
    elif changed_input == "music":
        current_music_sha256 = "b" * 64
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(changed_input)

    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=_ForbiddenInteractions())
    client.model_id = current_model
    with pytest.raises(
        ValueError,
        match="raw feature-plan reuse inputs differ",
    ):
        client.plan_feature_edit(
            catalog=current_catalog,
            brief=current_brief,
            uploaded=None,
            music_sha256=current_music_sha256,
            prompt_template=current_prompt,
            run_id="reuse-run",
            run_dir=run_dir,
            reuse_raw_output=True,
        )


def _semantic_music_fixture() -> tuple[Any, Any, str]:
    music_sha256 = "c" * 64
    music_definition_sha256 = "d" * 64
    visual_sha256 = "e" * 64
    music_lock = SimpleNamespace(
        music_id=f"sha256:{music_sha256}",
        definition_sha256=music_definition_sha256,
        duration_ms=10_000,
        bpm=120.0,
        meter=4,
        sections=[],
        cues=[],
    )
    visual_map = SimpleNamespace(
        project_duration_ms=10_000,
        aspect_ratio="16:9",
        points=[],
        model_dump=lambda mode: {
            "project_duration_ms": 10_000,
            "aspect_ratio": "16:9",
            "points": [],
        },
    )
    return music_lock, visual_map, visual_sha256


def _semantic_music_output(
    *,
    music_lock: Any,
    visual_sha256: str,
) -> str:
    return json.dumps(
        {
            "contract_version": "semantic-music-pairing-v1",
            "music_id": music_lock.music_id,
            "music_definition_sha256": music_lock.definition_sha256,
            "visual_sync_map_sha256": visual_sha256,
            "global_strategy": "Preserve the observable musical flow.",
            "section_interpretations": [],
            "pairings": [],
            "uncertainties": [],
            "requires_human_review": True,
            "model_provenance": ModelProvenance(
                model_id=MODEL_ID,
                api="gemini_interactions",
                sdk="google-genai",
                sdk_version="test",
                run_id="semantic-paid",
                generated_at="2026-07-23T00:00:00+00:00",
            ).model_dump(mode="json"),
        }
    )


def test_semantic_music_raw_reuse_requires_exact_definition_binding(
    tmp_path: Path,
) -> None:
    music_lock, visual_map, visual_sha256 = _semantic_music_fixture()
    interactions = _CompletedFeatureInteractions(
        _semantic_music_output(
            music_lock=music_lock,
            visual_sha256=visual_sha256,
        )
    )
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = MODEL_ID
    uploaded_audio = SimpleNamespace(
        uri="https://example.invalid/music",
        mime_type="audio/wav",
    )
    run_dir = tmp_path / "semantic-music"
    client.plan_music_semantic_pairing(
        music_lock=music_lock,
        visual_map=visual_map,
        visual_sync_map_sha256=visual_sha256,
        uploaded_audio=uploaded_audio,
        prompt_template="Interpret only the supplied audible evidence.",
        run_id="fresh-semantic-run",
        run_dir=run_dir,
    )
    assert interactions.calls == 1
    binding_path = (
        run_dir / "semantic_music_pairing.raw_output_binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["music_media_sha256"] == "c" * 64
    assert binding["music_definition_sha256"] == "d" * 64
    assert binding["visual_sync_map_sha256"] == visual_sha256
    assert {
        "prompt_template_sha256",
        "causal_prompt_sha256",
        "system_instruction_sha256",
        "response_schema_sha256",
        "model_id_sha256",
        "request_definition_sha256",
        "full_request_sha256",
        "definition_sha256",
    }.issubset(binding)

    client.client = SimpleNamespace(interactions=_ForbiddenInteractions())
    reused = client.plan_music_semantic_pairing(
        music_lock=music_lock,
        visual_map=visual_map,
        visual_sync_map_sha256=visual_sha256,
        uploaded_audio=uploaded_audio,
        prompt_template="Interpret only the supplied audible evidence.",
        run_id="reuse-semantic-run",
        run_dir=run_dir,
        reuse_raw_output=True,
    )
    assert reused.model_provenance.interaction_id == "paid-feature-interaction"

    with pytest.raises(
        ValueError,
        match="semantic music raw reuse inputs differ",
    ):
        client.plan_music_semantic_pairing(
            music_lock=music_lock,
            visual_map=visual_map,
            visual_sync_map_sha256=visual_sha256,
            uploaded_audio=uploaded_audio,
            prompt_template="A changed semantic music prompt.",
            run_id="changed-semantic-run",
            run_dir=run_dir,
            reuse_raw_output=True,
        )


def test_live_client_disables_hidden_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(gemini_module.genai, "Client", _Client)
    client = GeminiLabClient(api_key="test-key")
    try:
        retry_options = captured["http_options"].retry_options
        assert retry_options is not None
        assert retry_options.attempts == 1
    finally:
        client.close()


def test_paid_responses_are_preserved_as_immutable_attempts(tmp_path: Path) -> None:
    class _PaidInteraction:
        id = "interaction-reused-by-test-double"

        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "id": self.id,
                "model": "gemini-3.6-flash",
                "output_text": "{}",
                "usage": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 10,
                    "total_thought_tokens": 0,
                },
            }

    interaction = _PaidInteraction()
    for _ in range(2):
        gemini_module._record_interaction_attempt(
            run_dir=tmp_path,
            operation="contract_test",
            canonical_filename="contract_test.raw_interaction.json",
            interaction=interaction,
        )

    assert len(list((tmp_path / "attempts").glob("*.raw_interaction.json"))) == 2
    assert (tmp_path / "contract_test.raw_interaction.json").is_file()
    summary = summarize_usage_and_list_price(tmp_path)
    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1


def _capture_request(
    tmp_path: Path,
    name: str,
    request_filename: str,
    invoke: Callable[[GeminiLabClient, Path], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = tmp_path / name
    run_dir.mkdir()
    client, interactions = _client()
    with pytest.raises(_StopRequest, match="request captured"):
        invoke(client, run_dir)
    assert interactions.request is not None
    saved = json.loads((run_dir / request_filename).read_text(encoding="utf-8"))
    return interactions.request, saved


def test_all_candidate_and_frame_observation_calls_use_visual_evidence_instruction(
    tmp_path: Path,
) -> None:
    media = SimpleNamespace(asset_id="sha256:" + "a" * 64, duration_ms=10_000)
    uploaded = SimpleNamespace(uri="https://example.invalid/video", mime_type="video/mp4")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"test image payload")

    event = SimpleNamespace(
        event_id="event-generic",
        grounding_targets=[],
        model_dump_json=lambda indent=None: "{}",
    )
    dense_catalog = SimpleNamespace(
        source_asset_id=media.asset_id,
        frames=[SimpleNamespace(frame_id="DF000001")],
        contact_sheet_paths=[str(image)],
        contact_sheet_hashes=["b" * 64],
    )

    calls: list[tuple[str, str, Callable[[GeminiLabClient, Path], Any]]] = [
        (
            "targets",
            "target_candidates.request.json",
            lambda client, run_dir: client.suggest_targets(
                media=media,
                uploaded=uploaded,
                prompt_template="Find observable candidate targets.",
                run_id="run-targets",
                run_dir=run_dir,
            ),
        ),
        (
            "storyboard",
            "indexed_storyboard.request.json",
            lambda client, run_dir: client.analyze_indexed_storyboard(
                media=media,
                frames=[
                    {
                        "frame_id": "F000001",
                        "frame_pts": 0,
                        "frame_time_ms": 0,
                        "image_path": str(image),
                        "image_hash": "c" * 64,
                    }
                ],
                prompt_template="Describe only the supplied frames.",
                run_id="run-storyboard",
                run_dir=run_dir,
            ),
        ),
        (
            "moments",
            "direct_moments.request.json",
            lambda client, run_dir: client.analyze_direct_moments(
                media=media,
                uploaded=uploaded,
                prompt_template="Suggest observable moments.",
                run_id="run-moments",
                run_dir=run_dir,
                locked_target_id="entity-generic",
                locked_target_description="the user-selected physical object",
            ),
        ),
        (
            "dense",
            "dense_selection.request.json",
            lambda client, run_dir: client.select_dense_event_frames(
                event=event,
                catalog=dense_catalog,
                prompt_template="Select only supplied frame IDs.",
                run_id="run-dense",
                run_dir=run_dir,
            ),
        ),
    ]

    for name, filename, invoke in calls:
        api_request, saved_request = _capture_request(
            tmp_path, name, filename, invoke
        )
        assert api_request["system_instruction"] == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
        assert saved_request["system_instruction"] == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
        assert api_request["model"] == MODEL_ID
        assert not {"temperature", "top_p", "top_k"}.intersection(
            api_request["generation_config"]
        )
        assert api_request["generation_config"]["thinking_level"] in {"low", "high"}


def test_edit_planning_calls_separate_intent_from_media_evidence(tmp_path: Path) -> None:
    uploaded = SimpleNamespace(uri="https://example.invalid/reel", mime_type="video/mp4")
    catalog = SimpleNamespace(
        catalog_id="catalog-generic",
        frames=[SimpleNamespace(frame_id="RF000001")],
    )
    brief = SimpleNamespace(
        project_id="project-generic",
        model_dump_json=lambda indent=None: '{"chapters": []}',
    )

    calls: list[tuple[str, str, Callable[[GeminiLabClient, Path], Any]]] = [
        (
            "rushes",
            "rushes_edit_plan.request.json",
            lambda client, run_dir: client.plan_rushes_edit(
                catalog=catalog,
                uploaded=uploaded,
                prompt_template="Plan an evidence-backed edit.",
                project_id="project-generic",
                run_id="run-rushes",
                run_dir=run_dir,
            ),
        ),
        (
            "feature",
            "feature_edit_plan.request.json",
            lambda client, run_dir: client.plan_feature_edit(
                catalog=catalog,
                brief=brief,
                uploaded=uploaded,
                prompt_template="Match the brief to supported footage.",
                run_id="run-feature",
                run_dir=run_dir,
            ),
        ),
    ]

    for name, filename, invoke in calls:
        api_request, saved_request = _capture_request(
            tmp_path, name, filename, invoke
        )
        assert api_request["system_instruction"] == EDITORIAL_SYSTEM_INSTRUCTION
        assert saved_request["system_instruction"] == EDITORIAL_SYSTEM_INSTRUCTION
        assert api_request["model"] == MODEL_ID
        assert not {"temperature", "top_p", "top_k"}.intersection(
            api_request["generation_config"]
        )
        assert api_request["generation_config"]["thinking_level"] in {"low", "high"}

    assert "不證明素材中存在相符畫面" in EDITORIAL_SYSTEM_INSTRUCTION
    assert "不得選擇不相符素材補位" in EDITORIAL_SYSTEM_INSTRUCTION
    assert "OPPO" not in EDITORIAL_SYSTEM_INSTRUCTION
    assert "Reno" not in EDITORIAL_SYSTEM_INSTRUCTION


def test_live_request_sources_do_not_use_deprecated_sampling_parameters() -> None:
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "src", root / "scripts"):
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert '"temperature"' not in source, path
            assert '"top_p"' not in source, path
            assert '"top_k"' not in source, path


def test_ground_frame_interleaves_content_addressed_identity_references(
    tmp_path: Path,
) -> None:
    target_frame = tmp_path / "target-frame.png"
    target_frame.write_bytes(b"target frame bytes")
    target_hash = hashlib.sha256(target_frame.read_bytes()).hexdigest()
    positive = tmp_path / "positive.png"
    positive.write_bytes(b"same locked instance")
    negative = tmp_path / "negative.png"
    negative.write_bytes(b"explicit confuser")

    references = (
        GroundingIdentityReference(
            reference_id="anchor-positive",
            role="positive",
            target_id="subject.primary",
            description="same reviewer-selected instance",
            path=positive,
            sha256=hashlib.sha256(positive.read_bytes()).hexdigest(),
        ),
        GroundingIdentityReference(
            reference_id="anchor-negative",
            role="negative",
            target_id="subject.primary",
            description="similar instance that must be excluded",
            path=negative,
            sha256=hashlib.sha256(negative.read_bytes()).hexdigest(),
        ),
    )
    media = SimpleNamespace(asset_id="sha256:" + "a" * 64)
    frame = SimpleNamespace(
        path=str(target_frame),
        frame_time_ms=1250,
        frame_pts=30,
        frame_hash=target_hash,
        width=1920,
        height=1080,
    )

    api_request, saved_request = _capture_request(
        tmp_path,
        "ground-references",
        "grounding.request.json",
        lambda client, run_dir: client.ground_frame(
            media=media,
            frame=frame,
            event_id="event-generic",
            event_description="identity-only exact-frame grounding",
            entity_id="subject.primary",
            target_description="the locked foreground instance",
            prompt_template=(
                "Target {{target_description}} in event {{event_description}} "
                "for {{entity_id}} at {{frame_time_ms}}."
            ),
            run_id="run-ground-references",
            output_dir=run_dir,
            identity_references=references,
        ),
    )

    api_input = api_request["input"]
    labels = [item["text"] for item in api_input if item["type"] == "text"]
    assert any("anchor-positive" in label and "role=positive" in label for label in labels)
    assert any("anchor-negative" in label and "role=negative" in label for label in labels)
    assert labels[-1].startswith("FRAME_TO_GROUND")
    assert sum(item["type"] == "image" for item in api_input) == 3

    recorded_images = [
        item for item in saved_request["input"] if item["type"] == "image"
    ]
    assert [item.get("reference_role") for item in recorded_images[:-1]] == [
        "positive",
        "negative",
    ]
    assert recorded_images[-1]["image_role"] == "frame_to_ground"
    assert all("data" not in item for item in recorded_images)
    saved_text = json.dumps(saved_request, ensure_ascii=False)
    assert str(positive) not in saved_text
    assert str(negative) not in saved_text


def test_ground_frame_rejects_tampered_identity_reference_before_network(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"actual bytes")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"frame bytes")
    client, interactions = _client()

    with pytest.raises(ValueError, match="hash mismatch"):
        client.ground_frame(
            media=SimpleNamespace(asset_id="sha256:" + "a" * 64),
            frame=SimpleNamespace(
                path=str(frame_path),
                frame_time_ms=0,
                frame_pts=0,
                frame_hash=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                width=640,
                height=360,
            ),
            event_id="event-generic",
            event_description="identity-only",
            entity_id="subject.primary",
            target_description="the locked instance",
            prompt_template="Ground {{target_description}}.",
            run_id="run-tampered-reference",
            output_dir=tmp_path / "tampered",
            identity_references=(
                GroundingIdentityReference(
                    reference_id="tampered",
                    role="positive",
                    target_id="subject.primary",
                    description="same instance",
                    path=reference_path,
                    sha256="0" * 64,
                ),
            ),
        )
    assert interactions.request is None


def test_ground_frame_rejects_reference_for_another_requested_target(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"valid reference bytes")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"frame bytes")
    client, interactions = _client()

    with pytest.raises(ValueError, match="requested entity_id"):
        client.ground_frame(
            media=SimpleNamespace(asset_id="sha256:" + "a" * 64),
            frame=SimpleNamespace(
                path=str(frame_path),
                frame_time_ms=0,
                frame_pts=0,
                frame_hash=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                width=640,
                height=360,
            ),
            event_id="event-generic",
            event_description="identity-only",
            entity_id="subject.requested",
            target_description="the requested instance",
            prompt_template="Ground {{target_description}}.",
            run_id="run-wrong-target-reference",
            output_dir=tmp_path / "wrong-target",
            identity_references=(
                GroundingIdentityReference(
                    reference_id="wrong-target",
                    role="positive",
                    target_id="subject.other",
                    description="another instance",
                    path=reference_path,
                    sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                ),
            ),
        )
    assert interactions.request is None


def test_identity_checkpoint_is_exact_frame_verify_only_request(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "verify-frame.png"
    frame_path.write_bytes(b"frame to verify")
    reference_path = tmp_path / "verify-reference.png"
    reference_path.write_bytes(b"locked reference")
    frame_hash = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()

    api_request, saved_request = _capture_request(
        tmp_path,
        "identity-checkpoint",
        "identity_checkpoint.request.json",
        lambda client, run_dir: client.verify_identity_checkpoint(
            frame=SimpleNamespace(
                path=str(frame_path),
                frame_time_ms=2250,
                frame_pts=54,
                frame_hash=frame_hash,
            ),
            target_id="subject.primary",
            target_description="the reviewer-locked foreground instance",
            run_id="identity-checkpoint-run",
            output_dir=run_dir,
            identity_references=(
                GroundingIdentityReference(
                    reference_id="positive-anchor",
                    role="positive",
                    target_id="subject.primary",
                    description="same locked instance",
                    path=reference_path,
                    sha256=reference_hash,
                ),
            ),
        ),
    )

    assert api_request["system_instruction"] == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
    assert api_request["generation_config"] == SEMANTIC_IDENTITY_GENERATION_CONFIG
    assert api_request["response_format"]["schema"]["properties"].keys() >= {
        "verdict",
        "evidence",
    }
    texts = [
        item["text"] for item in api_request["input"] if item["type"] == "text"
    ]
    assert texts[0].startswith("## Mode: VERIFY_IDENTITY")
    assert "不得輸出或修改 bounding box" in texts[0]
    assert texts[-1].startswith("FRAME_TO_VERIFY")
    assert all(
        "data" not in item
        for item in saved_request["input"]
        if item["type"] == "image"
    )


def test_content_map_transport_failure_never_triggers_paid_schema_repair(
    tmp_path: Path,
) -> None:
    calls = 0

    def fail_request(**_request: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("503 service unavailable")

    client = object.__new__(GeminiLabClient)
    client.model_id = MODEL_ID
    client.client = SimpleNamespace(
        interactions=SimpleNamespace(create=fail_request)
    )
    media = MediaInfo(
        path=str(tmp_path / "source.mp4"),
        sha256="a" * 64,
        asset_id="sha256:" + "a" * 64,
        format_name="mp4",
        duration_ms=1000,
        size_bytes=1,
        format_metadata={},
        video=VideoStreamInfo(
            index=0,
            codec_name="h264",
            coded_width=1920,
            coded_height=1080,
            display_width=1920,
            display_height=1080,
            rotation_degrees=0,
            average_frame_rate=Rational(numerator=30, denominator=1),
            real_frame_rate=Rational(numerator=30, denominator=1),
            time_base=Rational(numerator=1, denominator=30),
            start_pts=0,
            duration_ts=30,
            metadata={},
        ),
    )

    with pytest.raises(RuntimeError, match="503"):
        client.analyze_video(
            media=media,
            uploaded=SimpleNamespace(uri="files/source", mime_type="video/mp4"),
            prompt_template="Analyze only visible evidence.",
            run_id="transport-failure",
            run_dir=tmp_path / "run",
            repair_attempts=3,
        )

    assert calls == 1
    validation = json.loads(
        (tmp_path / "run" / "content_map.schema_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["attempts"][0]["failure_stage"] == "interaction_request"
    assert validation["attempts"][0]["paid_repair_allowed"] is False


def test_content_map_schema_repair_is_text_only_and_video_is_low_resolution(
    tmp_path: Path,
    content_map: ContentMap,
) -> None:
    requests: list[dict[str, Any]] = []
    valid = content_map.model_copy(
        update={
            "model_provenance": content_map.model_provenance.model_copy(
                update={
                    "model_id": MODEL_ID,
                    "interaction_id": None,
                    "run_id": "text-repair",
                }
            )
        }
    )
    responses = iter(
        (
            _FakeInteraction(
                interaction_id="invalid",
                output_text="{}",
            ),
            _FakeInteraction(
                interaction_id="repaired",
                output_text=valid.model_dump_json(),
            ),
        )
    )

    def create(**request: Any) -> _FakeInteraction:
        requests.append(request)
        return next(responses)

    client = object.__new__(GeminiLabClient)
    client.model_id = MODEL_ID
    client.client = SimpleNamespace(
        interactions=SimpleNamespace(create=create)
    )
    media = MediaInfo(
        path=str(tmp_path / "source.mp4"),
        sha256="a" * 64,
        asset_id="sha256:" + "a" * 64,
        format_name="mp4",
        duration_ms=10_000,
        size_bytes=1,
        format_metadata={},
        video=VideoStreamInfo(
            index=0,
            codec_name="h264",
            coded_width=1920,
            coded_height=1080,
            display_width=1920,
            display_height=1080,
            rotation_degrees=0,
            average_frame_rate=Rational(numerator=30, denominator=1),
            real_frame_rate=Rational(numerator=30, denominator=1),
            time_base=Rational(numerator=1, denominator=30),
            start_pts=0,
            duration_ts=300,
            metadata={},
        ),
    )

    result = client.analyze_video(
        media=media,
        uploaded=SimpleNamespace(
            uri="files/source",
            mime_type="video/mp4",
        ),
        prompt_template="Analyze visible evidence.",
        run_id="text-repair",
        run_dir=tmp_path / "run",
        repair_attempts=3,
    )

    assert result.model_provenance.interaction_id == "repaired"
    assert len(requests) == 2
    assert requests[0]["input"][0]["type"] == "video"
    assert requests[0]["input"][0]["media_resolution"] == "low"
    assert [item["type"] for item in requests[1]["input"]] == ["text"]
    assert requests[1]["generation_config"]["thinking_level"] == "minimal"


def test_full_clip_card_normalization_clamps_half_open_keyframe_endpoint() -> None:
    raw = json.dumps(
        {
            "events": [
                {
                    "event_id": "pose",
                    "start_mmss": "00:00",
                    "end_mmss": "00:02",
                    "recommended_keyframe_mmss": "00:02",
                },
                {
                    "event_id": "hold",
                    "start_mmss": "00:05",
                    "end_mmss": "00:06",
                    "recommended_keyframe_mmss": "00:06",
                },
            ]
        }
    )

    canonical, changes = normalize_full_clip_card_output_text(raw)
    events = json.loads(canonical)["events"]

    assert [item["recommended_keyframe_mmss"] for item in events] == [
        "00:01",
        "00:05",
    ]
    assert [item["reason"] for item in changes] == [
        "half_open_event_interval_clamp",
        "half_open_event_interval_clamp",
    ]
