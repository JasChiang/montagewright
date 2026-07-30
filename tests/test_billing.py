from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jascue_video_lab.billing import (
    BudgetExceeded,
    BudgetLedger,
    PaidDispatchAlreadyRecorded,
    adopt_paid_dispatch_journals,
    dispatch_paid_interaction,
    estimate_paid_call,
    summarize_usage_and_list_price,
)


def _cheap_estimate(stage: str):
    return estimate_paid_call(
        stage=stage,
        model_id="gemini-3.6-flash",
        text_input_tokens=1,
        max_output_tokens=1,
        thinking_level="minimal",
    )


def test_default_budget_hold_preserves_and_automatically_consumes_final_qa_slot(
) -> None:
    ledger = BudgetLedger(max_cost_usd=10.0, max_interactions=3)

    ledger.reserve(_cheap_estimate("candidate_reel_plan"))
    ledger.reserve(_cheap_estimate("exact_event_group"))
    with pytest.raises(BudgetExceeded, match="interaction reserve"):
        ledger.reserve(_cheap_estimate("multi_target_grounding"))

    # The typed QA stage consumes the default final_qa hold even when the
    # caller does not opt into the legacy recovery_call escape hatch.
    ledger.reserve(_cheap_estimate("final_qa:autonomous_final_9x16"))

    report = ledger.report()
    assert report["mandatory_stage_minimums"] == {"final_qa": 1}
    assert report["remaining_held_interactions"] == 0
    assert report["available_unheld_interactions"] == 0
    assert report["committed_interactions"] == 3


def test_multiple_mandatory_stage_holds_cannot_be_spent_by_other_calls() -> None:
    ledger = BudgetLedger(
        max_cost_usd=10.0,
        max_interactions=6,
        mandatory_stage_minimums={
            "exact_event_group": 2,
            "final_qa": 1,
        },
    )

    for _ in range(3):
        ledger.reserve(_cheap_estimate("candidate_reel_plan"))
    with pytest.raises(BudgetExceeded, match="interaction reserve"):
        ledger.reserve(_cheap_estimate("multi_target_grounding"))

    ledger.reserve(_cheap_estimate("exact_event_group"))
    ledger.reserve(_cheap_estimate("final_qa:autonomous_final_9x16"))
    report = ledger.report()
    assert report["remaining_held_interactions"] == 1
    assert report["mandatory_interaction_holds"]["exact_event_group"] == {
        "minimum_interactions": 2,
        "committed_interactions": 1,
        "remaining_held_interactions": 1,
    }

    ledger.reserve(_cheap_estimate("exact_event_group"))
    assert ledger.report()["remaining_held_interactions"] == 0


def test_cancelling_mandatory_reservation_restores_its_stage_hold() -> None:
    ledger = BudgetLedger(
        max_cost_usd=10.0,
        max_interactions=2,
        mandatory_stage_minimums={"exact_event_group": 1},
    )
    mandatory = ledger.reserve(_cheap_estimate("exact_event_group"))
    ledger.cancel_before_dispatch(mandatory.reservation_id)

    report = ledger.report()
    assert report["committed_interactions"] == 0
    assert report["remaining_held_interactions"] == 1
    ledger.reserve(_cheap_estimate("candidate_reel_plan"))
    with pytest.raises(BudgetExceeded, match="interaction reserve"):
        ledger.reserve(_cheap_estimate("multi_target_grounding"))


def test_resumed_and_conservatively_adopted_calls_consume_stage_holds() -> None:
    ledger = BudgetLedger(
        max_cost_usd=10.0,
        max_interactions=3,
        mandatory_stage_minimums={
            "exact_event_group": 1,
            "final_qa": 1,
        },
    )
    ledger.adopt_reconciled_usage(
        stage="final_qa:autonomous_final_9x16",
        model_id="gemini-3.6-flash",
        usage={
            "total_input_tokens": 10,
            "total_cached_tokens": 0,
            "total_output_tokens": 1,
            "total_thought_tokens": 0,
        },
    )
    adopted = ledger.adopt_conservative_dispatch(
        dispatch_id="dispatch-exact-event",
        estimate=_cheap_estimate("exact_event_group"),
    )
    assert (
        ledger.adopt_conservative_dispatch(
            dispatch_id="dispatch-exact-event",
            estimate=_cheap_estimate("exact_event_group"),
        )
        is adopted
    )

    report = ledger.report()
    assert report["committed_interactions"] == 2
    assert report["remaining_held_interactions"] == 0
    assert report["available_unheld_interactions"] == 1


def test_resume_cannot_adopt_nonmandatory_work_into_a_mandatory_hold() -> None:
    ledger = BudgetLedger(max_cost_usd=10.0, max_interactions=2)
    usage = {
        "total_input_tokens": 10,
        "total_cached_tokens": 0,
        "total_output_tokens": 1,
        "total_thought_tokens": 0,
    }
    ledger.adopt_reconciled_usage(
        stage="candidate_reel_plan",
        model_id="gemini-3.6-flash",
        usage=usage,
    )

    with pytest.raises(BudgetExceeded, match="mandatory future"):
        ledger.adopt_reconciled_usage(
            stage="multi_target_grounding",
            model_id="gemini-3.6-flash",
            usage=usage,
        )

    assert ledger.committed_interactions == 1
    assert ledger.report()["remaining_held_interactions"] == 1


def test_usage_summary_prices_input_output_and_thought_tokens(tmp_path) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 1_000,
            "total_output_tokens": 100,
            "total_thought_tokens": 20,
            "input_tokens_by_modality": [
                {"modality": "VIDEO", "tokens": 700},
                {"modality": "TEXT", "tokens": 300},
            ],
        }
    }
    path = tmp_path / "test.raw_interaction.json"
    path.write_text(json.dumps(interaction), encoding="utf-8")
    summary = summarize_usage_and_list_price(tmp_path)
    assert summary["request_count"] == 1
    assert summary["input_tokens_by_modality"] == {"TEXT": 300, "VIDEO": 700}
    assert summary["billed_output_tokens"] == 120
    assert summary["model"] == "gemini-3.6-flash"
    assert summary["estimated_total_cost_usd"] == 0.0024


def test_usage_summary_deduplicates_copied_raw_interactions(tmp_path) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 1_000,
            "total_output_tokens": 100,
            "total_thought_tokens": 20,
        }
    }
    for name in ("attempt.raw_interaction.json", "canonical.raw_interaction.json"):
        (tmp_path / name).write_text(json.dumps(interaction), encoding="utf-8")

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 1
    assert summary["duplicate_artifact_count"] == 1
    assert summary["estimated_total_cost_usd"] == 0.0024


def test_usage_summary_counts_identical_immutable_attempts_separately(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 1_000,
            "total_output_tokens": 100,
            "total_thought_tokens": 20,
        },
    }
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    for name in ("first.raw_interaction.json", "second.raw_interaction.json"):
        (attempts / name).write_text(json.dumps(interaction), encoding="utf-8")
    (tmp_path / "canonical.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1
    assert summary["estimated_total_cost_usd"] == 0.0048


def test_usage_summary_deduplicates_copied_immutable_attempt_uuid(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    attempt_name = (
        "grounding.unknown."
        "0123456789abcdef0123456789abcdef.raw_interaction.json"
    )
    for branch in ("original", "trim-recompile"):
        directory = tmp_path / branch / "attempts"
        directory.mkdir(parents=True)
        (directory / attempt_name).write_text(
            json.dumps(interaction),
            encoding="utf-8",
        )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 1
    assert summary["duplicate_artifact_count"] == 1


def test_usage_summary_counts_identical_attempts_in_nested_attempts_tree(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    for branch in ("first", "second"):
        directory = tmp_path / "attempts" / "nested" / branch
        directory.mkdir(parents=True)
        (directory / "response.raw_interaction.json").write_text(
            json.dumps(interaction), encoding="utf-8"
        )
    (tmp_path / "canonical.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1
    assert summary["duplicate_artifact_paths"] == [
        "canonical.raw_interaction.json"
    ]


def test_usage_summary_counts_identical_numbered_attempt_directories(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    for attempt in ("attempt-1", "attempt-02"):
        directory = tmp_path / "variant" / attempt
        directory.mkdir(parents=True)
        (directory / "response.raw_interaction.json").write_text(
            json.dumps(interaction), encoding="utf-8"
        )
    (tmp_path / "canonical.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1


def test_usage_summary_counts_identical_legacy_attempt_filenames(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    for name in (
        "content_map.attempt-1.raw_interaction.json",
        "content_map.attempt-02.raw_interaction.json",
    ):
        (tmp_path / name).write_text(json.dumps(interaction), encoding="utf-8")
    (tmp_path / "content_map.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1
    assert summary["duplicate_artifact_paths"] == [
        "content_map.raw_interaction.json"
    ]


def test_usage_summary_prices_mixed_models_per_response(tmp_path) -> None:
    for name, model in (
        ("old.raw_interaction.json", "gemini-3.5-flash"),
        ("new.raw_interaction.json", "gemini-3.6-flash"),
    ):
        (tmp_path / name).write_text(
            json.dumps(
                {
                    "model": model,
                    "usage": {
                        "total_input_tokens": 1_000,
                        "total_output_tokens": 100,
                        "total_thought_tokens": 20,
                    },
                }
            ),
            encoding="utf-8",
        )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["model"] == "mixed"
    assert summary["estimated_total_cost_usd"] == 0.00498
    assert summary["models"]["gemini-3.5-flash"]["estimated_total_cost_usd"] == 0.00258
    assert summary["models"]["gemini-3.6-flash"]["estimated_total_cost_usd"] == 0.0024


def test_usage_summary_applies_cached_input_discount(tmp_path) -> None:
    (tmp_path / "cached.raw_interaction.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.6-flash",
                "usage": {
                    "total_input_tokens": 1_000,
                    "total_cached_tokens": 800,
                    "total_output_tokens": 0,
                    "total_thought_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["total_cached_input_tokens"] == 800
    assert summary["total_uncached_input_tokens"] == 200
    assert summary["estimated_total_cost_usd"] == 0.00042


def test_usage_summary_refuses_unpriced_or_missing_model(tmp_path) -> None:
    (tmp_path / "unknown.raw_interaction.json").write_text(
        json.dumps({"model": "unknown-model", "usage": {"total_input_tokens": 1}}),
        encoding="utf-8",
    )

    try:
        summarize_usage_and_list_price(tmp_path)
    except ValueError as error:
        assert "no Standard pricing is registered" in str(error)
    else:
        raise AssertionError("unknown model must fail closed")


def test_missing_usage_is_reported_as_unpriced_not_silently_free(tmp_path) -> None:
    path = tmp_path / "attempts" / "attempt-000001" / "raw_interaction.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"id": "response-without-usage", "model": "gemini-3.6-flash"}
        ),
        encoding="utf-8",
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 1
    assert summary["priced_request_count"] == 0
    assert summary["unpriced_request_count"] == 1
    assert summary["pricing_complete"] is False
    assert summary["cost_interpretation"] == "lower_bound_incomplete_usage_metadata"
    assert summary["unpriced_request_paths"] == [
        "attempts/attempt-000001/raw_interaction.json"
    ]


def test_ambiguous_dispatch_is_adopted_once_and_exact_request_never_replayed(
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    def fail_create(**request):
        calls.append(request)
        raise RuntimeError("503 transport unavailable")

    client = SimpleNamespace(
        interactions=SimpleNamespace(create=fail_create)
    )
    estimate = estimate_paid_call(
        stage="exact_event_group",
        model_id="gemini-3.6-flash",
        image_count=8,
        media_resolution="high",
        text_input_tokens=200,
        max_output_tokens=2048,
        thinking_level="low",
    )
    first_ledger = BudgetLedger(
        max_cost_usd=1.25,
        max_interactions=25,
    )
    request = {"model": "gemini-3.6-flash", "input": [{"type": "text", "text": "x"}]}

    with pytest.raises(RuntimeError, match="503"):
        dispatch_paid_interaction(
            client=client,
            request=request,
            request_record=request,
            journal_dir=tmp_path / "picture" / "exact",
            estimate=estimate,
            budget_ledger=first_ledger,
        )

    assert len(calls) == 1
    assert first_ledger.committed_interactions == 1
    assert first_ledger.actual_cost_usd == estimate.worst_case_cost_usd
    resumed_ledger = BudgetLedger(
        max_cost_usd=1.25,
        max_interactions=25,
    )
    adopted = adopt_paid_dispatch_journals(
        budget_ledger=resumed_ledger,
        root=tmp_path,
        allowed_top_level={"picture"},
    )
    assert len(adopted) == 1
    assert resumed_ledger.committed_interactions == 1
    assert resumed_ledger.actual_cost_usd == estimate.worst_case_cost_usd

    with pytest.raises(PaidDispatchAlreadyRecorded):
        dispatch_paid_interaction(
            client=client,
            request=request,
            request_record=request,
            journal_dir=tmp_path / "picture" / "exact",
            estimate=estimate,
            budget_ledger=resumed_ledger,
        )

    assert len(calls) == 1
    assert resumed_ledger.committed_interactions == 1
