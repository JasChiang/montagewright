from __future__ import annotations

from pathlib import Path

from jascue_video_lab.delivery_pipeline import _write_status
from jascue_video_lab.models import (
    FeatureCutExecutionProfile,
    FeatureCutRunState,
)
from jascue_video_lab.storage import read_json


def test_legacy_execution_profiles_remain_the_baseline() -> None:
    values = [profile.value for profile in FeatureCutExecutionProfile]
    assert values[:2] == [
        "review_preview",
        "production_review",
    ]
    assert FeatureCutRunState.READY_FOR_HUMAN_REVIEW.value == (
        "ready_for_human_review"
    )
    assert FeatureCutRunState.DELIVERY_ELIGIBLE.value == "delivery_eligible"


def test_legacy_delivery_status_never_claims_automatic_eligibility(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-status.json"
    _write_status(
        path,
        stage="completed",
        terminal=True,
        state="ready_for_human_review",
    )

    status = read_json(path)

    assert status["state"] == "ready_for_human_review"
    assert status["delivery_eligible"] is False
