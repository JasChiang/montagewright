from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import jascue_video_lab.final_edit_qa as final_edit_qa_module
from jascue_video_lab.final_edit_qa import (
    CanonicalSegmentQa,
    execute_final_edit_qa,
    load_cached_final_edit_qa,
    prepare_final_edit_qa,
)
from jascue_video_lab.storage import read_json, write_json


def _write_video(path: Path, *, vertical: bool, audio: bool) -> None:
    size = "180x320" if vertical else "320x180"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s={size}:r=24:d=2",
    ]
    if audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=2",
                "-shortest",
            ]
        )
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if audio:
        command.extend(["-c:a", "aac"])
    else:
        command.append("-an")
    command.append(str(path))
    subprocess.run(command, check=True)


def _finding(
    observation: str,
    *,
    modality: str = "visual",
) -> dict[str, Any]:
    return {
        "assessment": "acceptable",
        "observation": observation,
        "evidence_modality": modality,
        "correction_suggestion": None,
    }


class _FakeInteraction:
    def __init__(self, output: dict[str, Any], *, model: str) -> None:
        self.output_text = json.dumps(output, ensure_ascii=False)
        self.id = "interaction-test"
        self._model = model

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self._model,
            "output_text": self.output_text,
            "usage": {
                "total_input_tokens": 2_000,
                "total_output_tokens": 500,
                "total_thought_tokens": 100,
                "total_cached_tokens": 0,
                "input_tokens_by_modality": [
                    {"modality": "VIDEO", "tokens": 1_800},
                    {"modality": "TEXT", "tokens": 200},
                ],
            },
        }


class _FakeInteractions:
    def __init__(self, interaction: _FakeInteraction) -> None:
        self.interaction = interaction
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> _FakeInteraction:
        self.requests.append(request)
        return self.interaction


class _FakeClient:
    def __init__(self, interaction: _FakeInteraction) -> None:
        self.interactions = _FakeInteractions(interaction)


def _canonical_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    render = tmp_path / "canonical.mp4"
    _write_video(render, vertical=False, audio=True)
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "horizontal": {
                "chapters": [
                    {
                        "segment_id": "opening",
                        "feature_id": "brief-opening",
                        "expected_semantics": "Introduce the subject and result.",
                        "duration_ms": 1_000,
                    },
                    {
                        "segment_id": "demonstration",
                        "feature_id": "brief-demo",
                        "expected_semantics": "Show a complete generic action.",
                        "duration_ms": 1_000,
                    },
                ]
            }
        },
    )
    brief = tmp_path / "brief.json"
    write_json(
        brief,
        {
            "title": "Generic demonstration",
            "items": [
                {"id": "brief-opening", "intent": "Introduce the subject"},
                {"id": "brief-demo", "intent": "Show the result"},
            ],
        },
    )
    return render, manifest, brief


def _canonical_output(prepared: Any) -> dict[str, Any]:
    segments = []
    for item in prepared.segment_contract:
        segments.append(
            {
                "order": item["order"],
                "segment_id": item["segment_id"],
                "brief_item_id": item["brief_item_id"],
                "brief_delivery": _finding("The intended idea is visibly present."),
                "action_completeness": _finding(
                    "The action reaches a visible result."
                ),
                "dwell_quality": _finding(
                    "The action and result have enough observable breathing room."
                ),
                "transition_quality": _finding(
                    "The relation to the adjacent unit remains understandable.",
                    modality="sequence",
                ),
                "repetition_relation": _finding(
                    "Any repeated view adds complementary information.",
                    modality="sequence",
                ),
                "music_relationship": _finding(
                    "The audible music supports the visible emphasis.",
                    modality="visual_and_audio",
                ),
                "segment_summary": "Usable observation for human review.",
            }
        )
    hashes = prepared.input_hashes
    return {
        "contract_version": "final-edit-qa-v1",
        "mode": "canonical_16x9",
        "render_sha256": hashes["render_sha256"],
        "proxy_sha256": hashes["proxy_sha256"],
        "manifest_sha256": hashes["manifest_sha256"],
        "brief_sha256": hashes["brief_sha256"],
        "segments": segments,
        "global_review": {
            "hook": _finding("The opening establishes visible interest."),
            "pacing": _finding(
                "Information and action progress with observable variation.",
                modality="sequence",
            ),
            "music_flow": _finding(
                "The music remains continuous and supports the sequence.",
                modality="visual_and_audio",
            ),
            "ending": _finding(
                "The final picture and audible ending resolve together.",
                modality="visual_and_audio",
            ),
            "disposition": "ready_for_human_review",
            "priority_corrections": [],
        },
        "limitations": ["This remains a sampled model observation."],
        "requires_human_review": True,
    }


def test_canonical_final_qa_is_one_read_only_structured_request(
    tmp_path: Path,
) -> None:
    render, manifest, brief = _canonical_files(tmp_path)
    output_dir = tmp_path / "qa"
    prepared = prepare_final_edit_qa(
        mode="canonical_16x9",
        render_path=render,
        manifest_path=manifest,
        brief_path=brief,
        output_dir=output_dir,
        model_id="gemini-3.6-flash",
    )
    interaction = _FakeInteraction(
        _canonical_output(prepared),
        model="gemini-3.6-flash",
    )
    client = _FakeClient(interaction)

    execution = execute_final_edit_qa(
        prepared=prepared,
        client=client,
        uploaded_video={"uri": "files/final-video", "mime_type": "video/mp4"},
        output_dir=output_dir,
    )

    assert len(client.interactions.requests) == 1
    request = client.interactions.requests[0]
    assert request["store"] is False
    assert request["generation_config"].get("temperature") is None
    assert request["input"][1]["type"] == "video"
    schema_text = json.dumps(request["response_format"]["schema"])
    for field in (
        "brief_delivery",
        "action_completeness",
        "dwell_quality",
        "transition_quality",
        "repetition_relation",
        "music_relationship",
        "hook",
        "pacing",
        "music_flow",
        "ending",
    ):
        assert field in schema_text
    assert execution.result.requires_human_review is True
    assert (execution.attempt_dir / "raw_interaction.json").exists()
    assert read_json(execution.attempt_dir / "schema_validation.json")["ok"] is True
    assert read_json(execution.run_dir / "input_hashes.json") == prepared.input_hashes
    pricing = read_json(execution.run_dir / "pricing.json")
    assert pricing["request_count"] == 1
    assert pricing["estimated_total_cost_usd"] > 0

    cached = load_cached_final_edit_qa(prepared, output_dir=output_dir)
    assert cached is not None
    assert cached.cache_hit is True


def test_human_review_gate_is_application_owned_without_paid_retry(
    tmp_path: Path,
) -> None:
    render, manifest, brief = _canonical_files(tmp_path)
    output_dir = tmp_path / "qa"
    prepared = prepare_final_edit_qa(
        mode="canonical_16x9",
        render_path=render,
        manifest_path=manifest,
        brief_path=brief,
        output_dir=output_dir,
        model_id="gemini-3.6-flash",
    )
    output = _canonical_output(prepared)
    output["requires_human_review"] = False
    client = _FakeClient(
        _FakeInteraction(output, model="gemini-3.6-flash")
    )

    execution = execute_final_edit_qa(
        prepared=prepared,
        client=client,
        uploaded_video={"uri": "files/final-video", "mime_type": "video/mp4"},
        output_dir=output_dir,
    )

    assert len(client.interactions.requests) == 1
    assert execution.result.requires_human_review is True
    normalization = read_json(
        execution.attempt_dir / "contract_normalization.json"
    )
    policy = normalization["application_owned_fields"]["requires_human_review"]
    assert policy["model_value"] is False
    assert policy["normalized_value"] is True
    raw_output = read_json(execution.attempt_dir / "raw_output.json")
    assert '"requires_human_review": false' in raw_output["output_text"]


def test_validator_only_bump_reuses_paid_raw_output_without_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render, manifest, brief = _canonical_files(tmp_path)
    output_dir = tmp_path / "qa"
    original = prepare_final_edit_qa(
        mode="canonical_16x9",
        render_path=render,
        manifest_path=manifest,
        brief_path=brief,
        output_dir=output_dir,
        model_id="gemini-3.6-flash",
    )
    paid_client = _FakeClient(
        _FakeInteraction(
            _canonical_output(original),
            model="gemini-3.6-flash",
        )
    )
    paid = execute_final_edit_qa(
        prepared=original,
        client=paid_client,
        uploaded_video={"uri": "files/final-video", "mime_type": "video/mp4"},
        output_dir=output_dir,
    )
    assert len(paid_client.interactions.requests) == 1

    monkeypatch.setattr(
        final_edit_qa_module,
        "FINAL_EDIT_QA_VALIDATOR_VERSION",
        "final-edit-qa-validator-v2-test",
    )
    revised = prepare_final_edit_qa(
        mode="canonical_16x9",
        render_path=render,
        manifest_path=manifest,
        brief_path=brief,
        output_dir=output_dir,
        model_id="gemini-3.6-flash",
    )
    assert revised.cache_key == original.cache_key
    assert (
        revised.input_hashes["validator_version"]
        != original.input_hashes["validator_version"]
    )
    forbidden_client = _FakeClient(
        _FakeInteraction({}, model="gemini-3.6-flash")
    )

    recovered = execute_final_edit_qa(
        prepared=revised,
        client=forbidden_client,
        uploaded_video={"uri": "files/unused", "mime_type": "video/mp4"},
        output_dir=output_dir,
    )

    assert recovered.cache_hit is True
    assert recovered.run_dir == paid.run_dir
    assert forbidden_client.interactions.requests == []
    validation = read_json(recovered.run_dir / "schema_validation.json")
    assert validation["validator_version"] == "final-edit-qa-validator-v2-test"
    assert validation["additional_api_request_count"] == 0


def test_dwell_quality_contract_rejects_fixed_second_rules() -> None:
    payload = {
        "order": 1,
        "segment_id": "segment-001",
        "brief_item_id": "brief-001",
        "brief_delivery": _finding("Visible."),
        "action_completeness": _finding("Complete."),
        "dwell_quality": _finding("Every shot should always last 3 seconds."),
        "transition_quality": _finding("Visible."),
        "repetition_relation": _finding("Complementary."),
        "music_relationship": _finding("Audible.", modality="audio"),
        "segment_summary": "Summary.",
    }
    with pytest.raises(ValidationError, match="fixed-seconds"):
        CanonicalSegmentQa.model_validate(payload)


def test_crop_only_mode_accepts_silent_video_and_excludes_editorial_music_fields(
    tmp_path: Path,
) -> None:
    render = tmp_path / "vertical.mp4"
    _write_video(render, vertical=True, audio=False)
    manifest = tmp_path / "vertical-manifest.json"
    write_json(
        manifest,
        {
            "vertical": {
                "chapters": [
                    {
                        "segment_id": "crop-a",
                        "feature_id": "unit-a",
                        "target_description": "The main visible subject",
                        "important_text": "Required visible heading",
                        "applied_strategy": "tracked_crop",
                        "duration_ms": 2_000,
                    }
                ]
            }
        },
    )
    output_dir = tmp_path / "crop-qa"
    prepared = prepare_final_edit_qa(
        mode="crop_only_9x16",
        render_path=render,
        manifest_path=manifest,
        output_dir=output_dir,
        model_id="gemini-3.6-flash",
    )
    schema_text = json.dumps(prepared.schema)
    for field in (
        "crop_containment",
        "subject_visibility",
        "text_integrity",
        "tracking_stability",
    ):
        assert field in schema_text
    for excluded in (
        "brief_delivery",
        "dwell_quality",
        "music_relationship",
        "music_flow",
    ):
        assert excluded not in schema_text
    assert prepared.input_hashes["brief_sha256"] is None
    assert prepared.input_hashes["proxy_contract"]["audio"] == "omitted"
    assert prepared.segment_contract[0]["tracking_expected"] is True
    hashes = prepared.input_hashes
    crop_output = {
        "contract_version": "final-edit-qa-v1",
        "mode": "crop_only_9x16",
        "render_sha256": hashes["render_sha256"],
        "proxy_sha256": hashes["proxy_sha256"],
        "manifest_sha256": hashes["manifest_sha256"],
        "segments": [
            {
                "order": 1,
                "segment_id": "crop-a",
                "crop_containment": _finding("The required subject remains contained."),
                "subject_visibility": _finding("The subject stays identifiable."),
                "text_integrity": _finding("The required heading remains visible."),
                "tracking_stability": _finding(
                    "The dynamic crop follows without an observable jump.",
                    modality="sequence",
                ),
                "segment_summary": "Crop-only evidence is available for review.",
            }
        ],
        "global_review": {
            "overall_crop_quality": _finding(
                "The crop preserves the required visual regions.",
                modality="sequence",
            ),
            "disposition": "ready_for_human_review",
            "priority_corrections": [],
        },
        "limitations": ["Audio was intentionally excluded and not evaluated."],
        "requires_human_review": True,
    }
    client = _FakeClient(
        _FakeInteraction(crop_output, model="gemini-3.6-flash")
    )
    execution = execute_final_edit_qa(
        prepared=prepared,
        client=client,
        uploaded_video={"uri": "files/crop-video", "mime_type": "video/mp4"},
        output_dir=output_dir,
    )
    assert execution.result.mode == "crop_only_9x16"
    assert len(client.interactions.requests) == 1


def test_canonical_mode_rejects_a_silent_render(tmp_path: Path) -> None:
    render = tmp_path / "silent-horizontal.mp4"
    _write_video(render, vertical=False, audio=False)
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {"horizontal": {"chapters": [{"segment_id": "one"}]}},
    )
    brief = tmp_path / "brief.json"
    write_json(brief, {"intent": "generic"})

    with pytest.raises(ValueError, match="requires the final render with its music"):
        prepare_final_edit_qa(
            mode="canonical_16x9",
            render_path=render,
            manifest_path=manifest,
            brief_path=brief,
            output_dir=tmp_path / "qa",
            model_id="gemini-3.6-flash",
        )


def test_invalid_model_output_preserves_raw_usage_cost_and_validation(
    tmp_path: Path,
) -> None:
    render, manifest, brief = _canonical_files(tmp_path)
    output_dir = tmp_path / "qa-invalid"
    prepared = prepare_final_edit_qa(
        mode="canonical_16x9",
        render_path=render,
        manifest_path=manifest,
        brief_path=brief,
        output_dir=output_dir,
        model_id="gemini-3.6-flash",
    )
    invalid = _canonical_output(prepared)
    invalid["render_sha256"] = "0" * 64
    client = _FakeClient(
        _FakeInteraction(invalid, model="gemini-3.6-flash")
    )

    with pytest.raises(ValueError, match="render hash"):
        execute_final_edit_qa(
            prepared=prepared,
            client=client,
            uploaded_video={"uri": "files/invalid", "mime_type": "video/mp4"},
            output_dir=output_dir,
        )

    run_dir = output_dir / "runs" / prepared.cache_key
    attempt_dir = sorted((run_dir / "attempts").glob("attempt-*"))[-1]
    assert (attempt_dir / "raw_interaction.json").exists()
    assert (attempt_dir / "raw_output.json").exists()
    assert (attempt_dir / "usage.json").exists()
    assert read_json(attempt_dir / "schema_validation.json")["ok"] is False
    assert read_json(attempt_dir / "pricing.json")["request_count"] == 1
    assert read_json(run_dir / "pricing.json")["estimated_total_cost_usd"] > 0
