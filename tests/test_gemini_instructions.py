from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import jascue_video_lab.gemini as gemini_module
from jascue_video_lab.billing import summarize_usage_and_list_price

from jascue_video_lab.gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    GroundingIdentityReference,
    MODEL_ID,
    SEMANTIC_IDENTITY_GENERATION_CONFIG,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    GeminiLabClient,
    canonicalize_feature_edit_plan_output,
)
from jascue_video_lab.models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    ModelProvenance,
    RushClip,
    RushFrame,
    RushesCatalog,
)


class _StopRequest(RuntimeError):
    pass


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
