from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from montagewright.cost import Ledger
from montagewright.cost import BudgetSpent
from montagewright.fal_openrouter import FalOpenRouterClient, _content
from montagewright.gemini import structured_json, video_content
from montagewright.planner import ask
from montagewright.uploads import UploadCache


def test_fal_request_maps_auth_schema_and_checkpoints_paid_response(monkeypatch, tmp_path):
    sent = []

    def fake_urlopen(request, *, timeout):
        sent.append((request, timeout))
        return io.BytesIO(json.dumps({
            "id": "fal-generation-1",
            "choices": [{"message": {"content": '{"ok":true}', "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}]}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.002},
        }).encode())

    monkeypatch.setattr("montagewright.fal_openrouter.urlopen", fake_urlopen)
    client = FalOpenRouterClient("test-key")
    journal = tmp_path / "spend.jsonl"
    ledger = Ledger(cap_usd=1, journal_path=journal)
    request = dict(
        model="gemini-3.8-flash", store=False, input="hello",
        generation_config={"max_output_tokens": 32},
        response_format=structured_json({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
    )
    first = ask(client, ledger=ledger, budget_stage="smoke", **request)
    second = ask(client, ledger=ledger, budget_stage="smoke", **request)

    assert first.output_text == second.output_text == '{"ok":true}'
    assert len(sent) == 1
    http_request, timeout = sent[0]
    assert http_request.full_url.endswith("/openrouter/router/openai/v1/chat/completions")
    assert http_request.get_header("Authorization") == "Key test-key"
    body = json.loads(http_request.data)
    assert body["model"] == "google/gemini-3.8-flash"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["response_format"]["json_schema"]["schema"] == request["response_format"]["schema"]
    assert body["max_tokens"] == 32
    assert ledger.spent_usd == pytest.approx(0.002)
    saved = list((tmp_path / "work" / "responses").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["provider"] == "fal_openrouter"
    assert json.loads(saved[0].read_text())["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "opaque"}]


def test_fal_media_is_read_at_dispatch_and_cache_reuses_exact_local_file(tmp_path):
    picture = tmp_path / "產品.jpg"
    picture.write_bytes(b"image bytes")
    client = FalOpenRouterClient("test-key")
    cache = UploadCache.load(tmp_path / "uploads.json")
    uri, hit = cache.uri_for(picture, client, mime_type="image/jpeg")
    again, second_hit = cache.uri_for(picture, client, mime_type="image/jpeg")
    assert not hit and second_hit and uri == again
    assert uri == picture.as_uri()
    assert _content([{"type": "image", "mime_type": "image/jpeg", "uri": uri}]) == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aW1hZ2UgYnl0ZXM="}}
    ]
    picture.write_bytes(b"changed")
    with pytest.raises(FileNotFoundError):
        _content([{"type": "image", "mime_type": "image/jpeg", "uri": picture.with_name("gone.jpg").as_uri()}])


def test_video_maps_processing_to_openrouter_video_url(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video bytes")
    part = video_content(video.as_uri(), processing="agentic")
    mapped = _content([part])
    assert mapped[0]["type"] == "video_url"
    assert mapped[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert mapped[0]["video_url"]["processing"] == "agentic"
    static = _content([video_content(video.as_uri(), processing={"type": "static", "fps": 4})])
    assert static[0]["video_url"]["processing"] == "static"
    assert "fps" not in static[0]["video_url"]
    with pytest.raises(ValueError, match="offsets"):
        _content([video_content(video.as_uri(), processing={
            "type": "static", "fps": 4, "start_offset": "1s",
        })])


def test_fal_media_budget_stops_before_dispatch(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video bytes")
    monkeypatch.setattr("montagewright.fal_openrouter.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="10.0\n"))
    monkeypatch.setattr("montagewright.fal_openrouter.urlopen", lambda *a, **k: pytest.fail("paid request was sent"))
    client = FalOpenRouterClient("test-key")
    ledger = Ledger(cap_usd=0.001, journal_path=tmp_path / "spend.jsonl")
    with pytest.raises(BudgetSpent):
        ask(client, ledger=ledger, budget_stage="clip_cards", model="gemini-3.8-flash",
            input=[video_content(video.as_uri())],
            generation_config={"max_output_tokens": 64})


def test_fal_checkpoint_key_tracks_current_media_bytes(tmp_path):
    from montagewright.checkpoints import response_path

    image = tmp_path / "still.jpg"
    image.write_bytes(b"first")
    ledger = Ledger(cap_usd=1, journal_path=tmp_path / "spend.jsonl")
    request = {"model": "gemini-3.8-flash", "input": [{
        "type": "image", "uri": image.as_uri(), "mime_type": "image/jpeg",
    }]}
    first = response_path(ledger, "reference", request, provider="fal_openrouter")
    image.write_bytes(b"second")
    second = response_path(ledger, "reference", request, provider="fal_openrouter")
    direct = response_path(ledger, "reference", request)
    assert first != second != direct


def test_oversize_inline_request_stops_before_network(monkeypatch):
    monkeypatch.setattr(
        "montagewright.fal_openrouter.urlopen",
        lambda *a, **k: pytest.fail("oversize paid request was sent"),
    )
    client = FalOpenRouterClient("test-key")
    with pytest.raises(ValueError, match="inline request"):
        client.interactions.create(
            model="gemini-3.8-flash", input="x" * 19_000_000,
        )
