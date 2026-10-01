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
    # Agentic on this route either 503s or answers empty with zero input
    # tokens (measured 2026-09-30), so it is sent as static.
    assert mapped[0]["video_url"]["processing"] == "static"
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
            model="gemini-3.8-flash", input="x" * 20_000_000,
        )


def test_a_video_answer_with_zero_input_tokens_is_refused(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video bytes")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *a):
            return json.dumps({
                "choices": [{"message": {"content": "{\"words\": []}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 9},
            }).encode()

    monkeypatch.setattr(
        "montagewright.fal_openrouter.urlopen", lambda *a, **k: _Response()
    )
    client = FalOpenRouterClient("test-key")
    with pytest.raises(RuntimeError, match="zero input tokens"):
        client.interactions.create(
            model="gemini-3.8-flash",
            input=[video_content(video.as_uri(), processing="static"),
                   {"type": "text", "text": "what is shown"}],
        )


def test_an_oversized_video_is_reencoded_on_the_same_timeline(tmp_path, monkeypatch):
    import subprocess as sp

    from montagewright import fal_openrouter as fo

    video = tmp_path / "long.mp4"
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=s=1280x720:r=30:d=12,noise=alls=60:allf=t",
            "-c:v", "libx264", "-b:v", "3M", str(video)], check=True)
    monkeypatch.setattr(fo, "FAL_VIDEO_MAX_BYTES", 2_000_000)
    monkeypatch.setattr(fo.tempfile, "gettempdir", lambda: str(tmp_path))
    sized = fo._fal_sized(video)
    assert sized != video and sized.stat().st_size <= 2_000_000
    assert abs(fo._duration_seconds(sized) - fo._duration_seconds(video)) < 0.2


def test_many_videos_that_sum_past_the_body_are_all_shrunk_not_dropped(
    tmp_path, monkeypatch,
):
    import subprocess as sp

    from montagewright import fal_openrouter as fo

    videos = []
    for index in range(4):
        video = tmp_path / f"take{index}.mp4"
        sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f",
                "lavfi", "-i",
                f"testsrc2=s=640x360:r=30:d=6,noise=alls={40 + index}:allf=t",
                "-c:v", "libx264", "-b:v", "1500k", str(video)], check=True)
        videos.append(video)
    total = sum(one.stat().st_size for one in videos)
    monkeypatch.setattr(fo, "FAL_REQUEST_MAX_BYTES", int(total * 1.34 * 0.6))
    monkeypatch.setattr(fo.tempfile, "gettempdir", lambda: str(tmp_path))
    sent = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *a):
            return json.dumps({
                "choices": [{"message": {"content": "{}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2},
            }).encode()

    def _urlopen(request, **kwargs):
        sent["body"] = request.data
        return _Response()

    monkeypatch.setattr(fo, "urlopen", _urlopen)
    client = FalOpenRouterClient("test-key")
    client.interactions.create(
        model="gemini-3.8-flash",
        input=[video_content(one.as_uri(), processing="static") for one in videos]
        + [{"type": "text", "text": "compare"}],
    )
    body = json.loads(sent["body"])
    assert len(sent["body"]) <= fo.FAL_REQUEST_MAX_BYTES
    assert sum(
        part["type"] == "video_url" for part in body["messages"][0]["content"]
    ) == 4
