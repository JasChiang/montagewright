from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jascue_video_lab.delivery_pipeline as pipeline
import pytest
from jascue_video_lab.media import sha256_file


@pytest.mark.parametrize(
    ("picture_ready", "expected_state"),
    [
        (True, "ready_for_human_review"),
        (False, "review_required"),
    ],
)
def test_delivery_pipeline_hash_binds_mux_and_runs_qa_on_final_media(
    tmp_path: Path,
    monkeypatch,
    picture_ready: bool,
    expected_state: str,
) -> None:
    brief = tmp_path / "brief.json"
    music = tmp_path / "music.wav"
    lock = tmp_path / "music-lock.json"
    picture = tmp_path / "picture.mp4"
    render_manifest = tmp_path / "render-manifest.json"
    for path, payload in (
        (brief, b"{}"),
        (music, b"music"),
        (lock, b"{}"),
        (picture, b"picture"),
        (render_manifest, b"{}"),
    ):
        path.write_bytes(payload)

    monkeypatch.setattr(
        pipeline,
        "run_feature_cut_experiment",
        lambda **_kwargs: {
            "ready_for_human_review": picture_ready,
            "media_rendered": True,
            "run_state": (
                "ready_for_human_review"
                if picture_ready
                else "review_preview"
            ),
            "horizontal_output": str(picture),
            "vertical_output": None,
            "manifest_path": str(render_manifest),
        },
    )
    fake_lock = SimpleNamespace(music_id=f"sha256:{sha256_file(music)}")
    monkeypatch.setattr(
        pipeline.MusicMapLock,
        "model_validate",
        lambda _payload: fake_lock,
    )
    monkeypatch.setattr(pipeline, "read_json", lambda _path: {})
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda _path: SimpleNamespace(duration_ms=60_000),
    )
    plan = SimpleNamespace()
    monkeypatch.setattr(
        pipeline,
        "plan_single_interval_music_assembly",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        pipeline,
        "write_music_assembly_artifacts",
        lambda *_args, **_kwargs: None,
    )

    def fake_music_render(_source, _plan, output, output_dir):
        del output_dir
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered-music")
        manifest = output.parent / "music-assembly-render.json"
        manifest.write_bytes(b"{}")
        return SimpleNamespace(output_audio_path=output, manifest_path=manifest)

    monkeypatch.setattr(
        pipeline,
        "render_single_interval_music_assembly",
        fake_music_render,
    )

    def fake_delivery(**kwargs):
        output = kwargs["output_path"]
        manifest = kwargs["manifest_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"final-mux")
        manifest.write_bytes(b"{}")
        return SimpleNamespace(output_path=output, manifest_path=manifest)

    monkeypatch.setattr(pipeline, "assemble_music_only_delivery", fake_delivery)
    prepared = SimpleNamespace(
        proxy_path=tmp_path / "qa-proxy.mp4",
        input_hashes={"proxy_sha256": "a" * 64},
    )
    prepared.proxy_path.write_bytes(b"proxy")
    monkeypatch.setattr(
        pipeline,
        "prepare_final_edit_qa",
        lambda **_kwargs: prepared,
    )
    qa = SimpleNamespace(
        result=SimpleNamespace(
            global_review=SimpleNamespace(
                disposition="ready_for_human_review"
            )
        ),
        run_dir=tmp_path / "qa-run",
        cache_hit=False,
    )
    monkeypatch.setattr(
        pipeline,
        "execute_final_edit_qa",
        lambda **_kwargs: qa,
    )

    class FakeClient:
        client = object()

        def __init__(self, *, model_id: str) -> None:
            assert model_id

        def ensure_video_upload(self, path: Path, artifact_dir: Path):
            assert path == prepared.proxy_path
            assert artifact_dir.name == "a" * 64
            return {"uri": "files/qa", "mime_type": "video/mp4"}, True

        def close(self) -> None:
            return None

    monkeypatch.setattr(pipeline, "GeminiLabClient", FakeClient)

    result = pipeline.run_feature_delivery_pipeline(
        feature_cut_kwargs={"catalog_path": tmp_path / "catalog.json"},
        brief_path=brief,
        music_path=music,
        music_lock_path=lock,
        output_dir=tmp_path / "delivery",
    )

    assert result["state"] == expected_state
    assert result["picture_ready_for_human_review"] is picture_ready
    assert result["delivery_eligible"] is False
    assert result["human_approval_status"] == "not_run"
    assert Path(result["aspects"]["horizontal"]["final_output"]).read_bytes() == (
        b"final-mux"
    )


def test_delivery_pipeline_stops_before_music_when_picture_media_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "run_feature_cut_experiment",
        lambda **_kwargs: {
            "ready_for_human_review": False,
            "media_rendered": False,
            "run_state": "partial",
        },
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("music stage must not run")

    monkeypatch.setattr(
        pipeline,
        "plan_single_interval_music_assembly",
        forbidden,
    )
    with pytest.raises(
        pipeline.DeliveryPipelineBlocked,
        match="did not produce reviewable picture media",
    ):
        pipeline.run_feature_delivery_pipeline(
            feature_cut_kwargs={},
            brief_path=tmp_path / "brief.json",
            music_path=tmp_path / "music.wav",
            music_lock_path=tmp_path / "music-lock.json",
            output_dir=tmp_path / "delivery",
        )
    assert called is False
