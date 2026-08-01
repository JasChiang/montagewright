from pathlib import Path
from types import SimpleNamespace

from jascue_video_lab.gemini import GeminiLabClient
from jascue_video_lab.media import sha256_file
from jascue_video_lab.storage import write_json


def _client_without_network() -> GeminiLabClient:
    return object.__new__(GeminiLabClient)


def test_ensure_upload_reuses_active_saved_file(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    source = tmp_path / "video.mp4"
    source.write_bytes(b"same immutable video bytes")
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/active"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": sha256_file(source)},
    )
    client = _client_without_network()
    uploaded = SimpleNamespace(name="files/active")
    calls = {"upload": 0}
    client.resume_video_upload = lambda *_args, **_kwargs: uploaded  # type: ignore[method-assign]

    def unexpected_upload(*_args, **_kwargs):
        calls["upload"] += 1
        raise AssertionError("ACTIVE file must not be uploaded again")

    client.upload_video = unexpected_upload  # type: ignore[method-assign]
    result, reused = client.ensure_video_upload(source, upload_dir)
    assert result is uploaded
    assert reused is True
    assert calls["upload"] == 0


def test_ensure_upload_reuploads_after_confirmed_file_api_expiry(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    source = tmp_path / "video.mp4"
    source.write_bytes(b"expired source bytes")
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/expired"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": sha256_file(source)},
    )
    client = _client_without_network()
    uploaded = SimpleNamespace(name="files/new")

    class NotFoundError(RuntimeError):
        code = 404

    def expired(*_args, **_kwargs):
        raise NotFoundError("file expired")

    client.resume_video_upload = expired  # type: ignore[method-assign]
    client.upload_video = lambda *_args, **_kwargs: uploaded  # type: ignore[method-assign]
    result, reused = client.ensure_video_upload(source, upload_dir)
    assert result is uploaded
    assert reused is False
    assert list((upload_dir / "history").glob("*/file_upload_initial.json"))


def test_ensure_upload_reuploads_after_file_api_403(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    source = tmp_path / "video.mp4"
    source.write_bytes(b"inaccessible cached source bytes")
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/old"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": sha256_file(source)},
    )
    client = _client_without_network()
    replacement = SimpleNamespace(name="files/reuploaded")

    class PermissionDeniedError(RuntimeError):
        code = 403

    client.resume_video_upload = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        PermissionDeniedError("PERMISSION_DENIED for cached File API object")
    )
    client.upload_video = lambda *_args, **_kwargs: replacement  # type: ignore[method-assign]

    result, reused = client.ensure_video_upload(source, upload_dir)

    assert result is replacement
    assert reused is False
    record = (upload_dir / "file_cache.json").read_text(encoding="utf-8")
    assert "expired_deleted_or_inaccessible" in record


def test_ensure_upload_reuploads_after_confirmed_file_api_failed_state(
    tmp_path: Path,
) -> None:
    upload_dir = tmp_path / "upload"
    source = tmp_path / "video.mp4"
    source.write_bytes(b"failed remote processing")
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/failed"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": sha256_file(source)},
    )
    client = _client_without_network()
    replacement = SimpleNamespace(name="files/reuploaded")
    client.resume_video_upload = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("Gemini File API ended in state FAILED")
    )
    client.upload_video = lambda *_args, **_kwargs: replacement  # type: ignore[method-assign]

    result, reused = client.ensure_video_upload(source, upload_dir)

    assert result is replacement
    assert reused is False


def test_ensure_upload_replaces_noncanonical_audio_mime_alias(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "music.wav"
    audio_path.write_bytes(b"RIFF")
    upload_dir = tmp_path / "upload"
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/alias"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": sha256_file(audio_path)},
    )
    client = _client_without_network()
    cached = SimpleNamespace(name="files/alias", mime_type="audio/x-wav")
    replacement = SimpleNamespace(name="files/canonical", mime_type="audio/wav")
    client.resume_video_upload = lambda *_args, **_kwargs: cached  # type: ignore[method-assign]
    client.upload_video = lambda *_args, **_kwargs: replacement  # type: ignore[method-assign]

    result, reused = client.ensure_video_upload(audio_path, upload_dir)

    assert result is replacement
    assert reused is False
    cache_record = (upload_dir / "file_cache.json").read_text(encoding="utf-8")
    assert "saved_file_api_mime_type_is_not_canonical" in cache_record


def test_ensure_upload_does_not_duplicate_on_transient_error(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    source = tmp_path / "video.mp4"
    source.write_bytes(b"transient source bytes")
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/unknown"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": sha256_file(source)},
    )
    client = _client_without_network()

    def transient(*_args, **_kwargs):
        raise RuntimeError("temporary network failure")

    client.resume_video_upload = transient  # type: ignore[method-assign]
    client.upload_video = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("must not upload after an ambiguous error")
    )

    try:
        client.ensure_video_upload(source, upload_dir)
    except RuntimeError as error:
        assert "temporary network failure" in str(error)
    else:
        raise AssertionError("transient error should be preserved")


def test_ensure_upload_rejects_active_cache_for_changed_local_bytes(
    tmp_path: Path,
) -> None:
    upload_dir = tmp_path / "upload"
    source = tmp_path / "video.mp4"
    source.write_bytes(b"new bytes")
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/stale"})
    write_json(
        upload_dir / "local_source_binding.json",
        {"sha256": "0" * 64},
    )
    client = _client_without_network()
    replacement = SimpleNamespace(name="files/replacement")
    client.resume_video_upload = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("stale remote object must not be resumed")
    )
    client.upload_video = lambda *_args, **_kwargs: replacement  # type: ignore[method-assign]

    result, reused = client.ensure_video_upload(source, upload_dir)

    assert result is replacement
    assert reused is False
    record = (upload_dir / "file_cache.json").read_text(encoding="utf-8")
    assert "local_source_hash_mismatch" in record
