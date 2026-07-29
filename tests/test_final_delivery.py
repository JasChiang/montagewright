from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jascue_video_lab.final_delivery import (
    FinalDeliveryError,
    assemble_music_only_delivery,
    assemble_picture_only_delivery,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.music import (
    CuePriority,
    LockedMusicCue,
    MusicMapLock,
    MusicMapReview,
    MusicSectionCandidate,
)
from jascue_video_lab.music_assembly import (
    MUSIC_ASSEMBLY_RENDER_FILENAME,
    plan_single_interval_music_assembly,
    render_single_interval_music_assembly,
    write_music_assembly_artifacts,
)
from jascue_video_lab.storage import read_json, write_json


def _picture(path: Path, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _music(path: Path, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            str(duration),
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )


def test_picture_only_delivery_is_explicitly_silent_and_hash_bound(
    tmp_path: Path,
) -> None:
    picture = tmp_path / "picture.mp4"
    _picture(picture, 1.0)
    result = assemble_picture_only_delivery(
        picture_path=picture,
        output_path=tmp_path / "silent.mp4",
        manifest_path=tmp_path / "silent.json",
        aspect_ratio="16:9",
        artifact_bindings={"render": "a" * 64},
    )

    payload = read_json(result.manifest_path)
    assert payload["audio_policy"] == "explicitly_absent"
    assert payload["output_sha256"] == sha256_file(result.output_path)
    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(result.output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert {
        stream["codec_type"] for stream in probe["streams"]
    } == {"video"}


def _assembled_music(
    tmp_path: Path,
    *,
    duration: float,
) -> tuple[Path, Path]:
    source = tmp_path / f"source-{duration}.wav"
    _music(source, duration)
    sample_rate = 48_000
    duration_samples = round(sample_rate * duration)
    lock_path = tmp_path / f"music-{duration}.lock.json"
    proposal_sha256 = "a" * 64
    review = MusicMapReview(
        proposal_sha256=proposal_sha256,
        reviewer="test-editor",
        reviewed_at="2026-07-23T00:00:00+00:00",
        decision="approved",
        bpm=120.0,
        first_downbeat_sample=0,
        meter=4,
    )
    cues = [
        LockedMusicCue(
            cue_id=f"locked-cue-{index + 1:05d}",
            kind="downbeat" if index % 4 == 0 else "beat",
            sample_index=sample,
            time_ms=round(sample * 1000 / sample_rate),
            strength=0.9 if index % 4 == 0 else 0.6,
            priority=(
                CuePriority.PREFERRED
                if index % 4 == 0
                else CuePriority.OPTIONAL
            ),
        )
        for index, sample in enumerate(range(0, duration_samples, 24_000))
    ]
    music_lock = MusicMapLock(
        music_id=f"sha256:{sha256_file(source)}",
        proposal_path="/not/needed.json",
        proposal_sha256=proposal_sha256,
        review=review,
        master_sample_rate=sample_rate,
        duration_samples=duration_samples,
        duration_ms=round(duration * 1000),
        bpm=120.0,
        meter=4,
        first_downbeat_sample=0,
        cues=cues,
        sections=[
            MusicSectionCandidate(
                section_id="section-001",
                start_sample=0,
                end_sample=duration_samples,
                label="section_001",
                boundary_source="whole_track",
                confidence=0.5,
            )
        ],
        definition_sha256="c" * 64,
    )
    write_json(lock_path, music_lock)
    target_ms = round(duration * 1000)
    plan = plan_single_interval_music_assembly(
        music_lock,
        music_lock_path=lock_path,
        target_duration_ms=target_ms,
        minimum_duration_ms=max(1, target_ms - 100),
        maximum_duration_ms=target_ms + 100,
        preferred_phrase_bars=(1,),
    )
    artifact_dir = tmp_path / f"assembly-{duration}"
    write_music_assembly_artifacts(plan, output_dir=artifact_dir)
    rendered = tmp_path / f"rendered-{duration}.wav"
    render_single_interval_music_assembly(
        source,
        plan,
        rendered,
        artifact_dir,
    )
    return rendered, artifact_dir


def test_delivery_replaces_rush_audio_with_one_continuous_music_stream(
    tmp_path: Path,
) -> None:
    picture = tmp_path / "picture.mp4"
    output = tmp_path / "delivery.mp4"
    manifest = tmp_path / "delivery.json"
    _picture(picture, 2)
    music, artifact_dir = _assembled_music(tmp_path, duration=2)

    result = assemble_music_only_delivery(
        picture_path=picture,
        music_path=music,
        output_path=output,
        manifest_path=manifest,
        music_assembly_artifact_dir=artifact_dir,
        aspect_ratio="16:9",
    )

    saved = json.loads(manifest.read_text())
    assert result.output_path == output.resolve()
    assert saved["music"]["join_count"] == 0
    assert saved["music"]["assembly_evidence"]["status"] == "validated"
    assert saved["synchronization"]["delivery_music_trim_applied"] is False
    assert saved["synchronization"]["picture_extension_applied"] is False
    assert saved["synchronization"]["ffmpeg_shortest_used"] is False
    assert saved["synchronization"]["per_stream_qc_passed"] is True
    assert "-shortest" not in saved["ffmpeg"]["command"]
    assert saved["audio_policy"]["source_audio"] == "excluded"
    streams = saved["output"]["probe"]["streams"]
    assert sum(item["codec_type"] == "video" for item in streams) == 1
    assert sum(item["codec_type"] == "audio" for item in streams) == 1


def test_delivery_refuses_to_hide_a_material_duration_mismatch(
    tmp_path: Path,
) -> None:
    picture = tmp_path / "picture.mp4"
    _picture(picture, 2)
    music, artifact_dir = _assembled_music(tmp_path, duration=4)

    with pytest.raises(FinalDeliveryError, match="hard-cutting"):
        assemble_music_only_delivery(
            picture_path=picture,
            music_path=music,
            output_path=tmp_path / "delivery.mp4",
            manifest_path=tmp_path / "delivery.json",
            music_assembly_artifact_dir=artifact_dir,
            aspect_ratio="16:9",
        )


def test_delivery_refuses_music_when_render_manifest_hash_is_not_bound(
    tmp_path: Path,
) -> None:
    picture = tmp_path / "picture.mp4"
    _picture(picture, 2)
    music, artifact_dir = _assembled_music(tmp_path, duration=2)
    render_path = artifact_dir / MUSIC_ASSEMBLY_RENDER_FILENAME
    changed = read_json(render_path)
    changed["output_audio_sha256"] = "0" * 64
    write_json(render_path, changed)

    with pytest.raises(
        FinalDeliveryError,
        match="render manifest output hash",
    ):
        assemble_music_only_delivery(
            picture_path=picture,
            music_path=music,
            output_path=tmp_path / "delivery.mp4",
            manifest_path=tmp_path / "delivery.json",
            music_assembly_artifact_dir=artifact_dir,
            aspect_ratio="16:9",
        )
