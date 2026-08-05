from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

from jascue_video_lab.media import (
    create_analysis_proxy,
    extract_frame,
    extract_frame_at_pts,
    extract_frames_bounded,
    probe_video,
)


def test_probe_and_extract_preserve_semantic_request_vs_pts(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=c=blue:s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    assert media.video.coded_width == 320
    assert media.video.display_height == 180
    assert media.duration_ms == 2000
    frame = extract_frame(video, 555, tmp_path / "frame.png")
    assert frame.requested_time_ms == 555
    assert frame.frame_time_ms == 600
    assert frame.frame_pts != frame.frame_time_ms
    assert (frame.width, frame.height) == (320, 180)


def test_extract_frame_at_pts_reselects_exact_decoded_frame(tmp_path: Path) -> None:
    video = tmp_path / "exact-pts.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    semantic = extract_frame(video, 555, tmp_path / "semantic.png")

    exact = extract_frame_at_pts(
        video,
        semantic.frame_pts,
        tmp_path / "exact.png",
    )

    assert exact.frame_pts == semantic.frame_pts
    assert exact.frame_time_ms == semantic.frame_time_ms
    assert exact.requested_time_ms == exact.frame_time_ms


def test_extract_frame_near_container_eof_uses_last_decodable_pts(
    tmp_path: Path,
) -> None:
    video = tmp_path / "eof.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )

    frame = extract_frame(video, 1_999, tmp_path / "eof.png")

    assert frame.requested_time_ms == 1_999
    assert frame.frame_time_ms == 1_900
    assert frame.frame_time_ms < frame.requested_time_ms


def test_bounded_batch_matches_single_frame_selection_for_cfr(
    tmp_path: Path,
) -> None:
    video = tmp_path / "batch-cfr.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    requests = (0, 555, 1_250)

    batch = extract_frames_bounded(
        video,
        requests,
        tmp_path / "batch-cfr",
        window_start_ms=0,
        window_end_ms=2_000,
    )
    singles = tuple(
        extract_frame(video, request, tmp_path / f"single-{index}.png")
        for index, request in enumerate(requests)
    )

    assert [frame.requested_time_ms for frame in batch] == list(requests)
    assert [frame.frame_pts for frame in batch] == [
        frame.frame_pts for frame in singles
    ]
    assert [frame.frame_time_ms for frame in batch] == [
        frame.frame_time_ms for frame in singles
    ]


def test_bounded_batch_matches_single_frame_selection_for_vfr(
    tmp_path: Path,
) -> None:
    video = tmp_path / "batch-vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=5:d=1",
            "-filter_complex",
            (
                "[0:v]setpts=PTS-STARTPTS[v0];"
                "[1:v]setpts=PTS-STARTPTS[v1];"
                "[v0][v1]concat=n=2:v=1:a=0[v]"
            ),
            "-map",
            "[v]",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    requests = (150, 950, 1_150, 1_750)

    batch = extract_frames_bounded(
        video,
        requests,
        tmp_path / "batch-vfr",
        window_start_ms=0,
        window_end_ms=media.duration_ms,
    )
    singles = tuple(
        extract_frame(video, request, tmp_path / f"vfr-single-{index}.png")
        for index, request in enumerate(requests)
    )

    assert [frame.frame_pts for frame in batch] == [
        frame.frame_pts for frame in singles
    ]


def test_bounded_batch_preserves_nonzero_source_pts(
    tmp_path: Path,
) -> None:
    video = tmp_path / "batch-nonzero-start.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=10:d=2",
            "-vf",
            "setpts=PTS+5/TB",
            "-copyts",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    assert media.video.start_pts is not None
    assert media.video.start_pts > 0
    requests = (555, 1_250)

    batch = extract_frames_bounded(
        video,
        requests,
        tmp_path / "batch-nonzero",
        window_start_ms=0,
        window_end_ms=2_000,
    )
    singles = tuple(
        extract_frame(video, request, tmp_path / f"nonzero-single-{index}.png")
        for index, request in enumerate(requests)
    )

    assert [frame.frame_pts for frame in batch] == [
        frame.frame_pts for frame in singles
    ]
    assert all(frame.frame_pts > media.video.start_pts for frame in batch)


def test_bounded_batch_preserves_ffmpeg_display_orientation(
    tmp_path: Path,
) -> None:
    base = tmp_path / "orientation-base.mp4"
    video = tmp_path / "orientation-rotated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(base),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(base),
            "-c",
            "copy",
            str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    assert media.video.rotation_degrees == 90

    (batch,) = extract_frames_bounded(
        video,
        (555,),
        tmp_path / "batch-orientation",
        window_start_ms=0,
        window_end_ms=2_000,
    )
    single = extract_frame(video, 555, tmp_path / "single-orientation.png")

    assert batch.frame_pts == single.frame_pts
    assert (batch.width, batch.height) == (single.width, single.height)
    assert (batch.width, batch.height) == (180, 320)


def test_bounded_batch_preserves_narrow_eof_fallback(
    tmp_path: Path,
) -> None:
    video = tmp_path / "batch-eof.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )

    (batch_frame,) = extract_frames_bounded(
        video,
        (1_999,),
        tmp_path / "batch-eof",
        window_start_ms=0,
        window_end_ms=2_000,
    )
    single = extract_frame(video, 1_999, tmp_path / "single-eof.png")

    assert batch_frame.requested_time_ms == 1_999
    assert batch_frame.frame_pts == single.frame_pts
    assert batch_frame.frame_time_ms == single.frame_time_ms == 1_900


def test_probe_preserves_non_square_sample_aspect_ratio(tmp_path: Path) -> None:
    video = tmp_path / "anamorphic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=720x576:r=10:d=0.2",
            "-vf",
            "setsar=16/15",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    media = probe_video(video)

    assert media.video.sample_aspect_ratio.numerator == 16
    assert media.video.sample_aspect_ratio.denominator == 15
    assert media.video.display_sample_aspect_ratio == media.video.sample_aspect_ratio


def test_extract_frame_maps_nonzero_stream_pts_to_local_playback_time(
    tmp_path: Path,
) -> None:
    video = tmp_path / "nonzero-start.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=10:d=2",
            "-vf",
            "setpts=PTS+5/TB",
            "-copyts",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    assert media.video.start_pts is not None
    assert media.video.start_pts > 0

    frame = extract_frame(video, 555, tmp_path / "nonzero-frame.png")

    source_time_base = Fraction(
        media.video.time_base.numerator,
        media.video.time_base.denominator,
    )
    local_time_ms = round(
        Fraction(frame.frame_pts - media.video.start_pts) * source_time_base * 1000
    )
    assert frame.requested_time_ms == 555
    assert frame.frame_time_ms == local_time_ms == 600
    assert frame.frame_pts > media.video.start_pts


def test_analysis_proxy_preserves_duration_and_has_independent_identity(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=c=purple:s=640x360:r=20:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    source = probe_video(video)
    proxy, record = create_analysis_proxy(video, tmp_path / "analysis-proxy.mp4", max_side=480)
    assert proxy.video.display_width == 480
    assert abs(proxy.duration_ms - source.duration_ms) <= 100
    assert proxy.asset_id != source.asset_id
    assert record["source_asset_id"] == source.asset_id
    assert record["proxy_asset_id"] == proxy.asset_id
