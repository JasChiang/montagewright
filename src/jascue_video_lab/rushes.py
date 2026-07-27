from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from time import monotonic
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .billing import summarize_usage_and_list_price
from .gemini import GeminiLabClient
from .media import extract_frame, probe_video, sha256_file
from .models import RushClip, RushFrame, RushesCatalog, RushesEditPlan
from .shots import ShotManifest, detect_shots_ffmpeg
from .storage import read_json, utc_now, write_json


def _probe_clip(path: Path) -> dict[str, Any]:
    media = probe_video(path)
    rate = media.video.average_frame_rate or media.video.real_frame_rate
    return {
        "duration_ms": media.duration_ms,
        "size_bytes": media.size_bytes,
        "width": media.video.display_width,
        "height": media.video.display_height,
        "frame_rate": (
            f"{rate.numerator}/{rate.denominator}" if rate is not None else "0/0"
        ),
    }


def _format_mmss(milliseconds: int) -> str:
    seconds = max(0, milliseconds // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _catalog_fingerprint(clips: list[RushClip], interval_ms: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"interval_ms={interval_ms}\n".encode())
    for clip in clips:
        digest.update(
            f"{clip.clip_id}|{clip.sha256}|{clip.duration_ms}|{clip.width}x{clip.height}\n".encode()
        )
    return f"sha256:{digest.hexdigest()}"


def _label_frame(source_path: Path, output_path: Path, label: str) -> None:
    with Image.open(source_path).convert("RGB") as source:
        image = source.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(18, image.width // 30))
    height = max(48, image.height // 8)
    draw.rectangle((0, 0, image.width, height), fill="#080b10")
    draw.text((14, 10), label, fill="#ffffff", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def _render_contact_sheets(frame_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = 4, 4
    cell_width, cell_height = 320, 180
    page_size = columns * rows
    for page_index, start in enumerate(range(0, len(frame_paths), page_size), start=1):
        canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "#101418")
        for local_index, path in enumerate(frame_paths[start : start + page_size]):
            with Image.open(path).convert("RGB") as source:
                frame = source.resize((cell_width, cell_height))
            x = (local_index % columns) * cell_width
            y = (local_index // columns) * cell_height
            canvas.paste(frame, (x, y))
        canvas.save(output_dir / f"page-{page_index:03d}.jpg", quality=88)


def _extract_catalog_frames(
    clip: RushClip,
    raw_dir: Path,
    *,
    sample_interval_ms: int,
    max_width: int,
) -> list[tuple[Path, Any]]:
    """Extract immutable decoded frames and preserve their real source PTS."""

    records: list[tuple[Path, Any]] = []
    requested_times = list(range(0, clip.duration_ms, sample_interval_ms)) or [0]
    for local_index, requested_time_ms in enumerate(requested_times):
        output = raw_dir / f"{local_index:06d}.png"
        frame = extract_frame(
            Path(clip.path),
            requested_time_ms,
            output,
            max_width=max_width,
        )
        records.append((output, frame))
    return records


def _discover_sources(
    source_directory: Path,
    *,
    excluded_directory: Path | None = None,
) -> list[Path]:
    root = source_directory.expanduser().resolve(strict=True)
    excluded = (
        excluded_directory.expanduser().resolve()
        if excluded_directory is not None
        else None
    )
    sources: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".mp4",
            ".mov",
            ".m4v",
        }:
            continue
        resolved = path.resolve()
        if excluded is not None and (
            resolved == excluded or excluded in resolved.parents
        ):
            continue
        sources.append(resolved)
    return sorted(sources, key=lambda path: str(path.relative_to(root)).casefold())


def _clip_id_for_path(
    path: Path,
    *,
    source_directory: Path,
    used: set[str],
) -> str:
    relative = str(path.relative_to(source_directory.resolve()))
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "clip"
    candidate = base
    if candidate in used:
        suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        candidate = f"{base}-{suffix}"
    used.add(candidate)
    return candidate


def validate_rushes_catalog_sources(
    catalog: RushesCatalog,
    *,
    source_directory: Path | None = None,
    sample_interval_ms: int | None = None,
    excluded_directory: Path | None = None,
) -> dict[str, Any]:
    """Reject a stale catalog before model calls or rendering."""

    # A deliberately empty catalog can still be used to produce an auditable
    # missing-evidence review package. It has no source bytes whose lineage can
    # become stale, so do not require its informational root path to exist.
    if not catalog.clips:
        return {
            "contract_version": "rushes-catalog-source-validation-v1",
            "catalog_id": catalog.catalog_id,
            "source_directory": catalog.source_directory,
            "status": "validated_empty_catalog",
            "source_count": 0,
            "added": [],
            "removed": [],
            "changed": [],
        }

    expected_root = Path(catalog.source_directory).expanduser().resolve(strict=True)
    if source_directory is not None:
        supplied_root = source_directory.expanduser().resolve(strict=True)
        if supplied_root != expected_root:
            raise ValueError("catalog source directory differs from the requested source")
    if sample_interval_ms is not None and sample_interval_ms != catalog.sample_interval_ms:
        raise ValueError("catalog sample interval differs from the requested interval")
    catalog_by_path = {
        str(Path(clip.path).expanduser().resolve()): clip for clip in catalog.clips
    }
    current_paths = _discover_sources(
        expected_root,
        excluded_directory=excluded_directory,
    )
    current_set = {str(path) for path in current_paths}
    catalog_set = set(catalog_by_path)
    added = sorted(current_set - catalog_set)
    removed = sorted(catalog_set - current_set)
    changed: list[dict[str, str]] = []
    for path_string in sorted(current_set & catalog_set):
        current_hash = sha256_file(Path(path_string))
        saved_hash = catalog_by_path[path_string].sha256
        if current_hash != saved_hash:
            changed.append(
                {
                    "path": path_string,
                    "catalog_sha256": saved_hash,
                    "current_sha256": current_hash,
                }
            )
    result = {
        "contract_version": "rushes-catalog-source-validation-v1",
        "catalog_id": catalog.catalog_id,
        "source_directory": str(expected_root),
        "added_paths": added,
        "removed_paths": removed,
        "changed_paths": changed,
        "valid": not (added or removed or changed),
        "validated_at": utc_now(),
    }
    if not result["valid"]:
        raise ValueError(
            "rushes catalog is stale: source files were added, removed or changed"
        )
    return result


def create_rushes_catalog(
    source_directory: Path,
    output_dir: Path,
    *,
    sample_interval_ms: int = 2000,
    max_width: int = 640,
) -> RushesCatalog:
    if sample_interval_ms < 500:
        raise ValueError("sample_interval_ms must be at least 500")
    resolved_source_directory = source_directory.expanduser().resolve(strict=True)
    sources = _discover_sources(
        resolved_source_directory,
        excluded_directory=output_dir,
    )
    if not sources:
        raise ValueError(f"no video files found in {source_directory}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[RushClip] = []
    used_clip_ids: set[str] = set()
    for path in sources:
        metadata = _probe_clip(path)
        clips.append(
            RushClip(
                clip_id=_clip_id_for_path(
                    path,
                    source_directory=resolved_source_directory,
                    used=used_clip_ids,
                ),
                path=str(path.resolve()),
                sha256=sha256_file(path),
                **metadata,
            )
        )
    frames_dir = output_dir / "catalog-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[RushFrame] = []
    frame_paths: list[Path] = []
    next_frame_number = 1
    for clip in clips:
        raw_dir = output_dir / "catalog-raw" / clip.clip_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_paths = _extract_catalog_frames(
            clip,
            raw_dir,
            sample_interval_ms=sample_interval_ms,
            max_width=max_width,
        )
        for local_index, (raw_path, extracted) in enumerate(raw_paths):
            requested_time_ms = local_index * sample_interval_ms
            if requested_time_ms >= clip.duration_ms:
                continue
            frame_id = f"RF{next_frame_number:06d}"
            output_path = frames_dir / f"{frame_id}.jpg"
            _label_frame(
                raw_path,
                output_path,
                f"{frame_id}  |  {clip.clip_id}  |  {_format_mmss(requested_time_ms)}",
            )
            frames.append(
                RushFrame(
                    frame_id=frame_id,
                    clip_id=clip.clip_id,
                    requested_time_ms=requested_time_ms,
                    image_path=str(Path("catalog-frames") / output_path.name),
                    source_image_path=str(
                        Path("catalog-raw") / clip.clip_id / raw_path.name
                    ),
                    frame_time_ms=extracted.frame_time_ms,
                    frame_pts=extracted.frame_pts,
                    frame_hash=extracted.frame_hash,
                )
            )
            frame_paths.append(output_path)
            next_frame_number += 1
    if not frames:
        raise RuntimeError("catalog extraction produced no frames")
    reel_path = output_dir / "analysis-reel.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-start_number",
            "1",
            "-i",
            str(frames_dir / "RF%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(reel_path),
        ],
        check=True,
    )
    _render_contact_sheets(frame_paths, output_dir / "contact-sheets")
    catalog = RushesCatalog(
        catalog_id=_catalog_fingerprint(clips, sample_interval_ms),
        source_directory=str(resolved_source_directory),
        sample_interval_ms=sample_interval_ms,
        total_duration_ms=sum(clip.duration_ms for clip in clips),
        clips=clips,
        frames=frames,
        analysis_reel_path=str(reel_path.resolve()),
        generated_at=utc_now(),
    )
    write_json(output_dir / "catalog.json", catalog)
    return catalog


def _crop_filter(aspect_ratio: str, focus: str) -> tuple[str, tuple[int, int]]:
    if aspect_ratio == "16:9":
        return (
            "fps=30,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
            (1920, 1080),
        )
    x_expression = {"left": "0", "center": "(iw-ow)/2", "right": "iw-ow"}[focus]
    return f"fps=30,scale=-2:1920,crop=1080:1920:{x_expression}:0", (1080, 1920)


def _segment_bounds(
    *, center_ms: int, requested_duration_ms: int, clip_duration_ms: int, shot: Any
) -> tuple[int, int]:
    start = center_ms - requested_duration_ms // 2
    end = center_ms + requested_duration_ms // 2
    if start < shot.start_time_ms:
        end += shot.start_time_ms - start
        start = shot.start_time_ms
    if end > shot.end_time_ms:
        start -= end - shot.end_time_ms
        end = shot.end_time_ms
    start = max(shot.start_time_ms, 0, start)
    end = min(shot.end_time_ms, clip_duration_ms, end)
    if end - start < 1000:
        raise ValueError("selected representative frame has less than one second inside its shot")
    return start, end


def render_rushes_edit(
    catalog: RushesCatalog,
    plan: RushesEditPlan,
    output_dir: Path,
    *,
    scdet_threshold: float = 4.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    shot_cache: dict[str, ShotManifest] = {}
    render_result: dict[str, Any] = {"timelines": []}
    for timeline in plan.timelines:
        slug = timeline.aspect_ratio.replace(":", "x")
        timeline_dir = output_dir / slug
        segments_dir = timeline_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[dict[str, Any]] = []
        for index, selected in enumerate(timeline.shots):
            frame = frames[selected.representative_frame_id]
            clip = clips[frame.clip_id]
            if clip.clip_id not in shot_cache:
                shot_cache[clip.clip_id] = detect_shots_ffmpeg(
                    Path(clip.path),
                    threshold=scdet_threshold,
                    output_path=output_dir / "shots" / f"{clip.clip_id}.json",
                )
            shot = next(
                item
                for item in shot_cache[clip.clip_id].shots
                if item.start_time_ms <= frame.requested_time_ms < item.end_time_ms
            )
            start_ms, end_ms = _segment_bounds(
                center_ms=frame.requested_time_ms,
                requested_duration_ms=round(selected.suggested_duration_seconds * 1000),
                clip_duration_ms=clip.duration_ms,
                shot=shot,
            )
            filter_graph, dimensions = _crop_filter(
                timeline.aspect_ratio, selected.vertical_focus
            )
            segment_path = segments_dir / f"{index:03d}-{selected.select_id}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start_ms / 1000:.3f}",
                    "-i",
                    clip.path,
                    "-t",
                    f"{(end_ms - start_ms) / 1000:.3f}",
                    "-vf",
                    filter_graph,
                    "-an",
                    "-c:v",
                    "h264_videotoolbox",
                    "-b:v",
                    "8M",
                    "-pix_fmt",
                    "yuv420p",
                    str(segment_path),
                ],
                check=True,
            )
            rendered.append(
                {
                    "select_id": selected.select_id,
                    "representative_frame_id": frame.frame_id,
                    "source_clip_id": clip.clip_id,
                    "source_path": clip.path,
                    "source_in_ms": start_ms,
                    "source_out_ms": end_ms,
                    "source_shot_id": shot.shot_id,
                    "vertical_focus": selected.vertical_focus,
                    "output_dimensions": list(dimensions),
                    "segment_path": str(segment_path.resolve()),
                }
            )
        concat_path = segments_dir / "concat.txt"
        concat_path.write_text(
            "".join(f"file '{Path(item['segment_path']).name}'\n" for item in rendered),
            encoding="utf-8",
        )
        output_path = timeline_dir / f"rough-cut-{slug}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path.resolve()),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path.resolve()),
            ],
            check=True,
        )
        render_result["timelines"].append(
            {
                "aspect_ratio": timeline.aspect_ratio,
                "output_path": str(output_path.resolve()),
                "shots": rendered,
            }
        )
    write_json(output_dir / "render-manifest.json", render_result)
    _render_review_html(catalog, plan, render_result, output_dir)
    return render_result


def _render_review_html(
    catalog: RushesCatalog,
    plan: RushesEditPlan,
    render_result: dict[str, Any],
    output_dir: Path,
) -> None:
    timelines = {item["aspect_ratio"]: item for item in render_result["timelines"]}
    sections: list[str] = []
    for timeline in plan.timelines:
        rendered = timelines[timeline.aspect_ratio]
        video_rel = Path(rendered["output_path"]).relative_to(output_dir.resolve())
        rows = []
        for selected, item in zip(timeline.shots, rendered["shots"], strict=True):
            frame = next(
                frame for frame in catalog.frames if frame.frame_id == selected.representative_frame_id
            )
            frame_rel = Path("..") / frame.image_path
            rows.append(
                "<tr>"
                f"<td><img src=\"{html.escape(str(frame_rel))}\"></td>"
                f"<td>{html.escape(item['source_clip_id'])}</td>"
                f"<td>{item['source_in_ms']/1000:.3f}–{item['source_out_ms']/1000:.3f}s</td>"
                f"<td>{html.escape(selected.role)}</td>"
                f"<td>{html.escape(selected.visual_description)}</td>"
                f"<td>{html.escape('; '.join(selected.quality_risks) or 'none')}</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{timeline.aspect_ratio} — {html.escape(timeline.title)}</h2>"
            f"<p>{html.escape(timeline.editorial_intent)}</p>"
            f"<video controls src=\"{html.escape(str(video_rel))}\"></video>"
            "<table><thead><tr><th>frame</th><th>clip</th><th>source range</th><th>role</th><th>description</th><th>risks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    (output_dir / "index.html").write_text(
        """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Rushes rough-cut review</title>
<style>body{font:15px system-ui;background:#101214;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}section{background:#1b1f24;padding:20px;margin:20px 0;border-radius:12px}video{width:min(100%,960px);max-height:70vh;background:#000}table{border-collapse:collapse;width:100%;margin-top:18px}th,td{border:1px solid #3b424a;padding:8px;text-align:left;vertical-align:top}img{width:200px}code{color:#7cf}</style>
<h1>Rushes rough-cut review</h1><p>這是 Gemini frame-ID selects 經本機 FFmpeg shot-boundary 驗證後的實驗 rough cut，不是 final edit。</p>"""
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )


def run_rushes_experiment(
    source_directory: Path,
    output_dir: Path,
    *,
    prompt_template: str,
    sample_interval_ms: int = 2000,
    scdet_threshold: float = 4.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = monotonic()
    catalog_path = output_dir / "catalog.json"
    if catalog_path.exists():
        catalog = RushesCatalog.model_validate(read_json(catalog_path))
        validation = validate_rushes_catalog_sources(
            catalog,
            source_directory=source_directory,
            sample_interval_ms=sample_interval_ms,
            excluded_directory=output_dir,
        )
        write_json(output_dir / "catalog-source-validation.json", validation)
    else:
        stage = monotonic()
        catalog = create_rushes_catalog(
            source_directory,
            output_dir,
            sample_interval_ms=sample_interval_ms,
        )
        timings["catalog_seconds"] = round(monotonic() - stage, 3)
    reel_path = Path(catalog.analysis_reel_path)
    reel_media = probe_video(reel_path)
    upload_dir = output_dir / "file-cache" / reel_media.sha256 / "upload"
    client = GeminiLabClient()
    try:
        stage = monotonic()
        uploaded, reused = client.ensure_video_upload(reel_path, upload_dir)
        timings["file_api_seconds"] = round(monotonic() - stage, 3)
        stage = monotonic()
        plan = client.plan_rushes_edit(
            catalog=catalog,
            uploaded=uploaded,
            prompt_template=prompt_template,
            project_id="rushes-selects",
            run_id=f"rushes-{uuid.uuid4().hex[:8]}",
            run_dir=output_dir / "gemini",
        )
        timings["gemini_plan_seconds"] = round(monotonic() - stage, 3)
    finally:
        client.close()
    stage = monotonic()
    render_result = render_rushes_edit(
        catalog,
        plan,
        output_dir / "renders",
        scdet_threshold=scdet_threshold,
    )
    timings["render_seconds"] = round(monotonic() - stage, 3)
    timings["total_seconds"] = round(monotonic() - started, 3)
    pricing = summarize_usage_and_list_price(output_dir / "gemini")
    write_json(output_dir / "pricing.json", pricing)
    write_json(
        output_dir / "timing.json",
        {**timings, "file_api_reused": reused, "generated_at": utc_now()},
    )
    result = {
        "catalog_path": str(catalog_path.resolve()),
        "plan_path": str((output_dir / "gemini" / "rushes_edit_plan.json").resolve()),
        "review_path": str((output_dir / "renders" / "index.html").resolve()),
        "renders": render_result,
        "timing": timings,
        "pricing": pricing,
    }
    write_json(output_dir / "result.json", result)
    return result
