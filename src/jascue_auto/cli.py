"""One command from a folder of rushes to a finished cut.

Every stage writes what it decided next to the output, because a run nobody
can inspect afterwards is a run nobody can argue with. The report is the
deliverable as much as the file is.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from jascue_auto.clipcard import build_library, load_card, subjects_from_card
from jascue_auto.cost import Ledger
from jascue_auto.grounding import load_beat_grid
from jascue_auto.pipeline import probe, run
from jascue_auto.planner import MaterialItem, decide_direction, select_shots
from jascue_auto.schema import EDL, Clip, Reframe, Subject
from jascue_auto.uploads import UploadCache, default_cache_path

ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV"}


def _client():
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is required")
    from jascue_auto.planner import _http_options

    return genai.Client(api_key=key, http_options=_http_options(types))


def _make_proxy(source: Path, destination: Path) -> Path:
    """A small copy for the model to watch.

    Sending 4K masters would cost more than the rest of the run put together
    and tell the model nothing extra: it is judging what is in the frame, not
    how sharp it is.
    """

    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vf", "scale=640:-2", "-r", "15",
            "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "64k", str(destination),
        ],
        check=True,
    )
    return destination


def command_render(args: argparse.Namespace) -> int:
    rushes = args.rushes.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    client = _client()
    cache = UploadCache.load(args.upload_cache or default_cache_path())
    ledger = Ledger(cap_usd=args.budget)

    sources_paths = sorted(
        path for path in rushes.iterdir() if path.suffix in VIDEO_SUFFIXES
    )
    if not sources_paths:
        raise SystemExit(f"no video files in {rushes}")
    print(f"{len(sources_paths)} clips in {rushes.name}", flush=True)

    proxies = {
        path.stem: _make_proxy(path, work / "proxies" / f"{path.stem}.mp4")
        for path in sources_paths
    }
    cards, stats = build_library(
        proxies, work / "cards", client=client, cache=cache
    )
    ledger.record(
        "clip_cards",
        input_tokens=stats["input"],
        output_tokens=stats["output"],
    )
    print(
        f"cards: {stats['written']} written, {stats['reused']} reused, "
        f"{stats['failed']} failed  (${ledger.spent_usd:.4f})",
        flush=True,
    )

    material = []
    for source_id, proxy in proxies.items():
        card = load_card(cards[source_id]) if source_id in cards else None
        if card is not None and not card.get("usable", True):
            continue
        material.append(
            MaterialItem(
                source_id=source_id,
                duration_seconds=_duration(proxy),
                summary=(card or {}).get("summary", ""),
                proxy=proxy,
                composition=(card or {}).get("composition", ""),
                camera_moves=bool((card or {}).get("camera_moves", False)),
                subjects=tuple(
                    box.label for box in subjects_from_card(card or {})
                ),
            )
        )

    brief = args.brief.read_text(encoding="utf-8") if args.brief else ""
    ledger.check()
    direction, usage_direction = decide_direction(
        material, brief=brief, music=args.music, cache=cache, client=client
    )
    ledger.record(
        "direction",
        input_tokens=usage_direction.input_tokens,
        output_tokens=usage_direction.output_tokens
        + usage_direction.thought_tokens,
    )
    print(
        f"direction: {direction['target_seconds']:.0f}s {direction['aspect']}, "
        f"{len(direction.get('unusable', []))} ruled out",
        flush=True,
    )

    ledger.check()
    selection, usage_selection = select_shots(
        material, direction, brief=brief, cache=cache, client=client
    )
    ledger.record(
        "selection",
        input_tokens=usage_selection.input_tokens,
        output_tokens=usage_selection.output_tokens
        + usage_selection.thought_tokens,
    )
    print(f"selection: {len(selection['shots'])} shots", flush=True)

    edl = _edl_from_selection(selection, rushes)
    sources = {
        shot["source_id"]: probe(
            shot["source_id"], rushes / f"{shot['source_id']}{_suffix(rushes, shot['source_id'])}"
        )
        for shot in selection["shots"]
    }
    grid = load_beat_grid(args.music_map) if args.music_map else None
    if grid is None:
        raise SystemExit("--music-map is required; run analyze-music first")

    aspect = ASPECTS[args.aspect]
    result, plan, report, resolved = run(
        edl,
        sources,
        grid,
        output,
        target_aspect=aspect,
        intent=direction["direction"],
        music=args.music,
        cards=cards,
        checkpoint=args.sam_checkpoint,
        ledger=ledger,
        client=client,
    )
    # The direction set a length; somebody has to compare it with what came
    # out. Three layers each made a defensible call last run and delivered
    # 17.9 seconds against 30, with nothing in the report saying so.
    report.target_seconds = float(direction["target_seconds"])

    _write_report(
        output,
        direction=direction,
        selection=selection,
        report=report,
        plan=plan,
        result=result,
        usages=[usage_direction, usage_selection],
    )
    print(f"\n{report.summary()}", flush=True)
    for stage, usd in sorted(
        report.spend().get("by_stage", {}).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {stage:12s} ${usd:.4f}", flush=True)
    print(f"deliverable {result.deliverable}", flush=True)
    print(f"preview     {result.preview}", flush=True)
    return 0


def _suffix(rushes: Path, stem: str) -> str:
    for suffix in VIDEO_SUFFIXES:
        if (rushes / f"{stem}{suffix}").exists():
            return suffix
    return ".mp4"


def _duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def _edl_from_selection(selection: dict, rushes: Path) -> EDL:
    clips = []
    for index, shot in enumerate(selection["shots"]):
        reframe = Reframe(
            subject=Subject(
                description=shot["subject"],
                coarse_position=shot["subject_position"],
                # Text and UI have to survive whole or they say nothing. The
                # first run cropped a Galaxy Unpacked wordmark down to "y
                # Unpacked", which is worse than not using the shot.
                min_visible=1.0 if shot.get("must_be_whole") else 0.85,
            ),
            intent=shot.get("why", "")[:120],
            camera_move=shot.get("camera_move", "hold"),
            framing=shot.get("framing", "thirds"),
            camera_energy="active",
        )
        if shot.get("then_subject"):
            reframe = reframe.model_copy(
                update={
                    "then_subject": Subject(
                        description=shot["then_subject"],
                        coarse_position=shot.get(
                            "then_subject_position", "center"
                        ),
                    )
                }
            )
        clips.append(
            Clip(
                clip_id=f"k{index:02d}",
                source_id=shot["source_id"],
                approx_in_seconds=float(shot["start_seconds"]),
                approx_out_seconds=float(shot["start_seconds"]) + 4.0,
                in_looks_like=shot["subject"],
                energy_intent=shot.get("energy", "medium"),
                reframe=reframe,
            )
        )
    return EDL(project_id=rushes.name, clips=clips)


def _write_report(output: Path, **parts) -> None:
    """The account of what was decided and what it cost."""

    report = parts["report"]
    payload = {
        "direction": parts["direction"],
        "selection": parts["selection"],
        "cuts_on_music": f"{report.aligned_cuts}/{report.total_cuts}",
        "shots_following": report.following_shots,
        "shots_held": report.static_shots,
        "upscales": {k: round(v, 3) for k, v in report.upscales.items()},
        "subject_notes": report.subject_notes,
        "degradations": [
            {
                "clip_id": step.clip_id,
                "ladder": step.ladder_other or step.ladder,
                "trigger": step.trigger,
                "measured": step.measured,
                "adjudication": step.adjudication,
            }
            for step in report.degradations
        ],
        "tokens": {
            "input": report.input_tokens
            + sum(u.input_tokens for u in parts["usages"]),
            "output": report.output_tokens
            + sum(u.output_tokens + u.thought_tokens for u in parts["usages"]),
        },
        "duration_seconds": round(parts["result"].duration_seconds, 3),
        "target_seconds": report.target_seconds,
        "duration_shortfall_seconds": report.duration_shortfall,
        "extended_for_moves": report.extended_for_moves,
        "spend": report.spend(),
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jascue-auto")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Cut a folder of rushes into a film")
    render.add_argument("rushes", type=Path)
    render.add_argument("--brief", type=Path)
    render.add_argument("--music", type=Path)
    render.add_argument("--music-map", type=Path)
    render.add_argument("--aspect", choices=sorted(ASPECTS), default="9:16")
    render.add_argument("--sam-checkpoint", type=Path)
    render.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="Total spend ceiling in USD. Reaching it stops the run with the "
        "best cut so far rather than making the next step cheaper.",
    )
    render.add_argument(
        "--upload-cache",
        type=Path,
        help="Where uploaded-media URIs are remembered. Defaults to a shared "
        "location, because the key is the file's content and a per-run store "
        "re-uploads material that has not changed.",
    )
    render.add_argument("--output", type=Path, required=True)
    render.set_defaults(handler=command_render)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
