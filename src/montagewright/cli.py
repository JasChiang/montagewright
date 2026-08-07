"""One command from a folder of rushes to a finished cut.

Every stage writes what it decided next to the output, because a run nobody
can inspect afterwards is a run nobody can argue with. The report is the
deliverable as much as the file is.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from montagewright.clipcard import (
    action_beats,
    build_library,
    find_subject,
    load_card,
    snap_to_action,
    subjects_from_card,
)
from montagewright.cost import BudgetSpent, Ledger
from montagewright.grounding import analyse_track, load_beat_grid, shots_in
from montagewright.pipeline import probe, run
from montagewright.review import (
    Round,
    actionable_keys,
    adjudicate,
    review_cut,
    review_shots,
    should_continue,
)
from montagewright.planner import (
    MaterialItem,
    decide_direction,
    replan_shots,
    select_shots,
)
from montagewright.schema import EDL, Clip, move_of_shot, reframe_of, subject_of
from montagewright.uploads import (
    UploadCache,
    default_cache_path,
    default_library,
)

ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}

# Long enough that it is worth asking whether this is one take or many. Under
# it, scene detection costs more than it can save.
SPLIT_ABOVE_SECONDS = 90.0
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV"}


def _client():
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is required")
    from montagewright.planner import _http_options

    return genai.Client(api_key=key, http_options=_http_options(types))


def _travel(source: Path, target_aspect: float) -> tuple[float, float]:
    """Horizontal and vertical room to move, as fractions of the frame.

    Not the proxy, for the same reason `push_room` is not: this looked like a
    question about shape and is a question about resolution. A crop only has
    to be as tall as the delivery, so what is left over is room to move -- and
    a 640-wide proxy has nothing left over and would report that no clip
    anywhere can be tilted.
    """

    from montagewright.executor import delivery_size
    from montagewright.measure.media import probe_video
    from montagewright.reframe import travel_room

    try:
        shape = probe_video(source).video
        wide, tall = int(shape.display_width), int(shape.display_height)
    except Exception:
        return 0.0, 0.0
    if not wide or not tall:
        return 0.0, 0.0
    out_w, out_h = delivery_size(target_aspect)
    return travel_room(
        source_width=wide, source_height=tall, target_aspect=target_aspect,
        output_width=out_w, output_height=out_h,
    )


def _push_room(proxy: Path, target_aspect: float) -> float:
    """How far this source can be pushed into, as a zoom factor.

    Read off the file that will actually be cut, so it is a fact about this
    clip rather than a rule about clips. A 4K take at 9:16 has room for
    about 1.5x; a 1080 one has none, and saying so is what stops a push
    being asked for where it cannot be given.

    Not the proxy: that is 640 pixels wide and would report that nothing
    anywhere can be pushed into.
    """

    from montagewright.executor import delivery_size
    from montagewright.measure.media import probe_video
    from montagewright.reframe import zoom_budget

    try:
        shape = probe_video(proxy).video
        wide = int(shape.display_width)
        tall = int(shape.display_height)
    except Exception:
        return 1.0
    if not wide or not tall:
        return 1.0
    out_w, out_h = delivery_size(target_aspect)
    budget = zoom_budget(
        source_width=wide, source_height=tall, source_aspect=wide / tall,
        target_aspect=target_aspect, output_width=out_w, output_height=out_h,
    )
    return round(1.0 / max(budget, 1e-6), 2)


def _make_proxy(
    source: Path, destination: Path, *, library: Path | None = None
) -> Path:
    """A small copy for the model to watch.

    Sending 4K masters would cost more than the rest of the run put together
    and tell the model nothing extra: it is judging what is in the frame, not
    how sharp it is.

    A proxy is a pure function of the bytes it was made from, so it is kept
    where the cards it feeds are kept -- named for those bytes, shared across
    runs. It used to live in the output directory, which meant a second cut
    of the same rushes re-encoded seventy-four 4K files before it could ask
    the first question. The run still gets one under the source's own name,
    because everything downstream looks it up that way; it is just a link
    now.
    """

    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if library is not None:
        from montagewright.uploads import content_hash

        kept = library / "proxies" / f"{content_hash(source)[:20]}.mp4"
        if kept.exists():
            try:
                destination.hardlink_to(kept)
            except OSError:
                shutil.copy2(kept, destination)
            return destination
        kept.parent.mkdir(parents=True, exist_ok=True)
        _encode_proxy(source, kept)
        try:
            destination.hardlink_to(kept)
        except OSError:
            shutil.copy2(kept, destination)
        return destination

    _encode_proxy(source, destination)
    return destination


def _encode_proxy(source: Path, destination: Path) -> None:
    """Shrink the frame, keep the clock.

    This forced 15fps for a while, which nothing asked for. Gemini samples
    video at one frame a second whatever it is given, so the extra frames
    were never looked at; SAM tracks the original rather than this file, at
    its own rate; and the only thing left watching a proxy at full speed is
    the web preview, which is happier at the source rate anyway.

    What the resampling did cost was a clock that no longer matched. At
    15fps a duration has to land on a multiple of 1/15, so a 12.012s take
    came out 12.133s -- and cards describe this file while the edit cuts the
    original, which left two clocks a tenth of a second apart for no reason.
    Dropping the flag makes them the same to the millisecond, for about nine
    percent more bytes and no extra encoding time.

    The width is capped rather than set, because `scale=640` is a demand and
    not a limit: handed a 320x240 clip it produced a 640x480 one, larger than
    the file it came from and blurrier than the picture it describes. Nothing
    is gained by enlarging a proxy -- the API caps each video frame at 70
    tokens whatever it is sent, so the extra pixels are discarded before the
    model ever sees them.

    `-2` rounds the height to the nearest even number, which H.264 requires,
    so the aspect can shift by up to one pixel. 16:9 and 4:3 divide cleanly
    and come out exact; the worst measured case is a 3840x1600 source at
    0.25%, since a short proxy gives that one pixel more to be worth. It only
    reaches anything through the card's own subject boxes, and only when
    those are used without a client to ask again -- the grounding and
    tracking path never touches this file.
    """

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vf", "scale='min(640,iw)':-2",
            "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "64k", str(destination),
        ],
        check=True,
    )


def _tee_output(destination: Path) -> None:
    """Send everything printed to the terminal and to a file.

    Line buffered, because the point is to be readable while the run is
    still going.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = destination.open("a", encoding="utf-8", buffering=1)

    class Both:
        def __init__(self, first, second):
            self.first, self.second = first, second

        def write(self, text):
            self.first.write(text)
            try:
                self.second.write(text)
            except (OSError, ValueError):
                pass
            return len(text)

        def flush(self):
            self.first.flush()
            try:
                self.second.flush()
            except (OSError, ValueError):
                pass

    sys.stdout = Both(sys.stdout, handle)
    sys.stderr = Both(sys.stderr, handle)


def _make_findable(output: Path) -> None:
    """Let the interface list a cut written anywhere.

    It only scans its own runs folder, so `--output ~/cut` -- which is what
    the README tells people to type -- produced a film, a report and two
    timelines that nothing could open. Rather than teach every path in the
    interface that a run might live elsewhere, the runs folder gets a link
    pointing at it, and everything downstream carries on believing the
    layout it already believes.
    """

    from montagewright.webapp import RUNS_ROOT

    root = RUNS_ROOT.expanduser()
    if output.is_relative_to(root):
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        stem = output.parent.name if output.name == "out" else output.name
        folder = root / (stem or "cut")
        count = 2
        while folder.exists() and not (folder / "out").is_symlink():
            folder = root / f"{stem}-{count}"
            count += 1
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / "out"
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(output, target_is_directory=True)
    except OSError:
        # Not being listable is not a reason to refuse to cut.
        pass


def command_render(args: argparse.Namespace) -> int:
    rushes = args.rushes.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # What produced this, written before anything is attempted. The
    # interface lists every cut in the runs folder and could only see the
    # ones it had started itself, so a render from the command line left a
    # report, a film and two timelines that nothing could open. Written at
    # the end it would have covered only the runs that finished -- and the
    # ones worth finding are the others: three died partway today, each
    # after paying for cards, direction and selection, each resumable from
    # what was already on disk, and none of them openable.
    # Everything printed also goes beside the output, so a run started here
    # can be read in the interface. The note said which command produced the
    # run and the interface could offer to resume it -- but with no log there
    # was nothing on the page except the word "中斷", which reads as broken
    # rather than as unfinished.
    _tee_output(output / "run.log")

    (output / "command.json").write_text(
        json.dumps({
            "source": str(rushes),
            "command": [sys.executable, "-u", "-m", "montagewright.cli"]
            + sys.argv[1:],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    _make_findable(output)

    work = output / "work"
    # Cards and transcripts describe the material, so they belong to the
    # material rather than to one attempt at cutting it. Keeping them beside
    # the output meant every new run over the same rushes paid for them again
    # -- forty-four cents of cards before anything was decided.
    library = (args.library or default_library()).expanduser()
    client = _client()
    cache = UploadCache.load(args.upload_cache or default_cache_path())
    ledger = Ledger(cap_usd=args.budget)

    sources_paths = sorted(
        path for path in rushes.iterdir() if path.suffix in VIDEO_SUFFIXES
    )
    if not sources_paths:
        raise SystemExit(f"no video files in {rushes}")
    print(f"{len(sources_paths)} clips in {rushes.name}", flush=True)

    # Something already cut is one file holding many takes. Handing it over
    # whole means one card for five minutes, one transcript, and a planner
    # choosing windows out of a single source as though the cuts inside it
    # were not there -- so a long file is opened along the boundaries it
    # already has. A continuous take comes back as itself.
    rushes_paths: list[Path] = []
    for path in sources_paths:
        spans = (
            shots_in(path)
            if _duration(path) >= SPLIT_ABOVE_SECONDS
            else [(0.0, 0.0)]
        )
        if len(spans) < 2:
            rushes_paths.append(path)
            continue
        print(
            f"{path.name}: already cut, opening into {len(spans)} shots",
            flush=True,
        )
        pieces = work / "shots"
        pieces.mkdir(parents=True, exist_ok=True)
        for index, (start, end) in enumerate(spans):
            piece = pieces / f"{path.stem}-{index:02d}{path.suffix}"
            if not piece.exists():
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                        "-i", str(path), "-c:v", "libx264", "-crf", "18",
                        "-preset", "veryfast", "-c:a", "aac", str(piece),
                    ],
                    check=True,
                )
            rushes_paths.append(piece)
    sources_paths = rushes_paths

    proxies = {
        path.stem: _make_proxy(
            path, work / "proxies" / f"{path.stem}.mp4", library=library
        )
        for path in sources_paths
    }
    # The originals, by source id. How far a shot can be pushed into is a
    # fact about the file that will be cut, and the proxy is 640 pixels wide
    # -- asking it says every clip has no room at all.
    originals = {path.stem: path for path in sources_paths}
    def wrote(index: int, total: int, source_id: str) -> None:
        print(f"  card {index}/{total}  {source_id}", flush=True)

    cards, stats = build_library(
        proxies, library / "cards", client=client, cache=cache, progress=wrote
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
    for line in stats.get("failures", [])[:5]:
        print(f"  card failed — {line[:140]}", flush=True)

    # Only where the speech is the content. A transcript costs a call and a
    # minute per clip, and on b-roll it answers a question nobody asked --
    # so the card, which already watched the clip with its audio, says which
    # ones need one rather than a flag somebody has to remember.
    transcripts: dict[str, dict] = {}
    speaking = [
        source_id for source_id in proxies
        if (load_card(cards[source_id]) if source_id in cards else {})
        and (load_card(cards[source_id]) or {}).get("speech") == "content"
    ]
    if speaking and args.speech != "never":
        from montagewright.transcript import describe as transcribe
        from montagewright.transcript import load as load_transcript
        from montagewright.transcript import save as save_transcript

        print(
            f"speech: {len(speaking)} clips carry it as content",
            flush=True,
        )
        for source_id in speaking:
            from montagewright.uploads import content_hash

            destination = (
                library / "transcripts"
                / f"{content_hash(proxies[source_id])[:20]}.json"
            )
            card = load_transcript(destination)
            if card is None:
                # Before the call, not after: a transcript on a long clip is
                # one of the more expensive things here, and the cap is meant
                # to stop work rather than to describe it afterwards.
                ledger.check()
                try:
                    card, usage = transcribe(
                        proxies[source_id], client=client,
                        locale=args.locale, cache=cache,
                    )
                except Exception as error:
                    print(
                        f"  {source_id} — no transcript: "
                        f"{type(error).__name__}: {error}"[:150],
                        flush=True,
                    )
                    continue
                ledger.record(
                    "transcript",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens + usage.thought_tokens,
                )
                save_transcript(card, destination)
            transcripts[source_id] = card
        print(f"  transcribed, running total ${ledger.spent_usd:.4f}", flush=True)

    # Measured once per source rather than twice per field, and off the
    # original: how far a crop can travel is a fact about resolution.
    _room = {
        source_id: _travel(originals.get(source_id, proxy), ASPECTS[args.aspect])
        for source_id, proxy in proxies.items()
    }
    material = []
    # Which takes were set aside before anything was planned, and why. The
    # card gives a reason and it was being dropped on the floor, so a run
    # that quietly worked from sixty-six of seventy-four clips looked
    # identical to one that had all of them -- and "why didn't it use the
    # good coin shot" had no answer anywhere in the output.
    set_aside: dict[str, str] = {}
    for source_id, proxy in proxies.items():
        card = load_card(cards[source_id]) if source_id in cards else None
        if card is not None and not card.get("usable", True):
            set_aside[source_id] = str(
                card.get("unusable_reason") or "no reason given"
            )
            continue
        material.append(
            MaterialItem(
                source_id=source_id,
                duration_seconds=_duration(proxy),
                summary=(card or {}).get("summary", ""),
                proxy=proxy,
                composition=(card or {}).get("composition", ""),
                camera_moves=bool((card or {}).get("camera_moves", False)),
                camera_motion=str((card or {}).get("camera_motion", "") or ""),
                shot_size=str((card or {}).get("shot_size", "") or ""),
                facing=str((card or {}).get("facing", "") or ""),
                usable_from=float((card or {}).get("usable_from_seconds", 0.0)),
                usable_to=float((card or {}).get("usable_to_seconds", 0.0)),
                push_room=_push_room(
                    originals.get(source_id, proxy), ASPECTS[args.aspect]
                ),
                pan_room=_room[source_id][0],
                tilt_room=_room[source_id][1],
                action=tuple(
                    f"{beat.what} {beat.starts_seconds:.1f}-{beat.ends_seconds:.1f}s"
                    for beat in action_beats(card or {})[:4]
                ),
                needs=tuple(
                    f"{entry.get('what')}（{entry.get('why', '')[:60]}）"
                    for entry in (card or {}).get("needs", [])
                ),
                subjects=tuple(
                    _subject_line(box, _aspect(proxy), ASPECTS[args.aspect])
                    for box in subjects_from_card(card or {})
                ),
                speech=_speech_lines(transcripts.get(source_id)),
            )
        )

    if set_aside:
        print(
            f"set aside: {len(set_aside)} clips the cards called unusable",
            flush=True,
        )
        for source_id, why in list(set_aside.items())[:5]:
            print(f"  {source_id} — {why[:110]}", flush=True)

    brief = args.brief.read_text(encoding="utf-8") if args.brief else ""
    asked = _asked(
        ",".join(sorted(item.source_id for item in material)),
        brief, args.aspect, str(args.music or ""),
    )
    direction = _decided(work, "direction", asked)
    if direction is None:
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
        _decide(work, "direction", asked, direction)
    else:
        print("direction: reused from the last attempt", flush=True)
    print(
        f"direction: {direction['target_seconds']:.0f}s {direction['aspect']}, "
        f"{len(direction.get('unusable', []))} ruled out",
        flush=True,
    )

    chose = _asked(asked, json.dumps(direction, sort_keys=True, ensure_ascii=False))
    selection = _decided(work, "selection", chose)
    if selection is None:
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
        _decide(work, "selection", chose, selection)
    else:
        print("selection: reused from the last attempt", flush=True)
    print(f"selection: {len(selection['shots'])} shots", flush=True)

    edl, snaps = _edl_from_selection(selection, rushes, cards)
    if snaps:
        print(f"cut on action: {len(snaps)} in-points moved", flush=True)
    found = {path.stem: path for path in sources_paths}
    sources = {
        shot["source_id"]: probe(shot["source_id"], found[shot["source_id"]])
        for shot in selection["shots"]
        if shot["source_id"] in found
    }
    if args.music_map:
        grid = load_beat_grid(args.music_map)
    elif args.music:
        # A reviewed lock proves which analysis a delivery was cut against.
        # Requiring one before the tool will run at all turns "let me see
        # what this does" into a two-step errand, and the measurement is the
        # same either way.
        print("no music map given; measuring the track", flush=True)
        grid = analyse_track(args.music)
    else:
        # A cut carried by what people say does not need a bed, and refusing
        # to run without one was the tool making an editorial decision on the
        # way in. Without a grid every length is content-led, which is what a
        # speech cut wants anyway.
        grid = None
        print("no music; lengths will be led by content", flush=True)

    rhythm_context = _rhythm_context(selection, cards)

    aspect = ASPECTS[args.aspect]

    def cut(edl, sources, rhythm_context):
        """Render this plan. One definition, because a replan renders again.

        It was written out twice, and only the first copy learned to keep the
        voice -- so a run that revised anything delivered the same film with
        the speech thrown away and nothing but the bed left.
        """

        return run(
            edl,
            sources,
            grid,
            output,
            target_aspect=aspect,
            intent=direction["direction"],
            brief=brief,
            rhythm_context=rhythm_context,
            target_seconds=float(direction["target_seconds"]),
            music=args.music,
            cards=cards,
            checkpoint=args.sam_checkpoint,
            ledger=ledger,
            keep_voice=bool(transcripts),
            under_speech=str(direction.get("music_under_speech") or "duck"),
            client=client,
        )

    result, plan, report, resolved = cut(edl, sources, rhythm_context)
    # The direction set a length; somebody has to compare it with what came
    # out. Three layers each made a defensible call last run and delivered
    # 17.9 seconds against 30, with nothing in the report saying so.
    report.target_seconds = float(direction["target_seconds"])

    rounds: list[Round] = []
    shot_verdicts: dict[str, dict] = {}
    stopped = "review not requested"
    if args.review:
        # Every round renders before it reviews, so a stopping condition
        # always leaves a finished film rather than a half-planned one.
        while True:
            keep_going, stopped = should_continue(rounds, ledger=ledger)
            if not keep_going:
                break
            # Two questions, two viewings. The shots say whether each one did
            # what its plan promised; the cut says whether the eight of them
            # are a film. Only the second was ever asked, and it is the one
            # that cannot see a clipped wordmark going past in three seconds.
            try:
                shot_verdicts = review_shots(
                    {
                        f"k{index:02d}": path
                        for index, path in enumerate(result.segment_paths)
                    },
                    selection["shots"],
                    seconds={
                        clip_id: entry["seconds"]
                        for clip_id, entry in report.rhythm_decisions.items()
                    },
                    degradations=report.degradations,
                    client=client,
                    cache=cache,
                    ledger=ledger,
                )
            except BudgetSpent as error:
                stopped = str(error)
                break
            missed = [
                f"{clip_id}: {entry.get('note', '')}"
                for clip_id, entry in shot_verdicts.items()
                if not entry.get("delivered", True)
            ]
            print(
                f"shots: {len(shot_verdicts) - len(missed)}/"
                f"{len(shot_verdicts)} delivered what they planned",
                flush=True,
            )
            for line in missed:
                print(f"  {line[:110]}", flush=True)
            try:
                verdict = review_cut(
                    result.preview,
                    brief=brief,
                    direction=direction["direction"],
                    client=client,
                    cache=cache,
                    ledger=ledger,
                )
            except BudgetSpent as error:
                stopped = str(error)
                break
            rounds.append(
                Round(
                    index=len(rounds) + 1,
                    verdict=verdict,
                    actionable=actionable_keys(verdict),
                )
            )
            print(
                f"review {len(rounds)}: {verdict.verdict} "
                f"({len(verdict.issues)} issues) — {verdict.overall[:70]}",
                flush=True,
            )
            keep_going, stopped = should_continue(rounds, ledger=ledger)
            if not keep_going:
                break
            failing = [
                (index, shot, shot_verdicts[f"k{index:02d}"].get("note", ""))
                for index, shot in enumerate(selection["shots"])
                if not shot_verdicts.get(f"k{index:02d}", {}).get(
                    "delivered", True
                )
            ]
            if not failing:
                # The cut reviewer wants a change nobody can point at a shot.
                stopped = (
                    "revision asked for, but no shot was named as undelivered"
                )
                break
            try:
                ledger.check()
                replanned, usage = replan_shots(
                    failing,
                    material,
                    direction,
                    brief=brief,
                    context="\n".join(
                        f"第 {index + 1} 顆（{shot['source_id']}）："
                        f"{shot.get('why', '')}"
                        for index, shot in enumerate(selection["shots"])
                    ),
                    cache=cache,
                    client=client,
                )
            except BudgetSpent as error:
                stopped = str(error)
                break
            ledger.record(
                "replan",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens + usage.thought_tokens,
            )
            fresh = replanned.get("shots", [])
            if len(fresh) != len(failing):
                stopped = (
                    f"replan returned {len(fresh)} shots for "
                    f"{len(failing)} that needed one"
                )
                break
            for (index, old, _), new in zip(failing, fresh):
                print(
                    f"  replan k{index:02d}: {old['source_id']} "
                    f"{move_of_shot(old)} → {new['source_id']} "
                    f"{move_of_shot(new)} — {new.get('why', '')[:70]}",
                    flush=True,
                )
                selection["shots"][index] = new
            # Everything downstream is rebuilt from the amended selection, so
            # the next round renders a different film rather than re-reading
            # the same one.
            edl, snaps = _edl_from_selection(selection, rushes, cards)
            sources = {
                shot["source_id"]: probe(
                    shot["source_id"], found[shot["source_id"]]
                )
                for shot in selection["shots"]
                if shot["source_id"] in found
            }
            rhythm_context = _rhythm_context(selection, cards)
            try:
                ledger.check()
                result, plan, report, resolved = cut(
                    edl, sources, rhythm_context
                )
            except BudgetSpent as error:
                stopped = str(error)
                break
            report.target_seconds = float(direction["target_seconds"])
        if rounds:
            report.degradations = adjudicate(
                report.degradations, rounds[-1].verdict, shot_verdicts
            )

    _write_report(
        output,
        snaps=snaps,
        set_aside=set_aside,
        material_ids=[item.source_id for item in material],
        rounds=rounds,
        shot_verdicts=shot_verdicts,
        stopped_because=stopped,
        direction=direction,
        selection=selection,
        report=report,
        plan=plan,
        result=result,
        usages=[],
    )
    print(f"\n{report.summary()}", flush=True)
    for stage, usd in sorted(
        report.spend().get("by_stage", {}).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {stage:12s} ${usd:.4f}", flush=True)
    # The words, once there is a cut for them to sit on. Written from the
    # same lines the timeline shows, so what was corrected in the interface
    # is what gets burned.
    if transcripts and args.subtitles != "none":
        from montagewright.transcript import (
            against_cut, to_srt, words_against_cut,
        )

        said = against_cut(
            selection["shots"],
            {k: v for k, v in report.rhythm_decisions.items()},
            transcripts,
        )
        try:
            from montagewright.subtitles import as_cues

            wide, tall = (1920, 1080) if aspect >= 1.0 else (1080, 1920)
            said = as_cues(said, args.aspect, wide, tall)
        except Exception:
            pass
        if said:
            (output / "subtitles.srt").write_text(
                to_srt(said, with_speaker=True), encoding="utf-8"
            )
            print(f"subtitles   {output / 'subtitles.srt'}", flush=True)
            if args.subtitles == "burn":
                from montagewright import subtitles as typeset
                from montagewright.subtitles import NoFontHere, burn

                if args.subtitle_font:
                    typeset.CHOSEN = str(args.subtitle_font.expanduser())

                try:
                    burned = burn(
                        result.deliverable, said,
                        output / "deliverable-subtitled.mp4",
                        aspect=args.aspect, work=work / "subs",
                        style=typeset.look(args.subtitle_look),
                        words=words_against_cut(
                            selection["shots"],
                            report.rhythm_decisions,
                            transcripts,
                        ),
                    )
                    print(f"burned in   {burned}", flush=True)
                except NoFontHere as error:
                    # The cut is finished either way; say what is missing
                    # rather than failing a render over a font.
                    print(f"not burned  {error}", flush=True)

    print(f"deliverable {result.deliverable}", flush=True)
    print(f"preview     {result.preview}", flush=True)

    # Off unless asked for. A rendered file is what most runs want; a
    # timeline is for the run where somebody intends to open it and disagree
    # with one shot, and writing one every time is clutter in every other.
    if args.timeline != "none":
        from montagewright.timeline import to_fcpxml, to_xmeml

        payload = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )
        width, height = (
            (1920, 1080) if ASPECTS[args.aspect] >= 1.0 else (1080, 1920)
        )
        for flavour, suffix, build in (
            ("premiere", "xml", to_xmeml),
            ("finalcut", "fcpxml", to_fcpxml),
        ):
            if args.timeline in {flavour, "both"}:
                path = output / f"timeline.{suffix}"
                path.write_text(
                    build(plan, payload, name=output.name,
                          width=width, height=height, music=args.music),
                    encoding="utf-8",
                )
                print(f"timeline    {path}", flush=True)
    return 0


def _suffix(rushes: Path, stem: str) -> str:
    for suffix in VIDEO_SUFFIXES:
        if (rushes / f"{stem}{suffix}").exists():
            return suffix
    return ".mp4"


def _speech_lines(card: dict | None, limit: int = 40) -> tuple[str, ...]:
    """The soundbites, as the planner needs to read them.

    A window out of an interview is chosen because of a sentence, so the
    sentence has to be visible when the choice is made -- with who said it,
    because a shot of the person not talking is the obvious way to get this
    wrong, and with its seconds, because they are what the shot's length is.
    """

    if not card:
        return ()
    from montagewright.transcript import lines_of

    return tuple(
        f"{line.starts_seconds:.1f}-{line.ends_seconds:.1f}s"
        f"（{line.speaker or '未標'}）{line.text}"
        for line in lines_of(card)[:limit]
    )


def _subject_line(box, source_aspect: float, target_aspect: float) -> str:
    """A subject, and how much of it the delivery frame can hold.

    Selection was marking a wordmark "must be whole" on a 16:9 plate being
    delivered 9:16, where the widest possible crop covers a third of it. The
    planner was not being careless -- it had no way to know, so the promise
    was unkeepable at the moment it was made. Told the fraction, it can move
    the camera across the subject, pick a tighter shot of the same thing, or
    accept a partial view on purpose.
    """

    fits = min(1.0, (target_aspect / source_aspect) / max(box.width, 1e-9))
    if fits >= 0.995:
        return box.label
    return f"{box.label}（交付比例裡最多只能露出 {fits * 100:.0f}%）"


def _duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def _aspect(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return float(stream["width"]) / float(stream["height"])


def _decided(work: Path, name: str, key: str) -> dict | None:
    """A decision already paid for, if it was the same question.

    Cards and transcripts survive a crash because they are keyed by the
    content they describe. Direction and selection were not kept at all, so a
    run that died after them -- on a quota, on a bad path -- paid for them
    again on the way back. They are keyed the same way: the same material,
    brief and aspect is the same question, and a different one is a different
    key rather than a stale answer.
    """

    path = work / f"{name}.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return saved.get("value") if saved.get("key") == key else None


def _decide(work: Path, name: str, key: str, value: dict) -> dict:
    path = work / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"key": key, "value": value}, ensure_ascii=False),
        encoding="utf-8",
    )
    return value


def _asked(*parts: str) -> str:
    import hashlib

    return hashlib.sha256("\u0000".join(parts).encode("utf-8")).hexdigest()[:16]


def _rhythm_context(
    selection: dict, cards: dict[str, Path]
) -> dict[str, dict]:
    """Why each shot exists, what moves in it, how much frame it holds.

    A length is a judgement about purpose, and the rhythm pass was being
    asked to make it from a description and an energy label.
    """

    context: dict[str, dict] = {}
    for index, shot in enumerate(selection["shots"]):
        clip_id = f"k{index:02d}"
        card = (
            load_card(cards[shot["source_id"]])
            if shot["source_id"] in cards
            else None
        )
        entry: dict[str, object] = {"why": shot.get("why", "")}
        if card is not None:
            box = find_subject(card, subject_of(shot))
            if box is not None:
                entry["subject_share"] = round(box.width * box.height, 4)
            beats = action_beats(card)
            if beats:
                entry["action"] = "；".join(
                    f"{beat.what} {beat.starts_seconds:.1f}-"
                    f"{beat.ends_seconds:.1f}s"
                    for beat in beats[:3]
                )
        context[clip_id] = entry
    return context


def _edl_from_selection(
    selection: dict, rushes: Path, cards: dict[str, Path]
) -> tuple[EDL, dict[str, str]]:
    clips = []
    snaps: dict[str, str] = {}
    for index, shot in enumerate(selection["shots"]):
        # Text and UI have to survive whole or they say nothing. The first
        # run cropped a Galaxy Unpacked wordmark down to "y Unpacked", which
        # is worse than not using the shot. That, and everything else a
        # reframe is made of, is decided in one place now -- the rebuild had
        # its own copy and the copy was missing a field.
        reframe = reframe_of(shot)
        clip_id = f"k{index:02d}"
        start = float(shot["start_seconds"])
        # What selection said this shot needs to do its job. It used to be a
        # flat four seconds for every shot, which meant the layer that chose
        # the material had no say in how long it ran and the layer that chose
        # the length started from a constant.
        wanted = float(shot.get("seconds_needed") or 0.0) or 4.0
        card = load_card(cards[shot["source_id"]]) if shot["source_id"] in cards else None
        if card is not None:
            start, note = snap_to_action(card, start, wanted)
            if note:
                snaps[clip_id] = note
        clips.append(
            Clip(
                clip_id=clip_id,
                source_id=shot["source_id"],
                approx_in_seconds=start,
                approx_out_seconds=start + wanted,
                in_looks_like=subject_of(shot),
                energy_intent=shot.get("energy", "medium"),
                reframe=reframe,
            )
        )
    return EDL(project_id=rushes.name, clips=clips), snaps


def _write_report(output: Path, **parts) -> None:
    """The account of what was decided and what it cost."""

    report = parts["report"]
    payload = {
        "direction": parts["direction"],
        "selection": parts["selection"],
        # Against how many asked for a beat, not how many cuts there are. A
        # speech cut where every length is content-led was reading as 0/13,
        # which is what missing every beat would look like.
        "cuts_on_music": (
            f"{report.aligned_cuts}/"
            f"{sum(1 for e in report.rhythm_decisions.values() if e.get('cut_on_beat'))}"
            if report.rhythm_decisions
            else f"{report.aligned_cuts}/{report.total_cuts}"
        ),
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
                # Who settled it and on what grounds. Without this the report
                # says "replan" and nothing about why, which is the same
                # position the reviewer was in before they could see the shot.
                "adjudication_reason": step.adjudication_reason,
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
        "moves_too_short": report.moves_too_short,
        "spend": report.spend(),
        "cut_on_action": parts.get("snaps", {}),
        "set_aside": parts.get("set_aside", {}),
        # Everything that was on the table. Without it there is no way to
        # say which clips were simply passed over, which is how most of a
        # folder does not reach the film.
        "material_ids": parts.get("material_ids", []),
        "rhythm": report.rhythm_decisions,
        # Per shot, against the plan that asked for it. The whole-cut verdict
        # below answers a different question and has never once caught a
        # composition fault, because at thirty seconds the next shot arrives
        # before the fault registers.
        "shots": parts.get("shot_verdicts", {}),
        "review": {
            "stopped_because": parts.get("stopped_because"),
            "rounds": [
                {
                    "verdict": entry.verdict.verdict,
                    "overall": entry.verdict.overall,
                    "issues": [
                        {
                            "clip_id": issue.clip_id,
                            "at_seconds": issue.at_seconds,
                            "type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "fix": issue.fix,
                        }
                        for issue in entry.verdict.issues
                    ],
                }
                for entry in parts.get("rounds", [])
            ],
        },
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def command_transcribe(args: argparse.Namespace) -> int:
    """Subtitle a video without touching the edit.

    This is the transcript half on its own, because subtitling something
    already finished is a real job and has nothing to do with cutting. The
    same card feeds the editorial passes when there is an edit.
    """

    from montagewright.transcript import describe, lines_of, load, save, to_srt
    from montagewright.uploads import content_hash

    client = _client()
    cache = UploadCache.load(args.upload_cache or default_cache_path())
    ledger = Ledger(cap_usd=args.budget)

    sources = (
        sorted(p for p in args.source.iterdir() if p.suffix in VIDEO_SUFFIXES)
        if args.source.is_dir()
        else [args.source]
    )
    if not sources:
        raise SystemExit(f"no video files in {args.source}")

    output = (args.output or args.source.parent).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    library = (args.library or default_library()).expanduser()
    work = output / "work"

    for source in sources:
        # The same proxy a render would make, keyed the same way. Both
        # commands transcribe the same material and this one kept its answer
        # beside its own output under the file's name, so rendering the same
        # folder afterwards found nothing and paid for all of it again -- at
        # twenty cents a clip, on the most expensive call in the tool.
        proxy = _make_proxy(
            source, work / "proxies" / f"{source.stem}.mp4", library=library
        )
        destination = (
            library / "transcripts" / f"{content_hash(proxy)[:20]}.json"
        )
        card = load(destination)
        if card is None:
            card, usage = describe(
                proxy, client=client, locale=args.locale, cache=cache
            )
            ledger.record(
                "transcript",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens + usage.thought_tokens,
            )
            save(card, destination)
        lines = lines_of(card)
        (output / f"{source.stem}.srt").write_text(
            to_srt(lines), encoding="utf-8"
        )
        changed = sum(1 for line in lines if line.corrected)
        print(
            f"{source.name}: {len(lines)} lines, {changed} corrected, "
            f"{card.get('language')} — {output / (source.stem + '.srt')}",
            flush=True,
        )
        for note in card.get("uncertain", [])[:3]:
            print(f"  unsure — {str(note)[:120]}", flush=True)

    print(f"transcript spend ${ledger.spent_usd:.4f}", flush=True)
    return 0


def command_timeline(args: argparse.Namespace) -> int:
    """Write a timeline for a cut that already exists.

    Choosing the flavour before the run and finding out afterwards that you
    wanted one meant running the whole thing again. Everything a timeline
    needs is in the report and the material, and rebuilding the plan costs
    nothing -- the subject positions come out of the cards.
    """

    from montagewright.clipcard import card_map
    from montagewright.executor import plan_render
    from montagewright.pipeline import Report, follow_subjects, probe
    from montagewright.timeline import to_fcpxml, to_xmeml

    output = args.output.expanduser().resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    shots = report.get("selection", {}).get("shots", [])
    rhythm = report.get("rhythm", {})
    aspect = ASPECTS.get(report.get("direction", {}).get("aspect", "9:16"), 9 / 16)
    library = (args.library or default_library()).expanduser()
    cards = card_map(output / "work" / "proxies", library / "cards")

    hunting = [output / "work" / "shots"] + (
        [args.rushes.expanduser()] if args.rushes else []
    )
    clips, sources = [], {}
    for index, shot in enumerate(shots):
        source_id = shot["source_id"]
        if source_id not in sources:
            match = next(
                (
                    path
                    for folder in hunting if folder.exists()
                    for path in folder.iterdir()
                    if path.stem == source_id
                ),
                None,
            )
            if match is None:
                raise SystemExit(
                    f"cannot find {source_id}; pass --rushes with its folder"
                )
            sources[source_id] = probe(source_id, match)
        start = float(shot.get("start_seconds", 0.0))
        clips.append(
            Clip(
                clip_id=f"k{index:02d}", source_id=source_id,
                approx_in_seconds=start,
                approx_out_seconds=start
                + float(rhythm.get(f"k{index:02d}", {}).get("seconds", 0.0)),
                in_looks_like=subject_of(shot),
                energy_intent=shot.get("energy", "medium"),
                reframe=reframe_of(shot),
            )
        )
    edl = EDL(project_id=output.name, clips=clips)
    paths = follow_subjects(
        edl, sources, target_aspect=aspect, report=Report(),
        cards=cards, checkpoint=None, client=None,
    )
    plan = plan_render(edl, sources, target_aspect=aspect, crop_paths=paths)
    width, height = (1920, 1080) if aspect >= 1.0 else (1080, 1920)
    for flavour, suffix, build in (
        ("premiere", "xml", to_xmeml), ("finalcut", "fcpxml", to_fcpxml)
    ):
        if args.flavour in {flavour, "both"}:
            path = output / f"timeline.{suffix}"
            path.write_text(
                build(plan, report, name=output.name,
                      width=width, height=height),
                encoding="utf-8",
            )
            print(f"timeline    {path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="montagewright")
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
    render.add_argument(
        "--review",
        action="store_true",
        help="Watch the finished cut and report what it would change.",
    )
    render.add_argument("--output", type=Path, required=True)
    render.add_argument(
        "--timeline", choices=["none", "premiere", "finalcut", "both"],
        default="none",
        help="also write an editable timeline (off unless asked for)",
    )
    render.add_argument(
        "--subtitles", choices=["none", "sidecar", "burn"], default="sidecar",
        help=(
            "sidecar writes an SRT beside the cut; burn also puts the words "
            "on the picture, inside the safe area for the delivery aspect"
        ),
    )
    render.add_argument(
        "--subtitle-look", choices=["plain", "speakers", "spoken", "plate"],
        default="plain",
        help="how burned subtitles look: plain, a colour per speaker, "
             "filling as it is said, or on a plate",
    )
    render.add_argument(
        "--subtitle-font", type=Path,
        help="a font file to set the subtitles in; the system is asked if "
             "this is not given",
    )
    render.add_argument(
        "--speech", choices=["auto", "never"], default="auto",
        help="transcribe clips whose card calls the speech content",
    )
    render.add_argument("--locale", default="zh-TW")
    render.add_argument(
        "--library", type=Path,
        help="where cards and transcripts live; shared across runs",
    )
    render.set_defaults(handler=command_render)

    speak = sub.add_parser(
        "transcribe", help="Subtitle a video, or a folder of them"
    )
    speak.add_argument("source", type=Path)
    speak.add_argument("--locale", default="zh-TW")
    speak.add_argument("--output", type=Path)
    speak.add_argument("--budget", type=float, default=5.0)
    speak.add_argument("--upload-cache", type=Path)
    speak.add_argument(
        "--library", type=Path,
        help="where cards and transcripts live; shared across runs",
    )
    speak.set_defaults(handler=command_transcribe)

    lay = sub.add_parser(
        "timeline", help="Write a timeline for a cut that already exists"
    )
    lay.add_argument("output", type=Path)
    lay.add_argument(
        "--flavour", choices=["premiere", "finalcut", "both"], default="both"
    )
    lay.add_argument(
        "--rushes", type=Path,
        help="where the sources are, if they are not under the output",
    )
    lay.add_argument("--library", type=Path)
    lay.set_defaults(handler=command_timeline)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
