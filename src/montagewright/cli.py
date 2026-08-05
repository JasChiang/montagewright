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
from montagewright.schema import EDL, Clip, Reframe, Subject
from montagewright.uploads import UploadCache, default_cache_path

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
            destination = work / "transcripts" / f"{source_id}.json"
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
                usable_from=float((card or {}).get("usable_from_seconds", 0.0)),
                usable_to=float((card or {}).get("usable_to_seconds", 0.0)),
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
                    f"{old.get('camera_move')} → {new['source_id']} "
                    f"{new.get('camera_move')} — {new.get('why', '')[:70]}",
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
        rounds=rounds,
        shot_verdicts=shot_verdicts,
        stopped_because=stopped,
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
                          width=width, height=height),
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
            box = find_subject(card, shot["subject"])
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
        reframe = Reframe(
            subject=Subject(
                description=shot["subject"],
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
                    )
                }
            )
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
                in_looks_like=shot["subject"],
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

    for source in sources:
        destination = output / f"{source.stem}.transcript.json"
        card = load(destination)
        if card is None:
            card, usage = describe(
                source, client=client, locale=args.locale, cache=cache
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
        "--speech", choices=["auto", "never"], default="auto",
        help="transcribe clips whose card calls the speech content",
    )
    render.add_argument("--locale", default="zh-TW")
    render.set_defaults(handler=command_render)

    speak = sub.add_parser(
        "transcribe", help="Subtitle a video, or a folder of them"
    )
    speak.add_argument("source", type=Path)
    speak.add_argument("--locale", default="zh-TW")
    speak.add_argument("--output", type=Path)
    speak.add_argument("--budget", type=float, default=5.0)
    speak.add_argument("--upload-cache", type=Path)
    speak.set_defaults(handler=command_transcribe)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
