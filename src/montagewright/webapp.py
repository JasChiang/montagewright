"""Drop a folder in, watch it cut, read what it decided.

The CLI already prints every stage and writes the whole account to
report.json. What it cannot do is let someone check the result against the
material without a terminal and a video player: the question after a run is
never "did it finish", it is "which take is that, and why is it framed like
that". So this serves the finished film beside the decisions that produced
it -- source file, in and out, the move, the subject, the reason, the
degradations, and the per-shot verdict.

The run itself is the CLI in a subprocess. Importing the pipeline here would
be faster and would put a long-running job in the request thread; a
subprocess is killable, its stdout is already the progress log, and a crash
takes the run down instead of the server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from montagewright.uploads import default_library

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV", ".avi", ".mkv"}
AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".aiff", ".MP3", ".M4A", ".WAV"}
# The names a request may ask for, and what each one is as a ratio. These
# were two different shapes with one name -- a tuple to validate against and
# a dict to look up -- and the lookup silently returned nothing.
ASPECTS = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 4 / 5}
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

# Runs live somewhere they survive a restart. They were in a temp directory
# keyed by an in-memory dict, so closing the server threw away every finished
# cut -- and comparing this run against the last one is most of what anybody
# does with a tool like this.
RUNS_ROOT = Path(
    os.environ.get("MONTAGEWRIGHT_RUNS", Path.home() / ".cache" / "montagewright" / "runs")
)
# A browser upload of local material copies it into the browser and writes it
# back out; the bytes were already on disk. Uploading stays for the case where
# they genuinely are not, and that case has a ceiling.
MAX_UPLOAD_BYTES = int(os.environ.get("MONTAGEWRIGHT_MAX_UPLOAD", 4 * 1024**3))


@dataclass
class Run:
    """One cut in progress or finished, and everything it said on the way."""

    run_id: str
    root: Path
    lines: list[str] = field(default_factory=list)
    state: str = "running"
    returncode: int | None = None
    process: subprocess.Popen | None = None
    started_at: float = field(default_factory=time.time)
    source: str = ""
    # What was run, so it can be run again into the same place. Everything
    # already paid for -- cards, transcripts, the direction, the selection --
    # is keyed on disk, so a second attempt picks up where the first stopped.
    command: list[str] = field(default_factory=list)

    @property
    def output(self) -> Path:
        return self.root / "out"

    def remember(self) -> None:
        """Write enough to rebuild this run after a restart."""

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "run.json").write_text(
            json.dumps({
                "run_id": self.run_id,
                "state": self.state,
                "started_at": self.started_at,
                "source": self.source,
                "command": self.command,
                "log": self.lines,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    def report(self) -> dict | None:
        path = self.output / "report.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Being written right now. A half-read report is not an error,
            # it is a "not yet".
            return None


RUNS: dict[str, Run] = {}


def recall() -> None:
    """Pick up runs left by an earlier server."""

    if not RUNS_ROOT.exists():
        return
    for folder in sorted(RUNS_ROOT.iterdir()):
        if folder.name in RUNS or not folder.is_dir():
            continue
        # Either the note this server wrote, or the one the command line
        # leaves beside its output. A cut is a cut however it was started.
        note = folder / "run.json"
        spare = folder / "out" / "command.json"
        if not note.exists() and not spare.exists():
            continue
        try:
            saved = json.loads(
                (note if note.exists() else spare).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not note.exists():
            saved.setdefault("state", "done")
            saved.setdefault(
                "started_at", (folder / "out").stat().st_mtime
            )
        RUNS[folder.name] = Run(
            run_id=folder.name,
            root=folder,
            lines=saved.get("log", []),
            # Anything found on disk is finished as far as this process is
            # concerned: the thing that was running died with the last server.
            state=saved.get("state", "done")
            if saved.get("state") != "running"
            else "interrupted",
            started_at=float(saved.get("started_at", 0.0)),
            source=saved.get("source", ""),
            command=saved.get("command", []),
        )


def _collect(run: Run) -> None:
    """Drain the child's output into the run, and mark how it ended."""

    assert run.process is not None and run.process.stdout is not None
    for line in run.process.stdout:
        text = line.rstrip("\n")
        if text:
            run.lines.append(text)
    run.returncode = run.process.wait()
    run.state = "done" if run.returncode == 0 else "failed"
    run.remember()


def _transcript_map(run) -> dict:
    """Which transcript belongs to which source, for this run.

    Transcripts moved into the shared library when they became worth keeping
    across runs, and they are named for the bytes they describe -- the same
    rule as the cards, for the same reason. Two readers here were still
    looking in the output directory, which nothing has written since, and
    keying by filename, which is a hash. So they found nothing and said
    nothing: the transcript tab was empty for every run, and the subtitles
    were empty for every run, and both looked like "this cut has no speech".
    """

    from montagewright.clipcard import card_map
    from montagewright.transcript import load

    found = card_map(
        run.output / "work" / "proxies",
        default_library() / "transcripts",
    )
    return {
        source_id: card
        for source_id, path in found.items()
        if (card := load(path)) is not None
    }


def _subtitle_lines(run, *, edits: bool = True) -> "list":
    """What should appear on screen, edits included.

    The derived lines come from the transcripts and the running order. An
    edited set, if there is one, replaces them wholesale -- it was derived
    from the same cut and then corrected, so merging the two would mean
    re-deriving a line somebody had already fixed.
    """

    from montagewright.transcript import Line, against_cut

    edited = run.output / "work" / "subtitles.json"
    if edits and edited.exists():
        try:
            saved = json.loads(edited.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = []
        if saved:
            return [
                Line(
                    text=str(one.get("text", "")),
                    starts_seconds=float(one.get("at", 0.0)),
                    ends_seconds=float(one.get("until", 0.0)),
                    heard=str(one.get("heard", "")),
                    speaker=str(one.get("speaker", "")),
                )
                for one in saved
            ]

    report = run.report() or {}
    return against_cut(
        report.get("selection", {}).get("shots", []),
        report.get("rhythm", {}),
        _transcript_map(run),
    )


class _AlreadyHave(Exception):
    """The run recorded its crops, so there is nothing to rebuild."""


def _typed_path(raw: str) -> Path | None:
    """Whatever a person pasted, as a path.

    Dragging out of Finder or copying from a browser gives a `file://` URL
    with the spaces and the Chinese percent-encoded; quoting a path in a
    terminal leaves the quotes on. All of that arrives here looking like a
    path and is not one -- `Path("file:/Users/...")` is a relative directory
    called "file:", so the run started, spent four minutes writing cards, and
    only then failed on a track that was never there.
    """

    from urllib.parse import unquote, urlparse

    text = raw.strip().strip('"').strip("'")
    if not text:
        return None
    if text.startswith("file:"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
    return Path(text).expanduser()


def _save(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def create_app() -> FastAPI:
    app = FastAPI(title="montagewright")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.post("/api/runs")
    async def start(
        rushes: list[UploadFile] | None = None,
        music: UploadFile | None = None,
        source_path: str = Form(""),
        music_path: str = Form(""),
        brief: str = Form(""),
        aspect: str = Form("9:16"),
        budget: float = Form(6.0),
        review: bool = Form(True),
        timeline: str = Form("none"),
        speech: str = Form("auto"),
        locale: str = Form("zh-TW"),
    ) -> JSONResponse:
        if aspect not in ASPECTS:
            raise HTTPException(
                400, f"aspect must be one of {sorted(ASPECTS)}"
            )

        run_id = uuid.uuid4().hex[:12]
        root = RUNS_ROOT / run_id
        root.mkdir(parents=True, exist_ok=True)

        # A path is the ordinary case: this runs beside the material, and
        # pushing a folder of 4K through the browser to write it back to disk
        # a directory away is work nobody asked for.
        typed = _typed_path(source_path)
        if typed is not None:
            rush_dir = typed
            if not rush_dir.exists():
                raise HTTPException(400, f"{rush_dir} is not there")
            if rush_dir.is_file():
                holder = root / "rushes"
                holder.mkdir(parents=True, exist_ok=True)
                link = holder / rush_dir.name
                if not link.exists():
                    try:
                        os.link(rush_dir, link)
                    except OSError:
                        link.symlink_to(rush_dir)
                rush_dir = holder
            kept = sum(
                1 for path in rush_dir.iterdir()
                if path.suffix in VIDEO_SUFFIXES
            )
        else:
            rush_dir = root / "rushes"
            rush_dir.mkdir(parents=True, exist_ok=True)
            kept = 0
            budgeted = MAX_UPLOAD_BYTES
            for upload in rushes or []:
                # A folder upload arrives with its paths; only the leaf
                # matters, and anything that is not footage is not ours to
                # guess about.
                name = Path(upload.filename or "").name
                if not name or Path(name).suffix not in VIDEO_SUFFIXES:
                    continue
                written = _save(upload, rush_dir / name)
                budgeted -= written.stat().st_size
                if budgeted < 0:
                    shutil.rmtree(root, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"more than {MAX_UPLOAD_BYTES // 1024**3} GB uploaded; "
                        "give a path on this machine instead",
                    )
                kept += 1
        if not kept:
            raise HTTPException(400, "no video files there")

        command = [
            sys.executable, "-u", "-m", "montagewright.cli", "render",
            str(rush_dir), "--aspect", aspect,
            "--budget", str(budget),
            "--output", str(root / "out"),
        ]
        track = _typed_path(music_path)
        if track is not None:
            if not track.exists():
                shutil.rmtree(root, ignore_errors=True)
                raise HTTPException(400, f"{track} is not there")
            command += ["--music", str(track)]
        elif music is not None and music.filename:
            track = _save(music, root / Path(music.filename).name)
            command += ["--music", str(track)]
        if speech in {"auto", "never"}:
            command += ["--speech", speech]
        if locale.strip():
            command += ["--locale", locale.strip()]
        if brief.strip():
            brief_path = root / "brief.md"
            brief_path.write_text(brief, encoding="utf-8")
            command += ["--brief", str(brief_path)]
        if review:
            command += ["--review"]
        if timeline in {"premiere", "finalcut", "both"}:
            command += ["--timeline", timeline]
        checkpoint = Path("artifacts/models/sam2.1_hiera_tiny.pt").resolve()
        if checkpoint.exists():
            command += ["--sam-checkpoint", str(checkpoint)]

        run = Run(
            run_id=run_id, root=root, source=str(rush_dir), command=command
        )
        run.lines.append(f"{kept} clips from {rush_dir}")
        run.remember()
        run.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        RUNS[run_id] = run
        threading.Thread(target=_collect, args=(run,), daemon=True).start()
        return JSONResponse({"run_id": run_id})

    @app.get("/api/browse")
    def browse(path: str = "", kind: str = "video") -> JSONResponse:
        """List folders so a path can be clicked instead of typed.

        A browser will not hand over a real filesystem path -- a directory
        picker gives relative names and nothing else -- so typing one out was
        the only way to point this at material sitting next to the server.
        This serves the listing instead. It binds to localhost, and the person
        using it owns the disk.
        """

        here = Path(path).expanduser() if path.strip() else Path.home()
        try:
            here = here.resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(404, f"{path} is not there")
        if not here.is_dir():
            here = here.parent

        looking = AUDIO_SUFFIXES if kind == "audio" else VIDEO_SUFFIXES
        folders = []
        for entry in sorted(here.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                clips = sum(
                    1 for child in entry.iterdir()
                    if child.suffix in looking
                )
            except PermissionError:
                continue
            folders.append({"name": entry.name, "path": str(entry), "clips": clips})
        loose = [
            {"name": entry.name, "path": str(entry)}
            for entry in sorted(here.iterdir())
            if entry.is_file() and entry.suffix in looking
        ]
        return JSONResponse({
            "here": str(here),
            "parent": str(here.parent) if here.parent != here else None,
            "folders": folders,
            "videos": loose,
        })

    @app.get("/api/runs")
    def history(limit: int = 30) -> JSONResponse:
        """Earlier cuts, so this one can be compared against them."""

        recall()
        rows = sorted(
            RUNS.values(), key=lambda run: run.started_at, reverse=True
        )[:limit]
        out = []
        for run in rows:
            report = run.report() or {}
            out.append({
                "run_id": run.run_id,
                "state": run.state,
                "started_at": run.started_at,
                "source": Path(run.source).name if run.source else "",
                "source_path": run.source,
                "seconds": report.get("duration_seconds"),
                "shots": len(report.get("selection", {}).get("shots", [])),
                "spend": round(
                    sum(report.get("spend", {}).get("by_stage", {}).values()), 4
                ) or None,
                "delivered": sum(
                    1 for entry in (report.get("shots") or {}).values()
                    if entry.get("delivered")
                ) or None,
            })
        return JSONResponse({"runs": out})

    @app.get("/api/runs/{run_id}/waveform/{which}")
    def waveform(run_id: str, which: str, width: int = 2000):
        """The sound as a picture, so the tracks can be read.

        A cut with speech under music is two things happening at once and the
        page could only play it. Seeing where the voice sits and where the bed
        steps back is the difference between trusting the mix and checking
        it.
        """

        run = _run(run_id)
        if which == "voice":
            # picture.mp4 is the cut before the bed goes under it, so it is
            # the voice alone. Older runs kept only the deliverable.
            source = next(
                (
                    run.output / name
                    for name in ("picture.mp4", "deliverable.mp4")
                    if (run.output / name).exists()
                ),
                run.output / "picture.mp4",
            )
        elif which == "music":
            source = None
            if "--music" in run.command:
                candidate = Path(run.command[run.command.index("--music") + 1])
                source = candidate if candidate.exists() else None
            if source is None:
                raise HTTPException(404, "this cut has no music")
        else:
            raise HTTPException(400, "voice or music")
        if not source.exists():
            raise HTTPException(404, f"no {which} to draw")

        width = max(400, min(width, 6000))
        drawn = run.output / f"wave-{which}-{width}.png"
        if not drawn.exists():
            report = run.report() or {}
            seconds = float(report.get("duration_seconds") or 0.0)
            trim = ["-t", f"{seconds:.3f}"] if seconds and which == "music" else []
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
                + trim + ["-i", str(source), "-filter_complex",
                          f"aformat=channel_layouts=mono,"
                          f"showwavespic=s={width}x120:colors=#8b8880",
                          "-frames:v", "1", str(drawn)],
                check=False,
            )
        if not drawn.exists():
            raise HTTPException(500, "could not draw it")
        return FileResponse(drawn, media_type="image/png")

    @app.get("/api/runs/{run_id}/timeline-data")
    def timeline_data(run_id: str) -> JSONResponse:
        """Each shot's place on the timeline, and how far it can be pulled.

        The handles were rendered for exactly this and nothing could reach
        them: half a second either side, invisible, unusable.
        """

        from montagewright.pipeline import probe

        run = _run(run_id)
        report = run.report() or {}
        shots = report.get("selection", {}).get("shots", [])
        rhythm = report.get("rhythm", {})
        verdicts = report.get("shots", {})

        found: dict[str, float] = {}
        crops: dict[str, list] = {}

        # What the render actually used, if the run left it behind. Only a
        # held frame can be re-derived afterwards -- a follow came out of a
        # propagation nothing here can repeat -- so recomputing was the
        # interface drawing every move as a static box and presenting it as
        # evidence. Runs from before this was written still fall through.
        recorded = False
        trail = run.output / "work" / "crops.json"
        if trail.exists():
            try:
                crops = json.loads(trail.read_text(encoding="utf-8"))
                recorded = bool(crops)
            except (OSError, ValueError) as error:
                print(f"timeline-data: unreadable crops.json ({error})", True)

        try:
            if crops:
                raise _AlreadyHave
            plan, _, _ = _rebuild(run)
            for segment in plan.segments:
                path = segment.crop_path
                keys = (
                    [(k.seconds, k.crop) for k in path.keyframes]
                    if path is not None
                    else ([(0.0, segment.crop)] if segment.crop else [])
                )
                crops[segment.clip_id] = [
                    {
                        "at": round(at, 3), "x": round(box.x, 5),
                        "y": round(box.y, 5), "w": round(box.width, 5),
                        "h": round(box.height, 5),
                    }
                    for at, box in keys
                ]
        except _AlreadyHave:
            pass
        except Exception as error:
            # The reel is still useful without the boxes, but a silent except
            # here is how "0/13 have a crop path" looked like a fact about the
            # cut rather than a broken lookup.
            print(f"timeline-data: no crop paths ({error})", flush=True)
            crops = {}

        blocks, cursor = [], 0.0
        for index, shot in enumerate(shots):
            key = f"k{index:02d}"
            seconds = float(rhythm.get(key, {}).get("seconds", 0.0))
            source_id = shot.get("source_id", "")
            if source_id not in found:
                match = next(
                    (
                        path for path in
                        list((run.output / "work" / "shots").glob("*"))
                        + list(Path(run.source).glob("*"))
                        if path.stem == source_id
                    ),
                    None,
                )
                found[source_id] = (
                    probe(source_id, match).duration_seconds if match else 0.0
                )
            blocks.append({
                "clip_id": key,
                # Which shot of the report this is, which is what the take
                # endpoint is keyed by. Its position in the reel is not the
                # same number the moment anything is reordered or dropped,
                # and peeking at the take asked for /source/undefined.
                "index": index,
                # Where the crop actually sat, keyframe by keyframe. Without
                # it "it followed the subject" is a claim in a report; with
                # it you can watch the box move over the original.
                "crop": crops.get(key, []),
                "source_id": source_id,
                "at": round(cursor, 3),
                "seconds": round(seconds, 3),
                "in_seconds": float(shot.get("start_seconds", 0.0)),
                "source_seconds": round(found[source_id], 3),
                "subject": shot.get("subject", ""),
                "camera_move": shot.get("camera_move", "hold"),
                "why": shot.get("why", ""),
                "delivered": verdicts.get(key, {}).get("delivered"),
                "note": verdicts.get(key, {}).get("note", ""),
            })
            cursor += seconds
        return JSONResponse({
            "blocks": blocks,
            "seconds": round(cursor, 3),
            # Whether these boxes are what the render used or the best that
            # could be worked out afterwards. A rebuilt box is a guess with
            # the same shape as evidence, and this view exists to be
            # evidence -- so it has to say which it is holding.
            "crops_are": "recorded" if recorded else "rebuilt",
        })

    def _rebuild(run: Run, wanted: list[dict] | None = None):
        """The render plan again, from what the report already records.

        Nothing here costs anything: the subject positions come out of the
        cards, so this is arithmetic. It is what lets a timeline be asked for
        after the fact, and a running order be changed without re-planning
        the film.
        """

        from montagewright.clipcard import card_map
        from montagewright.executor import plan_render
        from montagewright.pipeline import Report, follow_subjects, probe
        from montagewright.schema import EDL, Clip, reframe_of

        report = run.report() or {}
        original = report.get("selection", {}).get("shots", [])
        rhythm = report.get("rhythm", {})
        if wanted is None:
            wanted = [
                {
                    "index": i,
                    "in_seconds": float(shot.get("start_seconds", 0.0)),
                    "seconds": float(
                        rhythm.get(f"k{i:02d}", {}).get("seconds", 0.0)
                    ),
                }
                for i, shot in enumerate(original)
            ]
        aspect = ASPECTS.get(
            report.get("direction", {}).get("aspect", "9:16"), 9 / 16
        )
        cards = card_map(
            run.output / "work" / "proxies",
            default_library() / "cards",
        )
        clips, sources = [], {}
        for index, entry in enumerate(wanted):
            plan = original[int(entry["index"])]
            source_id = plan["source_id"]
            if source_id not in sources:
                match = next(
                    (
                        path for path in
                        list((run.output / "work" / "shots").glob("*"))
                        + list(Path(run.source).glob("*"))
                        if path.stem == source_id
                    ),
                    None,
                )
                if match is None:
                    raise HTTPException(404, f"{source_id} is gone")
                sources[source_id] = probe(source_id, match)
            start = float(entry["in_seconds"])
            clips.append(Clip(
                clip_id=f"k{index:02d}", source_id=source_id,
                approx_in_seconds=start,
                approx_out_seconds=start + float(entry["seconds"]),
                in_looks_like=plan.get("subject", ""),
                energy_intent=plan.get("energy", "medium"),
                reframe=reframe_of(plan),
            ))
        edl = EDL(project_id=run.run_id, clips=clips)
        paths = follow_subjects(
            edl, sources, target_aspect=aspect, report=Report(),
            cards=cards, checkpoint=None, client=None,
        )
        return (
            plan_render(edl, sources, target_aspect=aspect, crop_paths=paths),
            report, aspect,
        )

    @app.post("/api/runs/{run_id}/recut")
    async def recut(run_id: str, request: Request) -> JSONResponse:
        """Render an amended running order. No model calls, so no cost.

        Everything that was decided stays decided -- the crop, the subject,
        the move. This moves cuts and drops shots, which is the part a person
        wants after watching it once and does not want to re-plan a whole
        film for.
        """

        from montagewright.renderer import render as render_cut

        run = _run(run_id)
        wanted = (await request.json()).get("shots", [])
        if not wanted:
            raise HTTPException(400, "nothing left to cut")
        plan, report, _ = _rebuild(run, wanted)

        # The bed and whether the voice survives were decided when the run
        # was set up; a recut is a different running order, not a different
        # film, so both carry over.
        music = None
        if "--music" in run.command:
            candidate = Path(run.command[run.command.index("--music") + 1])
            music = candidate if candidate.exists() else None
        result = render_cut(
            plan, run.output, music=music, keep_segments=True,
            keep_voice=bool(
                list((default_library() / "transcripts").glob("*.json"))
            ),
            under_speech=str(
                report.get("direction", {}).get("music_under_speech") or "duck"
            ),
        )
        # What was rendered, from the plan that rendered it. This read a
        # name belonging to the rebuild's own scope, so a recut that had
        # already re-encoded every segment raised on the way to saying so.
        return JSONResponse({
            "seconds": round(result.duration_seconds, 3),
            "shots": len(plan.segments),
        })

    @app.get("/api/runs/{run_id}/transcripts")
    def transcripts(run_id: str) -> JSONResponse:
        """What was heard, per source, before anything was cut.

        Worth reading before the editorial passes are paid for: a transcript
        that got the product name wrong sends every later decision after the
        wrong sentence.
        """

        from montagewright.transcript import lines_of

        run = _run(run_id)
        out = {}
        for source_id, card in sorted(_transcript_map(run).items()):
            out[source_id] = {
                "language": card.get("language"),
                "summary": card.get("summary", ""),
                "lines": [
                    {
                        "text": line.text,
                        "heard": line.heard,
                        "speaker": line.speaker,
                        "starts_seconds": line.starts_seconds,
                        "ends_seconds": line.ends_seconds,
                        "corrected": line.corrected,
                    }
                    for line in lines_of(card)
                ],
            }
        return JSONResponse({"sources": out})

    def _run(run_id: str) -> Run:
        run = RUNS.get(run_id)
        if run is None:
            # Runs are picked up off disk lazily, and only the listing was
            # doing it -- so after a restart every other endpoint answered
            # "no such run" until something happened to ask for the list.
            recall()
            run = RUNS.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    @app.get("/api/runs/{run_id}")
    def status(run_id: str, since: int = 0) -> JSONResponse:
        run = _run(run_id)
        return JSONResponse({
            "state": run.state,
            "returncode": run.returncode,
            "lines": run.lines[since:],
            "total_lines": len(run.lines),
            "report": run.report() if run.state != "running" else None,
            "has_video": (run.output / "preview.mp4").exists(),
        })

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str) -> JSONResponse:
        """Run it again into the same place.

        Nothing already paid for is paid for twice: the cards, transcripts,
        the direction and the selection are all keyed on disk, so this picks
        up where the quota or the bad path stopped it.
        """

        run = _run(run_id)
        if not run.command:
            # Recorded before the command was kept. What carries the value is
            # the work directory -- cards, transcripts, the direction, the
            # selection -- and that belongs to the output, so a plain run
            # pointed back at it picks all of them up. The options it was
            # started with are lost; the money is not.
            if not run.source or not Path(run.source).exists():
                raise HTTPException(
                    400, "this run did not record how it started"
                )
            run.command = [
                sys.executable, "-u", "-m", "montagewright.cli", "render",
                run.source, "--output", str(run.output), "--review",
            ]
            run.lines.append(
                "— 這一輪是舊版存的，沒有記下當初的選項；"
                "用預設值續跑，已經算好的東西都會沿用 —"
            )
        if run.process is not None and run.process.poll() is None:
            raise HTTPException(409, "it is still going")
        run.lines.append("— 續跑 —")
        run.state = "running"
        run.process = subprocess.Popen(
            run.command,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        run.remember()
        threading.Thread(target=_collect, args=(run,), daemon=True).start()
        return JSONResponse({"state": run.state})

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str) -> JSONResponse:
        run = _run(run_id)
        if run.process is not None and run.process.poll() is None:
            run.process.terminate()
            run.state = "stopped"
        return JSONResponse({"state": run.state})

    @app.get("/api/runs/{run_id}/video")
    def video(run_id: str):
        path = _run(run_id).output / "preview.mp4"
        if not path.exists():
            raise HTTPException(404, "no preview yet")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/deliverable")
    def deliverable(run_id: str):
        path = _run(run_id).output / "deliverable.mp4"
        if not path.exists():
            raise HTTPException(404, "no deliverable yet")
        return FileResponse(
            path, media_type="video/mp4", filename=f"{run_id}.mp4"
        )

    @app.get("/api/runs/{run_id}/timeline/{flavour}")
    def timeline(run_id: str, flavour: str):
        suffix = {"premiere": "xml", "finalcut": "fcpxml"}.get(flavour)
        if suffix is None:
            raise HTTPException(400, "premiere or finalcut")
        run = _run(run_id)
        path = run.output / f"timeline.{suffix}"
        if not path.exists():
            # Built on request. Asking for one up front and finding out
            # afterwards that you wanted it meant running the whole thing
            # again, and everything it needs is already written down.
            from montagewright.timeline import to_fcpxml, to_xmeml

            plan, report, aspect = _rebuild(run)
            width, height = (1920, 1080) if aspect >= 1.0 else (1080, 1920)
            build = to_xmeml if flavour == "premiere" else to_fcpxml
            path.write_text(
                build(plan, report, name=run.output.name,
                      width=width, height=height),
                encoding="utf-8",
            )
        return FileResponse(
            path, media_type="application/xml",
            filename=f"{run_id}.{suffix}",
        )

    @app.get("/api/runs/{run_id}/subtitle-track")
    def subtitle_track(run_id: str) -> JSONResponse:
        """The subtitles as a track, so they can be read against the picture.

        Same lines the file gets. A line that is wrong is wrong in both, and
        the place to notice is next to the shot it is under.
        """

        run = _run(run_id)
        # What the transcripts and the running order produce, before anyone
        # touched it. Sent so the track can mark the lines that were changed
        # -- comparing against `heard` marked almost all of them, because
        # correcting what the recogniser misheard is the whole point of the
        # pass that produced them.
        derived = {
            round(line.starts_seconds, 3): line.text
            for line in _subtitle_lines(run, edits=False)
        }
        return JSONResponse({
            "lines": [
                {
                    "at": round(line.starts_seconds, 3),
                    "until": round(line.ends_seconds, 3),
                    "text": line.text,
                    "speaker": line.speaker,
                    "heard": line.heard,
                    "derived": derived.get(round(line.starts_seconds, 3), ""),
                }
                for line in _subtitle_lines(run)
            ],
            "edited": (run.output / "work" / "subtitles.json").exists(),
        })

    @app.put("/api/runs/{run_id}/subtitle-track")
    async def edit_subtitle_track(run_id: str, request: Request) -> JSONResponse:
        """Keep an edited set of lines beside the ones that were derived.

        Gemini fixes most of what the recogniser mishears and not all of it,
        and the name of a product is exactly the kind of word it gets wrong.
        Edits are kept separately rather than written back over the
        transcript: the transcript is what was heard, which stays true, and
        this is what should appear on screen.
        """

        run = _run(run_id)
        sent = (await request.json()).get("lines", [])
        kept = [
            {
                "at": round(float(one.get("at", 0.0)), 3),
                "until": round(float(one.get("until", 0.0)), 3),
                "text": str(one.get("text", "")),
                "speaker": str(one.get("speaker", "")),
                "heard": str(one.get("heard", "")),
            }
            for one in sent
            if str(one.get("text", "")).strip()
        ]
        kept.sort(key=lambda one: one["at"])
        destination = run.output / "work" / "subtitles.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return JSONResponse({"lines": len(kept)})

    @app.get("/api/runs/{run_id}/subtitles")
    def subtitles(run_id: str):
        """Every transcript in the run, as one SRT against the finished cut.

        The lines are timed against their own source, so they are shifted
        onto the timeline the shots landed on -- a subtitle file that needs
        the reader to work out which take a line came from is not one.
        """

        from montagewright.transcript import to_srt

        run = _run(run_id)
        timed = _subtitle_lines(run)
        if not timed:
            raise HTTPException(404, "nothing was transcribed in this run")
        path = run.output / "subtitles.srt"
        # Several people in one cut; the file says which is which.
        path.write_text(to_srt(timed, with_speaker=True), encoding="utf-8")
        return FileResponse(
            path, media_type="text/plain", filename=f"{run_id}.srt"
        )

    @app.get("/api/runs/{run_id}/source/{index}")
    def source_clip(run_id: str, index: int):
        """The take this shot was cut from, uncropped.

        Checking that a crop followed anything means seeing what it was
        moving across. The rendered segment cannot show that -- it is the
        answer, not the working.
        """

        run = _run(run_id)
        report = run.report() or {}
        shots = report.get("selection", {}).get("shots", [])
        if index >= len(shots):
            raise HTTPException(404, "no such shot")
        source_id = shots[index]["source_id"]

        # The proxy, when there is one. This view exists to show where the
        # crop sat, and the proxy is the same framing at a five-hundredth of
        # the bytes -- the original is 128MB of 4K that the browser has to
        # decode in full to draw a rectangle on, and every scrub reopens it.
        proxy = run.output / "work" / "proxies" / f"{source_id}.mp4"
        if proxy.exists():
            return FileResponse(proxy, media_type="video/mp4")

        match = next(
            (
                path for path in
                list((run.output / "work" / "shots").glob("*"))
                + list(Path(run.source).glob("*"))
                if path.stem == source_id
            ),
            None,
        )
        if match is None:
            raise HTTPException(404, f"{source_id} is gone")
        return FileResponse(match, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/shot/{index}")
    def shot(run_id: str, index: int):
        """One rendered shot on its own. Faults hide in the whole cut."""

        run = _run(run_id)
        matches = sorted((run.output / "segments").glob(f"{index:03d}-*.mp4"))
        real = [p for p in matches if not p.name.endswith(".handles.mp4")]
        if not real:
            raise HTTPException(404, "no such shot")
        return FileResponse(real[0], media_type="video/mp4")

    return app


app = create_app()


def main() -> int:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("MONTAGEWRIGHT_HOST", "127.0.0.1"),
        port=int(os.environ.get("MONTAGEWRIGHT_PORT", "8765")),
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
