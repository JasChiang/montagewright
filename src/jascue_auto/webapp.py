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

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV", ".avi", ".mkv"}
ASPECTS = ("9:16", "16:9", "1:1", "4:5")
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

# Runs live somewhere they survive a restart. They were in a temp directory
# keyed by an in-memory dict, so closing the server threw away every finished
# cut -- and comparing this run against the last one is most of what anybody
# does with a tool like this.
RUNS_ROOT = Path(
    os.environ.get("JASCUE_RUNS", Path.home() / ".cache" / "jascue-auto" / "runs")
)
# A browser upload of local material copies it into the browser and writes it
# back out; the bytes were already on disk. Uploading stays for the case where
# they genuinely are not, and that case has a ceiling.
MAX_UPLOAD_BYTES = int(os.environ.get("JASCUE_MAX_UPLOAD", 4 * 1024**3))


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
        note = folder / "run.json"
        if folder.name in RUNS or not note.exists():
            continue
        try:
            saved = json.loads(note.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
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


def _save(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def create_app() -> FastAPI:
    app = FastAPI(title="jascue-auto")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.post("/api/runs")
    async def start(
        rushes: list[UploadFile] = None,
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
            raise HTTPException(400, f"aspect must be one of {ASPECTS}")

        run_id = uuid.uuid4().hex[:12]
        root = RUNS_ROOT / run_id
        root.mkdir(parents=True, exist_ok=True)

        # A path is the ordinary case: this runs beside the material, and
        # pushing a folder of 4K through the browser to write it back to disk
        # a directory away is work nobody asked for.
        if source_path.strip():
            rush_dir = Path(source_path.strip()).expanduser()
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
            sys.executable, "-u", "-m", "jascue_auto.cli", "render",
            str(rush_dir), "--aspect", aspect,
            "--budget", str(budget),
            "--output", str(root / "out"),
        ]
        if music_path.strip():
            command += ["--music", str(Path(music_path.strip()).expanduser())]
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

        run = Run(run_id=run_id, root=root, source=str(rush_dir))
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

    @app.get("/api/runs/{run_id}/transcripts")
    def transcripts(run_id: str) -> JSONResponse:
        """What was heard, per source, before anything was cut.

        Worth reading before the editorial passes are paid for: a transcript
        that got the product name wrong sends every later decision after the
        wrong sentence.
        """

        from jascue_auto.transcript import lines_of, load

        run = _run(run_id)
        folder = run.output / "work" / "transcripts"
        out = {}
        for path in sorted(folder.glob("*.json")):
            card = load(path)
            if card is None:
                continue
            out[path.stem] = {
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
        path = _run(run_id).output / f"timeline.{suffix}"
        if not path.exists():
            raise HTTPException(404, "no timeline was asked for")
        return FileResponse(
            path, media_type="application/xml",
            filename=f"{run_id}.{suffix}",
        )

    @app.get("/api/runs/{run_id}/subtitles")
    def subtitles(run_id: str):
        """Every transcript in the run, as one SRT against the finished cut.

        The lines are timed against their own source, so they are shifted
        onto the timeline the shots landed on -- a subtitle file that needs
        the reader to work out which take a line came from is not one.
        """

        from jascue_auto.transcript import Line, lines_of, load, to_srt

        run = _run(run_id)
        report = run.report() or {}
        shots = report.get("selection", {}).get("shots", [])
        rhythm = report.get("rhythm", {})
        cards = {
            path.stem: load(path)
            for path in (run.output / "work" / "transcripts").glob("*.json")
        }
        timed: list[Line] = []
        cursor = 0.0
        for index, shot in enumerate(shots):
            seconds = float(
                rhythm.get(f"k{index:02d}", {}).get("seconds", 0.0)
            )
            card = cards.get(shot.get("source_id", ""))
            start = float(shot.get("start_seconds", 0.0))
            for line in lines_of(card or {}):
                if line.ends_seconds <= start:
                    continue
                if line.starts_seconds >= start + seconds:
                    continue
                timed.append(
                    Line(
                        text=(
                            f"{line.speaker}：{line.text}"
                            if line.speaker
                            else line.text
                        ),
                        starts_seconds=cursor
                        + max(0.0, line.starts_seconds - start),
                        ends_seconds=cursor
                        + min(seconds, line.ends_seconds - start),
                        heard=line.heard,
                    )
                )
            cursor += seconds
        if not timed:
            raise HTTPException(404, "nothing was transcribed in this run")
        path = run.output / "subtitles.srt"
        path.write_text(to_srt(timed), encoding="utf-8")
        return FileResponse(
            path, media_type="text/plain", filename=f"{run_id}.srt"
        )

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
        host=os.environ.get("JASCUE_HOST", "127.0.0.1"),
        port=int(os.environ.get("JASCUE_PORT", "8765")),
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
