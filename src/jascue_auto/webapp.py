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
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV", ".avi", ".mkv"}
ASPECTS = ("9:16", "16:9", "1:1", "4:5")
PAGE = Path(__file__).resolve().parent / "web" / "index.html"


@dataclass
class Run:
    """One cut in progress or finished, and everything it said on the way."""

    run_id: str
    root: Path
    lines: list[str] = field(default_factory=list)
    state: str = "running"
    returncode: int | None = None
    process: subprocess.Popen | None = None

    @property
    def output(self) -> Path:
        return self.root / "out"

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


def _collect(run: Run) -> None:
    """Drain the child's output into the run, and mark how it ended."""

    assert run.process is not None and run.process.stdout is not None
    for line in run.process.stdout:
        text = line.rstrip("\n")
        if text:
            run.lines.append(text)
    run.returncode = run.process.wait()
    run.state = "done" if run.returncode == 0 else "failed"


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
        rushes: list[UploadFile],
        music: UploadFile | None = None,
        brief: str = Form(""),
        aspect: str = Form("9:16"),
        budget: float = Form(6.0),
        review: bool = Form(True),
    ) -> JSONResponse:
        if aspect not in ASPECTS:
            raise HTTPException(400, f"aspect must be one of {ASPECTS}")

        run_id = uuid.uuid4().hex[:12]
        root = Path(tempfile.mkdtemp(prefix=f"jascue-{run_id}-"))
        rush_dir = root / "rushes"
        rush_dir.mkdir(parents=True)

        kept = 0
        for upload in rushes:
            # A folder upload arrives with its paths; only the leaf matters,
            # and anything that is not footage is not ours to guess about.
            name = Path(upload.filename or "").name
            if not name or Path(name).suffix not in VIDEO_SUFFIXES:
                continue
            _save(upload, rush_dir / name)
            kept += 1
        if not kept:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(400, "no video files in what was uploaded")

        command = [
            sys.executable, "-u", "-m", "jascue_auto.cli", "render",
            str(rush_dir), "--aspect", aspect,
            "--budget", str(budget),
            "--output", str(root / "out"),
        ]
        if music is not None and music.filename:
            track = _save(music, root / Path(music.filename).name)
            command += ["--music", str(track)]
        if brief.strip():
            brief_path = root / "brief.md"
            brief_path.write_text(brief, encoding="utf-8")
            command += ["--brief", str(brief_path)]
        if review:
            command += ["--review"]
        checkpoint = Path("artifacts/models/sam2.1_hiera_tiny.pt").resolve()
        if checkpoint.exists():
            command += ["--sam-checkpoint", str(checkpoint)]

        run = Run(run_id=run_id, root=root)
        run.lines.append(f"{kept} clips uploaded")
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
