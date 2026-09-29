"""One isolated Codex repair for a Python wiring exception, then checkpointed resume.

Editorial rejection, provider failures and budget exhaustion never enter here.
The production process is not relaunched until the original test suite passes
against the staged source. No model is given rushes or a production command.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

from montagewright.checkpoints import read_json, write_json


ATTEMPT_ENV = "MONTAGEWRIGHT_TECHNICAL_REPAIR_ATTEMPT"


def eligible(error):
    return isinstance(error, (NameError, AttributeError, TypeError)) and any(
        "montagewright" in frame.filename and "site-packages" not in frame.filename
        for frame in traceback.extract_tb(error.__traceback__))


def codex_executable():
    override = os.environ.get("MONTAGEWRIGHT_CODEX_BIN")
    if override:
        return shutil.which(override)
    candidates = [shutil.which("codex"),
                  "/Applications/ChatGPT.app/Contents/Resources/codex",
                  "/Applications/Codex.app/Contents/Resources/codex"]
    available = []
    for candidate in dict.fromkeys(candidates):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            try:
                version = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=5)
                found = re.search(r"codex-cli (\d+)\.(\d+)\.(\d+)", version.stdout)
                if not version.returncode and found:
                    available.append((tuple(map(int, found.groups())), candidate))
            except (OSError, subprocess.TimeoutExpired):
                continue
    return max(available)[1] if available else None


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".txt", ".html"}}


def attempt(error, *, output, argv, repository=None):
    """Return resumed exit status, or None when no validated repair was applied."""
    if os.environ.get(ATTEMPT_ENV) or not eligible(error):
        return None
    repository = Path(repository or Path(__file__).resolve().parents[2])
    if not (repository / "tests").is_dir():
        return None  # packaged installs cannot self-validate a source patch
    executable = codex_executable()
    root = Path(output) / "work" / "technical-repair"
    status_path = root / "status.json"
    if read_json(status_path):
        return None  # one attempt per output, including interruption/failed repair
    if not executable:
        write_json(status_path, {"status": "unavailable", "reason": "codex executable unavailable"})
        return None
    before = _files(repository / "src")
    root.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"status": "repairing", "executable": executable, "billing": "Codex account usage, separate from Gemini ledger"})
    print("technical repair: Codex is checking an isolated source copy; paid video results are preserved", flush=True)
    with tempfile.TemporaryDirectory(prefix="montagewright-repair-") as scratch:
        stage = Path(scratch)
        for name in ("src", "tests", "scripts"):
            if (repository / name).is_dir():
                shutil.copytree(repository / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("pyproject.toml", "README.md"):
            if (repository / name).exists():
                shutil.copy2(repository / name, stage / name)
        schema = stage / "repair-schema.json"
        write_json(schema, {"type": "object", "properties": {"fixed": {"type": "boolean"},
            "explanation": {"type": "string"}}, "required": ["fixed", "explanation"], "additionalProperties": False})
        trace = "".join(traceback.format_exception(error)).replace(str(repository), "<project>")
        prompt = ("Fix only the Python wiring exception below in this isolated repository copy. "
            "Do not change editorial policies, Gemini/ASR authority, budgets, credentials, model selection, "
            "release gates or tests. Do not run Gemini, Codex children, render jobs or access production files. "
            "Edit only existing src/montagewright Python files. Preserve public contracts. "
            "Return fixed=false if this cannot be repaired as a narrow wiring fix.\n" + trace)
        try:
            with (root / "codex.jsonl").open("w") as log:
                response = subprocess.run([executable, "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
                    "--json", "--output-schema", str(schema), "--output-last-message", str(stage / "answer.json"),
                    "-C", str(stage), "-"], input=prompt, text=True, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            answer = read_json(stage / "answer.json") or {}
            if response.returncode or answer.get("fixed") is not True:
                raise ValueError("Codex did not produce a successful repair")
            after = _files(stage / "src")
            changed = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
            if not changed or len(changed) > 3 or any(key not in before or key not in after or not key.endswith(".py") for key in changed):
                raise ValueError("repair must change one to three existing Python files")
            protected = {"cost.py", "budget_plan.py", "gemini.py", "checkpoints.py", "release.py", "technical_repair.py"}
            if any(Path(key).name in protected for key in changed):
                raise ValueError("repair attempted to change a protected budget, API or release boundary")
            # Tests are evidence, not something the repair agent can weaken.
            shutil.rmtree(stage / "tests")
            shutil.copytree(repository / "tests", stage / "tests", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(repository / "pyproject.toml", stage / "pyproject.toml")
            environment = dict(os.environ, PYTHONPATH=str(stage / "src") + os.pathsep + str(stage), **{ATTEMPT_ENV: "1"})
            with (root / "validation.log").open("w") as log:
                validation = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=stage,
                    env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            if validation.returncode:
                raise ValueError("original test suite failed against the staged repair")
            if _files(repository / "src") != before:
                raise ValueError("production source changed concurrently; repair was not applied")
            for key in changed:
                backup = root / "before" / key
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(before[key])
            for key in changed:
                target = repository / "src" / key
                temporary = target.with_suffix(".repair-tmp")
                temporary.write_bytes(after[key])
                temporary.replace(target)
            usage = []
            for line in (root / "codex.jsonl").read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "turn.completed" and event.get("usage"):
                    usage.append(event["usage"])
            write_json(status_path, {"status": "applied", "files": changed, "explanation": answer.get("explanation"),
                "executable": executable, "usage": usage,
                "billing": "Codex account usage, separate from Gemini ledger",
                "original_sha256": {key: hashlib.sha256(before[key]).hexdigest() for key in changed}})
        except (OSError, ValueError, subprocess.TimeoutExpired) as failure:
            write_json(status_path, {"status": "failed", "reason": str(failure)})
            print(f"technical repair not applied: {failure}", flush=True)
            return None
    # Exception tracebacks retain the output lease. Release only the current
    # process's lock before handing execution to a fresh Python interpreter.
    from montagewright.release import OutputLease
    OutputLease(Path(output) / ".montagewright.lock").release()
    environment = dict(os.environ, PYTHONPATH=str(repository / "src") + os.pathsep + str(repository), **{ATTEMPT_ENV: "1"})
    result = subprocess.run([sys.executable, "-m", "montagewright.cli", *argv], cwd=repository, env=environment)
    write_json(root / "resume.json", {"returncode": result.returncode, "checkpointed": True})
    return result.returncode
