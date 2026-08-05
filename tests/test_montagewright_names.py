"""Names that are used and never defined.

Three runs died on this in one afternoon, each after paying for cards,
direction and selection -- the point where the tool has spent everything and
delivered nothing. All three were a moved import or a moved variable, none
was reachable from a unit test without building most of a run, and every one
of them is visible from the source alone in under a second.

pyflakes is the whole check. This is here so it runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "montagewright"

# An unused import is a tidiness question and this is not about tidiness.
IGNORED = ("imported but unused", "unable to detect undefined names")


def _pyflakes() -> list[str]:
    found = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(PACKAGE)],
        capture_output=True,
        text=True,
    )
    if "No module named pyflakes" in found.stderr:
        pytest.skip("pyflakes is not installed here")
    return [
        line
        for line in (found.stdout + found.stderr).splitlines()
        if line.strip() and not any(skip in line for skip in IGNORED)
    ]


def test_nothing_uses_a_name_that_was_never_defined() -> None:
    complaints = [line for line in _pyflakes() if "undefined name" in line]
    assert not complaints, "\n".join(complaints)


def test_nothing_else_pyflakes_finds_either() -> None:
    """Kept separate so an undefined name is never lost in a longer list."""

    complaints = [line for line in _pyflakes() if "undefined name" not in line]
    assert not complaints, "\n".join(complaints)
