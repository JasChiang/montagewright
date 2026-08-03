"""The import boundary between the new pipeline and the old package.

`jascue_video_lab` holds two very different kinds of code. Some of it computes
things -- shot boundaries, beat grids, quality measurements, mask propagation
-- and is worth keeping. The rest decides whether work should happen at all,
and that is precisely what the rebuild replaces.

Nothing enforces the difference at runtime, and the two live in the same
package, so reaching for a decision module is a one-line mistake that reads
like reuse. This test is the enforcement.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "jascue_auto"

# Modules that calculate. Each is imported as a library; §2.1 of the work
# order also strips the internal safety margins on the way in, leaving one
# outermost bound rather than several compounding ones.
ALLOWED = frozenset(
    {
        "jascue_video_lab.geometry",
        "jascue_video_lab.media",
        "jascue_video_lab.music",
        "jascue_video_lab.music_cues",
        "jascue_video_lab.sam_tracking",
        "jascue_video_lab.shot_quality",
        "jascue_video_lab.shots",
    }
)


def _imported_old_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("jascue_video_lab"):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module of ours to check.
            if node.level or not node.module:
                continue
            if node.module.startswith("jascue_video_lab"):
                found.add(node.module)
    return found


def _python_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_the_package_exists() -> None:
    """A vacuous pass would be indistinguishable from a passing scan."""

    assert _python_sources(), f"no sources found under {PACKAGE_ROOT}"


@pytest.mark.parametrize(
    "source", _python_sources(), ids=lambda path: path.name
)
def test_only_calculation_modules_are_imported(source: Path) -> None:
    offenders = _imported_old_modules(source) - ALLOWED
    assert not offenders, (
        f"{source.relative_to(PACKAGE_ROOT)} imports {sorted(offenders)} from "
        "the old package. Only the calculation modules are carried over; "
        "anything that decides whether work happens is rewritten. If one of "
        "these really does just compute, move it to ALLOWED deliberately and "
        "say why in the commit."
    )


def test_the_guard_would_catch_a_decision_import(tmp_path: Path) -> None:
    """The scan has to fail on the thing it exists to prevent."""

    offending = tmp_path / "tempting.py"
    offending.write_text(
        "from jascue_video_lab.feature_cut import run_feature_cut\n"
        "import jascue_video_lab.reframe_policy\n",
        encoding="utf-8",
    )
    assert _imported_old_modules(offending) - ALLOWED == {
        "jascue_video_lab.feature_cut",
        "jascue_video_lab.reframe_policy",
    }


def test_the_guard_allows_a_calculation_import(tmp_path: Path) -> None:
    permitted = tmp_path / "fine.py"
    permitted.write_text(
        "from jascue_video_lab.shots import detect_shots_ffmpeg\n"
        "from jascue_video_lab.media import probe_video\n"
        "from pathlib import Path\n",
        encoding="utf-8",
    )
    assert not _imported_old_modules(permitted) - ALLOWED
