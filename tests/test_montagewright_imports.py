"""The boundary between deciding and measuring.

`montagewright.measure` holds the code that computes things -- shot
boundaries, beat grids, mask propagation, geometry. Everything above it
decides: what the film is, which shots, how long, what the camera does.

The whole design rests on those staying apart, and nothing enforces it at
runtime. A measurement module reaching up into the planner would be a
one-line mistake that reads like convenience, and it is how the previous
system arrived at a decision layer nobody could find the edges of. This test
is the enforcement.

It used to guard the other direction, against the new pipeline importing
decision code out of the old package. That package is gone; only its
measurements survive, and they live here now.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "montagewright"

# What the layer above is made of. A measurement that needs any of these is
# not a measurement.
DECIDING = frozenset(
    {
        "montagewright.planner",
        "montagewright.pipeline",
        "montagewright.review",
        "montagewright.clipcard",
        "montagewright.transcript",
        "montagewright.capabilities",
        "montagewright.cli",
        "montagewright.webapp",
    }
)


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            # A relative import inside measure/ stays inside measure/.
            if node.level or not node.module:
                continue
            found.add(node.module)
    return found


def test_nothing_that_measures_reaches_up_into_what_decides() -> None:
    offenders: dict[str, set[str]] = {}
    for source in sorted((PACKAGE / "measure").rglob("*.py")):
        reached = {
            name
            for name in _imports(source)
            if any(name == deciding or name.startswith(f"{deciding}.")
                   for deciding in DECIDING)
        }
        if reached:
            offenders[source.name] = reached
    assert not offenders, (
        f"measurement reaching into the decision layer: {offenders}"
    )


def test_the_guard_would_catch_it(tmp_path: Path) -> None:
    """The scan has to fail on the thing it exists to prevent."""

    offending = tmp_path / "tempting.py"
    offending.write_text(
        "from montagewright.planner import decide_rhythm\n"
        "import montagewright.review\n",
        encoding="utf-8",
    )
    reached = {
        name
        for name in _imports(offending)
        if any(name == d or name.startswith(f"{d}.") for d in DECIDING)
    }
    assert reached == {"montagewright.planner", "montagewright.review"}


def test_the_guard_allows_a_measurement_import(tmp_path: Path) -> None:
    permitted = tmp_path / "fine.py"
    permitted.write_text(
        "from montagewright.measure.shots import detect_shots_ffmpeg\n"
        "from montagewright.measure.media import probe_video\n"
        "from pathlib import Path\n",
        encoding="utf-8",
    )
    reached = {
        name
        for name in _imports(permitted)
        if any(name == d or name.startswith(f"{d}.") for d in DECIDING)
    }
    assert not reached


def test_the_old_package_is_gone() -> None:
    """Its measurements moved in; its decisions did not come with them."""

    assert not (PACKAGE.parent / "jascue_video_lab").exists()
    for source in sorted(PACKAGE.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        assert "jascue_video_lab" not in text, source.name
