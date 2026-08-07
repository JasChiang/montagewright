"""The interface has to parse before any assertion about it means anything.

Every other test about this page reads it as text: does it contain this
function, does it mention that field. All of those pass on a file that the
browser refuses to run, and that is not a hypothetical -- an HTML comment was
added inside a JavaScript template literal, the backticks in its prose closed
the string early, and the whole application failed to load. The suite stayed
green, because the string it was looking for was right there in the file.

`node --check` is the whole check. This is here so it runs.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "src" / "montagewright" / "web" / "index.html"
SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.S)


def _node() -> str:
    found = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if found.returncode != 0:
        pytest.skip("node is not installed here")
    return found.stdout.strip()


def test_the_page_javascript_parses():
    _node()
    body = "\n".join(SCRIPT.findall(PAGE.read_text(encoding="utf-8")))
    assert body.strip(), "the page has no script to check"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(body)
        where = handle.name
    checked = subprocess.run(
        ["node", "--check", where], capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr


def test_the_check_would_have_caught_a_backtick_in_a_comment():
    """The exact mistake, so this guards the thing it was written for."""

    _node()
    broken = (
        "function inspect() {\n"
        "  return `<span>${esc(x)}</span>\n"
        "    <!-- a note mentioning `camera_move` in prose -->\n"
        "    <span>${esc(y)}</span>`;\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(broken)
        where = handle.name
    checked = subprocess.run(
        ["node", "--check", where], capture_output=True, text=True
    )
    assert checked.returncode != 0
    assert "camera_move" in checked.stderr
