"""Every field a prompt names is a field its schema has.

This bug has now landed three times: `camera_move` removed from the selection
schema while six readers still asked for it, `then_subject` taught in prose
after it stopped existing, and the direction prompt telling the model to fill
`material_notes` with a `supersedes` key -- against a schema whose only list
is `unusable`, keyed `superseded_by`, pointing the other way.

None of them raised. A model handed instructions for a field that is not in
its response schema writes something plausible into the fields it does have,
and the run completes, and the guidance silently did nothing.

The existing prompt tests assert that particular sentences are present, which
catches a deletion and nothing else -- a sentence can stay word for word
correct while the field it names is renamed underneath it. This asserts the
join instead: pull every `backticked` identifier out of each prompt and
require the paired schema to define it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from montagewright import clipcard, planner, review, transcript

PROMPTS = Path(planner.__file__).resolve().parent / "prompts"

# `looks[].must_be_whole` is one name written as a path; `.` and `[]` are
# punctuation around the parts, not part of any one of them.
NAME = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:(?:\[\])?\.[A-Za-z_][A-Za-z0-9_]*)*)`")


def names_in(prompt: str) -> set[str]:
    found: set[str] = set()
    for whole in NAME.findall(prompt):
        found.update(part for part in whole.replace("[]", "").split(".") if part)
    return found


def names_of(schema: dict) -> set[str]:
    """Every property name and every enum value anywhere in the schema.

    Both, because a prompt legitimately names either: `unusable` is a field
    and `bed` is one of the values `music_under_speech` accepts, and the
    prompt has to be able to say both without the test caring which is which.
    """

    found: set[str] = set()
    if not isinstance(schema, dict):
        return found
    for value in schema.get("enum", []) or []:
        if isinstance(value, str):
            found.add(value)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, child in properties.items():
            found.add(key)
            found |= names_of(child)
    for key in ("items", "additionalProperties"):
        found |= names_of(schema.get(key))
    return found


# Words a prompt backticks that are not the model's to write: things it is
# told about rather than asked for. Each is here because it is read from
# somewhere else, not because the check was inconvenient.
SPOKEN_OF = {
    # The transcript's per-line times and `heard` are written by local code
    # after the answer comes back, and the shot review names the ladder rungs
    # the executor records rather than any field of its own.
    "clipcard_zh-TW.txt": set(),
    "direction_zh-TW.txt": set(),
    # Selection reads the card's vocabulary off the material listing and
    # never writes it: shot size, which way a subject faces, and how the
    # content is laid out are all decided one step earlier. Taken from the
    # card schema rather than listed, so renaming one of those values fails
    # here too instead of quietly joining the allowlist.
    "selection_zh-TW.txt": "card",
    "rhythm_zh-TW.txt": {"beats", "chorus_1_start"},
    "replan_zh-TW.txt": set(),
    "review_zh-TW.txt": set(),
    "shotreview_zh-TW.txt": {"none_recorded"},
    "transcript_zh-TW.txt": set(),
}

PAIRED = [
    ("clipcard_zh-TW.txt", lambda: clipcard.card_schema()),
    ("direction_zh-TW.txt", lambda: planner._direction_schema()),
    ("selection_zh-TW.txt", lambda: planner._selection_schema(["A"])),
    ("rhythm_zh-TW.txt", lambda: planner._rhythm_schema(["k00"])),
    ("replan_zh-TW.txt", lambda: planner._selection_schema(["A"])),
    ("review_zh-TW.txt", lambda: review._verdict_schema()),
    ("shotreview_zh-TW.txt", lambda: review._shot_schema(["k00"])),
    ("transcript_zh-TW.txt", lambda: transcript._schema()),
]


def _allowed(filename: str) -> set[str]:
    spoken = SPOKEN_OF[filename]
    return names_of(clipcard.card_schema()) if spoken == "card" else spoken


@pytest.mark.parametrize("filename,build", PAIRED, ids=[name for name, _ in PAIRED])
def test_a_prompt_only_names_fields_its_schema_defines(filename, build):
    prompt = (PROMPTS / filename).read_text(encoding="utf-8")
    unknown = names_in(prompt) - names_of(build()) - _allowed(filename)
    assert not unknown, (
        f"{filename} instructs the model about "
        f"{sorted(unknown)}, which its response schema does not define"
    )


def test_the_check_would_notice_a_renamed_field():
    """Reintroducing the bug has to fail this, or it guards nothing."""

    schema = {"properties": {"unusable": {"type": "array"}}}
    assert names_in("填 `material_notes`，用 `supersedes` 指出") - names_of(schema) == {
        "material_notes",
        "supersedes",
    }


def test_the_microphone_is_not_offered_as_evidence_of_who_is_speaking():
    """It was, and in an interview it points the other way.

    The prompt listed three cues for identifying a speaker -- whose mouth is
    moving, who is holding the microphone, who the camera is on -- and two of
    them are unreliable in exactly the material this runs on. The microphone
    in a street interview is held by the person asking and extended toward
    the person answering, so "who holds it" identifies the one not talking;
    the camera is on the listener during every reaction shot.

    This matters beyond the label. Selection is told to frame whoever is
    speaking using these same words, so a wrong attribution becomes a talking
    shot pointed at the wrong person.
    """

    prompt = (PROMPTS / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    speaking = prompt[prompt.index("## 誰在講"):]
    speaking = speaking[: speaking.index("## 切成字幕")]

    # The microphone and the camera are named as traps, not as evidence.
    assert "拿著麥克風的人通常不是正在講的那個" in speaking
    assert "鏡頭對著誰不代表誰在講" in speaking
    # And the reliable cue is the one that does not need the picture at all.
    assert "最可靠的線索是這段話本身" in speaking
    # Not knowing is an available answer, because guessing here is invisible.
    assert "uncertain" in speaking

    # The schema says the same thing: a model reads the field description
    # even when the prompt scrolls past.
    from montagewright.transcript import _schema

    said = _schema()["properties"]["lines"]["items"]["properties"]["speaker"]
    assert "麥克風都不是證據" in said["description"]


def test_a_talking_shot_is_checked_against_who_is_talking():
    """The prompt called it the most obvious error and nothing looked for it.

    Neither reviewer mentioned the speaker. The one that watches a single
    shot against its own plan is the only one that can see it -- in a
    finished cut the next shot covers it in a second and a half.
    """

    shots = (PROMPTS / "shotreview_zh-TW.txt").read_text(encoding="utf-8")
    assert "正在講話的那一個" in shots
    assert "delivered` 就是 false" in shots
    # Off-screen speech is not the same fault, or every voiceover fails.
    assert "不在畫面裡" in shots
