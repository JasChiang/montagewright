"""What the execution layer can actually do, in one place.

The planner cannot ask for a move it has never been told exists. Selection was
returning a subject and a position and nothing else, so a row of three
handsets -- which wants a sweep across them, not a follow of any one -- came
back as a single group subject whose centre never moves, and rendered as a
hold. The capability was missing from the menu, not from the material.

This list is the menu. It is rendered into the prompt so the model chooses
from what exists, and it is the vocabulary the executor dispatches on, so the
two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    name: str
    when: str
    needs_subject: bool


CAMERA_MOVES: tuple[Capability, ...] = (
    Capability(
        name="hold",
        when=(
            "畫面本身已經成立，或主體靜止且資訊集中。不動是一個決定，"
            "不是沒有決定。"
        ),
        needs_subject=True,
    ),
    Capability(
        name="follow",
        when=(
            "主體在畫面裡實際移動，鏡頭跟著它。主體不動時選這個會得到 hold，"
            "因為沒有東西可跟。"
        ),
        needs_subject=True,
    ),
    Capability(
        name="sweep_left",
        when=(
            "畫面是靜態的陳列或並排（一排手錶、三台手機、一整面產品牆），"
            "要讓觀眾逐一看過去。這是設計出來的運鏡，跟主體動不動無關；"
            "從右邊開始往左掃。"
        ),
        needs_subject=False,
    ),
    Capability(
        name="sweep_right",
        when="同上，從左邊開始往右掃。",
        needs_subject=False,
    ),
    Capability(
        name="push_in",
        when=(
            "要把注意力收束到一個細節上——鏡頭模組、螢幕上的數值、"
            "一個材質。畫面尺寸夠大才會好看。"
        ),
        needs_subject=True,
    ),
    Capability(
        name="pull_out",
        when="從細節退開，交代它在什麼脈絡裡，通常用在一段的收尾。",
        needs_subject=True,
    ),
)

MOVE_NAMES: tuple[str, ...] = tuple(move.name for move in CAMERA_MOVES)


def describe_for_prompt() -> str:
    """The menu as the planner reads it."""

    lines = ["本機執行層目前做得到這些運鏡，請從中選："]
    for move in CAMERA_MOVES:
        subject = "需要指定主體" if move.needs_subject else "不需要主體"
        lines.append(f"- `{move.name}`（{subject}）：{move.when}")
    lines.append(
        "選了做不到的組合——例如對靜止主體選 follow——本機會照實記錄實際"
        "做到什麼，不會假裝執行了。"
    )
    return "\n".join(lines)
