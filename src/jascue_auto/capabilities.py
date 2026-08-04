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

# Where the subject sits when it does not fill the output ratio.
#
# A subject small in a clean frame is a composition, not a shortfall. Product
# work on a white sweep is mostly negative space, and the question it asks is
# where the subject sits in that space -- not how to get rid of it. Enlarging
# the picture to fill the frame answers a question nobody asked and softens
# the result to do it.
FRAMING_INTENTS: tuple[tuple[str, str], ...] = (
    (
        "thirds",
        "主體放在三分線上，留白在另一側。乾淨背景、產品陳列、"
        "帶情緒的空鏡都適合。這是大部分情況的預設。",
    ),
    (
        "centre",
        "主體置中。對稱構圖、正面對鏡、或主體本身就是畫面全部時使用。",
    ),
    (
        "fill",
        "主體盡量佔滿畫面。細節特寫、螢幕上的數值、材質質感這類"
        "要看清楚的鏡頭用這個；本機仍會守住放大上限，不會為了填滿而糊掉。",
    ),
)

INTENT_NAMES: tuple[str, ...] = tuple(name for name, _ in FRAMING_INTENTS)


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
    lines.append("")
    lines.append("主體沒有填滿輸出比例時，它擺哪裡由你決定（framing）：")
    for name, when in FRAMING_INTENTS:
        lines.append(f"- `{name}`：{when}")
    lines.append(
        "留白本身是合法的構圖，不是要被消滅的東西。本機不會為了填滿畫面"
        "而放大到糊，實際放大倍數會回報成數字。"
    )
    return "\n".join(lines)
