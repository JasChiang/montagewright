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
    # The shortest this move can be and still register as a move at all --
    # a technical floor, not an answer to "how long should this shot be".
    # How long a sweep needs depends on how far it travels and how much has
    # to be taken in on the way, and that is visible in the material: the
    # same 2.5 seconds that reads a two-word logo is not enough for three
    # handsets, and is twice what a short push needs. The planner watches the
    # shot, so the planner sets the length; this is what the executor reports
    # against when the length it was given cannot contain the move.
    min_seconds: float = 0.0


CAMERA_MOVES: tuple[Capability, ...] = (
    Capability(
        name="hold",
        when=(
            "畫面本身已經成立，或主體靜止且資訊集中。"
            "素材本身已經有攝影機運動時也用這個——讓它自己的推軌或搖攝演完，"
            "再疊一層數位運鏡兩者會打架。不動是一個決定，不是沒有決定。"
        ),
        needs_subject=True,
        min_seconds=0.8,
    ),
    Capability(
        name="pan",
        when=(
            "水平帶過去。指定一個主體就跟著它走；指定兩個就從第一個帶到"
            "第二個；一排東西（三台手機、一列手錶）就從頭掃到尾。"
            "主體不動又只有一個時會變成 hold，因為沒有東西可跟。"
        ),
        needs_subject=True,
        min_seconds=2.5,
    ),
    Capability(
        name="tilt",
        when=(
            "垂直帶過去。主體上下移動時用——放進水裡、從桌面拿起、"
            "沿著機身從上往下看。橫式素材出直式時上下沒有空間可移，"
            "本機會照實回報做不到。"
        ),
        needs_subject=True,
        min_seconds=2.5,
    ),
    Capability(
        name="push_in",
        when=(
            "收緊到一個細節上——鏡頭模組、螢幕上的數值、一個材質。"
            "本機依來源解析度決定收多緊，不會為了框滿而放大到糊。"
        ),
        needs_subject=True,
        min_seconds=2.5,
    ),
    Capability(
        name="pull_out",
        when="從細節退開，交代它在什麼脈絡裡，通常用在一段的收尾。",
        needs_subject=True,
        min_seconds=2.5,
    ),
)

# These name what a crop can do, not what a camera can do. A dolly, a crane or
# a true track all change viewpoint, which produces parallax -- foreground and
# background moving at different rates -- and no amount of moving a window
# over an already-shot frame produces that. Offering `dolly_zoom` would get it
# chosen for an effect the executor cannot deliver, which is the same failure
# as any other capability declared and not built.

MOVE_NAMES: tuple[str, ...] = tuple(move.name for move in CAMERA_MOVES)
MOVE_FLOORS: dict[str, float] = {
    move.name: move.min_seconds for move in CAMERA_MOVES
}

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
        lines.append(
            f"- `{move.name}`（{subject}，短於 {move.min_seconds:g} 秒就"
            f"不成立）：{move.when}"
        )
    lines.append(
        "選了做不到的組合——例如對靜止主體選 follow——本機會照實記錄實際"
        "做到什麼，不會假裝執行了。"
    )
    lines.append("")
    lines.append(
        "括號裡的秒數是這個運鏡在技術上還算得上運鏡的底線，不是它需要多久。"
        "需要多久你看素材決定：鏡頭要走多遠、路上有幾樣東西要讓觀眾看進去、"
        "字有多長要讀完。同樣是橫掃，兩個字的 logo 跟一排三台手機需要的時間"
        "差很多。你看得到這顆畫面，所以把這個判斷寫進 `seconds_needed`。"
        "時間不夠的運鏡看不懂——三台手機橫掃只有 1.5 秒，觀眾還沒看到第二台"
        "就切了。本機不會替你補時間，只會在做不到的時候照實回報。"
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


# What this tool does not do, in the same place and for the same reason as
# what it does. A reviewer judging the cut against a brief has no way to know
# the difference between "this was done badly" and "this cannot be done here"
# -- so it reported a missing title card as a fault every round, which is a
# paid call spent on something no replan can ever fix, and a verdict of
# "revise" on a cut that was as asked.
CANNOT: tuple[tuple[str, str], ...] = (
    ("字卡、標題、下標", "會燒字幕，但沒有其他文字圖層"),
    ("轉場特效", "每一次都是硬切；長度與切點是唯一的節奏工具"),
    ("調色與濾鏡", "畫面只做裁切、縮放與響度處理"),
    ("速度變化", "沒有慢動作或加速"),
    ("畫面合成", "不疊圖、不分割畫面、不做子母畫面"),
)


def describe_limits_for_prompt() -> str:
    """The same list, for a prompt that is about to judge the result."""

    return "\n".join(
        f"- {what}：{why}" for what, why in CANNOT
    )
