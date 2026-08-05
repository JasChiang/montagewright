"""Per-asset facts, written once and reused forever.

A card describes one clip in isolation: what is in it, where the subjects sit,
how the frame is composed, whether the take failed. Everything here is true
without knowing what the clip will be used for, which is exactly why the brief
is not an input. A card written against one brief would have to be rewritten
for the next deliverable made from the same shoot; a card written against the
material alone is good until the material changes.

Subject boxes live here for the same reason. They are expensive to compute,
they never change, and asking for them once per render instead means paying
for the same answer on every rhythm tweak, every second aspect, and every
review round -- eleven calls a run for something that was already known.

What a card deliberately cannot carry is anything comparative or sequential.
Whether this take beats the other two, whether it serves the brief, how long
it should hold: none of that is visible from inside one clip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from montagewright.planner import MAX_OUTPUT_TOKENS

from montagewright.uploads import upload_now

CARD_VERSION = "montagewright-clip-card-v1"


class CardLibraryEmpty(RuntimeError):
    """Every clip failed to describe, so there is nothing to plan from."""


@dataclass(frozen=True)
class Beat:
    """One movement in a clip, and when it runs."""

    what: str
    starts_seconds: float
    ends_seconds: float


def action_beats(card: dict[str, Any]) -> list[Beat]:
    beats: list[Beat] = []
    for entry in card.get("action", []) or []:
        try:
            start = float(entry["starts_seconds"])
            end = float(entry["ends_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        # A span shorter than a few frames is not a span. Twenty-six of
        # forty-one beats in one library came back under a quarter of a
        # second -- "the models rotate their phones, 0.02 to 0.06s" -- and
        # every consumer downstream treated them as real: in-points were
        # snapped onto them and the rhythm pass was shown them as how long
        # the action takes. A guessed timestamp is worse than none.
        if end - start >= 0.25:
            beats.append(Beat(str(entry.get("what", "")), start, end))
    return sorted(beats, key=lambda beat: beat.starts_seconds)


def snap_to_action(
    card: dict[str, Any], wanted_start: float, duration: float
) -> tuple[float, str | None]:
    """Move a planned in-point onto the nearest action that contains it.

    A cut placed by arithmetic lands wherever the seconds fall, which is
    usually the middle of a gesture. An editor entering a shot goes in as the
    movement starts. This only moves the in-point when there is an action
    close enough to be the one meant -- half the shot's length -- so a static
    shot keeps the timing it was given.
    """

    beats = action_beats(card)
    if not beats:
        return wanted_start, None

    tolerance = max(0.5, duration / 2.0)
    nearest = min(beats, key=lambda beat: abs(beat.starts_seconds - wanted_start))
    drift = nearest.starts_seconds - wanted_start
    if abs(drift) > tolerance:
        return wanted_start, None
    return nearest.starts_seconds, (
        f"moved {drift:+.2f}s onto '{nearest.what}'" if abs(drift) > 0.05 else None
    )


@dataclass(frozen=True)
class SubjectBox:
    """One nameable thing in the frame, with where it sits."""

    label: str
    centre_x: float
    centre_y: float
    width: float
    height: float
    moves: bool
    # When this position was true. A box is a moment, and a moment is the
    # whole answer only for a locked-off frame.
    at_seconds: float = 0.0

    @property
    def is_horizontal(self) -> bool:
        return self.width > self.height


def card_schema() -> dict[str, Any]:
    """Flat, because a nested one was what made the old plan unservable."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "usable",
            "composition",
            "subjects",
            "action",
            "camera_motion",
            "speech",
            "usable_from_seconds",
            "usable_to_seconds",
            "needs",
        ],
        "properties": {
            "summary": {
                "type": "string",
                "description": "What happens in this clip, in a line or two.",
            },
            "usable": {
                "type": "boolean",
                "description": (
                    "False only for a take that failed outright -- an action "
                    "cut off, a screen that slept, a misfire. Handheld "
                    "movement, soft focus and unusual framing are styles "
                    "available to an edit, not defects."
                ),
            },
            "unusable_reason": {"type": "string"},
            "composition": {
                "type": "string",
                "enum": ["horizontal", "vertical", "square", "mixed"],
                "description": (
                    "How the content is laid out in the frame. A row of "
                    "handsets across a table is horizontal and will fight a "
                    "vertical crop; a standing person is vertical and will "
                    "sit in one happily. Recorded here so the edit knows "
                    "before it commits an aspect."
                ),
            },
            "usable_from_seconds": {
                "type": "number",
                "description": (
                    "從第幾秒起這支才真的可用。開頭常有攝影機還在甩、還在"
                    "對焦、或畫面還沒穩定的部分——那些不是可以剪進片子的"
                    "素材，寫出可用的起點比說整支不能用有用得多。"
                    "整支都可用就填 0。"
                ),
            },
            "usable_to_seconds": {
                "type": "number",
                "description": (
                    "到第幾秒為止還可用。結尾常有鏡頭已經移開、"
                    "人已經走出畫面、或開始收東西的部分。"
                    "整支都可用就填總長。"
                ),
            },
            "needs": {
                "type": "array",
                "description": (
                    "這支素材要用的話需要什麼處理。這是你看過畫面之後的"
                    "理解，不是規則：主體太小就要推近，重點偏在一側就要"
                    "裁切重新構圖，內容橫向鋪開就要橫搖帶過，"
                    "只有一段可用就要修剪。什麼都不需要就留空。"
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["what", "why"],
                    "properties": {
                        "what": {
                            "type": "string",
                            "enum": ["trim", "crop", "zoom", "pan", "tilt"],
                        },
                        "why": {
                            "type": "string",
                            "description": (
                                "用畫面上看得到的東西說明。「主體只佔畫面"
                                "一小塊，直式輸出要推近才看得清楚」"
                                "比「需要 zoom」有用。"
                            ),
                        },
                    },
                },
            },
            "action": {
                "type": "array",
                "description": (
                    "Where things actually happen in this clip. An editor "
                    "cuts on action -- into a gesture as it begins, out as it "
                    "completes -- and a cut placed by arithmetic lands "
                    "mid-movement, which reads as a mistake even to someone "
                    "who cannot say why. Empty for a static shot."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["what", "starts_seconds", "ends_seconds"],
                    "properties": {
                        "what": {
                            "type": "string",
                            "description": (
                                "The movement, in a few words: 'the hand "
                                "reaches in', 'the phone opens', 'the watch "
                                "is lowered'."
                            ),
                        },
                        "starts_seconds": {
                            "type": "number",
                            "description": (
                                "動作開始的那一秒——手還沒碰到之前、機身還"
                                "沒開始翻之前。"
                            ),
                        },
                        "ends_seconds": {
                            "type": "number",
                            "description": (
                                "動作完成的那一秒。這是一段時間，不是一個"
                                "瞬間：起訖相同或只差零點幾秒，等於沒有指出"
                                "任何東西，後面就沒辦法把畫面切在動作上。"
                                "看不出來動作在哪裡結束，就不要寫這一筆——"
                                "留空比給一個猜的時間有用。"
                            ),
                        },
                    },
                },
            },
            "camera_moves": {
                "type": "boolean",
                "description": "Whether the camera itself moves in this take.",
            },
            "speech": {
                "type": "string",
                "enum": ["none", "ambient", "content"],
                "description": (
                    "這支素材裡的說話是不是內容本身。`content`：訪談、"
                    "受訪者的回答、對鏡頭講話、旁白——把聲音拿掉這顆就"
                    "沒有意義了。`ambient`：現場環境音、旁邊路人的交談、"
                    "聽不清楚的背景人聲，剪掉不影響。`none`：沒有人聲。"
                    "填 content 的素材後面才會去做逐字稿，那要花錢也花"
                    "時間，所以不確定的時候看的是「這顆的意思靠不靠聲音"
                    "成立」，不是「有沒有人在講話」。"
                ),
            },
            "camera_motion": {
                "type": "string",
                "description": (
                    "攝影機怎麼動，以及動了之後畫面裡多了什麼、少了什麼。"
                    "「往右平移，右邊會有第三台白色手機進畫面」、"
                    "「緩慢推近到機身鉸鏈」、「繞著產品轉，背面轉出來」。"
                    "這決定的是要不要再加一層數位運鏡：素材自己會把東西"
                    "帶進來時，框住不動讓它演完就好，兩個運鏡疊在一起會"
                    "打架。攝影機不動就留空。"
                ),
            },
            "subjects": {
                "type": "array",
                "description": (
                    "The things worth framing, described so each can be told "
                    "from anything similar beside it, with where it sits."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "label",
                        "centre_x",
                        "centre_y",
                        "width",
                        "height",
                        "moves",
                        "at_seconds",
                    ],
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": (
                                "'the left, grey handset', not 'the handset'."
                            ),
                        },
                        "centre_x": {
                            "type": "number",
                            "description": (
                                "Fraction of frame width, 0.0 left to 1.0 "
                                "right. Never pixels."
                            ),
                        },
                        "centre_y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "at_seconds": {
                            "type": "number",
                            "description": (
                                "你是看第幾秒說出這個位置的。攝影機或主體"
                                "在動的時候，位置只在那一刻成立——後面要拿"
                                "這個框去框別的時間點，得先知道它是什麼時候"
                                "量的。"
                            ),
                        },
                        "moves": {
                            "type": "boolean",
                            "description": (
                                "Whether this subject moves within the shot."
                            ),
                        },
                    },
                },
            },
        },
    }


def load_card(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("version") == CARD_VERSION else None


def save_card(path: Path, card: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**card, "version": CARD_VERSION}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def subjects_from_card(card: dict[str, Any]) -> list[SubjectBox]:
    boxes: list[SubjectBox] = []
    for entry in card.get("subjects", []):
        try:
            boxes.append(
                SubjectBox(
                    label=str(entry["label"]),
                    centre_x=float(entry["centre_x"]),
                    centre_y=float(entry["centre_y"]),
                    width=float(entry["width"]),
                    height=float(entry["height"]),
                    moves=bool(entry.get("moves", False)),
                    at_seconds=float(entry.get("at_seconds", 0.0) or 0.0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def find_subject(card: dict[str, Any], description: str) -> SubjectBox | None:
    """Match a planner's subject description to a box the card already holds.

    Exact wording will not match, so this looks for the card's label inside
    the description or the other way round. A miss returns None and the caller
    grounds the subject the expensive way -- a card that cannot answer is not
    a reason to fail, only a reason to pay.
    """

    boxes = subjects_from_card(card)
    lowered = description.lower()
    for box in boxes:
        label = box.label.lower()
        if label and (label in lowered or lowered in label):
            return box
    # Fall back to any shared distinguishing word of reasonable length.
    for box in boxes:
        for word in box.label.split():
            if len(word) >= 2 and word in description:
                return box
    return None


def describe_clip(
    proxy: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Watch one clip and write its card.

    Called once per asset, ever. The result is content-addressed, so a card
    survives every rerun, every second aspect and every review round -- which
    is the whole reason the subject boxes belong here rather than being
    grounded again on each render.
    """

    from montagewright.planner import MODEL_ID, Usage, _parse, _to_frame_fractions

    prompts = Path(__file__).resolve().parent / "prompts"
    instruction = (prompts / "clipcard_zh-TW.txt").read_text(encoding="utf-8")

    if cache is None:
        uploaded = upload_now(proxy, client)
        uri = uploaded.uri
    else:
        uri, _ = cache.uri_for(proxy, client, mime_type="video/mp4")

    interaction = client.interactions.create(
        model=model_id or MODEL_ID,
        store=False,
        input=[
            {"type": "text", "text": instruction},
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "media_resolution": "low",
            },
        ],
        generation_config={"thinking_level": "low", "max_output_tokens": MAX_OUTPUT_TOKENS},
        response_format={
            "mime_type": "application/json",
            "schema": card_schema(),
        },
    )
    card = _parse(interaction, what="clip card")
    # The model answers boxes in its native 0..1000 space for some clips
    # whatever the field says, so the conversion happens on receipt.
    card["subjects"] = _to_frame_fractions(card.get("subjects", []))
    return card, Usage.from_interaction(interaction)


def build_library(
    proxies: dict[str, Path],
    card_dir: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write a card for every asset that does not already have one.

    Resumable by construction: an existing card of the current version is left
    alone. A run interrupted halfway costs nothing to restart, which matters
    when the alternative is seventy-four paid calls.
    """

    card_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    stats = {"written": 0, "reused": 0, "failed": 0, "input": 0, "output": 0}
    failures: list[str] = []

    for source_id, proxy in sorted(proxies.items()):
        destination = card_dir / f"{source_id}.json"
        if load_card(destination) is not None:
            paths[source_id] = destination
            stats["reused"] += 1
            continue
        try:
            card, usage = describe_clip(
                proxy, client=client, cache=cache, model_id=model_id
            )
        except Exception as error:
            # One unreadable clip is not a reason to abandon the library.
            # Losing why it failed is a different matter: a NameError in the
            # request took all seventy-four down and reported it as the
            # routine "74 failed ($0.0000)" line, and the run went on to
            # plan a film from an empty library.
            failures.append(f"{source_id}: {type(error).__name__}: {error}")
            stats["failed"] += 1
            continue
        save_card(destination, card)
        paths[source_id] = destination
        stats["written"] += 1
        stats["input"] += usage.input_tokens
        stats["output"] += usage.output_tokens + usage.thought_tokens
    stats["failures"] = failures
    if failures and not paths:
        # Not a degradation. A library with nothing in it is the input
        # missing, and everything downstream reads its absence as "these
        # clips have no description" rather than as an error -- one run
        # picked an aspect, chose sixteen shots and spent $1.80 planning a
        # film out of empty cards before anything said so.
        raise CardLibraryEmpty(
            f"every one of the {len(failures)} clips failed to describe; "
            f"first: {failures[0]}"
        )
    return paths, stats
