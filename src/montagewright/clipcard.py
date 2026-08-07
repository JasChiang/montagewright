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

from montagewright.planner import MAX_OUTPUT_TOKENS, ask

from montagewright.uploads import upload_now

def _card_version() -> str:
    """A version that changes when the card's shape does.

    Cards are keyed by the bytes they describe, so adding a field does not
    invalidate them -- only the version does. Two required fields were added
    without touching it, so every cached card stayed and the fields were
    silently absent from every listing that had asked for them.

    Deriving it from the required fields removes the step somebody has to
    remember. Adding, removing or renaming one rewrites the library on the
    next run; changing only a description does not, which is right, because
    a description change does not make an old card wrong.
    """

    import hashlib

    shape = ",".join(sorted(card_schema()["required"]))
    return f"montagewright-clip-card-{hashlib.sha256(shape.encode()).hexdigest()[:8]}"


CARD_VERSION = ""  # set below, once card_schema is defined


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
            "shot_size",
            "facing",
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
            "shot_size": {
                "type": "string",
                "enum": ["wide", "medium", "close", "extreme_close"],
                "description": (
                    "主體在畫面裡佔多大。`wide`：主體連同它所在的環境，"
                    "人是全身或更遠。`medium`：主體是畫面主角但還看得到周圍，"
                    "人約半身。`close`：主體填滿大部分畫面，人是肩上。"
                    "`extreme_close`：只有一個局部——鏡頭模組、鉸鏈、"
                    "螢幕上的一個數字、眼睛。\n"
                    "這是剪接會用到的事實：兩顆景別太接近接在一起會跳，"
                    "而一段戲通常要從遠往近推進。看的是主體佔畫面的比例，"
                    "不是攝影機離它多遠。"
                ),
            },
            "facing": {
                "type": "string",
                "enum": ["left", "right", "toward", "away", "flat"],
                "description": (
                    "主體朝向或移動的方向，從觀眾的角度看。`left`／`right`："
                    "人面向那一側、或東西往那一側走。`toward`：朝鏡頭來。"
                    "`away`：離鏡頭去。`flat`：正對鏡頭、對稱擺放、"
                    "或看不出方向。\n"
                    "這是銀幕方向：兩顆都朝右的對談鏡頭接在一起，"
                    "觀眾會以為兩個人在對同一邊說話；一個往右走的東西"
                    "下一顆變成往左，會讀成它掉頭了。這件事只有看畫面"
                    "才知道，量不出來。"
                ),
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


def clip_seconds(path: Path) -> float:
    """How long this clip is, measured rather than asked about."""

    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


# How far past the measured end a timestamp may land and still be believed.
#
# Gemini samples video at one frame a second, so what it can say is roughly
# integer seconds -- on a clip lasting 12.012s the last frame it holds is at
# 12, and "the action ends at 13" is a rounding artefact rather than a
# mistake. A tolerance of half a second, which is what this had first, threw
# that away and deleted a real action.
#
# There is a lot of room to be generous here: the smallest possible MM:SS
# collision is 1:01 written as 101 on a clip just past a minute, which
# overshoots by forty seconds. Anything between one second and forty is not a
# notation problem, and nothing has produced one yet.
#
# The proxy is also about 0.1s longer than the file it was made from, since
# re-encoding pads the tail. Cards describe the proxy and the edit cuts the
# original, so a usable_to taken at face value can sit a frame past the end
# of the source. The renderer clamps there; it is noted here because this is
# where the two clocks are closest to being confused for each other.
SLOP = 1.5


def _as_mmss(value: float) -> float | None:
    """Read a number back as the MM:SS it was probably written from.

    `1:53` comes back either as `1.53` or as `153`, depending on whether the
    colon became a decimal point or vanished. Both are recoverable, and both
    are only worth trying when the plain reading has already failed.
    """

    for minutes, seconds in (
        (int(value), round((value - int(value)) * 100)),   # 1.53 -> 1m 53s
        (int(value) // 100, int(value) % 100),             # 110  -> 1m 10s
    ):
        if 0 < minutes < 60 and 0 <= seconds < 60:
            return minutes * 60 + seconds
    return None


def times_on_receipt(card: dict[str, Any], duration: float) -> dict[str, Any]:
    """Put the card's seconds back on a clock that matches the file.

    Gemini reads video in MM:SS, and the card asks for plain seconds -- so on
    any clip past a minute the two notations collide. Both clips over a minute
    in this library came back wrong, in opposite directions: a 71.1s take said
    `usable_to: 110.0`, which is 1:10 with the colon dropped, and a 113.4s take
    said `usable_to: 1.53`, which is 1:53 with the colon turned into a decimal
    point. The second is the dangerous one, because 1.53 is smaller than the
    duration and so passes every range check while claiming a two-minute take
    is usable for a second and a half.

    This is the same shape as the subject boxes arriving in 0..1000 space
    whatever the field says, and it gets the same treatment: the model answers
    in its own units, and the conversion happens here, against a duration that
    was measured locally rather than asked for.

    Anything still out of range afterwards is dropped rather than clamped. A
    missing action is a static shot, which is a fine thing to be; an action at
    a wrong second puts a cut in the wrong place.
    """

    if duration <= 0:
        return card

    def readings(value: Any) -> list[float]:
        """Every way of reading this number that lands inside the clip.

        Plain seconds first, so an unambiguous number keeps its obvious
        meaning and only a number that cannot be what it says gets reread.
        """

        try:
            plain = float(value)
        except (TypeError, ValueError):
            return []
        out = []
        if 0 <= plain <= duration + SLOP:
            out.append(min(plain, duration))
        again = _as_mmss(plain)
        if again is not None and again <= duration + SLOP and again not in out:
            out.append(min(again, duration))
        return out

    def span(
        raw_start: Any, raw_end: Any, least: float
    ) -> tuple[float, float] | None:
        """The start and end of one interval, read the same way at both ends.

        A span is what makes the notation visible: 1.1 and 1.13 are both
        readable as plain seconds, and read that way they describe a
        thirty-millisecond action, which is not a thing that happens. Read as
        MM:SS they are 1:10 to 1:13, which is. So the readings are scored
        together, mixed ones are penalised, and a degenerate span loses to a
        real one -- but if the plain reading is the only one available it is
        kept whatever its length, because a genuinely short window is the
        model's to report.
        """

        starts = readings(raw_start) or [0.0]
        ends = readings(raw_end)
        pairs = [
            ((i != j, max(i, j), i + j), s, e)
            for i, s in enumerate(starts)
            for j, e in enumerate(ends)
            if e > s
        ]
        if not pairs:
            return None
        real = [one for one in pairs if one[2] - one[1] >= least]
        return min(real or pairs)[1:]

    # A usable window is judged against the clip: a couple of seconds out of
    # two minutes is the shape MM:SS-as-a-decimal leaves behind.
    window = span(
        card.get("usable_from_seconds"),
        card.get("usable_to_seconds"),
        max(1.0, duration * 0.1),
    ) or (0.0, duration)
    card["usable_from_seconds"] = round(window[0], 3)
    card["usable_to_seconds"] = round(window[1], 3)

    kept = []
    for beat in card.get("action") or []:
        # The same floor `action_beats` uses. Below it a beat cannot place a
        # cut anyway, so there is nothing to preserve by keeping it.
        found = span(beat.get("starts_seconds"), beat.get("ends_seconds"), 0.25)
        if found is None:
            continue
        kept.append(
            dict(beat, starts_seconds=round(found[0], 3), ends_seconds=round(found[1], 3))
        )
    card["action"] = kept

    for subject in card.get("subjects") or []:
        found = readings(subject.get("at_seconds"))
        subject["at_seconds"] = round(found[0] if found else 0.0, 3)
    return card


CARD_VERSION = _card_version()


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

    # The card asks for "the whole length" without ever saying what it is, and
    # the model reads video in MM:SS -- so it was converting a notation it was
    # never told to convert, against a total it had to guess. Both are stated
    # here now, and checked again on receipt.
    duration = clip_seconds(proxy)
    if duration > 0:
        instruction += (
            f"\n\n## 這支素材的長度\n\n"
            f"{duration:.1f} 秒（也就是 {int(duration) // 60}:"
            f"{duration - 60 * (int(duration) // 60):04.1f}）。\n"
            f"所有秒數欄位都要填**從頭算起的純秒數**，不要用「分:秒」。"
            f"例如 1 分 53 秒要寫 113.0，不能寫 1.53 也不能寫 153。\n"
        )

    if cache is None:
        uploaded = upload_now(proxy, client)
        uri = uploaded.uri
    else:
        uri, _ = cache.uri_for(proxy, client, mime_type="video/mp4")

    interaction = ask(
        client,
        model=model_id or MODEL_ID,
        store=False,
        # The video first, the question after it. Google's own guidance for
        # a single video is to put the text last, and this had it the other
        # way round since the card writer was first written.
        input=[
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "resolution": "low",
            },
            {"type": "text", "text": instruction},
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
    card = times_on_receipt(card, duration)
    return card, Usage.from_interaction(interaction)


def card_map(proxies: Path, card_dir: Path) -> dict[str, Path]:
    """Which card describes which source, rebuilt after the fact.

    `build_library` hands this map back when it writes the cards, and anything
    asking for it later has to derive it the same way -- a card is named for
    the bytes it describes, not for the source those bytes came from. Two
    callers took the filename to be the source id instead. Every lookup
    missed, so every shot was reframed with no subject to aim at, so every
    crop sat dead centre while the report went on describing the subject it
    had followed. It looked like a working cut. That is the whole reason this
    is a function and not two dict comprehensions.
    """

    from montagewright.uploads import content_hash

    found: dict[str, Path] = {}
    for proxy in sorted(proxies.glob("*.mp4")):
        card = card_dir / f"{content_hash(proxy)[:20]}.json"
        if card.exists():
            found[proxy.stem] = card
    return found


def build_library(
    proxies: dict[str, Path],
    card_dir: Path,
    *,
    client,
    cache=None,
    model_id: str | None = None,
    progress=None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Write a card for every asset that does not already have one.

    Resumable by construction: an existing card of the current version is left
    alone. A run interrupted halfway costs nothing to restart, which matters
    when the alternative is seventy-four paid calls.
    """

    card_dir.mkdir(parents=True, exist_ok=True)
    # Named by what they describe, not by where the file happened to sit. A
    # card is only worth caching because it stays true, and it stays true for
    # the same bytes under any name in any folder -- keeping them per run
    # meant "content-addressed" and "written once per output directory" at
    # the same time, so a second cut of the same rushes rewrote all of them.
    from montagewright.uploads import content_hash
    paths: dict[str, Path] = {}
    stats: dict[str, Any] = {
        "written": 0, "reused": 0, "failed": 0, "input": 0, "output": 0,
    }
    failures: list[str] = []

    total = len(proxies)
    for index, (source_id, proxy) in enumerate(sorted(proxies.items()), start=1):
        destination = card_dir / f"{content_hash(proxy)[:20]}.json"
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
        # Seventy-four of these is four minutes with nothing on screen, which
        # is indistinguishable from a hang to anyone watching.
        if progress is not None:
            progress(index, total, source_id)
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
