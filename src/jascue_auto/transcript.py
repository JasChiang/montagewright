"""What was said, when, and what it actually was.

A transcript card is its own artifact, not more fields on a clip card. A clip
card describes what a take looks like and is worth caching because that stays
true; a transcript is only worth paying for when the speech matters, and it is
useful on its own -- subtitling a finished video is a job that never touches
the edit.

Two halves, split the way everything else here is split. The system recogniser
is very good at when a word was said and reliably wrong about what several of
them were: it emits characters converted from a simplified model (乾 as 幹,
回 as 迴, 面 as 麵), it mishears proper nouns, and it has never heard of the
product on screen. Gemini gets the audio, the picture, and the recogniser's
own text, and corrects the words. Timings are never touched by the model --
they are measured, and a model asked for a timestamp will invent one.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jascue_auto.uploads import upload_now

CARD_VERSION = "jascue-auto-transcript-v1"
TOOL = Path(__file__).resolve().parents[2] / "tools" / "transcribe" / "transcribe"

# Below this a "word" is usually the recogniser splitting one syllable, and a
# subtitle cannot sit on it.
MIN_WORD_SECONDS = 0.04


class TranscriberMissing(RuntimeError):
    """The Swift tool has not been built for this machine."""


@dataclass(frozen=True)
class Word:
    text: str
    starts_seconds: float
    ends_seconds: float
    confidence: float | None = None


@dataclass(frozen=True)
class Line:
    """One subtitle: what to show, and the window it belongs in."""

    text: str
    starts_seconds: float
    ends_seconds: float
    heard: str = ""

    @property
    def duration(self) -> float:
        return self.ends_seconds - self.starts_seconds

    @property
    def corrected(self) -> bool:
        return bool(self.heard) and self.heard != self.text


def extract_audio(source: Path, destination: Path) -> Path:
    """Mono 16k PCM, which is what the recogniser wants and nothing more."""

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )
    return destination


def hear(source: Path, *, locale: str = "zh-TW") -> dict[str, Any]:
    """Run the system recogniser over one file.

    Returns its answer unaltered. Whatever is wrong with the words is wrong
    at this point and is corrected later against the picture; rewriting them
    here would mean guessing without having seen anything.
    """

    if not TOOL.exists():
        raise TranscriberMissing(
            f"{TOOL} is not built. Run:\n"
            f"  swiftc -parse-as-library -O -o {TOOL} {TOOL.with_suffix('.swift').parent}/Transcribe.swift"
        )
    with tempfile.TemporaryDirectory() as work:
        wav = extract_audio(source, Path(work) / "audio.wav")
        completed = subprocess.run(
            [str(TOOL), str(wav), locale],
            capture_output=True, text=True, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"transcription failed for {source.name}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return json.loads(completed.stdout)


def words_of(payload: dict[str, Any]) -> list[Word]:
    """Every word the recogniser placed, in order."""

    words: list[Word] = []
    for utterance in payload.get("utterances", []) or []:
        for entry in utterance.get("words", []) or []:
            try:
                start = float(entry["starts_seconds"])
                end = float(entry["ends_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(entry.get("text", "")).strip()
            if not text or end - start < MIN_WORD_SECONDS:
                continue
            confidence = entry.get("confidence")
            words.append(
                Word(
                    text=text,
                    starts_seconds=start,
                    ends_seconds=end,
                    confidence=(
                        float(confidence) if confidence is not None else None
                    ),
                )
            )
    return sorted(words, key=lambda word: word.starts_seconds)


def gaps(words: list[Word], *, at_least: float = 0.35) -> list[float]:
    """Silences long enough to be somewhere a cut could go.

    A pause is not a sentence ending -- hesitating, thinking and being
    interrupted all make pauses, and cutting on one of those reads as a
    mistake. This says where the speaker stopped, not where a thought did;
    which of them is a boundary is a question for whoever can hear the
    sentence, and the answers get snapped back onto these.
    """

    found = []
    for earlier, later in zip(words, words[1:]):
        if later.starts_seconds - earlier.ends_seconds >= at_least:
            found.append(
                round((earlier.ends_seconds + later.starts_seconds) / 2.0, 3)
            )
    return found


def snap(seconds: float, candidates: list[float], *, within: float = 0.6) -> float:
    """Put a spoken boundary on the nearest measured silence.

    The model hears where a sentence ends; the recogniser measured where the
    sound stopped. Taking the model's second directly puts the cut a syllable
    early or late, and taking only the silences puts it in the middle of a
    thought -- the same split as a subject box seeded by name and measured by
    the tracker.
    """

    if not candidates:
        return seconds
    nearest = min(candidates, key=lambda point: abs(point - seconds))
    return nearest if abs(nearest - seconds) <= within else seconds


def to_srt(lines: list[Line]) -> str:
    """Standard subtitles, so this is useful without the rest of the tool."""

    def stamp(seconds: float) -> str:
        milli = max(0, round(seconds * 1000))
        hours, milli = divmod(milli, 3_600_000)
        minutes, milli = divmod(milli, 60_000)
        secs, milli = divmod(milli, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milli:03d}"

    blocks = []
    for index, line in enumerate(lines, start=1):
        blocks.append(
            f"{index}\n"
            f"{stamp(line.starts_seconds)} --> {stamp(line.ends_seconds)}\n"
            f"{line.text}\n"
        )
    return "\n".join(blocks)


def load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("version") == CARD_VERSION else None


def save(card: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**card, "version": CARD_VERSION}, ensure_ascii=False,
                   indent=1),
        encoding="utf-8",
    )
    return path


def lines_of(card: dict[str, Any]) -> list[Line]:
    return [
        Line(
            text=str(entry.get("text", "")),
            starts_seconds=float(entry.get("starts_seconds", 0.0)),
            ends_seconds=float(entry.get("ends_seconds", 0.0)),
            heard=str(entry.get("heard", "")),
        )
        for entry in card.get("lines", []) or []
        if entry.get("text")
    ]


def _schema() -> dict[str, Any]:
    """Flat, and with no field the model has to invent a number for."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["language", "lines", "uncertain", "summary"],
        "properties": {
            "language": {
                "type": "string",
                "description": (
                    "BCP-47 for what is actually spoken. The recogniser was "
                    "run with a guess; this is the answer."
                ),
            },
            "summary": {"type": "string"},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "text", "heard", "starts_seconds", "ends_seconds"
                    ],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "改正後的字幕文字。",
                        },
                        "heard": {
                            "type": "string",
                            "description": (
                                "辨識器原本給的同一段文字。沒改就填一樣的"
                                "——後面要靠這個看出你改了什麼。"
                            ),
                        },
                        "starts_seconds": {"type": "number"},
                        "ends_seconds": {"type": "number"},
                    },
                },
            },
            "uncertain": {
                "type": "array",
                "items": {"type": "string"},
                "description": "聽不清楚或無法確定的地方，各一句說明。",
            },
        },
    }


def describe(
    source: Path,
    *,
    client,
    locale: str = "zh-TW",
    cache=None,
    model_id: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Hear it locally, then have the words corrected against the picture."""

    from jascue_auto.planner import (
        MAX_OUTPUT_TOKENS,
        MODEL_ID,
        PROMPTS,
        Usage,
        _parse,
    )

    heard = hear(source, locale=locale)
    words = words_of(heard)
    silences = gaps(words)
    rough = "\n".join(
        f"{word.starts_seconds:.2f} {word.text}" for word in words
    )

    if cache is None:
        uri = upload_now(source, client).uri
    else:
        uri, _ = cache.uri_for(source, client, mime_type="video/mp4")

    instruction = (PROMPTS / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    interaction = client.interactions.create(
        model=model_id or MODEL_ID,
        store=False,
        input=[
            {
                "type": "text",
                "text": (
                    f"{instruction}\n\n## 辨識器給的逐字稿"
                    f"（{locale}，秒數 + 字）\n\n{rough}\n"
                ),
            },
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                "media_resolution": "low",
            },
        ],
        generation_config={
            "thinking_level": "high",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        response_format={
            "mime_type": "application/json",
            "schema": _schema(),
        },
    )
    payload = _parse(interaction, what="transcript")

    # The model heard where the sentences end; the recogniser measured where
    # the sound stopped. Neither alone puts a cut in the right place.
    lines = []
    for entry in payload.get("lines", []) or []:
        try:
            start = float(entry["starts_seconds"])
            end = float(entry["ends_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        lines.append({
            "text": str(entry.get("text", "")).strip(),
            "heard": str(entry.get("heard", "")).strip(),
            "starts_seconds": round(snap(start, silences), 3),
            "ends_seconds": round(snap(end, silences), 3),
        })

    card = {
        "summary": payload.get("summary", ""),
        "language": payload.get("language", locale),
        "heard_with": heard.get("locale", locale),
        "duration_seconds": heard.get("duration_seconds", 0.0),
        "lines": [line for line in lines if line["text"]],
        "uncertain": payload.get("uncertain", []) or [],
        # Where the speaker actually stopped. A cut placed anywhere else in a
        # talking shot lands mid-syllable, and every consumer of this card
        # needs them, not just the one that wrote them.
        "silences": silences,
    }
    return card, Usage.from_interaction(interaction)
