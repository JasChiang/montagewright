"""Recover missed speech on Apple's clock, never the model's coarse clock."""
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
import subprocess
import tempfile


def recover(said, words, blocks, *, audio: Path, locale: str, root: Path | None):
    from montagewright.backfill import across_lines
    from montagewright.transcript import hear, words_of
    from montagewright.spans import seconds_of
    from montagewright.checkpoints import key_for, read_json, write_json
    from montagewright.uploads import content_hash
    missing = []
    for text, (start, end, pieces) in zip(said, across_lines(said, words)):
        if end <= start or any(p.ends_seconds <= p.starts_seconds and p.text.isalnum() for p in pieces):
            missing.append(text)
    if not missing:
        return words, []
    recovered = list(words)
    attempts = []
    windows = set()
    for text in missing:
        matches = sorted(blocks, key=lambda b: SequenceMatcher(
            None, text, str(b.get("said", "")), autojunk=False).ratio(), reverse=True)
        if not matches:
            continue
        block = matches[0]
        if SequenceMatcher(None, text, str(block.get("said", "")), autojunk=False).ratio() < 0.4:
            continue
        try:
            start = max(0, seconds_of(str(block["from"])) - 1)
            end = seconds_of(str(block["to"])) + 1
        except (KeyError, ValueError, TypeError):
            continue
        if end <= start or (start, end) in windows:
            continue
        windows.add((start, end))
        path = (root / (key_for({"source": content_hash(audio), "locale": locale,
                                "start": start, "end": end, "recovery": 1}) + ".json")
                if root else None)
        payload = read_json(path) if path else None
        if payload is None:
            with tempfile.TemporaryDirectory() as folder:
                clip = Path(folder) / "speech.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio),
                    "-ss", str(start), "-t", str(end-start), "-vn", "-ac", "1",
                    "-ar", "16000", str(clip)], check=True, capture_output=True)
                payload = hear(clip, locale=locale)
            if path:
                write_json(path, payload)
        added = 0
        for word in words_of(payload):
            word = replace(word, starts_seconds=word.starts_seconds+start,
                           ends_seconds=word.ends_seconds+start)
            # Keep original measured words immutable. Recovery fills only holes.
            if any(word.starts_seconds < w.ends_seconds and word.ends_seconds > w.starts_seconds
                   for w in recovered):
                continue
            recovered.append(word)
            added += 1
        attempts.append({"start_seconds": start, "end_seconds": end,
                         "words_added": added, "timing_source": "apple_local_retry",
                         "evidence": str(path) if path else None})
    return sorted(recovered, key=lambda w: w.starts_seconds), attempts
