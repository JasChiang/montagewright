"""Recover missed speech on Apple's clock, never the model's coarse clock."""
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
import subprocess
import tempfile


def _unheard_gaps(said, words, *, minimum_seconds=0.45, maximum_attempts=8):
    """Find measured gaps beside corrected characters with no Apple time."""
    from montagewright.backfill import across_lines

    zero_times = {
        piece.starts_seconds
        for _text, (_start, _end, pieces) in zip(said, across_lines(said, words))
        for piece in pieces
        if piece.ends_seconds <= piece.starts_seconds
        and any(char.isalnum() for char in piece.text)
    }
    ordered = sorted(words, key=lambda word: word.starts_seconds)
    gaps = []
    for left, right in zip(ordered, ordered[1:]):
        if right.starts_seconds - left.ends_seconds < minimum_seconds:
            continue
        if any(left.ends_seconds <= point <= right.starts_seconds for point in zero_times):
            window = (max(0.0, left.ends_seconds - 0.25), right.starts_seconds + 0.25)
            if not gaps or gaps[-1] != window:
                gaps.append(window)
        if len(gaps) >= maximum_attempts:
            break
    return gaps


def _unresolved_character_count(said, words):
    from montagewright.backfill import across_lines

    return sum(
        1 for _text, (_start, _end, pieces) in zip(said, across_lines(said, words))
        for piece in pieces
        if piece.ends_seconds <= piece.starts_seconds
        and any(char.isalnum() for char in piece.text)
    )


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
        candidates = []
        for word in words_of(payload):
            word = replace(word, starts_seconds=word.starts_seconds+start,
                           ends_seconds=word.ends_seconds+start)
            # Keep original measured words immutable. Recovery fills only holes.
            if any(word.starts_seconds < w.ends_seconds and word.ends_seconds > w.starts_seconds
                   for w in recovered):
                continue
            candidates.append(word)
        before = _unresolved_character_count(said, recovered)
        proposed = sorted([*recovered, *candidates], key=lambda word: word.starts_seconds)
        added = len(candidates) if _unresolved_character_count(said, proposed) < before else 0
        if added:
            recovered = proposed
        attempts.append({"start_seconds": start, "end_seconds": end,
                         "words_added": added, "timing_source": "apple_local_retry",
                         "evidence": str(path) if path else None})
    recovered.sort(key=lambda word: word.starts_seconds)
    # The regular retry uses the same signal and a Gemini-sized window. If
    # Apple still left a genuine gap, listen once more to just that gap with
    # local level normalization. Never replace measured words or invent times.
    for start, end in _unheard_gaps(said, recovered):
        path = (root / (key_for({"source": content_hash(audio), "locale": locale,
                "start": start, "end": end, "recovery": 2}) + ".json")
                if root else None)
        payload = read_json(path) if path else None
        if payload is None:
            with tempfile.TemporaryDirectory() as folder:
                clip = Path(folder) / "speech.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio),
                    "-ss", str(start), "-t", str(end-start), "-vn", "-ac", "1",
                    "-ar", "16000", "-af", "highpass=f=80,dynaudnorm=f=75:g=9",
                    str(clip)], check=True, capture_output=True)
                payload = hear(clip, locale=locale)
            if path:
                write_json(path, payload)
        candidates = []
        for word in words_of(payload):
            word = replace(word, starts_seconds=word.starts_seconds+start,
                           ends_seconds=word.ends_seconds+start)
            if word.starts_seconds < start + 0.25 or word.ends_seconds > end - 0.25:
                continue
            if any(word.starts_seconds < existing.ends_seconds
                   and word.ends_seconds > existing.starts_seconds
                   for existing in recovered):
                continue
            candidates.append(word)
        before = _unresolved_character_count(said, recovered)
        proposed = sorted([*recovered, *candidates], key=lambda word: word.starts_seconds)
        added = len(candidates) if _unresolved_character_count(said, proposed) < before else 0
        if added:
            recovered = proposed
        attempts.append({"start_seconds": start, "end_seconds": end,
                         "words_added": added,
                         "timing_source": "apple_normalized_gap_retry",
                         "evidence": str(path) if path else None})
    return sorted(recovered, key=lambda w: w.starts_seconds), attempts
