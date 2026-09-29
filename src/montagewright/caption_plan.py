"""Gemini chooses semantic breaks; Apple-derived character spans own time."""
from dataclasses import replace
import json


def save_cues(path, lines):
    from montagewright.checkpoints import write_json
    from dataclasses import asdict
    write_json(path, [{"at": line.starts_seconds, "until": line.ends_seconds,
                       "text": line.text, "heard": line.heard, "speaker": line.speaker,
                       "timing_source": line.timing_source, "timing_confidence": line.timing_confidence,
                       "timing_locked": line.timing_locked,
                       "timed_text": [asdict(mark) for mark in line.timed_text]} for line in lines])


def repair_cues(lines, words, *, video, feedback, aspect, width, height, client, cache, ledger):
    """Re-listen to disputed delivered speech; keep Apple as timing authority."""
    from montagewright.planner import ask, _parse, MODEL_ID
    from montagewright.gemini import video_content, static_video_processing, structured_json
    from montagewright.transcript import Line
    from montagewright.backfill import across_lines
    from montagewright.subtitles import as_cues
    from montagewright.uploads import upload_now
    uri = cache.uri_for(video, client, mime_type="video/mp4")[0] if cache else upload_now(video, client).uri
    schema = {"type": "object", "required": ["lines"], "properties": {
        "lines": {"type": "array", "items": {"type": "object", "required": ["text", "speaker"],
                   "properties": {"text": {"type": "string"}, "speaker": {"type": "string"}}}}}}
    response = ask(client, model=MODEL_ID, store=False, upload_cache=cache, ledger=ledger,
        budget_stage="caption_repair", generation_config={"thinking_level": "high", "max_output_tokens": 8192},
        input=[video_content(uri, resolution="high", processing=static_video_processing(2)),
               {"type": "text", "text": "回聽成片原音並修正字幕，不能依燒錄文字猜答案。保留完整發言，不改寫句意。"
                "依原音可增刪錯漏字，不產生时间。非文字問題保留原文，不能改字掩蓋構圖問題。\n"
                + json.dumps({"review": feedback, "current_lines": [{"text": x.text, "speaker": x.speaker} for x in lines]}, ensure_ascii=False)}],
        response_format=structured_json(schema))
    rows = _parse(response, what="caption repair").get("lines", [])
    if not rows or not words:
        raise ValueError("caption repair requires speech and Apple timing evidence")
    texts = [str(row["text"]).strip() for row in rows]
    timings = across_lines(texts, words)
    fixed = []
    for row, text, (start, end, marks) in zip(rows, texts, timings, strict=True):
        if end <= start or any(mark.ends_seconds <= mark.starts_seconds and any(c.isalnum() for c in mark.text) for mark in marks):
            raise ValueError("caption repair inserted speech without an Apple timing anchor; retain pending evidence")
        fixed.append(Line(text, start, end, speaker=row["speaker"], timed_text=tuple(marks)))
    return as_cues(fixed, aspect, width, height, words=words, client=client, ledger=ledger,
                   context=json.dumps(feedback, ensure_ascii=False))


def semantic_cues(lines, *, aspect, face, room, client, ledger, context=""):
    from montagewright.planner import ask, _parse, MODEL_ID
    from montagewright.gemini import structured_json
    from montagewright.subtitles import _width, _character_marks
    schema = {"type": "object", "required": ["lines"], "properties": {
        "lines": {"type": "array", "items": {"type": "object",
            "required": ["index", "pieces"], "properties": {
                "index": {"type": "integer"},
                "pieces": {"type": "array", "items": {"type": "string"}},
            }}}}}
    prompt = (
        "為成片做繁體中文字幕語意斷句。只能拆開原文字，不增刪改字，標點與空白也保留。"
        "每個 index 都要出現且順序不變。pieces 串接必須逐字等於原文。"
        "保護專有名詞、數字單位、否定詞與完整詞組；不可留下單字孤行。"
        "不用時間碼。每個 piece 是一個依序出現的單行 cue。"
        f"比例 {aspect}，最大字寬 {room}px；字體量測的每字寬如下。"
        "避免過短閃爍或長句讀不完，換人不合併。\n"
        + context + "\n" + json.dumps([
            {"index": i, "text": line.text, "speaker": line.speaker,
             "duration": line.ends_seconds-line.starts_seconds,
             "character_widths": [round(_width(c, face), 2) for c in line.text]}
            for i, line in enumerate(lines)
        ], ensure_ascii=False)
    )
    faults = []
    for attempt in range(2):
        response = ask(client, model=MODEL_ID, store=False,
            input=[{"type": "text", "text": prompt + (
                "\n上一版未通過：" + "; ".join(faults) if faults else "")}],
            generation_config={"thinking_level": "high", "max_output_tokens": 16384},
            response_format=structured_json(schema), ledger=ledger, budget_stage="subtitle_layout")
        payload = _parse(response, what="semantic subtitle layout")
        rows = payload.get("lines") or []
        faults = []
        if [r.get("index") for r in rows] != list(range(len(lines))):
            faults.append("index 遺漏、重複或順序錯誤")
        else:
            out = []
            for line, row in zip(lines, rows):
                pieces = row.get("pieces") or []
                if not pieces or any(not p for p in pieces) or "".join(pieces) != line.text:
                    faults.append(f"{row['index']} 文字被改動")
                    continue
                if any(_width(p, face) > room for p in pieces):
                    faults.append(f"{row['index']} 超過 {room}px")
                    continue
                marks = _character_marks(line)
                if len(pieces) > 1 and not marks:
                    faults.append(f"{row['index']} 缺 Apple 文字時間對應，不能猜測斷句時間")
                    continue
                cursor = 0
                for i, piece in enumerate(pieces):
                    end = cursor + len(piece)
                    starts = next((a for lo, hi, a, b in marks if hi > cursor), line.starts_seconds)
                    finishes = next((a for lo, hi, a, b in marks if lo >= end), line.ends_seconds)
                    if i == 0:
                        starts = line.starts_seconds
                    if finishes-starts < 0.6 and len(pieces) > 1:
                        faults.append(f"{row['index']}「{piece}」不足 0.6 秒，請重新組句")
                    # Slice character evidence together with text, never duplicate a
                    # whole word's glyphs into both visual fragments.
                    clock = tuple(line.timed_text[cursor:end])
                    out.append(replace(line, text=piece, starts_seconds=starts,
                                       ends_seconds=finishes, timed_text=clock))
                    cursor = end
            if not faults:
                return out
    raise ValueError("字幕語意排版未通過：" + "; ".join(faults))
