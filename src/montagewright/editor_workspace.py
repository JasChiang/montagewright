"""Durable editorial context and bounded, source-clock video tools.

The model chooses what to inspect. These tools own files, clocks and preview
geometry; an inspection never silently changes the accepted edit.
"""
from __future__ import annotations

import json
import math
import subprocess
import shutil
import hashlib
from pathlib import Path

from montagewright.checkpoints import key_for, read_json, write_json
from montagewright.uploads import content_hash


ASPECTS = {"16:9": (640, 360), "9:16": (360, 640), "1:1": (480, 480), "4:5": (384, 480)}


def material_index(material):
    return [{"source_id": m.source_id, "seconds": m.duration_seconds,
             "summary": m.summary, "composition": m.composition,
             "subjects": list(m.subjects), "motion": m.camera_motion,
             "identity_absent_targets": list(m.identity_absent_targets),
             "speech": list(m.speech), "audio_spans": list(m.audio_spans),
             "action_windows": list(m.action_windows),
             "geometry": {"push_room": m.push_room, "pan_room": m.pan_room, "tilt_room": m.tilt_room},
             "spans": [{"span_id": s.span_id, "start": s.starts_seconds,
                        "end": s.ends_seconds, "why": s.why} for s in m.spans]}
            for m in material]


class EditorWorkspace:
    def __init__(self, root: Path, material, *, brief, direction, selection,
                 preview: Path | None = None, timeline=None, problems=()):
        self.root = root
        self.material = {m.source_id: m for m in material}
        self.preview = preview
        self.aspect = direction.get("aspect", "9:16")
        if self.aspect not in ASPECTS:
            raise ValueError(f"unresolved preview aspect {self.aspect}")
        self.context = {"version": "editor-workspace-v1", "brief": brief,
                        "direction": direction, "selection": selection,
                        "timeline": timeline, "problems": list(problems),
                        "materials": material_index(material),
                        "source_hashes": {m.source_id: content_hash(m.proxy)
                            for m in material if m.proxy and m.proxy.exists()},
                        "preview_hash": content_hash(preview) if preview and preview.exists() else None}
        # Make source-level identity observations available to targeted
        # Agentic investigation, without treating an index as viewed footage.
        coverage_path = root.parent / "identity-screen-coverage.json"
        coverage = read_json(coverage_path) if coverage_path.exists() else None
        if coverage:
            self.context["grounding_index"] = [{
                "source_id": row["source_id"],
                "instances": [{key: candidate.get(key) for key in (
                    "candidate_id", "target_id", "start_ms", "end_ms", "identity_status", "visible_state")}
                    for candidate in (row.get("discovery") or {}).get("candidates", [])],
                "status": row.get("status"),
            } for row in coverage.get("sources", []) if row.get("source_id") in self.material]
        blocked_path = root.parent / "grounding-blocked.json"
        if blocked_path.exists():
            self.context["grounding_failures"] = read_json(blocked_path).get("failures", [])
        self.revision = key_for(self.context)
        self.context["revision"] = self.revision
        self.directory = root / "revisions" / self.revision
        write_json(self.directory / "context.json", self.context)
        write_json(root / "current.json", {"revision": self.revision})

    def record(self, kind, payload):
        value = {"revision": self.revision, "kind": kind, "payload": payload}
        write_json(self.directory / "events" / f"{key_for(value)}.json", value)

    def inspect(self, request):
        """Render a bounded source or timeline interval, keeping its clock map."""
        allowed = {"inspect_source", "inspect_cut", "preview_framing"}
        operation = request.get("operation")
        if operation not in allowed:
            raise ValueError(f"unknown editorial tool {operation}")
        if operation == "inspect_cut":
            path = self.preview
            if path is None or not path.exists():
                raise ValueError("no current cut exists")
            info = _probe(path)
            duration = float(info["format"]["duration"])
            source_id = "current_cut"
        else:
            source_id = request.get("source_id")
            if source_id not in self.material:
                raise ValueError("source must come from the material index")
            item = self.material[source_id]
            path, duration = item.proxy, item.duration_seconds
            if path is None or not path.exists():
                raise ValueError("source video is unavailable")
        from montagewright.spans import seconds_of
        parsed_start, parsed_end = seconds_of(request["start"]), seconds_of(request["end"])
        if parsed_start is None or parsed_end is None:
            raise ValueError("interval requires M:SS clock readings")
        start, end = float(parsed_start), float(parsed_end)
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration + .001:
            raise ValueError(f"interval must lie in [0, {duration}]")
        if end - start > 20:
            raise ValueError("inspect at most 20 seconds per tool call; request another interval if needed")
        mode = request.get("framing", "source") if operation == "preview_framing" else "source"
        if mode not in {"source", "fit", "fill", "pan_left_to_right", "pan_right_to_left"}:
            raise ValueError("unsupported preview framing")
        identity = {"revision": self.revision, "source": source_id, "start": start,
                    "end": end, "framing": mode, "aspect": self.aspect}
        artifact = self.root / "previews" / f"{key_for(identity)}.mp4"
        if not artifact.exists():
            artifact.parent.mkdir(parents=True, exist_ok=True)
            filters = framing_filter(mode, self.aspect, end-start)
            partial = artifact.with_suffix(".partial.mp4")
            try:
                completed = subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(path),
                    "-t", str(end-start), "-map", "0:v:0", "-map", "0:a?",
                    "-vf", filters, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-movflags", "+faststart", str(partial)], capture_output=True, text=True, timeout=120)
                if completed.returncode:
                    raise RuntimeError("preview render failed: " + completed.stderr[-1500:])
                partial.replace(artifact)
            finally:
                partial.unlink(missing_ok=True)
        result = {**identity, "operation": operation, "source_sha256": content_hash(path), "path": str(artifact), "sha256": content_hash(artifact),
                  "clock": "timeline" if operation == "inspect_cut" else "source",
                  "mapping": "preview time + start = original time; speed 1",
                  "purpose": "diagnostic preview, not an accepted replacement",
                  "tradeoffs": _tradeoffs(mode)}
        write_json(artifact.with_suffix(".json"), result)
        self.record("tool_result", result)
        return result


def _probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))


def _tradeoffs(mode):
    return {"source": "uncropped original context", "fit": "preserves the full image with padding; details become smaller",
            "fill": "fills the canvas but may remove subjects or words at the sides",
            "pan_left_to_right": "sequential view, not simultaneous visibility; inspect subject and text readability",
            "pan_right_to_left": "sequential view, not simultaneous visibility; inspect subject and text readability"}[mode]


def framing_filter(mode, aspect, seconds):
    w, h = ASPECTS[aspect]
    if mode == "source":
        return "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps=30"
    if mode == "fit":
        return f"scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
    scale = f"scale={w}:{h}:force_original_aspect_ratio=increase:force_divisible_by=2"
    x = "(iw-ow)/2"
    if mode.startswith("pan_"):
        progress = f"min(t/{seconds:.6f},1)"
        x = f"(iw-ow)*({progress})" if mode == "pan_left_to_right" else f"(iw-ow)*(1-{progress})"
    return f"{scale},crop={w}:{h}:x='{x}':y='(ih-oh)/2',setsar=1,fps=30"


def _tool_schema():
    return {"type": "object", "required": ["requests", "ready", "reason"], "properties": {
        "ready": {"type": "boolean"}, "reason": {"type": "string"},
        "requests": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "required": ["operation", "source_id", "start", "end", "framing"],
            "properties": {"operation": {"type": "string", "enum": ["inspect_source", "inspect_cut", "preview_framing"]},
                "source_id": {"type": "string"}, "start": {"type": "string", "description": "M:SS.sss original clock"}, "end": {"type": "string", "description": "M:SS.sss original clock"},
                "framing": {"type": "string", "enum": ["source", "fit", "fill", "pan_left_to_right", "pan_right_to_left"]}}}}}}


def inspected_selection_faults(shots, results):
    faults = []
    for shot in shots:
        start = float(shot.get("start_seconds", 0))
        end = start + float(shot.get("seconds_needed", 0)) * float(shot.get("speed") or 1)
        windows = sorted((r["start"], r["end"]) for r in results
                         if r["source"] == shot.get("source_id") and r["clock"] == "source")
        covered = start
        for left, right in windows:
            if left <= covered + .001:
                covered = max(covered, right)
        if covered + .001 < end:
            faults.append(f"{shot.get('replace_clip_id', 'shot')} chooses source time {start}-{end} outside inspected footage")
    return faults


def decoded_digest(path):
    """Compare picture AND sound, independent of container metadata."""
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a?",
        "-vf", "scale=160:160:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v", "rawvideo", "-c:a", "pcm_s16le", "-f", "framemd5", "-"],
        capture_output=True, check=True, timeout=120)
    return hashlib.sha256(result.stdout).hexdigest()


def record_render(root, *, material, brief, direction, selection, preview, timeline):
    old = read_json(root / "last-render.json")
    state = EditorWorkspace(root, material, brief=brief, direction=direction,
                            selection=selection, preview=preview, timeline=timeline)
    snapshot = state.directory / "preview.mp4"
    if not snapshot.exists():
        temporary = snapshot.with_suffix(".partial.mp4")
        shutil.copy2(preview, temporary)
        temporary.replace(snapshot)
    digest = decoded_digest(snapshot)
    result = {"revision": state.revision, "decoded_digest": digest,
              "preview": str(snapshot), "previous": old,
              "changed": old is None or digest != old["decoded_digest"]}
    # History is stored by revision; avoid recursively copying all old history.
    if old:
        result["previous"] = ({k: old[k] for k in ("revision", "decoded_digest", "preview")}
                              if result["changed"] else old.get("previous"))
    state.record("render_result", result)
    write_json(root / "last-render.json", result)
    return result


def gather_evidence(workspace, *, client, cache, ledger, initial=(), rounds=3, max_evidence_seconds=180):
    """Gemini navigates a full catalog while only requested video is attached.

    The JSON tool protocol deliberately uses the existing checkpointed ask()
    boundary. It does not require persistent remote conversations or arbitrary
    shell/function execution. Every turn reconstructs the current context.
    """
    from montagewright.planner import ask, _parse, MODEL_ID
    from montagewright.gemini import structured_json, video_content, static_video_processing
    from montagewright.uploads import upload_now
    results, history = [], []
    seen = set()
    evidence_seconds = 0.0
    def execute(request):
        nonlocal evidence_seconds
        if not isinstance(request, dict):
            history.append({"error": "tool request must be an object"})
            return
        signature = key_for(request)
        if signature in seen:
            return
        seen.add(signature)
        try:
            from montagewright.spans import seconds_of
            start, end = seconds_of(request.get("start")), seconds_of(request.get("end"))
            if start is None or end is None or not math.isfinite(end-start) or end <= start:
                raise ValueError("invalid inspection interval")
            if evidence_seconds + end-start > max_evidence_seconds:
                raise ValueError("inspection evidence limit reached; choose from inspected footage")
            results.append(workspace.inspect(request))
            evidence_seconds += end-start
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            history.append({"request": request, "error": str(error)})
    for request in initial:
        execute(request)
    def parts():
        body = [{"type": "text", "text": json.dumps({"context": workspace.context, "tool_history": history}, ensure_ascii=False)}]
        for result in results:
            path = Path(result["path"])
            uri = cache.uri_for(path, client, mime_type="video/mp4")[0] if cache else upload_now(path, client).uri
            body.extend([{"type": "text", "text": json.dumps(result, ensure_ascii=False)},
                         video_content(uri, resolution="high", processing=("agentic" if result.get("operation") == "inspect_source" else static_video_processing(4)))])
        return body
    for turn in range(min(rounds, 3)):
        prompt = (
            "你是剪輯師。保留全片敘事與使用者硬條件，先診斷問題，再選擇需要的本機工具。"
            "完整素材目錄始終可查；不要把摘要當作看過影片。inspect_source 回看來源區間，"
            "inspect_cut 看目前成片（source_id=current_cut），preview_framing 試構圖並與原片比較。"
            "時間填 M:SS.sss 來源時鐘或成片時鐘，不是 span offset。每次最多三個區間，每個最長20秒。"
            "fit 保留全景但縮小，fill 會裁切，pan 是依序展示不是同時展示。"
            "這些是診斷候選；最終仍須提交可由剪輯引擎執行的替換計畫。"
            "grounding_index 是候選觀察而非最終證明；同 target_id 可以有多台不同實體，不能把其他同型號自動視為排除機種。"
            "核驗失敗要查明是回覆截斷、身份不確定、追蹤失敗或構圖放不下；不能取消使用者的主體限制。"
            "可選擇其他素材但先要求回看要選的區間。證據夠了就 ready=true，requests=[]。"
            "不要重複請求已看過的區間。剩餘工具輪數：" + str(rounds-turn)
        )
        interaction = ask(client, model=MODEL_ID, store=False,
            input=parts()+[{"type": "text", "text": prompt}], upload_cache=cache,
            generation_config={"thinking_level": "low", "max_output_tokens": 8192},
            response_format=structured_json(_tool_schema()), ledger=ledger, budget_stage="editor_tools")
        answer = _parse(interaction, what="editor tools")
        from montagewright.checkpoints import capture
        workspace.record("video_processing", capture(interaction, "editor_tools", MODEL_ID))
        history.append(answer)
        workspace.record("tool_decision", {"turn": turn, "decision": answer})
        requests = answer.get("requests", [])
        if not isinstance(requests, list) or len(requests) > 3:
            raise ValueError("editor returned more than three tool requests")
        if answer.get("ready") and not requests:
            break
        before = len(seen)
        for request in requests:
            execute(request)
        if len(seen) == before:
            break
    workspace.record("evidence_complete", {"results": results, "history": history})
    return parts(), results
