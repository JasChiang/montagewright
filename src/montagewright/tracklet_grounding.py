"""Tracklet grounding: local code proposes every instance, the model picks.

The previous shot-window path asked Gemini two things at once on every exact
frame -- is this the locked identity, and where exactly is it -- and handed
the answer to SAM as a seed. Identity is the model's strength; a box in
0..1000 space is not, and asking per frame made each frame its own
independent guess. Two phones of the same SKU collapsed to one, a lookalike
next to the target could win a frame, and a group the edit asked for was
cropped because only one member was ever located.

Here the division follows the span contract that fixed time: the model may
only refer to things that exist.

1. An open-vocabulary detector boxes every candidate instance on frames
   sampled across the cut. It knows categories, not identities.
2. Detections are linked across frames into tracklets, each with a number.
3. One contact sheet of the cut, the numbers drawn on it, goes to Gemini
   with the reference images. It answers which numbers are the target,
   which are other products, and which numbers the edit needs in frame
   together. It never writes a coordinate.
4. The chosen tracklets become the per-frame boxes the crop path already
   reads, so nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DETECTOR_ID = "IDEA-Research/grounding-dino-base"
# What the detector is asked to find. Categories only: identity is the
# model's answer, not the detector's, so this list stays deliberately broad.
# A lookalike must still be proposed, otherwise it cannot be named and
# excluded.
DEFAULT_PHRASES = (
    "a phone", "a smartphone", "a foldable phone", "a tablet", "a smartwatch",
)
SAMPLES_PER_SECOND = 4.0
MAX_SAMPLES = 32
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
NMS_IOU = 0.6
LINK_IOU = 0.25
MAX_GAP_SAMPLES = 2
MIN_PRESENCE = 0.2
MIN_AREA = 0.0015
SHEET_MOMENTS = (0.15, 0.5, 0.85)
PICK_VERSION = "tracklet-pick-v1"
PICK_OUTPUT_TOKENS = 3072

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in 0..1


@dataclass
class Detection:
    sample: int
    box: Box
    score: float
    label: str


@dataclass
class Tracklet:
    number: int
    detections: dict[int, Detection] = field(default_factory=dict)

    def presence(self, samples: int) -> float:
        return len(self.detections) / max(1, samples)

    def box_at(self, sample: int) -> Box | None:
        """The detected box, or a straight interpolation across a short gap."""

        if sample in self.detections:
            return self.detections[sample].box
        seen = sorted(self.detections)
        before = [one for one in seen if one < sample]
        after = [one for one in seen if one > sample]
        if not before or not after:
            return None
        a, b = before[-1], after[0]
        if b - a > MAX_GAP_SAMPLES + 1:
            return None
        share = (sample - a) / (b - a)
        box_a, box_b = self.detections[a].box, self.detections[b].box
        return tuple(  # type: ignore[return-value]
            va + (vb - va) * share for va, vb in zip(box_a, box_b)
        )

    def mean_area(self) -> float:
        return sum(
            (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
            for d in self.detections.values()
        ) / max(1, len(self.detections))


class TrackletGroundingError(RuntimeError):
    pass


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )
    return inter / union if union > 0 else 0.0


def sample_frames(
    source: Path, start: float, end: float, work: Path
) -> list[tuple[float, Path]]:
    """Frames across the cut on the source clock, one ffmpeg pass."""

    span = max(0.1, end - start)
    fps = min(SAMPLES_PER_SECOND, MAX_SAMPLES / span)
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("s_*.jpg"):
        stale.unlink()
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", str(source),
            "-vf", f"fps={fps:.4f},scale=960:-2", "-q:v", "3",
            str(work / "s_%03d.jpg"),
        ],
        check=True,
    )
    frames = sorted(work.glob("s_*.jpg"))
    # The fps filter emits the frame nearest each tick starting at the
    # window's first frame; the n-th output is at start + n / fps.
    return [(start + index / fps, path) for index, path in enumerate(frames)]


_DETECTOR: tuple[Any, Any, str] | None = None


def _detector() -> tuple[Any, Any, str]:
    global _DETECTOR
    if _DETECTOR is None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(DETECTOR_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTOR_ID
        ).to(device).eval()
        _DETECTOR = (processor, model, device)
    return _DETECTOR


def _nms(detections: list[Detection]) -> list[Detection]:
    """Class-agnostic: one physical thing proposed as 'phone' and as
    'foldable phone' is one proposal."""

    kept: list[Detection] = []
    for one in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(_iou(one.box, other.box) < NMS_IOU for other in kept):
            kept.append(one)
    return kept


def detect(
    frames: Sequence[tuple[float, Path]],
    phrases: Sequence[str] = DEFAULT_PHRASES,
) -> list[list[Detection]]:
    import torch
    from PIL import Image

    processor, model, device = _detector()
    text = ". ".join(phrases) + "."
    out: list[list[Detection]] = []
    for sample, (_, path) in enumerate(frames):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]
        labels = result.get("text_labels", result.get("labels"))
        found = []
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            x0, y0, x1, y1 = (float(v) for v in box)
            normalised = (
                max(0.0, x0 / width), max(0.0, y0 / height),
                min(1.0, x1 / width), min(1.0, y1 / height),
            )
            area = (normalised[2] - normalised[0]) * (normalised[3] - normalised[1])
            if area < MIN_AREA:
                continue
            found.append(Detection(sample, normalised, float(score), str(label)))
        out.append(_nms(found))
    return out


def link(per_frame: Sequence[Sequence[Detection]]) -> list[Tracklet]:
    """Join detections frame to frame by overlap; a short miss is a gap,
    not a new object."""

    from scipy.optimize import linear_sum_assignment

    tracklets: list[Tracklet] = []
    last_seen: dict[int, int] = {}
    for sample, detections in enumerate(per_frame):
        live = [
            one for one in tracklets
            if sample - last_seen[one.number] <= MAX_GAP_SAMPLES + 1
        ]
        assigned: set[int] = set()
        if live and detections:
            cost = [
                [
                    1.0 - _iou(
                        one.detections[last_seen[one.number]].box, detection.box
                    )
                    for detection in detections
                ]
                for one in live
            ]
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if 1.0 - cost[row][col] >= LINK_IOU:
                    live[row].detections[sample] = detections[col]
                    last_seen[live[row].number] = sample
                    assigned.add(col)
        for index, detection in enumerate(detections):
            if index in assigned:
                continue
            fresh = Tracklet(number=len(tracklets) + 1)
            fresh.detections[sample] = detection
            tracklets.append(fresh)
            last_seen[fresh.number] = sample
    samples = len(per_frame)
    kept = [
        one for one in tracklets
        if one.presence(samples) >= MIN_PRESENCE or len(one.detections) >= 3
    ]
    # Renumber densely so the sheet never shows a gap that reads as a
    # missing answer.
    for number, one in enumerate(kept, start=1):
        one.number = number
    return kept


_COLOURS = (
    "#ffd400", "#00e5ff", "#ff4fd8", "#7cff4f", "#ff8a00",
    "#9d7bff", "#ffffff", "#ff3b3b", "#3bff9d", "#4f8aff",
)


def contact_sheet(
    frames: Sequence[tuple[float, Path]],
    tracklets: Sequence[Tracklet],
    destination: Path,
) -> list[int]:
    """Three moments of the cut side by side, every tracklet outlined and
    numbered where it is. Returns the sample indices shown."""

    from PIL import Image, ImageDraw, ImageFont

    count = len(frames)
    shown = sorted({
        min(count - 1, max(0, round(share * (count - 1))))
        for share in SHEET_MOMENTS
    })
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26
        )
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for sample in shown:
        image = Image.open(frames[sample][1]).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for one in tracklets:
            box = one.box_at(sample)
            if box is None:
                continue
            colour = _COLOURS[(one.number - 1) % len(_COLOURS)]
            x0, y0, x1, y1 = (
                box[0] * width, box[1] * height, box[2] * width, box[3] * height
            )
            draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
            tag = f"#{one.number}"
            left, top, right, bottom = draw.textbbox((0, 0), tag, font=font)
            tag_w, tag_h = right - left + 10, bottom - top + 8
            ty = y0 - tag_h if y0 - tag_h >= 0 else y0
            draw.rectangle([x0, ty, x0 + tag_w, ty + tag_h], fill=colour)
            draw.text((x0 + 5, ty + 2), tag, fill="black", font=font)
        draw.rectangle([0, 0, 150, 34], fill="black")
        draw.text((6, 3), f"{frames[sample][0]:.1f}s", fill="white", font=font)
        tiles.append(image)
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * len(tiles), height), "black")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * width, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=88)
    return shown


def _pick_schema(numbers: Sequence[int]) -> dict[str, Any]:
    # Gemini's response schema accepts enums of strings only; an integer
    # enum is dropped on the way through and the items come back empty.
    labels = [str(one) for one in numbers]
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "description": "Exactly one entry per numbered outline.",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "string", "enum": labels},
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "target", "other_product", "not_a_product",
                                "uncertain",
                            ],
                        },
                        "evidence": {"type": "string"},
                    },
                    "required": ["number", "verdict", "evidence"],
                },
            },
            "required_numbers": {
                "type": "array",
                "items": {"type": "string", "enum": labels},
                "description": (
                    "Numbers that must be in frame together for the shot "
                    "to deliver what the edit asks for. Empty when the "
                    "target is not in this cut."
                ),
            },
            "target_visible_without_outline": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": [
            "assignments", "required_numbers",
            "target_visible_without_outline", "note",
        ],
    }


PICK_PROMPT = """你是剪輯助理，負責在一顆鏡頭裡指認產品身份。

上面是參考圖與身份目錄（target 的 identity cues、stable exclusions）。
最後一張圖是同一顆鏡頭的三個時刻並排（左到右時間遞增，左上角是來源秒數）。
本機偵測器把畫面裡每一個候選物件框起來並編號 #1、#2…；同一個號碼在三個時刻是同一個物件。
偵測器只懂類別，不懂型號，而且可能框錯（例如把手臂上的圖案當成手錶）。

請做兩件事，不要寫任何座標：

1. 對每一個號碼判斷：
   - target：符合 target 的身份（依 identity_semantics；同 SKU 的不同實機都算 target）
   - other_product：是產品，但屬於 stable exclusions 或其他型號
   - not_a_product：框錯了，不是產品
   - uncertain：看不清，無法判斷
   evidence 寫你看得到的具體依據（機身比例、相機排列、折疊方式…）。

2. 這顆鏡頭在剪輯上要呈現：{intent}
   在 required_numbers 列出「必須同時留在畫面裡」才能完成這個呈現的號碼。
   只要一台就能完成時只列一台；要並排比較、多色展示、手持互動時，列出所有參與的號碼。
   只能列 verdict 是 target 的號碼，或和 target 直接互動而不可切掉的物件。
   target 不在這顆鏡頭裡就回空陣列。

如果你看到 target 但它沒有被任何框框住，target_visible_without_outline 填 true。
"""


def pick(
    spec: Any,
    target_id: str,
    sheet: Path,
    numbers: Sequence[int],
    intent: str,
    *,
    client: Any,
    cache: Any | None,
    ledger: Any | None,
) -> tuple[dict[str, Any], Any]:
    from montagewright.planner import MODEL_ID, Usage, ask
    from montagewright.gemini import structured_json
    from montagewright.reference_grounding import (
        _parse_payload,
        reference_prompt_parts,
    )

    parts = reference_prompt_parts(
        spec, client=client, cache=cache, target_ids=[target_id],
        resolution="high",
    )
    parts.append({"type": "text", "text": "SHOT CONTACT SHEET follows."})
    parts.append({
        "type": "image",
        "mime_type": "image/jpeg",
        "uri": sheet.resolve().as_uri() if cache is None else _sheet_uri(
            sheet, client, cache
        ),
        "resolution": "high",
    })
    parts.append({
        "type": "text",
        "text": PICK_PROMPT.format(intent=intent or "清楚呈現 target")
        + f"\nNUMBERS={list(numbers)} TARGET_ID={target_id}",
    })
    interaction = ask(
        client,
        upload_cache=cache,
        model=MODEL_ID,
        store=False,
        input=parts,
        patience_seconds=180.0,
        generation_config={
            "thinking_level": "low",
            "max_output_tokens": PICK_OUTPUT_TOKENS,
        },
        response_format=structured_json(_pick_schema(numbers)),
        ledger=ledger,
        budget_stage="tracklet_pick",
    )
    payload = _parse_payload(interaction, "tracklet pick")
    return payload, Usage.from_interaction(interaction)


def _sheet_uri(sheet: Path, client: Any, cache: Any) -> str:
    uri, _ = cache.uri_for(sheet, client, mime_type="image/jpeg")
    return uri


def validate_pick(
    payload: dict[str, Any], numbers: Sequence[int]
) -> dict[str, Any]:
    allowed = set(numbers)
    verdicts: dict[int, dict[str, str]] = {}
    for one in payload.get("assignments") or []:
        try:
            number = int(str(one.get("number", "")).lstrip("#"))
        except ValueError:
            number = -1
        if number not in allowed or number in verdicts:
            raise TrackletGroundingError(
                f"pick named #{number}, which is not a proposed outline "
                "or was answered twice"
            )
        verdicts[number] = {
            "verdict": str(one.get("verdict")),
            "evidence": str(one.get("evidence") or ""),
        }
    # An outline left unanswered is not evidence of anything. It becomes
    # `uncertain`: it can never be the target, and it does not throw away
    # the answers the model did give about the others.
    for number in allowed - set(verdicts):
        verdicts[number] = {"verdict": "uncertain", "evidence": "unanswered"}
    required = []
    for one in payload.get("required_numbers") or []:
        try:
            value = int(str(one).lstrip("#"))
        except ValueError:
            continue
        if value in allowed:
            required.append(value)
    targets = [n for n, v in verdicts.items() if v["verdict"] == "target"]
    # A required number must be the target or something the target is being
    # used with. An other_product can never be required -- that is the
    # lookalike standing in for the target, the failure this path exists to
    # stop.
    required = [
        n for n in required if verdicts[n]["verdict"] != "other_product"
    ]
    return {
        "verdicts": verdicts,
        "targets": sorted(targets),
        "required": sorted(set(required)),
        "unboxed_target": bool(payload.get("target_visible_without_outline")),
        "note": str(payload.get("note") or ""),
    }


def union_box(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def cache_key(
    source: Path, start: float, end: float, target_id: str,
    spec_sha: str, intent: str,
) -> str:
    stat = source.stat()
    material = json.dumps({
        "version": PICK_VERSION, "detector": DETECTOR_ID,
        "phrases": DEFAULT_PHRASES, "source": str(source.resolve()),
        "size": stat.st_size, "mtime": int(stat.st_mtime),
        "start": round(start, 3), "end": round(end, 3),
        "target": target_id, "spec": spec_sha, "intent": intent,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def ground_cut(
    source: Path,
    start: float,
    end: float,
    target_id: str,
    *,
    spec: Any,
    intent: str,
    client: Any,
    cache: Any | None,
    ledger: Any | None,
    work: Path,
    memory: Path | None = None,
) -> dict[str, Any]:
    """Locate the target in one cut. Returns per-sample geometry and the
    record of how identity was decided; raises when nothing can be proved.

    The returned samples are the union of the required tracklets at each
    sampled moment, so a comparison or a colour line-up is framed as the
    group the edit asked for rather than whichever member was found first.
    """

    spec_sha = (
        spec.definition_sha256() if hasattr(spec, "definition_sha256")
        else hashlib.sha256(repr(spec).encode("utf-8")).hexdigest()
    )
    key = cache_key(source, start, end, target_id, spec_sha, intent)
    remembered = (memory / f"tracklet-{key[:24]}.json") if memory else None
    if remembered is not None and remembered.exists():
        saved = json.loads(remembered.read_text())
        if saved.get("key") == key:
            return saved["result"]
    if client is None:
        # Replays and recuts run without a key. Without a remembered pick
        # there is no identity answer to give, and detecting candidates
        # nobody can name only costs time.
        raise TrackletGroundingError(
            "no client and no remembered pick for this cut"
        )

    frames = sample_frames(source, start, end, work / f"tracklets-{key[:12]}")
    if len(frames) < 2:
        raise TrackletGroundingError(
            f"only {len(frames)} frame(s) decoded in {start:.2f}-{end:.2f}s"
        )
    tracklets = link(detect(frames))
    usage = None
    if not tracklets:
        result = {
            "status": "no_candidates",
            "times": [t for t, _ in frames],
            "samples": [],
            "tracklets": [],
            "pick": None,
        }
    else:
        sheet = work / f"tracklets-{key[:12]}" / "sheet.jpg"
        shown = contact_sheet(frames, tracklets, sheet)
        numbers = [one.number for one in tracklets]
        payload, usage = pick(
            spec, target_id, sheet, numbers, intent,
            client=client, cache=cache, ledger=ledger,
        )
        decided = validate_pick(payload, numbers)
        if decided["targets"] and not set(decided["required"]) & set(
            decided["targets"]
        ):
            # Nothing the model required is the target: fall back to the
            # most present target, never to a lookalike.
            by_presence = max(
                (one for one in tracklets if one.number in decided["targets"]),
                key=lambda one: (len(one.detections), one.mean_area()),
            )
            decided["required"] = [by_presence.number]
        chosen = [one for one in tracklets if one.number in decided["required"]]
        samples = []
        for index, (at, _) in enumerate(frames):
            boxes = [b for b in (one.box_at(index) for one in chosen) if b]
            if not boxes:
                samples.append({"at": at, "present": False})
                continue
            x0, y0, x1, y1 = union_box(boxes)
            samples.append({
                "at": at, "present": True,
                "box": [x0, y0, x1, y1],
                "members": len(boxes), "of": len(chosen),
            })
        excluded = []
        for one in tracklets:
            if decided["verdicts"][one.number]["verdict"] != "other_product":
                continue
            for index, detection in sorted(one.detections.items()):
                excluded.append({
                    "at_seconds": round(frames[index][0], 3),
                    "box_xyxy_1000": [round(v * 1000) for v in detection.box],
                    "reason": decided["verdicts"][one.number]["evidence"],
                    "tracklet": one.number,
                })
        result = {
            "status": (
                "target_located" if decided["required"] else "target_absent"
            ),
            "times": [t for t, _ in frames],
            "samples": samples,
            "tracklets": [
                {
                    "number": one.number,
                    "presence": round(one.presence(len(frames)), 3),
                    "labels": sorted({d.label for d in one.detections.values()}),
                    **decided["verdicts"][one.number],
                }
                for one in tracklets
            ],
            "pick": {
                "required": decided["required"],
                "targets": decided["targets"],
                "unboxed_target": decided["unboxed_target"],
                "note": decided["note"],
                "sheet": str(sheet),
                "sheet_samples": shown,
            },
            "excluded_instances": excluded,
        }
    result["usage"] = (
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        } if usage is not None else None
    )
    if remembered is not None:
        remembered.parent.mkdir(parents=True, exist_ok=True)
        remembered.write_text(json.dumps(
            {"key": key, "result": result}, ensure_ascii=False, indent=1
        ))
    result["_usage_object"] = usage
    return result
