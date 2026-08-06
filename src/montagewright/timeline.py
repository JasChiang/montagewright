"""The cut as a timeline, so somebody can disagree with one shot.

A rendered file is the machine's answer and the whole of it. Everything this
pipeline gets wrong is the same shape -- the execution was faithful and the
plan was a poor call, a shot framed on the person who was not talking -- and
no amount of self-review fixes taste. Handing over a timeline moves the tool
from final render to first assembly, which is the thing it is actually good
at: watching five minutes, finding the thirteen sentences, aligning them,
working out the crop. Then a person opens it and changes one shot.

Two things already exist and nothing can use them. Every segment is rendered
with half a second of handle either side, for exactly this, and consumed by
nobody. And every shot carries why it was chosen, why it runs that long and
what degraded -- written into report.json, which is a debugging artifact, not
something an editor reads. Here the handles are just the source being longer
than the clip, and the reasons are markers sitting on the shots they explain.

Two flavours because the two applications disagree. FCPXML is what Final Cut
reads properly; Premiere has always been happier with the older xmeml, which
Resolve also takes.

One honest limit: the crop travels along an eased path evaluated per frame,
and an NLE interpolates between keyframes. The keyframes here land exactly
where they land in the render; the acceleration between them will not match.
Same in-point, same out-point, same framing at each key, a slightly different
feel in the middle -- and the rendered file is there when that matters.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from montagewright.executor import CropBox, RenderPlan, Segment

FPS = 30


def _frames(seconds: float, fps: int = FPS) -> int:
    return max(0, round(seconds * fps))


def _keys(segment: Segment) -> list[tuple[float, CropBox]]:
    """The crop at each moment the path names, seconds from the shot's start."""

    if segment.crop_path is not None:
        return [
            (key.seconds, key.crop) for key in segment.crop_path.keyframes
        ]
    if segment.crop is not None:
        return [(0.0, segment.crop)]
    return []


def _placement(
    crop: CropBox, source_aspect: float, target_aspect: float
) -> tuple[float, float, float]:
    """Scale and offset that put this crop full-frame in the sequence.

    Worked in the sequence's own terms rather than the source's: an NLE fits
    the clip first, so the question is how much bigger than that fit the
    source has to be for the crop to fill the frame, and how far to slide it
    so the crop's centre lands on the frame's.
    """

    scale = 1.0 / max(crop.width, 1e-9)
    # How large the fitted source is, in frame widths, once scaled.
    fitted_height = (target_aspect / source_aspect) * scale
    centre_x = crop.x + crop.width / 2.0
    centre_y = crop.y + crop.height / 2.0
    return (
        scale,
        (0.5 - centre_x) * scale,
        (0.5 - centre_y) * fitted_height,
    )


def _notes(clip_id: str, index: int, report: dict[str, Any]) -> list[str]:
    """Why this shot is here, in the words the layers used at the time."""

    shots = report.get("selection", {}).get("shots", [])
    shot = shots[index] if index < len(shots) else {}
    rhythm = report.get("rhythm", {}).get(clip_id, {})
    verdict = report.get("shots", {}).get(clip_id, {})

    lines = []
    if shot.get("why"):
        lines.append(f"選片：{shot['why']}")
    if rhythm.get("why"):
        lines.append(f"長度：{rhythm['why']}")
    if shot.get("camera_move"):
        lines.append(f"運鏡：{shot['camera_move']}／{shot.get('framing', '')}")
    for step in report.get("degradations", []):
        if step.get("clip_id") == clip_id:
            lines.append(
                f"降級：{step.get('ladder')}"
                f"（{step.get('adjudication', 'unadjudicated')}）"
            )
    if verdict.get("note"):
        mark = "做到" if verdict.get("delivered") else "沒做到"
        lines.append(f"驗收（{mark}）：{verdict['note']}")
    return lines


def to_xmeml(
    plan: RenderPlan,
    report: dict[str, Any],
    *,
    name: str,
    width: int,
    height: int,
    fps: int = FPS,
) -> str:
    """FCP7 XML: what Premiere and Resolve open without complaint."""

    files: dict[str, str] = {}
    items: list[str] = []
    markers: list[str] = []
    cursor = 0

    for index, segment in enumerate(plan.segments):
        source = segment.source
        length = _frames(segment.duration_seconds, fps)
        file_id = f"file-{source.source_id}"
        if file_id not in files:
            files[file_id] = (
                f'<file id="{escape(file_id)}">'
                f"<name>{escape(source.path.name)}</name>"
                f"<pathurl>{escape(source.path.resolve().as_uri())}</pathurl>"
                f'<rate><timebase>{fps}</timebase></rate>'
                f"<duration>{_frames(source.duration_seconds, fps)}</duration>"
                f"<media><video><samplecharacteristics>"
                f"<width>{source.width}</width>"
                f"<height>{source.height}</height>"
                f"</samplecharacteristics></video><audio/></media></file>"
            )
            file_ref = files[file_id]
        else:
            file_ref = f'<file id="{escape(file_id)}"/>'

        motion = ""
        keys = _keys(segment)
        if keys:
            aspect = source.aspect_ratio
            target = width / height
            scale_keys, centre_keys = [], []
            for seconds, crop in keys:
                scale, offset_x, offset_y = _placement(crop, aspect, target)
                at = _frames(seconds, fps)
                scale_keys.append(
                    f"<keyframe><when>{at}</when>"
                    f"<value>{scale * 100:.4f}</value></keyframe>"
                )
                centre_keys.append(
                    f"<keyframe><when>{at}</when><value>"
                    f"<horiz>{offset_x:.6f}</horiz>"
                    f"<vert>{-offset_y:.6f}</vert>"
                    f"</value></keyframe>"
                )
            motion = (
                "<filter><effect><name>Basic Motion</name>"
                "<effectid>basic</effectid>"
                "<effectcategory>motion</effectcategory>"
                "<effecttype>motion</effecttype><mediatype>video</mediatype>"
                "<parameter><parameterid>scale</parameterid>"
                "<name>Scale</name><valuemin>0</valuemin>"
                f"<valuemax>1000</valuemax>{''.join(scale_keys)}</parameter>"
                "<parameter><parameterid>center</parameterid>"
                f"<name>Center</name>{''.join(centre_keys)}</parameter>"
                "</effect></filter>"
            )

        items.append(
            f'<clipitem id="{escape(segment.clip_id)}">'
            f"<name>{escape(source.path.stem)}</name>"
            f"<start>{cursor}</start><end>{cursor + length}</end>"
            f"<in>{_frames(segment.in_seconds, fps)}</in>"
            f"<out>{_frames(segment.out_seconds, fps)}</out>"
            f"{file_ref}{motion}</clipitem>"
        )
        for note in _notes(segment.clip_id, index, report):
            markers.append(
                f"<marker><name>{escape(note[:60])}</name>"
                f"<comment>{escape(note)}</comment>"
                f"<in>{cursor}</in><out>{cursor + length}</out></marker>"
            )
        cursor += length

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE xmeml>\n<xmeml version="5"><sequence>'
        f"<name>{escape(name)}</name><duration>{cursor}</duration>"
        f"<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>"
        f"<media><video><format><samplecharacteristics>"
        f"<width>{width}</width><height>{height}</height>"
        f"</samplecharacteristics></format>"
        f"<track>{''.join(items)}</track></video><audio/></media>"
        f"{''.join(markers)}</sequence></xmeml>\n"
    )


def to_fcpxml(
    plan: RenderPlan,
    report: dict[str, Any],
    *,
    name: str,
    width: int,
    height: int,
    fps: int = FPS,
) -> str:
    """FCPXML: what Final Cut reads properly."""

    def rational(seconds: float) -> str:
        return f"{_frames(seconds, fps) * 1001}/{fps * 1001}s"

    assets: dict[str, str] = {}
    clips: list[str] = []
    cursor = 0.0
    # One format per distinct source geometry, and the sequence's own. An
    # asset with no format leaves Final Cut to guess what shape the media
    # is, which it does by opening the file -- and guesses wrongly when the
    # file is not where the XML says.
    shapes: dict[tuple[int, int], str] = {}

    for index, segment in enumerate(plan.segments):
        source = segment.source
        asset_id = f"r{len(assets) + 2}"
        existing = next(
            (
                key
                for key, value in assets.items()
                if f'name="{html.escape(source.path.stem)}"' in value
            ),
            None,
        )
        if existing is None:
            shape = (int(source.width), int(source.height))
            if shape not in shapes:
                shapes[shape] = f"f{len(shapes) + 1}"
            assets[asset_id] = (
                f'<asset id="{asset_id}" name="{html.escape(source.path.stem)}" '
                f'start="0s" hasVideo="1" hasAudio="1" '
                f'format="{shapes[shape]}" audioSources="1" audioChannels="2" '
                f'duration="{rational(source.duration_seconds)}">'
                f'<media-rep kind="original-media" '
                f'src="{html.escape(source.path.resolve().as_uri())}"/>'
                f"</asset>"
            )
        else:
            asset_id = existing

        adjust = ""
        keys = _keys(segment)
        if keys:
            target = width / height
            placed = [
                (rational(seconds),
                 _placement(crop, source.aspect_ratio, target))
                for seconds, crop in keys
            ]
            if len(placed) == 1:
                # A still frame is two attributes. It was written as <param>
                # elements carrying a time, which is not a thing FCPXML has:
                # Final Cut refused the whole file with "no declaration for
                # attribute time of element param" and imported nothing.
                _, (scale, offset_x, offset_y) = placed[0]
                adjust = (
                    f'<adjust-transform '
                    f'position="{offset_x * width:.3f} {offset_y * height:.3f}" '
                    f'scale="{scale:.5f} {scale:.5f}"/>'
                )
            else:
                # A move is keyframes, and keyframes live inside a
                # keyframeAnimation inside the param they belong to.
                moves = []
                for name, pick in (
                    ("position",
                     lambda p: f"{p[1] * width:.3f} {p[2] * height:.3f}"),
                    ("scale", lambda p: f"{p[0]:.5f} {p[0]:.5f}"),
                ):
                    frames = "".join(
                        f'<keyframe time="{at}" value="{pick(where)}"/>'
                        for at, where in placed
                    )
                    moves.append(
                        f'<param name="{name}">'
                        f"<keyframeAnimation>{frames}</keyframeAnimation>"
                        f"</param>"
                    )
                adjust = (
                    "<adjust-transform>" + "".join(moves) + "</adjust-transform>"
                )

        notes = "".join(
            f'<marker start="{rational(cursor)}" duration="{rational(0.1)}" '
            f'value="{html.escape(note[:180])}"/>'
            for note in _notes(segment.clip_id, index, report)
        )
        clips.append(
            f'<asset-clip name="{html.escape(source.path.stem)}" '
            f'ref="{asset_id}" offset="{rational(cursor)}" '
            f'start="{rational(segment.in_seconds)}" '
            f'duration="{rational(segment.duration_seconds)}">'
            f"{adjust}{notes}</asset-clip>"
        )
        cursor += segment.duration_seconds

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE fcpxml>\n<fcpxml version="1.9"><resources>'
        # No name. FFVideoFormat is the prefix Apple gives its built-in
        # presets -- FFVideoFormat1080p30 and the like -- so a bare
        # "FFVideoFormat" sends Final Cut looking for a preset that does not
        # exist, and it warns that the sequence's format is an unexpected
        # value. A custom size does not claim to be a preset; it states its
        # own dimensions and says what colour it is in.
        f'<format id="r1" width="{width}" height="{height}" '
        f'frameDuration="1001/{fps * 1001}s" '
        f'colorSpace="1-1-1 (Rec. 709)"/>'
        + "".join(
            f'<format id="{ident}" width="{shape[0]}" height="{shape[1]}" '
            f'frameDuration="1001/{fps * 1001}s" '
            f'colorSpace="1-1-1 (Rec. 709)"/>'
            for shape, ident in shapes.items()
        )
        + f"{''.join(assets.values())}</resources>"
        f'<library><event name="{html.escape(name)}">'
        f'<project name="{html.escape(name)}"><sequence format="r1" '
        f'duration="{rational(cursor)}" tcStart="0s">'
        f"<spine>{''.join(clips)}</spine></sequence></project>"
        "</event></library></fcpxml>\n"
    )


def write_timelines(
    plan: RenderPlan,
    report: dict[str, Any],
    output_dir: Path,
    *,
    name: str,
    width: int,
    height: int,
) -> tuple[Path, Path]:
    """Both flavours, beside the render they describe."""

    premiere = output_dir / f"{name}.xml"
    finalcut = output_dir / f"{name}.fcpxml"
    premiere.write_text(
        to_xmeml(plan, report, name=name, width=width, height=height),
        encoding="utf-8",
    )
    finalcut.write_text(
        to_fcpxml(plan, report, name=name, width=width, height=height),
        encoding="utf-8",
    )
    return premiere, finalcut
