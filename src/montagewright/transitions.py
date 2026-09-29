"""Centered picture transitions that preserve the established audio/caption clock."""
from pathlib import Path


def apply_transitions(plan, segments, picture: Path) -> Path:
    from montagewright.renderer import _run, RenderError
    from montagewright.executor import allocate_timeline_frames
    from montagewright.checkpoints import write_json
    fps = plan.output_fps
    bounds = allocate_timeline_frames([s.screen_duration_seconds for s in plan.segments], fps)
    current = picture
    receipts = []
    for index, incoming in enumerate(plan.segments):
        kind = incoming.transition_in
        if kind == "cut":
            continue
        if index == 0:
            raise RenderError("the opening shot has no preceding transition partner")
        half_frames = max(1, round(incoming.transition_seconds * fps / 2))
        half = half_frames / fps
        duration = 2 * half
        boundary = bounds[index][0] / fps
        at = boundary-half
        if any((b-a)/fps < duration for a, b in bounds[index-1:index+1]):
            raise RenderError("transition overlaps a neighbouring shot boundary")
        destination = picture.with_name(f"transition-{index:03d}.mp4")
        command = ["ffmpeg", "-v", "error", "-y", "-i", str(current)]
        if kind == "dissolve":
            previous = plan.segments[index-1]
            before, handles_a, _ = segments[index-1]
            after, handles_b, _ = segments[index]
            if previous.speed_ratio != 1 or incoming.speed_ratio != 1:
                raise RenderError("dissolve requires recorded-speed source handles")
            if handles_a.tail_seconds < half or handles_b.head_seconds < half:
                raise RenderError("dissolve requires real source handles on both sides")
            if previous.picture_role == "speaker" or incoming.picture_role == "speaker":
                raise RenderError("dissolve cannot overlap a lip-sync speaker commitment")
            a_start = handles_a.head_seconds + (bounds[index-1][1]-bounds[index-1][0])/fps-half
            b_start = handles_b.head_seconds-half
            command += ["-i", str(before.with_name(before.stem+".handles.mp4")),
                        "-i", str(after.with_name(after.stem+".handles.mp4"))]
            graph = (
                f"[1:v]trim=start={a_start}:duration={duration},setpts=PTS-STARTPTS,settb=AVTB[a];"
                f"[2:v]trim=start={b_start}:duration={duration},setpts=PTS-STARTPTS,settb=AVTB[b];"
                f"[a][b]xfade=transition=fade:duration={duration}:offset=0,"
                f"trim=duration={duration},setpts=PTS-STARTPTS+{at}/TB[bridge];"
                f"[0:v][bridge]overlay=eof_action=pass:repeatlast=0:enable='gte(t,{at})*lt(t,{at+duration})'[v]"
            )
        elif kind == "dip_black":
            graph = (
                f"[0:v]split=3[base][a][b];"
                f"[a]trim=start={at}:duration={half},setpts=PTS-STARTPTS,fade=t=out:st=0:d={max(1/fps, half-1/fps)}[out];"
                f"[b]trim=start={boundary}:duration={half},setpts=PTS-STARTPTS,fade=t=in:st=0:d={half}[in];"
                f"[out][in]concat=n=2:v=1:a=0,setpts=N/{fps}/TB+{at}/TB[bridge];"
                f"[base][bridge]overlay=eof_action=pass:repeatlast=0:enable='gte(t,{at})*lt(t,{at+duration})'[v]"
            )
        else:
            raise RenderError(f"unsupported transition {kind}")
        command += ["-filter_complex_threads", "1", "-filter_complex", graph,
                    "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-colorspace", "bt709",
                    "-c:a", "copy", "-r", str(fps),
                    "-frames:v", str(bounds[-1][1]), str(destination)]
        _run(command)
        current = destination
        receipts.append({"incoming_clip": incoming.clip_id, "kind": kind,
                         "start_frame": bounds[index][0]-half_frames,
                         "end_frame": bounds[index][0]+half_frames,
                         "audio": "unchanged", "timeline_frames": bounds[-1][1]})
    write_json(picture.parent / "transitions.json", receipts)
    return current
