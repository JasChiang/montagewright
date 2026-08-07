"""The join between corrected words and measured time.

What is being protected here is one property: a caption's edges are moments
somebody actually spoke, because the recogniser measured them. Text may come
from anywhere -- a model, a person typing -- and the clock stays the
recogniser's. These check that it does.
"""

from montagewright.backfill import Timed, across_lines, align, drift, what_was_heard
from montagewright.transcript import Word


def said(*pairs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, starts_seconds=a, ends_seconds=b) for t, a, b in pairs]


HEARD = said(
    ("在", 0.0, 0.2), ("夏", 0.2, 0.4), ("天", 0.4, 0.6),
    ("吹", 0.6, 0.8), ("頭", 0.8, 1.0), ("發", 1.0, 1.3),
)


def test_text_the_recogniser_got_right_keeps_the_time_it_was_measured_with():
    timed = align("在夏天吹頭發", HEARD)

    assert [one.starts_seconds for one in timed] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert all(one.measured for one in timed)


def test_a_corrected_character_takes_the_span_of_the_one_it_replaced():
    # 發 → 髮 is the correction the recogniser cannot make and the model can.
    timed = align("在夏天吹頭髮", HEARD)

    assert timed[-1].text == "髮"
    assert (timed[-1].starts_seconds, timed[-1].ends_seconds) == (1.0, 1.3)
    # Flagged as worked out rather than measured, even though the span is
    # exactly the replaced character's -- nobody was recorded saying 髮.
    assert not timed[-1].measured
    assert all(one.measured for one in timed[:-1])


def test_punctuation_the_correction_added_takes_no_time():
    timed = align("在夏天，吹頭髮。", HEARD)

    comma = timed[3]
    assert comma.text == "，"
    assert comma.starts_seconds == comma.ends_seconds
    # And it has not pushed the real characters off their measured times.
    assert timed[4].text == "吹" and timed[4].starts_seconds == 0.6


def test_a_line_never_drifts_from_what_was_measured():
    for text in ("在夏天吹頭發", "在夏天吹頭髮", "在夏天，吹頭髮。", "今夏天吹頭髮"):
        assert drift(align(text, HEARD), HEARD) == 0.0


def test_heard_is_read_from_the_recogniser_not_reported_by_the_model():
    # The model, asked to correct errors and quote them unchanged in the same
    # breath, corrects both. The recogniser's own output is on disk.
    assert what_was_heard(HEARD, 0.6, 1.3) == "吹頭發"


def test_lines_are_timed_without_reading_the_models_timestamps():
    starts_and_ends = across_lines(["在夏天", "吹頭髮"], HEARD)

    assert [(a, b) for a, b, _ in starts_and_ends] == [(0.0, 0.6), (0.6, 1.3)]


def test_editing_one_line_leaves_its_neighbours_where_they_were():
    lines = ["在夏天", "吹頭髮"]
    before = across_lines(lines, HEARD)

    after = across_lines(["在夏天", "吹頭髮啦"], HEARD)

    assert after[0][:2] == before[0][:2]


def test_splitting_a_line_puts_the_break_on_a_measured_moment():
    whole = across_lines(["在夏天吹頭髮"], HEARD)
    halves = across_lines(["在夏天", "吹頭髮"], HEARD)

    assert halves[0][0] == whole[0][0]
    assert halves[-1][1] == whole[0][1]
    # The new edge is a word boundary the recogniser measured, not a
    # proportional guess at where half the characters land.
    assert halves[0][1] == 0.6


def test_a_card_that_never_stored_words_gives_back_nothing_rather_than_a_guess():
    assert align("在夏天", []) == []
    assert across_lines(["在夏天", "吹頭髮"], []) == [(0.0, 0.0, []), (0.0, 0.0, [])]


def test_a_line_the_recogniser_missed_entirely_is_marked_not_measured():
    timed = align("完全沒說過的話", said(("嗯", 0.0, 0.1)))

    assert timed and not any(one.measured for one in timed)


def test_an_english_word_spreads_across_its_own_characters():
    timed = align("hello", said(("hello", 0.0, 0.5)))

    assert timed[0].starts_seconds == 0.0
    assert round(timed[-1].ends_seconds, 6) == 0.5
    assert timed[1].starts_seconds > timed[0].starts_seconds


def test_drift_of_nothing_is_nothing():
    assert drift([], HEARD) == 0.0
    assert drift([Timed("在", 0.0, 0.2)], []) == 0.0


def test_an_edit_saved_from_the_browser_comes_back_on_the_measured_clock(
    tmp_path,
) -> None:
    """The round trip, not the mechanism.

    The browser sends the times a line had before it was edited. If the
    server keeps them, a line whose length changed sits at a moment nobody
    said it. This checks the edited text is re-timed and that the new times
    are sent back, because the browser cannot draw what it is not told.
    """

    import json

    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was, held = web.RUNS_ROOT, web._transcript_map
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        here = web.RUNS_ROOT / "r1"
        (here / "out" / "work").mkdir(parents=True)
        (here / "run.json").write_text(
            json.dumps({"state": "done", "started_at": 0.0}), encoding="utf-8"
        )

        # A run whose words were measured, and whose shot is the whole of it.
        (here / "out" / "report.json").write_text(json.dumps({
            "selection": {"shots": [{"source_id": "a", "start_seconds": 0.0}]},
            "rhythm": {"k00": {"seconds": 1.3}},
            "direction": {"aspect": "9:16"},
        }), encoding="utf-8")

        def cards(run):
            return {"a": {
                "lines": [{
                    "text": "在夏天吹頭發",
                    "starts_seconds": 0.0, "ends_seconds": 1.3,
                }],
                "words": [
                    {
                        "text": one.text,
                        "starts_seconds": one.starts_seconds,
                        "ends_seconds": one.ends_seconds,
                    }
                    for one in HEARD
                ],
                "silences": [],
            }}

        web._transcript_map = cards
        client = TestClient(web.create_app())

        # Somebody splits the line in two, keeping the old times on both.
        saved = client.put("/api/runs/r1/subtitle-track", json={"lines": [
            {"at": 0.0, "until": 1.3, "text": "在夏天"},
            {"at": 0.0, "until": 1.3, "text": "吹頭髮"},
        ]})

        assert saved.status_code == 200
        back = saved.json()["timed"]
        assert [(one["at"], one["until"]) for one in back] == [
            (0.0, 0.6), (0.6, 1.3)
        ]
    finally:
        web.RUNS_ROOT = was
        web._transcript_map = held
        web.RUNS.pop("r1", None)


# --- Gemini reads video in MM:SS; the card asks for seconds ---------------

def _card(**over):
    base = {
        "usable_from_seconds": 0.0, "usable_to_seconds": 10.0,
        "action": [], "subjects": [],
    }
    base.update(over)
    return base


def test_a_colon_that_became_a_decimal_point_is_read_back():
    """1:53 arrived as 1.53 on a clip lasting 113.4 seconds.

    The dangerous case: 1.53 is inside the clip, so it passes every range
    check while claiming a two-minute take is usable for a second and a half.
    """

    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(_card(usable_to_seconds=1.53), 113.4)

    assert got["usable_to_seconds"] == 113.0


def test_a_colon_that_vanished_is_read_back():
    # 1:10 arrived as 110 on a clip lasting 71.1 seconds.
    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(_card(usable_to_seconds=110.0), 71.1)

    assert got["usable_to_seconds"] == 70.0


def test_both_ends_of_an_action_are_read_the_same_way():
    # 1.1 and 1.13 are each readable as plain seconds, and read that way they
    # describe a 30ms action. Read as MM:SS they are 1:10 to 1:13.
    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(action=[{"what": "x", "starts_seconds": 1.1, "ends_seconds": 1.13}]),
        113.4,
    )

    assert [(a["starts_seconds"], a["ends_seconds"]) for a in got["action"]] == [
        (70.0, 73.0)
    ]


def test_a_clip_that_never_had_the_problem_is_left_alone():
    from montagewright.clipcard import times_on_receipt

    was = _card(
        usable_to_seconds=27.2,
        action=[{"what": "x", "starts_seconds": 2.5, "ends_seconds": 4.5}],
        subjects=[{"label": "x", "at_seconds": 2.0}],
    )

    got = times_on_receipt(dict(was), 27.2)

    assert got["usable_from_seconds"] == 0.0 and got["usable_to_seconds"] == 27.2
    assert got["action"] == was["action"]


def test_a_genuinely_short_window_is_the_models_to_report():
    # Five usable seconds out of a hundred is a strong claim, but it is a
    # claim -- and 5.0 has no MM:SS reading that lands inside the clip, so
    # there is nothing to prefer over it.
    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(_card(usable_to_seconds=5.0), 100.0)

    assert got["usable_to_seconds"] == 5.0


def test_an_action_that_cannot_be_read_into_the_clip_is_dropped():
    # Not clamped. A missing action is a static shot, which is a fine thing
    # to be; an action at a wrong second puts a cut in the wrong place.
    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(action=[{"what": "x", "starts_seconds": 400.0, "ends_seconds": 480.0}]),
        30.0,
    )

    assert got["action"] == []


def test_a_clip_whose_length_is_unknown_is_not_second_guessed():
    from montagewright.clipcard import times_on_receipt

    was = _card(usable_to_seconds=1.53)

    assert times_on_receipt(dict(was), 0.0) == was


def test_a_timestamp_rounded_up_past_the_end_is_kept_not_deleted():
    """Gemini samples at one frame a second, so it answers in whole seconds.

    On a clip lasting 12.012s the last frame it holds is at 12, and "ends at
    13" is that rounding rather than a mistake. Half a second of tolerance,
    which is what this had first, deleted the action instead.
    """

    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(action=[{"what": "x", "starts_seconds": 10.0, "ends_seconds": 13.0}]),
        12.012,
    )

    assert [(a["starts_seconds"], a["ends_seconds"]) for a in got["action"]] == [
        (10.0, 12.012)
    ]


def test_the_slop_is_nowhere_near_a_notation_error():
    # The smallest possible MM:SS collision is 1:01 as 101 on a clip just
    # past a minute, overshooting by forty seconds. Rounding is one second.
    from montagewright.clipcard import SLOP, times_on_receipt

    assert SLOP < 40

    got = times_on_receipt(_card(usable_to_seconds=101.0), 61.0)

    assert got["usable_to_seconds"] == 61.0


# --- the proxy is a smaller copy, never a larger one ----------------------

def _clip(tmp_path, width, height, name="in.mp4"):
    import subprocess

    made = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size={width}x{height}:rate=30:duration=1",
         "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p", str(made)],
        check=True,
    )
    return made


def _size(path):
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return tuple(int(one) for one in out.split(","))


def test_a_clip_narrower_than_the_proxy_width_is_left_alone(tmp_path):
    """`scale=640` is a demand, not a limit.

    Handed a 320x240 clip it produced a 640x480 one -- bigger than the file
    it came from, blurrier than the picture it describes, and no more use to
    a model that caps the frame at 70 tokens regardless.
    """

    from montagewright.cli import _encode_proxy

    small = _clip(tmp_path, 320, 240)
    made = tmp_path / "out.mp4"
    _encode_proxy(small, made)

    assert _size(made) == (320, 240)


def test_a_clip_wider_than_the_proxy_width_is_shrunk_to_it(tmp_path):
    from montagewright.cli import _encode_proxy

    big = _clip(tmp_path, 1920, 1080)
    made = tmp_path / "out.mp4"
    _encode_proxy(big, made)

    assert _size(made) == (640, 360)


def test_the_proxy_keeps_the_length_it_was_made_from(tmp_path):
    """Forcing a frame rate made the duration land on a multiple of it.

    Cards describe the proxy and the edit cuts the original, so the two ran
    a tenth of a second apart until the rate was left alone.
    """

    from montagewright.cli import _encode_proxy
    from montagewright.clipcard import clip_seconds

    source = _clip(tmp_path, 1920, 1080)
    made = tmp_path / "out.mp4"
    _encode_proxy(source, made)

    assert abs(clip_seconds(made) - clip_seconds(source)) < 0.005


def test_the_resolution_key_is_the_one_the_api_reads():
    """The knob has to be connected to something.

    Every video part carried `media_resolution`, which the Interactions API
    does not define -- its field is `resolution`. The SDK dropped it on the
    floor and forwarded the unknown key, so five call sites looked like they
    were controlling frame detail and were not. It went unnoticed because
    `low` and the default are both 70 tokens a frame for video, so the
    setting that never applied would not have changed anything if it had.

    Checked against the SDK rather than a string, so the day the field is
    renamed this fails here instead of silently in a paid call.
    """

    import re
    from pathlib import Path

    from google.genai._gaos.types.interactions.videocontent import VideoContent

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    sending = [
        path for path in root.rglob("*.py")
        if re.search(r'"type":\s*"video"', path.read_text(encoding="utf-8"))
    ]
    assert sending, "no video parts found; this test has lost its subject"

    for path in sending:
        text = path.read_text(encoding="utf-8")
        assert '"media_resolution"' not in text, (
            f"{path.name} sets media_resolution, which the Interactions API "
            f"ignores; the field is `resolution`"
        )

    # And the key that is used actually lands on the model's own field.
    part = VideoContent(
        type="video", mime_type="video/mp4", uri="files/x", resolution="low"
    )
    assert part.resolution == "low"


# --- thinking is spent from the answer's budget --------------------------

def test_a_pass_that_spent_its_budget_thinking_says_so():
    """An exhausted budget produces no text at all.

    Not truncated JSON -- nothing. The old message was "returned no text",
    which reads as the model declining to answer rather than as a ceiling
    to raise. The API marks the run `incomplete`; that is worth reading
    instead of guessing from the shape of the output.
    """

    import pytest

    from montagewright.planner import PlannerError, _parse

    class Ran:
        status = "incomplete"
        output_text = ""
        usage = {"total_thought_tokens": 45, "total_output_tokens": 0}

    with pytest.raises(PlannerError) as raised:
        _parse(Ran(), what="selection")

    assert "output budget" in str(raised.value)
    assert "45" in str(raised.value)


def test_a_finished_pass_is_parsed_normally():
    from montagewright.planner import _parse

    class Ran:
        status = "completed"
        output_text = '{"shots": []}'
        usage = {}

    assert _parse(Ran(), what="selection") == {"shots": []}


def test_the_ceiling_is_the_models_own():
    # 65536 for gemini-3.6-flash. Half of it was still a ration, and the
    # billing is on what is produced rather than on what is allowed.
    from montagewright.planner import MAX_OUTPUT_TOKENS

    assert MAX_OUTPUT_TOKENS == 65536


# --- a description belongs beside its own footage ------------------------

def _material(tmp_path, ids, missing=()):
    from montagewright.planner import MaterialItem

    out = []
    for one in ids:
        proxy = tmp_path / f"{one}.mp4"
        if one not in missing:
            proxy.write_bytes(b"not really a video")
        out.append(
            MaterialItem(
                source_id=one, duration_seconds=10.0, summary=f"{one} 的內容",
                proxy=proxy, composition="horizontal",
            )
        )
    return out


class _Cache:
    def uri_for(self, path, client, *, mime_type):
        return f"files/{path.stem}", None


def test_each_clip_is_described_next_to_its_own_video(tmp_path):
    from montagewright.planner import _attach_material

    parts = _attach_material(_material(tmp_path, ["a", "b", "c"]), _Cache(), None)

    # text, video, text, video, text, video -- and each text names the clip
    # whose uri follows it.
    assert [one["type"] for one in parts] == ["text", "video"] * 3
    for said, shown in zip(parts[::2], parts[1::2]):
        assert shown["uri"].split("/")[-1] in said["text"]


def test_a_missing_proxy_takes_its_description_with_it(tmp_path):
    """The failure this shape exists to prevent.

    With the listing in the prompt and the videos after it, a clip that
    failed to encode was skipped among the videos while its line stayed in
    the listing -- so every clip after it was described against the wrong
    picture, and nothing raised.
    """

    from montagewright.planner import _attach_material

    parts = _attach_material(
        _material(tmp_path, ["a", "b", "c"], missing={"b"}), _Cache(), None
    )

    assert [one["type"] for one in parts] == ["text", "video"] * 2
    assert "b" not in "".join(
        one["text"] for one in parts if one["type"] == "text"
    ).replace("的內容", "")
    for said, shown in zip(parts[::2], parts[1::2]):
        assert shown["uri"].split("/")[-1] in said["text"]


def test_the_prompt_no_longer_carries_a_second_copy_of_the_listing():
    # Described twice is worse than described once in the wrong place: the
    # two copies can disagree, and only one of them sits by the footage.
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "planner.py"
    ).read_text(encoding="utf-8")

    assert "依序附上影片" not in text


# --- a move needs somewhere to happen ------------------------------------

def test_travel_room_comes_from_resolution_not_only_from_shape():
    """The first version of this asked only about aspect and got it wrong.

    It assumed the crop is always the largest that fits, so a 4K take
    delivering 9:16 was reported as having no vertical room at all. The crop
    only has to be as tall as the delivery -- 1920 of the 2160 it has -- and
    the 240 left over are room to tilt.
    """

    from montagewright.reframe import travel_room

    def room(w, h, ow, oh):
        return travel_room(
            source_width=w, source_height=h, target_aspect=ow / oh,
            output_width=ow, output_height=oh,
        )

    across, up = room(3840, 2160, 1080, 1920)
    assert round(across, 3) == 0.719
    assert round(up, 3) == 0.111

    # A source with nothing spare really does have nowhere to go vertically.
    across, up = room(1920, 1080, 1080, 1920)
    assert round(across, 3) == 0.684
    assert up == 0.0

    # And delivering the aspect the source already is only has no room when
    # the resolution matches too -- 4K to FHD can put the frame anywhere.
    assert room(3840, 2160, 1920, 1080) == (0.5, 0.5)
    assert room(1920, 1080, 1920, 1080) == (0.0, 0.0)


def test_the_planner_is_told_which_moves_have_nowhere_to_go():
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            push_room=1.5, pan_room=0.684, tilt_room=0.0,
        )
    ])

    assert "橫向可移 68%" in said
    assert "tilt 做不到" in said


def test_a_source_already_at_the_delivery_aspect_says_neither_move_works():
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            push_room=1.0, pan_room=0.0, tilt_room=0.0,
        )
    ])

    assert "pan 跟 tilt 都做不到" in said


# --- a push is the one move whose frame shrinks --------------------------

def _zoom(**over):
    from montagewright.reframe import build_zoom_path, zoom_budget

    budget = zoom_budget(
        source_width=3840, source_height=2160, source_aspect=16 / 9,
        target_aspect=1080 / 1920, output_width=1080, output_height=1920,
    )
    args = dict(
        source_aspect=16 / 9, target_aspect=1080 / 1920, duration_seconds=3.0,
        direction="push_in", centre_x=0.5, centre_y=0.5, energy="active",
        framing="fill", budget=budget, subject_height=0.35, clip_id="k00",
    )
    args.update(over)
    return build_zoom_path(**args)


def _where(crop, subject_x):
    """Where the subject sits inside the crop: 0 is the left edge, 1 the right."""

    return (subject_x - crop.x) / crop.width


def test_a_push_follows_a_subject_that_moves_while_the_frame_closes():
    """Aiming at the mean of five samples loses a walking subject.

    Measured before this: a subject crossing from 0.35 to 0.65 starts hard
    against the left edge and finishes at 1.22 -- outside the frame -- and
    nothing recorded that it had happened.
    """

    walk = [(3.0 * i / 4, 0.35 + 0.30 * i / 4, 0.5) for i in range(5)]
    degradations = []

    path = _zoom(track=walk, degradations=degradations)

    assert round(_where(path.keyframes[0].crop, 0.35), 2) == 0.5
    assert round(_where(path.keyframes[-1].crop, 0.65), 2) == 0.5
    assert any(
        one.ladder_other == "zoom_followed_subject" for one in degradations
    )


def test_a_push_given_no_track_aims_where_it_was_told_to():
    """Passing no track has to change nothing about a static push.

    This asserted "two keyframes" when it was written, which was an
    implementation detail rather than the property -- a later change gave
    every designed move a rest at each end, and the test failed for a
    reason that had nothing to do with what it was guarding.
    """

    without = _zoom()
    explicit_none = _zoom(track=None)

    assert [
        (one.seconds, one.crop.x, one.crop.width) for one in without.keyframes
    ] == [
        (one.seconds, one.crop.x, one.crop.width)
        for one in explicit_none.keyframes
    ]
    # Aimed at the centre it was given, not at the middle of the frame.
    assert without.keyframes[0].crop.x == _zoom(centre_x=0.5).keyframes[0].crop.x


def test_a_subject_that_only_jitters_is_not_chased():
    # A track that wanders by less than the deadband would make the push
    # wobble, which reads worse than aiming at one point.
    still = [(3.0 * i / 4, 0.5 + 0.001 * i, 0.5) for i in range(5)]

    assert len(_zoom(track=still).keyframes) == 2


# --- what a crop cannot measure ------------------------------------------

def test_the_card_asks_for_shot_size_and_facing():
    """Neither is derivable from geometry, and both decide what can follow
    what: two neighbouring shots at the same size read as a jump, and two
    facing the same way read as both people addressing the same side."""

    from montagewright.clipcard import card_schema

    schema = card_schema()

    assert "shot_size" in schema["required"]
    assert "facing" in schema["required"]
    assert schema["properties"]["shot_size"]["enum"] == [
        "wide", "medium", "close", "extreme_close"
    ]
    assert schema["properties"]["facing"]["enum"] == [
        "left", "right", "toward", "away", "flat"
    ]


def test_the_planner_is_shown_size_and_facing():
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            shot_size="close", facing="right",
        )
    ])

    assert "景別close" in said and "朝向right" in said


def test_a_shot_with_no_direction_says_nothing_about_direction():
    # `flat` is "no direction to preserve", which is not a fact worth a line
    # in a listing the planner has to read seventy-four of.
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            shot_size="wide", facing="flat",
        )
    ])

    assert "朝向" not in said


def test_selection_is_told_what_to_do_with_them():
    # A field nobody is told to use is a field nobody fills honestly.
    from pathlib import Path

    prompt = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "prompts" / "selection_zh-TW.txt"
    ).read_text(encoding="utf-8")

    assert "景別太接近會跳" in prompt
    assert "銀幕方向" in prompt or "朝向決定" in prompt


def test_the_overlay_reports_the_move_that_happened():
    """The crop overlay exists to check whether the move happened.

    It printed `camera_move`, which is what was asked for -- so a pan that
    degraded to a hold drew a motionless box captioned 橫搖, in the one view
    whose whole job is catching exactly that.
    """

    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "web" / "index.html"
    ).read_text(encoding="utf-8")

    assert "function cropDid(" in page
    # The label is built from what the keyframes did, and only mentions the
    # plan when the two disagree.
    assert "const did = cropDid(keys);" in page
    assert "if (planned !== did)" in page


def test_selection_is_told_a_row_needs_two_endpoints():
    """A pan with one static subject is the one combination that cannot work.

    The prompt advised `pan` for a subject too wide to frame without saying
    it needs both ends named, so a row of watches came back as one subject,
    could not be followed, and rendered as a hold.
    """

    from pathlib import Path

    prompt = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "prompts" / "selection_zh-TW.txt"
    ).read_text(encoding="utf-8")

    assert "then_subject" in prompt
    assert "就要給兩個端點" in prompt


# --- a move has to arrive somewhere and stay there -----------------------

def _handoff(seconds, energy="calm", degradations=None):
    from montagewright.reframe import build_handoff_path

    return build_handoff_path(
        source_aspect=16 / 9, target_aspect=1080 / 1920,
        duration_seconds=seconds, from_centre=0.15, to_centre=0.85,
        from_width=0.10, to_width=0.10, energy=energy,
        clip_id="k00", degradations=degradations,
    )


def _centre(keyframe):
    return keyframe.crop.x + keyframe.crop.width / 2


def test_a_designed_move_rests_at_both_ends():
    """Two keyframes means moving in every frame of the shot.

    That is a pan with its first and last seconds cut off, and it reads as
    one: the eye never gets a still frame to recognise where it started or
    where it ended up.
    """

    path = _handoff(5.0, energy="active")

    assert len(path.keyframes) == 4
    # Still at the start, still at the end, travelling in between.
    assert _centre(path.keyframes[0]) == _centre(path.keyframes[1])
    assert _centre(path.keyframes[2]) == _centre(path.keyframes[3])
    assert path.keyframes[1].seconds > 0.0
    assert path.keyframes[2].seconds < 5.0


def test_a_move_that_cannot_cross_in_the_time_says_so():
    """It used to stop partway and report nothing.

    A 1.2s calm pan across 0.700 of frame arrived 43% of the way, and the
    destination -- usually the point of the shot -- never appeared.
    """

    degradations = []
    _handoff(1.2, energy="calm", degradations=degradations)

    assert [one.ladder_other for one in degradations] == [
        "move_does_not_fit_the_time"
    ]
    measured = degradations[0].measured
    assert measured["needed_speed_vw_s"] > measured["max_speed_vw_s"]


def test_a_move_with_room_to_spare_reports_nothing():
    degradations = []
    _handoff(5.0, energy="active", degradations=degradations)

    assert degradations == []


def test_a_shot_too_short_to_rest_in_still_moves():
    # Settling is capped at a share of the shot, so a brief take is not all
    # settling and no move.
    path = _handoff(0.6, energy="active")

    assert _centre(path.keyframes[-1]) > _centre(path.keyframes[0])


def test_a_push_rests_too():
    # Same argument: a push that starts on the first frame and ends on the
    # last reads as cut out of a longer one.
    from montagewright.reframe import build_zoom_path, zoom_budget

    budget = zoom_budget(
        source_width=3840, source_height=2160, source_aspect=16 / 9,
        target_aspect=1080 / 1920, output_width=1080, output_height=1920,
    )
    path = build_zoom_path(
        source_aspect=16 / 9, target_aspect=1080 / 1920, duration_seconds=4.0,
        direction="push_in", centre_x=0.5, centre_y=0.5, energy="active",
        framing="fill", budget=budget, subject_height=0.35,
    )

    assert len(path.keyframes) == 4
    assert path.keyframes[0].crop.width == path.keyframes[1].crop.width
    assert path.keyframes[2].crop.width == path.keyframes[3].crop.width


# --- one builder for every shape a list of looks can be -----------------

WIDE, TIGHT = 0.3164, 0.20


def _looks(stops, seconds=4.0, degradations=None):
    from montagewright.reframe import build_look_path

    return build_look_path(
        stops, source_aspect=16 / 9, target_aspect=1080 / 1920,
        duration_seconds=seconds, energy="active", clip_id="k00",
        degradations=degradations,
    )


def _mids(path):
    return [round(k.crop.x + k.crop.width / 2, 2) for k in path.keyframes]


def test_one_look_is_a_hold():
    path = _looks([(0.0, 0.5, 0.5, WIDE)])

    assert path.is_static


def test_two_looks_are_a_move_that_rests_at_both_ends():
    path = _looks([(0.0, 0.2, 0.5, WIDE), (0.0, 0.8, 0.5, WIDE)])

    assert _mids(path) == [0.2, 0.2, 0.8, 0.8]


def test_two_looks_at_one_thing_are_a_push():
    path = _looks([(0.0, 0.5, 0.5, WIDE), (0.0, 0.5, 0.5, TIGHT)])

    widths = [round(k.crop.width, 2) for k in path.keyframes]
    assert widths == [0.32, 0.32, 0.20, 0.20]
    assert _mids(path) == [0.5, 0.5, 0.5, 0.5]


def test_three_looks_stop_on_the_way():
    """The shape the old menu could not express at all.

    `then_subject` named exactly two endpoints, so a row of three watches
    had no way to be introduced one at a time.
    """

    path = _looks(
        [(0.6, 0.15, 0.5, WIDE), (0.6, 0.5, 0.5, WIDE), (0.8, 0.85, 0.5, WIDE)],
        seconds=5.0,
    )

    assert len(path.keyframes) == 6
    assert _mids(path) == [0.16, 0.16, 0.5, 0.5, 0.84, 0.84]


def test_a_move_and_a_push_at_once():
    # Needed a builder of its own before; now it is just two looks that
    # disagree about both position and size.
    path = _looks([(0.0, 0.2, 0.5, WIDE), (0.0, 0.8, 0.5, TIGHT)])

    assert _mids(path)[0] != _mids(path)[-1]
    assert path.keyframes[0].crop.width > path.keyframes[-1].crop.width


def test_too_many_looks_for_the_time_is_reported():
    degradations = []
    _looks(
        [(0.6, 0.15, 0.5, WIDE), (0.6, 0.5, 0.5, WIDE), (0.8, 0.85, 0.5, WIDE)],
        seconds=2.0, degradations=degradations,
    )

    assert [one.ladder_other for one in degradations] == [
        "looks_do_not_fit_the_time"
    ]
    measured = degradations[0].measured
    assert measured["needed_speed_vw_s"] > measured["max_speed_vw_s"]


def test_resting_never_eats_the_whole_shot():
    # Three looks asking for a second each, in a shot lasting two.
    path = _looks(
        [(1.0, 0.15, 0.5, WIDE), (1.0, 0.5, 0.5, WIDE), (1.0, 0.85, 0.5, WIDE)],
        seconds=2.0,
    )

    assert _mids(path)[0] < _mids(path)[-1]
    assert path.keyframes[-1].seconds <= 2.0


def test_the_move_is_read_off_the_looks_not_taken_from_the_plan():
    """Removing camera_move from the schema made every shot a hold.

    reframe_of still read the field, the new schema no longer sends it, and
    the default was "hold" -- so the whole cut would have rendered
    motionless with nothing raising. The name is now observed rather than
    chosen, which is also why it cannot disagree with the looks.
    """

    from montagewright.schema import reframe_of

    def move(looks):
        return reframe_of({"looks": looks, "why": "x"}).camera_move

    assert move([{"at": "the coin"}]) == "hold"
    assert move([{"at": "the left one"}, {"at": "the right one"}]) == "pan"
    assert move(
        [{"at": "the coin", "framing": "thirds"},
         {"at": "the coin", "framing": "fill"}]
    ) == "push_in"
    assert move(
        [{"at": "the table", "framing": "fill"},
         {"at": "the table", "framing": "thirds"}]
    ) == "pull_out"
    assert move([{"at": "a"}, {"at": "b"}, {"at": "c"}]) == "pan"


def test_a_plan_written_before_looks_still_says_what_it_meant():
    from montagewright.schema import reframe_of

    was = reframe_of({
        "subject": "the left handset", "camera_move": "pan",
        "then_subject": "the right handset", "why": "x",
    })

    assert was.camera_move == "pan"
    assert [one.at for one in was.looks] == [
        "the left handset", "the right handset"
    ]


def _measured(monkeypatch, looks, places):
    """Run _measure_looks against fake grounding, counting the calls."""

    from montagewright import pipeline
    from montagewright.executor import Source

    asked = []

    def located(frames, description, *, client):
        asked.append(description)

        class Used:
            input_tokens = output_tokens = thought_tokens = 0

        return [{
            "present": True, "centre_x": places[description], "centre_y": 0.5,
            "width": 0.1, "height": 0.3, "frame_index": 0,
        }], Used()

    monkeypatch.setattr(pipeline, "locate_subject", located)
    monkeypatch.setattr(pipeline, "_sample_frames", lambda *a, **k: ([], []))
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    class Clip:
        clip_id = "k00"
        approx_in_seconds, approx_out_seconds = 0.0, 6.0

    stops, missing = pipeline._measure_looks(
        looks,
        Source(source_id="s", path=None, duration_seconds=6.0,
               width=3840, height=2160),
        Clip(), None, pipeline.Report(), object(), 1080 / 1920,
    )
    return stops, missing, asked


def test_three_looks_reach_the_renderer_as_three_stops(monkeypatch):
    """They used to be truncated to two with nothing recorded.

    `pan` read `subject` and `then_subject`; a third look had nowhere to go,
    so a row of three watches lost its middle stop silently.
    """

    from montagewright.schema import Look

    stops, missing, _ = _measured(
        monkeypatch,
        [Look(at="A", seconds=0.6), Look(at="B", seconds=0.6),
         Look(at="C", seconds=0.8)],
        {"A": 0.15, "B": 0.5, "C": 0.85},
    )

    assert missing == ""
    assert [round(one[1], 2) for one in stops] == [0.15, 0.5, 0.85]
    assert [one[0] for one in stops] == [0.6, 0.6, 0.8]


def test_one_subject_looked_at_twice_is_measured_once(monkeypatch):
    # Two looks at one thing is how a push is written, and grounding the
    # same description twice would buy one answer twice.
    from montagewright.schema import Look

    stops, _, asked = _measured(
        monkeypatch,
        [Look(at="A", framing="thirds"), Look(at="A", framing="fill")],
        {"A": 0.4},
    )

    assert asked == ["A"]
    assert len(stops) == 2
    # Same place, tighter crop -- which is what a push in is.
    assert stops[0][1] == stops[1][1]
    assert stops[1][3] < stops[0][3]


def test_a_subject_nobody_can_find_is_named_rather_than_guessed(monkeypatch):
    from montagewright import pipeline
    from montagewright.executor import Source
    from montagewright.schema import Look

    def nothing(frames, description, *, client):
        class Used:
            input_tokens = output_tokens = thought_tokens = 0

        return [{"present": False}], Used()

    monkeypatch.setattr(pipeline, "locate_subject", nothing)
    monkeypatch.setattr(pipeline, "_sample_frames", lambda *a, **k: ([], []))
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    class Clip:
        clip_id = "k00"
        approx_in_seconds, approx_out_seconds = 0.0, 6.0

    stops, missing = pipeline._measure_looks(
        [Look(at="the ghost")],
        Source(source_id="s", path=None, duration_seconds=6.0,
               width=3840, height=2160),
        Clip(), None, pipeline.Report(), object(), 1080 / 1920,
    )

    assert stops == [] and missing == "the ghost"


def test_only_one_place_knows_what_shape_a_shot_is_written_in():
    """Six readers survived the field being removed from the schema.

    Two raised -- one of them after direction and selection had been paid
    for -- and four returned an empty string or "hold" and carried on. The
    lesson had been written an hour earlier and not applied: when a field
    goes, search for who still reads it, not who still writes it.

    So there is one reader, and this is what keeps it that way.
    """

    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    reading = re.compile(
        r'\[["\'](?:subject|camera_move|then_subject|must_be_whole)["\']\]'
        r'|\.get\(\s*["\'](?:subject|camera_move|then_subject|must_be_whole)["\']'
    )
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "schema.py":
            continue  # the one place allowed to know both shapes
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if reading.search(line):
                offenders.append(f"{path.name}:{number}  {line.strip()}")

    assert not offenders, (
        "these read a selection shot's old fields directly instead of going "
        "through schema.looks_of / subject_of / move_of_shot:\n  "
        + "\n  ".join(offenders)
    )


def test_a_three_look_plan_actually_reaches_the_look_builder(monkeypatch):
    """The wiring, not the parts.

    build_look_path had seven tests and _measure_looks three, and every one
    passed while the branch that calls them was unreachable: the
    two-subject pan sat above it, and `then_subject` is set exactly when
    there is a second look, so it caught every pan first. Three watches went
    on losing their middle stop.

    Testing the pieces is not testing that anything calls them.
    """

    from montagewright import pipeline
    from montagewright.executor import Source
    from montagewright.schema import Clip, EDL, reframe_of

    called = {}

    def look_path(stops, **kw):
        from montagewright.reframe import CropBox, CropPath, Keyframe

        called["stops"] = stops
        return CropPath([Keyframe(0.0, CropBox(0.0, 0.0, 0.3, 1.0))])

    def located(frames, description, *, client):
        class Used:
            input_tokens = output_tokens = thought_tokens = 0

        return [{
            "present": True, "centre_x": {"A": 0.2, "B": 0.5, "C": 0.8}[description],
            "centre_y": 0.5, "width": 0.1, "height": 0.3, "frame_index": 0,
        }], Used()

    monkeypatch.setattr(pipeline, "build_look_path", look_path)
    monkeypatch.setattr(pipeline, "locate_subject", located)
    monkeypatch.setattr(pipeline, "_sample_frames", lambda *a, **k: ([], []))
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    # Built the way production builds it. Constructing a Reframe by hand
    # gave one with no `then_subject`, and `then_subject` is precisely what
    # the shadowing branch tested for -- so the test passed with the bug
    # reintroduced. A fake that does not have the shape of the real thing
    # cannot catch a bug about that shape.
    clip = Clip(
        clip_id="k00", source_id="s", approx_in_seconds=0.0,
        approx_out_seconds=5.0,
        reframe=reframe_of({
            "looks": [{"at": "A"}, {"at": "B"}, {"at": "C"}], "why": "x",
        }),
    )
    pipeline.follow_subjects(
        EDL(project_id="t", clips=[clip]),
        {"s": Source(source_id="s", path=None, duration_seconds=5.0,
                     width=3840, height=2160)},
        target_aspect=1080 / 1920,
        report=pipeline.Report(),
        client=object(),
    )

    assert "stops" in called, (
        "a three-look plan never reached build_look_path -- something above "
        "the looks branch is catching it first"
    )
    assert len(called["stops"]) == 3
