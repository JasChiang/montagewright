"""Every camera move has to reach its own builder.

A follow branch once lost its grounding call entirely when two branches were
inserted above it, and the whole suite stayed green: nothing exercised
camera_move at all, and the planner happened not to choose follow for several
runs. The gap was found by watching a render.
"""

from __future__ import annotations

import pytest

from montagewright.capabilities import INTENT_NAMES, MOVE_NAMES
from montagewright.reframe import (
    MAX_UPSCALE,
    Observation,
    achieved_upscale,
    build_crop_path,
    build_handoff_path,
    build_sweep_path,
    build_zoom_path,
    visible_fraction,
    zoom_budget,
)
from montagewright.executor import CropBox

WIDE, TALL = 16 / 9, 9 / 16


def _moving(count: int = 5) -> list[Observation]:
    return [
        Observation(
            seconds=index * 0.5,
            centre_x=0.20 + 0.08 * index,
            centre_y=0.5,
            width=0.18,
            height=0.4,
        )
        for index in range(count)
    ]


def _still(count: int = 5) -> list[Observation]:
    return [
        Observation(
            seconds=index * 0.5,
            centre_x=0.5,
            centre_y=0.5,
            width=0.18,
            height=0.4,
        )
        for index in range(count)
    ]


def test_every_declared_move_has_a_builder() -> None:
    """The menu the planner reads and the code that dispatches cannot drift."""

    from montagewright import pipeline

    dispatch = pipeline.follow_subjects.__doc__ or ""
    source = pytest.importorskip("inspect").getsource(pipeline.follow_subjects)
    for name in MOVE_NAMES:
        assert name in source, f"{name} is offered but never dispatched"
    assert dispatch  # the helper is documented


class TestFollow:
    def test_a_moving_subject_is_followed(self) -> None:
        path = build_crop_path(
            _moving(), source_aspect=WIDE, target_aspect=TALL, energy="active"
        )
        assert not path.is_static
        assert path.travel() > 0.0

    def test_a_still_subject_holds_and_says_so(self) -> None:
        degradations: list = []
        path = build_crop_path(
            _still(),
            source_aspect=WIDE,
            target_aspect=TALL,
            energy="active",
            clip_id="k00",
            degradations=degradations,
        )
        assert path.is_static
        assert any(step.ladder == "static_on_subject" for step in degradations)

    def test_a_subject_that_returns_is_not_chased(self) -> None:
        """Out and back inside one shot reads as a wobble, not a move."""

        there_and_back = [
            Observation(seconds=index * 0.4, centre_x=x, centre_y=0.5, width=0.18, height=0.4)
            for index, x in enumerate([0.30, 0.25, 0.20, 0.21, 0.26, 0.30])
        ]
        degradations: list = []
        path = build_crop_path(
            there_and_back,
            source_aspect=WIDE,
            target_aspect=TALL,
            energy="active",
            clip_id="k01",
            degradations=degradations,
        )
        assert path.is_static
        assert degradations, "a substitution has to be recorded, not silent"

    def test_a_sweep_that_hesitates_still_follows(self) -> None:
        """One pause must not be mistaken for indecision."""

        hesitating = [
            Observation(seconds=index * 0.4, centre_x=x, centre_y=0.5, width=0.18, height=0.4)
            for index, x in enumerate([0.20, 0.30, 0.32, 0.44, 0.50, 0.60])
        ]
        path = build_crop_path(
            hesitating, source_aspect=WIDE, target_aspect=TALL, energy="active"
        )
        assert not path.is_static


class TestSweep:
    @pytest.mark.parametrize("direction", ["sweep_left", "sweep_right"])
    def test_a_sweep_moves_without_a_subject(self, direction: str) -> None:
        path = build_sweep_path(
            source_aspect=WIDE,
            target_aspect=TALL,
            duration_seconds=2.5,
            direction=direction,
            energy="active",
        )
        assert not path.is_static

    def test_the_two_directions_are_opposite(self) -> None:
        left = build_sweep_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.5,
            direction="sweep_left", energy="active",
        )
        right = build_sweep_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.5,
            direction="sweep_right", energy="active",
        )
        went_left = left.keyframes[-1].crop.x - left.keyframes[0].crop.x
        went_right = right.keyframes[-1].crop.x - right.keyframes[0].crop.x
        assert went_left < 0 < went_right


class TestZoom:
    @pytest.mark.parametrize("direction", ["push_in", "pull_out"])
    def test_a_zoom_changes_the_crop_size(self, direction: str) -> None:
        path = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction=direction, energy="dynamic", budget=0.5,
        )
        assert not path.is_static
        first, last = path.keyframes[0].crop, path.keyframes[-1].crop
        if direction == "push_in":
            assert last.width < first.width
        else:
            assert last.width > first.width

    def test_the_zoom_aims_at_the_subject(self) -> None:
        """Pushing at the centre of the frame wastes the only vertical move."""

        low = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", centre_y=0.75, energy="dynamic",
            budget=0.5, framing="centre",
        )
        middle = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", centre_y=0.5, energy="dynamic",
            budget=0.5, framing="centre",
        )
        assert low.keyframes[-1].crop.y > middle.keyframes[-1].crop.y

    def test_the_source_bounds_the_push(self) -> None:
        """A 4K source affords a push a 1080 one does not."""

        uhd = zoom_budget(
            source_width=3840, source_height=2160, source_aspect=WIDE,
            target_aspect=TALL, output_width=1080, output_height=1920,
        )
        hd = zoom_budget(
            source_width=1920, source_height=1080, source_aspect=WIDE,
            target_aspect=TALL, output_width=1080, output_height=1920,
        )
        assert uhd < hd, "more pixels should allow a tighter crop"

    def test_the_delivered_enlargement_is_reported(self) -> None:
        """Sharpness is measurable, so nobody should judge it from a preview."""

        budget = zoom_budget(
            source_width=3840, source_height=2160, source_aspect=WIDE,
            target_aspect=TALL, output_width=1080, output_height=1920,
        )
        path = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", energy="dynamic", budget=budget,
        )
        tightest = min(path.keyframes, key=lambda k: k.crop.width).crop
        upscale = achieved_upscale(
            tightest, source_width=3840, source_height=2160,
            output_width=1080, output_height=1920,
        )
        assert upscale <= MAX_UPSCALE + 1e-6


class TestHandoff:
    def test_it_pans_between_measured_centres(self) -> None:
        """Not a cut, and not a guess at where the subjects are."""

        path = build_handoff_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=3.0,
            from_centre=0.359, to_centre=0.635, energy="calm",
        )
        assert not path.is_static
        first, last = path.keyframes[0].crop, path.keyframes[-1].crop
        assert first.x + first.width / 2 == pytest.approx(0.359, abs=0.02)
        assert last.x + last.width / 2 == pytest.approx(0.635, abs=0.02)

    def test_it_does_not_overshoot_the_subjects(self) -> None:
        """Aiming at nine-box extremes ran past both handsets into background."""

        path = build_handoff_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=3.0,
            from_centre=0.359, to_centre=0.635, energy="calm",
        )
        assert path.travel() < 0.4, "the move should be the gap, not the frame"


class TestFit:
    def test_a_subject_too_wide_is_reported(self) -> None:
        oversized = [
            Observation(seconds=i * 0.5, centre_x=0.5, centre_y=0.5, width=0.55, height=0.6)
            for i in range(4)
        ]
        degradations: list = []
        build_crop_path(
            oversized, source_aspect=WIDE, target_aspect=TALL,
            clip_id="k02", min_visible=0.85, degradations=degradations,
        )
        assert any(
            step.ladder_other == "subject_larger_than_crop"
            for step in degradations
        ), "a shot showing half its subject is a fact review needs"

    def test_a_subject_that_fits_is_not_reported(self) -> None:
        degradations: list = []
        build_crop_path(
            _moving(), source_aspect=WIDE, target_aspect=TALL,
            clip_id="k03", min_visible=0.85, degradations=degradations,
        )
        assert not [
            step for step in degradations
            if step.ladder_other == "subject_larger_than_crop"
        ]

    def test_visible_fraction_measures_both_axes(self) -> None:
        crop = CropBox(0.4, 0.0, 0.2, 1.0)
        inside = Observation(seconds=0, centre_x=0.5, centre_y=0.5, width=0.1, height=0.5)
        outside = Observation(seconds=0, centre_x=0.9, centre_y=0.5, width=0.1, height=0.5)
        assert visible_fraction(crop, inside) == pytest.approx(1.0)
        assert visible_fraction(crop, outside) == pytest.approx(0.0)


def test_framing_intents_are_all_known_to_the_builders() -> None:
    for intent in INTENT_NAMES:
        path = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", energy="calm", budget=0.6, framing=intent,
        )
        assert path.keyframes


def test_a_subject_wider_than_the_delivery_is_recorded_under_every_move() -> None:
    """The fit check belongs to the clip, not to one branch of the dispatch.

    It lived inside the hold branch, so a wordmark 0.88 of a 16:9 frame wide
    was swept across when the planner asked for a hold and silently cropped to
    "Galaxy Unpac" when it asked for a push -- same material, same impossible
    promise, one of them unrecorded. None of the five path builders except
    build_crop_path reports fit, so nothing else caught it either.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    fact_at = source.index("subject_wider_than_delivery")
    # Every branch that dispatches on what the shot is, whatever those
    # happen to be called today. This named the two-subject pan branch,
    # which has since been deleted -- the property is "before all of them",
    # not "before these two". Guards that only skip or warn do not count.
    branches = [
        line
        for line in source.splitlines()
        if line.startswith("            if ")
        and ("move ==" in line or "move in " in line or "reframe.looks" in line)
    ]
    assert len(branches) >= 3, branches
    for branch in branches:
        assert source.index(branch) > fact_at, (
            f"the fit check must run before {branch!r} dispatches, or that "
            "move delivers a clipped subject with nothing in the report"
        )


def test_a_subject_that_cannot_fit_says_so_before_it_is_chosen() -> None:
    """`must_be_whole` is only answerable if the ceiling travels with the subject."""

    from montagewright.cli import _subject_line
    from montagewright.clipcard import SubjectBox

    wide = SubjectBox(
        label="寫著 Galaxy Unpacked 的螢幕畫面",
        centre_x=0.52, centre_y=0.49, width=0.88, height=0.81, moves=False,
    )
    line = _subject_line(wide, WIDE, TALL)
    assert "36%" in line, line

    small = SubjectBox(
        label="硬幣", centre_x=0.5, centre_y=0.5, width=0.10, height=0.2,
        moves=False,
    )
    # No fraction, because this one fits whole. Whether it moves is still
    # said: that question has an answer for every subject, not only the ones
    # the delivery frame cannot hold.
    fits = _subject_line(small, WIDE, TALL)
    assert fits.startswith("硬幣（")
    assert "%" not in fits


def test_a_pan_onto_a_small_subject_centres_it_rather_than_hugging_an_edge() -> None:
    """Both edge bounds exist for a subject that fills the crop.

    A folded Flip 0.19 of the frame wide, panned to inside a 0.316 crop,
    cannot touch both edges: "do not travel past the far edge" and "do not
    start far from the near edge" describe an empty interval. Resolving that
    by taking one bound put the phone a tenth of the way in with a third of
    the frame wall behind it, which reads on screen as the pan overshooting.
    """

    path = build_handoff_path(
        source_aspect=WIDE,
        target_aspect=TALL,
        duration_seconds=4.0,
        from_centre=0.28,
        to_centre=0.69,
        from_width=0.20,
        to_width=0.19,
    )
    end = path.keyframes[-1].crop
    where = (0.69 - end.x) / end.width
    assert 0.4 <= where <= 0.6, (
        f"the destination sits at {where:.2f} of the frame, not centred"
    )


def test_rhythm_is_told_the_length_it_is_dividing_up() -> None:
    """Eight lengths decided in isolation summed to 26s of a 45s film."""

    import inspect

    from montagewright import planner

    source = inspect.getsource(planner.decide_rhythm)
    assert "target_seconds" in source
    assert "定調要" in source


def test_selection_is_told_that_shot_count_is_a_length_decision() -> None:
    """Told only "35 seconds", selection picked sixteen shots.

    Every length downstream then had to be two seconds, which fits a static
    product view and does not fit a gesture playing out or a screen being
    read -- so the film hit its target duration by cutting away from more
    things sooner.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "顆數與長度是同一個決定" in prompt
    assert "seconds_needed" in prompt


def test_the_layer_that_picks_a_shot_says_how_long_it_needs() -> None:
    """Length started from a constant, not from the shot.

    Every clip left selection with a flat four-second window, so the layer
    that knew what the shot was for had no say in how long it ran, and the
    layer that set the length began from a number nobody chose.
    """

    import inspect

    from montagewright import cli
    from montagewright.planner import _selection_schema

    shot = _selection_schema(["C1"])["properties"]["shots"]["items"]
    assert "seconds_needed" in shot["required"]

    source = inspect.getsource(cli._edl_from_selection)
    assert "seconds_needed" in source
    assert "start + 4.0" not in source


def test_a_move_floor_reports_rather_than_lengthens() -> None:
    """The length is the planner's; the floor says what could not happen.

    A flat 2.5s raised any pan to 2.5s, which reads as a safeguard and is a
    length decision made by a constant -- how long a sweep needs depends on
    how far it travels and what is on the way, and only the layer that
    watched the shot knows that. The menu now says local code will not add
    time, so it must not.
    """

    import inspect

    from montagewright import grounding

    source = inspect.getsource(grounding._requested_duration)
    assert "MOVE_FLOORS" not in source, (
        "the requested length must come back whole, floor applied nowhere"
    )
    assert "move_too_short" in inspect.getsource(grounding.ground_timeline)


def test_the_menu_hands_the_timing_judgement_to_the_planner() -> None:
    from montagewright.capabilities import describe_for_prompt

    menu = describe_for_prompt()
    assert "seconds_needed" in menu
    # The property, not the wording: length is the planner's call and the
    # executor reports a shortfall rather than quietly padding it. This
    # asserted one sentence verbatim and broke when the menu was rewritten
    # around looks, though nothing it guards had changed.
    assert "照實記一筆" in menu or "照實回報" in menu
    assert "只有看過這顆畫面的人知道" in menu
    assert "至少" not in menu


def test_the_shot_reviewer_sees_the_shot_and_settles_its_degradations() -> None:
    """Adjudication rested on a viewer who never saw the shot in question.

    "The subject is 0.88 of frame wide and can show 36% of itself" is not
    judgeable from a thirty-second film and a line of numbers. Whoever
    watched that one shot settles it.
    """

    from montagewright.review import adjudicate
    from montagewright.schema import DegradationStep, Issue, ReviewVerdict

    step = DegradationStep(
        clip_id="k00",
        ladder="other",
        ladder_other="subject_wider_than_delivery",
        trigger="wider than any crop at the delivery aspect",
        measured={"subject_width_vw": 0.88},
    )
    silent = ReviewVerdict(verdict="approve", overall="", issues=[])

    kept = adjudicate([step], silent, {"k00": {
        "degradation_verdict": "acceptable", "note": "字完整讀得到",
    }})
    assert kept[0].adjudication == "accept"
    assert "字完整讀得到" in kept[0].adjudication_reason

    sent_back = adjudicate([step], silent, {"k00": {
        "degradation_verdict": "replan", "note": "字被裁掉一角",
    }})
    assert sent_back[0].adjudication == "replan"

    # No shot verdict: the whole-cut reviewer's silence still decides.
    assert adjudicate([step], silent, {})[0].adjudication == "accept"


def test_segments_survive_the_render_so_they_can_be_reviewed() -> None:
    import inspect

    from montagewright import pipeline

    assert "keep_segments=True" in inspect.getsource(pipeline.run)


def test_the_report_says_why_a_degradation_was_settled() -> None:
    """"replan" with no grounds leaves the reader where the reviewer was."""

    import inspect

    from montagewright import cli

    assert "adjudication_reason" in inspect.getsource(cli._write_report)


def test_replanning_is_a_new_plan_rather_than_a_softer_fallback() -> None:
    """A ladder answers a failed shot with a less obvious version of itself.

    Push less far, sweep more slowly, crop a little wider -- none of those
    ask why the shot failed. A coin that fell outside the frame is not
    recovered by a gentler push; it wants a different take, a different
    subject, or the admission that the shot was about the handset edge.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "replan_zh-TW.txt").read_text(encoding="utf-8")
    assert "不是把原本的做法縮水" in prompt
    assert "放棄這顆" in prompt
    # It must also be able to stand its ground: the shot reviewer sees one
    # shot with no context, and "swept past without stopping to be read" is
    # sometimes exactly what was wanted.
    assert "也可能是規劃本來就沒問題" in prompt


def test_a_sweep_is_judged_against_its_own_intent_not_legibility() -> None:
    """Reading the text is one valid outcome of a pan, not the only one.

    Leading the eye across a wall to open a scene is a different job from
    letting a viewer read the wall, and a reviewer holding every sweep to
    the second standard sends back shots that did what they meant to.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "shotreview_zh-TW.txt").read_text(encoding="utf-8")
    assert "判準來自這顆自己的宣告，不是一套通用標準" in prompt
    assert "本來就不是問題" in prompt or "完全不是問題" in prompt


def test_the_executor_does_not_swap_the_move_it_was_given() -> None:
    """A substitution the planner cannot see is a decision it cannot argue with.

    A replan chose hold for a wide title, reasoning that travelling across it
    was what cut it in the first place -- and the executor swapped the hold
    for a sweep, so the next review described a sweep across a clipped title
    and the loop spent a round fighting itself.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    assert "build_sweep_path" not in source.split('if move == "hold"')[1], (
        "the hold branch must hold; the fit is already recorded"
    )
    assert "subject_wider_than_delivery" in source


def test_must_be_whole_is_described_as_a_requirement_not_a_lever() -> None:
    """The planner replanned two shots by only setting this flag.

    Its stated reasoning was that the crop engine would scale to preserve
    the whole wordmark. Nothing does that -- shrinking a subject to fit is
    pillarboxing, which this pipeline will not do -- so the flag produced a
    record and an identical frame, and the loop spent a round on it twice.
    """

    from montagewright.planner import PROMPTS, _selection_schema

    # Lives on each look now rather than on the shot: a shot can settle on
    # a wordmark that must be whole and then on a face that need not be.
    described = _selection_schema(["C1"])["properties"]["shots"]["items"][
        "properties"
    ]["looks"]["items"]["properties"]["must_be_whole"]["description"]
    assert "does not fulfil" in described

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "是一句宣告，不是一個開關" in prompt
    replan = (PROMPTS / "replan_zh-TW.txt").read_text(encoding="utf-8")
    assert "不是一個做法" in replan


def test_no_pass_rations_its_output_ceiling() -> None:
    """A flat ceiling truncates the whole pass, not its tail.

    Twenty-two shots stopped mid-token at 8192 and the run died: the answer
    is one decision plus one sentence per shot, so it scales with the cut.
    """

    import inspect

    from montagewright import planner

    source = inspect.getsource(planner.decide_rhythm)
    assert "MAX_OUTPUT_TOKENS" in source
    assert '"max_output_tokens": 8192' not in source
    # Billing is on tokens produced, so rationing the ceiling bought nothing
    # and cost a whole pass. One generous constant, everywhere.
    assert planner.MAX_OUTPUT_TOKENS >= 32768


def test_an_action_beat_has_to_be_long_enough_to_be_one() -> None:
    """Twenty-six of forty-one beats in one library were under 0.25s.

    "The models rotate their phones, 0.02 to 0.06s" is not a span, and the
    two fields carrying every timestamp in the schema were the only ones
    with no description telling the model what a good answer looks like.
    Downstream, in-points were snapped onto those numbers.
    """

    from montagewright.clipcard import action_beats, card_schema

    entry = card_schema()["properties"]["action"]["items"]["properties"]
    assert entry["ends_seconds"].get("description"), (
        "the field that carries the timing must say what it wants"
    )

    card = {"action": [
        {"what": "翻轉手機", "starts_seconds": 0.02, "ends_seconds": 0.06},
        {"what": "手伸進畫面", "starts_seconds": 2.0, "ends_seconds": 3.4},
    ]}
    kept = action_beats(card)
    assert [beat.what for beat in kept] == ["手伸進畫面"]


def test_a_library_that_wrote_nothing_stops_the_run() -> None:
    """An empty card library is missing input, not a degradation.

    A NameError in the request took all seventy-four cards down, was
    reported as the routine "74 failed ($0.0000)" line, and the run went on
    to pick an aspect, choose sixteen shots and spend $1.80 planning a film
    out of nothing -- every downstream layer reading the absence as "these
    clips have no description" rather than as a failure.
    """

    import tempfile
    from pathlib import Path

    from montagewright.clipcard import CardLibraryEmpty, build_library

    class Refuses:
        class files:
            @staticmethod
            def upload(**_):
                raise RuntimeError("no")

    with tempfile.TemporaryDirectory() as work:
        clip = Path(work) / "a.mp4"
        clip.write_bytes(b"not really a video")
        try:
            build_library(
                {"a": clip}, Path(work) / "cards", client=Refuses()
            )
        except CardLibraryEmpty as error:
            assert "RuntimeError" in str(error), "the reason has to survive"
        else:
            raise AssertionError("an empty library must not pass silently")


def test_clip_cards_can_be_written_at_all() -> None:
    """The request referenced a name the module never imported."""

    import inspect

    from montagewright import clipcard

    assert "MAX_OUTPUT_TOKENS" in dir(clipcard)
    compile(inspect.getsource(clipcard), "clipcard.py", "exec")


def test_the_card_says_what_the_source_camera_does() -> None:
    """"The camera moves" does not support the decision it feeds.

    A reference cut held a static frame on the left-hand handset and let the
    take's own move bring a third one in from the right. Choosing that needs
    to know what the move reveals; a boolean cannot say, and the planner was
    left to either ignore the movement or lay a digital one over it.
    """

    from montagewright.clipcard import card_schema
    from montagewright.planner import MaterialItem, _describe_material

    schema = card_schema()
    assert "camera_motion" in schema["required"]

    described = _describe_material([
        MaterialItem(
            source_id="C1",
            duration_seconds=9.0,
            summary="兩台摺疊機並排",
            camera_moves=True,
            camera_motion="往右平移，右邊會有第三台手機進畫面",
        )
    ])
    assert "第三台" in described, described


def test_a_card_box_is_only_reused_when_nothing_moved() -> None:
    """The box says where, and said nothing about when.

    `moves` was parsed off every subject and read by no one, so a held frame
    on a take whose camera pans was aimed at wherever the card happened to
    look -- and the subject walked out of it. The card knows which kind of
    shot this is; the reuse now depends on it.
    """

    import inspect

    from montagewright import pipeline
    from montagewright.clipcard import card_schema, subjects_from_card

    box = card_schema()["properties"]["subjects"]["items"]
    assert "at_seconds" in box["required"], "a position needs its moment"

    parsed = subjects_from_card({"subjects": [{
        "label": "手機", "centre_x": 0.5, "centre_y": 0.5,
        "width": 0.2, "height": 0.4, "moves": True, "at_seconds": 1.2,
    }]})
    assert parsed[0].at_seconds == 1.2
    assert parsed[0].moves

    source = inspect.getsource(pipeline.follow_subjects)
    assert "not box.moves" in source, (
        "a moving subject must be measured over the shot, not reused"
    )


def test_takes_set_aside_before_planning_say_why() -> None:
    """A run working from sixty-six of seventy-four looked like a full one.

    The card gives a reason a take failed and the filter dropped it, so
    "why wasn't the good coin shot used" had no answer anywhere in the
    output -- not in the report, not on the console.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert "unusable_reason" in source
    assert "set_aside" in inspect.getsource(cli._write_report)


def test_the_web_run_reports_what_it_decided_not_just_the_file() -> None:
    """The question after a run is which take that is and why it looks like that."""

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert {"/api/runs", "/api/runs/{run_id}/video",
            "/api/runs/{run_id}/shot/{index}"} <= paths

    # What each shot is and why it looks like that. These used to be column
    # headings; the table became rows when it turned out seven columns of
    # very different lengths made every shot six hundred pixels tall.
    page = PAGE.read_text(encoding="utf-8")
    for shown in (
        "shotcard", "s.source_id", "s.camera_move", "s.subject", "s.why",
        "tellDegradation", "實際做到什麼",
    ):
        assert shown in page, shown


def test_a_track_can_be_measured_without_a_reviewed_lock() -> None:
    """Requiring a lock file first turns "see what this does" into an errand.

    The lock proves which analysis a delivery was cut against, which is worth
    its ceremony for a delivery and is pure friction for a trial. Downbeats
    and section boundaries are derived while locking, so deriving them here
    too is what keeps "land on the chorus" resolvable.
    """

    import inspect

    from montagewright import cli, grounding

    source = inspect.getsource(grounding.analyse_track)
    assert "downbeat" in source and "section_boundary" in source
    assert "analyse_track" in inspect.getsource(cli.command_render)


def test_a_transcript_is_its_own_card() -> None:
    """Subtitling something finished has nothing to do with cutting it.

    A clip card describes what a take looks like and is cached because that
    stays true. Speech is only worth paying for when it matters, and it is
    useful with no edit at all -- so it is a separate artifact, not more
    fields on the clip card.
    """

    from montagewright import clipcard, transcript

    assert "transcript" not in clipcard.card_schema()["properties"]
    assert transcript.CARD_VERSION != clipcard.CARD_VERSION

    fields = transcript._schema()["properties"]
    assert "language" in fields, "the locale was a guess; this is the answer"
    assert "speaker" in fields["lines"]["items"]["required"], (
        "a talking shot framed on whoever is not talking is the fault this "
        "is here to make fixable"
    )
    assert "heard" in fields["lines"]["items"]["properties"], (
        "what the recogniser said has to survive, or a correction is invisible"
    )


def test_spoken_boundaries_land_on_a_measured_break() -> None:
    """The model hears where a sentence ends; the recogniser marked where the
    sound broke. Neither alone puts the cut in the right place."""

    from montagewright.transcript import Word, gaps, snap

    words = [
        Word("溼", 0.0, 0.30), Word("了", 0.30, 0.62),
        Word("。", 0.62, 1.01),
        Word("回", 1.01, 1.66), Word("家", 1.66, 1.98),
    ]
    breaks = gaps(words)
    assert breaks == [1.01]
    assert snap(1.2, breaks) == 1.01
    # Too far to be the same boundary: a model second that lands nowhere near
    # a break is left alone rather than dragged across a word.
    assert snap(3.0, breaks) == 3.0


def test_every_upload_waits_until_the_file_can_be_used() -> None:
    """An upload returns before the service has finished with it.

    The cached path waited; the uncached branch in five other modules did
    not, so it worked on short clips and failed on the first long one with
    "not in an ACTIVE state".
    """

    import re
    from pathlib import Path

    for module in Path("src/montagewright").glob("*.py"):
        if module.name == "uploads.py":
            continue
        text = module.read_text(encoding="utf-8")
        assert not re.search(r"client\.files\.upload\(", text), (
            f"{module.name} uploads without waiting; use upload_now"
        )


def test_the_transcriber_is_reachable_as_its_own_command() -> None:
    from montagewright.cli import main

    try:
        main(["transcribe", "--help"])
    except SystemExit as exit_code:
        assert exit_code.code == 0


def test_music_goes_under_a_voice_rather_than_over_it() -> None:
    """Throwing the source audio away is right for b-roll and ruins an
    interview, where what was said is the whole content.

    A fixed lower level is not the answer either: quiet enough never to bury
    a sentence is too quiet to be doing anything in the gaps. Measured on one
    street interview, the voice as sidechain trigger pulled the bed down 8.2
    dB against the same bed with a silent trigger.
    """

    import inspect

    from montagewright import renderer

    source = inspect.getsource(renderer._mux_music)
    assert "sidechaincompress" in source
    assert "keep_voice" in inspect.signature(renderer.render).parameters
    # The voice has to reach the output, not only the compressor's key input.
    assert "asplit" in source and "[voice][ducked]amix" in source


def test_a_transcript_is_only_paid_for_where_speech_is_the_content() -> None:
    """A transcript costs a call and a minute a clip.

    On b-roll it answers a question nobody asked, and a flag somebody has to
    remember is a flag somebody forgets -- so the card, which already watched
    the clip with its audio, says which clips need one.
    """

    import inspect

    from montagewright import cli
    from montagewright.clipcard import card_schema

    speech = card_schema()["properties"]["speech"]
    assert speech["enum"] == ["none", "ambient", "content"]
    assert "speech" in card_schema()["required"]

    source = inspect.getsource(cli.command_render)
    assert '"speech") == "content"' in source
    assert "keep_voice=bool(transcripts)" in source


def test_the_planner_sees_the_sentences_it_is_choosing_between() -> None:
    """A window out of an interview is chosen because of a sentence."""

    from montagewright.cli import _speech_lines
    from montagewright.planner import MaterialItem, _describe_material
    from montagewright.transcript import CARD_VERSION

    lines = _speech_lines({
        "version": CARD_VERSION,
        "lines": [{
            "text": "夏天最崩潰的是流汗完又下雨",
            "speaker": "穿灰藍色T恤的受訪男子",
            "starts_seconds": 4.0, "ends_seconds": 9.2,
        }],
    })
    assert "穿灰藍色T恤的受訪男子" in lines[0]
    assert "4.0-9.2s" in lines[0]

    described = _describe_material([
        MaterialItem(source_id="S01", duration_seconds=70.0,
                     summary="街訪", speech=lines)
    ])
    assert "說了什麼" in described and "流汗完又下雨" in described


def test_a_cut_that_never_asked_for_a_beat_is_not_a_missed_one() -> None:
    """A speech-led cut read as 0/13 aligned.

    Thirteen shots, every one deliberately off the grid so a sentence could
    finish, and the fallback for "no rhythm pass ran" turned that into total
    failure in the one line anyone reads.
    """

    from montagewright.pipeline import Report

    speech = Report(total_cuts=13, aligned_cuts=0)
    speech.rhythm_decisions = {
        f"k{i:02d}": {"cut_on_beat": False} for i in range(13)
    }
    assert "0/0 cuts on a musical event (13 content-led by choice)" in (
        speech.summary()
    )

    silent = Report(total_cuts=4, aligned_cuts=4)
    assert "4/4 cuts on a musical event" in silent.summary()


def test_an_already_cut_file_is_opened_along_its_own_boundaries() -> None:
    """One file holding many takes is not one take.

    Handed over whole it becomes one card describing five minutes, one
    transcript, and a planner choosing windows out of a single source as
    though the cuts inside it were not there. A continuous take comes back
    as itself, which is the honest answer for a locked-off interview.
    """

    import inspect

    from montagewright import cli, grounding

    assert "shots_in" in inspect.getsource(cli.command_render)
    source = inspect.getsource(grounding.shots_in)
    assert "scene" in source
    # The split pieces keep the name they came from, so a report traces back.
    assert '{path.stem}-{index:02d}' in inspect.getsource(cli.command_render)


def test_music_is_not_required_to_make_a_cut() -> None:
    """Refusing to run without a bed was the tool deciding on the way in.

    A cut carried by what people say does not need one, and without a grid
    every length is content-led -- which is what a speech cut wants.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert "--music or --music-map is required" not in source
    assert "lengths will be led by content" in source


def test_a_chinese_filename_can_be_uploaded() -> None:
    """The uploader puts the name in a header, and a header is latin-1.

    Every clip in a folder named in Chinese -- which is most of what this is
    pointed at -- failed with UnicodeEncodeError, and the material's own name
    is not part of the bytes being sent.
    """

    import tempfile
    from pathlib import Path

    from montagewright.uploads import _ascii_named

    work = Path(tempfile.mkdtemp())
    chinese = work / "夏日街訪_夏天最崩潰的事-00.mp4"
    chinese.write_bytes(b"x" * 64)
    with _ascii_named(chinese) as sendable:
        sendable.name.encode("ascii")
        assert sendable.suffix == ".mp4"
        assert sendable.read_bytes() == chinese.read_bytes()

    plain = work / "C8371.MP4"
    plain.write_bytes(b"y")
    with _ascii_named(plain) as sendable:
        assert sendable == plain, "an ASCII name needs no detour"


def test_a_timeline_is_written_only_when_asked_for() -> None:
    """Most runs want a file. A timeline is for the run where somebody
    intends to open it and disagree with one shot."""

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert 'args.timeline != "none"' in source
    parser_source = inspect.getsource(cli.main)
    assert '"--timeline"' in parser_source and 'default="none"' in parser_source


def test_a_timeline_carries_the_reasons_and_the_original_media() -> None:
    """Handles and reasons both existed and neither could be used.

    Every segment renders with half a second either side for exactly this
    and nothing consumed it; every shot carries why it was chosen, in a
    debugging artifact no editor reads. Referencing the original source is
    what makes the handle a thing you can drag.
    """

    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.timeline import to_fcpxml, to_xmeml

    source = Source(
        source_id="S00", path=Path("/tmp/夏日街訪-00.mp4"),
        duration_seconds=48.0, width=1920, height=1080,
    )
    plan = RenderPlan(project_id="t", segments=[
        Segment(clip_id="k00", source=source, in_seconds=4.0,
                out_seconds=7.0, crop=CropBox(0.34, 0.0, 0.3164, 1.0))
    ])
    report = {
        "selection": {"shots": [{"why": "受訪者回答核心問題",
                                 "camera_move": "hold"}]},
        "rhythm": {"k00": {"why": "讓句子講完"}},
        "shots": {"k00": {"delivered": True, "note": "框住講話的人"}},
        "degradations": [],
    }
    for build in (to_xmeml, to_fcpxml):
        xml = build(plan, report, name="cut", width=1080, height=1920)
        assert "受訪者回答核心問題" in xml, "the reason has to travel"
        assert "讓句子講完" in xml
        # The original file, not the rendered segment: trimming outward is
        # only possible against material the timeline can still reach.
        assert "-00.mp4" in xml and "segments" not in xml


def test_rhythm_is_decided_whether_or_not_there_is_music() -> None:
    """This asserted the opposite, on reasoning that turned out to be wrong.

    "Its whole job is reconciling a length against a track" -- but what it
    reconciles is the sequence against itself. Gated on having a grid, a film
    with no music had nothing deciding its pacing at all: every length was
    whatever selection guessed for that shot alone, and nothing ever asked
    whether eight in a row had a shape. Speech-led cuts, which need shaping
    most, got none of it.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.run)

    assert "decide_rhythm_first and grid is not None" not in source
    assert "if decide_rhythm_first:" in source


def test_a_film_with_no_track_is_told_so_rather_than_shown_an_empty_grid() -> None:
    from montagewright.planner import decide_rhythm
    import inspect

    said = inspect.getsource(decide_rhythm)

    assert "這支片沒有配樂" in said
    assert "沒有拍點要對" in said


def test_how_music_sits_under_speech_is_decided_not_computed() -> None:
    """A compressor ducks on signal and knows nothing about the film.

    On a cut that is speech end to end that means the bed climbs into every
    breath and is pushed down by the next line -- busier than sitting behind
    it steadily. Which of the two a film wants is an editorial call.
    """

    import inspect

    from montagewright import renderer
    from montagewright.planner import PROMPTS, _direction_schema

    field = _direction_schema()["properties"]["music_under_speech"]
    assert field["enum"] == ["bed", "duck", "none"]
    assert "music_under_speech" in _direction_schema()["required"]

    prompt = (PROMPTS / "direction_zh-TW.txt").read_text(encoding="utf-8")
    assert "music_under_speech" in prompt

    source = inspect.getsource(renderer._mux_music)
    assert 'under_speech == "bed"' in source
    assert "sidechaincompress" in source, "ducking is still available"


def test_the_bed_is_placed_under_the_voice_not_under_the_music() -> None:
    """"The music minus 12" lands wherever that track was mastered.

    A mastered track sits near -12 dBFS and a street interview averages
    around -22, so a fixed subtraction put the bed at exactly the level of
    the speech it was meant to be beneath, and the reviewer reported the
    voice as missing from a cut that contained it.
    """

    import inspect

    from montagewright import renderer

    assert not hasattr(renderer, "MUSIC_UNDER_VOICE_DB")
    assert renderer.BED_BELOW_VOICE_DB > 0
    source = inspect.getsource(renderer._mux_music)
    assert "_level(picture) - _level(music) - BED_BELOW_VOICE_DB" in source


def test_pauses_come_from_the_punctuation_the_recogniser_wrote() -> None:
    """The space between words is always zero; the punctuation is not.

    The transcriber segments a stream continuously, so each word's end is the
    next word's start -- ninety-five per cent of inter-word gaps in one
    interview were exactly 0.000s, and pause candidates built on them found
    four points in seventy seconds. A 。 is the recogniser saying it heard a
    break, and the token carries that break's span.
    """

    from montagewright.transcript import Word, gaps, snap_end

    words = [
        Word("行", 12.9, 13.14),
        Word("。", 13.14, 13.50),   # the break itself, 0.36s of it
        Word("對", 13.50, 13.68),
    ]
    assert gaps(words) == [13.5]
    # The out-point was landing on the last syllable's nominal end, which is
    # where the pause starts -- the word's decay is still to come.
    assert snap_end(13.14, gaps(words)) == 13.5
    # Never backwards: the pause before the final word is often the nearer one.
    assert snap_end(13.60, [13.5]) == 13.60


def test_the_direction_only_promises_what_the_tools_can_do() -> None:
    """It asked for keyword titles and jump cuts, and nothing downstream
    speaks either -- so the reviewer reported the film as failing to do what
    the film had asked of itself."""

    import inspect

    from montagewright import planner
    from montagewright.planner import PROMPTS

    assert "describe_for_prompt()" in inspect.getsource(planner.decide_direction)
    prompt = (PROMPTS / "direction_zh-TW.txt").read_text(encoding="utf-8")
    assert "只承諾做得到的事" in prompt


def test_cutting_before_a_sentence_ends_stays_available() -> None:
    """Snapping a card's line to the break is not a rule about the cut.

    The card records where the sentence ends, which is a fact. What the shot
    does with it is selection's -- `seconds_needed` is used as given, with no
    second snap on the way to the EDL -- so cutting away on the highest word
    and leaving the answer for the next shot is expressible. Only the prompt
    was talking anyone out of it, in words that could not tell a deliberate
    cliffhanger from a sentence hacked short to fit.
    """

    import inspect

    from montagewright import cli
    from montagewright.planner import PROMPTS

    source = inspect.getsource(cli._edl_from_selection)
    assert "snap_end" not in source, "the shot's out-point is selection's"

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "留懸念" in prompt
    assert "為了湊秒數砍掉半句是意外" in prompt


def test_the_voice_is_levelled_before_anything_goes_under_it() -> None:
    """One bed cannot sit under two speakers fourteen decibels apart.

    A street interview runs from a shouted answer to a mumbled one. Placed
    against the average, the bed sat comfortably under the loud speaker and
    five decibels under the quiet one -- close enough that the reviewer
    reported the voice as inaudible and, unable to hear the opening line,
    also reported the hook as missing.
    """

    import inspect

    from montagewright import renderer

    assert "speechnorm" in renderer.VOICE_LEVELLER
    source = inspect.getsource(renderer._mux_music)
    # Both paths: the steady bed and the ducked one.
    assert source.count("VOICE_LEVELLER") == 2


def test_a_replan_renders_the_same_way_the_first_pass_did() -> None:
    """The render call was written out twice and only one copy kept up.

    The first learned to keep the voice and lay the bed under it; the second,
    which runs after a replan, did not -- so any run that revised anything
    delivered the film with the speech thrown away and nothing but music
    left, having sounded correct in the round before.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert source.count("keep_voice=bool(transcripts)") == 1
    assert source.count("resolved = run(") == 0, (
        "both renders go through one definition"
    )
    assert source.count("= cut(") == 2


def test_dropping_the_question_depends_on_something_carrying_the_premise() -> None:
    """"The answer implies the question" holds only when a title says it.

    Written as a flat rule it removed every host line from a street-interview
    Shorts that has no text cards, so the opening became a punchline with
    nothing to be the punchline of -- "來月經吧" is only startling after
    somebody asks what the worst thing about summer is.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "問句通常不用剪進去" not in prompt
    assert "觀眾從哪裡知道題目" in prompt


def test_the_page_takes_a_path_and_remembers_what_it_ran() -> None:
    """Two things a local tool should not have made anyone do.

    Uploading material that is already on the disk beside the server copies
    it into the browser to write it back out a directory away -- 836 MB of it
    for one interview. And runs lived in a temp directory keyed by an
    in-memory dict, so closing the server threw away every finished cut,
    when comparing this one against the last is most of the work.
    """

    from montagewright.webapp import MAX_UPLOAD_BYTES, PAGE, RUNS_ROOT, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/transcripts" in paths
    assert MAX_UPLOAD_BYTES > 0, "an uncapped upload can fill the disk"
    assert "runs" in str(RUNS_ROOT)

    page = PAGE.read_text(encoding="utf-8")
    for control in ("source_path", "music_path", "speech", "locale"):
        assert control in page, control
    # The panes the browser column offers. They were drawer tabs with longer
    # names when the drawer ran the width of the screen.
    assert "pane-past" in page and "先前" in page
    assert "pane-tx" in page and "逐字稿" in page


def test_a_folder_can_be_clicked_instead_of_typed() -> None:
    """A browser will not hand over a real path.

    A directory picker gives relative names and nothing else, so pointing
    this at material sitting next to the server meant typing the path out.
    The server lists the folders instead -- it binds to localhost, and the
    person using it owns the disk.
    """

    from fastapi.testclient import TestClient

    from montagewright.webapp import PAGE, create_app

    listing = TestClient(create_app()).get("/api/browse").json()
    assert listing["here"], "somewhere to start from"
    assert "folders" in listing and "videos" in listing
    # The count is what makes a listing useful: it says which folder holds
    # the rushes without descending into every one of them.
    assert all("clips" in folder for folder in listing["folders"])

    page = PAGE.read_text(encoding="utf-8")
    assert "browseTo" in page and "用這個資料夾" in page


def test_every_paid_stage_checks_the_cap_before_spending() -> None:
    """"Call before dispatching, so the cap stops work rather than paying
    for it" -- and three stages did not.

    Transcription, replanning and the re-render after a replan all recorded
    against the ledger and none of them asked it first, so a run at $5.90 of
    a $6 cap would still fire a replan and land past it. The accounting was
    right; the stopping was not.
    """

    import inspect
    import re

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    # Each of these is a paid call site; each must be preceded by a check.
    for call in ("transcribe(", "replan_shots("):
        for match in re.finditer(re.escape(call), source):
            before = source[max(0, match.start() - 700):match.start()]
            assert "ledger.check()" in before, f"{call} spends unchecked"

    # Inside the render, the subject pass is one call per shot, so a cap read
    # only between stages lets a whole plan through after it is reached.
    from montagewright import pipeline

    inner = inspect.getsource(pipeline.follow_subjects)
    assert inner.count("_afford(report)") == inner.count("locate_subject(")


def test_a_pasted_path_is_taken_as_a_path() -> None:
    """Finder and browsers hand over a URL, terminals leave the quotes on.

    `Path("file:/Users/...")` is a relative directory called "file:", so a run
    started with one spent four minutes writing cards and only then failed on
    a track that had never been there.
    """

    from montagewright.webapp import _typed_path

    assert str(_typed_path("file:///Users/j/a%20b/%E5%A4%8F.mp3")) == (
        "/Users/j/a b/夏.mp3"
    )
    assert str(_typed_path("file:/Users/j/x.mp3")) == "/Users/j/x.mp3"
    assert str(_typed_path(' "/Users/j/y.mp3" ')) == "/Users/j/y.mp3"
    assert _typed_path("   ") is None


def test_the_long_silent_stage_reports_itself() -> None:
    """Seventy-four cards is four minutes with nothing on screen, which looks
    exactly like a hang."""

    import inspect

    from montagewright import cli, clipcard

    assert "progress" in inspect.signature(clipcard.build_library).parameters
    assert "card {index}/{total}" in inspect.getsource(cli.command_render)


def test_a_new_run_clears_the_last_one_from_the_page() -> None:
    """Leaving the previous result up while cards are written reads as though
    the new run had already finished."""

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    start = page.index("runId = (await res.json()).run_id;")
    assert "classList.add('hide')" in page[start:start + 600]


def test_the_page_says_what_each_stage_is_doing() -> None:
    """"cards: 74 written" does not explain four minutes of nothing.

    The raw output is what the pipeline says to itself. Someone watching a
    run wants to know which part is happening and what that part is for --
    especially during the long silent one.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "const STEPS" in page and "paintSteps" in page
    for said in ("Apple 辨識器給逐字時間", "SAM 逐幀追出在哪裡",
                 "先逐顆對照它自己的計畫"):
        assert said in page, said
    # The machine output is still there, folded away.
    assert "原始輸出" in page


def test_music_can_be_picked_the_same_way_as_the_rushes() -> None:
    """Typing was the only way, and a path pasted from Finder is a URL."""

    from fastapi.testclient import TestClient

    from montagewright.webapp import PAGE, create_app

    audio = TestClient(create_app()).get(
        "/api/browse", params={"kind": "audio"}
    ).json()
    assert "videos" in audio  # the listing switches what it looks for
    page = PAGE.read_text(encoding="utf-8")
    assert "browse-music" in page and "openPicker" in page


def test_a_failed_run_says_so_where_it_can_be_seen() -> None:
    """A run died on the project spend cap and the page showed ticked steps.

    The traceback was in the raw output, folded away, and the stage list
    marked everything up to the failure as complete and everything after it
    as pending -- which is what a run still in progress looks like.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "showFault" in page
    assert "Google 專案的月度支出上限滿了" in page
    assert "ai.studio/spend" in page
    # The step it died on is marked, not ticked.
    assert "broke" in page and ".steps li.broke .name" in page


def test_a_run_that_died_can_be_picked_up_where_it_stopped() -> None:
    """The quota ran out after seventy-four cards and a selection.

    The cards survived because they are keyed by the content they describe.
    The direction and the selection were not kept at all, so the second
    attempt paid for them again -- and they are the same question whenever
    the material, the brief and the aspect are the same.
    """

    import inspect

    from montagewright import cli
    from montagewright.webapp import PAGE, create_app

    source = inspect.getsource(cli.command_render)
    assert '_decided(work, "direction"' in source
    assert '_decided(work, "selection"' in source
    # A different brief is a different question, not a stale answer.
    assert "brief, args.aspect" in source

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/resume" in paths
    assert "繼續跑" in PAGE.read_text(encoding="utf-8")


def test_cards_belong_to_the_material_not_to_one_attempt() -> None:
    """"Content-addressed" and "written once per output directory" at once.

    Cards and transcripts describe the clip, which is why they are worth
    keeping -- and they lived beside the output, so a second cut of the same
    rushes rewrote all seventy-four of them for forty-four cents before
    anything had been decided. They are named by the bytes now and live in
    one library, so any run over the same material finds them.
    """

    import inspect

    from montagewright import cli, clipcard
    from montagewright.uploads import default_library

    assert "library" in str(default_library())
    assert "content_hash(proxy)" in inspect.getsource(clipcard.build_library)

    source = inspect.getsource(cli.command_render)
    assert 'library / "cards"' in source
    assert 'library / "transcripts"' in source


def test_the_cut_can_be_adjusted_without_replanning_it() -> None:
    """The four things anyone wants after watching it once.

    A sentence cut short, a shot that runs long, an order that reads better
    the other way, one shot that should go. None of those need the film
    re-planned, and re-planning them costs money and changes everything
    else. The strip trims, reorders and drops; the recut renders from the
    amended order with no model calls, so it is free.
    """

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/recut" in paths
    assert "/api/runs/{run_id}/timeline-data" in paths

    page = PAGE.read_text(encoding="utf-8")
    for piece in ("paintStrip", "data-edge", "ondrop", "data-kill", "undo-cut"):
        assert piece in page, piece
    # Pulling the head earlier eats into the handle rather than sliding the
    # shot, so the out-point stays where the edit put it.
    assert "eats into the handle" in page
    # One shot at a time, beside the viewer, instead of a table to scroll.
    assert "function inspect" in page and 'id="inspector"' in page


def test_a_degradation_is_shown_in_words_with_its_number() -> None:
    """"static_on_subject　accept" names the code that raised it.

    A degradation is worth recording rather than hiding because it carries
    the measurement that forced it. Printing the enum and the verdict hides
    exactly that -- the reader learns there was a fallback and nothing about
    what happened or how far off it was.
    """

    from montagewright.schema import DegradationStep
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "tellDegradation" in page
    # Every ladder the schema can produce has something to say.
    ladders = DegradationStep.model_fields["ladder"].annotation.__args__
    for name in ladders:
        if name != "other":
            assert f"{name}:" in page, name
    for named in ("subject_wider_than_delivery", "tracking_lost_most_frames",
                  "trim_window_clamped_to_source", "subject_larger_than_crop"):
        assert f"{named}:" in page, named
    assert "measured" in page and "已改動" in page


def test_the_adjudication_says_who_looked_and_what_they_concluded() -> None:
    """"accept" is a word from the schema, not an account of anything.

    It means somebody watched that shot on its own, with the degradation
    beside it, and decided the picture still works. Which layer did the
    watching is the part that matters, and the part that changed: the
    whole-cut reviewer never saw the shot it was ruling on.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "看過畫面：可以這樣交" in page
    assert "逐顆驗收單獨看了這一顆" in page
    assert "沒有人看過" in page


def test_an_empty_panel_says_what_empty_means() -> None:
    """A panel showing only its subtitle reads as broken.

    Nothing set aside means every clip was usable; no transcript means the
    speech was not the content. Both are answers, and both looked like a
    failure to load.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "emptyRow" in page
    assert "每一支素材都用上了" in page
    assert "這一輪沒有做逐字稿" in page
    assert "還沒有跑過任何一輪" in page


def test_earlier_runs_are_grouped_by_the_material() -> None:
    """One folder of rushes, several cuts out of it.

    That is the unit of the work and the unit the cards are cached by, so
    runs in a group are the cheap ones -- which is the thing worth seeing
    when deciding whether to cut the same material again.
    """

    from montagewright.webapp import PAGE, create_app
    from fastapi.testclient import TestClient

    listed = TestClient(create_app()).get("/api/runs").json()["runs"]
    assert all("source_path" in row for row in listed)
    page = PAGE.read_text(encoding="utf-8")
    assert "bySource" in page and "支剪成" in page


def test_the_slowest_stage_reports_each_shot() -> None:
    """A grounding call and sometimes a propagation, per shot.

    SAM writes its progress with carriage returns that never reach a log, so
    minutes passed with the last visible line still being about the music --
    and a process at 0% CPU waiting on the network looks exactly like a hung
    one from the page.
    """

    import inspect

    from montagewright import pipeline
    from montagewright.webapp import PAGE

    assert "subject {index}/{total}" in inspect.getsource(
        pipeline.follow_subjects
    )
    page = PAGE.read_text(encoding="utf-8")
    # How long the current stage has been going, so waiting reads as waiting.
    assert "stepSince" in page and "已經 ${since(" in page
    assert "CPU 沒有動是正常的" in page


def test_every_way_a_clip_misses_the_film_is_shown() -> None:
    """Only the rarest of three was on screen.

    A card can call a take unusable, the direction can rule one out, and
    selection can simply pass one over -- and the last of those is how most
    of a folder does not reach the film. Showing only the first made a run
    that ruled out three takes and passed over sixty report that every clip
    had been usable.
    """

    import inspect

    from montagewright import cli
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    for how in ("卡片判定不能用", "定調排除", "選片沒挑"):
        assert how in page, how
    # "Passed over" can only be told from what was on the table.
    assert "material_ids" in page
    assert "material_ids" in inspect.getsource(cli._write_report)


def test_the_timeline_has_tracks_on_one_time_scale() -> None:
    """A cut with speech under music is two things happening at once.

    The page could only play it. Seeing where the voice sits and where the
    bed steps back is the difference between trusting the mix and checking
    it -- and a strip laid out by proportion rather than by time cannot line
    up with anything.
    """

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/waveform/{which}" in paths

    page = PAGE.read_text(encoding="utf-8")
    for piece in ("paintRuler", "paintWaves", "playhead",
                  "lane-voice", "lane-music"):
        assert piece in page, piece
    # Clicking the reel seeks; a timeline that cannot be scrubbed is a chart.
    assert "$('reel').onclick" in page


def test_a_run_is_found_after_a_restart_without_asking_for_the_list() -> None:
    """Runs are picked up off disk lazily and only the listing did it.

    Every other endpoint answered "no such run" until something happened to
    fetch the list first, which is an ordering nobody could see.
    """

    import inspect

    from montagewright import webapp

    source = inspect.getsource(webapp.create_app)
    guts = source[source.index("def _run(run_id"):]
    assert "recall()" in guts[:400]


def test_the_crop_can_be_checked_against_the_take_it_came_from() -> None:
    """"push_in on (0.507, 0.484)" is a claim in a report.

    Playing the take underneath with the box drawn on it is that claim being
    checked, which is the whole job here. The rendered segment cannot show it
    -- it is the answer, not the working.
    """

    from montagewright.webapp import PAGE, create_app
    from fastapi.testclient import TestClient

    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/source/{which}" in paths

    # The crop path travels with the timeline, keyframe by keyframe.
    page = PAGE.read_text(encoding="utf-8")
    assert "cropAt" in page and "drawCrop" in page
    assert "原素材＋裁切框" in page
    # A label built from the crop width said the same sentence on every
    # shot -- 9:16 out of 16:9 is 0.316 wide always. It has to say where the
    # box is pointed and whether it travelled.
    assert "cropSays" in page and "across(" in page
    assert "偏左" in page and "偏右" in page and "置中" in page


def test_a_card_is_found_by_what_it_describes_not_by_its_filename(
    tmp_path,
) -> None:
    """The rebuild used to key cards by filename and miss every one.

    A card is named for the hash of the proxy it describes. Taking that name
    to be the source id meant the map was empty in a way nothing could see:
    every lookup missed, every shot reframed with no subject, every crop dead
    centre -- and the report still described the subject it had followed.
    """

    from montagewright.clipcard import card_map
    from montagewright.uploads import content_hash

    proxies = tmp_path / "proxies"
    proxies.mkdir()
    cards = tmp_path / "cards"
    cards.mkdir()

    proxy = proxies / "C8329.mp4"
    proxy.write_bytes(b"not really a video, but it hashes")
    named_for_its_bytes = cards / f"{content_hash(proxy)[:20]}.json"
    named_for_its_bytes.write_text("{}", encoding="utf-8")

    found = card_map(proxies, cards)
    assert found == {"C8329": named_for_its_bytes}
    # The old way. It is what the bug looked like from the inside.
    assert "C8329" not in {path.stem: path for path in cards.glob("*.json")}


def test_a_proxy_with_no_card_is_left_out_rather_than_guessed_at(
    tmp_path,
) -> None:
    from montagewright.clipcard import card_map

    (tmp_path / "proxies").mkdir()
    (tmp_path / "cards").mkdir()
    (tmp_path / "proxies" / "C0001.mp4").write_bytes(b"unanalysed")
    assert card_map(tmp_path / "proxies", tmp_path / "cards") == {}


def test_the_planner_is_told_to_name_a_subject_the_card_already_measured(
) -> None:
    """A reworded subject is a subject that has to be located again.

    The listing hands the planner every subject the card measured, with its
    box. Free-form naming meant six shots in nine described theirs in wording
    the card never used -- sometimes in another language entirely -- and each
    miss fell through to a paid grounding call for a position already sitting
    in the library.
    """

    from montagewright.planner import _selection_schema

    schema = _selection_schema(["C8330", "C8332"])
    said = str(schema)
    assert "可框住的主體" in said
    assert "copy" in said and "exactly" in said


def test_a_block_carries_the_shot_it_came_from() -> None:
    """Peeking at the take needs the shot's index, not the reel's.

    The two are the same number until a recut drops or reorders anything, and
    the block never carried either -- so the take never loaded, and the crop
    box had no picture to be drawn on.
    """

    from pathlib import Path

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    # Addressed by the take now. The position was the bug this test was
    # written for and then became one of its own: it changes at every cut
    # even when the next shot comes out of the same file, so the browser
    # dropped a take it already had and fetched it again under a new name.
    assert "/source/${encodeURIComponent(b.source_id)}" in page
    # The endpoint still answers to a position, because a block carries one
    # and an older page may still ask that way.
    from montagewright import webapp

    import inspect

    resolve = inspect.getsource(webapp.create_app)
    assert "which.isdigit()" in resolve
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert '"index": index,' in source


def test_the_reel_says_whether_its_boxes_are_the_render_s_or_a_rebuild(
) -> None:
    """A guess shaped like evidence is worse than no evidence.

    Only a held frame can be re-derived after the fact, and only when a card
    can name the subject. Where it cannot, the rebuild centres -- and a
    centred box drawn over the take, unlabelled, says the render centred
    too. This view exists to be checkable, so it has to say which it holds.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "crops_are" in page and "cropsAre" in page
    assert "推測" in page
    assert "不等於實際裁切" in page


def test_peeking_does_not_seek_the_take_every_frame() -> None:
    """A seek is a decode from the nearest keyframe.

    Doing one per frame to keep two players aligned pinned a core and played
    like a slideshow, when a playing video keeps its own time for free.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "raw.play()" in page
    assert "cut.paused ? 0.08 : 0.35" in page


def test_the_take_is_served_as_a_proxy_when_there_is_one() -> None:
    """128MB of 4K to draw a rectangle on, where 256KB says the same thing."""

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'proxy = run.output / "work" / "proxies" / f"{source_id}.mp4"' in source
    assert "if proxy.exists():" in source


def test_one_reframe_builder_for_the_run_and_for_the_rebuild() -> None:
    """The rebuild's copy was missing then_subject.

    A pan between two subjects in one frame is the only move that reads it,
    so the handoff branch was never reached and a pan measured at
    0.278 -> 0.696 was rebuilt as a held centre crop.
    """

    from montagewright.schema import reframe_of

    both = reframe_of({
        "subject": "the left, black smartphone",
        "then_subject": "the right, white smartphone",
        "camera_move": "pan",
    })
    assert both.then_subject is not None
    assert both.then_subject.description == "the right, white smartphone"
    assert reframe_of({"subject": "a phone"}).then_subject is None


def test_a_proxy_is_kept_where_the_cards_it_feeds_are_kept(tmp_path) -> None:
    """Re-encoding seventy-four 4K files to ask the first question.

    A proxy is a pure function of the bytes it was made from -- the same
    reason the card built from it is content-addressed and shared. Keeping it
    in the output directory meant a second cut of the same rushes paid the
    whole encode again before anything was decided.
    """

    from montagewright.cli import _make_proxy
    from montagewright.uploads import content_hash

    library = tmp_path / "library"
    source = tmp_path / "C0001.MP4"
    source.write_bytes(b"pretend this is a take")

    kept = library / "proxies" / f"{content_hash(source)[:20]}.mp4"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"already encoded once")

    first = _make_proxy(source, tmp_path / "a" / "C0001.mp4", library=library)
    assert first.read_bytes() == b"already encoded once"

    # A second run over the same rushes finds it too, under its own name.
    second = _make_proxy(source, tmp_path / "b" / "C0001.mp4", library=library)
    assert second.read_bytes() == b"already encoded once"
    assert second.name == "C0001.mp4"


def test_the_zoom_slider_does_not_redraw_the_waveform_per_pixel() -> None:
    """Drawing a waveform decodes the whole cut.

    It is cached by the width asked for, and the slider asked at a width
    derived from its exact position -- a miss every time, two ffmpeg passes
    over the film per pixel of drag. The width is rounded to something the
    cache can hold, and the request waits for the drag to settle.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "Math.ceil(asked / 500) * 500" in page
    assert "paintWavesWhenSettled" in page
    assert "clearTimeout(wavesSoon)" in page


def test_the_run_can_write_down_the_crops_it_used(tmp_path) -> None:
    """The record has to survive a real path, not a plausible one.

    It was written against a field name Keyframe does not have, which no
    test touched -- so it raised only after twelve paid grounding calls, at
    the moment the run had everything it needed and was about to render.
    """

    import json

    from montagewright.executor import CropBox
    from montagewright.pipeline import write_crops
    from montagewright.reframe import CropPath, Keyframe

    path = CropPath(keyframes=[
        Keyframe(seconds=0.0, crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0)),
        Keyframe(seconds=2.5, crop=CropBox(x=0.41, y=0.05, width=0.26, height=0.82)),
    ])
    out = tmp_path / "deep" / "crops.json"
    write_crops({"k00": path}, out)

    back = json.loads(out.read_text(encoding="utf-8"))
    assert [k["at"] for k in back["k00"]] == [0.0, 2.5]
    assert back["k00"][1]["w"] == 0.26


def test_the_selection_becomes_an_edl_without_reaching_for_a_missing_name(
    tmp_path,
) -> None:
    """Two runs died here on names that were not defined.

    Both were a moved import, and both raised only after the run had paid for
    cards, direction and selection -- the point where an editing tool has
    spent everything and delivered nothing. Nothing exercised this function,
    so nothing said so until it was expensive.
    """

    from montagewright.cli import _edl_from_selection

    selection = {
        "shots": [
            {
                "source_id": "C0001",
                "subject": "the left, black smartphone",
                "then_subject": "the right, white smartphone",
                "camera_move": "pan",
                "start_seconds": 1.5,
                "seconds_needed": 3.0,
                "why": "hand off between the two handsets",
                "energy": "medium",
            },
            {
                "source_id": "C0002",
                "subject": "the coin beside the hinge",
                "camera_move": "hold",
                "start_seconds": 0.0,
                "must_be_whole": True,
            },
        ]
    }

    edl, snaps = _edl_from_selection(selection, tmp_path, cards={})

    assert [clip.clip_id for clip in edl.clips] == ["k00", "k01"]
    first, second = edl.clips
    assert first.reframe.camera_move == "pan"
    assert first.reframe.then_subject is not None
    assert first.approx_out_seconds - first.approx_in_seconds == 3.0
    # No seconds_needed means the fallback length, not zero.
    assert second.approx_out_seconds > second.approx_in_seconds
    assert second.reframe.subject.min_visible == 1.0
    assert snaps == {}


def test_the_overlay_eases_the_way_the_render_does() -> None:
    """The box is checked against the picture, so it has to move like it.

    The crop expression ramps on smoothstep -- the camera takes up the move
    and sets it down. Interpolating the drawn box linearly put it in the
    wrong place through the middle of every move, which is the part of a
    move anyone is checking.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "u * u * (3 - 2 * u)" in page

    # The same curve the expression builder writes.
    from montagewright.reframe import _eased

    assert "*(3-2*" in _eased(0.0, 1.0, 0.0, 1.0)


def test_the_reel_moves_without_relaying_itself_out_every_frame() -> None:
    """A one-pixel line, sixty times a second, at the cost of a full layout.

    Writing `left` on the playhead relayouts the reel under it; a transform
    is composited. Reading the scroller's geometry after writing that style
    forces the browser to flush the layout it was told to do. And the clock
    reads in seconds, so writing its text every frame is fifty-nine text
    measurements a second that change nothing.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "translateX(${x}px)" in page
    assert "if (now !== clockSaid)" in page
    # Every measurement read before anything is written.
    move = page[page.index("function movePlayhead()"):]
    move = move[:move.index("\n}")]
    assert move.index("box.scrollLeft, wide") < move.index("style.transform")


def test_the_crop_overlay_does_not_measure_the_page_every_frame() -> None:
    """Two getBoundingClientRect calls a frame, to move one rectangle.

    Where the take sits inside the frame changes when the window changes or
    another take loads, and at no other time.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "let placed = null" in page
    assert "function forgetPlacement()" in page
    assert "loadedmetadata" in page


def test_the_reviewer_is_told_what_this_tool_cannot_do() -> None:
    """A reviewer cannot tell "done badly" from "cannot be done here".

    A brief asked for title cards. There is no text layer, so the reviewer
    reported their absence as a fault, returned "revise" on a cut that was
    exactly as planned, and sent a paid round after something no replan can
    ever fix. The list of what the executor can do already lives in one
    place and is rendered into the prompt that chooses; the list of what it
    cannot now sits beside it, rendered into the prompt that judges.
    """

    from montagewright.capabilities import CANNOT, describe_limits_for_prompt
    from montagewright.review import PROMPTS

    said = describe_limits_for_prompt()
    assert "字卡" in said and "轉場" in said
    assert len(CANNOT) == said.count("\n") + 1

    prompt = (PROMPTS / "review_zh-TW.txt").read_text(encoding="utf-8")
    assert "{limits}" in prompt
    assert "不要把它們當成缺點寫進 issues" in prompt

    # And it is actually substituted, not left as a literal brace.
    filled = prompt.replace("{limits}", said)
    assert "{limits}" not in filled and "字卡" in filled


def test_a_run_made_from_the_command_line_is_openable(tmp_path) -> None:
    """The interface listed only the cuts it had started itself.

    A run from the command line left a report, a film and two timelines, and
    nothing that could open them -- so the one made to check the crops with
    could not be looked at.
    """

    import json

    import montagewright.webapp as web

    folder = tmp_path / "check-0805"
    (folder / "out").mkdir(parents=True)
    (folder / "out" / "command.json").write_text(
        json.dumps({"source": "/rushes", "command": ["render", "/rushes"]}),
        encoding="utf-8",
    )
    (folder / "out" / "report.json").write_text("{}", encoding="utf-8")

    was_root, was_runs = web.RUNS_ROOT, dict(web.RUNS)
    try:
        web.RUNS_ROOT = tmp_path
        web.RUNS.clear()
        web.recall()
        assert "check-0805" in web.RUNS
        found = web.RUNS["check-0805"]
        assert found.source == "/rushes"
        assert found.state == "done"
        assert found.started_at > 0
    finally:
        web.RUNS_ROOT = was_root
        web.RUNS.clear()
        web.RUNS.update(was_runs)


def test_a_cut_written_anywhere_can_still_be_listed(tmp_path) -> None:
    """The interface only scans its own runs folder.

    `--output ~/cut` is what the README tells people to type, and it produced
    a film, a report and two timelines that nothing could open. The runs
    folder gets a link to wherever the cut actually went, so every path in
    the interface carries on believing the layout it already believes.
    """

    import montagewright.cli as command
    import montagewright.webapp as web

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        elsewhere = tmp_path / "somewhere" / "mycut"
        elsewhere.mkdir(parents=True)
        (elsewhere / "report.json").write_text("{}", encoding="utf-8")

        command._make_findable(elsewhere)

        link = tmp_path / "runs" / "mycut" / "out"
        assert link.is_symlink()
        assert link.resolve() == elsewhere.resolve()
        assert (link / "report.json").exists()

        # Twice is not two links to the same place, nor a crash.
        command._make_findable(elsewhere)
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["mycut"]
    finally:
        web.RUNS_ROOT = was


def test_a_cut_already_in_the_runs_folder_is_left_alone(tmp_path) -> None:
    import montagewright.cli as command
    import montagewright.webapp as web

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        inside = tmp_path / "runs" / "abc123" / "out"
        inside.mkdir(parents=True)
        command._make_findable(inside)
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["abc123"]
    finally:
        web.RUNS_ROOT = was


def test_a_line_lands_where_the_cut_put_the_take_it_came_from() -> None:
    """A line is timed against its take; the cut kept part of that take.

    This lived inside the SRT endpoint because that was the only thing that
    needed it. The track on the timeline needs it, and burning it into the
    picture will need it, and three copies of "where does this line land" is
    three answers to it.
    """

    from montagewright.transcript import against_cut

    lines = against_cut(
        [
            {"source_id": "A", "start_seconds": 2.0},
            {"source_id": "B", "start_seconds": 0.0},
        ],
        {"k00": {"seconds": 3.0}, "k01": {"seconds": 2.0}},
        {
            "A": {"lines": [
                {"text": "早安", "starts_seconds": 2.5, "ends_seconds": 4.0,
                 "speaker": "主持人"},
                # Spoken in the take, after the part that was used.
                {"text": "太早了", "starts_seconds": 9.0, "ends_seconds": 10.0},
            ]},
            "B": {"lines": [
                {"text": "第二顆", "starts_seconds": 0.2, "ends_seconds": 1.4},
            ]},
        },
    )

    assert [line.text for line in lines] == ["早安", "第二顆"]
    assert lines[0].starts_seconds == 0.5      # 2.5 in a take entered at 2.0
    assert lines[0].speaker == "主持人"
    assert lines[1].starts_seconds == 3.2      # after a three-second shot


def test_who_said_it_is_a_decision_rather_than_a_default() -> None:
    """Two callers had already drifted -- one prefixed, one did not."""

    from montagewright.transcript import Line, to_srt

    lines = [
        Line(text="早安", starts_seconds=0.5, ends_seconds=2.0,
             speaker="主持人"),
        Line(text="你好", starts_seconds=2.0, ends_seconds=3.0),
    ]
    named = to_srt(lines, with_speaker=True)
    plain = to_srt(lines)

    assert "主持人：早安" in named and "主持人" not in plain
    # And neither loses the numbering or the timestamps.
    for made in (named, plain):
        assert made.startswith("1\n00:00:00,500 --> 00:00:02,000\n")
        assert "\n2\n00:00:02,000 --> 00:00:03,000\n" in made


def test_an_edited_line_is_what_appears_and_the_transcript_stays_true(
    tmp_path, monkeypatch,
) -> None:
    """Gemini fixes most of what the recogniser mishears, not all of it.

    A product name is exactly the kind of word it gets wrong. The correction
    is kept beside the transcript rather than written over it: the transcript
    is what was heard, which stays true, and this is what should be on
    screen.

    Built the way a run builds it -- a proxy per source, and a transcript in
    the shared library named for that proxy's bytes. Two readers were looking
    in the output directory and keying by filename, so every run had an empty
    transcript tab and no subtitles, and both read as "no speech here".
    """

    import json
    from dataclasses import dataclass

    from montagewright.transcript import save
    from montagewright.uploads import content_hash
    from montagewright.webapp import _subtitle_lines

    monkeypatch.setenv("MONTAGEWRIGHT_LIBRARY", str(tmp_path / "library"))

    @dataclass
    class Pretend:
        output: Path

        def report(self):
            return {
                "selection": {"shots": [{"source_id": "A",
                                         "start_seconds": 0.0}]},
                "rhythm": {"k00": {"seconds": 3.0}},
            }

    proxies = tmp_path / "out" / "work" / "proxies"
    proxies.mkdir(parents=True)
    proxy = proxies / "A.mp4"
    proxy.write_bytes(b"a take with someone talking in it")

    save({"lines": [
        {"text": "Galaxy Z 佛的", "starts_seconds": 0.0,
         "ends_seconds": 2.0, "heard": "Galaxy Z 佛的"},
    ]}, tmp_path / "library" / "transcripts"
        / f"{content_hash(proxy)[:20]}.json")

    run = Pretend(output=tmp_path / "out")
    assert [line.text for line in _subtitle_lines(run)] == ["Galaxy Z 佛的"]

    (tmp_path / "out" / "work" / "subtitles.json").write_text(
        json.dumps([{"at": 0.0, "until": 2.0, "text": "Galaxy Z Fold",
                     "heard": "Galaxy Z 佛的"}], ensure_ascii=False),
        encoding="utf-8",
    )
    fixed = _subtitle_lines(run)
    assert [line.text for line in fixed] == ["Galaxy Z Fold"]
    # What was actually heard survives the correction.
    assert fixed[0].heard == "Galaxy Z 佛的"


def test_the_safe_area_is_a_property_of_where_the_film_is_going() -> None:
    """A 9:16 cut is watched inside an app that draws over its own bottom.

    The handle, the caption and the button rail sit in the lower fifth, so a
    subtitle where a subtitle traditionally goes is a subtitle nobody reads.
    A 16:9 cut has none of that. One number for both would be the execution
    layer deciding something about distribution.
    """

    from montagewright.subtitles import SAFE_AREAS, safe_area

    tall = safe_area("9:16")
    wide = safe_area("16:9")
    assert tall.up_from_bottom > wide.up_from_bottom * 2
    assert tall.side_margin > wide.side_margin
    # Every aspect the render flag offers has one.
    assert set(SAFE_AREAS) == {"9:16", "4:5", "1:1", "16:9"}
    # An aspect nobody planned for gets the most constrained band, not none.
    assert safe_area("21:9") == SAFE_AREAS["9:16"]


def test_a_subtitle_never_loses_a_word_to_fit(tmp_path) -> None:
    """Cutting the overflow at max_lines dropped the end of a sentence and
    left a cut that looked finished.

    Fitting is tried in this order: split the sentence into separate cues,
    then set it smaller, and only then let it take a third row. What is
    never tried is saying less.
    """

    from montagewright.subtitles import _face, draw_line, safe_area, wrap

    area = safe_area("9:16")
    room = round(1080 * (1 - area.side_margin * 2))
    # Nowhere to break it into cues -- no punctuation anywhere -- and too
    # long for two rows even at the smallest size this will set.
    said = (
        "夏天最崩潰的事情就是流汗然後又曬傷然後又中暑然後還要擠捷運"
        "然後回到家發現冷氣壞掉真的是非常非常痛苦的一件事情啊"
        "而且隔天起來還要再經歷一次一模一樣的事情"
    )
    asked = round(1920 * area.text_height)
    assert len(wrap(said, _face(asked), room)) > area.max_lines

    made = draw_line(
        said, width=1080, height=1920, area=area, into=tmp_path / "one.png",
    )
    assert made is not None
    drawn, left, top = made
    assert drawn.exists() and top > 0

    # Whatever size it settled on, every character is still on the picture.
    for attempt in range(6):
        size = max(12, round(asked * (1 - attempt * 0.06)))
        rows = wrap(said, _face(size), room)
        if len(rows) <= area.max_lines:
            break
    assert "".join(rows) == said


def test_no_line_begins_with_a_mark_that_closes_one() -> None:
    """Breaking before a comma hangs it under the line above.

    It is the one typographic mistake in Chinese that everybody notices, and
    it was in the first frame that came out of this.
    """

    from montagewright.subtitles import NEVER_STARTS, _face, safe_area, wrap

    area = safe_area("9:16")
    room = round(360 * (1 - area.side_margin * 2))
    for said in (
        "對，然後你就…你就已經濕漉漉了，然後慢慢被自己蒸乾",
        "在夏天讓你最崩潰的事情是什麼？我覺得是流汗，還有曬傷。",
    ):
        for line in wrap(said, _face(round(640 * area.text_height)), room)[1:]:
            assert line[0] not in NEVER_STARTS, line


def test_two_lines_are_balanced_rather_than_filled() -> None:
    """A full line and an orphan reads as a mistake on screen."""

    from montagewright.subtitles import _face, wrap

    face = _face(40)
    said = "對，然後你就已經濕漉漉了，然後慢慢被自己蒸乾"
    wide = face.getbbox(said)[2]
    lines = wrap(said, face, round(wide * 0.6))

    assert len(lines) == 2
    shorter = min(face.getbbox(one)[2] for one in lines)
    longer = max(face.getbbox(one)[2] for one in lines)
    assert shorter > longer * 0.6, lines


def test_a_long_sentence_becomes_several_cues_not_more_rows() -> None:
    """The transcript's idea of a line is a sentence.

    The median is thirteen characters and the tail runs to fifty-six.
    Wrapping the long ones put fifty characters of Chinese over somebody's
    face, which is not a subtitle, it is a paragraph -- and setting it
    smaller only made it a smaller paragraph.
    """

    from montagewright.subtitles import _face, safe_area, split_cues, wrap
    from montagewright.transcript import Line

    area = safe_area("9:16")
    face = _face(round(1920 * area.text_height))
    room = round(1080 * (1 - area.side_margin * 2))
    said = (
        "哦，如果是這種…這種就是如果今天洗完澡，然後出來又是剛好冷氣"
        "又壞掉的話，應該就是會蠻…蠻不開心、蠻不爽的呀，對。"
    )

    cues = split_cues([Line(text=said, starts_seconds=10.0,
                            ends_seconds=17.0)], face, room)

    assert len(cues) > 1
    # Every one of them fits on a single row.
    for cue in cues:
        assert len(wrap(cue.text, face, room)) == 1, cue.text
    # Nothing said twice, nothing lost, and the window is the one it had.
    assert "".join(cue.text for cue in cues) == said
    assert cues[0].starts_seconds == 10.0
    assert abs(cues[-1].ends_seconds - 17.0) < 1e-6
    # And they run in order, without gaps or overlaps.
    for before, after in zip(cues, cues[1:]):
        assert abs(before.ends_seconds - after.starts_seconds) < 1e-6


def test_a_brief_cue_is_joined_only_while_it_still_fits_one_row() -> None:
    """Two rules pull against each other, and one of them wins.

    A cue too short to read is a flicker, so it gets joined to the one
    before. But joining up to two rows traded that fault for a worse one --
    a wall of text where a quick line was wanted. So the join happens only
    while the result still fits on a single row, and a short cue that
    cannot be absorbed stays as it is.
    """

    from montagewright.subtitles import _face, _width, split_cues
    from montagewright.transcript import Line

    face = _face(40)
    room = _width("十二個字的一行寬度啊啊", face)
    said = "好，對，是，嗯，然後呢，就這樣，真的很誇張啊我跟你講"
    cues = split_cues(
        [Line(text=said, starts_seconds=0.0, ends_seconds=1.2)], face, room,
        least=0.7,
    )

    # Some joining happened: fewer cues than there are places to break.
    assert 1 < len(cues) < said.count("，") + 1
    # None of them overflows the row.
    for cue in cues:
        assert _width(cue.text, face) <= room, cue.text
    assert "".join(cue.text for cue in cues) == said


def test_a_shot_that_catches_part_of_a_sentence_shows_that_part() -> None:
    """The window was clipped to the shot and the words were not.

    So a shot holding one second of a ten-second sentence put the whole
    sentence on screen for one second. There are no word timings kept, so
    the share of the window stands in for the share of the words, and the
    ends are nudged to where the sentence pauses.
    """

    from montagewright.transcript import Line, _within

    line = Line(
        text="對，然後你就…你就已經濕漉漉了，然後慢慢被自己…被下午的太陽弄乾",
        starts_seconds=63.95,
        ends_seconds=74.03,
    )

    # The shot catches only the first second of it.
    opening = _within(line, from_seconds=63.95, to_seconds=64.95)
    assert opening and len(opening) < len(line.text) / 3
    assert line.text.startswith(opening.rstrip("…，"))

    # Wholly inside the shot: untouched, not re-cut.
    assert _within(line, from_seconds=60.0, to_seconds=80.0) == line.text

    # A sliver too short to read is nothing, rather than two characters.
    assert _within(line, from_seconds=63.95, to_seconds=64.05) == ""


def test_clipping_a_line_never_inverts_the_slice() -> None:
    """Snapping the head past the tail produced an empty string.

    rfind counts from the end when given a negative start, so an unclamped
    search window looked at the wrong part of the line -- and the subtitle
    it produced simply vanished, which nothing would have reported.
    """

    from montagewright.transcript import Line, _within

    line = Line(
        text="對，然後你就…你就已經濕漉漉了，然後慢慢被自己…被慢慢被下午的太陽弄乾了",
        starts_seconds=0.0,
        ends_seconds=10.0,
    )
    # Every window of a reasonable size gives something or nothing on
    # purpose; none of them gives an accidental empty string.
    for at in range(0, 9):
        got = _within(line, from_seconds=float(at), to_seconds=at + 2.0)
        assert got == "" or len(got) >= 3, (at, got)
        assert got in line.text or got.strip("…，。") in line.text, (at, got)


def test_a_refused_run_leaves_nothing_behind(tmp_path) -> None:
    """The folder was made before the input was checked.

    So every mistyped path left an empty directory in the runs folder that
    nothing would ever open, list or clean up.
    """

    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        client = TestClient(web.create_app())

        gone = client.post("/api/runs", data={"source_path": "/no/such/one"})
        assert gone.status_code == 400

        empty = tmp_path / "no-footage"
        empty.mkdir()
        (empty / "notes.txt").write_text("not a video", encoding="utf-8")
        barren = client.post("/api/runs", data={"source_path": str(empty)})
        assert barren.status_code == 400

        wrong = client.post(
            "/api/runs", data={"source_path": "/tmp", "aspect": "3:2"}
        )
        assert wrong.status_code == 400

        made = list((tmp_path / "runs").iterdir()) if (
            tmp_path / "runs"
        ).exists() else []
        assert made == [], made
    finally:
        web.RUNS_ROOT = was


def test_nothing_cut_yet_is_an_invitation_not_an_empty_editor() -> None:
    """A black rectangle, empty tracks and an empty inspector.

    That is what a first run saw, and it reads as broken rather than as new.
    The only readable thing on the page was a line at the bottom of a drawer.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="nothing-yet"' in page
    assert "還沒有剪過東西" in page
    assert "function showWorkspace(" in page
    # Off at startup, on only when a run is opened.
    assert "showWorkspace(false);" in page
    assert "showWorkspace(true);" in page


def test_the_setup_offers_what_the_command_line_offers() -> None:
    """A flag the interface cannot set is a flag most people never find."""

    from pathlib import Path

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="subtitles"' in page
    assert "body.append('subtitles'" in page

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'subtitles: str = Form("sidecar")' in source
    assert 'command += ["--subtitles", subtitles]' in source


def test_the_system_is_asked_for_a_font_before_paths_are_guessed() -> None:
    """A hard-coded list of paths is a guess about somebody else's machine.

    It was wrong on this one in both directions: it missed the font the
    system would have named, and the font the system names first is one
    FreeType cannot open. So asking is the start of the search, not the end.
    """

    from montagewright import subtitles as typeset

    face = typeset._face(40, text="夏天最崩潰的事")
    assert face is not None
    assert typeset._can_draw(face, "夏天最崩潰的事")

    # An explicit choice wins over anything found.
    was = typeset.CHOSEN
    try:
        typeset.CHOSEN = "/System/Library/Fonts/STHeiti Medium.ttc"
        assert typeset._face(40, text="夏天").path.endswith("STHeiti Medium.ttc")
    finally:
        typeset.CHOSEN = was


def test_a_font_is_chosen_for_the_language_not_for_one_stray_character(
) -> None:
    """One emoji in a line means no Chinese font draws everything.

    Taking the first candidate that did set a whole street interview in a
    maths font, which had the emoji and not the language.
    """

    from montagewright import subtitles as typeset

    with_emoji = typeset._face(40, text="好熱😀真的")
    plain = typeset._face(40, text="好熱真的")
    assert with_emoji.path == plain.path
    assert typeset._can_draw(with_emoji, "好熱真的")


def test_characters_the_font_cannot_spell_are_named() -> None:
    """Pillow draws a missing glyph as an empty box and says nothing, so a
    name it cannot spell reaches a finished film."""

    from montagewright.subtitles import cannot_spell

    assert cannot_spell("夏天最崩潰的事") == ""
    assert "😀" in cannot_spell("好熱😀真的")


def test_one_shot_can_be_turned_down_without_re_planning_anything() -> None:
    """Levelling makes every speaker the same loudness.

    That is not the same as every speaker being right: one of them stood
    next to a road. The gain belongs to the segment, survives a re-cut, and
    costs nothing because nothing has to be decided again.
    """

    from pathlib import Path

    from montagewright.executor import Segment, Source
    from montagewright.webapp import PAGE

    made = Segment(
        clip_id="k00",
        source=Source(source_id="A", path=Path("/nowhere.mp4"), width=1920,
                      height=1080, duration_seconds=10.0),
        in_seconds=0.0, out_seconds=2.0,
    )
    assert made.gain_db == 0.0

    renderer = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "renderer.py"
    ).read_text(encoding="utf-8")
    assert 'f"volume={segment.gain_db:.2f}dB"' in renderer
    # Silence costs a filter nobody needs.
    assert "abs(segment.gain_db) > 0.01" in renderer

    web = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'gains[f"k{index:02d}"]' in web
    assert "segment.gain_db = gains.get(segment.clip_id, 0.0)" in web

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="gain"' in page and "gain_db: b.gain_db || 0" in page


def test_the_regional_face_and_a_weight_that_reads_over_a_picture() -> None:
    """A .ttc holds several faces and index 0 is whatever it lists first.

    For PingFang that is Hong Kong Regular, so a Taiwanese cut was being set
    in the wrong regional character forms, in a weight too thin to hold up
    against a moving picture.
    """

    from montagewright import subtitles as typeset

    for lang, region in (("zh-tw", "TC"), ("zh-cn", "SC"), ("zh-hk", "HK")):
        family, style = typeset._face(40, text="夏天", lang=lang).getname()
        if not family.startswith("PingFang"):
            continue  # another machine, another font; nothing to assert
        assert family.endswith(region), (lang, family)
        assert style in ("Medium", "Semibold"), (lang, style)


def test_the_words_move_onto_the_cut_with_the_lines() -> None:
    """Otherwise a line asked to fill as it is said fills to the rhythm of
    a completely different part of the interview."""

    from montagewright.transcript import words_against_cut

    cards = {"A": {"words": [
        {"text": "在", "starts_seconds": 10.0, "ends_seconds": 10.3},
        {"text": "夏", "starts_seconds": 10.3, "ends_seconds": 10.6},
        # Spoken in the take, outside the part that was used.
        {"text": "後", "starts_seconds": 30.0, "ends_seconds": 30.4},
    ]}}
    moved = words_against_cut(
        [{"source_id": "A", "start_seconds": 10.0}],
        {"k00": {"seconds": 2.0}},
        cards,
    )
    assert [word.text for word in moved] == ["在", "夏"]
    assert moved[0].starts_seconds == 0.0
    assert abs(moved[1].starts_seconds - 0.3) < 1e-6


def test_a_cue_fills_at_the_speed_it_was_actually_said() -> None:
    """Character by character, against measured timings rather than a pace.

    That is the difference between reading along and watching a progress
    bar. When the two cannot be lined up, the cue is drawn whole rather
    than guessed at.
    """

    from montagewright.subtitles import spans_in
    from montagewright.transcript import Word

    said = "在夏天讓你最崩潰的事情是什麼？"
    words = [
        Word(text=ch, starts_seconds=i * 0.3, ends_seconds=(i + 1) * 0.3)
        for i, ch in enumerate("在夏天讓你最崩潰的事情是什麼")
    ]
    marks = spans_in(said, words, 0.0, 5.0)

    assert len(marks) == len(words)
    # Each mark says how much of the line has been said by when.
    assert marks[0] == (1, 0.3)
    # The question mark is revealed with the character before it.
    assert marks[-1][0] == len(said)
    # Nothing measurable in this window: draw it in one piece.
    assert spans_in(said, words, 90.0, 95.0) == []


def test_the_interface_offers_the_fonts_this_machine_has() -> None:
    """A flag on the command line is a flag most people never find, and
    typing a path to a font file is worse than that."""

    from pathlib import Path

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    # Both places a subtitle gets set: before a run, and when burning one.
    assert 'id="setup-font"' in page and 'id="font"' in page
    assert 'id="setup-look"' in page and 'id="look"' in page
    assert "'/api/fonts'" in page and "function loadFonts(" in page
    assert "body.append('subtitle_font'" in page

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert '@app.get("/api/fonts")' in source
    assert 'command += ["--subtitle-font", subtitle_font]' in source
    # A chosen font applies to one render and does not leak into the next.
    assert "was, typeset.CHOSEN = typeset.CHOSEN, (font or None)" in source
    assert "typeset.CHOSEN = was" in source


def test_the_font_list_leaves_out_what_nobody_should_pick() -> None:
    """LastResort is the font that draws the boxes, and the dotted names
    are interface variants the system keeps for itself."""

    from montagewright.subtitles import fonts_here

    found = fonts_here()
    if not found:
        return  # no fontconfig on this machine; nothing to check
    assert all(not one["family"].startswith(".") for one in found)
    assert all("LastResort" not in one["file"] for one in found)
    assert len({one["family"] for one in found}) == len(found)


def test_the_preview_places_subtitles_where_the_burn_will() -> None:
    """Correcting wording without seeing it in place is guessing.

    So the player draws each cue over the picture as it plays -- and it is
    driven by the band the render actually uses, sent with the timeline. A
    second opinion about where the words sit would make the preview a lie
    about the thing it exists to preview.
    """

    from pathlib import Path

    from montagewright.subtitles import safe_area
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="burnt"' in page and "function showBurnt(" in page
    # The client reads the server's numbers rather than keeping its own.
    assert "safeArea = data.safe_area" in page
    for field in ("side_margin", "text_height", "up_from_bottom"):
        assert f"safeArea.{field}" in page
    # Nothing is placed against an element that has not loaded: before
    # metadata it is 300x150, which set a whole line at four pixels.
    assert "if (!video.videoWidth) { box.classList.add('hide'); return; }" in page

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert '"safe_area": _safe_area_of(report)' in source

    # And the numbers it sends are the ones the burn is built from.
    from montagewright.webapp import _safe_area_of

    for aspect in ("9:16", "16:9", "1:1", "4:5"):
        sent = _safe_area_of({"direction": {"aspect": aspect}})
        band = safe_area(aspect)
        assert sent["up_from_bottom"] == band.up_from_bottom
        assert sent["side_margin"] == band.side_margin
        assert sent["text_height"] == band.text_height


def test_the_font_list_follows_the_language_of_the_cut() -> None:
    """Offering Chinese faces to somebody cutting a Japanese interview is a
    list with nothing they want in it."""

    from montagewright.subtitles import fonts_here

    chinese = {one["family"] for one in fonts_here("zh-tw")}
    japanese = {one["family"] for one in fonts_here("ja")}
    if not chinese or not japanese:
        return  # no fontconfig here
    assert chinese != japanese


def test_the_zoom_budget_guards_the_size_that_is_actually_delivered() -> None:
    """It was calibrated against an output that never existed.

    zoom_budget assumed 1080x1920 while the renderer scaled every segment to
    whatever the opening crop happened to measure -- 1214x2160 off a 4K
    source at 9:16. So a push the report called 1.35x enlargement was 1.52x
    on disk, and the one number deciding how far a shot may push was
    protecting a file nobody was making.
    """

    from montagewright.executor import delivery_size
    from montagewright.reframe import MAX_UPSCALE, zoom_budget

    for aspect, expected in (
        (9 / 16, (1080, 1920)), (16 / 9, (1920, 1080)),
        (1.0, (1080, 1080)), (0.8, (1080, 1350)),
    ):
        assert delivery_size(aspect) == expected

    wide, tall = delivery_size(9 / 16)
    budget = zoom_budget(
        source_width=3840, source_height=2160, source_aspect=3840 / 2160,
        target_aspect=9 / 16, output_width=wide, output_height=tall,
    )
    # The tightest crop this allows, delivered at that size, enlarges by
    # exactly the budget -- not by half as much again.
    base_w = (9 / 16) / (3840 / 2160) * 3840
    tightest = base_w * budget
    assert abs((wide / tightest) - MAX_UPSCALE) < 0.01


def test_a_segment_is_scaled_to_the_delivery_not_to_its_own_crop() -> None:
    from pathlib import Path

    renderer = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "renderer.py"
    ).read_text(encoding="utf-8")
    # Both paths -- the moving crop and the still one -- land on one size.
    assert renderer.count('f"scale={output_size[0]}:{output_size[1]}"') == 2
    assert "keyframes[0].crop.to_pixels(" not in renderer


def test_the_planner_is_told_how_far_each_clip_can_be_pushed() -> None:
    """The execution layer was answering an editorial question alone.

    How far a shot may push before it softens is measured from that file's
    own dimensions -- it is a fact about the clip, not a rule about clips.
    Nobody was telling the planner, so it asked for pushes that could not be
    given and found out afterwards, in a degradation, with a constant in the
    executor deciding how much softness was acceptable.
    """

    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(source_id="BIG", duration_seconds=6.2, summary="4K",
                     push_room=1.52),
        MaterialItem(source_id="SMALL", duration_seconds=4.0, summary="1080p",
                     push_room=1.0),
    ])
    assert "最多推近 1.52×" in said
    assert "推近沒有空間" in said

    # And the prompt says what to do about it, or the number is decoration.
    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "最多推近" in prompt
    # What has to survive rewording: that "no room" is a fact about the file
    # rather than a caution, and that the answer is a different take rather
    # than a smaller version of the same move.
    assert "沒有空間" in prompt and "糊" in prompt
    assert "別把計畫打折" in prompt or "不要把原本的計畫打折" in prompt


def test_push_room_is_read_from_the_file_that_gets_cut(tmp_path) -> None:
    """The proxy is 640 pixels wide and would report that nothing anywhere
    can be pushed into."""

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "cli.py"
    ).read_text(encoding="utf-8")
    assert "originals = {path.stem: path for path in sources_paths}" in source
    assert "originals.get(source_id, proxy)" in source


def test_a_finished_cut_can_be_taken_away() -> None:
    """The interface could make a film, show it, prove the crop followed
    something -- and offer no way to get any of it out.

    The download links lived in the drawer along the bottom, and the drawer
    was deleted when the layout became columns. Nothing failed, nothing was
    reported: the routes stayed, and the only thing that went was every way
    of reaching them.
    """

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert {
        "/api/runs/{run_id}/deliverable",
        "/api/runs/{run_id}/timeline/{flavour}",
        "/api/runs/{run_id}/subtitles",
        "/api/runs/{run_id}/burned",
    } <= paths

    page = PAGE.read_text(encoding="utf-8")
    for reached in ("/deliverable", "/timeline/premiere", "/timeline/finalcut",
                    "/subtitles", "/burned"):
        assert reached in page, reached
    # Offered only when there is one -- a link that answers 404 reads as a
    # fault rather than as an absence.
    assert "$('dl-srt').classList.toggle('hide', !spoken)" in page


def test_final_cut_will_parse_what_we_write() -> None:
    """It refused the whole file and imported nothing.

    "No declaration for attribute time of element param" -- because the
    keyframes were written as <param time="..."/>, which is not a thing
    FCPXML has. A still frame is two attributes on adjust-transform; a move
    is keyframes inside a keyframeAnimation inside the param they belong to.
    """

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.reframe import CropPath, Keyframe
    from montagewright.timeline import to_fcpxml

    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=20.0, width=3840, height=2160)
    still = Segment(clip_id="k00", source=where, in_seconds=0.0,
                    out_seconds=2.0,
                    crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0))
    moving = Segment(
        clip_id="k01", source=where, in_seconds=3.0, out_seconds=6.0,
        crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0),
        crop_path=CropPath(keyframes=[
            Keyframe(seconds=0.0,
                     crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0)),
            Keyframe(seconds=3.0,
                     crop=CropBox(x=0.55, y=0.0, width=0.32, height=1.0)),
        ]),
    )
    made = to_fcpxml(
        RenderPlan(project_id="p", segments=[still, moving]), {},
        name="p", width=1080, height=1920,
    )

    root = ET.fromstring(made)
    assert not [
        one for one in root.iter("param") if "time" in one.attrib
    ], "param carries no time attribute in FCPXML"

    adjusts = list(root.iter("adjust-transform"))
    assert len(adjusts) == 2
    # The still one says it in attributes.
    assert adjusts[0].get("position") and adjusts[0].get("scale")
    assert list(adjusts[0]) == []
    # The moving one wraps its keyframes.
    named = {one.get("name") for one in adjusts[1].iter("param")}
    assert named == {"position", "scale"}
    for one in adjusts[1].iter("param"):
        frames = list(one.iter("keyframe"))
        assert len(frames) == 2
        assert all(f.get("time") and f.get("value") for f in frames)


def test_the_sequence_format_is_a_shape_not_a_preset_name() -> None:
    """Final Cut warned that the sequence's format was an unexpected value.

    FFVideoFormat is the prefix Apple gives its built-in presets --
    FFVideoFormat1080p30 and the like -- so a bare "FFVideoFormat" sent it
    looking for a preset that does not exist. A custom size does not claim
    to be a preset; it states its dimensions. And every asset names a format
    of its own, or Final Cut is left to work out the shape of the media by
    opening it.
    """

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.timeline import to_fcpxml

    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=20.0, width=3840, height=2160)
    made = to_fcpxml(
        RenderPlan(project_id="p", segments=[
            Segment(clip_id="k00", source=where, in_seconds=0.0,
                    out_seconds=2.0,
                    crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0)),
        ]),
        {}, name="p", width=1080, height=1920,
    )
    root = ET.fromstring(made)

    shapes = {one.get("id"): one for one in root.iter("format")}
    assert all(one.get("name") is None for one in shapes.values())
    # The sequence's own shape, and one for the media it cuts from.
    assert shapes["r1"].get("width") == "1080"
    assert any(one.get("width") == "3840" for one in shapes.values())

    for asset in root.iter("asset"):
        assert asset.get("format") in shapes, asset.get("id")
    assert root.find(".//sequence").get("format") in shapes


def test_keyframes_are_on_the_clip_s_own_clock() -> None:
    """A clip's clock starts at its source in-point, not at zero.

    Written from zero, a move began before the shot did and ended before it
    ended -- so the head and tail of every moving shot rendered with no
    transform, which for a 16:9 source in a 9:16 sequence is the picture
    letterboxed in black. That is the black somebody saw.
    """

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.reframe import CropPath, Keyframe
    from montagewright.timeline import to_fcpxml

    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=30.0, width=3840, height=2160)
    moving = Segment(
        clip_id="k00", source=where, in_seconds=4.0, out_seconds=7.0,
        crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0),
        crop_path=CropPath(keyframes=[
            Keyframe(seconds=0.0,
                     crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0)),
            Keyframe(seconds=3.0,
                     crop=CropBox(x=0.55, y=0.0, width=0.32, height=1.0)),
        ]),
    )
    root = ET.fromstring(to_fcpxml(
        RenderPlan(project_id="p", segments=[moving]), {},
        name="p", width=1080, height=1920,
    ))

    def ticks(stamp: str) -> float:
        top, _, bottom = stamp.rstrip("s").partition("/")
        return float(top) / float(bottom or 1)

    clip = root.find(".//asset-clip")
    began, ran = ticks(clip.get("start")), ticks(clip.get("duration"))
    for frame in root.iter("keyframe"):
        at = ticks(frame.get("time"))
        assert began - 1e-6 <= at <= began + ran + 1e-6, (
            f"keyframe at {at} is outside the clip's {began}..{began + ran}"
        )
    # And they span it, rather than sitting in a corner of it.
    times = sorted({ticks(f.get("time")) for f in root.iter("keyframe")})
    assert abs(times[0] - began) < 1e-6
    assert abs(times[-1] - (began + ran)) < 1e-6


def test_the_timeline_carries_the_bed(tmp_path) -> None:
    """It opened as a silent film with no sign there had been a track."""

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.timeline import to_fcpxml

    track = tmp_path / "bed.m4a"
    track.write_bytes(b"pretend this is music")
    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=20.0, width=3840, height=2160)
    root = ET.fromstring(to_fcpxml(
        RenderPlan(project_id="p", segments=[
            Segment(clip_id="k00", source=where, in_seconds=0.0,
                    out_seconds=3.0,
                    crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0)),
        ]),
        {}, name="p", width=1080, height=1920, music=track,
    ))

    bed = [
        one for one in root.iter("asset-clip")
        if one.get("audioRole") == "music"
    ]
    assert len(bed) == 1
    assert bed[0].get("lane") == "-1"       # under the picture
    assert bed[0].get("offset") == "0s"
    assert bed[0].get("ref") in {a.get("id") for a in root.iter("asset")}


def test_frames_are_not_pulled_when_there_is_nothing_to_ask() -> None:
    """Sampling frames runs ffmpeg over the take.

    Three branches did it and then checked whether a client existed to send
    them to -- so rebuilding a plan, which never has a client, extracted
    frames for every shot and discarded all of them. Opening a finished cut
    took eight and a half seconds of that before anything appeared.
    """

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "pipeline.py"
    ).read_text(encoding="utf-8")

    lines = source.splitlines()
    # Every place frames are pulled has a check that asking is possible
    # close above it -- never only below it.
    sites = [
        i for i, line in enumerate(lines)
        if "_sample_frames(" in line and not line.lstrip().startswith("def ")
    ]
    assert sites, "the sampling calls moved; this guard needs rewriting"
    for at in sites:
        # Above for a guard clause or the branch it sits in, just below for
        # the conditional form -- `_sample_frames(...) if _may_ask(...)`.
        near = "\n".join(lines[max(0, at - 30):at + 8])
        assert "_may_ask(client)" in near, (
            f"line {at + 1} pulls frames with no check for a client:"
            f"\n{near}"
        )


def test_a_file_is_measured_once(tmp_path) -> None:
    """Reading a file's shape costs an ffprobe -- a whole process.

    Drawing the timeline asked for the same dozen sources every time, which
    was most of the second and a half before a cut appeared.
    """

    import montagewright.pipeline as works

    calls = []
    real = works.subprocess.run

    def counted(command, *args, **kw):
        if command and command[0] == "ffprobe":
            calls.append(command[-1])
        return real(command, *args, **kw)

    made = tmp_path / "one.mp4"
    import subprocess as sp
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
            str(made)], check=True)

    works.subprocess.run = counted
    try:
        first = works.probe("A", made)
        again = works.probe("A", made)
        # A second id for the same bytes is still not a second ffprobe.
        other = works.probe("B", made)
    finally:
        works.subprocess.run = real

    assert len(calls) == 1, calls
    assert first.duration_seconds == again.duration_seconds
    assert other.source_id == "B"
    assert other.duration_seconds == first.duration_seconds


def test_the_subtitle_panel_has_the_operations_captioning_needs() -> None:
    """A lane you can drag is not an editor.

    Four things come up over and over when captioning: put a line in, cut
    one in two where the speaker paused, join two that were split too
    finely, and take one out. Everything else is typing.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="pane-subs"' in page
    for control in ("cue-add", "cue-split", "cue-join", "cue-del"):
        assert f"$('{control}')" in page, control

    # Timecodes are editable, not only draggable.
    assert "function unstamp(" in page and "function stamp(" in page
    # A cue may not end before it starts.
    assert "Math.min(want, subs[i].until - 0.1)" in page
    assert "Math.max(want, subs[i].at + 0.1)" in page
    # Splitting lands on a pause, near by, and never leaves a sliver.
    assert "Math.max(2, Math.min(cut, line.text.length - 2))" in page
    # The line being spoken is marked without rebuilding rows being typed in.
    assert "function followCue()" in page


def test_a_still_subject_says_so_before_the_shot_is_planned():
    """The executor was answering this after the plan was already spent.

    A frame told to follow something that stands still becomes a hold, and
    that substitution was happening in `reframe`, one pass too late to change
    anything, and recorded as a degradation -- forty-five of them in one run.
    The card has always known; the listing simply did not say.
    """

    from montagewright.cli import _subject_line
    from montagewright.clipcard import SubjectBox

    still = SubjectBox(
        label="the row of watches", centre_x=0.5, centre_y=0.5,
        width=0.56, height=0.3, moves=False, at_seconds=1.0,
    )
    walking = SubjectBox(
        label="the model", centre_x=0.5, centre_y=0.5,
        width=0.2, height=0.8, moves=True, at_seconds=1.0,
    )
    said = _subject_line(still, 16 / 9, 9 / 16)
    assert "定鏡" in said
    # And the fraction it can show is still there: two facts, one line.
    assert "57%" in said
    assert "移動" in _subject_line(walking, 16 / 9, 9 / 16)


def test_the_quoted_travel_time_is_the_one_the_render_will_take():
    """A price the planner budgets against, read from the executor's table.

    Selection is told `seconds_needed` must cover "the travel between" the
    looks and was never told the speed, so it could not price a move and
    stopped asking for them -- twenty-three shots, twenty-three single looks.
    The quote has to be the real number or it is worse than none.
    """

    from montagewright.planner import _travel_seconds
    from montagewright.reframe import ENERGY_LIMITS, seconds_needed_for
    from montagewright.schema import LOOK_ENERGIES

    room = 0.684
    quoted = _travel_seconds(room)
    for label, energy in LOOK_ENERGIES.items():
        # What the executor will actually charge for the same distance, with
        # no dwell at either end, is travel alone.
        stops = [(0.0, 0.0, 0.0, 0.3164), (0.0, room, 0.0, 0.3164)]
        charged = seconds_needed_for(stops, energy) - 2 * 0.35
        assert f"{label} {charged:.1f}s" in quoted, (label, quoted, charged)

    # All three, because the speed follows the energy the same answer picks.
    assert set(LOOK_ENERGIES) == {"low", "medium", "high"}
    assert len(set(ENERGY_LIMITS[e]["max_speed"] for e in LOOK_ENERGIES.values())) == 3


def test_the_energy_a_shot_asks_for_reaches_the_camera():
    """Two vocabularies, and until now nothing joined them.

    The shot says low/medium/high; the speed table is keyed calm/active/
    dynamic. `reframe_of` hard-coded "active", so a shot's own label had no
    effect on anything and a calm passage swept at the same rate as a frantic
    one.
    """

    from montagewright.reframe import ENERGY_LIMITS
    from montagewright.schema import look_energy, reframe_of

    for asked, expected in (("low", "calm"), ("medium", "active"), ("high", "dynamic")):
        assert look_energy(asked) == expected
        shot = {"looks": [{"at": "a", "seconds": 1.0}], "energy": asked}
        assert reframe_of(shot).camera_energy == expected

    # An absent or unknown label still has to land on a real speed.
    for junk in (None, "", "brisk"):
        assert look_energy(junk) in ENERGY_LIMITS


def test_the_planner_has_to_say_whether_the_frame_moves():
    """The question the looks refactor deleted, asked again without the menu.

    `camera_move` was a required enum, so every shot answered "does this one
    move?" before it could be written down. An array with a minimum length of
    one turned that into an option rather than a question, and the cheapest
    valid answer is a single look -- twenty-three shots, no move anywhere.
    """

    from montagewright.planner import _selection_schema

    shot = _selection_schema(["A"])["properties"]["shots"]["items"]
    assert "frame" in shot["required"]
    assert shot["properties"]["frame"]["enum"] == ["settles", "travels"]

    # Not the old menu under a new name. The two answers are not moves, and
    # the description does not hand back the vocabulary the refactor removed
    # -- "hold" is left out of this check because it is also an ordinary
    # English verb, which is exactly why it stopped being a move name.
    from montagewright.capabilities import MOVE_NAMES

    written = repr(shot["properties"]["frame"])
    menu = [name for name in MOVE_NAMES if name != "hold"]
    assert not [name for name in menu if name in written]
    assert not set(shot["properties"]["frame"]["enum"]) & set(MOVE_NAMES)

    # Asked before the looks are written, which is the working part -- a
    # model that has just written "travels" writes what follows in the
    # presence of that word.
    order = shot["required"]
    assert order.index("frame") < order.index("looks")
    assert list(shot["properties"]).index("frame") < list(
        shot["properties"]
    ).index("looks")


def test_a_plan_that_says_one_thing_and_describes_another_is_reported():
    """Prose and structure disagreed and nothing anywhere compared them.

    One shot's `why` said the frame sweeps across a row; its looks named a
    single place; the rhythm pass then repeated the sweep in its own
    reasoning; the film held still.
    """

    from montagewright.planner import frame_disagreements

    one = {"at": "a"}
    assert frame_disagreements([
        {"frame": "travels", "looks": [one]},
        {"frame": "settles", "looks": [one, one]},
    ]) == [
        "k00 said travels and gave one look",
        "k01 said settles and gave 2 looks",
    ]

    # Agreement is silent, in both directions.
    assert frame_disagreements([
        {"frame": "settles", "looks": [one]},
        {"frame": "travels", "looks": [one, one, one]},
    ]) == []

    # The looks still decide what is rendered: this only reports.
    from montagewright.schema import move_of_shot

    assert move_of_shot({"frame": "travels", "looks": [one]}) == "hold"


def test_the_speed_budget_covers_every_axis_not_just_across():
    """`_limit_speed` read `x` and copied the rest through unchanged.

    So a tilt arrived at whatever speed the keyframes asked for and a push
    changed size instantly, while the report said the energy budget had been
    applied. It was named for the budget and enforced it on the one move that
    existed when it was written.
    """

    from montagewright.reframe import ENERGY_LIMITS, Keyframe, _limit_speed
    from montagewright.executor import CropBox

    ceiling = ENERGY_LIMITS["calm"]["max_speed"]

    def travelled(start: CropBox, end: CropBox) -> tuple[float, float, float]:
        limited, _ = _limit_speed(
            [Keyframe(0.0, start), Keyframe(1.0, end)], ENERGY_LIMITS["calm"]
        )
        last = limited[-1].crop
        return (
            abs(last.x - start.x), abs(last.y - start.y),
            abs(last.width - start.width),
        )

    # Straight down, far further than a second of calm allows.
    base = CropBox(0.0, 0.0, 0.5, 0.5)
    _, down, _ = travelled(base, CropBox(0.0, 0.5, 0.5, 0.5))
    assert down <= ceiling + 1e-6, down

    # Closing in, ditto. The width may not collapse in one step.
    _, _, closed = travelled(base, CropBox(0.0, 0.0, 0.1, 0.1))
    assert closed <= ceiling + 1e-6, closed

    # A diagonal keeps its direction: clamping each axis on its own would
    # cut the longer component and leave the shorter one, bending the path.
    across, down, _ = travelled(base, CropBox(0.4, 0.2, 0.5, 0.5))
    assert across > 1e-6 and down > 1e-6
    assert abs(across / down - 2.0) < 0.05, (across, down)


def test_a_multi_look_path_cannot_crop_past_the_resolution_budget():
    """The looks builder was given two aspect ratios and no pixels.

    So it cropped as tightly as a framing asked and the pipeline measured the
    upscale afterwards, which is a record of a soft shot rather than a
    prevention of one -- and the shot has been spent by the time it is read.
    """

    from montagewright.reframe import (
        MAX_UPSCALE,
        achieved_upscale,
        build_look_path,
    )

    # A framing asking for a tenth of the frame width, on 4K delivered
    # 1080x1920 -- room to push, but nothing like this much.
    stops = [(0.4, 0.3, 0.5, 0.10), (0.4, 0.7, 0.5, 0.10)]
    common = dict(
        source_aspect=16 / 9, target_aspect=9 / 16, duration_seconds=3.0,
        energy="active",
    )
    pixels = dict(
        source_width=3840, source_height=2160,
        output_width=1080, output_height=1920,
    )

    degradations: list = []
    guarded = build_look_path(
        stops, clip_id="k00", degradations=degradations, **pixels, **common
    )
    tightest = min(guarded.keyframes, key=lambda one: one.crop.width).crop
    assert achieved_upscale(tightest, **pixels) <= MAX_UPSCALE + 1e-3
    assert [one.ladder for one in degradations] == ["reduced_zoom"]

    # And the same call without the pixel dimensions is the old behaviour,
    # which is what this guards against coming back.
    loose = build_look_path(stops, clip_id="k00", **common)
    assert achieved_upscale(
        min(loose.keyframes, key=lambda one: one.crop.width).crop, **pixels
    ) > MAX_UPSCALE

    # A source is not opened out further than its own pixels require: the
    # same framing on the same file delivered smaller keeps more of the push.
    smaller = build_look_path(
        stops, clip_id="k01", degradations=[],
        source_width=3840, source_height=2160,
        output_width=540, output_height=960, **common,
    )
    assert (
        min(one.crop.width for one in smaller.keyframes)
        < min(one.crop.width for one in guarded.keyframes)
    )


def test_the_production_path_passes_the_output_size_to_the_looks_builder():
    """A guard that only covers the builder guards the wrong caller.

    The last refactor's own test named a branch by string and went green
    when the branch was deleted; this asserts the call the pipeline makes.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    call = source[source.index("build_look_path("):]
    call = call[: call.index(")\n")]
    for given in ("source_width=", "source_height=", "output_width=", "output_height="):
        assert given in call, given


def test_an_approving_film_review_does_not_bury_a_shot_that_missed():
    """Two reviewers, two questions, and only one verdict was being read.

    A run finished with three shots that did not do what they planned,
    forty-five degradations and five seconds missing -- and the whole-film
    reviewer said "approve (0 issues)", which returned before the shot
    reviewer's findings were even collected. The replan loop those findings
    exist to drive had never run.
    """

    from montagewright.review import Round, ReviewVerdict, should_continue

    approved = Round(
        index=1,
        verdict=ReviewVerdict(verdict="approve", overall="looks good", issues=[]),
        actionable=[],
    )

    # Nothing missed: an approval still ends the loop.
    assert should_continue([approved]) == (False, "approved")

    # Something missed: the film reviewer cannot see a promise it was never
    # told about, so its approval is not a veto over the shot reviewer.
    keep, why = should_continue([approved], undelivered=3)
    assert keep and "3" in why

    # And the limits still win, or a shot nobody can fix spends the budget
    # one replan at a time.
    from montagewright.review import MAX_ROUNDS

    capped = [approved] * MAX_ROUNDS
    assert should_continue(capped, undelivered=3)[0] is False


def test_the_shot_verdicts_are_collected_before_the_gate_that_reads_them():
    """The ordering is the bug, so the ordering is what is asserted."""

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert source.index("failing = [") < source.rindex("should_continue(")
    # Both gates read it. The one at the top of the loop decides whether a
    # replanned cut is looked at again, and reading a pre-replan verdict
    # there would stop on it.
    assert source.count("undelivered=undelivered") == 2


def test_the_takes_the_direction_ruled_out_are_accounted_for():
    """`set_aside` answered "why is this take missing" for one of two ways.

    The cards' `usable: false` was recorded; the direction's `unusable` list
    was applied to selection and written down nowhere, so a run that removed
    five clips reported none set aside and seventy-four in play.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    ruling = source[source.index('for entry in direction.get("unusable"'):]
    ruling = ruling[: ruling.index("chose = ")]
    assert "set_aside[source_id]" in ruling
    assert "superseded_by" in ruling
    # And it happens after the direction exists, not before it.
    assert source.index("set_aside: dict[str, str] = {}") < source.index(ruling[:40])


def test_the_side_by_side_pane_keeps_its_layout_while_a_take_reloads():
    """The flash was the page relaying out, not the video going black.

    `width:auto` lays a video out from its own intrinsic size, and a video
    whose src is being swapped has none -- it falls back to 300x150 until
    metadata arrives. In a centred flex row that resized both halves and
    re-centred the pair, so the whole picture area jumped and jumped back at
    every cut. It never happened on the finished cut because that is one
    element whose src never changes.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")

    # Each half owns its width whatever the element inside it currently
    # knows about itself.
    assert ".frame.both > .half {" in page
    assert "flex: 1 1 0" in page
    # And is invisible to layout otherwise, so single-pane mode is unchanged.
    assert ".frame > .half { display: contents; }" in page

    # A second element to swap to, so crossing a cut does not call load().
    assert 'id="raw-next"' in page
    assert "function showTake(" in page and "function warmNext(" in page
    # Swapped by exchanging which one answers to the visible id, because
    # everything else on the page looks the take up that way.
    swap = page[page.index("function showTake("):]
    swap = swap[: swap.index("\n}\n")]
    assert "spare.id = 'raw-video'" in swap
    assert "raw.id = 'raw-next'" in swap
    # Only reloads when nothing already has it.
    assert swap.index("spare.getAttribute('src') === want") < swap.index("raw.load()")

    # Listeners are bound to the pane, not to one element: the two swap, and
    # a listener attached to the element would follow the wrong one.
    assert "$('raw-video').addEventListener" not in page
    assert "$('frame').addEventListener" in page


def test_a_crop_path_survives_a_round_trip_through_disk():
    """`write_crops` had no reader, so the record was written and ignored."""

    import tempfile
    from pathlib import Path

    from montagewright.executor import CropBox
    from montagewright.pipeline import read_crops, write_crops
    from montagewright.reframe import CropPath, Keyframe

    was = {
        "k00": CropPath([
            Keyframe(0.0, CropBox(0.10, 0.0, 0.3164, 1.0)),
            Keyframe(1.5, CropBox(0.42, 0.0, 0.3164, 1.0)),
            Keyframe(3.0, CropBox(0.68, 0.0, 0.3164, 1.0)),
        ]),
        "k01": CropPath([Keyframe(0.0, CropBox(0.0, 0.0, 0.5, 0.5))]),
    }
    with tempfile.TemporaryDirectory() as work:
        where = Path(work) / "crops.json"
        write_crops(was, where)
        back = read_crops(where)

    assert sorted(back) == ["k00", "k01"]
    assert [round(k.seconds, 3) for k in back["k00"].keyframes] == [0.0, 1.5, 3.0]
    assert [round(k.crop.x, 5) for k in back["k00"].keyframes] == [0.1, 0.42, 0.68]

    # A run that kept no record says so rather than raising: those exist.
    assert read_crops(Path(work) / "gone.json") == {}


def test_the_timeline_is_written_from_what_the_render_did():
    """FCPXML that disagrees with the film opens as a different cut.

    The exports rebuilt the crop paths from the cards with no client and no
    checkpoint. For a held frame that is the same arithmetic; for anything
    that followed a subject it is a guess, because that path came out of a
    mask propagation nothing there can repeat.
    """

    import inspect

    from montagewright import cli, webapp

    exporting = inspect.getsource(cli.command_timeline)
    assert exporting.index("read_crops(") < exporting.index("follow_subjects(")
    # And says so when there is no record, rather than quietly guessing.
    assert "will differ from the film" in exporting

    # The interface rebuilds per shot, because it also serves a recut and a
    # recut changes lengths -- a stored path is a set of times, so it holds
    # only while the shot is as long as it was.
    rebuilding = inspect.getsource(webapp.create_app)
    rebuilding = rebuilding[rebuilding.index("stored = read_crops("):]
    rebuilding = rebuilding[: rebuilding.index("plan = plan_render(")]
    assert 'abs(covered - float(entry["seconds"])) <= 0.05' in rebuilding
    assert "if stale:" in rebuilding


def test_a_walking_subject_is_followed_rather_than_averaged():
    """The looks path collapsed every observation into a mean.

    Frames are pulled across the shot and the boxes come back one per frame,
    and all of them were reduced to one point before anything downstream saw
    them -- so a subject that walked across the frame was handed on as a
    place in the middle of its own path, and a shot planned to follow it held
    there while the subject left.
    """

    from montagewright.reframe import build_look_path

    walked = [(0.0, 0.20, 0.5), (1.0, 0.50, 0.5), (2.0, 0.80, 0.5)]
    common = dict(
        source_aspect=16 / 9, target_aspect=9 / 16, duration_seconds=2.0,
        energy="dynamic",
    )
    # One look, resting the whole shot, on something that does not stay put.
    stops = [(2.0, 0.50, 0.5, 0.3164)]

    followed = build_look_path(stops, tracks=[walked], **common)
    centres = [k.crop.x + k.crop.width / 2 for k in followed.keyframes]
    assert len(followed.keyframes) >= 3, followed.keyframes
    assert centres == sorted(centres)
    assert max(centres) - min(centres) > 0.25, centres

    # Without the track it is the old behaviour: one place, held.
    held = build_look_path(stops, **common)
    still = {round(k.crop.x, 4) for k in held.keyframes}
    assert len(still) == 1

    # A subject that barely moved is not chased -- below the deadband the
    # frame would only jitter, and holding is what it should look like.
    from montagewright.reframe import DEADBAND

    twitch = [(0.0, 0.50, 0.5), (1.0, 0.50 + DEADBAND / 4, 0.5)]
    steady = build_look_path(stops, tracks=[twitch], **common)
    assert max(k.crop.x for k in steady.keyframes) - min(
        k.crop.x for k in steady.keyframes
    ) < DEADBAND


def test_a_track_is_read_where_the_frame_is_actually_looking():
    """The window a stop occupies is not the whole shot.

    The frame arrives at a subject partway through and leaves before the end,
    so sampling the track into the window by position would run the subject's
    movement at the wrong speed.
    """

    from montagewright.reframe import _across

    track = [(0.0, 0.0, 0.5), (1.0, 0.5, 0.5), (2.0, 1.0, 0.5)]

    # A window over the second half sees the second half of the walk.
    across = _across(track, 1.0, 2.0)
    assert [round(one[0], 3) for one in across] == [1.0, 2.0]
    assert [round(one[1], 3) for one in across] == [0.5, 1.0]

    # Edges are pinned and interior samples kept, so the shape survives.
    whole = _across(track, 0.0, 2.0)
    assert [round(one[0], 3) for one in whole] == [0.0, 1.0, 2.0]

    # A window past either end clamps rather than extrapolating.
    assert round(_across(track, 3.0, 4.0)[0][1], 3) == 1.0


def test_the_production_path_hands_the_tracks_to_the_builder():
    """A guard on the builder alone guards the wrong caller."""

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    call = source[source.index("build_look_path("):]
    assert "tracks=tracks" in call[: call.index(")\n")]

    # And the sampler's timestamps reach the measurement, which is what
    # makes a box tie to a moment. They were being dropped by the caller.
    measuring = inspect.getsource(pipeline._measure_looks)
    assert "frames, times = _sample_frames(" in measuring
    assert "frame_index" in measuring
