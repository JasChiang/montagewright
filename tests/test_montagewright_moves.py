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
    for branch in ('if (\n                move == "pan"', 'if move in {"push_in"'):
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
    assert _subject_line(small, WIDE, TALL) == "硬幣"


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
    assert "本機不會替你補時間" in menu
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

    described = _selection_schema(["C1"])["properties"]["shots"]["items"][
        "properties"
    ]["must_be_whole"]["description"]
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
        try:
            build_library(
                {"a": Path("a.mp4")}, Path(work), client=Refuses()
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

    page = PAGE.read_text(encoding="utf-8")
    for shown in ("素材", "運鏡", "主體", "為什麼用這顆", "降級", "逐顆驗收"):
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


def test_no_music_means_no_rhythm_pass() -> None:
    """Its whole job is reconciling a length against a track.

    Called with no grid it went looking for section boundaries in music that
    was never supplied, and every length was already the one selection asked
    for -- so there was nothing to reconcile even if it had survived.
    """

    import inspect

    from montagewright import pipeline

    assert "decide_rhythm_first and grid is not None" in inspect.getsource(
        pipeline.run
    )


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
    assert "先前的剪輯" in page
    assert "聽到了什麼" in page


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
