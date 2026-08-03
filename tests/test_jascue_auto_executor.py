"""The execution layer's contract: it carries out plans, it does not judge them."""

from __future__ import annotations

from pathlib import Path

import pytest

from jascue_auto.executor import (
    CROP_MARGIN,
    CropBox,
    MissingSource,
    Source,
    plan_render,
)
from jascue_auto.schema import EDL, Clip, Reframe, Subject

UHD = Source("uhd", Path("uhd.mp4"), duration_seconds=10.0, width=3840, height=2160)
PORTRAIT = 9 / 16


def _clip(clip_id: str, start: float, end: float, **kwargs: object) -> Clip:
    return Clip(
        clip_id=clip_id,
        source_id="uhd",
        approx_in_seconds=start,
        approx_out_seconds=end,
        **kwargs,
    )


def _edl(*clips: Clip) -> EDL:
    return EDL(project_id="test", clips=list(clips))


def test_every_clip_becomes_a_segment() -> None:
    """The whole point. A plan of five beats renders five beats."""

    edl = _edl(*[_clip(f"c{index}", index, index + 1) for index in range(5)])
    plan = plan_render(edl, {"uhd": UHD})
    assert [segment.clip_id for segment in plan.segments] == [
        "c0", "c1", "c2", "c3", "c4"
    ]


def test_a_window_past_the_end_still_renders() -> None:
    """An impossible trim degrades to a real one; it does not vanish.

    A missing beat leaves a hole nobody can review. A short one can be seen,
    argued with, and replanned.
    """

    plan = plan_render(_edl(_clip("late", 50.0, 60.0)), {"uhd": UHD})

    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.duration_seconds > 0.0
    assert segment.out_seconds <= UHD.duration_seconds

    step = next(
        step for step in plan.degradations if step.clip_id == "late"
    )
    assert step.ladder_other == "trim_window_clamped_to_source"
    # The numbers behind the claim travel with it.
    assert step.measured["requested_out"] == 60.0
    assert step.measured["source_duration"] == 10.0


def test_a_partly_long_window_is_trimmed_without_being_called_a_degradation() -> None:
    """Running past the end by a little is a note, not a rung down."""

    plan = plan_render(_edl(_clip("over", 8.0, 20.0)), {"uhd": UHD})

    assert plan.segments[0].out_seconds == 10.0
    assert not plan.degradations
    assert any("out-point trimmed" in note for note in plan.notes)


def test_a_subject_anchors_the_crop_off_centre() -> None:
    """A subject on the left is not framed by cropping to the middle."""

    def crop_x(position: str) -> int:
        clip = _clip(
            position,
            0.0,
            2.0,
            reframe=Reframe(
                subject=Subject(
                    description="the left, grey handset",
                    coarse_position=position,
                )
            ),
        )
        plan = plan_render(
            _edl(clip), {"uhd": UHD}, target_aspect=PORTRAIT
        )
        assert plan.segments[0].crop is not None
        return plan.segments[0].crop.to_pixels(3840, 2160)[0]

    assert crop_x("mid_left") < crop_x("center") < crop_x("mid_right")


def test_anchoring_keeps_a_margin_at_the_extremes() -> None:
    """An edge subject does not put the frame edge through itself."""

    clip = _clip(
        "edge",
        0.0,
        2.0,
        reframe=Reframe(
            subject=Subject(
                description="the leftmost handset", coarse_position="mid_left"
            )
        ),
    )
    plan = plan_render(_edl(clip), {"uhd": UHD}, target_aspect=PORTRAIT)
    crop = plan.segments[0].crop
    assert crop is not None
    assert crop.x > 0.0, "a left anchor still keeps some frame to its left"
    assert crop.x <= (1.0 - crop.width) * CROP_MARGIN + 1e-9


def test_a_stated_subject_is_not_a_degradation() -> None:
    """Doing what the plan asked is not a fall back."""

    clip = _clip(
        "framed",
        0.0,
        2.0,
        reframe=Reframe(
            subject=Subject(description="the watch face", coarse_position="center")
        ),
    )
    plan = plan_render(_edl(clip), {"uhd": UHD}, target_aspect=PORTRAIT)
    assert not plan.degradations


def test_a_missing_subject_centres_and_says_so() -> None:
    plan = plan_render(
        _edl(_clip("bare", 0.0, 2.0)), {"uhd": UHD}, target_aspect=PORTRAIT
    )
    step = next(step for step in plan.degradations if step.clip_id == "bare")
    assert step.ladder == "center_crop"
    assert "no subject" in step.trigger


def test_a_matching_aspect_is_not_cropped() -> None:
    plan = plan_render(
        _edl(_clip("wide", 0.0, 2.0)), {"uhd": UHD}, target_aspect=16 / 9
    )
    assert plan.segments[0].crop is None


def test_a_missing_source_is_the_one_refusal() -> None:
    """Not a judgement about the material -- the file simply is not there."""

    edl = EDL(
        project_id="test",
        clips=[
            Clip(
                clip_id="orphan",
                source_id="never_supplied",
                approx_in_seconds=0.0,
                approx_out_seconds=1.0,
            )
        ],
    )
    with pytest.raises(MissingSource, match="never_supplied"):
        plan_render(edl, {"uhd": UHD})


def test_the_plan_reports_its_own_cost() -> None:
    plan = plan_render(
        _edl(_clip("a", 0.0, 2.0), _clip("b", 50.0, 60.0)), {"uhd": UHD}
    )
    assert plan.duration_seconds == pytest.approx(
        sum(segment.duration_seconds for segment in plan.segments)
    )
    assert plan.degraded_clip_ids == {"b"}


class TestCropBox:
    def test_pixel_origin_of_zero_stays_zero(self) -> None:
        assert CropBox(0.0, 0.0, 1.0, 0.5625).to_pixels(3840, 2160)[:2] == (0, 0)

    def test_extent_never_runs_off_the_frame(self) -> None:
        x, y, width, height = CropBox(0.9, 0.9, 0.2, 0.2).to_pixels(3840, 2160)
        assert x + width <= 3840
        assert y + height <= 2160

    def test_pixel_values_are_even(self) -> None:
        for value in CropBox(0.3333, 0.1111, 0.4444, 0.7777).to_pixels(1920, 1080):
            assert value % 2 == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.5},
            {"x": 0.0, "y": 0.0, "width": 1.5, "height": 0.5},
            {"x": -0.1, "y": 0.0, "width": 0.5, "height": 0.5},
        ],
    )
    def test_nonsense_extents_are_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            CropBox(**kwargs)
