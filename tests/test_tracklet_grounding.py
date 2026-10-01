from pathlib import Path
from types import SimpleNamespace

import pytest

from montagewright import tracklet_grounding as tg


def _det(sample, box, score=0.5):
    return tg.Detection(sample, box, score, "a phone")


def test_link_keeps_two_side_by_side_units_apart_and_bridges_a_short_miss():
    left = (0.10, 0.40, 0.30, 0.80)
    right = (0.60, 0.40, 0.80, 0.80)
    frames = [
        [_det(0, left), _det(0, right)],
        [_det(1, right)],  # the left one is missed for one sample
        [_det(2, left), _det(2, right)],
        [_det(3, left), _det(3, right)],
    ]
    tracklets = tg.link(frames)
    assert len(tracklets) == 2
    by_x = sorted(tracklets, key=lambda one: one.detections[0].box[0])
    assert sorted(by_x[0].detections) == [0, 2, 3]
    # The gap is interpolated, so the crop does not lose the unit for a beat.
    assert by_x[0].box_at(1) == pytest.approx(left)
    assert sorted(by_x[1].detections) == [0, 1, 2, 3]


def test_nms_merges_one_thing_proposed_under_two_phrases():
    box = (0.2, 0.2, 0.4, 0.6)
    kept = tg._nms([_det(0, box, 0.4), _det(0, (0.21, 0.2, 0.4, 0.61), 0.3)])
    assert len(kept) == 1 and kept[0].score == 0.4


def test_pick_answers_are_string_numbers_and_a_lookalike_is_never_required():
    decided = tg.validate_pick(
        {
            "assignments": [
                {"number": "1", "verdict": "target", "evidence": "two rings"},
                {"number": "#2", "verdict": "other_product", "evidence": "flip"},
            ],
            "required_numbers": ["1", "2"],
            "target_visible_without_outline": False,
            "note": "",
        },
        [1, 2, 3],
    )
    assert decided["targets"] == [1]
    assert decided["required"] == [1]
    # #3 was never answered: uncertain, not a failure of the whole shot.
    assert decided["verdicts"][3]["verdict"] == "uncertain"


def test_pick_rejects_a_number_that_was_never_proposed():
    with pytest.raises(tg.TrackletGroundingError):
        tg.validate_pick(
            {"assignments": [{"number": "9", "verdict": "target",
                              "evidence": ""}],
             "required_numbers": [], "target_visible_without_outline": False,
             "note": ""},
            [1, 2],
        )


def test_union_box_frames_the_whole_group():
    assert tg.union_box([(0.1, 0.3, 0.2, 0.5), (0.5, 0.2, 0.7, 0.6)]) == (
        0.1, 0.2, 0.7, 0.6,
    )


def test_clientless_cut_without_a_remembered_pick_never_loads_the_detector(
    tmp_path, monkeypatch,
):
    source = tmp_path / "take.mp4"
    source.write_bytes(b"not decoded")
    monkeypatch.setattr(
        tg, "_detector",
        lambda: (_ for _ in ()).throw(AssertionError("detector loaded")),
    )
    with pytest.raises(tg.TrackletGroundingError, match="no client"):
        tg.ground_cut(
            source, 0.0, 2.0, "device.x", spec=SimpleNamespace(), intent="",
            client=None, cache=None, ledger=None, work=tmp_path,
            memory=tmp_path / "memory",
        )


def test_remembered_pick_is_replayed_without_a_client(tmp_path):
    source = tmp_path / "take.mp4"
    source.write_bytes(b"x")
    spec = SimpleNamespace()
    spec_sha = __import__("hashlib").sha256(repr(spec).encode()).hexdigest()
    key = tg.cache_key(source, 0.0, 2.0, "device.x", spec_sha, "one unit")
    memory = tmp_path / "memory"
    memory.mkdir()
    saved = {"status": "target_located", "samples": [], "times": []}
    (memory / f"tracklet-{key[:24]}.json").write_text(
        __import__("json").dumps({"key": key, "result": saved})
    )
    result = tg.ground_cut(
        source, 0.0, 2.0, "device.x", spec=spec, intent="one unit",
        client=None, cache=None, ledger=None, work=tmp_path, memory=memory,
    )
    assert result["status"] == "target_located"


def _fake_cut(width, required, sheet=None):
    samples = [
        {"at": 0.5 * i, "present": True,
         "box": [0.25, 0.3, 0.25 + width, 0.7], "members": len(required),
         "of": len(required)}
        for i in range(6)
    ]
    return {
        "status": "target_located", "times": [s["at"] for s in samples],
        "samples": samples, "tracklets": [],
        "pick": {"required": required, "targets": required,
                 "unboxed_target": False, "note": "",
                 "sheet": str(sheet) if sheet else None},
        "excluded_instances": [], "usage": None,
    }


def _call(monkeypatch, tmp_path, result, **kwargs):
    from montagewright import pipeline
    from montagewright.executor import Source

    monkeypatch.setattr(tg, "ground_cut", lambda *a, **k: dict(result))
    source = Source("C1", tmp_path / "C1.mp4", 10.0, 1920, 1080)
    clip = SimpleNamespace(clip_id="k00", approx_in_seconds=0.0,
                           approx_out_seconds=3.0)
    report = pipeline.Report()
    return pipeline._tracklet_subject_samples(
        source, clip, "device.x", spec=SimpleNamespace(), client=object(),
        upload_cache=None, report=report, work=tmp_path,
        output=tmp_path / "out", memory=None, intent="two side by side",
        target_aspect=9 / 16, **kwargs,
    ), report


def test_a_required_pair_wider_than_a_vertical_crop_goes_back_to_the_planner(
    monkeypatch, tmp_path,
):
    from montagewright.pipeline import ReferenceGroupDoesNotFit

    # 9:16 out of 16:9 keeps 0.316 of the width; the pair spans 0.50.
    with pytest.raises(ReferenceGroupDoesNotFit, match="canvas_mode fit"):
        _call(monkeypatch, tmp_path, _fake_cut(0.50, [1, 2]))


def test_a_pair_that_fits_or_a_look_that_reads_across_is_delivered(
    monkeypatch, tmp_path,
):
    (boxes, _, _), _ = _call(monkeypatch, tmp_path, _fake_cut(0.25, [1, 2]))
    assert len(boxes) == 6
    (boxes, _, _), _ = _call(
        monkeypatch, tmp_path, _fake_cut(0.50, [1, 2]), reads_across=True,
    )
    assert len(boxes) == 6


def test_the_contact_sheet_is_kept_with_the_output(monkeypatch, tmp_path):
    sheet = tmp_path / "scratch-sheet.jpg"
    sheet.write_bytes(b"jpeg")
    _, report = _call(monkeypatch, tmp_path, _fake_cut(0.2, [1], sheet))
    kept = tmp_path / "out" / "k00-device.x-tracklets.jpg"
    assert kept.read_bytes() == b"jpeg"
    record = report.reference_grounding["k00"]
    assert record["contact_sheet"] == str(kept)


def test_the_pick_is_told_the_shot_purpose_not_only_where_it_settles():
    from montagewright.pipeline import _shot_intent

    look = SimpleNamespace(at="中間的淡紫色機身", entity_id="device.x",
                           co_visible_entity_ids=())
    clip = SimpleNamespace(
        music_sync=SimpleNamespace(rhythm_reason="收尾回到三色並排陳列的 hero shot"),
        story_point="結尾",
    )
    intent = _shot_intent(clip, SimpleNamespace(looks=[look]), "device.x")
    assert "中間的淡紫色機身" in intent
    assert "三色並排" in intent and "結尾" in intent
