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
