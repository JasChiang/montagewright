"""Duration contracts must survive entry points, source clocks and release."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from montagewright import cli, webapp
from montagewright.job import Delivery, EditJob, load_job, write_job
from montagewright.planner import duration_padding_disagreements, repeated_image_clip_indices
from montagewright.release import technical_qc_faults


@pytest.mark.parametrize("mode,bounds", [(None, (58, 62)), ("approx", (58, 62)), ("at_most", (58, 60))])
def test_cli_job_and_web_resolve_the_same_sixty_second_contract(tmp_path, monkeypatch, mode, bounds):
    data = {"seconds": 60, **({"duration_mode": mode} if mode else {})}
    expected = Delivery.model_validate(data)
    assert expected.duration_mode == "range"
    assert (expected.minimum_seconds, expected.maximum_seconds) == bounds
    path = write_job(tmp_path / "job.json", EditJob(delivery=expected))
    assert load_job(path).delivery == expected

    captured = []
    monkeypatch.setattr(cli, "command_render", lambda args: captured.append(args) or 0)
    arguments = ["render", str(tmp_path), "--output", str(tmp_path / "out"), "--seconds", "60"]
    if mode:
        arguments.extend(["--duration-mode", mode])
    assert cli.main(arguments) == 0
    resolved = cli._resolved_job(captured[-1], tmp_path, tmp_path / "out")
    assert resolved.delivery.minimum_seconds == bounds[0]
    assert resolved.delivery.maximum_seconds == bounds[1]

    import io
    class Finished:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO("")
        def wait(self):
            return 0
        def poll(self):
            return 0
    (tmp_path / "take.mp4").touch()
    monkeypatch.setattr(webapp, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(webapp.subprocess, "Popen", Finished)
    response = TestClient(webapp.create_app()).post("/api/runs", data={
        "source_path": str(tmp_path), **data, "review": "false",
        "delivery_variants_json": json.dumps([{"variant_id": "square", "delivery": {**data, "aspect": "1:1"}}]),
    })
    assert response.status_code == 200, response.text
    saved = load_job(Path(response.json()["job"]))
    for delivery in (saved.delivery, saved.variants[1].delivery):
        assert (delivery.minimum_seconds, delivery.maximum_seconds) == bounds


def test_mode_override_clears_persisted_range_bounds(tmp_path, monkeypatch):
    path = write_job(tmp_path / "job.json", EditJob(rushes=str(tmp_path), output=str(tmp_path / "out"), delivery=Delivery(seconds=60)))
    captured = []
    monkeypatch.setattr(cli, "command_render", lambda args: captured.append(args) or 0)
    assert cli.main(["render", "--job", str(path), "--seconds", "30", "--duration-mode", "at_most"]) == 0
    delivery = cli._resolved_job(captured[-1], tmp_path, tmp_path / "out").delivery
    assert (delivery.minimum_seconds, delivery.maximum_seconds) == (28, 30)


@pytest.mark.parametrize("duration,allowed", [(57.9, False), (58, True), (59, True), (61, True), (62, True), (62.1, False)])
def test_encoded_duration_gate_uses_the_resolved_range(tmp_path, monkeypatch, duration, allowed):
    # Mock only the media probe, not the release rule. Other stream faults
    # are independent of these boundary assertions.
    from montagewright import release
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stderr="", stdout=json.dumps({"streams": [], "format": {"duration": duration}}),
    ))
    faults = technical_qc_faults(tmp_path / "movie.mp4", EditJob(delivery=Delivery(seconds=60)))
    assert (not any("ranged delivery" in fault for fault in faults)) == allowed


def test_repeats_use_source_clock_across_span_ids_and_playback_rates():
    shots = [
        {"source_id": "a", "span_id": "a:s0", "start_seconds": 0, "seconds_needed": 3, "speed": 2},
        {"source_id": "a", "span_id": "a:s1", "start_seconds": 5, "seconds_needed": 2},
    ]
    assert repeated_image_clip_indices(shots) == {1}
    shots[1].update(intentional_repeat=True, intentional_repeat_reason="為了湊足秒數")
    assert repeated_image_clip_indices(shots) == {1}
    assert duration_padding_disagreements(shots)
    shots[1]["intentional_repeat_reason"] = "bookend on the hero shot"
    assert repeated_image_clip_indices(shots) == set()
    shots[1].pop("intentional_repeat")
    shots[0].update(intentional_repeat=True, intentional_repeat_reason="bookend")
    assert repeated_image_clip_indices(shots) == {1}  # An earlier flag cannot excuse a later repeat.
    shots[0]["speed"] = 0.5
    assert repeated_image_clip_indices(shots) == set()


def test_explicit_slow_motion_padding_is_rejected_but_action_detail_is_allowed():
    assert duration_padding_disagreements([{"speed": .5, "why": "拖慢以湊滿秒數"}])
    assert not duration_padding_disagreements([{"speed": .5, "why": "放慢鉸鏈動作以看清結構，不為湊秒數"}])


def test_no_duration_stays_automatic_and_explicit_exact_stays_exact():
    assert Delivery().duration_mode == "preferred"
    assert Delivery(seconds=60, duration_mode="exact").minimum_seconds is None
    with pytest.raises(ValueError, match="positive seconds"):
        Delivery(duration_mode="at_most")
