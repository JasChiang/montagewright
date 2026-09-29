import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from montagewright.cost import Ledger, BudgetSpent


def test_paid_result_survives_parser_failure_and_zero_budget_resume(tmp_path):
    from montagewright.planner import ask, _parse
    class Provider:
        calls = 0
        def create(self, **kw):
            self.calls += 1
            return SimpleNamespace(status="completed", output_text="broken JSON",
                usage={"total_input_tokens": 100, "total_output_tokens": 50})
    provider = Provider()
    client = SimpleNamespace(interactions=provider)
    journal = tmp_path / "spend.jsonl"
    request = dict(model="gemini-3.8-flash", input=[{"type":"text", "text":"test"}],
                   generation_config={"max_output_tokens":100})
    first = ask(client, ledger=Ledger(1, journal_path=journal), budget_stage="transcript", **request)
    with pytest.raises(ValueError):
        json.loads(first.output_text)
    before = journal.read_text()
    resumed = ask(client, ledger=Ledger(0, journal_path=journal, cumulative_budget=True),
                  budget_stage="transcript", **request)
    assert resumed.output_text == first.output_text
    assert provider.calls == 1 and journal.read_text() == before


def test_media_uri_rotation_keeps_checkpoint_identity(tmp_path):
    from montagewright.checkpoints import response_path
    ledger = Ledger(1, journal_path=tmp_path / "spend.jsonl")
    a = response_path(ledger, "cards", {"input":[{"type":"video", "uri":"old"}]},
                      SimpleNamespace(entries={"abc":{"uri":"old"}}))
    b = response_path(ledger, "cards", {"input":[{"type":"video", "uri":"new"}]},
                      SimpleNamespace(entries={"abc":{"uri":"new"}}))
    assert a == b


def test_completion_reserve_cannot_be_spent_by_an_earlier_stage():
    ledger = Ledger(1, completion_reserve={"review": .8})
    with pytest.raises(BudgetSpent):
        ledger.reserve("clip_cards", input_tokens=0, max_output_tokens=100000)
    receipt = ledger.reserve("review", input_tokens=0, max_output_tokens=100000)
    assert receipt


def test_semantic_caption_split_preserves_text_and_apple_clock(monkeypatch):
    from montagewright.caption_plan import semantic_cues
    from montagewright.transcript import Line, CharacterTiming
    import montagewright.planner as planner
    import montagewright.subtitles as subtitles
    monkeypatch.setattr(subtitles, "_width", lambda text, face: len(text)*10)
    requests = []
    def answer(client, **kw):
        requests.append(kw)
        return SimpleNamespace(output_text=json.dumps({"lines":[
            {"index":0,"pieces":["今天介紹", "新的手機"]}]}))
    monkeypatch.setattr(planner, "ask", answer)
    text = "今天介紹新的手機"
    clock = tuple(CharacterTiming(c,i*.3,(i+1)*.3,True) for i,c in enumerate(text))
    line = Line(text,0,2.4,timed_text=clock)
    cues = semantic_cues([line],aspect="9:16",face=None,room=40,client=object(),ledger=None)
    assert [c.text for c in cues] == ["今天介紹", "新的手機"]
    assert cues[1].starts_seconds == 1.2
    assert "".join(c.text for cue in cues for c in cue.timed_text) == text
    assert "9:16" in requests[0]["input"][0]["text"]


@pytest.mark.parametrize("kind", ["dissolve", "dip_black"])
def test_transition_renders_real_intermediate_frames_without_changing_clock(tmp_path, monkeypatch, kind):
    from montagewright.executor import Source, Segment, RenderPlan
    from montagewright.renderer import render
    import montagewright.renderer as renderer
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")
    sources = []
    for colour in ("red", "blue"):
        path = tmp_path / f"{colour}.mp4"
        subprocess.run(["ffmpeg","-v","error","-y","-f","lavfi","-i",
            f"color={colour}:s=160x90:r=30:d=3","-c:v","libx264",str(path)],check=True)
        sources.append(Source(colour,path,3,160,90))
    plan = RenderPlan("transition", [Segment("a", sources[0],.5,1.5),
        Segment("b", sources[1],.5,1.5,transition_in=kind,transition_seconds=.4)],
        output_size=(160,90))
    result = render(plan,tmp_path/"out")
    info = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=nb_frames,duration","-of","json",str(result.deliverable)]))["streams"][0]
    assert int(info["nb_frames"]) == 60
    assert float(info["duration"]) == pytest.approx(2,abs=.001)
    pixel = subprocess.check_output(["ffmpeg","-v","error","-ss","1","-i",str(result.deliverable),
        "-frames:v","1","-vf","scale=1:1","-pix_fmt","rgb24","-f","rawvideo","-"])
    r,g,b = pixel[:3]
    if kind == "dissolve":
        assert 50 < r < 205 and 50 < b < 205
    else:
        assert max(r,g,b) < 35


def test_web_proposal_and_budget_resume_are_connected(tmp_path, monkeypatch):
    import io
    import montagewright.webapp as web
    from fastapi.testclient import TestClient
    class Process:
        def __init__(self,*args,**kw): self.stdout=io.StringIO("")
        def poll(self): return 0
        def wait(self): return 0
    monkeypatch.setattr(web,"RUNS_ROOT",tmp_path/"runs")
    monkeypatch.setattr(web.subprocess,"Popen",Process)
    web.RUNS.clear()
    rushes=tmp_path/"rushes";rushes.mkdir();(rushes/"a.mp4").touch()
    client=TestClient(web.create_app())
    response=client.post("/api/runs",data={"source_path":str(rushes),"mode":"propose",
        "aspect":"auto","budget":"8","target_budget":"5"})
    assert response.status_code==200,response.text
    value=response.json();job=json.loads(Path(value["job"]).read_text())
    assert job["run"]["mode"]=="propose" and job["run"]["target_budget_usd"]==5
    run=web.RUNS[value["run_id"]];run.output.mkdir(exist_ok=True)
    (run.output/"proposal.md").write_text("# 方案")
    assert client.get(f"/api/runs/{run.run_id}/proposal").status_code==200
    resumed=client.post(f"/api/runs/{run.run_id}/resume",data={"mode":"edit","budget":"10"})
    assert resumed.status_code==200
    assert run.command[-4:]==["--budget","10.0","--mode","edit"]


def test_saved_semantic_cues_survive_web_reload_with_character_clock(tmp_path):
    from montagewright.webapp import _subtitle_lines
    work = tmp_path / 'work'; work.mkdir()
    (work / 'subtitles.json').write_text(json.dumps([{
        'at':1, 'until':2, 'text':'手機', 'timing_source':'apple_audio_time_range',
        'timed_text':[{'text':'手','starts_seconds':1,'ends_seconds':1.5,'measured':True},
                      {'text':'機','starts_seconds':1.5,'ends_seconds':2,'measured':True}]
    }]))
    cues = _subtitle_lines(SimpleNamespace(output=tmp_path))
    assert cues[0].text == '手機' and cues[0].timed_text[1].starts_seconds == 1.5


def test_default_upload_cache_falls_back_before_upload(tmp_path, monkeypatch):
    import montagewright.cli as cli
    occupied = tmp_path / 'not-a-directory'; occupied.write_text('existing')
    monkeypatch.setattr(cli, 'default_cache_path', lambda: occupied / 'uploads.json')
    cache = cli._writable_upload_cache(None, tmp_path / 'out')
    assert cache.path == tmp_path / 'out/work/uploads.json'
    assert cache.path.exists() and occupied.read_text() == 'existing'
    with pytest.raises(OSError):
        cli._writable_upload_cache(occupied / 'explicit.json', tmp_path / 'out')
