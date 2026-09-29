import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from montagewright.editor_workspace import EditorWorkspace, gather_evidence, inspected_selection_faults, record_render
from montagewright.planner import MaterialItem
from montagewright.spans import Span


@pytest.fixture
def workspace(tmp_path):
    video = tmp_path / "rush.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=5", "-c:v", "libx264", str(video)], check=True)
    material = [MaterialItem("a", 5, "phone", proxy=video,
                            spans=(Span("a:s00", "a", 0, 5),)),
                MaterialItem("b", 5, "other take", proxy=video,
                            spans=(Span("b:s00", "b", 0, 5),))]
    return EditorWorkspace(tmp_path / "editor", material, brief="keep both phones",
        direction={"aspect": "9:16"}, selection={"shots": [{"source_id": "a", "why": "compare"}]})


def test_preview_preserves_source_clock_and_canvas(workspace):
    result = workspace.inspect({"operation": "preview_framing", "source_id": "a",
                                "start": 1, "end": 3, "framing": "fit"})
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", result["path"]]))
    assert (info["streams"][0]["width"], info["streams"][0]["height"]) == (360, 640)
    assert float(info["streams"][0]["duration"]) == pytest.approx(2)
    assert result["start"] == 1 and result["clock"] == "source"
    assert "padding" in result["tradeoffs"]
    assert workspace.inspect({"operation": "preview_framing", "source_id": "a", "start": 1,
                              "end": 3, "framing": "fit"})["sha256"] == result["sha256"]


def test_tool_loop_retains_full_context_and_can_find_alternate(workspace, monkeypatch):
    import montagewright.planner as planner
    calls = []
    responses = iter([
        {"ready": False, "reason": "look at alternative", "requests": [
            {"operation": "inspect_source", "source_id": "b", "start": 1, "end": 3, "framing": "source"}]},
        {"ready": True, "reason": "choose b", "requests": []},
    ])
    def ask(client, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(next(responses)))
    monkeypatch.setattr(planner, "ask", ask)
    cache = SimpleNamespace(uri_for=lambda path, *_a, **_k: (str(path), False))
    parts, results = gather_evidence(workspace, client=object(), cache=cache, ledger=None)
    assert len(calls) == 2 and len(results) == 1
    for call in calls:
        context = json.loads(call["input"][0]["text"])["context"]
        assert context["brief"] == "keep both phones"
        assert [m["source_id"] for m in context["materials"]] == ["a", "b"]
    assert any(p["type"] == "video" for p in calls[1]["input"])
    assert not inspected_selection_faults([{"source_id": "b", "start_seconds": 1, "seconds_needed": 2}], results)
    assert inspected_selection_faults([{"source_id": "b", "start_seconds": 0, "seconds_needed": 4}], results)


def test_invalid_source_and_clock_never_reach_ffmpeg(workspace):
    with pytest.raises(ValueError, match="material index"):
        workspace.inspect({"operation": "inspect_source", "source_id": "/etc/passwd", "start": 0, "end": 2})
    with pytest.raises(ValueError, match="interval"):
        workspace.inspect({"operation": "inspect_source", "source_id": "a", "start": float("nan"), "end": 2})


def test_render_revisions_preserve_previous_and_detect_no_change(workspace):
    args = dict(material=list(workspace.material.values()), brief="test", direction={"aspect": "9:16"},
                preview=workspace.material["a"].proxy, timeline={"shots": []})
    first = record_render(workspace.root, selection={"why": "first"}, **args)
    second = record_render(workspace.root, selection={"why": "renamed only"}, **args)
    assert first["changed"] and not second["changed"]
    assert first["revision"] != second["revision"]
    assert Path(first["preview"]).exists() and Path(second["preview"]).exists()


def test_tool_evidence_limit_blocks_extra_video_before_render(workspace, monkeypatch):
    import montagewright.planner as planner
    calls = []
    def ask(client, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps({'ready':True, 'reason':'enough', 'requests':[]}))
    monkeypatch.setattr(planner, 'ask', ask)
    cache = SimpleNamespace(uri_for=lambda path, *_a, **_k: (str(path), False))
    parts, results = gather_evidence(workspace, client=object(), cache=cache, ledger=None, max_evidence_seconds=2,
        initial=[{'operation':'inspect_source','source_id':'a','start':0,'end':2},
                 {'operation':'inspect_source','source_id':'b','start':0,'end':2}])
    assert len(results) == 1
    assert 'evidence limit' in json.loads(parts[0]['text'])['tool_history'][0]['error']


@pytest.mark.parametrize('size', [(640,360), (360,640), (480,480)])
def test_fit_survives_compiler_and_real_renderer(workspace, tmp_path, size):
    from montagewright.schema import Clip, EDL
    from montagewright.pipeline import probe
    from montagewright.executor import plan_render
    from montagewright.renderer import render
    source = probe('a', workspace.material['a'].proxy)
    edl = EDL(project_id='fit', clips=[Clip(clip_id='k00',source_id='a',approx_in_seconds=1,approx_out_seconds=2,canvas_mode='fit')])
    plan = plan_render(edl, {'a':source}, target_aspect=size[0]/size[1], output_size=size)
    assert plan.segments[0].crop is None and plan.segments[0].crop_path is None
    assert plan.segments[0].canvas_mode == 'fit'
    made = render(plan, tmp_path / 'out')
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(made.deliverable)]))
    stream = next(s for s in info['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height']) == size
    assert float(stream['duration']) == pytest.approx(1, abs=.04)
    assert stream['color_space'] == 'bt709'


def test_full_frame_fit_cannot_inherit_identity_proof_for_a_crop(workspace):
    from montagewright.schema import Clip, EDL
    from montagewright.pipeline import Report, follow_subjects, probe
    clip = Clip(clip_id='k00',source_id='a',approx_in_seconds=0,approx_out_seconds=1,canvas_mode='fit')
    report = Report()
    paths = follow_subjects(EDL(project_id='identity-fit',clips=[clip]), {'a':probe('a',workspace.material['a'].proxy)},
                           target_aspect=9/16,report=report,grounding_spec=object())
    assert paths == {} and report.static_shots == 1
    assert any('full-frame identity' in x for x in report.plan_disagreements)


def test_approved_picture_sound_and_brief_reuse_review_at_zero_budget(workspace, monkeypatch):
    import montagewright.review as review
    import montagewright.renderer as renderer
    from montagewright.cost import Ledger
    calls = []
    def ask(client, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps({'verdict':'approve','overall':'all three visible','issues':[]}))
    monkeypatch.setattr(review,'ask',ask)
    monkeypatch.setattr(renderer,'_peak',lambda path: -100)
    cache = SimpleNamespace(uri_for=lambda *a,**k: ('file://video',False))
    arguments = dict(preview=workspace.material['a'].proxy,brief='keep all three',direction='simple',client=object(),cache=cache,
                     complete_pass=True,ledger=Ledger(cap_usd=0,journal_path=workspace.root/'spend.jsonl'))
    first = review.review_cut(**arguments)
    second = review.review_cut(**arguments,already=[review.Round(index=1,verdict=first,actionable=())])
    assert first == second and len(calls) == 1
    review.review_cut(**{**arguments,'brief':'keep only one'})
    assert len(calls) == 2
    assert review.should_continue([review.Round(index=1,verdict=first,actionable=())],ledger=arguments['ledger']) == (False,'approved')


def test_agentic_source_and_static_preview_keep_video_evidence_each_turn(workspace,monkeypatch):
    import montagewright.planner as planner
    calls=[]
    answers=iter([{'ready':False,'reason':'compare framing','requests':[{
        'operation':'preview_framing','source_id':'a','start':0,'end':2,'framing':'fit'}]},
        {'ready':True,'reason':'seen both','requests':[]}])
    def ask(client,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(next(answers)),steps=[
            {'type':'processing_call','id':'p1'}, {'type':'processing_result','call_id':'p1'}])
    monkeypatch.setattr(planner,'ask',ask)
    cache=SimpleNamespace(uri_for=lambda p,*a,**k:(str(p),False))
    parts,results=gather_evidence(workspace,client=object(),cache=cache,ledger=None,
        initial=[{'operation':'inspect_source','source_id':'a','start':0,'end':2,'framing':'source'}])
    first=[p for p in calls[0]['input'] if p['type']=='video']
    second=[p for p in calls[1]['input'] if p['type']=='video']
    assert first[0]['processing']=='agentic'
    assert len(second)==2 and second[0]['processing']=='agentic'
    assert second[1]['processing']!= 'agentic'
    assert all(r.get('source_sha256') for r in results)
    events=[json.loads(p.read_text()) for p in (workspace.directory/'events').glob('*.json')]
    assert any(e['kind']=='video_processing' and len(e['payload']['steps'])==2 for e in events)
