import json
from types import SimpleNamespace

import pytest

from montagewright.caption_plan import repair_cues, save_cues
from montagewright.transcript import Line, Word, CharacterTiming


def test_caption_correction_keeps_apple_clock_and_survives_saved_track(tmp_path, monkeypatch):
    import montagewright.planner as planner
    import montagewright.subtitles as subtitles
    calls = []
    def ask(client, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps({'lines':[{'text':'頭髮','speaker':'person'}]}))
    monkeypatch.setattr(planner, 'ask', ask)
    monkeypatch.setattr(subtitles, 'as_cues', lambda lines, *a, **k: lines)
    cache = SimpleNamespace(uri_for=lambda *a, **k: ('file://video',False))
    words = [Word('頭',1,1.4), Word('發',1.4,2)]
    fixed = repair_cues([Line('頭發',1,2)], words, video=tmp_path/'video.mp4', feedback={'verdict':'revise'},
                       aspect='9:16',width=360,height=640,client=object(),cache=cache,ledger=None)
    assert fixed[0].text == '頭髮'
    assert [(m.starts_seconds,m.ends_seconds) for m in fixed[0].timed_text] == [(1,1.4),(1.4,2)]
    path = tmp_path/'subtitles.json'
    save_cues(path,fixed)
    marks = [CharacterTiming(**one) for one in json.loads(path.read_text())[0]['timed_text']]
    assert marks[-1].text == '髮' and not marks[-1].measured
    assert calls[0]['input'][0]['type'] == 'video'


def test_caption_repair_refuses_to_invent_missing_time(tmp_path, monkeypatch):
    import montagewright.planner as planner
    monkeypatch.setattr(planner,'ask',lambda *a,**k: SimpleNamespace(output_text=json.dumps({'lines':[{'text':'新增的話','speaker':''}]})))
    cache = SimpleNamespace(uri_for=lambda *a,**k: ('file://video',False))
    with pytest.raises(ValueError,match='Apple timing'):
        repair_cues([],[],video=tmp_path/'video.mp4',feedback={},aspect='1:1',width=480,height=480,client=object(),cache=cache,ledger=None)
