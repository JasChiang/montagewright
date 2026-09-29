import copy
from types import SimpleNamespace
import pytest
from montagewright.grounding_validation import validate_composition, visible_fraction
from montagewright.grounding_recovery import GroundingBlocked, merge_replacements, recover


def fixture():
    contract={'source_sha256':'source-a','source_interval':[0,2],'checkpoint_times':[0,1,2],
      'required_instances':[{'instance_id':'purple','target_id':'fold8'},{'instance_id':'white','target_id':'fold8'}],
      'forbidden_targets':['flip8']}
    instances=[{'instance_id':name,'target_id':'fold8','identity_status':'confirmed','box':box}
      for name,box in [('purple',[.05,.2,.3,.8]),('white',[.7,.2,.95,.8])]]
    frames=[{'source_seconds':t,'source_sha256':'source-a','crop':[0,0,1,1],'instances':copy.deepcopy(instances)} for t in [0,1,2]]
    return contract,frames


def test_same_sku_instances_need_independent_visibility():
    c,f=fixture();assert validate_composition(c,f)['passed']
    f[1]['crop']=[.34,0,.66,1]
    result=validate_composition(c,f)
    assert not result['passed'] and len(result['faults'])==2


def test_identity_success_cannot_hide_source_clock_or_instance_failure():
    c,f=fixture();f[1]['source_sha256']='other'
    assert not validate_composition(c,f)['passed']
    c,f=fixture();f.pop(1)
    assert not validate_composition(c,f)['passed']
    c,f=fixture();f[1]['instances'][0]['identity_status']='uncertain'
    assert not validate_composition(c,f)['passed']


def test_forbidden_instance_is_checked_in_actual_crop():
    c,f=fixture();f[0]['instances'].append({'instance_id':'flip','target_id':'flip8','box':[.4,.2,.6,.8]})
    assert not validate_composition(c,f)['passed']
    with pytest.raises(ValueError): visible_fraction([0,0,float('nan'),1],[0,0,1,1])


def test_recovery_never_removes_requested_identity():
    old={'source_id':'a','looks':[{'entity_id':'fold8'}],'cut_on_beat':True}
    selection={'shots':[old],'music_from_seconds':16}
    failing=[(0,old,'unproved')]
    bad={'shots':[{'replace_clip_id':'k00','looks':[{'entity_id':'none'}]}]}
    with pytest.raises(GroundingBlocked,match='removed'):merge_replacements(selection,failing,bad)
    assert old['looks'][0]['entity_id']=='fold8'
    good={'shots':[{'replace_clip_id':'k00','source_id':'b','looks':[{'entity_id':'fold8'}]}]}
    made=merge_replacements(selection,failing,good)
    assert made['shots'][0]['cut_on_beat'] and made['music_from_seconds']==16
    assert selection['shots'][0]['source_id']=='a'


def test_repeat_failure_persists_evidence_without_another_paid_call(tmp_path,monkeypatch):
    import montagewright.planner as planner
    monkeypatch.setattr(planner,'replan_shots',lambda *a,**k:pytest.fail('must not pay again'))
    fault=SimpleNamespace(clip_id='k00',entity_id='fold8')
    selection={'shots':[{'looks':[{'entity_id':'fold8'}]}]}
    with pytest.raises(GroundingBlocked):
        recover(selection,[fault],material=[],direction={},brief='',work=tmp_path,
                client=object(),cache=None,ledger=None,grounding_spec=None,attempt=1)
    assert (tmp_path/'grounding-blocked.json').exists()
    assert selection['shots'][0]['looks'][0]['entity_id']=='fold8'


def test_web_reloads_grounding_blocked_state(tmp_path):
    import json
    from montagewright.webapp import _state_of_a_foreign_run
    (tmp_path/'run-state.json').write_text(json.dumps({'state':'grounding_blocked'}))
    assert _state_of_a_foreign_run(tmp_path)=='grounding_blocked'


def test_resume_restores_only_legacy_failed_target_constraints():
    from montagewright.grounding_recovery import restore_legacy_constraints
    selection={'shots':[{'identity_status':'needs_review','identity_target_id':'fold8','looks':[{'entity_id':'none'}]},
                        {'looks':[{'entity_id':'none'}]}]}
    restore_legacy_constraints(selection)
    assert selection['shots'][0]['looks'][0]['entity_id']=='fold8'
    assert selection['shots'][0]['identity_status']=='unverified'
    assert selection['shots'][1]['looks'][0]['entity_id']=='none'


def test_recovery_reuses_saved_proposal_and_preserves_music(tmp_path, monkeypatch):
    import montagewright.planner as planner
    calls=[]
    def replan(*args, **kwargs):
        calls.append(kwargs)
        return {'shots':[{'replace_clip_id':'k00','source_id':'b','looks':[{'entity_id':'fold8'}]}]}, None
    monkeypatch.setattr(planner,'replan_shots',replan)
    selection={'shots':[{'source_id':'a','looks':[{'entity_id':'fold8'}],'cut_on_beat':True}],
               'music_from_seconds':16}
    fault=SimpleNamespace(clip_id='k00',entity_id='fold8')
    options=dict(material=[],direction={},brief='',work=tmp_path,client=object(),
                 cache=None,ledger=None,grounding_spec=None,attempt=0)
    first=recover(selection,[fault],**options)
    second=recover(selection,[fault],**options)
    assert first==second and len(calls)==1
    assert first['music_from_seconds']==16 and first['shots'][0]['cut_on_beat']
    assert calls[0]['editor_selection']==selection


def test_processing_evidence_survives_checkpoint_without_recharging():
    from montagewright.checkpoints import capture,replay
    original=SimpleNamespace(status='completed',output_text='{}',id='response',usage={},
        steps=[{'type':'processing_call','id':'p'}, {'type':'processing_result','id':'p'}])
    saved=capture(original,'editor_tool','test-model')
    cached=replay(saved)
    assert cached.saved_processing_steps==original.steps
    assert cached.steps==[] and cached.usage=={} and cached.checkpoint_reused
