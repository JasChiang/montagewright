"""Bounded editorial recovery without relaxing the requested identity."""
from __future__ import annotations
import copy
from pathlib import Path
from montagewright.checkpoints import key_for, read_json, write_json


class GroundingBlocked(RuntimeError):
    pass


def required_targets(shot):
    return {look.get('entity_id') for look in shot.get('looks', [])
            if look.get('entity_id') not in (None, '', 'none')}


def merge_replacements(selection, failing, proposal):
    expected = {f'k{i:02d}': (i, old) for i, old, _ in failing}
    shots = proposal.get('shots', [])
    ids = [s.get('replace_clip_id') for s in shots]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise GroundingBlocked('identity repair must replace each failed clip exactly once')
    result = copy.deepcopy(selection)
    for shot in shots:
        index, old = expected[shot['replace_clip_id']]
        if not required_targets(old) <= required_targets(shot):
            raise GroundingBlocked('identity repair removed a required target')
        new = copy.deepcopy(shot)
        new.setdefault('cut_on_beat', old.get('cut_on_beat', False))
        result['shots'][index] = new
    return result


def recover(selection, faults, *, material, direction, brief, work: Path,
            client, cache, ledger, grounding_spec, attempt):
    from montagewright.planner import replan_shots
    failing = []
    for fault in faults:
        index = next((i for i in range(len(selection['shots'])) if f'k{i:02d}' == fault.clip_id), None)
        if index is not None:
            failing.append((index, selection['shots'][index], str(fault)))
    evidence = {'selection': selection, 'failures': [dict(clip_id=f.clip_id, target_id=f.entity_id,
                reason=str(f)) for f in faults], 'attempt': attempt,
                'policy': 'keep identity constraints; inspect source and propose executable replacement'}
    write_json(work/'grounding-blocked.json', evidence)
    if attempt or not failing:
        raise GroundingBlocked('grounding remained unproved after one bounded editorial repair; saved evidence')
    receipt = work/'grounding-repairs'/f'{key_for(evidence)}.json'
    proposal = read_json(receipt)
    if proposal is None:
        try:
            proposal, _ = replan_shots(failing, material, direction, brief=brief,
                context='Identity grounding failed. Do not clear targets or use an unconstrained center crop. '
                        'Inspect earlier/later source moments or alternative footage. Preserve the required product identity.',
                client=client, cache=cache, ledger=ledger, grounding_spec=grounding_spec,
                editor_selection=selection)
        except ValueError as error:
            raise GroundingBlocked(f'grounding repair was not executable: {error}') from error
        write_json(receipt, proposal)
    return merge_replacements(selection, failing, proposal)


def restore_legacy_constraints(selection):
    """Reinstate explicitly recorded targets removed by the old draft fallback."""
    for shot in selection.get('shots',[]):
        target=shot.get('identity_target_id')
        if shot.get('identity_status')!='needs_review' or target in (None,'','none'):
            continue
        if not shot.get('looks'):
            raise GroundingBlocked('legacy identity failure has no recoverable look contract')
        restored=False
        for look in shot['looks']:
            if look.get('entity_id') in (None,'','none'):
                look['entity_id']=target
                restored=True
        if restored:
            shot['identity_status']='unverified'
            shot['identity_constraint_restored']=True
