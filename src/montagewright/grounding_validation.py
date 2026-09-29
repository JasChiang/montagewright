"""Offline composition contract checks and a reproducible grounding case viewer.

No provider calls. Saved model evidence is not treated as human ground truth.
"""
from __future__ import annotations
import argparse
import html
import json
import math
import subprocess
from pathlib import Path
from montagewright.checkpoints import write_json, key_for
from montagewright.uploads import content_hash


def visible_fraction(box, crop):
    for coordinates in (box, crop):
        if len(coordinates) != 4 or not all(math.isfinite(x) for x in coordinates):
            raise ValueError('boxes require four finite normalized coordinates')
        x0,y0,x1,y1=coordinates
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('invalid normalized box')
    x0,y0,x1,y1=box; a,b,c,d=crop
    return max(0,min(x1,c)-max(x0,a))*max(0,min(y1,d)-max(y0,b))/((x1-x0)*(y1-y0))


def validate_composition(contract, observations):
    """Require physical instance IDs, source-clock evidence and crop visibility.

    Two Fold8 instances are two requirements even though their SKU is equal.
    Observations are explicit checkpoints; a pass does not certify unseen frames.
    """
    required = contract['required_instances']
    if len({x['instance_id'] for x in required}) != len(required):
        raise ValueError('required physical instance IDs must be unique')
    if not required:
        raise ValueError('at least one required instance is needed')
    start,end = contract['source_interval']
    if not 0 <= start < end or not math.isfinite(end):
        raise ValueError('invalid source interval')
    faults=[]
    expected=contract['checkpoint_times']
    if not expected or any(not start <= t <= end for t in expected):
        raise ValueError('checkpoints must lie in the source interval')
    for time in expected:
        frames=[f for f in observations if abs(f['source_seconds']-time)<1e-6]
        if len(frames)!=1:
            faults.append(f'{time}: missing or ambiguous exact-frame evidence');continue
        frame=frames[0]
        if frame.get('source_sha256')!=contract['source_sha256']:
            faults.append(f'{time}: source hash mismatch');continue
        instances=frame['instances'];ids=[x['instance_id'] for x in instances]
        if len(ids)!=len(set(ids)):
            faults.append(f'{time}: ambiguous physical instance IDs');continue
        by_id={x['instance_id']:x for x in instances}
        for wanted in required:
            actual=by_id.get(wanted['instance_id'])
            if not actual or actual.get('identity_status')!='confirmed' or actual.get('target_id')!=wanted['target_id']:
                faults.append(f"{time}: unproved instance {wanted['instance_id']}");continue
            if visible_fraction(actual['box'],frame['crop']) < wanted.get('minimum_visible_fraction',.85):
                faults.append(f"{time}: crop loses required instance {wanted['instance_id']}")
        for instance in instances:
            if instance.get('target_id') in contract.get('forbidden_targets',[]) and visible_fraction(instance['box'],frame['crop'])>contract.get('maximum_forbidden_fraction',0):
                faults.append(f"{time}: forbidden target visible: {instance['instance_id']}")
    return {'passed':not faults,'faults':faults,'scope':'declared source-clock checkpoints only; continuous tracking and editorial QA remain separate'}


def build_cases(run: Path, output: Path):
    from montagewright.editor_workspace import framing_filter
    report=json.loads((run/'report.json').read_text())
    timeline=json.loads((run/'work/current-timeline.json').read_text())
    job=json.loads((run/'edit-job.json').read_text())
    selection=json.loads((run/'work/resolved-selection.json').read_text())['value']
    review_path=run/'work/resumed-shot-review.json'
    reviews=json.loads(review_path.read_text()) if review_path.exists() else report.get('shots',{})
    output.mkdir(parents=True,exist_ok=True)
    cases=[];sections=[];hashes={}
    for i,shot in enumerate(selection['shots']):
        key=f'k{i:02d}'; row=timeline['shots'][i];source=Path(job['rushes'])/f"{shot['source_id']}.MP4"
        if not source.exists():
            raise FileNotFoundError(source)
        length=min(5,row['frame_count']/timeline['output_fps']);start=row['in_seconds']
        variants=[('source','source','16:9',source,start),('actual','source','9:16',run/'segments'/f'{i:03d}-{key}.mp4',0)]
        variants += [(f'{aspect.replace(":","-")}-{mode}',mode,aspect,source,start)
                     for aspect in ('16:9','9:16','1:1') for mode in ('fit','fill')]
        videos=[]
        for name,mode,aspect,path,offset in variants:
            if path not in hashes: hashes[path]=content_hash(path)
            digest=key_for([hashes[path],offset,length,mode,aspect,'validation-view-v1'])[:12]
            dest=output/f'{key}-{name}-{digest}.mp4'
            if not dest.exists():
                partial=dest.with_suffix('.partial.mp4')
                subprocess.run(['ffmpeg','-v','error','-y','-ss',str(offset),'-i',str(path),'-t',str(length),
                    '-an','-vf',framing_filter(mode,aspect,length),'-c:v','libx264','-preset','veryfast','-crf','28',
                    '-movflags','+faststart',str(partial)],check=True)
                partial.replace(dest)
            videos.append(f'<figure><figcaption>{html.escape(name)}</figcaption><video controls muted preload="none" src="{dest.name}"></video></figure>')
        case={'clip_id':key,'source_id':shot['source_id'],'source_interval':[start,start+length],
              'planned_looks':shot.get('looks',[]),'grounding':report.get('reference_grounding',{}).get(key),
              'model_review':reviews.get(key),'human_annotation':None,
              'note':'Candidate previews are center-fill/full-fit diagnostics, not accepted edits or ground truth.'}
        cases.append(case)
        sections.append(f'<section><h2>{key} · {html.escape(shot["source_id"])} · {start:.3f}–{start+length:.3f}s</h2>'
            +f'<p>{html.escape(str(reviews.get(key,{}).get("note","")))}</p><div class="clips">'+''.join(videos)+'</div>'
            +'<details><summary>保存的計畫與 grounding 證據</summary><pre>'+html.escape(json.dumps(case,ensure_ascii=False,indent=2))+'</pre></details></section>')
    write_json(output/'cases.json',{'version':1,'run':str(run),'paid_calls':0,'human_ground_truth_complete':False,'cases':cases})
    page='''<!doctype html><meta charset="utf-8"><title>Grounding 獨立驗證</title><style>body{font:15px system-ui;background:#10151c;color:#e6edf4;margin:24px}section{border-top:1px solid #445;padding:16px 0}.clips{display:flex;overflow:auto;gap:12px}figure{margin:0;flex:0 0 230px}video{width:230px;height:250px;background:#000}pre{white-space:pre-wrap}p{max-width:1100px;line-height:1.6}</style><h1>Grounding 獨立驗證</h1><p>先比較原片與實際成片，再查看三種比例的置中滿版／完整留邊候選。候選未使用主體追蹤，不代表推薦剪法。下方判語來自既有模型審查，人工基準尚未建立；不能把 74/74 有回覆當成辨識準確率。</p>'''+''.join(sections)
    (output/'index.html').write_text(page)
    return output/'index.html'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(build_cases(args.run.resolve(),args.output.resolve()))

if __name__=='__main__':main()
