"""Fresh suffix factuality by display policy; paired source-family contrasts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from probity_benchmark.grid6_protocol import POLICIES

def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(labels,metadata,bootstraps=5000):
 meta={r['id']:r for r in metadata}
 if len(meta)!=len(metadata):raise ValueError('duplicate live turn metadata')
 families=sorted({r['family'] for r in metadata});turns=defaultdict(list)
 seen=set()
 for r in labels:
  if r['turn_id'] not in meta:raise ValueError('label outside fresh suffix population')
  key=(r['turn_id'],r['unit_index'])
  if key in seen:raise ValueError('duplicate labeled unit')
  if not 0<=r['unit_index']<meta[r['turn_id']]['unit_count']:raise ValueError('unit index outside frozen turn')
  seen.add(key);turns[r['turn_id']].append(r)
 records=[]
 for turn,m in meta.items():
  items=turns[turn];eligible=[r['primary_label'] for r in items if r['primary_label'] is not None]
  records.append({**m,'units':len(items),'eligible':len(eligible),'annotated':sum(r['primary_annotated'] for r in items),
    'expected_units':m['unit_count'],
    'complete':len(items)==m['unit_count'] and {r['unit_index'] for r in items}==set(range(m['unit_count'])) and all(r['primary_annotated'] for r in items),
    'contradiction_fraction':float(np.mean(eligible)) if eligible else None,
    'eligible_coverage':len(eligible)/len(items) if items else None})
 complete=[];incomplete=[]
 for family in families:
  subset=[r for r in records if r['family']==family]
  expected={(policy,index) for policy in POLICIES for index in range(2,6)}
  actual={(r['policy'],r['turn_index']) for r in subset}
  if len(subset)==len(expected) and actual==expected and all(r['complete'] for r in subset):complete.append(family)
  else:incomplete.append({'family':family,'present_turns':len(subset),'expected_turns':len(expected),'complete_labeled_turns':sum(r['complete'] for r in subset),'missing_policy_turns':[list(x) for x in sorted(expected-actual)]})
 results=[]
 for role in ['all','correct_advocate','wrong_advocate']:
  selected=[r for r in records if r['family'] in complete and (role=='all' or r['role']==role)]
  lookup={(r['family'],r['policy'],r['turn_index']):r for r in selected}
  for baseline in ['none','neutral']:
   for policy in POLICIES:
    if policy==baseline:continue
    for metric in ['contradiction_fraction','eligible_coverage']:
     points=[];matched=0
     for family in complete:
      paired=[]
      for index in range(2,6):
       a=lookup.get((family,baseline,index));b=lookup.get((family,policy,index))
       if a is None or b is None or a[metric] is None or b[metric] is None:continue
       paired.append([a[metric],b[metric]])
      if paired:points.append(np.mean(paired,axis=0));matched+=len(paired)
     record={'role':role,'baseline':baseline,'policy':policy,'metric':metric,'families':len(points),'matched_turns':matched}
     if points:
      arr=np.array(points);delta=arr[:,1]-arr[:,0];rng=np.random.default_rng(61)
      draws=delta[rng.integers(len(delta),size=(bootstraps,len(delta)))].mean(axis=1)
      record.update(baseline_mean=float(arr[:,0].mean()),policy_mean=float(arr[:,1].mean()),delta=float(delta.mean()),delta_ci95=np.quantile(draws,[.025,.975]).tolist())
     else:record.update(baseline_mean=None,policy_mean=None,delta=None,delta_ci95=None)
     results.append(record)
 coverage=[]
 for role in ['all','correct_advocate','wrong_advocate']:
  for policy in POLICIES:
   rr=[r for r in records if r['policy']==policy and (role=='all' or r['role']==role)]
   coverage.append({'role':role,'policy':policy,'families':len({r['family'] for r in rr}),'turns':len(rr),'units':sum(r['units'] for r in rr),'eligible_units':sum(r['eligible'] for r in rr),'primary_annotated_units':sum(r['annotated'] for r in rr),'missing_or_incomplete_turns':sum(not r['complete'] for r in rr),'all_ineligible_turns':sum(r['complete'] and r['eligible']==0 for r in rr)})
 return {'family_denominator':len(families),'complete_paired_seven_policy_families':complete,'incomplete_families':incomplete,'results':results,'coverage':coverage},records

def plot(summary,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=False)
 names={'none':'No feedback','neutral':'Neutral formatting','bundle':'Full bundle','shuffled_bundle':'Shuffled scores','turn_scores':'Turn scores','speaker_means':'Speaker means','recommendation':'Recommendation only'}
 policies=list(POLICIES);index={p:i for i,p in enumerate(policies)}
 for row,role in enumerate(['all','correct_advocate','wrong_advocate']):
  for col,metric in enumerate(['contradiction_fraction','eligible_coverage']):
   ax=axes[row,col]
   for baseline,color,shift in [('none','#3366aa',-.12),('neutral','#dd7722',.12)]:
    subset=[r for r in summary['results'] if r['role']==role and r['metric']==metric and r['baseline']==baseline]
    labeled=False
    for r in subset:
     y=index[r['policy']]+shift
     if r['delta'] is None:continue
     value=r['delta']*100;lo,hi=np.array(r['delta_ci95'])*100
     ax.plot([lo,hi],[y,y],color=color,lw=1.4);ax.scatter([value],[y],s=22,color=color,label='vs '+names[baseline] if not labeled else None);labeled=True
   ax.set_yticks(range(len(policies)),[names.get(p,p) for p in policies]);ax.invert_yaxis();ax.axvline(0,c='gray',ls='--');ax.grid(axis='x',alpha=.2)
   title='Contradicted fraction among eligible sentences' if metric=='contradiction_fraction' else 'Eligible fraction of all sentences'
   ax.set_title(role.replace('_',' ')+' · '+title,fontsize=10);ax.set_xlabel('Paired change (percentage points), family-bootstrap 95% CI');ax.legend(fontsize=8)
 fig.suptitle('Exploratory fresh suffix factuality after shared opening round\n'+str(len(summary['complete_paired_seven_policy_families']))+' complete paired families · automated primary reader · unknowns excluded from conditional factuality',fontsize=12)
 fig.tight_layout(rect=(0,0,1,.96))
 for ext in ['png','pdf']:fig.savefig(out/f'live_factuality.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--planning',type=Path,default=ROOT/'runs/iteration6/live_label_planning');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('immutable output required')
 metadata_path=a.planning/'turn_metadata_private.jsonl';plan_path=a.planning/'annotation_plan.json';plan=json.loads(plan_path.read_text())
 if digest(metadata_path)!=plan['files_sha256'][metadata_path.name]:raise ValueError('live metadata changed')
 summary,records=summarize(rows(a.labels),rows(metadata_path))
 summary.update(schema='iteration6_live_factuality_v1',weighting='Match identical suffix turn indices within source family and policy pair; average matched turns within family, then equal family weight. Only complete labeled seven-policy families enter contrasts.',
  multiplicity='Exploratory marginal95% intervals; no correction across policies, roles or outcomes.',
  limits='Automatic sentence factuality, not human truth or intent. Unknowns excluded from contradicted fraction, eligibility coverage shown separately. These are fresh generated suffixes under fixed original detector, not counterfactual replay.',
  input_sha256={str(p):digest(p) for p in [a.labels,metadata_path,plan_path,Path(__file__)]})
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.out/'turns.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));plot(summary,a.out)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['results','coverage']},indent=2))
if __name__=='__main__':main()
