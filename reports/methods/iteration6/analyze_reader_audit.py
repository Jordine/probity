"""Describe independent reader agreement on the preselected family sample."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from probity_benchmark.grid6_evaluate import metadata, read_rows
from probity_benchmark.grid6_labels import PRIMARY, SECONDARY, eligible_label

def analyze(labels, turn_meta, families):
 selected=[r for r in labels if turn_meta[r['turn_id']]['family'] in families]
 if not selected or any(not r['primary_annotated'] or not r['secondary_annotated'] for r in selected):
  raise ValueError('independent family audit must be completely annotated by both readers')
 by_family=defaultdict(list);confusion=Counter();family_confusion=defaultdict(Counter);disagreements=[]
 for r in selected:
  a=r['checks_by_model'][PRIMARY]['check'];b=r['checks_by_model'][SECONDARY]['check']
  la=eligible_label(a,r['unit']['display_text']);lb=eligible_label(b,r['unit']['display_text'])
  by_family[turn_meta[r['turn_id']]['family']].append({'verdict_agreement':float(a['verdict']==b['verdict']),
    'eligibility_agreement':float((la is None)==(lb is None)),
    'primary_eligible':float(la is not None),'secondary_eligible':float(lb is not None),
    'eligible_verdict_agreement':float(la==lb) if la is not None and lb is not None else None})
  confusion[a['verdict'],b['verdict']]+=1
  family_confusion[turn_meta[r['turn_id']]['family']][a['verdict'],b['verdict']]+=1
  if a['verdict']!=b['verdict'] or (la is None)!=(lb is None):disagreements.append(r)
 result={}
 for metric in next(iter(by_family.values()))[0]:
  points=[]
  for family,rows in sorted(by_family.items()):
   values=[r[metric] for r in rows if r[metric] is not None]
   if values:points.append(float(np.mean(values)))
  if not points:
   result[metric]={'estimate':None,'family_bootstrap_ci95':None,'families':0};continue
  values=np.array(points);rng=np.random.default_rng(61)
  draws=values[rng.integers(len(values),size=(5000,len(values)))].mean(axis=1)
  result[metric]={'estimate':float(values.mean()),'family_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist(),'families':len(values)}
 return {'families':len(by_family),'units':len(selected),'metrics':result,
   'raw_sentence_confusion':[{ 'primary':a,'secondary':b,'count':n} for (a,b),n in sorted(confusion.items())],
   'family_balanced_confusion':[{'primary':a,'secondary':b,'fraction':float(np.mean([counts[a,b]/sum(counts.values()) for counts in family_confusion.values()]))} for a in ['supported','contradicted','unknown'] for b in ['supported','contradicted','unknown']],
   'administrative_waived_checks_in_audit':sum(isinstance(c.get('provenance'),dict) and 'administrative_waiver' in c['provenance'] for r in selected for c in r['checks_by_model'].values()),
   'disagreement_units':len(disagreements)},disagreements

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True)
 p.add_argument('--audit-plan',type=Path,default=ROOT/'runs/iteration6/planning/secondary_audit_plan.json');p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise ValueError('immutable output required')
 meta=metadata(ROOT/'runs/iteration5/judge_grid_v1/package.json',ROOT/'runs/iteration5/judge_grid_v1/pairs.json')
 plan=json.loads(a.audit_plan.read_text());result,disagreements=analyze(read_rows(a.labels),meta,set(plan['selected_families']))
 assert result['units']==plan['new_units']+plan['old_dual_reader_units'] and result['families']==plan['family_count']
 result.update(schema='iteration6_independent_reader_audit_v1',
   interpretation='Agreement between automated readers, not accuracy against human ground truth. Primary reader and consensus are separate tiers.',
   administrative_waiver='Explicitly marked checks may exceed only the600-character rationale limit; full original rationale and strict failure receipt retained. Every substantive label/text/evidence rule remains unchanged; no trimming or relabeling.',
   weighting='Average sentence agreement within each selected source family, then equal weight across families. Bootstrap resamples families.',
   selection='Frozen hash-ranked 12-family sample; excludes all legacy dual labels outside the selected sample.',
   input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.labels,a.audit_plan,Path(__file__)]})
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 (a.out/'disagreements_private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in disagreements))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,4));names=['verdict_agreement','eligibility_agreement','eligible_verdict_agreement'];titles=['Verdict agreement (all sentences)','Eligibility agreement (all sentences)','Verdict agreement (both eligible)']
 for i,name in enumerate(names):
  m=result['metrics'][name]
  if not m['families']:
   ax.text(50,i,'No jointly eligible sentences',ha='center');continue
  lo,hi=np.array(m['family_bootstrap_ci95'])*100;value=m['estimate']*100
  ax.plot([lo,hi],[i,i],lw=2);ax.scatter([value],[i],s=40);ax.text(value,i-.12,f'{value:.1f}%',ha='center',fontsize=9)
 ax.set_yticks(range(len(names)),titles);ax.set_xlim(0,105);ax.invert_yaxis();ax.set_ylim(len(names)-.6,-.4);ax.grid(axis='x',alpha=.2)
 ax.set_xlabel('Agreement %, equal family weights and family-bootstrap 95% CI');ax.set_title('Independent factual-label audit · 12 preselected source families\nQwen primary vs Opus secondary; automated agreement is not human accuracy',fontsize=11)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(a.out/f'reader_agreement.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 names=['supported','contradicted','unknown'];values=np.array([[next(r['fraction'] for r in result['family_balanced_confusion'] if r['primary']==a and r['secondary']==b) for b in names] for a in names])*100
 counts={(r['primary'],r['secondary']):r['count'] for r in result['raw_sentence_confusion']}
 fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(values,cmap='Blues',vmin=0,vmax=100)
 for i,a_name in enumerate(names):
  for j,b_name in enumerate(names):ax.text(j,i,f"{values[i,j]:.1f}%\n(n={counts.get((a_name,b_name),0)})",ha='center',va='center',color='white' if values[i,j]>50 else 'black')
 ax.set_xticks(range(3),names);ax.set_yticks(range(3),names);ax.set_xlabel('Secondary reader: Opus');ax.set_ylabel('Primary reader: Qwen');ax.set_title('Reader verdicts on the frozen 12-family sample\nPercentages use equal family weights; n is raw sentence count',fontsize=11)
 fig.colorbar(im,ax=ax,label='Family-balanced fraction of all sentences (%)');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(a.out/f'reader_confusion.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
