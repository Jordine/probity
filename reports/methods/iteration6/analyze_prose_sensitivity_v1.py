"""Posthoc complete-family sensitivity for manually confirmed incomplete prose."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from probity_benchmark import grid6_live as live
from probity_benchmark.grid6_protocol import POLICIES
from analyze_judges_v1 import interval
FAMILY='c315c58070ae394a5d522c55860a5391f8f1f7714b8ca205a2865e3a2899c14b'

def audit(destination):
 m,ps,bs,_=live.load_run(ROOT/'runs/iteration6/live/display24_v1')
 if not all(b['complete'] for b in bs):raise ValueError('full final display population required')
 screen=[];seenprefix=set()
 for b in bs:
  for i,t in enumerate(b['turns']):
   if i<2:
    key=(b['family_id'],i)
    if key in seenprefix:continue
    seenprefix.add(key)
   text=t['argument'].rstrip()
   if text[-1] in '.!?\"\u201d\u2019\'':continue
   rec={'family_id':b['family_id'],'branch_id':b['branch_id'],'policy':b['policy'],'turn_index':i,'argument':t['argument'],'mechanical_nonterminal':True,'manually_confirmed_incomplete':False}
   if i>=2:
    r=b['continuation_receipts'][i-2]['receipt'];raw=json.loads(r['response_body']);content=raw['choices'][0]['message']['content'];parsed=json.loads(content)
    if parsed['argument']!=t['argument']:raise ValueError('raw argument differs')
    rec.update(request_sha256=r['request_sha256'],response_body_sha256=r['response_body_sha256'],provider=raw.get('provider'),finish_reason=r['finish_reason'],usage=r['usage'],max_tokens=r['request']['max_tokens'],raw_content=content)
   if b['family_id']==FAMILY and text.endswith('with the notation'):
    rec['manually_confirmed_incomplete']=True;rec['manual_basis']='Full raw paragraph and source quotation reviewed in qualitative_case2_v1.md: quotation introduction ends before its quoted content. Exact raw provider text retained; no parser clipping.'
   screen.append(rec)
 if any(not r['manually_confirmed_incomplete'] for r in screen):raise ValueError('additional screened endings require manual review before sensitivity: '+str([(r['family_id'],r['turn_index']) for r in screen if not r['manually_confirmed_incomplete']]))
 out={'schema':'display_prose_audit_v1','fresh_arguments':672,'unique_shared_prefix_arguments':48,'mechanical_screen_count':len(screen),'confirmed_incomplete_arguments':sum(r['manually_confirmed_incomplete'] for r in screen),'excluded_families':sorted({r['family_id'] for r in screen if r['manually_confirmed_incomplete']}),'cases':screen,'selection':'Mechanical punctuation screen is a discovery aid, not an exhaustive semantic completeness test. Confirmation is separate manual source/raw inspection. Exclude an entire family if any argument in any of seven policies is confirmed incomplete; never selectively remove the affected policy.','run_manifest_sha256':live.file_hash(ROOT/'runs/iteration6/live/display24_v1/manifest.json')}
 destination.mkdir(parents=True,exist_ok=False);live.durable(destination/'audit.json',out);print(json.dumps({k:out[k] for k in ['mechanical_screen_count','confirmed_incomplete_arguments','excluded_families']}))

def analyze(auditpath,destination):
 audit=live.read(auditpath);removed=set(audit['excluded_families'])
 def csvrows(p):return list(csv.DictReader(p.open()))
 behavior=csvrows(ROOT/'reports/data/iteration6/display_behavior_v1/trajectory_metrics.csv');judges=[r for r in csvrows(ROOT/'reports/data/iteration6/live_judges_v1/judgments.csv') if r['group']=='display']
 families=sorted({r['family'] for r in behavior}-removed)
 if len(families)+len(removed)!=24:raise ValueError('removed family outside population')
 effects=[]
 for name,rows,metrics in [('behavior',behavior,['suffix_words','own_exact_repeat_fraction','own_jaccard','assigned_answer_deviations','authentic_role_score','incorrect_role_score']),('judge',judges,['haiku_correct','native_correct'])]:
  lookup={(r['family'],r['policy']):r for r in rows}
  for f in families:
   if any((f,p) not in lookup for p in POLICIES):raise ValueError('incomplete retained family')
  for metric in metrics:
   for baseline in ['none','neutral']:
    for policy in POLICIES:
     if policy==baseline:continue
     values=[float(lookup[f,policy][metric])-float(lookup[f,baseline][metric]) for f in families]
     effects.append({'population':name,'metric':metric,'baseline':baseline,'policy':policy,**interval(values)})
 destination.mkdir(parents=True,exist_ok=False)
 with (destination/'effects.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
 result={'primary_families':24,'retained_families':len(families),'excluded_families':sorted(removed),'retained_branches':7*len(families),'audit_sha256':live.file_hash(auditpath),'effects':effects,'interpretation':'Posthoc descriptive sensitivity, not replacement of primary 24-family estimates. Whole-family exclusion preserves all seven conditions. 10,000 family bootstrap draws; no generation-seed uncertainty; completeness screen not exhaustive.'};live.durable(destination/'summary.json',result)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11,4.5));colors={'haiku_correct':'#ae5b1a','native_correct':'#216d91'}
 for ax,baseline in zip(axs,['none','neutral']):
  policies=[p for p in POLICIES if p!=baseline]
  for metric,offset in [('haiku_correct',-.13),('native_correct',.13)]:
   for i,p in enumerate(policies):
    c=next(x for x in effects if x['metric']==metric and x['baseline']==baseline and x['policy']==p)
    ax.plot([100*c['lo95'],100*c['hi95']],[i+offset]*2,color=colors[metric]);ax.scatter(100*c['estimate'],i+offset,color=colors[metric],label=('Haiku 5.5' if metric.startswith('haiku') else 'Native Qwen3.5-4B') if i==0 else None)
  ax.set_yticks(range(len(policies)),policies);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Accuracy difference vs '+baseline+' (pp)');ax.spines[['top','right']].set_visible(False)
 axs[0].legend();fig.suptitle(f'Post-hoc prose-quality sensitivity · {len(families)} complete paired families\nExclude all seven policies of every family with a confirmed incomplete argument');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(destination/('judge_sensitivity.'+ext),dpi=180,bbox_inches='tight')
 lines=['# Prose-quality sensitivity','',result['interpretation'],'',f"Retained {len(families)}/24 families, {7*len(families)} branches per judge. Removed families: "+', '.join(sorted(removed))+'.','','The primary published display estimates retain all 24 families. This sensitivity excludes the same entire families for all display conditions and both judges; it never silently replaces malformed prose with regenerated text. Both behavior and judge contrasts against no feedback and neutral feedback are retained in effects.csv.']
 (destination/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'retained_families':len(families),'destination':str(destination)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','analyze']);p.add_argument('--destination',type=Path,required=True);p.add_argument('--audit',type=Path);a=p.parse_args()
 if a.mode=='audit':audit(a.destination)
 else:analyze(a.audit,a.destination)
