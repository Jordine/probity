"""Read-only iteration5 Qwen3.5 paired behavior analysis; run with python3."""
from pathlib import Path
import csv, hashlib, html, json, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'reports/data/iteration6/behavior'
FIG=ROOT/'reports/figures/iteration6/behavior'
OUT=ROOT/'runs/iteration6/behavior'
MODEL='qwen/qwen3.5-27b'
INPUTS={}
def load(path):
 p=ROOT/path;INPUTS[str(path)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def csvout(name,rows):
 with (DATA/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(rows,key):
 d=defaultdict(list)
 for r in rows:
  if r[key] is not None:d[r['family']].append(float(r[key]))
 x=np.array([np.mean(d[k]) for k in sorted(d)]);rng=np.random.default_rng(4166)
 b=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
 return {'mean':float(x.mean()),'lo':float(np.quantile(b,.025)),'hi':float(np.quantile(b,.975)),'families':len(x)}
def percentile(x,ref):return round(100*(sum(v<x for v in ref)+.5*sum(v==x for v in ref))/len(ref),3)
def words(x):return re.findall(r"\b[\w’']+\b",x.lower())
def jaccard(a,b):
 a,b=set(words(a)),set(words(b));return len(a&b)/len(a|b)

def run():
 for p in [DATA,FIG,OUT]:p.mkdir(parents=True,exist_ok=True)
 pairs=[x for x in load('runs/iteration5/judge_grid_v1/pairs.json') if x['participant_model']==MODEL]
 assert len(pairs)==166 and len({x['family'] for x in pairs})==116
 bundles={a:load('runs/iteration5/response_bundles/adapted_primary_'+s+'_v1.json') for a,s in [('A','baseline'),('B','feedback')]}
 scoremaps={a:{(r['episode_id'],r['turn_index']):r for r in b['rows']} for a,b in bundles.items()}
 turns=[];episodes={};trajectory=[];paired=[]
 for p in pairs:
  per={}
  for arm in ['A','B']:
   reused=p['generation_provenance']=='reused_iteration3_and_4'
   directory=('runs/iteration3/final/debates_v1' if reused else 'runs/iteration5/debates_v4') if arm=='A' else ('runs/iteration4/feedback_v1' if reused else 'runs/iteration5/feedback_v1')
   eid=p['no_feedback_episode_id' if arm=='A' else 'feedback_episode_id']
   e=load(f'{directory}/episode_{eid}.json');episodes[eid]=e
   assert e['complete'] and len(e['turns'])==6 and e['model']==MODEL
   texts=defaultdict(list);scores=defaultdict(list);tr=[]
   for i,t in enumerate(e['turns']):
    s=scoremaps[arm][eid,i];argument=t['argument'];h=hashlib.sha256(argument.encode()).hexdigest()
    assert s['argument_sha256']==h==p['no_feedback_argument_sha256' if arm=='A' else 'feedback_argument_sha256'][i]
    raw=s['scores'][0];pct=percentile(raw,bundles[arm]['references'][MODEL]['scores'])
    if arm=='B':assert pct==t['probe_feedback']['percentile'] and abs(raw-t['probe_feedback']['raw_score'])<1e-9
    speaker=t['speaker'];previous=texts[speaker];own=scores[speaker]
    r={'question':p['matched_question_id'],'family':p['family'],'family_name':p['family_name'],'arm':arm,'episode_id':eid,'turn':i+1,'round':t['round'],'speaker':speaker,'role':'correct_answer_advocate' if t['assigned_answer']==e['gold'] else 'wrong_answer_advocate','raw_score':raw,'percentile':pct,'words':len(argument.split()),'characters':len(argument),'prior_public_turns':i,'prior_feedback_scores':i if arm=='B' else 0,'own_prior_percentile':own[-1] if own else None,'own_score_change':pct-own[-1] if own else None,'same_as_previous_own':int(argument==previous[-1]) if previous else None,'jaccard_previous_own':jaccard(argument,previous[-1]) if previous else None,'role_switch':int(t['answer']!=t['assigned_answer']),'request_sha256':t['request_sha256'],'argument_sha256':h}
    turns.append(r);tr.append(r);texts[speaker].append(argument);scores[speaker].append(pct)
   sr={'question':p['matched_question_id'],'family':p['family'],'arm':arm,'episode_id':eid,'words':np.mean([x['words'] for x in tr]),'characters':np.mean([x['characters'] for x in tr]),'any_exact_repeat':int(any(len(set(v))<3 for v in texts.values())),'all_three_same_speakers':sum(len(set(v))==1 for v in texts.values()),'jaccard_previous_own':np.mean([x['jaccard_previous_own'] for x in tr if x['jaccard_previous_own'] is not None]),'wrong_percentile':np.mean([x['percentile'] for x in tr if x['role']=='wrong_answer_advocate']),'correct_percentile':np.mean([x['percentile'] for x in tr if x['role']=='correct_answer_advocate'])}
   trajectory.append(sr);per[arm]=sr
  assert episodes[p['no_feedback_episode_id']]['private_preparations']==episodes[p['feedback_episode_id']]['private_preparations']
  paired.append({'question':p['matched_question_id'],'family':p['family'],**{k:float(per['B'][k]-per['A'][k]) for k in ['words','characters','any_exact_repeat','all_three_same_speakers','jaccard_previous_own','wrong_percentile','correct_percentile']}})
 assert len(turns)==1992 and sum(x['arm']=='A' for x in turns)==996
 assert sum(x['any_exact_repeat'] for x in trajectory if x['arm']=='A')==116
 assert sum(x['any_exact_repeat'] for x in trajectory if x['arm']=='B')==88
 csvout('turn_metrics.csv',turns);csvout('trajectory_metrics.csv',trajectory);csvout('paired_metrics.csv',paired)
 summary={'model':MODEL,'questions':166,'families':116,'trajectories':332,'turns':1992,'bootstrap':{'seed':4166,'resamples':10000,'unit':'source family','estimand':'equal-weight mean of within-family question means','limitation':'one trajectory per arm; no generation-seed uncertainty; unadjusted intervals'},'paired_B_minus_A':{k:ci(paired,k) for k in list(paired[0])[2:]},'by_arm':{},'score_round':[]}
 for arm in ['A','B']:
  rr=[r for r in trajectory if r['arm']==arm]
  summary['by_arm'][arm]={k:ci(rr,k) for k in ['words','any_exact_repeat','jaccard_previous_own','wrong_percentile','correct_percentile']}
  summary['by_arm'][arm]['exact_repeat_count']=sum(r['any_exact_repeat'] for r in rr)
  for role in ['correct_answer_advocate','wrong_answer_advocate']:
   for rd in [1,2,3]:
    rr=[r for r in turns if r['arm']==arm and r['role']==role and r['round']==rd]
    summary['score_round'].append({'arm':arm,'role':role,'round':rd,**ci(rr,'percentile')})
 # Round 3 minus round 1 changes retain the same question/speaker, then family average.
 change=[]
 for arm in ['A','B']:
  for role in ['correct_answer_advocate','wrong_answer_advocate']:
   groups=defaultdict(dict)
   for t in turns:
    if t['arm']==arm and t['role']==role:groups[t['question']][t['round']]=t
   rr=[{'family':v[1]['family'],'change':v[3]['percentile']-v[1]['percentile']} for v in groups.values()]
   change.append({'arm':arm,'role':role,**ci(rr,'change')})
 summary['round3_minus_round1']=change
 judges=[]
 jp=ROOT/'runs/iteration5/analysis_v1/paired_episodes.jsonl';INPUTS[str(jp.relative_to(ROOT))]=hashlib.sha256(jp.read_bytes()).hexdigest()
 for l in jp.read_text().splitlines():
  r=json.loads(l)
  if r['participant_model']==MODEL:judges.append({'judge':r['judge'],'question':r['matched_question_id'],'family':r['family'],**{c:int(r[c]) for c in 'ABCD'},'B_minus_A':int(r['B'])-int(r['A'])})
 assert len(judges)==332
 csvout('judge_pairs.csv',judges);summary['judges']={}
 for j in sorted({r['judge'] for r in judges}):
  rr=[r for r in judges if r['judge']==j]
  summary['judges'][j]={c:ci(rr,c) for c in ['A','B','C','D','B_minus_A']}
  summary['judges'][j]['paired_counts']={'fixes':sum(r['B']>r['A'] for r in rr),'harms':sum(r['B']<r['A'] for r in rr),'both_correct':sum(r['A'] and r['B'] for r in rr),'both_wrong':sum(not r['A'] and not r['B'] for r in rr)}
 summary['role_switches']=sum(t['role_switch'] for t in turns)
 dump(DATA/'summary.json',summary)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for role,col in [('correct_answer_advocate','#187f80'),('wrong_answer_advocate','#c05b36')]:
  for arm,ls in [('A','--'),('B','-')]:
   rr=[r for r in summary['score_round'] if r['arm']==arm and r['role']==role]
   axs[0,0].plot([r['round'] for r in rr],[r['mean'] for r in rr],ls+'o',color=col,label=f'{arm}: '+role.replace('_answer_advocate',' advocate'))
   axs[0,0].fill_between([1,2,3],[r['lo'] for r in rr],[r['hi'] for r in rr],color=col,alpha=.08)
 axs[0,0].set(title='Assigned-role probe signal by round',xlabel='Speaker round',ylabel='Mean reference percentile',xticks=[1,2,3],ylim=(0,80));axs[0,0].legend(fontsize=8)
 for arm,col in [('A','#666666'),('B','#4c73b7')]:
  vals=[]
  for rd in [1,2,3]:vals.append(ci([t for t in turns if t['arm']==arm and t['round']==rd],'words'))
  axs[0,1].errorbar([1,2,3],[v['mean'] for v in vals],yerr=[[v['mean']-v['lo'] for v in vals],[v['hi']-v['mean'] for v in vals]],fmt='o-',color=col,label=arm)
 axs[0,1].set(title='Argument length',xlabel='Speaker round',ylabel='Words per turn',xticks=[1,2,3]);axs[0,1].legend()
 rr=[summary['by_arm'][a]['any_exact_repeat'] for a in ['A','B']]
 axs[1,0].bar(['A: no feedback','B: live feedback'],[100*r['mean'] for r in rr],color=['#888888','#4c73b7'],yerr=[[100*(r['mean']-r['lo']) for r in rr],[100*(r['hi']-r['mean']) for r in rr]],capsize=5)
 axs[1,0].set(title='Trajectories with any exact same-speaker repeat',ylabel='Family-macro percentage',ylim=(0,100))
 for ix,(j,v) in enumerate(summary['judges'].items()):
  rr=[v[c] for c in ['A','B']];xx=np.array([0,1])+.06*ix
  axs[1,1].errorbar(xx,[100*r['mean'] for r in rr],yerr=[[100*(r['mean']-r['lo']) for r in rr],[100*(r['hi']-r['mean']) for r in rr]],fmt='o-',label=j)
 axs[1,1].set(title='Unassisted final judge: A versus B',xticks=[0,1],xticklabels=['A: no feedback','B: live feedback'],ylabel='Family-macro accuracy (%)',ylim=(50,100));axs[1,1].legend(fontsize=8)
 fig.suptitle('Qwen3.5-27B: 166 matched questions / 116 source families\nFamily-macro means and 95% family-bootstrap intervals; A scores computed retrospectively',fontsize=12)
 for ext in ['png','pdf']:fig.savefig(FIG/f'paired_behavior.{ext}',dpi=180)
 plt.close(fig)
 # Reuse outcome-independent inspection selections frozen before expansion.
 sample=load('runs/iteration5/reader_reviews/inspection_pack_v5/evidence.json')
 units=[u for u in sample['units'] if u['pair']['participant_model']==MODEL];assert len(units)==2
 rows=[];body=['<!doctype html><meta charset="utf-8"><title>Qwen3.5 feedback and next response</title><style>body{max-width:1200px;margin:auto;font:16px system-ui}pre{white-space:pre-wrap;background:#eee;padding:12px}td{width:50%;vertical-align:top;padding:12px}table{width:100%;table-layout:fixed}</style><h1>Fixed Qwen3.5 qualitative sample</h1><p>Two questions selected before expansion outcomes. Exact saved private feedback blocks and public arguments. This is not a representative rate estimate. Full original messages, private plans and provider envelopes remain in the linked inspection pack.</p>']
 for u in units:
  body.append('<h2>'+html.escape(u['pair']['family_name'])+'</h2>')
  for i in range(6):
   a=u['arms']['baseline']['turns'][i];b=u['arms']['feedback']['turns'][i]
   msg=b['submitted_messages'][-1]['content'];start=msg.index('BEGIN PRIVATE PAST-TURN PROBE FEEDBACK');feedback=msg[start:]
   row={'question':u['pair']['matched_question_id'],'family_name':u['pair']['family_name'],'turn':i+1,'speaker':b['episode_turn']['speaker'],'feedback_exact':feedback,'A_argument':a['episode_turn']['argument'],'B_argument':b['episode_turn']['argument'],'B_result_percentile':b['score_result']['percentile'],'B_request_sha256':b['episode_turn']['request_sha256']};rows.append(row)
   body.append(f'<h3>Turn {i+1}: {html.escape(row["speaker"])}</h3><details><summary>Exact private feedback before B response</summary><pre>{html.escape(feedback)}</pre></details><table><tr><th>A: no feedback</th><th>B: live feedback</th></tr><tr><td>{html.escape(row["A_argument"])}</td><td>{html.escape(row["B_argument"])}</td></tr></table><p>B response score after generation: {row["B_result_percentile"]}; request {row["B_request_sha256"]}</p>')
 (OUT/'qualitative_sample.html').write_text('\n'.join(body));dump(OUT/'qualitative_sample.json',rows)
 dump(OUT/'manifest.json',{'schema':'iteration6_qwen35_behavior_v1','input_hashes':INPUTS,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'counts':{'pairs':166,'families':116,'trajectories':332,'turns':1992,'judge_pairs':332},'qualitative_sample_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'qualitative_sample.html',OUT/'qualitative_sample.json']},'outputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [DATA,FIG] for p in sorted(folder.iterdir()) if p.is_file()},'paid_calls':0})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
