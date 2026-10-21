"""Offline display screen behavior, paired within the frozen shared-prefix families."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from probity_benchmark import grid6_live as live
from probity_benchmark.grid6_protocol import POLICIES, visible_packet, digest

def token_set(s):return set(s.lower().split())
def row(branch):
    suffix=branch['turns'][2:];prior={};repeats=[];similarity=[]
    for t in branch['turns']:
        old=prior.get(t['speaker'])
        if old is not None:
            repeats.append(int(old.strip()==t['argument'].strip()))
            a,b=token_set(old),token_set(t['argument']);similarity.append(len(a&b)/max(1,len(a|b)))
        prior[t['speaker']]=t['argument']
    r={'family':branch['family_id'],'policy':branch['policy'],'branch_id':branch['branch_id'],'suffix_words':float(np.mean([len(t['argument'].split()) for t in suffix])),'own_exact_repeat_fraction':float(np.mean(repeats)),'own_jaccard':float(np.mean(similarity)),'assigned_answer_deviations':sum(t['answer']!=t['assigned_answer'] for t in suffix)}
    for name,authentic in [('authentic_role',True),('incorrect_role',False)]:
        ts=[t for t in suffix if (t['assigned_answer']==branch['gold'])==authentic]
        r[name+'_score']=float(np.mean([t['probe_feedback']['percentile'] for t in ts]))
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'runs/iteration6/live/display24_v1');p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    m,problems,branches,_=live.load_run(a.out)
    if not all(b['complete'] for b in branches):raise ValueError('all168 branches must be complete for final descriptive report')
    a.destination.mkdir(parents=True,exist_ok=False);rows=[row(b) for b in branches]
    with (a.destination/'trajectory_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    families=sorted({b['family_id'] for b in branches});lookup={(r['family'],r['policy']):r for r in rows}
    if len(families)!=24 or any((f,p) not in lookup for f in families for p in POLICIES):raise ValueError('paired24x7 grid incomplete')
    rng=np.random.default_rng(24166);contrasts=[]
    metrics=['suffix_words','own_exact_repeat_fraction','own_jaccard','assigned_answer_deviations','authentic_role_score','incorrect_role_score']
    draws=rng.integers(0,len(families),(10000,len(families)))
    for metric in metrics:
        for baseline in ('none','neutral'):
            for policy in POLICIES:
                if policy==baseline:continue
                vals=np.array([lookup[f,policy][metric]-lookup[f,baseline][metric] for f in families]);boots=vals[draws].mean(axis=1)
                contrasts.append({'metric':metric,'baseline':baseline,'policy':policy,'contrast':'policy_minus_'+baseline,'families':24,'estimate':float(vals.mean()),'lo95':float(np.percentile(boots,2.5)),'hi95':float(np.percentile(boots,97.5))})
    selection=sorted(families,key=lambda f:hashlib.sha256(('grid6_display_qualitative_v1:'+f).encode()).hexdigest())[:2]
    packets=[]
    for b in branches:
        if b['family_id'] not in selection:continue
        entries=[]
        for item in b['continuation_receipts']:
            receipt=item['receipt'];entries.append({'exact_messages':receipt['request']['messages'],'raw_provider_response':json.loads(receipt['response_body']),'request_sha256':receipt['request_sha256'],'probe_result':item.get('score_result',item.get('score'))})
        packets.append({'family':b['family_id'],'policy':b['policy'],'branch_id':b['branch_id'],'gold':b['gold'],'source_question':problems[b['problem_id']].question,'shared_first_round':b['turns'][:2],'fresh_turns':b['turns'][2:],'continuation_contexts':entries,'initial_branch_artifact':str(a.out/'states'/b['branch_id']/'turns_02.json'),'initial_round_raw_note':'Inherited historical first round; exact old request hashes preserved. Refer to iteration5 raw archive for its provider envelopes. Every new suffix raw envelope is included here.'})
    live.durable(a.destination/'qualitative_packets.json',{'selection':'First 2 families sorted SHA256(grid6_display_qualitative_v1:+family), all 7 policies; outcome-independent selection. Includes exact system/user/private feedback and returned response envelopes for every fresh turn.','family_ids':selection,'packets':packets})
    shuffled=[]
    for b in branches:
        if b['policy']!='shuffled_bundle':continue
        for i in range(2,6):
            real=visible_packet(b['turns'][:i],'bundle',b['shuffle_seed']);fake=visible_packet(b['turns'][:i],'shuffled_bundle',b['shuffle_seed'])
            shuffled.append({'family':b['family_id'],'turn_index':i,'all_values_unchanged':digest(real['units'])==digest(fake['units']),'recommendation_unchanged':real['detector_only_recommendation']==fake['detector_only_recommendation']})
    judge_counts={'haiku_expected':168,'haiku_valid':sum(live.read(p).get('valid') is True for p in (a.out.parent/'haiku24_v1/requests').glob('*.json')),'native_expected':168,'native_valid':0}
    native=ROOT/'runs/iteration6/gpu/native_results_v1/judge_v1/evaluations.jsonl'
    branch_ids={b['branch_id'] for b in branches}
    if native.exists():
        native_jobs=ROOT/'runs/iteration6/gpu/native_bundle_v1/judge_jobs.jsonl'
        selected={r['job_id'] for r in map(json.loads,native_jobs.read_text().splitlines()) if r.get('branch_id',r.get('episode_id')) in branch_ids}
        judge_counts['native_valid']=sum(r.get('valid') is True and r['job_id'] in selected for r in map(json.loads,native.read_text().splitlines()))
    summary={'judge_counts_at_analysis':judge_counts,'shuffled_control':{'packets':len(shuffled),'unchanged_numeric_packets':sum(x['all_values_unchanged'] for x in shuffled),'unchanged_recommendations':sum(x['recommendation_unchanged'] for x in shuffled),'limitation':'Permutation need not be a derangement; some packets or recommendations remain unchanged. This is a weak negative control, not guaranteed uninformative or adversarial feedback.'},'branches':168,'families':24,'fresh_suffix_turns':672,'shared_first_round_turns_per_family':2,'contrasts':contrasts,'interpretation':'All effects conditional on shared original A first round and historical private plans. Initial prefix is identical across seven policies. Probe scores summarize assigned roles, not sentence factuality. Live feedback changes subsequent contexts, so suffix cannot be counterfactually replayed as equivalent. One generated suffix per policy; paired family bootstrap does not estimate seed variance.','provenance':{'run_manifest_sha256':live.file_hash(a.out/'manifest.json'),'script_sha256':live.file_hash(__file__)},'qualitative_selection':selection}
    live.durable(a.destination/'summary.json',summary)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    for ax,metric,label in zip(axs,['suffix_words','own_exact_repeat_fraction','incorrect_role_score'],['Words per new response','Exact self-repeat fraction','Incorrect-role probe percentile']):
        cs=[c for c in contrasts if c['metric']==metric and c['baseline']=='none']
        for i,c in enumerate(cs):
            ax.plot([c['lo95'],c['hi95']],[i,i],color='#286782',lw=2);ax.scatter(c['estimate'],i,color='#286782',s=25)
        ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(cs)),[c['policy'] for c in cs]);ax.invert_yaxis();ax.set_xlabel(label+' difference');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Display policy minus no feedback · 24 paired families\n95% family-bootstrap intervals; fresh rounds 2–3 conditional on shared round 1',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(a.destination/('display_behavior.'+ext),dpi=180,bbox_inches='tight')
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    for ax,metric,label in zip(axs,['suffix_words','own_exact_repeat_fraction','incorrect_role_score'],['Words per new response','Exact self-repeat fraction','Incorrect-role probe percentile']):
        cs=[c for c in contrasts if c['metric']==metric and c['baseline']=='neutral']
        for i,c in enumerate(cs):
            ax.plot([c['lo95'],c['hi95']],[i,i],color='#865a93',lw=2);ax.scatter(c['estimate'],i,color='#865a93',s=25)
        ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(cs)),[c['policy'] for c in cs]);ax.invert_yaxis();ax.set_xlabel(label+' difference');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Display policy minus neutral feedback · 24 paired families\n95% family-bootstrap intervals; neutral controls for receiving an extra feedback message',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(a.destination/('display_behavior_vs_neutral.'+ext),dpi=180,bbox_inches='tight')
    text=['# Shared-prefix display behavior','',summary['interpretation'],'','These are descriptive response changes. Factual-label metrics and final-judge effects are separate analyses.','', '| Display | Words vs none | Repeat fraction vs none | Incorrect-role score vs none |','|---|---:|---:|---:|']
    for policy in POLICIES[1:]:
        values=[next(c for c in contrasts if c['policy']==policy and c['metric']==metric and c['baseline']=='none') for metric in ['suffix_words','own_exact_repeat_fraction','incorrect_role_score']]
        text.append('|'+policy+'|'+'|'.join(f"{v['estimate']:.3f} [{v['lo95']:.3f}, {v['hi95']:.3f}]" for v in values)+'|')
    text+=['', 'The machine-readable summary also provides every policy minus neutral-formatting control; no-feedback and neutral are separate baselines.','', 'Shuffled control: '+str(summary['shuffled_control']), '', 'Final judge counts available at analysis time: '+str(judge_counts)+'. Missing native judgments are pending, not incorrect outcomes.', '', 'Raw reading packets use two fixed hash-selected families and include all policies, every fresh request message, feedback and complete returned response. The inherited initial round is identified by exact historical request hashes; its provider envelopes remain in the iteration5 archive.','', 'Runtime reproducibility: the repaired worker reproduces the original scalar implementation exactly on 12 same-process cases and passes four frozen historical checks. Some other historical scores differ slightly on this runtime; historical score drift is audited separately.']
    (a.destination/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps({'families':24,'branches':168,'contrasts':len(contrasts),'destination':str(a.destination)},indent=2))
if __name__=='__main__':main()
