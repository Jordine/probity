"""Pair native4B and Haiku judgments on the exact same new debate exports."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from probity_benchmark import grid6_live as live,grid6_judge as haiku,grid6_prompt as prompt
from probity_benchmark.grid6_protocol import POLICIES

def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def interval(values):
    x=np.array(values,dtype=float)
    if not len(x):raise ValueError('empty paired population')
    rng=np.random.default_rng(62166);draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return {'estimate':float(x.mean()),'lo95':float(np.percentile(draws,2.5)),'hi95':float(np.percentile(draws,97.5)),'families':len(x)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    base=ROOT/'runs/iteration6';bundle=base/'gpu/native_bundle_v1';native=base/'gpu/native_results_v1'
    complete=live.read(native/'native_complete_v1.json');ready=live.read(bundle/'ready.json')
    if complete.get('complete') is not True or live.file_hash(bundle/'judge_jobs.jsonl')!=complete['jobs_sha256'] or complete['jobs_sha256']!=ready['jobs_sha256']:raise ValueError('native job provenance mismatch')
    if live.file_hash(native/'judge_v1/evaluations.jsonl')!=complete['evaluations_sha256'] or live.file_hash(native/'judge_v1/manifest.json')!=complete['manifest_sha256']:raise ValueError('native evaluation/manifest hash mismatch')
    jobs=rows(bundle/'judge_jobs.jsonl');evals=rows(native/'judge_v1/evaluations.jsonl');ej={r['job_id']:r for r in evals}
    if len(jobs)!=ready['expected_jobs'] or len(evals)!=len(jobs) or len(ej)!=len(jobs) or set(ej)!={j['job_id'] for j in jobs}:raise ValueError('native exact coverage incomplete')
    if not all(r['valid'] is True and r['finish_reason']=='stop' for r in evals):raise ValueError('invalid native result')
    nj={j['branch_id']:j for j in jobs}
    dm,_,db,_=live.load_run(base/'live/display24_v1');hm,hjobs=haiku.load(base/'live/haiku24_v1');hby={j['branch_id']:j for j in hjobs}
    records=[]
    def add(branch,job,receipt,group):
        bid=branch['branch_id'];j=nj[bid];nr=ej[j['job_id']]
        if j['messages']!=job['request']['messages'] or j['gold']!=job['gold']:raise ValueError('judges did not see identical public context')
        if nr['parsed']['answer'] not in ('A','B') or nr['correct']!=(nr['parsed']['answer']==j['gold']):raise ValueError('native answer/gold mismatch')
        records.append({'group':group,'family':branch['family_id'],'branch_id':bid,'policy':branch['policy'],'level':branch.get('level'),'gold':j['gold'],'haiku_answer':receipt['parsed']['answer'],'native_answer':nr['parsed']['answer'],'haiku_correct':int(receipt['parsed']['answer']==j['gold']),'native_correct':int(nr['correct']),'native_job_id':j['job_id'],'haiku_request_sha256':receipt['request_sha256'],'native_request_sha256':nr['request_sha256']})
    for b in db:
        if not b['complete']:raise ValueError('display branch incomplete')
        j=hby[b['branch_id']];r=live.read(base/'live/haiku24_v1/requests'/f"{j['job_id']}.json");haiku.validate_job_receipt(j,r);add(b,j,r,'display')
    if not ready['prompt_skipped']:
        pm,problems,pb,_=prompt.load(base/'live/prompt8_v1')
        for b in pb:
            if not b['complete']:raise ValueError('prompt branch incomplete')
            j=prompt.judge_job(pm,problems[b['problem_id']],b);r=prompt.recover(base/'live/prompt8_v1',j)
            if r is None:raise ValueError('missing final Haiku prompt judgment')
            add(b,j,r,'prompt:'+b['level'])
    if len(records)!=len(jobs):raise ValueError('cross-judge branch population mismatch')
    estimates=[];effects=[];interactions=[]
    for group in sorted({r['group'] for r in records}):
        rr=[r for r in records if r['group']==group];families=sorted({r['family'] for r in rr});policies=POLICIES if group=='display' else prompt.POLICIES
        expected=24 if group=='display' else 8
        lookup={(r['family'],r['policy']):r for r in rr}
        if len(families)!=expected or len(rr)!=expected*len(policies) or any((f,c) not in lookup for f in families for c in policies):raise ValueError('paired condition grid incomplete')
        for model in ('haiku','native'):
            for policy in policies:
                estimates.append({'group':group,'judge':model,'policy':policy,**interval([lookup[f,policy][model+'_correct'] for f in families])})
            for baseline in (('none','neutral') if group=='display' else ('none',)):
                for policy in policies:
                    if policy==baseline:continue
                    before=[lookup[f,baseline][model+'_correct'] for f in families];after=[lookup[f,policy][model+'_correct'] for f in families]
                    effects.append({'group':group,'judge':model,'policy':policy,'baseline':baseline,'fixes':sum(a>b for a,b in zip(after,before)),'harms':sum(a<b for a,b in zip(after,before)),**interval([a-b for a,b in zip(after,before)])})
        for policy in policies:
            if policy=='none':continue
            values=[(lookup[f,policy]['native_correct']-lookup[f,'none']['native_correct'])-(lookup[f,policy]['haiku_correct']-lookup[f,'none']['haiku_correct']) for f in families]
            interactions.append({'group':group,'policy':policy,'contrast':'native_uplift_minus_haiku_uplift',**interval(values)})
    result={'schema':'grid6_paired_final_judges_v1','display_branches':168,'prompt_branches':0 if ready['prompt_skipped'] else 32,'per_judge_total':len(records),'prompt_skipped':ready['prompt_skipped'],'accuracy':estimates,'policy_effects':effects,'judge_uplift_interactions':interactions,'uncertainty':'10,000 source-family bootstrap draws, seed 62166. Same cases/arguments for both judges; 24 families per display condition, 8 per instruction condition. No generation-seed, annotator, or provider variation modeled. Small-sample bootstrap intervals can collapse at boundary outcomes; this does not establish certainty. Exploratory simultaneous comparisons, no multiple-testing selection claim.','context':'Both judges saw identical source-free question/options/public arguments and original answer-only system instructions. No score packet, private plans or source story provided to either judge.','provenance':{'native_jobs_sha256':complete['jobs_sha256'],'native_evaluations_sha256':complete['evaluations_sha256'],'native_manifest_sha256':complete['manifest_sha256'],'native_manifest':live.read(native/'judge_v1/manifest.json'),'haiku_manifest_sha256':live.file_hash(base/'live/haiku24_v1/manifest.json'),'script_sha256':live.file_hash(__file__)}}
    a.destination.mkdir(parents=True,exist_ok=False)
    for name,data in [('judgments',records),('accuracy',estimates),('policy_effects',effects),('judge_interactions',interactions)]:
        if data:
            with (a.destination/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    live.durable(a.destination/'summary.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,5));colors={'haiku':'#ae5b1a','native':'#216d91'};names={'haiku':'Haiku 5.5','native':'Native Qwen3.5-4B'}
    for model,offset in [('haiku',-.13),('native',.13)]:
        for i,policy in enumerate(POLICIES):
            c=next(x for x in estimates if x['group']=='display' and x['judge']==model and x['policy']==policy)
            axs[0].plot([100*c['lo95'],100*c['hi95']],[i+offset]*2,color=colors[model]);axs[0].scatter(100*c['estimate'],i+offset,color=colors[model],s=25,label=names[model] if i==0 else None)
        for i,policy in enumerate(POLICIES[1:]):
            c=next(x for x in effects if x['group']=='display' and x['judge']==model and x['policy']==policy and x['baseline']=='none')
            axs[1].plot([100*c['lo95'],100*c['hi95']],[i+offset]*2,color=colors[model]);axs[1].scatter(100*c['estimate'],i+offset,color=colors[model],s=25)
    for ax,policies in [(axs[0],POLICIES),(axs[1],POLICIES[1:])]:ax.set_yticks(range(len(policies)),policies);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axs[0].set_xlabel('Final judge accuracy (%)');axs[0].set_xlim(-3,103);axs[0].legend(loc='lower left');axs[1].set_xlabel('Paired uplift vs no feedback (percentage points)');axs[1].axvline(0,color='gray',lw=.8)
    fig.suptitle('Same 168 display debates, two unassisted final judges\n24 paired families per condition; 95% family-bootstrap intervals');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(a.destination/('display_judges.'+ext),dpi=180,bbox_inches='tight')
    fig,ax=plt.subplots(figsize=(8,4.5));nonneutral=[c for c in POLICIES if c!='neutral']
    for model,offset in [('haiku',-.13),('native',.13)]:
        for i,policy in enumerate(nonneutral):
            c=next(x for x in effects if x['group']=='display' and x['judge']==model and x['policy']==policy and x['baseline']=='neutral')
            ax.plot([100*c['lo95'],100*c['hi95']],[i+offset]*2,color=colors[model]);ax.scatter(100*c['estimate'],i+offset,color=colors[model],s=25,label=names[model] if i==0 else None)
    ax.set_yticks(range(len(nonneutral)),nonneutral);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Paired final accuracy vs neutral feedback (percentage points)');ax.legend();ax.set_title('Neutral feedback control · 24 paired families\n95% family-bootstrap intervals');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(a.destination/('display_judges_vs_neutral.'+ext),dpi=180,bbox_inches='tight')
    if not ready['prompt_skipped']:
        fig,ax=plt.subplots(figsize=(8,3.4))
        for model,offset in [('haiku',-.12),('native',.12)]:
            for i,level in enumerate(prompt.LEVELS):
                c=next(x for x in effects if x['group']=='prompt:'+level and x['judge']==model and x['baseline']=='none')
                ax.plot([100*c['lo95'],100*c['hi95']],[i+offset]*2,color=colors[model]);ax.scatter(100*c['estimate'],i+offset,color=colors[model],s=30,label=names[model] if i==0 else None)
        ax.set_yticks(range(2),prompt.LEVELS);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Feedback-on minus off final accuracy (percentage points)');ax.set_title('Same 32 fresh prompt debates; 8 paired families per level\n95% family-bootstrap intervals');ax.legend();ax.spines[['top','right']].set_visible(False);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(a.destination/('prompt_judges.'+ext),dpi=180,bbox_inches='tight')
    lines=['# Paired final judges','',result['context'],'',result['uncertainty'],'',f"Each judge evaluated {len(records)} identical debate transcripts: 168 display branches and {result['prompt_branches']} prompt branches.",'','| Population | Condition | Judge | Correct/families | Accuracy |','|---|---|---|---:|---:|']
    for c in estimates:lines.append(f"|{c['group']}|{c['policy']}|{names[c['judge']]}|{round(c['estimate']*c['families'])}/{c['families']}|{100*c['estimate']:.1f}%|")
    lines+=['','Plots show paired policy effects against no feedback. CSV/JSON also retain the neutral-control comparisons and native-minus-Haiku uplift interactions. A smaller effect for one judge is a measured interaction here, not an assumption about model strength.','', 'These participant-display comparisons occupy the public-arguments-only judge column of the historical 2×2 grid.']
    (a.destination/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'per_judge':len(records),'display':168,'prompt':result['prompt_branches'],'destination':str(a.destination)},indent=2))
if __name__=='__main__':main()
