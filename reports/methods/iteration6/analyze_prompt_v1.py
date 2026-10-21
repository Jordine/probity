"""Offline paired descriptive summary of completed prompt ablation; fixed family resampling."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from probity_benchmark import grid6_prompt as prompt,grid6_live as live

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'runs/iteration6/live/prompt8_v1');p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    m,problems,branches,_=prompt.load(a.out)
    if not prompt.status(a.out)['complete']:raise ValueError('all32 debates and Haiku judgments required for final analysis')
    a.destination.mkdir(parents=True,exist_ok=False)
    rows=[];packets=[]
    chosen=m['families'][:2]
    for b in branches:
        if not b['complete']:continue
        j=prompt.judge_job(m,problems[b['problem_id']],b);r=prompt.recover(a.out,j)
        row={'family':b['family_id'],'branch_id':b['branch_id'],'level':b['level'],'feedback':b['policy'],'judge_correct':None if r is None else int(r['parsed']['answer']==b['gold']),'words':float(np.mean([len(t['argument'].split()) for t in b['turns']])),'max_words':max(len(t['argument'].split()) for t in b['turns']),'exact_self_repeat':int(any(t['argument'].strip()==prior['argument'].strip() for i,t in enumerate(b['turns']) for prior in b['turns'][:i] if prior['speaker']==t['speaker'])),'assigned_answer_deviations':sum(t['answer']!=t['assigned_answer'] for t in b['turns'])}
        for role in ('authentic_assignment','incorrect_assignment'):
            ts=[t for t in b['turns'] if (t['assigned_answer']==b['gold'])==(role=='authentic_assignment')]
            row[role+'_mean_percentile']=float(np.mean([t['probe_feedback']['percentile'] for t in ts]))
        rows.append(row)
        if b['family_id'] in chosen:
            turns=[]
            for t in b['turns']:
                receipt=live.read(a.out/'requests'/f"{t['event_id']}.json")
                turns.append({'turn':t,'raw_receipt_path':str(a.out/'requests'/f"{t['event_id']}.json"),'exact_messages':receipt['request']['messages'],'raw_response':json.loads(receipt['response_body'])})
            plans={}
            for plan in m['plans']:
                if plan['pair_id']==b['pair_id']:
                    job=prompt.plan_job(m,problems[plan['problem_id']],plan);plans[plan['speaker']]=prompt.recover(a.out,job)
            packets.append({'private_preparation_receipts':plans,'family':b['family_id'],'level':b['level'],'feedback':b['policy'],'sample_rule':'first 2 preregistered hash-selected families; all 4 conditions/all 6 turns','turns':turns})
    if rows:
        with (a.destination/'trajectories.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lookup={(r['family'],r['level'],r['feedback']):r for r in rows}
    complete=[f for f in m['families'] if all((f,l,c) in lookup and lookup[(f,l,c)]['judge_correct'] is not None for l in prompt.LEVELS for c in prompt.POLICIES)]
    contrasts=[];rng=np.random.default_rng(8166)
    for metric in ('judge_correct','words','exact_self_repeat','assigned_answer_deviations','authentic_assignment_mean_percentile','incorrect_assignment_mean_percentile'):
        for level in prompt.LEVELS:
            d=np.array([lookup[f,level,'bundle'][metric]-lookup[f,level,'none'][metric] for f in complete])
            if not len(d):continue
            boots=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
            contrasts.append({'metric':metric,'level':level,'contrast':'feedback_on_minus_off','families':len(d),'estimate':float(d.mean()),'lower95':float(np.percentile(boots,2.5)),'upper95':float(np.percentile(boots,97.5))})
    summary={'status':prompt.status(a.out),'complete_four_cell_families':len(complete),'family_ids':complete,'contrasts':contrasts,'uncertainty':'10,000 paired family bootstrap draws, seed 8166; 8 families maximum, one independently generated trajectory per condition; exploratory, no seed variance estimate. Scores grouped by assigned role do not constitute local factual labels.','scope':m['scope'],'qualitative_family_ids':chosen,'provenance':{'run_manifest_sha256':live.file_hash(a.out/'manifest.json'),'script_sha256':live.file_hash(__file__)}}
    live.durable(a.destination/'summary.json',summary);live.durable(a.destination/'qualitative_packets.json',packets)
    if contrasts:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(11,3.4))
        for ax,metric,label in zip(axs,('judge_correct','words','incorrect_assignment_mean_percentile'),('Judge accuracy difference (pp)','Mean words difference','Incorrect-role score difference')):
            cs=[c for c in contrasts if c['metric']==metric];scale=100 if metric=='judge_correct' else 1
            for i,c in enumerate(cs):
                v=c['estimate']*scale;ax.errorbar(v,i,xerr=[[v-c['lower95']*scale],[c['upper95']*scale-v]],fmt='o',capsize=4,color='#1f6f8b')
            ax.set_yticks(range(len(cs)),[c['level'] for c in cs]);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel(label);ax.set_title('Feedback on minus off');ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'Fresh prompt comparison: {len(complete)} paired families; 95% family bootstrap intervals',fontsize=11);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(a.destination/('paired_prompt_effects.'+ext),dpi=180,bbox_inches='tight')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
