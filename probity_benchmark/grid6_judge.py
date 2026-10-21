"""Source-free final Haiku judgments for completed grid6 branches; no score display.

Frozen original answer-only judge prompt/template, durable receipts, separate stage
ceiling inside the shared $60 campaign ledger. Init and analyze never call APIs.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from collections import defaultdict
import aiohttp
import numpy as np

from . import grid6_live as live
from .api import ANSWER, object_schema
from .budget import PilotBudget
from .build_native_judge_jobs import JUDGE, OUTPUT
from .grid4_generate import validate_saved_response
from .grid5_generate import audit_receipt
from .grid6_protocol import digest, POLICIES

MODEL='anthropic/claude-haiku-5.5'
SCHEMA='grid6_unassisted_judge_v1'

def make_job(problem,branch,template):
    if not branch['complete']:raise ValueError('final judge requires all six turns')
    messages,gold=problem.view(False,branch['swapped'],structured=True)
    if gold!=branch['gold']:raise ValueError('final answer mapping changed')
    messages[0]['content']=JUDGE+OUTPUT
    messages[1]['content']+='\n\nPublic debate:\n'+'\n\n'.join(f"Turn {i+1}. {t['speaker']} (answer {t['answer']}):\n"+t['argument'] for i,t in enumerate(branch['turns']))
    request={**copy.deepcopy(template),'messages':messages}
    trajectory=digest([{'speaker':t['speaker'],'answer':t['answer'],'argument':t['argument']} for t in branch['turns']])
    key=digest([SCHEMA,branch['branch_id'],trajectory,request])
    return {'job_id':key,'branch_id':branch['branch_id'],'family_id':branch['family_id'],'parent_problem_id':branch['parent_problem_id'],'policy':branch['policy'],'gold':gold,'trajectory_sha256':trajectory,'request':request}

def initialize(out,live_out,catalog,legacy_receipts):
    out=Path(out)
    if out.exists():raise ValueError('new versioned judge directory required')
    lm,_,_,_=live.load_run(live_out)
    spec=next(s for s in live.read(catalog)['data'] if s['id']==MODEL)
    rows=[json.loads(l) for l in Path(legacy_receipts).read_text().splitlines() if l]
    accepted=[r for r in rows if r.get('valid') and r['model']==MODEL]
    if not accepted:raise ValueError('missing original accepted judge request')
    legacy=accepted[0];audit_receipt(legacy)
    if legacy['request']['messages'][0]['content']!=JUDGE+OUTPUT:raise ValueError('original judge instruction differs')
    template={k:copy.deepcopy(v) for k,v in legacy['request'].items() if k!='messages'}
    expected_schema={'type':'json_schema','json_schema':{'name':'diagnostic','strict':True,'schema':object_schema({'answer':ANSWER})}}
    if template['response_format']!=expected_schema or template['reasoning']!={'enabled':False}:raise ValueError('original judge response contract differs')
    prices={k:float(spec['pricing'].get(k,0)) for k in ['prompt','completion','request']}
    template['provider']['max_price']={'prompt':prices['prompt']*1e6,'completion':prices['completion']*1e6,'request':prices['request']}
    out.mkdir(parents=True)
    for name in ['intents','raw_responses','requests','jobs','analysis']:(out/name).mkdir()
    manifest={'schema':SCHEMA,'model':MODEL,'model_spec':spec,'request_template':template,
        'live_out':str(Path(live_out).resolve()),'live_manifest_sha256':live.file_hash(Path(live_out)/'manifest.json'),
        'expected_branches':lm['branches'],'expected_families':lm['families'],'source_code_sha256':live.file_hash(__file__),
        'original_receipts_sha256':live.file_hash(legacy_receipts),'catalog_sha256':live.file_hash(catalog),
        'prompt':JUDGE+OUTPUT,'stage_limit_usd':1.5,'created_at_unix':time.time(),
        'scope':'New final unassisted judgments; no source story, private plan, role metadata, probe scores or recommendation in messages. Frozen original judge instruction/output contract.'}
    live.durable(out/'manifest.json',manifest);return {'expected_branches':lm['branches'],'model':MODEL,'paid_calls':0}

def load(out):
    out=Path(out);m=live.read(out/'manifest.json')
    if m['schema']!=SCHEMA or live.file_hash(__file__)!=m['source_code_sha256']:raise ValueError('frozen judge implementation changed')
    if live.file_hash(Path(m['live_out'])/'manifest.json')!=m['live_manifest_sha256']:raise ValueError('live source manifest changed')
    lm,problems,branches,_=live.load_run(m['live_out'])
    jobs=[make_job(problems[b['problem_id']],b,m['request_template']) for b in branches if b['complete']]
    return m,jobs

def validate_job_receipt(job,receipt):
    audit_receipt(receipt)
    if receipt['request']!=job['request'] or receipt['tag']!={'stage':SCHEMA,'job_id':job['job_id'],'branch_id':job['branch_id'],'trajectory_sha256':job['trajectory_sha256']}:raise ValueError('judge receipt belongs to another trajectory/context')
    return receipt

def finalize_raw(record):
    record=copy.deepcopy(record)
    try:validate_saved_response(record)
    except Exception as exc:
        record.update(valid=False,error=str(exc),error_type=type(exc).__name__)
        record.setdefault('cost_usd',record['reserved_usd']);record.setdefault('cost_basis','reservation_upper_bound')
    return record

def recover(out,job,ledger=None):
    out=Path(out);path=out/'requests'/f"{job['job_id']}.json";raw=out/'raw_responses'/path.name
    if not path.exists() and raw.exists():live.durable(path,finalize_raw(live.read(raw)))
    if not path.exists():return None
    receipt=live.read(path)
    if ledger:live.settle_receipt(receipt,ledger)
    if receipt.get('valid') is not True:raise RuntimeError('failed judge receipt; no automatic retry')
    return validate_job_receipt(job,receipt)

async def judge_one(out,job,manifest,allocation,ledger,session):
    out=Path(out)
    if recover(out,job,ledger) is not None:return 0
    key=job['job_id'];intent=out/'intents'/f'{key}.json'
    if intent.exists():raise RuntimeError('unresolved judge intent; no automatic retry')
    bound,upper=live.bound_request(job['request'],manifest['model_spec'])
    if live.stage_committed(out)+upper>allocation['judge_cap_usd']:raise RuntimeError('final judge stage cap exhausted')
    budget_key='grid6-judge:'+str(out.resolve())+':'+key
    if budget_key in live.read(ledger)['entries']:raise RuntimeError('orphan judge ledger reservation')
    record={'model':MODEL,'request':job['request'],'request_sha256':hashlib.sha256(json.dumps(job['request'],sort_keys=True).encode()).hexdigest(),
        'tag':{'stage':SCHEMA,'job_id':key,'branch_id':job['branch_id'],'trajectory_sha256':job['trajectory_sha256']},
        'budget_key':budget_key,'input_token_upper_bound':bound,'reserved_usd':upper,'valid':None,'attempt':0,
        'allocation_sha256':digest(allocation),'created_at_unix':time.time()}
    PilotBudget(ledger).reserve(budget_key,upper,'api');live.durable(intent,record);live.durable(out/'jobs'/f'{key}.json',job)
    start=time.monotonic()
    try:
        async with session.post('https://openrouter.ai/api/v1/chat/completions',json=job['request']) as response:
            record.update(http_status=response.status,response_body=await response.text())
        record['duration_seconds']=time.monotonic()-start;live.durable(out/'raw_responses'/f'{key}.json',record)
        record=finalize_raw(record)
    except Exception as exc:
        record.update(valid=False,error=str(exc),error_type=type(exc).__name__)
        record.setdefault('cost_usd',upper);record.setdefault('cost_basis','reservation_upper_bound')
    live.durable(out/'requests'/f'{key}.json',record);live.settle_receipt(record,ledger)
    if not record['valid']:raise RuntimeError('judge failed; receipt and conservative cost retained')
    validate_job_receipt(job,record);return 1

async def step(out,allocation_path,ledger,max_new=8,timeout=600):
    out=Path(out)
    if type(max_new) is not int or max_new<0:raise ValueError('nonnegative bounded judge count required')
    with live.lock(out):
        m,jobs=load(out);a=live.read(allocation_path)
        if a.get('stage')!='grid6_judge' or a.get('execute_authorized') is not True or a.get('run_manifest_sha256')!=live.file_hash(out/'manifest.json'):raise ValueError('missing or mismatched judge stage allocation')
        if Path(a['ledger_path']).resolve()!=Path(ledger).resolve() or live.read(ledger)['cap_usd']!=60:raise ValueError('judge must share exact $60 campaign ledger')
        cap=a['judge_cap_usd']
        if type(cap) not in (int,float) or not math.isfinite(cap) or not 0<cap<=1.5:raise ValueError('judge allocation exceeds reserved $1.5')
        pending=[]
        for job in jobs:
            if recover(out,job,ledger) is None:
                live.bound_request(job['request'],m['model_spec']);pending.append(job)
        generated=0
        if pending and max_new:
            if not os.environ.get('OPENROUTER_API_KEY'):raise RuntimeError('OPENROUTER_API_KEY unavailable')
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']}) as session:
                sem=asyncio.Semaphore(4);failed=asyncio.Event()
                async def one(job):
                    async with sem:
                        if failed.is_set():return 0
                        try:return await judge_one(out,job,m,a,ledger,session)
                        except Exception:failed.set();raise
                outcomes=await asyncio.gather(*(one(j) for j in pending[:max_new]),return_exceptions=True)
                errors=[x for x in outcomes if isinstance(x,BaseException)];generated=sum(x for x in outcomes if isinstance(x,int))
                if errors:raise RuntimeError('judge wave drained after failure: '+str(errors[0]))
        return {'generated_this_step':generated,'complete_live_branches':len(jobs),'expected_branches':m['expected_branches'],'accepted_judgments':sum((out/'requests'/f"{j['job_id']}.json").exists() for j in jobs),'judge_committed_usd':live.stage_committed(out)}

def analyze(out,destination):
    out=Path(out);destination=Path(destination);m,jobs=load(out);rows=[]
    for job in jobs:
        receipt=recover(out,job)
        if receipt is not None:rows.append({**{k:job[k] for k in ['job_id','branch_id','family_id','parent_problem_id','policy','gold']},'answer':receipt['parsed']['answer'],'correct':int(receipt['parsed']['answer']==job['gold']),'request_sha256':receipt['request_sha256']})
    grouped=defaultdict(dict)
    for r in rows:
        if r['policy'] in grouped[r['family_id']]:raise ValueError('duplicate family/policy')
        grouped[r['family_id']][r['policy']]=r
    complete={f:v for f,v in grouped.items() if set(v)==set(POLICIES)}
    results=[]
    if complete:
        families=sorted(complete);rng=np.random.default_rng(624);indices=rng.integers(len(families),size=(10000,len(families)))
        baseline=np.array([complete[f]['none']['correct'] for f in families])
        for p in POLICIES:
            x=np.array([complete[f][p]['correct'] for f in families]);d=x-baseline;boots=d[indices].mean(axis=1)
            results.append({'policy':p,'accuracy':float(x.mean()),'paired_minus_none':float(d.mean()),'lo':float(np.quantile(boots,.025)),'hi':float(np.quantile(boots,.975))})
    if destination.exists():raise ValueError('new versioned analysis destination required')
    destination.mkdir(parents=True)
    summary={'schema':'grid6_live_judge_analysis_v1','accepted_judgments':len(rows),'expected':m['expected_branches'],'complete_grid_families':len(complete),'expected_families':m['expected_families'],'complete':len(rows)==m['expected_branches'],'results':results,'seed':624,'bootstrap_samples':10000,'limits':'One shared prefix/continuation seed; historically exposed hash-selected24families; unadjusted paired family bootstrap; excludes incomplete family grids from contrasts; no factual label or local-probe claim.'}
    live.durable(destination/'summary.json',summary)
    (destination/'judgments.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if results:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        labels=[r['policy'].replace('_',' ') for r in results];y=np.arange(len(results))
        axes[0].barh(y,[100*r['accuracy'] for r in results]);axes[0].set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Unassisted judge accuracy (%)')
        ds=[100*r['paired_minus_none'] for r in results];axes[1].errorbar(ds,y,xerr=[[100*(r['paired_minus_none']-r['lo']) for r in results],[100*(r['hi']-r['paired_minus_none']) for r in results]],fmt='o');axes[1].axvline(0,color='gray',lw=1);axes[1].set(yticks=y,yticklabels=labels,xlabel='Difference from no feedback (pp), 95% interval')
        fig.suptitle(f"Qwen3.5 display screen: {len(complete)} complete families; Haiku without probe scores")
        for ext in ['png','pdf']:fig.savefig(destination/f'policy_accuracy.{ext}',dpi=180)
        plt.close(fig)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('init');a.add_argument('--out',type=Path,required=True);a.add_argument('--live',type=Path,required=True);a.add_argument('--catalog',type=Path,required=True);a.add_argument('--legacy-receipts',type=Path,required=True)
    a=s.add_parser('step');a.add_argument('--out',type=Path,required=True);a.add_argument('--allocation',type=Path,required=True);a.add_argument('--ledger',type=Path,required=True);a.add_argument('--max-new',type=int,default=8);a.add_argument('--timeout',type=float,default=600)
    a=s.add_parser('analyze');a.add_argument('--out',type=Path,required=True);a.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    if a.command=='init':r=initialize(a.out,a.live,a.catalog,a.legacy_receipts)
    elif a.command=='step':r=asyncio.run(step(a.out,a.allocation,a.ledger,a.max_new,a.timeout))
    else:r=analyze(a.out,a.destination)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
