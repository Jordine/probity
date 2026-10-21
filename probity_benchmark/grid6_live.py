"""Durable bounded shared-prefix live debate executor.

No paid work on init/status/ingest/export. A step needs coordinator stage allocation,
shares the campaign ledger, writes a durable intent before POST, and never retries
an unresolved action. Each generated suffix turn is scored before the next turn.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import aiohttp
from .api import strict_json
from .budget import PilotBudget
from .grid4_generate import validate_saved_response, validate_score
from .grid5_generate import audit_receipt
from .grid6_protocol import (MODEL, PROTOCOL, STATIC_FIELDS, POLICIES, digest, publish,
                            next_request, continuation_tag, append_turn, validate_branch)
from .prepare_debate_corpus import returned_reasoning
from .preflight import load_problems

SCHEMA='grid6_live_executor_v1'

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return strict_json(Path(path).read_text())
def durable(path,value,immutable=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        if read(path)!=value:raise ValueError('immutable artifact differs: '+str(path))
        return
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

@contextmanager
def lock(out):
    with (Path(out)/'.executor.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def logical_id(branch):return digest(['grid6-live-event',continuation_tag(branch)])
def bound_request(body,spec):
    schema=body['response_format']['json_schema']['schema']
    bound=len(json.dumps([body['messages'],schema],ensure_ascii=False).encode())+2048
    context=spec['top_provider'].get('context_length') or spec['context_length']
    if bound+body['max_tokens']>context:raise ValueError('full context plus completion allowance exceeds frozen model context')
    p=body['provider']['max_price']
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in p.values()):raise ValueError('invalid price ceilings')
    upper=bound*p['prompt']/1e6+body['max_tokens']*p['completion']/1e6+p.get('request',0)
    if upper<=0:raise ValueError('positive reservation required')
    return bound,upper

def initialize(out,drafts,source,bundle_path,catalog_path):
    out,drafts=Path(out),Path(drafts)
    if out.exists():raise ValueError('new versioned live output directory required')
    draft_manifest=read(drafts/'manifest.json')
    if draft_manifest['schema']!=PROTOCOL:raise ValueError('draft protocol differs')
    problems,audit=load_problems(source)
    if audit['errors']:raise ValueError('invalid source')
    problems={p.problem_id:p for p in problems}
    catalog=read(catalog_path)
    specs={x['id']:x for x in catalog['data']} if 'data' in catalog else catalog['models']
    spec=specs[MODEL]
    if not {'structured_outputs','temperature'}<=set(spec['supported_parameters']):raise ValueError('generation contract parameters unsupported')
    prices={k:float(spec['pricing'].get(k,0)) for k in ['prompt','completion','request']}
    ceiling={'prompt':prices['prompt']*1e6,'completion':prices['completion']*1e6,'request':prices['request']}
    detector=read(bundle_path);identity=digest([detector['head_sha256'],detector['reference_sha256']])
    initial=[];roots={};input_hashes={str(Path(p).resolve()):file_hash(p) for p in [source,bundle_path,catalog_path,drafts/'manifest.json']}
    for job in draft_manifest['jobs']:
        path=drafts/'branches'/f"{job['branch_id']}.json";input_hashes[str(path.resolve())]=file_hash(path)
        original=read(path)['branch'];branch=copy.deepcopy(original)
        validate_branch(problems[branch['problem_id']],branch)
        if branch['detector_identity']!=identity:raise ValueError('detector differs from draft shared-prefix scores')
        if len(branch['turns'])!=2:raise ValueError('draft contains a continuation')
        branch['generation_parameters']['provider']['max_price']=ceiling
        branch['contract_sha256']=digest({k:branch[k] for k in STATIC_FIELDS})
        branch['branch_id']=digest([PROTOCOL,branch['contract_sha256']])
        validate_branch(problems[branch['problem_id']],branch)
        bound_request(next_request(problems[branch['problem_id']],branch),spec)
        initial.append(branch)
        roots.setdefault(branch['parent_episode_id'],[]).append(branch)
    if len(initial)!=draft_manifest['branches'] or len({b['branch_id'] for b in initial})!=len(initial):raise ValueError('branch coverage mismatch')
    for branches in roots.values():
        if set(b['policy'] for b in branches)!=set(POLICIES) or len({b['shared_prefix_sha256'] for b in branches})!=1:raise ValueError('incomplete or mismatched shared-prefix grid')
    out.mkdir(parents=True)
    (out/'source.jsonl').write_bytes(Path(source).read_bytes())
    durable(out/'detector.json',detector)
    for sub in ['states','intents','raw_responses','requests','score_requests','score_results','exports']:(out/sub).mkdir()
    for branch in initial:durable(out/'states'/branch['branch_id']/'turns_02.json',branch)
    dependencies=['grid6_live.py','grid6_protocol.py','grid4_generate.py','grid5_generate.py','environment3_evidence.py','budget.py','api.py','prepare_debate_corpus.py']
    manifest={'schema':SCHEMA,'protocol':PROTOCOL,'created_at_unix':time.time(),'input_sha256':input_hashes,
        'source_sha256':file_hash(out/'source.jsonl'),'detector_sha256':file_hash(out/'detector.json'),
        'source_code_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in dependencies},
        'model_spec':spec,'model':MODEL,'branches':len(initial),'families':len(roots),'policies':list(POLICIES),
        'branch_ids':[b['branch_id'] for b in initial],'new_turns_required':4*len(initial),
        'initial_sha256':{b['branch_id']:file_hash(out/'states'/b['branch_id']/'turns_02.json') for b in initial},
        'causal_contract':'Every suffix turn including control arms waits for its exact contextual scalar score before next generation.',
        'api_contract':'Frozen catalog-price ceiling; unpinned provider routing. Full raw provider envelopes retained, no hidden reasoning recovery claim.',
        'scope':'Exploratory shared-prefix display screen; no historical suffix reuse; source/dev exposure retained.'}
    durable(out/'manifest.json',manifest)
    return status(out)

def load_run(out):
    out=Path(out);manifest=read(out/'manifest.json')
    if manifest['schema']!=SCHEMA:raise ValueError('unknown executor schema')
    for name,h in manifest['source_code_sha256'].items():
        if file_hash(Path(__file__).with_name(name))!=h:raise ValueError('frozen executor dependency changed: '+name)
    if file_hash(out/'source.jsonl')!=manifest['source_sha256'] or file_hash(out/'detector.json')!=manifest['detector_sha256']:raise ValueError('frozen runtime input changed')
    problems,audit=load_problems(out/'source.jsonl')
    if audit['errors']:raise ValueError('invalid frozen source')
    problems={p.problem_id:p for p in problems};branches=[]
    for bid in manifest['branch_ids']:
        directory=out/'states'/bid
        if file_hash(directory/'turns_02.json')!=manifest['initial_sha256'][bid]:raise ValueError('initial branch changed')
        states=sorted(directory.glob('turns_*.json'))
        if [int(p.stem[-2:]) for p in states]!=list(range(2,2+len(states))):raise ValueError('noncontiguous immutable branch state history')
        branch=read(states[-1]);validate_branch(problems[branch['problem_id']],branch)
        if branch['branch_id']!=bid:raise ValueError('state belongs to another branch')
        branches.append(branch)
    return manifest,problems,branches,read(out/'detector.json')

def scoring_request(branch,receipt,detector):
    audit_receipt(receipt);body=receipt['request']
    if receipt['tag']!=continuation_tag(branch):raise ValueError('receipt does not belong to current state')
    message=json.loads(receipt['response_body'])['choices'][0]['message']
    index=len(branch['turns']);step=branch['turn_schedule'][index]
    return {'schema':'grid4_probe_request_v1','job_id':'grid6-'+logical_id(branch),
        'episode_id':branch['branch_id'],'matched_no_feedback_episode_id':branch['parent_episode_id'],
        'turn_index':index,'model':MODEL,'parent_problem_id':branch['parent_problem_id'],'split':'dev',
        'speaker':step['speaker'],'argument':receipt['parsed']['argument'],
        'argument_sha256':hashlib.sha256(receipt['parsed']['argument'].encode()).hexdigest(),
        'messages':copy.deepcopy(body['messages']),'response':message['content'],
        'response_reasoning':returned_reasoning(receipt),'source_request_sha256':receipt['request_sha256'],
        'response_body_sha256':hashlib.sha256(receipt['response_body'].encode()).hexdigest(),
        'source_receipt':copy.deepcopy(receipt),'head_sha256':detector['head_sha256'],'reference_sha256':detector['reference_sha256']}

def finish_raw(record):
    record=copy.deepcopy(record)
    try:
        validate_saved_response(record)
        returned_reasoning(record)
    except Exception as exc:
        record.update(valid=False,error=str(exc),error_type=type(exc).__name__)
        record.setdefault('cost_usd',record['reserved_usd']);record.setdefault('cost_basis','reservation_upper_bound')
    return record

def settle_receipt(receipt,ledger_path):
    ledger=PilotBudget(ledger_path);state=read(ledger_path)['entries'].get(receipt['budget_key'])
    if state is None:raise ValueError('paid receipt lacks a ledger reservation')
    if state['status']=='reserved':ledger.settle(receipt['budget_key'],receipt['cost_usd'],'Grid6 full raw action receipt; failures conservatively charged, no automatic retry.')
    elif state['status']!='settled' or abs(state['amount_usd']-receipt['cost_usd'])>1e-9:raise ValueError('receipt cost differs from ledger settlement')

def recover_action(out,branch,ledger_path=None):
    out=Path(out);key=logical_id(branch);path=out/'requests'/f'{key}.json';raw=out/'raw_responses'/f'{key}.json'
    if not path.exists() and raw.exists():durable(path,finish_raw(read(raw)))
    if not path.exists():return None
    receipt=read(path)
    if ledger_path:settle_receipt(receipt,ledger_path)
    if receipt.get('valid') is not True:raise RuntimeError('saved generation failure; no automatic retry: '+str(path))
    audit_receipt(receipt)
    if receipt['tag']!=continuation_tag(branch):raise ValueError('saved event state mismatch')
    return receipt

def ingest(out,ledger_path=None):
    out=Path(out);manifest,problems,branches,detector=load_run(out);advanced=0
    for branch in branches:
        if branch['complete']:continue
        receipt=recover_action(out,branch,ledger_path)
        if receipt is None:continue
        if receipt['request']!=next_request(problems[branch['problem_id']],branch):raise ValueError('saved exact action request differs')
        request=scoring_request(branch,receipt,detector);durable(out/'score_requests'/f"{request['job_id']}.json",request)
        path=out/'score_results'/f"{request['job_id']}.json"
        if not path.exists():continue
        result=read(path);validate_score(request,result,detector)
        enriched={**result,'branch_id':branch['branch_id'],'detector_identity':branch['detector_identity'],'messages_sha256':digest(receipt['request']['messages'])}
        updated=append_turn(problems[branch['problem_id']],branch,receipt,enriched)
        durable(out/'states'/branch['branch_id']/f"turns_{len(updated['turns']):02d}.json",updated);advanced+=1
    return advanced

def status(out):
    out=Path(out);manifest,_,branches,_=load_run(out)
    suffix=sum(len(b['turns'])-2 for b in branches)
    return {'schema':SCHEMA,'branches':len(branches),'complete_branches':sum(b['complete'] for b in branches),
        'scored_suffix_turns':suffix,'required_suffix_turns':manifest['new_turns_required'],
        'generation_receipts':len(list((out/'requests').glob('*.json'))),
        'score_requests':len(list((out/'score_requests').glob('*.json'))),
        'score_results':len(list((out/'score_results').glob('*.json'))),
        'complete':all(b['complete'] for b in branches)}

def check_allocation(out,allocation_path,ledger_path):
    allocation=read(allocation_path);ledger=read(ledger_path)
    if allocation.get('stage')!='grid6_live' or allocation.get('execute_authorized') is not True:raise ValueError('coordinator live stage allocation missing')
    if allocation.get('run_manifest_sha256')!=file_hash(Path(out)/'manifest.json'):raise ValueError('allocation does not pin live run')
    if Path(allocation['ledger_path']).resolve()!=Path(ledger_path).resolve():raise ValueError('allocation ledger differs')
    cap=allocation['generation_cap_usd']
    if type(cap) not in (int,float) or not math.isfinite(cap) or cap<=0 or cap>60:raise ValueError('invalid generation allocation')
    if ledger['cap_usd']!=60.0:raise ValueError('iteration6 hard TOTAL campaign ceiling must be $60')
    return allocation

def stage_committed(out):
    total=0.
    for path in (Path(out)/'intents').glob('*.json'):
        intent=read(path);receipt=Path(out)/'requests'/path.name
        total+=(read(receipt).get('cost_usd',intent['reserved_usd']) if receipt.exists() else intent['reserved_usd'])
    return total

async def generate_action(out,branch,problem,manifest,allocation,ledger_path,session):
    out=Path(out);key=logical_id(branch);body=next_request(problem,branch)
    if recover_action(out,branch,ledger_path) is not None:return False
    intent=out/'intents'/f'{key}.json'
    if intent.exists():raise RuntimeError('unresolved generation intent; reconcile before any retry')
    bound,upper=bound_request(body,manifest['model_spec'])
    if stage_committed(out)+upper>allocation['generation_cap_usd']:raise RuntimeError('generation stage allocation exhausted')
    budget_key='grid6-live:'+str(out.resolve())+':'+key
    if budget_key in read(ledger_path)['entries']:raise RuntimeError('orphaned ledger reservation; no automatic retry')
    record={'tag':continuation_tag(branch),'model':MODEL,'request':body,
        'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
        'logical_id':key,'assigned_answer':branch['turn_schedule'][len(branch['turns'])]['assigned_answer'],
        'reserved_usd':upper,'input_token_upper_bound':bound,'budget_key':budget_key,
        'valid':None,'attempt':0,'allocation_sha256':digest(allocation),'created_at_unix':time.time()}
    PilotBudget(ledger_path).reserve(budget_key,upper,'api')
    durable(intent,record)
    start=time.monotonic()
    try:
        async with session.post('https://openrouter.ai/api/v1/chat/completions',json=body) as response:
            record.update(http_status=response.status,response_body=await response.text())
        record['duration_seconds']=time.monotonic()-start
        durable(out/'raw_responses'/f'{key}.json',record)
        record=finish_raw(record)
    except Exception as exc:
        record.update(valid=False,error=str(exc),error_type=type(exc).__name__,duration_seconds=time.monotonic()-start)
        record.setdefault('cost_usd',upper);record.setdefault('cost_basis','reservation_upper_bound')
    durable(out/'requests'/f'{key}.json',record);settle_receipt(record,ledger_path)
    if not record['valid']:raise RuntimeError('generation failed; raw receipt/upper liability retained without retry')
    return True

async def step(out,allocation_path,ledger_path,max_new=8,timeout=600):
    if type(max_new) is not int or max_new<0:raise ValueError('nonnegative bounded action count required')
    out=Path(out)
    with lock(out):
        allocation=check_allocation(out,allocation_path,ledger_path)
        advanced=ingest(out,ledger_path)
        manifest,problems,branches,detector=load_run(out)
        ready=[b for b in branches if not b['complete'] and not (out/'requests'/f'{logical_id(b)}.json').exists()]
        generated=0
        if ready and max_new:
            if not os.environ.get('OPENROUTER_API_KEY'):raise RuntimeError('OPENROUTER_API_KEY unavailable')
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']}) as session:
                # Four simultaneous HTTP calls; synchronous intent/reservation publication precedes every await.
                sem=asyncio.Semaphore(4);failed=asyncio.Event()
                async def one(branch):
                    async with sem:
                        if failed.is_set():return 0
                        try:
                            made=await generate_action(out,branch,problems[branch['problem_id']],manifest,allocation,ledger_path,session)
                            request=scoring_request(branch,read(out/'requests'/f'{logical_id(branch)}.json'),detector)
                            durable(out/'score_requests'/f"{request['job_id']}.json",request)
                            return int(made)
                        except Exception:
                            failed.set();raise
                outcomes=await asyncio.gather(*(one(b) for b in ready[:max_new]),return_exceptions=True)
                errors=[e for e in outcomes if isinstance(e,BaseException)]
                generated=sum(e for e in outcomes if isinstance(e,int))
                if errors:raise RuntimeError('bounded wave stopped after failure; all started calls drained and saved: '+str(errors[0]))
        return {**status(out),'advanced_this_step':advanced,'generated_this_step':generated,'generation_committed_usd':stage_committed(out)}

def export(out,destination):
    """Export completed transcripts and answer-only judge jobs, without paying."""
    from .build_native_judge_jobs import JUDGE, OUTPUT
    from .api import ANSWER,object_schema
    out=Path(out);destination=Path(destination);manifest,problems,branches,detector=load_run(out)
    if not all(b['complete'] for b in branches):raise ValueError('all branches must be complete before canonical export')
    if destination.exists():raise ValueError('new versioned export directory required')
    destination.mkdir(parents=True);episodes=[]
    for b in branches:
        e={**b,'episode_id':b['branch_id'],'protocol':PROTOCOL,'parent_problem_id':b['parent_problem_id'],'split':'dev'}
        episodes.append(e);durable(destination/'episodes'/f"episode_{e['episode_id']}.json",e)
    jobs=[]
    for e in episodes:
        messages,gold=problems[e['problem_id']].view(False,e['swapped'],structured=True)
        if gold!=e['gold']:raise ValueError('judge gold mapping changed')
        messages[0]['content']=JUDGE+OUTPUT
        transcript='\n\n'.join(f"Turn {i+1}. {t['speaker']} (answer {t['answer']}):\n"+t['argument'] for i,t in enumerate(e['turns']))
        messages[1]['content']+='\n\nPublic debate:\n'+transcript
        jobs.append({'job_id':digest([e['episode_id'],'unassisted-final-judge',messages]),
            'episode_id':e['episode_id'],'branch_id':e['branch_id'],'policy':e['policy'],'family':e['family_id'],
            'participant_model':MODEL,'kind':'none','group':e['parent_problem_id'],'split':'dev','gold':e['gold'],'messages':messages,'schema':object_schema({'answer':ANSWER}),
            'argument_sha256':[hashlib.sha256(t['argument'].encode()).hexdigest() for t in e['turns']]})
    with (destination/'judge_jobs.jsonl').open('w') as f:
        for row in jobs:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    durable(destination/'manifest.json',{'schema':'grid6_live_export_v1','source_manifest_sha256':file_hash(out/'manifest.json'),'branches':len(episodes),'judge_jobs_sha256':file_hash(destination/'judge_jobs.jsonl'),'scope':'Unassisted judges only; no source/private plans/scores in messages; native and API runners may consume identical jobs.'})
    return {'episodes':len(episodes),'judge_jobs':len(jobs)}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('init');a.add_argument('--out',type=Path,required=True);a.add_argument('--drafts',type=Path,required=True);a.add_argument('--source',type=Path,required=True);a.add_argument('--bundle',type=Path,required=True);a.add_argument('--catalog',type=Path,required=True)
    for name in ['status','ingest']:
        a=sub.add_parser(name);a.add_argument('--out',type=Path,required=True)
        if name=='ingest':a.add_argument('--ledger',type=Path)
    a=sub.add_parser('step');a.add_argument('--out',type=Path,required=True);a.add_argument('--allocation',type=Path,required=True);a.add_argument('--ledger',type=Path,required=True);a.add_argument('--max-new',type=int,default=8);a.add_argument('--timeout',type=float,default=600)
    a=sub.add_parser('export');a.add_argument('--out',type=Path,required=True);a.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    if a.command=='init':result=initialize(a.out,a.drafts,a.source,a.bundle,a.catalog)
    elif a.command=='status':result=status(a.out)
    elif a.command=='ingest':
        with lock(a.out):result={'advanced':ingest(a.out,a.ledger),**status(a.out)}
    elif a.command=='export':result=export(a.out,a.destination)
    else:result=asyncio.run(step(a.out,a.allocation,a.ledger,a.max_new,a.timeout))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
