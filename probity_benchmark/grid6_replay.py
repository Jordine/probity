"""Qwen3.5 exact-context multi-probe replay with priority live scalar scoring.

Two unpadded native forwards share all compatible heads/layers. Only response
and forced-DYL-readout token states leave the GPU, live in RAM briefly, and are
pooled/projected immediately. Output files contain scores and offsets, never
activation tensors. Provider-internal activation equivalence is not claimed.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from . import grid4_score as frozen
from .grid5_generate import audit_receipt
from .local3_annotate_units import display_units
from .prepare_debate_corpus import returned_reasoning
from .evaluate_debate_transfer import argument_mapping
from .preflight import sha256
from .qwen_extract import LAYERS, Runtime, overlapping_tokens, serialize

MODEL = 'qwen/qwen3.5-27b'
SCHEMA = 'grid6_multi_probe_replay_v1'
VERSIONS = {'torch': '2.9.1+cu128', 'transformers': '5.9.0',
            'flash-linear-attention': '0.5.2', 'causal-conv1d': '1.7.0'}
REFERENCE = 'runs/iteration3/final/response_bundles/adapted_primary.json'
FIBS_SOURCE_COMMIT = '3941138776ea8e9784769de59298b006d2b1a3ac'
FIBS_WEIGHTS_REVISION = '05def1c13a70237cda3b717d785813918b8a22fa'
HEAD_FILES = {
    'adapted': 'runs/iteration2/fitted_response_qwen35/heads.pt',
    'apollo_data': 'runs/iteration2/fitted_apollo_qwen35/heads.pt',
    'local2': 'runs/iteration2/fitted_local_qwen35_adjudicated/heads.pt',
    'local3': 'runs/iteration3/local_probes/fitted_qwen35_v2/heads.pt',
    'local3_candidates': 'runs/iteration3/local_probes/all_candidates_qwen35_v1/candidate_heads.pt',
}


def read(path):
    return json.loads(Path(path).read_text())


def check_hash(path, expected):
    if frozen.file_hash(path) != expected:
        raise ValueError('artifact identity changed: ' + str(path))


def project(values, head):
    """Frozen CPU float32 pooling upstream, float64 projection downstream."""
    if values.device.type != 'cpu' or values.shape[-1] != head['weight'].numel():
        raise ValueError('head dimensions or CPU projection contract differ')
    result = values.double() @ head['weight'].double() + head['bias']
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite probe projection')
    return result


def prepare(root, out):
    """Compile compact gzip transport from audited original receipts, no inference."""
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('prepared replay plan already exists; select a new output')
    audit_dir = root/'runs/iteration6/raw_audit'
    audit_manifest = read(audit_dir/'manifest.json')
    index_path = audit_dir/'turn_index.jsonl'
    check_hash(index_path, audit_manifest['outputs_sha256']['turn_index.jsonl'])
    index = [json.loads(line) for line in index_path.read_text().splitlines()]
    if len(index) != 1992 or len({(r['episode_id'],r['turn_index']) for r in index}) != 1992:
        raise ValueError('requires all1992 audited Qwen3.5 turns')
    reference = read(root/REFERENCE); frozen.validate_reference(reference)
    package_path = root/'runs/iteration5/judge_grid_v1/package.json'
    package = read(package_path)
    completion = read(root/'runs/iteration5/coordinator/final_analysis_complete_v1.json')
    check_hash(package_path, completion['package_sha256'])
    episodes = {}
    for name, expected in package['input_sha256'].items():
        path = Path(name)
        if path.name.startswith('episode_') and path.suffix == '.json':
            ep = read(path)
            if ep['model'] == MODEL:
                check_hash(path, expected); episodes[ep['episode_id']] = ep
    pairs = read(root/'runs/iteration5/judge_grid_v1/pairs.json')
    meta = {p[k]:p for p in pairs if p['participant_model']==MODEL for k in ['no_feedback_episode_id','feedback_episode_id']}
    expected = {}
    for arm, filename in [('A','adapted_primary_baseline_v1.json'),('B','adapted_primary_feedback_v1.json')]:
        path=root/'runs/iteration5/response_bundles'/filename
        check_hash(path,package['input_sha256'][str(path)])
        for row in read(path)['rows']:
            expected[(arm,row['episode_id'],row['turn_index'])]=row['scores'][0]
    locations = {}
    for row in index:
        locations.setdefault(row['receipt']['path'],set()).add(row['request_sha256'])
    receipts = {}
    for rel, keys in locations.items():
        path=root/rel;check_hash(path,audit_manifest['inputs_sha256'][rel])
        stream=path.open()
        texts=stream if path.suffix=='.jsonl' else [stream.read()]
        for text in texts:
            value=json.loads(text)
            if value.get('request_sha256') in keys and value.get('valid'):
                receipts[value['request_sha256']]=value
        stream.close()
    rows=[]
    for row in index:
        ep=episodes[row['episode_id']];i=row['turn_index'];turn=ep['turns'][i]
        receipt=receipts[row['request_sha256']];audit_receipt(receipt)
        message=json.loads(receipt['response_body'])['choices'][0]['message']
        request={'schema':'grid4_probe_request_v1','job_id':'grid6-offline-'+sha256(ep['episode_id']+':'+str(i)),
            'episode_id':ep['episode_id'],'matched_no_feedback_episode_id':meta[ep['episode_id']]['no_feedback_episode_id'],
            'turn_index':i,'model':MODEL,'parent_problem_id':ep['parent_problem_id'],'split':ep['split'],
            'speaker':turn['speaker'],'argument':turn['argument'],'argument_sha256':sha256(turn['argument']),
            'messages':receipt['request']['messages'],'response':message['content'],
            'response_reasoning':returned_reasoning(receipt),'source_request_sha256':receipt['request_sha256'],
            'response_body_sha256':sha256(receipt['response_body']),'source_receipt':receipt,
            'head_sha256':reference['head_sha256'],'reference_sha256':reference['reference_sha256']}
        frozen.validate_request(request,reference,MODEL)
        rows.append({'request':request,'arm':row['arm'],'family':meta[ep['episode_id']]['family'],
                     'expected_raw_score':expected[(row['arm'],ep['episode_id'],i)]})
    rows.sort(key=lambda r:r['request']['job_id'])
    path=out/'requests.jsonl.gz'
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as handle:
        for row in rows:handle.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    parity=[]
    for arm in ['A','B']:
        candidates=sorted([r for r in rows if r['arm']==arm],key=lambda r:len(json.dumps(r['request']['messages']))+len(r['request']['response']))
        parity.extend([candidates[0],candidates[-1]])
    frozen.atomic_json(out/'parity.json',{'cases':parity,'absolute_score_tolerance':frozen.PARITY_SCORE_ATOL,
                                      'require_identical_percentile':True})
    heads={}
    for name,rel in HEAD_FILES.items():
        path=root/rel; heads[name]={'path':rel,'sha256':frozen.file_hash(path)}
    fibs_dir=root/'runs/iteration6/gpu/published_fibs_qwen35_linear'
    fibs_files={str(p.relative_to(root)):frozen.file_hash(p) for p in fibs_dir.rglob('*') if p.is_file()}
    vendor=root/'runs/iteration6/gpu/probe-inference-source/src'
    vendor_files={str(p.relative_to(root)):frozen.file_hash(p) for p in vendor.rglob('*.py')}
    runtime_sources={str(p.relative_to(root)):frozen.file_hash(p) for p in [Path(__file__).resolve(),root/'probity_benchmark/grid4_score.py',root/'probity_benchmark/qwen_extract.py',root/'probity_benchmark/local3_annotate_units.py',root/'probity_benchmark/evaluate_debate_transfer.py']}
    plan={'schema':SCHEMA,'model':MODEL,'native_model':frozen.MODELS[MODEL],
          'rows':len(rows),'episodes':len(episodes),'source_families':len({r['family'] for r in rows}),
          'request_file':'requests.jsonl.gz','request_file_sha256':frozen.file_hash(out/'requests.jsonl.gz'),
          'parity_sha256':frozen.file_hash(out/'parity.json'),'heads':heads,
          'reference_path':REFERENCE,'reference_sha256':frozen.file_hash(root/REFERENCE),
          'fibs_path':str(fibs_dir.relative_to(root)),'fibs_weights_revision':FIBS_WEIGHTS_REVISION,
          'fibs_files_sha256':fibs_files,'fibs_source_commit':FIBS_SOURCE_COMMIT,
          'fibs_source_sha256':vendor_files,'runtime_sources_sha256':runtime_sources,
          'versions':VERSIONS,'no_raw_activations_persisted':True,
          'execution':'Two complete unpadded Yes./No. forwards; shared native weights and projections. Live requests receive priority.',
          'output':'Response and unit scores, period readout scores by layer, offsets/hashes/token counts. No hidden-state tensors.',
          'training':'No fitting. All heads frozen before iteration5; layer candidates remain explicitly separate candidate results.',
          'scope':'Historical B input preserves original historical feedback; alternate scores do not change the old trajectory.'}
    frozen.atomic_json(out/'plan.json',plan)
    return {k:v for k,v in plan.items() if k in ['schema','rows','episodes','source_families','no_raw_activations_persisted']}


class MultiRuntime(Runtime):
    """Capture only response tokens plus one DYL token per layer, never prompt states."""
    def __init__(self):
        super().__init__(frozen.MODELS[MODEL]['model'],frozen.MODELS[MODEL]['model_revision'])
        if sha256(self.tokenizer.chat_template)!=frozen.MODELS[MODEL]['chat_template_sha256']:
            raise ValueError('native chat template differs from original extraction')
        for handle in self.handles:handle.remove()
        self.handles=[self.model.model.layers[layer].register_forward_hook(self.capture(layer)) for layer in LAYERS]
        self.positions=None;self.capture_values={}

    def capture(self,layer):
        def hook(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            if value.ndim!=3 or value.shape[0]!=1 or value.shape[2]!=5120 or self.positions is None:
                raise ValueError('unexpected decoder activation shape')
            selected=value[0,self.positions].detach().to('cpu',copy=True)
            if not bool(torch.isfinite(selected).all()):raise ValueError('nonfinite selected states')
            self.capture_values[layer]=selected
        return hook

    @torch.inference_mode()
    def capture_serial(self,serial):
        ids=serial['ids'];pos=serial['response_positions']
        if not ids or len(ids)>self.context_length:raise ValueError('complete context exceeds native limit; no truncation permitted')
        if not pos or not 0<=min(pos)<=max(pos)<serial['dyl_position']<len(ids):
            raise ValueError('invalid response/DYL alignment')
        self.positions=[*pos,serial['dyl_position']];self.capture_values={}
        input_ids=torch.tensor([ids],device=self.model.device)
        mask=torch.ones_like(input_ids);positions=(mask.long().cumsum(-1)-1).clamp(min=0)
        try:
            self.model(input_ids=input_ids,attention_mask=mask,position_ids=positions,
                       past_key_values=None,use_cache=False,logits_to_keep=1)
            values=self.capture_values
            if set(values)!=set(LAYERS) or any(v.shape!=(len(pos)+1,5120) for v in values.values()):
                raise ValueError('incomplete selected-layer capture')
            return values
        finally:
            self.positions=None;self.capture_values={}


def project_features(yes_values,no_values,units,offsets,argument_range,heads,fibs):
    """Pure CPU projection; temporary token states never enter returned output."""
    units=copy.deepcopy(units)
    unit_indices=[overlapping_tokens(offsets,u['start'],u['end']) for u in units]
    arg_indices=overlapping_tokens(offsets,*argument_range)
    response_scores={};local_scores={};causality={}
    for name,head in heads.items():
        layer=head['layer'];feature=head['feature'];yes=yes_values[layer][:-1]
        if feature=='response_mean':values=yes.float().mean(0)
        elif feature=='argument_mean':values=yes[arg_indices].float().mean(0)
        elif feature=='dyl_no':values=no_values[layer][-1].float()
        elif feature=='dyl_difference':values=no_values[layer][-1].float()-yes_values[layer][-1].float()
        elif feature in ('span_mean','span_last'):
            pooled=torch.stack([yes[idx].float().mean(0) if feature=='span_mean' else yes[idx[-1]].float() for idx in unit_indices])
            local_scores[name]=project(pooled,head).tolist();continue
        else:raise ValueError('unsupported frozen feature '+feature)
        response_scores[name]=float(project(values,head))
    for layer in LAYERS:
        yes=yes_values[layer][:-1].float();no=no_values[layer][:-1].float()
        if yes.shape!=no.shape:raise ValueError('DYL changed public response token alignment')
        ratio=float((yes-no).square().mean().sqrt()/yes.square().mean().sqrt())
        if not math.isfinite(ratio) or ratio>.03:raise ValueError('future DYL arm materially changed original response states')
        causality[str(layer)]=ratio
    for i,unit in enumerate(units):
        unit['token_positions_within_response']=unit_indices[i]
        unit['scores']={name:values[i] for name,values in local_scores.items()}
        unit['scores']['response_broadcast']=response_scores['adapted:response_mean']
        unit['scores']['position']=unit['start']/max(end for _,end in offsets)
        unit['scores']['length']=len(unit_indices[i])
    fibs_outputs={}
    if fibs is not None:
        if tuple(fibs.used_layers)!=tuple(LAYERS) or fibs.d_model!=5120 or fibs.arch!='linear':
            raise ValueError('published FIBS head contract differs')
        mask=torch.ones(1,dtype=torch.bool)
        for arm,values in [('yes',yes_values),('no',no_values)]:
            acts={layer:values[layer][-1:] for layer in LAYERS}
            fibs_outputs[arm]=fibs.score(acts,mask)
            per_layer=fibs.layer_logits({l:v.unsqueeze(0) for l,v in acts.items()},mask.unsqueeze(0))
            fibs_outputs[arm+'_layer_logits']={str(l):float(v[0]) for l,v in per_layer.items()}
        fibs_outputs['no_minus_yes']=fibs_outputs['no']-fibs_outputs['yes']
        fibs_outputs['layer_no_minus_yes_probability']={str(l):float(torch.sigmoid(torch.tensor(fibs_outputs['no_layer_logits'][str(l)],dtype=torch.float64))-torch.sigmoid(torch.tensor(fibs_outputs['yes_layer_logits'][str(l)],dtype=torch.float64))) for l in LAYERS}
        response_scores['published_fibs:no_minus_yes']=fibs_outputs['no_minus_yes']
    return {'response_scores':response_scores,'units':units,'fibs':fibs_outputs,
            'dyl_causality_relative_rms':causality}


class MultiScorer:
    def __init__(self,root,plan):
        self.root=Path(root);self.plan=plan
        for rel,expected in {**plan['runtime_sources_sha256'],**plan['fibs_files_sha256'],**plan['fibs_source_sha256']}.items():
            check_hash(self.root/rel,expected)
        observed={k:importlib.metadata.version(k) for k in VERSIONS}
        if observed!=VERSIONS:raise ValueError('native runtime version mismatch: '+json.dumps(observed))
        check_hash(self.root/plan['reference_path'],plan['reference_sha256'])
        self.bundle=read(self.root/plan['reference_path']);frozen.validate_reference(self.bundle)
        self.contract_hash=frozen.canonical_hash(plan)
        self.heads={}
        self.head_provenance={}
        for source,artifact in plan['heads'].items():
            path=self.root/artifact['path'];check_hash(path,artifact['sha256'])
            saved=torch.load(path,weights_only=True,map_location='cpu')
            contract=saved.get('__contract__')
            if contract is not None and any(contract[k]!=frozen.MODELS[MODEL][k] for k in ['model','model_revision']):
                raise ValueError('head model/revision contract differs')
            for key,head in saved.items():
                if key.startswith('__'):continue
                if head['layer'] not in LAYERS or tuple(head['weight'].shape)!=(5120,):raise ValueError('head layer/dimension mismatch')
                if not bool(torch.isfinite(head['weight']).all()) or not math.isfinite(head['bias']):raise ValueError('nonfinite frozen head')
                name=source+':'+key;self.heads[name]=head
                self.head_provenance[name]={'source':source,'layer':head['layer'],'feature':head['feature'],'artifact_sha256':artifact['sha256']}
        self.head=frozen.load_head(self.root/plan['heads']['adapted']['path'],self.bundle,MODEL)
        sys.path.insert(0,str(self.root/'runs/iteration6/gpu/probe-inference-source/src'))
        from probe_inference import load_probe
        self.fibs=load_probe(self.root/plan['fibs_path'])
        self.runtime=MultiRuntime();self.model=MODEL
        self.provenance={**frozen.MODELS[MODEL],'selected_feature':'response_mean',
            'adapted_heads_sha256':plan['heads']['adapted']['sha256'],
            'reference_artifact_sha256':plan['reference_sha256'],'head_sha256':self.bundle['head_sha256'],
            'reference_sha256':self.bundle['reference_sha256'],'scorer_sha256':frozen.file_hash(__file__),
            'original_runtime_sha256':frozen.file_hash(Path(__file__).with_name('qwen_extract.py')),
            'gpu':torch.cuda.get_device_name(0),'versions':observed,'dtype':'bfloat16',
            'allow_bf16_reduced_precision_reduction':False,
            'execution':'Complete unpadded Yes/No teacher-forced native replay; shared six-layer hooks',
            'pooling':'CPU float32 complete-response mean then float64 linear projection, matching frozen scalar.',
            'persistent_output':'Scores/offsets/provenance only; no activations','scope':frozen.SCOPE,
            'replay_contract_sha256':self.contract_hash}

    def score(self,request):
        request_hash=frozen.validate_request(request,self.bundle,MODEL);started=time.monotonic()
        yes=serialize(self.runtime.tokenizer,request['messages'],request['response'],'Yes.',reasoning=request['response_reasoning'])
        no=serialize(self.runtime.tokenizer,request['messages'],request['response'],'No.',reasoning=request['response_reasoning'])
        if yes['ids'][:yes['followup_start']]!=no['ids'][:no['followup_start']]:raise ValueError('DYL alters original tokens')
        if yes['response_positions']!=no['response_positions']:raise ValueError('DYL changes response positions')
        argument,mapping=argument_mapping(request['response']);_,units=display_units(request['response'],['sentence','clause'])
        offsets=[(yes['offsets'][i][0]-yes['response_start'],yes['offsets'][i][1]-yes['response_start']) for i in yes['response_positions']]
        yes_values=self.runtime.capture_serial(yes);no_values=self.runtime.capture_serial(no)
        projected=project_features(yes_values,no_values,units,offsets,(mapping[0][0],mapping[-1][1]),self.heads,self.fibs)
        del yes_values,no_values
        raw=projected['response_scores']['adapted:response_mean']
        result={'schema':'grid4_probe_result_v1',**{key:request[key] for key in frozen.IDENTITY_FIELDS},
            'score_request_sha256':request_hash,'raw_score':raw,
            'percentile':frozen.percentile(raw,self.bundle['references'][MODEL]['scores']),
            'input_tokens':len(yes['ids']),'response_tokens':len(yes['response_positions']),
            'serialized_yes_sha256':sha256(yes['text']),'returned_reasoning_sha256':yes['returned_reasoning_sha256'],
            'returned_reasoning_range':yes['returned_reasoning_range'],'termination':'teacher_forced_complete_sequence',
            'elapsed_seconds':time.monotonic()-started,'provenance':self.provenance}
        multi={'schema':SCHEMA,'replay_contract_sha256':self.contract_hash,'job_id':request['job_id'],'episode_id':request['episode_id'],'turn_index':request['turn_index'],
            'score_request_sha256':request_hash,'source_request_sha256':request['source_request_sha256'],
            'argument_sha256':request['argument_sha256'],'response_body_sha256':request['response_body_sha256'],
            'messages_sha256':frozen.canonical_hash(request['messages']),
            'response_offsets':offsets,'response_token_ids':[yes['ids'][i] for i in yes['response_positions']],
            'response_absolute_token_positions':yes['response_positions'],
            'dyl_yes_position':yes['dyl_position'],'dyl_no_position':no['dyl_position'],
            'dyl_readout_token':'.','serialized_yes_sha256':sha256(yes['text']),'serialized_no_sha256':sha256(no['text']),
            'input_tokens_yes':len(yes['ids']),'input_tokens_no':len(no['ids']),
            'head_provenance':self.head_provenance,**projected}
        frozen.validate_result(result,request,self.bundle)
        return result,multi


def serve(args):
    root=Path(args.root).resolve();os.chdir(root);plan_dir=Path(args.plan).resolve();plan=read(plan_dir/'plan.json')
    check_hash(plan_dir/'requests.jsonl.gz',plan['request_file_sha256']);check_hash(plan_dir/'parity.json',plan['parity_sha256'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    worker_lock=(out/'.worker.lock').open('a')
    try:fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:raise RuntimeError('another scorer owns this output') from exc
    if time.time()+120>=args.deadline_unix:raise ValueError('insufficient time before fixed scorer deadline')
    torch.set_num_threads(4)
    scorer=MultiScorer(root,plan)
    start=time.monotonic();parity=[]
    for case in read(plan_dir/'parity.json')['cases']:
        result,multi=scorer.score(case['request']);delta=abs(result['raw_score']-case['expected_raw_score'])
        equal_percentile=result['percentile']==frozen.percentile(case['expected_raw_score'],scorer.bundle['references'][MODEL]['scores'])
        parity.append({'job_id':result['job_id'],'absolute_score_delta':delta,'identical_percentile':equal_percentile,
                       'passed':delta<=frozen.PARITY_SCORE_ATOL and equal_percentile,'result':result})
    parity_receipt={'passed':all(p['passed'] for p in parity),'cases':parity,'at_unix':time.time(),
                    'plan_sha256':frozen.file_hash(plan_dir/'plan.json')}
    parity_path=out/('parity_'+str(time.time_ns())+'.json');frozen.atomic_json(parity_path,parity_receipt)
    if not parity_receipt['passed']:raise ValueError('frozen adapted scalar parity failed; live scoring disabled')
    ready={'passed':True,'model':MODEL,'parity_receipt':str(parity_path),'plan_sha256':frozen.file_hash(plan_dir/'plan.json'),
           'provenance':scorer.provenance,'at_unix':time.time()}
    frozen.atomic_json(out/('ready_'+str(time.time_ns())+'.json'),ready)
    with gzip.open(plan_dir/'requests.jsonl.gz','rt') as stream:offline=[json.loads(line) for line in stream]
    live_requests=Path(args.live_requests) if args.live_requests else None
    live_results=Path(args.live_results) if args.live_results else None
    for name in ['scalars','multi','live_multi']:(out/name).mkdir(exist_ok=True)
    processed_offline=set();processed_live=set();position=0
    while time.time()<args.deadline_unix:
        if args.stop_file and Path(args.stop_file).exists():break
        pending=[]
        if live_requests:
            live_requests.mkdir(parents=True,exist_ok=True);live_results.mkdir(parents=True,exist_ok=True)
            pending=[p for p in sorted(live_requests.glob('*.json')) if p.stem not in processed_live]
        if pending:
            path=pending[0];request=read(path)
            if path.stem!=request['job_id']:raise ValueError('live queue filename/request mismatch')
            scalar_path=live_results/path.name;multi_path=out/'live_multi'/path.name;row=None
        else:
            if position>=len(offline):
                frozen.atomic_json(out/'offline_complete.json',{'complete':True,'rows':len(offline),
                                    'plan_sha256':frozen.file_hash(plan_dir/'plan.json')})
                if not live_requests:break
                time.sleep(2);continue
            row=offline[position];position+=1;request=row['request']
            scalar_path=out/'scalars'/(request['job_id']+'.json');multi_path=out/'multi'/scalar_path.name
        if scalar_path.exists() and multi_path.exists():
            result=read(scalar_path);multi=read(multi_path)
            frozen.validate_result(result,request,scorer.bundle)
            if multi['score_request_sha256']!=frozen.canonical_hash(request) or multi.get('replay_contract_sha256')!=scorer.contract_hash:raise ValueError('multi-probe cached context/contract differs')
        else:
            result,multi=scorer.score(request)
            if row is not None:
                delta=abs(result['raw_score']-row['expected_raw_score'])
                if delta>frozen.PARITY_SCORE_ATOL:raise ValueError('historical scalar parity drift on '+request['job_id'])
                multi.update(arm=row['arm'],family=row['family'],historical_scalar_absolute_difference=delta)
            frozen.atomic_json(multi_path,multi)
            frozen.atomic_json(scalar_path,result)
        (processed_live if pending else processed_offline).add(request['job_id'])
        status={'at_unix':time.time(),'offline_completed_this_process':len(processed_offline),
                'live_completed_this_process':len(processed_live),'offline_expected':len(offline),
                'elapsed_seconds':time.monotonic()-start,'last_job_id':request['job_id'],
                'last_input_tokens':result['input_tokens'],'last_score_seconds':result['elapsed_seconds']}
        temporary=out/'status.json.tmp';temporary.write_text(json.dumps(status)+'\n');temporary.replace(out/'status.json')
        if (len(processed_offline)+len(processed_live))%25==0:print(json.dumps(status),flush=True)
    print(json.dumps({'worker_stopped':True,'at_unix':time.time(),'offline_processed':len(processed_offline),'live_processed':len(processed_live)}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',default='.');p.add_argument('--out',required=True)
    p=sub.add_parser('serve');p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--live-requests');p.add_argument('--live-results');p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--stop-file')
    args=ap.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(args.root,args.out),indent=2))
    else:
        if bool(args.live_requests)!=bool(args.live_results):raise ValueError('live request and result directories must be supplied together')
        serve(args)

if __name__=='__main__':main()
