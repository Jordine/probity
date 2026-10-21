import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probity_benchmark import grid6_live as live
from probity_benchmark.grid6_protocol import MODEL,POLICIES,PROTOCOL,make_branch,digest,next_request
from probity_benchmark.preflight import load_problems

class FakeResponse:
    status=200
    def __init__(self,body,event):self.body=body;self.event=event
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    async def text(self):
        return json.dumps({'id':'fake-event-'+str(self.event),'provider':'test-only',
            'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'argument':'Fresh argument '+str(self.event)+'.','answer':'A'})}}],
            'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.001}})
class FakeSession:
    calls=[]
    def __init__(self,*args,**kwargs):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    def post(self,url,json):
        self.calls.append(copy.deepcopy(json));return FakeResponse(json,len(self.calls))

class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);FakeSession.calls=[]
        source=self.root/'source.jsonl'
        source.write_text(json.dumps({'synthetic_story':'SECRET SOURCE Ada arrived.','question':'Who arrived?', 'option_a':'Ada','option_b':'Bo','correct_answer':'A'})+'\n')
        self.problem=load_problems(source)[0][0]
        detector={'head_sha256':'head','reference_sha256':'reference','references':{MODEL:{'scores':[0.,1.]}}}
        bundle=self.root/'bundle.json';live.durable(bundle,detector)
        identity=digest(['head','reference'])
        prep={'evidence_items':[],'anticipated_challenge':'Private challenge','response_to_challenge':'Private response','request_sha256':'prep'}
        original={'episode_id':'old','problem_id':self.problem.problem_id,'parent_problem_id':'parent','model':MODEL,'complete':True,'protocol':'explicit_reasoned_evidence_sequential','gold':'A','swapped':False,'private_preparations':{'Red':prep,'Blue':copy.deepcopy(prep)},'turns':[]}
        for i in range(6):
            text='Old argument '+str(i)+'.';original['turns'].append({'speaker':'Red' if i%2==0 else 'Blue','round':i//2+1,'assigned_answer':'A' if i%2==0 else 'B','answer':'A' if i%2==0 else 'B','argument':text,'request_sha256':'old'+str(i),'probe_feedback':{'turn_index':i,'source_request_sha256':'old'+str(i),'argument_sha256':hashlib.sha256(text.encode()).hexdigest(),'detector_identity':identity,'percentile':50}})
        drafts=self.root/'drafts';jobs=[]
        for policy in POLICIES:
            b=make_branch(original,policy,identity,'family',self.problem)
            live.durable(drafts/'branches'/f"{b['branch_id']}.json",{'branch':b});jobs.append({'branch_id':b['branch_id']})
        live.durable(drafts/'manifest.json',{'schema':PROTOCOL,'jobs':jobs,'branches':7})
        catalog=self.root/'catalog.json';live.durable(catalog,{'data':[{'id':MODEL,'supported_parameters':['structured_outputs','temperature'],'pricing':{'prompt':.0000001,'completion':.0000002,'request':0},'top_provider':{'context_length':1000000},'context_length':1000000}]})
        self.out=self.root/'live';live.initialize(self.out,drafts,source,bundle,catalog)
        self.ledger=self.root/'budget.json';live.durable(self.ledger,{'cap_usd':60.0,'entries':{}})
        self.allocation=self.root/'allocation.json';live.durable(self.allocation,{'stage':'grid6_live','execute_authorized':True,'run_manifest_sha256':live.file_hash(self.out/'manifest.json'),'ledger_path':str(self.ledger),'generation_cap_usd':2})
    def tearDown(self):self.tmp.cleanup()
    def step(self,n=7):
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'fake-not-a-key'}),patch.object(live.aiohttp,'ClientSession',FakeSession):
            return asyncio.run(live.step(self.out,self.allocation,self.ledger,n))
    def add_scores(self,mutate=None):
        for p in (self.out/'score_requests').glob('*.json'):
            q=live.read(p);path=self.out/'score_results'/p.name
            if path.exists():continue
            result={k:q[k] for k in ['job_id','episode_id','matched_no_feedback_episode_id','turn_index','model','argument_sha256','source_request_sha256','response_body_sha256','head_sha256','reference_sha256']}
            result.update(schema='grid4_probe_result_v1',score_request_sha256=digest(q),raw_score=.5,percentile=50.)
            if mutate:mutate(result)
            live.durable(path,result)
    def test_full_grid_waits_for_scores_and_preserves_exact_prefix(self):
        first=self.step();self.assertEqual(first['generated_this_step'],7);self.assertEqual(first['scored_suffix_turns'],0)
        self.assertEqual(self.step()['generated_this_step'],0);self.assertEqual(len(FakeSession.calls),7)
        for _ in range(3):self.add_scores();self.step()
        self.add_scores();done=self.step(0)
        self.assertTrue(done['complete']);self.assertEqual(done['scored_suffix_turns'],28);self.assertEqual(len(FakeSession.calls),28)
        _,_,branches,_=live.load_run(self.out);self.assertEqual(len({digest(b['turns'][:2]) for b in branches}),1)
        self.assertEqual(len({r['receipt']['response_body'] for b in branches for r in b['continuation_receipts']}),28)
        entries=live.read(self.ledger)['entries'];self.assertEqual(len(entries),28);self.assertTrue(all(e['status']=='settled' for e in entries.values()))
        self.assertAlmostEqual(sum(e['amount_usd'] for e in entries.values()),.028)
        dest=self.root/'exports';self.assertEqual(live.export(self.out,dest)['judge_jobs'],7)
        jobs=[json.loads(l) for l in (dest/'judge_jobs.jsonl').read_text().splitlines()]
        for j in jobs:
            self.assertNotIn('SECRET SOURCE',str(j['messages']));self.assertNotIn('PRIVATE PAST-TURN',str(j['messages']))
    def test_generation_receipts_are_not_reused_across_logical_branches(self):
        self.step();records=[live.read(p) for p in (self.out/'requests').glob('*.json')]
        self.assertEqual(len({r['logical_id'] for r in records}),7)
        self.assertEqual(len({r['tag']['branch_id'] for r in records}),7)
    def test_raw_receipt_recovery_does_not_pay_twice(self):
        self.step(1);path=next((self.out/'requests').glob('*.json'));path.unlink()
        self.step(0);self.assertTrue(path.exists());self.assertEqual(len(FakeSession.calls),1)
        self.assertEqual(len(live.read(self.ledger)['entries']),1)
    def test_unresolved_intent_blocks_before_transport(self):
        _,_,branches,_=live.load_run(self.out);b=branches[0]
        live.durable(self.out/'intents'/f'{live.logical_id(b)}.json',{'reserved_usd':.01})
        with self.assertRaisesRegex(RuntimeError,'unresolved generation intent'):self.step(1)
        self.assertFalse(FakeSession.calls)
    def test_wrong_context_score_cannot_advance(self):
        self.step(1);self.add_scores(lambda r:r.update(source_request_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.step(0)
        self.assertEqual(live.status(self.out)['scored_suffix_turns'],0)
    def test_changed_stage_manifest_or_total_cap_blocks(self):
        a=live.read(self.allocation);a['run_manifest_sha256']='changed';live.durable(self.allocation,a,False)
        with self.assertRaisesRegex(ValueError,'does not pin'):self.step()
        self.assertFalse(FakeSession.calls)
    def test_unknown_transport_liability_blocks_retry(self):
        class Failing(FakeSession):
            def post(self,*args,**kwargs):raise TimeoutError('test timeout')
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'fake'}),patch.object(live.aiohttp,'ClientSession',Failing):
            with self.assertRaisesRegex(RuntimeError,'failure'):asyncio.run(live.step(self.out,self.allocation,self.ledger,1))
        e=next(iter(live.read(self.ledger)['entries'].values()));self.assertEqual(e['status'],'settled');self.assertGreater(e['amount_usd'],0)
        with self.assertRaisesRegex(RuntimeError,'saved generation failure'):self.step(1)
        self.assertFalse(FakeSession.calls)
    def test_stage_and_total_budget_exhaustion_never_send(self):
        a=live.read(self.allocation);a['generation_cap_usd']=.0000001;live.durable(self.allocation,a,False)
        with self.assertRaisesRegex(RuntimeError,'allocation exhausted'):self.step(1)
        self.assertFalse(FakeSession.calls);self.assertFalse(list((self.out/'intents').glob('*.json')))
        a['generation_cap_usd']=2;live.durable(self.allocation,a,False)
        ledger=live.read(self.ledger);ledger['entries']['other-stage']={'status':'settled','amount_usd':59.999999};live.durable(self.ledger,ledger,False)
        with self.assertRaisesRegex(RuntimeError,'cumulative pilot budget'):self.step(1)
        self.assertFalse(FakeSession.calls);self.assertFalse(list((self.out/'intents').glob('*.json')))
    def test_total_campaign_ceiling_cannot_be_raised(self):
        ledger=live.read(self.ledger);ledger['cap_usd']=61;live.durable(self.ledger,ledger,False)
        with self.assertRaisesRegex(ValueError,'TOTAL campaign ceiling'):self.step(1)
        self.assertFalse(FakeSession.calls)
    def test_full_context_bound_is_checked(self):
        manifest,problems,branches,_=live.load_run(self.out)
        spec=copy.deepcopy(manifest['model_spec']);spec['top_provider']['context_length']=100
        with self.assertRaisesRegex(ValueError,'full context'):live.bound_request(next_request(self.problem,branches[0]),spec)

if __name__=='__main__':unittest.main()
