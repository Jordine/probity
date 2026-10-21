import asyncio
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from probity_benchmark import grid6_judge as judge
from probity_benchmark import grid6_live as live
from probity_benchmark.api import ANSWER,object_schema
from probity_benchmark.build_native_judge_jobs import JUDGE,OUTPUT
from tests import test_grid6_live as fixture
FakeSession=fixture.FakeSession
FakeResponse=fixture.FakeResponse

class JudgeResponse(FakeResponse):
    async def text(self):
        return json.dumps({'id':'fake-judge-'+str(self.event),'provider':'mock','choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'answer':'A'})}}],'usage':{'prompt_tokens':100,'completion_tokens':8,'cost':.001}})
class JudgeSession(FakeSession):
    calls=[]
    def post(self,url,json):
        self.calls.append(copy.deepcopy(json));return JudgeResponse(json,len(self.calls))

class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixture.LiveTests();self.fx.setUp()
        self.fx.step()
        for _ in range(3):self.fx.add_scores();self.fx.step()
        self.fx.add_scores();self.fx.step(0)
        self.root=self.fx.root;self.out=self.root/'judge';JudgeSession.calls=[]
        m,problems,branches,_=live.load_run(self.fx.out)
        template=copy.deepcopy(branches[0]['generation_parameters']);template['model']=judge.MODEL
        template['response_format']['json_schema']['schema']=object_schema({'answer':ANSWER})
        fake=judge.make_job(problems[branches[0]['problem_id']],branches[0],template)
        body=fake['request'];payload={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'answer':'A'})}}],'usage':{'prompt_tokens':100,'completion_tokens':8,'cost':.001}}
        legacy={'model':judge.MODEL,'request':body,'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'valid':True,'http_status':200,'finish_reason':'stop','response_body':json.dumps(payload),'parsed':{'answer':'A'},'usage':payload['usage']}
        receipts=self.root/'legacy.jsonl';receipts.write_text(json.dumps(legacy)+'\n')
        catalog=live.read(self.root/'catalog.json');haiku=copy.deepcopy(catalog['data'][0]);haiku['id']=judge.MODEL;catalog['data'].append(haiku);live.durable(self.root/'judge_catalog.json',catalog)
        judge.initialize(self.out,self.fx.out,self.root/'judge_catalog.json',receipts)
        self.allocation=self.root/'judge_allocation.json';live.durable(self.allocation,{'stage':'grid6_judge','execute_authorized':True,'run_manifest_sha256':live.file_hash(self.out/'manifest.json'),'ledger_path':str(self.fx.ledger),'judge_cap_usd':1.5})
    def tearDown(self):self.fx.tearDown()
    def step(self,n=7):
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'fake'}),patch.object(judge.aiohttp,'ClientSession',JudgeSession):
            return asyncio.run(judge.step(self.out,self.allocation,self.fx.ledger,n))
    def test_source_free_exact_original_prompt_and_idempotent_paid_execution(self):
        m,jobs=judge.load(self.out)
        self.assertEqual(len(jobs),7)
        for job in jobs:
            messages=job['request']['messages'];self.assertEqual(messages[0]['content'],JUDGE+OUTPUT)
            for forbidden in ['SECRET SOURCE','Private challenge','PRIVATE PAST-TURN','signal_percentile']:
                self.assertNotIn(forbidden,str(messages))
        self.assertEqual(self.step()['generated_this_step'],7)
        self.assertEqual(self.step()['generated_this_step'],0)
        self.assertEqual(len(JudgeSession.calls),7)
        self.assertAlmostEqual(live.stage_committed(self.out),.007)
        report=judge.analyze(self.out,self.root/'analysis')
        self.assertTrue(report['complete']);self.assertEqual(report['complete_grid_families'],1)
        self.assertEqual(len(report['results']),7)
        self.assertTrue(all(r['paired_minus_none']==0 for r in report['results']))
    def test_partial_grids_do_not_produce_paired_effects(self):
        self.step(1);report=judge.analyze(self.out,self.root/'partial')
        self.assertFalse(report['complete']);self.assertEqual(report['accepted_judgments'],1);self.assertEqual(report['results'],[])
    def test_judge_cap_cannot_exceed_reservation(self):
        a=live.read(self.allocation);a['judge_cap_usd']=1.51;live.durable(self.allocation,a,False)
        with self.assertRaisesRegex(ValueError,'reserved'):self.step()
        self.assertFalse(JudgeSession.calls)
    def test_raw_recovery_does_not_repeat_paid_judgment(self):
        self.step(1);path=next((self.out/'requests').glob('*.json'));path.unlink()
        self.step(0);self.assertTrue(path.exists());self.assertEqual(len(JudgeSession.calls),1)
    def test_trajectory_mismatch_receipt_rejected(self):
        self.step(1);p=next((self.out/'requests').glob('*.json'));r=live.read(p);r['tag']['trajectory_sha256']='other';live.durable(p,r,False)
        with self.assertRaisesRegex(ValueError,'another trajectory'):self.step(0)

if __name__=='__main__':unittest.main()
