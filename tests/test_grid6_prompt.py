import asyncio, copy, hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from probity_benchmark import grid6_prompt as prompt, grid6_live as live
from probity_benchmark.grid6_protocol import digest, MODEL
from probity_benchmark.api import object_schema, ANSWER
from probity_benchmark.preflight import load_problems
from tests import test_grid6_live as helpers

class Response(helpers.FakeResponse):
    async def text(self):
        props=self.body['response_format']['json_schema']['schema']['properties']
        if 'evidence_items' in props:parsed={'evidence_items':[{'claim':'Ada arrived','quotation':'Ada arrived.'}]*3,'anticipated_challenge':'Other answer?','response_to_challenge':'Ada is named.'}
        elif 'argument' in props:parsed={'argument':f'Fresh argument{self.event}.','answer':'A'}
        else:parsed={'answer':'A'}
        message={'role':'assistant','content':json.dumps(parsed)}
        if self.body.get('reasoning',{}).get('enabled'):message.update(reasoning='Private reasoning.',reasoning_details=[{'type':'reasoning.text','text':'Private reasoning.'}])
        return json.dumps({'id':f'unique-{self.event}','choices':[{'finish_reason':'stop','message':message}],'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.001}})
class Session(helpers.FakeSession):
    calls=[]
    def post(self,url,json):self.calls.append(copy.deepcopy(json));return Response(json,len(self.calls))

class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.out=self.root/'prompt';Session.calls=[]
        display=self.root/'display';display.mkdir();source=display/'source.jsonl'
        source.write_text(json.dumps({'synthetic_story':'SECRET SOURCE Ada arrived.','question':'Who arrived?','option_a':'Ada','option_b':'Bo','correct_answer':'A'})+'\n')
        self.problem=load_problems(source)[0][0]
        detector={'head_sha256':'head','reference_sha256':'reference','references':{MODEL:{'scores':[0.,1.]}}}
        parents=[{'policy':'none','family_id':str(i),'problem_id':self.problem.problem_id,'parent_problem_id':'parent','swapped':False,'gold':'A','detector_identity':digest(['head','reference']),'turn_schedule':[{'speaker':'Red' if j%2==0 else 'Blue','round':j//2+1,'assigned_answer':'A' if j%2==0 else 'B'} for j in range(6)]} for i in range(8)]
        spec={'supported_parameters':['structured_outputs','temperature'],'pricing':{'prompt':.0000001,'completion':.0000002,'request':0},'top_provider':{'context_length':1000000},'context_length':1000000}
        catalog=self.root/'catalog.json';live.durable(catalog,{'data':[dict(spec,id=MODEL),dict(spec,id='anthropic/claude-haiku-5.5')]})
        template=prompt.generation_parameters();template['model']='anthropic/claude-haiku-5.5';template['response_format']['json_schema']['schema']=object_schema({'answer':ANSWER});template['provider']['max_price']={'prompt':.1,'completion':.2,'request':0}
        live.durable(self.root/'haiku24_v1/manifest.json',{'model':template['model'],'request_template':template})
        auth=self.root/'auth.json';live.durable(auth,{'cap':3})
        prepbody=prompt.generation_parameters();prepbody.update(messages=[{'role':'system','content':'private prep'}],max_tokens=16384,reasoning={'enabled':True});prepbody['response_format']['json_schema']['schema']=prompt.preparation_schema();prepbody['provider']['max_price']={'prompt':.1,'completion':.2,'request':0}
        prep=live.finish_raw({'request':prepbody,'request_sha256':hashlib.sha256(json.dumps(prepbody,sort_keys=True).encode()).hexdigest(),'model':MODEL,'response_body':asyncio.run(Response(prepbody,0).text()),'http_status':200,'reserved_usd':1,'input_token_upper_bound':100000})
        preparations=self.root/'preparations.jsonl';preparations.write_text(json.dumps(prep)+'\n')
        with patch.object(live,'load_run',return_value=({}, {},parents,detector)):
            prompt.initialize(self.out,display,catalog,Path(__file__).resolve().parents[1]/'debate/debate_manager.py',auth,preparations)
        self.ledger=self.root/'budget.json';live.durable(self.ledger,{'cap_usd':60,'entries':{}})
        self.allocation=self.root/'allocation.json';live.durable(self.allocation,{'stage':prompt.SCHEMA,'execute_authorized':True,'api_cap_usd':3,'manifest_sha256':live.file_hash(self.out/'manifest.json'),'ledger_path':str(self.ledger)})
    def tearDown(self):self.tmp.cleanup()
    def step(self,n=32):
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'fake'}),patch.object(prompt.aiohttp,'ClientSession',Session):return asyncio.run(prompt.step(self.out,self.allocation,self.ledger,n))
    def scores(self,mutate=None):
        for path in (self.out/'score_requests').glob('*.json'):
            dst=self.out/'score_results'/path.name
            if dst.exists():continue
            q=live.read(path);r={k:q[k] for k in ['job_id','episode_id','matched_no_feedback_episode_id','turn_index','model','argument_sha256','source_request_sha256','response_body_sha256','head_sha256','reference_sha256']}
            r.update(schema='grid4_probe_result_v1',score_request_sha256=digest(q),raw_score=.5,percentile=50.)
            if mutate:mutate(r)
            live.durable(dst,r)
    def test_full32fresh_grid_plans_scoring_and_final_judges(self):
        self.assertEqual(self.step()['complete_plans'],32)
        self.assertEqual(self.step()['generated_public_turns'],32)
        self.assertEqual(self.step()['generated_public_turns'],32)
        for _ in range(6):self.scores();self.step()
        s=prompt.status(self.out);self.assertTrue(s['complete']);self.assertEqual(s['scored_public_turns'],192)
        self.assertEqual(len(Session.calls),256)
        self.assertEqual(len(live.read(self.ledger)['entries']),256)
        m,problems,branches,_=prompt.load(self.out)
        for b in branches:
            j=prompt.judge_job(m,self.problem,b)
            self.assertNotIn('SECRET SOURCE',str(j['request']['messages']))
            self.assertNotIn('private preparation',str(j['request']['messages']))
            self.assertEqual(len(b['turns']),6)
        self.assertEqual(len(list((self.out/'score_requests').glob('grid6-prompt-*.json'))),192)
    def test_literal_selection_and_plan_pairing(self):
        self.step();m,_,branches,_=prompt.load(self.out)
        self.assertEqual(set(m['instructions']),{'honest','selective','explicit'})
        self.assertEqual(m['preparation_parameters']['reasoning'],{'enabled':True})
        self.assertEqual(m['preparation_parameters']['max_tokens'],16384)
        for b in branches:
            job=prompt.action_job(self.out,m,self.problem,b)
            for other in branches:
                if b['pair_id']==other['pair_id']:
                    j2=prompt.action_job(self.out,m,self.problem,other)
                    self.assertEqual(job['plan_event_id'],j2['plan_event_id'])
                    self.assertEqual(job['request'],j2['request'])
                    if b['policy']!=other['policy']:self.assertNotEqual(job['event_id'],j2['event_id'])
        false_plan=next(p for p in m['plans'] if p['speaker']=='Blue' and p['level']=='selective')
        system=prompt.plan_job(m,self.problem,false_plan)['request']['messages'][0]['content']
        self.assertIn(m['instructions']['selective'],system);self.assertNotIn(m['instructions']['explicit'],system)
        self.assertNotIn('create a coherent fabricated',system)
    def test_raw_recovery_no_double_charge(self):
        self.step(1);p=next((self.out/'requests').glob('*.json'));p.unlink();self.step(0)
        self.assertTrue(p.exists());self.assertEqual(len(Session.calls),1)
    def test_wrong_score_blocks_and_waits_for_second_round_feedback(self):
        self.step();self.step();self.scores(lambda r:r.update(source_request_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.step()
        self.assertEqual(prompt.status(self.out)['scored_public_turns'],0)
    def test_unknown_intent_blocks_and_total_cap_cannot_change(self):
        m,_,_,_=prompt.load(self.out);j=prompt.plan_job(m,self.problem,m['plans'][0])
        live.durable(self.out/'intents'/f"{j['event_id']}.json",{'reserved_usd':.1})
        with self.assertRaisesRegex(RuntimeError,'in-flight'):self.step(1)
        self.assertFalse(Session.calls)
        d=live.read(self.ledger);d['cap_usd']=61;live.durable(self.ledger,d,False)
        with self.assertRaisesRegex(ValueError,'exact60'):self.step()
    def test_saved_score_or_context_corruption_blocks_resume(self):
        self.step();self.step();self.scores();self.step(0)
        path=next((self.out/'states').glob('*/turns_01.json'));b=live.read(path)
        b['turns'][0]['probe_feedback']['percentile']=99;live.durable(path,b,False)
        with self.assertRaises(ValueError):prompt.load(self.out)
    def test_stage_cap_checked_before_transport(self):
        live.durable(self.out/'intents'/'prior.json',{'reserved_usd':3})
        with self.assertRaisesRegex(RuntimeError,'cap exhausted'):self.step(1)
        self.assertFalse(Session.calls)

if __name__=='__main__':unittest.main()
