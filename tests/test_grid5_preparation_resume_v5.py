"""Proposal-only v5 checks: isolated fake transport, no real authorization or POST."""
import asyncio
from collections import Counter
import copy
import importlib.util
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from probity_benchmark import api
from tests.test_grid5_preparation_liability import World

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r5_tests',ROOT/'runs/iteration5/coordinator/preparation_resume_v5.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);g=r.load_liability()

class V5RealHistory(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  ps,rows,_,_=r.generation.frozen_inventory(r.SOURCE,r.INVENTORY,r.INVENTORY_SHA)
  cls.jobs=r.make_jobs(ps,rows,r.read(r.RUN/'preparations_v1/api_manifest.json')['models'])
  cls.history=r.audit_history([r.RUN/f'preparations_v{i}' for i in range(1,9)],cls.jobs,r.read(r.META/'budget_before_resume_v5.json'))
 def test_real_history_preserves_560_and_nine_unknowns(self):
  self.assertEqual(r.stopped_history(self.jobs),r.read(r.HISTORY_MANIFEST))
  self.assertEqual((len(self.history.canonical),len(self.history.missing)),(560,8))
  r.verify_canonical(self.history,r.read(r.HISTORY_MANIFEST))
  policy=r.read(r.LIABILITY_POLICY)
  with patch.object(g,'load_policy',return_value=policy):
   report=g.validate_liability_ledger(r.META/'budget_before_resume_v5.json',r.LIABILITY_POLICY)
  self.assertEqual(report['unknown_liability_entries'],9)
  self.assertEqual(report['additional_liability_usd'],5.750784)
  self.assertAlmostEqual(report['preparation_committed_usd'],12.342478856)
  self.assertEqual(r.batch_budget(self.history,report['additional_liability_usd']),1.65)
 def test_eight_routing_changes_only_and_560_original_bodies(self):
  planned=r.routed_jobs(self.jobs);missing=set(r.route_proofs())
  self.assertEqual(missing,self.history.missing)
  for key,job in planned.items():
   old=self.jobs[key]
   if key not in missing:self.assertEqual(job,old)
   else:
    restored=copy.deepcopy(job['request'])
    self.assertEqual(restored['provider'].pop('order'),['deepinfra/fp8'])
    self.assertIs(restored['provider'].pop('allow_fallbacks'),False)
    self.assertEqual(restored,old['request'])
    self.assertEqual(job['request_sha256'],r.route_proofs()[key]['candidate_request_sha256'])
 def test_prior_full_lineage_and_fail_closed_draft(self):
  r.prior.verify_freezes(r.PRIOR_RECOVERY_FREEZE_SHA)
  with TemporaryDirectory() as tmp:
   p=Path(tmp)/'policy.json';v=r.read(r.LIABILITY_POLICY);v['ready']=False
   p.write_text(json.dumps(v))
   with self.assertRaisesRegex(ValueError,'unapproved'):g.load_policy(p)
   v['ready']=True;v['authorization_path']=str(Path(tmp)/'missing_authorization.json')
   p.write_text(json.dumps(v))
   with self.assertRaises(FileNotFoundError):g.load_policy(p)
 def test_template_cannot_authorize_even_if_approved_bit_is_flipped(self):
  with TemporaryDirectory() as tmp:
   policy=r.read(r.LIABILITY_POLICY);policy['ready']=True
   auth=r.read(r.META/'user_resume_authorization_TEMPLATE_v5.json');auth['approved']=True
   ap=Path(tmp)/'template.json';ap.write_text(json.dumps(auth));policy['authorization_path']=str(ap)
   pp=Path(tmp)/'policy.json';pp.write_text(json.dumps(policy))
   with self.assertRaisesRegex(ValueError,'templates cannot'):g.load_policy(pp)
 def test_numeric_one_is_not_real_boolean_approval(self):
  with TemporaryDirectory() as tmp:
   policy=r.read(r.LIABILITY_POLICY);policy['ready']=True
   auth=r.read(r.META/'user_resume_authorization_TEMPLATE_v5.json')
   auth.update(approved=1,template_only=False,user_message='TEST ONLY',recorded_at_unix=1)
   ap=Path(tmp)/'bad_approval.json';ap.write_text(json.dumps(auth));policy['authorization_path']=str(ap)
   pp=Path(tmp)/'policy.json';pp.write_text(json.dumps(policy))
   with self.assertRaisesRegex(ValueError,'authorization differs'):g.load_policy(pp)
 def test_v10_rejected_before_receipts(self):
  with self.assertRaisesRegex(ValueError,'only one additional'):
   r.audit_history([r.RUN/'preparations_v10'],self.jobs,{'cap_usd':45,'entries':{}})
 def test_original_client_method_not_mutated(self):
  self.assertNotIn('route_body',inspect.getsource(api.BudgetedChat.complete))
  self.assertIsNot(r.RoutedBudgetedChat.complete,api.BudgetedChat.complete)

class V5FakeTransport(unittest.TestCase):
 def scenario(self,behavior='normal',double=False,cache=False):
  with TemporaryDirectory() as tmp:
   w=World(tmp,count=4,model='qwen/qwen3.5-27b');w.behavior=behavior
   original_payload=w.payload
   def payload():
    value=original_payload()
    if behavior != 'missing_provider':value['provider']='WrongProvider' if behavior == 'wrong_provider' else 'DeepInfra'
    return value
   w.payload=payload
   jobs=copy.deepcopy(w.jobs);first=next(iter(jobs));proofs={}
   for key,j in jobs.items():
    body=copy.deepcopy(j['request']);body['provider'].update(order=['deepinfra/fp8'],allow_fallbacks=False)
    proofs[key]={'tag':j['tag'],'original_request_sha256':j['request_sha256'],
     'candidate_request_sha256':r.object_hash(body),'prior_actual_attempts':3 if key==first else 0}
   if cache:proofs.pop(first)
   auth=w.run/'fake_authorization.json';authorization=r.read(r.META/'user_resume_authorization_TEMPLATE_v5.json');authorization.update(approved=True,template_only=False,user_message="TEST FIXTURE ONLY: simulated approval",recorded_at_unix=1);g.atomic(auth,authorization)
   policy=g.read(w.policy);policy.update(schema='grid5_preparation_liability_policy_v5',preparation_cap_usd=14,concurrency=1,
    authorization_path=str(auth),route_proposal_sha256=authorization['route_proposal_sha256'],
    missing_jobs={json.dumps(p['tag'],sort_keys=True):p for p in proofs.values()})
   g.atomic(w.policy,policy)
   async def execute():
    reuse=[]
    if cache:
     old=api.BudgetedChat(w.run/'preparations_v1',list(w.catalog),1.65,max_tokens=16384,
      reasoning={'enabled':True},schema_retries=2)
     async with old:await w.call(old,jobs[first])
     reuse=[old.out/'requests.jsonl']
    raw=r.RoutedBudgetedChat(w.run/'preparations_v9',list(w.catalog),1.65,max_tokens=16384,
     reasoning={'enabled':True},schema_retries=2,reuse_receipts=reuse)
    raw.original_jobs=jobs
    history=SimpleNamespace(cost=0,attempts=Counter({first:1 if cache else 3}),new_transport_failures=2)
    proxy=g.wrap_client(raw,r.routed_jobs(jobs),lambda:None,w.policy,history)
    raw.complete=r.ordered_completion(raw,SimpleNamespace(missing=set(jobs)),raw.complete)
    results=[]
    try:
     async with proxy:
      calls=list(jobs.values())
      if double:calls=[calls[0],calls[0]]
      results=await asyncio.gather(*[w.call(raw,j) for j in calls],return_exceptions=True)
    except ValueError as exc:
     if not double:raise
     results.append(exc)  # Duplicate concurrent work fails closed and retains reservation.
    return raw,results
   with w.patches(),patch.object(r,'route_proofs',return_value=proofs):raw,results=asyncio.run(execute())
   self.assertEqual(w.peak,1)
   for receipt in raw.records:
    self.assertEqual(receipt['request_sha256'],g.body_hash(receipt['request']))
    if 'reused_from' in receipt:
     self.assertNotIn('order',receipt['request']['provider'])
     self.assertEqual(receipt['request'],jobs[r.job_key(receipt['tag'])]['request'])
     self.assertEqual(receipt['cost_usd'],0)
    else:
     self.assertEqual(receipt['request']['provider']['order'],['deepinfra/fp8'])
     self.assertFalse(receipt['request']['provider']['allow_fallbacks'])
   return w.posts,raw.records,results
 def test_all_requests_routed_before_hash_and_post_serially(self):
  posts,records,results=self.scenario();self.assertEqual(len(posts),4);self.assertTrue(all(r['valid'] for r in records))
 def test_original_cached_body_reused_without_route_or_paid_regeneration(self):
  posts,records,results=self.scenario(cache=True)
  self.assertEqual(len(posts),4)  # One original fixture POST, then only three missing.
  cached=[x for x in records if 'reused_from' in x]
  self.assertEqual(len(cached),1)
  self.assertEqual(cached[0]['cost_usd'],0)
 def test_fourth_total_attempt_only_no_fifth(self):
  posts,records,results=self.scenario(double=True);self.assertEqual(len(posts),1)
  self.assertTrue(any(isinstance(v,(RuntimeError,ValueError)) for v in results))
 def test_timeout_or_schema_failure_stops_waiters_and_retries(self):
  for behavior in ('timeout','schema_once','old_allowance_overrun','wrong_provider','missing_provider'):
   posts,records,results=self.scenario(behavior);self.assertEqual(len(posts),1);self.assertFalse(records[0]['valid'])

class TransportObservationV5(unittest.TestCase):
 def observe(self, failure=None):
  with TemporaryDirectory() as tmp:
   chunks=[b'{"answer":"caf',b'\xc3\xa9"}']
   expected=b''.join(chunks)
   class Content:
    def __init__(self):self.index=0
    async def readany(self):
     if self.index==1 and failure is not None:raise failure
     if self.index>=len(chunks):return b''
     value=chunks[self.index];self.index+=1;return value
   class Response:
    status=200
    headers={'Content-Type':'application/json','X-Request-ID':'safe-id',
     'Authorization':'secret-auth','Set-Cookie':'secret-cookie'}
    def __init__(self):self.content=Content();self.closed=False;self.raw=None
    async def __aenter__(self):return self
    async def text(self):
     result=[]
     while True:
      chunk=await self.content.readany()
      if not chunk:break
      result.append(chunk)
     self.raw=b''.join(result)
     return self.raw.decode('utf-8')
    async def __aexit__(self,*args):self.closed=True;return False
   response=Response()
   class Session:
    def __init__(self):self.posts=0
    def post(self,url,**kwargs):self.posts+=1;return response
   original=Session();observed=r.ObservedSession(original,Path(tmp));body={'unchanged':'request'}
   async def execute():
    async with observed.post('https://openrouter.ai/api/v1/chat/completions',json=body) as result:
     return await result.text()
   if failure is not None:
    with self.assertRaises(type(failure)):asyncio.run(execute())
   else:
    result=asyncio.run(execute());self.assertEqual(result,expected.decode('utf-8'))
    self.assertEqual(response.raw,expected)
   events=list(map(json.loads,(Path(tmp)/'transport_observations_v5.jsonl').read_text().splitlines()))
   self.assertEqual(original.posts,1);self.assertTrue(response.closed)
   serialized=json.dumps(events)
   self.assertNotIn('secret-auth',serialized);self.assertNotIn('secret-cookie',serialized)
   self.assertNotIn('café',serialized);self.assertNotIn('unchanged',serialized)
   self.assertEqual(events[1]['state'],'response_headers')
   self.assertEqual(events[1]['response_headers'],{'content-type':'application/json','x-request-id':'safe-id'})
   return events,expected,chunks[0]
 def test_original_response_bytes_and_decoding_unchanged(self):
  events,expected,_=self.observe();self.assertEqual(events[-1]['state'],'response_body_complete')
  self.assertEqual(events[-1]['body_bytes_received'],len(expected))
  self.assertEqual(events[-1]['received_prefix_sha256'],r.generation.sha256(expected.decode()))
 def test_partial_body_timeout_retains_progress_and_reraises(self):
  events,_,prefix=self.observe(TimeoutError())
  self.assertEqual(events[-1]['state'],'request_exception')
  self.assertEqual(events[-1]['error_type'],'TimeoutError')
  self.assertEqual(events[-1]['body_bytes_received'],len(prefix))
  self.assertEqual(events[-1]['failed_phase'],'body_progress')
  self.assertFalse(events[-1]['cancelled'])
 def test_partial_body_cancellation_propagates(self):
  events,_,prefix=self.observe(asyncio.CancelledError())
  self.assertTrue(events[-1]['cancelled'])
  self.assertEqual(events[-1]['body_bytes_received'],len(prefix))

class RealAiohttpLoopbackV5(unittest.TestCase):
 def check_loopback(self, timeout=False):
  from aiohttp import web, ClientSession, ClientTimeout
  async def scenario():
   with TemporaryDirectory() as tmp:
    request_bodies=[]
    payload='{"answer":"café"}'.encode('utf-8')
    async def handler(request):
     request_bodies.append(await request.read())
     response=web.StreamResponse(headers={'Content-Type':'application/json','Set-Cookie':'never-log'})
     await response.prepare(request)
     await response.write(payload[:8])
     await asyncio.sleep(.15 if timeout else .01)
     try:
      await response.write(payload[8:]);await response.write_eof()
     except ConnectionResetError:pass
     return response
    app=web.Application();app.router.add_post('/test',handler)
    runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    try:
     async with ClientSession(timeout=ClientTimeout(total=.06 if timeout else 2)) as session:
      observed=r.ObservedSession(session,Path(tmp))
      async def call():
       async with observed.post(f'http://127.0.0.1:{port}/test',json={'test':'value'}) as response:
        value=await response.text()
        self.assertEqual(value,payload.decode());self.assertEqual(response._body,payload)
      if timeout:
       with self.assertRaises(TimeoutError):await call()
      else:await call()
     self.assertEqual(request_bodies,[b'{"test": "value"}'])
     events=list(map(json.loads,(Path(tmp)/'transport_observations_v5.jsonl').read_text().splitlines()))
     states=[x['state'] for x in events]
     self.assertIn('connection_start',states);self.assertIn('connection_ready',states)
     self.assertIn('response_headers',states);self.assertIn('body_progress',states)
     self.assertEqual(events[-1]['state'],'request_exception' if timeout else 'response_body_complete')
     self.assertEqual(events[-1]['body_bytes_received'],8 if timeout else len(payload))
     self.assertNotIn('never-log',json.dumps(events))
    finally:await runner.cleanup()
  asyncio.run(scenario())
 def test_real_slotted_reader_original_bytes(self):self.check_loopback()
 def test_real_partial_payload_timeout(self):self.check_loopback(timeout=True)
 def test_header_observation_failure_cleans_acquired_response(self):
  async def scenario():
   with TemporaryDirectory() as tmp:
    cleaned=[]
    async def enter():return SimpleNamespace(status=200,headers={})
    async def cleanup(*args):cleaned.append(True);return False
    original=SimpleNamespace(__aenter__=enter,__aexit__=cleanup)
    owner=SimpleNamespace(out=Path(tmp),active={})
    manager=r.ObservedRequest(owner,original,{})
    def event(state,**kwargs):
     if state=='response_headers':raise OSError('mock header log failure')
    with patch.object(manager.observation,'event',side_effect=event):
     with self.assertRaises(OSError):await manager.__aenter__()
    self.assertEqual(cleaned,[True]);self.assertFalse(owner.active)
  asyncio.run(scenario())
 def test_logging_failure_still_cleans_original_context(self):
  async def scenario():
   with TemporaryDirectory() as tmp:
    original=SimpleNamespace()
    cleaned=[]
    async def cleanup(*args):cleaned.append(True);return False
    original.__aexit__=cleanup
    owner=SimpleNamespace(out=Path(tmp),active={})
    manager=r.ObservedRequest(owner,original,{})
    manager.task=asyncio.current_task();owner.active[manager.task]=manager.observation
    with patch.object(manager.observation,'event',side_effect=OSError('mock disk failure')):
     with self.assertRaises(OSError):await manager.__aexit__(None,None,None)
    self.assertEqual(cleaned,[True]);self.assertFalse(owner.active)
  asyncio.run(scenario())

if __name__=='__main__':unittest.main()
