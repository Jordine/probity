"""Offline checks for additive serial recovery; never contacts a provider."""
import asyncio
from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resume4_test', ROOT/'runs/iteration5/coordinator/preparation_resume_v4.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
g = r.load_liability()

class RecoveryV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        problems, rows, _, _ = r.generation.frozen_inventory(r.SOURCE, r.INVENTORY, r.INVENTORY_SHA)
        cls.jobs = r.make_jobs(problems, rows, r.read(r.RUN/'preparations_v1/api_manifest.json')['models'])
        cls.dirs = [r.RUN/f'preparations_v{i}' for i in range(1,7)]
        cls.history = r.audit_history(cls.dirs, cls.jobs, r.read(r.RUN/'budget.json'))

    def test_exact_stopped_history_preserves_560_and_all_attempts(self):
        self.assertEqual(r.stopped_history(self.jobs), r.read(r.HISTORY_MANIFEST))
        self.assertEqual((len(self.history.canonical), len(self.history.missing)), (560,8))
        self.assertEqual(self.history.new_transport_failures,0)
        self.assertEqual(len(self.history.quarantine),1)
        r.verify_canonical(self.history,r.read(r.HISTORY_MANIFEST))
        self.assertEqual({k[1] for k in self.history.missing},{'qwen/qwen3.5-27b'})

    def test_unchanged_prior_freeze_chain(self):
        r.prior.verify_freezes(r.PRIOR_RECOVERY_FREEZE_SHA)

    def test_liability_ledger_retains_seven_topups_and_caps(self):
        report=g.validate_liability_ledger(r.RUN/'budget.json',r.LIABILITY_POLICY)
        self.assertEqual(report['unknown_liability_entries'],7)
        self.assertEqual(report['additional_liability_usd'],4.472832)
        self.assertEqual(report['preparation_committed_usd'],10.959547696)
        self.assertEqual(r.batch_budget(self.history,report['additional_liability_usd']),2.04)
        policy=g.load_policy(r.LIABILITY_POLICY)
        self.assertEqual((policy['concurrency'],policy['campaign_cap_usd'],policy['preparation_cap_usd']),(1,45,13))
        ledger={'cap_usd':45,'entries':{'other':{'amount_usd':45.01}}}
        with self.assertRaisesRegex(ValueError,'cap exceeded'):g._caps(ledger,policy)

    def test_zero_post_exception_exact_directory_and_hash(self):
        directory=self.dirs[-1]
        receipts=r.generation.read_rows(directory/'requests.jsonl')
        summary=r.read(directory/'summary.json')
        self.assertTrue(r.reviewed_zero_post_budget_stop(directory,receipts,summary))
        self.assertFalse(r.reviewed_zero_post_budget_stop(r.RUN/'preparations_v7',receipts,summary))
        with patch.object(r.generation,'file_hash',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'v6 evidence changed'):
                r.reviewed_zero_post_budget_stop(directory,receipts,summary)
        changed=dict(summary,errors=summary['errors']+['unexpected'])
        self.assertFalse(r.reviewed_zero_post_budget_stop(directory,receipts,changed))

    def test_v8_without_v7_transport_and_any_v9_are_rejected(self):
        for version in (8,9):
            with self.assertRaisesRegex(ValueError,'v8 permitted only'):
                r.audit_history([r.RUN/f'preparations_v{version}'],self.jobs,
                    {'cap_usd':45,'entries':{}})

    def make_guard(self):
        guard=object.__new__(g.GuardedClient)
        guard.client=SimpleNamespace(stopped=False)
        guard.condition=asyncio.Condition()
        guard.by_task={};guard.leases={}
        guard.limit=g.number(2.04);guard.accounted=g.number(0);guard.extra=g.number(0)
        guard.broken=False;guard.cancelled=False
        guard.historical_attempts=Counter();guard.new_attempts={}
        guard.prior_new_transport_failures=0;guard.new_transport_failures=0
        guard.sync_records=lambda:None;guard.event=lambda *a,**k:None
        return guard

    def test_serial_lease_blocks_second_until_first_settles(self):
        jobs=[self.jobs[k] for k in sorted(self.history.missing)][:2]
        async def scenario():
            guard=self.make_guard()
            first=await guard.acquire(jobs[0]['request'],jobs[0]['tag'],0)
            task=asyncio.create_task(guard.acquire(jobs[1]['request'],jobs[1]['tag'],0))
            await asyncio.sleep(.01)
            self.assertFalse(task.done())
            self.assertEqual(len(guard.leases),1)
            await guard.finish(first)
            second=await asyncio.wait_for(task,1)
            self.assertEqual(len(guard.leases),1)
            await guard.finish(second)
        asyncio.run(scenario())

    def test_three_attempt_and_second_failure_guards_precede_lease(self):
        job=self.jobs[next(iter(self.history.missing))]
        key=r.job_key(job['tag'])
        async def scenario():
            guard=self.make_guard();guard.historical_attempts[key]=3
            with self.assertRaisesRegex(RuntimeError,'three actual'):
                await guard.acquire(job['request'],job['tag'],0)
            self.assertFalse(guard.leases)
            guard=self.make_guard();guard.prior_new_transport_failures=1;guard.new_transport_failures=1
            with self.assertRaisesRegex(RuntimeError,'one new unanswered'):
                await guard.acquire(job['request'],job['tag'],0)
            self.assertFalse(guard.leases)
        asyncio.run(scenario())

class FrozenClientIntegrationV4(unittest.TestCase):
    def scenario(self, behavior, prior_failures=0):
        from tempfile import TemporaryDirectory
        from tests.test_grid5_preparation_liability import World
        from probity_benchmark import api
        with TemporaryDirectory() as tmp:
            world=World(tmp,count=4,model="qwen/qwen3.5-27b")
            policy=g.read(world.policy)
            approval=world.run/'authorization.json'
            g.atomic(approval,g.read(r.META/'user_resume_authorization_v4.json'))
            policy.update(schema='grid5_preparation_liability_policy_v4',preparation_cap_usd=13,
                concurrency=1,authorization_path=str(approval),
                authorization_sha256=g.digest(approval.read_bytes()))
            g.atomic(world.policy,policy)
            world.behavior=behavior
            async def execute():
                raw=api.BudgetedChat(world.run/'preparations_v7',list(world.catalog),2.04,
                    max_tokens=16384,reasoning={'enabled':True},schema_retries=2)
                history=SimpleNamespace(cost=0,attempts=Counter(),new_transport_failures=prior_failures)
                proxy=g.wrap_client(raw,world.jobs,lambda:None,world.policy,history)
                async with proxy:
                    outcomes=await asyncio.gather(*[world.call(raw,j) for j in world.jobs.values()],
                        return_exceptions=True)
                return outcomes
            with world.patches(): outcomes=asyncio.run(execute())
            proof=g.read(world.run/'preparations_v7/liability_settlement_v3.json')
            self.assertEqual(proof['schema'],'grid5_preparation_liability_settlement_v4')
            self.assertEqual(world.peak,1)
            self.assertEqual(proof['posts'],len(world.posts))
            self.assertEqual(world.posts,[j['request'] for j in list(world.jobs.values())[:len(world.posts)]])
            return len(world.posts),outcomes,proof

    def test_frozen_client_four_exact_posts_serial(self):
        count,outcomes,proof=self.scenario('normal')
        self.assertEqual(count,4)
        self.assertFalse(any(isinstance(x,Exception) for x in outcomes))
        self.assertEqual(proof['additional_unknown_liability_usd'],0)

    def test_new_schema_error_stops_recursive_retry_and_waiters(self):
        count,outcomes,proof=self.scenario('schema_once')
        self.assertEqual(count,1)
        self.assertTrue(any(isinstance(x,Exception) for x in outcomes))

    def test_first_or_second_timeout_drains_and_stops_all_waiters(self):
        for prior in (0,1):
            count,outcomes,proof=self.scenario('timeout',prior)
            self.assertEqual(count,1)
            self.assertEqual(proof['additional_unknown_liability_usd'],.638976)

if __name__=='__main__': unittest.main()
