"""Reprocess 24 completed source reviews without any network access."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import jsonschema
from probity_benchmark import environment3_source_review as module
from probity_benchmark.api import assert_no_exposed_truncation, strict_json

ROOT = Path('/root/projects/probity-benchmark')
OLD = ROOT / 'runs/iteration3/final/source_ambiguity_review_v1'
OUT = ROOT / 'runs/iteration3/final/source_ambiguity_review_v2'
source_receipts = [json.loads(s) for s in (OLD / 'requests.jsonl').read_text().splitlines()]
assert len(source_receipts) == 24

class ReplayOnly:
    def __init__(self, out, models, budget, **kwargs):
        assert out == OUT and not out.exists()
        out.mkdir(parents=True)
        self.out = out
        self.records = []
        self.spent = 0
        self.by_tag = {json.dumps(r['tag'], sort_keys=True): r for r in source_receipts}
        assert len(self.by_tag) == 24
        manifest = json.loads((OLD / 'api_manifest.json').read_text())
        manifest.update(offline_replay_only=True, budget=0, receipt_reuse_identity='exact original request body and canonical job tag; no catalog lookup or new API requests', original_manifest_sha256=hashlib.sha256((OLD/'api_manifest.json').read_bytes()).hexdigest())
        (out / 'api_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False
    async def complete(self, model, messages, schema, tag):
        prior = self.by_tag.pop(json.dumps(tag, sort_keys=True))
        assert prior['valid'] and prior['finish_reason'] == 'stop'
        request = prior['request']
        assert prior['model'] == model == request['model']
        assert request['messages'] == messages
        assert request['response_format']['json_schema']['schema'] == schema
        assert request['max_tokens'] == 8192 and request['reasoning'] == {'enabled': False}
        assert hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() == prior['request_sha256']
        payload = json.loads(prior['response_body'])
        assert_no_exposed_truncation(payload)
        parsed = strict_json(payload['choices'][0]['message']['content'])
        jsonschema.validate(parsed, schema)
        assert parsed == prior['parsed']
        record = dict(prior, cost_usd=0.0, original_cost_usd=prior['cost_usd'], cost_basis='previously_accounted_receipt', reused_from=str(OLD/'requests.jsonl'))
        self.records.append(record)
        with (self.out / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        return record

module.BudgetedChat = ReplayOnly
asyncio.run(module.run(SimpleNamespace(dataset=ROOT/'runs/iteration3/coordinator/holdout_families_source_v4.jsonl',out=OUT,budget=0,reuse_receipts=[])))
