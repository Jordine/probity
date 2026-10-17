"""CPU tests of queue identity, strict replay caching and non-GPU preflight."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probity_benchmark import grid5_score as score
from probity_benchmark.preflight import sha256
from tests.test_grid4_score import fixture


def complete_result(request, bundle):
    _, result, _ = fixture()
    result.update({k: request[k] for k in score.frozen.IDENTITY_FIELDS})
    result.update(score_request_sha256=score.frozen.canonical_hash(request), input_tokens=100,
                  response_tokens=12, termination="teacher_forced_complete_sequence", elapsed_seconds=.01)
    spec = score.frozen.MODELS[request['model']]
    result['provenance'].update({k: spec[k] for k in ('model', 'model_revision', 'layer', 'chat_template_sha256')})
    result['provenance'].update(scorer_sha256=score.SOURCE_SHA256['grid4_score.py'],
        original_runtime_sha256=score.SOURCE_SHA256['qwen_extract.py'],
        reference_artifact_sha256=score.REFERENCE_SHA256, dtype='bfloat16',
        allow_bf16_reduced_precision_reduction=False, versions=score.VERSIONS)
    return result


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request, _, self.bundle = fixture()
        self.reference = self.root / 'reference.json'
        score.frozen.atomic_json(self.reference, self.bundle)
        self.queue = self.root / 'queue'
        for name in ('requests', 'jobs', 'results', 'parity_receipts', 'worker_receipts'):
            (self.queue / name).mkdir(parents=True)
        self.plan = {'schema': score.SCHEMA, 'adapter_sha256': score.frozen.file_hash(score.__file__),
                     'reference_artifact_sha256': score.frozen.file_hash(self.reference),
                     'head_artifact_sha256': {m: 'a' * 64 for m in score.frozen.MODELS},
                     'native_judge_spec_sha256': 'b' * 64,
                     'frozen_inference_sources': score.SOURCE_SHA256,
                     'parity_plan_sha256': score.PARITY_SHA256,
                     'expected': {m: {'baseline': 1, 'feedback': 1} for m in score.frozen.MODELS}}
        score.frozen.atomic_json(self.queue / 'plan.json', self.plan)

    def baseline(self, name='baseline-1'):
        r = copy.deepcopy(self.request)
        r.update(job_id=name, episode_id=name, matched_no_feedback_episode_id=name)
        return r

    def episode(self, r):
        return {'episode_id': r['episode_id'], 'model': r['model'], 'complete': True,
                'parent_problem_id': r['parent_problem_id'], 'split': r['split'],
                'turns': [{'speaker': r['speaker'], 'argument': r['argument'],
                           'request_sha256': r['source_request_sha256']}]}

    def test_baseline_builder_retains_receipt_and_exact_response(self):
        r = self.baseline()
        actual = score.baseline_request(self.episode(r), 0, r['source_receipt'], self.bundle)
        for key in ('messages', 'response', 'response_reasoning', 'source_receipt', 'argument'):
            self.assertEqual(actual[key], r[key])
        self.assertEqual(actual['episode_id'], actual['matched_no_feedback_episode_id'])

    def test_arm_contract_duplicate_turn_and_complete_context_identity(self):
        with self.assertRaisesRegex(ValueError, 'pairing'):
            score.publish(self.queue, self.request, 'baseline', self.bundle)
        score.publish(self.queue, self.request, 'feedback', self.bundle)
        self.assertEqual(len(score.publish_many(self.queue, [self.request], 'feedback', self.bundle)), 1)
        other = copy.deepcopy(self.request)
        other['job_id'] = 'other-job'
        with self.assertRaisesRegex(ValueError, 'duplicate episode/turn'):
            score.publish(self.queue, other, 'feedback', self.bundle)
        changed = copy.deepcopy(self.request)
        changed['messages'][0]['content'] += ' Additional complete context.'
        changed['source_receipt']['request']['messages'] = copy.deepcopy(changed['messages'])
        digest = sha256(json.dumps(changed['source_receipt']['request'], sort_keys=True))
        changed['source_request_sha256'] = changed['source_receipt']['request_sha256'] = digest
        with self.assertRaisesRegex(ValueError, 'different request'):
            score.publish(self.queue, changed, 'feedback', self.bundle)

    def test_batch_overflow_publishes_nothing_and_partial_is_not_work(self):
        with self.assertRaisesRegex(ValueError, 'exceeds expected'):
            score.publish_many(self.queue, [self.baseline('b1'), self.baseline('b2')], 'baseline', self.bundle)
        self.assertEqual(list((self.queue / 'jobs').iterdir()), [])
        score.frozen.atomic_json(self.queue / 'requests/uncommitted.json', self.baseline())
        self.assertEqual(score.read_items(self.queue, self.plan, self.bundle), [])

    def test_feedback_priority_and_published_mutation_rejected(self):
        baseline = self.baseline()
        score.publish(self.queue, baseline, 'baseline', self.bundle)
        score.publish(self.queue, self.request, 'feedback', self.bundle)
        cache = {}
        items = score.read_items(self.queue, self.plan, self.bundle, cache)
        self.assertEqual(score.next_job(items, set(), self.request['model'])[0]['arm'], 'feedback')
        path = self.queue / 'requests' / (baseline['job_id'] + '.json')
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            score.read_items(self.queue, self.plan, self.bundle, cache)

    def test_export_selects_only_arm_and_preserves_original_metadata(self):
        baseline = self.baseline()
        score.publish(self.queue, baseline, 'baseline', self.bundle)
        score.publish(self.queue, self.request, 'feedback', self.bundle)
        score.frozen.atomic_json(self.queue / 'results' / (baseline['job_id'] + '.json'), complete_result(baseline, self.bundle))
        episodes = self.root / 'baseline'
        score.frozen.atomic_json(episodes / 'episode_1.json', self.episode(baseline))
        out = self.root / 'scores.json'
        result = score.export_arm(self.queue, episodes, self.reference, 'baseline', out)
        self.assertEqual(result['turns'], 1)
        actual = score.read(out)
        self.assertEqual({k: v for k, v in actual.items() if k != 'rows'},
                         {k: v for k, v in self.bundle.items() if k != 'rows'})
        self.assertEqual(actual['rows'][0]['episode_id'], baseline['episode_id'])

    def test_strict_result_rejects_runtime_or_termination_drift(self):
        r = complete_result(self.request, self.bundle)
        score.check_result(r, self.request, self.bundle)
        for field, value in [('dtype', 'float16'), ('versions', {}), ('allow_bf16_reduced_precision_reduction', True)]:
            altered = copy.deepcopy(r)
            altered['provenance'][field] = value
            with self.assertRaisesRegex(ValueError, 'provenance'):
                score.check_result(altered, self.request, self.bundle)
        r['termination'] = 'length'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            score.check_result(r, self.request, self.bundle)

    def test_worker_reuses_result_loads_once_and_keeps_only_json(self):
        baseline = self.baseline()
        score.publish(self.queue, baseline, 'baseline', self.bundle)
        score.publish(self.queue, self.request, 'feedback', self.bundle)
        score.frozen.atomic_json(self.queue / 'results' / (baseline['job_id'] + '.json'), complete_result(baseline, self.bundle))
        args = SimpleNamespace(queue=self.queue, reference=self.reference, heads=self.root / 'heads.pt',
            native_spec=self.root / 'native.json', parity=self.root / 'parity.json', model=self.request['model'],
            max_seconds=10, idle_timeout=2, poll_seconds=.001)
        actual_hash = score.frozen.file_hash
        def fake_hash(path):
            path = Path(path)
            return {'heads.pt': 'a' * 64, 'native.json': 'b' * 64,
                    'parity.json': score.PARITY_SHA256[args.model]}.get(path.name) or actual_hash(path)
        with patch.object(score, 'assert_frozen_assets', return_value=self.bundle), \
             patch.object(score.frozen, 'file_hash', side_effect=fake_hash), \
             patch.object(score.frozen, 'run_parity', return_value={'passed': True, 'cases': []}), \
             patch.object(score.frozen, 'Scorer') as constructor:
            constructor.return_value.provenance = {}
            constructor.return_value.score.side_effect = lambda r: complete_result(r, self.bundle)
            summary = score.serve(args)
        self.assertEqual(summary['completed'], {'baseline': 1, 'feedback': 1})
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(constructor.return_value.score.call_count, 1)
        self.assertEqual(constructor.return_value.score.call_args.args[0]['job_id'], self.request['job_id'])
        self.assertTrue(all(p.suffix in ('.json', '.lock') for p in self.queue.rglob('*') if p.is_file()))

    def test_tokenizer_preflight_never_constructs_model_and_checks_exact_scope(self):
        score.publish(self.queue, self.request, 'feedback', self.bundle)
        template = 'pinned template'
        spec = copy.deepcopy(score.frozen.MODELS)
        spec[self.request['model']]['chat_template_sha256'] = sha256(template)
        tokenizer = SimpleNamespace(chat_template=template)
        config = SimpleNamespace(max_position_embeddings=128)
        serial = {'ids': list(range(100)), 'response_positions': [90, 91], 'text': 'complete', 'returned_reasoning_sha256': None}
        with patch.dict(score.frozen.MODELS, spec), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
             patch('transformers.AutoConfig.from_pretrained') as cfg, \
             patch.object(score.frozen, 'serialize', return_value=serial) as serializer, \
             patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=AssertionError('weights forbidden')):
            cfg.return_value.get_text_config.return_value = config
            result = score.tokenizer_preflight(self.queue, self.reference, self.root / 'preflight.json', 'qwen35', True)
            self.assertEqual(result['checked_requests'], 1)
            serializer.assert_called_once_with(tokenizer, self.request['messages'], self.request['response'],
                                               'Yes.', reasoning=self.request['response_reasoning'])
            config.max_position_embeddings = 99
            with self.assertRaisesRegex(ValueError, 'exceeds pinned context'):
                score.tokenizer_preflight(self.queue, self.reference, self.root / 'too-long.json', 'qwen35', True)


if __name__ == '__main__':
    unittest.main()
