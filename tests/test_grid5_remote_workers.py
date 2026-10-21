"""Pure remote launch preflights; no model weights, network or subprocesses."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from probity_benchmark import grid5_score as score
from probity_benchmark.preflight import sha256

path = Path(__file__).resolve().parents[1] / 'runs/iteration5/coordinator/remote_workers.py'
spec = importlib.util.spec_from_file_location('grid5_remote_workers', path)
workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workers)


class Tokenizer:
    chat_template = 'fixed'
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(c) for c in ids)


class RemoteTests(unittest.TestCase):
    def test_full_job_marker_rejects_partial_or_modified_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = [{'job_id': str(i), 'messages': [{'role': 'user', 'content': str(i)}]} for i in range(1328)]
            path = root / 'jobs.jsonl'
            path.write_text(''.join(json.dumps(j) + '\n' for j in jobs))
            marker = {'jobs_path': 'jobs.jsonl', 'jobs_sha256': score.frozen.file_hash(path), 'expected_jobs': 1328}
            self.assertEqual(len(workers.validate_native_marker(marker, root)[1]), 1328)
            with self.assertRaisesRegex(ValueError, 'count/identity'):
                workers.validate_native_marker({**marker, 'expected_jobs': 1136}, root)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'transfer hash'):
                workers.validate_native_marker(marker, root)

    def test_native_exact_cache_coverage_context_and_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = {'model': 'Qwen/test', 'revision': 'fixed'}
            tokenizer = Tokenizer()
            contract = {'model': model['model'], 'revision': model['revision'], 'dtype': 'bfloat16',
                'backend': 'transformers native generate, unpadded, greedy, thinking disabled',
                'template_sha256': sha256(tokenizer.chat_template), 'max_new_tokens': 8192,
                'runner_sha256': score.SOURCE_SHA256['run_local_judge.py'], 'answer_only': True}
            jobs = [{'job_id': str(i), 'messages': [{'role': 'user', 'content': str(i)}]} for i in range(1328)]
            score.frozen.atomic_json(root / 'manifest.json', {'contract': contract})
            rows = [{'valid': True, 'parsed': {'answer': 'A'}, 'request_sha256':
                sha256(json.dumps({'contract': contract, 'text': str(i)}, sort_keys=True))} for i in range(192)]
            (root / 'evaluations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            result = workers.native_cache_preflight(jobs, [root], tokenizer, model, 16384)
            self.assertEqual((result['exact_cache_hits'], result['fresh_requests']), (192, 1136))
            with self.assertRaisesRegex(ValueError, 'completion allowance'):
                workers.native_cache_preflight(jobs, [root], tokenizer, model, 8192)
            with self.assertRaisesRegex(ValueError, 'cache contract'):
                workers.native_cache_preflight(jobs, [root], tokenizer, {**model, 'revision': 'changed'}, 16384)
            (root / 'evaluations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError, '192 exact cache hits'):
                workers.native_cache_preflight(jobs, [root], tokenizer, model, 16384)


if __name__ == '__main__':
    unittest.main()
