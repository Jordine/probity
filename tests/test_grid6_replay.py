import copy
import json
from pathlib import Path
import sys
import unittest

import torch

from probity_benchmark.grid6_replay import LAYERS, project, project_features
from probity_benchmark.grid4_score import project_response
from probity_benchmark.evaluate_debate_transfer import argument_mapping
from probity_benchmark.local3_annotate_units import display_units
from probity_benchmark.qwen_extract import overlapping_tokens


class SharedProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)
        self.raw=json.dumps({'argument':'Ada said "No." Café. Bo said yes.', 'answer':'A'})
        argument,mapping=argument_mapping(self.raw)
        _,self.units=display_units(self.raw,['sentence','clause'])
        self.arg_range=(mapping[0][0],mapping[-1][1])
        self.offsets=[(i,i+1) for i in range(len(self.raw))]
        self.yes={l:torch.randn(len(self.raw)+1,8).bfloat16() for l in LAYERS}
        self.no={l:v.clone() for l,v in self.yes.items()}
        for l in LAYERS:self.no[l][-1]+=1
        weight=torch.randn(8,dtype=torch.float64)
        self.heads={
            'adapted:response_mean':{'layer':43,'feature':'response_mean','weight':weight,'bias':.2},
            'apollo_data:dyl_no':{'layer':19,'feature':'dyl_no','weight':weight,'bias':-.1},
            'apollo_data:dyl_difference':{'layer':43,'feature':'dyl_difference','weight':weight,'bias':.3},
            'argument':{'layer':35,'feature':'argument_mean','weight':weight,'bias':.1},
            'local':{'layer':35,'feature':'span_mean','weight':weight,'bias':.4},
            'last':{'layer':51,'feature':'span_last','weight':weight,'bias':.5},
        }

    def result(self):
        return project_features(self.yes,self.no,self.units,self.offsets,self.arg_range,self.heads,None)

    def test_adapted_projection_exactly_matches_frozen_scalar(self):
        result=self.result()
        expected=project_response(self.yes[43][:-1],self.heads['adapted:response_mean'])
        self.assertEqual(result['response_scores']['adapted:response_mean'],expected)

    def test_local_unit_mapping_and_pooling(self):
        result=self.result()
        for unit in result['units']:
            indices=overlapping_tokens(self.offsets,unit['start'],unit['end'])
            expected=float(project(self.yes[35][indices].float().mean(0),self.heads['local']))
            self.assertAlmostEqual(unit['scores']['local'],expected,places=12)
            self.assertEqual(self.raw[unit['start']:unit['end']],unit['text'])
            self.assertEqual(unit['token_positions_within_response'],indices)
            expected_last=float(project(self.yes[51][indices[-1]].float(),self.heads['last']))
            self.assertAlmostEqual(unit['scores']['last'],expected_last,places=12)

    def test_dyl_uses_period_not_response_end(self):
        result=self.result()
        expected=float(project(self.no[19][-1].float(),self.heads['apollo_data:dyl_no']))
        self.assertEqual(result['response_scores']['apollo_data:dyl_no'],expected)
        delta=self.no[43][-1].float()-self.yes[43][-1].float()
        self.assertEqual(result['response_scores']['apollo_data:dyl_difference'],float(project(delta,self.heads['apollo_data:dyl_difference'])))

    def test_output_is_json_and_contains_no_activation_tensors(self):
        result=self.result()
        encoded=json.dumps(result,allow_nan=False)
        self.assertNotIn('response_tokens',encoded)
        self.assertNotIn('weight',encoded)
        self.assertEqual(result['units'][0]['scores']['response_broadcast'],result['response_scores']['adapted:response_mean'])

    def test_original_arrays_units_unchanged(self):
        units=copy.deepcopy(self.units);before={l:v.clone() for l,v in self.yes.items()}
        self.result()
        self.assertEqual(self.units,units)
        self.assertTrue(all(torch.equal(before[l],v) for l,v in self.yes.items()))

    def test_future_arm_change_guard(self):
        self.no[35][0]+=20
        with self.assertRaisesRegex(ValueError,'future DYL'):
            self.result()

    def test_nonfinite_projection_rejected(self):
        self.yes[43][1,1]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.result()

    def test_invalid_pooling_contract_rejected(self):
        self.heads['argument']['feature']='wrong'
        with self.assertRaisesRegex(ValueError,'unsupported frozen feature'):
            self.result()

    def test_json_escaped_quotation_and_unicode_offsets(self):
        self.assertIn('\\"',self.raw)
        self.assertIn('\\u00e9',self.raw)
        result=self.result()
        self.assertTrue(any('Café' in u['display_text'] for u in result['units']))
        self.assertTrue(all(self.raw[u['start']:u['end']]==u['text'] for u in result['units']))


class PublishedReadoutTests(unittest.TestCase):
    def test_compact_one_token_equals_full_masked_fibs(self):
        root=Path(__file__).resolve().parents[1]
        vendor=root/'runs/iteration6/gpu/probe-inference-source/src'
        weights=root/'runs/iteration6/gpu/published_fibs_qwen35_linear'
        if not vendor.is_dir() or not weights.is_dir():self.skipTest('private cached frozen FIBS package unavailable')
        sys.path.insert(0,str(vendor))
        from probe_inference import load_probe
        probe=load_probe(weights)
        torch.manual_seed(19)
        full={l:torch.randn(5,5120).bfloat16() for l in LAYERS}
        compact={l:v[3:4] for l,v in full.items()}
        mask=torch.tensor([False,False,False,True,False])
        expected=probe.score(full,mask)
        self.assertEqual(probe.score(compact,torch.ones(1,dtype=torch.bool)),expected)
        layers=probe.layer_logits({l:v.unsqueeze(0) for l,v in compact.items()},torch.ones(1,1,dtype=torch.bool))
        self.assertEqual(set(layers),set(LAYERS))


if __name__=='__main__':unittest.main()
