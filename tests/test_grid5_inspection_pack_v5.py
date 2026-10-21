"""Offline v5 adapter/route provenance tests; no real expanded pack is built."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reports import build_grid5_inspection_pack_v5 as pack


def route_fixture(root):
    entries=[{'tag':{'problem_id':str(i),'model':'qwen/qwen3.5-27b','speaker':speaker}}
             for i in range(4) for speaker in ('Red','Blue')]
    ready={'route_proposal_sha256':'proposal','routed_missing_requests':entries}
    route={'schema':'grid5_analysis_route_provenance_v5','recovery_freeze_sha256':'freeze',
           **ready,'affected_trajectories':[]}
    for arm,directory in [('baseline','debates_v1'),('feedback','feedback_v1')]:
        for i in range(4):
            path=root/'runs/iteration5'/directory/f'episode_{i}.json';path.parent.mkdir(parents=True,exist_ok=True)
            episode={'episode_id':arm+str(i),'problem_id':str(i),'model':'qwen/qwen3.5-27b','complete':True,'turns':[{}]*6}
            path.write_text(json.dumps(episode))
            route['affected_trajectories'].append({k:episode[k] for k in ('episode_id','problem_id','model')} |
                {'arm':arm,'path':str(path.relative_to(root)),'sha256':pack.original.digest(path)})
    analysis=root/'runs/iteration5/analysis_v1/analysis.json';analysis.parent.mkdir();analysis.write_text('{}')
    route['analysis_sha256']={str(analysis.relative_to(root)):pack.original.digest(analysis)}
    return ready,route


class AdapterTests(unittest.TestCase):
    def test_only_ready_filename_constant_changes(self):
        source=Path(pack.original.__file__).read_text()
        original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='collect')
        adapted=pack.readiness_ast(source).body[0]
        changed=[n for n in ast.walk(adapted) if isinstance(n,ast.Constant) and n.value==pack.READY_NAME]
        self.assertEqual(len(changed),1)
        changed[0].value='preparations_recovery_ready_v4.json'
        self.assertEqual(ast.dump(original,include_attributes=True),ast.dump(adapted,include_attributes=True))
        self.assertEqual(pack.original.digest(pack.original.__file__),pack.FROZEN_BUILDER_SHA)
    def test_changed_builder_rejected(self):
        with patch.object(pack.original,'digest',return_value='different'):
            with self.assertRaisesRegex(ValueError,'frozen inspection builder'):pack.adapted_collect()
    def test_original_completion_gate_still_prevents_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta=root/'coordinator';meta.mkdir()
            selection=meta/'qualitative_reader_selection_v1.json';selection.write_text('{}')
            with patch.object(pack.original,'RUN',root):
                collect=pack.adapted_collect()
                args=SimpleNamespace(out=root/'reader_reviews/new',selection_sha256=pack.original.digest(selection),completion_sha256='absent')
                with self.assertRaises(FileNotFoundError):collect(args)
                self.assertFalse(args.out.exists())
    def test_route_exact_stratum_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ready,route=route_fixture(root)
            self.assertEqual(len(pack.validate_route(route,ready,'freeze',root)),8)
    def test_route_mapping_freeze_coverage_and_analysis_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ready,route=route_fixture(root)
            for mutation in ('schema','freeze','proposal','mapping','omit','duplicate','analysis','outside'):
                changed=copy.deepcopy(route)
                if mutation=='schema':changed['schema']='v4'
                if mutation=='freeze':changed['recovery_freeze_sha256']='other'
                if mutation=='proposal':changed['route_proposal_sha256']='other'
                if mutation=='mapping':changed['routed_missing_requests'][0]['tag']['speaker']='other'
                if mutation=='omit':changed['affected_trajectories'].pop()
                if mutation=='duplicate':changed['affected_trajectories'][1]=changed['affected_trajectories'][0]
                if mutation=='analysis':changed['analysis_sha256']={}
                if mutation=='outside':changed['affected_trajectories'][0]['path']='../../outside'
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    pack.validate_route(changed,ready,'freeze',root)
    def test_truncated_trajectory_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ready,route=route_fixture(root);row=route['affected_trajectories'][0];path=root/row['path']
            episode=json.loads(path.read_text());episode['turns'].pop();path.write_text(json.dumps(episode));row['sha256']=pack.original.digest(path)
            with self.assertRaisesRegex(ValueError,'identity/completion'):pack.validate_route(route,ready,'freeze',root)
    def test_v5_audit_is_mandatory_before_original_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta=Path(tmp)
            for name in ('final_analysis_complete_v1.json',pack.READY_NAME,pack.ROUTE_NAME,'execution_recovery_freeze_v5.json'):
                (meta/name).write_text('{}')
            sha=pack.original.digest(meta/pack.READY_NAME)
            args=SimpleNamespace(completion_sha256=sha,preparation_ready_sha256=sha,route_provenance_sha256=sha,recovery_freeze_sha256=sha)
            fake=SimpleNamespace(verify_freezes=lambda _: {},audit_recovery=lambda *_: (_ for _ in ()).throw(ValueError('canonical560 altered')))
            with patch.object(pack,'META',meta),patch.object(pack,'load_prework',return_value=fake),patch.object(pack,'adapted_collect') as builder:
                with self.assertRaisesRegex(ValueError,'canonical560 altered'):pack.collect(args)
                builder.return_value.assert_not_called()


if __name__=='__main__':unittest.main()
