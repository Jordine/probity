"""Private v5 version adapter for the frozen inspection builder; offline only.

The sole compiled change is the readiness filename literal in collect(). All
original completion/package/selection/source/turn/judge checks remain intact.
Additional v5 gates independently verify 560 retained preparations, eight exact
routing deltas, the operational freeze, and final analysis route provenance.
"""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path

from reports import build_grid5_inspection_pack as original

ROOT = original.ROOT
RUN = original.RUN
META = RUN / 'coordinator'
FROZEN_BUILDER_SHA = 'f3180205568f350106210a41d87e9d8d22f486c003dd3a0fa7ba6a07085928bc'
READY_NAME = 'preparations_recovery_ready_v5.json'
ROUTE_NAME = 'analysis_route_provenance_v5.json'


def adapted_collect():
    """Hash-pin source, then substitute exactly one readiness path constant."""
    path = Path(original.__file__)
    original.require(original.digest(path) == FROZEN_BUILDER_SHA, 'frozen inspection builder changed')
    module = readiness_ast(path.read_text())
    namespace = dict(vars(original))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['collect']


def readiness_ast(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect']
    original.require(len(functions) == 1, 'single original collect function required')
    fn = copy.deepcopy(functions[0])
    hits = [n for n in ast.walk(fn) if isinstance(n, ast.Constant)
            and n.value == 'preparations_recovery_ready_v4.json']
    original.require(len(hits) == 1, 'single explicit readiness substitution required')
    hits[0].value = READY_NAME
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    return module


def load_prework():
    path = META / 'prework_resume_v5.py'
    spec = importlib.util.spec_from_file_location('inspection_v5_prework_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_route(route, ready, freeze_sha, root=ROOT):
    """Bind the complete failure-selected stratum and every analysis artifact."""
    require = original.require
    require(route.get('schema') == 'grid5_analysis_route_provenance_v5'
            and route.get('recovery_freeze_sha256') == freeze_sha
            and route.get('route_proposal_sha256') == ready['route_proposal_sha256']
            and route.get('routed_missing_requests') == ready['routed_missing_requests'],
            'final route provenance differs from reviewed v5 readiness')
    affected = {(e['tag']['problem_id'], e['tag']['model']) for e in ready['routed_missing_requests']}
    require(len(affected) == 4 and len(ready['routed_missing_requests']) == 8,
            'exact four-question/eight-plan routing stratum required')
    seen = set(); hashes = {}
    trajectories = route.get('affected_trajectories', [])
    require(len(trajectories) == 8, 'all eight affected trajectories required')
    for row in trajectories:
        arm = row['arm']
        require(arm in ('baseline','feedback'), 'unknown route provenance arm')
        rel = Path(row['path'])
        directory = 'debates_v1' if arm == 'baseline' else 'feedback_v1'
        require(not rel.is_absolute() and '..' not in rel.parts
                and rel.parent == Path('runs/iteration5') / directory
                and rel.name.startswith('episode_') and rel.suffix == '.json', 'unexpected trajectory path')
        path = root / rel
        require(not path.is_symlink() and original.digest(path) == row['sha256'], 'affected trajectory changed')
        episode = original.read(path)
        identity = (arm, row['problem_id'], row['model'])
        require(identity not in seen and (row['problem_id'],row['model']) in affected,
                'duplicate or unrelated route trajectory')
        seen.add(identity)
        require(all(episode[k] == row[k] for k in ('episode_id','problem_id','model'))
                and episode.get('complete') is True and len(episode['turns']) == 6,
                'route trajectory identity/completion differs')
        hashes[str(path)] = row['sha256']
    require(seen == {(arm,*key) for arm in ('baseline','feedback') for key in affected},
            'incomplete route trajectory stratum')
    analyses = {str(p.relative_to(root)):original.digest(p)
                for p in (root/'runs/iteration5/analysis_v1').rglob('*') if p.is_file()}
    require(analyses and route.get('analysis_sha256') == analyses, 'complete final analysis provenance differs')
    return hashes


def collect(args):
    # Fail before any raw expansion if full completion/review hashes are absent.
    for path, sha in ((META/'final_analysis_complete_v1.json', args.completion_sha256),
                      (META/READY_NAME, args.preparation_ready_sha256),
                      (META/ROUTE_NAME, args.route_provenance_sha256),
                      (META/'execution_recovery_freeze_v5.json', args.recovery_freeze_sha256)):
        original.require(original.digest(path) == sha, 'reviewed v5 inspection evidence changed: '+path.name)
    builder = adapted_collect()
    prework = load_prework()
    freezes = prework.verify_freezes(args.recovery_freeze_sha256)
    ready = original.read(META/READY_NAME)
    _, audit = prework.audit_recovery(ready, freezes)
    original.require(audit['complete_preparations'] == 568 and audit['unchanged_prior_preparations'] == 560,
                     'complete v5 preparation audit required')
    route = original.read(META/ROUTE_NAME)
    affected_hashes = validate_route(route, ready, args.recovery_freeze_sha256)
    out, report = builder(args)
    original.require(all(report['input_sha256'].get(k) == v for k,v in affected_hashes.items()),
                     'affected trajectories not bound into complete benchmark package')
    report['schema'] = 'grid5_private_inspection_pack_v5'
    report['adapter_sha256'] = original.digest(__file__)
    report['version_adapter'] = {'frozen_builder_sha256': FROZEN_BUILDER_SHA,
        'sole_original_function_change':'preparations_recovery_ready_v4.json -> preparations_recovery_ready_v5.json',
        'v5_preparation_audit':audit, 'analysis_route_provenance':route,
        'interpretation':'The eight routed plans form a failure-selected operational amendment, not a randomized provider comparison. Private diagnostics never enter this pack as canonical preparations.'}
    for path in (META/READY_NAME, META/ROUTE_NAME, META/'execution_recovery_freeze_v5.json',
                 META/'prework_resume_v5.py',Path(__file__)):
        report['input_sha256'][str(path)] = original.digest(path)
    return out,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','completion','selection','preparation-ready','route-provenance','recovery-freeze'):
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    out,report=collect(parser.parse_args())
    page=original.render(report)
    os.umask(0o077)
    out.mkdir(parents=True,exist_ok=False)
    (out/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'index.html').write_text(page)
    manifest={'schema':report['schema'],'label':original.LABEL,'private_only':True,
        'units':4,'transcripts':8,'public_turns':48,'judge_cells':16,
        'builder_sha256':report['builder_sha256'],'adapter_sha256':report['adapter_sha256'],
        'input_sha256':report['input_sha256'],
        'output_sha256':{n:original.digest(out/n) for n in ('evidence.json','index.html')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'private_inspection_pack':str(out),'paid_calls':0}))


if __name__=='__main__':main()
