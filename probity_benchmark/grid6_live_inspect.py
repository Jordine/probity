"""Render outcome-blind examples with exact saved live requests and responses."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path

from .grid6_live import load_run, file_hash


def pre(value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    return '<pre>' + html.escape(text) + '</pre>'


def details(title, value):
    return '<details><summary>' + html.escape(title) + '</summary>' + pre(value) + '</details>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--families', type=int, default=3)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new inspector output required')
    manifest, problems, branches, detector = load_run(a.run)
    by_family = defaultdict(list)
    for branch in branches:
        by_family[branch['family_id']].append(branch)
    chosen = sorted(by_family, key=lambda family: hashlib.sha256(('grid6-inspection-v1:' + family).encode()).hexdigest())[:a.families]
    receipts = {row['request_sha256']: (path, row) for path in (a.run / 'requests').glob('*.json')
                if (row := json.loads(path.read_text())).get('valid')}
    a.out.mkdir(parents=True)
    style = '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em;color:#17202a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f7;padding:1em;font:13px monospace}details{margin:1em 0}summary{cursor:pointer}article{border-top:2px solid #ccd1d1;margin-top:2em}a{color:#15568b}.note{background:#fff7dd;padding:1em}</style>'
    pages, hashes = [], {}
    for family in chosen:
        for b in sorted(by_family[family], key=lambda b: b['policy']):
            body = ['<!doctype html><meta charset="utf-8">', style,
                    '<h1>' + html.escape(b['policy']) + '</h1>',
                    '<p class="note">This branch shares the first two historical A turns, then regenerates every later turn. Private feedback is part of each saved debater request. It is not in the public argument unless the debater repeats it. Public reasoning was disabled; provider-internal reasoning is unavailable.</p>',
                    '<p>Family: ' + html.escape(family) + '<br>Branch: ' + b['branch_id'] +
                    '<br>Complete: ' + str(b['complete']) + '</p>',
                    details('Private structured preparations (not preparation thinking)', b['private_preparations'])]
            problem = problems[b['problem_id']]
            body.append(details('Complete source problem', problem.__dict__))
            for i, turn in enumerate(b['turns']):
                body.extend(['<article><h2>Turn ' + str(i + 1) + ': ' + html.escape(turn['speaker']) + '</h2>',
                             pre(turn['argument']), details('Saved turn, assigned role, and resulting score', turn)])
                if i < 2:
                    body.append('<p>Shared historical prefix. Full original raw context is indexed in iteration6/raw_audit/turn_index.jsonl under request hash ' + turn['request_sha256'] + '.</p>')
                else:
                    path, receipt = receipts[turn['request_sha256']]
                    hashes[str(path)] = file_hash(path)
                    body.extend([details('Exact submitted request: system, source, private plan, history, feedback, settings', receipt['request']),
                                 details('Raw provider response body (verbatim)', receipt['response_body']),
                                 details('Full receipt and provenance', receipt)])
                body.append('</article>')
            filename = b['branch_id'] + '.html'
            (a.out / filename).write_text('\n'.join(body))
            pages.append({'family': family, 'policy': b['policy'], 'branch_id': b['branch_id'],
                          'complete': b['complete'], 'page': filename})
    index = ['<!doctype html><meta charset="utf-8">', style, '<h1>Live feedback: exact model-visible records</h1>',
             '<p>Examples selected by a fixed family hash, without inspecting outcomes. These are qualitative examples, not prevalence estimates. All seven display policies from each selected family are shown.</p><ul>']
    for item in pages:
        index.append('<li><a href="' + item['page'] + '">' + html.escape(item['policy'] + ' · family ' + item['family'][:12]) + '</a> · complete=' + str(item['complete']) + '</li>')
    index.append('</ul><p>All branch states and raw records: <a href="' + html.escape(os.path.relpath(a.run, a.out)) + '">production run directory</a>.</p>')
    (a.out / 'index.html').write_text('\n'.join(index))
    (a.out / 'manifest.json').write_text(json.dumps({'selection': 'hash rank grid6-inspection-v1:<family>',
        'selected_families': chosen, 'pages': pages, 'production_manifest_sha256': file_hash(a.run / 'manifest.json'),
        'receipt_sha256': hashes, 'script_sha256': file_hash(__file__)}, indent=2) + '\n')
    print(json.dumps({'families': len(chosen), 'pages': len(pages), 'complete_branches_shown': sum(p['complete'] for p in pages)}))


if __name__ == '__main__':
    main()
