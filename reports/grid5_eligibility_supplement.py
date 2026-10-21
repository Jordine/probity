"""Offline eligibility-only sensitivity from reviewed complete grid5 evidence.

Only aggregate JSON/CSV and hashes leave this module. Never calls model services,
changes source keys, or modifies the frozen analysis or diagnostic artifacts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from probity_benchmark.grid4_evaluate import grid_statistics, PAIR_CONTRASTS

COHORT = 'secondary:eligible_only'
REFERENCE_COHORTS = ('all', 'secondary:source_supported', 'secondary:source_supported_and_eligible')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(row):
    """Unknown diagnostic eligibility is excluded, even with otherwise passing checks."""
    model = row['participant_model']
    return (row['secondary_diagnostics']['eligibility_by_model'][model] is True
            and row['no_feedback_knowledge_correct'] is True
            and row['feedback_knowledge_correct'] is True
            and not row['no_feedback_role_switches']
            and not row['feedback_role_switches'])


def validate_paired(rows, diagnostics, questions, families, units, judges):
    require(diagnostics.get('schema') == 'grid5_secondary_diagnostics_v1', 'wrong diagnostics schema')
    require(diagnostics.get('outcome_blind_to_judges') is True, 'judge-blind diagnostic provenance required')
    by_question = {}
    for entry in diagnostics['rows']:
        q = entry['matched_question_id']
        require(q not in by_question, 'duplicate diagnostic question')
        by_question[q] = entry
    require(len(by_question) == questions, 'diagnostic coverage mismatch')
    groups, identities, model_set, family_map = {}, set(), set(), {}
    for row in rows:
        for key in ('judge', 'family', 'participant_model', 'matched_question_id',
                    'no_feedback_episode_id', 'feedback_episode_id'):
            require(isinstance(row.get(key), str) and row[key], 'missing paired identity: ' + key)
        judge, question, model = row['judge'], row['matched_question_id'], row['participant_model']
        identity = judge, question, model
        require(identity not in identities, 'duplicate question/model/judge unit')
        identities.add(identity); model_set.add(model)
        for cell in 'ABCD':
            require(type(row.get(cell)) is bool, 'missing/nonboolean paired cell: ' + cell)
        for check in ('no_feedback_knowledge_correct', 'feedback_knowledge_correct'):
            require(type(row.get(check)) is bool, 'missing/nonboolean knowledge check')
        for check in ('no_feedback_role_switches', 'feedback_role_switches'):
            require(type(row.get(check)) is list, 'missing/non-list role checks')
        require(question in by_question and row.get('secondary_diagnostics') == by_question[question],
                'embedded diagnostics do not match reviewed diagnostics')
        entry = by_question[question]
        require(entry['source_status'] in ('supported', 'contradicted', 'ambiguous', 'unreviewed'), 'unknown source status')
        require(model in entry['eligibility_by_model'], 'missing model eligibility')
        value = entry['eligibility_by_model'][model]
        require(value is None or type(value) is bool, 'eligibility must be boolean or unknown')
        if value is True:
            require(eligible(row), 'diagnostic eligibility contradicts actual knowledge/role checks')
        require(question not in family_map or family_map[question] == row['family'], 'question changes source family')
        family_map[question] = row['family']
        groups.setdefault(judge, []).append(row)
    require(set(groups) == set(judges) and len(groups) == 2, 'complete two-judge coverage required')
    require(len(model_set) == 2, 'complete two-model coverage required')
    require(len(family_map) == questions and len(set(family_map.values())) == families, 'source-family/question coverage mismatch')
    for entry in by_question.values():
        require(set(entry['eligibility_by_model']) == model_set, 'diagnostic model coverage mismatch')
    reference = None
    for judge, selected in groups.items():
        require(len(selected) == units == questions * 2, 'complete question/model units required')
        require({(r['matched_question_id'], r['participant_model']) for r in selected} ==
                {(q, m) for q in by_question for m in model_set}, 'incomplete question/model cross-product')
        for field in ('no_feedback_episode_id', 'feedback_episode_id'):
            require(len({r[field] for r in selected}) == units, 'duplicate episode identity')
        provenance = {(r['matched_question_id'], r['participant_model']):
                      {k: v for k, v in r.items() if k not in ('judge', 'A', 'B', 'C', 'D')}
                      for r in selected}
        require(reference is None or reference == provenance, 'paired provenance differs across judges')
        reference = provenance
    return groups


def summarize(rows, judge, cohort):
    """Use the original family-macro paired/bootstrap implementation unchanged."""
    if not rows:
        return [], []
    pairs, outcomes = [], {}
    for i, row in enumerate(rows):
        jobs = {cell: f'{i}:{cell}' for cell in 'ABCD'}
        pairs.append({**row, 'jobs': jobs})
        outcomes.update({jobs[cell]: {'correct': row[cell]} for cell in 'ABCD'})
    accuracy, contrasts, _ = grid_statistics(pairs, outcomes, judge)
    for target in (accuracy, contrasts):
        for result in target:
            chosen = [r for r in rows if result['participant_model'] == 'all'
                      or r['participant_model'] == result['participant_model']]
            result.update(cohort=cohort, questions=len({r['matched_question_id'] for r in chosen}))
            if 'cell' in result:
                result.update(question_weighted_accuracy=result['accuracy'],
                              accuracy=result['family_macro_accuracy'])
            else:
                name = result['contrast']
                if name in PAIR_CONTRASTS:
                    pos, neg = PAIR_CONTRASTS[name]
                    micro = sum(int(r[pos]) - int(r[neg]) for r in chosen) / len(chosen)
                else:
                    micro = sum(int(r['D']) - int(r['B']) - int(r['C']) + int(r['A']) for r in chosen) / len(chosen)
                result.update(family_macro_difference=result['difference'], question_weighted_difference=micro)
    return accuracy, contrasts


def equal(actual, expected, label):
    if isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-10), 'reference mismatch: ' + label)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), 'reference mismatch: ' + label)
        for a, e in zip(actual, expected):
            equal(a, e, label)
    else:
        require(actual == expected, 'reference mismatch: ' + label)


def verify_reference(analysis, groups):
    for judge, rows in groups.items():
        supported = [r for r in rows if r['secondary_diagnostics']['source_status'] == 'supported']
        subsets = {'all': rows, 'secondary:source_supported': supported,
                   'secondary:source_supported_and_eligible': [r for r in supported if eligible(r)]}
        for cohort, selected in subsets.items():
            computed = summarize(selected, judge, cohort)
            for field, expected_rows, label in zip(('accuracy', 'contrasts'), computed, ('cell', 'contrast')):
                stored = [r for r in analysis[field] if r['judge'] == judge and r['cohort'] == cohort]
                require(len(stored) == len(expected_rows), 'reference cohort row coverage mismatch')
                index = {(r['participant_model'], r[label]): r for r in stored}
                require(len(index) == len(stored), 'duplicate reference aggregate')
                for expected in expected_rows:
                    key = expected['participant_model'], expected[label]
                    require(key in index, 'missing reference aggregate')
                    for k, value in expected.items():
                        require(k in index[key], 'missing reference statistic: ' + k)
                        equal(index[key][k], value, cohort + '/' + k)


def calculate(analysis, rows, diagnostics):
    """Pure calculation, including reference agreement; filesystem wrapper requires completion."""
    require(analysis.get('schema') == 'grid5_corpus_evaluation_v1', 'wrong analysis schema')
    require(analysis['bootstrap']['resamples'] == 10000 and analysis['bootstrap']['seed'] == 2026101422
            and analysis['bootstrap']['interval'] == 'unadjusted descriptive 95%', 'unexpected bootstrap contract')
    judges = sorted({r['judge'] for r in analysis['accuracy'] if r['cohort'] == 'all'})
    groups = validate_paired(rows, diagnostics, analysis['questions'], analysis['families'],
                             analysis['matched_units_per_judge'], judges)
    verify_reference(analysis, groups)
    accuracy, contrasts, composition = [], [], []
    for judge, all_rows in groups.items():
        retained = [r for r in all_rows if eligible(r)]
        a, c = summarize(retained, judge, COHORT)
        accuracy.extend(a); contrasts.extend(c)
        for model in ['all', *sorted({r['participant_model'] for r in all_rows})]:
            chosen = [r for r in retained if model == 'all' or r['participant_model'] == model]
            original = [r for r in all_rows if model == 'all' or r['participant_model'] == model]
            composition.append(dict(judge=judge, cohort=COHORT, participant_model=model,
                families=len({r['family'] for r in chosen}), questions=len({r['matched_question_id'] for r in chosen}),
                retained_units=len(chosen), planned_units=len(original), excluded_units=len(original)-len(chosen),
                unknown_eligibility_units=sum(r['secondary_diagnostics']['eligibility_by_model'][r['participant_model']] is None for r in original)))
    return {'schema': 'grid5_eligibility_only_supplement_v1', 'accuracy': accuracy, 'contrasts': contrasts,
            'composition': composition, 'bootstrap': analysis['bootstrap'],
            'reference_cohorts_verified': list(REFERENCE_COHORTS),
            'definition': 'Initial knowledge passes and no public answer-field role switches in either paired trajectory, with affirmative reviewed model-specific eligibility. Unknown is excluded. No source-support restriction; provided keys unchanged.',
            'interpretation': 'Descriptive post-generation selected secondary analysis; not a causal eligibility effect. Feedback inherits the initial knowledge observation. Families receive equal weight after averaging retained question/model units; retained model composition can differ. Unadjusted exploratory intervals do not estimate repeated-generation variation. Empty cohorts have zero composition counts and no invented estimates.'}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        if not rows:
            stream.write('# No eligible observations; no estimates.\n')
            return
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def build(analysis_path, paired_path, diagnostics_path, completion_path, reviewed_hashes, out):
    paths = dict(analysis=Path(analysis_path), paired=Path(paired_path),
                 diagnostics=Path(diagnostics_path), completion=Path(completion_path))
    require(set(reviewed_hashes) == set(paths), 'four explicit reviewed input hashes required')
    actual_hashes = {key: digest(path) for key, path in paths.items()}
    require(actual_hashes == reviewed_hashes, 'reviewed input hash mismatch')
    analysis, diagnostics, completion = [json.loads(paths[k].read_text()) for k in ('analysis', 'diagnostics', 'completion')]
    require((analysis['questions'], analysis['families'], analysis['matched_units_per_judge']) == (166, 116, 332),
            'actual output requires completed full expansion')
    require(completion['analysis_sha256'] == actual_hashes['analysis']
            and completion['diagnostics_sha256'] == actual_hashes['diagnostics']
            and completion['package_sha256'] == analysis['package_sha256'], 'completion marker does not bind inputs')
    require(analysis['secondary_diagnostics']['sha256'] == actual_hashes['diagnostics'], 'analysis diagnostic hash mismatch')
    require(completion.get('plot_manifest_sha256') and completion.get('public_summary_sha256')
            and completion.get('api_stage_receipts'), 'completed final-analysis marker required')
    require(completion['campaign_spending']['unsettled_entries'] == 0, 'unsettled completed run')
    rows = [json.loads(line) for line in paths['paired'].read_text().splitlines() if line.strip()]
    result = calculate(analysis, rows, diagnostics)
    # Detect any changing input before writing, as well as prior reviewed-hash agreement.
    require({key: digest(path) for key, path in paths.items()} == actual_hashes, 'inputs changed during analysis')
    result.update(input_sha256=actual_hashes, module_sha256=digest(__file__),
                  evidence_status='completed reviewed inputs; aggregate-only offline supplement')
    out = Path(out)
    require(not out.exists(), 'new immutable supplement directory required')
    out.mkdir(parents=True)
    (out / 'eligibility_only.json').write_text(json.dumps(result, indent=2) + '\n')
    for field in ('accuracy', 'contrasts', 'composition'):
        write_csv(out / (field + '.csv'), result[field])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis', 'paired', 'diagnostics', 'completion'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.analysis, args.paired, args.diagnostics, args.completion,
          {k: getattr(args, k + '_sha256') for k in ('analysis', 'paired', 'diagnostics', 'completion')}, args.out)
