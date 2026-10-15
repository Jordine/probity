# Frozen judge comparisons

`judge3_protocol.py` creates complete native judge jobs. Its raw baseline and
labelled oracle use exactly the same system instruction permitting optional
auxiliary signals. Every public argument remains complete. Public headers use
the response's answer field, never the private assigned answer.

The primary detector contrast is **real minus matched neutral**. Report the
ordinary raw baseline too, and the neutral presentation's effect relative to
it. Report real minus shuffled and shuffled minus neutral separately: damage
from shuffled advice must not be called uplift. The analysis pairs episodes
and weights parent stories equally before bootstrap resampling. Corrected and
harmed judgments are reported separately. No outcome-dependent exclusions are
allowed.

The labelled role oracle supplies the known correct answer and tests whether
the judge can use the display. It is not a deployable detector or a local-fact
oracle. The optional antioracle is a development-only compliance diagnostic;
it presents the opposite answer as fallible advice. Neither is evidence that
a probe recognizes deception.

Build candidate baseline/oracle jobs with a source JSON list:

```json
[
  {
    "protocol": "prepared_sequential_v1",
    "dataset": "runs/iteration3/environment/worlds_dev2.jsonl",
    "episodes": "runs/iteration3/environment/prepared_sequential_v1"
  },
  {
    "protocol": "legacy_v3",
    "dataset": "runs/iteration2/paired_worlds_v2.jsonl",
    "episodes": "runs/iteration2/debates_v3",
    "parent_source_lines": [90, 112]
  }
]
```

```bash
python3 -m probity_benchmark.judge3_protocol build \
  --sources sources.json --splits dev --out candidate_jobs
python3 -m probity_benchmark.run_local_judge \
  --jobs candidate_jobs/jobs.jsonl --model-spec qwen4b.json \
  --answer-only --out candidate_results
python3 -m probity_benchmark.judge3_protocol analyse \
  --jobs candidate_jobs/jobs.jsonl \
  --evaluations candidate_results/evaluations.jsonl \
  --out candidate_analysis.json
```

Full-story and question-only native competence checks remain separate; the
existing `build_native_judge_jobs` creates them. Full-source competence and
the debater's actual private-context knowledge must both be checked. A source
check performed before adding a confusing decoy packet does not validate
knowledge after that packet.

Add one or more `--score-bundles` for frozen detectors. A bundle is a JSON
object with this structure (the example numbers and hashes are placeholders):

```json
{
  "schema": "judge3_score_bundle_v1",
  "detector": "adapted_response",
  "target": "wrong_answer_advocacy",
  "granularity": "response",
  "orientation": "higher_more_unreliable",
  "head_sha256": "hash of the selected frozen head artifact",
  "reference_sha256": "hash of the development reference artifact",
  "references": {
    "qwen/qwen3.5-27b": {
      "split": "dev",
      "parents": ["development parent identifiers"],
      "scores": [-1.0, 0.0, 1.0]
    }
  },
  "rows": [
    {
      "episode_id": "episode identifier",
      "turn_index": 0,
      "argument_sha256": "sha256 of exact complete decoded argument",
      "spans": [[0, 1234]],
      "scores": [0.75]
    }
  ]
}
```

Every covered episode must contain every turn. The exact argument hash and
span coordinates are checked. `response` has exactly one full-argument span.
For local detectors use `target: local_factual_contradiction` and
`granularity: sentence` or `clause`; spans must equal `partition(argument,
granularity)` exactly. This fixed text-only partition includes every
character once. Annotation-selected claims, omitted unlabelled sentences,
and cherry-picked peaks are rejected. The local scores are associated with
contradiction, not automatically intentional lies or calibrated probabilities.

Local neutral, real, and shuffled displays contain the same complete units
and text, with absent scores represented by null. Thus merely selecting and
showing a claim cannot reveal its label. Response signals include speaker
means and a fallible recommendation from the lower mean. Local signals do
not automatically recommend an answer: supported claims can appear in a
misleading argument and contradicted details in a correct-answer argument.

Normalizers must be frozen development score distributions. Test parents
overlapping their reference parents are rejected. The pipeline must also
retain the upstream model/layer/head-selection provenance; the renderer
cannot establish that a declared artifact was selected without test access.

Response-score controls permute complete trajectories across parents within
model, protocol, split, and turn-count cells. Local controls permute scalar
unit scores across parents within model, protocol, and split cells, preserving
the exact score multiset. The latter does not preserve trajectory correlation
because complete arguments have unequal sentence counts. Neither shuffle
uses truth labels or searches for a desirable outcome. A parent contributing
more than half a cell makes a strictly cross-parent permutation impossible;
the builder stops instead of silently accepting same-parent donors.

Candidate selection precedes the final holdout: require unambiguous valid
questions, strong full-source competence, correct private-context knowledge,
prose adherence, and source-supported correct-answer advocacy. Review those
checks before judge/probe outcomes. A lower judge score obtained by confusing
the debaters about which source is authentic is a failed candidate, not
useful deception headroom. B−A and D−B require new feedback trajectories;
these offline C−A comparisons do not estimate them.
