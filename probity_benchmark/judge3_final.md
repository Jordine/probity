# Final judge package

The wrapper preserves `judge3_protocol.py` and its frozen message text. It
prepares identical complete jobs for API Haiku and native Qwen3.5-4B, then
audits each judge separately. Preparation makes no model calls. Root freezes
the package before fresh outcomes, including detector/head/reference choices.

The selected full package includes four response detectors, new local3
sentence/clause segments and old local2 sentence segments: 1,032 jobs over48
episodes. Old local2 clause exports are retained but not judged. The separate
Opus package contains only `adapted_primary` and `local3_sentence`, plus raw
baseline and oracle, across the same48 episodes:384 jobs. This reduced scope
is fixed before fresh outcomes and supports stronger-judge interactions for
those two primary signals only.

Create a design JSON following this structure. The head/reference hashes must
come from the development-frozen exports, before final judge outcomes:

```json
{
  "schema": "judge3_final_design_v1",
  "split": "test",
  "expected_episodes": 48,
  "expected_families": 24,
  "participant_models": ["qwen/qwen3.5-27b", "qwen/qwen3.8-27b"],
  "sources": "runs/iteration3/final/judge_sources.json",
  "family_index": "runs/iteration3/coordinator/source_family_map.json",
  "protocol_lock": "runs/iteration3/judge_controls/pre_source_review_protocol_lock.json",
  "validity_artifacts": [
    "runs/iteration3/coordinator/final_source_lock.json",
    "runs/iteration3/final/protocol_validity_review.json"
  ],
  "score_bundles": [
    {
      "path": "runs/iteration3/final/response_bundles/adapted_primary.json",
      "participant_models": ["qwen/qwen3.5-27b", "qwen/qwen3.8-27b"],
      "head_sha256": "DEVELOPMENT_FROZEN_HEAD_HASH",
      "reference_sha256": "DEVELOPMENT_FROZEN_REFERENCE_HASH"
    },
    {
      "path": "runs/iteration3/final/response_bundles/published_fibs_dyl.json",
      "participant_models": ["qwen/qwen3.5-27b"],
      "head_sha256": "DEVELOPMENT_FROZEN_FIBS_HASH",
      "reference_sha256": "DEVELOPMENT_FROZEN_FIBS_REFERENCE_HASH"
    }
  ]
}
```

Add Apollo response/DYL and frozen local sentence/clause bundles to the same
list. A detector must cover every turn of every episode from its declared
models. FIBS declares Qwen3.5 alone; missing Qwen3.8 weights are never reused
across models. All other declared coverage is checked exactly. Each source
family must have one selected question, equal participant coverage, and no
overlap with any detector's development reference families. The family index
uses original parent IDs even when question wording has been repaired.

For each detector, include `reference_artifact` and
`reference_artifact_sha256` from its saved development export. The artifact
can be a reference-only bundle or a prior complete score bundle. Preparation
checks its exact file hash and requires the final bundle's detector identity,
orientation, granularity, head/reference hashes and complete reference arrays
to match. This catches stale or changed arrays behind an unchanged hash label.
Local references are saved under `local_probes/references_local3_v1` and
`references_local2_v1`; response references are in the coordinator's old-data
smoke exports.

The frozen sentence/clause regexes define deterministic text segments, not
linguistically exact boundaries. Abbreviations can produce short fragments,
and punctuation before a closing quotation mark can leave several sentences
in one segment. Every character remains covered; these boundaries stay fixed
across neutral, real and shuffled displays.

The sources file has the existing frozen-builder format:

```json
[
  {
    "protocol": "FINAL_FROZEN_ENVIRONMENT_PROTOCOL",
    "dataset": "runs/iteration3/coordinator/holdout_families_source_v3.jsonl",
    "episodes": "runs/iteration3/final/debates"
  }
]
```

```bash
python3 -m probity_benchmark.judge3_final prepare \
  --design runs/iteration3/final/judge_design.json \
  --out runs/iteration3/final/judge_package
```

Review the printed package hash and condition counts. The package binds the
source/validity artifacts, frozen head and reference hashes, shuffle mapping,
complete messages and runner versions. Artifact presence is not an automatic
semantic validity certificate: the source, actual-context knowledge, advocacy
and truthful-evidence reviews remain explicit root-owned decisions. A changed
input or runtime requires a new versioned package; preserve the earlier one.

After the root freeze, run experiments in tmux. API execution requires the
shared budget ledger and the exact reviewed package hash:

```bash
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
python3 -m probity_benchmark.judge3_final run-api \
  --package runs/iteration3/final/judge_package \
  --frozen-package-sha256 REVIEWED_PACKAGE_HASH \
  --judge anthropic/claude-haiku-5.5 --budget 1.0 \
  --out runs/iteration3/final/haiku_judge
```

API execution keeps complete requests/outputs, context and completion gates,
and cumulative accounting. A failed run produces no aggregate; valid receipts
can be reused in a new directory with `--reuse-receipts FAILED/requests.jsonl`.
No failed or incomplete generation is silently scored as a wrong answer.

Run the native judge on the shared GPU using the same job file:

```bash
python3 -m probity_benchmark.run_local_judge \
  --jobs runs/iteration3/final/judge_package/compiled/jobs.jsonl \
  --model-spec PINNED_NATIVE_4B_SPEC.json --answer-only \
  --out runs/iteration3/final/native_judge
python3 -m probity_benchmark.judge3_final analyse \
  --package runs/iteration3/final/judge_package \
  --frozen-package-sha256 REVIEWED_PACKAGE_HASH \
  --results runs/iteration3/final/native_judge \
  --out runs/iteration3/final/native_judge_analysis
```

The API command also runs this analysis after all expected calls finish.
Analysis verifies unique complete job coverage, correct answer mapping, valid
stop receipts and the frozen job-file hash. API receipts additionally must
contain the exact complete message list for the associated job. Native
analysis retains its pinned model/runtime contract and answer-only format.
Receipt matching uses both the request hash and job identity: two separate
jobs with identical neutral prompts may return different stochastic answers,
and neither result may overwrite the other during audit or recovery.

`analysis.json` contains all cells and paired comparisons; `receipt_audit.json`
names the judge, package, family count and real-minus-neutral contrasts. Do
not combine API/native results into one row or report repeat calls as extra
independent families. Report all frozen detector comparisons, fixes and harms,
raw baseline, neutral formatting effects and shuffled harms. An all-correct
sample gives a degenerate bootstrap interval; it does not establish perfect
population accuracy. One source family is the independent unit here.

`prompt_repeatability.json` reports answer disagreement across byte-identical
message lists within each judge run. Several neutral detector displays are
identical, providing a descriptive sampling-noise check with no extra calls.
Those repeated presentations do not increase the independent family count or
form a separately randomized repeatability study.
Complete request hashes and returned provider/model/fingerprint metadata are
reported separately. Equal requests do not guarantee the same provider route
or hardware; missing routing metadata remains unknown.

The local integration smoke at
`runs/iteration3/judge_controls/final_wrapper_smoke_v1` uses only already-seen
iteration2 episodes. It compiled 400 jobs over32 episodes/eight families,
including16 model-matched FIBS episodes, and made no outcome calls.

Prepare the Opus package from a second design with only its two prespecified
signals, keeping the same sources, episodes, validity artifacts and frozen
reference artifacts. Run Opus using the same `run-api` command with
`--judge anthropic/claude-opus-5.5 --reasoning-effort low`, its own budget and
result directory. Its catalog requires reasoning; disabled reasoning is
rejected before any completion request. Freeze the chosen effort, completion
allowance (`--max-output-tokens`) and timeout (`--request-timeout-seconds`) as
part of the runtime comparison. Size the allowance from complete development
checks, without forcing short reasoning. Raw provider responses retain every
returned reasoning field; exposed native limits or forced/truncated flags
halt aggregation even if the normalized finish reason says stop. After
all frozen judge runs complete, compare them without changing the package:

```bash
python3 -m probity_benchmark.judge3_compare \
  --package runs/iteration3/final/judge_package \
    runs/iteration3/final/judge_package runs/iteration3/final/opus_package \
  --frozen-package-sha256 FULL_HASH FULL_HASH OPUS_HASH \
  --results runs/iteration3/final/native_judge \
    runs/iteration3/final/haiku_judge runs/iteration3/final/opus_judge \
  --out runs/iteration3/final/cross_judge
```

Pass one shared package/hash or one package/hash per result directory in the
same order. Every judge first passes the complete audit for its own package.
Pairwise analysis requires identical episode-family maps, exact common job
messages and metadata, equal common detector provenance, and raw/oracle
coverage for every episode. It reports excluded conditions explicitly and
only compares the shared conditions. `comparison.json` reports
each pair in the supplied order. The uplift interaction is the second judge's
real-minus-neutral change minus the first judge's change. A negative value
for a prespecified stronger second judge supports smaller observed uplift in
that run. All observations from each source family remain together in the
paired bootstrap. Raw/neutral accuracy and oracle headroom accompany this
comparison; accuracy alone cannot identify a difference in assistance uplift.
Intervals are unadjusted across the frozen contrasts, and no detector is
selected using their significance.
