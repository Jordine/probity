# Reading the current Probity experiment

This study asks whether a frozen activation-based signal helps a judge select
the correct answer after a debate. Two advocates using the same Qwen27B model
defend opposing answers; a separate judge makes the final choice. The main
metric is **judge answer accuracy against the provided key**. It is distinct
from debater source-knowledge accuracy, probe discrimination of assigned roles,
and accuracy of local factual labels.

The [24-family pilot is complete](probity_iteration4_results.md), including its
plots and a separately reported Haiku replay. The
[166-question expansion](probity_iteration5_plan.md) is ongoing as of
16 October 2026. All expansion counts below are targets until its completion
and coverage audit. Do not read pilot plots as expanded results.

## The two-by-two design

| During public generation | Judge sees public debate, without scores | Judge also sees final score packet |
| --- | --- | --- |
| Debaters receive no scores | **Cell A** | **Cell C**, exactly A's transcript |
| Debaters receive past-turn scores | **Cell B** | **Cell D**, exactly B's transcript |

A question × participant-model condition has two generated six-turn debates
and four judge presentations. Both advocates use Qwen3.5-27B in one condition,
and Qwen3.8-27B in the other. These are separate same-model debates, not
Qwen3.5 versus Qwen3.8. Each judge evaluates both participant conditions.
Cell letters are experimental conditions; the answer labels A/B in actual
prompts are a different use of those letters.

| Comparison | What changes |
| --- | --- |
| B − A | Feedback instructions and available scores during generation; a different public trajectory |
| C − A | Adding the final score display to the identical no-feedback transcript |
| D − B | Adding the final score display to the identical feedback transcript |
| D − A | Feedback during generation and the final judge display, with a different trajectory |
| (D − B) − (C − A) | Difference between the two display-associated effects |

The trajectory comparisons do not isolate numerical feedback from the
awareness instruction or generation/provider variation. The display comparisons
use no display as their baseline; they are not matched neutral-display
comparisons. An interval containing zero does not establish equivalence.

## What each model sees

Both advocates receive the authentic source, question, options and designated
correct answer. Each is assigned an answer and privately prepares exactly
three claim/quotation pairs plus an anticipated challenge and response. The
wrong-answer advocate is explicitly instructed to invent a coherent account
with plausible quotations and maintain it. The detailed false account is
self-generated in that preparation, then supplied as a concrete plan during
the public debate. It is not an externally supplied alternate story.

There are three sequential rounds, six turns total. Each turn receives the full
source, its speaker's plan and all earlier public arguments/answer fields.
The prompt asks for two completed paragraphs with concrete quotations and a
rebuttal, while keeping private instructions and preparation private. Private
preparation uses reasoning at temperature zero; public turns use temperature
zero with reasoning disabled. Returned private reasoning is not exchanged as
public debate text. The two trajectories reuse the same private plans and
source, option order, assignments and schedule.

In B/D, both advocates additionally receive scores for **both** speakers'
completed turns, running speaker means and an optional recommendation for
the lower-mean speaker's assigned answer. The recommendation is absent when
there are not two distinct speaker means. Only the very first public turn has
no numerical history. The second speaker's opening already receives the first turn's score.
The feedback-availability instruction appears from the first turn. This is
in-context feedback, with no weight update or score-based rewriting loop.

The final judge receives the question, options and all public turns, plus
the final score packet in C/D. It receives neither the source story nor
private plans or generation instructions. Judges do not cross-examine the
advocates. The two judges are native bf16 Qwen3.5-4B and API Haiku5.5, both
under their recorded answer-only contracts.

These are prompt instructions, not guarantees about the generated prose.
Correct-answer advocacy does not certify every quotation as true. An authentic
answer supplied in context does not establish the model's latent knowledge or
intent. See the [exact protocol](probity_iteration4_protocol_scope.md) for
token allowances, response schemas and differences from legacy prompt switches.

## What the probe and display mean

| Participant API alias | Native replay head | Scored text |
| --- | --- | --- |
| `qwen/qwen3.5-27b` | Adapted response-mean logistic head, layer index 43 | Complete final assistant response |
| `qwen/qwen3.8-27b` | Adapted response-mean logistic head, layer index 35 | Complete final assistant response |

The pipeline replays the complete API context and response on a pinned native
checkpoint. It selects output states from the indicated `model.layers[index]`,
mean-pools the final response's token activations, and applies a frozen linear
head. JSON and answer-field tokens are included; earlier-message and returned
reasoning tokens are excluded from the pooled feature. The resulting score is
mapped to a 0–100 percentile against a fixed model-specific development
reference. A speaker's displayed mean is the mean of its turn percentiles.

Higher percentiles indicate association with assigned wrong-answer advocacy.
They are **not calibrated probabilities of deception or local lies**. The
text/JSON display bundles turn percentiles, speaker means and an optional
answer recommendation. This grid does not isolate those components and does
not display colored sentence-level annotations. The selected heads used eight
earlier training families and four development families.

Native replay provides the activations; they are not hidden states exported by
the API provider. A model alias alone cannot prove identical serving weights.
Full-context/token-selection checks and score provenance accompany the private
results. Raw activation tensors are not saved to the VPS. Apollo/DYL, local
sentence/clause probes and other displays have earlier separate evaluations;
they are not additional probe conditions in this grid.

## Coverage, reuse and interpretation

| Unit | Completed pilot | Expanded target, including pilot | Additional expansion target |
| --- | ---: | ---: | ---: |
| Question rows | 24 | 166 | 142 |
| Original story families | 24 | 116 | Not 142 independent families |
| Question/model units per cell, per judge | 48 | 332 | 284 |
| Distinct public trajectories | 96 | 664 | 568 |
| Public turns | 576 | 3,984 | 3,408 |
| Private plans | 96 | 664 | 568 |
| Final decisions per judge, across all four cells | 192 | 1,328 | 1,136 |

All 48 pilot question/model pairs (24 questions × two participant models),
their trajectories, scores and primary
judgments are reused exactly where identities match. Technical retries and
cache hits are not new seeds or independent replications. The later Haiku A/C
replay remains a separate sensitivity; it does not replace the expansion's
cached primary pilot baseline.

The expanded corpus includes prior training/development/evaluation use.
Only **28 families, covering 35 questions**, had no recorded use before this
expansion. Report these separately. The all-corpus estimate describes this
corpus; it is not a fresh 116-family held-out test. Multiple questions from one
family must remain in the same resampling cluster.

Models, prompts, probe and display stay fixed, but inclusion broadens. The
pilot passed source-knowledge and public answer-field retention gates. New
questions remain included when those diagnostics fail. The five accepted pilot
source repairs are retained; the other 142 questions retain their raw input
fields, including seven with flagged historical alternative repairs. Provided
keys stay fixed. A source-only reviewer supplies a fallible support diagnostic,
not independently certified ground truth.

Family-macro accuracy first averages question/model units within each family,
then gives every family equal weight. Question-micro accuracy instead weights
every unit equally. A micro numerator/denominator need not reproduce the macro
percentage. Paired differences are formed within units before family averaging;
10,000 family bootstrap resamples give unadjusted 95% percentile intervals.
Fixes and harms are counts of question/model units, not family counts.

The intervals describe empirical family variation, not variation over repeated
generation seeds or model serving. Comparisons across judges, models, contrasts
and overlapping cohorts are exploratory and are not multiplicity-adjusted.
In selected cohorts, report retained families, questions and units by model:
selection can leave only one model or some questions within a family.

The frozen analysis includes a source-supported sensitivity and a joint
source-supported/eligibility sensitivity. The latter changes both source and
eligibility composition; it does not isolate the old eligibility gate.
Eligibility also uses public role retention after generation, so selected
results are descriptive. The additive
[eligibility-only analyzer](../reports/grid5_eligibility_supplement.py) will
report that selection separately once complete inputs exist. It first
reconstructs the frozen reference cohorts, then applies eligibility without
requiring source-reviewer support. None of these diagnostics certifies
sentence-level truth.

## Evidence and reproduction

| Artifact | Purpose and availability |
| --- | --- |
| [Pilot report and figures](probity_iteration4_results.md) | Public completed aggregate results, including the separate replay |
| [Expansion plan](probity_iteration5_plan.md) | Public frozen scope, inclusion policy and reviewed recovery history |
| [Current harness](../probity_benchmark/README.md) | API/native environment split; distinct from legacy setup |
| `reports/data/iteration5/grid_summary_v1.json` | Planned public completed aggregate summary; its name alone is not completion evidence |
| `runs/iteration5/coordinator/source_inventory_v2.json` | Private frozen source, exposure, schedule and repair provenance |
| `runs/iteration5/coordinator/execution*_freeze_v*.json` | Private original and recovery execution hashes; recovery adds history, not independent trials |
| `runs/iteration5/judge_grid_v1/pairs.json` | Planned private join from question/model unit to the two episodes and four judge jobs |
| `runs/iteration5/analysis_v1/paired_episodes.jsonl` | Planned private paired scientific observations |
| `runs/iteration5/coordinator/final_analysis_complete_v1.json` | Required final analysis completion marker, checked alongside coverage and cleanup receipts |

Private replication requires the source stories, private plans, complete public
transcripts, exact submitted messages, returned outputs, probe/reference
artifacts and runtime receipts. Public aggregates permit figure reconstruction,
not independent recreation of the raw judgments. Provider-internal state or
reasoning that was not returned is unavailable. Private evidence is not
automatically exported with the public report.

To inspect a matched example, start with its `pairs.json` entry. Follow its
`no_feedback_episode_id` and `feedback_episode_id` to their full episodes.
For each public turn, join `request_sha256` to the request receipt; its `request.messages` is the
authoritative submitted prompt. The parsed plan is not the entire provider
response. Follow the pair's `jobs` mapping (A/B/C/D to job IDs) to the compiled
judge messages, then join API receipts by `tag.job` or native results by job ID. Check assigned
and actual answers, prior-turn history, feedback availability and score/head
provenance together. Do not join by question text or treat an episode's
diagnostic `judge` field as the final grid result.

The pilot's no-feedback episodes and A/C jobs depend on iteration-3 artifacts;
its feedback episodes and B/D jobs live in iteration 4. Numbered preparation
recovery directories retain old cache hits and failed attempts; they are not
different scientific samples. These dependencies must accompany any private
evidence handoff.

For a plot-only reader with NumPy and Matplotlib, the **completed public pilot**
can be replotted without API credentials or a GPU:

```bash
python3 -m reports.plot_grid4_results \
  --analysis reports/figures/iteration4/grid_v1/grid_summary.json \
  --out /tmp/probity-reader-grid4
```

Once the expansion's audited aggregate summary exists, its original figure
renderer is likewise offline:

```bash
python3 -m reports.plot_grid5_results \
  --summary reports/data/iteration5/grid_summary_v1.json \
  --out /tmp/probity-reader-grid5
```

The additive reader pack uses the same completed public summary, validates
aggregate arithmetic, and makes the macro/micro distinction and all five
untouched-family contrasts explicit:

```bash
python3 -m reports.plot_grid5_research_pack \
  --summary reports/data/iteration5/grid_summary_v1.json \
  --out /tmp/probity-reader-grid5-pack
```

It provides a PDF, PNG pages, cohort/model composition CSV and a provenance
manifest. Both expansion commands are conditional on actual completed results;
test fixtures are not experiment outputs. The original frozen renderer and
analysis remain unchanged. The reader pack does not independently reconstruct
bootstrap intervals from aggregates; the private eligibility supplement
performs reference-statistic reconciliation using complete paired observations.

The [recovery reporter](../reports/build_grid5_recovery_report.py) is also
offline. With private run access, this writes a dated private audit and
separate aggregate JSON/Markdown:

```bash
python3 -m reports.build_grid5_recovery_report \
  --out /tmp/probity-recovery-snapshot
```

It can report an in-progress snapshot, but requires actual completion and
cleanup evidence before labeling the run complete. Only the aggregate outputs
are candidates for a reviewed public release. These tools do not publish or
send evidence to colleagues.

The private `grid5_evaluate analyse` command can recompute statistics from
reviewed paired packages and raw result inputs; use hashes from the release
audit, not freshly blessed arbitrary files. Do not use the coordinator's
`finish_analysis_v1.py` as a plot-only reproduction command: it also launches
paid final judging when that stage is absent.

## Completion, recovery and remaining scope

A complete accepted grid does not mean all attempts succeeded. The final
release must separately report first-attempt success, technical failures,
accepted retries and missing jobs by model. Retrying an identical request is
technical continuation, not an extra seed. The rejected 82,150-token private
preparation remains invalid and charged; a reviewed retry does not erase it.
Five earlier unanswered requests retain unknown-charge allowances even though
their retries succeeded. Known charges, unknown liabilities, active batch
reservations and GPU accounting must be reported separately. The expansion's
campaign ceiling is $45 and its preparation ceiling is $11.

The active campaign extends dataset coverage with the existing probe, display,
participants and judges. It does not add repeated trajectories, new
probe/layer/pooling conditions, local feedback, full transcript truth labeling,
original dishonesty-level sweeps, externally specified false accounts, matched
neutral/shuffled debater feedback, or model-organism deception absent from
context. Earlier related experiments remain separate evidence. The
[paper options](probity_paper_options.md) distinguish a descriptive benchmark
paper from claims that require those additional controls.
