# Qwen3.5 debate feedback and probe ablations

Status: complete. The historical audit/behavior analysis, all 13,029 primary
factuality labels, the independent reader audit, 168 display debates, 32 prompt
debates, all live/offline probe scores, both final-judge passes, sensitivity
analyses and plots are complete. The campaign spent $30.20 under the
user-approved $60 ceiling; all owned rentals are released. This specification records the
user's request of 18 October 2026.
The scope is Qwen3.5-27B on the existing debate environment. Other participant
models and environments are deferred.

## Questions and priority

1. What did the latest live probe feedback change in the debaters' arguments?
2. Which probe type and layer detect false or unsupported statements in these
   debates, including mixed truthful/false responses?
3. Which feedback display improves the subsequent debate and final judgment?
4. How do those findings depend on fabrication and defense instructions?

The first deliverable is a paired analysis and inspectable full-context record
of the existing run. It does not wait for new model generation.

Available deliverables:

- [Latest-run Qwen3.5 behavior report and plots](probity_iteration6_behavior.md).
- [Full raw-context coverage audit](probity_iteration6_raw_context_audit.md).
- [Expandable raw request/response examples](../runs/iteration6/raw_audit/index.html).
- [A/B arguments with exact preceding feedback](../runs/iteration6/behavior/qualitative_sample.html).
- [Probe, layer, prompt, and cost inventory](probity_iteration6_ablation_inventory.md).

The audit verified all 1,992 public requests/responses and all 332 private
preparations. Existing B trajectories show shorter and less repetitive text;
unassisted judge differences are uncertain and wrong-answer scores do not
decrease. Historical factuality labeling and its paired analysis are now complete.
See [the factuality report](probity_iteration6_factuality.md) for conditional
contradiction rates, label eligibility and independent-reader disagreement.
Fresh generation and scoring are tracked separately in their production coordinators.


## Implemented campaign scope

The stage descriptions below record the broader research plan. This bounded
campaign implemented the following subset; the reading guide distinguishes the
executed stages from deferred work.

- Historical Qwen3.5-27B: all 166 questions, 116 source families, 332 debates,
  1,992 public turns and 13,029 sentence labels; a 12-family independent-reader
  audit and offline comparisons of frozen probe heads and layers.
- Display screen: 24 fixed-hash families × seven participant displays. All
  168 branches share the original A first round within question and regenerate
  four suffix turns. All use the adapted layer-43 response probe for live feedback.
- Prompt screen: eight fixed-hash families × selective/explicit instruction
  blocks × feedback off/on. It generates new private plans and all six turns.
  On/off pairs share their newly prepared plans. It removes the inherited
  coherent-fabrication mandate and does not supply an external false account.
- Final judging: Haiku and native Qwen3.5-4B evaluate identical public arguments
  without score packets. The new screens occupy the left column of the earlier
  2×2 grid; historical C/D comparisons remain separate.
- Fresh display suffixes receive new primary factuality labels and offline
  probe evaluation. Prompt-screen factuality labeling is not included.

Deferred: fresh live trajectories for alternate probe types/layers; new local-head
training; sentence/phrase highlight displays; scored final-judge displays on the
new branches; calibrated thresholds/FPR analyses; externally specified false
accounts; further instruction levels; multiple generation seeds; and semantic
interaction labels. The 57 recorded score readouts include controls, duplicate
views and contrasts, not 57 independent detector families.

## Existing population and provenance

Use the frozen iteration5 source/inclusion inventory and exact canonical paths:
166 question rows from 116 normalized original source-title families; 166 A
and 166 B Qwen3.5 trajectories; six alternating turns each, 1,992 public turns.
A/C reuse one transcript; B/D reuse another. Five inherited question/source
repairs remain explicit. The entire corpus has now been examined: no portion
of this new analysis may be advertised as an untouched confirmatory test set.

All earlier source, receipt, head, normalization, and result files remain
immutable. New outputs go under iteration6. Reuse requires matching complete
context and artifact hashes, not identical visible arguments alone.

## What replay means

Offline scoring holds recorded messages and response text fixed. It can compare
alternative probes/layers on both A and B. For B, the original probe feedback
must remain in the recorded input. A new score measures detection conditional
on that historical policy; it is not a continuation under the new policy.

Final judge displays can change on fixed text. Changes to any debater-visible
probe, layer, display, or instruction require fresh generation from the first
changed request onward, fresh causal scoring, and new annotations of changed
text. No later historical turn can be spliced into a changed branch.

## Exact model-visible context

Audit every retained Qwen3.5 public request and both speakers' private preparation
requests. Export the original structured request, raw response body, request
and response hashes, provider identity when supplied, reasoning settings,
sampling parameters, usage, finish reason, and source receipt locator.

In historical B the system prompt announces private feedback from the opening
turn onward. Each turn's user message contains the complete source, question,
speaker's private preparation, public history, and private past-turn score
packet. That packet includes both speakers' completed turns; the current turn
has no score yet. Thus turn 1 has an empty packet and turn 2 already sees turn
1's score. Private feedback is not included in the public transcript shown to
the judge unless the final display condition supplies scores separately.

The preparation process requests thinking; public generation requests thinking
disabled. Preserve every provider-returned reasoning field verbatim and mark
unreturned reasoning as unavailable. Private structured plans and internal
reasoning are different artifacts. Exact submitted API messages can be
recovered; provider-internal serialization, undisclosed reasoning, and exact
deterministic regeneration cannot be guaranteed.

## Stage 1: latest-run analysis

Analyze all 166 Qwen3.5 A/B pairs. Report question-weighted counts and equal-weight
family means with paired family-bootstrap intervals. Keep model identities,
speaker assignment, opening order, source version, and judge separate.

Quantities: judge A/B/C/D accuracy and paired changes; percentile trajectories
by assigned role; per-speaker next-turn score changes; exact and lexical
repetition; length; observed answer switches; disclosure flags. These are
descriptive. Assigned wrong-answer advocacy is not a local deception label.

For the first report reuse the two Qwen3.5 inspection cases chosen before the
expansion outcomes, and label them illustrative rather than prevalence evidence.
Use a fixed hash-based sample for the larger subsequent semantic audit. Display
complete A/B histories, the exact prior feedback, and the next
response. Review corrections, fabricated quotation, contradictions, misleading
inferences, opponent engagement, and apparent attention to feedback. Do not
report keyword counts as semantic judgments or selected cases as error rates.

## Stage 2: factual annotation and offline probe comparison

Compile full-source annotation jobs for every Qwen3.5 A/B turn, reusing prior
annotations only on exact source/text/unit/rubric identity. Annotators do not
receive assigned role, probe values, experimental arm, or authentic-answer
instructions. Executed factuality jobs contain the full source and current
complete argument, without previous public turns. Claims about an opponent
therefore cannot be resolved from earlier debate context in this rubric;
contextual attribution and interaction labeling remain separate future work.

Start with fixed sentence spans to preserve comparability with previous labels.
Record supported / contradicted / unresolved verdicts, source evidence, quotation
accuracy, and mixed-fact/interpretive scope. Mixed or unresolved units must not
silently become truthful negatives. Add claim decomposition as a separate
versioned analysis if sentence ambiguity prevents useful local evaluation.
An independent annotator audits a fixed random family-stratified subset;
agent inspection is not human adjudication. Save disagreements and coverage.

Report AUROC/AUPRC where both classes exist, within-response discrimination,
per-role results, uncertainty clustered by source family, and performance on
mixed responses. Include token position and shuffled/random-head controls.
The source prompts explicitly disclose truth and assignment to each advocate,
so a detector may identify its instructed role without identifying false claims.
Report supported and contradicted units within each role and mixed response;
any prompt-only diagnostic is a separate context ablation, not faithful replay.
Calibrate thresholds on development families only; report false-positive rates
and coverage alongside discrimination. Never select on the reported test fold.

Candidate probe types: existing adapted response head, Apollo response head,
Apollo DYL readouts/contrasts, compatible published FIBS/DYL, existing local
sentence/clause heads. Inventory compatibility before promising a full matrix.
Layer ablations require heads actually trained for each layer and matching
model/feature contracts; a head cannot be moved to a different layer merely
because dimensions match. Candidate extracted layers begin with the existing
six-layer contract. Distinguish fixed published heads from retrained heads.

Replay complete original context without truncation. Check token mappings,
returned reasoning, model revision, chat template and pooling scope. Batch
compatible projections during the same forwards. Persist scores, unit offsets,
hashes and needed compact pooled features; do not persist raw token activations
to VPS disk. Run parity checks against the frozen selected head first.

## Stage 3: fresh live feedback experiment

Historical A/B results remain their own comparison. A new shared-prefix design
answers a narrower question: what happens when feedback is introduced after
the same first round? Reuse the first two turns of A (one per speaker), private
plans, roles and source, then regenerate all four remaining turns for every
arm, including a contemporaneous no-feedback control. No branch has feedback
awareness before the fork. This differs explicitly from historical B, which
has awareness at turn 1 and numeric feedback at turn 2.

Use identical generation settings and interleave arm requests to reduce route
and time confounding. Pin provider route where feasible and record actual
provider/revision availability. At each subsequent turn: generate, validate
and save the complete response; score it with that branch's selected probe;
only then construct the next request. Never include a current/future turn score.

Initial display conditions: no feedback; matched neutral packet; current
bundle; per-turn percentiles only; speaker means only; recommendation only;
shuffled-control bundle. Use a deterministic predeclared within-prefix shuffle
as an explicitly weak control (it can be an identity permutation or preserve
speaker means); do not describe it as guaranteed destruction of signal.
Local highlights are added when the appropriate local probe and offsets pass
validation. Controls must not use gold labels or future trajectory scores.

Screen on a deterministic family sample independent of outcome and score;
hold other families out of new fitting/selection. All-corpus follow-up remains
exploratory because of historical exposure. Sampling uncertainty from multiple
generation replicates is a later axis; temperature-zero duplicates are not
independent seeds. Candidate decisions and uncertainty are reported even when
no condition improves the outcome.

Use both existing final judges to separate weak/strong judge behavior. Judge
sees question/options and public debate, never the authentic source or private
preparation. The broader follow-up plan evaluates raw and scored views on identical branch
text, preserving the 2×2 mapping. This campaign executes only the unassisted
judge views on new branches; scored new-branch views are deferred.

## Stage 4: prompt ablations and selected interactions

Inventory original Probity prompt literals and current instructions first.
Changing preparation instructions, the externally supplied fabricated account,
or defense strength requires new plans and a new matched A/B pair under that
prompt condition. A shared historical prefix is invalid for those changes.

Compare current self-generated fabricated account with a precisely specified
false account and selected original defense levels. Keep truth access, question,
role assignment, token budget, and output contract matched where possible;
record unavoidable differences. Test leading probe/display combinations under
each prompt, including a common reference combination. Separate one-axis
screening from follow-up interactions: no claim of universal ranking from one
probe under one prompt. Local-head training uses source-family-separated folds.

The inspected original `debate_manager.py` supplies symmetric, persuasive,
selective, and explicit instructions; these functions do not supply a detailed
external false account. An externally specified fabrication is therefore a new
intervention unless the intended historical template is located. Do not call
it a reproduction of an original condition without that provenance.

## Outcomes, execution gates, and deliverables

Primary live outcome: unassisted judge accuracy on completed branches, paired
by question/prefix. Co-primary behavioral evidence: factual-error changes
after the fork under blinded source-based annotation. Secondary: assisted judge
accuracy, within-response detector performance, evasion conditional on false
claims, corrections, contradictions, length, repetition and role adherence.
Include missingness, ambiguity, technical failures, route heterogeneity and
all hypotheses tested. A score decrease alone is not evidence of less deception.

Required checks: exact prefix and source identity; no future-score access;
branch-specific receipt/score hashes; no imported later turns; byte-preserved
returned responses; full context/token checks; family-aware analysis; closed
budget liabilities and rental cleanup. Technical retries keep all attempts and
do not select on answer accuracy. Scientific validity failures remain outcomes.

Deliverables: versioned spec; complete raw-context browser/exports; paired
behavior figures/tables and qualitative examples; runnable offline campaign
manifest; tested display/branch compiler; annotation and probe evaluation
artifacts; live ablation results and final researcher-readable report as each
paid stage completes. Finished, prepared, and not-yet-run work are listed
separately. A new budget must be bounded before paid execution; historical
iteration5 headroom is not silently reassigned to this expanded campaign.

## Initial paid-stage proposal (dated design record)

This section preserves the pre-execution gates and estimates. The executed
scope and final state are summarized above and in the results report; phrases
such as "prepared" below describe the state when this plan was frozen.

The user approved a **$60 hard ceiling** ("60 USD cap allowed."). The shared
ledger is `runs/iteration6/budget.json`; prior iteration5 spending stays separate.
Planning allowances are $8 for primary complete-source labels, $12 for a roughly
10% second-model audit, $25 for GPU replay/live scoring, $5 for continuation
generation and final judging, and $10 for recovery. These are portions of the
single $60 cap, not independent authorizations to exceed it. These are planning bounds
from historical receipts, not current provider quotes or charges. Verify prices
and measured token/throughput costs before committing; stop at the cap and
report unfinished axes instead of silently exceeding it.

The prepared display screen chooses 24 families and one question per family
by deterministic hashes before inspecting new outcomes. Seven conditions give
168 branches, each needing four fresh turns: 672 new public generations.
Additional probe/layer live branches and full prompt sweeps are later stages;
the $60 ceiling is not an estimate for the full Cartesian product. Freeze
exact candidate and stopping decisions before their paid results are seen.

Run offline preparation from the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_grid6*.py' -v
python3 -m probity_benchmark.grid6_protocol --root . \
  --out runs/iteration6/protocol/screen24_v2 --families 24
python3 runs/iteration6/raw_audit/audit_raw_context.py
python3 runs/iteration6/behavior/analyze.py
python3 runs/iteration6/planning/run_annotation_jobs.py
```

The branch compiler writes complete first-continuation messages and immutable
prefix identities. Its output is explicitly `prepared_not_generated`; it is
not a generation runner or evidence that any new branch has completed.
The v2 contract fixes source/plan/policy mutation and historical-suffix reuse
gaps found in independent review. It preserves the v1 selected questions and
first messages; v1 remains an unused draft. Twenty-five unit tests cover the
branch contract and label-tier assembler; six additional annotation-runner
tests cover authorization, blinding, exact text, and durable resume. The shuffled control is weak: it is
unchanged for 17 of 24 initial forks and may preserve aggregate information.
Do not interpret this control alone as a successful signal-destruction test.

The first annotation compilation contains 13,029 fixed sentence units from
1,992 turns: 967 units in 144 turns have earlier exact-context label reuse,
and 12,062 units in 1,848 turns need 379 new primary annotation jobs. The
unchanged sentence rubric measures source factuality. Semantic interaction
labels (responding to an opponent, retracting a claim, self-contradiction) need
a separate contextual rubric and are not supplied by these factual jobs.

`probity_benchmark.grid6_labels` assembles uniform primary-reader labels and
separate two-reader consensus labels. Legacy dual-reader judgments must not be
mixed with new single-reader labels under one consensus metric. In the pre-execution snapshot,
`runs/iteration6/labels/existing_only_v2/` assembly marks all 12,062 new units
unannotated. Within the 967 old units, the primary reader has 707 eligible
labels (379 supported, 328 contradicted); two-reader consensus has 548 (257
supported, 291 contradicted). Disagreements and invalid/unknown scope remain
visible, and the corpus is explicitly incomplete.

Historical annotation execution is complete. The executor requires pinned
job and runner hashes, an explicit model allowance and campaign cap, and a
campaign ledger. It saves raw receipts and discovers previous logical job/model
completions independently of mutable price metadata. Unknown in-flight costs
block resubmission. Context/cost bounds are checked before calls. The historical
cheap annotator is Qwen3.8; that is a labeler, not an additional debater model.
Current prices and context bounds were checked before paid calls. The independent
audit completed all 36 new Opus jobs for $8.35340, including smoke calls, with no
failed or repeated calls. The complete primary population is assembled: 5,156
supported, 3,893 contradicted and 3,980 excluded/unknown-scope sentence units.
Invalid returned schema items are retained with their raw receipts; valid units
can be salvaged only after independent exact-text/evidence validation. Versioned
repair jobs cover unresolved units, without changing the original population.

## Execution additions and numerical reproducibility

All 1,992 historical contexts passed native tokenizer/offset checks. Maximum
input length is 11,036 tokens; no truncation is permitted. The 96 GB rentals
used for scoring had fixed cleanup deadlines and are now released. Persistent probe outputs contain scores, offsets,
token IDs and provenance, not raw activation tensors.

The initial four historical reference cases passed exact scalar parity. Further
rows exposed small differences from saved historical raw scores. On 12 diagnostic
cases, the original single-head scorer and new shared-forward scorer were
bit-identical on the current GPU. Historical serialized-input, request, response,
model, head, reference, version and numerical-setting identities also matched.
Four sampled historical deviations were at most 0.00432 raw-score units and did
not change displayed percentiles. This is consistent with native numerical
variation across runs; it does not identify a particular kernel as the cause.

The additive operational replay runner preserves the frozen scoring code and
records historical raw/percentile differences on every row. Fresh live work is
gated by the current 12-case scorer-equivalence check, four historical reference
cases, worker liveness, and a pause on large displayed historical drift. The old
B feedback and calibration references are unchanged. Do not claim bit-identical
reproduction of every historical score.

A separate instruction screen was authorized for at most $3 of the same $60:
eight fixed-hash families, selective versus explicit original instruction blocks,
feedback off/on, 32 new private plans and 32 fresh six-turn debates. It ran
after the display experiment completed. Private planning retained reasoning enabled and a 16,384-token
completion ceiling; public turns retain reasoning disabled and an 8,192-token
ceiling. These are original instruction blocks in the current harness, not an
end-to-end reproduction of the old code. No externally specified false-account
condition is included. See `runs/iteration6/live/prompt8_spec_v1.md` for the frozen
design and provenance.

Fresh display suffix factual labeling has a separate $3 allocation within $60,
conditional on complete 672-turn exports. This measures factual changes in the
new arguments; historical labels cannot be transferred to changed text. Neither
conditional allocation is extra funding above the campaign cap.

Consensus-tier coverage combines earlier convenience labels with the new fixed-hash
12-family audit. It is not representative of the complete corpus. The independent
reader-agreement report restricts its denominator to the frozen 12 audit families;
primary-tier factuality estimates use the consistent primary reader throughout.

The completed replay diagnostic covers all 1,992 turns: 1,556 raw scores match
exactly, 1,650 differ by at most 0.0001, and the maximum absolute raw difference
is 0.01504. Twelve displayed percentiles change, by at most 2.084 points; 1,980
remain identical. No five-point pause threshold was reached. These results
supersede the earlier four-deviation sample above. Original historical feedback
was retained in every input.

Fresh-label process recovery preserves the interrupted batch's full $3 cost
bound, including eight unreturned in-flight jobs and all 77 returned receipts.
An additional $2 was allocated from campaign recovery within the unchanged $60
ceiling, raising this stage's conservative bound to $5. The original $3 is a
conservative liability, not a verified charge. Exact accepted units are reused;
only unresolved work is requested again. Original logs and job-journal state
remain archived.
