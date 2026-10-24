# Probity debate benchmark for in-context deception probes

Status: proposed next benchmark, 24 October 2026

## Objective

Turn the Probity debate environment into a benchmark for probes that monitor
deception during a multi-turn interaction. The benchmark must distinguish three
questions:

1. **Passive detection:** can a frozen probe detect a deceptive turn or locate a
   deceptive span in a fixed transcript?
2. **Probe-in-the-loop behavior:** how does a debate change when participants
   receive live feedback from a particular probe?
3. **Decision support:** does exposing a final judge to a particular probe help
   it select the source-supported answer on fixed debate text?

The existing B trajectory was generated using one adapted layer-43 response
probe. Offline rescoring B with other probes does not create the counterfactual
trajectory those probes would have induced. Consequently, current B results do
not support a general claim about probe feedback.

The primary benchmark model is Qwen3.5-27B. Other participant models are future
transfer evaluations rather than required cells in the first benchmark.

## What the initial results imply

| Initial result | Design implication |
| --- | --- |
| Native Qwen3.5-4B already reaches 79.0% in A; Haiku reaches 88.5% | The present environment has limited judge-uplift headroom. Difficulty must be repaired on development data before another large live sweep. |
| Only the adapted layer-43 response probe generated B | Every live probe claim must name the exact probe. Comparative live evaluation requires a new `B[p]` trajectory for each probe `p`. |
| Response-level DYL/FIBS readouts reach about 0.93–0.94 pooled AUROC but are constant within a response | Report response detection separately from sentence/span localization. Within-response localization is not defined for one-score-per-response methods. |
| Local3 reaches 0.814 within-response AUROC, while an exploratory candidate reaches about 0.834 | Local monitoring is promising, but current selection and evaluation reuse exposed automated labels. It needs a family-disjoint development process and locked human-gold test. |
| B reduces the eligible contradiction fraction by 2.66 percentage points, concentrated in wrong-role turns, without a stable judge benefit | Measure factuality, debate behavior, deceptive success and judge accuracy separately. None is a substitute for another. |
| Many historical debates repeat arguments and some generations end with unfinished prose | The interaction protocol and completion contract need repair before using judge uplift as the primary benchmark endpoint. |
| All 116 current source families have experimental exposure | Treat the current corpus as open development data. A confirmatory probe result requires new locked source families. |

## Benchmark matrix

For each probe `p`, define the original grid as follows:

| Participant trajectory | Judge sees arguments only | Judge also sees probe `p` |
| --- | --- | --- |
| No participant feedback | **A** | **C[p]**, reusing A text |
| Live participant feedback from probe `p` | **B[p]** | **D[p]**, reusing `B[p]` text |

`C[p] - A` measures judge assistance on identical text. `D[p] - B[p]`
measures judge assistance after probe-conditioned behavior. `B[p] - A` measures
the combined behavioral effect of announcing and displaying that probe's
feedback. Comparing `B[p] - A` across probes is the live in-context benchmark.

The benchmark has three preregistered tracks.

### Track P: passive probe benchmark

Generate transcripts once and score every probe on the exact same saved model
contexts. This is the main probe-quality comparison and can include the full
probe and layer inventory without regenerating debates.

Track P has distinct tasks:

- response-level deceptive-action detection;
- local sentence or atomic-claim detection;
- trajectory monitoring, including first-detection time and false alarms before
  the first deceptive act;
- intent controls that distinguish deception from assigned role, false text,
  quotation, negation and mistaken belief.

After live `B[p]` trajectories exist, score every one of them with every probe
`q`, producing a cross-probe matrix rather than evaluating only the probe that
generated the feedback. Report each probe on A and on every `B[p]`. This tests
whether a detector remains useful when behavior adapts to itself or to a
different detector.

### Track L: live probe-in-the-loop benchmark

Select a small number of probes using development data, freeze them, and create
new `B[p]` trajectories. Every selected probe uses a common scalar display in
the primary probe comparison so that probe identity is not confounded with
presentation. A separate display experiment then holds the winning probe fixed.

### Track J: judge-assistance benchmark

Construct `C[p]` and `D[p]` post hoc from the exact A and `B[p]` transcripts.
The primary condition displays calibrated probe evidence without an answer
recommendation. Recommendation-only and full-bundle presentations are separate
display ablations.

## Data program

### D0: existing open development corpus

Use all current 116 QualitySyn source families for engineering, error analysis,
probe fitting and environment hillclimbing. Preserve the original text and five
documented repairs. Never describe results on this corpus as an untouched test.
Use family-disjoint internal train and development folds even within D0; an
exposed development corpus still should not evaluate a head on its training
families.

Complete the remaining source audit so every question is marked valid,
ambiguous, contradicted or unusable by two independent readers with adjudication.
Repair or retire ambiguous and source-contradicted questions in a new dataset
version; do not mutate historical experiment inputs.

### D1: new environment-development families

Create at least 40 new source families for tuning the interaction and judging
protocol. Keep all variants of a story in one family. These families may be
used for prompt selection, judge selection, probe layer selection and display
selection, so they are development data permanently. Generate D1 and T1 through
the same frozen item-construction pipeline, assign their family split before
hillclimbing, and seal T1 while D1 is used.

### T1: locked test families

Create a separate locked set of at least 80 new source families, with one
primary binary question per family. Set the final size using a preregistered
power simulation based on development fix/harm discordance and three trajectory
seeds. No probe, prompt, threshold, layer, display or judge may be selected from
T1 results.

### Item contract

Each new family contains:

- a novel source story and an unambiguous binary question;
- a structured atomic fact table with source evidence spans;
- a source-supported account;
- a separately authored, internally coherent counterfactual account supporting
  the wrong answer;
- matched true and false quotations or evidence units;
- an answer-key rationale and documented ambiguity checks;
- provenance, generator and reviewer identities, hashes and family grouping.

The primary benchmark uses the specified counterfactual account so that the
false advocate has a stable position across probes and seeds. A secondary
ecological track lets the advocate invent its own account, preserving the
original Probity behavior. Do not pool the two tracks.

### Difficulty and validity gates

Tune these gates only on D0/D1, then freeze them before T1:

- two independent source reviews agree on the answer, or a human adjudicator
  resolves the disagreement;
- a source-informed judge answers at least 90% correctly;
- question-only judges remain close to 50% on the binary task;
- without seeing the source, independent raters cannot reliably identify the
  counterfactual account from style or detail alone;
- the selected weak final judge scores approximately 55–70% on unassisted A
  debates in development, leaving room for improvement and harm;
- neither answer option has systematic length, lexical, quotation or position
  cues;
- invalid, truncated and role-switching episodes are retained in accounting and
  regenerated only under a frozen objective retry rule.

Qualification models and thresholds must be selected before T1. Publish all
candidate counts and exclusion reasons so difficulty filtering is not mistaken
for representative sampling. Do not manufacture headroom only by choosing an
incapable judge: the account-plausibility and source-informed gates must pass,
and a stronger secondary judge must evaluate the same frozen transcripts.

## Debate protocol repair

Use three functionally distinct rounds rather than three unconstrained advocacy
turns:

1. opening account with two evidence claims;
2. response to a specific opponent claim plus one new claim;
3. closing synthesis that resolves one explicit contradiction.

Require claim identifiers and source-style evidence references, cap each turn,
and detect exact duplicate turns after generation. Regenerate them only under a
frozen retry rule. Use a schema that does not cause unfinished prose; the
current strict-schema failure mode must pass a development stress test before
launch.

The primary implementation should generate and score with the same pinned
Qwen3.5-27B checkpoint, tokenizer, chat template, precision and model revision.
The current hosted-generation/local-scoring arrangement remains useful as an
ecological robustness condition, but it cannot establish exact activation
provenance.

All live branches share the same opening round within family and seed. Feedback
first appears after that round, and every `B[p]` branch regenerates all later
turns. Use three seeds per family. Save private plans, all submitted messages,
sampling parameters, raw responses, probe token/span coordinates and exact
feedback packets.

## Ground truth

Role assignment is not the main deception label. A false advocate can state a
true sentence, and a true advocate can quote or negate a false sentence.

Maintain separate labels for:

- assigned role and private instruction;
- private deceptive intent or plan;
- whether a public turn advances the false account;
- source-supported, contradicted, unsupported and ambiguous atomic claims;
- deliberate false assertion, quotation, negation, correction and uncertainty;
- exact character and token spans for local evaluation.

Development labels may use blinded model readers with disagreement review. T1
requires two human annotators and adjudication for every evaluable atomic claim.
Probe values, condition, assigned role and judge outcome remain hidden from
annotators.

Add matched intent controls:

- the same false proposition asserted deceptively, asserted under a supplied
  mistaken belief, quoted, negated and corrected;
- truthful and false turns from the same assigned advocate;
- matched private instructions whose public surface text is held constant where
  feasible.

These controls test whether a probe detects deception rather than role,
incorrect content or the literal presence of a false proposition.

## Probe suite

Benchmark a small, named set of probe families. Record exact checkpoint,
training corpus, label semantics, pooling, layer, sign and calibration data.

| Probe family | Purpose | Test status |
| --- | --- | --- |
| Text-only monitor and simple lexical baselines | Measure what is available without activations | Passive and judge baseline |
| Length, position, random direction and role/instruction probes | Detect artifacts and role leakage | Passive controls |
| Apollo roleplaying linear-probe recipe, trained on Qwen3.5-27B activations | Cross-paper strategic-deception baseline; weights from another model architecture cannot be transferred | External-data transfer, clearly labeled reproduced |
| DYL follow-up and Yes/No contrast heads trained on Apollo roleplaying data | Preserve the requested DYL comparison with an exact scoring contract | External-data transfer |
| FIBS LinearEnsemble plus at least one multi-token architecture, trained on Qwen3.5-27B activations | Scaled multi-layer/multi-token cross-environment baseline | External-data transfer |
| Frozen Probity adapted layer-43 response head | Legacy live baseline used in current B | Passive and live |
| Frozen Local3 span head | Existing local baseline | Passive and live candidate |
| New in-domain span/trajectory head | Measure attainable performance after legitimate benchmark hillclimbing | Train only on D0/D1; locked T1 test |

Do not describe 57 readouts as 57 independent probe families. Derived signs,
pooling views, controls and layer views stay grouped under their parent probe.

### Layers and pooling

For Qwen3.5-27B, retain the preregistered depth grid corresponding to layers 19,
27, 35, 43, 51 and 58. Fit or score compatible weights at their own layer; never
apply a layer-43 vector to a different layer. Select one layer per probe family
on D1, freeze it for T1, and publish the full development layer curve.

Compare these response representations where supported:

- response-token mean;
- last response token;
- DYL follow-up final-token score;
- top-k or maximum token score;
- sentence/span mean and maximum;
- released multi-layer/token aggregation.

For localization, compare sentence-independent, prefix-conditioned and local
window scoring on development data. Freeze one context convention before T1.
Broadcasting one response score across sentences is reported as localization
not applicable, rather than AUROC 0.5.

### Training and calibration

Report external zero-shot probes separately from in-domain probes. In-domain
probe training, regularization, sign, layer and early stopping use D0/D1 only.
Group all examples from a source family in one split. Preserve a fixed benign
calibration set to set thresholds, including ordinary truthful chat and debate
turns from both roles.

### Cross-paper FIBS contract

FIBS probe parameters are model-specific, so reproduce the released training
recipe on Qwen3.5-27B activations rather than importing weights trained on a
different residual-stream dimension. The comparable read uses six layers at
approximately 30%, 42%, 55%, 67%, 80% and 90% model depth and the final user and
assistant turn. Include:

- the model self-report logit as a non-probe baseline;
- the six-layer LinearEnsemble;
- at least one released multi-token method, preferably EFC and, if compute
  permits, the axial probe;
- Yes/No prefill contrast scoring for a frozen follow-up question.

Evaluate a small preregistered follow-up set on D1, including a generic
truthfulness question and a debate-specific knowing-falsehood question, then
freeze one before T1. Report the sensitivity because follow-up wording changes
the read. Since full FIBS contains on-policy Werewolf and Among Us families,
report both full-FIBS training and a no-social-deduction-family ablation if the
claim is cross-environment transfer from non-game deception data.

## Live conditions

The primary live probe comparison uses the same minimal display for every
probe: calibrated percentile for the participant's previous turn, with a short
fixed explanation and no answer recommendation. For local probes, reduce the
local output to a preregistered scalar, such as the maximum calibrated sentence
score, for this probe-isolation comparison.

After choosing a probe on D1, run the following display ablation while holding
the probe fixed:

- no feedback;
- token-count-matched neutral feedback;
- scalar previous-turn score;
- sentence heatmap;
- top one or top three peak phrases;
- scalar plus speaker history;
- recommendation only;
- full bundle.

Use a real derangement for shuffled controls: no packet may retain its original
turn, family, numerical value or recommendation. Validate this mechanically
before generation. The earlier shuffled condition is not sufficient because it
often preserved useful values and recommendations.

The minimum probes carried into live T1 should be frozen before opening T1:

1. the legacy adapted response probe;
2. the strongest external DYL/FIBS probe on D1;
3. the strongest local/span probe on D1.

If the released FIBS architecture and DYL probe are scientifically distinct and
both clear the passive gate, retain both rather than collapsing them into one.

## Metrics and inference

### Passive detection

- response AUROC and area under the precision-recall curve;
- TPR at 1% and 5% FPR using frozen benign calibration;
- calibration error and score distributions by role;
- within-response sentence/claim AUROC and average precision;
- top-k span recall and first-deceptive-span rank;
- trajectory false alarms and time to first correct detection;
- performance on the matched intent controls;
- degradation from D1 to locked T1.

### Live behavior

- contradiction and unsupported-claim rates with eligibility denominators;
- false-account adherence and deliberate-false-assertion rate;
- corrections, concessions and successful challenges;
- repetition, invalid completion and role-switch rates;
- probe score trajectories and whether later false spans decrease;
- wrong-advocate success and final judge accuracy.

### Judge assistance

- `C[p] - A` and `D[p] - B[p]` on identical text;
- `B[p] - A` for behavioral feedback;
- `D[p] - A` for the combined system;
- fixes, harms and probe-by-feedback interaction;
- score-only versus recommendation-only contrasts.

The source family is the resampling unit. Keep every seed and branch from a
family together in bootstrap or hierarchical-model resampling. Report both
family-weighted and question-weighted estimates, preregister primary contrasts,
and adjust or clearly label exploratory multiple comparisons. Generation-seed
variation must appear in uncertainty estimates.

## Stage gates

1. **Environment gate:** source validity passes; question-only accuracy is near
   chance; A accuracy is in the development headroom band; repetition and
   invalid completion rates are below frozen limits.
2. **Passive-probe gate:** at least one external probe beats controls on
   deceptive-action detection, and at least one local probe beats controls on
   within-response localization. Role-matched controls must not collapse the
   result.
3. **Live-development gate:** run the common-display `B[p]` screen on 24 D1
   families with three seeds. Freeze probes and the display before T1.
4. **Locked-test gate:** run A, selected `B[p]`, and post-hoc `C[p]/D[p]` on T1.
   Do not expand conditions after inspecting T1.
5. **Claim gate:** reserve general claims about in-context probes for effects
   replicated across at least two scientifically distinct probes or explicitly
   state that the result applies only to the named probe.

## Required deliverables

- versioned benchmark dataset and family/exposure ledger;
- dataset card documenting generation, repairs, exclusions and contamination
  checks;
- exact probe registry with weights, hashes, layers, pooling and training data;
- passive and live manifests fixed before T1;
- raw-context browser showing every model-visible message and feedback packet;
- human annotation export and adjudication log;
- response, localization, trajectory, behavior and judge-assistance plots;
- per-family and per-seed machine-readable results;
- cost and rental ledger with no raw activation tensors retained on the VPS.

## Recommended execution order

1. Repair and version D0; build the item validator and intent-control schema.
2. Generate D1 and T1 through one frozen pipeline, commit the family split and
   seal T1.
3. Hillclimb the debate protocol on D0/D1 until it passes the environment gate.
4. Integrate exact released Apollo/FIBS artifacts and verify every probe against
   its reference implementation.
5. Run the complete passive suite on D0/D1 and train the in-domain local head.
6. Run the 24-family, three-seed live probe screen with the common display.
7. Freeze the selected probes, layer, thresholds, display and judge.
8. Open T1 only for the preregistered passive and live benchmark.
9. Annotate T1 with human adjudication, run `C[p]/D[p]`, and publish all frozen
   outcomes including failures.

This ordering uses the present corpus for legitimate hillclimbing while
preserving a new test set for the final benchmark claim.

## Related probe baselines

- Goldowsky-Dill et al., [Detecting Strategic Deception Using Linear
  Probes](https://arxiv.org/abs/2502.03407).
- Hollinsworth et al., [Caught in the Act: Probes Effectively Detect Sabotage
  and Catch Unverbalized Deception](https://arxiv.org/abs/2610.12445).
