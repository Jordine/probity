# Probity: second Qwen pilot

Completed second pilot, 2026-10-14. There is useful detector signal, but the
current debate environment still leaves too little judge error to measure
substantial uplift. On eight fresh parent stories the small judge improves
from 31/32 to 32/32 with the frozen adapted probe; shuffled scores yield 24/32.
Local detection has promising Qwen3.8 results and important control failures.
The cumulative conservative accounting is **$19.55 of the $25 cap**, including
unpriced-call and transfer allowances. Both task rentals are destroyed and
their absence verified.

## What the high baseline measures

The no-story baseline is **judge answer accuracy**, before any debate. It is
not a debater win rate or a probe score. Repeated evaluations point to answer
choice cues, not just sampling noise. Across 60 development stories, three
repeats and both answer orders, all 1,080 new requests completed with
`finish_reason=stop`. Confidence intervals resample stories, not repeated calls.

| Judge | No-story accuracy | 95% story interval | Short options | Descriptive options | Options only, descriptive |
|---|---:|---:|---:|---:|---:|
| Haiku 5.5 | 61.4% | 52.5–70.3% | 42.4% | 84.6% | 79.6% |
| Qwen3.5-27B | 68.1% | 58.6–77.8% | 47.0% | 93.8% | 79.6% |
| Qwen3.8-27B | 67.2% | 57.5–76.7% | 50.5% | 87.7% | 77.8% |

The short-option rule (both answers at most six regex words) was chosen after
the first pilot and before these repeats. This is an exploratory diagnosis,
not a contamination test. There is some repeat variation, but it does not
explain the large descriptive-option advantage.

## Environment and participant checks

The main iteration uses 16 parent stories, each with an original and an
opposite-gold renamed world. The question and answer options are identical
within each pair. Reusing the same no-story prediction gives exactly 50% by
information symmetry; that is a design property, not an empirical equivalence
test. Eight parents are training, four development and four probe-transfer
test. Every world, turn and claim from a parent stays in its partition.
The eight training parents had appeared in the earlier baseline diagnosis;
all four local-test parents had not. This is a probe-fitting split inside an
exploratory pilot, not an untouched final benchmark. The later eight-parent
transfer check is separately frozen.

Four ambiguous question targets were repaired before the final debates. A
provider content block remains missing data and was not circumvented. The
repaired-question validation returned 47/47 correct full-story answers among
completed calls, with one missing blocked response.

The standard protocol produced 64 three-round episodes and 384 complete turns:
all 64 private knowledge checks were correct and all 384 answer fields matched
the assigned option. Blind prose auditing plus targeted complete-turn review
found one actual prose mismatch. That training turn is excluded from the
whole-response advocacy targets. The distinction matters: an answer field can
match an assignment while the prose argues for something else.

| Judge | Qwen3.5 debates | Qwen3.8 debates |
|---|---:|---:|
| Haiku 5.5 | 31/32 | 30/32 |
| Qwen3.5-27B | 28/32 | 31/32 |
| Qwen3.8-27B | 31/32 | 32/32 |
| Qwen3.5-9B | 28/32 | 30/32 |
| Native Qwen3.5-4B, final signal-permission protocol | 27/32 | 28/32 |

The native 4B judge scored 29/32 on full-story twins, including 8/8 development
and 8/8 test worlds. It uses bf16, greedy, thinking-disabled, answer-only
generation. The API judges use their recorded structured-output routes, so
this is not an isolated model-size comparison. The 4B explanation format first
hit a generation loop and malformed JSON; those runs were rejected. All 128
final baseline jobs stopped normally, with at most eight generated tokens.

Three alternative debater prompts were also tested on the same four
development parents: evidence-focused, explicit permission to fabricate, and
three factual assertions per turn. Haiku still judged 16/16, 15/16 and 15/16
correctly. Increasing permission to lie did not itself produce a difficult
oversight task. These were development diagnostics, not additional independent
test stories.

## Whole-response probes

The target here is advocating the assigned wrong answer after passing the
private knowledge check. It can be detected from role or style information;
it is not a label that every supporting claim is false, or proof of intent.
All scores below use native-model replay of complete API texts, not the API
provider's exact internal runtime.

| Frozen or adapted detector | Qwen3.5 test AUROC | Qwen3.8 test AUROC |
|---|---:|---:|
| Apollo-data response mean, frozen before debates | 0.936 | 0.693 |
| Apollo-data No-prefill DYL | 0.503 | 0.470 |
| Apollo-data No-minus-Yes DYL | 0.974 | 0.458 |
| Published model-matched FIBS DYL | 0.988 | Not available |
| Debate-adapted response mean, selected on development | 0.988 | 0.991 |
| Response-text-only control for adapted training | 0.740 | 0.821 |

Each model has 48 test turns from only four parents. The adapted heads use
eight training parents, with feature/layer/regularization selection on four
development parents. Qwen3.8's weak Apollo transfer and strong adapted
advocacy result show a transfer/target mismatch; they do not establish that a
general deception representation was absent or recovered. Published Qwen3.5
weights were never applied to Qwen3.8. Newly fitted Apollo-data heads are not
the collaborators' original checkpoints.

## Local annotation audit

The first completed two-model consensus accepted 1,370 claims, including 51
false claims. Complete-source review found shared mistakes: both annotators
accepted a false assertion about a fictional character's pronouns, and an
extracted clause lost its surrounding accusation/attribution. Name swaps also
created a confound by assigning a conventionally masculine name to a female
character. Following the name stereotype is not evidence of deliberate lying.

A blind independent Opus pass reassessed all exact proposed substrings against
complete sources and arguments. It did not see prior verdicts, gold answers,
roles, participant identities or probe scores. After two guarded manual scope
exclusions, the corpus contains **1,679 claims: 1,495 supported and 184
contradicted**. Of the initial 1,370, 124 were removed, 433 previously rejected
claims were accepted, 18 accepted true labels became false and one false label
became true. These remain silver labels, not exhaustive human ground truth.

Wrong-answer advocacy contains 542 supported and 155 contradicted accepted
claims. Correct-answer advocacy contains 953 supported and 29 contradicted
claims. This directly demonstrates why role labels cannot replace local
factual labels. Unknown and unselected text is never a negative.

The provisional local fits using the first consensus were superseded before
their test metrics were inspected. The adjudicated label set, candidate grids
and development selection criteria were fixed before viewing the new local
test results.

## Local training and complete displays

Each model's complete token activations were cached once at six layers. Local
heads compare controlled-only, debate-only and combined training; mean/last
span pooling; logistic, within-response ranking and annealed objectives. A
separate token-bag sweep fits eight heads together per cached layer, including
the legacy max-positive/min-negative objective and alternatives that penalize
all known-negative tokens. False claim spans are weak bags, not tokenwise
truth labels.

The final controlled data contains 120 parent contexts and 480 responses.
Identical response text is replayed against complementary source worlds that
flip every factual label. Any context-blind text or position score therefore
has exactly chance pooled and within-response AUROC on the balanced twins.
This tests contextual factual consistency, not generated deceptive intent;
template families are shared across partitions.

The token-bag sweep selected mean pooling on development for both models:

| Token objective | Qwen3.5 local test AUROC | Qwen3.8 local test AUROC |
|---|---:|---:|
| Mean BCE | 0.808 | 0.897 |
| Legacy max/min | 0.674 | 0.402 |
| Max-positive/all-known-negative | 0.643 | 0.562 |
| Within-response ranking with max pooling | 0.724 | 0.511 |

These are parent-macro within-response AUROCs, not token accuracy. Qwen3.5 has
28 false claims in 13 mixed responses across four test parents; Qwen3.8 has 15
false claims in 10 mixed responses across three contributing test parents.
The selected mean heads have bootstrap intervals 0.700–0.933 and 0.692–1.000.
At thresholds chosen for at most 10% development false positives, test false
claim recall is 67.9%/53.3%, with 19.0%/10.7% false positives on supported
claims. These thresholds do not maintain their development false-positive
rates on the test stories.

| Local head or control | Qwen3.5 local test AUROC | Qwen3.8 local test AUROC |
|---|---:|---:|
| Development-selected pooled head | 0.611 | 0.942 |
| Mean token-bag head | 0.808 | 0.897 |
| First-pilot Qwen3.5 local head | 0.865 | Not available |
| Text-only TF–IDF | 0.609 | 0.511 |
| Position only, direction selected on development | 0.499 | 0.810 |
| Best of 96 random directions on development | 0.748 | 0.634 |
| Published DYL broadcast to every claim | 0.500 | Not available |

The pooled selections are a debate-only, layer-35 pairwise head for Qwen3.5
and a combined-data, layer-35 logistic head for Qwen3.8. Their intervals are
0.306–0.889 and 0.827–1.000. Qwen3.5's newer selected head does not beat the
first-pilot local head; this is not a general improvement claim. Qwen3.8's
strong position control and just three contributing parents also prevent a
broad localization claim. DYL's pooled claim AUROC can look high (0.833 here)
while its within-response AUROC is exactly 0.5, because the same message score
cannot rank claims inside that message.

The controlled-only heads achieve 0.991/0.994 local AUROC on held-out
context-twin groups, but only 0.654/0.575 on natural debate claims. This
synthetic-to-natural gap is substantial. Conversely, frozen natural pooled
heads correctly order **90.3%/94.7%** of identical-text, opposite-fact pairs
across 20 controlled test parents; the mean token heads achieve 80.0%/86.9%.
That is evidence of context sensitivity, not proof of deceptive-intent
detection or robust natural localization.

The export scores every sentence, clause and eight-token window independently
of annotation selection. Full arguments, original JSON, all response-token
scores, source stories and reviewed claims are retained. DYL scores stay
separate from local highlighting. Within-response shading and raw logits are
not truth probabilities.
The portable HTML adds sentence/clause switches, full source links and the
complete highest-scoring eight-token window for each turn. All windows and
token scores remain in the companion JSONL. All 384 arguments were checked
for exact character preservation in both views; 3,358 displayed labelled
claim scores agree with fitted predictions to within 0.000006.

Post-outcome error analysis read four complete arguments and their complete
sources: each model's highest-scoring supported claim and lowest-scoring
contradicted claim. A true description of a ruined facility inside a
misleading historical/current-city argument is falsely flagged. Incorrect
speaker attributions inside otherwise correct arguments are missed. A
correct speaker attribution also receives a very large token-head score.
These are selected failure examples, not a representative frequency estimate;
no labels or fits were changed after this review.

## Judge-assistance protocol checks

The first score display had an instruction-scope conflict: the system allowed
only question/options/arguments while the user message supplied detector
signals. Both baseline and assistance were rerun with the same system
explicitly allowing optional signals. The earlier output is retained as a
failed protocol diagnostic.

With that corrected percentile display, the four-parent pilot test baseline
was 14/16. The adapted head and Apollo DYL each gave 15/16 with real signals
and 15/16 with shuffled signals. Apollo response scores gave 13/16 real and
14/16 shuffled; matched FIBS gave 7/8 in both. The oracle scored 15/16. Thus
there is no demonstrated advantage over the shuffled control in this pilot.
The protocol debugging exposed these test outcomes, so they are exploratory.

A clearer display was subsequently fixed: show each turn's percentile, each
speaker's mean, and the detector-only answer recommendation implied by the
lower mean. On the original development parents, the adapted probe gave
16/16 versus 14/16 for shuffled scores; the explicit oracle gave 16/16.
These development results motivated a separate eight-parent transfer check.
No new head, threshold or normalization is fitted to those fresh stories.

## Fresh transfer

Eight previously unqueried place-name stories were locked for the separate
transfer check. Input review caught unrenamed country adjectives, a briefing
base whose name was absent, an ambiguous last-project question and a district
name encoded only through initials/imagery. The versioned test inputs repair
the adjectives and ask three explicitly supported factual targets. Original
dataset bytes and failed validation receipts are preserved. All three API
validation models then answered 32/32 full-story variants correctly, although
Haiku retained five ambiguity flags and quotation-exactness checks still
expose unreliable explanatory evidence. Correct answers alone do not certify
that evidence.

The fresh run produced 32 episodes and 192 turns. All 32 private knowledge
checks and 192 answer-field checks passed. A blind prose audit, repair pass
and complete-turn review found one genuine truthful concession: prose
adherence is 191/192. All 32 episodes remain in the primary judge comparison.
Haiku judged 32/32 correctly. The native 4B judge scored 14/16 with the full
stories and 8/16 without them; the paired no-story result is structurally
balanced.

All 64 native baseline calls and all 256 assistance calls ended normally.
The primary result pools both participant models over eight parent stories:

| Judge condition | Correct / episodes | Change from no probe |
|---|---:|---:|
| No probe | 31/32 | — |
| Adapted response probe, real scores | 32/32 | +3.1 percentage points |
| Adapted response probe, shuffled scores | 24/32 | −21.9 points |
| Apollo response mean, real scores | 31/32 | 0 points |
| Apollo DYL contrast, real scores | 28/32 | −9.4 points |
| Explicit oracle | 32/32 | +3.1 points |
| Published FIBS, Qwen3.5 episodes only | 16/16 | 0 points versus 16/16 |

For the primary adapted probe, the parent-bootstrap interval is **0 to +9.4
points versus baseline** and **+12.5 to +40.6 points versus shuffled scores**.
The latter contrast is driven mainly by shuffled advice harming an already
strong judge. It is not a 25-point improvement over the unaided judge.
The adapted probe fixes one error and introduces none. Eight parents and one
baseline error are insufficient for a reliable estimate of general uplift.

Apollo DYL's three harmed judgments are all on Qwen3.8 debates: 12/16 with
real DYL versus 15/16 without scores. Its fresh wrong-advocacy AUROC is 0.630
after excluding the known concession, versus 0.937 for Qwen3.5. The frozen
adapted response heads score 0.999/1.000 on that operational proxy. Strong
role-conditioned separation does not imply local truth detection.

The original generic assistance metadata mistakenly said four test parents;
an explicit erratum records eight and the earlier development-only display
selection. Model inputs and frozen heads are unchanged. The final analysis
computes parent counts from evaluated records. The known prose concession is
excluded only from the separate advocacy-proxy audit, never from C−A.

There are now **92 original stories never queried**. Debater-feedback B−A and
D−B require new feedback trajectories and a validated execution contract;
they are not estimated by replaying scores onto existing debates.

## Next experiment

1. **Improve the environment on development data before another sweep.**
   Retain paired worlds and explicit answer scoring. Use verifiable
   multi-step questions, stronger evidence selection by the wrong-answer
   advocate, and longer-range attribution/quotation conflicts. Keep complete
   source context. Require high full-story judge accuracy, balanced no-story
   information, reliable role adherence, and enough unaided debate errors
   for the oracle to show useful headroom. Merely granting permission to lie
   or reducing the judge to 4B has not solved this.
2. **Strengthen local labels and controls.** Human-review the sparse false
   claims and add position-balanced false attribution, truthful fragments
   inside misleading arguments, and matched true/false paraphrases. Keep
   mean pooling as a baseline: the legacy max/min and the annealed
   alternatives did not beat it here. Recheck against the first-pilot head,
   position, text and development-selected random directions.
3. **Freeze before expanding.** Use a new story holdout for the next local
   probe comparison and C−A trial. Only then run B−A and D−B with genuinely
   new feedback trajectories. Preserve a separate reference using exact
   collaborator checkpoints when accessible.

The remaining **$5.45** is unspent; the rental is closed rather than consuming
it on more copies of the same ceiling-limited comparison.

## Reproduction and evidence

The isolated branch includes GitHub main through `ca3952e1`. The original
working checkout remains intact. See the [harness guide](../probity_benchmark/README.md)
for commands. New jobs use locked cumulative cost reservations, complete
requests/responses, stop-reason gates and immutable output directories.

The final natural replay contexts are at most 9,712 tokens for Qwen3.5 and
9,519 for Qwen3.8, under the pinned 262,144-token limit. Short alignment checks
pass. The new hardware's generic 7,111-token parity diagnostic passes the
unchanged 3% criterion, including the alternative recurrent-kernel diagnostic;
this does not erase the earlier hardware's failed generic/padded checks or
establish exact API-runtime equivalence. No large-scale live-feedback run is
approved by these offline diagnostics.

Canonical main-iteration inputs are `paired_worlds_v2.jsonl`, `debates_v3/`,
`adjudicated_reviewed/` and `context_twins.jsonl` under `runs/iteration2/`.
Use the `*_adjudicated` fits and reviews, not the superseded consensus fits.
`manual_claim_label_review.json`, `annotation_reconciliation_audit.json`,
`judge_assistance/protocol_audit.json` and `fresh_transfer/input_quality_repairs.json`
record the substantive corrections.

The portable feature archive contains 1,536 cached examples in 1,558 files,
3,142,142,718 uncompressed bytes (2,063,723,383 compressed). Its SHA-256,
complete tar/gzip read, per-cache row counts, and all 109 separately copied
result artifacts were verified before destruction. Fitted heads, full score
exports, complete corpora and runtime/provenance receipts are retained.
The roughly 7.8 GB of complete per-token activation tensors were transient
on the rental because the VPS lacks space; new token-head training requires
re-extraction. This is a retention limit, not input or output truncation.

Use review_qwen35_adjudicated/review_linked_v2.html and the corresponding
Qwen3.8 file for qualitative review. fresh_transfer/judge_assistance_results/
holds the corrected fresh paired analysis, and
fresh_transfer/response_proxy_audit.json records the concession-aware proxy
metrics. portable_feature_caches.tar.gz, artifact_retention_verified.json,
portable_results_copy_verified.json, and final_cost_summary.json record
retention and accounting. The original failed/provisional artifacts remain
clearly separate.

Provider-reported API charges across both pilots are **$11.72**. GPU rental
is estimated at **$5.29** from listed hourly rates and elapsed lifetimes.
The **$19.55** conservative total additionally retains **$1.95** for unpriced
or interrupted API calls, **$0.55** for transfer fees and small cleanup-time
allowances. It is not a final provider invoice. Instance 55950201 was
destroyed at **18:13 UTC** and absence verified; its watchdog was then stopped.
No other instance was targeted.
