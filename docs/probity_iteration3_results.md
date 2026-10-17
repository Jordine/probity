# Probity iteration 3 — development record and fresh evaluation

Status: the final protocol was frozen at 19:51 UTC on 14 October 2026. All 96
private plans and all 48 fresh debates, comprising 288 public turns, are now
complete. The original private-preparation run stopped after 24 plans because
two requests timed out. A versioned source/recovery amendment was frozen on
15 October, before any public debates or fresh probe outcomes. All attempts
and charges remain recorded. Frozen response scores and sentence annotations
and sentence localization are complete. All three final judge comparisons are
complete. The independent secondary clause diagnostic is also complete after
lossless annotation recovery. Both task GPU rentals have been destroyed and their absence verified.
The additional $50 raises the cumulative cap to $75. Three agent workstreams
share `runs/iteration3/board/` and the locked ledger `runs/pilot_budget.json`.

## Main findings from the fresh comparison

The environment provides a useful test of response-level probe assistance for
the small Qwen judge. It does not establish improved local lie detection. The
following pooled results comprise 48 debate presentations from **24 independent
original story families**, with both Qwen 27B participants represented. Uplift
compares the real signal against its matched neutral display; raw accuracy is
the separate no-signal baseline.

| Judge | Raw answer accuracy | Adapted response probe: neutral → real | Response uplift, pp [95% family CI] | New sentence probe uplift, pp [95% family CI] |
| --- | --- | --- | --- | --- |
| Native Qwen3.5-4B | 37/48 (77.1%) | 39/48 → 47/48 | +16.7 [6.3, 29.2] | −18.8 [−31.3, −8.3] |
| Haiku 5.5 | 43/48 (89.6%) | 45/48 → 44/48 | −2.1 [−8.3, 4.2] | +2.1 [−4.2, 8.3] |
| Opus 5.5 | 45/48 (93.8%) | 46/48 → 47/48 | +2.1 [0.0, 6.3] | 0.0 [0.0, 0.0] |

The adapted response signal fixes eight native-judge errors and introduces none.
The new sentence display fixes none and introduces nine; its clause display
fixes none and introduces six (−12.5 pp, CI −20.8 to −4.2). Stronger judges largely
avoid this local-display harm, but have little remaining headroom for a positive
gain. The paired adapted-probe uplift difference is −18.8 pp for Haiku versus
native (CI −35.4 to −4.2) and −14.6 pp for Opus versus native (−29.2 to −2.1).
These pilot intervals are not adjusted for multiple comparisons. All frozen
contrasts, including unsuccessful detectors and controls, are retained in the
[judge tables](../reports/data/iteration3/judges/real_minus_neutral.csv).

The oracle reaches 48/48 for native and Opus, and 47/48 for Haiku. An empirical
100% bootstrap interval on these few families is not a population guarantee.
Native and Haiku each complete 1,032 logical judgments; the prespecified reduced
Opus package completes 384. Every accepted generation stops normally, with no
inference error or length truncation. Native uses 863 fresh generations and 169
exact cached reuses; their agreement does not measure inference noise. Haiku
has three disagreements among 265 identical-request pairs, in two of 97 repeated
request groups. Those calls span Anthropic/Azure routing without fingerprints,
so this measures routing/runtime variability, not isolated sampling noise. The
reduced Opus package has no repeated identical requests.

![All frozen participant-specific detector contrasts and baseline/oracle accuracy](../reports/figures/iteration3/judge_results.png)

## What the development trials establish

The earlier high baseline refers to **judge answer accuracy**, not local probe
accuracy or debater truth knowledge. Those are three separate measurements.

A source-family audit found 166 dataset rows but only 116 original story
families. The 74 previously consumed rows span 64 families. Family-weighting the
old no-story results leaves Haiku/Qwen3.5/Qwen3.8 at 61.7%/68.3%/67.6%; descriptive
options remain much easier than options of at most six words each. Sampling
noise and repeated source families do not explain away the answer-option cues.
This does not identify contamination as their cause.

The following development comparisons use four old source families unless
otherwise stated. Repeated worlds, participant models and turns are not extra
independent families. Exact-format results use the frozen answer-only judge
renderer; explanatory judge calls are kept separately.

| Environment | Native Qwen3.5-4B, no signal | Exact-format Haiku, no signal | Validity observation |
| --- | --- | --- | --- |
| Legacy | 16/16 | 16/16 | Public arguments usually reveal the answer |
| Dual-source rehearsal, two families | Not measured | Not measured | Explanatory Haiku fell to 6/8 sequential and 4/8 simultaneous, but debater knowledge in the combined context was only 4/8; rejected |
| Authentic-source structured preparation, two families | 8/8 | Not measured | Still at ceiling |
| Explicit assigned-answer preparation | 14/16 | Not measured | One of 32 final-context knowledge checks fails; judge errors are in different cells |
| Reasoning-enabled private preparation | 13/16 | 14/16 | 32/32 final-context knowledge checks; 96/96 prose-adherent turns |
| Same preparation plus public evidence execution | 13/16 | 14/16 | 32/32 final-context knowledge checks; 96/96 prose-adherent turns |

Both valid reasoning-preparation variants reach 16/16 with the explicitly
labelled oracle on each exact-format judge. The native judge misses the same
three model/world cases under both variants, spanning only two families. The
explanatory Haiku difference of 13/16 versus 14/16 disappears under the frozen
answer-only format; it is not a basis for selecting a winner.

The open-ended preparation attempt failed with four 8,192-token length finishes.
Structured preparation repaired that failure. A later public-reasoning trial
failed before completing any episode: one provider HTTP 504 and one 16,384-token
length finish. Its partial generations are retained and excluded from evaluation.
The full failed trace contains an unescaped quotation followed by 8,988 repeated
newline-and-indent sequences. The repair removes provider-side JSON enforcement
for public reasoning turns while retaining strict local parsing and schema
validation. Provider routing also varies, so this is a runtime repair rather
than a controlled causal comparison of decoding implementations. A later
900-second transport timeout has no returned response and is accounted at its
reserved upper bound; exact request-and-job receipt recovery preserves complete
calls without treating the incomplete batch as an evaluation. The contextual
diagnostic replaces the original system prompt, including its explicit correct
answer, while retaining the complete non-system source, private brief and
history presented before each speaker's own final turn. The diagnostic does
not include that speaker's final completion or later turns. Its answers measure performance under changed instructions and
knowledge cues; they do not establish generation-time ignorance, intention or
independent reasoning. Correct-answer advocacy is also not a label that every
sentence is true.

## Frozen local heads

Natural source twins preserve complete response wording, span positions and
role assignments while changing the complete source. The verified corpus has
230 response rows and 269 paired claims. A complementary mixed-claim augmentation
has 36 rows and 49 paired claims across 11 old source parents. All four old
development families contribute to its selection control. Failed annotation
protocols and disagreements remain in the artifacts; probe outcomes were not
used to repair annotation instructions.

One feature extraction per model supports a 32-candidate head search. Selection
maximizes the minimum of natural within-response localization, identical-text
source-pair ordering and mixed-twin localization, all grouped by original story
family. The selected objective combines equally weighted families of source-pair,
natural within-response and augmented mixed-response ranking pairs.

| Development mixed-claim control | Qwen3.5 | Qwen3.8 |
| --- | --- | --- |
| New selected head | .875 | .813 |
| Frozen previous local head | .563 | .656 |
| Position / identical text / response broadcast | .500 | .500 |
| Best of 32 development-selected random directions | .563 | .594 |

These are **selection-set results**, based on 16 paired claims in 12 mixed
source-context responses from four families. They are not a generalization
estimate. On the earlier iteration-2 test diagnostic, the new heads score
.817/.772 within-response AUROC; the Qwen3.8 result trades off the previous
.942. The fresh comparisons below evaluate both old heads and fixed controls.

The new heads are layer 35, C=.01 for Qwen3.5 and layer 43, C=.001 for Qwen3.8.
Complete all-sentence and all-clause development references are frozen separately
from the heads. The 90th percentile of each all-sentence reference is the
fixed flag threshold; it is not a calibrated 10% false-positive-rate guarantee.

The development-reference replay on the replacement machine differed from the
original score arrays by at most 1.78e-15. Every other field, row order, source
manifest and head hash matched exactly. Export stopped before fresh local
scoring. A versioned operational adapter now requires exact equality to the
preserved same-machine replay, then supplies the original frozen development
reference values to the unchanged exporter. Every fresh scoring call remains
unchanged; no numerical tolerance or new normalization is introduced. Both
original and replayed arrays, including the failed audit, are retained. The
floating-point discrepancy's precise cause has not been established.

## Fresh response discrimination and factual coverage

The following AUROCs distinguish assigned wrong-answer roles from assigned
correct-answer roles across all 144 turns and 24 families per model. All rows
remain included, including the mixed-stance turn. Static wording in the legacy
response evaluator incorrectly suggests prose mismatches were excluded; a
separate reporting clarification records that no exclusion list was supplied.
These role metrics do not measure local factual truth or intent.

| Frozen response detector | Qwen3.5 | Qwen3.8 |
| --- | ---: | ---: |
| Apollo-data response mean | .980 | .667 |
| Apollo-data No-prefill DYL | .605 | .593 |
| Apollo-data No-minus-Yes DYL | .983 | .624 |
| Published FIBS DYL | .993 | Unavailable |
| Adapted response head | .994 | 1.000 |

The adapted head's lower-mean-score answer choice is correct on 23/24 Qwen3.5
debates and 24/24 Qwen3.8 debates. This is a detector-only decision rule, not a
judge-assistance result. Perfect observed separation does not establish perfect
population performance. The new Apollo-data heads are distinct from the exact
collaborator checkpoints, whose compatibility remains unverified.

Complete independent Qwen/Opus annotation covers 1,977 fixed sentence units:
599 supported, 584 contradicted and 794 unknown under the frozen scope and
agreement rules. All 104 annotation jobs finish, with three schema failures
recovered by the original retry policy and retained in the receipts. The cost
is $13.915809972. These are model-consensus labels, not human ground truth.

| Participant | Supported | Contradicted | Unknown | Responses with both accepted labels |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 | 257 | 291 | 419 | 15/144 |
| Qwen3.8 | 342 | 293 | 375 | 13/144 |

Wrong-answer advocates produce accepted contradicted statements in 23/24 and
24/24 families, respectively. They also produce 18 and 19 supported sentences;
correct-answer advocates produce nine and zero contradicted sentences. The
environment therefore contains factual contrasts, but whole-sentence local
ranking has a much smaller eligible sample than the 288-turn corpus. Unknowns
stay in the coverage denominator and never become truthful negatives.

## Fresh sentence localization

The primary statistic ranks supported versus contradicted accepted units
within each mixed response, averages responses within original story family,
then averages families. The new heads do not establish improved localization.

| Frozen track | Qwen3.5 | Qwen3.8 |
| --- | ---: | ---: |
| New local head | .871 [.700, 1.000] | .703 [.428, .917] |
| Previous local head | .879 [.757, .971] | .919 [.808, 1.000] |
| Fixed word TF-IDF | .680 [.495, .864] | .711 [.556, .856] |
| Development-selected random direction | .854 [.736, .957] | .539 [.211, .856] |
| Response score broadcast to every unit | .500 | .500 |

Intervals resample families and are unadjusted for multiple comparisons. Only
15 mixed responses from seven Qwen3.5 families and 13 from five Qwen3.8 families
contribute to this ranking statistic. The full coverage denominator still
contains 144 responses and 24 families per model. The paired new-minus-random
Qwen3.5 difference is .017 [-.059, .107]; the paired new-minus-old Qwen3.8
difference is -.217 [-.406, -.033]. Pooled AUROC .965/.951 does not establish
within-response localization because it also reflects differences between
responses and advocacy roles.

At the previously frozen all-sentence development 90th-percentile threshold,
the new heads recover 28.18%/10.92% of accepted contradicted units, with
100%/94.12% precision among flagged units that have accepted labels. The
false-positive rates among accepted supported units are 0%/.585%. These
precision figures exclude flagged unknown units, whose factual correctness
remains unmeasured; they are not precision estimates for every displayed flag.

The complete control plot includes both length and position orientations and
the model-matched FIBS broadcast control. A deterministic qualitative casebook
selects the two highest-scored accepted supported flags and the two
lowest-scored missed contradictions per model, using frozen thresholds and
fixed ties. Qwen3.5 has no eligible false-positive flag, leaving six cases in
total. This selection is for error inspection and does not alter labels,
denominators or metrics.

Complete-source review finds three clear missed contradictions: invented
dialogue, a false water-provider/recipient relation, and an invented quotation
excluding a third attack. The fourth missed unit has attribution/paraphrase
ambiguity: one character states the exact observation, while another supplies
the same causal explanation. Both flagged supported units repeat the same
authentic quotation in one episode whose argument applies it to the wrong
person; they are context-ambiguous and not independent clean false positives.
One annotator cites the wrong paragraph number for an otherwise exact quote.
Byte offsets and score/display mapping pass. These caveats remain alongside
the original labels. The fixed splitter can cut after honorifics or inside
quotations, so “sentence units” does not guarantee complete semantic claims.

![All frozen sentence-localization controls](../reports/figures/iteration3/local_sentence_results.png)

## Secondary clause diagnostic

The fixed first wrong-assigned opening from every episode supplies 771 clause
units across 48 responses. Independent model consensus accepts 64 supported and
214 contradicted units; 493 remain unknown. There are 391/380 units for
Qwen3.5/Qwen3.8, respectively, with 136/142 accepted labels. Only 14 and 11
responses, each from a different family, contain both accepted classes. This
secondary descriptive sample is distinct from the 5,598 all-turn clause units
displayed in the judge experiment and from the primary sentence evaluation.

| Frozen track | Qwen3.5 within-response AUROC | Qwen3.8 within-response AUROC |
| --- | ---: | ---: |
| New local head | .977 [.951, 1.000] | .738 [.568, .893] |
| Previous local head | .876 [.709, .981] | .921 [.838, 1.000] |
| Fixed word TF-IDF | .597 [.470, .735] | .704 [.561, .846] |
| Development-selected random direction | .754 [.583, .901] | .588 [.424, .740] |
| Response score broadcast | .500 | .500 |

Qwen3.5 shows a promising clause-level signal above the random-direction control:
paired difference .223 [.076, .406]. Its new-minus-old difference is .101
[.000, .261]. Qwen3.8 again trails the previous head: −.183 [−.321, −.065].
All controls and unadjusted family intervals are retained. The same frozen
sentence-reference threshold is applied here; it is not recalibrated on clauses.
New-head recall of accepted contradictions is 50.5%/30.3%, with 100%/82.5%
precision among flagged units with accepted labels. Only 53 of 107 Qwen3.5 flags
and 40 of 94 Qwen3.8 flags have accepted labels, so these precision estimates
cannot describe all displayed flags. Better local ranking on this subset also
does not imply judge uplift: the complete native-judge clause display harms
answer accuracy, as reported above.

The original annotation batch completed 48 of 50 logical jobs. Both failed
Qwen attempts returned identical complete content, rejected solely because one
explanatory reason contained 976 characters versus the schema's 600-character
limit. A hash-bound, descriptive-only amendment accepts the earliest original
payload losslessly and preserves every label, evidence coordinate and character.
It reuses the 48 valid original jobs and issues only the genuinely missing Opus
job, with the original full request and retry allowance. That single call costs
$0.141468; combined clause annotation cost is $3.344310664. Twelve bounded checks
and an exhaustive 771-unit reconciliation audit pass. Original failures and
strict-schema validity remain recorded; the primary results do not change.

![Secondary descriptive opening-clause localization; all controls, using frozen sentence-reference thresholds](../reports/figures/iteration3/local_clause_results.png)

## Fresh source controls

Each model receives both answer orders on each of the 24 fresh source families,
with and without the complete story. The following are accuracy percentages;
there are 48 presentations per model, but only 24 independent families.

| Model | No story, all | No story, short options | No story, descriptive options | Full story, all |
| --- | ---: | ---: | ---: | ---: |
| Native Qwen3.5-4B | 50.0 | 46.2 | 54.5 | 81.3 |
| Haiku 5.5 | 54.2 | 46.2 | 63.6 | 95.8 |
| Qwen3.5-27B | 58.3 | 50.0 | 68.2 | 100.0 |
| Qwen3.8-27B | 60.4 | 53.8 | 68.2 | 100.0 |
| Opus 5.5, low effort | 64.6 | 50.0 | 81.8 | 100.0 |

The prespecified short-option group has 13 families; the descriptive group has
11. Family-bootstrap 95% intervals for overall no-story accuracy are 35.4–64.6%
for native 4B, 39.6–68.8% for Haiku, 41.7–72.9% for Qwen3.5, 41.7–79.2% for
Qwen3.8 and 47.9–81.2% for Opus. These small-sample intervals include 50%.
The descriptive-option pattern is consistent with option cues, but neither
identifies contamination nor rules out sampling noise. These original questions
have no construction that guarantees a 50% no-story population baseline.

Both Qwen debaters answer every full-story control correctly, separately from
the judges' no-story scores. Perfect observed accuracy and a degenerate
bootstrap interval do not establish perfect population accuracy. Debate
accuracy and probe assistance are separate measurements, reported above and below.

This table describes the corrected source-control runs. A full-source
audit found generator metadata appended to source 114: its question, options and
literal answer key. This is the only such footer found in all 166 original
stories. The other 23 fresh families' full-story accuracy is 39/46 for native
4B, 44/46 for Haiku and 46/46 for both Qwens and Opus. Thus the perfect Qwen
source scores also hold on those 23 sources without an appended answer key.

The v5 source amendment removes exactly 175 non-narrative footer characters,
preserving all 28,622 preceding narrative characters and every other row byte.
Questions, options, gold answers, source-family membership and schedules are
unchanged. All eight cleaned-source API followup calls answer both orders
correctly; corrected Haiku full-story accuracy is 46/48. The two native
followup calls both answer incorrectly, giving corrected native accuracy 39/48
(family-bootstrap interval 64.6–95.8%). Every no-story prompt is unchanged. Original
receipts and the exact removed footer remain available; no historical receipt
is relabeled. The corrected API calls cost $0.086790802; native inference is
accounted separately in the GPU rental.

All five original models have complete, valid generation receipts. The first Opus batch
stopped issuing calls after 83 of 96 because its conservative in-flight budget
reservation would not fit. A separate recovery reused those exact receipts and
issued the remaining 13 with unchanged prompts and runtime settings; no valid
answer was rerun. The complete results and plot are saved in
`runs/iteration3/final/source_control_results.json` and `source_controls.png`.
The corrected report is `final/source_control_corrected_all_v1.json`; its
30-cell table and figures are in `final/source_controls_corrected_v1/`.
The shared bootstrap keeps its fixed seed and 10,000 draws. Canonical sorting
of corrected records changes finite Monte Carlo interval endpoints even for
unchanged no-story observations; a separate provenance review records both
orders. Original results are preserved.

## Frozen fresh debate comparison

The final reserve contains 24 unseen original source families, one question per
family. Source-based repairs affect questions/options at source lines 41, 80,
116 and 165. Story text remains unchanged apart from the non-narrative source-114
footer removed in the documented amendment above. A pre-outcome metadata
amendment balances displayed gold and first speaker to six families in each
of the four cells, identically across participant models. Both answer orders
are evaluated in the separate no-story/full-story controls. There is no
constructed 50% no-story claim for these original, unpaired stories.

All 48 embedded source-only participant checks and all 288 public answer fields
pass. The separate contextual diagnostic originally accepts 90 of 96 jobs:
88 correct and two wrong. The six remaining jobs return `quotation` instead of
the required `evidence` key. A documented read-only alias interpretation
preserves the 90 accepted records exactly and selects the earliest complete
alias response for each unresolved job, independently of correctness. This
gives 94/96 correct answers, with 83 exact source-evidence strings. All 118 raw
attempts, strict schema failures and charges remain intact. Both wrong answers
are Qwen3.5 on source 116. The changed-system scope above prevents interpreting
these as proof of generation-time ignorance. No family or episode is excluded.

Blind prose checking covers all 288 turns and repairs five non-exact evidence
quotations. Manual review finds one genuine mixed-stance turn: Qwen3.8 on
source 8 concludes the opposing answer despite retaining its assigned answer
field. A second residual flag, on source 41, is an annotator error. These
distinctions are recorded separately; neither the outputs nor original labels
are silently rewritten. Role adherence is not factual truth.

The primary local truth labels cover 1,977 fixed whole-text sentence units across
all 288 turns, with
independent Qwen and Opus checks, explicit scope/unknown labels and reported
coverage. A predeclared secondary clause diagnostic covers the first
wrong-assigned opening in every episode: 48 responses, chosen without probe
scores or truth labels. This addresses a development coverage warning: eight
fixed wrong-advocate openings contain 35 sentence units, with 16 contradicted,
19 unknown and no fully supported units under the strict mixed-conjunct rule.
Within-response AUROC is undefined on that selected diagnostic sample. Its
role-aware manual labels are not a fresh generalization estimate. Labels never
select which text is shown to judges. All displayed characters and all turns
are preserved. The final comparison separates a raw baseline, matched neutral
display, real signals, shuffled signals and an explicitly labelled oracle.
Before any fresh labels, the complete payload recount prompted an operational
amendment: a 32,768-token output allowance and a 1,200-second client timeout for
both annotators, with a $20 primary-stage ceiling. The original source modules,
all 52 batches, complete prompts, schemas, model identities, reasoning settings
and retry count remain unchanged. Observed smoke throughput showed that the
old 300-second timeout could end a large response early; a fresh valid Qwen
response subsequently took 361.8 seconds. The optional 771-clause diagnostic
was subsequently released under its original $5 ceiling after the actual
primary cost and conservative rental bound left sufficient headroom. None of
these limits authorizes exceeding the $75 total cap.

Real-minus-neutral is the primary assistance contrast; shuffled damage is not
uplift. The experiment measures offline judge assistance, not debater-feedback
B−A or D−B.

Native Qwen3.5-4B and Haiku receive the full detector package. Opus receives
a predeclared smaller package with the two primary signals, to compare
the same paired uplift across judge strengths within the compute cap. Opus
uses explicitly requested low-effort reasoning because its API does not offer
the same disabled-reasoning contract. Eight old-data smoke calls all finish
normally with 11 completion tokens and zero reported reasoning tokens; this
does not establish hidden reasoning use. Repeated identical API prompts provide
a descriptive answer-disagreement diagnostic, with provider route differences
recorded and no increase in the independent-family count. The native runner
caches identical prompts. Its cached agreement is reported separately and
cannot measure sampling noise. A read-only reporting amendment resolves each
logical job to its original generation and preserves every reuse flag.

API recovery normally binds both the exact request and complete job tag. This
prevents distinct identical-prompt jobs from inheriting one another's recorded
stochastic answers. The integration suite at the original freeze passes 96 tests, including
recovery, full-context contracts and exact control-result coverage.

The first next-day preparation recovery exposed a catalog-dependent exception:
Qwen3.8's provider price ceiling changed, so seven already-complete jobs were
sent again despite unchanged inference settings. The interrupted attempt and
all charges remain recorded; its duplicate responses never replace earlier
valid responses. A separate operational adapter now selects the earliest valid
receipt by complete job identity and permits only the price-ceiling field to
differ. It preserves the original request, hash and response, rejects any
changed inference field, and never charges a reused receipt again. The frozen
API and preparation driver are unchanged. Twenty-two canonical plans are
reused, and 74 are needed, including four new-input plans for cleaned source
114. The client transport timeout is 1,800 seconds; the full 16,384-token
completion allowance remains unchanged. The amendment records that original
source-control outcomes already existed; it does not claim an earlier freeze.

The relevant CPU integration suite now passes 122 tests, including recovery,
preparation completion, contextual alias and generation-provenance regressions.
An unrelated legacy pytest/torch pipeline module could not load
on this CPU VPS and is outside that count. All 109 files bound by the original
freeze remain byte-identical. The source-review sample also preserves an
attribution ambiguity on parent 124 and other interpretive caveats, without
removing families or tuning on judge/probe outcomes.

A further operational gate handles the frozen API's existing schema retries.
It requires one accepted, complete result per canonical preparation job, while
retaining and charging every failed raw attempt. This corrects the launcher's
overly strict requirement that every attempted response be schema-valid.
The source, inference settings, retry allowance and valid responses remain
unchanged. Five focused gate regressions pass, including a failed schema attempt
followed by valid completion and an unresolved job that blocks continuation.

Returned public reasoning is now replayed in full when present, ahead of the
scored final response. All 32 checked returned traces round-trip under the
pinned Qwen tokenizers; final-response pooling excludes reasoning, and both DYL
arms preserve the original context. A two-model GPU smoke also completes.
This remains an explicitly defined unpadded teacher-forced replay, not verified
equivalence to opaque provider prompts, hidden reasoning or live execution.

Fresh extraction completes 144/144 turns per model, with maximum complete
contexts of 10,177 and 10,545 tokens and exact response/source alignment.
Both short alignment diagnostics pass, and the DYL causality discrepancy is
zero on all fresh rows. Long-context diagnostics fail only the layer-58
left-padding RMS bound: .03363 for Qwen3.5 and .04018 for Qwen3.8 versus the
unchanged .03 criterion. All incremental and cosine checks pass. The latter
padding discrepancy exceeds the historical range; it is retained and reported,
without changing tolerances or excluding cases. The frozen experiment uses
unpadded replay. Generic ten-token diagnostic continuations do not establish
all-token equivalence for real responses or provider execution.

The portable transcript inspector preserves every displayed character, uses
the exact frozen score percentiles and shows complete source stories, private
preparations and Unicode character coordinates on demand. All 1,344 combinations
of fresh episode, detector, answer visibility and color mode preserve the full
story and argument text in Chromium at each of two viewport widths. Actual
desktop and phone screenshots have been inspected. Its highest-scored-unit navigation
is an inspection aid, not an additional evaluated judge condition.

## Next experiments and retention

The next environment test should reuse balanced pairs of coherent source worlds
with identical question/options and opposite gold answers. A future-only scaffold
contains two already-consumed development families, four complete sources and a
review of all 31 changed paragraphs. Eight offline checks cover source isolation,
answer aliases, options and consumed-family boundaries. This construction was
already present in the development machinery; the scaffold makes its controls
explicit. A fixed no-story answer scores exactly 50% across an opposite-gold pair
by construction, which is not evidence that the debate task is hard. Empirical
debater knowledge and judge headroom must still pass before a larger run. The
28 remaining unused families have not been consumed by this scaffold.

For local probes, prioritize quote/attribution-aware units and explicit atomic
relations, followed by independent annotation and source-matched controls.
Qwen3.5 opening clauses justify a focused follow-up; the primary sentence results
and Qwen3.8 regression argue against a broad new-head sweep. For assistance,
test a smaller number of displays on the native judge, retaining neutral and
shuffled controls and a smaller strong-judge comparison. Do not tune a display
on these same 24 families and call the result a new generalization estimate.

A future-only quotation/honorific scaffold now stages 45 already-consumed
development responses from four families. Its proposed boundaries change
257 old sentence units to 274 units across 29 responses while preserving every
source, argument and raw-JSON character. Eight test methods cover 12 boundary
fixtures, nine invalid-contract mutations and 80 seeded character-conservation
cases. The proposed annotation schema separates quotation content, attributed
speaker/actor and provider/recipient/object relations. No API calls, truth labels,
probe fits or current-result changes were made. These mechanical checks do not
establish complete claim enumeration, semantic entailment or improved probe
accuracy; natural-unit review, independent development annotation and a new
evaluation freeze are still required.

The cumulative ledger is fully settled at **$68.0954558278 of $75**, leaving
$6.9045441722 unused. This is conservative accounting, including timeout and
rental upper bounds, rather than a claim about an exact final provider invoice.
Final Haiku/Opus judging costs $0.6082176/$8.042816. The replacement GPU settlement
is $3.5097756166, covering intent-to-verified-absence runtime at the hourly upper
bound plus the full traffic reserve. Both task rentals are absent; unrelated
instances remain untouched. No paid work remains active.

The code, aggregate tables and figures form the public review snapshot. Full
source texts, private preparations, raw provider receipts, annotated units,
scores, heads and the transcript inspector are retained separately. Both fresh
selected-feature archives were split into four parts, each independently
downloaded from Drive and checked for exact length and SHA256. The whole archive
identities are `00e8500ddf9bdb1c4fa49e0f4f428a0395d1b4d06adb494dd50d8d739019367b`
(Qwen3.5, 255,416,055 bytes) and
`565f94868f1360fc43e60bfc4490212c3dbb87d3a90efb12ef010aec6099e436`
(Qwen3.8, 259,533,332 bytes). The raw feature-provenance archive is 2,937,781 bytes,
SHA256 `926dfafda05a25b3110a3bc478c3e9f4f714414915cd70d458c7c41812f4188a`,
and also passes an independent Drive download check.

Retained tensors cover all fixed sentence units and wrong-opening clause units
at all six extracted layers, with original dtypes, full text/coordinates and
response features. They are not the complete transient all-token activation
cache. Other feature subsets require re-extraction from pinned inputs. Public
GPU/VPS package inventories and pinned model/image identities record the runtime
without claiming equivalence to hidden provider execution. Public aggregate
inputs reproduce the source-control, sentence and judge PNGs byte-for-byte;
vector files can differ in timestamps and element identifiers.
