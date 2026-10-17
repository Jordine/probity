# Probity paper options and evidence plan

The strongest current direction is a **benchmark and evaluation paper** about
when probe assistance improves a judge's decisions in explicitly instructed
debates. The completed 2×2 grid supplies a concrete result and an auditable
protocol. The dataset-only expansion can show how that result varies across
questions, source families and prior exposure. It will not by itself establish
a new general deception detector or the causal effect of numerical feedback.

This is a planning note, not a report of expansion results. The
[completed results](probity_iteration4_results.md),
[exact protocol and gaps](probity_iteration4_protocol_scope.md),
[earlier detector comparisons](probity_iteration3_results.md) and
[costed run options](probity_cost_plan.md) remain the supporting records.

## Possible paper contributions

| Direction | Evidence already available | What the 166-question expansion adds | What would still be needed |
| --- | --- | --- | --- |
| **Main: environment validity and matched probe-assistance benchmark** | Source/no-source controls; two Qwen27B participant models; native4B and Haiku judges; full A/B/C/D grid on 24 families; earlier neutral, shuffled and oracle judge displays; exact-context and token/cache audits | Coverage of the supplied corpus under the same protocol, exposure-stratified results, family-level heterogeneity and more paired outcomes | New-family source/no-source controls if claiming validity of the expanded environment; a genuinely independent corpus or later locked replication for broad generalization; neutral/shuffled feedback arms for a mechanism claim |
| **Conditional: local truth localization or a new probe method** | Apollo-data response/DYL and other offline comparisons; local pooling/logistic/ranking/annealed searches; 1,977 sentence annotations and 771 opening-clause annotations; mixed-response evaluation with inconsistent gains across models | More existing-probe scores and examples only if explicitly collected; the current dataset-only grid does not automatically provide factual labels or a new head comparison | Validated atomic units, a random human-adjudicated subset, a held-out-family training/selection contract, and reliable gains over previous heads and controls within mixed responses on both models |
| **Secondary: behavioral changes under probe feedback** | Paired old/new trajectories with complete score histories; outcome fixes/harms; limited qualitative review; no automatic role-switch or disclosure flags in the 288 feedback turns | More paired trajectories under the fixed intervention, including fresh no-feedback/feedback pairs for newly added questions | Blinded factual/prose annotation, contemporaneous randomized generation, trajectory repeats and awareness-only/neutral/shuffled feedback controls before attributing changes specifically to informative scores |

The local-method direction should remain contingent on new evidence. High
pooled role classification is not proof that a probe can find a false sentence
inside an otherwise truthful response. The existing local comparisons and
negative results belong in the benchmark paper even if no new method succeeds.

## Candidate main figures and claim boundaries

| Figure or table | Existing evidence and defensible claim | Planned expansion analysis | Remaining qualification |
| --- | --- | --- | --- |
| Protocol diagram and environment headroom | A/C share one transcript; B/D share another. Judges see the question and debate, not the story. Question-only accuracy on the selected 24 families ranges from 50% to 64.6% across five models | Show question/family coverage, lengths, exposure and any separately collected source/no-source checks | These controls do not prove chance-level solvability. High A accuracy is judge answer accuracy after debate, not debater knowledge or probe accuracy |
| Two judge-specific 2×2 grids | Native correct counts A/B/C/D = 37/40/47/47 out of 48; Haiku = 43/44/44/47. The adapted response proxy helps the native judge on this selected sample | Family-macro accuracy for all 116 families, both participant models, and exposure strata; question-weighted counts alongside | Existing 24 families were reused after favorable detector selection; full-corpus results include training and development exposure |
| Paired effect forest plot and fixes/harms | Native C−A = +20.83 percentage points; Haiku C−A = +2.08. B−A intervals include zero for both judges | B−A, C−A, D−B, D−A and interaction, with paired family intervals and model-specific estimates | No reliable feedback-only improvement is established by the pilot; larger sample size alone does not remove intervention or time confounds |
| Detector/display and local-validation comparison | Earlier seven-track offline comparison includes Apollo/DYL, response and local displays; new local heads did not improve localization consistently across models | Retain these prior results with their original denominators; any expanded comparison must be declared separately | The dataset-only expansion covers one adapted response head, not a full probe/display sweep; 771 annotated opening clauses are not all 5,598 displayed clauses |
| Reproducibility and sensitivity panel | All 96 Haiku A/C requests freshly replayed; two answers changed in one family. Haiku D−A shifts from +8.33 to +6.25 points and its interval includes zero | Separate reused and newly generated outcomes; report request/runtime/provider identities, failures, exact reuse and missingness | Replay was post-result. It cannot separate provider, time and sampling variation; four native controls are diagnostics, not a replication |

Use judge-specific panels rather than implying all judges benefit equally.
Native and API judges have different serving and reasoning contracts. The full
grid has two judges; the earlier reduced Opus package does not supply Opus B/D.

## Dataset-only expansion and exposure

Keep the current prompt, two participant models, frozen response heads and
development percentiles, one generation schedule and two final judges fixed.
With all **166 questions from 116 original story families**, the design contains
332 matched question/model units, 664 six-turn transcripts and 1,328 logical
presentations per judge. If every existing identity remains exact, 48 matched
units and 192 presentations per judge are already available: the incremental
work is 284 matched units, 568 transcripts and 1,136 presentations per judge.
These are target counts, conditional on completed valid runs. Cache hits are
reuse, never independent replicates.

Freeze the family map and exposure ledger before inspecting new outcomes.
Retain separate flags for detector training, hyperparameter/reference
development, environment/prompt/baseline tuning, the existing 24-family results,
and no prior use. The adapted head used eight training families and four
development families. Other earlier work reached 64 families in total; the
iteration-3 reporting set added 24. **Only 28 wholly unused families, containing
35 question rows, remain before this expansion.** A new question from a used
story is not a new family.

Training/development/tuning flags can overlap. Publish those flags and use an
explicit mutually exclusive reporting hierarchy: training, then development or
reference, then other earlier tuning, then the existing 24 reporting families,
then untouched families. Report the actual family and question counts from the
frozen manifest rather than subtracting assumed disjoint counts. Group all
questions and rewritten versions of a named original story together; grouping
by source identity does not prove semantic independence between different
stories.

The full corpus is an exploratory coverage analysis. The untouched-family
stratum is the cleanest prospective generalization estimate available within
this dataset, provided no new fitting or selection uses its results. Its 28
families are not 35 independent questions. Consuming it now also means that
later tuning will require another independent source for confirmation. No head,
threshold or percentile reference is refitted on any expansion row.

## Planned inference and causal interpretation

The primary full-corpus estimator should give each original story family equal
weight: first average paired correctness or correctness differences across its
questions and both participant models, then average families. Report each
participant model separately as well. Show question-weighted accuracy and exact
numerators/denominators as secondary summaries; they need not equal the family
estimate when families have different numbers of questions.

Use a paired bootstrap of original families, retaining every question, both
models and all four cells together in each resample. Use the same convention
within each exposure stratum. Report 95% intervals, fix/harm counts and the
number of families contributing to every panel. A preregistered descriptive
analysis can use the existing 10,000-resample convention; unadjusted intervals
must not be presented as multiplicity-controlled significance. Predeclare the
five contrasts above and both judges, and report all of them. Degenerate
intervals at a ceiling are a finite-sample property, not certainty about future
stories. Rows with technical or protocol failures must remain accounted for;
any clean-subset analysis is secondary to the full declared inclusion set.

C−A and D−B compare judge presentations of identical text. B−A and D−A also
change the generated trajectory. In the reused 24-family stratum, they compare
old no-feedback generations with later feedback generations. Record generation
dates and scheduling for the new pairs and distinguish that stratum; do not
retroactively describe the whole study as contemporaneous or randomized.
The feedback treatment bundles its announcement, past response scores, speaker
means and a recommendation. Its components require separate ablations.

The detector measures association with assigned wrong-answer advocacy under
explicit in-context fabrication instructions. Native checkpoint replay provides
the scored activations; they are not hidden states exported by the API provider.
Neither role labels nor machine-annotated contradictions establish hidden
intent, knowing deception, or transfer to model organisms with unprompted
deceptive behavior. Those remain separate future questions.

## Reuse and implementation boundary

The original compiler deliberately requires one question per family and excludes
normalization-reference families. The 166-question analysis therefore needs a
new versioned compiler and explicit exposure contract; changing a count or
disabling a guard in the frozen iteration-4 code is insufficient. Preserve the
old transcripts, scores, primary judge outcomes and request/job identities.
Keep the fresh Haiku A/C diagnostic separate from primary cached outcomes.

Native judging can reuse exact rendered prompts under its pinned model,
template and runner contract. API judging can reuse only identical complete
requests with the same canonical job tag, including schema, reasoning and price
ceiling settings. A/C and B/D transcript sharing saves generation and annotation
work; it does not make raw and scored judge presentations the same request.
Stream scalar projections and compact pooled features rather than saving raw
activation tensors. The budget alternatives in the cost plan are separate
campaign choices; the present dataset-only expansion does not imply three
sampled seeds, exhaustive new annotation or additional probes.
