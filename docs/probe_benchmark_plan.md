# Probity deception-probe benchmark plan

The benchmark asks two separate questions: can a detector identify source-grounded false claims inside a debate, and does showing its output improve a source-blind judge's answer? Whole-response discrimination, local factual classification and judge uplift are different measurements. Assigned advocacy roles are never substituted for local truth labels.

This public methodology summary replaces the original working note. Historical pilot results are described in [iteration 2](probity_iteration2_results.md); the current design and limitations are in the [iteration 3 plan](probity_iteration3_plan.md) and [iteration 3 results](probity_iteration3_results.md). Private correspondence and external working-document extracts are omitted.

## Environment validation

Use the existing synthetic reading-comprehension setting as a controlled environment. Audit source structure, question/answer consistency, embedded answer metadata and source-family overlap before interpreting model behavior. Split at the original story-family level: different questions or paraphrases from one story are not independent test examples.

Measure no-story and full-source answer controls with both displayed answer orders. Keep source access identical within each comparison and distinguish judge performance from debater knowledge. A high no-story baseline may reflect option-content cues, position preference, familiar material or inference noise; order changes alone do not isolate those causes. A strong full-source judge is a positive control, not the source-blind debate condition.

Iterate with Qwen 27B participants. The selected iteration-3 protocol uses reasoned private evidence preparation followed by three rounds of public arguments. Both advocates receive the authentic source, an explicit verified answer and an assigned advocacy target. This measures instructed factual misrepresentation with strong in-context cues; it does not establish hidden-goal deception or latent intention.

Retain complete source, preparation, preceding public history, returned text and request identity. Check actual prose as well as answer fields. Preserve mixed advocacy, correct-role overclaims, true statements in wrong-role arguments and source ambiguity. The later context diagnostic replaces the original system instruction, including its verified-answer cue, and therefore is not an identical generation-time knowledge test.

## Comparable and correctly aligned probes

Keep published FIBS weights, separately trained Apollo-data heads, DYL signals and new local heads clearly identified. Exact collaborator checkpoints remain unverified. Record participant checkpoint, tokenizer revision, layer, normalization, training split, pooling and score orientation. Do not apply a model-specific head to an incompatible hidden-state space.

For DYL, retain the complete action context and specify the exact follow-up, prefill, scored tokens and yes/no contrast. Check token coordinates, padding, chat wrappers, reasoning exclusion and full-context handling. Native provider generations and local replay are distinct systems; successful serialization checks do not by themselves prove backend equivalence.

Compare complete-response, sentence and clause aggregation under frozen rules. Matched neutral displays must use the same presentation as real detector scores so display effects are not mistaken for detector utility. Raw logits are not calibrated probabilities.

## Local factual supervision

Train and select local heads on development families, including minimally changed truthful/deceptive pairs and mixed responses where authentic statements occur beside false or misleading ones. Evaluate on untouched source families with independent, source-grounded annotations of fixed complete units. Preserve unknown, disputed, quoted, hypothetical and mixed claims rather than forcing binary labels from speaker identity.

Use held-out selection to compare objectives, layers, normalization and pooling. Freeze heads, reference distributions, thresholds and display rules before fresh outcomes. Inspect deterministic false-positive and false-negative examples qualitatively after evaluation; do not use that inspection to relabel the test set or improve reported thresholds.

## Judge utility and reproducibility

Separate offline assistance on fixed transcripts from interventions that change a debater's subsequent behavior. The latter require newly generated trajectories. For fixed transcripts, compare real signals with matched neutral signals and report the raw baseline and labelled oracle separately. An oracle provides a usability ceiling, not evidence that a deployable probe achieves that uplift.

Prespecify paired comparisons, cluster uncertainty by source family, retain all valid cases and report operational failures. Save exact request/response identities, all failed attempts, accounting and versioned recovery amendments. Reuse accepted outputs only under the audited request/tag contract; never regenerate valid wrong answers selectively.

Use the separate API and GPU dependency specifications in `probity_benchmark/`, preserve model/data revisions and publish code, aggregate results and artifact hashes. Full evidence and feature caches remain separate from the Git repository. A reproducibility package must distinguish reanalysis of retained evidence from a new stochastic provider run.
