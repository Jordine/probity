# Iteration 6: Qwen3.5 debate ablation inventory

The offline planner has inventoried 166 questions, 116 source families, 332 Qwen3.5-27B A/B trajectories and 1,992 public turns. This inventory launches no API calls or rentals. Reproduce it from the repository root:

```bash
python3 runs/iteration6/planning/build_plan.py
```

The generated `runs/iteration6/planning/turn_tasks.jsonl` contains one source-bound task per public turn. `summary.json`, `probe_inventory.json`, `prompt_inventory.json`, `screening_proposal.json`, and `cost_plan.json` record counts, immutable artifact hashes, exact prompt function sources, proposed family selection, and historical costs.

## What can be replayed

Both A and B can be scored offline on their actual saved contexts. For B this means retaining the exact prior-feedback packet in each original request, along with the system instructions, private plan, story, public history, and returned completion. Alternative detector scores on these contexts measure detection on behavior induced by the original detector. They do not tell us how the debater would have behaved with alternative feedback.

A new live probe, layer, display, or prompt requires a new continuation. The proposed shared-prefix design starts with the same two public turns, then forks into four more turns per branch. This estimates effects conditional on a common first round. It differs deliberately from iteration5, where B was independently generated and anticipated feedback from its first turn. Changing the fabrication prompt also changes private preparation/openings; prefix sharing is valid within a prompt condition, not across different private accounts.

The replay scorer must join the audited saved inference contexts rather than regenerate prompts from current Python code. Verify response token boundaries, complete context length, model/tokenizer revision, selected layer, and training-time pooling. Do not remove original B feedback to construct an imagined counterfactual. Private reasoning returned by the provider is distinct from the private prepared account and from text subsequently fed back to the model; preserve that distinction.

## Available probes and layer ablations

| Probe family | Saved feature | Selected Qwen3.5 layer | Availability |
| --- | --- | ---: | --- |
| Adapted advocacy | response mean | 43 | Frozen weight |
| Adapted advocacy | argument mean | 43 | Frozen weight |
| Adapted advocacy | No-prefill DYL | 51 | Frozen weight |
| Adapted advocacy | No-minus-Yes DYL | 35 | Frozen weight |
| Apollo-data | response mean | 43 | Frozen weight |
| Apollo-data | No-prefill DYL | 19 | Frozen weight |
| Apollo-data | No-minus-Yes DYL | 43 | Frozen weight |
| Published FIBS | Published DYL contrast | 19,27,35,43,51,58 | Published multilayer contract; verify loader/weights before execution |
| Local2 | Previous selected local readout | Check frozen contract | Frozen weight |
| Local3 | Sentence/clause span mean | 35 | Frozen weight |
| Local3 candidate grid | Layer-specific local vectors | 27,35,43,51 | All 32 trained candidates retained |

The Apollo-data heads were trained locally using the Apollo roleplaying data; they are not verified copies of collaborators' checkpoints. The published FIBS head is a separate published artifact. Exact hashes and native revision are in the JSON inventory.

Adapted and Apollo trainers evaluated layers 19,27,35,43,51,58 but saved only the winner for each feature. Candidate performance records are not candidate weights. To ablate their layer, reproduce per-layer fitting on the original training examples and original family-disjoint development selection. Do not apply the layer-43 weight to other layers. The existing local3 candidate archive permits a real layer ablation immediately after compatible feature extraction. It does not make alternate response heads available.

`probity_benchmark/qwen_extract.py`, `extract_iteration.py`, `cache_dyl_denials.py`, `local3_score_units.py`, and `score_frozen_response.py` are existing components, not a ready-made combined iteration6 runner. The older extractor writes feature caches. Use GPU-local temporary storage for any required pooled caches and stream scalar outputs with text/span offsets to the VPS; never transfer raw per-token hidden states to VPS disk. An integrated scorer must respect the qwen alignment preflight and refuse silent truncation. Response/argument/local heads can share compatible forward passes. Yes/No DYL continuations add work and cannot be costed as free extra linear heads.

## Annotation and selection

144 A turns have verified reusable earlier fixed-sentence consensus annotations: exact request SHA, submitted messages, complete source, returned response, unit coordinates, original annotation system rubric and item schema all match. The compiler also verifies every old annotation unit is covered. These labels assess source factuality, not interaction behavior. The other 1,848 turns require new sentence labels. All 1,992 turns need the new contextual behavior rubric if measuring concessions, self-contradiction, responding to challenges, and changing a claim after feedback.

Existing annotation implementation: `probity_benchmark/local3_annotate_units.py`; corpus: `runs/iteration3/final/local_sentence_annotations_v1/corpus.jsonl`. Its input loader expects complete single-source directories/receipts. Iteration6 must adapt this to the audited combined A/B inventory rather than supply directories opportunistically. Annotation input must contain complete source, exact argument, and (for interaction labels) the relevant public prefix, while hiding probe scores, assigned truth role, and outcome. Unknown/unverifiable labels remain in coverage and must not be silently turned into truthful negatives. LLM labels remain silver labels, including where models agree.

The proposal consumes the authoritative root `screen24_v2` manifest, which retains the v1 scientific selection of 24 source families by a fixed hash ordering and one question per family, without using outcomes in the selector. Remaining 92 families are disjoint from this new screen. All 116 families have already been observed in iteration5; a fresh split does not make them historically untouched. Carry all variants of a source family into the same partition. Before local training, audit original training-family overlap and keep probe fitting/development separate from current evaluation.

## Staged campaign

1. Audit exact contexts and latest A/B behavior, reuse verified old labels, annotate new sentences and interactions. Review at least a stratified subset independently; reserve extra review for disagreements without changing the representative sampling denominator.
2. Run a small native GPU preflight and measure throughput. Replay frozen heads first; recover missing per-layer heads on old training data and freeze selection before scoring the remaining families.
3. On 24 screening families, start with seven branches: no feedback, matched neutral formatting, real current bundle, shuffled-score bundle, turn scores without recommendation, speaker means only, and recommendation only. Two shared opening turns and four continuation turns imply 672 new turn calls and 168 judgments per judge. Define causal shuffled-score controls before launch: no future scores from the same debate may enter a current request.
4. Hold display fixed to compare a small set of valid probes/layers; hold probe fixed to compare local display and prompts. Choose branches by transcript-label performance and behavior, not role-proxy AUROC alone. Freeze candidate selection and family partition before the larger comparison.
5. Confirm selected conditions on remaining families and then report all 166 questions with the family weighting and selection provenance. Multiple seeds and the full Cartesian product are separate later expansions.

The original verified `debate/debate_manager.py` exposes symmetric, persuasive, selective and explicit defense instructions. None of those functions supplies a detailed prewritten false account. The current `environment3_prepared.py` asks the model to prepare a coherent fabricated account; `environment3_evidence.py` then asks it to use the invented quotations. A proposed externally specified false account should be described as a new intervention until the intended original template is located. Copying the entire old prompt would additionally change word limits, quote rules, and judging instructions, so targeted prompt substitutions are needed to isolate defense framing.

## Cost evidence and authorized limit

The earlier 288-turn sentence-label job cost $0.604594 for Qwen3.8 and $13.311216 for Opus. Scaling linearly to all 1,992 turns gives about $4.18 for the cheap annotator and $92.07 for full Opus duplication, or $9.21 for a 10% Opus second-reader sample. Reusing 144 old turns can reduce factual-label calls, while richer interaction labels may cost more. These are historical same-protocol extrapolations, not a guarantee for the new rubric or current prices.

Historical 27B rental rates were approximately $1.32–1.42/hour; the much lower $0.20–0.25 rates in the ledger were for the smaller judge and should not be used to budget 27B replay. Multi-layer hooks share a forward pass, but DYL variants and long contexts add compute. GPU wall time must be estimated from a measured preflight.

The user subsequently authorized a **$60 hard total iteration6 cap**, recorded in `runs/iteration6/authorization_v1.json` and the shared ledger. Primary historical factual labeling is capped at $8; independent Opus audit at $12; fresh display-suffix factual labeling has a separate $3 allocation within the same $60. GPU and generation stages share the remaining campaign limit through coordinator reservations. The earlier $50 proposal is superseded. This buys an exploratory sequence, not every axis over all 166 questions. The previous $45 dataset-only ledger remains separate; its unused balance is not transferred.

## Compiled annotation execution contract

The offline annotation compiler now resolves the audited canonical receipts directly, including both JSONL receipts and individual feedback JSON receipts:

```bash
python3 runs/iteration6/planning/compile_annotation_jobs.py
python3 runs/iteration6/planning/run_annotation_jobs.py
```

It compiled 13,029 fixed sentence units across all 1,992 turns. Of these, 967 units on 144 turns reuse exact verified old labels; 12,062 units on 1,848 turns form 379 new source-grouped jobs of at most 40 units. `annotation_requests.jsonl` contains only a job hash, blinded messages, and response schema. Separate private metadata maps returned indices to original turns/spans. System prompts, private plans, scores, arm, debater-model identity, gold answer, and assigned truth role are excluded from annotator messages. Public argument wording can still reveal clues, which metadata blinding cannot remove.

The unchanged factuality rubric has complete source plus complete current argument, without previous turns. It does not label concession, self-correction, prior-turn attribution, or feedback response; those need a separate contextual interaction rubric and new labels. These are full-sentence labels, not atomic claim disaggregation.

`run_annotation_jobs.py` validates offline by default. Paid execution requires `--execute`, an explicit positive batch `--budget`, a global ledger, an authorization JSON pinning `annotation_jobs_sha256` and `annotation_runner_sha256`, allowed annotation models, and a new output directory. It uses the existing locked ledger and `BudgetedChat`; saves raw provider request/response receipts and accepted annotations; validates copied text, evidence indices, and schema; and quarantines isolated normal-stop schema errors with known charges after at most one identical-request retry. Other independent jobs continue; later versioned repair jobs contain only uncovered exact units. Transport, truncation, unknown liability, budget gates, or a systematic validation failure rate stop new calls. Resume requires settled prior liability and a fresh output directory inside the ledger campaign. The runner automatically discovers all previous raw receipts and validates logical job/model identity, so changed catalog prices or an omitted reuse flag cannot trigger repeat billing. A locked, durable campaign journal marks each job before submission; unresolved in-flight or failed logical jobs block continuation until explicit recovery. Duplicate models are rejected. All pending job context bounds and the maximum single-job reservation are preflighted before the first submission; a bounded campaign continuation can resume after known charged schema failures, within the remaining $8 primary allocation. Unknown transport liability cannot be automatically restarted. Eleven runner tests, six unit-recovery tests, and eight assembler tests passed. All raw failed responses remain available.

Individual reader checks and two-reader consensus are separate label tiers. A cheap primary reader over new units must be compared with the same primary reader extracted from the old annotations; old consensus must not be pooled with new single-reader labels. `probity_benchmark/grid6_labels.py` assembles those tiers and records missingness. Reused units have their two original checks verified against referenced raw annotation receipts, source evidence paragraphs, and recomputed reconciliation; both old corpus and receipt hashes are pinned in the annotation plan.

The independent reader audit is complete: 36/36 new Opus jobs accepted for $8.35340 including the initial three-call smoke test. Its frozen hash-ranked sample covers 12 of 116 source families, 192 turns, and 1,230 units including 65 previously dual-labeled units. Agreement analysis must use this sample rather than all conveniently available old dual labels. `analyze_reader_audit.py` reports family-weighted agreement and family-bootstrap intervals; agreement is not accuracy against human ground truth.


Historical primary labeling is now complete: all 13,029 units have primary annotations, with 12,062 new units costing $4.007514254 including failed attempts. Final labels are in `runs/iteration6/labels/historical_v1/`. Four checks use an explicitly approved waiver only for rationale length above 600 characters; full original rationale and strict-failure receipts remain, and every substantive text, evidence and verdict check still applies. Two affected checks are in the independent audit. No raw wording or verdict was rewritten.

The completed [historical factuality report](probity_iteration6_factuality.md) links the paired A/B estimates and final reader-audit figures. All sentence units are annotated, but unknown or out-of-scope units remain ineligible for binary factuality. This distinction, primary versus consensus tiers, and separate coverage denominators remain explicit. The new display suffixes require their own factual labels because their arguments changed.
