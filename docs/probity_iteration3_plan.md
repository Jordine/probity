# Probity iteration 3: parallel development and a fresh comparison

The user authorized an additional $50 on 2026-10-14, raising the cumulative
experiment cap to $75. The previous conservative accounting was $19.5502.
All API calls and rental reservations use the same process-locked ledger at
`runs/pilot_budget.json`; workstream allowances are ceilings, not spending targets.

The runtime permits four concurrent agents including the coordinator, so three
persistent research agents work alongside the coordinator. Their durable message
board is `runs/iteration3/board/`: immutable messages, one notes file per owner,
and explicit handoffs for shared code. The machine-readable allocation is
`runs/iteration3/fleet_plan.json`.

| Owner | Question | Initial API ceiling |
| --- | --- | --- |
| Local probes | Can factual contradictions be localized after controlling wording, position, source context, and annotation scope? | $8 |
| Environment | Can knowledgeable Qwen 27B debaters produce coherent false advocacy that creates useful judge headroom? | $10 |
| Judge controls | Is difficulty due to missing information, role cues, confusing inputs, or a judge unable to use reliable assistance? | $5 |
| Coordinator | Shared GPU, new holdout, frozen evaluation, artifact retention, integration and accounting | Remaining authorized budget |

The first development batch compares coherent counterfactual rehearsal packets,
sealed versus sequential turns, authentic-source private preparation, and explicit
assigned-answer preparation. Failed controls remain part of the report. Lower
judge accuracy caused by debaters misunderstanding the authentic source is not
accepted as useful headroom. Full-source competence, private-context knowledge,
answer fields, prose adherence, and local factual claims are separate checks.

The local workstream pairs identical complete responses with different complete
source stories. Role assignments, response wording, and positions are unchanged.
Only independently verified claim-label flips enter training. Unsupported or
scope-dependent substrings remain unlabelled. Features may be extracted before
verification, but final training uses a separately hashed projection of verified
labels. Candidate heads share one activation cache per model. A small development
search compares source-pair discrimination and within-response localization,
including frozen old heads, position controls, and random directions.

The first reserve sampled 24 unused question/story rows with seed 2026101430.
A subsequent source-family audit found that 166 rows represent 116 original
story families, and the 74 previously consumed rows span 64 families. Before any
fresh no-story, debate or probe outcomes, the reserve was amended to 24 previously
unseen families, one question per family. The first eligible scheduled question
per family was retained; eight additional unseen families were sampled with seed
2026101431. Initial and amended locks, all exclusions, source identities and
dataset hashes remain under `runs/iteration3/coordinator/holdout*_lock*.json`.
The final sampling unit and bootstrap unit are the original story family.

These original stories are not automatically counterfactual twins: there is no
constructed 50% no-story claim for this pool. Source-only validation and documented
source-based question repairs occur before debate generation. Repairs change
questions/options only when supported by the complete source, and avoid adding
answer-revealing facts to the question. Full source strings are preserved. No
input is dropped because of its no-story, debate, or probe outcome. Baseline
accuracy and the predeclared short-option subgroup (both answers at most six
words) remain measured outcomes; order counterbalancing cannot remove semantic
answer-option cues.

Freeze the environment, heads, score orientation, normalization references,
display and comparison before final outcomes. The primary assistance contrast
is real scores minus a matched neutral display, paired within episodes and
bootstrapped by parent story. Also report the ordinary baseline, neutral-layout
effect, shuffled-score harm, real-minus-shuffled, an explicitly labelled oracle,
and corrected/harmed judgments. A shuffle advantage alone is not uplift. Local
displays partition every argument independently of annotation labels and retain
every character. Whole-response role scores do not establish local lie detection.

All contexts and generations remain complete. A length finish or invalid parse
stops dependent analysis, with full failed receipts retained. Native replay stays
explicitly distinct from the provider's live runtime. GPU jobs share one lock;
the rental has a fixed deadline and an independent VPS shutdown watchdog. Task
artifacts are copied and verified before destroying the rental. On 15 October,
the user also authorized pushing the code and reports to GitHub. Credentials,
private reference material and raw activation archives remain outside Git.
External collaborator messages are not part of this work plan.

## Execution amendments, 15 October

The initial ceilings above describe development allocations. The cumulative
$75 ceiling is unchanged. The complete fresh comparison uses source dataset v5
and global freeze v2. A source-only audit removed a 175-character generator
footer from source 114 before public debate generation, preserving all narrative
bytes. Recovery retains original attempts and charges, while selecting canonical
complete receipts without rerunning valid jobs.

The 48 debates and 288 turns are complete. The separate contextual knowledge
diagnostic replaces the original system message, including its explicit correct
answer, and retains non-system inputs before each speaker's own last turn. Its
errors cannot establish generation-time ignorance. All families remain in the
primary comparison, including ambiguous-source and mixed-advocacy cases.

Primary independent sentence annotation has a separately released $20 ceiling,
32,768-token allowance and 1,200-second transport timeout, justified by complete
payload counts and measured development throughput before fresh labels. The
optional clause stage remains held pending actual costs. Final judge API runs
require exact package costing against the live ledger. Read-only reporting
amendments separate cached native answers from distinct model generations and
preserve original result bytes. See [the results record](probity_iteration3_results.md)
and the versioned receipts under `runs/iteration3/final/` for the current state.
