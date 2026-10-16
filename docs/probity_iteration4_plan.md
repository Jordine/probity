# Complete the matched 2×2 feedback experiment

The user requested completion of the grid on 2026-10-16, explicitly including
D−A and reuse of the original stories. The previous cumulative $75 spending
ceiling remains in force; $68.09545582782046 was accounted before this work.

| Cell | Debaters receive past probe scores | Judge receives probe scores | Transcript |
| --- | --- | --- | --- |
| A | No | No | Existing iteration-3 trajectory |
| B | Yes | No | New feedback trajectory |
| C | No | Yes | Exactly the A trajectory |
| D | Yes | Yes | Exactly the B trajectory |

Reuse every one of the original 24 source families, one question per family,
with both Qwen3.5-27B and Qwen3.8-27B: 48 matched trajectory pairs. Freeze source,
question, options, answer order, speaker assignment/order, private preparation,
three-round protocol, and public generation settings from
`runs/iteration3/final/debates_v1`. Public reasoning remains disabled as in that
final corpus. Only the documented private feedback intervention changes. Both
speakers can see the scores of all earlier public turns, never a future or
current turn. The instruction requires keeping scores out of public arguments.
Retain and audit any disclosure rather than silently rewriting a transcript.

Use the already frozen adapted response probe for this first complete grid:
the model-specific response-mean heads and development reference populations
from iteration 2, with the same score orientation and display as iteration-3 C.
This detector predicts wrong-answer advocacy; it is not a calibrated local-lie
probability. Its favorable previous result motivated this follow-up, so the
completed grid is exploratory on reused stories, not a fresh confirmatory test.

Score each action using its actual full generation request, including earlier
feedback. Run pinned bf16 native replay and verify agreement on old scored
actions before releasing new feedback. Keep one model loaded while processing
the six causal turn stages. Persist scores, exact contexts, hashes, complete
API receipts and metadata; do not persist new raw activation tensors to the VPS.

Reuse original A/C jobs and judgments. Judge each new B/D transcript in fresh
independent contexts with and without scores, using the original prompts and
runtime contract. Native Qwen3.5-4B and Haiku5.5 are first priority; complete Opus
as well if the remaining authorized budget permits. Reuse byte-identical
requests only when the complete runtime/cache contract matches. Receipt reuse
is an optimization, not an independent repeated measurement.

Report A, B, C and D accuracy, B−A, C−A, D−B, D−A, and
`(D−B)−(C−A)`, pooled and by participant/judge. Resample the 24 original story
families together across both participants and all cells. Report corrections
and harms for the paired differences. Keep real-minus-neutral display effects
separate from the raw four-cell comparisons. Full context, parse, stance and
score-disclosure checks accompany qualitative transcript review.

All paid work uses the existing process-locked `runs/pilot_budget.json` ledger.
The task rental has a fixed shutdown deadline and independent VPS watchdog.
Unrelated rentals are untouched. Generation, scoring and evaluation agents
coordinate through `runs/iteration4/board/`; the coordinator owns spending,
infrastructure, integration, review and publication.
