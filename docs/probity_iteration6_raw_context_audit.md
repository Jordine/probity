# Qwen3.5 debate context and raw-record audit

The completed iteration5 Qwen3.5-27B subset has **166 questions, 116 source-story families, 332 A/B trajectories, and 1,992 public turns**. Every retained public turn has its exact caller-submitted request, including the system and user messages, and its complete raw provider response body. All 332 private preparation calls have the same records, including provider-exposed reasoning.

This is stronger than preserving arguments alone. It is not a claim that API serving internals, hidden computation, or the provider's final tokenized prompt are recoverable.

## What happens during a debate

1. Each speaker privately prepares a plan using the full authentic story, the question, both answer options, the verified correct answer, and its assigned answer. The false-answer speaker is explicitly instructed to invent a coherent factual account. The plan contains three claim/quotation pairs plus an anticipated challenge and response. The two private plans are identical between the matched A and B trajectories.
2. Each public turn is a **fresh two-message API request**: a system instruction and a user message. The user message includes the entire authentic story, question/options, that speaker's own private plan, and all prior public arguments. Previous arguments are rendered as text inside the user message; this is not a persistent sequence of assistant-role messages.
3. In A, the speaker generates its next argument using that context.
4. In B, the system message additionally explains the feedback protocol. The user message ends with `BEGIN PRIVATE PAST-TURN PROBE FEEDBACK`, an explanation of the signal, and a JSON packet. The packet contains completed public turns' response percentiles, running means by speaker, and an optional lower-mean-speaker answer recommendation. This is appended text in the existing user message, rather than an extra message with a new role.
5. After each B public turn, the generated response is scored before the next turn may proceed. The next speaker sees the new score along with earlier scores. Therefore the first public turn receives an empty packet; the second public turn already receives one score. Feedback is updated **after each turn**, not only after a two-speaker round.
6. Each advocate sees scores for both speakers' prior public arguments, and sees only its own private plan. Each B advocate is instructed to keep the feedback private. Judges receive the public arguments, question and options; C/D additionally receive the final packet. The judges do not receive the story or private plans.

The historical B prompt also told advocates from the first turn that feedback would arrive. A/B therefore differ before any numeric feedback is available, as well as in later feedback and their separately generated text.

## Audit results

The offline audit reads canonical episode paths bound by the completed iteration5 judge package. It joins each turn to its accepted receipt by request hash, validates the request hash and provider response, reconstructs the expected prompt from the source and historical prior turns, and requires exact message equality. For B, it reconstructs the packet and validates the causal chain of prior request hashes. It also joins every private plan to its original preparation receipt and reconstructs its exact prompt.

| Check | Verified |
| --- | ---: |
| A exact public requests and raw responses | 996/996 |
| B exact public requests and raw responses | 996/996 |
| B packets matched to historical prior turns | 996/996 |
| B packets containing at least one score | 830/996 |
| B packets excluding current/future turns | 996/996 |
| Unique private preparation requests and responses | 332/332 |
| Paired A/B units reusing identical private plans | 166/166 |
| Public requests with exactly `system,user` roles | 1,992/1,992 |

The packet exclusion check verifies which turns supplied its scores. It is not an independent detector-quality test. The exact prompt check covers full story text, private preparation, public history, and feedback; no missing caller-visible context was found.

Every public request used `temperature=0`, `max_tokens=8192`, and `reasoning.enabled=false`; no seed was supplied. Every preparation request used `temperature=0`, `max_tokens=16384`, and `reasoning.enabled=true`.

**Thinking needs a precise distinction.** All 332 preparation responses contain nonempty provider-exposed `reasoning` and `reasoning_details`; the original response bodies retain them. All 1,992 public-turn responses have absent or empty reasoning fields and report zero reasoning tokens. There is therefore no public-turn thinking trace to export. The structured private plan is supplied to public generation; the preparation's separate reasoning trace is not appended to the public request.

Providers varied between arms: A used DeepInfra for 715 turns, Alibaba for 91, and SiliconFlow for 190; B used DeepInfra for 637, Alibaba for 242, and SiliconFlow for 117. Exact request parameters and returned provider/model/usage fields remain in the receipts. This operational routing difference was not randomized.

## Inspect or reproduce

Run the audit from the repository root:

```bash
python3 runs/iteration6/raw_audit/audit_raw_context.py
```

This performs no model calls and no rentals, and does not modify iteration5. Outputs are private run artifacts:

- [Complete raw inspector](../runs/iteration6/raw_audit/index.html): two matched Qwen3.5 examples, one original pilot and one new iteration5 question; expand each private preparation or public turn to see the complete request and provider response.
- [Raw example dossiers](../runs/iteration6/raw_audit/representative_raw_dossiers.json): the same four trajectories, machine readable.
- [All-turn receipt index](../runs/iteration6/raw_audit/turn_index.jsonl): all 1,992 public turns with original receipt file and JSONL line where applicable.
- [Audit summary](../runs/iteration6/raw_audit/summary.json): counts, settings, provider counts and reasoning coverage.
- [Input/output hash manifest](../runs/iteration6/raw_audit/manifest.json).
- [Audit implementation](../runs/iteration6/raw_audit/audit_raw_context.py).

Canonical receipt locations are `runs/iteration5/debates_v4/requests.jsonl`, `runs/iteration5/feedback_v1/requests/`, and `runs/iteration5/preparations_v9/requests.jsonl`. The 24 reused pilot questions use `runs/iteration3/final/debates_v1/requests.jsonl`, `runs/iteration4/feedback_v1/requests/`, and `runs/iteration3/final/preparations_v3/requests.jsonl`. Historical iterations and failed attempts remain separate from the accepted receipts indexed here.

The pre-existing [iteration5 inspector](../runs/iteration5/reader_reviews/inspection_pack_v5/index.html) also contains exact compiled judge messages, API judge receipts, and native judge results for its fixed four-unit sample. The native judge's final serialized chat-template text was not retained there; compiled messages are retained.

## What replay can establish

An already completed B debate can be **rescored** under a different detector or layer using its authentic historical messages and fixed generated response. Its prompts must retain the original feedback, because that is the context in which that response was produced. Such a replay measures the new detector's scores on a historically feedback-conditioned corpus. It cannot show how debaters would have responded to the new detector.

Changing the live probe, displayed packet or feedback instruction requires generating a new continuation from the first affected prompt. Past prefixes may be cached only while they are truly identical. Existing A/B text can still support controlled final-judge display experiments, since those displays occur after generation is over.

Exact request reconstruction also does not guarantee exact regeneration. Hosted provider tokenization/templates, internal prompts, revisions, kernels, and hidden reasoning were not exposed, and the request omitted an explicit seed. Native GPU rescoring is a documented reconstruction using the local model and chat template, not proof that the hosted provider used identical internal activations.

## Complete receipts can still contain unfinished prose

A later sentence-ending screen found 14 of the 1,992 historical public arguments
ending in an incomplete clause or quotation introduction. Inspection of all 14
raw envelopes verified that the stored argument exactly matches the provider's
content. Every call reports normal `stop` and uses only 57–233 completion tokens,
below the 8,192-token allowance. Both DeepInfra and SiliconFlow appear. These
are retained generated-prose defects, not parser clipping or evidence that the
saved context was truncated.

The detailed evidence is in
`runs/iteration6/raw_audit/historical_incomplete_prose_audit_v1.json`. The screen
uses absence of final sentence punctuation to find candidates; that heuristic
alone does not prove incompleteness. The listed cases were also read in context.
No original argument or label was silently repaired.

The new display experiment exposed the same kind of ending immediately before
a quotation. In a separate four-call diagnostic on two selected affected
contexts, both fresh strict-schema calls reproduced the unfinished ending and
both calls with `response_format` omitted returned complete arguments. All four
reported DeepInfra and normal termination; even the unconstrained calls returned
valid JSON. This supports an output-format-associated failure in those selected
contexts. It does not reveal provider internals or estimate the population-wide
failure rate. Exact requests and responses are retained in
`runs/iteration6/live/output_format_diagnostic_v1/`. Production branches remain
unchanged.
