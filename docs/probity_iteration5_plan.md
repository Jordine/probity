# Extend dataset coverage with the existing feedback grid

The user requested on 16 October 2026: “Lets do all 166 families first on what
you’ve alr run (so we are only extending this one axis) — seeds and further can
be added later. Note this estimation and save the plans — as well as provide
options for what we should include in the paper.”

The corpus contains **166 questions from 116 original story families**. This
campaign extends question coverage; it does not create 166 independent families.
The previous [cost estimates](probity_cost_plan.md) and
[machine-readable forecasts](../reports/costs/probity_cost_forecast_v1.json)
remain saved. Later seed, prompt, probe, display and judge sweeps are deferred.

## Fixed experiment

Use the completed [iteration-4 protocol](probity_iteration4_protocol_scope.md):
Qwen3.5-27B versus itself and Qwen3.8-27B versus itself; one run per question;
temperature zero; thinking-enabled private evidence plans; three sequential
public rounds with public thinking disabled; and the same explicit truth and
fabricated-evidence instructions. Preserve the 16,384-token private preparation
and 8,192-token public completion allowances, complete contexts and stop checks.

Use the frozen adapted response-mean head at layer 43 for Qwen3.5 and layer 35
for Qwen3.8, with the existing development reference values. Keep the existing
whole-response percentile, speaker-mean and recommendation packet. A/C share
the no-feedback transcript; B/D share the live-feedback transcript. Final
judges remain Haiku5.5 and native bf16 Qwen3.5-4B under their original answer-only
contracts. No probe fitting, recalibration or new local-truth labeling sweep is
part of this campaign.

Reuse the 24 repaired pilot questions, their exact option/opening schedules,
private plans, 48 no-feedback debates, 48 feedback debates and matching scores
and judgments. New rows receive a deterministic schedule frozen before model
outcomes are inspected. Keep source-question repairs and their provenance
explicit; never silently replace an original source or discard a hard question.

The source freeze preserves the five accepted pilot repairs and uses the raw
model-input fields for the other 142 questions. Seven additional questions have
historical alternative repairs; these remain recorded validity flags rather
than new source or gold changes in this dataset-only extension. Run the same
outcome-blind complete-source ambiguity diagnostic, then report provided-key
results alongside a source-validity sensitivity analysis. The source reviewer
is a diagnostic, not independent ground truth.

Target coverage is **664 distinct six-turn debates and 1,328 final decisions per
judge**. Exact pilot reuse leaves **568 new debates, 3,408 new public turns,
568 new private preparation calls and 1,136 new decisions per judge**.

## Inclusion and analysis

Record historical exposure for every original family: probe training/reference,
other development or tuning, the previous 24-family evaluation, and the 28
previously untouched families. Multiple questions from a family remain together
through every analysis. Do not label the entire expanded set a fresh test set.

The all-question expansion retains diagnostic failures as outcomes. In
particular, an incorrect initial source-knowledge answer or a public role switch
does not trigger a replacement sample, a changed prompt or silent deletion.
The authentic answer remains explicitly supplied in the unchanged private and
public prompts. The old 24 questions passed the former knowledge/answer-field
eligibility gates; retaining failures on new questions changes inclusion, not
their prompt treatment. Report an eligibility-matched secondary analysis as
well as the complete-corpus descriptive result. Invalid/truncated API responses
remain technical failures with complete attempts and bounded identical-request
recovery, never successful observations.

Freeze all settings before generating or judging the untouched subset. Report
that subset separately, together with exposed-family results and all-corpus
descriptive results. Keep the same A, B, C and D definitions, paired B−A, C−A,
D−B, D−A and interaction `(D−B)−(C−A)`. Use family-macro estimates with matching
family-bootstrap intervals, and label question-weighted accuracy separately.
Include paired fixes/harms, role switches, feedback disclosure, source ambiguity,
parse/completion failures and qualitative transcript examples.

The versioned expansion adapters allow multiple questions per family and
explicitly exposed rows. They preserve the old experiment’s source and checks;
they must not bypass contamination guards by pretending calibration families
are new test families. Publish the expansion’s new inclusion policy explicitly.

## Execution and costs

The saved receipt-derived model estimates **$34.48–39.61 additional** for the
142 new questions under this narrower one-seed, selected-head protocol. This
includes preparation/generation, validity diagnostics, final judging, setup,
GPU/API waiting and 25% contingency. The internal ceiling for this authorized
dataset-only campaign is **$45**. It is a scope-limited execution allowance,
not authorization for later experimental axes. The previous pilot ledger and
its $73.776736 conservative spending record remain unchanged; new spending is
recorded separately in `runs/iteration5/budget.json` and combined in reports.

1. Freeze source, exposure, schedule, reuse and request-runtime manifests.
2. Complete source diagnostics, new private plans and no-feedback debates on
   the VPS before renting a GPU. Prepare feedback openings in advance.
3. Rent one suitable 96 GB GPU with a bounded price/runtime reservation and an
   independent shutdown watchdog. Verify old score parity first. Process each
   participant model while it is loaded, then run the native 4B judge.
4. Retain scalar scores, complete messages/receipts, hashes, compact features
   where necessary, and source/transcript provenance. Do not save new raw
   activation tensors to the VPS. Check every full context without truncation.
5. Finish API judging and validity checks, audit paired coverage and exact reuse,
   inspect transcripts, produce plots and report exposed/untouched strata.
6. Verify rental destruction and reconcile actual costs before publication.

The expected additional GPU work is about 5–8 payable hours including setup.
The scheduler should keep GPU waiting low by completing independent API work
first. The exact rental offer, rate, deadline and reserve will be recorded before
creation. A failure to fit the execution ceiling pauses new paid calls rather
than silently increasing the budget.

See [paper options](probity_paper_options.md) for claims, figures, current
limitations and evidence that would justify a broader paper later.

The frozen source/inclusion inventory SHA256 is
`06a4e542fba39a187638e5dbddfd93fd0af744c35431f940accdf4f7bc2567b5`.
Its private sources and complete receipts remain in the versioned iteration-5
run directory; the public plan does not expose source text or credentials.

## Transport recovery, 16 October 2026

The first preparation batch stopped at 18:19 UTC after one request reached its
1,800-second transport timeout without returning a response. It retained 135
accepted private plans; 432 requests had not started. All 142 complete-source
diagnostics had finished. No debate generation, GPU rental or final judging had
started. Accounted spending was $1.463401471, including the unanswered request's
full $0.04767828 cost reservation.

Continue the 433 missing plans in separate, versioned batches. Verify all
request bodies and canonical job identities against the frozen source and
request contract before making paid calls, and reuse every accepted response
exactly. A changed catalog price ceiling must not silently cause a cache miss
and regenerate an accepted plan. Preserve original attempts, stopped-run
markers, charges and receipts.

Allow at most six new recovery batches and three transport attempts per missing
request, including previous attempts, within the original $11 total preparation
envelope and $45 campaign ceiling. Continue only after each preceding batch
has drained and its liability is accounted for. Schema, completion, context,
source and budget failures still require review. The 1,800-second timeout and
all model, prompt, sampling and token settings remain unchanged. These are
technical continuations, not additional seeds or independent observations;
report their attempts and missingness in the final analysis.

## Reviewed preparation recovery, 16 October 2026

Preparation stopped again with 316 of 568 new plans accepted. All five earlier
transport timeouts had accepted identical-request retries. One later Alibaba
response reported 82,150 completion tokens: 81,920 thinking tokens and 230 answer
tokens. The unchanged runner rejected it for exceeding its cost reservation;
it would also fail the existing combined-completion token check. The original
response, invalid status and full $0.128880375 charge are retained.

Alibaba documents `max_tokens` as an answer-token limit for this model, with a
separate thinking budget. This differs from the runner's combined-token
assumption; the receipt alone does not establish a provider contract violation.
See the [Alibaba API reference](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)
and [OpenRouter reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
The request remains out of protocol and is not an accepted observation.

An explicit review permits an identical-request technical retry of that one
recorded failure, retaining all 316 accepted plans exactly. No general exception
for over-limit responses is introduced: another nontransport failure stops for
review. Across the entire history, permit at most three actual paid attempts per
job and six recovery batches after the original batch. Report first-attempt
failure, retries, and eventual missingness separately; accepted retries must not
be described as a sample without retry selection.

Five unanswered requests also have unresolved thinking-token costs. Preserve
their original allowances and add $0.638976 of conservative liability per
request, or $3.19488 total. This is an allowance for unknown charges, not measured
spending. A later successful retry does not resolve the earlier request's cost.
Before resumption, recorded costs plus these additional allowances are
$6.812461936 for preparation and $6.981508436 for the campaign.

Reserve every new preparation attempt against its prompt byte bound and the
frozen model's global context ceiling as a conservative billed-output bound:
262,144 tokens for Qwen3.5 and 1,000,000 for Qwen3.8, at the unchanged sent price
ceilings. Admit a request only when recorded costs, unresolved liabilities and
all concurrent requests fit the remaining original $11 preparation envelope.
Retain unknown-call allowances after a drained batch; incomplete drainage keeps
the full batch reservation. These bounds assume the service respects the sent
price ceilings and published model context limits.

Dispatch unfinished Qwen3.8 preparations before Qwen3.5 preparations so that
large per-request reservations are feasible while more headroom remains.
Concurrency may fall below four when the financial gate requires it. This
changes operational scheduling only: prompts, models, request bodies, sampling,
the existing acceptance checks and the $45 campaign ceiling remain fixed.
Coverage within the preparation envelope is not guaranteed; an unaffordable
next request stops before being sent. Controlling provider-specific thinking
budgets explicitly is a future protocol change, outside this dataset expansion.
