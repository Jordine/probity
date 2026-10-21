# Probity experiment cost plan — 16 October 2026

The useful next expansion is affordable: **about $185–210 for all 166 questions,
both Qwen 27B models and three sampled seeds**, including one full 2×2 grid,
Qwen sentence labels, validity diagnostics, shared six-layer probe extraction,
Haiku and native 4B judging, and a 25% reserve. Adding Opus annotation on a random
20% audit raises that to **$320–350**; independently annotating everything with
both Qwen and Opus raises it to **$875–900**.

These are new-campaign estimates, not spending authorization. No paid calls or
rentals were made for this forecast. Existing spending remains $73.776736 of the
previous $75 authorization. Historical artifacts are not automatically deducted:
changing the sampling protocol can make a fresh run preferable.

## What one grid costs

One question × participant-model configuration × seed × prompt condition needs
two six-turn debates: one without feedback for A/C, and one with feedback for
B/D. A and C share their transcript; B and D share theirs. Two private speaker
preparations are shared by both trajectories. The estimate resamples those
preparations for each seed.

For 166 questions × two models × three seeds, this is 996 matched units,
**1,992 distinct transcripts, 11,952 public turns and 3,984 final decisions per
judge**. The sample has 116 source-story families; its questions are correlated.
Only 28 families, containing 35 question rows, remain untouched by earlier work.
Using all 166 is an exploratory corpus expansion, not a fresh confirmatory test.

| Component, full 166-question / three-seed grid | Expected USD before reserve |
| --- | ---: |
| Two private preparations per matched unit + both debate trajectories | $56.23 |
| Prose/role and contextual-knowledge checks, source/question checks | $11.36 |
| Exhaustive sentence labels: Qwen alone | $43.17 |
| Exhaustive sentence labels: Qwen + Opus on random 20% | $153.65 |
| Exhaustive sentence labels: both on every transcript | $595.59 |
| Haiku final decisions, raw or response-level score display | $1.39 |
| Six-layer response/local/DYL extraction + live feedback | $31.37–51.54 |
| Native Qwen3.5-4B final decisions | $0.90–1.49 |
| One cold setup / model loading / transfer allowance | $1.41–2.15 |

The annotation rows are alternatives. Cheap validity diagnostics do not provide
local factual labels. Qwen-only labels are not dual consensus or human ground
truth; the 20% Opus option leaves the other 80% single-annotator. Select that
audit independently of probe scores, and report model disagreement separately.

## Marginal costs and reuse by axis

| Axis | Work that increases | What can be reused |
| --- | --- | --- |
| One to three sampled seeds | About 3× public generation, labels, scoring and decisions; preparations also 3× if resampled | Source/question checks and frozen probes |
| 24 selected questions to all 166 | 6.92× trajectories before length differences | Frozen probes and protocol; family-aware analysis remains necessary |
| Every problem in a story | One trajectory pair per question, even if the story repeats | Story text and compatible prefix caching; no cache discount assumed here |
| Four legacy prompt levels + current protocol | About 5× trajectories and their downstream work | Frozen probes; any byte-identical source controls |
| Additional probe heads, offline | Linear projections and head fitting; new views if a judge evaluates them | Same response/Yes/No forwards, labels and transcripts |
| One to six layers, offline | Layer hooks, pooled features and fits at compatible layers | Same base-model forwards; not 6× model inference |
| Response / sentence / clause / peak displays, judge only | New final-judge requests and display-specific token counts | Debates, factual labels and underlying scores |
| Changing any debater-visible probe/layer/display | A new feedback trajectory, annotation and decisions for each policy | Original no-feedback transcript and compatible private preparations |
| Additional final judge model | New final decisions only | All debates, annotations and scores |
| Real / neutral / shuffled judge displays | Roughly 3× scored-view decisions | Debates and labels; identical neutral requests can be deduplicated |
| Both answer orders × both opening speakers on every question | 4× trajectories; only 2× preparations when the opening order is the sole change | Compatible preparations and source checks |
| Mixed-model speaker pairings | Additional participant configurations and their trajectories | Source checks and same-model probe checkpoints, with correct speaker mapping |

There is exactly one supplied wrong option in each of the 166 binary questions;
there is no extra three-distractor multiplier. Balance answer/order assignments
across questions if exhaustive within-question counterbalancing is unnecessary.
Three seeds cannot exactly balance four assignment/order combinations per item.

The existing runs used temperature 0 and no explicit seed. Independent sampled
replicates require a frozen nonzero-temperature sampling protocol, explicit
seed where supported, and distinct generation provenance. Merely replaying a
cached request or changing a seed at temperature 0 does not establish variation.
Holding the private plans fixed across three seeds saves approximately **$18
before reserve** on the full 166-question grid, but measures public-debate
variation conditional on those plans.

## Annotation and final-judge unit prices

These amortized prices use the observed batch sizes and text lengths, repriced
at the current catalog rates without relying on cache discounts.

| Operation | USD |
| --- | ---: |
| Prose/assigned-answer check per transcript, including observed repairs | $0.00063 |
| Two contextual speaker-knowledge checks per transcript | $0.00473 |
| Source/question validity check, shared across its repeats | $0.00132 |
| Exhaustive sentence labels per transcript, Qwen alone | $0.02167 |
| Same, Opus alone | $0.27732 |
| Same, Qwen + Opus | $0.29899 |
| Opening-only clause labels, both models | $0.07142 |
| Haiku decision: raw / response / sentence / clause display | $0.000329 / $0.000370 / $0.000800 / $0.001135 |
| Opus decision: raw / response / sentence display | $0.01331 / $0.01485 / $0.03206 |

The existing sentence corpus contains 1,977 units across 48 transcripts. These
rates imply about **$0.53 per 1,000 units with Qwen or $7.26 with both** when
batching and text lengths are comparable. Human adjudication is additional.
Opening-clause labels cover one selected turn per debate; they are not full
six-turn clause coverage. Using the observed 771 annotated opening clauses and
5,598 all-turn display clauses gives a rough **$0.52/transcript** dual-annotation
proxy for exhaustive clauses, with substantial uncertainty about batching and
unit validity. Atomic factual claims need a pilot to establish their count and
label quality; do not treat mechanical clauses as equivalent to atomic claims.

The current 48 feedback transcripts still need exhaustive factual annotation.
At current rates, that is approximately **$1.04 Qwen-only, $3.70 with a random
20% Opus audit, or $14.35 dual**, before reserve. Labels on the original 48
transcripts are reusable if the same unit definition is retained.

## Probe and GPU costs

The existing all-feature extractor uses **two full forwards per public action**,
for the DYL Yes and No suffixes. The Yes pass also supplies response/local
features, so response + Yes + No is not three independent passes in this
implementation. Six layers and many linear heads share those forwards.

Measured selected-head scoring took 536 seconds for all 288 feedback actions.
Live generation also holds the GPU while API responses arrive. The forecast
therefore allows 24–48 payable GPU seconds per offline six-turn transcript and
60–90 seconds per live-feedback transcript for shared all-feature extraction.
Native final judging allows 0.6–1.0 seconds per decision, plus cold setup, for
individual response/sentence/clause views comparable to those already measured.
Substantially more verbose displays need a new timing measurement.
These are planning allowances, not an assertion that a full-scale run has been
benchmarked. A campaign is approximately 25–40 GPU-hours; API annotation may run
separately, and wall-clock completion depends on concurrency and provider limits.

The illustrative GPU rate is $1.35/hour. A read-only Vast search on 16 October
found 96 GB RTX PRO 6000 workstation offers around $1.28–1.41/hour with 180 GiB
disk included. Transfer is charged separately and offer availability varies.
See [Vast pricing](https://docs.vast.ai/guides/instances/pricing).
The setup allowance applies per rental. Very large campaigns spread across
multiple machines or interrupted rentals must budget that setup again.

Eight exported score/display bundles exist, seven of which were judged in
iteration 3. Sentence and clause bundles can use the same local head. These
cover adapted response, Apollo response, Apollo DYL difference, new local,
previous local and published FIBS; the No-only Apollo DYL diagnostic is also
available in response evaluation. **FIBS is model-matched only for Qwen3.5.**
Retained local-head searches have 32 candidates/model across four layers;
arbitrary six-layer combinations are not all ready or validated. Published
weights cannot simply be moved to another layer or model. Additional linear
fits can reuse compact training features; new training examples and factual
labels cost extra under the same unit rates above.

Stream score projections and pooled span features. A naive six-layer raw
response-activation cache would be roughly 202 GB for this one 1,992-transcript
campaign. The forecast does not assume saving those tensors to the VPS.
Full-corpus story lengths average 4,943 tokens versus 5,422 in the selected 24;
the forecast keeps the observed longer-context average rather than claiming
an unmeasured length discount. Maximum complete contexts still need explicit
checks in the actual run; expanded prompts must never silently truncate.

## Campaign choices

All ranges below include 25% reserve, current-price generation with fresh private
plans per seed, validity checks, shared feature extraction, and Haiku + native
4B final judging. They exclude human labor and new training-corpus development.

| Campaign | Distinct transcripts | Planned USD |
| --- | ---: | ---: |
| Current 24 questions, three seeds, one prompt; 20% Opus audit | 288 | $48–53 |
| Untouched 35 questions / 28 families, three seeds; 20% Opus audit | 420 | $69–76 |
| All 166, three seeds, one prompt; validity only | 1,992 | $130–160 |
| Same, exhaustive Qwen sentence labels | 1,992 | $185–210 |
| Same, Qwen labels + random 20% Opus audit | 1,992 | $320–350 |
| Same, dual sentence labels throughout | 1,992 | $875–900 |
| All 166 × three seeds × five prompt regimes; 20% Opus audit | 9,960 | $1,590–1,725 |
| Same five-prompt campaign, dual sentence labels | 9,960 | $4,350–4,490 |
| One prompt, 108 offline judge views on both transcript arms; 20% Opus audit | 1,992 | $585–655 |
| Five prompts, 108 distinct feedback policies; 20% Opus audit | 542,820 | $83,000–91,100 |
| Same full feedback sweep, dual sentence labels | 542,820 | $233,500–241,600 |

The 108-view illustration means six candidate probe configurations × six
layers × three displays. It is a hypothetical compatible-head sweep, not a claim
that all those checkpoints/displays exist today. It excludes incompatible FIBS
combinations. The feedback version uses a matched judge display per policy;
crossing all 108 judge views onto every feedback transcript would cost more.
It also extrapolates current feedback text lengths: verbose feedback policies
could increase generation and GPU costs beyond this workload allowance.

Adding Opus as a final decision judge to the ordinary full grid adds about
**$70 with reserve** for raw/response displays. Its cost grows with dense local
displays. Adding it to the illustrative 108-policy/five-prompt sweep brings the
20%-audit campaign to approximately **$133,500–141,500**, or
**$284,000–292,000** with exhaustive dual annotation. Opus clause-judge costs
are inferred from observed clause token lengths and its sentence-output length;
no Opus clause-judge run has been measured.

## Counting rule and recommended sequence

Let `N = questions × participant configurations × seeds × prompt regimes`.
Let `F` be distinct debater-feedback policies. Then generated transcripts are
`N × (1 + F)`, and compatible shared preparations cost `2N` requests.
If `C` distinct judge displays are matched to those policies, a judge needs
`N × (1 + C + 2F)` decisions. With a distinct display for each policy, this is
`N × (1 + 3F)`. Crossing `S` non-baseline judge views onto every transcript costs
`N × (1 + F) × (1 + S)` decisions per judge. Add judge replicates and control
views separately; reuse byte-identical neutral displays where appropriate.

Start by finishing factual labels on the existing feedback transcripts and
checking local probe validity. Use consumed families to narrow prompt variants,
layer choices and display methods; fix the legacy `explicit_v2` routing issue
before calling a sweep faithful to the original levels. Run the broad probe
comparison offline on shared transcripts, then carry only a few selected probe
and display policies into live feedback. Preserve untouched families for a
frozen confirmatory comparison. This sequence answers the main research
questions without paying for the enormous full feedback cross-product.

## Evidence and scope of the estimate

The machine-readable [cost forecast](../reports/costs/probity_cost_forecast_v1.json)
contains unit prices, exact scenario counts, cost components and evidence
hashes. Private audit files retain the full receipt accounting. Generation
accounting deduplicates 817 receipt rows to 783 actual attempts and separates
two unverified timeout liabilities from provider-reported charges. Annotation
prices include observed failed requests and repairs; the reserve also allows
for future retries, routing variation and changed output lengths.

Current [OpenRouter catalog](https://openrouter.ai/api/v1/models) input/output
rates per million tokens are Qwen3.5-27B $0.26/$2.60, Qwen3.8-27B $0.24/$2.20,
Haiku5.5 $0.10/$0.50 and Opus5.5 $4/$20. Historical Qwen routes were often
cheaper. The estimate deliberately uses the recorded token volumes at these
catalog rates, not the cheapest advertised quantized route. Provider execution
and native bf16 replay are not assumed equivalent. API credit-purchase fees,
tax, human review, implementation time and open-ended probe hillclimbing are
excluded. Ranges are planning allowances, not statistical confidence intervals.
