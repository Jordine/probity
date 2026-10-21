# Reading the Qwen3.5 probe and feedback experiments

This campaign separates three questions: **does the probe detect factual errors,
does feedback change what advocates say, and does that help the final judge?**
Those require different measurements. A higher score for the advocate assigned
the wrong answer does not by itself show that a detector can locate false claims.

The campaign is complete. It spent $30.20 under its $60 total ceiling, and all
owned GPU rentals are released. This guide describes the frozen design and
completed outputs. Intermediate plots explicitly marked as pipeline checks are
not results.

## Which debates belong to which experiment?

| Experiment | Data and generation | What the comparison can answer |
| --- | --- | --- |
| Historical A/B behavior and factuality | 166 matched questions from 116 source-story families; 332 six-turn Qwen3.5-27B debates | How the existing no-feedback and feedback trajectories differ |
| Historical probe comparison | All 1,992 saved public turns, with their actual original contexts | Which frozen readouts discriminate source-supported versus source-contradicted sentences on these observed debates |
| New participant-display screen | 24 families; one question each; seven conditions; 168 debates sharing their first two turns within question | How different private displays change four newly generated continuation turns |
| New instruction screen | Eight families; selective/explicit instructions × feedback off/on; 32 new six-turn debates | How these instruction blocks and feedback interact in a small exploratory screen |

The 166 figure counts **questions**, not independent stories. Several questions
come from the same source family. Related questions remain together in analysis
and resampling. All families have earlier experimental exposure; these results
must not be described as a new untouched test set.

The original repository's QualitySyn input supplies these questions. The current
campaign inherits five documented earlier source repairs; it does not generate
a new story dataset. The selected 24 and eight families are subsets of that
corpus, chosen by fixed hashes rather than current outcomes.

## Where this fits in the two-by-two grid

The earlier grid crosses feedback to debaters with scores shown to the judge:

| Debater input | Judge gets public arguments only | Judge also gets scores |
| --- | --- | --- |
| No private score feedback | A | C: same arguments as A |
| Private past-turn score feedback | B | D: same arguments as B |

The new display and instruction screens study the **left column**. Their final
judges receive no score packet. Here “display” means how scores are privately
presented to the advocates. It does not mean a changed display to the judge.
The historical A/B/C/D results remain available separately; their C−A and D−B
contrasts can reuse identical arguments because the intervention occurs after
generation.

Changing a live feedback condition requires new generation from the point where
that condition first changes the advocate's input. Offline rescoring preserves
the observed context, including historical feedback. It cannot produce the
debate that would have occurred under another probe.

## What an advocate actually sees

An advocate receives the authentic story, question and options, its assigned
answer, its own private prepared account, and the preceding public arguments.
In the historical/display harness, the wrong-answer advocate is instructed to
prepare a coherent fabricated account itself. The new instruction screen instead
uses the selected original instruction block without that inherited fabrication
mandate; a plan may selectively quote or reframe rather than invent facts. No
condition supplies an externally written alternate story. Each public call is an explicit
system/user request, not an opaque persistent chat.

Historical B announced feedback availability from turn one and supplied the
previous turn's score before turn two. The new display experiment instead uses
the same original no-feedback first round in every condition. Its intervention
begins before turn three. The new instruction experiment generates all six
turns afresh, sharing each newly prepared plan between its feedback-on/off pair.
Its on/off first rounds are separate API generations, not an identical copied
prefix. Numerical feedback and the feedback instruction first appear at turn 3,
after one complete round.

In the prompt screen, the correct-answer advocate always receives the honest
instruction. The wrong-answer advocate receives either the selective block,
which permits selective interpretation and downplaying, or the explicit block,
which also permits out-of-context quotation, misdirection, and a
win-at-all-costs stance. Neither block supplies a detailed alternate story.
This instruction × feedback screen is a different 2×2 from A/B/C/D. Its eight
families are a subset of the 24 display families, so the two screens are
correlated exploratory analyses rather than independent replications.

The seven private display conditions are no feedback, neutral formatting, the
full score bundle, past-turn scores only, speaker means only, an answer
recommendation only, and a shuffled-score bundle. The shuffled condition is a
weak control: 17/24 initial packets are unchanged. Across all 96 newly supplied
shuffled feedback packets, 23 retain identical numerical values and 47 retain the
unshuffled answer recommendation for their own prior history. Shuffling can
preserve useful aggregate information; it does not guarantee removal of the signal.

All live conditions use the same frozen adapted response probe. Alternative
Apollo-trained, DYL, published FIBS and local heads are evaluated offline on the
recorded behavior. This campaign does **not** generate separate live debates
for every probe type or layer.

## How factuality and probe plots should be read

The primary reader labels fixed sentences against the authentic source. It
does not see assigned advocacy roles, probe values, condition metadata or judge
outcomes. Unsupported, ambiguous and invalid assessments remain distinct from
supported and contradicted labels; excluded sentences are never treated as
truthful negatives. These are automated source-factuality judgments, not proof
of deceptive intent or a human gold standard.

Qwen3.8-27B supplies the primary factuality labels; it is not a debater in this
campaign. A separate Opus reader audits a frozen 12-family sample. Agreement analysis
uses those families. Other legacy dual-reader labels are convenient additional
coverage; pooling them into a consensus tier does not make that tier
representative of the whole corpus.

Across-sentence AUROC gives each source family equal total weight in the
eligible population. Within-argument AUROC uses only arguments containing both
supported and contradicted sentences, then averages arguments within family
and families equally. Read its smaller argument/family denominator. A score
that is constant throughout an argument has no ability to locate a particular
false sentence even if it separates roles or whole arguments well.

Confidence intervals resample source families. They do not include uncertainty
from repeated generation seeds, annotator mistakes or provider variation.
Layer/head comparisons are exploratory; selecting the highest current score
does not create a separately confirmed winner. Probe signs remain frozen.

## Raw evidence and reproduction limits

The historical [raw-context audit](probity_iteration6_raw_context_audit.md)
verifies all 1,992 public calls and 332 private preparations. New generation
receipts retain the full submitted request, feedback, provider response and
request/response hashes. Private preparation requests enable reasoning and
retain provider-returned reasoning. Public calls disable reasoning. Unreturned
provider internals cannot be reconstructed from these records.

The [live-screen raw-context supplement](probity_iteration6_live_context_audit.md)
separately verifies the display and prompt receipt populations and explains
canonical versus transported requests during provider-route recovery.

Public arguments are generated through the hosted Qwen API. A rented GPU then
scores the exact saved request and returned text with a pinned local Qwen3.5-27B
checkpoint before the next live action receives its feedback. This is live
feedback, but the activations come from the local scoring forward, not from the
provider's private inference process. Provider-internal weight revisions,
quantization and serialization are not independently verified.

Native probe replay checks full tokenization, offsets, model revision and head
provenance, and refuses truncation. DYL forwards force `Yes.` and `No.`; the
frozen readout is the final period immediately before the assistant end token,
not the Yes/No word or the end token. Token positions are recorded per turn. Every historical input passed the token
preflight; the largest was 11,036 tokens. Saved outputs contain scalar scores,
token IDs and offsets, not raw activation tensors. Model weights were loaded on
ephemeral GPU rentals, which have been released.

The shared-forward scorer matched the original scorer exactly on the current
GPU in a 12-case check. The complete 1,992-turn replay preserves the historical displayed percentile in
1,980 turns; 12 change by at most 2.084 percentile points. Raw scores match
exactly in 1,556 turns, with maximum absolute difference 0.01504. Per-turn raw
and displayed differences are retained in the [replay diagnostic](../reports/data/iteration6/replay_diagnostic_v1/summary.json).
This is not a claim of bit-identical cross-run or API/native reproduction.

Normal provider termination does not guarantee complete prose. Fourteen of the
1,992 historical arguments and 12 of the 672 new display suffix arguments end
with unfinished clauses or quote introductions in the raw provider text. All
12 new cases belong to one family. The primary display analysis retains them;
a post-hoc sensitivity excludes that entire family across all seven conditions.
In a four-call diagnostic on two selected affected contexts, both strict-schema
requests reproduced an unfinished ending and both otherwise identical requests
without `response_format` completed. This supports a format-associated failure
for those contexts, not a population rate or a proven provider mechanism.
Production outputs are preserved rather than replaced.

## What remains outside this campaign

There is no new local-probe training, full live probe-type × layer × display ×
prompt factorial, multi-seed expansion, external false-account condition, or
corpus-wide semantic interaction labeling. Qualitative examples can illustrate
concessions or persistent fabricated quotations, but source-factuality labels
alone do not measure those conversational behaviors.

The [specification](probity_iteration6_spec.md) and
[probe inventory](probity_iteration6_ablation_inventory.md) provide exact scope
and artifact provenance. The [historical behavior report](probity_iteration6_behavior.md)
is already complete and is separate from the new continuation results.

## Available readings

- [Canonical results narrative](probity_iteration6_results.md) and
  [figure/artifact index](../reports/data/iteration6/README.md).
- [Historical factuality and reader agreement](probity_iteration6_factuality.md):
  complete source-based labeling, paired A/B estimates and annotation limitations.
- [Completed display behavior](../reports/data/iteration6/display_behavior_v1/interpretation.md):
  no-feedback and neutral controls, illustrative transcript readings and unfinished-prose caveat.
- [Full display condition table](../reports/data/iteration6/display_behavior_v1/report.md)
  and [behavior plot](../reports/data/iteration6/display_behavior_v1/display_behavior.png).
- [Fresh display factuality](../reports/data/iteration6/live_factuality_v1/live_factuality.png),
  [probe/layer benchmark](../reports/data/iteration6/probe_factuality_v2/probe_factuality.png),
  [local-control comparison](../reports/data/iteration6/local_control_comparison_v1/local_control_differences.png),
  and [paired final judges](../reports/data/iteration6/live_judges_v1/report.md).
- [Prompt behavior](../reports/data/iteration6/prompt_behavior_v1/paired_prompt_effects.png)
  and [whole-family prose sensitivity](../reports/data/iteration6/prompt_prose_sensitivity_v4/prompt_sensitivity.png).
- The local [display raw-context browser](../runs/iteration6/live/inspection_v1/index.html)
  shows every policy for three fixed-hash families. Full raw archives remain on
  the experiment host; these local archive links do not resolve in a public Git checkout.
- The local [prompt raw-context browser](../runs/iteration6/live/prompt_inspection_v2/index.html)
  shows all four cells for two fixed-manifest families, including private plans,
  provider-returned preparation reasoning, every public request/response,
  feedback and score. `prompt8_v1/export_v1` is a judge/analysis episode view,
  not the canonical raw archive.
