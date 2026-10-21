# Probity iteration 6: probe feedback in Qwen3.5-27B debates

Iteration 6 is complete. It separates three questions that had previously been
easy to conflate: whether a readout distinguishes source-supported from
source-contradicted text, whether showing that readout changes the debate, and
whether the changed debate helps a final judge. The campaign spent **$30.20**
of its authorized **$60** ceiling. All owned GPU rentals are released.

## What was run

| Study | Paired unit | Generated debates or evaluated turns | Intervention |
| --- | --- | ---: | --- |
| Historical A/B | 166 questions in 116 source-story families | 332 debates; 1,992 turns | Historical feedback off/on |
| Historical probe benchmark | 116 source-story families | 13,029 labeled sentences from all 1,992 turns | 57 frozen readouts, including controls and layer views |
| Display screen | 24 families | 168 debates; 672 new suffix turns | Seven private feedback displays |
| Prompt screen | 8 families | 32 fully new debates; 192 turns | Selective/explicit defense instruction × feedback off/on |
| Final judges | Same 200 new transcripts | 200 Haiku and 200 native Qwen3.5-4B decisions | Public arguments only; no probe scores |

The **166** count is the number of binary questions in the current QualitySyn
expansion, after five inherited documented source repairs. Those questions come
from **116** source stories and are analyzed and resampled by source family. The
older **14-family** number is the subset of mixed-label responses in an
iteration-3 local-clause analysis, not the full dataset or an alternative main
corpus denominator.

## Main results

In the historical A/B corpus, feedback-conditioned debates have a lower
source-contradiction fraction among reader-eligible sentences. The family-weighted conditional contradiction rate is
41.61% without feedback and 38.95% with feedback, a **−2.66 percentage-point**
difference (95% paired family-bootstrap interval −4.72 to −0.55). The change is
concentrated among advocates assigned the wrong answer: **−5.94 points**
(−10.33 to −1.86); the correct-answer advocates change by +0.17 points
(−2.56 to +2.98). This historical contrast also changes the opening instruction
and regenerates the full debate, so it does not isolate the numeric packet.

The probe benchmark shows why argument-level and local detection must remain
separate. Broadcast DYL/FIBS-style scores have high pooled sentence AUROC
(roughly 0.93–0.94), but they are constant within an argument and therefore have
within-argument AUROC 0.5 by construction. The frozen local3 head reaches
**0.814 within-argument AUROC** across 378 mixed-label arguments in 83 families
(0.771–0.857). It exceeds a development-selected random-direction control by
**0.109** (0.050–0.167). This is a strong exploratory local result on previously
exposed automated labels, not a clean held-out or human-gold estimate.

The 24-family display screen does not identify a robust winning presentation.
Relative to no feedback, estimated changes in the contradicted fraction among
reader-eligible sentences range from −0.8 to −5.2 points across the five
substantive or shuffled displays, and every interval includes zero. Eligibility
coverage also varies by condition and is reported beside this conditional
outcome. Relative to the neutral-feedback control, the full bundle changes the
conditional contradicted fraction by +0.20 points (−4.43 to +5.23). Several displays also shorten
or reduce lexical repetition, but neutral feedback alone causes some of those
changes. The final-judge result is likewise model-dependent: full-bundle
feedback changes Haiku accuracy from 19/24 to 21/24 (**+8.3 points**, 0 to
20.8), while native Qwen3.5-4B remains 18/24 (**0 points**, −16.7 to +16.7).

The eight-family prompt screen is smaller and more exploratory. For both the
selective and explicit original instruction blocks, feedback changes Haiku
accuracy from 7/8 to 8/8 (**+12.5 points**, 0 to 37.5). Native Qwen3.5-4B is
unchanged at 7/8 under explicit instructions and falls from 7/8 to 5/8 under
selective instructions (**−25 points**, −62.5 to 0). Feedback raises the
incorrect-role mean probe percentile by 4.69 points under selective instructions
(0.78 to 9.11) and 1.39 under explicit instructions (−1.48 to 3.99). One
feedback-on argument ends in an unfinished clause. Excluding its entire family
leaves the qualitative behavior conclusions unchanged; the retained sample is
seven families.

The complete prompt 2×2 also compares the wrong-advocate instruction axis.
Haiku's explicit-minus-selective effect is 0/8 under feedback and is formed by
one fix and one harm without feedback; its interaction is therefore zero on
average. Native accuracy is identical across instructions without feedback,
while explicit exceeds selective by 2/8 with feedback, producing a +25-point
interaction. These empirical bootstrap contrasts rest on eight families; the
selective judge divergence involves three changed family outcomes across the
two judges and should be treated as a hypothesis for replication.

These screens support a narrow conclusion: probe feedback can change debate
text, and some configurations improve one judge, but the benefit is not stable
across displays or judge models at current sample sizes. The local probe is a
promising locator of factual contradictions; that does not yet establish that
showing its output causes better debates.

## Relation to the A/B/C/D grid

| Debater input | Judge sees arguments only | Judge also sees probe scores |
| --- | --- | --- |
| No private feedback | A | C, using the A transcript |
| Private feedback | B | D, using the B transcript |

The completed historical A/B analysis and both new screens are in the
arguments-only judge column. Display and prompt refer to what the **debaters**
privately receive. The Haiku and native judges in iteration 6 saw the same
source-free public transcripts and no scores, private plans, or story. The
earlier C/D results can reuse A/B transcripts because judge assistance is
post-hoc. A full new C/D sweep over the display and prompt branches remains a
separate experiment.

## Raw records and quality checks

The canonical raw records preserve model-visible system and user messages,
sampling settings, request/response hashes, raw provider bodies, finish reasons,
usage, and accepted state transitions. In routed prompt receipts, `request` is
the canonical scientific body and `transport_request` is the actual submitted
body; routing changed while messages, schema and sampling stayed identical.
Private plan calls retain the reasoning
returned by the provider. Public calls disabled reasoning. Every public feedback
packet is visible in the exact saved request that caused the next argument.
Provider-internal state that was never returned cannot be reconstructed.

The display run contains 672 new public action receipts and reuses the first two
historical A turns per family. The prompt run contains 32 private plan, 192 public
action, and 32 Haiku judge receipts. Its `export_v1` directory is a judge/analysis
view of episodes; it is not the canonical raw archive. Canonical records are in
the run's `requests`, `raw_responses`, `jobs`, and `states` directories. Failed
attempts and transparent provider-route amendments remain retained.

Native replay refused truncation; the longest historical input was 11,036
tokens. Shared and original scorer implementations matched exactly on the same
GPU. Across rentals, 1,980/1,992 historical displayed percentiles were identical;
the remaining 12 differed by at most 2.084 percentile points. No raw activation
tensors were stored on the VPS.

The local raw browsers are:

- `runs/iteration6/raw_audit/index.html` for historical requests and responses;
- `runs/iteration6/live/inspection_v1/index.html` for three display families;
- `runs/iteration6/live/prompt_inspection_v2/index.html` for two prompt families,
  all four cells, private plans, all turns, feedback and scores.

The `runs/` tree is intentionally excluded from Git. A checksummed private
handoff archive must accompany the public branch when colleagues need raw text.
The current host-side archive is
`runs/iteration6/handoff/iteration6_private_evidence_v2.tar.gz` (SHA-256
`cf023966d1f87f62ca834a95f5beab45bc2acfd5337a694ea3070812ae20974d`).

## Limits and next decisions

All estimates use one generated trajectory per cell. Family bootstrap intervals
do not capture generation-seed, provider, or annotation uncertainty. Factuality
comes from automated readers; the 12-family second-reader audit finds 74.0%
overall verdict agreement and 97.5% agreement when both readers consider a
sentence eligible. Every source family had earlier experimental exposure.

The clean next step is a preregistered multi-seed confirmation of a small number
of conditions: no feedback, neutral feedback, full bundle, and one local
sentence/phrase display. It should add human factuality adjudication and both an
unassisted and score-assisted judge. The current data do not justify the full
probe × layer × display × prompt factorial. New live debates are required for
every debater-visible probe or display change; saved feedback-conditioned turns
cannot be replayed as the counterfactual debate.

Start with the [reading guide](probity_iteration6_reading_guide.md), then use the
[factuality report](probity_iteration6_factuality.md), [specification](probity_iteration6_spec.md),
and [artifact index](../reports/data/iteration6/README.md) for exact denominators,
plots, CSVs and provenance.
