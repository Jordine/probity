# Probity iteration 5: completed 166-question debate grid

Iteration 5 is complete. It evaluates whether a frozen activation-based score
packet helps an answer-only judge choose the correct answer after a debate. The
completed population contains 166 questions from 116 source families, two
same-model participant conditions, 332 question/model units per grid cell, and
664 distinct six-turn debate trajectories. Of those trajectories, 568 were
newly generated and 96 were exact pilot reuse.

The clearest result is a large descriptive uplift when the final judge sees the
bundled score packet. Live feedback to debaters, considered by itself through
the separately generated B-versus-A trajectories, has a small and uncertain
association with judge accuracy.

## The completed 2×2

| Cell | Public trajectory | What the final judge sees |
| --- | --- | --- |
| **A** | No feedback during generation | Debate without scores |
| **B** | Feedback-conditioned generation | Debate without scores |
| **C** | The exact A transcript | Debate plus final score packet |
| **D** | The exact B transcript | Debate plus final score packet |

A and C contain identical public text. B and D contain identical public text.
The C−A and D−B comparisons therefore hold the transcript fixed. B−A compares
one no-feedback trajectory with one separately generated feedback-conditioned
trajectory for the same question, participant model and role assignment. There
is exactly one retained trajectory per arm per unit, so this is not a
repeated-seed estimate or an isolated effect of a single feedback field.

The packet uses the frozen adapted response-mean logistic head. These heads were
selected in earlier work for favorable assigned-role discrimination using eight
training and four development families; iteration-5 outcomes were not used to
fit, normalize, or reselect them. The scorer mean-pools
the complete structured assistant response at layer 43 for Qwen3.5-27B and
layer 35 for Qwen3.8-27B. It displays prior-turn reference percentiles, running
speaker means, and an optional recommendation for the lower-mean speaker's
assigned answer. The grid does not separate those components. Higher scores
mean greater association with assigned wrong-answer advocacy under this
explicit protocol; they are not calibrated probabilities of deception or
sentence-level lies.

## Main results

Accuracy below is the primary equal-weight source-family macro. Intervals are
unadjusted 95% percentile family-bootstrap intervals from 10,000 resamples.
They describe family variation. The experiment has one debate seed per unit,
so the intervals do not include repeated-generation uncertainty.

| Judge | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Native bf16 Qwen3.5-4B | 76.7% [70.9, 82.0] | 80.9% [75.8, 85.8] | 98.1% [96.3, 99.4] | 98.1% [96.3, 99.6] |
| API Claude Haiku 5.5 | 86.1% [81.5, 90.4] | 85.8% [81.1, 90.3] | 94.4% [91.2, 97.2] | 95.8% [93.2, 98.0] |

| Judge | B−A: feedback-conditioned trajectory | C−A: packet on A text | D−B: packet on B text | D−A: combined | Interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native bf16 Qwen3.5-4B | +4.2 pp [−0.1, +8.6] | +21.4 [+16.2, +26.9] | +17.2 [+12.5, +22.1] | +21.4 [+16.2, +26.9] | −4.2 [−8.7, +0.2] |
| API Claude Haiku 5.5 | −0.3 pp [−5.3, +4.7] | +8.3 [+4.2, +12.8] | +9.9 [+5.7, +14.4] | +9.6 [+5.0, +14.5] | +1.7 [−3.7, +7.3] |

The same-text display contrasts are positive for both judges. Baseline judge
accuracy is already high, especially for Haiku, and the packet includes an
answer recommendation. These results support a bundled judge-assistance
effect. They do not establish that the numerical probe value alone caused the
uplift. B−A is near zero or uncertain for both judges, so this single-seed grid
does not show a clear benefit from changing the public debate with live
feedback.

The companion [research figure pack](../reports/figures/iteration5/research_pack_v2/research_pack.pdf)
shows the full grid, paired effects, untouched-family sensitivity, and source
review sensitivities. The [environment-quality plot](../reports/figures/iteration5/environment_quality_v1/environment_quality.png)
reports exact response repetition.

## Prompt and information contract

Both advocates receive the authentic source, question, options, and explicitly
verified correct answer. Each advocate is assigned one answer. The wrong-answer
advocate is directly instructed to invent a coherent account with plausible
quotations and maintain it. Each speaker privately prepares three
claim/quotation pairs plus an anticipated challenge and response.

The detailed fabricated account is self-generated during private preparation
and then supplied concretely to that speaker during the public debate. It is
not an externally authored false story. This run uses one strong custom
fabrication protocol; it does not sweep the original repository's dishonesty
levels or alternative prompting styles.

The participant conditions are Qwen3.5-27B debating itself and Qwen3.8-27B
debating itself. They are not cross-model debates. The final judges are native
bf16 Qwen3.5-4B and API Claude Haiku 5.5. Judges see the question, answer
options, and public turns. They do not see the source story or private plans.

OpenRouter serving was heterogeneous and unrandomized. Across the 1,704 new
no-feedback turns, the leading reported providers were Near AI (711),
DeepInfra (683), SiliconFlow (149), Reka (129), and Alibaba (23), with nine
other responses. Across the 1,704 new feedback turns, they were Near AI (693),
DeepInfra (559), Alibaba (195), Reka (143), and SiliconFlow (101), with 13
other responses. Haiku A/C comprised 608 Azure and 56 Anthropic responses;
Haiku B/D comprised 664 Azure responses. Provider routing was not randomized,
and provider fingerprints and revisions were unavailable.

## Coverage and execution checks

| Quantity | Completed |
| --- | ---: |
| Questions / original source families | 166 / 116 |
| Question/model units per cell per judge | 332 |
| Distinct debate trajectories | 664 |
| New / reused pilot trajectories | 568 / 96 |
| Probe-scored public turns | 3,984 |
| New scalar replay jobs | 3,408 |
| Decisions per judge / both judges | 1,328 / 2,656 |

All retained canonical public-turn generations ended with `stop`. New public completions
used at most 549 tokens without feedback and 527 with feedback. The longest
complete serialized scalar-replay context was 11,036 tokens. Native judging
covered 1,328/1,328 decisions, with maximum input 3,329 tokens and maximum
completion eight tokens. These checks found no truncation gap in the retained
run.

The native judge reused 192 exact pilot jobs and ran 1,136 fresh jobs. Each
Haiku half reused 96 decisions and ran 568 fresh ones. Cache reuse preserves
historical observations; it does not add seeds.

The campaign ledger accounts for **$33.5044004652** under the $45 ceiling,
with zero unsettled entries and $11.4955995348 headroom. This accounting total
includes $5.750784 in conservatively retained unknown-provider liabilities, so
it is not a claim about final provider billing. All owned GPU rentals were
destroyed and settlement evidence was retained.

## Environment and validity diagnostics

The six-turn format was followed structurally, but exact within-speaker
repetition is common. The additive environment-quality analysis reports the
following full-corpus counts:

| Diagnostic | No-feedback | Feedback-conditioned |
| --- | ---: | ---: |
| Trajectories where either speaker repeats an argument exactly | 278/332 | 242/332 |
| Speaker-trajectories with all three turns identical | 137/664 | 90/664 |
| Turns containing a blank-line two-paragraph form | 263/1,992 | 97/1,992 |

These are exact-string diagnostics, not semantic redundancy measures. They
show that many retained samples behave more like repeated advocacy than an
iterative exchange. This limits the interpretation of B−A and is a plausible
contributor to high judge baselines.

Public answer fields showed zero role switches, but that does not certify prose
adherence or quotation truth. A source-only reviewer marked 98 questions
supported, 25 ambiguous, six contradicted, and 37 unreviewed. Only seven new
questions received complete manual source reads. Full-corpus findings are
therefore descriptive; the figure pack includes source-supported and joint
source-supported/eligibility sensitivities.

The automatic disclosure heuristic marked 32 turns in seven feedback episodes
because they contained the ordinary word `probe` or `probes`. Qualitative
review found that every occurrence referred to task-domain spacecraft,
scientific equipment, or the verb “probe”; **zero were verified disclosures of
the activation probe, detector score, percentile, or private feedback**. Ten
additional `score`/`scored` matches across three episodes were already marked
task-domain false positives. The formal additive review preserves every match
and its source hash without altering the canonical outcomes.

## What this run does not cover

This completed grid has no repeated generation seeds, prompt sweep, externally
specified fabricated account, dishonesty-level sweep, mixed-model debate,
additional probes or layers, pooling comparison, local-truth probe feedback,
sentence/clause highlighting, peak-phrase display, neutral or shuffled
feedback, awareness-only control, exhaustive transcript truth annotation,
Opus judge, or independent repeated-generation replication on fresh families.
Apollo/DYL and newly trained local
probes were evaluated elsewhere; they are not hidden conditions in this grid.

The most useful next experiment is a smaller factorial replication that first
fixes the repetitive debate behavior, then separates the score value from the
speaker means and answer recommendation. Multiple debate seeds are needed
before drawing a stronger conclusion about live feedback changing the public
trajectory.

## Evidence and reproduction

- Machine-readable aggregate summary: [grid_summary_v1.json](../reports/data/iteration5/grid_summary_v1.json)
- Improved standalone figures: [research_pack_v2](../reports/figures/iteration5/research_pack_v2/)
- Eligibility-only sensitivity: [eligibility_only_v1](../reports/data/iteration5/eligibility_only_v1/)
- Environment-quality diagnostic: [environment_quality_v1](../reports/data/iteration5/environment_quality_v1/)
- Public disclosure review: [disclosure_review_v1](../reports/data/iteration5/disclosure_review_v1/)
- Audited run recovery summary: [public_summary.md](../runs/iteration5/reader_reviews/recovery_final_v1/public_summary.md)
- Private fixed-sample transcript inspector: `runs/iteration5/reader_reviews/inspection_pack_v5/index.html`
- Final immutable completion marker: `runs/iteration5/coordinator/final_analysis_complete_v1.json`

The public aggregates reproduce figures and tables. Full replication requires
private source stories, private plans, public transcripts, submitted messages,
raw model responses, probe artifacts, and runtime receipts.
