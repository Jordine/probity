# Probity iteration 5: one-page result

The completed experiment asks whether a frozen activation-based score packet
helps a final judge choose the correct answer after a six-turn debate. It covers
166 questions from 116 source families, two same-model participant conditions
(Qwen3.5-27B and Qwen3.8-27B), 332 question/model units per grid cell, and 664
distinct debate trajectories. There are 568 newly generated trajectories and
96 exact pilot reuses. Each of two judges made 1,328 decisions.

| Cell | Debate generation | Judge presentation |
| --- | --- | --- |
| A | No feedback | No score packet |
| B | Feedback-conditioned | No score packet |
| C | Exact A transcript | Final score packet |
| D | Exact B transcript | Final score packet |

The packet uses one frozen adapted response-mean logistic head, mean-pooled over
the whole structured response: Qwen3.5 layer 43 and Qwen3.8 layer 35. It bundles
prior-turn percentiles, running speaker means, and an optional recommendation.
The experiment cannot attribute an effect to one component.

Family-macro accuracy was:

| Judge | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Native bf16 Qwen3.5-4B | 76.7% | 80.9% | 98.1% | 98.1% |
| API Claude Haiku 5.5 | 86.1% | 85.8% | 94.4% | 95.8% |

Adding the packet to identical text produced a large descriptive uplift. For
the native judge, C−A was +21.4 percentage points (95% family-bootstrap interval
+16.2 to +26.9) and D−B was +17.2 (+12.5 to +22.1). For Haiku, C−A was +8.3
(+4.2 to +12.8) and D−B was +9.9 (+5.7 to +14.4). B−A, which compares separate
single-seed trajectories, was uncertain: +4.2 (−0.1 to +8.6) for the native
judge and −0.3 (−5.3 to +4.7) for Haiku.

This supports a bundled final-display assistance effect. It does not establish
that the numerical probe alone caused the uplift, and it provides weak evidence
that live feedback improves the generated debate trajectory.

The environment has a major quality limitation: either speaker repeats an
argument exactly in 278/332 no-feedback trajectories and 242/332
feedback-conditioned trajectories. All three turns from a speaker are identical
in 137/664 and 90/664 speaker-trajectories respectively. These debates often
behave like repeated advocacy rather than iterative exchange.

Other caveats: there is one debate seed; intervals are unadjusted 10,000-sample
family bootstraps and do not capture generation variance; provider routing was
heterogeneous and unrandomized; only 28 families/35 questions were untouched
before expansion; source review classified 98 questions supported, 25
ambiguous, six contradicted, and 37 unreviewed. A lexical leak heuristic marked
32 turns, but manual review found zero actual disclosures of probe scores,
percentiles, recommendations, or private feedback.

See [the full results and methods](../../../docs/probity_iteration5_results.md),
the [figure pack](../../figures/iteration5/research_pack_v2/research_pack.pdf),
and the [environment diagnostic](../../figures/iteration5/environment_quality_v1/environment_quality.png).
