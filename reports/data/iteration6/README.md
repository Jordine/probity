# Iteration 6 artifact index

Start with the [results narrative](../../../docs/probity_iteration6_results.md),
[reading guide](../../../docs/probity_iteration6_reading_guide.md), and
[follow-on debate-probe benchmark specification](../../../docs/probity_debate_probe_benchmark_spec.md).
Only one probe generated live feedback trajectories in iteration 6;
alternative probe and layer results below are passive/offline evaluations. Each
directory contains machine-readable JSON/CSV next to PNG/PDF figures.

| Question | Artifact | Population |
| --- | --- | --- |
| Did historical feedback change factuality? | [Historical paired factuality](factual_behavior_v1/factual_behavior.png) and [summary](factual_behavior_v1/summary.json) | 166 questions, 116 families |
| How reliable are the labels? | [Reader agreement](reader_audit_v2/reader_agreement.png), [confusion](reader_audit_v2/reader_confusion.png), [summary](reader_audit_v2/summary.json) | 12-family audit, 1,230 sentences |
| Which probes and layers detect contradictions? | [Pooled/local probe plot](probe_factuality_v2/probe_factuality.png), [layer plot](probe_factuality_v2/probe_layers.png), [metrics](probe_factuality_v2/metrics.csv) | All 13,029 historical sentences |
| Does local3 beat simple controls? | [Paired control differences](local_control_comparison_v1/local_control_differences.png) and [summary](local_control_comparison_v1/summary.json) | 378 mixed-label arguments, 83 families |
| How did display change behavior? | [Behavior plot](display_behavior_v1/display_behavior.png), [neutral-control plot](display_behavior_v1/display_behavior_vs_neutral.png), [report](display_behavior_v1/report.md) | 24 families × 7 displays |
| How did display change factuality? | [Display factuality](live_factuality_v1/live_factuality.png) and [summary](live_factuality_v1/summary.json) | 4,640 sentences in 672 new suffix turns |
| Do probes still work on new display turns? | [Fresh-turn probe evaluation](live_probe_factuality_v2/live_probe_factuality.png) and [summary](live_probe_factuality_v2/summary.json) | Same 4,640 sentences |
| Do final-judge results depend on judge model? | [Display judges](live_judges_v1/display_judges.png), [neutral-control judges](live_judges_v1/display_judges_vs_neutral.png), [prompt judges](live_judges_v1/prompt_judges.png), [report](live_judges_v1/report.md) | 200 identical transcripts per judge |
| How did instruction and feedback interact? | [Prompt effects](prompt_behavior_v1/paired_prompt_effects.png) and [summary](prompt_behavior_v1/summary.json) | 8 families × 4 cells |
| What are all axes and the interaction in the prompt 2×2? | [Complete judge contrasts](prompt_factorial_v3/prompt_factorial_judges.png) and [all behavior/judge contrasts](prompt_factorial_v3/contrasts.csv) | 8 paired families × 4 cells |
| Does incomplete prose drive prompt results? | [Prompt sensitivity](prompt_prose_sensitivity_v4/prompt_sensitivity.png) and [summary](prompt_prose_sensitivity_v4/summary.json) | Primary n=8; sensitivity n=7; both judges retained |
| Does incomplete prose drive display results? | [Display sensitivity](display_prose_sensitivity_v1/judge_sensitivity.png) and [summary](display_prose_sensitivity_v1/summary.json) | Primary n=24; sensitivity n=23 |
| Is native replay numerically stable? | [Replay differences](replay_diagnostic_v1/replay_differences.png) and [summary](replay_diagnostic_v1/summary.json) | All 1,992 historical turns |

Intervals generally resample source families. Read each summary's `limits`,
`uncertainty`, and denominator fields before quoting a point estimate. The raw
receipts and HTML browsers live under the gitignored `runs/iteration6/` tree and
must be transferred through the checksummed private handoff archive.
