A: no feedback / no packet · B: feedback-conditioned / no packet · C: exact A transcript / packet · D: exact B transcript / packet.
Packet: frozen adapted response-mean logistic head, whole structured-response mean pooling; Qwen3.5 layer 43 / Qwen3.8 layer 35; prior-turn percentiles, speaker means, optional recommendation.
Equal-weight source-family means. Intervals: unadjusted 95% percentile family bootstrap, 10,000 resamples.
They describe family variation; one debate seed gives no repeated-generation uncertainty. B−A/D−A change trajectories; C−A/D−B hold public text fixed.

Completed corpus: 664 trajectories = 568 newly generated + 96 reused pilot; 332 matched question/model units per cell per judge; 1,328 decisions per judge.
Composition is reported by judge, cohort and participant model in cohort_composition.csv.
Pooled selected cohorts average retained units within family, then families equally; model balance can differ.
The field named question_weighted_accuracy in the v1 source is a retained-unit-weighted rate; in the joint selected cohort one question retains only one model.
Source support is a fallible reviewer diagnostic, not certified gold or local sentence truth.
Joint eligibility additionally requires the initial knowledge check and public answer-field role retention
in both trajectories; feedback inherits the initial knowledge observation. Selection is descriptive and
post-generation, and is not an eligibility-only analysis. Cohorts overlap.
Untouched means no recorded family use before this expansion, not an independent repeated-generation study.
This pack does not contain attempt/missingness accounting: attach the audited recovery supplement before sharing.
Intervals cannot be independently recomputed from aggregate summaries alone.
