# Shared-prefix display behavior

All effects conditional on shared original A first round and historical private plans. Initial prefix is identical across seven policies. Probe scores summarize assigned roles, not sentence factuality. Live feedback changes subsequent contexts, so suffix cannot be counterfactually replayed as equivalent. One generated suffix per policy; paired family bootstrap does not estimate seed variance.

These are descriptive response changes. Factual-label metrics and final-judge effects are separate analyses.

| Display | Words vs none | Repeat fraction vs none | Incorrect-role score vs none |
|---|---:|---:|---:|
|neutral|-12.344 [-20.010, -4.666]|-0.083 [-0.188, 0.021]|0.174 [-1.780, 2.214]|
|bundle|-8.823 [-20.948, 2.334]|-0.125 [-0.188, -0.062]|3.125 [1.215, 5.252]|
|turn_scores|-6.646 [-13.667, -0.041]|-0.104 [-0.208, 0.000]|0.781 [-0.998, 2.691]|
|speaker_means|-5.208 [-16.990, 4.823]|-0.083 [-0.177, 0.010]|1.823 [-0.174, 4.037]|
|recommendation|-10.875 [-17.917, -3.740]|-0.062 [-0.135, 0.000]|1.823 [0.217, 3.602]|
|shuffled_bundle|-11.750 [-21.157, -2.729]|-0.135 [-0.229, -0.042]|1.172 [-1.085, 3.385]|

The machine-readable summary also provides every policy minus neutral-formatting control; no-feedback and neutral are separate baselines.

Shuffled control: {'packets': 96, 'unchanged_numeric_packets': 23, 'unchanged_recommendations': 47, 'limitation': 'Permutation need not be a derangement; some packets or recommendations remain unchanged. This is a weak negative control, not guaranteed uninformative or adversarial feedback.'}

This behavior analysis was frozen after all 168 Haiku judgments and before the native pass. The later paired final-judge report contains all 168 valid native judgments; see `../live_judges_v1/report.md`.

Raw reading packets use two fixed hash-selected families and include all policies, every fresh request message, feedback and complete returned response. The inherited initial round is identified by exact historical request hashes; its provider envelopes remain in the iteration5 archive.

Runtime reproducibility: the repaired worker reproduces the original scalar implementation exactly on 12 same-process cases and passes four frozen historical checks. Some other historical scores differ slightly on this runtime; historical score drift is audited separately.
