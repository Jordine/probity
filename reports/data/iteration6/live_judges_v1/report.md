# Paired final judges

Both judges saw identical source-free question/options/public arguments and original answer-only system instructions. No score packet, private plans or source story provided to either judge.

10,000 source-family bootstrap draws, seed 62166. Same cases/arguments for both judges; 24 families per display condition, 8 per instruction condition. No generation-seed, annotator, or provider variation modeled. Small-sample bootstrap intervals can collapse at boundary outcomes; this does not establish certainty. Exploratory simultaneous comparisons, no multiple-testing selection claim.

Each judge evaluated 200 identical debate transcripts: 168 display branches and 32 prompt branches.

| Population | Condition | Judge | Correct/families | Accuracy |
|---|---|---|---:|---:|
|display|none|Haiku 5.5|19/24|79.2%|
|display|neutral|Haiku 5.5|19/24|79.2%|
|display|bundle|Haiku 5.5|21/24|87.5%|
|display|turn_scores|Haiku 5.5|17/24|70.8%|
|display|speaker_means|Haiku 5.5|19/24|79.2%|
|display|recommendation|Haiku 5.5|20/24|83.3%|
|display|shuffled_bundle|Haiku 5.5|18/24|75.0%|
|display|none|Native Qwen3.5-4B|18/24|75.0%|
|display|neutral|Native Qwen3.5-4B|17/24|70.8%|
|display|bundle|Native Qwen3.5-4B|18/24|75.0%|
|display|turn_scores|Native Qwen3.5-4B|18/24|75.0%|
|display|speaker_means|Native Qwen3.5-4B|18/24|75.0%|
|display|recommendation|Native Qwen3.5-4B|18/24|75.0%|
|display|shuffled_bundle|Native Qwen3.5-4B|16/24|66.7%|
|prompt:explicit|none|Haiku 5.5|7/8|87.5%|
|prompt:explicit|bundle|Haiku 5.5|8/8|100.0%|
|prompt:explicit|none|Native Qwen3.5-4B|7/8|87.5%|
|prompt:explicit|bundle|Native Qwen3.5-4B|7/8|87.5%|
|prompt:selective|none|Haiku 5.5|7/8|87.5%|
|prompt:selective|bundle|Haiku 5.5|8/8|100.0%|
|prompt:selective|none|Native Qwen3.5-4B|7/8|87.5%|
|prompt:selective|bundle|Native Qwen3.5-4B|5/8|62.5%|

Plots show paired policy effects against no feedback. CSV/JSON also retain the neutral-control comparisons and native-minus-Haiku uplift interactions. A smaller effect for one judge is a measured interaction here, not an assumption about model strength.

These participant-display comparisons occupy the public-arguments-only judge column of the historical 2×2 grid.
