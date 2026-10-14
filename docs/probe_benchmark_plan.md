# Probity probe benchmark: validation plan and first audit

Status: 2026-10-14. Initial pilot ceiling: **$25 total**, across API calls, GPU rental and storage. No new GPU has been rented for this task. Model-based validation is blocked by the connected OpenRouter account returning HTTP 402, insufficient credits. Nothing below is a claim of measured probe uplift.

Source correction: the first audit used a local base that omitted three commits on [Jordine/probity main](https://github.com/Jordine/probity/commits/main). GitHub `main` at `ca3952e17af8469be6d0b4ea8843727cd98a291f` has now been fetched, its newer token-probe work reviewed, and all three commits merged into this working branch. The updated assessment below includes that evidence.

The immediate question is whether the debate environment can measure useful detection of knowing falsehoods. Judge accuracy, role adherence, whole-response detection, local lie detection and downstream uplift are separate measurements. A failure in one must not be reported as a failure in another.

The original request is preserved in the [private research brief](https://chatgpt.com/space/page_01b1d87bb01081919db3dd0815fc95ef). The design follows the [debate benchmark spec](https://docs.google.com/document/d/1worLKbiRDYb3q5xOz6WH30VTnfr2-o9RGav_p_RWBNs/edit), with the decisions and remaining gaps below.

## Work completed

- Created an isolated worktree at `/root/projects/probity-benchmark`, branch `codex/probe-benchmark-20261014`, initially based on `831c44f4ff4c1d7e71c86f149062e65be24947c3` and now including GitHub `main` through `ca3952e1`. The original checkout contains substantial uncommitted changes and a nested older checkout; it has been preserved.
- Added a strict dataset preflight and a budgeted API baseline runner. Prompts include only the allowed fields; both answer orders are tested; full requests/responses and run manifests are saved. Errors, malformed answers and incomplete generations stop the pilot. Summary confidence intervals resample stories, not answer-order variants.
- Fixed an off-by-one in `FastLocalModelProvider`: generation hooks previously attached the state predicting a token to that token. They now capture the state after consuming that token. New metadata marks this as `post_token_v1`; old activation caches must not be mixed with these results.
- Removed silent output-allowance shrinking in that provider. Context overflow and a generation that ends without EOS now fail. The capture loop supports multiple EOS IDs and validates one activation per returned token.
- Fixed baseline transcript task/speaker metadata, rejected unknown task types before judging, and removed winner guessing by mention counts. Legacy `correct` is explicitly labelled `assigned_role_selection`; it is not an independent test of the answer actually endorsed in the debate.
- Sixteen targeted CPU tests pass. No real-model activation equivalence test, detector evaluation or new accuracy measurement has yet run.

## Evidence that changes the plan

### Dataset passes structural checks, not behavioral validation

The complete `data/quality_synthetic/stories.jsonl` contains 166 valid problems, with no duplicate problem hashes or repeated story hashes. All 4,636,993 bytes were read. The gold labels are A: 71 and B: 95, so answer-order counterbalancing is necessary. Story lengths are 10,218–38,853 characters; median 22,681, p90 32,670 and p95 34,510. These are character counts, not tokenizer context measurements.

Dataset SHA-256: `3c773e22849131571f5e88a774fe6001cb4bb41ca28576e00d3d634bc02b0863`.

### Archived debate results have invalid evaluation inputs

The archived `baseline_explicit_v2_20251226_192544` run contains 165 saved transcripts and 990 turns. All 165 omit the task type, structured problem data and explicit model mapping expected by the judge builder. None of the turns saves a finish reason. The old evaluator interpreted missing task type as APPS/code safety, and built an unknown-problem header.

Two complete transcripts were read qualitatively: `synthetic_103_Where_did_the_narrat` and `synthetic_109_How_many_years_ago_w`. They were selected as the two smallest saved transcripts for an initial inspection, not as a representative sample. In both, the supposedly dishonest speaker argues for the correct answer and the other speaker agrees. Some judges choose that speaker and are scored incorrect by role. Judge explanations explicitly discuss code/security criteria during the story task. This establishes concrete failures, not their prevalence across all debate content.

The archived prompts and generation termination metadata were not saved, so these records cannot establish whether role failure came from the model, its exact prompt, or an older serialization/mapping bug. Preserve them as audit evidence; do not relabel them into a clean benchmark.

### Newer GitHub probe results provide a baseline worth reproducing

The three GitHub-only commits were `0b8e2a68` (token-level training and sentence evaluation), `36cb6b47` (validation/leakage checks) and `ca3952e1` (70B validation and saved probes). They add per-token span labels, attention-mask handling, a direct Hugging Face extraction path, and more recent positive results. The earlier characterization based only on `LOCALIZATION_AUDIT_FINDINGS.md` and `PROJECT_STATUS.md` was incomplete.

The [committed 70B run log](https://github.com/Jordine/probity/blob/ca3952e17af8469be6d0b4ea8843727cd98a291f/results/70b_validation_results.log) reports Llama3.3-70B layer 55 IID `nauroc=0.9699` on 60 validation conversations, debate response-mean AUROC `0.7867`, and higher dishonest than honest scores in `14/15` scenario pairs. This is promising evidence for an existing probe baseline. The 93% figure is paired score ranking, not judge accuracy, validated false-span localization or downstream uplift. These are historical reported results, not newly reproduced measurements.

Review of the [70B script](https://github.com/Jordine/probity/blob/ca3952e17af8469be6d0b4ea8843727cd98a291f/scripts/run_70b_hf.py) and new validation scripts identifies concrete checks before reuse:

- The IID metric includes all nonpadding prompt and response tokens; prompt tokens are negative labels. Recompute on assistant content only and on mixed true/false response spans. A positive affine z-score transform preserves each example's AUROC, so the `norm_auroc` name does not independently remove this issue.
- Large within-turn score variance shows that scores vary; it does not establish that peaks coincide with knowing falsehoods. The generated-debate metric compares role-conditioned response means without independently labelled false spans.
- The 70B tokenizer calls allow input truncation at 1,024 training tokens and 8,192 debate-input tokens, generation has a 200-token cap without a completion gate, and printed examples are clipped. The 7B validation scripts explicitly cut stories to 2,000–3,000 characters. Their outputs do not establish compliance with the current no-truncation rule.
- The HF extraction helper does not pass an attention mask; the saved run logs warnings about missing masks. Its hooks read decoder-block outputs, whereas the TransformerLens scripts use `hook_resid_pre`. Preserve and test those distinct layer conventions.
- The 70B separation bootstrap resamples honest and dishonest arrays independently despite matched scenarios; use paired story resampling. Layer selection uses the same IID validation set as the reported best-layer metric; retain a separate final test.

The plan is to reuse and harden this newer loader/trainer/extraction work, then port the method to the target Qwens with newly matched weights. Do not assume either that local probing has failed or that the logged results already validate this benchmark. Historical label/boundary findings remain relevant to the older paths, not a complete verdict on the April additions.

Additional unresolved code risks: special display tokens can be removed without removing their score/activation positions; word-keyed score dictionaries overwrite repeated words; some probe exceptions are suppressed; some formatting defaults manufacture a neutral 0.5 score. Legacy judge instructions also overstate independence and local reliability. These paths are not approved for the new benchmark yet.

## Track 1 — establish a usable environment

1. **Small diagnostic pilot:** 12 distinct stories, both answer orders, with and without the full story, for Haiku 5.5, Qwen3.5-27B and Qwen3.8-27B: 144 planned calls. These are development examples, permanently excluded from a final test. The current runner has not completed any successful inference.
2. Record prompt length, completion length, finish reason, raw answer, parser validity, actual API route, model identifier and all generation settings. Explicitly disable reasoning for this first baseline; it is a configuration choice, not a hidden reasoning-token cutoff. Repeat the knowledge check with the eventual local debater's actual reasoning/template settings before comparing probe experiments.
3. Check that no-story accuracy is compatible with chance, full-story accuracy is high, and answer-order effects are small. Twelve stories can diagnose gross defects; a nonsignificant result is not evidence of equivalence to 50%. Pre-register a tolerance and interval criterion on a larger independent set before calling this a benchmark. A proposed final criterion is a 95% interval contained within 40–60% for the balanced no-story baseline, and at least 85% full-story point accuracy with its interval reported. These are proposed design criteria, not achieved results.
4. Before assigning advocacy roles, privately test each debater's knowledge of the story answer. Keep that diagnostic out of the live debate history. Separately measure whether the assigned wrong-answer advocate actually advances a knowingly false claim. Refusal, truthful concession, mistakes and failed role adherence are distinct outcomes. Report all episodes and the pre-specified knowledge-qualified subset; do not silently filter bad episodes after seeing uplift.
5. Generate a small fresh set of three-round debates with weak Haiku judging. The judge sees the question, answer options and public turns, never the story, role labels or private prompts. Score the judge's explicit answer choice against the answer key. Record speaker preference as a separate diagnostic.
6. Review complete transcripts, including true/false claims, fabricated or misleading quotations, refusals, consensus, score leakage and incomplete output. Stop if these failures dominate. Compare quote strings to the story as an auxiliary check; string match alone cannot establish truthful use of evidence.

Use a story-grouped, versioned train/dev/test manifest before any probe fitting or prompt tuning. Lock all pilot stories into development. If the remaining 154 stories are split among training, tuning and testing, precision will be limited; use separate training sources or generate new held-out stories rather than repeatedly tuning on the same test set.

## Track 2 — make the detectors comparable and correct

The [downstream meeting notes](https://docs.google.com/document/d/1z1t2hHzCwJ4l80I7uw6ilALB-lXpLkV2lTN4a8BwMiw/edit) emphasize DYL, raw scores, long-context dilution and the distinction between role detection and actual lying. The [current slide deck](https://docs.google.com/presentation/d/10EGvdjlgQeoDdiHpM1OybY87G94VPpPEke-GoAnoZ_o/edit) also highlights speaker-position bias. Its auditor comparison changes both player count and prompt, so it cannot isolate a probe effect.

| Detector | Intended first comparison | Required before use |
|---|---|---|
| Apollo roleplay probe | Shared reference baseline | Exact training source, checkpoint hash, supported model revision, layer/hook, pooling and score orientation |
| Collaborator DYL probe | Shared yes/no follow-up protocol | Exact private follow-up messages, prefill bytes/token IDs, final scored position, Yes/No contrast formula and original action context |
| Published FIBS probes | Additional reproducible Qwen3.5 reference | Published inference contract and pinned weight revision; label separately from Apollo roleplay training |
| New local probes | Within-response false-span detection | Verified span labels, train/dev separation, position-safe feature cache and held-out localization metrics |

The [FIBS paper, arXiv:2610.12445v1](https://arxiv.org/html/2610.12445v1) supports testing probes on both contextual falsehoods and model-organism behavior, with model size, pooling and architecture treated as explicit variables. Its broad training mixture is not synonymous with the Apollo roleplay dataset. Its labels concern knowing falsehoods, which do not cover every form of deceptive intent.

The [official implementation](https://github.com/AlignmentResearch/caught-in-the-act-probes) makes the token contract consequential: linear/MLP heads and full-window architectures use different extraction and output conventions. The last non-EOT token of a `Yes.` prefill can be the period. Do not describe that as the `Yes` token or substitute it for a collaborator's last-Yes/No setup. Preserve raw logits or probabilities with an explicit score type; never sigmoid an already probabilistic score. The published Qwen3.5 weights are a reference, not evidence that those weights apply to Qwen3.8.

Before any real probe result:

- Pin model/weights/tokenizer revisions, chat-template hash, dtype/quantization, layer numbering, pre/post-block hook, attention mask, response boundaries, prompt/reasoning policy and detector checkpoint hash.
- Fork DYL from the exact context that generated the action; discard its follow-up from the public trajectory. Distinguish the model's generated self-report from a probe applied to a forced prefill.
- Compare incremental generation states to a teacher-forced replay of the same token sequence. Test left padding, multiple EOS tokens, special tokens, Unicode, repeated words, first/last response tokens and long prompts. Require exact token IDs/masks and numerically justified activation tolerance.
- Use tokenizer offsets plus occurrence indices for displays. Assert equal counts for all IDs, offsets, activations and token scores. Missing scores are errors, not zeros or 0.5.
- Count the actual full prompt plus output allowance with the selected tokenizer. Fail oversized inputs and cut-off thinking/output. Save finish-reason distributions before accuracy metrics. Never silently clip context, completions, labels, plots or printed examples.
- Check raw score separation against transcript-level truth/lie labels, then check localization within mixed-truth responses. Include role-only, length-only, random-weight and shuffled-score controls.

The accessible [Cadenza project plan](https://docs.google.com/document/d/1RjSXfBEz5gr80HSebG7HxdvF1zprs0UklrD7cLbVUYk/edit) identifies [Cadenza-Labs/downstream-environment](https://github.com/Cadenza-Labs/downstream-environment). That repository returns 404 for the authenticated GitHub account `Jordine`. Documents are available; code/checkpoint compatibility remains unconfirmed. No message has been sent to collaborators.

## Track 3 — hillclimb local lie detection

First reproduce the newer GitHub token-level baseline with assistant-only labels, complete contexts and explicit hook/score contracts; reuse its existing training and leakage-control work. Start with a manually checked diagnostic set covering false assertions, true assertions in dishonest turns, faithful and fabricated quotations, misleading use of true evidence, uncertainty, mistakes and truthful concessions. Annotate factual falsity, evidence of knowledge and deceptive intent separately; preserve uncertain labels. Validate offsets by checking the complete labelled text against the original transcript. An LLM labeler may assist but cannot define ground truth without checks.

Train an inexpensive linear token/span baseline first, then test within-response ranking, span pooling and the existing annealing/NTML objectives. Include both true and false spans within the same response so that role identity alone cannot solve the objective. Contrastive source pairs, paraphrases, stories and prompt templates stay in one data partition. Fit scaling and thresholds on train/dev only.

Collect selected-layer features once per model/context and fit all regularization, loss and pooling variants from that cache. Use the existing multi-probe sweep principle rather than re-running a 27B model for every head. Cache keys include dataset bytes, model revision, tokenizer/template, hook, dtype, mask, position policy and generation version. Historical caches with uncertain alignment are not reused.

Report sample AUROC separately from within-transcript AUROC/AUPRC, span overlap, top-span recall and false positives on true spans. Bootstrap by story/transcript, never by token. Compare mean, max, length-normalized top-quantile and contiguous-window peak scores. A single high score on a long response is not automatically a local detection success. Define peak phrase width and threshold on development data, then freeze them.

Keep every displayed sentence/word occurrence and its offsets, including repeats. Show a full transcript with optional highlighted spans; provide whole-response and sentence views as explicit alternative summaries. A DYL message score must never be painted onto individual phrases as if it localized them.

## Track 4 — estimate uplift with paired controls

| Cell | Debaters receive scores | Judge receives scores | Trajectory source |
|---|---|---|---|
| A | No | No | No-feedback debate |
| B | Yes | No | Feedback debate |
| C | No | Yes | Same frozen debate as A |
| D | Yes | Yes | Same frozen debate as B |

The primary contrast from the spec is **B − A**: does detector access help the debaters produce a more informative debate for an unaided judge? **C − A** and **D − B** estimate judge assistance conditional on a frozen trajectory. Also report the interaction `(D − B) − (C − A)`.

Generate separate no-feedback and feedback trajectories on matched stories; do not retrofit feedback into an already frozen conversation. Judge each trajectory in fresh sessions with and without scores. Counterbalance correct speaker, speaker labels, speaking order and answer order. In B, explicitly prohibit disclosing probe outputs and audit for such disclosures.

Start with one fixed Qwen model and Haiku judge. Add Qwen3.5 versus Qwen3.8 and Opus judging only after the environment and detector gates pass. Stronger judges may have less headroom, but this is a hypothesis to test, not a premise. Compare no detector, real detector, distribution-matched shuffled scores, random weights and a clearly labelled oracle positive control. Match formatting and extra context in controls. Freeze aggregation/display choices before held-out comparison, and report paired story-level intervals and role-failure rates alongside uplift.

## Implementation and compute decisions

Continue on the isolated branch with a small new `probity_benchmark` harness. Separate immutable episode data, provider adapters, activation extraction, probe training, score rendering and analysis. Preserve useful existing code and historical artifacts. Decide whether to extract a new repository after the first valid end-to-end pilot reveals which interfaces are worth keeping; a wholesale rewrite now would obscure whether behavior or infrastructure changed.

Provisional allocation: **$2 environment API checks, $15 single-GPU feature extraction/probe pilots, $3 judge comparisons, $5 reserve**. This is an allocation, not a quote or spending commitment. Maintain one cumulative receipt across stages; the runner's `--budget` is only a per-invocation limit. The two failed API attempts reserved at most $0.0479584 in total; their original `cost_usd` fields are conservative reservations, not provider-reported charges. Preserve the original receipts and use `runs/pilot_accounting_20261014.json` for that correction.

Rent only after inputs, checkpoint contracts, runnable scripts and a current price fit the remaining budget. Profile weights plus KV cache and long contexts before choosing an instance; 27B weights alone do not establish memory fit. Prefer one sufficiently large GPU for a short, watched extraction job, reuse activations for CPU/lightweight head sweeps, checkpoint progress, and terminate only the instance created for this task. The pre-existing 4×H100 instance has not been used or changed.

The API runner reserves cost before concurrent requests and constrains provider token/request prices using OpenRouter's documented [provider price filter](https://openrouter.ai/docs/guides/routing/provider-selection#max-price). It stops on invalid output, unknown failures and accounting anomalies. API routes are recorded but are not equivalent to pinned local probe models.

## Reproduce the completed checks and resume the pilot

From `/root/projects/probity-benchmark`, with the existing Python environment:

```bash
python3 -m unittest tests.test_benchmark_preflight tests.test_generation_alignment tests.test_judge_protocol -v
python3 -m probity_benchmark.preflight --out runs/preflight_20261014/dataset_audit.json
```

After the provider account is funded, the next paid step is the same 12-story pilot with corrected request parameters, a new output directory and a $2 limit. Do not overwrite the failed-attempt receipts:

```bash
python3 -m probity_benchmark.run_baselines \
  --models anthropic/claude-haiku-5.5 qwen/qwen3.5-27b qwen/qwen3.8-27b \
  --n 12 --seed 20261014 --budget 2 --max-tokens 8192 --concurrency 4 \
  --out runs/baseline_pilot_20261014_v3
```

This command runs only environment baselines. It does not yet generate debates, train probes, establish local alignment on Qwen, or execute the 2×2 sweep. Those remain gated work above.
