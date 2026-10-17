# Iteration 3 reproduction records

This snapshot publishes experiment code, audited outcome tables, aggregate probe results, real figures and method/runtime records. The [results report](../../docs/probity_iteration3_results.md) explains the findings and their limits. `manifest.json` records public file hashes and the identities of separately retained evidence. Original records preserve their bytes, including historical logical paths.

The primary corpus contains 24 source families, 48 debates and 288 public turns. All cases remain in the denominator. Independent sentence annotation covered 1,977 fixed units; accepted labels and unknown units remain distinct. The context diagnostic has 90 originally schema-accepted results and six documented lossless key aliases. Its 94/96 correct answers were obtained after replacing the original system message and removing its verified-answer cue; this is not a generation-time knowledge or intention test.

From the repository root, the following commands redraw the public charts without source stories, API credentials or GPU inference. Use a fresh output directory. Matplotlib 3.10.9 is the recorded rendering version; SVG metadata and renderer/library differences can change file bytes without changing the plotted statistics.

```bash
python3 reports/plot_local_probe_results.py \
  --results reports/data/iteration3/local_sentence_summary.json \
  --granularity sentence \
  --out-prefix reproduced_figures/local_sentence

python3 reports/plot_local_probe_results.py \
  --results reports/data/iteration3/local_clause_summary.json \
  --granularity clause \
  --out-prefix reproduced_figures/local_clause

python3 reproduction/iteration3/judge_controls/plot_corrected_source_controls_v1.py \
  --report reports/data/iteration3/source_control_corrected_all_v1.json \
  --out reproduced_figures/source_controls

python3 reports/plot_frozen_judge_results.py \
  --table Qwen3.5-4B reports/data/iteration3/judges/native \
  --table Haiku5.5 reports/data/iteration3/judges/haiku \
  --table Opus5.5 reports/data/iteration3/judges/opus \
  --out-prefix reproduced_figures/judge_results
```

These commands project existing audited statistics; they do not rerun annotations, bootstrap analyses or model inference. The source-control summary includes 480 outcome-only observations and all 30 aggregate cells. The judge figure includes all 30 prespecified effect cells across its three judge models. The secondary clause chart covers the 771 units from 48 fixed wrong-assigned opening responses, separately from the 5,598 all-turn clause units used for judge displays. Figure manifests identify real results; layout fixtures are excluded.

Further reproduction requires the corresponding retained artifacts:

| Operation | Required inputs beyond the public code and summaries |
| --- | --- |
| Recompute statistical analyses | Exact source-family mapping, complete episode/annotation/score records, frozen analysis specifications and original failed-attempt records. |
| Rescore local probes from cached activations | Frozen probe heads, original development reference arrays, retained same-machine reference replay, reference and fresh activation caches, and both small fitted `text_control.joblib` artifacts. |
| Extract fresh activations | Complete original source/preparation/generation contexts, model and tokenizer revisions, probe artifacts, full-sequence extraction contract and a compatible GPU runtime. |
| Run fresh API generation or judging | Complete source and input packages, explicit API configuration, a new budget ledger and new output paths. Retain every failed attempt and completion; new provider outputs are a new experiment. |
| Build a transcript inspector | Private source/episode/probe bundles supplied to `reports/build_transcript_inspector.py`. Its generated HTML contains source and transcript text and is not part of this public package. |

`reports/summarize_response_probes.py` recomputes the descriptive assigned-role companion from the retained `response_probes_*` and `fitted_response_*` outputs plus the source-family map. It preserves all 288 turns. Those role labels are different from independently judged sentence truth and from hidden intent. Published FIBS weights and separately trained Apollo-data heads are separate tracks; exact collaborator checkpoints have not been verified. No Qwen3.8 published FIBS counterpart is supplied by this run.

The private reproduction inventory includes complete raw requests/jobs, frozen heads and the required Qwen3.5 and Qwen3.8 `text_control.joblib` files under their original `runs/iteration3/local_probes/fitted_*_v2/` paths. Their hashes are in `local_probes/text_control_provenance_v1.json`. Essential structured request/job records are not excluded by file size. The separately listed `local_fresh_feature_provenance_private_v1.tar.gz` preserves 15 original files byte for byte, including complete raw corpora, extraction manifests and alignment receipts. It omits only a 74-byte readiness marker containing an infrastructure identifier. The original 16-member `local_fresh_feature_provenance_v1.tar.gz` remains unchanged and separately retained with independent download verification. Both hash receipts are public; both archives remain private because they contain full source and prompt text. Neither contains activation tensors. The new derivative has not been separately uploaded at inventory time; its preserved originals remain recoverable from the retained original archive.

The two old-reference archives `local_probes/retained_verified_features_qwen35_v2.tar.xz` and `local_probes/retained_verified_features_qwen38_v2.tar.xz` total 359,719,884 bytes. Fresh retained activation segments are identified as `final/retained_features_qwen35_v1.tar.gz` and `final/retained_features_qwen38_v1.tar.gz`, totaling 514,949,387 bytes. These four archives and model weights are separate dependencies; the lightweight bundle alone cannot reproduce GPU-free probe rescoring. The fresh archives were retained separately in four parts each, with all eight parts independently downloaded and hash verified. The original archive-creation receipts remain unchanged; later retention proofs establish external preservation. `manifest.json` records the safe hashes, part counts and VPS presence at inventory time; retrieval identifiers remain in the private dependency inventory. Hashes identify artifacts and do not provide download access.

Older development caches are separately retained too. The iteration-2 portable archive contains 2,063,723,383 compressed bytes in 31 verified parts. The initial-pilot archive contains 1,279,922,324 compressed bytes in 20 verified parts and preserves all 916 original members, including 908 tensor files and eight small metadata files. Local tensor copies were removed only after external part readback and original-byte checks; the eight metadata files remain local. Restore pointers identify the exact whole-archive and member hashes. The initial-pilot archive's `/dev/shm` path identifies its temporary staging location, not a required restore location: unpack its members at their recorded repository paths. None of these archive containers is included in the lightweight bundle.

`final/gpu_runtime_packages_v1.json` and `final/vps_runtime_packages_v1.json` are public-safe package inventories captured after execution, not complete solver lockfiles. The GPU runtime records Python 3.11.14, PyTorch 2.9.1+cu128, Transformers 5.9.0 and the pinned container image; the VPS inventory records the analysis/runtime environment separately. The repository's legacy package configuration is not the Qwen extraction environment.

The original `final/features_qwen35/manifest.json` and `final/features_qwen38/manifest.json` bind extraction to these checkpoints:

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Preserve tokenizer/checkpoint revisions, bfloat16 execution, layers 19/27/35/43/51/58, pooling, score orientation, normalization populations, full context and exact unit coordinates. The extraction uses unpadded teacher-forced replay with returned reasoning included before the scored final response. It does not establish equivalence to opaque provider internals or live generation activations. The report retains the long-context left-padding discrepancy and its original bound.

The exact-reference adapter `local_probes/run_frozen_reference_export_v1.py` records a specific recovery: recomputed development references must match the retained same-machine replay exactly, then their original frozen normalization arrays remain authoritative. It never substitutes fresh evaluation scores. Both the original roundoff failure and the operational release/receipt remain recorded. Reusing that adapter requires its hash-bound heads, frozen/replayed arrays and reference features; silently loosening a tolerance or replacing a frozen array does not reproduce this run.

Historical runners under `environment/`, `coordinator/`, `local_probes/` and `judge_controls/` can refer to `/root/projects/probity-benchmark`, `/workspace`, prior attempts and immutable output directories. In a separate reproduction checkout, restore dependencies at the logical paths recorded in the manifests, or document a relocation wrapper while keeping frozen file bytes unchanged. Inspect the relevant stage's release, recovery amendment and completion receipt before executing its runner. Do not rerun a historical command into retained output directories. New inference requires new paths and its own receipts, and provider outputs need not match byte for byte.

The future `environment/counterfactual_pairs_scaffold_v1/` contains offline construction/validation code, protocol and audit metadata only. Full worked source examples and prompt previews remain private; its tests require those original consumed-development inputs and the family/consumption audits at their recorded paths. The scaffold uses already-consumed development families, groups both worlds by original source family and leaves unused families untouched. Its no-story paired 50% score is a construction property, not measured improvement or evidence of judge headroom.

The separate `local_probes/future_units_v1/` proposal binds the offline quote/honorific parser in `probity_benchmark/local3_future_units.py`. On 45 already-consumed responses from four development families it changes 257 units to 274 while preserving all source/argument characters. Public code, structural tests and audit metadata are supplied; full natural units and annotation payloads stay private. No models were called, labels created or current metrics replaced. New boundaries require compatible token pools or fresh extraction, independent annotation and a new freeze before claiming improvement. The inherited iteration-2 portable activation archive is separately identified for historical reproduction; it is not needed by the frozen fresh-export specs once their retained reference caches are restored.

The code retains the repository's existing MIT notice. Data, model weights and probe artifacts are supplied separately; private evidence access is separate from this public repository.
