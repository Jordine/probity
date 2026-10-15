# Probity analysis plots

[Download the 11-page PDF](figures/iteration3/analysis_plots_v1/probity_analysis_plots.pdf) · [Download all PNG, SVG and PDF exports](figures/iteration3/analysis_plots_v1/probity_analysis_plots.zip)

These figures use the completed pilot's audited aggregates. The independent unit is the original story family: 48 debate presentations cover 24 families, with both Qwen 27B debater models represented. Intervals resample families and are not adjusted for multiple comparisons. No new inference or statistical fitting was performed for this collection.

The [full results report](../docs/probity_iteration3_results.md) gives the methods and limitations. The [plot manifest](figures/iteration3/analysis_plots_v1/manifest.json) records every input hash, plotted summary and output hash.

## Primary judge comparisons

Uplift is real probe assistance minus the matched neutral display. The adapted response probe helps the small judge; the new local sentence display harms it. Both comparisons were prespecified.

![Primary probe uplift by judge](figures/iteration3/analysis_plots_v1/01_primary_uplift.png)

## Baseline accuracy and headroom

The high baseline refers to the judge's answer accuracy. Neutral displays have their own baselines, so subtracting the no-signal score would answer a different question.

![Judge accuracy for no signal, neutral displays, probe assistance and oracle](figures/iteration3/analysis_plots_v1/02_judge_accuracy.png)

## All detector effects

Every measured pooled detector comparison is shown. FIBS covers Qwen 3.5 debaters only; Opus evaluated only the two primary signals. Unevaluated cells are distinct from zero effects.

![All detector effects for all judges](figures/iteration3/analysis_plots_v1/03_all_detector_uplift.png)

## Errors fixed and introduced

These are paired changes relative to each detector's neutral control. Unchanged answers are omitted. The denominator is 48 presentations except for the 24-presentation FIBS track.

![Paired corrections and new errors](figures/iteration3/analysis_plots_v1/04_fixes_and_harms.png)

## Response role detection

These labels identify the assigned correct-answer or incorrect-answer role. They do not establish the truth of individual sentences or hidden intent.

![Response probe assigned-role AUROC](figures/iteration3/analysis_plots_v1/05_response_role_detection.png)

## Local truth label coverage

All 1,977 fixed sentence units remain in the denominator, including unknown and unaccepted labels. Accepted labels come from independent model judgments and evidence, not human ground truth.

![Supported, contradicted and unknown sentence units by assigned role](figures/iteration3/analysis_plots_v1/06_sentence_truth_coverage.png)

## Repeatability and cache reuse

Only Haiku supplies distinct generations of repeated identical requests in this comparison. Three of 265 answer pairs differ, within two of 97 prompt groups. Provider routing also varies; this is not an isolated measurement of sampling noise. Native cached agreement does not measure generation variability, and Opus has no identical-request repeats.

![Fresh generations, cached reuses and observed Haiku disagreements](figures/iteration3/analysis_plots_v1/07_repeatability.png)

## Primary local sentence evaluation

The full control comparison uses within-response AUROC, averaged by original source family. Only responses with both accepted supported and contradicted units contribute: 15 responses across seven families for Qwen 3.5, and 13 across five families for Qwen 3.8. Pooled sentence AUROC would not establish localization.

![All primary sentence localization controls](figures/iteration3/local_sentence_results.png)

## Secondary opening-clause diagnostic

This is a separate descriptive evaluation of 771 clauses from 48 wrong-assigned opening responses. It is distinct from the all-turn clause displays used by judges. The new Qwen 3.5 probe performs better here, while Qwen 3.8 remains worse than the previous probe; this does not establish a general improvement in local detection or judge uplift.

![All secondary opening-clause localization controls](figures/iteration3/local_clause_results.png)

## Source controls

Corrected question-only and full-story controls distinguish answer-option cues from source comprehension. The short-option subgroup is shown separately; all original story families remain in the overall comparison.

![Fresh source controls with and without the story](figures/iteration3/source_controls.png)

## Judge effects by debater model

These detailed panels preserve all frozen participant-specific effects and show raw baseline and oracle accuracy separately.

![Judge effects separated by debater model](figures/iteration3/judge_results.png)

## Reproduction

From the repository root, use a fresh output directory:

```bash
python3 reports/plot_analysis_summary.py --out reproduced_analysis_plots
```

The renderer uses Matplotlib, NumPy, CairoSVG and PyMuPDF. It reads only the published aggregate files and the original detailed SVG figures. It checks effect arithmetic, family and presentation counts, and truth-label denominators before plotting. The PDF preserves vector figures; the ZIP includes PNG, SVG, PDF and the numeric manifest. Original experiment files and prior figures are unchanged.
