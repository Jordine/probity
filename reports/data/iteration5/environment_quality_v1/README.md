# Iteration-5 environment-quality diagnostic

This additive diagnostic reads all 664 canonical public trajectories joined by
`runs/iteration5/judge_grid_v1/pairs.json`. It does not change the frozen grid,
probe scores, judge inputs, outcomes, or inclusion decisions.

## Measures

- **Trajectory with any exact repeat:** either speaker has fewer than three
  distinct public `argument` strings across its three turns.
- **Speaker all-three identical:** all three argument strings for one speaker
  are exactly equal. There are two speaker-trajectories per debate trajectory.
- **Blank-line turn:** the argument contains `\n`, optional whitespace, and a
  second `\n`. This is a mechanical proxy for the requested two-paragraph
  format, not a semantic assessment.

String equality is exact. The audit does not normalize whitespace, punctuation,
or case, and therefore does not count paraphrased repetition.

## Pooled result

- No-feedback A/C transcript arm: **278/332**
  trajectories contain an exact within-speaker repeat; **137/664**
  speaker-trajectories repeat the same argument all three times; **263/1992**
  turns contain a blank line.
- Live-feedback B/D transcript arm: **242/332**
  trajectories contain an exact within-speaker repeat; **90/664**
  speaker-trajectories repeat the same argument all three times; **97/1992**
  turns contain a blank line.

`summary.csv` gives pooled, participant-model, provenance, and model-by-provenance
breakdowns. `trajectory_rows.csv` retains one text-free diagnostic row per
trajectory. The PNG presents the principal pooled and stratified rates.

## Limitations

Exact repetition is a narrow lower bound on repetitive behavior. This audit does
not measure semantic redundancy, quotation validity, whether a rebuttal engages
the opponent, factuality, deception, or overall debate quality. A blank line can
occur without two good paragraphs, and two semantic paragraphs can be formatted
without a blank line. Comparisons between arms are descriptive: the public
trajectories were separately generated and API provider routing was not
randomized. The reused 24-question pilot and 142-question expansion were created
at different times and are shown separately rather than treated as repetitions.

## Reproduction

From the repository root:

```bash
python3 reports/build_grid5_environment_quality.py
```

The command requires a new output directory by default. Remove or redirect the
existing additive output directories only when intentionally rebuilding them.
`manifest.json` records the input, script, table, README, and figure hashes.
