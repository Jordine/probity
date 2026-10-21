# Analysis code for iteration6

These are byte-identical copies of the analysis scripts used in the experiment.
`source_manifest.json` maps each copy to its original location and hash. The
canonical detector, labeling, branch-generation and evaluation implementations
are in `probity_benchmark/grid6_*.py`. The historical numerical replay diagnostic
is `reports/diagnose_grid6_replay.py`.

Run from the repository root. These scripts require the corresponding private
`runs/iteration5` and `runs/iteration6` archives: publishing code and aggregate
results does not publish the raw API receipts. Most scripts accept `--help` and
require a new destination. The historical behavior script instead writes its
fixed report paths; do not rerun it over a preserved result without first making
a separate working copy of the archive.

For the experimental design and completed/deferred distinctions, start with
[the reading guide](../../../docs/probity_iteration6_reading_guide.md).
