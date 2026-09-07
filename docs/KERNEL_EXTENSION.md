# Kernel theory and minimal-change calibration

Updated 2026-09-07. The title, original profiles, components, k=10 voting and
existing benchmark results are preserved. IMDB calibration is unchanged.

## Experiments and dependency order

After the original pipeline and `run_regularized_selection.py` have completed:

```bash
python -m dwasim.experiments.kernel_target --dataset all
python -m dwasim.experiments.power --dataset all
python -m dwasim.reporting.assets --tables-only
python -m unittest discover -s tests -v
```

The new runners preserve existing result and checkpoint files. Use
`--output-dir NEW_DIRECTORY` for a separate rerun. The default asset builder
reads the default `results/kernel_target/` and `results/kernel_power/` folders.
Do not silently replace them with selected reruns.

Kernel-target fitting uses reference-only centered Gram matrices and a
class-balanced label kernel. Every inner fold rebuilds the target from its own
reference labels. The positive ridge grid is fixed. All six full/deleted/single
component families receive the same grid. The nonnegative least-squares solver
checks its KKT residual and returns coefficients on the original uncentered
affinities. This is not a new neural representation or decision rule.

Power calibration applies a shared p in {1,2,4} separately to each within-path
affinity before fusion, not after fusion and not as a matrix product. Component
selection remains unchanged; each deletion repeats the restricted selection.
There are 45 fusion candidates for every family. Including component selection,
full DWASim has 90 candidates, each deletion 60, and magnitude/cosine 45.
Existing component-selection grids are reused only for verified matching
outer IDs and inner seeds. Each p=1 full-method result must reproduce the
frozen original metrics. These cached runs are not end-to-end cost benchmarks.

## Scientific interpretation

The three components and nonnegative fusion are PSD under the stated zero
conventions. sqrt(2(1-S)) is a Hilbert pseudometric; raw Bray--Curtis is not
claimed to satisfy the triangle inequality. Integer powers retain PSD and each
individual path ranking but can change fused rankings. Bounded perturbation
guarantees require explicit score margins and frozen parameters.

Power calibration improves the four official ACM/DBLP endpoint point estimates
relative to original DWASim. The DBLP F1 difference is small; ACM outer mean
retrieval declines. Component deletion remains tied or nearly tied in some
settings. Do not claim uniform generalization or that every component is
essential. Kernel-target fitting has a unique optimum but reduces official F1;
all its outcomes are retained in the paper's supplementary material.

Both experiments are revision-stage development on previously inspected
benchmarks. Parameters are selected from training labels only. Paired-query
bootstrap intervals condition on frozen predictions and the shared graph,
are not multiplicity-adjusted, and do not establish independent-graph risk.
The five outer partitions overlap and are not five independent datasets.

The original three figure bitmaps and the approved quantitative figures are
unchanged. The public code bundle contains code, tests and Markdown only,
not data, results or manuscript assets. Full local outputs remain in the
separate reproducibility archive.
