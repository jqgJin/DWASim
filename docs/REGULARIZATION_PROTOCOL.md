# Component screening and path-weight shrinkage

## Fixed development experiment

This is a revision-stage experiment on previously inspected ACM and DBLP test
sets, not an untouched external evaluation or preregistration. It changes only
the selection rule, not the component affinities, three complete paths per
dataset, official partitions, deterministic ranking, or ten-neighbor voting.
IMDB calibration is not changed by this experiment.

1. Evaluate the 15 quarter-simplex component candidates on each path.
2. For screening, retain the minimum nonzero count among candidates within
   0.005 of the best internal mean Macro-F1. Break ties by Macro-F1, NDCG@10,
   then magnitude weight. The tolerance is a practical rule, not a confidence
   bound or an equivalence margin.
3. Fit the 15 quarter-simplex path weights on these fixed component affinities.
4. For shrinkage, interpolate toward uniform path weights at strengths
   0, 0.25, 0.5, 0.75, and 1. Choose the largest strength within 0.005 of the
   best internal mean Macro-F1 among these candidates. Reuse the strength-zero
   evaluation and perform four extra configurations. Comparison slack is 1e-12.

Internal reference/validation splits are stratified 80/20 partitions with
seeds 20250803--20250812. Complete outer refits use seeds 20260910--20260914
and the same 80/20 ratio. The outer partitions overlap; their mean and sample
SD describe partition sensitivity, not independent graph replications.

All rules are fixed before this experiment is run: original selection,
original-component uniform fusion, shrinkage alone, screening alone,
screened uniform fusion, both rules, cosine, single-component selection,
cosine with shrinkage, and single-component selection with shrinkage.

Cosine uses 15 configurations, single-component selection 24, original and
screened selection 60. Shrinkage adds four. Uniform fusion with fitted
components uses 46. Actual computation is shared across rules; per-method
budgets describe the corresponding standalone selection work. Their search
spaces are not equal-dimensional.

No official test label is passed to fitting functions. Only training matrices,
training IDs, training labels and internal partitions are accepted. Existing
data loaders can read the official test labels, but they do not participate
in any selection stage. All outer fits and final training selection finish
before official-test metrics are evaluated.

The combined rule is compared with original, uniform, cosine and single
selection using 2,000 category-stratified paired query bootstrap resamples,
seeds 20260930--20260933 in that order. Intervals are unadjusted and conditional
on the fitted predictions; graph dependence and selection uncertainty are
outside their scope. No significance ranking is asserted. Prediction arrays,
all candidate scores and partitions are generated locally, never committed.

## Commands

After benchmark preparation:

```bash
python -m dwasim.experiments.regularization --dataset all
python -m dwasim.reporting.regularization
python -m unittest discover -s tests -v
```

Completed final result records are not overwritten. For an independent rerun:

```bash
python -m dwasim.experiments.regularization --dataset all --output-dir results/rerun
python -m dwasim.reporting.regularization --results results/rerun --output manuscript_assets/rerun
```

Checkpoint records preserve finished outer partitions for inspection; the
runner does not automatically resume from them. A rerun recomputes all fits
and must use a new output directory after a completed run. Cache reuse changes
runtime, not selection rules. The runner's audit duration is not a comparison
of full-pipeline costs.

## Interpretation

Do not change the tolerance, strengths, seeds or selected rule in response to
the reported test ranking. Retain coincident and unfavorable outcomes. Fewer
active terms do not by themselves prove unique information or lower total
runtime. Screening evaluates the full grid even if it selects one component.
The original method remains a separate reference throughout the manuscript.
