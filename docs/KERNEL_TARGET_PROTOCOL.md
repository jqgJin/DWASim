# Theory-guided, minimal-change revision — protocol locked before evaluation

Date: 2026-09-06. This is a revision-stage experiment, not a preregistered study.
The official test benchmarks have been inspected in earlier development.

## Argument and scope

DWASim's existing nonnegative profile affinities admit a positive-semidefinite
kernel interpretation. This motivates learning the same nine coefficients by a
regularized, centered kernel-target objective rather than a coarse two-stage
grid. Typed paths, complete profiles, component definitions, k=10, tie rules,
datasets and official partitions remain unchanged. The new selector is an
explicit extension, not a relabeling of previous results. IMDB's calibrated
multi-label protocol remains unchanged in this experiment.

## Locked terminology

| Term | Meaning | Decision |
|---|---|---|
| DWASim | Existing support/magnitude/direction affinity and path fusion | Keep title and method name |
| Original selection | Existing two-stage quarter-simplex search | Retain its results |
| Kernel-target selection | Regularized nonnegative centered-kernel fitting | New weight-learning variant |
| Magnitude-only | One minus Bray–Curtis dissimilarity | Same optimizer as full variant |
| Cosine | Directional affinity | Same optimizer as full variant |
| PSD | Positive semidefinite | Not strictly positive definite |
| Stability | Deterministic perturbation bound | Not a graph-independent risk guarantee |

## Frozen experiment

- ACM: PAP, PSP, PTP. DBLP: APA, APTPA, APVPA.
- Families: all nine components; magnitude-only; cosine-only; no support;
  no magnitude; no direction. Every family has the same regularization grid.
- On each fitting reference set, center every component Gram matrix and
  normalize it by its Frobenius norm. Construct the centered, unit-Frobenius
  label kernel from inverse-square-root-frequency-scaled one-hot labels.
- Let G be the Gram matrix of normalized centered component matrices, and a
  their inner products with the normalized label kernel. Solve
  min_{v >= 0} 0.5 v^T (G + eta I) v - a^T v.
- eta in {0.001, 0.01, 0.1, 1, 10}; select mean validation Macro-F1, then
  NDCG@10, then larger eta for exact ties. No test-driven grid changes.
- Convert to coefficients on the ORIGINAL uncentered affinities as
  gamma_r = (v_r / norm_r) / sum_s(v_s / norm_s).
  Centering is for weight learning only, not a hidden change to query scoring.
- Centered-constant components are excluded using a relative floating-point
  tolerance. If all label alignment vanishes, use uniform eligible weights;
  record the fallback. Do not claim general information loss from a
  training-constant component.
- Each of the ten inner splits (seeds 20250803–20250812) refits its label target
  and optimizer using reference labels only. Validation labels never enter
  that fit. Refit the selected eta on all available training references.
- Five outer partitions: seeds 20260910–20260914, 80/20 stratified training-only
  splits, with complete inner reselection. Record every family and partition.
- Official test evaluation occurs only after fitting. Keep every outcome,
  including negative ablations. Report Macro-F1 and NDCG@10, weights, KKT
  residual, wall time and conditional paired bootstrap differences (2000
  replicates). Outer partitions overlap; they are not independent graphs.
- Original and screening+shrinkage records are retained as frozen historical
  comparators. No result is overwritten; new output must use an empty folder.

## Theory evidence map

1. PSD of all components: direct feature/integral construction, including zero
   profiles; general RKHS basis from Aronszajn (1950).
2. sqrt(2(1-S)) is a Hilbert pseudometric and preserves affinity rankings;
   the raw Bray–Curtis discrepancy is NOT claimed to be a metric.
3. Regularized kernel-target fitting: centered alignment framework of Cortes,
   Mohri & Rostamizadeh (2012), with the regularized nonnegative quadratic
   objective stated explicitly. Classical convex optimization supports unique
   solutions; a deterministic perturbation bound is proved for this objective.
4. Ranking preservation requires an explicit score margin; weight stability
   alone cannot guarantee improved Macro-F1 or generalization on dependent
   graph nodes.

No theorem will be described as proving empirical superiority. No new neural
encoder, path family, diffusion step, or decision rule is introduced.
