# Bounded follow-up: path-wise kernel power calibration

This follow-up was specified after the centered-target experiment showed
classification/retrieval mismatch in ACM training-only outer evaluations and
its official descriptive ACM results were also available. It is exploratory
revision-stage method development, not a new untouched confirmation set.
The target-fitting outcomes are retained regardless of this follow-up.

Motivation: a global monotone transform cannot improve a ranking. A monotone
transform applied BEFORE multi-path fusion can change cross-path evidence
balance. Integer entrywise powers of PSD kernels remain PSD by tensor-product
feature maps, and preserve each individual path ranking. No graph/profile,
component formula, k, prediction rule or original figure is changed.

- Apply the SAME exponent p in {1,2,4} to every selected within-path affinity.
- Select p and the original 15 quarter-simplex path weights together, using
  existing ten inner training-only partitions and Macro-F1, then NDCG@10;
  exact remaining ties prefer smaller p, then fewer nonzero path coefficients.
- Original within-path component selection is unchanged. Component-deletion
  controls rerun that selection over the corresponding restricted grid.
- Families: full DWASim, magnitude-only, cosine-only, no support, no magnitude,
  no direction. All six have the same 45 fusion candidates.
- Report p=1 controls for EVERY family, not merely a previously weaker baseline.
- Reuse only exact component-selection score records whose reference IDs and
  fixed seeds match the existing frozen five outer partitions and final fit.
  Fresh fusion selection uses current reference affinity matrices.
- Preserve original stored-precision arithmetic; assert p=1 full affinity
  reproduces the existing official original DWASim F1 and NDCG to 1e-12.
- Five outer partitions and ten inner partitions as in PROTOCOL.md. All data
  and paths unchanged. Save all candidates, folds, parameters, predictions,
  metrics, timings and paired query-bootstrap intervals (2000 replicates).
- Do not change this candidate set in response to official results. Report
  null/negative outcomes and avoid claiming an across-the-board improvement.
- Count 45 component + 45 fusion candidates for full DWASim; restrictions have
  their own stated component budgets. Cache reuse accelerates this local run;
  it is NOT an end-to-end runtime benchmark.

Only a small number of supported results should enter the main paper. Complete
search and ablation tables belong in the SI; no new decorative plots are needed.
