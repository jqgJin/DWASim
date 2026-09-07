# Experiment programs

Run commands from the repository root after preparing the benchmark records.
Each program writes generated artifacts to an ignored local directory.

## Primary evaluations

| Program | Purpose | Generated location |
|---|---|---|
| `run_support_retrieval_stress.py` | Three-path component selection, strong similarity baselines, Macro-F1, and NDCG evaluation | `results/` and `cache/` |
| `plot_support_retrieval_stress.py` | Component-weight and paired-difference figure | `figures/` |
| `run_real_multipath_fusion.py` | Matched two-path normalization and fusion evaluation | `results/` and `cache/` |
| `run_unified_normalization_validation.py` | Pair-relative normalization and fixed-component audit | `results/` and `cache/` |
| `plot_real_multipath_fusion.py` | Multi-path fusion figure | `figures/` |
| `run_imdb_external_validation.py` | Multi-label evaluation on HGBn-IMDB | `results/` and `cache/` |
| `run_revision_controls.py` | Matched simple controls, retuned deletion, and outer training-only stability | `results/` and `cache/` |
| `run_representation_controls.py` | Complete/half profile and discrepancy-by-scale factorial controls | `results/` and `cache/` |
| `run_controlled_mechanisms.py` | Disjoint controlled support, magnitude, direction, complementary, and conflicting regimes | `results/` |
| `benchmark_revision_cost.py` | Isolated query-batch timing and process-memory diagnostic | `results/` |
| `run_path_fusion_audit.py` | Shared-component best-single-path, uniform, and learned fusion; paired intervals and complete outer reselection | `results/` and `cache/` |
| `benchmark_pipeline_cost.py` | Serial fresh-derived full-pipeline costs and exact zero-weight inference checks | `results/` |
| `build_revision_assets.py` | Generate current manuscript tables and publication figures from completed records | `manuscript_assets/revision_tables/` and `manuscript_assets/` |

## Supporting analyses

| Program | Purpose |
|---|---|
| `run_corrected_protocol.py` | Leakage-free single-path evaluation |
| `tune_effective_weight.py` | Training-only selection of the effective support weight |
| `run_nested_multipath_optimization.py` | Nested development analysis without official test-label selection |
| `run_historical_table.py` | Audits the earlier sampling protocol |
| `search_historical_seed.py` | Searches documented seeds for historical-protocol diagnostics |
| `reproduce_original.py` | Shared path construction, historical formulas, and data access |
| `similarity_baselines.py` | Audited PathSim and HeteSim implementations |

## Recommended order

```bash
python -m dwasim.datasets --dataset all
python -m dwasim.experiments.single_path --dataset all --k 10 --lambda-value 0.5
python -m dwasim.historical.weight_sweep --dataset all --k 10 --splits 10
python -m dwasim.experiments.normalization --dataset all --k 10 --splits 10
python -m dwasim.experiments.normalization_controls --dataset all --bootstrap-iterations 2000
python -m dwasim.experiments.main --dataset all --splits 10 --bootstrap-iterations 2000
python -m dwasim.experiments.multilabel --bootstrap-iterations 2000
python -m dwasim.experiments.ablation --dataset all --outer-repeats 5 --bootstrap-iterations 2000
python -m dwasim.experiments.representation
python -m dwasim.experiments.mechanisms
python -m dwasim.experiments.kernel_cost
python -m dwasim.experiments.path_fusion --dataset all --outer-repeats 5 --bootstrap-iterations 2000
python -m dwasim.experiments.pipeline_cost --dataset all --method all
python -m dwasim.experiments.regularization --dataset all
python -m dwasim.reporting.assets
python -m dwasim.historical.plot_normalization
python -m dwasim.historical.plot_components
```

Some evaluations are computationally intensive. Intermediate matrices and
pairwise components are cached automatically so an interrupted workflow can
reuse completed work.

`benchmark_pipeline_cost.py` intentionally bypasses derived profile and
component caches, but leaves them intact. It requires the completed primary
ACM/DBLP and IMDB results above only for a post-selection correctness assertion.
No previous score is consulted during fitting. The six workers run serially;
do not launch other heavy programs during timing. Imports, initial raw-data
conversion, metrics, bootstrap, output writing, and the subsequent active-query
audit are outside the recorded full-pipeline time. OS file caching is allowed.
The extra active pass omits only exactly zero weights and verifies identical
scores, rankings, and predictions. It does not use an approximation tolerance.

The asset builder accepts `--tables-only` and `--output PATH`. Its default output
is `manuscript_assets/`, not `figures/`. The older plotting commands above generate
supporting development visualizations; the current paper uses the two figures
from `build_revision_assets.py`. Original conceptual diagrams are author-provided
manuscript assets, not generated benchmark plots and not bundled with this
code-only repository.

## Evidence boundary

The additional regularization experiment is defined in
[`REGULARIZATION_PROTOCOL.md`](REGULARIZATION_PROTOCOL.md).
`run_regularized_selection.py` generates all ten rules, five complete outer
refits, final training selections, conditional paired effects, and frozen
predictions. `build_regularized_assets.py` regenerates its tables without
modifying figures; the general asset builder invokes it when both completed
dataset records are available. An alternative result directory can be passed
to the runner with `--output-dir` and to the table builder with `--results`.
Neither script downloads benchmark data. The runtime recorded by the runner
is an audit duration with cached affinities, not a full-pipeline benchmark.


- Training labels may be used to select component, path, and task parameters.
- Official test labels are used only after all selections are fixed.
- Revision-stage diagnostics are not described as preregistered or untouched
  external holdouts; all generated records and protocol parameters are retained.
- Test comparisons use identical candidates, queries, and tie handling.
- Node-bootstrap intervals condition on the observed graph and do not model
  dependence induced by shared edges.
- Historical-protocol diagnostics are kept separate from corrected held-out
  evaluations.
