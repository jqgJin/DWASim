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
python src/prepare_hgb.py --dataset all
python src/run_corrected_protocol.py --dataset all --k 10 --lambda-value 0.5
python src/tune_effective_weight.py --dataset all --k 10 --splits 10
python src/run_real_multipath_fusion.py --dataset all --k 10 --splits 10
python src/run_unified_normalization_validation.py --dataset all --bootstrap-iterations 2000
python src/run_support_retrieval_stress.py --dataset all --splits 10 --bootstrap-iterations 2000
python src/run_imdb_external_validation.py --bootstrap-iterations 2000
python src/plot_real_multipath_fusion.py
python src/plot_support_retrieval_stress.py
```

Some evaluations are computationally intensive. Intermediate matrices and
pairwise components are cached automatically so an interrupted workflow can
reuse completed work.

## Evidence boundary

- Training labels may be used to select component, path, and task parameters.
- Official test labels are used only after all selections are fixed.
- Test comparisons use identical candidates, queries, and tie handling.
- Node-bootstrap intervals condition on the observed graph and do not model
  dependence induced by shared edges.
- Historical-protocol diagnostics are kept separate from corrected held-out
  evaluations.
