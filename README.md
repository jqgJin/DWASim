# DWASim

This repository contains the implementation accompanying the manuscript
**“DWASim: Dynamic Weight Adjustment Similarity Method between Interactive
Entities and Their Connectivity.”**

DWASim constructs interpretable affinities between same-type entities in a
heterogeneous information network. The implementation supports global and
pair-relative normalization, path-specific combinations of support,
relative-magnitude, and directional evidence, multi-path fusion, nearest-
neighbour evaluation, and the PathSim, HeteSim, Jaccard, Bray--Curtis, cosine,
and Bhattacharyya comparison methods used by the study.

## Repository scope

The public repository intentionally contains source code and documentation
only. Benchmark records, processed matrices, cached computations, generated
metrics, and figures are not committed. They are obtained or generated
locally by the commands below and are excluded by `.gitignore`.

```text
DWASim/
├── src/                  experiment and plotting programs
├── tests/                data-independent unit tests
├── docs/
│   ├── DATASETS.md       benchmark provenance and local layout
│   └── EXPERIMENTS.md    script-to-experiment map
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment

The reference environment uses Python 3.10 on CPU. A CUDA device is not
required.

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Benchmark setup

The HGBn-ACM, HGBn-DBLP, and HGBn-IMDB records are public third-party
benchmarks from the Heterogeneous Graph Benchmark project. They are not part
of this repository. Download them from the source described in
[`docs/DATASETS.md`](docs/DATASETS.md), and place the extracted records under:

```text
data/raw/ACM/
data/raw/DBLP/
data/raw/IMDB/
```

Each dataset directory must contain `node.dat`, `link.dat`, `label.dat`,
`label.dat.test`, and `info.dat`. Convert the records into the sparse relation
matrices used by the experiment programs:

```bash
python src/prepare_hgb.py --dataset all
```

The conversion validates node counts, verifies disjoint training and test
label sets, checks reverse-relation transposes, and records source-file hashes
in locally generated manifests.

## Main experiment commands

Run the three-path component and retrieval evaluation:

```bash
python src/run_support_retrieval_stress.py --dataset all --splits 10 --bootstrap-iterations 2000
python src/plot_support_retrieval_stress.py
```

Run the matched two-path normalization and fusion evaluations:

```bash
python src/run_real_multipath_fusion.py --dataset all --k 10 --splits 10
python src/run_unified_normalization_validation.py --dataset all --bootstrap-iterations 2000
python src/plot_real_multipath_fusion.py
```

Run the external multi-label evaluation:

```bash
python src/run_imdb_external_validation.py --bootstrap-iterations 2000
```

Additional audit and development programs are documented in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

All parameter and path-weight selection is performed with training labels.
Official test labels are reserved for final held-out scoring. Tie handling and
random seeds are deterministic in the corrected evaluation programs.

## Tests

The repository includes unit tests that do not require benchmark or result
files:

```bash
python -m unittest discover -s tests -v
```

These tests cover the PathSim and HeteSim formulas, rectangular HeteSim,
deterministic ranking and voting, simplex weights, pair-relative affinities,
multi-label decisions, NDCG, and cache-key stability.

## Generated directories

The programs create the following directories when needed:

- `data/processed/` for converted sparse relations and label mappings;
- `cache/` for reusable intermediate computations;
- `results/` for generated evaluation records;
- `figures/` for generated visualizations.

These artifacts remain local and are intentionally excluded from version
control.
