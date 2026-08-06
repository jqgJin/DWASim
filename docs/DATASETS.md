# Benchmark datasets

## Source and ownership

The implementation uses the public HGBn-ACM, HGBn-DBLP, and HGBn-IMDB
benchmarks distributed by the Heterogeneous Graph Benchmark (HGB) project:

- HGB repository: <https://github.com/THUDM/HGB>
- Public benchmark folder linked by HGB:
  <https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX>

The datasets are third-party resources and are not created, redistributed, or
claimed by this repository. OpenHGNN is a separate software project that can
load heterogeneous graph benchmarks; it is not the source of the DWASim
method or implementation.

## Local directory layout

After downloading and extracting the official archives, use this layout:

```text
data/
└── raw/
    ├── ACM/
    │   ├── node.dat
    │   ├── link.dat
    │   ├── label.dat
    │   ├── label.dat.test
    │   └── info.dat
    ├── DBLP/
    │   └── ... same required files ...
    └── IMDB/
        └── ... same required files ...
```

Do not commit the downloaded archives or extracted records. The repository's
`.gitignore` excludes the complete `data/` tree.

## Deterministic conversion

Run:

```bash
python src/prepare_hgb.py --dataset all
```

The converter reads the official global identifiers and creates one sparse
matrix for each typed relation under `data/processed/<DATASET>/`. It preserves
the official training/test label partitions and keeps IMDB as a multi-label
task. The generated manifest records relation shapes, source hashes, label
counts, and reverse-relation consistency checks.

The processed matrices and manifests are derived artifacts. They are created
locally and are not included in the GitHub repository.
