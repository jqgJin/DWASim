# DWASim

Code accompanying **DWASim: Dynamic Weight Adjustment Similarity Method between
Interactive Entities and Their Connectivity**.

DWASim compares typed interaction profiles using support, relative magnitude,
and direction, then selects and combines semantic paths. The repository contains
the original method, matched controls, and the kernel-power and kernel-target
extensions. A mathematical guarantee is not a claim of universal prediction gains.

## Start here

Use Python 3.10 and install `requirements.txt` (or `requirements-lock.txt` for
the recorded reference environment). All experiments run on CPU.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python prepare_data.py --help
python run_experiments.py --help
python make_figures.py --help
```

Activate the virtual environment before installing and running. Only three
root-level programs are needed:

| Entry | Purpose |
|---|---|
| `prepare_data.py` | Convert externally downloaded benchmark records |
| `run_experiments.py` | Select a named experiment and pass its options |
| `make_figures.py` | Generate the current manuscript tables and figures |

Download ACM, DBLP, and IMDB from the third-party HGB source described in
[DATASETS](docs/DATASETS.md). This repository does not distribute datasets,
prediction records, caches, figures, or manuscript PDFs.

```bash
python prepare_data.py --dataset all
python run_experiments.py main --dataset all --splits 10 --bootstrap-iterations 2000
python run_experiments.py imdb --bootstrap-iterations 2000
```

For the complete manuscript, follow the dependency order in
[RUNNING](docs/RUNNING.md); the commands above alone do not generate every
supplementary analysis. To inspect an experiment's own options:

```bash
python run_experiments.py power --help
```

## Structure

```text
dwasim/
  data.py             relations, profiles, partitions and component caches
  datasets.py         raw HGB conversion and validation
  similarity.py       DWASim component formulas and exact active inference
  baselines.py        PathSim and HeteSim definitions
  fusion.py           component mixing and path calibration
  selection.py        shared grids and reference-only model selection
  kernels.py          regularized centered-kernel solver
  evaluation.py       ranking, voting, retrieval and paired effects
  diagnostics.py      descriptive component/query diagnostics
  experiments/        single-label, multi-label and control protocols
  reporting/          manuscript tables and figures
  historical/         earlier protocols and visualizations, not default runs
tests/                data-independent regression tests
docs/                 data, experiment and theory documentation
archive/flat_v1/      byte-identical pre-migration sources for provenance
```

Current numerical routines have explicit package imports; shared data,
selection, fusion and evaluation functions no longer live inside experiment
entry scripts. Protocol-specific implementations remain separate, particularly
IMDB and serial timing. The archive is not imported by current code.

## Local artifacts

By default, `data/`, `cache/`, `results/`, and `manuscript_assets/` are located
beside the entry scripts. To use another workspace, set `DWASIM_WORKDIR` before
starting Python. For example, in PowerShell:

```powershell
$env:DWASIM_WORKDIR = 'D:\research\dwasim_runs'
python run_experiments.py power --help
```

The chosen workspace must contain the required inputs and earlier experiment
records. This setting relocates artifacts; it does not download data or change
the protocol. Existing outputs should be preserved. Use a separate workspace or
an experiment's supported output option for a new run.

## Validation and provenance

```bash
python -m unittest discover -s tests -v
python make_figures.py --tables-only
```

The second command needs completed experiment records. Tests need neither
benchmark data nor recorded results. The restructuring preserves equations,
grids, seed order, tie rules and output schemas; validation details appear in
[STRUCTURE](docs/STRUCTURE.md). Archive fingerprints identify earlier source
versions, not the reorganized implementation.

Development included AI assistance. Code organization does not change that
provenance; authors remain responsible for verification and accurate reporting.
