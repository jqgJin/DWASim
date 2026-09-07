# Running the structured repository

Only three root entry scripts are needed. Each experiment retains its original
parser, default grid, output schema and training/test boundary. Run `--help`
after an experiment name to see its options. IMDB is a separate multi-label
protocol: `main --dataset all` means ACM and DBLP, not IMDB.

## Complete current manuscript workflow

Prepare HGB data as described in DATASETS.md. Set `DWASIM_WORKDIR` first if data
and outputs belong in a separate workspace. Run from the repository root:

```bash
python prepare_data.py --dataset all
python run_experiments.py main --dataset all --splits 10 --bootstrap-iterations 2000
python run_experiments.py imdb --bootstrap-iterations 2000
python run_experiments.py ablation --dataset all --outer-repeats 5 --bootstrap-iterations 2000
python run_experiments.py representation
python run_experiments.py mechanisms
python run_experiments.py paths --dataset all --outer-repeats 5 --bootstrap-iterations 2000
python run_experiments.py regularization --dataset all
python run_experiments.py kernel-target --dataset all
python run_experiments.py power --dataset all
python run_experiments.py kernel-cost
python run_experiments.py cost --dataset all --method all
python make_figures.py
```

These are expensive research runs, not installation tests. Run timing protocols
alone, without other intensive work. Power calibration checks its p=1 outputs
against completed regularization records, so regularization must run first.
The cost protocol checks its final predictions against completed main and IMDB
records. The two extension protocols preserve completed/partial outputs and
require a new output directory for a new run.

Some earlier runners retain their original overwrite behavior. For a fresh
experiment, use a separate workspace rather than rerunning over a completed
research archive. Parameter options are not normalized across distinct
protocols: unsupported options produce parser errors rather than being ignored.
The fixed `representation` and `mechanisms` protocols take no options.

`make_figures.py --tables-only` regenerates tables without redrawing figures.
`--output PATH` directs the assets to another folder. The figures are built from
completed experiment records, not from values manually entered in the plotting
program. Original conceptual diagrams are manuscript assets and are not included
in the code-only repository.

## Additional analyses

| Experiment | Purpose |
|---|---|
| `single-path` | Earlier corrected held-out single-path protocol |
| `normalization` | Two-path normalization/fusion comparison |
| `normalization-controls` | Additional matched normalization controls |
| `development` | Earlier nested development analysis |

These are retained for traceability and do not replace the current main
experiment. Detailed original protocol notes remain in EXPERIMENTS.md and the
regularization/kernel protocol documents; their module commands are supported.

## Historical runs

Historical protocols and older plotting routines are explicitly separated in
`dwasim.historical`. For example:

```bash
python -m dwasim.historical.original --help
python -m dwasim.historical.controlled
python -m dwasim.historical.multipath
```

The last two run fixed historical simulations; they are not installation checks
and do not use the current benchmark protocol. Run controlled before multipath
so its output folder exists. Historical query self-retrieval and seed searches
are not part of the current main evaluation and must not support current
held-out performance claims.

`archive/flat_v1/` contains the byte-identical pre-migration source snapshot.
Current code never imports it. To reproduce that version with its original
layout, copy `archive/flat_v1` into a separate workspace, install the same
dependencies, and supply the required data there. The nested archived kernel
versions identify the source hashes used by the original extension experiments.

## Smoke tests

```bash
python prepare_data.py --help
python run_experiments.py --help
python run_experiments.py power --help
python make_figures.py --help
python -m unittest discover -s tests -v
```

Help commands and tests do not require benchmark downloads or generate research
results. Merely importing the historical multipath module no longer starts its
experiment; execution is now behind a standard main guard.
