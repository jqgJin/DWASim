# Repository restructuring, 2026-09-07

The public interface now consists of `prepare_data.py`, `run_experiments.py`,
and `make_figures.py`. This is a source-organization change, not a change in
the manuscript method. The number of internal files is not a scientific quality
measure: independent protocols, tests and historical sources remain separate.

## Shared implementation

Thirty-three existing definitions were extracted, with unchanged syntax trees,
from experiment programs into four shared modules:

| Module | Responsibilities |
|---|---|
| `dwasim.data` | Relations, profiles, official partitions, component caches and fold slicing |
| `dwasim.fusion` | Component mixing, path affinity and integer powers |
| `dwasim.selection` | Shared simplex grids, splits and reference-only selection |
| `dwasim.evaluation` | Deterministic ranking/voting, classification, retrieval and paired effects |

These modules do not import experiment or historical modules. The component
formulas, classical baselines, kernel solver and descriptive diagnostics have
their own library modules. Experiment programs import these functions instead
of defining a second implementation. Their re-exports preserve internal call
compatibility where needed.

## Paths and process execution

`dwasim.paths` is the common artifact-location policy. The package directory is
not an output directory. `DWASIM_WORKDIR` selects a separate research workspace;
the default is the repository root. All three entry scripts work when invoked
by absolute path from another current directory. The timing workers launch with
package module names and inherit the package search path.

## Reproducibility boundary

The migration preserves formula bodies, candidate grids, seeds, tie rules and
result schemas. It does not rewrite historical results or their source hashes.
The previous flat implementation is retained under `archive/flat_v1/`; it is not
a dependency of the current code. Historical experiments remain separately
available under `dwasim.historical` and are not run by the primary entry point.

The structure validation covers:

- unchanged syntax trees for all 33 extracted definitions;
- exact complete inner-selection comparisons for power and kernel-target fitting;
- exact comparisons of the ten original component-control families;
- all 24 retained ACM/DBLP power models' official affinities, predictions and metrics;
- reconstruction of manuscript tables with the new asset entry;
- all three entry help screens and every registered experiment's help;
- the existing 53 data-independent tests and seven additional routing tests.

This validation is not a fresh full-benchmark refit, and does not measure a new
runtime advantage. Numerical result files and manuscript PDFs remain unchanged.
Authors should upload the new repository structure as a replacement checkout,
not layer it over the old flat tree. Preserve old releases through version
history or the supplied source archive.
