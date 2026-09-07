# Code maintenance: 2026-09-07

This pass improves readability in the kernel-target solver and the two kernel
experiment runners. Dense nested expressions were expanded into explicit loops,
reference and validation indices were named consistently, and the power-candidate
tie rule was given a named function. Public interfaces, search grids, seeds,
reference-only fitting boundaries, result schemas, and integrity checks remain.

This is a maintenance refactor, not a new algorithm or experiment. Development
included AI assistance; the refactor does not change that provenance. Authors
remain responsible for reviewing the code and reporting actual tool use.

The exact three source files used before this refactor are retained under
`archive/flat_v1/archive/kernel_extension_v1/`. Source fingerprints in historical experiment
records refer to those versions, not the newly formatted/refactored files.
Recorded experiment outputs and their source fingerprints must not be rewritten
to make them appear to originate from a later source version. To rerun the
archived implementation, use a separate project copy and replace its three
matching source files with the archived versions before following the protocol.

Validation for this maintenance pass compares old and new reference statistics,
all thirty kernel-target solver configurations, complete inner selection on a
deterministic fixture, and all 24 retained ACM/DBLP power-model official-query
affinity matrices. It also runs the local and code-only test suites. This does
not constitute a fresh full-benchmark refit; runtime measurements are excluded
from equality comparisons. No manuscript statement or experimental result is
changed by this maintenance pass.
