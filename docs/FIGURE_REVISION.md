# Experimental presentation and figure revision

The current figures organize complementary evidence instead of repeating
absolute-score tables. No fitted model, prediction, split, or benchmark metric
is changed by the visualization revision.

## Regeneration

After completing the benchmark and regularization runs:

```powershell
python -m dwasim.reporting.assets --output manuscript_assets
```

The entry point generates conventional tables, then calls
`build_evidence_assets.py` for the current nonredundant tables and figures.
`--tables-only` also recomputes diagnostic statistics, but does not export figures.
The focused builder can be run independently with the same output option.
It reads locally prepared benchmark relations and existing model outputs; it
does not download datasets, select a new model, or upload anything.

- Figure 4: classification/retrieval scatter panels and conditional paired effects.
- Figure 5: all-pair path/component correlations, support saturation, and an exact
  empirical distribution of per-query retrieval differences.
- Figure outputs: editable PDF, SVG and EPS, plus 600 dpi PNG/TIFF, at 160 mm width.
- New diagnostic results are kept under the local generated results directory.
- All queries and every unordered training pair are retained, without sampling.

The fixed retrieval contrast is screening plus shrinkage minus original DWASim.
Every category is included in the accompanying table. Category abbreviations
are DB (databases), WC (wireless communication), DM (data mining), AI (artificial
intelligence), and IR (information retrieval). IMDB retains its separate
multi-label results; no unperformed regularization experiment is implied.

## Statistical precision correction

Previous diagnostic accumulation used single precision, which can produce
correlations above one and inaccurate variances for nearly constant components.
`evidence_diagnostics.py` promotes the exact predictor affinities to double
precision for statistical accumulation only. Undefined constant-column
correlations are null/masked, not zero. Model arithmetic is unchanged.
Historical result records remain intact; current manuscript diagnostic tables
are regenerated from the corrected analysis. The revision-control runner now
uses the stable diagnostic routine for future reruns.

Tests verify perfect/negative correlation, constant columns, numerical bounds,
invalid inputs, and conservation of all query counts. Recomputed Macro-F1 and
mean query NDCG must equal the existing benchmark records before figures export.

## Evidence and layout boundaries

These are descriptive revision-stage analyses. Correlation is not causal
importance; bootstrap intervals condition on a fixed graph and predictions.
Overlapping outer partitions are summarized once in consolidated supplementary
tables. Identical official-test outputs are noted, not counted as independent
improvements or assumed identical after outer refits. Scatter labels are offset
for readability; the data points are never jittered. ECDF curves are exact,
unsmoothed, and include every tied query.

The journal guide specifies 8–12 pt figure lettering. Current lettering is
Arial 8–11 pt with strokes at least 0.3 pt. The figure preflight's 160 mm width
warning is accepted because the manuscript uses a 160 mm text block. External
papers informed evidence organization only; no artwork or data were copied.
