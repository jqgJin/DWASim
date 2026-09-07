"""Tables for the frozen regularization experiment; no figure modifications."""
from pathlib import Path
import argparse
import json

from dwasim.paths import ROOT
NAMES = {"Original": "Original DWASim", "Uniform": "Uniform fusion", "Shrinkage": "Shrinkage only",
         "Screened": "Screening only", "ScreenedUniform": "Screened uniform", "ScreenedShrinkage": "Screening + shrinkage",
         "Cosine": "Cosine", "SingleSelector": "Single selector", "CosineShrinkage": "Cosine + shrinkage",
         "SingleSelectorShrinkage": "Selector + shrinkage"}


def table(folder, stem, caption, headings, rows, spec=None):
    spec = spec or "l" + "r"*(len(headings)-1)
    lines = [r"\begin{table}[!ht]", r"\caption{"+caption+r"}\label{tab:"+stem+"}",
             r"\centering\normalsize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}"+spec+"@{}}", r"\toprule",
             " & ".join(headings)+r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row))+r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", r"\FloatBarrier", ""]
    (folder / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")


def generate(output, results=ROOT / "results"):
    data = {ds: json.loads((results / f"regularized_selection_{ds}.json").read_text(encoding="utf-8")) for ds in ("ACM", "DBLP")}
    folder = output / "revision_tables"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ("Original", "Uniform", "Shrinkage", "Screened", "ScreenedShrinkage", "CosineShrinkage", "SingleSelectorShrinkage"):
        values = [NAMES[method]]
        for ds in data:
            metric = data[ds]["test_metrics"][method]
            values += [f'{metric["macro_f1"]:.4f}', f'{metric["ndcg_at_10"]:.4f}']
        rows.append(values)
    table(folder, "regularized_main", "Component screening and path-weight shrinkage on the official test partitions. NDCG denotes NDCG@10. Both rules use a fixed internal Macro-F1 tolerance of 0.005. The same shrinkage rule is applied to the simpler controls; no test-based method selection is performed.",
          ["Selection rule", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows)
    rows = []
    for method in NAMES:
        values = [NAMES[method]]
        for ds in data:
            model = data[ds]["fit"]["models"][method]
            values += [f'{model["alpha"]:.2f}', model["active_terms"]]
        values += [data["ACM"]["fit"]["models"][method]["candidate_evaluations"]]
        rows.append(values)
    table(folder, "regularized_selection", r"Final training-selected shrinkage strengths and nonzero joint terms. Terms count nonzero $w_k\theta_{kj}$, not independent sources of information. Budget counts candidate evaluations, each on ten internal partitions. Uniform rules have fixed $\alpha=1$; unshrunk rules have $\alpha=0$.",
          ["Rule", r"ACM $\alpha$", "Terms", r"DBLP $\alpha$", "Terms", "Budget"], rows)
    for ds, record in data.items():
        rows = [[NAMES[m], f'{record["test_metrics"][m]["macro_f1"]:.4f}', f'{record["test_metrics"][m]["ndcg_at_10"]:.4f}'] for m in NAMES]
        table(folder, f"regularized_all_{ds}", f"{ds} official-test results for all ten frozen selection rules. All candidates are retained, including rules whose scores coincide.", ["Rule", "Macro-F1", "NDCG@10"], rows)
        rows = []
        for m in NAMES:
            values = [NAMES[m]]
            for metric in ("macro_f1", "ndcg_at_10"):
                s = record["outer_summary"][m][metric]
                values.append(f'${s["mean"]:.4f}\\pm{s["sample_sd"]:.4f}$')
            rows.append(values)
        table(folder, f"regularized_outer_{ds}", f"{ds} regularization results after complete reselection in five overlapping outer training partitions: mean and sample standard deviation. These summarize partition sensitivity, not independent graph replication.", ["Rule", "Macro-F1", "NDCG@10"], rows)
        rows = []
        for m in ("Original", "Uniform", "Cosine", "SingleSelector"):
            for metric, label in (("macro_f1", "Macro-F1"), ("ndcg_at_k", "NDCG@10")):
                e = record["paired_combined_minus"][m][metric]
                rows.append([NAMES[m], label, f'{e["difference"]:.4f}', f'[{e["lower_95"]:.4f}, {e["upper_95"]:.4f}]'])
        table(folder, f"regularized_effects_{ds}", f"{ds} screening-plus-shrinkage minus comparator effects. Intervals use 2,000 paired category-stratified query resamples, are conditional on fixed predictions, and are not multiplicity-adjusted. They exclude graph and selection uncertainty.", ["Comparator", "Endpoint", "Difference", r"95\% interval"], rows, "llrr")
        rows = []
        for r in record["outer"]:
            values = [r["seed"]]
            values += [f'{r["metrics"][m]["macro_f1"]:.4f}' for m in ("Original", "ScreenedShrinkage", "CosineShrinkage", "SingleSelectorShrinkage")]
            rows.append(values)
        table(folder, f"regularized_partitions_{ds}", f"{ds} paired outer Macro-F1 values. Combined denotes screening plus shrinkage; cosine and selector columns both include shrinkage. Every partition refits all selection stages.", ["Seed", "Original", "Combined", "Cosine", "Selector"], rows)
    print("Generated regularization tables", folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript_assets")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    generate(args.output, args.results)
