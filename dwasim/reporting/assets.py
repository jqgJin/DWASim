"""Generate manuscript tables and evidence figures from completed revision runs.

Figure contract: joint performance, paired effects, and component diagnostics;
all observations retained, no coefficient-as-importance claims. Python backend,
160 mm final width, Arial 9--10 pt, editable vector text and 600 dpi raster.
"""
import argparse
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dwasim.paths import ROOT
NAMES = {"TriComponentDWASim": "DWASim", "OneComponentSelector": "Single selector",
         "FixedEqualMixture": "Fixed equal mixture", "SharedComponentMixture": "Shared mixture",
         "JointBudget60": "Joint search (60)", "NoSupport": "No support", "NoMagnitude": "No magnitude",
         "NoDirection": "No direction", "PTPNoSupport": "No PTP support", "MagnitudeOnly": "Magnitude only",
         "GlobalDWASim": "Global form", "RelativeDWASim": "Pair-relative form",
         "CardinalityCalibratedDWASim": "DWASim, two labels", "TrainingPrior": "Training prior",
         "TrainingPriorCardinality": "Prior, two labels", "EqualComponents": "Equal support/magnitude",
         "SupportOnly": "Support only", "ThreeComponents": "Equal three-component", "NativePathSim": "Native PathSim",
         "BestSinglePath": "Best single path", "UniformPathFusion": "Uniform path fusion",
         "LearnedPathFusion": "Learned path fusion"}


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def name(key):
    return NAMES.get(key, key)


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def table(folder, stem, caption, headings, rows, spec=None):
    spec = spec or ("l" + "r" * (len(headings)-1))
    content = [r"\begin{table}[!ht]", r"\caption{" + caption + r"}\label{tab:" + stem + "}",
               r"\centering\normalsize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.12}",
               r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + spec + "@{}}", r"\toprule", " & ".join(headings) + r" \\", r"\midrule"]
    content.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    content += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", r"\FloatBarrier", ""]
    (folder / f"{stem}.tex").write_text("\n".join(content), encoding="utf-8")


def generate_tables(output, controls):
    folder = output / "revision_tables"
    folder.mkdir(parents=True, exist_ok=True)
    stress = {row["dataset"]: row for row in read("support_retrieval_stress.json")["rows"]}
    rows = []
    for method in ("TriComponentDWASim", "Cosine", "MagnitudeOnly", "Jaccard", "Bhattacharyya", "PathSim", "HeteSim"):
        row = [name(method)]
        for dataset in ("ACM", "DBLP"):
            m = stress[dataset]["test_metrics"][method]
            row += [f'{m["macro_f1"]:.4f}', f'{m["ndcg_at_10"]:.4f}']
        rows.append(row)
    table(folder, "main_benchmarks", "Official three-path test performance. Profile affinities use complete profiles; PathSim and HeteSim retain native half-path definitions. NDCG denotes NDCG@10. Higher is better; no significance ranking is implied.", ["Method", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows)
    rows = []
    for method in ("OneComponentSelector", "FixedEqualMixture", "SharedComponentMixture", "JointBudget60", "NoSupport", "NoMagnitude", "NoDirection", "PTPNoSupport"):
        row = [name(method), controls["ACM"]["selection"][method]["candidate_count"]]
        for dataset in ("ACM", "DBLP"):
            m = controls[dataset]["test_metrics"].get(method)
            row += [f'{m["macro_f1"]:.4f}', f'{m["ndcg_at_10"]:.4f}'] if m else ["--", "--"]
        rows.append(row)
    table(folder, "matched_controls", "Restricted-selection and retuned-deletion controls. Single selector denotes per-path single-component selection. Budget counts candidate configurations, each evaluated on ten internal partitions. DWASim uses 60; cosine uses 15. No PTP support applies only to ACM.", ["Control", "Budget", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows)
    rows = []
    for method in ("TriComponentDWASim", "Cosine", "OneComponentSelector", "NoSupport"):
        row = [name(method)]
        for dataset in ("ACM", "DBLP"):
            m = controls[dataset]["outer_summary"][method]["macro_f1"]
            row.append(f'${m["mean"]:.4f}\\pm{m["sample_sd"]:.4f}$')
        rows.append(row)
    table(folder, "outer_stability", "Macro-F1 after complete reselection within five outer training partitions (mean and sample standard deviation). Partitions overlap; standard deviations are descriptive partition variability.", ["Method", "ACM", "DBLP"], rows)
    representations = {ds: read(f"representation_controls_{ds}.json") for ds in controls}
    rows = []
    for representation in ("complete", "half"):
        for method in ("TriComponentDWASim", "Cosine", "MagnitudeOnly", "OneComponentSelector"):
            row = [representation.capitalize(), name(method)]
            for ds in controls:
                m = representations[ds]["representation"][representation][method]["metrics"]
                row += [f'{m["macro_f1"]:.4f}', f'{m["ndcg_at_10"]:.4f}']
            rows.append(row)
    table(folder, "representation", "Representation control with fixed path schemas and training-only selection. Half profiles end at the middle node type; complete profiles end at the original source type. Both metrics are official-test values; representations were not selected by their test scores.", ["Profile", "Method", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows, spec="llrrrr")
    rows = []
    for first, second in (("H", "GlobalL"), ("H", "Bray"), ("J", "GlobalL"), ("J", "Bray")):
        row = [f'${first}$', r'$L/B_1$' if second == "GlobalL" else "$B$"]
        for ds in controls:
            m = representations[ds]["factorial"][f"{first}_{second}"]["metrics"]
            row += [f'{m["macro_f1"]:.4f}', f'{m["ndcg_at_10"]:.4f}']
        rows.append(row)
    table(folder, "factorial", "Discrepancy-by-magnitude-scale diagnostic using the original two complete paths. H is normalized by its path-wide maximum; J and B are pair-relative. Component and two-path weights use the same 0.1 grid in all four settings.", ["First term", "Magnitude", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows)
    imdb = read("imdb_external_validation.json")
    imethods = ("TriComponentDWASim", "Cosine", "OneComponentSelector", "MagnitudeOnly", "FixedEqualMixture", "GlobalDWASim", "PathSim", "HeteSim", "CardinalityCalibratedDWASim", "TrainingPrior", "TrainingPriorCardinality")
    rows = []
    for method in imethods:
        m = imdb["external_metrics"][method]
        rows.append([name(method)] + [f'{m[k]:.4f}' for k in ("macro_f1", "micro_f1", "subset_accuracy", "hamming_loss")] + [f'{m["predicted_mean_cardinality"]:.2f}'])
    table(folder, "imdb", "Official IMDB multi-label evaluation. F1 values are macro/micro averages; exact denotes exact label-set accuracy. Lower Hamming loss is better. Count is the mean number of predicted labels; the observed mean is 1.73. Precision and recall appear in the SI.", ["Method", "Macro", "Micro", "Exact", "Hamming", "Count"], rows)
    m, c = [imdb["external_metrics"][k] for k in ("TriComponentDWASim", "CardinalityCalibratedDWASim")]
    text = (f'Three-component DWASim reaches {m["macro_f1"]:.4f} Macro-F1 and {m["micro_f1"]:.4f} Micro-F1. '
            f'Its held-out predictions are identical to cosine and single-component selection. Macro-precision is {m["macro_precision"]:.4f} and macro-recall is {m["macro_recall"]:.4f}; '
            f'the mean predicted label count is {m["predicted_mean_cardinality"]:.2f}, compared with {m["true_mean_cardinality"]:.2f} observed. '
            f'Fixing the prediction count at two changes Macro-F1 to {c["macro_f1"]:.4f}, exact accuracy to {c["subset_accuracy"]:.4f}, and Hamming loss to {c["hamming_loss"]:.4f}. '
            'Thus, applying the same construction across tasks does not establish an incremental mixture benefit.\n')
    (folder / "imdb_interpretation.tex").write_text(text, encoding="utf-8")
    simulation = read("controlled_mechanisms.json")
    rows = []
    methods = ("TriComponentDWASim", "Cosine", "MagnitudeOnly", "Jaccard", "OneComponentSelector", "FixedEqualMixture")
    for scenario in simulation["protocol"]["scenarios"]:
        row = [scenario]
        for method in methods:
            scores = [r["methods"][method]["metrics"]["macro_f1"] for r in simulation["runs"] if r["scenario"] == scenario]
            row.append(f'${np.mean(scores):.3f}\\pm{np.std(scores, ddof=1):.3f}$')
        rows.append(row)
    # Wide mechanism data are split into two coherent panels to retain readable text.
    table(folder, "mechanisms", "Controlled mechanism Macro-F1 across ten independently generated profile collections (mean and sample standard deviation). These deliberately identifiable regimes often reach ceiling performance.", ["Regime", "DWASim", "Cosine", "Magnitude"], [r[:4] for r in rows])
    table(folder, "mechanisms_other", "Additional controlled-mechanism baselines using the same observations and protocol. Selector chooses one component per path; equal fixes the three component weights to one third.", ["Regime", "Jaccard", "Selector", "Equal"], [[r[0]]+r[4:] for r in rows])
    cost = read("revision_cost.json")
    table(folder, "cost", "Query-batch comparison-stage cost on DBLP APTPA: median seconds over five repetitions and whole-process peak resident memory (MiB). Profiles are loaded before timing. Native PathSim uses 7,723-dimensional half profiles; other rows use 4,057-dimensional complete profiles.", ["Method", "Candidates", "Seconds", "Peak MiB"], [[name(r["method"]), r["candidates"], f'{r["median_seconds"]:.4f}', f'{r["process_peak_MiB"]:.1f}'] for r in cost["rows"]])
    # Supplementary selections and mechanism diagnostics.
    rows, diag, single = [], [], []
    for ds, record in controls.items():
        model = record["selection"]["TriComponentDWASim"]
        for i, path in enumerate(record["paths"]):
            rows.append([ds, path]+[f'{v:.2f}' for v in model["component_weights"][i]]+[f'{model["path_weights"][i]:.2f}'])
            d = record["component_diagnostics"][path]
            diag.append([ds, path, f'{100*d["nonzero_density"]:.5f}', f'{100*d["support_affinity_equal_one_fraction"]:.2f}']+[scientific(v) for v in d["component_variance"]])
            for method, m in record["single_path_metrics"][path].items():
                single.append([ds, path, name(method), f'{m["macro_f1"]:.4f}', f'{m["ndcg_at_10"]:.4f}'])
    table(folder, "selected_weights", "Main-method component and path weights from official training labels. Zero path weight removes that path regardless of its within-path selection.", ["Dataset", "Path", r'$\theta_s$', r'$\theta_m$', r'$\theta_d$', "$w$"], rows, spec="llrrrr")
    table(folder, "component_diagnostics", "Complete-profile nonzero density and fraction of unordered training pairs with unit support affinity (percent), followed by support, magnitude, and directional affinity variances. Coefficients cannot identify importance when a component is nearly constant.", ["Data", "Path", "Density", r"\shortstack{Unit\\support}", r'$v_s$', r'$v_m$', r'$v_d$'], diag, spec="llrrrrr")
    for ds in controls:
        table(folder, f"single_paths_{ds}", f"{ds} single-path official-test metrics, before across-path fusion. Classical methods retain native half-path inputs.", ["Dataset", "Path", "Method", "Macro-F1", "NDCG@10"], [row for row in single if row[0] == ds], spec="lllrr")
        corr = []
        for path, d in controls[ds]["component_diagnostics"].items():
            a = np.asarray(d["component_correlation"])
            corr.append([path, f'{a[0,1]:.4f}', f'{a[0,2]:.4f}', f'{a[1,2]:.4f}'])
        table(folder, f"correlation_{ds}", f"{ds} component-affinity Pearson correlations over unordered training pairs. A zero-variance component is assigned zero correlation as a computational placeholder, not as evidence of independence.", ["Path", r"\shortstack{Support/\\magnitude}", r"\shortstack{Support/\\direction}", r"\shortstack{Magnitude/\\direction}"], corr)
    table(folder, "imdb_precision_recall", "IMDB precision and recall. All methods use five binary genre indicators; no averaging over mutually exclusive categories is performed.", ["Method", "Macro-P", "Macro-R", "Micro-P", "Micro-R"], [[name(k)]+[f'{imdb["external_metrics"][k][metric]:.4f}' for metric in ("macro_precision", "macro_recall", "micro_precision", "micro_recall")] for k in imethods])
    table(folder, "imdb_categories", "Genre-specific IMDB F1. Labels are the actual benchmark genre names.", ["Genre", "DWASim", "Magnitude", "PathSim", "HeteSim"], [[genre]+[f'{imdb["external_metrics"][k]["per_class_f1"][str(i)]:.4f}' for k in ("TriComponentDWASim", "MagnitudeOnly", "PathSim", "HeteSim")] for i, genre in enumerate(imdb["classes"])])
    model = imdb["selections"]["TriComponentDWASim"]
    table(folder, "imdb_weights", "IMDB selected component and path weights. The actor path has zero fusion weight. Rank calibration can make different component choices yield identical downstream rankings.", ["Path", r'$\theta_s$', r'$\theta_m$', r'$\theta_d$', "$w$"], [[path]+[f'{v:.2f}' for v in model["selected_component_weights"][path]]+[f'{model["selected"]["path_weights"][i]:.2f}'] for i, path in enumerate(imdb["paths"])])


def generate_review_tables(output):
    folder = output / "revision_tables"
    fusion = {ds: read(f"path_fusion_audit_{ds}.json") for ds in ("ACM", "DBLP")}
    rows, outer, effects = [], [], []
    for method in ("BestSinglePath", "UniformPathFusion", "LearnedPathFusion"):
        row, variability = [name(method)], [name(method)]
        for ds, record in fusion.items():
            metrics = record["test_metrics"][method]
            row += [f'{metrics["macro_f1"]:.4f}', f'{metrics["ndcg_at_10"]:.4f}']
            for endpoint in ("macro_f1", "ndcg_at_10"):
                item = record["outer_summary"][method][endpoint]
                variability += [f'${item["mean"]:.4f}\\pm{item["sample_sd"]:.4f}$']
        rows.append(row)
        outer.append(variability)
    table(folder, "path_fusion", "Path-fusion controls with shared within-path component weights. Best single path is selected using training validation; uniform fusion fixes all three weights to one third. Learned fusion is the main DWASim model. F1 denotes Macro-F1 and NDCG denotes NDCG@10.", ["Fusion rule", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows)
    # Two metrics with mean/SD pairs are separated to avoid a wide, small-font table.
    for ds_index, ds in enumerate(fusion):
        table(folder, f"fusion_outer_{ds}", f"{ds} path-fusion results after complete reselection in five outer training partitions: mean and sample standard deviation. Overlapping partitions measure label-partition sensitivity, not independent graph replication.", ["Fusion rule", "Macro-F1", "NDCG@10"], [[r[0]] + r[1+ds_index*2:3+ds_index*2] for r in outer])
        effects = []
        for comparator in ("BestSinglePath", "UniformPathFusion"):
            for endpoint, label in (("macro_f1", "Macro-F1"), ("ndcg_at_k", "NDCG@10")):
                value = fusion[ds]["paired_learned_minus"][comparator][endpoint]
                effects.append([name(comparator), label, f'{value["difference"]:.4f}', f'[{value["lower_95"]:.4f}, {value["upper_95"]:.4f}]'])
        table(folder, f"fusion_effects_{ds}", f"{ds} learned-fusion minus comparator differences. Intervals use 2,000 paired category-stratified node resamples of frozen predictions and condition on this graph and selected model. They are descriptive, without multiple-comparison correction.", ["Comparator", "Endpoint", "Difference", "95\\% interval"], effects)
    rows = []
    stages = (("Profile construction", "profile_seconds"), ("Training affinities", "training_components_seconds"),
              ("Internal selection", "selection_seconds"), ("Query affinities", "query_components_seconds"),
              ("Ranking and voting", "ranking_prediction_seconds"), ("Total pipeline", "pipeline_wall_seconds"))
    for ds in ("ACM", "DBLP", "IMDB"):
        records = {method: read(f"pipeline_cost_{ds}_{method}.json") for method in ("DWASim", "Cosine")}
        for method, r in records.items():
            if not all(r["pruning_checks"].values()):
                raise ValueError(f"Inference identity check failed: {ds} {method}")
            rows.append([ds, method, f'{r["pipeline_wall_seconds"]:.2f}', f'{r["query_components_seconds"]:.2f}', f'{r["active_query_seconds"]:.2f}', f'{r["pipeline_peak_MiB"]/1024:.2f}'])
        table(folder, f"pipeline_stages_{ds}", f"{ds} full-pipeline stages in seconds, measured once per method in isolated processes with two computational threads. Total includes small orchestration and memory-release overheads beyond the listed stages.", ["Stage", "DWASim", "Cosine"], [[label]+[f'{records[m][key]:.3f}' for m in records] for label, key in stages])
    table(folder, "pipeline_summary", "Complete-pipeline and query-affinity costs in seconds. Total includes profile construction, training affinities, selection, query affinities, and prediction. Query computes all component/path inputs for the method; active query omits exactly zero-weight terms after selection. Peak is the whole-process high-water memory mark before the extra active-query audit. Each entry is one observed run.", ["Data", "Method", "Total (s)", "Query (s)", r"\shortstack{Active\\query (s)}", r"\shortstack{Peak\\(GiB)}"], rows, spec="llrrrr")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript_assets")
    parser.add_argument("--tables-only", action="store_true", help="Regenerate tables without changing approved figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
        "axes.linewidth": .6, "lines.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False, "savefig.dpi": 600})
    controls = {ds: read(f"revision_controls_{ds}.json") for ds in ("ACM", "DBLP")}
    generate_tables(args.output, controls)
    generate_review_tables(args.output)
    if all((ROOT / "results" / f"regularized_selection_{ds}.json").exists() for ds in ("ACM", "DBLP")):
        from dwasim.reporting.regularization import generate
        generate(args.output)
    from dwasim.reporting.evidence import generate as generate_evidence
    generate_evidence(args.output, figures=not args.tables_only)
    if all((ROOT / "results" / kind / f"{kind}_{ds}.json").exists()
           for kind in ("kernel_power", "kernel_target") for ds in ("ACM", "DBLP")):
        from dwasim.reporting.kernels import generate as generate_kernel
        generate_kernel(args.output)
    print("Generated manuscript assets:", args.output)


if __name__ == "__main__":
    main()
