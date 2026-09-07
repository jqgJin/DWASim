"""Evidence-led experimental figures and nonredundant manuscript tables.

All official queries and unordered training pairs are retained. Correlation is
descriptive, not feature importance. Existing fitted models are never modified.
Figure width: 160 mm; Arial 8--12 pt; editable vector and 600 dpi raster exports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score
from dwasim.diagnostics import affinity_statistics, nullable, query_effect_summary

from dwasim.paths import ROOT
DATASETS = ("ACM", "DBLP")
COLORS = {"ACM": "#294F70", "DBLP": "#A46027"}
CATEGORIES = {"ACM": ("DB", "WC", "DM"), "DBLP": ("DB", "DM", "AI", "IR")}
RULES = {"Original": "Original DWASim", "Uniform": "Uniform fusion", "Shrinkage": "Shrinkage only",
         "Screened": "Screening only", "ScreenedUniform": "Screened uniform", "ScreenedShrinkage": "Screening + shrinkage",
         "Cosine": "Cosine", "SingleSelector": "Single selector", "CosineShrinkage": "Cosine + shrinkage",
         "SingleSelectorShrinkage": "Selector + shrinkage"}


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_evidence():
    from dwasim.data import load_split
    from dwasim.data import load_components, STRESS_PATHS
    from dwasim.data import full_profile
    report = {"protocol": {"accumulation_dtype": "float64", "training_pairs": "all unordered, diagonal excluded",
              "correlation": "Pearson; constant columns undefined", "query_contrast": "ScreenedShrinkage minus Original",
              "query_exclusion_count": 0, "pair_sampling": False}, "datasets": {}}
    for ds in DATASETS:
        ids, labels, test_ids, truth = load_split(ds)
        upper = np.triu_indices(len(ids), k=1)
        columns, diagnostics = [], {}
        for path in STRESS_PATHS[ds]:
            components = load_components(ds, path)
            # Preserve the exact float32 affinity values used by the predictor;
            # promote only the statistical accumulation, not model arithmetic.
            values = np.stack([(1-components["train_jaccard"])[upper],
                               (1-components["train_bray"])[upper], components["train_cosine"][upper]], axis=1)
            variance, correlation = affinity_statistics(values)
            columns.append(values.astype(np.float64))
            profile = full_profile(ds, path)
            diagnostics[path] = {"component_variance": variance.tolist(), "component_correlation": nullable(correlation),
                "support_affinity_equal_one_fraction": float(np.mean(values[:, 0] == 1)),
                "training_unordered_pairs": len(values), "nonzero_density": float(profile.nnz / np.prod(profile.shape)),
                "full_support_rows": int(np.sum(np.diff(profile.indptr) == profile.shape[1])),
                "complete_profile_shape": list(profile.shape),
                "component_arrays_identical": bool(np.array_equal(values[:,0],values[:,1]) and np.array_equal(values[:,0],values[:,2]))}
        _, correlation = affinity_statistics(np.concatenate(columns, axis=1))
        record = read(f"regularized_selection_{ds}.json")
        path = ROOT / "results" / f"regularized_selection_{ds}.predictions.npz"
        with np.load(path) as predictions:
            np.testing.assert_array_equal(test_ids, predictions["test_ids"])
            np.testing.assert_array_equal(truth, predictions["truth"])
            for method, metric in record["test_metrics"].items():
                assert abs(f1_score(truth, predictions[f"prediction_{method}"], average="macro", zero_division=0)-metric["macro_f1"]) < 1e-12
                assert abs(predictions[f"ndcg_{method}"].mean()-metric["ndcg_at_10"]) < 1e-12
            for a,b in (("Uniform","Shrinkage"),("Screened","SingleSelector"),
                        ("ScreenedShrinkage","ScreenedUniform"),("ScreenedShrinkage","SingleSelectorShrinkage")):
                for prefix in ("prediction_","ndcg_"):
                    np.testing.assert_array_equal(predictions[prefix+a],predictions[prefix+b])
            delta = predictions["ndcg_ScreenedShrinkage"]-predictions["ndcg_Original"]
            category = {name: query_effect_summary(delta[truth == i]) for i, name in enumerate(CATEGORIES[ds])}
        report["datasets"][ds] = {"paths": list(STRESS_PATHS[ds]), "training_count": len(ids), "test_count": len(test_ids),
            "component_diagnostics": diagnostics, "cross_component_correlation": nullable(correlation),
            "query_effect": query_effect_summary(delta), "categories": category,
            "prediction_sha256": sha(path), "result_sha256": sha(ROOT / "results" / f"regularized_selection_{ds}.json")}
        report["datasets"][ds]["source_record_hashes"] = {name: sha(ROOT / "results" / name) for name in
            (f"revision_controls_{ds}.json",f"path_fusion_audit_{ds}.json",f"representation_controls_{ds}.json")}
        print(ds, "all-pair diagnostic and query audit complete", report["datasets"][ds]["query_effect"], flush=True)
    return report


def generate_tables(output, evidence):
    from dwasim.reporting.assets import table, scientific
    folder = output / "revision_tables"
    folder.mkdir(parents=True, exist_ok=True)
    regular = {ds: read(f"regularized_selection_{ds}.json") for ds in DATASETS}
    fusion = {ds: read(f"path_fusion_audit_{ds}.json") for ds in DATASETS}
    controls = {ds: read(f"revision_controls_{ds}.json") for ds in DATASETS}
    rows = []
    # One main-text table for path fusion and complexity control. Coincident
    # official-test predictions are documented, not hidden or counted as gains.
    for method, label in (("BestSinglePath", "Best single path"), ("Original", "Original DWASim"),
                          ("Uniform", r"Uniform / shrinkage$^{a}$"), ("Screened", r"Screening$^{b}$"),
                          ("ScreenedShrinkage", r"Screening + shrinkage$^{c}$"), ("CosineShrinkage", "Cosine + shrinkage")):
        row = [label]
        for ds in DATASETS:
            m = (fusion[ds] if method == "BestSinglePath" else regular[ds])["test_metrics"][method]
            row += [f'{m["macro_f1"]:.4f}', f'{m["ndcg_at_10"]:.4f}']
        rows.append(row)
    table(folder, "fusion_complexity", "Path fusion and complexity control on official test partitions. F1 is Macro-F1; NDCG is NDCG@10. All rules are selected using training labels.",
          ["Rule", "ACM F1", "NDCG", "DBLP F1", "NDCG"], rows)
    # Caption notes describe exact prediction equivalence only on official tests.
    p = folder / "fusion_complexity.tex"
    content = p.read_text(encoding="utf-8").replace(r"\end{tabular*}", r"\end{tabular*}" + "\n" +
        r"\par\vspace{4pt}\begin{minipage}{\linewidth}\normalsize " +
        r"$^{a}$Uniform fusion and shrinkage-only predictions coincide here. " +
        r"$^{b}$Screening matches the single selector. " +
        r"$^{c}$The combined rule matches screened uniform fusion and selector plus shrinkage. " +
        r"These identities do not imply equivalence after outer reselection.\end{minipage}")
    p.write_text(content, encoding="utf-8")
    rows = []
    for method, label in (("TriComponentDWASim", "DWASim"), ("Cosine", "Cosine"), ("MagnitudeOnly", "Magnitude only"), ("OneComponentSelector", "Single selector")):
        row = [label]
        for ds in DATASETS:
            r = read(f"representation_controls_{ds}.json")["representation"]
            row += [f'{r["half"][method]["metrics"][m]-r["complete"][method]["metrics"][m]:+.4f}' for m in ("macro_f1", "ndcg_at_10")]
        rows.append(row)
    table(folder, "representation_delta", "Half-profile minus complete-profile official-test differences, with path schemas and selection rules fixed. Positive values favor half profiles. Absolute results appear in the SI.",
          ["Method", r"ACM $\Delta$F1", r"$\Delta$NDCG", r"DBLP $\Delta$F1", r"$\Delta$NDCG"], rows)
    diagnostics = []
    for ds, r in evidence["datasets"].items():
        effect_rows = []
        for method, endpoints in controls[ds]["paired_final_minus_comparator"].items():
            from dwasim.reporting.assets import name
            values = [name(method)]
            for endpoint, label in (("macro_f1", "Macro-F1"),("ndcg_at_k", "NDCG@10")):
                e = endpoints[endpoint]
                values.append(f'{e["difference"]:.4f} [{e["lower_95"]:.4f}, {e["upper_95"]:.4f}]')
            effect_rows.append(values)
        table(folder, f"component_effects_{ds}", f"{ds} original DWASim minus restricted-selection and retuned-deletion controls. Descriptive 95\\% intervals use 2,000 paired category-stratified query resamples of frozen predictions, without multiplicity adjustment.",
              ["Comparator", r"Macro-F1 (95\% interval)", r"NDCG@10 (95\% interval)"], effect_rows, "lrr")
        corr = []
        for path, d in r["component_diagnostics"].items():
            diagnostics.append([ds, path, f'{100*d["nonzero_density"]:.5f}', f'{100*d["support_affinity_equal_one_fraction"]:.2f}']+
                               [scientific(v) for v in d["component_variance"]])
            values = np.asarray(d["component_correlation"], dtype=float)
            corr.append([path]+[f'{v:.4f}' if np.isfinite(v) else "--" for v in (values[0,1],values[0,2],values[1,2])])
        table(folder, f"correlation_{ds}", f"{ds} component-affinity Pearson correlations over every unordered training pair, accumulated in double precision. A dash denotes undefined correlation for a constant component.",
              ["Path", r"\shortstack{Support/\\magnitude}", r"\shortstack{Support/\\direction}", r"\shortstack{Magnitude/\\direction}"], corr)
        # A single comprehensive summary retains the two older unique controls.
        outer = []
        for method, label in list(RULES.items())+[("BestSinglePath", "Best single path"), ("NoSupport", "No support")]:
            source = fusion[ds] if method == "BestSinglePath" else controls[ds] if method == "NoSupport" else regular[ds]
            summary = source["outer_summary"][method]
            outer.append([label]+[f'${summary[m]["mean"]:.4f}\\pm{summary[m]["sample_sd"]:.4f}$' for m in ("macro_f1", "ndcg_at_10")])
        table(folder, f"outer_unified_{ds}", f"{ds} complete reselection across five overlapping outer training partitions (mean and sample standard deviation). All ten complexity rules and the two additional controls are retained. Variability is descriptive, not independent graph replication.", ["Rule", "Macro-F1", "NDCG@10"], outer)
    table(folder, "component_diagnostics", "Complete-profile density and training-pair unit-support fraction (percent), followed by support, magnitude, and directional variances. All pair statistics use double-precision accumulation.",
          ["Data", "Path", "Density", r"\shortstack{Unit\\support}", r'$v_s$', r'$v_m$', r'$v_d$'], diagnostics, "llrrrrr")
    rows = []
    for ds, r in evidence["datasets"].items():
        for label, q in r["categories"].items():
            rows.append([ds, label, q["n"], f'{q["mean"]:+.4f}', q["improved"], q["unchanged"], q["decreased"]])
    table(folder, "query_effect_categories", "Screening-plus-shrinkage minus original DWASim per-query NDCG@10, grouped by the actual query category. All official test queries are retained. Counts distinguish exact increases, ties, and decreases; category means are descriptive.",
          ["Data", "Category", "$n$", r"Mean $\Delta$", "Up", "Tie", "Down"], rows, "llrrrrr")


def style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "axes.linewidth": .6, "lines.linewidth": .85,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})


def export(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".eps"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def panel(ax, letter, dataset=None):
    ax.text(-.15, 1.045, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    if dataset:
        ax.text(1, 1.045, dataset, transform=ax.transAxes, ha="right", fontsize=9)


def plot_performance(output):
    regular = {ds: read(f"regularized_selection_{ds}.json") for ds in DATASETS}
    fusion = {ds: read(f"path_fusion_audit_{ds}.json") for ds in DATASETS}
    controls = {ds: read(f"revision_controls_{ds}.json") for ds in DATASETS}
    fig = plt.figure(figsize=(6.2992125984, 5.9448818898))
    axes = [fig.add_axes(b) for b in ((.14,.61,.345,.31),(.63,.61,.345,.31),(.25,.12,.31,.30),(.65,.12,.31,.30))]
    labels = {"BestSinglePath":"Single path", "Original":"DWASim", "Uniform":"Uniform", "Screened":"Screening",
              "ScreenedShrinkage":"Both rules", "Cosine":"Cosine", "CosineShrinkage":"Cosine + shrink."}
    locations = {"ACM": {"BestSinglePath":(.06,.08), "Original":(.55,.35), "Uniform":(.04,.81), "Screened":(.55,.51),
                            "ScreenedShrinkage":(.04,.96), "Cosine":(.62,.20), "CosineShrinkage":(.04,.66)},
                 "DBLP":{"BestSinglePath":(.04,.08), "Original":(.05,.40), "Uniform":(.60,.80), "Screened":(.48,.15),
                            "ScreenedShrinkage":(.49,.63), "Cosine":(.06,.95), "CosineShrinkage":(.06,.70)}}
    markers = dict(zip(labels, ("v","*","s","D","P","o","^")))
    for ax, ds, letter in zip(axes[:2], DATASETS, "ab"):
        metrics = dict(regular[ds]["test_metrics"], BestSinglePath=fusion[ds]["test_metrics"]["BestSinglePath"])
        points = [(metrics[m]["macro_f1"], metrics[m]["ndcg_at_10"]) for m in labels]
        xy = np.array(points)
        span = np.maximum(np.ptp(xy, axis=0), .001)
        ax.set_xlim(xy[:,0].min()-.12*span[0], xy[:,0].max()+.13*span[0])
        ax.set_ylim(xy[:,1].min()-.15*span[1], xy[:,1].max()+.15*span[1])
        for (method, label), (x,y) in zip(labels.items(), points):
            ax.scatter(x, y, s=62 if method == "Original" else 23, marker=markers[method], color=COLORS[ds], linewidths=.6, zorder=3)
            ax.annotate(label, (x,y), xytext=locations[ds][method], textcoords="axes fraction", fontsize=8,
                        va="center", arrowprops={"arrowstyle":"-", "color":"#737373", "lw":.5}, zorder=4)
        ax.set_xlabel("Macro-F1")
        ax.set_ylabel("NDCG@10")
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        panel(ax, letter, ds)
    comparators = ("BestSinglePath", "UniformPathFusion", "Cosine", "OneComponentSelector", "ScreenedShrinkage")
    row_labels = ("Single path", "Uniform fusion", "Cosine", "Single selector", "Both rules")
    for ax, metric, letter in zip(axes[2:], ("macro_f1","ndcg_at_k"), "cd"):
        ax.axvline(0, color="#777777", lw=.6, ls="--")
        for j, ds in enumerate(DATASETS):
            for y, method in enumerate(comparators):
                if method in ("BestSinglePath", "UniformPathFusion"):
                    e = fusion[ds]["paired_learned_minus"][method][metric]
                    point, low, high = e["difference"], e["lower_95"], e["upper_95"]
                elif method == "ScreenedShrinkage":
                    e = regular[ds]["paired_combined_minus"]["Original"][metric]
                    point, low, high = -e["difference"], -e["upper_95"], -e["lower_95"]
                else:
                    e = controls[ds]["paired_final_minus_comparator"][method][metric]
                    point, low, high = e["difference"], e["lower_95"], e["upper_95"]
                yy = y + (j-.5)*.25
                ax.plot([low,high], [yy,yy], color=COLORS[ds], lw=.85)
                ax.plot(point, yy, marker=("o","s")[j], ms=3.5, color=COLORS[ds])
        ax.set_yticks(range(5), row_labels if letter == "c" else [""]*5)
        ax.set_ylim(4.55,-.55)
        ax.set_xlim(-.035,.081)
        ax.set_xticks([-.02,0,.04,.08])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(r"$\Delta$ Macro-F1" if metric == "macro_f1" else r"$\Delta$ NDCG@10")
        panel(ax, letter)
    fig.legend([Line2D([],[],color=COLORS[ds],marker=m,lw=.85,ms=4) for ds,m in zip(DATASETS,("o","s"))],
               DATASETS, loc="lower center", ncol=2, bbox_to_anchor=(.60,.015))
    export(fig, output / "Fig4_performance_evidence")


def plot_mechanism(output, evidence):
    fig = plt.figure(figsize=(6.2992125984, 6.8110236220))
    axes = [fig.add_axes(b) for b in ((.14,.60,.345,.32),(.625,.60,.345,.32),(.16,.115,.31,.265),(.635,.115,.335,.265))]
    cmap = LinearSegmentedColormap.from_list("correlation", ["#9B522B","#FCFAF5","#294F70"])
    cmap.set_bad("#D2D2D2")
    for ax, ds, letter in zip(axes[:2], DATASETS, "ab"):
        r = evidence["datasets"][ds]
        correlation = np.asarray(r["cross_component_correlation"], dtype=float)
        im = ax.imshow(np.ma.masked_invalid(correlation), vmin=-1, vmax=1, cmap=cmap, interpolation="nearest")
        ticks = [f"{path}-{c}" for path in r["paths"] for c in ("S","M","D")]
        ax.set_xticks(range(9), ticks, rotation=90)
        ax.set_yticks(range(9), ticks)
        ax.tick_params(length=0, pad=2)
        for boundary in (2.5,5.5):
            ax.axhline(boundary, color="white", lw=.65)
            ax.axvline(boundary, color="white", lw=.65)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel(ax, letter, ds)
    cax = fig.add_axes((.30,.455,.48,.016))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1,-.5,0,.5,1])
    cb.set_label("Pearson correlation", labelpad=1)
    ax = axes[2]
    labels, percents = [], []
    for ds in DATASETS:
        for path,d in evidence["datasets"][ds]["component_diagnostics"].items():
            labels.append(path)
            percents.append(100*d["support_affinity_equal_one_fraction"])
    for i,(label,value) in enumerate(zip(labels,percents)):
        color = COLORS[DATASETS[i//3]]
        ax.plot([0,value],[i,i],lw=1,color=color)
        ax.plot(value,i,marker=("o","s")[i//3],color=color,ms=4)
        ax.annotate(f"{value:.2f}%", (value,i), xytext=(-5,6) if value>80 else (5,5), textcoords="offset points",
                    ha="right" if value>80 else "left", fontsize=8)
    ax.set_yticks(range(6),labels)
    ax.set_ylim(5.7,-.8)
    ax.set_xlim(-2,104)
    ax.set_xticks([0,50,100])
    ax.set_xlabel("Unit support affinity (%)")
    ax.tick_params(axis="y",length=0)
    panel(ax,"c")
    ax = axes[3]
    ax.axvline(0,color="#777777",lw=.6,ls="--")
    for ds,linestyle in zip(DATASETS,("-","--")):
        with np.load(ROOT / "results" / f"regularized_selection_{ds}.predictions.npz") as p:
            delta = p["ndcg_ScreenedShrinkage"]-p["ndcg_Original"]
        x,counts = np.unique(delta,return_counts=True)
        # Every observation contributes; duplicate values form exact ECDF jumps.
        ax.step(np.r_[min(-1,x[0]),x],np.r_[0,np.cumsum(counts)/len(delta)],where="post",color=COLORS[ds],ls=linestyle,lw=1.05,label=f"{ds} (n = {len(delta):,})")
    ax.set_xlim(-1,1)
    ax.set_ylim(0,1.02)
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_yticks([0,.5,1])
    ax.set_xlabel(r"Per-query $\Delta$NDCG@10")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(loc="lower right",fontsize=8)
    panel(ax,"d")
    export(fig,output / "Fig5_path_mechanism")


def generate(output, figures=True):
    output.mkdir(parents=True,exist_ok=True)
    evidence = collect_evidence()
    target = ROOT / "results" / "evidence_diagnostics.json"
    target.write_text(json.dumps(evidence,indent=2,allow_nan=False),encoding="utf-8")
    generate_tables(output,evidence)
    if figures:
        style()
        plot_performance(output)
        plot_mechanism(output,evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT / "manuscript_assets")
    parser.add_argument("--tables-only",action="store_true")
    args = parser.parse_args()
    generate(args.output,not args.tables_only)
