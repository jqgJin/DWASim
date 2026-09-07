"""Generate theory-extension tables from completed runs; never change figures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from dwasim.paths import RESULTS_ROOT

from dwasim.paths import ROOT
LABELS = {"PowerDWASim":"Full", "PowerMagnitude":"Magnitude", "PowerCosine":"Cosine",
          "PowerNoSupport":"No support", "PowerNoMagnitude":"No magnitude", "PowerNoDirection":"No direction",
          "KernelTarget":"Full", "MagnitudeTarget":"Magnitude", "CosineTarget":"Cosine",
          "NoSupportTarget":"No support", "NoMagnitudeTarget":"No magnitude", "NoDirectionTarget":"No direction"}


def load(kind, dataset):
    folder = RESULTS_ROOT / kind
    result = json.loads((folder/f"{kind}_{dataset}.json").read_text())
    path = folder/f"{kind}_{dataset}.predictions.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["prediction_sha256"]
    with np.load(path) as data:
        for name, metrics in result["test_metrics"].items():
            assert abs(f1_score(data["truth"],data[f"prediction_{name}"],average="macro",zero_division=0)-metrics["macro_f1"])<1e-12
            assert abs(data[f"ndcg_{name}"].mean()-metrics["ndcg_at_10"])<1e-12
    return result


def table(folder, name, caption, headers, rows, spec=None):
    spec = spec or "l"+"r"*(len(headers)-1)
    lines = [r"\begin{table}[!ht]", "\\caption{"+caption+"}\\label{tab:"+name+"}",
             r"\centering\normalsize",r"\setlength{\tabcolsep}{3pt}",r"\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}"+spec+"@{}}",r"\toprule",
             " & ".join(headers)+r" \\",r"\midrule"]
    lines.extend(" & ".join(row)+r" \\" for row in rows)
    lines.extend([r"\bottomrule",r"\end{tabular*}",r"\end{table}",r"\FloatBarrier",""])
    (folder/(name+".tex")).write_text("\n".join(lines),encoding="utf-8")


def generate(output):
    folder = output/"revision_tables"
    folder.mkdir(parents=True,exist_ok=True)
    power = {ds:load("kernel_power",ds) for ds in ("ACM","DBLP")}
    target = {ds:load("kernel_target",ds) for ds in ("ACM","DBLP")}
    summary = {}
    for ds,result in power.items():
        rows = []
        for name in (k for k in result["test_metrics"] if not k.endswith("P1")):
            m,base = result["test_metrics"][name],result["test_metrics"][name+"P1"]
            rows.append([LABELS[name],str(result["fit"]["models"][name]["power"]),f'{m["macro_f1"]:.4f}',
                         f'{m["macro_f1"]-base["macro_f1"]:+.4f}',f'{m["ndcg_at_10"]:.4f}',f'{m["ndcg_at_10"]-base["ndcg_at_10"]:+.4f}'])
        table(folder,f"kernel_power_{ds}",f"{ds} kernel-power calibration and matched ablations. Each difference is selected power minus that same family's retuned $p=1$ control. F1 is Macro-F1 and NDCG is NDCG@10; positive differences favor calibration.",
              ["Components","$p$","F1",r"$\Delta$F1","NDCG",r"$\Delta$NDCG"],rows)
        rows = []
        # Preserve selected and p=1 outer results in the same table.
        for name,stats in result["outer_summary"].items():
            label = LABELS[name.removesuffix("P1")]+(" ($p=1$)" if name.endswith("P1") else " (selected)")
            rows.append([label]+[f'${stats[m]["mean"]:.4f}\\pm{stats[m]["sample_sd"]:.4f}$' for m in ("macro_f1","ndcg_at_10")])
        table(folder,f"kernel_power_outer_{ds}",f"{ds} mean and sample standard deviation across five overlapping training-only outer partitions with complete reselection. These are not independent-graph confidence intervals.",
              ["Components / rule","Macro-F1","NDCG@10"],rows)
        rows = []
        for name in ("PowerDWASimP1","PowerMagnitude","PowerCosine","PowerNoSupport","PowerNoMagnitude","PowerNoDirection"):
            for metric,caption in (("macro_f1","F1"),("ndcg_at_k","NDCG")):
                value = result["paired_full_minus"][name][metric]
                label = "Original DWASim" if name.endswith("P1") else LABELS[name]
                rows.append([label,caption,f'{value["difference"]:+.4f}',f'[{value["lower_95"]:.4f}, {value["upper_95"]:.4f}]'])
        table(folder,f"kernel_power_effects_{ds}",f"{ds} full power-calibrated DWASim minus comparator. Intervals are descriptive 95\\% paired category-stratified query-bootstrap intervals (2,000 resamples), conditional on this graph and fitted models, without multiplicity adjustment.",
              ["Comparator","Endpoint","Difference","95\\% interval"],rows,spec="llrr")
        rows = []
        for name,stats in target[ds]["outer_summary"].items():
            rows.append([LABELS[name]]+[f'${stats[m]["mean"]:.4f}\\pm{stats[m]["sample_sd"]:.4f}$' for m in ("macro_f1","ndcg_at_10")])
        table(folder,f"kernel_target_outer_{ds}",f"{ds} kernel-target fitting with five overlapping training-only outer partitions: mean and sample standard deviation. Each inner fold reconstructs its label target from reference labels only.",
              ["Components","Macro-F1","NDCG@10"],rows)
        model = result["fit"]["models"]["PowerDWASim"]
        summary[ds] = {"power":model["power"],"path_weights":model["path_weights"],
                       "component_weights":model["component_weights"],"metrics":result["test_metrics"]["PowerDWASim"],
                       "paired_original":result["paired_full_minus"]["PowerDWASimP1"],
                       "outer":result["outer_summary"]["PowerDWASim"],
                       "outer_original":result["outer_summary"]["PowerDWASimP1"],
                       "target_metrics":target[ds]["test_metrics"]["KernelTarget"],
                       "target_max_kkt":max(m["kkt_residual"] for m in target[ds]["fit"]["models"].values()),
                       "target_fallbacks":[name for name,m in target[ds]["fit"]["models"].items() if m["fallback"]]}
    rows = [[LABELS[name]]+[f'{target[ds]["test_metrics"][name][metric]:.4f}' for ds in target for metric in ("macro_f1","ndcg_at_10")]
            for name in target["ACM"]["test_metrics"]]
    table(folder,"kernel_target_benchmarks","Kernel-target fitting on official test partitions. Full, single-component and deleted-component variants use the same regularization grid and reference-only fitting protocol.",
          ["Components","ACM F1","NDCG","DBLP F1","NDCG"],rows)
    fusion = folder/"fusion_complexity.tex"
    if fusion.exists():
        lines = [line for line in fusion.read_text(encoding="utf-8").splitlines() if not line.startswith("Kernel power &")]
        position = next(i for i,line in enumerate(lines) if line.startswith("Original DWASim &"))+1
        row = ["Kernel power"]+[f'{power[ds]["test_metrics"]["PowerDWASim"][metric]:.4f}' for ds in power for metric in ("macro_f1","ndcg_at_10")]
        lines.insert(position," & ".join(row)+r" \\")
        fusion.write_text("\n".join(lines)+"\n",encoding="utf-8")
    (output/"kernel_extension_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"manuscript_assets")
    generate(parser.parse_args().output)
