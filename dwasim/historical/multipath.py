import numpy as np

import dwasim.historical.controlled as exp


NOISES = (0.45, 0.55, 0.65)


from dwasim.similarity import safe_entropy_weights
import json


def entropy_weights(matrices):
    return safe_entropy_weights(matrices)


def main():
    records = {f"View {i + 1}": {"f1": [], "nmi": []} for i in range(3)}
    records["Uniform fusion"] = {"f1": [], "nmi": []}
    records["Entropy fusion"] = {"f1": [], "nmi": []}
    all_weights = []

    for seed in exp.SEEDS:
        matrices = []
        labels = None
        for view, noise in enumerate(NOISES):
            profiles, labels = exp.generate_profiles(seed + 1000 * view, 0.65, noise)
            h, l1, b0, b1 = exp.profile_discrepancies(profiles)
            matrix = exp.dwasim_from_discrepancies(h, l1, b0, b1, 0.5)
            matrices.append(matrix)
            records[f"View {view + 1}"]["f1"].append(exp.topk_macro_f1(matrix, labels))
            records[f"View {view + 1}"]["nmi"].append(exp.spectral_nmi(matrix, labels, seed))

        weights = entropy_weights(matrices)
        all_weights.append(weights)
        uniform = np.mean(matrices, axis=0)
        weighted = sum(weight * matrix for weight, matrix in zip(weights, matrices))
        for name, matrix in (("Uniform fusion", uniform), ("Entropy fusion", weighted)):
            records[name]["f1"].append(exp.topk_macro_f1(matrix, labels))
            records[name]["nmi"].append(exp.spectral_nmi(matrix, labels, seed))

    for name, metrics in records.items():
        f1 = np.asarray(metrics["f1"])
        nmi = np.asarray(metrics["nmi"])
        print(
            f"{name}: F1={f1.mean():.4f}+/-{f1.std(ddof=1):.4f}; "
            f"NMI={nmi.mean():.4f}+/-{nmi.std(ddof=1):.4f}"
        )

    weights = np.asarray(all_weights)
    print("Weights:", "; ".join(
        f"w{i + 1}={weights[:, i].mean():.4f}+/-{weights[:, i].std(ddof=1):.4f}"
        for i in range(weights.shape[1])
    ))

    (exp.ROOT / "multipath_controlled.json").write_text(json.dumps({"records": records, "weights": weights.tolist(), "seeds": list(exp.SEEDS), "noises": NOISES}, indent=2), encoding="utf-8")



if __name__ == "__main__":
    main()
