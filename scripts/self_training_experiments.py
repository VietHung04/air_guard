from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.classification_library import Paths, run_prepare, train_classifier, time_split, AQI_CLASSES
from src.semi_supervised_library import (
    SemiDataConfig,
    SelfTrainingConfig,
    mask_labels_time_aware,
    run_self_training,
)


def find_or_prepare_data(project_root: Path) -> pd.DataFrame:
    processed = project_root / "data" / "processed" / "cleaned.parquet"
    if processed.exists():
        df = pd.read_parquet(processed)
        return df

    # try to prepare (may require internet)
    print("Processed data not found; attempting to fetch and prepare dataset (may take time)...")
    run_prepare(Paths(project_root=project_root), use_ucimlrepo=True)
    if not processed.exists():
        raise FileNotFoundError(f"Processed file not found after prepare: {processed}")
    return pd.read_parquet(processed)


def run_experiments(project_root: Path, out_dir: Path, taus=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = find_or_prepare_data(project_root)

    data_cfg = SemiDataConfig()

    # mask labels to simulate scarcity (only in TRAIN portion)
    masked = mask_labels_time_aware(df.copy(), data_cfg, missing_fraction=0.95)

    # train/test split
    train_df, test_df = time_split(masked, cutoff=data_cfg.cutoff)
    # reference unmasked test from original df
    _, test_df_orig = time_split(df.copy(), cutoff=data_cfg.cutoff)

    # Baseline trained on labeled portion only
    print("Training baseline (supervised on labeled subset)...")
    baseline = train_classifier(train_df, test_df_orig)
    baseline_metrics = baseline["metrics"]

    results = []
    per_tau_reports = {}

    if taus is None:
        taus = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

    for tau in taus:
        print(f"Running self-training tau={tau}")
        st_cfg = SelfTrainingConfig(tau=float(tau), max_iter=10, min_new_per_iter=20, val_frac=0.2)
        out = run_self_training(masked.copy(), data_cfg=data_cfg, st_cfg=st_cfg)

        tm = out["test_metrics"]
        summary = {
            "tau": float(tau),
            "accuracy": float(tm["accuracy"]),
            "f1_macro": float(tm["f1_macro"]),
            "n_train": int(tm.get("n_train", 0)),
            "n_test": int(tm.get("n_test", 0)),
            "n_iters": len(out["history"]),
            "tau_used": float(st_cfg.tau),
        }
        results.append(summary)
        per_tau_reports[str(tau)] = {
            "metrics": tm,
            "history": out["history"],
            "pred_df": out.get("pred_df"),
        }

        # save per-tau history
        hist_df = pd.DataFrame(out["history"])
        hist_df.to_csv(out_dir / f"history_tau_{str(tau).replace('.', '_')}.csv", index=False)

    res_df = pd.DataFrame(results).sort_values("tau")
    res_df.to_csv(out_dir / "self_training_summary.csv", index=False)

    # baseline row
    baseline_row = {
        "tau": "baseline",
        "accuracy": baseline_metrics["accuracy"],
        "f1_macro": baseline_metrics["f1_macro"],
        "n_train": baseline_metrics.get("n_train", 0),
        "n_test": baseline_metrics.get("n_test", 0),
        "n_iters": 0,
    }
    pd.DataFrame([baseline_row]).to_csv(out_dir / "baseline_metrics.csv", index=False)

    # Plot accuracy and f1 vs tau
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=res_df, x="tau", y="accuracy", marker="o")
    plt.axhline(baseline_row["accuracy"], color="gray", linestyle="--", label="baseline")
    plt.title("Accuracy vs tau")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_tau.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=res_df, x="tau", y="f1_macro", marker="o")
    plt.axhline(baseline_row["f1_macro"], color="gray", linestyle="--", label="baseline")
    plt.title("F1 macro vs tau")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_macro_vs_tau.png")
    plt.close()

    # Per-class comparison between best tau and baseline
    best_idx = res_df["f1_macro"].astype(float).idxmax()
    best_tau = res_df.loc[best_idx, "tau"]
    best_report = per_tau_reports[str(best_tau)]["metrics"]["report"]
    baseline_report = baseline_metrics["report"]

    classes = list(baseline_report.keys())
    rows = []
    for c in AQI_CLASSES:
        if c in baseline_report and c in best_report:
            b_f1 = baseline_report[c].get("f1-score", np.nan)
            s_f1 = best_report[c].get("f1-score", np.nan)
            rows.append({"class": c, "baseline_f1": b_f1, "selftrain_f1": s_f1, "delta": s_f1 - b_f1})

    class_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    class_df.to_csv(out_dir / "per_class_f1_comparison.csv", index=False)

    # bar plot per-class delta
    plt.figure(figsize=(10, 5))
    sns.barplot(data=class_df, x="class", y="delta")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Per-class F1 delta (self-training tau={best_tau} - baseline)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_f1_delta.png")
    plt.close()

    # Save aggregated JSON
    summary_out = {
        "baseline": baseline_metrics,
        "results_table": res_df.to_dict(orient="records"),
        "best_tau": float(best_tau),
        "per_class_comparison": class_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf8") as fh:
        json.dump(summary_out, fh, indent=2, ensure_ascii=False)

    print(f"All outputs saved to: {out_dir}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / "results" / f"self_training_{ts}"
    run_experiments(repo_root, out_dir)


if __name__ == "__main__":
    main()
