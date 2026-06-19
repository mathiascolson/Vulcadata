# scripts/summarize_loo_regression.py

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def extract_fold_info(fold_dir: Path) -> dict:
    match = re.match(r"fold_(\d+)_test_(.+)", fold_dir.name)

    if match:
        return {
            "fold": int(match.group(1)),
            "test_event_id": match.group(2),
            "fold_dir": str(fold_dir),
        }

    return {
        "fold": None,
        "test_event_id": fold_dir.name,
        "fold_dir": str(fold_dir),
    }


def load_metrics(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    row = {
        "best_val_metric": payload.get("best_val_metric"),
        "best_val_score": payload.get("best_val_score"),
        "n_params": payload.get("n_params"),
        "n_trainable_params": payload.get("n_trainable_params"),
    }

    test_metrics = payload.get("test", {})

    for key, value in test_metrics.items():
        row[f"test_{key}"] = value

    return row


def main(args):
    loo_root = Path(args.loo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for fold_dir in sorted(loo_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        metrics_path = fold_dir / "model" / "metrics_cnn_bilstm_regression.json"

        row = extract_fold_info(fold_dir)

        if not metrics_path.exists():
            row["status"] = "missing_metrics"
            rows.append(row)
            continue

        row["status"] = "ok"
        row["metrics_path"] = str(metrics_path)
        row.update(load_metrics(metrics_path))
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"Aucun fold trouvé dans : {loo_root}")

    df = df.sort_values("fold", na_position="last")

    summary_path = output_dir / "loo_regression_summary.csv"
    df.to_csv(summary_path, index=False)

    metric_cols = [
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_business_score",
        "test_mae_36_48h",
        "test_mae_24_36h",
        "test_mae_12_24h",
        "test_mae_6_12h",
        "test_mae_0_6h",
    ]

    available_cols = [col for col in metric_cols if col in df.columns]
    completed = df[df["status"] == "ok"].copy()

    if not completed.empty:
        aggregate = completed[available_cols].agg(
            ["count", "mean", "std", "min", "median", "max"]
        )

        aggregate_path = output_dir / "loo_regression_aggregate_stats.csv"
        aggregate.to_csv(aggregate_path)

        print("\nStatistiques agrégées :")
        print(aggregate)

    display_cols = [
        "fold",
        "test_event_id",
        "status",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_business_score",
        "test_mae_36_48h",
        "test_mae_24_36h",
        "test_mae_12_24h",
        "test_mae_6_12h",
        "test_mae_0_6h",
    ]

    display_cols = [col for col in display_cols if col in df.columns]

    print("\nRésumé par fold :")
    print(df[display_cols].to_string(index=False))

    print(f"\nFichier écrit : {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loo-root",
        type=str,
        default="reports/loo_regression",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/loo_regression_summary",
    )

    args = parser.parse_args()
    main(args)