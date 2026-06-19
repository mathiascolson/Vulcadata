# scripts/optimize_alert_threshold.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def load_predictions(path: Path):
    data = np.load(path, allow_pickle=True)
    required = ["y_true", "y_proba"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Clés absentes du fichier predictions NPZ : {missing}. Clés disponibles : {list(data.keys())}")

    y_true = data["y_true"].astype(np.int64)
    y_proba = data["y_proba"].astype(np.float32)

    if y_proba.ndim != 2:
        raise ValueError(f"y_proba doit être de forme (N, n_classes), reçu : {y_proba.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true doit être de forme (N,), reçu : {y_true.shape}")
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(f"Tailles incohérentes : y_true={y_true.shape}, y_proba={y_proba.shape}")

    return y_true, y_proba


def alert_probability(y_proba: np.ndarray, min_class: int) -> np.ndarray:
    if min_class < 0 or min_class >= y_proba.shape[1]:
        raise ValueError(f"min_class invalide : {min_class} pour y_proba avec {y_proba.shape[1]} classes")
    return y_proba[:, min_class:].sum(axis=1)


def evaluate_thresholds(y_true, y_proba, min_class: int, thresholds: np.ndarray) -> pd.DataFrame:
    y_true_alert = (y_true >= min_class).astype(np.int64)
    p_alert = alert_probability(y_proba, min_class=min_class)

    rows = []
    for threshold in thresholds:
        y_pred_alert = (p_alert >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true_alert, y_pred_alert, labels=[0, 1]).ravel()

        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true_alert, y_pred_alert, zero_division=0)),
            "recall": float(recall_score(y_true_alert, y_pred_alert, zero_division=0)),
            "f1": float(f1_score(y_true_alert, y_pred_alert, zero_division=0)),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "positive_rate_pred": float(y_pred_alert.mean()),
            "positive_rate_true": float(y_true_alert.mean()),
        })

    return pd.DataFrame(rows)


def select_threshold(df: pd.DataFrame, mode: str, min_recall: float) -> dict:
    if df.empty:
        raise ValueError("Aucun seuil à sélectionner.")

    if mode == "max_f1":
        ranked = df.sort_values(
            by=["f1", "precision", "recall"],
            ascending=[False, False, False],
        )
        return ranked.iloc[0].to_dict()

    if mode == "recall_at_least":
        candidates = df[df["recall"] >= min_recall].copy()
        if candidates.empty:
            ranked = df.sort_values(by=["recall", "f1", "precision"], ascending=[False, False, False])
            selected = ranked.iloc[0].to_dict()
            selected["warning"] = f"Aucun seuil n'atteint recall >= {min_recall}. Meilleur recall sélectionné."
            return selected

        ranked = candidates.sort_values(by=["precision", "f1", "recall"], ascending=[False, False, False])
        return ranked.iloc[0].to_dict()

    raise ValueError(f"selection_mode invalide : {mode}")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_proba = load_predictions(Path(args.predictions_npz))

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    thresholds = np.round(thresholds, 6)

    results = evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba,
        min_class=args.min_class,
        thresholds=thresholds,
    )

    selected = select_threshold(
        df=results,
        mode=args.selection_mode,
        min_recall=args.min_recall,
    )

    alert_name = {
        3: "alert_24h",
        4: "alert_12h",
        5: "alert_6h",
    }.get(args.min_class, f"alert_min_class_{args.min_class}")

    csv_path = output_dir / f"threshold_scan_{alert_name}.csv"
    json_path = output_dir / f"best_threshold_{alert_name}.json"

    results.to_csv(csv_path, index=False)

    payload = {
        "predictions_npz": args.predictions_npz,
        "alert_name": alert_name,
        "min_class": int(args.min_class),
        "selection_mode": args.selection_mode,
        "min_recall": float(args.min_recall),
        "selected": selected,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Seuil sélectionné")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nFichiers écrits :")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions-npz",
        type=str,
        required=True,
        help="Fichier predictions_*.npz produit par le script de classification. Doit contenir y_true et y_proba.",
    )
    parser.add_argument("--output-dir", type=str, default="reports/threshold_optimization")
    parser.add_argument(
        "--min-class",
        type=int,
        default=3,
        help="Classe minimale pour définir l'alerte. 3=24h, 4=12h, 5=6h.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="max_f1",
        choices=["max_f1", "recall_at_least"],
    )
    parser.add_argument("--min-recall", type=float, default=0.80)

    args = parser.parse_args()
    main(args)
