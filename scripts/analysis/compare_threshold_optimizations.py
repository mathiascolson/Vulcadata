# scripts/compare_threshold_optimizations.py

import argparse
import json
from pathlib import Path

import pandas as pd


def find_threshold_csv(model_dir: Path) -> Path:
    """
    Recherche le CSV produit par optimize_alert_threshold.py.
    Le script accepte plusieurs noms possibles pour éviter une dépendance trop fragile
    au nom exact du fichier de sortie.
    """
    candidates = [
        model_dir / "threshold_results.csv",
        model_dir / "threshold_optimization_results.csv",
        model_dir / "alert_threshold_results.csv",
        model_dir / "results.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    csv_files = sorted(model_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    if len(csv_files) > 1:
        names = "\n".join(str(p) for p in csv_files)
        raise FileExistsError(
            f"Plusieurs CSV trouvés dans {model_dir}. "
            f"Impossible de choisir automatiquement :\n{names}"
        )

    raise FileNotFoundError(f"Aucun CSV de seuil trouvé dans : {model_dir}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes les plus probables.
    Objectif : rendre le script tolérant aux variantes de sortie.
    """
    rename_map = {}

    for col in df.columns:
        clean = col.lower().strip()

        if clean in {"threshold", "seuil", "alert_threshold"}:
            rename_map[col] = "threshold"

        elif clean in {
            "precision",
            "alert_precision",
            "alert_24h_precision",
            "precision_alert_24h",
            "test_alert_24h_precision",
        }:
            rename_map[col] = "precision"

        elif clean in {
            "recall",
            "alert_recall",
            "alert_24h_recall",
            "recall_alert_24h",
            "test_alert_24h_recall",
        }:
            rename_map[col] = "recall"

        elif clean in {
            "f1",
            "alert_f1",
            "alert_24h_f1",
            "f1_alert_24h",
            "test_alert_24h_f1",
        }:
            rename_map[col] = "f1"

        elif clean in {
            "tp",
            "true_positive",
            "true_positives",
        }:
            rename_map[col] = "tp"

        elif clean in {
            "fp",
            "false_positive",
            "false_positives",
        }:
            rename_map[col] = "fp"

        elif clean in {
            "fn",
            "false_negative",
            "false_negatives",
        }:
            rename_map[col] = "fn"

        elif clean in {
            "tn",
            "true_negative",
            "true_negatives",
        }:
            rename_map[col] = "tn"

    return df.rename(columns=rename_map)


def load_model_results(model_name: str, model_dir: Path) -> pd.DataFrame:
    csv_path = find_threshold_csv(model_dir)

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    required = ["threshold", "precision", "recall", "f1"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(
            f"Colonnes manquantes dans {csv_path} : {missing}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    df["model_name"] = model_name
    df["source_csv"] = str(csv_path)

    for col in ["threshold", "precision", "recall", "f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["threshold", "precision", "recall", "f1"]).copy()

    return df


def select_best_row(df: pd.DataFrame, min_recall: float) -> pd.Series:
    eligible = df[df["recall"] >= min_recall].copy()

    if eligible.empty:
        raise ValueError(
            f"Aucun seuil ne respecte recall >= {min_recall:.3f}."
        )

    eligible = eligible.sort_values(
        by=["precision", "f1", "threshold"],
        ascending=[False, False, False],
    )

    return eligible.iloc[0]


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {
        "cnn_transformer_alert_priority": Path(args.alert_priority_dir),
        "cnn_transformer_none": Path(args.none_dir),
    }

    frames = []

    for model_name, model_dir in model_specs.items():
        df_model = load_model_results(model_name=model_name, model_dir=model_dir)
        frames.append(df_model)

    all_results = pd.concat(frames, ignore_index=True)

    all_results_path = output_dir / "threshold_comparison_all_results.csv"
    all_results.to_csv(all_results_path, index=False)

    best_rows = []

    for model_name in all_results["model_name"].unique():
        df_model = all_results[all_results["model_name"] == model_name].copy()

        try:
            best = select_best_row(df_model, min_recall=args.min_recall)
            best_rows.append(best.to_dict())
        except ValueError:
            fallback = df_model.sort_values(
                by=["recall", "f1", "precision"],
                ascending=[False, False, False],
            ).iloc[0].to_dict()

            fallback["selection_warning"] = (
                f"Aucun seuil avec recall >= {args.min_recall:.3f}. "
                "Fallback = meilleur recall disponible."
            )
            best_rows.append(fallback)

    best_df = pd.DataFrame(best_rows)

    if "selection_warning" not in best_df.columns:
        best_df["selection_warning"] = ""

    best_df = best_df.sort_values(
        by=["recall", "precision", "f1"],
        ascending=[False, False, False],
    )

    eligible_best_df = best_df[best_df["recall"] >= args.min_recall].copy()

    if not eligible_best_df.empty:
        final_df = eligible_best_df.sort_values(
            by=["precision", "f1", "threshold"],
            ascending=[False, False, False],
        )
        winner = final_df.iloc[0].to_dict()
        decision_status = "eligible"
    else:
        final_df = best_df.sort_values(
            by=["recall", "f1", "precision"],
            ascending=[False, False, False],
        )
        winner = final_df.iloc[0].to_dict()
        decision_status = "fallback_no_model_reaches_min_recall"

    best_by_model_path = output_dir / "threshold_comparison_best_by_model.csv"
    best_df.to_csv(best_by_model_path, index=False)

    winner_path = output_dir / "threshold_comparison_winner.json"

    decision = {
        "decision_status": decision_status,
        "min_recall": args.min_recall,
        "winner": winner,
        "selection_rule": (
            "Parmi les modèles avec recall >= min_recall, sélection de la meilleure precision, "
            "puis meilleur F1, puis seuil le plus élevé. "
            "Si aucun modèle n'atteint min_recall, fallback sur meilleur recall, puis F1, puis precision."
        ),
        "outputs": {
            "all_results_csv": str(all_results_path),
            "best_by_model_csv": str(best_by_model_path),
            "winner_json": str(winner_path),
        },
    }

    with open(winner_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    print("\nComparaison des seuils terminée.")
    print(f"Résultats complets : {all_results_path}")
    print(f"Meilleurs par modèle : {best_by_model_path}")
    print(f"Décision : {winner_path}")

    print("\nMeilleur seuil par modèle :")
    display_cols = [
        "model_name",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "selection_warning",
    ]
    display_cols = [col for col in display_cols if col in best_df.columns]
    print(best_df[display_cols].to_string(index=False))

    print("\nModèle retenu :")
    print(f"  model_name : {winner.get('model_name')}")
    print(f"  threshold  : {winner.get('threshold')}")
    print(f"  precision  : {winner.get('precision')}")
    print(f"  recall     : {winner.get('recall')}")
    print(f"  f1         : {winner.get('f1')}")
    print(f"  status     : {decision_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alert-priority-dir",
        type=str,
        default="reports/threshold_optimization/cnn_transformer_alert_priority",
    )

    parser.add_argument(
        "--none-dir",
        type=str,
        default="reports/threshold_optimization/cnn_transformer_none",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/threshold_optimization/comparison",
    )

    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.80,
    )

    args = parser.parse_args()
    main(args)