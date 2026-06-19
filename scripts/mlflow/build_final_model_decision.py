# scripts/build_final_model_decision.py

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_regression_recommendation(path: Path) -> dict:
    payload = load_json(path)

    if "best_regression" not in payload:
        raise KeyError(
            f"Clé 'best_regression' absente de {path}. "
            f"Clés disponibles : {list(payload.keys())}"
        )

    best = payload["best_regression"]

    required = [
        "run_id",
        "run_name",
        "model_family",
        "feature_set",
        "input_npz",
        "loss_weighting",
        "test_business_score",
        "test_mae",
        "test_rmse",
        "test_r2",
    ]

    missing = [key for key in required if key not in best]
    if missing:
        raise KeyError(f"Champs manquants dans best_regression : {missing}")

    return best


def load_threshold_winner(path: Path) -> dict:
    payload = load_json(path)

    if "winner" not in payload:
        raise KeyError(
            f"Clé 'winner' absente de {path}. "
            f"Clés disponibles : {list(payload.keys())}"
        )

    winner = payload["winner"]

    required = [
        "model_name",
        "threshold",
        "precision",
        "recall",
        "f1",
    ]

    missing = [key for key in required if key not in winner]
    if missing:
        raise KeyError(f"Champs manquants dans winner : {missing}")

    return payload


def load_classification_ranked(path: Path, selected_model_name: str) -> dict | None:
    """
    Essaie de récupérer les informations du run classification retenu
    depuis classification_ranked.csv.

    Le selected_model_name issu de la comparaison de seuil est un nom simplifié,
    par exemple cnn_transformer_alert_priority.
    On matche donc de façon souple sur class_weighting.
    """
    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return None

    if selected_model_name == "cnn_transformer_alert_priority":
        candidates = df[
            (df.get("model_family", "") == "cnn_transformer")
            & (df.get("class_weighting", "") == "alert_priority")
        ].copy()
    elif selected_model_name == "cnn_transformer_none":
        candidates = df[
            (df.get("model_family", "") == "cnn_transformer")
            & (df.get("class_weighting", "") == "none")
        ].copy()
    else:
        candidates = df[df.astype(str).apply(
            lambda row: selected_model_name in " ".join(row.values),
            axis=1,
        )].copy()

    if candidates.empty:
        return None

    sort_col = None
    ascending = False

    if "test_business_score_classification" in candidates.columns:
        sort_col = "test_business_score_classification"
        ascending = False
    elif "test_alert_24h_f1" in candidates.columns:
        sort_col = "test_alert_24h_f1"
        ascending = False
    elif "test_macro_f1" in candidates.columns:
        sort_col = "test_macro_f1"
        ascending = False

    if sort_col:
        candidates[sort_col] = pd.to_numeric(candidates[sort_col], errors="coerce")
        candidates = candidates.sort_values(sort_col, ascending=ascending)

    row = candidates.iloc[0].to_dict()

    clean = {}
    for key, value in row.items():
        if pd.isna(value):
            clean[key] = None
        else:
            clean[key] = value

    return clean


def build_decision(
    regression_best: dict,
    threshold_payload: dict,
    classification_info: dict | None,
    loo_regression_summary_path: Path | None,
) -> dict:
    winner = threshold_payload["winner"]

    selected_model_name = winner.get("model_name")
    selected_threshold = float(winner.get("threshold"))

    regression_candidate = {
        "model_family": regression_best.get("model_family"),
        "model_type": regression_best.get("model_type"),
        "run_id": regression_best.get("run_id"),
        "run_name": regression_best.get("run_name"),
        "feature_set": regression_best.get("feature_set"),
        "dataset_group": regression_best.get("dataset_group"),
        "input_npz": regression_best.get("input_npz"),
        "loss_weighting": regression_best.get("loss_weighting"),
        "early_stopping_metric": regression_best.get("early_stopping_metric"),
        "test_business_score": regression_best.get("test_business_score"),
        "test_mae": regression_best.get("test_mae"),
        "test_rmse": regression_best.get("test_rmse"),
        "test_r2": regression_best.get("test_r2"),
        "test_mae_36_48h": regression_best.get("test_mae_36_48h"),
        "test_mae_24_36h": regression_best.get("test_mae_24_36h"),
        "test_mae_12_24h": regression_best.get("test_mae_12_24h"),
        "test_mae_6_12h": regression_best.get("test_mae_6_12h"),
        "test_mae_0_6h": regression_best.get("test_mae_0_6h"),
        "status": "candidate_exploratory",
        "note": (
            "Bonne performance sur split fixe. "
            "La robustesse inter-éruption doit être interprétée à partir du LOO."
        ),
    }

    if loo_regression_summary_path and loo_regression_summary_path.exists():
        regression_candidate["loo_summary_csv"] = str(loo_regression_summary_path)
        regression_candidate["loo_status"] = "available"
    else:
        regression_candidate["loo_status"] = "pending_or_not_found"

    classification_candidate = {
        "selection_name": selected_model_name,
        "model_family": "cnn_transformer",
        "task_type": "classification",
        "feature_set": "full",
        "dataset_group": "with_quiet",
        "alert_threshold_24h": selected_threshold,
        "min_class_alert": 3,
        "selection_min_recall": threshold_payload.get("min_recall"),
        "selection_status": threshold_payload.get("decision_status"),
        "precision_alert_24h": winner.get("precision"),
        "recall_alert_24h": winner.get("recall"),
        "f1_alert_24h": winner.get("f1"),
        "tp": winner.get("tp"),
        "fp": winner.get("fp"),
        "fn": winner.get("fn"),
        "tn": winner.get("tn"),
        "source_csv": winner.get("source_csv"),
        "status": "selected_for_alerting",
    }

    if classification_info:
        classification_candidate.update({
            "run_id": classification_info.get("run_id"),
            "run_name": classification_info.get("run_name"),
            "model_type": classification_info.get("model_type"),
            "input_npz": classification_info.get("input_npz"),
            "class_weighting": classification_info.get("class_weighting"),
            "label_smoothing": classification_info.get("label_smoothing"),
            "early_stopping_metric": classification_info.get("early_stopping_metric"),
            "test_business_score_classification": classification_info.get("test_business_score_classification"),
            "test_macro_f1": classification_info.get("test_macro_f1"),
            "test_balanced_accuracy": classification_info.get("test_balanced_accuracy"),
            "test_class_0_f1": classification_info.get("test_class_0_f1"),
            "test_class_1_f1": classification_info.get("test_class_1_f1"),
            "test_class_2_f1": classification_info.get("test_class_2_f1"),
            "test_class_3_f1": classification_info.get("test_class_3_f1"),
            "test_class_4_f1": classification_info.get("test_class_4_f1"),
            "test_class_5_f1": classification_info.get("test_class_5_f1"),
        })
    else:
        if selected_model_name == "cnn_transformer_alert_priority":
            classification_candidate["class_weighting"] = "alert_priority"
        elif selected_model_name == "cnn_transformer_none":
            classification_candidate["class_weighting"] = "none"
        else:
            classification_candidate["class_weighting"] = None

        classification_candidate["classification_run_metadata_status"] = "not_found"

    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regression_candidate": regression_candidate,
        "classification_candidate": classification_candidate,
        "decision": {
            "main_operational_model": "classification_candidate",
            "regression_role": "exploratory_time_to_eruption_estimation",
            "classification_role": "operational_alert_24h",
            "rationale": (
                "La classification est retenue comme modèle opérationnel principal "
                "car elle correspond mieux à l'objectif d'alerte. "
                "La régression est conservée comme estimation exploratoire du délai avant éruption."
            ),
        },
        "sources": {
            "threshold_winner_json": str(threshold_payload.get("outputs", {}).get("winner_json", "")),
            "threshold_all_results_csv": str(threshold_payload.get("outputs", {}).get("all_results_csv", "")),
            "threshold_best_by_model_csv": str(threshold_payload.get("outputs", {}).get("best_by_model_csv", "")),
        },
    }

    return decision


def print_human_summary(decision: dict) -> None:
    reg = decision["regression_candidate"]
    clf = decision["classification_candidate"]

    print("\n" + "=" * 100)
    print("PROPOSITION DE DÉCISION MODÈLE")
    print("=" * 100)

    print("\nRégression candidate")
    print(f"  model_family        : {reg.get('model_family')}")
    print(f"  run_name            : {reg.get('run_name')}")
    print(f"  input_npz           : {reg.get('input_npz')}")
    print(f"  loss_weighting      : {reg.get('loss_weighting')}")
    print(f"  test_business_score : {reg.get('test_business_score')}")
    print(f"  test_mae            : {reg.get('test_mae')}")
    print(f"  test_rmse           : {reg.get('test_rmse')}")
    print(f"  test_r2             : {reg.get('test_r2')}")
    print(f"  status              : {reg.get('status')}")
    print(f"  loo_status          : {reg.get('loo_status')}")

    print("\nClassification candidate")
    print(f"  selection_name      : {clf.get('selection_name')}")
    print(f"  run_name            : {clf.get('run_name')}")
    print(f"  class_weighting     : {clf.get('class_weighting')}")
    print(f"  threshold_alert_24h : {clf.get('alert_threshold_24h')}")
    print(f"  precision_alert_24h : {clf.get('precision_alert_24h')}")
    print(f"  recall_alert_24h    : {clf.get('recall_alert_24h')}")
    print(f"  f1_alert_24h        : {clf.get('f1_alert_24h')}")
    print(f"  TP / FP / FN / TN   : {clf.get('tp')} / {clf.get('fp')} / {clf.get('fn')} / {clf.get('tn')}")
    print(f"  status              : {clf.get('status')}")

    print("\nDécision")
    print(f"  main_operational_model : {decision['decision']['main_operational_model']}")
    print(f"  regression_role        : {decision['decision']['regression_role']}")
    print(f"  classification_role    : {decision['decision']['classification_role']}")

    print("\n" + "=" * 100)


def main(args):
    regression_best = load_regression_recommendation(Path(args.regression_recommendation_json))

    threshold_payload = load_threshold_winner(Path(args.threshold_winner_json))
    selected_model_name = threshold_payload["winner"].get("model_name")

    classification_info = load_classification_ranked(
        path=Path(args.classification_ranked_csv),
        selected_model_name=selected_model_name,
    )

    loo_summary_path = Path(args.loo_regression_summary_csv) if args.loo_regression_summary_csv else None

    decision = build_decision(
        regression_best=regression_best,
        threshold_payload=threshold_payload,
        classification_info=classification_info,
        loo_regression_summary_path=loo_summary_path,
    )

    print_human_summary(decision)

    output_path = Path(args.output_json)

    if not args.approve:
        print("\nAucun fichier final n'a été écrit.")
        print("Relancer avec --approve pour écrire :")
        print(f"  {output_path}")
        print("\nExemple :")
        print(
            "python scripts/build_final_model_decision.py "
            "--approve"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    print(f"\nFichier de décision écrit : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--regression-recommendation-json",
        type=str,
        default="reports/targeted_regression_analysis/best_models_recommendation.json",
    )

    parser.add_argument(
        "--classification-ranked-csv",
        type=str,
        default="reports/targeted_classification_analysis/classification_ranked.csv",
    )

    parser.add_argument(
        "--threshold-winner-json",
        type=str,
        default="reports/threshold_optimization/comparison/threshold_comparison_winner.json",
    )

    parser.add_argument(
        "--loo-regression-summary-csv",
        type=str,
        default="reports/loo_regression_summary/loo_regression_summary.csv",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="configs/final_model_decision.json",
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Écrit effectivement le fichier JSON final. Sans ce flag, le script affiche seulement la proposition.",
    )

    args = parser.parse_args()
    main(args)