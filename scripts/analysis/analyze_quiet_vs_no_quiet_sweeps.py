# scripts/analyze_quiet_vs_no_quiet_sweeps.py

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv(override=False)


# ============================================================
# MLflow
# ============================================================

def setup_mlflow(args):
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow n'est pas installé dans l'environnement courant."
        ) from exc

    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(args.experiment_name)

    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable : {args.experiment_name}")

    return mlflow, experiment


def load_runs(mlflow, experiment_id: str, max_results: int) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )

    if runs.empty:
        raise ValueError("Aucun run MLflow trouvé.")

    return runs


# ============================================================
# HELPERS
# ============================================================

def safe_get(row: pd.Series, candidates: list[str], default=np.nan):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def get_metric(row: pd.Series, metric_name: str):
    return safe_get(
        row,
        [
            f"metrics.{metric_name}",
            metric_name,
        ],
        default=np.nan,
    )


def get_param(row: pd.Series, param_name: str):
    return safe_get(
        row,
        [
            f"params.{param_name}",
            param_name,
        ],
        default=np.nan,
    )


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower()


def infer_model_family(row: pd.Series) -> str:
    """
    Important :
    - cnn_transformer doit être détecté avant transformer.
    - cnn_bilstm doit être détecté avant transformer.
    """
    text_parts = []

    for col in [
        "params.model_type",
        "tags.mlflow.runName",
        "params.run_name",
        "params.output_dir",
        "params.input_npz",
        "artifact_uri",
        "attributes.artifact_uri",
    ]:
        if col in row.index and pd.notna(row[col]):
            text_parts.append(str(row[col]))

    text = " ".join(text_parts).lower()
    compact = (
        text
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if (
        "cnntransformer" in compact
        or "cnn_transformer" in text
        or "cnn-transformer" in text
        or "cnntransformerclassifier" in compact
        or "cnntransformerregressor" in compact
    ):
        return "cnn_transformer"

    if (
        "cnnbilstm" in compact
        or "cnn_bilstm" in text
        or "cnn-bilstm" in text
        or "bilstm" in text
    ):
        return "cnn_bilstm"

    if (
        "transformerencoder" in compact
        or "transformer_classifier" in text
        or "transformer_volcano" in text
        or "transformer" in text
    ):
        return "transformer"

    return "unknown"


def infer_task_type(row: pd.Series) -> str:
    """
    Détection tâche :
    - classification si métriques F1 / accuracy / classe existent
    - régression si MAE/RMSE/R² existent
    """
    classification_indicators = [
        "metrics.test_macro_f1",
        "metrics.test_weighted_f1",
        "metrics.test_accuracy",
        "metrics.test_balanced_accuracy",
        "metrics.test_alert_f1",
        "metrics.test_alert_24h_f1",
        "params.n_classes",
    ]

    for col in classification_indicators:
        if col in row.index and pd.notna(row[col]):
            return "classification"

    regression_indicators = [
        "metrics.test_mae",
        "metrics.test_rmse",
        "metrics.test_r2",
        "metrics.best_val_mae",
    ]

    for col in regression_indicators:
        if col in row.index and pd.notna(row[col]):
            return "regression"

    return "unknown"


def infer_dataset_group(row: pd.Series) -> str:
    """
    Groupe dataset :
    - with_quiet si le chemin ou run name mentionne quiet / calme
    - no_quiet sinon
    """
    text_parts = []

    for col in [
        "params.input_npz",
        "tags.mlflow.runName",
        "params.run_name",
        "params.output_dir",
    ]:
        if col in row.index and pd.notna(row[col]):
            text_parts.append(str(row[col]))

    text = " ".join(text_parts).lower()

    if "with_quiet" in text or "quiet" in text or "calme" in text:
        return "with_quiet"

    return "no_quiet"


def is_finished_status(status) -> bool:
    value = str(status).upper()
    return value in {"FINISHED", ""}


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def to_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ============================================================
# EXTRACTION RUNS
# ============================================================

def extract_runs(runs_raw: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in runs_raw.iterrows():
        run_id = safe_get(row, ["run_id", "attributes.run_id"], default="")
        run_name = safe_get(row, ["tags.mlflow.runName"], default="")
        status = safe_get(row, ["status", "attributes.status"], default="")

        model_family = infer_model_family(row)
        task_type = infer_task_type(row)
        dataset_group = infer_dataset_group(row)

        record = {
            "run_id": run_id,
            "run_name": run_name,
            "status": status,
            "start_time": safe_get(row, ["start_time", "attributes.start_time"], default=pd.NaT),
            "artifact_uri": safe_get(row, ["artifact_uri", "attributes.artifact_uri"], default=""),

            "model_family": model_family,
            "task_type": task_type,
            "dataset_group": dataset_group,

            "input_npz": get_param(row, "input_npz"),
            "model_type": get_param(row, "model_type"),

            "seq_len": get_param(row, "seq_len"),
            "n_features": get_param(row, "n_features"),
            "n_classes": get_param(row, "n_classes"),

            "batch_size": get_param(row, "batch_size"),
            "learning_rate": get_param(row, "learning_rate"),
            "weight_decay": get_param(row, "weight_decay"),
            "dropout": get_param(row, "dropout"),
            "input_noise_std": get_param(row, "input_noise_std"),
            "class_weighting": get_param(row, "class_weighting"),
            "label_smoothing": get_param(row, "label_smoothing"),
            "early_stopping_metric": get_param(row, "early_stopping_metric"),

            # Transformer
            "d_model": get_param(row, "d_model"),
            "n_heads": get_param(row, "n_heads"),
            "nhead": get_param(row, "nhead"),
            "n_layers": get_param(row, "n_layers"),
            "num_layers": get_param(row, "num_layers"),
            "dim_feedforward": get_param(row, "dim_feedforward"),

            # CNN-BiLSTM
            "conv_channels": get_param(row, "conv_channels"),
            "lstm_hidden": get_param(row, "lstm_hidden"),
            "lstm_layers": get_param(row, "lstm_layers"),

            # Classification - validation
            "best_val_macro_f1": get_metric(row, "best_val_macro_f1"),
            "best_val_alert_24h_f1": get_metric(row, "best_val_alert_24h_f1"),
            "best_val_balanced_accuracy": get_metric(row, "best_val_balanced_accuracy"),

            # Classification - test
            "test_loss": get_metric(row, "test_loss"),
            "test_accuracy": get_metric(row, "test_accuracy"),
            "test_balanced_accuracy": get_metric(row, "test_balanced_accuracy"),
            "test_macro_precision": get_metric(row, "test_macro_precision"),
            "test_macro_recall": get_metric(row, "test_macro_recall"),
            "test_macro_f1": get_metric(row, "test_macro_f1"),
            "test_weighted_precision": get_metric(row, "test_weighted_precision"),
            "test_weighted_recall": get_metric(row, "test_weighted_recall"),
            "test_weighted_f1": get_metric(row, "test_weighted_f1"),

            "test_alert_precision": get_metric(row, "test_alert_precision"),
            "test_alert_recall": get_metric(row, "test_alert_recall"),
            "test_alert_f1": get_metric(row, "test_alert_f1"),

            "test_alert_24h_precision": get_metric(row, "test_alert_24h_precision"),
            "test_alert_24h_recall": get_metric(row, "test_alert_24h_recall"),
            "test_alert_24h_f1": get_metric(row, "test_alert_24h_f1"),

            "test_alert_12h_precision": get_metric(row, "test_alert_12h_precision"),
            "test_alert_12h_recall": get_metric(row, "test_alert_12h_recall"),
            "test_alert_12h_f1": get_metric(row, "test_alert_12h_f1"),

            "test_alert_6h_precision": get_metric(row, "test_alert_6h_precision"),
            "test_alert_6h_recall": get_metric(row, "test_alert_6h_recall"),
            "test_alert_6h_f1": get_metric(row, "test_alert_6h_f1"),

            "test_class_0_f1": get_metric(row, "test_class_0_f1"),
            "test_class_1_f1": get_metric(row, "test_class_1_f1"),
            "test_class_2_f1": get_metric(row, "test_class_2_f1"),
            "test_class_3_f1": get_metric(row, "test_class_3_f1"),
            "test_class_4_f1": get_metric(row, "test_class_4_f1"),
            "test_class_5_f1": get_metric(row, "test_class_5_f1"),

            # Régression - validation/test
            "best_val_mae": get_metric(row, "best_val_mae"),
            "test_mae": get_metric(row, "test_mae"),
            "test_rmse": get_metric(row, "test_rmse"),
            "test_mse": get_metric(row, "test_mse"),
            "test_r2": get_metric(row, "test_r2"),
        }

        records.append(record)

    df = pd.DataFrame(records)

    numeric_cols = [
        "seq_len",
        "n_features",
        "n_classes",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "input_noise_std",
        "label_smoothing",
        "d_model",
        "n_heads",
        "nhead",
        "n_layers",
        "num_layers",
        "dim_feedforward",
        "conv_channels",
        "lstm_hidden",
        "lstm_layers",

        "best_val_macro_f1",
        "best_val_alert_24h_f1",
        "best_val_balanced_accuracy",

        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_weighted_f1",

        "test_alert_precision",
        "test_alert_recall",
        "test_alert_f1",
        "test_alert_24h_precision",
        "test_alert_24h_recall",
        "test_alert_24h_f1",
        "test_alert_12h_precision",
        "test_alert_12h_recall",
        "test_alert_12h_f1",
        "test_alert_6h_precision",
        "test_alert_6h_recall",
        "test_alert_6h_f1",

        "test_class_0_f1",
        "test_class_1_f1",
        "test_class_2_f1",
        "test_class_3_f1",
        "test_class_4_f1",
        "test_class_5_f1",

        "best_val_mae",
        "test_mae",
        "test_rmse",
        "test_mse",
        "test_r2",
    ]

    df = to_numeric_columns(df, numeric_cols)

    return df


# ============================================================
# FILTRAGE
# ============================================================

def filter_runs(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    if args.only_finished:
        out = out[out["status"].apply(is_finished_status)].copy()

    if not args.keep_unknown:
        out = out[out["model_family"] != "unknown"].copy()
        out = out[out["task_type"] != "unknown"].copy()

    if args.model_families:
        wanted = set(args.model_families)
        out = out[out["model_family"].isin(wanted)].copy()

    if args.dataset_filter:
        mask = (
            out["input_npz"].fillna("").astype(str).str.contains(
                args.dataset_filter,
                case=False,
                regex=False,
            )
            | out["run_name"].fillna("").astype(str).str.contains(
                args.dataset_filter,
                case=False,
                regex=False,
            )
        )
        out = out[mask].copy()

    # Garde uniquement les runs avec au moins une métrique utile.
    useful_metrics = [
        "test_macro_f1",
        "test_alert_24h_f1",
        "test_mae",
        "test_r2",
    ]

    mask_useful = np.zeros(len(out), dtype=bool)

    for col in useful_metrics:
        if col in out.columns:
            mask_useful |= out[col].notna().to_numpy()

    out = out[mask_useful].copy()

    return out


# ============================================================
# SCORING
# ============================================================

def minmax_higher_is_better(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        return pd.Series(1.0, index=s.index)

    return (s - min_val) / (max_val - min_val)


def minmax_lower_is_better(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        return pd.Series(1.0, index=s.index)

    return 1.0 - ((s - min_val) / (max_val - min_val))


def add_classification_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    clf_mask = out["task_type"] == "classification"

    out["classification_score"] = np.nan

    if clf_mask.sum() == 0:
        return out

    clf = out.loc[clf_mask].copy()

    clf["score_macro_f1"] = minmax_higher_is_better(clf["test_macro_f1"])
    clf["score_balanced_accuracy"] = minmax_higher_is_better(clf["test_balanced_accuracy"])
    clf["score_alert_24h_f1"] = minmax_higher_is_better(clf["test_alert_24h_f1"])
    clf["score_alert_24h_recall"] = minmax_higher_is_better(clf["test_alert_24h_recall"])
    clf["score_alert_12h_f1"] = minmax_higher_is_better(clf["test_alert_12h_f1"])
    clf["score_quiet_f1"] = minmax_higher_is_better(clf["test_class_0_f1"])

    # Score orienté métier :
    # - macro_f1 : équilibre général multi-classe
    # - alert_24h_f1 : capacité à déclencher une alerte utile à 24h
    # - alert_24h_recall : limiter les faux négatifs d'alerte proche
    # - quiet_f1 : éviter une alerte permanente sur les périodes calmes
    # - balanced_accuracy : robustesse si classes déséquilibrées
    clf["classification_score"] = (
        0.25 * clf["score_macro_f1"].fillna(0.0)
        + 0.25 * clf["score_alert_24h_f1"].fillna(0.0)
        + 0.20 * clf["score_alert_24h_recall"].fillna(0.0)
        + 0.15 * clf["score_quiet_f1"].fillna(0.0)
        + 0.15 * clf["score_balanced_accuracy"].fillna(0.0)
    )

    out.loc[clf.index, "classification_score"] = clf["classification_score"]

    # Copie des scores intermédiaires dans out.
    for col in [
        "score_macro_f1",
        "score_balanced_accuracy",
        "score_alert_24h_f1",
        "score_alert_24h_recall",
        "score_alert_12h_f1",
        "score_quiet_f1",
    ]:
        out[col] = np.nan
        out.loc[clf.index, col] = clf[col]

    return out


def add_regression_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    reg_mask = out["task_type"] == "regression"

    out["regression_score"] = np.nan

    if reg_mask.sum() == 0:
        return out

    reg = out.loc[reg_mask].copy()

    reg["score_mae"] = minmax_lower_is_better(reg["test_mae"])
    reg["score_rmse"] = minmax_lower_is_better(reg["test_rmse"])
    reg["score_r2"] = minmax_higher_is_better(reg["test_r2"])

    val_gap = np.abs(reg["best_val_mae"] - reg["test_mae"])
    reg["val_test_gap"] = val_gap
    reg["score_stability"] = minmax_lower_is_better(val_gap)

    reg["regression_score"] = (
        0.40 * reg["score_mae"].fillna(0.0)
        + 0.25 * reg["score_rmse"].fillna(0.0)
        + 0.25 * reg["score_r2"].fillna(0.0)
        + 0.10 * reg["score_stability"].fillna(0.0)
    )

    out.loc[reg.index, "regression_score"] = reg["regression_score"]

    for col in [
        "score_mae",
        "score_rmse",
        "score_r2",
        "val_test_gap",
        "score_stability",
    ]:
        out[col] = np.nan
        out.loc[reg.index, col] = reg[col]

    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = add_classification_score(df)
    out = add_regression_score(out)

    out["overall_score"] = np.where(
        out["task_type"] == "classification",
        out["classification_score"],
        out["regression_score"],
    )

    out = out.sort_values(
        by=["task_type", "overall_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return out


# ============================================================
# SYNTHÈSES
# ============================================================

def rank_classification(df: pd.DataFrame) -> pd.DataFrame:
    clf = df[df["task_type"] == "classification"].copy()

    if clf.empty:
        return clf

    clf = clf.sort_values(
        by=[
            "classification_score",
            "test_alert_24h_f1",
            "test_macro_f1",
            "test_class_0_f1",
            "test_alert_24h_recall",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    clf["rank_classification"] = np.arange(1, len(clf) + 1)

    return clf


def rank_regression(df: pd.DataFrame) -> pd.DataFrame:
    reg = df[df["task_type"] == "regression"].copy()

    if reg.empty:
        return reg

    reg = reg.sort_values(
        by=[
            "regression_score",
            "test_r2",
            "test_mae",
            "test_rmse",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    reg["rank_regression"] = np.arange(1, len(reg) + 1)

    return reg


def summarize_classification_by_family(clf: pd.DataFrame) -> pd.DataFrame:
    if clf.empty:
        return pd.DataFrame()

    summary = (
        clf.groupby(["dataset_group", "model_family"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            best_classification_score=("classification_score", "max"),
            median_classification_score=("classification_score", "median"),

            best_test_macro_f1=("test_macro_f1", "max"),
            median_test_macro_f1=("test_macro_f1", "median"),

            best_test_balanced_accuracy=("test_balanced_accuracy", "max"),
            median_test_balanced_accuracy=("test_balanced_accuracy", "median"),

            best_test_alert_24h_f1=("test_alert_24h_f1", "max"),
            median_test_alert_24h_f1=("test_alert_24h_f1", "median"),

            best_test_alert_24h_recall=("test_alert_24h_recall", "max"),
            median_test_alert_24h_recall=("test_alert_24h_recall", "median"),

            best_test_class_0_f1=("test_class_0_f1", "max"),
            median_test_class_0_f1=("test_class_0_f1", "median"),
        )
        .reset_index()
        .sort_values(
            by=["dataset_group", "best_classification_score"],
            ascending=[True, False],
        )
    )

    return summary


def summarize_regression_by_family(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()

    summary = (
        reg.groupby(["dataset_group", "model_family"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            best_regression_score=("regression_score", "max"),
            median_regression_score=("regression_score", "median"),

            best_test_mae=("test_mae", "min"),
            median_test_mae=("test_mae", "median"),

            best_test_rmse=("test_rmse", "min"),
            median_test_rmse=("test_rmse", "median"),

            best_test_r2=("test_r2", "max"),
            median_test_r2=("test_r2", "median"),
        )
        .reset_index()
        .sort_values(
            by=["dataset_group", "best_regression_score"],
            ascending=[True, False],
        )
    )

    return summary


def compare_before_after(clf_summary: pd.DataFrame, reg_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Comparaison conceptuelle avant/après :
    - no_quiet = anciens runs régression
    - with_quiet = nouveaux runs classification

    Les métriques ne sont pas directement comparables.
    Le tableau met donc côte à côte les meilleurs indicateurs disponibles.
    """
    rows = []

    families = ["transformer", "cnn_transformer", "cnn_bilstm"]

    for family in families:
        reg_row = pd.DataFrame()
        clf_row = pd.DataFrame()

        if not reg_summary.empty:
            reg_row = reg_summary[
                (reg_summary["dataset_group"] == "no_quiet")
                & (reg_summary["model_family"] == family)
            ]

        if not clf_summary.empty:
            clf_row = clf_summary[
                (clf_summary["dataset_group"] == "with_quiet")
                & (clf_summary["model_family"] == family)
            ]

        rows.append(
            {
                "model_family": family,

                "before_task": "regression_without_quiet",
                "before_n_runs": (
                    int(reg_row["n_runs"].iloc[0])
                    if not reg_row.empty
                    else 0
                ),
                "before_best_test_mae": (
                    float(reg_row["best_test_mae"].iloc[0])
                    if not reg_row.empty and pd.notna(reg_row["best_test_mae"].iloc[0])
                    else np.nan
                ),
                "before_best_test_rmse": (
                    float(reg_row["best_test_rmse"].iloc[0])
                    if not reg_row.empty and pd.notna(reg_row["best_test_rmse"].iloc[0])
                    else np.nan
                ),
                "before_best_test_r2": (
                    float(reg_row["best_test_r2"].iloc[0])
                    if not reg_row.empty and pd.notna(reg_row["best_test_r2"].iloc[0])
                    else np.nan
                ),

                "after_task": "multiclass_classification_with_quiet",
                "after_n_runs": (
                    int(clf_row["n_runs"].iloc[0])
                    if not clf_row.empty
                    else 0
                ),
                "after_best_macro_f1": (
                    float(clf_row["best_test_macro_f1"].iloc[0])
                    if not clf_row.empty and pd.notna(clf_row["best_test_macro_f1"].iloc[0])
                    else np.nan
                ),
                "after_best_alert_24h_f1": (
                    float(clf_row["best_test_alert_24h_f1"].iloc[0])
                    if not clf_row.empty and pd.notna(clf_row["best_test_alert_24h_f1"].iloc[0])
                    else np.nan
                ),
                "after_best_alert_24h_recall": (
                    float(clf_row["best_test_alert_24h_recall"].iloc[0])
                    if not clf_row.empty and pd.notna(clf_row["best_test_alert_24h_recall"].iloc[0])
                    else np.nan
                ),
                "after_best_quiet_f1": (
                    float(clf_row["best_test_class_0_f1"].iloc[0])
                    if not clf_row.empty and pd.notna(clf_row["best_test_class_0_f1"].iloc[0])
                    else np.nan
                ),
                "comparison_note": (
                    "Les métriques avant/après ne sont pas directement comparables : "
                    "la tâche passe d'une régression horaire sans périodes calmes "
                    "à une classification multi-classes avec périodes calmes."
                ),
            }
        )

    return pd.DataFrame(rows)


def select_best_models(clf_ranked: pd.DataFrame, reg_ranked: pd.DataFrame) -> dict:
    result = {}

    if not clf_ranked.empty:
        best_clf = clf_ranked.iloc[0]
        result["best_classification_with_quiet"] = row_to_jsonable(best_clf)

    if not reg_ranked.empty:
        best_reg = reg_ranked.iloc[0]
        result["best_regression_overall"] = row_to_jsonable(best_reg)

        reg_no_quiet = reg_ranked[reg_ranked["dataset_group"] == "no_quiet"]
        if not reg_no_quiet.empty:
            result["best_regression_without_quiet"] = row_to_jsonable(reg_no_quiet.iloc[0])

    result["selection_logic"] = {
        "classification": (
            "Score composite orienté alerte : macro_f1, alert_24h_f1, "
            "alert_24h_recall, f1 de la classe calme et balanced_accuracy."
        ),
        "regression": (
            "Score composite : test_mae, test_rmse, test_r2 et stabilité validation/test."
        ),
        "important_warning": (
            "Les runs sans périodes calmes et avec périodes calmes ne mesurent pas "
            "la même tâche. Le modèle classification avec périodes calmes doit devenir "
            "le modèle principal d'alerte ; l'ancien modèle de régression peut rester "
            "une baseline d'estimation temporelle."
        ),
    }

    return result


def row_to_jsonable(row: pd.Series) -> dict:
    payload = {}

    keys = [
        "run_id",
        "run_name",
        "model_family",
        "task_type",
        "dataset_group",
        "input_npz",
        "model_type",
        "overall_score",
        "classification_score",
        "regression_score",

        "test_macro_f1",
        "test_balanced_accuracy",
        "test_alert_24h_f1",
        "test_alert_24h_precision",
        "test_alert_24h_recall",
        "test_alert_12h_f1",
        "test_alert_6h_f1",
        "test_class_0_f1",

        "test_mae",
        "test_rmse",
        "test_r2",

        "n_features",
        "n_classes",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "class_weighting",
        "label_smoothing",
        "early_stopping_metric",
        "d_model",
        "n_heads",
        "nhead",
        "n_layers",
        "num_layers",
        "dim_feedforward",
        "conv_channels",
        "lstm_hidden",
        "lstm_layers",
    ]

    for key in keys:
        if key not in row.index:
            continue

        value = row[key]

        if isinstance(value, (np.integer,)):
            payload[key] = int(value)
        elif isinstance(value, (np.floating,)):
            payload[key] = None if pd.isna(value) else float(value)
        elif pd.isna(value):
            payload[key] = None
        else:
            payload[key] = value

    return payload


# ============================================================
# EXPORT
# ============================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_main_results(clf_ranked, reg_ranked, clf_summary, reg_summary, comparison):
    print("\n" + "=" * 120)
    print("MEILLEUR MODÈLE CLASSIFICATION AVEC PÉRIODES CALMES")
    print("=" * 120)

    if clf_ranked.empty:
        print("Aucun run classification trouvé.")
    else:
        cols = [
            "rank_classification",
            "model_family",
            "run_name",
            "classification_score",
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_alert_24h_f1",
            "test_alert_24h_precision",
            "test_alert_24h_recall",
            "test_class_0_f1",
            "input_npz",
        ]
        cols = [c for c in cols if c in clf_ranked.columns]
        print(clf_ranked[cols].head(15).to_string(index=False))

    print("\n" + "=" * 120)
    print("MEILLEURS MODÈLES RÉGRESSION SANS / AVEC ANCIENS DATASETS")
    print("=" * 120)

    if reg_ranked.empty:
        print("Aucun run régression trouvé.")
    else:
        cols = [
            "rank_regression",
            "dataset_group",
            "model_family",
            "run_name",
            "regression_score",
            "test_mae",
            "test_rmse",
            "test_r2",
            "input_npz",
        ]
        cols = [c for c in cols if c in reg_ranked.columns]
        print(reg_ranked[cols].head(15).to_string(index=False))

    print("\n" + "=" * 120)
    print("SYNTHÈSE CLASSIFICATION PAR FAMILLE")
    print("=" * 120)
    if clf_summary.empty:
        print("Aucune synthèse classification.")
    else:
        print(clf_summary.to_string(index=False))

    print("\n" + "=" * 120)
    print("SYNTHÈSE RÉGRESSION PAR FAMILLE")
    print("=" * 120)
    if reg_summary.empty:
        print("Aucune synthèse régression.")
    else:
        print(reg_summary.to_string(index=False))

    print("\n" + "=" * 120)
    print("COMPARAISON AVANT / APRÈS PÉRIODES CALMES")
    print("=" * 120)
    print(comparison.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow, experiment = setup_mlflow(args)

    print(f"Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"Experiment   : {args.experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Output dir   : {output_dir}")

    runs_raw = load_runs(
        mlflow=mlflow,
        experiment_id=experiment.experiment_id,
        max_results=args.max_results,
    )

    raw_path = output_dir / "mlflow_runs_raw.csv"
    save_csv(runs_raw, raw_path)

    print(f"Runs bruts récupérés : {len(runs_raw)}")

    runs = extract_runs(runs_raw)
    runs_filtered = filter_runs(runs, args)
    runs_scored = add_scores(runs_filtered)

    if runs_scored.empty:
        raise ValueError(
            "Aucun run exploitable après filtrage. "
            "Vérifie les filtres dataset/model_family."
        )

    clf_ranked = rank_classification(runs_scored)
    reg_ranked = rank_regression(runs_scored)

    clf_summary = summarize_classification_by_family(clf_ranked)
    reg_summary = summarize_regression_by_family(reg_ranked)
    comparison = compare_before_after(clf_summary, reg_summary)

    best_models = select_best_models(clf_ranked, reg_ranked)

    # Exports globaux
    save_csv(runs, output_dir / "all_runs_extracted.csv")
    save_csv(runs_filtered, output_dir / "all_runs_filtered.csv")
    save_csv(runs_scored, output_dir / "all_runs_scored.csv")

    # Exports classification
    save_csv(clf_ranked, output_dir / "classification_with_quiet_ranked.csv")
    save_csv(clf_summary, output_dir / "classification_with_quiet_summary_by_family.csv")

    # Exports régression
    save_csv(reg_ranked, output_dir / "regression_ranked.csv")
    save_csv(reg_summary, output_dir / "regression_summary_by_family.csv")

    # Comparaison
    save_csv(comparison, output_dir / "quiet_vs_no_quiet_comparison.csv")
    save_json(best_models, output_dir / "best_models_recommendation.json")

    print_main_results(
        clf_ranked=clf_ranked,
        reg_ranked=reg_ranked,
        clf_summary=clf_summary,
        reg_summary=reg_summary,
        comparison=comparison,
    )

    print("\nFichiers écrits :")
    for path in [
        raw_path,
        output_dir / "all_runs_extracted.csv",
        output_dir / "all_runs_filtered.csv",
        output_dir / "all_runs_scored.csv",
        output_dir / "classification_with_quiet_ranked.csv",
        output_dir / "classification_with_quiet_summary_by_family.csv",
        output_dir / "regression_ranked.csv",
        output_dir / "regression_summary_by_family.csv",
        output_dir / "quiet_vs_no_quiet_comparison.csv",
        output_dir / "best_models_recommendation.json",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "Vulcadata"),
        help="Nom de l'expérience MLflow.",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Tracking URI MLflow. Si absent, utilise MLFLOW_TRACKING_URI.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/quiet_vs_no_quiet_sweep_analysis",
        help="Répertoire de sortie des rapports.",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=10000,
        help="Nombre maximal de runs MLflow à récupérer.",
    )

    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help=(
            "Filtre optionnel sur input_npz ou run_name. "
            "Exemples : with_quiet, stride5, full."
        ),
    )

    parser.add_argument(
        "--model-families",
        nargs="*",
        default=["transformer", "cnn_transformer", "cnn_bilstm"],
        choices=["transformer", "cnn_transformer", "cnn_bilstm", "unknown"],
        help="Familles de modèles à conserver.",
    )

    parser.add_argument(
        "--only-finished",
        action="store_true",
        help="Conserver uniquement les runs terminés.",
    )

    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Conserver les runs dont la famille ou tâche n'est pas reconnue.",
    )

    args = parser.parse_args()
    main(args)