# scripts/analyze_targeted_mlflow_results.py

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
# MLFLOW
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

    experiment_name = (
        args.experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "Vulcadata"
    )

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable : {experiment_name}")

    return mlflow, experiment, experiment_name


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
# HELPERS GÉNÉRIQUES
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


def normalize_for_search(value) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .lower()
        .replace("\\", "/")
        .replace("-", "_")
    )


def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def is_finished_status(status) -> bool:
    value = str(status).upper()
    return value in {"FINISHED", ""}


def to_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def safe_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if pd.isna(value):
            return None
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


# ============================================================
# INFÉRENCES
# ============================================================

def collect_text(row: pd.Series) -> str:
    text_parts = []

    for col in [
        "params.model_type",
        "params.task",
        "params.target",
        "params.input_npz",
        "params.output_dir",
        "params.run_name",
        "tags.mlflow.runName",
        "artifact_uri",
        "attributes.artifact_uri",
    ]:
        if col in row.index and pd.notna(row[col]):
            text_parts.append(str(row[col]))

    return " ".join(text_parts).lower()


def infer_model_family(row: pd.Series) -> str:
    text = collect_text(row)
    compact = (
        text
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("\\", "")
    )

    # Important : détecter cnn_transformer avant transformer.
    if (
        "cnntransformer" in compact
        or "cnn_transformer" in text
        or "cnn-transformer" in text
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
        or "transformerclassifier" in compact
        or "transformerregressor" in compact
        or "transformer" in text
    ):
        return "transformer"

    return "unknown"


def infer_task_type(row: pd.Series) -> str:
    text = collect_text(row)

    classification_indicators = [
        "metrics.test_business_score_classification",
        "metrics.test_macro_f1",
        "metrics.test_balanced_accuracy",
        "metrics.test_alert_24h_f1",
        "metrics.test_class_0_f1",
        "params.n_classes",
        "crossentropyloss",
        "multiclass",
        "classification",
        "classifier",
    ]

    for indicator in classification_indicators:
        if indicator.startswith("metrics.") or indicator.startswith("params."):
            if indicator in row.index and pd.notna(row[indicator]):
                return "classification"
        elif indicator in text:
            return "classification"

    regression_indicators = [
        "metrics.test_business_score",
        "metrics.test_mae",
        "metrics.test_rmse",
        "metrics.test_r2",
        "metrics.best_val_mae",
        "smoothl1loss",
        "regression",
        "regressor",
    ]

    for indicator in regression_indicators:
        if indicator.startswith("metrics."):
            if indicator in row.index and pd.notna(row[indicator]):
                return "regression"
        elif indicator in text:
            return "regression"

    return "unknown"


def infer_dataset_group(row: pd.Series) -> str:
    text = collect_text(row)

    if "with_quiet" in text or "quiet" in text or "calme" in text:
        return "with_quiet"

    return "no_quiet"


def infer_feature_set(row: pd.Series) -> str:
    text = collect_text(row)

    if "processed_core" in text or "_core_" in text or "/core/" in text or "core" in text:
        return "core"

    if "processed_full" in text or "_full_" in text or "/full/" in text or "full" in text:
        return "full"

    return "unknown"


def infer_sweep_type(row: pd.Series) -> str:
    text = collect_text(row)

    if "targeted" in text or "finetuning" in text or "fine_tuning" in text:
        return "targeted"

    if "sweep" in text:
        return "sweep"

    return "unknown"


# ============================================================
# EXTRACTION DES RUNS
# ============================================================

def extract_runs(runs_raw: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in runs_raw.iterrows():
        run_id = safe_get(row, ["run_id", "attributes.run_id"], default="")
        run_name = safe_get(row, ["tags.mlflow.runName"], default="")
        status = safe_get(row, ["status", "attributes.status"], default="")
        start_time = safe_get(row, ["start_time", "attributes.start_time"], default=pd.NaT)
        artifact_uri = safe_get(row, ["artifact_uri", "attributes.artifact_uri"], default="")

        model_family = infer_model_family(row)
        task_type = infer_task_type(row)
        dataset_group = infer_dataset_group(row)
        feature_set = infer_feature_set(row)
        sweep_type = infer_sweep_type(row)

        record = {
            "run_id": run_id,
            "run_name": run_name,
            "status": status,
            "start_time": start_time,
            "artifact_uri": artifact_uri,

            "model_family": model_family,
            "task_type": task_type,
            "dataset_group": dataset_group,
            "feature_set": feature_set,
            "sweep_type": sweep_type,

            "input_npz": get_param(row, "input_npz"),
            "model_type": get_param(row, "model_type"),
            "task_param": get_param(row, "task"),
            "target": get_param(row, "target"),

            "seq_len": get_param(row, "seq_len"),
            "n_features": get_param(row, "n_features"),
            "n_classes": get_param(row, "n_classes"),

            "batch_size": get_param(row, "batch_size"),
            "learning_rate": get_param(row, "learning_rate"),
            "weight_decay": get_param(row, "weight_decay"),
            "dropout": get_param(row, "dropout"),
            "input_noise_std": get_param(row, "input_noise_std"),
            "seed": get_param(row, "seed"),

            # Régression
            "loss": get_param(row, "loss"),
            "loss_weighting": get_param(row, "loss_weighting"),
            "huber_beta": get_param(row, "huber_beta"),
            "max_horizon_hours": get_param(row, "max_horizon_hours"),

            # Classification
            "class_weighting": get_param(row, "class_weighting"),
            "label_smoothing": get_param(row, "label_smoothing"),

            # Early stopping
            "early_stopping_metric": get_param(row, "early_stopping_metric"),
            "early_stopping_mode": get_param(row, "early_stopping_mode"),
            "early_stopping_patience": get_param(row, "early_stopping_patience"),
            "early_stopping_min_delta": get_param(row, "early_stopping_min_delta"),

            # Transformer / CNN-Transformer
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

            # Paramètres modèle
            "n_params": get_metric(row, "n_params"),
            "n_trainable_params": get_metric(row, "n_trainable_params"),

            # ====================================================
            # Métriques régression
            # ====================================================
            "best_val_mae": get_metric(row, "best_val_mae"),
            "best_val_business_score": get_metric(row, "best_val_business_score"),

            "test_loss": get_metric(row, "test_loss"),
            "test_mae": get_metric(row, "test_mae"),
            "test_mse": get_metric(row, "test_mse"),
            "test_rmse": get_metric(row, "test_rmse"),
            "test_r2": get_metric(row, "test_r2"),
            "test_business_score": get_metric(row, "test_business_score"),

            "test_n_0_6h": get_metric(row, "test_n_0_6h"),
            "test_mae_0_6h": get_metric(row, "test_mae_0_6h"),
            "test_rmse_0_6h": get_metric(row, "test_rmse_0_6h"),

            "test_n_6_12h": get_metric(row, "test_n_6_12h"),
            "test_mae_6_12h": get_metric(row, "test_mae_6_12h"),
            "test_rmse_6_12h": get_metric(row, "test_rmse_6_12h"),

            "test_n_12_24h": get_metric(row, "test_n_12_24h"),
            "test_mae_12_24h": get_metric(row, "test_mae_12_24h"),
            "test_rmse_12_24h": get_metric(row, "test_rmse_12_24h"),

            "test_n_24_36h": get_metric(row, "test_n_24_36h"),
            "test_mae_24_36h": get_metric(row, "test_mae_24_36h"),
            "test_rmse_24_36h": get_metric(row, "test_rmse_24_36h"),

            "test_n_36_48h": get_metric(row, "test_n_36_48h"),
            "test_mae_36_48h": get_metric(row, "test_mae_36_48h"),
            "test_rmse_36_48h": get_metric(row, "test_rmse_36_48h"),

            # ====================================================
            # Métriques classification
            # ====================================================
            "best_val_macro_f1": get_metric(row, "best_val_macro_f1"),
            "best_val_business_score_classification": get_metric(
                row,
                "best_val_business_score_classification",
            ),

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

            "test_business_score_classification": get_metric(
                row,
                "test_business_score_classification",
            ),

            "test_class_0_precision": get_metric(row, "test_class_0_precision"),
            "test_class_0_recall": get_metric(row, "test_class_0_recall"),
            "test_class_0_f1": get_metric(row, "test_class_0_f1"),
            "test_class_0_support": get_metric(row, "test_class_0_support"),

            "test_class_1_precision": get_metric(row, "test_class_1_precision"),
            "test_class_1_recall": get_metric(row, "test_class_1_recall"),
            "test_class_1_f1": get_metric(row, "test_class_1_f1"),
            "test_class_1_support": get_metric(row, "test_class_1_support"),

            "test_class_2_precision": get_metric(row, "test_class_2_precision"),
            "test_class_2_recall": get_metric(row, "test_class_2_recall"),
            "test_class_2_f1": get_metric(row, "test_class_2_f1"),
            "test_class_2_support": get_metric(row, "test_class_2_support"),

            "test_class_3_precision": get_metric(row, "test_class_3_precision"),
            "test_class_3_recall": get_metric(row, "test_class_3_recall"),
            "test_class_3_f1": get_metric(row, "test_class_3_f1"),
            "test_class_3_support": get_metric(row, "test_class_3_support"),

            "test_class_4_precision": get_metric(row, "test_class_4_precision"),
            "test_class_4_recall": get_metric(row, "test_class_4_recall"),
            "test_class_4_f1": get_metric(row, "test_class_4_f1"),
            "test_class_4_support": get_metric(row, "test_class_4_support"),

            "test_class_5_precision": get_metric(row, "test_class_5_precision"),
            "test_class_5_recall": get_metric(row, "test_class_5_recall"),
            "test_class_5_f1": get_metric(row, "test_class_5_f1"),
            "test_class_5_support": get_metric(row, "test_class_5_support"),
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
        "seed",
        "huber_beta",
        "max_horizon_hours",
        "label_smoothing",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "d_model",
        "n_heads",
        "nhead",
        "n_layers",
        "num_layers",
        "dim_feedforward",
        "conv_channels",
        "lstm_hidden",
        "lstm_layers",
        "n_params",
        "n_trainable_params",

        "best_val_mae",
        "best_val_business_score",
        "test_loss",
        "test_mae",
        "test_mse",
        "test_rmse",
        "test_r2",
        "test_business_score",

        "test_n_0_6h",
        "test_mae_0_6h",
        "test_rmse_0_6h",
        "test_n_6_12h",
        "test_mae_6_12h",
        "test_rmse_6_12h",
        "test_n_12_24h",
        "test_mae_12_24h",
        "test_rmse_12_24h",
        "test_n_24_36h",
        "test_mae_24_36h",
        "test_rmse_24_36h",
        "test_n_36_48h",
        "test_mae_36_48h",
        "test_rmse_36_48h",

        "best_val_macro_f1",
        "best_val_business_score_classification",
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
        "test_business_score_classification",

        "test_class_0_precision",
        "test_class_0_recall",
        "test_class_0_f1",
        "test_class_0_support",
        "test_class_1_precision",
        "test_class_1_recall",
        "test_class_1_f1",
        "test_class_1_support",
        "test_class_2_precision",
        "test_class_2_recall",
        "test_class_2_f1",
        "test_class_2_support",
        "test_class_3_precision",
        "test_class_3_recall",
        "test_class_3_f1",
        "test_class_3_support",
        "test_class_4_precision",
        "test_class_4_recall",
        "test_class_4_f1",
        "test_class_4_support",
        "test_class_5_precision",
        "test_class_5_recall",
        "test_class_5_f1",
        "test_class_5_support",
    ]

    return to_numeric_columns(df, numeric_cols)


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
        out = out[out["model_family"].isin(args.model_families)].copy()

    if args.task_types:
        out = out[out["task_type"].isin(args.task_types)].copy()

    if args.feature_sets:
        out = out[out["feature_set"].isin(args.feature_sets)].copy()

    if args.dataset_groups:
        out = out[out["dataset_group"].isin(args.dataset_groups)].copy()

    if args.dataset_filter:
        filter_text = args.dataset_filter.lower()

        mask = (
            out["input_npz"].fillna("").astype(str).str.lower().str.contains(
                filter_text,
                regex=False,
            )
            | out["run_name"].fillna("").astype(str).str.lower().str.contains(
                filter_text,
                regex=False,
            )
        )

        out = out[mask].copy()

    if args.run_name_filter:
        filter_text = args.run_name_filter.lower()

        mask = out["run_name"].fillna("").astype(str).str.lower().str.contains(
            filter_text,
            regex=False,
        )

        out = out[mask].copy()

    useful_metrics = [
        "test_business_score",
        "test_business_score_classification",
        "test_mae",
        "test_macro_f1",
        "test_alert_24h_f1",
    ]

    useful_mask = np.zeros(len(out), dtype=bool)

    for col in useful_metrics:
        if col in out.columns:
            useful_mask |= out[col].notna().to_numpy()

    out = out[useful_mask].copy()

    return out


# ============================================================
# RANKING
# ============================================================

def rank_regression(df: pd.DataFrame) -> pd.DataFrame:
    reg = df[df["task_type"] == "regression"].copy()

    if reg.empty:
        return reg

    if "test_business_score" not in reg.columns:
        reg["test_business_score"] = np.nan

    # Fallback : si business_score absent, on classe sur MAE.
    reg["ranking_business_score"] = reg["test_business_score"]
    reg.loc[reg["ranking_business_score"].isna(), "ranking_business_score"] = reg["test_mae"]

    reg = reg.sort_values(
        by=[
            "ranking_business_score",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_mae_36_48h",
            "test_mae_24_36h",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    reg["rank_regression"] = np.arange(1, len(reg) + 1)

    return reg


def rank_classification(df: pd.DataFrame) -> pd.DataFrame:
    clf = df[df["task_type"] == "classification"].copy()

    if clf.empty:
        return clf

    if "test_business_score_classification" not in clf.columns:
        clf["test_business_score_classification"] = np.nan

    # Fallback : si business_score_classification absent, on classe sur macro_f1.
    clf["ranking_business_score_classification"] = clf["test_business_score_classification"]
    clf.loc[
        clf["ranking_business_score_classification"].isna(),
        "ranking_business_score_classification",
    ] = clf["test_macro_f1"]

    clf = clf.sort_values(
        by=[
            "ranking_business_score_classification",
            "test_macro_f1",
            "test_class_1_f1",
            "test_class_2_f1",
            "test_class_0_f1",
            "test_alert_24h_f1",
            "test_alert_24h_recall",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    clf["rank_classification"] = np.arange(1, len(clf) + 1)

    return clf


# ============================================================
# SYNTHÈSES
# ============================================================

def summarize_regression(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()

    summary = (
        reg.groupby(
            [
                "dataset_group",
                "feature_set",
                "model_family",
                "loss_weighting",
            ],
            dropna=False,
        )
        .agg(
            n_runs=("run_id", "count"),
            best_business_score=("test_business_score", "min"),
            median_business_score=("test_business_score", "median"),
            best_mae=("test_mae", "min"),
            median_mae=("test_mae", "median"),
            best_rmse=("test_rmse", "min"),
            best_r2=("test_r2", "max"),
            best_mae_36_48h=("test_mae_36_48h", "min"),
            best_mae_24_36h=("test_mae_24_36h", "min"),
            best_mae_12_24h=("test_mae_12_24h", "min"),
            best_mae_6_12h=("test_mae_6_12h", "min"),
            best_mae_0_6h=("test_mae_0_6h", "min"),
        )
        .reset_index()
        .sort_values(
            by=[
                "best_business_score",
                "best_mae",
                "best_r2",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )

    return summary


def summarize_classification(clf: pd.DataFrame) -> pd.DataFrame:
    if clf.empty:
        return pd.DataFrame()

    summary = (
        clf.groupby(
            [
                "dataset_group",
                "feature_set",
                "model_family",
                "class_weighting",
            ],
            dropna=False,
        )
        .agg(
            n_runs=("run_id", "count"),
            best_business_score_classification=("test_business_score_classification", "max"),
            median_business_score_classification=("test_business_score_classification", "median"),
            best_macro_f1=("test_macro_f1", "max"),
            median_macro_f1=("test_macro_f1", "median"),
            best_balanced_accuracy=("test_balanced_accuracy", "max"),
            best_alert_24h_f1=("test_alert_24h_f1", "max"),
            best_alert_24h_recall=("test_alert_24h_recall", "max"),
            best_quiet_f1=("test_class_0_f1", "max"),
            best_class_1_f1=("test_class_1_f1", "max"),
            best_class_2_f1=("test_class_2_f1", "max"),
            best_class_3_f1=("test_class_3_f1", "max"),
            best_class_4_f1=("test_class_4_f1", "max"),
            best_class_5_f1=("test_class_5_f1", "max"),
        )
        .reset_index()
        .sort_values(
            by=[
                "best_business_score_classification",
                "best_macro_f1",
                "best_class_1_f1",
                "best_class_2_f1",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
    )

    return summary


def summarize_feature_set_effect(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()

    rows = []

    for task_type in sorted(ranked["task_type"].dropna().unique()):
        task_df = ranked[ranked["task_type"] == task_type].copy()

        if task_df.empty:
            continue

        for model_family in sorted(task_df["model_family"].dropna().unique()):
            fam_df = task_df[task_df["model_family"] == model_family].copy()

            for feature_set in sorted(fam_df["feature_set"].dropna().unique()):
                subset = fam_df[fam_df["feature_set"] == feature_set].copy()

                if subset.empty:
                    continue

                row = {
                    "task_type": task_type,
                    "model_family": model_family,
                    "feature_set": feature_set,
                    "n_runs": len(subset),
                }

                if task_type == "regression":
                    row.update({
                        "best_business_score": subset["test_business_score"].min(),
                        "best_mae": subset["test_mae"].min(),
                        "best_r2": subset["test_r2"].max(),
                        "best_mae_36_48h": subset["test_mae_36_48h"].min(),
                        "best_mae_24_36h": subset["test_mae_24_36h"].min(),
                    })
                else:
                    row.update({
                        "best_business_score_classification": subset[
                            "test_business_score_classification"
                        ].max(),
                        "best_macro_f1": subset["test_macro_f1"].max(),
                        "best_alert_24h_f1": subset["test_alert_24h_f1"].max(),
                        "best_class_0_f1": subset["test_class_0_f1"].max(),
                        "best_class_1_f1": subset["test_class_1_f1"].max(),
                        "best_class_2_f1": subset["test_class_2_f1"].max(),
                    })

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# BEST MODEL JSON
# ============================================================

def row_to_jsonable(row: pd.Series) -> dict:
    keys = [
        "run_id",
        "run_name",
        "status",
        "start_time",
        "artifact_uri",
        "model_family",
        "task_type",
        "dataset_group",
        "feature_set",
        "sweep_type",
        "input_npz",
        "model_type",
        "n_features",
        "seq_len",
        "n_classes",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "input_noise_std",
        "loss_weighting",
        "class_weighting",
        "label_smoothing",
        "early_stopping_metric",
        "conv_channels",
        "lstm_hidden",
        "lstm_layers",
        "d_model",
        "n_heads",
        "nhead",
        "n_layers",
        "num_layers",
        "dim_feedforward",
        "n_params",
        "n_trainable_params",

        "test_business_score",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_mae_36_48h",
        "test_mae_24_36h",
        "test_mae_12_24h",
        "test_mae_6_12h",
        "test_mae_0_6h",

        "test_business_score_classification",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_class_0_f1",
        "test_class_1_f1",
        "test_class_2_f1",
        "test_class_3_f1",
        "test_class_4_f1",
        "test_class_5_f1",
        "test_alert_24h_precision",
        "test_alert_24h_recall",
        "test_alert_24h_f1",
        "test_alert_12h_f1",
        "test_alert_6h_f1",
    ]

    payload = {}

    for key in keys:
        if key not in row.index:
            continue

        payload[key] = safe_json_value(row[key])

    return payload


def build_best_models_payload(reg_ranked: pd.DataFrame, clf_ranked: pd.DataFrame) -> dict:
    payload = {}

    if not reg_ranked.empty:
        best_reg = reg_ranked.iloc[0]
        payload["best_regression"] = row_to_jsonable(best_reg)

        for feature_set in ["full", "core"]:
            subset = reg_ranked[reg_ranked["feature_set"] == feature_set]
            if not subset.empty:
                payload[f"best_regression_{feature_set}"] = row_to_jsonable(subset.iloc[0])

    if not clf_ranked.empty:
        best_clf = clf_ranked.iloc[0]
        payload["best_classification"] = row_to_jsonable(best_clf)

        for feature_set in ["full", "core"]:
            subset = clf_ranked[clf_ranked["feature_set"] == feature_set]
            if not subset.empty:
                payload[f"best_classification_{feature_set}"] = row_to_jsonable(subset.iloc[0])

    payload["selection_logic"] = {
        "regression": (
            "Classement principal sur test_business_score croissant. "
            "Fallback sur test_mae si business_score absent. "
            "Le business_score favorise l'anticipation précoce sans ignorer les horizons proches."
        ),
        "classification": (
            "Classement principal sur test_business_score_classification décroissant. "
            "Fallback sur test_macro_f1 si business_score_classification absent. "
            "Le score classification valorise le calme, les classes précoces 36-48h / 24-36h, "
            "l'alerte 24h, macro_f1 et balanced_accuracy."
        ),
        "warning": (
            "Les résultats restent dépendants du split actuel. "
            "Un leave-one-eruption-out reste nécessaire pour valider la robustesse inter-éruption."
        ),
    }

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


# ============================================================
# AFFICHAGE CONSOLE
# ============================================================

def print_regression_results(reg_ranked: pd.DataFrame, top_n: int) -> None:
    print("\n" + "=" * 120)
    print("TOP RÉGRESSION")
    print("=" * 120)

    if reg_ranked.empty:
        print("Aucun run régression exploitable.")
        return

    cols = [
        "rank_regression",
        "model_family",
        "feature_set",
        "loss_weighting",
        "run_name",
        "test_business_score",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_mae_36_48h",
        "test_mae_24_36h",
        "test_mae_12_24h",
        "test_mae_6_12h",
        "test_mae_0_6h",
    ]

    cols = [c for c in cols if c in reg_ranked.columns]

    print(reg_ranked[cols].head(top_n).to_string(index=False))


def print_classification_results(clf_ranked: pd.DataFrame, top_n: int) -> None:
    print("\n" + "=" * 120)
    print("TOP CLASSIFICATION")
    print("=" * 120)

    if clf_ranked.empty:
        print("Aucun run classification exploitable.")
        return

    cols = [
        "rank_classification",
        "model_family",
        "feature_set",
        "class_weighting",
        "run_name",
        "test_business_score_classification",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_class_0_f1",
        "test_class_1_f1",
        "test_class_2_f1",
        "test_alert_24h_f1",
        "test_alert_24h_recall",
    ]

    cols = [c for c in cols if c in clf_ranked.columns]

    print(clf_ranked[cols].head(top_n).to_string(index=False))


def print_summaries(reg_summary: pd.DataFrame, clf_summary: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("SYNTHÈSE RÉGRESSION")
    print("=" * 120)

    if reg_summary.empty:
        print("Aucune synthèse régression.")
    else:
        print(reg_summary.to_string(index=False))

    print("\n" + "=" * 120)
    print("SYNTHÈSE CLASSIFICATION")
    print("=" * 120)

    if clf_summary.empty:
        print("Aucune synthèse classification.")
    else:
        print(clf_summary.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow, experiment, experiment_name = setup_mlflow(args)

    print(f"Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"Experiment   : {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Output dir   : {output_dir}")

    runs_raw = load_runs(
        mlflow=mlflow,
        experiment_id=experiment.experiment_id,
        max_results=args.max_results,
    )

    print(f"Runs bruts récupérés : {len(runs_raw)}")

    raw_path = output_dir / "mlflow_runs_raw.csv"
    save_csv(runs_raw, raw_path)

    runs_extracted = extract_runs(runs_raw)
    runs_filtered = filter_runs(runs_extracted, args)

    if runs_filtered.empty:
        raise ValueError(
            "Aucun run exploitable après filtrage. "
            "Vérifie --dataset-filter, --run-name-filter ou les familles de modèles."
        )

    reg_ranked = rank_regression(runs_filtered)
    clf_ranked = rank_classification(runs_filtered)

    reg_summary = summarize_regression(reg_ranked)
    clf_summary = summarize_classification(clf_ranked)

    combined_ranked = pd.concat(
        [reg_ranked, clf_ranked],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    feature_set_summary = summarize_feature_set_effect(combined_ranked)
    best_models = build_best_models_payload(reg_ranked, clf_ranked)

    save_csv(runs_extracted, output_dir / "all_runs_extracted.csv")
    save_csv(runs_filtered, output_dir / "all_runs_filtered.csv")
    save_csv(combined_ranked, output_dir / "all_runs_ranked.csv")

    save_csv(reg_ranked, output_dir / "regression_ranked.csv")
    save_csv(reg_summary, output_dir / "regression_summary.csv")

    save_csv(clf_ranked, output_dir / "classification_ranked.csv")
    save_csv(clf_summary, output_dir / "classification_summary.csv")

    save_csv(feature_set_summary, output_dir / "feature_set_summary.csv")
    save_json(best_models, output_dir / "best_models_recommendation.json")

    print_regression_results(reg_ranked, top_n=args.top_n)
    print_classification_results(clf_ranked, top_n=args.top_n)
    print_summaries(reg_summary, clf_summary)

    print("\n" + "=" * 120)
    print("FICHIERS ÉCRITS")
    print("=" * 120)

    for path in [
        raw_path,
        output_dir / "all_runs_extracted.csv",
        output_dir / "all_runs_filtered.csv",
        output_dir / "all_runs_ranked.csv",
        output_dir / "regression_ranked.csv",
        output_dir / "regression_summary.csv",
        output_dir / "classification_ranked.csv",
        output_dir / "classification_summary.csv",
        output_dir / "feature_set_summary.csv",
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
        default="reports/targeted_mlflow_analysis",
        help="Répertoire de sortie.",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=10000,
        help="Nombre maximal de runs MLflow à récupérer.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Nombre de runs affichés dans le terminal.",
    )

    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help=(
            "Filtre optionnel sur input_npz ou run_name. "
            "Exemples : stride5, with_quiet, targeted."
        ),
    )

    parser.add_argument(
        "--run-name-filter",
        type=str,
        default=None,
        help=(
            "Filtre optionnel uniquement sur le nom du run. "
            "Exemples : targeted, finetuning, with_quiet."
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
        "--task-types",
        nargs="*",
        default=None,
        choices=["regression", "classification", "unknown"],
        help="Types de tâche à conserver.",
    )

    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        choices=["full", "core", "unknown"],
        help="Types de features à conserver.",
    )

    parser.add_argument(
        "--dataset-groups",
        nargs="*",
        default=None,
        choices=["with_quiet", "no_quiet"],
        help="Groupes de dataset à conserver.",
    )

    parser.add_argument(
        "--only-finished",
        action="store_true",
        help="Conserver uniquement les runs terminés.",
    )

    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Conserver les runs dont la famille ou la tâche n'est pas reconnue.",
    )

    args = parser.parse_args()
    main(args)