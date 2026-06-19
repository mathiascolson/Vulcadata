# scripts/analyze_model_sweep_comparison.py

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
# CONFIG
# ============================================================

DEFAULT_MODEL_KEYWORDS = {
    "transformer": [
        "transformer",
        "TransformerEncoder",
    ],
    "cnn_transformer": [
        "cnn_transformer",
        "CNN_Transformer",
        "CNNTransformer",
        "cnn-transformer",
    ],
    "cnn_bilstm": [
        "cnn_bilstm",
        "CNN_BiLSTM",
        "bilstm",
        "BiLSTM",
    ],
}


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

    tracking_uri = (
        args.tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(args.experiment_name)

    if experiment is None:
        raise ValueError(
            f"Expérience MLflow introuvable : {args.experiment_name}"
        )

    return mlflow, experiment


def load_runs(mlflow, experiment_id: str, max_results: int = 5000) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )

    if runs.empty:
        raise ValueError("Aucun run MLflow trouvé dans cette expérience.")

    return runs


# ============================================================
# NORMALISATION COLONNES
# ============================================================

def safe_get(row: pd.Series, candidates: list[str], default=np.nan):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def infer_model_family(row: pd.Series) -> str:
    """
    Essaie d'identifier la famille du modèle à partir :
    - du paramètre model_type
    - du run_name
    - du nom du dossier input/output
    """
    text_parts = []

    for col in [
        "params.model_type",
        "tags.mlflow.runName",
        "params.run_name",
        "params.output_dir",
        "params.input_npz",
    ]:
        if col in row.index and pd.notna(row[col]):
            text_parts.append(str(row[col]))

    text = " ".join(text_parts).lower()

    for family, keywords in DEFAULT_MODEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                return family

    return "unknown"


def get_metric(row: pd.Series, metric_name: str):
    candidates = [
        f"metrics.{metric_name}",
        metric_name,
    ]
    return safe_get(row, candidates, default=np.nan)


def extract_clean_runs(runs: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in runs.iterrows():
        family = infer_model_family(row)

        run_id = safe_get(row, ["run_id", "attributes.run_id"], default=None)
        run_name = safe_get(row, ["tags.mlflow.runName"], default="")

        record = {
            "run_id": run_id,
            "run_name": run_name,
            "model_family": family,
            "status": safe_get(row, ["status", "attributes.status"], default=""),
            "start_time": safe_get(row, ["start_time", "attributes.start_time"], default=pd.NaT),
            "artifact_uri": safe_get(row, ["artifact_uri", "attributes.artifact_uri"], default=""),
            "input_npz": safe_get(row, ["params.input_npz"], default=""),
            "model_type": safe_get(row, ["params.model_type"], default=""),
            "seq_len": safe_get(row, ["params.seq_len"], default=np.nan),
            "n_features": safe_get(row, ["params.n_features"], default=np.nan),
            "batch_size": safe_get(row, ["params.batch_size"], default=np.nan),
            "learning_rate": safe_get(row, ["params.learning_rate"], default=np.nan),
            "weight_decay": safe_get(row, ["params.weight_decay"], default=np.nan),
            "dropout": safe_get(row, ["params.dropout"], default=np.nan),
            "seed": safe_get(row, ["params.seed"], default=np.nan),

            # Transformer params
            "d_model": safe_get(row, ["params.d_model"], default=np.nan),
            "n_heads": safe_get(row, ["params.n_heads"], default=np.nan),
            "n_layers": safe_get(row, ["params.n_layers"], default=np.nan),
            "dim_feedforward": safe_get(row, ["params.dim_feedforward"], default=np.nan),

            # CNN/LSTM params
            "conv_channels": safe_get(row, ["params.conv_channels"], default=np.nan),
            "lstm_hidden": safe_get(row, ["params.lstm_hidden"], default=np.nan),
            "lstm_layers": safe_get(row, ["params.lstm_layers"], default=np.nan),

            # Metrics finales test
            "test_mae": get_metric(row, "test_mae"),
            "test_rmse": get_metric(row, "test_rmse"),
            "test_mse": get_metric(row, "test_mse"),
            "test_r2": get_metric(row, "test_r2"),
            "test_loss": get_metric(row, "test_loss"),

            # Metrics pondérées si disponibles
            "test_weighted_mae": get_metric(row, "test_weighted_mae"),
            "test_weighted_rmse": get_metric(row, "test_weighted_rmse"),

            # Best validation
            "best_val_mae": get_metric(row, "best_val_mae"),
            "best_val_weighted_mae": get_metric(row, "best_val_weighted_mae"),
        }

        # Metrics par horizon si disponibles
        horizon_metrics = [
            "test_mae_0_6h",
            "test_mae_6_12h",
            "test_mae_12_24h",
            "test_mae_24_48h",
            "test_rmse_0_6h",
            "test_rmse_6_12h",
            "test_rmse_12_24h",
            "test_rmse_24_48h",
        ]

        for metric in horizon_metrics:
            record[metric] = get_metric(row, metric)

        records.append(record)

    clean = pd.DataFrame(records)

    # Conversion numérique robuste
    numeric_cols = [
        "seq_len",
        "n_features",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "seed",
        "d_model",
        "n_heads",
        "n_layers",
        "dim_feedforward",
        "conv_channels",
        "lstm_hidden",
        "lstm_layers",
        "test_mae",
        "test_rmse",
        "test_mse",
        "test_r2",
        "test_loss",
        "test_weighted_mae",
        "test_weighted_rmse",
        "best_val_mae",
        "best_val_weighted_mae",
        "test_mae_0_6h",
        "test_mae_6_12h",
        "test_mae_12_24h",
        "test_mae_24_48h",
        "test_rmse_0_6h",
        "test_rmse_6_12h",
        "test_rmse_12_24h",
        "test_rmse_24_48h",
    ]

    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    return clean


# ============================================================
# FILTRAGE
# ============================================================

def filter_runs(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    if args.keep_unknown is False:
        out = out[out["model_family"] != "unknown"].copy()

    if args.model_families:
        wanted = set(args.model_families)
        out = out[out["model_family"].isin(wanted)].copy()

    if args.dataset_filter:
        mask = out["input_npz"].fillna("").str.contains(
            args.dataset_filter,
            case=False,
            regex=False,
        )
        out = out[mask].copy()

    if args.run_name_filter:
        mask = out["run_name"].fillna("").str.contains(
            args.run_name_filter,
            case=False,
            regex=False,
        )
        out = out[mask].copy()

    if args.only_finished:
        # MLflow peut stocker FINISHED en attributes.status selon versions.
        out = out[
            out["status"].fillna("").astype(str).str.upper().isin(["FINISHED", ""])
        ].copy()

    # On garde uniquement les runs avec au moins une métrique test exploitable.
    metric_cols = [
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_weighted_mae",
    ]

    available_metric_mask = np.zeros(len(out), dtype=bool)

    for col in metric_cols:
        if col in out.columns:
            available_metric_mask |= out[col].notna().to_numpy()

    out = out[available_metric_mask].copy()

    return out


# ============================================================
# SCORING
# ============================================================

def minmax_lower_is_better(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        return pd.Series(1.0, index=s.index)

    # lower is better -> best = 1
    return 1.0 - ((s - min_val) / (max_val - min_val))


def minmax_higher_is_better(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        return pd.Series(1.0, index=s.index)

    # higher is better -> best = 1
    return (s - min_val) / (max_val - min_val)


def add_selection_score(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    # Score principal :
    # - Si weighted_mae existe, elle est prioritaire.
    # - Sinon fallback sur test_mae.
    if out["test_weighted_mae"].notna().any():
        out["primary_error"] = out["test_weighted_mae"].fillna(out["test_mae"])
        out["primary_error_name"] = np.where(
            out["test_weighted_mae"].notna(),
            "test_weighted_mae",
            "test_mae",
        )
    else:
        out["primary_error"] = out["test_mae"]
        out["primary_error_name"] = "test_mae"

    out["score_error"] = minmax_lower_is_better(out["primary_error"])
    out["score_rmse"] = minmax_lower_is_better(out["test_rmse"])
    out["score_r2"] = minmax_higher_is_better(out["test_r2"])

    # Stabilité validation/test :
    # si best_val_mae existe, pénalise les runs dont test_mae est très différent.
    if out["best_val_weighted_mae"].notna().any():
        val_reference = out["best_val_weighted_mae"].fillna(out["best_val_mae"])
    else:
        val_reference = out["best_val_mae"]

    out["val_test_gap"] = np.abs(val_reference - out["primary_error"])
    out["score_stability"] = minmax_lower_is_better(out["val_test_gap"])

    # Score par horizons proches, si disponible.
    # Plus faible MAE proche de l'éruption = mieux.
    if out["test_mae_0_6h"].notna().any() or out["test_mae_6_12h"].notna().any():
        close_error = (
            0.6 * out["test_mae_0_6h"].fillna(out["primary_error"])
            + 0.4 * out["test_mae_6_12h"].fillna(out["primary_error"])
        )
        out["close_horizon_error"] = close_error
        out["score_close_horizon"] = minmax_lower_is_better(close_error)
    else:
        out["close_horizon_error"] = np.nan
        out["score_close_horizon"] = np.nan

    # Pondérations :
    # - erreur principale : très important
    # - R² : important
    # - RMSE : important
    # - stabilité val/test : utile mais moins fiable vu le split par éruption
    # - horizon proche : prioritaire si disponible
    if out["score_close_horizon"].notna().any():
        out["selection_score"] = (
            0.35 * out["score_error"].fillna(0.0)
            + 0.20 * out["score_rmse"].fillna(0.0)
            + 0.20 * out["score_r2"].fillna(0.0)
            + 0.10 * out["score_stability"].fillna(0.0)
            + 0.15 * out["score_close_horizon"].fillna(0.0)
        )
    else:
        out["selection_score"] = (
            0.45 * out["score_error"].fillna(0.0)
            + 0.25 * out["score_rmse"].fillna(0.0)
            + 0.20 * out["score_r2"].fillna(0.0)
            + 0.10 * out["score_stability"].fillna(0.0)
        )

    out = out.sort_values(
        by=["selection_score", "test_r2", "test_mae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)

    return out


# ============================================================
# SYNTHÈSES
# ============================================================

def summarize_by_family(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("model_family", dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            best_selection_score=("selection_score", "max"),
            median_selection_score=("selection_score", "median"),
            best_test_mae=("test_mae", "min"),
            median_test_mae=("test_mae", "median"),
            best_test_rmse=("test_rmse", "min"),
            median_test_rmse=("test_rmse", "median"),
            best_test_r2=("test_r2", "max"),
            median_test_r2=("test_r2", "median"),
            best_primary_error=("primary_error", "min"),
            median_primary_error=("primary_error", "median"),
        )
        .reset_index()
        .sort_values(
            by=["best_selection_score", "best_test_r2", "best_test_mae"],
            ascending=[False, False, True],
        )
    )

    return agg


def build_recommendation(best_run: pd.Series, df: pd.DataFrame) -> dict:
    recommendation = {
        "selected_model_family": best_run["model_family"],
        "selected_run_name": best_run["run_name"],
        "selected_run_id": best_run["run_id"],
        "selection_score": float(best_run["selection_score"]),
        "primary_error_name": str(best_run["primary_error_name"]),
        "primary_error": float(best_run["primary_error"]) if pd.notna(best_run["primary_error"]) else None,
        "test_mae": float(best_run["test_mae"]) if pd.notna(best_run["test_mae"]) else None,
        "test_rmse": float(best_run["test_rmse"]) if pd.notna(best_run["test_rmse"]) else None,
        "test_r2": float(best_run["test_r2"]) if pd.notna(best_run["test_r2"]) else None,
        "test_weighted_mae": (
            float(best_run["test_weighted_mae"])
            if pd.notna(best_run["test_weighted_mae"])
            else None
        ),
        "best_val_mae": (
            float(best_run["best_val_mae"])
            if pd.notna(best_run["best_val_mae"])
            else None
        ),
        "best_val_weighted_mae": (
            float(best_run["best_val_weighted_mae"])
            if pd.notna(best_run["best_val_weighted_mae"])
            else None
        ),
        "input_npz": best_run["input_npz"],
        "model_type": best_run["model_type"],
        "hyperparameters": {
            "seq_len": safe_float_or_none(best_run.get("seq_len")),
            "n_features": safe_float_or_none(best_run.get("n_features")),
            "batch_size": safe_float_or_none(best_run.get("batch_size")),
            "learning_rate": safe_float_or_none(best_run.get("learning_rate")),
            "weight_decay": safe_float_or_none(best_run.get("weight_decay")),
            "dropout": safe_float_or_none(best_run.get("dropout")),
            "d_model": safe_float_or_none(best_run.get("d_model")),
            "n_heads": safe_float_or_none(best_run.get("n_heads")),
            "n_layers": safe_float_or_none(best_run.get("n_layers")),
            "dim_feedforward": safe_float_or_none(best_run.get("dim_feedforward")),
            "conv_channels": safe_float_or_none(best_run.get("conv_channels")),
            "lstm_hidden": safe_float_or_none(best_run.get("lstm_hidden")),
            "lstm_layers": safe_float_or_none(best_run.get("lstm_layers")),
        },
        "decision_rule": (
            "Le modèle est sélectionné selon un score composite : "
            "erreur principale, RMSE, R², stabilité validation/test et, si disponibles, "
            "métriques par horizon proche de l'éruption."
        ),
        "warning": (
            "La sélection reste dépendante du split actuel par éruption. "
            "Une validation leave-one-eruption-out serait nécessaire pour une conclusion robuste."
        ),
    }

    return recommendation


def safe_float_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def print_top_runs(df: pd.DataFrame, top_n: int):
    cols = [
        "rank",
        "model_family",
        "run_name",
        "selection_score",
        "primary_error_name",
        "primary_error",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_weighted_mae",
        "best_val_mae",
        "best_val_weighted_mae",
        "n_features",
        "input_npz",
        "run_id",
    ]

    available_cols = [c for c in cols if c in df.columns]

    print("\nTOP RUNS")
    print(df[available_cols].head(top_n).to_string(index=False))


def print_family_summary(summary: pd.DataFrame):
    print("\nSYNTHÈSE PAR FAMILLE")
    print(summary.to_string(index=False))


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

    runs_raw = load_runs(
        mlflow=mlflow,
        experiment_id=experiment.experiment_id,
        max_results=args.max_results,
    )

    raw_path = output_dir / "mlflow_runs_raw.csv"
    runs_raw.to_csv(raw_path, index=False)

    print(f"Runs bruts récupérés : {len(runs_raw)}")
    print(f"Export brut          : {raw_path}")

    clean = extract_clean_runs(runs_raw)
    filtered = filter_runs(clean, args)

    if filtered.empty:
        raise ValueError(
            "Aucun run exploitable après filtrage. "
            "Vérifie --dataset-filter, --run-name-filter ou les noms de modèles."
        )

    scored = add_selection_score(filtered, args)
    summary = summarize_by_family(scored)

    best_run = scored.iloc[0]
    recommendation = build_recommendation(best_run, scored)

    scored_path = output_dir / "model_sweep_comparison_ranked.csv"
    summary_path = output_dir / "model_sweep_summary_by_family.csv"
    recommendation_path = output_dir / "best_model_recommendation.json"

    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)

    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, ensure_ascii=False)

    print_top_runs(scored, args.top_n)
    print_family_summary(summary)

    print("\nMEILLEUR MODÈLE SÉLECTIONNÉ")
    print(f"Famille      : {recommendation['selected_model_family']}")
    print(f"Run name     : {recommendation['selected_run_name']}")
    print(f"Run ID       : {recommendation['selected_run_id']}")
    print(f"Score        : {recommendation['selection_score']:.4f}")
    print(f"Erreur princ.: {recommendation['primary_error_name']} = {recommendation['primary_error']}")
    print(f"Test MAE     : {recommendation['test_mae']}")
    print(f"Test RMSE    : {recommendation['test_rmse']}")
    print(f"Test R²      : {recommendation['test_r2']}")
    print(f"Input NPZ    : {recommendation['input_npz']}")

    print("\nFichiers écrits")
    print(f"  - {scored_path}")
    print(f"  - {summary_path}")
    print(f"  - {recommendation_path}")


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
        help=(
            "Tracking URI MLflow. "
            "Si absent, utilise MLFLOW_TRACKING_URI depuis .env."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/model_sweep_comparison",
        help="Répertoire de sortie des rapports.",
    )

    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help=(
            "Filtre optionnel sur params.input_npz. "
            "Exemples : stride3, stride5, core, full."
        ),
    )

    parser.add_argument(
        "--run-name-filter",
        type=str,
        default=None,
        help="Filtre optionnel sur le nom du run MLflow.",
    )

    parser.add_argument(
        "--model-families",
        nargs="*",
        default=None,
        choices=["transformer", "cnn_transformer", "cnn_bilstm", "unknown"],
        help="Familles de modèles à conserver.",
    )

    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Conserver les runs dont la famille de modèle n'est pas reconnue.",
    )

    parser.add_argument(
        "--only-finished",
        action="store_true",
        help="Conserver uniquement les runs terminés.",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=5000,
        help="Nombre maximal de runs à récupérer depuis MLflow.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Nombre de meilleurs runs affichés dans la console.",
    )

    args = parser.parse_args()
    main(args)