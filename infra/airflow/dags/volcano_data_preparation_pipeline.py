from __future__ import annotations

import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


DAG_ID = "volcano_data_preparation_pipeline"

PROJECT_ROOT = os.getenv("VULCADATA_PROJECT_ROOT", "/opt/vulcadata")

PERIODS_CSV = os.getenv(
    "VULCADATA_DATA_PREP_PERIODS_CSV",
    "data/metadata/extraction_periods.csv",
)
PROCESSED_CSV_DIR = os.getenv(
    "VULCADATA_PROCESSED_CSV_DIR",
    "data/extraction/processed_csv",
)
OUTPUT_DIR = os.getenv(
    "VULCADATA_DATA_PREP_OUTPUT_DIR",
    "data/preprocessing/processed",
)
TRAINING_PERIODS_CSV = os.getenv(
    "VULCADATA_TRAINING_PERIODS_RESOLVED_CSV",
    "reports/data_preparation/training_periods_for_preprocessing.csv",
)
CSV_INPUT_CHECK_JSON = os.getenv(
    "VULCADATA_CSV_INPUT_CHECK_JSON",
    "reports/data_preparation/processed_csv_input_check.json",
)
TRAINING_OUTPUT_NAME = os.getenv(
    "VULCADATA_TRAINING_OUTPUT_NAME",
    "volcano_multi.npz",
)
TRAINING_NPZ = os.getenv(
    "VULCADATA_TRAINING_NPZ",
    str(Path(OUTPUT_DIR) / TRAINING_OUTPUT_NAME),
)
NPZ_VALIDATION_JSON = os.getenv(
    "VULCADATA_TRAINING_NPZ_VALIDATION_JSON",
    "reports/validation/volcano_multi_npz_validation.json",
)
DATASET_METADATA_JSON = os.getenv(
    "VULCADATA_DATASET_METADATA_JSON",
    "reports/data_preparation/dataset_metadata_mlflow.json",
)

FEATURE_WINDOW_MINUTES = int(os.getenv("VULCADATA_FEATURE_WINDOW_MINUTES", "10"))
SEQ_LEN = int(os.getenv("VULCADATA_EXPECTED_SEQ_LEN", "120"))
SEQUENCE_STRIDE = int(os.getenv("VULCADATA_SEQUENCE_STRIDE", "5"))
MAX_HORIZON_HOURS = float(os.getenv("VULCADATA_MAX_HORIZON_HOURS", "48.0"))
ENTROPY_BINS = int(os.getenv("VULCADATA_ENTROPY_BINS", "20"))
N_CLASSES = int(os.getenv("VULCADATA_N_CLASSES", "6"))
SPLIT_STRATEGY = os.getenv("VULCADATA_SPLIT_STRATEGY", "chronological")
TRAIN_RATIO = float(os.getenv("VULCADATA_TRAIN_RATIO", "0.70"))
VAL_RATIO = float(os.getenv("VULCADATA_VAL_RATIO", "0.15"))

MLFLOW_ENABLED = os.getenv("VULCADATA_ENABLE_MLFLOW", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MLFLOW_REQUIRED = os.getenv("VULCADATA_MLFLOW_REQUIRED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    os.getenv("VULCADATA_MLFLOW_TRACKING_URI", "https://vartkirl-vulcadata-mlflow.hf.space/"),
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    os.getenv("VULCADATA_MLFLOW_EXPERIMENT_NAME", "Vulcadata"),
)
MLFLOW_RUN_NAME = os.getenv("VULCADATA_DATA_PREP_MLFLOW_RUN_NAME", "data_preparation")

RAW_SCHEDULE = os.getenv("VULCADATA_DATA_PREP_SCHEDULE", "manual").strip()
if RAW_SCHEDULE.lower() in {"", "none", "manual"}:
    DAG_SCHEDULE = None
else:
    DAG_SCHEDULE = RAW_SCHEDULE

CSV_SUFFIX = "_filtered_1_16Hz_aggregated_1min_with_fi.csv"


def quote(value: str | int | float) -> str:
    return shlex.quote(str(value))


def project_command(command: str) -> str:
    return (
        "set -euo pipefail; "
        f"cd {quote(PROJECT_ROOT)}; "
        f"export PYTHONPATH={quote(PROJECT_ROOT)}:${{PYTHONPATH:-}}; "
        f"{command}"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def as_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(PROJECT_ROOT) / path


def relative_to_project(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv_auto(path: Path):
    import pandas as pd

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_period_type(value: str) -> str:
    value = clean_text(value).lower()

    if value in {"eruption", "eruptive", "event"}:
        return "eruption"

    if value in {"quiet", "calm", "calme", "background", "non_eruptive"}:
        return "quiet"

    if value in {"inference", "predict", "prediction", "unknown"}:
        return "inference"

    raise ValueError(
        f"period_type invalide : {value}. Valeurs attendues : eruption, quiet ou inference."
    )


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_training_periods_file() -> None:
    periods_path = as_project_path(PERIODS_CSV)
    processed_csv_dir = as_project_path(PROCESSED_CSV_DIR)
    training_periods_path = as_project_path(TRAINING_PERIODS_CSV)
    check_report_path = as_project_path(CSV_INPUT_CHECK_JSON)

    if not periods_path.exists():
        raise FileNotFoundError(f"Fichier de périodes introuvable : {periods_path}")

    periods = read_csv_auto(periods_path)
    periods.columns = [str(column).strip() for column in periods.columns]

    required_columns = {"period_id", "period_type"}
    missing_columns = required_columns - set(periods.columns)
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le fichier de périodes : {sorted(missing_columns)}")

    for optional_column in ["eruption_start_utc", "eruption_end_utc", "split", "csv_path"]:
        if optional_column not in periods.columns:
            periods[optional_column] = ""

    selected_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    missing_csv: list[dict[str, Any]] = []

    for _, row in periods.iterrows():
        period_id = clean_text(row.get("period_id"))
        period_type = normalize_period_type(row.get("period_type"))

        if not period_id:
            raise ValueError("period_id vide dans le fichier de périodes.")

        csv_path_value = clean_text(row.get("csv_path"))
        if csv_path_value:
            csv_path = Path(csv_path_value)
            if not csv_path.is_absolute():
                csv_path = as_project_path(csv_path)
        else:
            csv_path = processed_csv_dir / f"{period_id}{CSV_SUFFIX}"

        if period_type == "inference":
            skipped_rows.append(
                {
                    "period_id": period_id,
                    "period_type": period_type,
                    "reason": "period_type_inference_excluded_from_training",
                    "csv_path": relative_to_project(csv_path),
                }
            )
            continue

        if period_type == "eruption" and not clean_text(row.get("eruption_start_utc")):
            raise ValueError(f"eruption_start_utc manquant pour la période éruptive : {period_id}")

        if not csv_path.exists():
            missing_csv.append(
                {
                    "period_id": period_id,
                    "period_type": period_type,
                    "expected_csv_path": relative_to_project(csv_path),
                }
            )
            continue

        selected_rows.append(
            {
                "period_id": period_id,
                "period_type": period_type,
                "eruption_start_utc": clean_text(row.get("eruption_start_utc")),
                "eruption_end_utc": clean_text(row.get("eruption_end_utc")),
                "split": clean_text(row.get("split")),
                "csv_path": str(csv_path),
            }
        )

    if missing_csv:
        report = {
            "status": "failed",
            "reason": "missing_aggregated_csv",
            "periods_csv": relative_to_project(periods_path),
            "processed_csv_dir": relative_to_project(processed_csv_dir),
            "selected_rows_count": len(selected_rows),
            "skipped_rows_count": len(skipped_rows),
            "missing_csv_count": len(missing_csv),
            "missing_csv": missing_csv,
            "generated_at_utc": utc_now_iso(),
        }
        write_json(report, check_report_path)
        raise FileNotFoundError(f"CSV agrégés manquants. Voir rapport : {check_report_path}")

    if not selected_rows:
        report = {
            "status": "failed",
            "reason": "no_training_period_available",
            "periods_csv": relative_to_project(periods_path),
            "processed_csv_dir": relative_to_project(processed_csv_dir),
            "selected_rows_count": 0,
            "skipped_rows_count": len(skipped_rows),
            "skipped_rows": skipped_rows,
            "generated_at_utc": utc_now_iso(),
        }
        write_json(report, check_report_path)
        raise ValueError("Aucune période training disponible après exclusion des périodes inference.")

    import pandas as pd

    training_periods_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(training_periods_path, index=False)

    report = {
        "status": "success",
        "periods_csv": relative_to_project(periods_path),
        "processed_csv_dir": relative_to_project(processed_csv_dir),
        "training_periods_csv": relative_to_project(training_periods_path),
        "selected_rows_count": len(selected_rows),
        "skipped_rows_count": len(skipped_rows),
        "missing_csv_count": 0,
        "selected_periods": [row["period_id"] for row in selected_rows],
        "skipped_rows": skipped_rows,
        "csv_suffix": CSV_SUFFIX,
        "generated_at_utc": utc_now_iso(),
    }
    write_json(report, check_report_path)


def load_validation_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": relative_to_project(path)}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json", "path": relative_to_project(path)}

    if not isinstance(payload, dict):
        return {"status": "not_a_json_object", "path": relative_to_project(path)}

    return payload


def summarize_npz(npz_path: Path) -> dict[str, Any]:
    import numpy as np

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ introuvable : {npz_path}")

    summary: dict[str, Any] = {
        "path": relative_to_project(npz_path),
        "size_bytes": int(npz_path.stat().st_size),
        "sha256": file_sha256(npz_path),
        "arrays": {},
    }

    with np.load(npz_path, allow_pickle=True) as payload:
        for key in payload.files:
            array = payload[key]
            summary["arrays"][key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }

    return summary


def log_dataset_metadata_to_mlflow() -> None:
    npz_path = as_project_path(TRAINING_NPZ)
    output_dir = as_project_path(OUTPUT_DIR)
    metadata_output_path = as_project_path(DATASET_METADATA_JSON)
    csv_input_check_path = as_project_path(CSV_INPUT_CHECK_JSON)
    npz_validation_path = as_project_path(NPZ_VALIDATION_JSON)
    preprocessing_config_path = output_dir / "preprocessing_config.json"
    feature_names_path = output_dir / "feature_names.txt"
    metadata_resolved_path = output_dir / "metadata_resolved_local_csv.csv"
    imputer_path = output_dir / "imputer.joblib"
    scaler_path = output_dir / "scaler.joblib"

    npz_summary = summarize_npz(npz_path)
    validation_report = load_validation_report(npz_validation_path)
    preprocessing_config = load_validation_report(preprocessing_config_path)
    csv_input_check = load_validation_report(csv_input_check_path)

    arrays = npz_summary.get("arrays", {})
    x_train_shape = arrays.get("X_train", {}).get("shape", [])
    x_val_shape = arrays.get("X_val", {}).get("shape", [])
    x_test_shape = arrays.get("X_test", {}).get("shape", [])

    feature_names_count = None
    if feature_names_path.exists():
        feature_names_count = len(
            [line for line in feature_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )

    metadata: dict[str, Any] = {
        "status": "success",
        "dataset_kind": "training_npz",
        "task_type": "classification",
        "npz": npz_summary,
        "periods_csv": PERIODS_CSV,
        "training_periods_csv": TRAINING_PERIODS_CSV,
        "processed_csv_dir": PROCESSED_CSV_DIR,
        "output_dir": OUTPUT_DIR,
        "feature_window_minutes": FEATURE_WINDOW_MINUTES,
        "seq_len": SEQ_LEN,
        "sequence_stride": SEQUENCE_STRIDE,
        "max_horizon_hours": MAX_HORIZON_HOURS,
        "entropy_bins": ENTROPY_BINS,
        "n_classes": N_CLASSES,
        "split_strategy": SPLIT_STRATEGY,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
        "feature_names_count": feature_names_count,
        "x_train_shape": x_train_shape,
        "x_val_shape": x_val_shape,
        "x_test_shape": x_test_shape,
        "artifacts": {
            "feature_names": relative_to_project(feature_names_path) if feature_names_path.exists() else None,
            "imputer": relative_to_project(imputer_path) if imputer_path.exists() else None,
            "scaler": relative_to_project(scaler_path) if scaler_path.exists() else None,
            "metadata_resolved": relative_to_project(metadata_resolved_path) if metadata_resolved_path.exists() else None,
            "preprocessing_config": relative_to_project(preprocessing_config_path) if preprocessing_config_path.exists() else None,
            "npz_validation": relative_to_project(npz_validation_path) if npz_validation_path.exists() else None,
            "csv_input_check": relative_to_project(csv_input_check_path) if csv_input_check_path.exists() else None,
        },
        "validation_report": validation_report,
        "preprocessing_config": preprocessing_config,
        "csv_input_check": csv_input_check,
        "generated_at_utc": utc_now_iso(),
    }

    mlflow_run_id = None
    mlflow_error = None

    if MLFLOW_ENABLED:
        try:
            import mlflow

            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            run_name = f"{MLFLOW_RUN_NAME}_{compact_utc_timestamp()}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow_run_id = run.info.run_id

                mlflow.set_tag("pipeline", DAG_ID)
                mlflow.set_tag("stage", "data_preparation")
                mlflow.set_tag("dataset_kind", "training_npz")
                mlflow.set_tag("storage_policy", "heavy_data_local_only")

                mlflow.log_param("periods_csv", PERIODS_CSV)
                mlflow.log_param("training_periods_csv", TRAINING_PERIODS_CSV)
                mlflow.log_param("processed_csv_dir", PROCESSED_CSV_DIR)
                mlflow.log_param("training_npz", TRAINING_NPZ)
                mlflow.log_param("npz_sha256", npz_summary["sha256"])
                mlflow.log_param("feature_window_minutes", FEATURE_WINDOW_MINUTES)
                mlflow.log_param("seq_len", SEQ_LEN)
                mlflow.log_param("sequence_stride", SEQUENCE_STRIDE)
                mlflow.log_param("max_horizon_hours", MAX_HORIZON_HOURS)
                mlflow.log_param("entropy_bins", ENTROPY_BINS)
                mlflow.log_param("n_classes", N_CLASSES)
                mlflow.log_param("split_strategy", SPLIT_STRATEGY)

                mlflow.log_metric("training_npz_size_bytes", npz_summary["size_bytes"])
                if len(x_train_shape) == 3:
                    mlflow.log_metric("n_train_sequences", int(x_train_shape[0]))
                    mlflow.log_metric("seq_len", int(x_train_shape[1]))
                    mlflow.log_metric("n_features", int(x_train_shape[2]))
                if len(x_val_shape) == 3:
                    mlflow.log_metric("n_val_sequences", int(x_val_shape[0]))
                if len(x_test_shape) == 3:
                    mlflow.log_metric("n_test_sequences", int(x_test_shape[0]))
                if feature_names_count is not None:
                    mlflow.log_metric("feature_names_count", int(feature_names_count))

                for artifact_path in [
                    csv_input_check_path,
                    npz_validation_path,
                    preprocessing_config_path,
                    feature_names_path,
                    metadata_resolved_path,
                ]:
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path), artifact_path="data_preparation")

        except Exception as exc:
            mlflow_error = repr(exc)
            if MLFLOW_REQUIRED:
                raise

    metadata["mlflow"] = {
        "enabled": MLFLOW_ENABLED,
        "required": MLFLOW_REQUIRED,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "run_id": mlflow_run_id,
        "error": mlflow_error,
    }

    write_json(metadata, metadata_output_path)


preprocess_training_dataset_command = project_command(
    "python -m src.preprocessing.preprocess_volcano_dataset "
    "--mode training "
    f"--periods {quote(TRAINING_PERIODS_CSV)} "
    f"--processed-csv-dir {quote(PROCESSED_CSV_DIR)} "
    f"--output-dir {quote(OUTPUT_DIR)} "
    f"--training-output-name {quote(TRAINING_OUTPUT_NAME)} "
    f"--feature-window-minutes {quote(FEATURE_WINDOW_MINUTES)} "
    f"--seq-len {quote(SEQ_LEN)} "
    f"--sequence-stride {quote(SEQUENCE_STRIDE)} "
    f"--max-horizon-hours {quote(MAX_HORIZON_HOURS)} "
    f"--entropy-bins {quote(ENTROPY_BINS)} "
    f"--n-classes {quote(N_CLASSES)} "
    f"--split-strategy {quote(SPLIT_STRATEGY)} "
    f"--train-ratio {quote(TRAIN_RATIO)} "
    f"--val-ratio {quote(VAL_RATIO)}"
)

validate_training_npz_command = project_command(
    "python scripts/preprocessing/validate_npz_dataset.py "
    f"--input-npz {quote(TRAINING_NPZ)} "
    "--task classification "
    f"--n-classes {quote(N_CLASSES)} "
    f"--max-horizon-hours {quote(MAX_HORIZON_HOURS)} "
    f"--output-json {quote(NPZ_VALIDATION_JSON)}"
)

with DAG(
    dag_id=DAG_ID,
    description="Prepare the local Vulcadata training dataset from existing aggregated CSV files.",
    start_date=datetime(2026, 1, 1),
    schedule=DAG_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["vulcadata", "volcano", "data-preparation", "preprocessing", "validation", "mlflow"],
) as dag:
    check_processed_csv_inputs = PythonOperator(
        task_id="check_processed_csv_inputs",
        python_callable=build_training_periods_file,
    )

    preprocess_training_dataset = BashOperator(
        task_id="preprocess_training_dataset",
        bash_command=preprocess_training_dataset_command,
    )

    validate_training_npz = BashOperator(
        task_id="validate_training_npz",
        bash_command=validate_training_npz_command,
    )

    log_dataset_metadata = PythonOperator(
        task_id="log_dataset_metadata_to_mlflow",
        python_callable=log_dataset_metadata_to_mlflow,
    )

    check_processed_csv_inputs >> preprocess_training_dataset >> validate_training_npz >> log_dataset_metadata
