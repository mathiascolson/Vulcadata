from __future__ import annotations

import json
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


DAG_ID = "volcano_inference_pipeline"

PROJECT_ROOT = os.getenv("VULCADATA_PROJECT_ROOT", "/opt/vulcadata")

INFERENCE_PERIODS_CSV = os.getenv(
    "VULCADATA_INFERENCE_PERIODS_CSV",
    "data/metadata/extraction_periods.csv",
)
PROCESSED_CSV_DIR = os.getenv(
    "VULCADATA_PROCESSED_CSV_DIR",
    "data/extraction/processed_csv",
)
REFERENCE_ARTIFACTS_DIR = os.getenv(
    "VULCADATA_REFERENCE_ARTIFACTS_DIR",
    "data/preprocessing/processed",
)
INFERENCE_OUTPUT_DIR = os.getenv(
    "VULCADATA_INFERENCE_PREPROCESSING_OUTPUT_DIR",
    "data/preprocessing/inference",
)
INFERENCE_SOURCE_NPZ_NAME = os.getenv(
    "VULCADATA_INFERENCE_SOURCE_NPZ_NAME",
    "inference_source.npz",
)

SOURCE_NPZ = str(Path(INFERENCE_OUTPUT_DIR) / INFERENCE_SOURCE_NPZ_NAME)
SOURCE_ARRAY_KEY = "X"

LATEST_BATCH_NPZ = str(Path(INFERENCE_OUTPUT_DIR) / "latest_batch.npz")

INFERENCE_CONFIG = os.getenv(
    "VULCADATA_INFERENCE_CONFIG",
    "configs/inference_config.yaml",
)
MODEL_DECISION = os.getenv(
    "VULCADATA_MODEL_DECISION",
    "configs/final_model_decision.json",
)
OUTPUT_JSON = os.getenv(
    "VULCADATA_INFERENCE_OUTPUT_JSON",
    "reports/inference/latest_result.json",
)

GX_VALIDATION_OUTPUT_JSON = os.getenv(
    "VULCADATA_GX_VALIDATION_OUTPUT_JSON",
    "reports/validation/latest_batch_gx_validation.json",
)

EVIDENTLY_REFERENCE_NPZ = os.getenv(
    "VULCADATA_EVIDENTLY_REFERENCE_NPZ",
    "data/preprocessing/processed/volcano_multi.npz",
)
EVIDENTLY_REFERENCE_ARRAY_KEY = os.getenv(
    "VULCADATA_EVIDENTLY_REFERENCE_ARRAY_KEY",
    "X_train",
)
EVIDENTLY_CURRENT_NPZ = LATEST_BATCH_NPZ
EVIDENTLY_CURRENT_ARRAY_KEY = "X"
EVIDENTLY_OUTPUT_DIR = os.getenv(
    "VULCADATA_EVIDENTLY_OUTPUT_DIR",
    "reports/monitoring/evidently",
)
EVIDENTLY_REPORT_NAME = os.getenv(
    "VULCADATA_EVIDENTLY_REPORT_NAME",
    "latest_data_drift",
)
EVIDENTLY_S3_PREFIX = os.getenv(
    "VULCADATA_EVIDENTLY_S3_PREFIX",
    "monitoring/evidently",
)
EVIDENTLY_MAX_ROWS = int(os.getenv("VULCADATA_EVIDENTLY_MAX_ROWS", "5000"))
EVIDENTLY_MAX_FEATURES = int(os.getenv("VULCADATA_EVIDENTLY_MAX_FEATURES", "50"))

S3_BUCKET = os.getenv("VULCADATA_S3_BUCKET", "vulcadata")
S3_LATEST_PREDICTION_KEY = os.getenv(
    "VULCADATA_S3_LATEST_PREDICTION_KEY",
    "predictions/latest/prediction.json",
)

EXPECTED_SEQ_LEN = int(os.getenv("VULCADATA_EXPECTED_SEQ_LEN", "120"))
EXPECTED_N_FEATURES = int(os.getenv("VULCADATA_EXPECTED_N_FEATURES", "992"))
LATEST_BATCH_SIZE = int(os.getenv("VULCADATA_LATEST_BATCH_SIZE", "1"))

FEATURE_WINDOW_MINUTES = int(os.getenv("VULCADATA_FEATURE_WINDOW_MINUTES", "10"))
SEQUENCE_STRIDE = int(os.getenv("VULCADATA_SEQUENCE_STRIDE", "5"))
ENTROPY_BINS = int(os.getenv("VULCADATA_ENTROPY_BINS", "20"))
N_CLASSES = int(os.getenv("VULCADATA_N_CLASSES", "6"))

MLFLOW_EXPERIMENT_NAME = os.getenv("VULCADATA_MLFLOW_EXPERIMENT_NAME", "Vulcadata")
MLFLOW_REPORT_JSON = os.getenv(
    "VULCADATA_INFERENCE_MLFLOW_REPORT_JSON",
    "reports/inference/inference_mlflow_log.json",
)

WRITE_S3_OUTPUTS = os.getenv("VULCADATA_WRITE_S3_OUTPUTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
WRITE_EVIDENTLY_S3 = os.getenv("VULCADATA_WRITE_EVIDENTLY_S3", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
VERIFY_S3_DASHBOARD = os.getenv("VULCADATA_VERIFY_S3_DASHBOARD", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

RAW_SCHEDULE = os.getenv("VULCADATA_INFERENCE_SCHEDULE", "manual").strip()

if RAW_SCHEDULE.lower() in {"", "none", "manual"}:
    DAG_SCHEDULE = None
else:
    DAG_SCHEDULE = RAW_SCHEDULE


def quote(value: str | int | float) -> str:
    return shlex.quote(str(value))


def project_command(command: str) -> str:
    return (
        "set -euo pipefail; "
        f"cd {quote(PROJECT_ROOT)}; "
        f"export PYTHONPATH={quote(PROJECT_ROOT)}:${{PYTHONPATH:-}}; "
        f"{command}"
    )


def as_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(PROJECT_ROOT) / path


def read_json_file(path: str | Path) -> dict[str, Any]:
    resolved_path = as_project_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {resolved_path}")
    return payload


def write_json_file(payload: dict[str, Any], path: str | Path) -> None:
    resolved_path = as_project_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def find_first_value(payload: Any, accepted_keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in accepted_keys:
                return value
        for value in payload.values():
            nested = find_first_value(value, accepted_keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = find_first_value(item, accepted_keys)
            if nested is not None:
                return nested
    return None


def to_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def verify_local_inference_output() -> None:
    payload = read_json_file(OUTPUT_JSON)
    prediction = payload.get("prediction") if isinstance(payload.get("prediction"), dict) else payload
    predicted_class = find_first_value(prediction, {"predicted_class"})
    p_alert_24h = find_first_value(prediction, {"p_alert_24h"})

    if predicted_class is None:
        raise ValueError(f"predicted_class not found in inference output: {OUTPUT_JSON}")

    if p_alert_24h is None:
        raise ValueError(f"p_alert_24h not found in inference output: {OUTPUT_JSON}")


def log_inference_metadata_to_mlflow() -> None:
    report: dict[str, Any] = {
        "status": "started",
        "dag_id": DAG_ID,
        "project_root": PROJECT_ROOT,
        "inference_periods_csv": INFERENCE_PERIODS_CSV,
        "processed_csv_dir": PROCESSED_CSV_DIR,
        "reference_artifacts_dir": REFERENCE_ARTIFACTS_DIR,
        "inference_output_dir": INFERENCE_OUTPUT_DIR,
        "source_npz": SOURCE_NPZ,
        "latest_batch_npz": LATEST_BATCH_NPZ,
        "output_json": OUTPUT_JSON,
        "gx_validation_output_json": GX_VALIDATION_OUTPUT_JSON,
        "evidently_output_dir": EVIDENTLY_OUTPUT_DIR,
        "write_s3_outputs": WRITE_S3_OUTPUTS,
        "write_evidently_s3": WRITE_EVIDENTLY_S3,
        "s3_bucket": S3_BUCKET if WRITE_S3_OUTPUTS else None,
        "s3_latest_prediction_key": S3_LATEST_PREDICTION_KEY if WRITE_S3_OUTPUTS else None,
        "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
    }

    try:
        import mlflow
    except Exception as exc:
        report.update(
            {
                "status": "skipped",
                "reason": "mlflow_import_failed",
                "error": repr(exc),
            }
        )
        write_json_file(report, MLFLOW_REPORT_JSON)
        return

    inference_payload = read_json_file(OUTPUT_JSON)
    gx_payload = read_json_file(GX_VALIDATION_OUTPUT_JSON)

    prediction_payload = (
        inference_payload.get("prediction")
        if isinstance(inference_payload.get("prediction"), dict)
        else inference_payload
    )

    predicted_class = to_int_or_none(
        find_first_value(prediction_payload, {"predicted_class"})
    )
    predicted_probability = to_float_or_none(
        find_first_value(prediction_payload, {"predicted_probability"})
    )
    p_alert_24h = to_float_or_none(
        find_first_value(prediction_payload, {"p_alert_24h"})
    )
    alert_24h_value = find_first_value(prediction_payload, {"alert_24h"})

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("VULCADATA_MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="airflow_inference_pipeline") as run:
        mlflow.log_param("dag_id", DAG_ID)
        mlflow.log_param("source_npz", SOURCE_NPZ)
        mlflow.log_param("latest_batch_npz", LATEST_BATCH_NPZ)
        mlflow.log_param("inference_periods_csv", INFERENCE_PERIODS_CSV)
        mlflow.log_param("reference_artifacts_dir", REFERENCE_ARTIFACTS_DIR)
        mlflow.log_param("expected_seq_len", EXPECTED_SEQ_LEN)
        mlflow.log_param("expected_n_features", EXPECTED_N_FEATURES)
        mlflow.log_param("latest_batch_size", LATEST_BATCH_SIZE)
        mlflow.log_param("write_s3_outputs", WRITE_S3_OUTPUTS)
        mlflow.log_param("write_evidently_s3", WRITE_EVIDENTLY_S3)
        mlflow.log_param("s3_bucket", S3_BUCKET if WRITE_S3_OUTPUTS else "")
        mlflow.log_param("s3_latest_prediction_key", S3_LATEST_PREDICTION_KEY if WRITE_S3_OUTPUTS else "")

        if predicted_class is not None:
            mlflow.log_metric("predicted_class", predicted_class)
        if predicted_probability is not None:
            mlflow.log_metric("predicted_probability", predicted_probability)
        if p_alert_24h is not None:
            mlflow.log_metric("p_alert_24h", p_alert_24h)
        if isinstance(alert_24h_value, bool):
            mlflow.log_metric("alert_24h", int(alert_24h_value))

        gx_success = gx_payload.get("gx_success")
        if isinstance(gx_success, bool):
            mlflow.log_metric("latest_batch_gx_success", int(gx_success))

        for artifact_path in [
            OUTPUT_JSON,
            GX_VALIDATION_OUTPUT_JSON,
            str(Path(EVIDENTLY_OUTPUT_DIR) / f"{EVIDENTLY_REPORT_NAME}.json"),
            str(Path(EVIDENTLY_OUTPUT_DIR) / f"{EVIDENTLY_REPORT_NAME}_summary.json"),
        ]:
            resolved_artifact_path = as_project_path(artifact_path)
            if resolved_artifact_path.exists():
                mlflow.log_artifact(str(resolved_artifact_path))

        report.update(
            {
                "status": "success",
                "mlflow_run_id": run.info.run_id,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_probability,
                "p_alert_24h": p_alert_24h,
                "alert_24h": alert_24h_value,
                "gx_success": gx_success,
            }
        )

    write_json_file(report, MLFLOW_REPORT_JSON)


preprocess_inference_dataset_command = project_command(
    "python -m src.preprocessing.preprocess_volcano_dataset "
    "--mode inference "
    f"--periods {quote(INFERENCE_PERIODS_CSV)} "
    f"--processed-csv-dir {quote(PROCESSED_CSV_DIR)} "
    f"--output-dir {quote(INFERENCE_OUTPUT_DIR)} "
    f"--reference-artifacts-dir {quote(REFERENCE_ARTIFACTS_DIR)} "
    f"--inference-output-name {quote(INFERENCE_SOURCE_NPZ_NAME)} "
    f"--feature-window-minutes {quote(FEATURE_WINDOW_MINUTES)} "
    f"--seq-len {quote(EXPECTED_SEQ_LEN)} "
    f"--sequence-stride {quote(SEQUENCE_STRIDE)} "
    f"--entropy-bins {quote(ENTROPY_BINS)} "
    f"--n-classes {quote(N_CLASSES)}"
)

prepare_latest_batch_command = project_command(
    "python -m src.inference.prepare_latest_batch "
    f"--source-npz {quote(SOURCE_NPZ)} "
    f"--source-array-key {quote(SOURCE_ARRAY_KEY)} "
    f"--output-npz {quote(LATEST_BATCH_NPZ)} "
    f"--batch-size {quote(LATEST_BATCH_SIZE)} "
    f"--expected-seq-len {quote(EXPECTED_SEQ_LEN)} "
    f"--expected-n-features {quote(EXPECTED_N_FEATURES)} "
    "--overwrite"
)

validate_latest_batch_with_gx_command = project_command(
    "python -m src.inference.validate_latest_batch_with_gx "
    f"--npz-path {quote(LATEST_BATCH_NPZ)} "
    "--array-key X "
    f"--expected-batch-size {quote(LATEST_BATCH_SIZE)} "
    f"--expected-seq-len {quote(EXPECTED_SEQ_LEN)} "
    f"--expected-n-features {quote(EXPECTED_N_FEATURES)} "
    f"--output-json {quote(GX_VALIDATION_OUTPUT_JSON)}"
)

run_inference_command = project_command(
    "python -m src.inference.run_inference "
    f"--npz-path {quote(LATEST_BATCH_NPZ)} "
    "--array-key X "
    f"--inference-config {quote(INFERENCE_CONFIG)} "
    f"--model-decision {quote(MODEL_DECISION)} "
    f"--output-json {quote(OUTPUT_JSON)} "
    "--output-is-logits"
    + (f" --write-s3 --s3-bucket {quote(S3_BUCKET)}" if WRITE_S3_OUTPUTS else "")
)

generate_evidently_report_command = project_command(
    "python -m src.monitoring.generate_evidently_report "
    f"--reference-npz {quote(EVIDENTLY_REFERENCE_NPZ)} "
    f"--current-npz {quote(EVIDENTLY_CURRENT_NPZ)} "
    f"--reference-array-key {quote(EVIDENTLY_REFERENCE_ARRAY_KEY)} "
    f"--current-array-key {quote(EVIDENTLY_CURRENT_ARRAY_KEY)} "
    f"--output-dir {quote(EVIDENTLY_OUTPUT_DIR)} "
    f"--report-name {quote(EVIDENTLY_REPORT_NAME)} "
    f"--max-rows {quote(EVIDENTLY_MAX_ROWS)} "
    f"--max-features {quote(EVIDENTLY_MAX_FEATURES)}"
    + (
        f" --write-s3 --s3-bucket {quote(S3_BUCKET)} --s3-prefix {quote(EVIDENTLY_S3_PREFIX)}"
        if WRITE_EVIDENTLY_S3
        else ""
    )
)

verify_dashboard_outputs_command = project_command(
    "python -m src.inference.verify_dashboard_outputs "
    f"--s3-bucket {quote(S3_BUCKET)} "
    f"--latest-prediction-key {quote(S3_LATEST_PREDICTION_KEY)}"
)

with DAG(
    dag_id=DAG_ID,
    description=(
        "Preprocess local aggregated CSV files into an inference batch, validate it, "
        "run volcano alert inference, publish lightweight dashboard outputs and log metadata."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=DAG_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=[
        "vulcadata",
        "volcano",
        "inference",
        "preprocessing",
        "mlflow",
        "s3-lightweight-outputs",
        "great-expectations",
        "evidently",
    ],
) as dag:
    preprocess_inference_dataset = BashOperator(
        task_id="preprocess_inference_dataset",
        bash_command=preprocess_inference_dataset_command,
    )

    prepare_latest_batch = BashOperator(
        task_id="prepare_latest_batch",
        bash_command=prepare_latest_batch_command,
    )

    validate_latest_batch_with_gx = BashOperator(
        task_id="validate_latest_batch_with_gx",
        bash_command=validate_latest_batch_with_gx_command,
    )

    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=run_inference_command,
    )

    verify_local_inference_output_task = PythonOperator(
        task_id="verify_local_inference_output",
        python_callable=verify_local_inference_output,
    )

    generate_evidently_report = BashOperator(
        task_id="generate_evidently_report",
        bash_command=generate_evidently_report_command,
    )

    log_inference_metadata_to_mlflow_task = PythonOperator(
        task_id="log_inference_metadata_to_mlflow",
        python_callable=log_inference_metadata_to_mlflow,
    )

    (
        preprocess_inference_dataset
        >> prepare_latest_batch
        >> validate_latest_batch_with_gx
        >> run_inference
        >> verify_local_inference_output_task
    )

    if WRITE_S3_OUTPUTS and VERIFY_S3_DASHBOARD:
        verify_dashboard_outputs = BashOperator(
            task_id="verify_dashboard_outputs",
            bash_command=verify_dashboard_outputs_command,
        )

        (
            verify_local_inference_output_task
            >> verify_dashboard_outputs
            >> generate_evidently_report
            >> log_inference_metadata_to_mlflow_task
        )
    else:
        (
            verify_local_inference_output_task
            >> generate_evidently_report
            >> log_inference_metadata_to_mlflow_task
        )