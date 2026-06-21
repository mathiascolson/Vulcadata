from __future__ import annotations

import os
import shlex
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


DAG_ID = "volcano_inference_pipeline"

PROJECT_ROOT = os.getenv("VULCADATA_PROJECT_ROOT", "/opt/vulcadata")

SOURCE_NPZ = os.getenv(
    "VULCADATA_SOURCE_NPZ",
    "data/preprocessing/processed/inference_source.npz",
)
SOURCE_ARRAY_KEY = os.getenv("VULCADATA_SOURCE_ARRAY_KEY", "X")

LATEST_BATCH_NPZ = os.getenv(
    "VULCADATA_LATEST_BATCH_NPZ",
    "data/preprocessing/processed/latest_batch.npz",
)

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
EVIDENTLY_CURRENT_NPZ = os.getenv(
    "VULCADATA_EVIDENTLY_CURRENT_NPZ",
    LATEST_BATCH_NPZ,
)
EVIDENTLY_CURRENT_ARRAY_KEY = os.getenv(
    "VULCADATA_EVIDENTLY_CURRENT_ARRAY_KEY",
    "X",
)
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

EXPECTED_SEQ_LEN = int(os.getenv("VULCADATA_EXPECTED_SEQ_LEN", "120"))
EXPECTED_N_FEATURES = int(os.getenv("VULCADATA_EXPECTED_N_FEATURES", "992"))
LATEST_BATCH_SIZE = int(os.getenv("VULCADATA_LATEST_BATCH_SIZE", "1"))

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
    "--output-is-logits "
    "--write-s3 "
    f"--s3-bucket {quote(S3_BUCKET)}"
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
    f"--max-features {quote(EVIDENTLY_MAX_FEATURES)} "
    "--write-s3 "
    f"--s3-bucket {quote(S3_BUCKET)} "
    f"--s3-prefix {quote(EVIDENTLY_S3_PREFIX)}"
)

verify_dashboard_outputs_command = project_command(
    "python -m src.inference.verify_dashboard_outputs "
    f"--s3-bucket {quote(S3_BUCKET)} "
    "--latest-prediction-key predictions/latest/prediction.json"
)

with DAG(
    dag_id=DAG_ID,
    description="Prepare the latest volcano batch, validate it, run inference and generate monitoring outputs.",
    start_date=datetime(2026, 1, 1),
    schedule=DAG_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["vulcadata", "volcano", "inference", "mlflow", "s3", "great-expectations", "evidently"],
) as dag:
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

    generate_evidently_report = BashOperator(
        task_id="generate_evidently_report",
        bash_command=generate_evidently_report_command,
    )

    verify_dashboard_outputs = BashOperator(
        task_id="verify_dashboard_outputs",
        bash_command=verify_dashboard_outputs_command,
    )

    (
        prepare_latest_batch
        >> validate_latest_batch_with_gx
        >> run_inference
        >> generate_evidently_report
        >> verify_dashboard_outputs
    )
