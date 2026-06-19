from __future__ import annotations

import json
import os
import shlex
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule


DAG_ID = "volcano_retraining_pipeline"

PROJECT_ROOT = os.getenv("VULCADATA_PROJECT_ROOT", "/opt/vulcadata")

READY_DIR = os.getenv("VULCADATA_RETRAINING_READY_DIR", "data/retraining/ready")
MERGED_DIR = os.getenv("VULCADATA_RETRAINING_MERGED_DIR", "data/retraining/merged")
ARCHIVE_DIR = os.getenv("VULCADATA_RETRAINING_ARCHIVE_DIR", "data/retraining/archive")

DETECTION_REPORT_PATH = os.getenv(
    "VULCADATA_NEW_FILES_DETECTION_OUTPUT_JSON",
    "reports/retraining/new_preprocessed_files_detection.json",
)
TRAINING_RESULT_PATH = os.getenv(
    "VULCADATA_CANDIDATE_TRAINING_RESULT_JSON",
    "reports/retraining/candidate_training_result.json",
)
DRIFT_OUTPUT_DIR = os.getenv(
    "VULCADATA_RETRAINING_DRIFT_OUTPUT_DIR",
    "reports/retraining/evidently",
)
DRIFT_SUMMARY_PATH = os.getenv(
    "VULCADATA_RETRAINING_DRIFT_SUMMARY_JSON",
    "reports/retraining/evidently/candidate_drift_summary.json",
)
COMPARISON_REPORT_PATH = os.getenv(
    "VULCADATA_CANDIDATE_COMPARISON_JSON",
    "reports/retraining/candidate_vs_champion_comparison.json",
)
ARCHIVE_REPORT_PATH = os.getenv(
    "VULCADATA_RETRAINING_ARCHIVE_REPORT_JSON",
    "reports/retraining/archive_processed_ready_files.json",
)
PROMOTION_REPORT_PATH = os.getenv(
    "VULCADATA_CANDIDATE_PROMOTION_RESULT_JSON",
    "reports/retraining/candidate_promotion_result.json",
)

CHAMPION_DECISION_PATH = os.getenv(
    "VULCADATA_FINAL_MODEL_DECISION_JSON",
    "configs/final_model_decision.json",
)
REFERENCE_NPZ_PATH = os.getenv(
    "VULCADATA_RETRAINING_REFERENCE_NPZ",
    "data/preprocessing/processed_full_stride5_with_quiet/volcano_multi.npz",
)

TRAINING_SCRIPT = os.getenv(
    "VULCADATA_CANDIDATE_TRAINING_SCRIPT",
    "scripts/training/train_cnn_transformer_classif_modified_Vfinetuning.py",
)
CANDIDATE_OUTPUT_DIR = os.getenv(
    "VULCADATA_CANDIDATE_OUTPUT_DIR",
    "models/retraining/cnn_transformer_candidate",
)

MIN_NEW_FILES_FOR_RETRAINING = int(os.getenv("VULCADATA_MIN_NEW_FILES_FOR_RETRAINING", "1"))
EXPECTED_SEQ_LEN = int(os.getenv("VULCADATA_EXPECTED_SEQ_LEN", "120"))
EXPECTED_N_FEATURES = int(os.getenv("VULCADATA_EXPECTED_N_FEATURES", "992"))

TRAINING_EPOCHS = int(os.getenv("VULCADATA_RETRAINING_EPOCHS", "1"))
TRAINING_BATCH_SIZE = int(os.getenv("VULCADATA_RETRAINING_BATCH_SIZE", "16"))
TRAINING_CLASS_WEIGHTING = os.getenv("VULCADATA_RETRAINING_CLASS_WEIGHTING", "alert_priority")
TRAINING_DEVICE_FLAG = os.getenv("VULCADATA_RETRAINING_DEVICE_FLAG", "--cpu")

MIN_EPOCHS_FOR_PROMOTION = int(os.getenv("VULCADATA_MIN_EPOCHS_FOR_PROMOTION", "2"))
MAX_BUSINESS_SCORE_DROP = float(os.getenv("VULCADATA_MAX_BUSINESS_SCORE_DROP", "0.0"))
MAX_ALERT_24H_F1_DROP = float(os.getenv("VULCADATA_MAX_ALERT_24H_F1_DROP", "0.02"))
MIN_ALERT_24H_RECALL = float(os.getenv("VULCADATA_MIN_ALERT_24H_RECALL", "0.70"))
MIN_ALERT_24H_PRECISION = float(os.getenv("VULCADATA_MIN_ALERT_24H_PRECISION", "0.40"))
MAX_CLASS_5_F1_DROP = float(os.getenv("VULCADATA_MAX_CLASS_5_F1_DROP", "0.05"))

LOCAL_CHAMPION_CHECKPOINT_PATH = os.getenv(
    "VULCADATA_LOCAL_CHAMPION_CHECKPOINT",
    "models/champion_classification_checkpoint/best_cnn_transformer_classifier.pt",
)
LOCAL_CHAMPION_ARCHIVE_DIR = os.getenv(
    "VULCADATA_LOCAL_CHAMPION_ARCHIVE_DIR",
    "models/champion_classification_checkpoint/archive",
)
DECISION_ARCHIVE_DIR = os.getenv(
    "VULCADATA_DECISION_ARCHIVE_DIR",
    "configs/model_decision_archive",
)
S3_BUCKET = os.getenv("VULCADATA_S3_BUCKET", "vulcadata")
S3_CHAMPION_KEY = os.getenv(
    "VULCADATA_S3_CHAMPION_CHECKPOINT_KEY",
    "models/champion_classification_checkpoint/best_cnn_transformer_classifier.pt",
)
S3_CHAMPION_ARCHIVE_PREFIX = os.getenv(
    "VULCADATA_S3_CHAMPION_ARCHIVE_PREFIX",
    "models/champion_classification_checkpoint/archive",
)
S3_DECISION_KEY = os.getenv(
    "VULCADATA_S3_DECISION_KEY",
    "model_decisions/final_model_decision.json",
)
PROMOTION_EXTRA_FLAGS = os.getenv("VULCADATA_PROMOTION_EXTRA_FLAGS", "")


def project_bash(command: str) -> str:
    return f"""
set -euo pipefail
cd {shlex.quote(PROJECT_ROOT)}
export PYTHONPATH={shlex.quote(PROJECT_ROOT)}:${{PYTHONPATH:-}}
{command}
"""


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def npz_file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative_to_project(path),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def load_and_validate_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    required_keys = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    arrays: dict[str, Any] = {}

    with np.load(path, allow_pickle=True) as payload:
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise ValueError(f"Missing keys in {path}: {missing_keys}")

        for key in required_keys:
            arrays[key] = payload[key]

        for split_name in ("train", "val", "test"):
            x = arrays[f"X_{split_name}"]
            y = arrays[f"y_{split_name}"]

            if x.ndim != 3:
                raise ValueError(f"Expected 3D array for X_{split_name} in {path}. Got {x.shape}.")

            if y.ndim != 1:
                raise ValueError(f"Expected 1D array for y_{split_name} in {path}. Got {y.shape}.")

            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Mismatched rows in {path} for split {split_name}: "
                    f"{x.shape[0]} sequences vs {y.shape[0]} labels."
                )

            if x.shape[1] != EXPECTED_SEQ_LEN or x.shape[2] != EXPECTED_N_FEATURES:
                raise ValueError(
                    f"Invalid shape for X_{split_name} in {path}: {x.shape}. "
                    f"Expected (*, {EXPECTED_SEQ_LEN}, {EXPECTED_N_FEATURES})."
                )

        extra_keys = [
            key
            for key in payload.keys()
            if key not in required_keys and key not in arrays
        ]
        for key in extra_keys:
            arrays[key] = payload[key]

    return arrays


def detect_and_merge_ready_npz() -> None:
    import numpy as np

    ready_dir = as_project_path(READY_DIR)
    merged_root = as_project_path(MERGED_DIR)
    detection_report_path = as_project_path(DETECTION_REPORT_PATH)

    ready_dir.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)
    detection_report_path.parent.mkdir(parents=True, exist_ok=True)

    source_files = sorted(ready_dir.glob("*.npz"))
    source_file_infos = [npz_file_info(path) for path in source_files]

    if len(source_files) < MIN_NEW_FILES_FOR_RETRAINING:
        report = {
            "status": "success",
            "should_retrain": False,
            "ready_dir": READY_DIR,
            "merged_dir": MERGED_DIR,
            "candidate_files_count": len(source_files),
            "min_new_files_for_retraining": MIN_NEW_FILES_FOR_RETRAINING,
            "selection_policy": "merge_all_ready_npz_on_sequence_axis",
            "concat_axis": 0,
            "expected_seq_len": EXPECTED_SEQ_LEN,
            "expected_n_features": EXPECTED_N_FEATURES,
            "source_files": source_file_infos,
            "files_to_process": [],
            "merged_npz_path": None,
            "generated_at_utc": utc_now_iso(),
        }
        detection_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    loaded_files = [load_and_validate_npz(path) for path in source_files]
    merged_payload: dict[str, Any] = {}

    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        merged_payload[key] = np.concatenate([payload[key] for payload in loaded_files], axis=0)

    reference_payload = loaded_files[0]
    for key, value in reference_payload.items():
        if key not in merged_payload:
            merged_payload[key] = value

    timestamp = utc_now_compact()
    merged_dir = merged_root / timestamp
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_npz_path = merged_dir / "merged_ready_batch.npz"
    np.savez_compressed(merged_npz_path, **merged_payload)

    merged_info = npz_file_info(merged_npz_path)
    merged_info["source_files_count"] = len(source_files)
    merged_info["source_files"] = source_file_infos

    report = {
        "status": "success",
        "should_retrain": True,
        "ready_dir": READY_DIR,
        "merged_dir": MERGED_DIR,
        "candidate_files_count": len(source_files),
        "min_new_files_for_retraining": MIN_NEW_FILES_FOR_RETRAINING,
        "selection_policy": "merge_all_ready_npz_on_sequence_axis",
        "concat_axis": 0,
        "expected_seq_len": EXPECTED_SEQ_LEN,
        "expected_n_features": EXPECTED_N_FEATURES,
        "source_files": source_file_infos,
        "files_to_process": [merged_info],
        "merged_npz_path": relative_to_project(merged_npz_path),
        "generated_at_utc": utc_now_iso(),
    }
    detection_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_retraining_path() -> str:
    report_path = as_project_path(DETECTION_REPORT_PATH)

    if not report_path.exists():
        raise FileNotFoundError(f"Detection report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if report.get("status") != "success":
        raise RuntimeError(f"Detection report status must be success. Got: {report.get('status')}")

    if report.get("should_retrain") is True:
        return "train_candidate_model"

    return "skip_retraining"


def archive_processed_source_files() -> None:
    detection_report_path = as_project_path(DETECTION_REPORT_PATH)
    training_result_path = as_project_path(TRAINING_RESULT_PATH)
    comparison_report_path = as_project_path(COMPARISON_REPORT_PATH)
    promotion_report_path = as_project_path(PROMOTION_REPORT_PATH)
    archive_report_path = as_project_path(ARCHIVE_REPORT_PATH)

    for required_path in [detection_report_path, training_result_path, comparison_report_path, promotion_report_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required report not found: {required_path}")

    detection_report = json.loads(detection_report_path.read_text(encoding="utf-8"))
    training_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    comparison_report = json.loads(comparison_report_path.read_text(encoding="utf-8"))
    promotion_report = json.loads(promotion_report_path.read_text(encoding="utf-8"))

    if training_result.get("status") != "success":
        raise RuntimeError("Training result must be success before archiving ready files.")

    if comparison_report.get("status") != "success":
        raise RuntimeError("Comparison report must be success before archiving ready files.")

    if promotion_report.get("status") != "success":
        raise RuntimeError("Promotion report must be success before archiving ready files.")

    source_files = detection_report.get("source_files", [])
    archive_dir = as_project_path(ARCHIVE_DIR) / utc_now_compact()
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_report_path.parent.mkdir(parents=True, exist_ok=True)

    archived_files = []

    for source in source_files:
        source_path_value = source.get("path") if isinstance(source, dict) else None
        if not source_path_value:
            continue

        source_path = as_project_path(source_path_value)
        if not source_path.exists():
            archived_files.append(
                {
                    "source_path": relative_to_project(source_path),
                    "archived_path": None,
                    "status": "missing_source_file",
                }
            )
            continue

        destination = archive_dir / source_path.name
        if destination.exists():
            destination = archive_dir / f"{source_path.stem}_{utc_now_compact()}{source_path.suffix}"

        shutil.move(str(source_path), str(destination))
        archived_files.append(
            {
                "source_path": relative_to_project(source_path),
                "archived_path": relative_to_project(destination),
                "status": "archived",
            }
        )

    report = {
        "status": "success",
        "archive_dir": relative_to_project(archive_dir),
        "archived_files_count": len([item for item in archived_files if item["status"] == "archived"]),
        "archived_files": archived_files,
        "merged_npz_path": detection_report.get("merged_npz_path"),
        "training_result_path": TRAINING_RESULT_PATH,
        "comparison_report_path": COMPARISON_REPORT_PATH,
        "promotion_report_path": PROMOTION_REPORT_PATH,
        "comparison_decision": comparison_report.get("decision"),
        "eligible_for_promotion": comparison_report.get("eligible_for_promotion"),
        "promotion_action": promotion_report.get("action"),
        "training_run_id": training_result.get("training_run_id"),
        "generated_at_utc": utc_now_iso(),
    }
    archive_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def optional_flag(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return value


def quote(value: str | int | float) -> str:
    return shlex.quote(str(value))


default_args = {
    "owner": "vulcadata",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id=DAG_ID,
    description="Conditional retraining pipeline for Vulcadata volcano alert model.",
    default_args=default_args,
    start_date=datetime(2026, 6, 17),
    schedule_interval=None,
    catchup=False,
    tags=["vulcadata", "retraining", "conditional"],
) as dag:
    start = EmptyOperator(task_id="start")

    detect_and_merge_ready_files = PythonOperator(
        task_id="detect_and_merge_ready_files",
        python_callable=detect_and_merge_ready_npz,
    )

    branch_should_retrain = BranchPythonOperator(
        task_id="branch_should_retrain",
        python_callable=choose_retraining_path,
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    train_candidate_model = BashOperator(
        task_id="train_candidate_model",
        bash_command=project_bash(
            f"""
python -m src.retraining.train_candidate_model \\
  --detection-report {quote(DETECTION_REPORT_PATH)} \\
  --training-script {quote(TRAINING_SCRIPT)} \\
  --output-dir {quote(CANDIDATE_OUTPUT_DIR)} \\
  --output-json {quote(TRAINING_RESULT_PATH)} \\
  --epochs {quote(TRAINING_EPOCHS)} \\
  --batch-size {quote(TRAINING_BATCH_SIZE)} \\
  --class-weighting {quote(TRAINING_CLASS_WEIGHTING)} {optional_flag(TRAINING_DEVICE_FLAG)}
"""
        ),
    )

    generate_retraining_evidently_report = BashOperator(
        task_id="generate_retraining_evidently_report",
        bash_command=project_bash(
            f"""
python -m src.retraining.generate_retraining_evidently_report \\
  --candidate-result {quote(TRAINING_RESULT_PATH)} \\
  --champion-decision {quote(CHAMPION_DECISION_PATH)} \\
  --reference-npz {quote(REFERENCE_NPZ_PATH)} \\
  --output-dir {quote(DRIFT_OUTPUT_DIR)}
"""
        ),
    )

    compare_candidate_to_champion = BashOperator(
        task_id="compare_candidate_to_champion",
        bash_command=project_bash(
            f"""
python -m src.retraining.compare_candidate_to_champion \\
  --candidate-result {quote(TRAINING_RESULT_PATH)} \\
  --champion-decision {quote(CHAMPION_DECISION_PATH)} \\
  --drift-summary {quote(DRIFT_SUMMARY_PATH)} \\
  --output-json {quote(COMPARISON_REPORT_PATH)} \\
  --min-epochs-for-promotion {quote(MIN_EPOCHS_FOR_PROMOTION)} \\
  --max-business-score-drop {quote(MAX_BUSINESS_SCORE_DROP)} \\
  --max-alert-24h-f1-drop {quote(MAX_ALERT_24H_F1_DROP)} \\
  --min-alert-24h-recall {quote(MIN_ALERT_24H_RECALL)} \\
  --min-alert-24h-precision {quote(MIN_ALERT_24H_PRECISION)} \\
  --max-class-5-f1-drop {quote(MAX_CLASS_5_F1_DROP)}
"""
        ),
    )

    promote_candidate_if_approved = BashOperator(
        task_id="promote_candidate_if_approved",
        bash_command=project_bash(
            f"""
python -m src.retraining.promote_candidate_if_approved \
  --project-root {quote(PROJECT_ROOT)} \
  --comparison-json {quote(COMPARISON_REPORT_PATH)} \
  --candidate-result {quote(TRAINING_RESULT_PATH)} \
  --decision-config {quote(CHAMPION_DECISION_PATH)} \
  --output-json {quote(PROMOTION_REPORT_PATH)} \
  --local-champion-checkpoint {quote(LOCAL_CHAMPION_CHECKPOINT_PATH)} \
  --local-champion-archive-dir {quote(LOCAL_CHAMPION_ARCHIVE_DIR)} \
  --decision-archive-dir {quote(DECISION_ARCHIVE_DIR)} \
  --s3-bucket {quote(S3_BUCKET)} \
  --s3-champion-key {quote(S3_CHAMPION_KEY)} \
  --s3-champion-archive-prefix {quote(S3_CHAMPION_ARCHIVE_PREFIX)} \
  --s3-decision-key {quote(S3_DECISION_KEY)} {optional_flag(PROMOTION_EXTRA_FLAGS)}
"""
        ),
    )

    archive_processed_source_files = PythonOperator(
        task_id="archive_processed_source_files",
        python_callable=archive_processed_source_files,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    start >> detect_and_merge_ready_files >> branch_should_retrain
    branch_should_retrain >> skip_retraining >> end
    (
        branch_should_retrain
        >> train_candidate_model
        >> generate_retraining_evidently_report
        >> compare_candidate_to_champion
        >> promote_candidate_if_approved
        >> archive_processed_source_files
        >> end
    )
