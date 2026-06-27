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

PERIODS_CSV = os.getenv(
    "VULCADATA_RETRAINING_PERIODS_CSV",
    "data/metadata/extraction_periods.csv",
)
PROCESSED_CSV_DIR = os.getenv(
    "VULCADATA_PROCESSED_CSV_DIR",
    "data/extraction/processed_csv",
)
RETRAINING_PERIODS_CSV = os.getenv(
    "VULCADATA_RETRAINING_PERIODS_RESOLVED_CSV",
    "reports/retraining/training_periods_for_preprocessing.csv",
)
RETRAINING_CSV_INPUT_CHECK_JSON = os.getenv(
    "VULCADATA_RETRAINING_CSV_INPUT_CHECK_JSON",
    "reports/retraining/processed_csv_input_check.json",
)
RETRAINING_OUTPUT_NAME = os.getenv(
    "VULCADATA_RETRAINING_OUTPUT_NAME",
    "volcano_multi_retraining_{{ ts_nodash }}.npz",
)
CSV_SUFFIX = os.getenv(
    "VULCADATA_AGGREGATED_CSV_SUFFIX",
    "_filtered_1_16Hz_aggregated_1min_with_fi.csv",
)

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
DECISION_MLFLOW_REPORT_PATH = os.getenv(
    "VULCADATA_RETRAINING_DECISION_MLFLOW_RESULT_JSON",
    "reports/retraining/retraining_decision_mlflow_result.json",
)
RETRAINING_GX_VALIDATION_OUTPUT_JSON = os.getenv(
    "VULCADATA_RETRAINING_GX_VALIDATION_JSON",
    "reports/retraining/retraining_dataset_gx_validation.json",
)


CHAMPION_DECISION_PATH = os.getenv(
    "VULCADATA_FINAL_MODEL_DECISION_JSON",
    "configs/final_model_decision.json",
)
REFERENCE_NPZ_PATH = os.getenv(
    "VULCADATA_RETRAINING_REFERENCE_NPZ",
    "data/preprocessing/processed/volcano_multi.npz",
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

FEATURE_WINDOW_MINUTES = int(os.getenv("VULCADATA_FEATURE_WINDOW_MINUTES", "10"))
SEQUENCE_STRIDE = int(os.getenv("VULCADATA_SEQUENCE_STRIDE", "5"))
MAX_HORIZON_HOURS = float(os.getenv("VULCADATA_MAX_HORIZON_HOURS", "48.0"))
ENTROPY_BINS = int(os.getenv("VULCADATA_ENTROPY_BINS", "20"))
SPLIT_STRATEGY = os.getenv("VULCADATA_SPLIT_STRATEGY", "chronological")
TRAIN_RATIO = float(os.getenv("VULCADATA_TRAIN_RATIO", "0.70"))
VAL_RATIO = float(os.getenv("VULCADATA_VAL_RATIO", "0.15"))

TRAINING_EPOCHS = int(os.getenv("VULCADATA_RETRAINING_EPOCHS", "2"))
TRAINING_BATCH_SIZE = int(os.getenv("VULCADATA_RETRAINING_BATCH_SIZE", "16"))
TRAINING_LEARNING_RATE = float(os.getenv("VULCADATA_RETRAINING_LEARNING_RATE", "3e-4"))
TRAINING_WEIGHT_DECAY = float(os.getenv("VULCADATA_RETRAINING_WEIGHT_DECAY", "1e-3"))
TRAINING_EARLY_STOPPING_PATIENCE = int(os.getenv("VULCADATA_RETRAINING_EARLY_STOPPING_PATIENCE", "3"))
TRAINING_EARLY_STOPPING_METRIC = os.getenv(
    "VULCADATA_RETRAINING_EARLY_STOPPING_METRIC",
    "business_score_classification",
)
TRAINING_CLASS_WEIGHTING = os.getenv("VULCADATA_RETRAINING_CLASS_WEIGHTING", "alert_priority")
TRAINING_N_CLASSES = int(os.getenv("VULCADATA_RETRAINING_N_CLASSES", "6"))
TRAINING_RUN_NAME = os.getenv("VULCADATA_RETRAINING_RUN_NAME", "airflow_cnn_transformer_candidate")
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

USE_MLFLOW = os.getenv("VULCADATA_RETRAINING_USE_MLFLOW", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
MLFLOW_TRACKING_URI = (
    os.getenv("VULCADATA_MLFLOW_TRACKING_URI")
    or os.getenv("MLFLOW_TRACKING_URI")
)
MLFLOW_EXPERIMENT_NAME = (
    os.getenv("VULCADATA_MLFLOW_EXPERIMENT_NAME")
    or os.getenv("MLFLOW_EXPERIMENT_NAME")
    or "Vulcadata"
)
DECISION_MLFLOW_RUN_NAME = os.getenv(
    "VULCADATA_RETRAINING_DECISION_MLFLOW_RUN_NAME",
    "airflow_retraining_decision",
)


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


def normalize_period_type(value: Any) -> str:
    normalized_value = clean_text(value).lower()

    if normalized_value in {"eruption", "eruptive", "event"}:
        return "eruption"

    if normalized_value in {"quiet", "calm", "calme", "background", "non_eruptive"}:
        return "quiet"

    if normalized_value in {"inference", "predict", "prediction", "unknown"}:
        return "inference"

    raise ValueError(
        f"Invalid period_type: {value}. Expected values: eruption, quiet or inference."
    )


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_retraining_periods_file() -> None:
    periods_path = as_project_path(PERIODS_CSV)
    processed_csv_dir = as_project_path(PROCESSED_CSV_DIR)
    retraining_periods_path = as_project_path(RETRAINING_PERIODS_CSV)
    check_report_path = as_project_path(RETRAINING_CSV_INPUT_CHECK_JSON)

    if not periods_path.exists():
        raise FileNotFoundError(f"Periods file not found: {periods_path}")

    periods = read_csv_auto(periods_path)
    periods.columns = [str(column).strip() for column in periods.columns]

    required_columns = {"period_id", "period_type"}
    missing_columns = required_columns - set(periods.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in periods file: {sorted(missing_columns)}")

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
            raise ValueError("Empty period_id in periods file.")

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
                    "reason": "period_type_inference_excluded_from_retraining",
                    "csv_path": relative_to_project(csv_path),
                }
            )
            continue

        if period_type == "eruption" and not clean_text(row.get("eruption_start_utc")):
            raise ValueError(f"Missing eruption_start_utc for eruptive period: {period_id}")

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
        raise FileNotFoundError(f"Missing aggregated CSV files. See report: {check_report_path}")

    if not selected_rows:
        report = {
            "status": "failed",
            "reason": "no_retraining_period_available",
            "periods_csv": relative_to_project(periods_path),
            "processed_csv_dir": relative_to_project(processed_csv_dir),
            "selected_rows_count": 0,
            "skipped_rows_count": len(skipped_rows),
            "skipped_rows": skipped_rows,
            "generated_at_utc": utc_now_iso(),
        }
        write_json(report, check_report_path)
        raise ValueError("No retraining period available after excluding inference periods.")

    import pandas as pd

    retraining_periods_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(retraining_periods_path, index=False)

    report = {
        "status": "success",
        "periods_csv": relative_to_project(periods_path),
        "processed_csv_dir": relative_to_project(processed_csv_dir),
        "retraining_periods_csv": relative_to_project(retraining_periods_path),
        "selected_rows_count": len(selected_rows),
        "skipped_rows_count": len(skipped_rows),
        "missing_csv_count": 0,
        "selected_periods": [row["period_id"] for row in selected_rows],
        "skipped_rows": skipped_rows,
        "csv_suffix": CSV_SUFFIX,
        "generated_at_utc": utc_now_iso(),
    }
    write_json(report, check_report_path)


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
    reference_npz_path = as_project_path(REFERENCE_NPZ_PATH)

    ready_dir.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)
    detection_report_path.parent.mkdir(parents=True, exist_ok=True)

    source_files = sorted(ready_dir.glob("*.npz"))
    source_file_infos = [npz_file_info(path) for path in source_files]

    base_reference_info: dict[str, Any] | None = None
    if reference_npz_path.exists():
        base_reference_info = npz_file_info(reference_npz_path)

    if len(source_files) < MIN_NEW_FILES_FOR_RETRAINING:
        report = {
            "status": "success",
            "should_retrain": False,
            "ready_dir": READY_DIR,
            "merged_dir": MERGED_DIR,
            "base_reference_npz": base_reference_info,
            "candidate_files_count": len(source_files),
            "min_new_files_for_retraining": MIN_NEW_FILES_FOR_RETRAINING,
            "selection_policy": "base_reference_plus_all_ready_npz_on_sequence_axis",
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

    if not reference_npz_path.exists():
        raise FileNotFoundError(f"Reference NPZ not found: {reference_npz_path}")

    reference_payload = load_and_validate_npz(reference_npz_path)
    ready_payloads = [load_and_validate_npz(path) for path in source_files]
    payloads_to_merge = [reference_payload, *ready_payloads]

    merged_payload: dict[str, Any] = {}

    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        merged_payload[key] = np.concatenate(
            [payload[key] for payload in payloads_to_merge],
            axis=0,
        )

    for key, value in reference_payload.items():
        if key not in merged_payload:
            merged_payload[key] = value

    timestamp = utc_now_compact()
    merged_dir = merged_root / timestamp
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_npz_path = merged_dir / "candidate_training_dataset.npz"
    np.savez_compressed(merged_npz_path, **merged_payload)

    merged_info = npz_file_info(merged_npz_path)
    merged_info["base_reference_npz"] = base_reference_info
    merged_info["new_source_files_count"] = len(source_files)
    merged_info["new_source_files"] = source_file_infos
    merged_info["selection_policy"] = "base_reference_plus_all_ready_npz_on_sequence_axis"

    split_shapes = {}
    for split_name in ("train", "val", "test"):
        split_shapes[split_name] = {
            "X_shape": list(merged_payload[f"X_{split_name}"].shape),
            "y_shape": list(merged_payload[f"y_{split_name}"].shape),
        }

    report = {
        "status": "success",
        "should_retrain": True,
        "ready_dir": READY_DIR,
        "merged_dir": MERGED_DIR,
        "base_reference_npz": base_reference_info,
        "candidate_files_count": len(source_files),
        "min_new_files_for_retraining": MIN_NEW_FILES_FOR_RETRAINING,
        "selection_policy": "base_reference_plus_all_ready_npz_on_sequence_axis",
        "concat_axis": 0,
        "expected_seq_len": EXPECTED_SEQ_LEN,
        "expected_n_features": EXPECTED_N_FEATURES,
        "source_files": source_file_infos,
        "files_to_process": [merged_info],
        "merged_npz_path": relative_to_project(merged_npz_path),
        "merged_split_shapes": split_shapes,
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


def mlflow_training_flags() -> str:
    if not USE_MLFLOW:
        return ""

    flags = ["--use-mlflow"]

    if MLFLOW_TRACKING_URI:
        flags.extend(["--mlflow-tracking-uri", quote(MLFLOW_TRACKING_URI)])

    if MLFLOW_EXPERIMENT_NAME:
        flags.extend(["--mlflow-experiment-name", quote(MLFLOW_EXPERIMENT_NAME)])

    return " ".join(flags)


def mlflow_decision_flags() -> str:
    flags: list[str] = []

    if MLFLOW_TRACKING_URI:
        flags.extend(["--mlflow-tracking-uri", quote(MLFLOW_TRACKING_URI)])

    if MLFLOW_EXPERIMENT_NAME:
        flags.extend(["--mlflow-experiment-name", quote(MLFLOW_EXPERIMENT_NAME)])

    return " ".join(flags)


def quote(value: str | int | float) -> str:
    return shlex.quote(str(value))


preprocess_retraining_dataset_command = project_bash(
    "python -m src.preprocessing.preprocess_volcano_dataset "
    "--mode training "
    f"--periods {quote(RETRAINING_PERIODS_CSV)} "
    f"--processed-csv-dir {quote(PROCESSED_CSV_DIR)} "
    f"--output-dir {quote(READY_DIR)} "
    f"--training-output-name {quote(RETRAINING_OUTPUT_NAME)} "
    f"--feature-window-minutes {quote(FEATURE_WINDOW_MINUTES)} "
    f"--seq-len {quote(EXPECTED_SEQ_LEN)} "
    f"--sequence-stride {quote(SEQUENCE_STRIDE)} "
    f"--max-horizon-hours {quote(MAX_HORIZON_HOURS)} "
    f"--entropy-bins {quote(ENTROPY_BINS)} "
    f"--n-classes {quote(TRAINING_N_CLASSES)} "
    f"--split-strategy {quote(SPLIT_STRATEGY)} "
    f"--train-ratio {quote(TRAIN_RATIO)} "
    f"--val-ratio {quote(VAL_RATIO)}"
)

RETRAINING_NPZ_PATH = str(Path(READY_DIR) / RETRAINING_OUTPUT_NAME)

validate_retraining_dataset_command = project_bash(
    "python -m src.retraining.validate_retraining_dataset "
    f"--npz-path {quote(RETRAINING_NPZ_PATH)} "
    f"--expected-seq-len {quote(EXPECTED_SEQ_LEN)} "
    f"--expected-n-features {quote(EXPECTED_N_FEATURES)} "
    f"--n-classes {quote(TRAINING_N_CLASSES)} "
    f"--output-json {quote(RETRAINING_GX_VALIDATION_OUTPUT_JSON)}"
)


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
    tags=["vulcadata", "retraining", "conditional", "great-expectations"],
) as dag:
    start = EmptyOperator(task_id="start")

    check_retraining_csv_inputs = PythonOperator(
        task_id="check_retraining_csv_inputs",
        python_callable=build_retraining_periods_file,
    )

    preprocess_retraining_dataset = BashOperator(
        task_id="preprocess_retraining_dataset",
        bash_command=preprocess_retraining_dataset_command,
    )

    validate_retraining_dataset = BashOperator(
        task_id="validate_retraining_dataset",
        bash_command=validate_retraining_dataset_command,
    )

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
python -m src.retraining.train_candidate_model \
  --detection-report {quote(DETECTION_REPORT_PATH)} \
  --training-script {quote(TRAINING_SCRIPT)} \
  --output-dir {quote(CANDIDATE_OUTPUT_DIR)} \
  --output-json {quote(TRAINING_RESULT_PATH)} \
  --epochs {quote(TRAINING_EPOCHS)} \
  --batch-size {quote(TRAINING_BATCH_SIZE)} \
  --learning-rate {quote(TRAINING_LEARNING_RATE)} \
  --weight-decay {quote(TRAINING_WEIGHT_DECAY)} \
  --early-stopping-patience {quote(TRAINING_EARLY_STOPPING_PATIENCE)} \
  --early-stopping-metric {quote(TRAINING_EARLY_STOPPING_METRIC)} \
  --class-weighting {quote(TRAINING_CLASS_WEIGHTING)} \
  --n-classes {quote(TRAINING_N_CLASSES)} \
  --run-name {quote(TRAINING_RUN_NAME)} {mlflow_training_flags()} {optional_flag(TRAINING_DEVICE_FLAG)}
"""
        ),
    )

    generate_retraining_evidently_report = BashOperator(
        task_id="generate_retraining_evidently_report",
        bash_command=project_bash(
            f"""
python -m src.retraining.generate_retraining_evidently_report \
  --candidate-result {quote(TRAINING_RESULT_PATH)} \
  --champion-decision {quote(CHAMPION_DECISION_PATH)} \
  --reference-npz {quote(REFERENCE_NPZ_PATH)} \
  --output-dir {quote(DRIFT_OUTPUT_DIR)}
"""
        ),
    )

    compare_candidate_to_champion = BashOperator(
        task_id="compare_candidate_to_champion",
        bash_command=project_bash(
            f"""
python -m src.retraining.compare_candidate_to_champion \
  --candidate-result {quote(TRAINING_RESULT_PATH)} \
  --champion-decision {quote(CHAMPION_DECISION_PATH)} \
  --drift-summary {quote(DRIFT_SUMMARY_PATH)} \
  --output-json {quote(COMPARISON_REPORT_PATH)} \
  --min-epochs-for-promotion {quote(MIN_EPOCHS_FOR_PROMOTION)} \
  --max-business-score-drop {quote(MAX_BUSINESS_SCORE_DROP)} \
  --max-alert-24h-f1-drop {quote(MAX_ALERT_24H_F1_DROP)} \
  --min-alert-24h-recall {quote(MIN_ALERT_24H_RECALL)} \
  --min-alert-24h-precision {quote(MIN_ALERT_24H_PRECISION)} \
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

    archive_processed_source_files_task = PythonOperator(
        task_id="archive_processed_source_files",
        python_callable=archive_processed_source_files,
    )

    log_retraining_decision_to_mlflow = BashOperator(
        task_id="log_retraining_decision_to_mlflow",
        bash_command=project_bash(
            f"""
python -m src.retraining.log_retraining_decision_to_mlflow \
  --candidate-result {quote(TRAINING_RESULT_PATH)} \
  --comparison-json {quote(COMPARISON_REPORT_PATH)} \
  --promotion-result {quote(PROMOTION_REPORT_PATH)} \
  --drift-summary {quote(DRIFT_SUMMARY_PATH)} \
  --detection-report {quote(DETECTION_REPORT_PATH)} \
  --archive-report {quote(ARCHIVE_REPORT_PATH)} \
  --output-json {quote(DECISION_MLFLOW_REPORT_PATH)} \
  --run-name {quote(DECISION_MLFLOW_RUN_NAME)} {mlflow_decision_flags()}
"""
        ),
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    (
        start
        >> check_retraining_csv_inputs
        >> preprocess_retraining_dataset
        >> validate_retraining_dataset
        >> detect_and_merge_ready_files
        >> branch_should_retrain
    )
    branch_should_retrain >> skip_retraining >> end
    (
        branch_should_retrain
        >> train_candidate_model
        >> generate_retraining_evidently_report
        >> compare_candidate_to_champion
        >> promote_candidate_if_approved
        >> archive_processed_source_files_task
        >> log_retraining_decision_to_mlflow
        >> end
    )
