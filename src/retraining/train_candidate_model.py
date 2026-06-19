from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DETECTION_REPORT_PATH = "reports/retraining/new_preprocessed_files_detection.json"
DEFAULT_TRAINING_SCRIPT = "scripts/train_cnn_transformer_classif_modified_Vfinetuning.py"
DEFAULT_OUTPUT_DIR = "models/retraining/cnn_transformer_candidate"
DEFAULT_OUTPUT_JSON = "reports/retraining/candidate_training_result.json"

DEFAULT_METRICS_NAME = "metrics_cnn_transformer_classifier.json"
DEFAULT_HISTORY_NAME = "history_cnn_transformer_classifier.json"
DEFAULT_BEST_MODEL_NAME = "best_cnn_transformer_classifier.pt"
DEFAULT_PREDICTIONS_NAME = "predictions_cnn_transformer_classifier.npz"
DEFAULT_CONFUSION_NAME = "confusion_matrix_cnn_transformer_classifier.npy"


class CandidateTrainingError(RuntimeError):
    """Raised when candidate model training cannot be executed or validated."""


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def compact_utc_timestamp() -> str:
    return utc_now_iso().replace("-", "").replace(":", "").replace("Z", "Z")


def normalize_local_path(path: str | Path) -> str:
    return str(Path(path).as_posix())


def read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise CandidateTrainingError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise CandidateTrainingError(f"JSON file must contain an object: {path}")

    return payload


def write_json_local(payload: dict[str, Any], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root:
        return Path(project_root).resolve()

    return Path.cwd().resolve()


def validate_detection_report(detection_report: dict[str, Any]) -> list[dict[str, Any]]:
    if detection_report.get("status") != "success":
        raise CandidateTrainingError(
            "Detection report status must be 'success'. "
            f"Got: {detection_report.get('status')}"
        )

    if detection_report.get("should_retrain") is not True:
        return []

    files_to_process = detection_report.get("files_to_process")

    if not isinstance(files_to_process, list):
        raise CandidateTrainingError(
            "Detection report must contain a list field 'files_to_process'."
        )

    if not files_to_process:
        raise CandidateTrainingError(
            "Detection report has should_retrain=true but files_to_process is empty."
        )

    for index, file_record in enumerate(files_to_process):
        if not isinstance(file_record, dict):
            raise CandidateTrainingError(f"files_to_process[{index}] must be an object.")

        if not file_record.get("path"):
            raise CandidateTrainingError(f"files_to_process[{index}] must contain 'path'.")

    return files_to_process


def select_input_npz(
    *,
    project_root: Path,
    files_to_process: list[dict[str, Any]],
    allow_multiple_files: bool,
) -> tuple[str, Path]:
    if len(files_to_process) > 1 and not allow_multiple_files:
        raise CandidateTrainingError(
            "Multiple files_to_process detected. Current wrapper expects one NPZ. "
            "Use --allow-multiple-files only after defining a merge policy."
        )

    selected_file = files_to_process[0]
    input_npz = str(selected_file["path"])
    input_npz_path = Path(input_npz)

    if not input_npz_path.is_absolute():
        input_npz_path = project_root / input_npz_path

    if not input_npz_path.exists():
        raise CandidateTrainingError(f"Input NPZ not found: {input_npz_path}")

    return input_npz, input_npz_path


def expected_artifact_paths(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)

    return {
        "best_model_path": normalize_local_path(output_dir / DEFAULT_BEST_MODEL_NAME),
        "history_path": normalize_local_path(output_dir / DEFAULT_HISTORY_NAME),
        "metrics_path": normalize_local_path(output_dir / DEFAULT_METRICS_NAME),
        "predictions_path": normalize_local_path(output_dir / DEFAULT_PREDICTIONS_NAME),
        "confusion_matrix_path": normalize_local_path(output_dir / DEFAULT_CONFUSION_NAME),
    }


def read_metrics_if_available(metrics_path: str | Path) -> dict[str, Any] | None:
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise CandidateTrainingError(f"Metrics file must contain an object: {metrics_path}")

    return payload


def build_training_command(
    *,
    training_script: str,
    input_npz: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_metric: str,
    class_weighting: str,
    n_classes: int,
    run_name: str,
    cpu: bool,
    use_amp: bool,
    use_mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str | None,
) -> list[str]:
    command = [
        sys.executable,
        training_script,
        "--input-npz",
        input_npz,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--early-stopping-patience",
        str(early_stopping_patience),
        "--early-stopping-metric",
        early_stopping_metric,
        "--class-weighting",
        class_weighting,
        "--n-classes",
        str(n_classes),
        "--run-name",
        run_name,
    ]

    if cpu:
        command.append("--cpu")

    if use_amp:
        command.append("--use-amp")

    if use_mlflow:
        command.append("--use-mlflow")

    if mlflow_tracking_uri:
        command.extend(["--mlflow-tracking-uri", mlflow_tracking_uri])

    if mlflow_experiment_name:
        command.extend(["--mlflow-experiment-name", mlflow_experiment_name])

    return command


def resolve_mlflow_run_id(
    *,
    run_name: str,
    use_mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str | None,
) -> str | None:
    if not use_mlflow:
        return None

    try:
        import mlflow
    except ImportError:
        return None

    tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = (
        mlflow_experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "Vulcadata"
    )

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )

        if not runs:
            return None

        return runs[0].info.run_id

    except Exception:
        return None


def train_candidate_model(
    *,
    project_root: str | Path | None,
    detection_report_path: str | Path,
    training_script: str | Path,
    output_dir: str | Path,
    output_json: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_metric: str,
    class_weighting: str,
    n_classes: int,
    run_name: str | None,
    cpu: bool,
    use_amp: bool,
    use_mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str | None,
    dry_run: bool,
    allow_multiple_files: bool,
) -> dict[str, Any]:
    project_root_path = resolve_project_root(project_root)

    if epochs < 1:
        raise CandidateTrainingError("epochs must be >= 1.")

    if batch_size < 1:
        raise CandidateTrainingError("batch_size must be >= 1.")

    detection_report = read_json_object(project_root_path / detection_report_path)
    files_to_process = validate_detection_report(detection_report)

    generated_at = utc_now_iso()
    orchestration_run_id = f"candidate_training_{compact_utc_timestamp()}"

    resolved_run_name = run_name or orchestration_run_id

    if not files_to_process:
        result = {
            "status": "skipped",
            "reason": "should_retrain is false",
            "dry_run": dry_run,
            "project_root": normalize_local_path(project_root_path),
            "detection_report_path": normalize_local_path(detection_report_path),
            "output_json": normalize_local_path(output_json),
            "generated_at_utc": generated_at,
        }
        write_json_local(result, project_root_path / output_json)
        return result

    input_npz, input_npz_path = select_input_npz(
        project_root=project_root_path,
        files_to_process=files_to_process,
        allow_multiple_files=allow_multiple_files,
    )

    training_script = normalize_local_path(training_script)
    training_script_path = project_root_path / training_script

    if not training_script_path.exists():
        raise CandidateTrainingError(f"Training script not found: {training_script_path}")

    output_dir = normalize_local_path(output_dir)
    artifact_paths = expected_artifact_paths(output_dir)

    command = build_training_command(
        training_script=training_script,
        input_npz=input_npz,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        class_weighting=class_weighting,
        n_classes=n_classes,
        run_name=resolved_run_name,
        cpu=cpu,
        use_amp=use_amp,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    base_result = {
        "status": "success",
        "dry_run": dry_run,
        "model_family": "cnn_transformer",
        "task_type": "multiclass_classification",
        "orchestration_run_id": orchestration_run_id,
        "training_run_id": orchestration_run_id,
        "mlflow_run_id": None,
        "run_name": resolved_run_name,
        "project_root": normalize_local_path(project_root_path),
        "training_script": training_script,
        "input_npz": input_npz,
        "input_npz_absolute_path": normalize_local_path(input_npz_path),
        "output_dir": output_dir,
        "detection_report_path": normalize_local_path(detection_report_path),
        "files_to_process_count": len(files_to_process),
        "files_to_process": files_to_process,
        "command": command,
        "artifacts": artifact_paths,
        "metrics": None,
        "generated_at_utc": generated_at,
    }

    if dry_run:
        write_json_local(base_result, project_root_path / output_json)
        return base_result

    completed = subprocess.run(
        command,
        cwd=project_root_path,
        check=False,
    )

    if completed.returncode != 0:
        raise CandidateTrainingError(
            f"Candidate training failed with return code {completed.returncode}."
        )

    metrics_path = project_root_path / artifact_paths["metrics_path"]
    metrics = read_metrics_if_available(metrics_path)

    missing_artifacts = [
        name
        for name, path in artifact_paths.items()
        if not (project_root_path / path).exists()
    ]

    if missing_artifacts:
        raise CandidateTrainingError(
            f"Missing expected training artifacts: {missing_artifacts}"
        )

    mlflow_run_id = resolve_mlflow_run_id(
        run_name=resolved_run_name,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    result = {
        **base_result,
        "training_run_id": mlflow_run_id or orchestration_run_id,
        "mlflow_run_id": mlflow_run_id,
        "metrics": metrics,
        "completed_at_utc": utc_now_iso(),
    }

    write_json_local(result, project_root_path / output_json)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a candidate Vulcadata model from the retraining detection report."
    )

    parser.add_argument(
        "--project-root",
        default=os.getenv("VULCADATA_PROJECT_ROOT"),
    )

    parser.add_argument(
        "--detection-report",
        default=os.getenv(
            "VULCADATA_NEW_FILES_DETECTION_OUTPUT_JSON",
            DEFAULT_DETECTION_REPORT_PATH,
        ),
    )

    parser.add_argument(
        "--training-script",
        default=os.getenv(
            "VULCADATA_CANDIDATE_TRAINING_SCRIPT",
            DEFAULT_TRAINING_SCRIPT,
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "VULCADATA_CANDIDATE_OUTPUT_DIR",
            DEFAULT_OUTPUT_DIR,
        ),
    )

    parser.add_argument(
        "--output-json",
        default=os.getenv(
            "VULCADATA_CANDIDATE_TRAINING_RESULT_JSON",
            DEFAULT_OUTPUT_JSON,
        ),
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-metric", type=str, default="business_score_classification")
    parser.add_argument("--class-weighting", type=str, default="early_warning_priority")
    parser.add_argument("--n-classes", type=int, default=6)

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-amp", action="store_true")

    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment-name", default=None)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-multiple-files", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = train_candidate_model(
            project_root=args.project_root,
            detection_report_path=args.detection_report,
            training_script=args.training_script,
            output_dir=args.output_dir,
            output_json=args.output_json,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_metric=args.early_stopping_metric,
            class_weighting=args.class_weighting,
            n_classes=args.n_classes,
            run_name=args.run_name,
            cpu=args.cpu,
            use_amp=args.use_amp,
            use_mlflow=args.use_mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            dry_run=args.dry_run,
            allow_multiple_files=args.allow_multiple_files,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as exc:
        project_root_path = resolve_project_root(args.project_root)

        failure_payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "dry_run": args.dry_run,
            "training_script": args.training_script,
            "detection_report_path": args.detection_report,
            "output_json": args.output_json,
            "generated_at_utc": utc_now_iso(),
        }

        write_json_local(failure_payload, project_root_path / args.output_json)

        print(json.dumps(failure_payload, indent=2, ensure_ascii=False))

        raise


if __name__ == "__main__":
    main()