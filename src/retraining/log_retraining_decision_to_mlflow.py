from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_RESULT = "reports/retraining/candidate_training_result.json"
DEFAULT_COMPARISON_JSON = "reports/retraining/candidate_vs_champion_comparison.json"
DEFAULT_PROMOTION_RESULT = "reports/retraining/candidate_promotion_result.json"
DEFAULT_DRIFT_SUMMARY = "reports/retraining/evidently/candidate_drift_summary.json"
DEFAULT_DETECTION_REPORT = "reports/retraining/new_preprocessed_files_detection.json"
DEFAULT_ARCHIVE_REPORT = "reports/retraining/archive_processed_ready_files.json"
DEFAULT_OUTPUT_JSON = "reports/retraining/retraining_decision_mlflow_result.json"
DEFAULT_RUN_NAME = "airflow_retraining_decision"
DEFAULT_EXPERIMENT_NAME = "Vulcadata"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_bool_metric(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    return None


def as_float_metric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric):
        return None
    return metric


def add_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    metric = as_float_metric(value)
    if metric is not None:
        metrics[name] = metric


def add_bool_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    metric = as_bool_metric(value)
    if metric is not None:
        metrics[name] = float(metric)


def collect_metrics(
    candidate_result: dict[str, Any],
    comparison: dict[str, Any],
    promotion: dict[str, Any],
    drift_summary: dict[str, Any],
    detection_report: dict[str, Any],
    archive_report: dict[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    add_bool_metric(metrics, "eligible_for_promotion", comparison.get("eligible_for_promotion"))
    add_bool_metric(metrics, "promotion_applied", promotion.get("action") == "candidate_promoted")
    add_bool_metric(metrics, "promotion_skipped", promotion.get("action") == "promotion_skipped")
    add_bool_metric(metrics, "dry_run", candidate_result.get("dry_run"))

    add_metric(metrics, "candidate_epochs", comparison.get("candidate_epochs"))
    add_metric(metrics, "files_to_process_count", candidate_result.get("files_to_process_count"))
    add_metric(metrics, "archived_files_count", archive_report.get("archived_files_count"))

    candidate_metrics = comparison.get("candidate_metrics") or {}
    champion_metrics = comparison.get("champion_metrics") or {}

    for key, value in candidate_metrics.items():
        add_metric(metrics, f"candidate_{key}", value)

    for key, value in champion_metrics.items():
        add_metric(metrics, f"champion_{key}", value)

    test_metrics = safe_get(candidate_result, ["metrics", "test"], {}) or {}
    for key, value in test_metrics.items():
        add_metric(metrics, f"candidate_test_{key}", value)

    add_metric(metrics, "candidate_best_val_score", safe_get(candidate_result, ["metrics", "best_val_score"]))
    add_metric(metrics, "candidate_n_params", safe_get(candidate_result, ["metrics", "n_params"]))
    add_metric(metrics, "candidate_n_trainable_params", safe_get(candidate_result, ["metrics", "n_trainable_params"]))

    add_bool_metric(metrics, "critical_drift_detected", drift_summary.get("critical_drift_detected"))
    add_bool_metric(metrics, "candidate_rejected_by_drift_check", drift_summary.get("candidate_rejected_by_drift_check"))
    add_bool_metric(metrics, "dataset_drift", drift_summary.get("dataset_drift"))
    add_bool_metric(metrics, "target_drift", drift_summary.get("target_drift"))
    add_bool_metric(metrics, "prediction_drift", drift_summary.get("prediction_drift"))

    new_source_files = detection_report.get("source_files") or []
    if isinstance(new_source_files, list):
        metrics["ready_source_files_count"] = float(len(new_source_files))

    return metrics


def collect_tags(
    candidate_result: dict[str, Any],
    comparison: dict[str, Any],
    promotion: dict[str, Any],
    drift_summary: dict[str, Any],
    archive_report: dict[str, Any],
) -> dict[str, str]:
    candidate_mlflow_run_id = (
        comparison.get("candidate_mlflow_run_id")
        or candidate_result.get("mlflow_run_id")
        or promotion.get("candidate_mlflow_run_id")
    )
    candidate_training_run_id = (
        comparison.get("candidate_training_run_id")
        or candidate_result.get("training_run_id")
        or promotion.get("candidate_training_run_id")
    )

    tags = {
        "pipeline": "volcano_retraining_pipeline",
        "run_role": "retraining_decision",
        "decision": str(comparison.get("decision")),
        "promotion_action": str(promotion.get("action")),
        "promotion_reason": str(promotion.get("reason")),
        "candidate_training_run_id": str(candidate_training_run_id),
        "candidate_mlflow_run_id": str(candidate_mlflow_run_id),
        "orchestration_run_id": str(candidate_result.get("orchestration_run_id")),
        "candidate_input_npz": str(candidate_result.get("input_npz")),
        "drift_reason": str(drift_summary.get("reason")),
        "archive_dir": str(archive_report.get("archive_dir")),
    }

    if candidate_mlflow_run_id:
        tags["mlflow.parentRunId"] = str(candidate_mlflow_run_id)

    return {key: value[:500] for key, value in tags.items() if value not in {"None", ""}}


def collect_params(
    candidate_result: dict[str, Any],
    comparison: dict[str, Any],
    promotion: dict[str, Any],
    drift_summary: dict[str, Any],
    detection_report: dict[str, Any],
) -> dict[str, str]:
    params = {
        "candidate_status": candidate_result.get("status"),
        "model_family": candidate_result.get("model_family"),
        "task_type": candidate_result.get("task_type"),
        "run_name": candidate_result.get("run_name"),
        "comparison_status": comparison.get("status"),
        "promotion_status": promotion.get("status"),
        "drift_status": drift_summary.get("status"),
        "selection_policy": detection_report.get("selection_policy"),
        "best_val_metric": safe_get(candidate_result, ["metrics", "best_val_metric"]),
    }
    return {key: str(value)[:250] for key, value in params.items() if value is not None}


def log_artifact_if_exists(mlflow: Any, path: Path, artifact_path: str = "retraining_reports") -> None:
    if path.exists() and path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def resolve_mlflow_artifact_uri(mlflow: Any, run_id: str | None) -> str | None:
    if not run_id:
        return None

    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
    except Exception:
        return None

    artifact_uri = getattr(run.info, "artifact_uri", None)
    if not artifact_uri:
        return None

    return str(artifact_uri)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_result_path = Path(args.candidate_result)
    comparison_path = Path(args.comparison_json)
    promotion_path = Path(args.promotion_result)
    drift_summary_path = Path(args.drift_summary)
    detection_report_path = Path(args.detection_report)
    archive_report_path = Path(args.archive_report)
    output_json_path = Path(args.output_json)

    candidate_result = read_json(candidate_result_path)
    comparison = read_json(comparison_path)
    promotion = read_json(promotion_path)
    drift_summary = read_json(drift_summary_path) if drift_summary_path.exists() else {}
    detection_report = read_json(detection_report_path) if detection_report_path.exists() else {}
    archive_report = read_json(archive_report_path) if archive_report_path.exists() else {}

    tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = args.mlflow_experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME

    candidate_mlflow_run_id = (
        comparison.get("candidate_mlflow_run_id")
        or candidate_result.get("mlflow_run_id")
        or promotion.get("candidate_mlflow_run_id")
    )
    candidate_mlflow_artifact_uri: str | None = None

    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        metrics = collect_metrics(
            candidate_result=candidate_result,
            comparison=comparison,
            promotion=promotion,
            drift_summary=drift_summary,
            detection_report=detection_report,
            archive_report=archive_report,
        )
        tags = collect_tags(
            candidate_result=candidate_result,
            comparison=comparison,
            promotion=promotion,
            drift_summary=drift_summary,
            archive_report=archive_report,
        )
        params = collect_params(
            candidate_result=candidate_result,
            comparison=comparison,
            promotion=promotion,
            drift_summary=drift_summary,
            detection_report=detection_report,
        )

        candidate_mlflow_run_id = tags.get("candidate_mlflow_run_id") or candidate_mlflow_run_id
        candidate_mlflow_artifact_uri = resolve_mlflow_artifact_uri(
            mlflow=mlflow,
            run_id=candidate_mlflow_run_id,
        )

        if candidate_mlflow_artifact_uri:
            tags["candidate_mlflow_artifact_uri"] = candidate_mlflow_artifact_uri[:500]
            params["candidate_mlflow_artifact_uri"] = candidate_mlflow_artifact_uri[:500]

        with mlflow.start_run(run_name=args.run_name, tags=tags) as active_run:
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)

            log_artifact_if_exists(mlflow, candidate_result_path)
            log_artifact_if_exists(mlflow, comparison_path)
            log_artifact_if_exists(mlflow, promotion_path)
            log_artifact_if_exists(mlflow, drift_summary_path)
            log_artifact_if_exists(mlflow, detection_report_path)
            log_artifact_if_exists(mlflow, archive_report_path)

            html_report = drift_summary.get("output_html")
            if html_report:
                log_artifact_if_exists(mlflow, Path(html_report), artifact_path="evidently")

            decision_mlflow_run_id = active_run.info.run_id
            decision_mlflow_artifact_uri = active_run.info.artifact_uri

        payload = {
            "status": "success",
            "decision_mlflow_run_id": decision_mlflow_run_id,
            "decision_mlflow_artifact_uri": decision_mlflow_artifact_uri,
            "candidate_mlflow_run_id": tags.get("candidate_mlflow_run_id"),
            "candidate_mlflow_artifact_uri": candidate_mlflow_artifact_uri,
            "candidate_training_run_id": tags.get("candidate_training_run_id"),
            "decision": comparison.get("decision"),
            "promotion_action": promotion.get("action"),
            "experiment_name": experiment_name,
            "tracking_uri": tracking_uri,
            "run_name": args.run_name,
            "logged_metrics_count": len(metrics),
            "logged_params_count": len(params),
            "generated_at_utc": utc_now_iso(),
        }
        write_json(payload, output_json_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    except Exception as exc:
        payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "candidate_mlflow_run_id": candidate_mlflow_run_id,
            "candidate_mlflow_artifact_uri": candidate_mlflow_artifact_uri,
            "decision": comparison.get("decision"),
            "promotion_action": promotion.get("action"),
            "experiment_name": experiment_name,
            "tracking_uri": tracking_uri,
            "run_name": args.run_name,
            "generated_at_utc": utc_now_iso(),
        }
        write_json(payload, output_json_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.allow_fail:
            return payload
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log the Vulcadata retraining comparison and promotion decision to MLflow."
    )
    parser.add_argument("--candidate-result", default=DEFAULT_CANDIDATE_RESULT)
    parser.add_argument("--comparison-json", default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--promotion-result", default=DEFAULT_PROMOTION_RESULT)
    parser.add_argument("--drift-summary", default=DEFAULT_DRIFT_SUMMARY)
    parser.add_argument("--detection-report", default=DEFAULT_DETECTION_REPORT)
    parser.add_argument("--archive-report", default=DEFAULT_ARCHIVE_REPORT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment-name", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
