import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_COMPARISON_JSON = "reports/retraining/candidate_vs_champion_comparison.json"
DEFAULT_CANDIDATE_RESULT = "reports/retraining/candidate_training_result.json"
DEFAULT_DECISION_CONFIG = "configs/final_model_decision.json"
DEFAULT_OUTPUT_JSON = "reports/retraining/candidate_promotion_result.json"
DEFAULT_LOCAL_CHAMPION_CHECKPOINT = "models/champion_classification_checkpoint/best_cnn_transformer_classifier.pt"
DEFAULT_LOCAL_CHAMPION_ARCHIVE_DIR = "models/champion_classification_checkpoint/archive"
DEFAULT_DECISION_ARCHIVE_DIR = "configs/model_decision_archive"
DEFAULT_S3_BUCKET = "vulcadata"
DEFAULT_S3_CHAMPION_KEY = "models/champion_classification_checkpoint/best_cnn_transformer_classifier.pt"
DEFAULT_S3_CHAMPION_ARCHIVE_PREFIX = "models/champion_classification_checkpoint/archive"
DEFAULT_S3_DECISION_KEY = "model_decisions/final_model_decision.json"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")

    return payload


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def resolve_path(path: str | Path, project_root: str | Path = ".") -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return Path(project_root) / path


def get_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(current, dict):
            return None

        if key not in current:
            return None

        current = current[key]

    return current


def first_existing_path(payload: dict[str, Any], paths: list[list[str]]) -> str | None:
    for path in paths:
        value = get_nested(payload, path)

        if isinstance(value, str) and value.strip():
            return value

    return None


def extract_candidate_checkpoint(candidate_result: dict[str, Any]) -> str:
    candidate_checkpoint = first_existing_path(
        candidate_result,
        [
            ["artifacts", "best_model_path"],
            ["best_model_path"],
            ["model_path"],
            ["checkpoint_path"],
        ],
    )

    if candidate_checkpoint:
        return candidate_checkpoint

    raise ValueError("Unable to locate candidate checkpoint in candidate result JSON.")


def get_candidate_training_run_id(
    comparison: dict[str, Any],
    candidate_result: dict[str, Any],
) -> str | None:
    comparison_value = comparison.get("candidate_training_run_id")

    if isinstance(comparison_value, str) and comparison_value.strip():
        return comparison_value

    candidate_value = candidate_result.get("training_run_id")

    if isinstance(candidate_value, str) and candidate_value.strip():
        return candidate_value

    orchestration_value = candidate_result.get("orchestration_run_id")

    if isinstance(orchestration_value, str) and orchestration_value.strip():
        return orchestration_value

    return None


def get_candidate_mlflow_run_id(
    comparison: dict[str, Any],
    candidate_result: dict[str, Any],
) -> str | None:
    comparison_value = comparison.get("candidate_mlflow_run_id")

    if isinstance(comparison_value, str) and comparison_value.strip():
        return comparison_value

    candidate_value = candidate_result.get("mlflow_run_id")

    if isinstance(candidate_value, str) and candidate_value.strip():
        return candidate_value

    return None


def find_champion_section(decision_config: dict[str, Any]) -> dict[str, Any]:
    classification_candidate = decision_config.get("classification_candidate")

    if isinstance(classification_candidate, dict):
        return classification_candidate

    champion = decision_config.get("champion")

    if isinstance(champion, dict):
        return champion

    return decision_config


def archive_local_file(
    source_path: Path,
    archive_dir: Path,
    timestamp: str,
    dry_run: bool,
) -> str | None:
    if not source_path.exists():
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{timestamp}_{source_path.name}"

    if not dry_run:
        shutil.copy2(source_path, archive_path)

    return str(archive_path)


def promote_local_checkpoint(
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    archive_dir: Path,
    timestamp: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not candidate_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_checkpoint}")

    archived_previous_checkpoint = archive_local_file(
        source_path=champion_checkpoint,
        archive_dir=archive_dir,
        timestamp=timestamp,
        dry_run=dry_run,
    )

    if not dry_run:
        champion_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_checkpoint, champion_checkpoint)

    return {
        "candidate_checkpoint": str(candidate_checkpoint),
        "local_champion_checkpoint": str(champion_checkpoint),
        "archived_previous_checkpoint": archived_previous_checkpoint,
        "local_checkpoint_copied": not dry_run,
    }


def load_boto3_client() -> Any:
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 import failed. S3 promotion requires boto3.") from exc

    return boto3.client("s3")


def s3_object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def promote_s3_checkpoint(
    candidate_checkpoint: Path,
    bucket: str,
    champion_key: str,
    archive_prefix: str,
    timestamp: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not candidate_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_checkpoint}")

    client = load_boto3_client()
    archive_key = f"{archive_prefix.rstrip('/')}/{timestamp}_{Path(champion_key).name}"
    previous_exists = s3_object_exists(client, bucket, champion_key)

    if not dry_run and previous_exists:
        client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": champion_key},
            Key=archive_key,
        )

    if not dry_run:
        client.upload_file(str(candidate_checkpoint), bucket, champion_key)

    return {
        "s3_bucket": bucket,
        "s3_champion_key": champion_key,
        "s3_champion_uri": f"s3://{bucket}/{champion_key}",
        "previous_s3_champion_existed": previous_exists,
        "archived_previous_s3_key": archive_key if previous_exists else None,
        "archived_previous_s3_uri": f"s3://{bucket}/{archive_key}" if previous_exists else None,
        "s3_checkpoint_uploaded": not dry_run,
    }


def upload_s3_file(
    local_path: Path,
    bucket: str,
    key: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found for S3 upload: {local_path}")

    if not dry_run:
        client = load_boto3_client()
        client.upload_file(str(local_path), bucket, key)

    return {
        "local_path": str(local_path),
        "s3_bucket": bucket,
        "s3_key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        "uploaded": not dry_run,
    }


def update_decision_config(
    decision_config: dict[str, Any],
    comparison: dict[str, Any],
    candidate_result: dict[str, Any],
    candidate_checkpoint: Path,
    champion_s3_uri: str,
    champion_s3_key: str,
    candidate_training_run_id: str | None,
    candidate_mlflow_run_id: str | None,
    promotion_report_path: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    candidate_metrics = comparison.get("candidate_metrics")

    if not isinstance(candidate_metrics, dict):
        candidate_metrics = {}

    champion_section = find_champion_section(decision_config)

    champion_section["model_uri"] = champion_s3_uri
    champion_section["s3_checkpoint_uri"] = champion_s3_uri
    champion_section["runtime_checkpoint_uri"] = champion_s3_uri
    champion_section["artifact_path"] = champion_s3_key
    champion_section["runtime_model_source"] = "s3_direct_pytorch_checkpoint"
    champion_section["flavor"] = "pytorch_checkpoint"
    champion_section["promoted_at_utc"] = generated_at_utc
    champion_section["promotion_source"] = "automatic_retraining_pipeline"
    champion_section["promotion_report_path"] = promotion_report_path
    champion_section["candidate_checkpoint_promoted_from"] = str(candidate_checkpoint)

    if candidate_training_run_id:
        champion_section["training_run_id"] = candidate_training_run_id
        champion_section["run_id"] = candidate_training_run_id

    if candidate_mlflow_run_id:
        champion_section["mlflow_run_id"] = candidate_mlflow_run_id

    metric_mapping = {
        "business_score_classification": "test_business_score_classification",
        "alert_24h_f1": "f1_alert_24h",
        "alert_24h_recall": "recall_alert_24h",
        "alert_24h_precision": "precision_alert_24h",
        "class_5_f1": "test_class_5_f1",
    }

    for source_key, target_key in metric_mapping.items():
        value = candidate_metrics.get(source_key)

        if value is not None:
            champion_section[target_key] = value
            champion_section[source_key] = value

    metrics = champion_section.get("metrics")

    if not isinstance(metrics, dict):
        metrics = {}

    test_metrics = metrics.get("test")

    if not isinstance(test_metrics, dict):
        test_metrics = {}

    for key, value in candidate_metrics.items():
        if value is not None:
            test_metrics[key] = value

    metrics["test"] = test_metrics
    champion_section["metrics"] = metrics

    decision_config["last_automatic_promotion"] = {
        "promoted_at_utc": generated_at_utc,
        "candidate_training_run_id": candidate_training_run_id,
        "candidate_mlflow_run_id": candidate_mlflow_run_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "champion_model_uri": champion_s3_uri,
        "promotion_report_path": promotion_report_path,
    }

    return decision_config


def promote_candidate_if_approved(args: argparse.Namespace) -> dict[str, Any]:
    generated_at_utc = utc_now_iso()
    timestamp = timestamp_utc()
    project_root = Path(args.project_root)

    comparison_path = resolve_path(args.comparison_json, project_root)
    candidate_result_path = resolve_path(args.candidate_result, project_root)
    decision_config_path = resolve_path(args.decision_config, project_root)
    output_json_path = resolve_path(args.output_json, project_root)

    comparison = read_json(comparison_path)

    if comparison.get("status") != "success":
        payload = {
            "status": "success",
            "action": "promotion_skipped",
            "reason": "comparison_status_is_not_success",
            "comparison_status": comparison.get("status"),
            "comparison_json": str(comparison_path),
            "generated_at_utc": generated_at_utc,
        }
        write_json(payload, output_json_path)
        return payload

    if comparison.get("decision") != "promote_candidate" or comparison.get("eligible_for_promotion") is not True:
        payload = {
            "status": "success",
            "action": "promotion_skipped",
            "reason": "candidate_not_eligible_for_promotion",
            "decision": comparison.get("decision"),
            "eligible_for_promotion": comparison.get("eligible_for_promotion"),
            "decision_reason": comparison.get("decision_reason"),
            "comparison_json": str(comparison_path),
            "generated_at_utc": generated_at_utc,
        }
        write_json(payload, output_json_path)
        return payload

    candidate_result = read_json(candidate_result_path)
    decision_config = read_json(decision_config_path)

    candidate_checkpoint = resolve_path(
        extract_candidate_checkpoint(candidate_result),
        project_root,
    )
    local_champion_checkpoint = resolve_path(args.local_champion_checkpoint, project_root)
    local_champion_archive_dir = resolve_path(args.local_champion_archive_dir, project_root)
    decision_archive_dir = resolve_path(args.decision_archive_dir, project_root)

    candidate_training_run_id = get_candidate_training_run_id(comparison, candidate_result)
    candidate_mlflow_run_id = get_candidate_mlflow_run_id(comparison, candidate_result)

    local_checkpoint_result = promote_local_checkpoint(
        candidate_checkpoint=candidate_checkpoint,
        champion_checkpoint=local_champion_checkpoint,
        archive_dir=local_champion_archive_dir,
        timestamp=timestamp,
        dry_run=args.dry_run,
    )

    if args.skip_s3:
        s3_result = {
            "s3_skipped": True,
            "s3_champion_uri": f"s3://{args.s3_bucket}/{args.s3_champion_key}",
            "s3_champion_key": args.s3_champion_key,
        }
    else:
        s3_result = promote_s3_checkpoint(
            candidate_checkpoint=candidate_checkpoint,
            bucket=args.s3_bucket,
            champion_key=args.s3_champion_key,
            archive_prefix=args.s3_champion_archive_prefix,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )

    archived_previous_decision_config = archive_local_file(
        source_path=decision_config_path,
        archive_dir=decision_archive_dir,
        timestamp=timestamp,
        dry_run=args.dry_run,
    )

    champion_s3_uri = f"s3://{args.s3_bucket}/{args.s3_champion_key}"
    updated_decision_config = update_decision_config(
        decision_config=decision_config,
        comparison=comparison,
        candidate_result=candidate_result,
        candidate_checkpoint=candidate_checkpoint,
        champion_s3_uri=champion_s3_uri,
        champion_s3_key=args.s3_champion_key,
        candidate_training_run_id=candidate_training_run_id,
        candidate_mlflow_run_id=candidate_mlflow_run_id,
        promotion_report_path=str(output_json_path),
        generated_at_utc=generated_at_utc,
    )

    if not args.dry_run:
        write_json(updated_decision_config, decision_config_path)

    s3_decision_upload = None

    if not args.skip_s3 and args.s3_decision_key:
        s3_decision_upload = upload_s3_file(
            local_path=decision_config_path,
            bucket=args.s3_bucket,
            key=args.s3_decision_key,
            dry_run=args.dry_run,
        )

    payload = {
        "status": "success",
        "action": "candidate_promoted" if not args.dry_run else "candidate_promotion_dry_run",
        "decision": comparison.get("decision"),
        "eligible_for_promotion": comparison.get("eligible_for_promotion"),
        "candidate_training_run_id": candidate_training_run_id,
        "candidate_mlflow_run_id": candidate_mlflow_run_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "comparison_json": str(comparison_path),
        "candidate_result_json": str(candidate_result_path),
        "decision_config_path": str(decision_config_path),
        "archived_previous_decision_config": archived_previous_decision_config,
        "local_checkpoint_promotion": local_checkpoint_result,
        "s3_checkpoint_promotion": s3_result,
        "s3_decision_config_upload": s3_decision_upload,
        "dry_run": args.dry_run,
        "generated_at_utc": generated_at_utc,
    }

    write_json(payload, output_json_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a retrained candidate checkpoint if the comparison gate approved it."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--comparison-json", default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--candidate-result", default=DEFAULT_CANDIDATE_RESULT)
    parser.add_argument("--decision-config", default=DEFAULT_DECISION_CONFIG)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--local-champion-checkpoint", default=DEFAULT_LOCAL_CHAMPION_CHECKPOINT)
    parser.add_argument("--local-champion-archive-dir", default=DEFAULT_LOCAL_CHAMPION_ARCHIVE_DIR)
    parser.add_argument("--decision-archive-dir", default=DEFAULT_DECISION_ARCHIVE_DIR)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--s3-champion-key", default=DEFAULT_S3_CHAMPION_KEY)
    parser.add_argument("--s3-champion-archive-prefix", default=DEFAULT_S3_CHAMPION_ARCHIVE_PREFIX)
    parser.add_argument("--s3-decision-key", default=DEFAULT_S3_DECISION_KEY)
    parser.add_argument("--skip-s3", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = promote_candidate_if_approved(args)

    if args.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
