from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DETECTION_REPORT_PATH = "reports/retraining/new_preprocessed_files_detection.json"
DEFAULT_MANIFEST_PATH = "data/retraining/manifests/preprocessed_files_manifest.json"
DEFAULT_OUTPUT_JSON = "reports/retraining/mark_preprocessed_files_integrated.json"


class PreprocessedFilesIntegrationError(RuntimeError):
    """Raised when preprocessed files cannot be marked as integrated."""


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_local_path(path: str | Path) -> str:
    return str(Path(path).as_posix())


def read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise PreprocessedFilesIntegrationError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise PreprocessedFilesIntegrationError(
            f"JSON file must contain an object: {path}"
        )

    return payload


def write_json_local(payload: dict[str, Any], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def validate_training_run_id(training_run_id: str) -> str:
    training_run_id = training_run_id.strip()

    if not training_run_id:
        raise PreprocessedFilesIntegrationError(
            "training_run_id must be a non-empty string."
        )

    return training_run_id


def extract_files_to_process(
    detection_report: dict[str, Any],
) -> list[dict[str, Any]]:
    if detection_report.get("status") != "success":
        raise PreprocessedFilesIntegrationError(
            "Detection report status must be 'success'."
        )

    files_to_process = detection_report.get("files_to_process")

    if files_to_process is None:
        files_to_process = detection_report.get("pending_files", [])

    if not isinstance(files_to_process, list):
        raise PreprocessedFilesIntegrationError(
            "Detection report field 'files_to_process' must be a list."
        )

    for index, file_record in enumerate(files_to_process):
        if not isinstance(file_record, dict):
            raise PreprocessedFilesIntegrationError(
                f"files_to_process[{index}] must be an object."
            )

        if "absolute_path" not in file_record:
            raise PreprocessedFilesIntegrationError(
                f"files_to_process[{index}] must contain 'absolute_path'."
            )

    return files_to_process


def get_manifest_files(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")

    if not isinstance(files, dict):
        raise PreprocessedFilesIntegrationError(
            "Manifest must contain a 'files' object."
        )

    return files

def resolve_manifest_file_key(
    *,
    manifest_files: dict[str, Any],
    file_record: dict[str, Any],
) -> str:
    relative_path = file_record.get("path")
    absolute_path = file_record.get("absolute_path")

    if relative_path and relative_path in manifest_files:
        return relative_path

    if absolute_path and absolute_path in manifest_files:
        return absolute_path

    for manifest_key, manifest_record in manifest_files.items():
        if not isinstance(manifest_record, dict):
            continue

        if relative_path and manifest_record.get("path") == relative_path:
            return manifest_key

        if absolute_path and manifest_record.get("absolute_path") == absolute_path:
            return manifest_key

    raise PreprocessedFilesIntegrationError(
        "File from detection report is missing from manifest: "
        f"path={relative_path}, absolute_path={absolute_path}"
    )

def mark_preprocessed_files_as_integrated(
    *,
    detection_report_path: str | Path,
    manifest_path: str | Path,
    training_run_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    training_run_id = validate_training_run_id(training_run_id)

    detection_report = read_json_object(detection_report_path)
    manifest = read_json_object(manifest_path)

    files_to_process = extract_files_to_process(detection_report)
    manifest_files = get_manifest_files(manifest)

    now = utc_now_iso()

    marked_files: list[dict[str, Any]] = []
    already_integrated_files: list[dict[str, Any]] = []

    for file_record in files_to_process:
        manifest_file_key = resolve_manifest_file_key(
            manifest_files=manifest_files,
            file_record=file_record,
        )

        manifest_record = manifest_files[manifest_file_key]

        current_status = manifest_record.get("status")
        previous_training_run_id = manifest_record.get("last_training_run_id")

        if current_status == "integrated":
            if previous_training_run_id == training_run_id:
                already_integrated_files.append(manifest_record)
                continue

            raise PreprocessedFilesIntegrationError(
                "File is already integrated with another training run. "
                f"file={absolute_path}, "
                f"existing_run_id={previous_training_run_id}, "
                f"new_run_id={training_run_id}"
            )

        updated_record = {
            **manifest_record,
            "status": "integrated",
            "integrated_at_utc": now,
            "last_training_run_id": training_run_id,
            "last_integration_report_path": normalize_local_path(detection_report_path),
        }

        manifest_files[manifest_file_key] = updated_record
        marked_files.append(updated_record)

    manifest["updated_at_utc"] = now

    if not dry_run:
        write_json_local(manifest, manifest_path)

    return {
        "status": "success",
        "dry_run": dry_run,
        "manifest_path": normalize_local_path(manifest_path),
        "detection_report_path": normalize_local_path(detection_report_path),
        "training_run_id": training_run_id,
        "files_to_process_count": len(files_to_process),
        "marked_files_count": len(marked_files),
        "already_integrated_files_count": len(already_integrated_files),
        "marked_file_paths": [record["path"] for record in marked_files],
        "already_integrated_file_paths": [
            record["path"] for record in already_integrated_files
        ],
        "marked_files": marked_files,
        "already_integrated_files": already_integrated_files,
        "generated_at_utc": now,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark local preprocessed files as integrated after successful retraining."
    )

    parser.add_argument(
        "--detection-report",
        default=os.getenv(
            "VULCADATA_NEW_FILES_DETECTION_OUTPUT_JSON",
            DEFAULT_DETECTION_REPORT_PATH,
        ),
    )

    parser.add_argument(
        "--manifest-path",
        default=os.getenv(
            "VULCADATA_PREPROCESSED_MANIFEST_PATH",
            DEFAULT_MANIFEST_PATH,
        ),
    )

    parser.add_argument(
        "--training-run-id",
        required=True,
        help="MLflow run id or orchestration run id associated with successful retraining.",
    )

    parser.add_argument(
        "--output-json",
        default=os.getenv(
            "VULCADATA_MARK_INTEGRATED_OUTPUT_JSON",
            DEFAULT_OUTPUT_JSON,
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the changes without writing the manifest.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = mark_preprocessed_files_as_integrated(
        detection_report_path=args.detection_report,
        manifest_path=args.manifest_path,
        training_run_id=args.training_run_id,
        dry_run=args.dry_run,
    )

    write_json_local(result, args.output_json)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()