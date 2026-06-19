from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_READY_DIR = "data/retraining/ready"
DEFAULT_MANIFEST_PATH = "data/retraining/manifests/preprocessed_files_manifest.json"
DEFAULT_OUTPUT_JSON = "reports/retraining/new_preprocessed_files_detection.json"
DEFAULT_MIN_NEW_FILES_FOR_RETRAINING = 1
SUPPORTED_EXTENSIONS = (".npz",)


class NewPreprocessedFilesDetectionError(RuntimeError):
    """Raised when detection of new local preprocessed files fails."""


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_local_path(path: str | Path) -> str:
    return str(Path(path).as_posix())


def default_manifest() -> dict[str, Any]:
    now = utc_now_iso()

    return {
        "version": 1,
        "storage": "local",
        "created_at_utc": now,
        "updated_at_utc": now,
        "files": {},
    }


def list_local_preprocessed_files(
    *,
    ready_dir: str | Path,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    ready_dir = Path(ready_dir)

    if not ready_dir.exists():
        raise NewPreprocessedFilesDetectionError(
            f"Ready directory not found: {ready_dir}"
        )

    if not ready_dir.is_dir():
        raise NewPreprocessedFilesDetectionError(
            f"Ready path must be a directory: {ready_dir}"
        )

    pattern = "**/*" if recursive else "*"

    detected_files: list[dict[str, Any]] = []

    for path in ready_dir.glob(pattern):
        if not path.is_file():
            continue

        if not path.name.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        stat = path.stat()
        absolute_path = path.resolve()

        detected_files.append(
            {
                "path": normalize_local_path(path),
                "absolute_path": normalize_local_path(absolute_path),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": int(stat.st_size),
                "modified_at_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                )
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    detected_files.sort(key=lambda item: item["path"])

    return detected_files


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return default_manifest()

    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise NewPreprocessedFilesDetectionError(
            "Manifest must be a JSON object."
        )

    if "files" not in payload or not isinstance(payload["files"], dict):
        raise NewPreprocessedFilesDetectionError(
            "Manifest must contain a 'files' object."
        )

    return payload


def write_json_local(payload: dict[str, Any], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_manifest(manifest: dict[str, Any], manifest_path: str | Path) -> None:
    write_json_local(manifest, manifest_path)


def has_file_changed(
    *,
    existing_record: dict[str, Any],
    detected_file: dict[str, Any],
) -> bool:
    return (
        existing_record.get("size_bytes") != detected_file.get("size_bytes")
        or existing_record.get("mtime_ns") != detected_file.get("mtime_ns")
        or existing_record.get("absolute_path") != detected_file.get("absolute_path")
    )


def update_manifest_with_detected_files(
    *,
    manifest: dict[str, Any],
    detected_files: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    now = utc_now_iso()

    manifest["storage"] = "local"
    manifest_files = manifest.setdefault("files", {})

    new_files: list[dict[str, Any]] = []
    pending_files: list[dict[str, Any]] = []

    for detected_file in detected_files:
        file_key = detected_file["path"]
        existing_record = manifest_files.get(file_key)

        if existing_record is None:
            record = {
                **detected_file,
                "status": "detected",
                "first_seen_at_utc": now,
                "last_seen_at_utc": now,
                "integrated_at_utc": None,
                "last_training_run_id": None,
            }

            manifest_files[file_key] = record
            new_files.append(record)
            pending_files.append(record)
            continue

        changed = has_file_changed(
            existing_record=existing_record,
            detected_file=detected_file,
        )

        existing_record.update(
            {
                **detected_file,
                "last_seen_at_utc": now,
            }
        )

        if changed and existing_record.get("status") == "integrated":
            existing_record["status"] = "detected"
            existing_record["integrated_at_utc"] = None
            existing_record["last_training_run_id"] = None

        if existing_record.get("status") != "integrated":
            pending_files.append(existing_record)

    manifest["updated_at_utc"] = now

    return manifest, new_files, pending_files


def detect_new_preprocessed_files(
    *,
    ready_dir: str | Path,
    manifest_path: str | Path,
    min_new_files_for_retraining: int,
    update_manifest: bool,
    recursive: bool = False,
) -> dict[str, Any]:
    if min_new_files_for_retraining < 1:
        raise NewPreprocessedFilesDetectionError(
            "min_new_files_for_retraining must be >= 1."
        )

    detected_files = list_local_preprocessed_files(
        ready_dir=ready_dir,
        recursive=recursive,
    )

    manifest = read_manifest(manifest_path)

    updated_manifest, new_files, pending_files = update_manifest_with_detected_files(
        manifest=manifest,
        detected_files=detected_files,
    )

    should_retrain = len(pending_files) >= min_new_files_for_retraining

    if update_manifest:
        write_manifest(updated_manifest, manifest_path)
    
    new_file_paths = [file_record["path"] for file_record in new_files]
    pending_file_paths = [file_record["path"] for file_record in pending_files]

    return {
        "status": "success",
        "storage": "local",
        "ready_dir": normalize_local_path(ready_dir),   
        "manifest_path": normalize_local_path(manifest_path),
        "detected_files_count": len(detected_files),
        "new_files_count": len(new_files),
        "pending_files_count": len(pending_files),
        "min_new_files_for_retraining": min_new_files_for_retraining,
        "should_retrain": should_retrain,
        "update_manifest": update_manifest,
        "recursive": recursive,
        "new_file_paths": new_file_paths,
        "pending_file_paths": pending_file_paths,
        "files_to_process": pending_files,
        "generated_at_utc": utc_now_iso(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect new local preprocessed Vulcadata files for retraining."
    )

    parser.add_argument(
        "--ready-dir",
        default=os.getenv("VULCADATA_RETRAINING_READY_DIR", DEFAULT_READY_DIR),
        help="Local directory containing preprocessed .npz files ready for retraining.",
    )

    parser.add_argument(
        "--manifest-path",
        default=os.getenv(
            "VULCADATA_PREPROCESSED_MANIFEST_PATH",
            DEFAULT_MANIFEST_PATH,
        ),
        help="Local JSON manifest tracking detected and integrated files.",
    )

    parser.add_argument(
        "--min-new-files-for-retraining",
        type=int,
        default=int(
            os.getenv(
                "VULCADATA_MIN_NEW_FILES_FOR_RETRAINING",
                str(DEFAULT_MIN_NEW_FILES_FOR_RETRAINING),
            )
        ),
    )

    parser.add_argument(
        "--output-json",
        default=os.getenv(
            "VULCADATA_NEW_FILES_DETECTION_OUTPUT_JSON",
            DEFAULT_OUTPUT_JSON,
        ),
    )

    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write the updated local manifest.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan ready-dir for .npz files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = detect_new_preprocessed_files(
        ready_dir=args.ready_dir,
        manifest_path=args.manifest_path,
        min_new_files_for_retraining=args.min_new_files_for_retraining,
        update_manifest=args.update_manifest,
        recursive=args.recursive,
    )

    write_json_local(result, args.output_json)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()