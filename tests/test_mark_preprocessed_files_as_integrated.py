from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retraining.mark_preprocessed_files_as_integrated import (
    PreprocessedFilesIntegrationError,
    mark_preprocessed_files_as_integrated,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_manifest(file_path: Path) -> dict:
    absolute_path = file_path.resolve().as_posix()

    return {
        "version": 1,
        "storage": "local",
        "created_at_utc": "2026-06-17T00:00:00Z",
        "updated_at_utc": "2026-06-17T00:00:00Z",
        "files": {
            absolute_path: {
                "path": file_path.as_posix(),
                "absolute_path": absolute_path,
                "filename": file_path.name,
                "suffix": ".npz",
                "size_bytes": 123,
                "modified_at_utc": "2026-06-17T00:00:00Z",
                "mtime_ns": 123456,
                "status": "detected",
                "first_seen_at_utc": "2026-06-17T00:00:00Z",
                "last_seen_at_utc": "2026-06-17T00:00:00Z",
                "integrated_at_utc": None,
                "last_training_run_id": None,
            }
        },
    }


def build_detection_report(file_path: Path) -> dict:
    absolute_path = file_path.resolve().as_posix()

    return {
        "status": "success",
        "storage": "local",
        "detected_files_count": 1,
        "new_files_count": 0,
        "pending_files_count": 1,
        "should_retrain": True,
        "files_to_process": [
            {
                "path": file_path.as_posix(),
                "absolute_path": absolute_path,
                "filename": file_path.name,
                "suffix": ".npz",
                "size_bytes": 123,
                "status": "detected",
            }
        ],
    }


def test_mark_preprocessed_files_as_integrated_updates_manifest(tmp_path) -> None:
    file_path = tmp_path / "ready" / "volcano_multi.npz"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"fake-npz")

    manifest_path = tmp_path / "manifest.json"
    detection_report_path = tmp_path / "detection.json"

    write_json(manifest_path, build_manifest(file_path))
    write_json(detection_report_path, build_detection_report(file_path))

    result = mark_preprocessed_files_as_integrated(
        detection_report_path=detection_report_path,
        manifest_path=manifest_path,
        training_run_id="mlflow_run_123",
        dry_run=False,
    )

    assert result["status"] == "success"
    assert result["dry_run"] is False
    assert result["marked_files_count"] == 1

    with manifest_path.open("r", encoding="utf-8") as file:
        updated_manifest = json.load(file)

    record = updated_manifest["files"][file_path.resolve().as_posix()]

    assert record["status"] == "integrated"
    assert record["integrated_at_utc"] is not None
    assert record["last_training_run_id"] == "mlflow_run_123"


def test_mark_preprocessed_files_as_integrated_dry_run_does_not_update_manifest(
    tmp_path,
) -> None:
    file_path = tmp_path / "ready" / "volcano_multi.npz"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"fake-npz")

    manifest_path = tmp_path / "manifest.json"
    detection_report_path = tmp_path / "detection.json"

    original_manifest = build_manifest(file_path)

    write_json(manifest_path, original_manifest)
    write_json(detection_report_path, build_detection_report(file_path))

    result = mark_preprocessed_files_as_integrated(
        detection_report_path=detection_report_path,
        manifest_path=manifest_path,
        training_run_id="mlflow_run_123",
        dry_run=True,
    )

    assert result["status"] == "success"
    assert result["dry_run"] is True
    assert result["marked_files_count"] == 1

    with manifest_path.open("r", encoding="utf-8") as file:
        unchanged_manifest = json.load(file)

    record = unchanged_manifest["files"][file_path.resolve().as_posix()]

    assert record["status"] == "detected"
    assert record["integrated_at_utc"] is None
    assert record["last_training_run_id"] is None


def test_mark_preprocessed_files_as_integrated_is_idempotent_for_same_run(
    tmp_path,
) -> None:
    file_path = tmp_path / "ready" / "volcano_multi.npz"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"fake-npz")

    manifest = build_manifest(file_path)
    record = manifest["files"][file_path.resolve().as_posix()]
    record["status"] = "integrated"
    record["integrated_at_utc"] = "2026-06-17T00:00:00Z"
    record["last_training_run_id"] = "mlflow_run_123"

    manifest_path = tmp_path / "manifest.json"
    detection_report_path = tmp_path / "detection.json"

    write_json(manifest_path, manifest)
    write_json(detection_report_path, build_detection_report(file_path))

    result = mark_preprocessed_files_as_integrated(
        detection_report_path=detection_report_path,
        manifest_path=manifest_path,
        training_run_id="mlflow_run_123",
        dry_run=False,
    )

    assert result["status"] == "success"
    assert result["marked_files_count"] == 0
    assert result["already_integrated_files_count"] == 1


def test_mark_preprocessed_files_as_integrated_rejects_missing_manifest_record(
    tmp_path,
) -> None:
    file_path = tmp_path / "ready" / "volcano_multi.npz"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"fake-npz")

    manifest_path = tmp_path / "manifest.json"
    detection_report_path = tmp_path / "detection.json"

    write_json(
        manifest_path,
        {
            "version": 1,
            "storage": "local",
            "files": {},
        },
    )
    write_json(detection_report_path, build_detection_report(file_path))

    with pytest.raises(
        PreprocessedFilesIntegrationError,
        match="missing from manifest",
    ):
        mark_preprocessed_files_as_integrated(
            detection_report_path=detection_report_path,
            manifest_path=manifest_path,
            training_run_id="mlflow_run_123",
            dry_run=False,
        )


def test_mark_preprocessed_files_as_integrated_rejects_empty_training_run_id(
    tmp_path,
) -> None:
    file_path = tmp_path / "ready" / "volcano_multi.npz"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"fake-npz")

    manifest_path = tmp_path / "manifest.json"
    detection_report_path = tmp_path / "detection.json"

    write_json(manifest_path, build_manifest(file_path))
    write_json(detection_report_path, build_detection_report(file_path))

    with pytest.raises(
        PreprocessedFilesIntegrationError,
        match="training_run_id",
    ):
        mark_preprocessed_files_as_integrated(
            detection_report_path=detection_report_path,
            manifest_path=manifest_path,
            training_run_id=" ",
            dry_run=False,
        )


def test_mark_preprocessed_files_as_integrated_rejects_failed_detection_report(
    tmp_path,
) -> None:
    file_path = tmp_path / "ready" / "volcano_multi.npz"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"fake-npz")

    manifest_path = tmp_path / "manifest.json"
    detection_report_path = tmp_path / "detection.json"

    failed_report = build_detection_report(file_path)
    failed_report["status"] = "failed"

    write_json(manifest_path, build_manifest(file_path))
    write_json(detection_report_path, failed_report)

    with pytest.raises(
        PreprocessedFilesIntegrationError,
        match="Detection report status",
    ):
        mark_preprocessed_files_as_integrated(
            detection_report_path=detection_report_path,
            manifest_path=manifest_path,
            training_run_id="mlflow_run_123",
            dry_run=False,
        )