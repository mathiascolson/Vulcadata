from __future__ import annotations

import json

import numpy as np
import pytest

from src.retraining.validate_retraining_dataset import (
    RetrainingDatasetValidationError,
    build_retraining_dataset_profile,
    validate_retraining_dataset,
    write_json,
)


def create_valid_payload() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "X_train": rng.normal(size=(3, 120, 992)).astype(np.float32),
        "y_train": np.array([0, 1, 2], dtype=np.int64),
        "X_val": rng.normal(size=(2, 120, 992)).astype(np.float32),
        "y_val": np.array([3, 4], dtype=np.int64),
        "X_test": rng.normal(size=(2, 120, 992)).astype(np.float32),
        "y_test": np.array([4, 5], dtype=np.int64),
    }


def write_npz(path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def test_build_retraining_dataset_profile_has_expected_rows_and_columns() -> None:
    payload = create_valid_payload()

    profile = build_retraining_dataset_profile(payload)

    assert list(profile["split_name"]) == ["train", "val", "test"]
    assert profile.shape[0] == 3
    assert set(profile["seq_len"]) == {120}
    assert set(profile["n_features"]) == {992}
    assert profile["x_nan_count"].sum() == 0
    assert profile["x_inf_count"].sum() == 0
    assert profile["y_nan_count"].sum() == 0
    assert profile["y_inf_count"].sum() == 0


def test_validate_retraining_dataset_accepts_valid_npz(tmp_path) -> None:
    npz_path = tmp_path / "candidate_training_dataset.npz"
    write_npz(npz_path, create_valid_payload())

    result = validate_retraining_dataset(
        npz_path=npz_path,
        expected_seq_len=120,
        expected_n_features=992,
        n_classes=6,
    )

    assert result["status"] == "success"
    assert result["gx_success"] is True
    assert result["gx_result"]["success"] is True
    assert len(result["split_profiles"]) == 3
    assert result["split_shapes"]["train"]["X_shape"] == [3, 120, 992]
    assert result["split_shapes"]["val"]["X_shape"] == [2, 120, 992]
    assert result["split_shapes"]["test"]["X_shape"] == [2, 120, 992]


def test_validate_retraining_dataset_rejects_missing_key(tmp_path) -> None:
    payload = create_valid_payload()
    payload.pop("y_test")
    npz_path = tmp_path / "candidate_training_dataset.npz"
    write_npz(npz_path, payload)

    with pytest.raises(RetrainingDatasetValidationError, match="Missing required keys"):
        validate_retraining_dataset(
            npz_path=npz_path,
            expected_seq_len=120,
            expected_n_features=992,
            n_classes=6,
        )


def test_validate_retraining_dataset_rejects_wrong_shape(tmp_path) -> None:
    payload = create_valid_payload()
    payload["X_train"] = np.random.default_rng(42).normal(size=(3, 119, 992)).astype(np.float32)
    npz_path = tmp_path / "candidate_training_dataset.npz"
    write_npz(npz_path, payload)

    with pytest.raises(RetrainingDatasetValidationError, match="Invalid X_train shape"):
        validate_retraining_dataset(
            npz_path=npz_path,
            expected_seq_len=120,
            expected_n_features=992,
            n_classes=6,
        )


def test_validate_retraining_dataset_rejects_mismatched_x_y_rows(tmp_path) -> None:
    payload = create_valid_payload()
    payload["y_train"] = np.array([0, 1], dtype=np.int64)
    npz_path = tmp_path / "candidate_training_dataset.npz"
    write_npz(npz_path, payload)

    with pytest.raises(RetrainingDatasetValidationError, match="row counts do not match"):
        validate_retraining_dataset(
            npz_path=npz_path,
            expected_seq_len=120,
            expected_n_features=992,
            n_classes=6,
        )


def test_validate_retraining_dataset_rejects_nan_values(tmp_path) -> None:
    payload = create_valid_payload()
    payload["X_train"][0, 0, 0] = np.nan
    npz_path = tmp_path / "candidate_training_dataset.npz"
    write_npz(npz_path, payload)

    with pytest.raises(RetrainingDatasetValidationError, match="NaN or infinite"):
        validate_retraining_dataset(
            npz_path=npz_path,
            expected_seq_len=120,
            expected_n_features=992,
            n_classes=6,
        )


def test_validate_retraining_dataset_rejects_invalid_labels(tmp_path) -> None:
    payload = create_valid_payload()
    payload["y_val"] = np.array([3, 6], dtype=np.int64)
    npz_path = tmp_path / "candidate_training_dataset.npz"
    write_npz(npz_path, payload)

    with pytest.raises(RetrainingDatasetValidationError, match="labels must be between"):
        validate_retraining_dataset(
            npz_path=npz_path,
            expected_seq_len=120,
            expected_n_features=992,
            n_classes=6,
        )


def test_write_json_creates_parent_directory(tmp_path) -> None:
    output_json = tmp_path / "reports" / "validation" / "retraining_gx.json"

    write_json({"status": "success"}, output_json)

    assert output_json.exists()

    with output_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data == {"status": "success"}
