from __future__ import annotations

import json

import numpy as np
import pytest

from src.inference.validate_latest_batch_with_gx import (
    LatestBatchValidationError,
    build_latest_batch_profile,
    validate_latest_batch_with_gx,
    write_json,
)


def create_valid_array() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(1, 120, 992)).astype(np.float32)


def test_build_latest_batch_profile_has_expected_columns() -> None:
    array = create_valid_array()

    profile = build_latest_batch_profile(array)

    assert profile.shape[0] == 1
    assert profile.loc[0, "sequence_index"] == 0
    assert profile.loc[0, "seq_len"] == 120
    assert profile.loc[0, "n_features"] == 992
    assert profile.loc[0, "nan_count"] == 0
    assert profile.loc[0, "inf_count"] == 0
    assert profile.loc[0, "finite_ratio"] == 1.0


def test_validate_latest_batch_with_gx_accepts_valid_npz(tmp_path) -> None:
    npz_path = tmp_path / "latest_batch.npz"
    np.savez_compressed(npz_path, X=create_valid_array())

    result = validate_latest_batch_with_gx(
        npz_path=npz_path,
        array_key="X",
        expected_batch_size=1,
        expected_seq_len=120,
        expected_n_features=992,
    )

    assert result["status"] == "success"
    assert result["shape"] == [1, 120, 992]
    assert result["dtype"] == "float32"
    assert result["gx_success"] is True
    assert result["gx_result"]["success"] is True
    assert result["gx_result"]["statistics"]["evaluated_expectations"] == 16


def test_validate_latest_batch_with_gx_rejects_missing_key(tmp_path) -> None:
    npz_path = tmp_path / "latest_batch.npz"
    np.savez_compressed(npz_path, Y=create_valid_array())

    with pytest.raises(LatestBatchValidationError, match="Array key 'X' not found"):
        validate_latest_batch_with_gx(
            npz_path=npz_path,
            array_key="X",
            expected_batch_size=1,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_validate_latest_batch_with_gx_rejects_wrong_shape(tmp_path) -> None:
    npz_path = tmp_path / "latest_batch.npz"

    invalid_array = np.random.default_rng(42).normal(
        size=(1, 119, 992)
    ).astype(np.float32)

    np.savez_compressed(npz_path, X=invalid_array)

    with pytest.raises(LatestBatchValidationError, match="Invalid array shape"):
        validate_latest_batch_with_gx(
            npz_path=npz_path,
            array_key="X",
            expected_batch_size=1,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_validate_latest_batch_with_gx_rejects_wrong_batch_size(tmp_path) -> None:
    npz_path = tmp_path / "latest_batch.npz"

    invalid_array = np.random.default_rng(42).normal(
        size=(2, 120, 992)
    ).astype(np.float32)

    np.savez_compressed(npz_path, X=invalid_array)

    with pytest.raises(LatestBatchValidationError, match="Invalid array shape"):
        validate_latest_batch_with_gx(
            npz_path=npz_path,
            array_key="X",
            expected_batch_size=1,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_validate_latest_batch_with_gx_rejects_nan_values(tmp_path) -> None:
    array = create_valid_array()
    array[0, 0, 0] = np.nan

    npz_path = tmp_path / "latest_batch.npz"
    np.savez_compressed(npz_path, X=array)

    with pytest.raises(LatestBatchValidationError, match="NaN or infinite"):
        validate_latest_batch_with_gx(
            npz_path=npz_path,
            array_key="X",
            expected_batch_size=1,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_validate_latest_batch_with_gx_rejects_infinite_values(tmp_path) -> None:
    array = create_valid_array()
    array[0, 0, 0] = np.inf

    npz_path = tmp_path / "latest_batch.npz"
    np.savez_compressed(npz_path, X=array)

    with pytest.raises(LatestBatchValidationError, match="NaN or infinite"):
        validate_latest_batch_with_gx(
            npz_path=npz_path,
            array_key="X",
            expected_batch_size=1,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_write_json_creates_parent_directory(tmp_path) -> None:
    output_json = tmp_path / "reports" / "validation" / "result.json"

    write_json({"status": "success"}, output_json)

    assert output_json.exists()

    with output_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data == {"status": "success"}