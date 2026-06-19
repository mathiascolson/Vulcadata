from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.inference.prepare_latest_batch import (
    default_metadata_path_for_npz,
    prepare_latest_batch,
)


def create_source_npz(path: Path) -> np.ndarray:
    x_test = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)

    np.savez_compressed(
        path,
        X_train=np.zeros((2, 4, 3), dtype=np.float32),
        X_test=x_test,
        y_test=np.arange(5, dtype=np.int64),
        test_times=np.array(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:20:00Z",
                "2026-01-01T00:30:00Z",
                "2026-01-01T00:40:00Z",
            ]
        ),
        test_eruption_ids=np.array(
            [
                "eruption_a",
                "eruption_a",
                "eruption_b",
                "eruption_b",
                "eruption_b",
            ]
        ),
        feature_names=np.array(["f0", "f1", "f2"]),
    )

    return x_test


def test_prepare_latest_batch_writes_last_sequence_with_standard_x_key(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    expected_x_test = create_source_npz(source_npz)

    output_npz = tmp_path / "latest_batch.npz"

    result = prepare_latest_batch(
        source_npz=source_npz,
        output_npz=output_npz,
        source_array_key="X_test",
        batch_size=1,
        expected_seq_len=4,
        expected_n_features=3,
    )

    with np.load(output_npz, allow_pickle=False) as output:
        assert output.files == ["X"]
        assert output["X"].shape == (1, 4, 3)
        np.testing.assert_array_equal(output["X"], expected_x_test[-1:])

    assert result.source_array_key == "X_test"
    assert result.output_array_key == "X"
    assert result.batch_size == 1
    assert result.source_batch_size == 5
    assert result.selected_start_index == 4
    assert result.selected_end_index == 4
    assert result.selected_times == ["2026-01-01T00:40:00Z"]
    assert result.selected_eruption_ids == ["eruption_b"]
    assert result.feature_names_count == 3

    metadata_path = default_metadata_path_for_npz(output_npz)
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["status"] == "success"
    assert metadata["source"]["array_key"] == "X_test"
    assert metadata["output"]["array_key"] == "X"
    assert metadata["selection"]["selected_start_index"] == 4
    assert metadata["selection"]["selected_end_index"] == 4
    assert metadata["selection"]["selected_times"] == ["2026-01-01T00:40:00Z"]
    assert metadata["selection"]["selected_eruption_ids"] == ["eruption_b"]
    assert metadata["features"]["feature_names_count"] == 3


def test_prepare_latest_batch_can_keep_multiple_latest_sequences(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    expected_x_test = create_source_npz(source_npz)

    output_npz = tmp_path / "latest_batch.npz"

    result = prepare_latest_batch(
        source_npz=source_npz,
        output_npz=output_npz,
        source_array_key="X_test",
        batch_size=2,
        expected_seq_len=4,
        expected_n_features=3,
    )

    with np.load(output_npz, allow_pickle=False) as output:
        assert output["X"].shape == (2, 4, 3)
        np.testing.assert_array_equal(output["X"], expected_x_test[-2:])

    assert result.selected_start_index == 3
    assert result.selected_end_index == 4
    assert result.selected_times == [
        "2026-01-01T00:30:00Z",
        "2026-01-01T00:40:00Z",
    ]
    assert result.selected_eruption_ids == ["eruption_b", "eruption_b"]


def test_prepare_latest_batch_can_infer_single_3d_array_when_unambiguous(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "single_array.npz"
    sequences = np.ones((3, 4, 3), dtype=np.float32)

    np.savez_compressed(
        source_npz,
        sequences=sequences,
        times=np.array(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:20:00Z",
            ]
        ),
    )

    output_npz = tmp_path / "latest_batch.npz"

    result = prepare_latest_batch(
        source_npz=source_npz,
        output_npz=output_npz,
        batch_size=1,
        expected_seq_len=4,
        expected_n_features=3,
    )

    assert result.source_array_key == "sequences"

    with np.load(output_npz, allow_pickle=False) as output:
        np.testing.assert_array_equal(output["X"], sequences[-1:])


def test_prepare_latest_batch_rejects_missing_source_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Source NPZ not found"):
        prepare_latest_batch(
            source_npz=tmp_path / "missing.npz",
            output_npz=tmp_path / "latest_batch.npz",
        )


def test_prepare_latest_batch_rejects_missing_source_key(tmp_path: Path) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    create_source_npz(source_npz)

    with pytest.raises(KeyError, match="Requested source array key"):
        prepare_latest_batch(
            source_npz=source_npz,
            output_npz=tmp_path / "latest_batch.npz",
            source_array_key="X_missing",
        )


def test_prepare_latest_batch_rejects_ambiguous_multiple_3d_arrays(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    create_source_npz(source_npz)

    with pytest.raises(ValueError, match="Multiple 3D arrays"):
        prepare_latest_batch(
            source_npz=source_npz,
            output_npz=tmp_path / "latest_batch.npz",
        )


def test_prepare_latest_batch_rejects_too_large_batch_size(tmp_path: Path) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    create_source_npz(source_npz)

    with pytest.raises(ValueError, match="exceeds source batch size"):
        prepare_latest_batch(
            source_npz=source_npz,
            output_npz=tmp_path / "latest_batch.npz",
            source_array_key="X_test",
            batch_size=6,
        )


def test_prepare_latest_batch_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    create_source_npz(source_npz)

    output_npz = tmp_path / "latest_batch.npz"
    output_npz.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_latest_batch(
            source_npz=source_npz,
            output_npz=output_npz,
            source_array_key="X_test",
        )


def test_prepare_latest_batch_overwrites_existing_output_when_requested(
    tmp_path: Path,
) -> None:
    source_npz = tmp_path / "volcano_multi.npz"
    expected_x_test = create_source_npz(source_npz)

    output_npz = tmp_path / "latest_batch.npz"
    output_npz.write_text("existing", encoding="utf-8")

    prepare_latest_batch(
        source_npz=source_npz,
        output_npz=output_npz,
        source_array_key="X_test",
        overwrite=True,
    )

    with np.load(output_npz, allow_pickle=False) as output:
        np.testing.assert_array_equal(output["X"], expected_x_test[-1:])