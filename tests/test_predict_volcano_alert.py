from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.common.s3 import download_json
from src.inference.predict_volcano_alert import (
    InferenceInput,
    VolcanoAlertInferenceResult,
    load_npz_sequences,
    normalize_model_output,
    predict_and_write_dashboard_outputs,
    predict_class_probabilities,
    predict_volcano_alert,
    predict_volcano_alert_from_npz,
    select_probability_vector,
    validate_sequence_batch,
)


class FakePredictModel:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.received_sequences: np.ndarray | None = None

    def predict(self, sequences: np.ndarray) -> Any:
        self.received_sequences = sequences
        return self.output


class FakeS3Body:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content


class FakeS3Error(Exception):
    def __init__(self, code: str, status_code: int) -> None:
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        }
        super().__init__(code)


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[tuple[str, str], str | None] = {}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]

        if (bucket, key) not in self.objects:
            raise FakeS3Error(code="NoSuchKey", status_code=404)

        return {"ContentLength": len(self.objects[(bucket, key)])}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        body = kwargs["Body"]
        content_type = kwargs.get("ContentType")

        if isinstance(body, str):
            body = body.encode("utf-8")

        self.objects[(bucket, key)] = body
        self.content_types[(bucket, key)] = content_type

        return {"ETag": "fake-etag"}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]

        if (bucket, key) not in self.objects:
            raise FakeS3Error(code="NoSuchKey", status_code=404)

        return {"Body": FakeS3Body(self.objects[(bucket, key)])}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")

        matching_keys = sorted(
            key
            for current_bucket, key in self.objects
            if current_bucket == bucket and key.startswith(prefix)
        )

        return {
            "Contents": [{"Key": key} for key in matching_keys],
            "IsTruncated": False,
        }

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        if (Bucket, Key) not in self.objects:
            raise FakeS3Error(code="NoSuchKey", status_code=404)

        Path(Filename).write_bytes(self.objects[(Bucket, Key)])


def make_sequences(
    batch_size: int = 3,
    seq_len: int = 120,
    n_features: int = 992,
) -> np.ndarray:
    return np.ones(
        shape=(batch_size, seq_len, n_features),
        dtype=np.float32,
    )


def test_load_npz_sequences_loads_default_x_key(tmp_path: Path) -> None:
    npz_path = tmp_path / "batch.npz"
    sequences = make_sequences()

    np.savez(npz_path, X=sequences)

    result = load_npz_sequences(npz_path)

    assert isinstance(result, InferenceInput)
    assert result.array_key == "X"
    assert result.source_path == str(npz_path)
    assert result.batch_size == 3
    assert result.seq_len == 120
    assert result.n_features == 992
    np.testing.assert_array_equal(result.sequences, sequences)


def test_load_npz_sequences_loads_explicit_key(tmp_path: Path) -> None:
    npz_path = tmp_path / "batch.npz"
    sequences = make_sequences(batch_size=2)

    np.savez(npz_path, custom_sequences=sequences)

    result = load_npz_sequences(npz_path, array_key="custom_sequences")

    assert result.array_key == "custom_sequences"
    np.testing.assert_array_equal(result.sequences, sequences)


def test_load_npz_sequences_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_npz_sequences(tmp_path / "missing.npz")


def test_validate_sequence_batch_accepts_expected_shape() -> None:
    sequences = make_sequences()

    result = validate_sequence_batch(
        sequences,
        expected_seq_len=120,
        expected_n_features=992,
    )

    assert result is sequences


def test_validate_sequence_batch_rejects_non_3d_array() -> None:
    sequences = np.ones((120, 992), dtype=np.float32)

    with pytest.raises(ValueError, match="3D array"):
        validate_sequence_batch(sequences)


def test_validate_sequence_batch_rejects_wrong_seq_len() -> None:
    sequences = make_sequences(seq_len=100)

    with pytest.raises(ValueError, match="Invalid seq_len"):
        validate_sequence_batch(
            sequences,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_validate_sequence_batch_rejects_wrong_n_features() -> None:
    sequences = make_sequences(n_features=900)

    with pytest.raises(ValueError, match="Invalid n_features"):
        validate_sequence_batch(
            sequences,
            expected_seq_len=120,
            expected_n_features=992,
        )


def test_validate_sequence_batch_rejects_non_finite_values() -> None:
    sequences = make_sequences()
    sequences[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        validate_sequence_batch(sequences)


def test_normalize_model_output_accepts_one_dimensional_probability_vector() -> None:
    result = normalize_model_output(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        n_classes=6,
    )

    assert result.shape == (1, 6)
    np.testing.assert_allclose(
        result,
        np.array([[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]),
    )


def test_normalize_model_output_accepts_two_dimensional_probability_matrix() -> None:
    result = normalize_model_output(
        [
            [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        ],
        n_classes=6,
    )

    assert result.shape == (2, 6)


def test_normalize_model_output_applies_softmax_when_output_is_logits() -> None:
    result = normalize_model_output(
        [[1.0, 2.0, 3.0]],
        n_classes=3,
        output_is_logits=True,
    )

    assert result.shape == (1, 3)
    assert result.sum() == pytest.approx(1.0)
    assert result[0, 2] > result[0, 1] > result[0, 0]


def test_normalize_model_output_rejects_wrong_number_of_classes() -> None:
    with pytest.raises(ValueError, match="must contain 6 classes"):
        normalize_model_output(
            [0.10, 0.20, 0.70],
            n_classes=6,
        )


def test_predict_class_probabilities_uses_model_predict_method() -> None:
    sequences = make_sequences()
    model = FakePredictModel(
        [
            [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
            [0.10, 0.10, 0.10, 0.20, 0.20, 0.30],
        ]
    )

    result = predict_class_probabilities(
        model=model,
        sequences=sequences,
        n_classes=6,
    )

    assert result.shape == (3, 6)
    assert model.received_sequences is sequences


def test_predict_class_probabilities_accepts_callable_model() -> None:
    sequences = make_sequences(batch_size=1)

    def model_callable(input_sequences: np.ndarray) -> list[list[float]]:
        assert input_sequences is sequences
        return [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]

    result = predict_class_probabilities(
        model=model_callable,
        sequences=sequences,
        n_classes=6,
    )

    assert result.shape == (1, 6)


def test_select_probability_vector_last_returns_last_row() -> None:
    matrix = np.array(
        [
            [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        ],
        dtype=float,
    )

    probabilities, selected_index = select_probability_vector(
        matrix,
        aggregation="last",
        min_class_alert=3,
    )

    assert probabilities == (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)
    assert selected_index == 1


def test_select_probability_vector_mean_returns_average_row() -> None:
    matrix = np.array(
        [
            [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
            [0.10, 0.15, 0.20, 0.20, 0.20, 0.15],
        ],
        dtype=float,
    )

    probabilities, selected_index = select_probability_vector(
        matrix,
        aggregation="mean",
        min_class_alert=3,
    )

    assert probabilities == pytest.approx(
        (0.20, 0.20, 0.20, 0.15, 0.15, 0.10)
    )
    assert selected_index is None


def test_select_probability_vector_max_alert_returns_highest_alert_row() -> None:
    matrix = np.array(
        [
            [0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
            [0.40, 0.20, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=float,
    )

    probabilities, selected_index = select_probability_vector(
        matrix,
        aggregation="max_alert",
        min_class_alert=3,
    )

    assert probabilities == (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)
    assert selected_index == 1


def test_predict_volcano_alert_returns_complete_result() -> None:
    sequences = make_sequences()
    model = FakePredictModel(
        [
            [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
            [0.10, 0.10, 0.10, 0.20, 0.20, 0.30],
        ]
    )

    result = predict_volcano_alert(
        model=model,
        sequences=sequences,
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
        expected_seq_len=120,
        expected_n_features=992,
        aggregation="last",
        created_at_utc="2026-06-17T06:30:00Z",
        eruption_id="eruption_2019_10_25",
        model_name="cnn_transformer",
        model_version="3",
        run_id="abc123",
        metadata={"dataset": "full_stride5"},
    )

    assert isinstance(result, VolcanoAlertInferenceResult)
    assert result.prediction.created_at_utc == "2026-06-17T06:30:00Z"
    assert result.prediction.alert_24h is True
    assert result.prediction.p_alert_24h == pytest.approx(0.70)
    assert result.prediction.predicted_class == 5
    assert result.prediction.model_name == "cnn_transformer"
    assert result.raw_model_output_shape == (3, 6)
    assert result.selected_sequence_index == 2
    assert result.metadata["batch_size"] == 3
    assert result.metadata["seq_len"] == 120
    assert result.metadata["n_features"] == 992
    assert result.metadata["dataset"] == "full_stride5"


def test_predict_volcano_alert_from_npz_runs_end_to_end(tmp_path: Path) -> None:
    npz_path = tmp_path / "batch.npz"
    sequences = make_sequences(batch_size=1)

    np.savez(npz_path, X=sequences)

    model = FakePredictModel(
        [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]
    )

    result = predict_volcano_alert_from_npz(
        model=model,
        npz_path=npz_path,
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
        expected_seq_len=120,
        expected_n_features=992,
        created_at_utc="2026-06-17T06:30:00Z",
    )

    assert result.prediction.alert_24h is True
    assert result.prediction.metadata["array_key"] == "X"
    assert result.prediction.metadata["source_path"] == str(npz_path)


def test_predict_and_write_dashboard_outputs_writes_s3_objects() -> None:
    client = FakeS3Client()
    sequences = make_sequences(batch_size=1)
    model = FakePredictModel(
        [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]
    )

    result = predict_and_write_dashboard_outputs(
        model=model,
        sequences=sequences,
        s3_client=client,
        bucket="vulcadata",
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
        expected_seq_len=120,
        expected_n_features=992,
        created_at_utc="2026-06-17T06:30:00Z",
        generated_at_utc="2026-06-17T06:31:00Z",
        eruption_id="eruption_2019_10_25",
        model_name="cnn_transformer",
        model_version="3",
        run_id="abc123",
        metadata={"dataset": "full_stride5"},
        extra_report_metadata={"airflow_dag_id": "volcano_inference_pipeline"},
    )

    assert result.dashboard_locations is not None
    assert result.dashboard_locations.latest_prediction.key == (
        "predictions/latest/prediction.json"
    )
    assert result.dashboard_locations.history_prediction.key == (
        "predictions/history/20260617T063000Z/prediction.json"
    )
    assert result.dashboard_locations.inference_report.key == (
        "reports/inference/20260617T063000Z/report.json"
    )

    latest_payload = download_json(
        client=client,
        bucket="vulcadata",
        key=result.dashboard_locations.latest_prediction.key,
    )

    assert latest_payload["alert"]["active"] is True
    assert latest_payload["model"] == {
        "model_name": "cnn_transformer",
        "model_version": "3",
        "run_id": "abc123",
    }