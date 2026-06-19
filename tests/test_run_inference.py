from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.common.s3 import download_json
from src.inference.run_inference import (
    InferenceRuntimeParameters,
    read_yaml_mapping,
    resolve_inference_runtime_parameters,
    result_to_json,
    run_volcano_inference,
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


def make_sequences() -> np.ndarray:
    return np.ones(
        shape=(1, 120, 992),
        dtype=np.float32,
    )


def write_test_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    npz_path = tmp_path / "batch.npz"
    inference_config_path = tmp_path / "inference_config.yaml"
    model_decision_path = tmp_path / "final_model_decision.json"

    np.savez(npz_path, X=make_sequences())

    inference_config_path.write_text(
        "\n".join(
            [
                "inference:",
                "  seq_len: 120",
                "  expected_n_features: 992",
                "  n_classes: 6",
                "  min_class_alert: 3",
                "  alert_threshold_24h: 0.35",
                "  aggregation: last",
                "  output_is_logits: false",
                "storage:",
                "  s3_bucket: vulcadata",
            ]
        ),
        encoding="utf-8",
    )

    model_decision_path.write_text(
        json.dumps(
            {
                "model_name": "cnn_transformer",
                "model_version": "3",
                "flavor": "pyfunc",
            }
        ),
        encoding="utf-8",
    )

    return npz_path, inference_config_path, model_decision_path


def test_read_yaml_mapping_reads_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "inference:",
                "  seq_len: 120",
                "  n_classes: 6",
            ]
        ),
        encoding="utf-8",
    )

    result = read_yaml_mapping(config_path)

    assert result == {
        "inference": {
            "seq_len": 120,
            "n_classes": 6,
        }
    }


def test_resolve_inference_runtime_parameters_from_config_and_decision() -> None:
    inference_config = {
        "inference": {
            "seq_len": 120,
            "expected_n_features": 992,
            "n_classes": 6,
            "min_class_alert": 3,
            "alert_threshold_24h": 0.35,
            "aggregation": "last",
            "output_is_logits": False,
        },
        "storage": {
            "s3_bucket": "vulcadata",
        },
    }
    model_decision = {
        "model_name": "cnn_transformer",
        "model_version": "3",
    }

    result = resolve_inference_runtime_parameters(
        inference_config=inference_config,
        model_decision=model_decision,
    )

    assert isinstance(result, InferenceRuntimeParameters)
    assert result.expected_seq_len == 120
    assert result.expected_n_features == 992
    assert result.n_classes == 6
    assert result.min_class_alert == 3
    assert result.alert_threshold_24h == 0.35
    assert result.aggregation == "last"
    assert result.output_is_logits is False
    assert result.s3_bucket == "vulcadata"


def test_resolve_inference_runtime_parameters_overrides_config_values() -> None:
    inference_config = {
        "inference": {
            "seq_len": 120,
            "expected_n_features": 992,
            "n_classes": 6,
            "min_class_alert": 3,
            "alert_threshold_24h": 0.35,
            "aggregation": "last",
        }
    }
    model_decision = {}

    result = resolve_inference_runtime_parameters(
        inference_config=inference_config,
        model_decision=model_decision,
        overrides={
            "aggregation": "max_alert",
            "s3_bucket": "custom-bucket",
        },
    )

    assert result.aggregation == "max_alert"
    assert result.s3_bucket == "custom-bucket"


def test_resolve_inference_runtime_parameters_rejects_missing_required_value() -> None:
    with pytest.raises(ValueError, match="expected_seq_len"):
        resolve_inference_runtime_parameters(
            inference_config={},
            model_decision={},
        )


def test_run_volcano_inference_uses_injected_loader(tmp_path: Path) -> None:
    npz_path, inference_config_path, model_decision_path = write_test_files(tmp_path)

    fake_model = FakePredictModel(
        [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]
    )

    loaded_uris: list[str] = []

    def fake_loader(uri: str) -> FakePredictModel:
        loaded_uris.append(uri)
        return fake_model

    result = run_volcano_inference(
        npz_path=npz_path,
        inference_config_path=inference_config_path,
        model_decision_path=model_decision_path,
        loader=fake_loader,
        write_s3=False,
        created_at_utc="2026-06-17T06:30:00Z",
        eruption_id="eruption_2019_10_25",
        extra_metadata={"dataset": "full_stride5"},
    )

    assert loaded_uris == ["models:/cnn_transformer/3"]
    assert result.prediction.alert_24h is True
    assert result.prediction.p_alert_24h == pytest.approx(0.70)
    assert result.prediction.created_at_utc == "2026-06-17T06:30:00Z"
    assert result.prediction.eruption_id == "eruption_2019_10_25"
    assert result.prediction.model_name == "cnn_transformer"
    assert result.prediction.model_version == "3"
    assert result.dashboard_locations is None
    assert result.prediction.metadata["dataset"] == "full_stride5"
    assert result.prediction.metadata["array_key"] == "X"


def test_run_volcano_inference_accepts_direct_model_object(tmp_path: Path) -> None:
    npz_path, inference_config_path, model_decision_path = write_test_files(tmp_path)

    model = FakePredictModel(
        [[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]]
    )

    result = run_volcano_inference(
        npz_path=npz_path,
        inference_config_path=inference_config_path,
        model_decision_path=model_decision_path,
        model=model,
        write_s3=False,
        created_at_utc="2026-06-17T06:30:00Z",
    )

    assert result.prediction.alert_24h is False
    assert result.prediction.p_alert_24h == pytest.approx(0.25)
    assert model.received_sequences is not None


def test_run_volcano_inference_writes_dashboard_outputs_when_enabled(tmp_path: Path) -> None:
    npz_path, inference_config_path, model_decision_path = write_test_files(tmp_path)

    client = FakeS3Client()
    model = FakePredictModel(
        [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]
    )

    result = run_volcano_inference(
        npz_path=npz_path,
        inference_config_path=inference_config_path,
        model_decision_path=model_decision_path,
        model=model,
        s3_client=client,
        write_s3=True,
        created_at_utc="2026-06-17T06:30:00Z",
        generated_at_utc="2026-06-17T06:31:00Z",
        extra_report_metadata={"airflow_dag_id": "volcano_inference_pipeline"},
    )

    assert result.dashboard_locations is not None
    assert result.dashboard_locations.latest_prediction.uri == (
        "s3://vulcadata/predictions/latest/prediction.json"
    )
    assert result.dashboard_locations.history_prediction.uri == (
        "s3://vulcadata/predictions/history/20260617T063000Z/prediction.json"
    )
    assert result.dashboard_locations.inference_report.uri == (
        "s3://vulcadata/reports/inference/20260617T063000Z/report.json"
    )

    latest_payload = download_json(
        client=client,
        bucket="vulcadata",
        key=result.dashboard_locations.latest_prediction.key,
    )

    assert latest_payload["alert"]["active"] is True
    assert latest_payload["classification"]["predicted_class"] == 4


def test_run_volcano_inference_requires_bucket_when_writing_s3(tmp_path: Path) -> None:
    npz_path, inference_config_path, model_decision_path = write_test_files(tmp_path)

    inference_config_path.write_text(
        "\n".join(
            [
                "inference:",
                "  seq_len: 120",
                "  expected_n_features: 992",
                "  n_classes: 6",
                "  min_class_alert: 3",
                "  alert_threshold_24h: 0.35",
            ]
        ),
        encoding="utf-8",
    )

    model = FakePredictModel(
        [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]
    )

    with pytest.raises(ValueError, match="s3_bucket is required"):
        run_volcano_inference(
            npz_path=npz_path,
            inference_config_path=inference_config_path,
            model_decision_path=model_decision_path,
            model=model,
            s3_client=FakeS3Client(),
            write_s3=True,
            created_at_utc="2026-06-17T06:30:00Z",
        )


def test_result_to_json_returns_json_string(tmp_path: Path) -> None:
    npz_path, inference_config_path, model_decision_path = write_test_files(tmp_path)

    model = FakePredictModel(
        [[0.05, 0.10, 0.15, 0.20, 0.25, 0.25]]
    )

    result = run_volcano_inference(
        npz_path=npz_path,
        inference_config_path=inference_config_path,
        model_decision_path=model_decision_path,
        model=model,
        write_s3=False,
        created_at_utc="2026-06-17T06:30:00Z",
    )

    json_result = result_to_json(result)
    parsed = json.loads(json_result)

    assert parsed["prediction"]["alert_24h"] is True
    assert parsed["prediction"]["created_at_utc"] == "2026-06-17T06:30:00Z"
    assert parsed["dashboard_locations"] is None