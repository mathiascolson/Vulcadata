from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.common.s3 import download_json
from src.inference.postprocess_predictions import postprocess_classification_prediction
from src.inference.write_dashboard_outputs import (
    DashboardOutputLocations,
    build_dashboard_payload,
    build_inference_report,
    build_inference_report_key,
    build_prediction_history_key,
    write_dashboard_outputs,
)


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


def make_prediction():
    return postprocess_classification_prediction(
        probabilities=[0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
        created_at_utc="2026-06-17T06:30:00Z",
        eruption_id="eruption_2019_10_25",
        model_name="cnn_transformer",
        model_version="3",
        run_id="abc123",
        metadata={"dataset": "full_stride5"},
    )


def test_build_prediction_history_key_uses_compact_utc_timestamp() -> None:
    result = build_prediction_history_key("2026-06-17T06:30:00Z")

    assert result == "predictions/history/20260617T063000Z/prediction.json"


def test_build_prediction_history_key_accepts_custom_prefix_and_filename() -> None:
    result = build_prediction_history_key(
        "2026-06-17T06:30:00Z",
        history_prefix="/custom/history/",
        filename="custom_prediction.json",
    )

    assert result == "custom/history/20260617T063000Z/custom_prediction.json"


def test_build_inference_report_key_uses_compact_utc_timestamp() -> None:
    result = build_inference_report_key("2026-06-17T06:30:00Z")

    assert result == "reports/inference/20260617T063000Z/report.json"


def test_build_dashboard_payload_returns_streamlit_friendly_structure() -> None:
    prediction = make_prediction()

    result = build_dashboard_payload(
        prediction,
        generated_at_utc="2026-06-17T06:31:00Z",
    )

    assert result["generated_at_utc"] == "2026-06-17T06:31:00Z"
    assert result["status"] == "success"
    assert result["alert"] == {
        "active": True,
        "p_alert_24h": pytest.approx(0.70),
        "threshold_24h": 0.35,
        "min_class_alert": 3,
    }
    assert result["classification"]["predicted_class"] == 4
    assert result["classification"]["predicted_probability"] == 0.25
    assert result["classification"]["n_classes"] == 6
    assert result["classification"]["probabilities_by_class"] == {
        "class_0": 0.05,
        "class_1": 0.10,
        "class_2": 0.15,
        "class_3": 0.20,
        "class_4": 0.25,
        "class_5": 0.25,
    }
    assert result["model"] == {
        "model_name": "cnn_transformer",
        "model_version": "3",
        "run_id": "abc123",
    }
    assert result["eruption_id"] == "eruption_2019_10_25"
    assert result["metadata"] == {"dataset": "full_stride5"}


def test_build_dashboard_payload_rejects_invalid_prediction_object() -> None:
    with pytest.raises(TypeError, match="ClassificationAlertPrediction"):
        build_dashboard_payload(  # type: ignore[arg-type]
            {"not": "a prediction"},
            generated_at_utc="2026-06-17T06:31:00Z",
        )


def test_build_inference_report_returns_operational_trace() -> None:
    prediction = make_prediction()

    locations = DashboardOutputLocations(
        latest_prediction=write_dashboard_outputs.__globals__["S3Location"](
            bucket="vulcadata",
            key="predictions/latest/prediction.json",
        ),
        history_prediction=write_dashboard_outputs.__globals__["S3Location"](
            bucket="vulcadata",
            key="predictions/history/20260617T063000Z/prediction.json",
        ),
        inference_report=write_dashboard_outputs.__globals__["S3Location"](
            bucket="vulcadata",
            key="reports/inference/20260617T063000Z/report.json",
        ),
    )

    result = build_inference_report(
        prediction=prediction,
        locations=locations,
        generated_at_utc="2026-06-17T06:31:00Z",
        extra_metadata={"airflow_dag_id": "volcano_inference_pipeline"},
    )

    assert result["generated_at_utc"] == "2026-06-17T06:31:00Z"
    assert result["status"] == "success"
    assert result["outputs"] == {
        "latest_prediction_uri": "s3://vulcadata/predictions/latest/prediction.json",
        "history_prediction_uri": "s3://vulcadata/predictions/history/20260617T063000Z/prediction.json",
        "inference_report_uri": "s3://vulcadata/reports/inference/20260617T063000Z/report.json",
    }
    assert result["prediction_summary"] == {
        "created_at_utc": "2026-06-17T06:30:00Z",
        "eruption_id": "eruption_2019_10_25",
        "alert_24h": True,
        "p_alert_24h": pytest.approx(0.70),
        "alert_threshold_24h": 0.35,
        "predicted_class": 4,
        "predicted_probability": 0.25,
    }
    assert result["model"] == {
        "model_name": "cnn_transformer",
        "model_version": "3",
        "run_id": "abc123",
    }
    assert result["metadata"] == {
        "airflow_dag_id": "volcano_inference_pipeline",
    }


def test_write_dashboard_outputs_writes_latest_history_and_report() -> None:
    client = FakeS3Client()
    prediction = make_prediction()

    locations = write_dashboard_outputs(
        client=client,
        bucket="vulcadata",
        prediction=prediction,
        generated_at_utc="2026-06-17T06:31:00Z",
        extra_report_metadata={"airflow_dag_id": "volcano_inference_pipeline"},
    )

    assert isinstance(locations, DashboardOutputLocations)

    assert locations.latest_prediction.uri == (
        "s3://vulcadata/predictions/latest/prediction.json"
    )
    assert locations.history_prediction.uri == (
        "s3://vulcadata/predictions/history/20260617T063000Z/prediction.json"
    )
    assert locations.inference_report.uri == (
        "s3://vulcadata/reports/inference/20260617T063000Z/report.json"
    )

    assert set(client.objects) == {
        ("vulcadata", "predictions/latest/prediction.json"),
        ("vulcadata", "predictions/history/20260617T063000Z/prediction.json"),
        ("vulcadata", "reports/inference/20260617T063000Z/report.json"),
    }

    assert client.content_types[
        ("vulcadata", "predictions/latest/prediction.json")
    ] == "application/json"
    assert client.content_types[
        ("vulcadata", "predictions/history/20260617T063000Z/prediction.json")
    ] == "application/json"
    assert client.content_types[
        ("vulcadata", "reports/inference/20260617T063000Z/report.json")
    ] == "application/json"


def test_write_dashboard_outputs_latest_and_history_have_same_dashboard_payload() -> None:
    client = FakeS3Client()
    prediction = make_prediction()

    locations = write_dashboard_outputs(
        client=client,
        bucket="vulcadata",
        prediction=prediction,
        generated_at_utc="2026-06-17T06:31:00Z",
    )

    latest_payload = download_json(
        client=client,
        bucket="vulcadata",
        key=locations.latest_prediction.key,
    )
    history_payload = download_json(
        client=client,
        bucket="vulcadata",
        key=locations.history_prediction.key,
    )

    assert latest_payload == history_payload
    assert latest_payload["alert"]["active"] is True
    assert latest_payload["alert"]["p_alert_24h"] == pytest.approx(0.70)
    assert latest_payload["prediction"]["created_at_utc"] == "2026-06-17T06:30:00Z"


def test_write_dashboard_outputs_report_references_written_locations() -> None:
    client = FakeS3Client()
    prediction = make_prediction()

    locations = write_dashboard_outputs(
        client=client,
        bucket="vulcadata",
        prediction=prediction,
        generated_at_utc="2026-06-17T06:31:00Z",
        extra_report_metadata={"airflow_run_id": "manual__2026-06-17"},
    )

    report_payload = download_json(
        client=client,
        bucket="vulcadata",
        key=locations.inference_report.key,
    )

    assert report_payload["outputs"] == locations.to_dict()
    assert report_payload["metadata"] == {
        "airflow_run_id": "manual__2026-06-17",
    }


def test_write_dashboard_outputs_accepts_custom_keys() -> None:
    client = FakeS3Client()
    prediction = make_prediction()

    locations = write_dashboard_outputs(
        client=client,
        bucket="vulcadata",
        prediction=prediction,
        latest_prediction_key="/custom/latest.json",
        history_prediction_prefix="/custom/history/",
        inference_report_prefix="/custom/reports/",
        generated_at_utc="2026-06-17T06:31:00Z",
    )

    assert locations.latest_prediction.key == "custom/latest.json"
    assert locations.history_prediction.key == (
        "custom/history/20260617T063000Z/prediction.json"
    )
    assert locations.inference_report.key == (
        "custom/reports/20260617T063000Z/report.json"
    )


def test_write_dashboard_outputs_rejects_invalid_prediction_object() -> None:
    client = FakeS3Client()

    with pytest.raises(TypeError, match="ClassificationAlertPrediction"):
        write_dashboard_outputs(
            client=client,
            bucket="vulcadata",
            prediction={"not": "a prediction"},  # type: ignore[arg-type]
            generated_at_utc="2026-06-17T06:31:00Z",
        )


def test_written_json_objects_are_valid_json() -> None:
    client = FakeS3Client()
    prediction = make_prediction()

    write_dashboard_outputs(
        client=client,
        bucket="vulcadata",
        prediction=prediction,
        generated_at_utc="2026-06-17T06:31:00Z",
    )

    for raw_content in client.objects.values():
        decoded = raw_content.decode("utf-8")
        parsed = json.loads(decoded)

        assert isinstance(parsed, dict)