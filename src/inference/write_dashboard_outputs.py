from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.s3 import (
    S3ClientProtocol,
    S3Location,
    normalize_s3_key,
    normalize_s3_prefix,
    upload_json,
)
from src.common.time_utils import format_compact_utc, format_utc_datetime, utc_now
from src.inference.postprocess_predictions import ClassificationAlertPrediction


DEFAULT_LATEST_PREDICTION_KEY = "predictions/latest/prediction.json"
DEFAULT_HISTORY_PREDICTION_PREFIX = "predictions/history"
DEFAULT_INFERENCE_REPORT_PREFIX = "reports/inference"
DEFAULT_PREDICTION_FILENAME = "prediction.json"
DEFAULT_REPORT_FILENAME = "report.json"


@dataclass(frozen=True)
class DashboardOutputLocations:
    """
    S3 locations written for dashboard consumption.
    """

    latest_prediction: S3Location
    history_prediction: S3Location
    inference_report: S3Location

    def to_dict(self) -> dict[str, str]:
        return {
            "latest_prediction_uri": self.latest_prediction.uri,
            "history_prediction_uri": self.history_prediction.uri,
            "inference_report_uri": self.inference_report.uri,
        }


def build_prediction_history_key(
    created_at_utc: str | datetime,
    *,
    history_prefix: str = DEFAULT_HISTORY_PREDICTION_PREFIX,
    filename: str = DEFAULT_PREDICTION_FILENAME,
) -> str:
    """
    Build the S3 key for an immutable historical prediction.

    Example:
    predictions/history/20260617T063000Z/prediction.json
    """
    normalized_prefix = normalize_s3_prefix(history_prefix)
    normalized_filename = normalize_s3_key(filename)
    timestamp = format_compact_utc(created_at_utc)

    if normalized_prefix:
        return f"{normalized_prefix.rstrip('/')}/{timestamp}/{normalized_filename}"

    return f"{timestamp}/{normalized_filename}"


def build_inference_report_key(
    created_at_utc: str | datetime,
    *,
    report_prefix: str = DEFAULT_INFERENCE_REPORT_PREFIX,
    filename: str = DEFAULT_REPORT_FILENAME,
) -> str:
    """
    Build the S3 key for an inference report.

    Example:
    reports/inference/20260617T063000Z/report.json
    """
    normalized_prefix = normalize_s3_prefix(report_prefix)
    normalized_filename = normalize_s3_key(filename)
    timestamp = format_compact_utc(created_at_utc)

    if normalized_prefix:
        return f"{normalized_prefix.rstrip('/')}/{timestamp}/{normalized_filename}"

    return f"{timestamp}/{normalized_filename}"


def build_dashboard_payload(
    prediction: ClassificationAlertPrediction,
    *,
    generated_at_utc: str | datetime | None = None,
) -> dict[str, Any]:
    """
    Build a dashboard-friendly JSON payload from a postprocessed prediction.

    This function performs no I/O. It only reshapes the prediction into a
    structure that Streamlit can consume directly later.
    """
    if not isinstance(prediction, ClassificationAlertPrediction):
        raise TypeError("prediction must be a ClassificationAlertPrediction instance.")

    generated_timestamp = format_utc_datetime(
        generated_at_utc if generated_at_utc is not None else utc_now()
    )

    prediction_dict = prediction.to_dict()

    return {
        "generated_at_utc": generated_timestamp,
        "status": "success",
        "alert": {
            "active": prediction.alert_24h,
            "p_alert_24h": prediction.p_alert_24h,
            "threshold_24h": prediction.alert_threshold_24h,
            "min_class_alert": prediction.min_class_alert,
        },
        "classification": {
            "predicted_class": prediction.predicted_class,
            "predicted_probability": prediction.predicted_probability,
            "n_classes": prediction.n_classes,
            "probabilities_by_class": prediction_dict["probabilities_by_class"],
        },
        "model": {
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "run_id": prediction.run_id,
        },
        "eruption_id": prediction.eruption_id,
        "prediction": prediction_dict,
        "metadata": prediction.metadata,
    }


def build_inference_report(
    prediction: ClassificationAlertPrediction,
    locations: DashboardOutputLocations,
    *,
    generated_at_utc: str | datetime | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a compact inference report.

    The report is not the dashboard payload itself. It is an operational trace
    of what was produced and where it was written.
    """
    if not isinstance(prediction, ClassificationAlertPrediction):
        raise TypeError("prediction must be a ClassificationAlertPrediction instance.")

    if not isinstance(locations, DashboardOutputLocations):
        raise TypeError("locations must be a DashboardOutputLocations instance.")

    generated_timestamp = format_utc_datetime(
        generated_at_utc if generated_at_utc is not None else utc_now()
    )

    return {
        "generated_at_utc": generated_timestamp,
        "status": "success",
        "outputs": locations.to_dict(),
        "prediction_summary": {
            "created_at_utc": prediction.created_at_utc,
            "eruption_id": prediction.eruption_id,
            "alert_24h": prediction.alert_24h,
            "p_alert_24h": prediction.p_alert_24h,
            "alert_threshold_24h": prediction.alert_threshold_24h,
            "predicted_class": prediction.predicted_class,
            "predicted_probability": prediction.predicted_probability,
        },
        "model": {
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "run_id": prediction.run_id,
        },
        "metadata": extra_metadata or {},
    }


def write_dashboard_outputs(
    client: S3ClientProtocol,
    *,
    bucket: str,
    prediction: ClassificationAlertPrediction,
    latest_prediction_key: str = DEFAULT_LATEST_PREDICTION_KEY,
    history_prediction_prefix: str = DEFAULT_HISTORY_PREDICTION_PREFIX,
    inference_report_prefix: str = DEFAULT_INFERENCE_REPORT_PREFIX,
    generated_at_utc: str | datetime | None = None,
    extra_report_metadata: dict[str, Any] | None = None,
) -> DashboardOutputLocations:
    """
    Write all JSON outputs needed by the future Streamlit dashboard.

    Written objects:
    - latest prediction:
      predictions/latest/prediction.json

    - immutable historical prediction:
      predictions/history/<timestamp>/prediction.json

    - operational inference report:
      reports/inference/<timestamp>/report.json

    The function uses an injected S3 client and can therefore be tested without
    real AWS credentials or network calls.
    """
    if not isinstance(prediction, ClassificationAlertPrediction):
        raise TypeError("prediction must be a ClassificationAlertPrediction instance.")

    normalized_latest_key = normalize_s3_key(latest_prediction_key)

    history_key = build_prediction_history_key(
        prediction.created_at_utc,
        history_prefix=history_prediction_prefix,
    )

    report_key = build_inference_report_key(
        prediction.created_at_utc,
        report_prefix=inference_report_prefix,
    )

    dashboard_payload = build_dashboard_payload(
        prediction,
        generated_at_utc=generated_at_utc,
    )

    latest_location = upload_json(
        client=client,
        bucket=bucket,
        key=normalized_latest_key,
        data=dashboard_payload,
    )

    history_location = upload_json(
        client=client,
        bucket=bucket,
        key=history_key,
        data=dashboard_payload,
    )

    preliminary_locations = DashboardOutputLocations(
        latest_prediction=latest_location,
        history_prediction=history_location,
        inference_report=S3Location(bucket=bucket, key=report_key),
    )

    report_payload = build_inference_report(
        prediction=prediction,
        locations=preliminary_locations,
        generated_at_utc=generated_at_utc,
        extra_metadata=extra_report_metadata,
    )

    report_location = upload_json(
        client=client,
        bucket=bucket,
        key=report_key,
        data=report_payload,
    )

    return DashboardOutputLocations(
        latest_prediction=latest_location,
        history_prediction=history_location,
        inference_report=report_location,
    )