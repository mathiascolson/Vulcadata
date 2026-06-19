from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv


DEFAULT_BUCKET = "vulcadata"
DEFAULT_LATEST_PREDICTION_KEY = "predictions/latest/prediction.json"


class DashboardOutputValidationError(ValueError):
    """Raised when the dashboard prediction payload is not usable."""


def load_project_dotenv(dotenv_path: str | Path | None = None) -> None:
    if dotenv_path is None:
        dotenv_path = Path(".env")

    dotenv_path = Path(dotenv_path)

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def read_json_from_s3(
    bucket: str,
    key: str,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    if s3_client is None:
        s3_client = boto3.client("s3")

    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    payload = json.loads(body.decode("utf-8"))

    if not isinstance(payload, dict):
        raise DashboardOutputValidationError(
            f"S3 payload must be a JSON object, got {type(payload).__name__}."
        )

    return payload


def extract_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Accept both:
    - a direct prediction payload;
    - a wrapper containing {"prediction": {...}}.
    """
    nested_prediction = payload.get("prediction")

    if isinstance(nested_prediction, dict):
        return nested_prediction

    return payload


def assert_number(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = payload.get(key)

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DashboardOutputValidationError(
            f"Field '{key}' must be numeric, got {type(value).__name__}."
        )

    value_float = float(value)

    if not math.isfinite(value_float):
        raise DashboardOutputValidationError(f"Field '{key}' must be finite.")

    if minimum is not None and value_float < minimum:
        raise DashboardOutputValidationError(
            f"Field '{key}' must be >= {minimum}, got {value_float}."
        )

    if maximum is not None and value_float > maximum:
        raise DashboardOutputValidationError(
            f"Field '{key}' must be <= {maximum}, got {value_float}."
        )

    return value_float


def validate_dashboard_prediction_payload(
    payload: dict[str, Any],
    *,
    probability_sum_tolerance: float = 1e-3,
) -> dict[str, Any]:
    prediction = extract_prediction_payload(payload)

    required_fields = [
        "created_at_utc",
        "model_name",
        "run_id",
        "n_classes",
        "min_class_alert",
        "predicted_class",
        "predicted_probability",
        "p_alert_24h",
        "alert_24h",
        "probabilities",
        "probabilities_by_class",
    ]

    missing_fields = [field for field in required_fields if field not in prediction]
    if missing_fields:
        raise DashboardOutputValidationError(
            f"Missing required dashboard fields: {missing_fields}"
        )

    if not isinstance(prediction["created_at_utc"], str) or not prediction["created_at_utc"]:
        raise DashboardOutputValidationError("Field 'created_at_utc' must be a non-empty string.")

    if not isinstance(prediction["model_name"], str) or not prediction["model_name"]:
        raise DashboardOutputValidationError("Field 'model_name' must be a non-empty string.")

    if not isinstance(prediction["run_id"], str) or not prediction["run_id"]:
        raise DashboardOutputValidationError("Field 'run_id' must be a non-empty string.")

    n_classes = prediction["n_classes"]
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes <= 0:
        raise DashboardOutputValidationError("Field 'n_classes' must be a positive integer.")

    min_class_alert = prediction["min_class_alert"]
    if (
        not isinstance(min_class_alert, int)
        or isinstance(min_class_alert, bool)
        or min_class_alert < 0
        or min_class_alert >= n_classes
    ):
        raise DashboardOutputValidationError(
            "Field 'min_class_alert' must be an integer between 0 and n_classes - 1."
        )

    predicted_class = prediction["predicted_class"]
    if (
        not isinstance(predicted_class, int)
        or isinstance(predicted_class, bool)
        or predicted_class < 0
        or predicted_class >= n_classes
    ):
        raise DashboardOutputValidationError(
            "Field 'predicted_class' must be an integer between 0 and n_classes - 1."
        )

    predicted_probability = assert_number(
        prediction,
        "predicted_probability",
        minimum=0.0,
        maximum=1.0,
    )
    p_alert_24h = assert_number(
        prediction,
        "p_alert_24h",
        minimum=0.0,
        maximum=1.0,
    )

    alert_24h = prediction["alert_24h"]
    if not isinstance(alert_24h, bool):
        raise DashboardOutputValidationError("Field 'alert_24h' must be boolean.")

    probabilities = prediction["probabilities"]
    if not isinstance(probabilities, list):
        raise DashboardOutputValidationError("Field 'probabilities' must be a list.")

    if len(probabilities) != n_classes:
        raise DashboardOutputValidationError(
            f"Field 'probabilities' must contain {n_classes} values, got {len(probabilities)}."
        )

    numeric_probabilities = []
    for index, probability in enumerate(probabilities):
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            raise DashboardOutputValidationError(
                f"Probability at index {index} must be numeric."
            )

        probability_float = float(probability)

        if not math.isfinite(probability_float):
            raise DashboardOutputValidationError(
                f"Probability at index {index} must be finite."
            )

        if probability_float < 0.0 or probability_float > 1.0:
            raise DashboardOutputValidationError(
                f"Probability at index {index} must be between 0 and 1."
            )

        numeric_probabilities.append(probability_float)

    probabilities_sum = sum(numeric_probabilities)

    if abs(probabilities_sum - 1.0) > probability_sum_tolerance:
        raise DashboardOutputValidationError(
            f"Probabilities must sum to 1. Got {probabilities_sum:.8f}."
        )

    probabilities_by_class = prediction["probabilities_by_class"]
    if not isinstance(probabilities_by_class, dict):
        raise DashboardOutputValidationError(
            "Field 'probabilities_by_class' must be an object."
        )

    expected_class_keys = {f"class_{class_index}" for class_index in range(n_classes)}
    actual_class_keys = set(probabilities_by_class.keys())

    if actual_class_keys != expected_class_keys:
        raise DashboardOutputValidationError(
            "Field 'probabilities_by_class' has invalid class keys. "
            f"Expected {sorted(expected_class_keys)}, got {sorted(actual_class_keys)}."
        )

    recomputed_predicted_class = max(
        range(n_classes),
        key=lambda class_index: numeric_probabilities[class_index],
    )

    if predicted_class != recomputed_predicted_class:
        raise DashboardOutputValidationError(
            "Field 'predicted_class' is inconsistent with probabilities. "
            f"Expected {recomputed_predicted_class}, got {predicted_class}."
        )

    if abs(predicted_probability - numeric_probabilities[predicted_class]) > 1e-6:
        raise DashboardOutputValidationError(
            "Field 'predicted_probability' is inconsistent with probabilities."
        )

    recomputed_p_alert_24h = sum(numeric_probabilities[min_class_alert:])

    if abs(p_alert_24h - recomputed_p_alert_24h) > 1e-6:
        raise DashboardOutputValidationError(
            "Field 'p_alert_24h' is inconsistent with probabilities and min_class_alert."
        )

    return {
        "status": "success",
        "created_at_utc": prediction["created_at_utc"],
        "model_name": prediction["model_name"],
        "run_id": prediction["run_id"],
        "n_classes": n_classes,
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
        "p_alert_24h": p_alert_24h,
        "alert_24h": alert_24h,
        "probabilities_sum": probabilities_sum,
    }


def verify_dashboard_outputs(
    *,
    s3_bucket: str,
    latest_prediction_key: str,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    payload = read_json_from_s3(
        bucket=s3_bucket,
        key=latest_prediction_key,
        s3_client=s3_client,
    )

    summary = validate_dashboard_prediction_payload(payload)
    summary["checked_uri"] = f"s3://{s3_bucket}/{latest_prediction_key}"

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Vulcadata dashboard prediction outputs stored on S3."
    )

    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("VULCADATA_S3_BUCKET", DEFAULT_BUCKET),
        help="S3 bucket containing dashboard prediction outputs.",
    )

    parser.add_argument(
        "--latest-prediction-key",
        default=os.getenv(
            "VULCADATA_LATEST_PREDICTION_KEY",
            DEFAULT_LATEST_PREDICTION_KEY,
        ),
        help="S3 key for the latest dashboard prediction JSON.",
    )

    parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Optional dotenv path used for local execution.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_project_dotenv(args.dotenv_path)

    summary = verify_dashboard_outputs(
        s3_bucket=args.s3_bucket,
        latest_prediction_key=args.latest_prediction_key,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()