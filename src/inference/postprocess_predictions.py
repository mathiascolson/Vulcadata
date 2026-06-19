from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Sequence

from src.common.time_utils import format_utc_datetime, utc_now


@dataclass(frozen=True)
class ClassificationAlertPrediction:
    """
    Postprocessed classification output for volcano alert inference.

    This object is intentionally JSON-compatible through `to_dict()`.
    """

    created_at_utc: str
    predicted_class: int
    predicted_probability: float
    probabilities: tuple[float, ...]
    p_alert_24h: float
    alert_24h: bool
    min_class_alert: int
    alert_threshold_24h: float
    n_classes: int
    eruption_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert prediction to a JSON-compatible dictionary.
        """
        result: dict[str, Any] = {
            "created_at_utc": self.created_at_utc,
            "predicted_class": self.predicted_class,
            "predicted_probability": self.predicted_probability,
            "probabilities": list(self.probabilities),
            "probabilities_by_class": {
                f"class_{class_index}": probability
                for class_index, probability in enumerate(self.probabilities)
            },
            "p_alert_24h": self.p_alert_24h,
            "alert_24h": self.alert_24h,
            "min_class_alert": self.min_class_alert,
            "alert_threshold_24h": self.alert_threshold_24h,
            "n_classes": self.n_classes,
            "eruption_id": self.eruption_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

        return result


def validate_classification_alert_parameters(
    *,
    n_classes: int,
    min_class_alert: int,
    alert_threshold_24h: float,
) -> None:
    """
    Validate runtime classification parameters.

    These parameters normally come from configuration files or from
    final_model_decision.json. This function only enforces consistency
    at inference time.
    """
    if not isinstance(n_classes, int) or isinstance(n_classes, bool):
        raise TypeError("n_classes must be an integer.")

    if n_classes < 2:
        raise ValueError("n_classes must be greater than or equal to 2.")

    if not isinstance(min_class_alert, int) or isinstance(min_class_alert, bool):
        raise TypeError("min_class_alert must be an integer.")

    if min_class_alert < 0:
        raise ValueError("min_class_alert must be greater than or equal to 0.")

    if min_class_alert >= n_classes:
        raise ValueError("min_class_alert must be strictly lower than n_classes.")

    if not isinstance(alert_threshold_24h, int | float) or isinstance(alert_threshold_24h, bool):
        raise TypeError("alert_threshold_24h must be a number.")

    if not isfinite(float(alert_threshold_24h)):
        raise ValueError("alert_threshold_24h must be finite.")

    if not 0 <= float(alert_threshold_24h) <= 1:
        raise ValueError("alert_threshold_24h must be between 0 and 1.")


def validate_probability_vector(
    probabilities: Sequence[float],
    *,
    n_classes: int,
    sum_tolerance: float = 1e-4,
) -> tuple[float, ...]:
    """
    Validate and normalize a probability vector as an immutable tuple.

    The vector must:
    - be a sequence;
    - have exactly `n_classes` values;
    - contain only finite numeric values;
    - contain values between 0 and 1;
    - sum approximately to 1.
    """
    if isinstance(probabilities, str | bytes):
        raise TypeError("probabilities must be a sequence of numbers, not a string.")

    try:
        probability_tuple = tuple(float(value) for value in probabilities)
    except TypeError as exc:
        raise TypeError("probabilities must be an iterable sequence of numbers.") from exc
    except ValueError as exc:
        raise TypeError("probabilities must contain only numeric values.") from exc

    if len(probability_tuple) != n_classes:
        raise ValueError(
            f"probabilities must contain exactly {n_classes} values; "
            f"got {len(probability_tuple)}."
        )

    if not isinstance(sum_tolerance, int | float) or isinstance(sum_tolerance, bool):
        raise TypeError("sum_tolerance must be a number.")

    if not isfinite(float(sum_tolerance)) or float(sum_tolerance) <= 0:
        raise ValueError("sum_tolerance must be a finite positive number.")

    for probability in probability_tuple:
        if not isfinite(probability):
            raise ValueError("probabilities must contain only finite values.")

        if not 0 <= probability <= 1:
            raise ValueError("probabilities must be between 0 and 1.")

    probability_sum = sum(probability_tuple)

    if abs(probability_sum - 1.0) > float(sum_tolerance):
        raise ValueError(
            "probabilities must sum to 1 within tolerance; "
            f"got {probability_sum:.8f}."
        )

    return probability_tuple


def compute_alert_probability(
    probabilities: Sequence[float],
    *,
    min_class_alert: int,
) -> float:
    """
    Compute the alert probability.

    Example with 6 classes and min_class_alert=3:
    p_alert_24h = P(class_3) + P(class_4) + P(class_5)
    """
    if not isinstance(min_class_alert, int) or isinstance(min_class_alert, bool):
        raise TypeError("min_class_alert must be an integer.")

    if min_class_alert < 0:
        raise ValueError("min_class_alert must be greater than or equal to 0.")

    if min_class_alert >= len(probabilities):
        raise ValueError("min_class_alert must be lower than the number of probabilities.")

    return float(sum(probabilities[min_class_alert:]))


def get_predicted_class(probabilities: Sequence[float]) -> tuple[int, float]:
    """
    Return the class index with the highest probability and its probability.
    """
    if not probabilities:
        raise ValueError("probabilities cannot be empty.")

    predicted_class = max(range(len(probabilities)), key=lambda index: probabilities[index])
    predicted_probability = float(probabilities[predicted_class])

    return predicted_class, predicted_probability


def postprocess_classification_prediction(
    probabilities: Sequence[float],
    *,
    n_classes: int,
    min_class_alert: int,
    alert_threshold_24h: float,
    created_at_utc: str | datetime | None = None,
    eruption_id: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ClassificationAlertPrediction:
    """
    Convert raw class probabilities into a business alert prediction.

    This function contains no model-loading logic and performs no I/O.
    It is therefore easy to unit-test and can be reused by Airflow,
    Streamlit or standalone scripts.
    """
    validate_classification_alert_parameters(
        n_classes=n_classes,
        min_class_alert=min_class_alert,
        alert_threshold_24h=alert_threshold_24h,
    )

    probability_tuple = validate_probability_vector(
        probabilities,
        n_classes=n_classes,
    )

    predicted_class, predicted_probability = get_predicted_class(probability_tuple)

    p_alert_24h = compute_alert_probability(
        probability_tuple,
        min_class_alert=min_class_alert,
    )

    alert_24h = p_alert_24h >= float(alert_threshold_24h)

    timestamp = format_utc_datetime(created_at_utc if created_at_utc is not None else utc_now())

    return ClassificationAlertPrediction(
        created_at_utc=timestamp,
        predicted_class=predicted_class,
        predicted_probability=predicted_probability,
        probabilities=probability_tuple,
        p_alert_24h=p_alert_24h,
        alert_24h=alert_24h,
        min_class_alert=min_class_alert,
        alert_threshold_24h=float(alert_threshold_24h),
        n_classes=n_classes,
        eruption_id=eruption_id,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        metadata=metadata or {},
    )