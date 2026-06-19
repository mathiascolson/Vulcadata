from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.inference.postprocess_predictions import (
    ClassificationAlertPrediction,
    compute_alert_probability,
    get_predicted_class,
    postprocess_classification_prediction,
    validate_classification_alert_parameters,
    validate_probability_vector,
)


def test_validate_classification_alert_parameters_accepts_valid_values() -> None:
    validate_classification_alert_parameters(
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
    )


def test_validate_classification_alert_parameters_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        validate_classification_alert_parameters(
            n_classes=1,
            min_class_alert=0,
            alert_threshold_24h=0.35,
        )


def test_validate_classification_alert_parameters_rejects_invalid_min_class_alert() -> None:
    with pytest.raises(ValueError, match="strictly lower than n_classes"):
        validate_classification_alert_parameters(
            n_classes=6,
            min_class_alert=6,
            alert_threshold_24h=0.35,
        )


def test_validate_classification_alert_parameters_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_classification_alert_parameters(
            n_classes=6,
            min_class_alert=3,
            alert_threshold_24h=1.2,
        )


def test_validate_probability_vector_accepts_valid_probabilities() -> None:
    result = validate_probability_vector(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        n_classes=6,
    )

    assert result == (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)


def test_validate_probability_vector_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 6 values"):
        validate_probability_vector(
            [0.1, 0.2, 0.7],
            n_classes=6,
        )


def test_validate_probability_vector_rejects_negative_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_probability_vector(
            [0.10, -0.10, 0.20, 0.20, 0.30, 0.30],
            n_classes=6,
        )


def test_validate_probability_vector_rejects_probability_sum_not_equal_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        validate_probability_vector(
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            n_classes=6,
        )


def test_validate_probability_vector_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="numeric values"):
        validate_probability_vector(
            [0.10, 0.20, "bad-value", 0.20, 0.20, 0.30],
            n_classes=6,
        )


def test_compute_alert_probability_sums_classes_from_min_class_alert() -> None:
    result = compute_alert_probability(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        min_class_alert=3,
    )

    assert result == pytest.approx(0.70)


def test_compute_alert_probability_rejects_min_class_alert_out_of_range() -> None:
    with pytest.raises(ValueError, match="lower than the number of probabilities"):
        compute_alert_probability(
            [0.05, 0.10, 0.15],
            min_class_alert=3,
        )


def test_get_predicted_class_returns_highest_probability_index() -> None:
    predicted_class, predicted_probability = get_predicted_class(
        [0.05, 0.10, 0.15, 0.20, 0.30, 0.20]
    )

    assert predicted_class == 4
    assert predicted_probability == 0.30


def test_get_predicted_class_rejects_empty_probabilities() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        get_predicted_class([])


def test_postprocess_classification_prediction_returns_alert_prediction() -> None:
    result = postprocess_classification_prediction(
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

    assert isinstance(result, ClassificationAlertPrediction)
    assert result.created_at_utc == "2026-06-17T06:30:00Z"
    assert result.predicted_class == 4
    assert result.predicted_probability == 0.25
    assert result.p_alert_24h == pytest.approx(0.70)
    assert result.alert_24h is True
    assert result.eruption_id == "eruption_2019_10_25"
    assert result.model_name == "cnn_transformer"
    assert result.model_version == "3"
    assert result.run_id == "abc123"
    assert result.metadata == {"dataset": "full_stride5"}


def test_postprocess_classification_prediction_returns_no_alert_when_below_threshold() -> None:
    result = postprocess_classification_prediction(
        probabilities=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
        created_at_utc="2026-06-17T06:30:00Z",
    )

    assert result.p_alert_24h == pytest.approx(0.25)
    assert result.alert_24h is False


def test_postprocess_classification_prediction_accepts_datetime_timestamp() -> None:
    result = postprocess_classification_prediction(
        probabilities=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
        n_classes=6,
        min_class_alert=3,
        alert_threshold_24h=0.35,
        created_at_utc=datetime(2026, 6, 17, 8, 30, tzinfo=UTC),
    )

    assert result.created_at_utc == "2026-06-17T08:30:00Z"


def test_prediction_to_dict_returns_json_compatible_structure() -> None:
    prediction = postprocess_classification_prediction(
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

    result = prediction.to_dict()

    assert result == {
        "created_at_utc": "2026-06-17T06:30:00Z",
        "predicted_class": 4,
        "predicted_probability": 0.25,
        "probabilities": [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
        "probabilities_by_class": {
            "class_0": 0.05,
            "class_1": 0.10,
            "class_2": 0.15,
            "class_3": 0.20,
            "class_4": 0.25,
            "class_5": 0.25,
        },
        "p_alert_24h": pytest.approx(0.70),
        "alert_24h": True,
        "min_class_alert": 3,
        "alert_threshold_24h": 0.35,
        "n_classes": 6,
        "eruption_id": "eruption_2019_10_25",
        "model_name": "cnn_transformer",
        "model_version": "3",
        "run_id": "abc123",
        "metadata": {"dataset": "full_stride5"},
    }