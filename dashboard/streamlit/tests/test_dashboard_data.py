from pathlib import Path
import sys


DASHBOARD_DIR = Path(__file__).resolve().parents[1]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from dashboard_config import CLASS_LABELS, DEFAULT_ALERT_THRESHOLD, DEFAULT_MIN_ALERT_CLASS
from dashboard_data import extract_prediction, normalize_probabilities


def test_class_labels_cover_all_prediction_classes():
    assert set(CLASS_LABELS.keys()) == {0, 1, 2, 3, 4, 5}


def test_normalize_probabilities_from_list():
    probabilities = normalize_probabilities([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])

    assert probabilities == [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]


def test_normalize_probabilities_from_class_dict():
    probabilities = normalize_probabilities(
        {
            "class_0": 0.1,
            "class_1": 0.2,
            "class_2": 0.3,
            "class_3": 0.15,
            "class_4": 0.15,
            "class_5": 0.1,
        }
    )

    assert probabilities == [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]


def test_extract_prediction_uses_real_dashboard_schema():
    payload = {
        "alert": {
            "active": False,
            "min_class_alert": 3,
            "p_alert_24h": 0.00380203643916838,
            "threshold_24h": 0.35,
        },
        "classification": {
            "n_classes": 6,
            "predicted_class": 1,
            "predicted_probability": 0.9836016430297109,
            "probabilities_by_class": {
                "class_0": 0.0015589072553659601,
                "class_1": 0.9836016430297109,
                "class_2": 0.011037413275754911,
                "class_3": 0.001332105854596177,
                "class_4": 0.0007119785885702108,
                "class_5": 0.001757951996001992,
            },
        },
        "prediction": {
            "alert_24h": False,
            "created_at_utc": "2026-06-18T16:31:09Z",
            "p_alert_24h": 0.00380203643916838,
            "predicted_class": 1,
            "predicted_probability": 0.9836016430297109,
            "probabilities": [
                0.0015589072553659601,
                0.9836016430297109,
                0.011037413275754911,
                0.001332105854596177,
                0.0007119785885702108,
                0.001757951996001992,
            ],
        },
        "metadata": {
            "aggregation": "last",
            "array_key": "X",
            "batch_size": 1,
            "model_output_shape": [1, 6],
        },
        "model": {
            "model_name": "cnn_transformer",
            "run_id": "5400525d69fe49029a27f3c36faa29bf",
        },
    }

    prediction = extract_prediction(payload)

    assert prediction["predicted_class"] == 1
    assert prediction["predicted_probability"] == 0.9836016430297109
    assert prediction["p_alert_24h"] == 0.00380203643916838
    assert prediction["alert_24h"] is False
    assert prediction["threshold_24h"] == DEFAULT_ALERT_THRESHOLD
    assert prediction["min_class_alert"] == DEFAULT_MIN_ALERT_CLASS
    assert prediction["created_at_utc"] == "2026-06-18T16:31:09Z"


def test_extract_prediction_recomputes_alert_probability_when_missing():
    payload = {
        "prediction": {
            "created_at_utc": "2026-06-18T16:31:09Z",
            "probabilities": [0.05, 0.10, 0.15, 0.20, 0.20, 0.30],
        },
        "alert": {
            "threshold_24h": 0.35,
            "min_class_alert": 3,
        },
    }

    prediction = extract_prediction(payload)

    assert prediction["predicted_class"] == 5
    assert prediction["predicted_probability"] == 0.30
    assert round(prediction["p_alert_24h"], 10) == 0.70
    assert prediction["alert_24h"] is True
