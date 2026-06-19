import json
from pathlib import Path


CONFIG_PATH = Path("configs/final_model_decision.json")


def test_final_model_decision_exists():
    assert CONFIG_PATH.exists(), f"Fichier introuvable : {CONFIG_PATH}"


def test_final_model_decision_required_sections():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "regression_candidate" in payload
    assert "classification_candidate" in payload
    assert "decision" in payload


def test_classification_decision_is_consistent():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    clf = payload["classification_candidate"]

    assert clf["model_family"] == "cnn_transformer"
    assert clf["task_type"] == "classification"
    assert clf["feature_set"] == "full"
    assert clf["dataset_group"] == "with_quiet"

    assert clf["class_weighting"] == "alert_priority"
    assert clf["min_class_alert"] == 3
    assert clf["alert_threshold_24h"] == 0.35

    assert 0.0 <= clf["precision_alert_24h"] <= 1.0
    assert 0.0 <= clf["recall_alert_24h"] <= 1.0
    assert 0.0 <= clf["f1_alert_24h"] <= 1.0


def test_operational_model_is_classification():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    decision = payload["decision"]

    assert decision["main_operational_model"] == "classification_candidate"
    assert decision["classification_role"] == "operational_alert_24h"