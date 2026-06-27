from __future__ import annotations

import argparse
import importlib
import json
import py_compile
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert isinstance(payload, dict), f"{path} must contain a JSON object"
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert isinstance(payload, dict), f"{path} must contain a YAML mapping"
    return payload


def walk_items(payload: Any):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key), value
            yield from walk_items(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from walk_items(item)


def values_for_keys(payload: Any, accepted_keys: set[str]) -> list[Any]:
    accepted = {key.lower() for key in accepted_keys}
    return [value for key, value in walk_items(payload) if key.lower() in accepted]


def first_int_for_keys(payload: Any, accepted_keys: set[str]) -> int | None:
    for value in values_for_keys(payload, accepted_keys):
        if isinstance(value, bool):
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def first_float_for_keys(payload: Any, accepted_keys: set[str]) -> float | None:
    for value in values_for_keys(payload, accepted_keys):
        if isinstance(value, bool):
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def first_list_for_keys(payload: Any, accepted_keys: set[str]) -> list[Any] | None:
    for value in values_for_keys(payload, accepted_keys):
        if isinstance(value, list):
            return value

    return None


def recursive_strings(payload: Any) -> list[str]:
    values: list[str] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            values.append(str(key))
            values.extend(recursive_strings(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(recursive_strings(item))
    elif isinstance(payload, str):
        values.append(payload)

    return values


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/pipeline_config.yaml",
        "configs/inference_config.yaml",
        "configs/final_model_decision.json",
    ],
)
def test_required_config_files_are_parseable(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path

    assert path.exists(), f"Missing config file: {relative_path}"

    if path.suffix == ".json":
        payload = read_json(path)
    else:
        payload = read_yaml(path)

    assert payload, f"Empty config file: {relative_path}"


def test_final_model_decision_operational_contract() -> None:
    decision_path = PROJECT_ROOT / "configs/final_model_decision.json"
    inference_config_path = PROJECT_ROOT / "configs/inference_config.yaml"

    decision_payload = read_json(decision_path)
    inference_payload = read_yaml(inference_config_path)

    combined_payload = {
        "final_model_decision": decision_payload,
        "inference_config": inference_payload,
    }

    classification_candidate = decision_payload.get("classification_candidate")
    assert isinstance(classification_candidate, dict), "final_model_decision.json must contain classification_candidate"

    all_strings = [value.lower() for value in recursive_strings(combined_payload)]

    assert any("cnn" in value and "transformer" in value for value in all_strings), "CNN-Transformer model reference not found"

    alert_threshold = first_float_for_keys(
        combined_payload,
        {"alert_threshold_24h", "alert_threshold", "classification_threshold"},
    )
    assert alert_threshold is not None, "Alert threshold not found"
    assert 0.0 < alert_threshold < 1.0, "Alert threshold must be between 0 and 1"

    min_class_alert = first_int_for_keys(combined_payload, {"min_class_alert"})
    assert min_class_alert is not None, "min_class_alert not found"
    assert min_class_alert >= 0, "min_class_alert must be non-negative"

    n_classes = first_int_for_keys(
        combined_payload,
        {"n_classes", "num_classes", "number_of_classes", "expected_n_classes"},
    )

    if n_classes is None:
        class_names = first_list_for_keys(combined_payload, {"class_names"})
        if class_names is not None:
            n_classes = len(class_names)

    assert n_classes == 6, "Operational classifier must use 6 classes"

    feature_dim = first_int_for_keys(
        combined_payload,
        {
            "feature_dim",
            "input_feature_dim",
            "n_features",
            "num_features",
            "number_of_features",
            "expected_n_features",
            "expected_feature_dim",
        },
    )
    assert feature_dim == 992, "Operational classifier must use 992 features"

    seq_len = first_int_for_keys(
        combined_payload,
        {"seq_len", "sequence_length", "expected_seq_len"},
    )

    if seq_len is not None:
        assert seq_len == 120, "Operational classifier must use sequences of length 120"

    assert any(
        "checkpoint" in value
        or ".pt" in value
        or "model_uri" in value
        or "runtime_model_source" in value
        for value in all_strings
    ), "Model runtime reference not found"


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/inference/load_model.py",
        "src/inference/run_inference.py",
        "src/retraining/train_candidate_model.py",
        "src/retraining/generate_retraining_evidently_report.py",
        "src/retraining/compare_candidate_to_champion.py",
        "src/retraining/promote_candidate_if_approved.py",
        "src/retraining/log_retraining_decision_to_mlflow.py",
        "infra/airflow/dags/volcano_inference_pipeline.py",
        "infra/airflow/dags/volcano_retraining_pipeline.py",
    ],
)
def test_critical_scripts_compile(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path

    assert path.exists(), f"Missing critical script: {relative_path}"

    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.inference.load_model",
        "src.inference.run_inference",
        "src.retraining.train_candidate_model",
        "src.retraining.generate_retraining_evidently_report",
        "src.retraining.compare_candidate_to_champion",
        "src.retraining.promote_candidate_if_approved",
        "src.retraining.log_retraining_decision_to_mlflow",
    ],
)
def test_critical_modules_are_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    "relative_path,report_type",
    [
        ("reports/retraining/retraining_decision_mlflow_result.json", "decision_mlflow"),
        ("reports/retraining/new_preprocessed_files_detection.json", "detection"),
        ("reports/retraining/candidate_training_result.json", "training"),
        ("reports/retraining/evidently/candidate_drift_summary.json", "drift"),
        ("reports/retraining/candidate_vs_champion_comparison.json", "comparison"),
        ("reports/retraining/candidate_promotion_result.json", "promotion"),
        ("reports/retraining/archive_processed_ready_files.json", "archive"),
    ],
)
def test_existing_retraining_report_contracts(relative_path: str, report_type: str) -> None:
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        pytest.skip(f"Report not generated yet: {relative_path}")

    payload = read_json(path)

    assert payload.get("status") == "success", f"{relative_path} must have status=success"

    if report_type == "detection":
        assert isinstance(payload.get("candidate_files_count"), int)
        assert payload["candidate_files_count"] >= 0

    if report_type == "training":
        assert payload.get("dry_run") is False
        assert isinstance(payload.get("metrics"), dict)
        assert isinstance(payload.get("artifacts"), dict)

    if report_type == "drift":
        assert isinstance(payload.get("critical_drift_detected"), bool)
        assert isinstance(payload.get("candidate_rejected_by_drift_check"), bool)

    if report_type == "comparison":
        assert payload.get("decision") in {"promote_candidate", "reject_candidate"}
        assert isinstance(payload.get("eligible_for_promotion"), bool)

    if report_type == "promotion":
        assert payload.get("action") in {"promotion_skipped", "candidate_promoted", "candidate_promotion_dry_run"}

    if report_type == "decision_mlflow":
        assert payload.get("run_name") == "airflow_retraining_decision"
        assert isinstance(payload.get("decision_mlflow_run_id"), str)
        assert payload["decision_mlflow_run_id"]
        assert isinstance(payload.get("decision_mlflow_artifact_uri"), str)
        assert payload["decision_mlflow_artifact_uri"]
        assert "candidate_mlflow_run_id" in payload
        if payload.get("candidate_mlflow_run_id"):
            assert isinstance(payload.get("candidate_mlflow_artifact_uri"), str)
            assert payload["candidate_mlflow_artifact_uri"]
        assert payload.get("decision") in {"promote_candidate", "reject_candidate"}
        assert payload.get("promotion_action") in {
            "promotion_skipped",
            "candidate_promoted",
            "candidate_promotion_dry_run",
        }

    if report_type == "archive":
        assert isinstance(payload.get("archived_files_count"), int)
        assert payload["archived_files_count"] >= 0


def test_promotion_is_skipped_when_candidate_is_rejected(tmp_path: Path) -> None:
    from src.retraining.promote_candidate_if_approved import promote_candidate_if_approved

    comparison_path = tmp_path / "candidate_vs_champion_comparison.json"
    output_path = tmp_path / "candidate_promotion_result.json"

    comparison_payload = {
        "status": "success",
        "decision": "reject_candidate",
        "eligible_for_promotion": False,
        "decision_reason": "unit test rejected candidate",
    }

    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison_payload, file)

    args = argparse.Namespace(
        project_root=str(tmp_path),
        comparison_json="candidate_vs_champion_comparison.json",
        candidate_result="candidate_training_result.json",
        decision_config="final_model_decision.json",
        output_json="candidate_promotion_result.json",
        local_champion_checkpoint="models/champion.pt",
        local_champion_archive_dir="models/archive",
        decision_archive_dir="configs/archive",
        s3_bucket="vulcadata",
        s3_champion_key="models/champion.pt",
        s3_champion_archive_prefix="models/archive",
        s3_decision_key="model_decisions/final_model_decision.json",
        skip_s3=True,
        dry_run=True,
        print_json=False,
    )

    result = promote_candidate_if_approved(args)

    assert result["status"] == "success"
    assert result["action"] == "promotion_skipped"
    assert result["reason"] == "candidate_not_eligible_for_promotion"
    assert output_path.exists()

    saved_result = read_json(output_path)

    assert saved_result["status"] == "success"
    assert saved_result["action"] == "promotion_skipped"