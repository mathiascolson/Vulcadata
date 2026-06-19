from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.inference.load_model import (
    ModelReference,
    build_registered_model_uri,
    build_runs_model_uri,
    infer_model_reference,
    load_model,
    load_model_from_decision_file,
    read_model_decision,
    validate_model_flavor,
    validate_non_empty_string,
)


def test_validate_non_empty_string_accepts_clean_value() -> None:
    assert validate_non_empty_string(" cnn_transformer ", "model_name") == "cnn_transformer"


def test_validate_non_empty_string_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_non_empty_string("   ", "model_name")


def test_validate_model_flavor_accepts_pyfunc_and_pytorch() -> None:
    assert validate_model_flavor("pyfunc") == "pyfunc"
    assert validate_model_flavor("pytorch") == "pytorch"


def test_validate_model_flavor_rejects_unsupported_flavor() -> None:
    with pytest.raises(ValueError, match="Unsupported model flavor"):
        validate_model_flavor("sklearn")


def test_build_registered_model_uri_with_version() -> None:
    result = build_registered_model_uri(
        model_name="cnn_transformer",
        model_version=3,
    )

    assert result == "models:/cnn_transformer/3"


def test_build_registered_model_uri_with_stage() -> None:
    result = build_registered_model_uri(
        model_name="cnn_transformer",
        model_stage="Production",
    )

    assert result == "models:/cnn_transformer/Production"


def test_build_registered_model_uri_with_alias() -> None:
    result = build_registered_model_uri(
        model_name="cnn_transformer",
        model_alias="champion",
    )

    assert result == "models:/cnn_transformer@champion"


def test_build_registered_model_uri_rejects_multiple_selectors() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        build_registered_model_uri(
            model_name="cnn_transformer",
            model_version=3,
            model_stage="Production",
        )


def test_build_runs_model_uri_returns_mlflow_runs_uri() -> None:
    result = build_runs_model_uri(
        run_id="abc123",
        artifact_path="/model/",
    )

    assert result == "runs:/abc123/model"


def test_model_reference_to_dict_returns_metadata() -> None:
    reference = ModelReference(
        uri="models:/cnn_transformer/3",
        flavor="pytorch",
        source_type="mlflow_registered_model",
        model_name="cnn_transformer",
        model_version="3",
        metadata={"dataset": "full_stride5"},
    )

    assert reference.to_dict() == {
        "uri": "models:/cnn_transformer/3",
        "flavor": "pytorch",
        "source_type": "mlflow_registered_model",
        "model_name": "cnn_transformer",
        "model_version": "3",
        "model_stage": None,
        "model_alias": None,
        "run_id": None,
        "artifact_path": None,
        "metadata": {"dataset": "full_stride5"},
    }


def test_read_model_decision_reads_json_object(tmp_path: Path) -> None:
    decision_path = tmp_path / "final_model_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "model_name": "cnn_transformer",
                "model_version": "3",
            }
        ),
        encoding="utf-8",
    )

    result = read_model_decision(decision_path)

    assert result == {
        "model_name": "cnn_transformer",
        "model_version": "3",
    }


def test_read_model_decision_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_model_decision(tmp_path / "missing.json")


def test_read_model_decision_rejects_non_object_json(tmp_path: Path) -> None:
    decision_path = tmp_path / "final_model_decision.json"
    decision_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        read_model_decision(decision_path)


def test_infer_model_reference_from_direct_model_uri() -> None:
    reference = infer_model_reference(
        {
            "model_uri": "models:/cnn_transformer/3",
            "flavor": "pytorch",
            "model_name": "cnn_transformer",
            "model_version": "3",
        }
    )

    assert reference.uri == "models:/cnn_transformer/3"
    assert reference.flavor == "pytorch"
    assert reference.source_type == "mlflow_registered_model"
    assert reference.model_name == "cnn_transformer"
    assert reference.model_version == "3"


def test_infer_model_reference_from_registered_model_name_and_version() -> None:
    reference = infer_model_reference(
        {
            "model_name": "cnn_transformer",
            "model_version": "3",
            "flavor": "pytorch",
        }
    )

    assert reference.uri == "models:/cnn_transformer/3"
    assert reference.flavor == "pytorch"
    assert reference.source_type == "mlflow_registered_model"
    assert reference.model_name == "cnn_transformer"
    assert reference.model_version == "3"


def test_infer_model_reference_from_run_id_and_artifact_path() -> None:
    reference = infer_model_reference(
        {
            "run_id": "abc123",
            "artifact_path": "model",
            "flavor": "pytorch",
        }
    )

    assert reference.uri == "runs:/abc123/model"
    assert reference.flavor == "pytorch"
    assert reference.source_type == "mlflow_run_artifact"
    assert reference.run_id == "abc123"
    assert reference.artifact_path == "model"


def test_infer_model_reference_from_nested_decision() -> None:
    reference = infer_model_reference(
        {
            "selected_model": {
                "model_name": "cnn_transformer",
                "model_alias": "champion",
                "flavor": "pytorch",
            },
            "dataset": "full_stride5",
        }
    )

    assert reference.uri == "models:/cnn_transformer@champion"
    assert reference.flavor == "pytorch"
    assert reference.source_type == "mlflow_registered_model"
    assert reference.model_name == "cnn_transformer"
    assert reference.model_alias == "champion"


def test_infer_model_reference_from_local_model_path() -> None:
    reference = infer_model_reference(
        {
            "model_path": "models/champion/model.pt",
            "flavor": "pytorch",
        }
    )

    assert reference.uri == "models/champion/model.pt"
    assert reference.flavor == "pytorch"
    assert reference.source_type == "local_or_generic_path"


def test_infer_model_reference_rejects_missing_model_location() -> None:
    with pytest.raises(ValueError, match="Could not infer model reference"):
        infer_model_reference(
            {
                "dataset": "full_stride5",
            }
        )


def test_load_model_uses_injected_loader_without_mlflow_dependency() -> None:
    reference = ModelReference(
        uri="models:/cnn_transformer/3",
        flavor="pytorch",
        source_type="mlflow_registered_model",
    )

    def fake_loader(uri: str) -> dict[str, Any]:
        return {
            "loaded": True,
            "uri": uri,
        }

    result = load_model(reference, loader=fake_loader)

    assert result == {
        "loaded": True,
        "uri": "models:/cnn_transformer/3",
    }


def test_load_model_rejects_invalid_reference_object() -> None:
    with pytest.raises(TypeError, match="ModelReference"):
        load_model(  # type: ignore[arg-type]
            {"uri": "models:/cnn_transformer/3"},
            loader=lambda uri: uri,
        )


def test_load_model_from_decision_file_uses_injected_loader(tmp_path: Path) -> None:
    decision_path = tmp_path / "final_model_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "model_name": "cnn_transformer",
                "model_version": "3",
                "flavor": "pytorch",
            }
        ),
        encoding="utf-8",
    )

    result = load_model_from_decision_file(
        decision_path,
        loader=lambda uri: {"loaded_uri": uri},
    )

    assert result == {
        "loaded_uri": "models:/cnn_transformer/3",
    }
    
def test_infer_model_reference_from_final_decision_uses_declared_operational_candidate() -> None:
    reference = infer_model_reference(
        {
            "generated_at_utc": "2026-06-16T14:26:40.479809+00:00",
            "regression_candidate": {
                "model_family": "cnn_bilstm",
                "task_type": "regression",
                "run_id": "673ffb5e04cf43fcba5b55a6a4422737",
            },
            "classification_candidate": {
                "selection_name": "cnn_transformer_alert_priority",
                "model_family": "cnn_transformer",
                "task_type": "classification",
                "run_id": "42cc5dd5722349c88b056eb3c6b77d63",
                "model_type": "CNNTransformerClassifier",
            },
            "decision": {
                "main_operational_model": "classification_candidate",
                "regression_role": "exploratory_time_to_eruption_estimation",
                "classification_role": "operational_alert_24h",
            },
        }
    )

    assert reference.uri == "runs:/42cc5dd5722349c88b056eb3c6b77d63/model"
    assert reference.flavor == "pyfunc"
    assert reference.source_type == "mlflow_run_artifact"
    assert reference.model_name == "cnn_transformer"
    assert reference.run_id == "42cc5dd5722349c88b056eb3c6b77d63"
    assert reference.artifact_path == "model"
    assert reference.metadata["selected_candidate_key"] == "classification_candidate"


def test_infer_model_reference_rejects_missing_declared_operational_candidate() -> None:
    with pytest.raises(ValueError, match="main_operational_model"):
        infer_model_reference(
            {
                "regression_candidate": {
                    "model_family": "cnn_bilstm",
                    "run_id": "673ffb5e04cf43fcba5b55a6a4422737",
                },
                "decision": {
                    "main_operational_model": "classification_candidate",
                },
            }
        )