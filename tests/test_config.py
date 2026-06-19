from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.common.config import (
    InferenceClassificationConfig,
    InferenceConfig,
    load_app_config,
    load_inference_config,
    load_monitoring_config,
    load_pipeline_config,
    load_training_config,
)


def write_yaml(path: Path, content: dict) -> None:
    path.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_load_app_config_from_project_configs() -> None:
    cfg = load_app_config()

    assert cfg.pipeline.project_name == "vulcadata"
    assert cfg.pipeline.storage.s3_bucket == "vulcadata"

    assert cfg.inference.input.seq_len > 0
    assert cfg.inference.input.expected_n_features > 0
    assert cfg.inference.classification.n_classes > 1
    assert (
        cfg.inference.classification.min_class_alert
        < cfg.inference.classification.n_classes
    )
    assert 0.0 <= cfg.inference.classification.alert_threshold_24h <= 1.0

    assert cfg.training.task_type in {"classification", "regression"}
    assert cfg.monitoring.evidently.full_report_hour_utc in range(24)


def test_individual_config_loaders_from_project_configs() -> None:
    pipeline_cfg = load_pipeline_config()
    inference_cfg = load_inference_config()
    training_cfg = load_training_config()
    monitoring_cfg = load_monitoring_config()

    assert pipeline_cfg.project_name == "vulcadata"
    assert inference_cfg.input.seq_len > 0
    assert training_cfg.training.batch_size > 0
    assert monitoring_cfg.thresholds.max_drifted_feature_share <= 1.0


def test_inference_classification_rejects_invalid_alert_threshold() -> None:
    with pytest.raises(ValidationError):
        InferenceClassificationConfig(
            n_classes=6,
            min_class_alert=3,
            alert_threshold_24h=1.5,
        )


def test_inference_classification_rejects_invalid_min_alert_class() -> None:
    with pytest.raises(ValidationError):
        InferenceClassificationConfig(
            n_classes=6,
            min_class_alert=6,
            alert_threshold_24h=0.35,
        )


def test_inference_classification_rejects_invalid_number_of_classes() -> None:
    with pytest.raises(ValidationError):
        InferenceClassificationConfig(
            n_classes=1,
            min_class_alert=0,
            alert_threshold_24h=0.35,
        )


def test_inference_config_rejects_too_short_inference_window() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(
            {
                "model_decision_path": "configs/final_model_decision.json",
                "input": {
                    "seq_len": 120,
                    "expected_n_features": 992,
                    "feature_window_minutes": 10,
                    "inference_window_hours": 1,
                },
                "classification": {
                    "n_classes": 6,
                    "min_class_alert": 3,
                    "alert_threshold_24h": 0.35,
                },
                "outputs": {
                    "write_batch_outputs": True,
                    "update_latest_outputs": True,
                    "update_history": True,
                    "predictions_filename": "predictions.parquet",
                    "batch_summary_filename": "batch_summary.json",
                    "station_coverage_filename": "station_coverage.json",
                    "model_info_filename": "model_info.json",
                },
            }
        )


def test_missing_required_pipeline_field_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    write_yaml(
        config_dir / "pipeline_config.yaml",
        {
            "project_name": "vulcadata",
            "timezone": "UTC",
            "paths": {
                "project_root": ".",
                "local_data_dir": "data",
                "local_reports_dir": "reports",
                "local_models_dir": "models",
            },
            "storage": {
                # s3_bucket volontairement absent
                "raw_prefix": "volcano/raw_mseed",
                "extraction_prefix": "volcano/extraction",
                "processed_prefix": "volcano/preprocessing/processed",
                "inference_prefix": "volcano/inference",
                "reports_prefix": "volcano/reports",
                "monitoring_prefix": "volcano/monitoring",
                "model_decisions_prefix": "volcano/model_decisions",
            },
            "mlflow": {
                "tracking_uri": "https://vartkirl-vulcadata-mlflow.hf.space/",
                "experiment_name": "Vulcadata",
                "registered_model_name": "vulcadata_alert_model",
            },
        },
    )

    with pytest.raises(ValidationError):
        load_pipeline_config(config_dir=config_dir)


def test_extra_unknown_field_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    write_yaml(
        config_dir / "inference_config.yaml",
        {
            "model_decision_path": "configs/final_model_decision.json",
            "unexpected_field": "must fail",
            "input": {
                "seq_len": 120,
                "expected_n_features": 992,
                "feature_window_minutes": 10,
                "inference_window_hours": 24,
            },
            "classification": {
                "n_classes": 6,
                "min_class_alert": 3,
                "alert_threshold_24h": 0.35,
            },
            "outputs": {
                "write_batch_outputs": True,
                "update_latest_outputs": True,
                "update_history": True,
                "predictions_filename": "predictions.parquet",
                "batch_summary_filename": "batch_summary.json",
                "station_coverage_filename": "station_coverage.json",
                "model_info_filename": "model_info.json",
            },
        },
    )

    with pytest.raises(ValidationError):
        load_inference_config(config_dir=config_dir)