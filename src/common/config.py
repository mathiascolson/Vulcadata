from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _strip_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _strip_prefix(value: str, field_name: str) -> str:
    value = _strip_non_empty(value, field_name)
    return value.strip("/")


class PathsConfig(StrictBaseModel):
    project_root: str
    local_data_dir: str
    local_reports_dir: str
    local_models_dir: str

    @field_validator("*")
    @classmethod
    def non_empty_string(cls, value: str, info) -> str:
        return _strip_non_empty(value, info.field_name)


class StorageConfig(StrictBaseModel):
    s3_bucket: str
    raw_prefix: str
    extraction_prefix: str
    processed_prefix: str
    inference_prefix: str
    reports_prefix: str
    monitoring_prefix: str
    model_decisions_prefix: str

    @field_validator("s3_bucket")
    @classmethod
    def valid_bucket(cls, value: str) -> str:
        return _strip_non_empty(value, "s3_bucket")

    @field_validator(
        "raw_prefix",
        "extraction_prefix",
        "processed_prefix",
        "inference_prefix",
        "reports_prefix",
        "monitoring_prefix",
        "model_decisions_prefix",
    )
    @classmethod
    def valid_prefix(cls, value: str, info) -> str:
        return _strip_prefix(value, info.field_name)


class MLflowConfig(StrictBaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str

    @field_validator("*")
    @classmethod
    def non_empty_string(cls, value: str, info) -> str:
        return _strip_non_empty(value, info.field_name)


class PipelineConfig(StrictBaseModel):
    project_name: str
    timezone: str
    paths: PathsConfig
    storage: StorageConfig
    mlflow: MLflowConfig

    @field_validator("project_name", "timezone")
    @classmethod
    def non_empty_string(cls, value: str, info) -> str:
        return _strip_non_empty(value, info.field_name)


class InferenceInputConfig(StrictBaseModel):
    seq_len: int = Field(gt=0)
    expected_n_features: int = Field(gt=0)
    feature_window_minutes: int = Field(gt=0)
    inference_window_hours: int = Field(gt=0)


class InferenceClassificationConfig(StrictBaseModel):
    n_classes: int = Field(gt=1)
    min_class_alert: int = Field(ge=0)
    alert_threshold_24h: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_min_alert_class(self) -> "InferenceClassificationConfig":
        if self.min_class_alert >= self.n_classes:
            raise ValueError("min_class_alert must be lower than n_classes.")
        return self


class InferenceOutputsConfig(StrictBaseModel):
    write_batch_outputs: bool
    update_latest_outputs: bool
    update_history: bool
    predictions_filename: str
    batch_summary_filename: str
    station_coverage_filename: str
    model_info_filename: str

    @field_validator(
        "predictions_filename",
        "batch_summary_filename",
        "station_coverage_filename",
        "model_info_filename",
    )
    @classmethod
    def non_empty_filename(cls, value: str, info) -> str:
        return _strip_non_empty(value, info.field_name)


class InferenceConfig(StrictBaseModel):
    model_decision_path: str
    input: InferenceInputConfig
    classification: InferenceClassificationConfig
    outputs: InferenceOutputsConfig

    @field_validator("model_decision_path")
    @classmethod
    def non_empty_model_decision_path(cls, value: str) -> str:
        return _strip_non_empty(value, "model_decision_path")

    @model_validator(mode="after")
    def validate_inference_window(self) -> "InferenceConfig":
        required_minutes = self.input.seq_len * self.input.feature_window_minutes
        available_minutes = self.input.inference_window_hours * 60

        if available_minutes < required_minutes:
            raise ValueError(
                "inference_window_hours is too short for the expected sequence length. "
                f"Required at least {required_minutes} minutes, got {available_minutes} minutes."
            )

        return self


class TrainingDatasetConfig(StrictBaseModel):
    seq_len: int = Field(gt=0)
    expected_n_features: int = Field(gt=0)
    n_classes: int = Field(gt=1)
    min_class_alert: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_min_alert_class(self) -> "TrainingDatasetConfig":
        if self.min_class_alert >= self.n_classes:
            raise ValueError("min_class_alert must be lower than n_classes.")
        return self


class TrainingRuntimeConfig(StrictBaseModel):
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)
    random_seed: int


class PromotionRulesConfig(StrictBaseModel):
    require_gx_passed: bool
    min_recall_alert_24h: float = Field(ge=0.0, le=1.0)
    min_precision_alert_24h: float = Field(ge=0.0, le=1.0)
    max_macro_f1_drop: float = Field(ge=0.0, le=1.0)
    min_precision_gain: float = Field(ge=0.0, le=1.0)


class TrainingConfig(StrictBaseModel):
    task_type: Literal["classification", "regression"]
    dataset: TrainingDatasetConfig
    training: TrainingRuntimeConfig
    promotion_rules: PromotionRulesConfig


class MonitoringReferenceConfig(StrictBaseModel):
    reference_dataset_s3_key: str

    @field_validator("reference_dataset_s3_key")
    @classmethod
    def non_empty_s3_key(cls, value: str) -> str:
        return _strip_prefix(value, "reference_dataset_s3_key")


class MonitoringCurrentConfig(StrictBaseModel):
    feature_snapshot_filename: str
    evidently_light_metrics_filename: str

    @field_validator("feature_snapshot_filename", "evidently_light_metrics_filename")
    @classmethod
    def non_empty_filename(cls, value: str, info) -> str:
        return _strip_non_empty(value, info.field_name)


class EvidentlyConfig(StrictBaseModel):
    run_light_metrics_each_batch: bool
    run_full_report_daily: bool
    full_report_hour_utc: int = Field(ge=0, le=23)
    reports_prefix: str
    monitoring_history_s3_key: str

    @field_validator("reports_prefix", "monitoring_history_s3_key")
    @classmethod
    def non_empty_s3_path(cls, value: str, info) -> str:
        return _strip_prefix(value, info.field_name)


class MonitoringThresholdsConfig(StrictBaseModel):
    max_drifted_feature_share: float = Field(ge=0.0, le=1.0)
    max_prediction_drift_score: float = Field(ge=0.0, le=1.0)


class MonitoringConfig(StrictBaseModel):
    reference: MonitoringReferenceConfig
    current: MonitoringCurrentConfig
    evidently: EvidentlyConfig
    thresholds: MonitoringThresholdsConfig


class AppConfig(StrictBaseModel):
    pipeline: PipelineConfig
    inference: InferenceConfig
    training: TrainingConfig
    monitoring: MonitoringConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_config_dir() -> Path:
    return get_project_root() / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(data, dict):
        raise TypeError(f"Configuration file must contain a YAML mapping: {path}")

    return data


def load_pipeline_config(config_dir: str | Path | None = None) -> PipelineConfig:
    config_dir = Path(config_dir) if config_dir else get_config_dir()
    return PipelineConfig.model_validate(load_yaml(config_dir / "pipeline_config.yaml"))


def load_inference_config(config_dir: str | Path | None = None) -> InferenceConfig:
    config_dir = Path(config_dir) if config_dir else get_config_dir()
    return InferenceConfig.model_validate(load_yaml(config_dir / "inference_config.yaml"))


def load_training_config(config_dir: str | Path | None = None) -> TrainingConfig:
    config_dir = Path(config_dir) if config_dir else get_config_dir()
    return TrainingConfig.model_validate(load_yaml(config_dir / "training_config.yaml"))


def load_monitoring_config(config_dir: str | Path | None = None) -> MonitoringConfig:
    config_dir = Path(config_dir) if config_dir else get_config_dir()
    return MonitoringConfig.model_validate(load_yaml(config_dir / "monitoring_config.yaml"))


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    return AppConfig(
        pipeline=load_pipeline_config(),
        inference=load_inference_config(),
        training=load_training_config(),
        monitoring=load_monitoring_config(),
    )


def clear_config_cache() -> None:
    load_app_config.cache_clear()