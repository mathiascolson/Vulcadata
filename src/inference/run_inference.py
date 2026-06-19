from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args

from src.common.s3 import S3ClientProtocol, create_s3_client
from src.inference.load_model import (
    ModelReference,
    infer_model_reference,
    load_model,
    read_model_decision,
)
from src.inference.predict_volcano_alert import (
    ProbabilityAggregation,
    VolcanoAlertInferenceResult,
    predict_volcano_alert_from_npz,
)
from src.inference.write_dashboard_outputs import write_dashboard_outputs


DEFAULT_INFERENCE_CONFIG_PATH = "configs/inference_config.yaml"
DEFAULT_MODEL_DECISION_PATH = "configs/final_model_decision.json"

def load_project_dotenv(dotenv_path: str = ".env") -> None:
    """
    Load local environment variables from .env when python-dotenv is available.

    This is required for local CLI execution because PowerShell does not load
    .env files automatically. Existing environment variables are not overridden.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(dotenv_path=dotenv_path, override=False)


@dataclass(frozen=True)
class InferenceRuntimeParameters:
    """
    Runtime parameters required by the batch inference adapter.

    Values are resolved from configuration files and/or final_model_decision.json.
    No business value is hard-coded here.
    """

    expected_seq_len: int
    expected_n_features: int
    n_classes: int
    min_class_alert: int
    alert_threshold_24h: float
    aggregation: ProbabilityAggregation = "last"
    output_is_logits: bool = False
    array_key: str | None = None
    s3_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_seq_len": self.expected_seq_len,
            "expected_n_features": self.expected_n_features,
            "n_classes": self.n_classes,
            "min_class_alert": self.min_class_alert,
            "alert_threshold_24h": self.alert_threshold_24h,
            "aggregation": self.aggregation,
            "output_is_logits": self.output_is_logits,
            "array_key": self.array_key,
            "s3_bucket": self.s3_bucket,
        }


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """
    Read a YAML file and return a mapping.

    PyYAML is imported lazily so unit tests and modules that do not need YAML
    are not coupled to it at import time.
    """
    yaml_path = Path(path)

    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML config file does not exist: {yaml_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configuration files.") from exc

    content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if content is None:
        return {}

    if not isinstance(content, dict):
        raise ValueError(f"YAML config file must contain a mapping: {yaml_path}")

    return content


def resolve_inference_runtime_parameters(
    *,
    inference_config: Mapping[str, Any],
    model_decision: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> InferenceRuntimeParameters:
    """
    Resolve runtime parameters from config, final model decision and overrides.

    Priority:
    1. explicit overrides;
    2. final_model_decision.json;
    3. inference_config.yaml.
    """
    if not isinstance(inference_config, Mapping):
        raise TypeError("inference_config must be a mapping.")

    if not isinstance(model_decision, Mapping):
        raise TypeError("model_decision must be a mapping.")

    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping or None.")

    sources: tuple[Mapping[str, Any], ...] = (
        overrides or {},
        model_decision,
        inference_config,
    )

    expected_seq_len = _required_int(
        sources=sources,
        keys=("expected_seq_len", "seq_len", "sequence_length"),
        field_name="expected_seq_len",
    )

    expected_n_features = _required_int(
        sources=sources,
        keys=("expected_n_features", "n_features", "num_features", "feature_count"),
        field_name="expected_n_features",
    )

    n_classes = _required_int(
        sources=sources,
        keys=("n_classes", "num_classes", "number_of_classes"),
        field_name="n_classes",
    )

    min_class_alert = _required_int(
        sources=sources,
        keys=("min_class_alert", "min_alert_class", "alert_min_class"),
        field_name="min_class_alert",
    )

    alert_threshold_24h = _required_probability(
        sources=sources,
        keys=(
            "alert_threshold_24h",
            "threshold_24h",
            "p_alert_threshold",
            "alert_threshold",
        ),
        field_name="alert_threshold_24h",
    )

    aggregation = _optional_string(
        _find_first_value(
            sources=sources,
            keys=("aggregation", "probability_aggregation"),
            default="last",
        )
    )

    if aggregation not in get_args(ProbabilityAggregation):
        raise ValueError("aggregation must be one of: 'last', 'mean', 'max_alert'.")

    output_is_logits = _optional_bool(
        _find_first_value(
            sources=sources,
            keys=("output_is_logits", "model_output_is_logits"),
            default=False,
        )
    )

    array_key = _optional_string(
        _find_first_value(
            sources=sources,
            keys=("array_key", "npz_array_key", "input_array_key"),
            default=None,
        )
    )

    s3_bucket = _optional_string(
        _find_first_value(
            sources=sources,
            keys=("s3_bucket", "bucket", "dashboard_bucket"),
            default=None,
        )
    )

    return InferenceRuntimeParameters(
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
        n_classes=n_classes,
        min_class_alert=min_class_alert,
        alert_threshold_24h=alert_threshold_24h,
        aggregation=aggregation,
        output_is_logits=output_is_logits,
        array_key=array_key,
        s3_bucket=s3_bucket,
    )


def run_volcano_inference(
    *,
    npz_path: str | Path,
    inference_config_path: str | Path = DEFAULT_INFERENCE_CONFIG_PATH,
    model_decision_path: str | Path = DEFAULT_MODEL_DECISION_PATH,
    model: Any | None = None,
    loader: Any | None = None,
    s3_client: S3ClientProtocol | None = None,
    write_s3: bool = False,
    s3_bucket: str | None = None,
    array_key: str | None = None,
    aggregation: ProbabilityAggregation | None = None,
    output_is_logits: bool | None = None,
    created_at_utc: str | None = None,
    generated_at_utc: str | None = None,
    eruption_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    extra_report_metadata: dict[str, Any] | None = None,
) -> VolcanoAlertInferenceResult:
    """
    Run the complete batch inference adapter.

    This function is intentionally independent from Airflow.
    Airflow will later call this function from a DAG task.
    """
    inference_config = read_yaml_mapping(inference_config_path)
    model_decision = read_model_decision(model_decision_path)

    runtime_overrides: dict[str, Any] = {}

    if s3_bucket is not None:
        runtime_overrides["s3_bucket"] = s3_bucket

    if array_key is not None:
        runtime_overrides["array_key"] = array_key

    if aggregation is not None:
        runtime_overrides["aggregation"] = aggregation

    if output_is_logits is not None:
        runtime_overrides["output_is_logits"] = output_is_logits

    runtime_parameters = resolve_inference_runtime_parameters(
        inference_config=inference_config,
        model_decision=model_decision,
        overrides=runtime_overrides,
    )

    model_reference = infer_model_reference(model_decision)

    loaded_model = model if model is not None else load_model(
        model_reference,
        loader=loader,
    )

    metadata: dict[str, Any] = {
        "runtime_parameters": runtime_parameters.to_dict(),
        "model_reference": model_reference.to_dict(),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    result = predict_volcano_alert_from_npz(
        model=loaded_model,
        npz_path=npz_path,
        n_classes=runtime_parameters.n_classes,
        min_class_alert=runtime_parameters.min_class_alert,
        alert_threshold_24h=runtime_parameters.alert_threshold_24h,
        expected_seq_len=runtime_parameters.expected_seq_len,
        expected_n_features=runtime_parameters.expected_n_features,
        array_key=runtime_parameters.array_key,
        output_is_logits=runtime_parameters.output_is_logits,
        aggregation=runtime_parameters.aggregation,
        created_at_utc=created_at_utc,
        eruption_id=eruption_id,
        model_name=model_reference.model_name,
        model_version=model_reference.model_version,
        run_id=model_reference.run_id,
        metadata=metadata,
    )

    if not write_s3:
        return result

    target_bucket = s3_bucket or runtime_parameters.s3_bucket

    if target_bucket is None:
        raise ValueError(
            "s3_bucket is required when write_s3=True. "
            "Provide it in config or through the s3_bucket argument."
        )

    resolved_s3_client = s3_client if s3_client is not None else create_s3_client()

    dashboard_locations = write_dashboard_outputs(
        client=resolved_s3_client,
        bucket=target_bucket,
        prediction=result.prediction,
        generated_at_utc=generated_at_utc,
        extra_report_metadata=extra_report_metadata,
    )

    return VolcanoAlertInferenceResult(
        prediction=result.prediction,
        probabilities=result.probabilities,
        raw_model_output_shape=result.raw_model_output_shape,
        aggregation=result.aggregation,
        selected_sequence_index=result.selected_sequence_index,
        dashboard_locations=dashboard_locations,
        metadata=result.metadata,
    )


def result_to_json(result: VolcanoAlertInferenceResult, *, indent: int = 2) -> str:
    """
    Serialize an inference result to JSON.
    """
    if not isinstance(result, VolcanoAlertInferenceResult):
        raise TypeError("result must be a VolcanoAlertInferenceResult instance.")

    return json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_project_dotenv()

    parser = argparse.ArgumentParser(
        description="Run Vulcadata batch volcano alert inference."
    )

    parser.add_argument(
        "--npz-path",
        required=True,
        help="Path to the input NPZ sequence batch.",
    )
    parser.add_argument(
        "--inference-config",
        default=DEFAULT_INFERENCE_CONFIG_PATH,
        help="Path to configs/inference_config.yaml.",
    )
    parser.add_argument(
        "--model-decision",
        default=DEFAULT_MODEL_DECISION_PATH,
        help="Path to configs/final_model_decision.json.",
    )
    parser.add_argument(
        "--write-s3",
        action="store_true",
        help="Write dashboard outputs to S3.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Override S3 bucket used for dashboard outputs.",
    )
    parser.add_argument(
        "--array-key",
        default=None,
        help="Override NPZ array key.",
    )
    parser.add_argument(
        "--aggregation",
        choices=list(get_args(ProbabilityAggregation)),
        default=None,
        help="Probability aggregation mode.",
    )
    parser.add_argument(
        "--output-is-logits",
        action="store_true",
        help="Apply softmax to model output before postprocessing.",
    )
    parser.add_argument(
        "--created-at-utc",
        default=None,
        help="Prediction timestamp, for example 2026-06-17T06:30:00Z.",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Dashboard generation timestamp.",
    )
    parser.add_argument(
        "--eruption-id",
        default=None,
        help="Optional eruption identifier.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional local path where the JSON result will be written.",
    )

    args = parser.parse_args(argv)

    result = run_volcano_inference(
        npz_path=args.npz_path,
        inference_config_path=args.inference_config,
        model_decision_path=args.model_decision,
        write_s3=args.write_s3,
        s3_bucket=args.s3_bucket,
        array_key=args.array_key,
        aggregation=args.aggregation,
        output_is_logits=args.output_is_logits,
        created_at_utc=args.created_at_utc,
        generated_at_utc=args.generated_at_utc,
        eruption_id=args.eruption_id,
    )

    json_result = result_to_json(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_result, encoding="utf-8")

    print(json_result)

    return 0


def _find_first_value(
    *,
    sources: tuple[Mapping[str, Any], ...],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for source in sources:
        value = _find_first_value_in_mapping(source, keys=keys, default=None)

        if value is not None:
            return value

    return default


def _find_first_value_in_mapping(
    mapping: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]

    for value in mapping.values():
        if isinstance(value, Mapping):
            nested_value = _find_first_value_in_mapping(
                value,
                keys=keys,
                default=None,
            )

            if nested_value is not None:
                return nested_value

    return default


def _required_int(
    *,
    sources: tuple[Mapping[str, Any], ...],
    keys: tuple[str, ...],
    field_name: str,
) -> int:
    value = _find_first_value(
        sources=sources,
        keys=keys,
        default=None,
    )

    if value is None:
        raise ValueError(f"Missing required inference parameter: {field_name}.")

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")

    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer.") from exc

    if int_value <= 0:
        raise ValueError(f"{field_name} must be strictly positive.")

    return int_value


def _required_probability(
    *,
    sources: tuple[Mapping[str, Any], ...],
    keys: tuple[str, ...],
    field_name: str,
) -> float:
    value = _find_first_value(
        sources=sources,
        keys=keys,
        default=None,
    )

    if value is None:
        raise ValueError(f"Missing required inference parameter: {field_name}.")

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a number.") from exc

    if not isfinite(float_value):
        raise ValueError(f"{field_name} must be finite.")

    if not 0 <= float_value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")

    return float_value


def _optional_bool(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        cleaned_value = value.strip().lower()

        if cleaned_value in {"true", "1", "yes", "y"}:
            return True

        if cleaned_value in {"false", "0", "no", "n"}:
            return False

    raise TypeError("Boolean value expected.")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    cleaned_value = value.strip()

    if not cleaned_value:
        return None

    return cleaned_value

if __name__ == "__main__":
    raise SystemExit(main())