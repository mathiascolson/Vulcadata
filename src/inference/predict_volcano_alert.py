from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.common.s3 import S3ClientProtocol
from src.inference.postprocess_predictions import (
    ClassificationAlertPrediction,
    postprocess_classification_prediction,
    validate_probability_vector,
)
from src.inference.write_dashboard_outputs import (
    DashboardOutputLocations,
    write_dashboard_outputs,
)


ProbabilityAggregation = Literal["last", "mean", "max_alert"]


@dataclass(frozen=True)
class InferenceInput:
    """
    Validated inference input loaded from a sequence batch.
    """

    sequences: np.ndarray
    source_path: str | None = None
    array_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.sequences.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.sequences.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.sequences.shape[2])


@dataclass(frozen=True)
class VolcanoAlertInferenceResult:
    """
    Full inference result returned by the volcano alert prediction pipeline.
    """

    prediction: ClassificationAlertPrediction
    probabilities: tuple[float, ...]
    raw_model_output_shape: tuple[int, ...]
    aggregation: ProbabilityAggregation
    selected_sequence_index: int | None = None
    dashboard_locations: DashboardOutputLocations | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "prediction": self.prediction.to_dict(),
            "probabilities": list(self.probabilities),
            "raw_model_output_shape": list(self.raw_model_output_shape),
            "aggregation": self.aggregation,
            "selected_sequence_index": self.selected_sequence_index,
            "metadata": self.metadata,
        }

        if self.dashboard_locations is not None:
            result["dashboard_locations"] = self.dashboard_locations.to_dict()
        else:
            result["dashboard_locations"] = None

        return result


def load_npz_sequences(
    npz_path: str | Path,
    *,
    array_key: str | None = None,
) -> InferenceInput:
    """
    Load a 3D sequence batch from an NPZ file.

    Expected shape:
    (batch_size, seq_len, n_features)

    If array_key is not provided, common keys are tried first:
    X, x, sequences, inputs, features.

    If none of these exists, the function selects the only 3D array available.
    """
    path = Path(npz_path)

    if not path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {path}")

    with np.load(path, allow_pickle=False) as npz_file:
        available_keys = list(npz_file.files)
        selected_key = _select_npz_array_key(
            available_keys=available_keys,
            npz_file=npz_file,
            requested_key=array_key,
        )
        sequences = np.asarray(npz_file[selected_key])

    return InferenceInput(
        sequences=sequences,
        source_path=str(path),
        array_key=selected_key,
        metadata={
            "available_npz_keys": available_keys,
        },
    )


def validate_sequence_batch(
    sequences: np.ndarray,
    *,
    expected_seq_len: int | None = None,
    expected_n_features: int | None = None,
) -> np.ndarray:
    """
    Validate the input sequence batch.

    The model expects a 3D numeric finite array:
    (batch_size, seq_len, n_features)
    """
    if not isinstance(sequences, np.ndarray):
        raise TypeError("sequences must be a numpy.ndarray.")

    if sequences.ndim != 3:
        raise ValueError(
            "sequences must be a 3D array with shape "
            "(batch_size, seq_len, n_features)."
        )

    if sequences.shape[0] <= 0:
        raise ValueError("sequences batch_size must be strictly positive.")

    if sequences.shape[1] <= 0:
        raise ValueError("sequences seq_len must be strictly positive.")

    if sequences.shape[2] <= 0:
        raise ValueError("sequences n_features must be strictly positive.")

    if not np.issubdtype(sequences.dtype, np.number):
        raise TypeError("sequences must contain numeric values.")

    if not np.all(np.isfinite(sequences)):
        raise ValueError("sequences must contain only finite values.")

    if expected_seq_len is not None:
        _validate_positive_integer(expected_seq_len, "expected_seq_len")

        if int(sequences.shape[1]) != expected_seq_len:
            raise ValueError(
                f"Invalid seq_len: expected {expected_seq_len}, "
                f"got {sequences.shape[1]}."
            )

    if expected_n_features is not None:
        _validate_positive_integer(expected_n_features, "expected_n_features")

        if int(sequences.shape[2]) != expected_n_features:
            raise ValueError(
                f"Invalid n_features: expected {expected_n_features}, "
                f"got {sequences.shape[2]}."
            )

    return sequences


def predict_class_probabilities(
    model: Any,
    sequences: np.ndarray,
    *,
    n_classes: int,
    output_is_logits: bool = False,
) -> np.ndarray:
    """
    Run model prediction and return a 2D probability matrix.

    Accepted model interfaces:
    - object with .predict(sequences)
    - callable model(sequences)

    Returned shape:
    (batch_size, n_classes)
    """
    _validate_positive_integer(n_classes, "n_classes")

    if hasattr(model, "predict"):
        raw_output = model.predict(sequences)
    elif callable(model):
        raw_output = model(sequences)
    else:
        raise TypeError("model must be callable or expose a predict method.")

    probability_matrix = normalize_model_output(
        raw_output,
        n_classes=n_classes,
        output_is_logits=output_is_logits,
    )

    return probability_matrix


def normalize_model_output(
    model_output: Any,
    *,
    n_classes: int,
    output_is_logits: bool = False,
    sum_tolerance: float = 1e-4,
) -> np.ndarray:
    """
    Convert a raw model output into a validated probability matrix.

    Accepted shapes:
    - (n_classes,)
    - (batch_size, n_classes)

    If output_is_logits=True, a softmax is applied row-wise.
    Otherwise, the output is expected to already contain probabilities.
    """
    _validate_positive_integer(n_classes, "n_classes")

    output_array = np.asarray(model_output, dtype=float)

    if output_array.ndim == 1:
        output_array = output_array.reshape(1, -1)

    if output_array.ndim != 2:
        raise ValueError("model output must be a 1D or 2D array.")

    if output_array.shape[1] != n_classes:
        raise ValueError(
            f"model output must contain {n_classes} classes; "
            f"got {output_array.shape[1]}."
        )

    if output_array.shape[0] <= 0:
        raise ValueError("model output batch_size must be strictly positive.")

    if not np.all(np.isfinite(output_array)):
        raise ValueError("model output must contain only finite values.")

    if output_is_logits:
        output_array = _softmax_2d(output_array)

    probability_rows: list[tuple[float, ...]] = []

    for row in output_array:
        probability_rows.append(
            validate_probability_vector(
                row.tolist(),
                n_classes=n_classes,
                sum_tolerance=sum_tolerance,
            )
        )

    return np.asarray(probability_rows, dtype=float)


def select_probability_vector(
    probability_matrix: np.ndarray,
    *,
    aggregation: ProbabilityAggregation = "last",
    min_class_alert: int,
) -> tuple[tuple[float, ...], int | None]:
    """
    Select or aggregate the probability vector used for the final alert.

    Supported modes:
    - last:
      use the last sequence of the batch.

    - mean:
      average probabilities over the batch.

    - max_alert:
      select the sequence with the highest alert probability.
    """
    if not isinstance(probability_matrix, np.ndarray):
        raise TypeError("probability_matrix must be a numpy.ndarray.")

    if probability_matrix.ndim != 2:
        raise ValueError("probability_matrix must be a 2D array.")

    if probability_matrix.shape[0] <= 0:
        raise ValueError("probability_matrix batch_size must be strictly positive.")

    if aggregation == "last":
        selected_index = int(probability_matrix.shape[0] - 1)
        selected = probability_matrix[selected_index]
        return tuple(float(value) for value in selected), selected_index

    if aggregation == "mean":
        selected = probability_matrix.mean(axis=0)
        return tuple(float(value) for value in selected), None

    if aggregation == "max_alert":
        if not isinstance(min_class_alert, int) or isinstance(min_class_alert, bool):
            raise TypeError("min_class_alert must be an integer.")

        if min_class_alert < 0 or min_class_alert >= probability_matrix.shape[1]:
            raise ValueError("min_class_alert must be within probability class range.")

        alert_probabilities = probability_matrix[:, min_class_alert:].sum(axis=1)
        selected_index = int(np.argmax(alert_probabilities))
        selected = probability_matrix[selected_index]
        return tuple(float(value) for value in selected), selected_index

    raise ValueError(
        "aggregation must be one of: 'last', 'mean', 'max_alert'."
    )


def predict_volcano_alert(
    model: Any,
    sequences: np.ndarray,
    *,
    n_classes: int,
    min_class_alert: int,
    alert_threshold_24h: float,
    expected_seq_len: int | None = None,
    expected_n_features: int | None = None,
    output_is_logits: bool = False,
    aggregation: ProbabilityAggregation = "last",
    created_at_utc: str | datetime | None = None,
    eruption_id: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VolcanoAlertInferenceResult:
    """
    Run volcano alert inference from an already loaded model and sequence batch.

    This function performs:
    - input validation;
    - model prediction;
    - model output normalization;
    - probability selection or aggregation;
    - alert postprocessing.

    It performs no S3 write and no model loading.
    """
    validated_sequences = validate_sequence_batch(
        sequences,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
    )

    probability_matrix = predict_class_probabilities(
        model=model,
        sequences=validated_sequences,
        n_classes=n_classes,
        output_is_logits=output_is_logits,
    )

    selected_probabilities, selected_sequence_index = select_probability_vector(
        probability_matrix,
        aggregation=aggregation,
        min_class_alert=min_class_alert,
    )

    runtime_metadata: dict[str, Any] = {
        "batch_size": int(validated_sequences.shape[0]),
        "seq_len": int(validated_sequences.shape[1]),
        "n_features": int(validated_sequences.shape[2]),
        "model_output_shape": list(probability_matrix.shape),
        "aggregation": aggregation,
        "selected_sequence_index": selected_sequence_index,
        "output_is_logits": output_is_logits,
    }

    if metadata:
        runtime_metadata.update(metadata)

    prediction = postprocess_classification_prediction(
        probabilities=selected_probabilities,
        n_classes=n_classes,
        min_class_alert=min_class_alert,
        alert_threshold_24h=alert_threshold_24h,
        created_at_utc=created_at_utc,
        eruption_id=eruption_id,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        metadata=runtime_metadata,
    )

    return VolcanoAlertInferenceResult(
        prediction=prediction,
        probabilities=selected_probabilities,
        raw_model_output_shape=tuple(int(value) for value in probability_matrix.shape),
        aggregation=aggregation,
        selected_sequence_index=selected_sequence_index,
        metadata=runtime_metadata,
    )


def predict_volcano_alert_from_npz(
    model: Any,
    npz_path: str | Path,
    *,
    n_classes: int,
    min_class_alert: int,
    alert_threshold_24h: float,
    expected_seq_len: int,
    expected_n_features: int,
    array_key: str | None = None,
    output_is_logits: bool = False,
    aggregation: ProbabilityAggregation = "last",
    created_at_utc: str | datetime | None = None,
    eruption_id: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VolcanoAlertInferenceResult:
    """
    Load sequences from an NPZ file and run volcano alert inference.
    """
    inference_input = load_npz_sequences(
        npz_path=npz_path,
        array_key=array_key,
    )

    merged_metadata: dict[str, Any] = {
        "source_path": inference_input.source_path,
        "array_key": inference_input.array_key,
        **inference_input.metadata,
    }

    if metadata:
        merged_metadata.update(metadata)

    return predict_volcano_alert(
        model=model,
        sequences=inference_input.sequences,
        n_classes=n_classes,
        min_class_alert=min_class_alert,
        alert_threshold_24h=alert_threshold_24h,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
        output_is_logits=output_is_logits,
        aggregation=aggregation,
        created_at_utc=created_at_utc,
        eruption_id=eruption_id,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        metadata=merged_metadata,
    )


def predict_and_write_dashboard_outputs(
    model: Any,
    sequences: np.ndarray,
    *,
    s3_client: S3ClientProtocol,
    bucket: str,
    n_classes: int,
    min_class_alert: int,
    alert_threshold_24h: float,
    expected_seq_len: int | None = None,
    expected_n_features: int | None = None,
    output_is_logits: bool = False,
    aggregation: ProbabilityAggregation = "last",
    created_at_utc: str | datetime | None = None,
    generated_at_utc: str | datetime | None = None,
    eruption_id: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_report_metadata: dict[str, Any] | None = None,
) -> VolcanoAlertInferenceResult:
    """
    Run inference and write dashboard outputs to S3.
    """
    result = predict_volcano_alert(
        model=model,
        sequences=sequences,
        n_classes=n_classes,
        min_class_alert=min_class_alert,
        alert_threshold_24h=alert_threshold_24h,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
        output_is_logits=output_is_logits,
        aggregation=aggregation,
        created_at_utc=created_at_utc,
        eruption_id=eruption_id,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        metadata=metadata,
    )

    dashboard_locations = write_dashboard_outputs(
        client=s3_client,
        bucket=bucket,
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


def _select_npz_array_key(
    *,
    available_keys: list[str],
    npz_file: Any,
    requested_key: str | None,
) -> str:
    if not available_keys:
        raise ValueError("NPZ file does not contain any array.")

    if requested_key is not None:
        cleaned_key = requested_key.strip()

        if not cleaned_key:
            raise ValueError("array_key cannot be empty.")

        if cleaned_key not in available_keys:
            raise KeyError(
                f"Array key {cleaned_key!r} not found in NPZ file. "
                f"Available keys: {available_keys}."
            )

        return cleaned_key

    preferred_keys = ("X", "x", "sequences", "inputs", "features")

    for key in preferred_keys:
        if key in available_keys:
            return key

    three_dimensional_keys = [
        key
        for key in available_keys
        if np.asarray(npz_file[key]).ndim == 3
    ]

    if len(three_dimensional_keys) == 1:
        return three_dimensional_keys[0]

    if len(three_dimensional_keys) > 1:
        raise ValueError(
            "Multiple 3D arrays found in NPZ file. "
            "Provide array_key explicitly."
        )

    raise ValueError(
        "No valid 3D sequence array found in NPZ file. "
        "Expected shape: (batch_size, seq_len, n_features)."
    )


def _softmax_2d(values: np.ndarray) -> np.ndarray:
    shifted_values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted_values)
    sums = exp_values.sum(axis=1, keepdims=True)

    if not np.all(np.isfinite(sums)) or np.any(sums <= 0):
        raise ValueError("Cannot apply softmax to model output.")

    return exp_values / sums


def _validate_positive_integer(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be strictly positive.")


def _validate_probability_threshold(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")

    float_value = float(value)

    if not isfinite(float_value):
        raise ValueError(f"{field_name} must be finite.")

    if not 0 <= float_value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")

    return float_value