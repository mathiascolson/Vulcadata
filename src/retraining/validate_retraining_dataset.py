from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import great_expectations as gx
import numpy as np
import pandas as pd


DEFAULT_NPZ_PATH = "data/retraining/ready/volcano_multi_retraining.npz"
DEFAULT_OUTPUT_JSON = "reports/retraining/retraining_dataset_validation.json"
DEFAULT_EXPECTED_SEQ_LEN = 120
DEFAULT_EXPECTED_N_FEATURES = 992
DEFAULT_N_CLASSES = 6

REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_KEYS = tuple(
    key
    for split_name in REQUIRED_SPLITS
    for key in (f"X_{split_name}", f"y_{split_name}")
)

PROFILE_COLUMNS = [
    "split_name",
    "n_sequences",
    "seq_len",
    "n_features",
    "x_dtype",
    "y_dtype",
    "x_nan_count",
    "x_inf_count",
    "y_nan_count",
    "y_inf_count",
    "x_finite_count",
    "x_total_values",
    "x_finite_ratio",
    "y_finite_count",
    "y_total_values",
    "y_finite_ratio",
    "x_min_value",
    "x_max_value",
    "x_mean_value",
    "x_std_value",
    "x_abs_max",
    "label_min",
    "label_max",
    "n_unique_labels",
]


class RetrainingDatasetValidationError(ValueError):
    """Raised when a retraining NPZ is not valid for candidate training."""


def write_json(payload: dict[str, Any], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_npz_payload(npz_path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise RetrainingDatasetValidationError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as payload:
        available_keys = set(payload.files)
        missing_keys = [key for key in REQUIRED_KEYS if key not in available_keys]

        if missing_keys:
            raise RetrainingDatasetValidationError(
                f"Missing required keys in {npz_path}: {missing_keys}. "
                f"Available keys: {sorted(available_keys)}"
            )

        return {key: payload[key] for key in REQUIRED_KEYS}


def validate_split_arrays(
    *,
    split_name: str,
    x: np.ndarray,
    y: np.ndarray,
    expected_seq_len: int,
    expected_n_features: int,
    n_classes: int,
) -> None:
    if not isinstance(x, np.ndarray):
        raise RetrainingDatasetValidationError(
            f"X_{split_name} must be a numpy.ndarray, got {type(x).__name__}."
        )

    if not isinstance(y, np.ndarray):
        raise RetrainingDatasetValidationError(
            f"y_{split_name} must be a numpy.ndarray, got {type(y).__name__}."
        )

    if x.ndim != 3:
        raise RetrainingDatasetValidationError(
            f"X_{split_name} must be 3D with shape (n_sequences, seq_len, n_features). "
            f"Got shape {x.shape}."
        )

    if y.ndim != 1:
        raise RetrainingDatasetValidationError(
            f"y_{split_name} must be 1D with shape (n_sequences,). Got shape {y.shape}."
        )

    if x.shape[0] != y.shape[0]:
        raise RetrainingDatasetValidationError(
            f"X_{split_name} and y_{split_name} row counts do not match: "
            f"{x.shape[0]} vs {y.shape[0]}."
        )

    if x.shape[0] <= 0:
        raise RetrainingDatasetValidationError(
            f"Split {split_name} must contain at least one sequence."
        )

    if x.shape[1] != expected_seq_len or x.shape[2] != expected_n_features:
        raise RetrainingDatasetValidationError(
            f"Invalid X_{split_name} shape {x.shape}. Expected "
            f"(*, {expected_seq_len}, {expected_n_features})."
        )

    if not np.issubdtype(x.dtype, np.number):
        raise RetrainingDatasetValidationError(
            f"X_{split_name} dtype must be numeric, got {x.dtype}."
        )

    if not np.issubdtype(y.dtype, np.number):
        raise RetrainingDatasetValidationError(
            f"y_{split_name} dtype must be numeric, got {y.dtype}."
        )

    if not np.isfinite(x).all():
        raise RetrainingDatasetValidationError(
            f"X_{split_name} contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise RetrainingDatasetValidationError(
            f"y_{split_name} contains NaN or infinite values."
        )

    if not np.all(np.equal(y, np.round(y))):
        raise RetrainingDatasetValidationError(
            f"y_{split_name} contains non-integer class labels."
        )

    label_min = int(np.min(y))
    label_max = int(np.max(y))

    if label_min < 0 or label_max >= n_classes:
        raise RetrainingDatasetValidationError(
            f"y_{split_name} labels must be between 0 and {n_classes - 1}. "
            f"Got min={label_min}, max={label_max}."
        )


def build_retraining_dataset_profile(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for split_name in REQUIRED_SPLITS:
        x = payload[f"X_{split_name}"]
        y = payload[f"y_{split_name}"]
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)

        x_finite_mask = np.isfinite(flat_x)
        y_finite_mask = np.isfinite(flat_y)
        x_finite_values = flat_x[x_finite_mask]
        y_finite_values = flat_y[y_finite_mask]

        row = {
            "split_name": split_name,
            "n_sequences": int(x.shape[0]),
            "seq_len": int(x.shape[1]),
            "n_features": int(x.shape[2]),
            "x_dtype": str(x.dtype),
            "y_dtype": str(y.dtype),
            "x_nan_count": int(np.isnan(flat_x).sum()),
            "x_inf_count": int(np.isinf(flat_x).sum()),
            "y_nan_count": int(np.isnan(flat_y).sum()),
            "y_inf_count": int(np.isinf(flat_y).sum()),
            "x_finite_count": int(x_finite_mask.sum()),
            "x_total_values": int(flat_x.size),
            "x_finite_ratio": float(x_finite_mask.sum() / flat_x.size),
            "y_finite_count": int(y_finite_mask.sum()),
            "y_total_values": int(flat_y.size),
            "y_finite_ratio": float(y_finite_mask.sum() / flat_y.size),
            "x_min_value": float(np.min(x_finite_values)),
            "x_max_value": float(np.max(x_finite_values)),
            "x_mean_value": float(np.mean(x_finite_values)),
            "x_std_value": float(np.std(x_finite_values)),
            "x_abs_max": float(np.max(np.abs(x_finite_values))),
            "label_min": int(np.min(y_finite_values)),
            "label_max": int(np.max(y_finite_values)),
            "n_unique_labels": int(len(np.unique(y))),
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def build_gx_context():
    return gx.get_context(mode="ephemeral")


def build_expectation_suite(
    context: Any,
    *,
    suite_name: str,
    expected_seq_len: int,
    expected_n_features: int,
    n_classes: int,
) -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(column_list=PROFILE_COLUMNS)
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToEqual(value=3))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="split_name",
            value_set=list(REQUIRED_SPLITS),
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="n_sequences",
            min_value=1,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="seq_len",
            min_value=expected_seq_len,
            max_value=expected_seq_len,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="n_features",
            min_value=expected_n_features,
            max_value=expected_n_features,
        )
    )

    for column in ["x_nan_count", "x_inf_count", "y_nan_count", "y_inf_count"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                max_value=0,
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="x_finite_ratio",
            min_value=1.0,
            max_value=1.0,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="y_finite_ratio",
            min_value=1.0,
            max_value=1.0,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="label_min",
            min_value=0,
            max_value=n_classes - 1,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="label_max",
            min_value=0,
            max_value=n_classes - 1,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="n_unique_labels",
            min_value=1,
            max_value=n_classes,
        )
    )

    return suite


def run_gx_validation(
    profile_df: pd.DataFrame,
    *,
    expected_seq_len: int,
    expected_n_features: int,
    n_classes: int,
) -> dict[str, Any]:
    context = build_gx_context()
    unique_suffix = uuid4().hex

    data_source = context.data_sources.add_pandas(
        name=f"vulcadata_retraining_pandas_source_{unique_suffix}"
    )
    data_asset = data_source.add_dataframe_asset(
        name=f"retraining_dataset_profile_{unique_suffix}"
    )
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"retraining_dataset_profile_batch_{unique_suffix}"
    )

    suite = build_expectation_suite(
        context,
        suite_name=f"vulcadata_retraining_dataset_suite_{unique_suffix}",
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
        n_classes=n_classes,
    )

    validation_definition = gx.ValidationDefinition(
        name=f"retraining_dataset_validation_{unique_suffix}",
        data=batch_definition,
        suite=suite,
    )

    validation_results = validation_definition.run(batch_parameters={"dataframe": profile_df})
    success = bool(validation_results.success)

    try:
        statistics = dict(getattr(validation_results, "statistics", {}) or {})
    except Exception:
        statistics = {}

    return {
        "gx_success": success,
        "gx_result": {
            "success": success,
            "statistics": statistics,
        },
    }


def validate_retraining_dataset(
    *,
    npz_path: str | Path,
    expected_seq_len: int,
    expected_n_features: int,
    n_classes: int,
) -> dict[str, Any]:
    payload = load_npz_payload(npz_path)

    for split_name in REQUIRED_SPLITS:
        validate_split_arrays(
            split_name=split_name,
            x=payload[f"X_{split_name}"],
            y=payload[f"y_{split_name}"],
            expected_seq_len=expected_seq_len,
            expected_n_features=expected_n_features,
            n_classes=n_classes,
        )

    profile_df = build_retraining_dataset_profile(payload)
    gx_summary = run_gx_validation(
        profile_df,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
        n_classes=n_classes,
    )

    if not gx_summary["gx_success"]:
        raise RetrainingDatasetValidationError(
            "Great Expectations validation failed. "
            f"Summary: {json.dumps(gx_summary, ensure_ascii=False)}"
        )

    split_profiles = profile_df.to_dict(orient="records")
    split_shapes = {
        split_name: {
            "X_shape": list(payload[f"X_{split_name}"].shape),
            "y_shape": list(payload[f"y_{split_name}"].shape),
        }
        for split_name in REQUIRED_SPLITS
    }

    return {
        "status": "success",
        "npz_path": str(npz_path),
        "required_keys": list(REQUIRED_KEYS),
        "expected_seq_len": expected_seq_len,
        "expected_n_features": expected_n_features,
        "n_classes": n_classes,
        "split_shapes": split_shapes,
        "split_profiles": split_profiles,
        **gx_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Vulcadata retraining NPZ dataset with Great Expectations."
    )
    parser.add_argument(
        "--npz-path",
        default=os.getenv("VULCADATA_RETRAINING_GX_NPZ", DEFAULT_NPZ_PATH),
    )
    parser.add_argument(
        "--expected-seq-len",
        type=int,
        default=int(os.getenv("VULCADATA_EXPECTED_SEQ_LEN", str(DEFAULT_EXPECTED_SEQ_LEN))),
    )
    parser.add_argument(
        "--expected-n-features",
        type=int,
        default=int(os.getenv("VULCADATA_EXPECTED_N_FEATURES", str(DEFAULT_EXPECTED_N_FEATURES))),
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=int(os.getenv("VULCADATA_RETRAINING_N_CLASSES", str(DEFAULT_N_CLASSES))),
    )
    parser.add_argument(
        "--output-json",
        default=os.getenv("VULCADATA_RETRAINING_GX_VALIDATION_JSON", DEFAULT_OUTPUT_JSON),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        summary = validate_retraining_dataset(
            npz_path=args.npz_path,
            expected_seq_len=args.expected_seq_len,
            expected_n_features=args.expected_n_features,
            n_classes=args.n_classes,
        )
        write_json(summary, args.output_json)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as exc:
        failure_summary = {
            "status": "failed",
            "npz_path": str(args.npz_path),
            "expected_seq_len": args.expected_seq_len,
            "expected_n_features": args.expected_n_features,
            "n_classes": args.n_classes,
            "gx_success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        write_json(failure_summary, args.output_json)
        print(json.dumps(failure_summary, indent=2, ensure_ascii=False))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
