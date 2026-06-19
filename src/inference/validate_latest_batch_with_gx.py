from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
import great_expectations as gx
import numpy as np
import pandas as pd


DEFAULT_NPZ_PATH = "data/preprocessing/processed/latest_batch.npz"
DEFAULT_ARRAY_KEY = "X"
DEFAULT_OUTPUT_JSON = "reports/validation/latest_batch_gx_validation.json"
DEFAULT_EXPECTED_BATCH_SIZE = 1
DEFAULT_EXPECTED_SEQ_LEN = 120
DEFAULT_EXPECTED_N_FEATURES = 992


PROFILE_COLUMNS = [
    "sequence_index",
    "seq_len",
    "n_features",
    "dtype",
    "nan_count",
    "inf_count",
    "finite_count",
    "total_values",
    "finite_ratio",
    "min_value",
    "max_value",
    "mean_value",
    "std_value",
    "zero_ratio",
    "abs_max",
]


class LatestBatchValidationError(ValueError):
    """Raised when latest_batch.npz is not valid for inference."""


def load_npz_array(npz_path: str | Path, array_key: str) -> np.ndarray:
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise LatestBatchValidationError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path) as data:
        available_keys = list(data.files)

        if array_key not in available_keys:
            raise LatestBatchValidationError(
                f"Array key '{array_key}' not found in {npz_path}. "
                f"Available keys: {available_keys}"
            )

        array = data[array_key]

    return array


def validate_numpy_array_contract(
    array: np.ndarray,
    *,
    expected_batch_size: int,
    expected_seq_len: int,
    expected_n_features: int,
) -> None:
    if not isinstance(array, np.ndarray):
        raise LatestBatchValidationError(
            f"Expected numpy.ndarray, got {type(array).__name__}."
        )

    if array.ndim != 3:
        raise LatestBatchValidationError(
            f"Expected a 3D array shaped (batch, seq_len, n_features), "
            f"got ndim={array.ndim}, shape={array.shape}."
        )

    expected_shape = (
        expected_batch_size,
        expected_seq_len,
        expected_n_features,
    )

    if array.shape != expected_shape:
        raise LatestBatchValidationError(
            f"Invalid array shape. Expected {expected_shape}, got {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise LatestBatchValidationError(
            f"Array dtype must be numeric, got {array.dtype}."
        )

    if not np.isfinite(array).all():
        raise LatestBatchValidationError(
            "Array contains NaN or infinite values."
        )


def build_latest_batch_profile(array: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    batch_size, seq_len, n_features = array.shape

    for sequence_index in range(batch_size):
        sequence = array[sequence_index]
        flat = sequence.reshape(-1)

        total_values = int(flat.size)
        finite_mask = np.isfinite(flat)
        finite_count = int(finite_mask.sum())
        nan_count = int(np.isnan(flat).sum())
        inf_count = int(np.isinf(flat).sum())

        if finite_count == 0:
            raise LatestBatchValidationError(
                f"Sequence {sequence_index} contains no finite value."
            )

        finite_values = flat[finite_mask]

        row = {
            "sequence_index": int(sequence_index),
            "seq_len": int(seq_len),
            "n_features": int(n_features),
            "dtype": str(array.dtype),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_count": finite_count,
            "total_values": total_values,
            "finite_ratio": float(finite_count / total_values),
            "min_value": float(np.min(finite_values)),
            "max_value": float(np.max(finite_values)),
            "mean_value": float(np.mean(finite_values)),
            "std_value": float(np.std(finite_values)),
            "zero_ratio": float(np.mean(flat == 0)),
            "abs_max": float(np.max(np.abs(finite_values))),
        }

        rows.append(row)

    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def build_gx_context():
    return gx.get_context(mode="ephemeral")


def build_expectation_suite(
    context: Any,
    *,
    suite_name: str,
    expected_batch_size: int,
    expected_seq_len: int,
    expected_n_features: int,
) -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=PROFILE_COLUMNS
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToEqual(
            value=expected_batch_size
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="sequence_index",
            min_value=0,
            max_value=expected_batch_size - 1,
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

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="nan_count",
            min_value=0,
            max_value=0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="inf_count",
            min_value=0,
            max_value=0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="finite_ratio",
            min_value=1.0,
            max_value=1.0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="zero_ratio",
            min_value=0.0,
            max_value=1.0,
        )
    )

    for column in [
        "min_value",
        "max_value",
        "mean_value",
        "std_value",
        "abs_max",
    ]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="std_value",
            min_value=0.0,
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="abs_max",
            min_value=0.0,
        )
    )

    return suite


def run_gx_validation(
    profile_df: pd.DataFrame,
    *,
    expected_batch_size: int,
    expected_seq_len: int,
    expected_n_features: int,
) -> dict[str, Any]:
    context = build_gx_context()

    unique_suffix = uuid4().hex

    data_source = context.data_sources.add_pandas(
        name=f"vulcadata_pandas_runtime_source_{unique_suffix}"
    )

    data_asset = data_source.add_dataframe_asset(
        name=f"latest_batch_profile_{unique_suffix}"
    )

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"latest_batch_profile_batch_{unique_suffix}"
    )

    suite = build_expectation_suite(
        context,
        suite_name=f"vulcadata_latest_batch_profile_suite_{unique_suffix}",
        expected_batch_size=expected_batch_size,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
    )

    validation_definition = gx.ValidationDefinition(
        name=f"latest_batch_profile_validation_{unique_suffix}",
        data=batch_definition,
        suite=suite,
    )

    validation_results = validation_definition.run(
        batch_parameters={"dataframe": profile_df}
    )

    success = bool(validation_results.success)

    try:
        statistics = dict(getattr(validation_results, "statistics", {}) or {})
    except Exception:
        statistics = {}

    result_description = {
        "success": success,
        "statistics": statistics,
    }

    return {
        "gx_success": success,
        "gx_result": result_description,
    }


def assert_profile_values_are_finite(profile_df: pd.DataFrame) -> None:
    numeric_columns = [
        "finite_ratio",
        "min_value",
        "max_value",
        "mean_value",
        "std_value",
        "zero_ratio",
        "abs_max",
    ]

    for column in numeric_columns:
        values = profile_df[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise LatestBatchValidationError(
                f"Profile column '{column}' contains non-finite values."
            )


def validate_latest_batch_with_gx(
    *,
    npz_path: str | Path,
    array_key: str,
    expected_batch_size: int,
    expected_seq_len: int,
    expected_n_features: int,
) -> dict[str, Any]:
    array = load_npz_array(npz_path=npz_path, array_key=array_key)

    validate_numpy_array_contract(
        array,
        expected_batch_size=expected_batch_size,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
    )

    profile_df = build_latest_batch_profile(array)
    assert_profile_values_are_finite(profile_df)

    gx_summary = run_gx_validation(
        profile_df,
        expected_batch_size=expected_batch_size,
        expected_seq_len=expected_seq_len,
        expected_n_features=expected_n_features,
    )

    if not gx_summary["gx_success"]:
        raise LatestBatchValidationError(
            "Great Expectations validation failed. "
            f"Summary: {json.dumps(gx_summary, ensure_ascii=False)}"
        )

    profile_records = profile_df.to_dict(orient="records")

    return {
        "status": "success",
        "npz_path": str(npz_path),
        "array_key": array_key,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "expected_batch_size": expected_batch_size,
        "expected_seq_len": expected_seq_len,
        "expected_n_features": expected_n_features,
        "profile": profile_records,
        **gx_summary,
    }


def write_json(payload: dict[str, Any], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate latest Vulcadata inference batch with Great Expectations."
    )

    parser.add_argument(
        "--npz-path",
        default=os.getenv("VULCADATA_LATEST_BATCH_NPZ", DEFAULT_NPZ_PATH),
    )

    parser.add_argument(
        "--array-key",
        default=DEFAULT_ARRAY_KEY,
    )

    parser.add_argument(
        "--expected-batch-size",
        type=int,
        default=int(
            os.getenv(
                "VULCADATA_LATEST_BATCH_SIZE",
                str(DEFAULT_EXPECTED_BATCH_SIZE),
            )
        ),
    )

    parser.add_argument(
        "--expected-seq-len",
        type=int,
        default=int(
            os.getenv(
                "VULCADATA_EXPECTED_SEQ_LEN",
                str(DEFAULT_EXPECTED_SEQ_LEN),
            )
        ),
    )

    parser.add_argument(
        "--expected-n-features",
        type=int,
        default=int(
            os.getenv(
                "VULCADATA_EXPECTED_N_FEATURES",
                str(DEFAULT_EXPECTED_N_FEATURES),
            )
        ),
    )

    parser.add_argument(
        "--output-json",
        default=os.getenv(
            "VULCADATA_GX_VALIDATION_OUTPUT_JSON",
            DEFAULT_OUTPUT_JSON,
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = validate_latest_batch_with_gx(
        npz_path=args.npz_path,
        array_key=args.array_key,
        expected_batch_size=args.expected_batch_size,
        expected_seq_len=args.expected_seq_len,
        expected_n_features=args.expected_n_features,
    )

    write_json(summary, args.output_json)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()