from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REFERENCE_NPZ = Path("data/preprocessing/processed_full_stride5_with_quiet/volcano_multi.npz")
DEFAULT_CURRENT_NPZ = Path("data/preprocessing/processed/latest_batch.npz")
DEFAULT_REFERENCE_ARRAY_KEY = "X_train"
DEFAULT_CURRENT_ARRAY_KEY = "X"
DEFAULT_OUTPUT_DIR = Path("reports/monitoring/evidently")
DEFAULT_REPORT_NAME = "latest_data_drift"
DEFAULT_MAX_ROWS = 5000
DEFAULT_MAX_FEATURES = 50
DEFAULT_S3_PREFIX = "monitoring/evidently"


@dataclass(frozen=True)
class EvidentlyOutputPaths:
    output_dir: Path
    html_path: Path
    json_path: Path
    summary_path: Path


def load_project_dotenv(dotenv_path: str = ".env") -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(dotenv_path=dotenv_path, override=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_output_paths(output_dir: Path, report_name: str) -> EvidentlyOutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)

    return EvidentlyOutputPaths(
        output_dir=output_dir,
        html_path=output_dir / f"{report_name}.html",
        json_path=output_dir / f"{report_name}.json",
        summary_path=output_dir / f"{report_name}_summary.json",
    )


def load_npz_array(npz_path: Path, array_key: str | None) -> np.ndarray:
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as npz_file:
        available_keys = list(npz_file.files)

        if array_key is not None:
            if array_key not in available_keys:
                raise KeyError(
                    f"Array key '{array_key}' not found in {npz_path}. "
                    f"Available keys: {available_keys}"
                )

            array = npz_file[array_key]
        else:
            array = select_first_numeric_array(npz_file=npz_file, available_keys=available_keys, npz_path=npz_path)

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Array must be numeric. Got dtype={array.dtype} in {npz_path}")

    return np.asarray(array, dtype=np.float32)


def select_first_numeric_array(npz_file: Any, available_keys: list[str], npz_path: Path) -> np.ndarray:
    preferred_keys = ["X", "X_test", "X_val", "X_train"]

    for key in preferred_keys:
        if key in available_keys and np.issubdtype(npz_file[key].dtype, np.number):
            return npz_file[key]

    for key in available_keys:
        if np.issubdtype(npz_file[key].dtype, np.number):
            return npz_file[key]

    raise ValueError(f"No numeric array found in {npz_path}. Available keys: {available_keys}")


def convert_array_to_2d_features(array: np.ndarray, source_name: str) -> np.ndarray:
    if array.ndim == 3:
        n_sequences, seq_len, n_features = array.shape

        if n_sequences <= 0 or seq_len <= 0 or n_features <= 0:
            raise ValueError(f"Invalid 3D array shape for {source_name}: {array.shape}")

        return array.reshape(n_sequences * seq_len, n_features)

    if array.ndim == 2:
        n_rows, n_features = array.shape

        if n_rows <= 0 or n_features <= 0:
            raise ValueError(f"Invalid 2D array shape for {source_name}: {array.shape}")

        return array

    if array.ndim == 1:
        if array.shape[0] <= 0:
            raise ValueError(f"Invalid 1D array shape for {source_name}: {array.shape}")

        return array.reshape(-1, 1)

    raise ValueError(
        f"Unsupported array shape for {source_name}: {array.shape}. "
        "Expected 1D, 2D or 3D numeric array."
    )


def limit_rows(array_2d: np.ndarray, max_rows: int) -> np.ndarray:
    if not isinstance(max_rows, int) or isinstance(max_rows, bool):
        raise TypeError("max_rows must be an integer.")

    if max_rows <= 0:
        raise ValueError("max_rows must be strictly positive.")

    if array_2d.shape[0] <= max_rows:
        return array_2d

    row_indices = np.linspace(0, array_2d.shape[0] - 1, num=max_rows, dtype=int)
    return array_2d[row_indices]


def select_feature_indices(reference_2d: np.ndarray, max_features: int) -> np.ndarray:
    if not isinstance(max_features, int) or isinstance(max_features, bool):
        raise TypeError("max_features must be an integer.")

    if max_features <= 0:
        raise ValueError("max_features must be strictly positive.")

    n_features = reference_2d.shape[1]

    if n_features <= max_features:
        return np.arange(n_features)

    cleaned_reference = np.where(np.isfinite(reference_2d), reference_2d, np.nan)
    variances = np.nanvar(cleaned_reference, axis=0)
    variances = np.nan_to_num(variances, nan=-1.0, posinf=-1.0, neginf=-1.0)

    selected_indices = np.argsort(variances)[-max_features:]
    return np.sort(selected_indices)


def build_monitoring_dataframe(array_2d: np.ndarray, feature_indices: np.ndarray) -> pd.DataFrame:
    selected_array = array_2d[:, feature_indices]
    selected_array = np.where(np.isfinite(selected_array), selected_array, np.nan)

    columns = [f"feature_{int(feature_index):04d}" for feature_index in feature_indices]
    dataframe = pd.DataFrame(selected_array, columns=columns)

    dataframe = dataframe.dropna(axis=1, how="all")

    if dataframe.empty:
        raise ValueError("Monitoring dataframe is empty after feature selection.")

    return dataframe


def prepare_evidently_datasets(
    reference_array: np.ndarray,
    current_array: np.ndarray,
    max_rows: int,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference_2d = convert_array_to_2d_features(reference_array, "reference")
    current_2d = convert_array_to_2d_features(current_array, "current")

    if reference_2d.shape[1] != current_2d.shape[1]:
        raise ValueError(
            "Reference and current arrays must have the same feature count. "
            f"Reference feature count: {reference_2d.shape[1]}. "
            f"Current feature count: {current_2d.shape[1]}."
        )

    reference_limited = limit_rows(reference_2d, max_rows=max_rows)
    current_limited = limit_rows(current_2d, max_rows=max_rows)

    feature_indices = select_feature_indices(reference_limited, max_features=max_features)

    reference_df = build_monitoring_dataframe(reference_limited, feature_indices)
    current_df = build_monitoring_dataframe(current_limited, feature_indices)

    common_columns = sorted(set(reference_df.columns).intersection(set(current_df.columns)))

    if not common_columns:
        raise ValueError("No common monitoring columns found between reference and current dataframes.")

    reference_df = reference_df[common_columns]
    current_df = current_df[common_columns]

    metadata = {
        "reference_original_shape": list(reference_array.shape),
        "current_original_shape": list(current_array.shape),
        "reference_2d_shape": list(reference_2d.shape),
        "current_2d_shape": list(current_2d.shape),
        "reference_monitoring_shape": list(reference_df.shape),
        "current_monitoring_shape": list(current_df.shape),
        "selected_feature_count": int(len(common_columns)),
        "selected_feature_columns": common_columns,
        "max_rows": int(max_rows),
        "max_features": int(max_features),
    }

    return reference_df, current_df, metadata


def import_evidently_components() -> tuple[Any, Any, str]:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        return Report, DataDriftPreset, "new"
    except ImportError:
        pass

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        return Report, DataDriftPreset, "legacy"
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required to generate monitoring reports. "
            "Install it with: pip install evidently"
        ) from exc


def run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    html_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    Report, DataDriftPreset, api_mode = import_evidently_components()

    if api_mode == "new":
        report = Report([DataDriftPreset()])
        report_result = report.run(current_data=current_df, reference_data=reference_df)
    else:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report_result = report

    save_report_html(report_result, html_path)
    save_report_json(report_result, json_path)

    summary = extract_evidently_summary(report_result)
    summary["evidently_api_mode"] = api_mode

    return summary


def save_report_html(report_result: Any, html_path: Path) -> None:
    if hasattr(report_result, "save_html"):
        report_result.save_html(str(html_path))
        return

    raise RuntimeError(
        "The installed Evidently report result does not provide save_html(). "
        "Use a compatible Evidently version or check the installed API."
    )


def save_report_json(report_result: Any, json_path: Path) -> None:
    if hasattr(report_result, "save_json"):
        report_result.save_json(str(json_path))
        return

    if hasattr(report_result, "json"):
        json_content = report_result.json()
        json_path.write_text(json_content, encoding="utf-8")
        return

    if hasattr(report_result, "as_dict"):
        json_path.write_text(
            json.dumps(report_result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    if hasattr(report_result, "dict"):
        json_path.write_text(
            json.dumps(report_result.dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    raise RuntimeError(
        "The installed Evidently report result cannot be exported to JSON. "
        "Use a compatible Evidently version or check the installed API."
    )


def report_result_to_dict(report_result: Any) -> dict[str, Any] | None:
    if hasattr(report_result, "dict"):
        try:
            candidate = report_result.dict()
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            pass

    if hasattr(report_result, "as_dict"):
        try:
            candidate = report_result.as_dict()
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            pass

    if hasattr(report_result, "json"):
        try:
            candidate = json.loads(report_result.json())
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            pass

    return None


def extract_evidently_summary(report_result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_drift": None,
        "number_of_drifted_columns": None,
        "share_of_drifted_columns": None,
        "drifted_columns": [],
    }

    report_dict = report_result_to_dict(report_result)

    if report_dict is None:
        return summary

    for value in iter_nested_values(report_dict):
        if not isinstance(value, dict):
            continue

        normalized_items = {str(key).lower(): nested_value for key, nested_value in value.items()}

        if summary["dataset_drift"] is None:
            summary["dataset_drift"] = normalized_items.get("dataset_drift", normalized_items.get("datasetdrift"))

        if summary["number_of_drifted_columns"] is None:
            summary["number_of_drifted_columns"] = normalized_items.get(
                "number_of_drifted_columns",
                normalized_items.get("numberofdriftedcolumns"),
            )

        if summary["share_of_drifted_columns"] is None:
            summary["share_of_drifted_columns"] = normalized_items.get(
                "share_of_drifted_columns",
                normalized_items.get("shareofdriftedcolumns"),
            )

        drift_by_columns = normalized_items.get("drift_by_columns", normalized_items.get("driftbycolumns"))

        if isinstance(drift_by_columns, dict) and not summary["drifted_columns"]:
            drifted_columns = []

            for column_name, column_payload in drift_by_columns.items():
                if isinstance(column_payload, dict) and bool(column_payload.get("drift_detected")):
                    drifted_columns.append(str(column_name))

            summary["drifted_columns"] = drifted_columns

    return summary


def iter_nested_values(value: Any) -> Any:
    yield value

    if isinstance(value, dict):
        for nested_value in value.values():
            yield from iter_nested_values(nested_value)

    if isinstance(value, list):
        for nested_value in value:
            yield from iter_nested_values(nested_value)


def upload_file_to_s3(local_path: Path, bucket: str, key: str, content_type: str | None = None) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required to upload Evidently reports to S3. "
            "Install it with: pip install boto3"
        ) from exc

    client = boto3.client("s3")

    extra_args = {}

    if content_type:
        extra_args["ContentType"] = content_type

    if extra_args:
        client.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
    else:
        client.upload_file(str(local_path), bucket, key)

    return f"s3://{bucket}/{key}"


def upload_reports_to_s3(
    paths: EvidentlyOutputPaths,
    bucket: str,
    prefix: str,
    report_name: str,
) -> dict[str, str]:
    cleaned_prefix = prefix.strip("/")

    html_key = f"{cleaned_prefix}/{report_name}.html"
    json_key = f"{cleaned_prefix}/{report_name}.json"
    summary_key = f"{cleaned_prefix}/{report_name}_summary.json"

    return {
        "html_s3_uri": upload_file_to_s3(
            paths.html_path,
            bucket,
            html_key,
            content_type="text/html",
        ),
        "json_s3_uri": upload_file_to_s3(
            paths.json_path,
            bucket,
            json_key,
            content_type="application/json",
        ),
        "summary_s3_uri": upload_file_to_s3(
            paths.summary_path,
            bucket,
            summary_key,
            content_type="application/json",
        ),
    }


def generate_evidently_report(
    reference_npz: Path,
    current_npz: Path,
    reference_array_key: str | None,
    current_array_key: str | None,
    output_dir: Path,
    report_name: str,
    max_rows: int,
    max_features: int,
    write_s3: bool,
    s3_bucket: str | None,
    s3_prefix: str,
) -> dict[str, Any]:
    generated_at_utc = utc_now_iso()
    paths = build_output_paths(output_dir=output_dir, report_name=report_name)

    reference_array = load_npz_array(reference_npz, reference_array_key)
    current_array = load_npz_array(current_npz, current_array_key)

    reference_df, current_df, dataset_metadata = prepare_evidently_datasets(
        reference_array=reference_array,
        current_array=current_array,
        max_rows=max_rows,
        max_features=max_features,
    )

    evidently_summary = run_evidently_report(
        reference_df=reference_df,
        current_df=current_df,
        html_path=paths.html_path,
        json_path=paths.json_path,
    )

    result: dict[str, Any] = {
        "status": "success",
        "generated_at_utc": generated_at_utc,
        "report_name": report_name,
        "reference_npz": str(reference_npz),
        "current_npz": str(current_npz),
        "reference_array_key": reference_array_key,
        "current_array_key": current_array_key,
        "local_outputs": {
            "html_path": str(paths.html_path),
            "json_path": str(paths.json_path),
            "summary_path": str(paths.summary_path),
        },
        "dataset_metadata": dataset_metadata,
        "evidently_summary": evidently_summary,
        "s3_outputs": None,
    }

    paths.summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if write_s3:
        if not s3_bucket:
            raise ValueError("--s3-bucket is required when --write-s3 is used.")

        s3_outputs = upload_reports_to_s3(
            paths=paths,
            bucket=s3_bucket,
            prefix=s3_prefix,
            report_name=report_name,
        )

        result["s3_outputs"] = s3_outputs

        paths.summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Evidently data drift report for Vulcadata inference batches."
    )

    parser.add_argument(
        "--reference-npz",
        type=Path,
        default=DEFAULT_REFERENCE_NPZ,
        help=f"Reference NPZ file. Default: {DEFAULT_REFERENCE_NPZ}",
    )

    parser.add_argument(
        "--current-npz",
        type=Path,
        default=DEFAULT_CURRENT_NPZ,
        help=f"Current NPZ file. Default: {DEFAULT_CURRENT_NPZ}",
    )

    parser.add_argument(
        "--reference-array-key",
        default=DEFAULT_REFERENCE_ARRAY_KEY,
        help=f"Reference array key. Default: {DEFAULT_REFERENCE_ARRAY_KEY}",
    )

    parser.add_argument(
        "--current-array-key",
        default=DEFAULT_CURRENT_ARRAY_KEY,
        help=f"Current array key. Default: {DEFAULT_CURRENT_ARRAY_KEY}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help=f"Report base name without extension. Default: {DEFAULT_REPORT_NAME}",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Maximum rows per dataset after sequence flattening. Default: {DEFAULT_MAX_ROWS}",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help=f"Maximum monitored features selected by reference variance. Default: {DEFAULT_MAX_FEATURES}",
    )

    parser.add_argument(
        "--write-s3",
        action="store_true",
        help="Upload HTML, JSON and summary reports to S3.",
    )

    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("VULCADATA_S3_BUCKET"),
        help="S3 bucket. Default: VULCADATA_S3_BUCKET environment variable.",
    )

    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help=f"S3 prefix. Default: {DEFAULT_S3_PREFIX}",
    )

    return parser.parse_args()


def main() -> int:
    load_project_dotenv()

    args = parse_args()

    try:
        result = generate_evidently_report(
            reference_npz=args.reference_npz,
            current_npz=args.current_npz,
            reference_array_key=args.reference_array_key,
            current_array_key=args.current_array_key,
            output_dir=args.output_dir,
            report_name=args.report_name,
            max_rows=args.max_rows,
            max_features=args.max_features,
            write_s3=args.write_s3,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
        )
    except Exception as exc:
        print(f"Erreur Evidently monitoring : {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())