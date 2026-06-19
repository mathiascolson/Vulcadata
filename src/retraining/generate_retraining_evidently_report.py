from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_RESULT = "reports/retraining/candidate_training_result.json"
DEFAULT_CHAMPION_DECISION = "configs/final_model_decision.json"
DEFAULT_REFERENCE_NPZ = "data/preprocessing/processed_full_stride5_with_quiet/volcano_multi.npz"
DEFAULT_OUTPUT_DIR = "reports/retraining/evidently"
DEFAULT_OUTPUT_HTML = "candidate_drift_report.html"
DEFAULT_OUTPUT_JSON = "candidate_drift_summary.json"
DEFAULT_EVIDENTLY_JSON = "candidate_drift_report_evidently_raw.json"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")

    return payload


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def resolve_path(path: str | Path, project_root: str | Path = ".") -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return Path(project_root) / path


def get_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(current, dict):
            return None

        if key not in current:
            return None

        current = current[key]

    return current


def first_existing_path(payload: dict[str, Any], paths: list[list[str]]) -> str | None:
    for path in paths:
        value = get_nested(payload, path)

        if isinstance(value, str) and value.strip():
            return value

    return None


def extract_candidate_npz(candidate_result: dict[str, Any]) -> str:
    input_npz = first_existing_path(
        candidate_result,
        [
            ["input_npz_absolute_path"],
            ["input_npz"],
        ],
    )

    if input_npz:
        return input_npz

    files_to_process = candidate_result.get("files_to_process")

    if isinstance(files_to_process, list) and files_to_process:
        first_file = files_to_process[0]

        if isinstance(first_file, dict):
            path = first_file.get("path")

            if isinstance(path, str) and path.strip():
                return path

    raise ValueError(
        "Unable to locate candidate NPZ path in candidate training result."
    )


def extract_reference_npz(
    champion_decision: dict[str, Any],
    fallback_reference_npz: str,
) -> str:
    reference_npz = first_existing_path(
        champion_decision,
        [
            ["reference_npz"],
            ["dataset_npz"],
            ["npz_path"],
            ["input_npz"],
            ["classification_candidate", "reference_npz"],
            ["classification_candidate", "dataset_npz"],
            ["classification_candidate", "npz_path"],
            ["classification_candidate", "input_npz"],
            ["champion", "reference_npz"],
            ["champion", "dataset_npz"],
            ["champion", "npz_path"],
            ["champion", "input_npz"],
        ],
    )

    if reference_npz:
        return reference_npz

    return fallback_reference_npz


def extract_predictions_npz(candidate_result: dict[str, Any]) -> str | None:
    predictions_path = first_existing_path(
        candidate_result,
        [
            ["artifacts", "predictions_path"],
            ["predictions_path"],
        ],
    )

    return predictions_path


def choose_indices(n_rows: int, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_rows, size=max_rows, replace=False)
    return np.sort(indices)


def summarize_split(
    x: np.ndarray,
    y: np.ndarray,
    split_name: str,
    max_sequences: int | None,
    seed: int,
) -> pd.DataFrame:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D X array for split {split_name}. Got {x.shape}")

    if y.ndim != 1:
        raise ValueError(f"Expected 1D y array for split {split_name}. Got {y.shape}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y row counts differ for split {split_name}: "
            f"{x.shape[0]} vs {y.shape[0]}"
        )

    indices = choose_indices(x.shape[0], max_sequences, seed)
    x = np.asarray(x[indices], dtype=np.float32)
    y = np.asarray(y[indices], dtype=np.int64)

    denominator = float(x.shape[1] * x.shape[2])

    sequence_mean = x.mean(axis=(1, 2))
    sequence_std = x.std(axis=(1, 2))
    sequence_min = x.min(axis=(1, 2))
    sequence_max = x.max(axis=(1, 2))
    sequence_abs_mean = np.abs(x).mean(axis=(1, 2))
    sequence_energy_mean = np.einsum("ijk,ijk->i", x, x) / denominator
    first_timestep_mean = x[:, 0, :].mean(axis=1)
    last_timestep_mean = x[:, -1, :].mean(axis=1)
    temporal_delta_mean = (x[:, -1, :] - x[:, 0, :]).mean(axis=1)
    feature_std_mean = x.std(axis=1).mean(axis=1)
    timestep_std_mean = x.std(axis=2).mean(axis=1)

    return pd.DataFrame(
        {
            "split": split_name,
            "target": y,
            "sequence_mean": sequence_mean,
            "sequence_std": sequence_std,
            "sequence_min": sequence_min,
            "sequence_max": sequence_max,
            "sequence_abs_mean": sequence_abs_mean,
            "sequence_energy_mean": sequence_energy_mean,
            "first_timestep_mean": first_timestep_mean,
            "last_timestep_mean": last_timestep_mean,
            "temporal_delta_mean": temporal_delta_mean,
            "feature_std_mean": feature_std_mean,
            "timestep_std_mean": timestep_std_mean,
        }
    )


def summarize_npz(
    npz_path: str | Path,
    max_sequences_per_split: int | None,
    seed: int,
) -> pd.DataFrame:
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    frames: list[pd.DataFrame] = []

    with np.load(npz_path, allow_pickle=True) as data:
        for split_index, split_name in enumerate(("train", "val", "test")):
            x_key = f"X_{split_name}"
            y_key = f"y_{split_name}"

            if x_key not in data or y_key not in data:
                continue

            x = data[x_key]
            y = data[y_key]
            frame = summarize_split(
                x=x,
                y=y,
                split_name=split_name,
                max_sequences=max_sequences_per_split,
                seed=seed + split_index,
            )
            frames.append(frame)

    if not frames:
        raise ValueError(f"No train/val/test arrays found in NPZ: {npz_path}")

    return pd.concat(frames, axis=0, ignore_index=True)


def load_targets_from_npz(npz_path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(npz_path)

    targets: dict[str, np.ndarray] = {}

    with np.load(npz_path, allow_pickle=True) as data:
        for split_name in ("train", "val", "test"):
            y_key = f"y_{split_name}"

            if y_key in data:
                targets[split_name] = np.asarray(data[y_key], dtype=np.int64)

    return targets


def class_distribution(values: np.ndarray, labels: list[int]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.int64)

    if values.size == 0:
        return {str(label): 0.0 for label in labels}

    counts = {label: 0 for label in labels}

    unique_values, unique_counts = np.unique(values, return_counts=True)

    for value, count in zip(unique_values, unique_counts):
        value = int(value)

        if value not in counts:
            counts[value] = 0

        counts[value] += int(count)

    total = float(sum(counts.values()))

    if total <= 0:
        return {str(label): 0.0 for label in labels}

    return {
        str(label): float(counts.get(label, 0) / total)
        for label in sorted(counts)
    }


def total_variation_distance(
    left_distribution: dict[str, float],
    right_distribution: dict[str, float],
) -> float:
    keys = sorted(set(left_distribution) | set(right_distribution))
    total = 0.0

    for key in keys:
        total += abs(
            float(left_distribution.get(key, 0.0))
            - float(right_distribution.get(key, 0.0))
        )

    return 0.5 * total


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"target", "predicted_class"}
    columns = []

    for column in frame.columns:
        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)

    return columns


def dataset_drift_summary(
    reference_data: pd.DataFrame,
    candidate_data: pd.DataFrame,
    max_standardized_mean_diff: float,
    max_share_drifted_features: float,
) -> dict[str, Any]:
    columns = numeric_feature_columns(reference_data)
    feature_results: dict[str, Any] = {}
    drifted_features = 0

    for column in columns:
        reference_values = reference_data[column].astype(float).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        candidate_values = candidate_data[column].astype(float).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if reference_values.empty or candidate_values.empty:
            feature_results[column] = {
                "drifted": True,
                "reason": "empty_reference_or_candidate_values",
            }
            drifted_features += 1
            continue

        reference_mean = float(reference_values.mean())
        candidate_mean = float(candidate_values.mean())
        reference_std = float(reference_values.std(ddof=0))
        candidate_std = float(candidate_values.std(ddof=0))

        if math.isclose(reference_std, 0.0, abs_tol=1e-12):
            standardized_mean_diff = (
                0.0
                if math.isclose(reference_mean, candidate_mean, abs_tol=1e-12)
                else float("inf")
            )
        else:
            standardized_mean_diff = abs(candidate_mean - reference_mean) / reference_std

        drifted = standardized_mean_diff > max_standardized_mean_diff

        if drifted:
            drifted_features += 1

        feature_results[column] = {
            "drifted": drifted,
            "reference_mean": reference_mean,
            "candidate_mean": candidate_mean,
            "reference_std": reference_std,
            "candidate_std": candidate_std,
            "standardized_mean_diff": (
                None
                if math.isinf(standardized_mean_diff)
                else float(standardized_mean_diff)
            ),
            "threshold": max_standardized_mean_diff,
        }

    total_features = len(columns)
    share_drifted_features = (
        float(drifted_features / total_features)
        if total_features > 0
        else 1.0
    )
    dataset_drift = share_drifted_features > max_share_drifted_features

    return {
        "dataset_drift": dataset_drift,
        "drifted_features_count": drifted_features,
        "total_features": total_features,
        "share_drifted_features": share_drifted_features,
        "max_standardized_mean_diff": max_standardized_mean_diff,
        "max_share_drifted_features": max_share_drifted_features,
        "features": feature_results,
    }


def target_drift_summary(
    reference_data: pd.DataFrame,
    candidate_data: pd.DataFrame,
    max_target_distribution_distance: float,
) -> dict[str, Any]:
    labels = sorted(
        set(reference_data["target"].astype(int).tolist())
        | set(candidate_data["target"].astype(int).tolist())
    )

    reference_distribution = class_distribution(
        reference_data["target"].to_numpy(),
        labels,
    )
    candidate_distribution = class_distribution(
        candidate_data["target"].to_numpy(),
        labels,
    )
    distance = total_variation_distance(reference_distribution, candidate_distribution)
    drifted = distance > max_target_distribution_distance

    return {
        "target_drift": drifted,
        "distance": distance,
        "max_target_distribution_distance": max_target_distribution_distance,
        "reference_distribution": reference_distribution,
        "candidate_distribution": candidate_distribution,
    }


def find_prediction_classes(predictions_npz_path: str | Path) -> tuple[np.ndarray | None, str | None]:
    predictions_npz_path = Path(predictions_npz_path)

    if not predictions_npz_path.exists():
        return None, None

    class_keys = [
        "y_pred",
        "predictions",
        "predicted_class",
        "predicted_classes",
        "test_predictions",
        "test_y_pred",
    ]
    score_keys = [
        "probabilities",
        "probas",
        "y_proba",
        "test_probabilities",
        "test_probas",
        "logits",
        "test_logits",
    ]

    with np.load(predictions_npz_path, allow_pickle=True) as data:
        for key in class_keys:
            if key in data:
                values = np.asarray(data[key])

                if values.ndim == 1:
                    return values.astype(np.int64), key

                if values.ndim >= 2:
                    return values.argmax(axis=-1).reshape(-1).astype(np.int64), key

        for key in score_keys:
            if key in data:
                values = np.asarray(data[key])

                if values.ndim >= 2:
                    return values.argmax(axis=-1).reshape(-1).astype(np.int64), key

        for key in data.keys():
            lower_key = key.lower()
            values = np.asarray(data[key])

            if values.ndim >= 2 and (
                "prob" in lower_key
                or "proba" in lower_key
                or "logit" in lower_key
                or "pred" in lower_key
            ):
                return values.argmax(axis=-1).reshape(-1).astype(np.int64), key

    return None, None


def prediction_drift_summary(
    candidate_predictions: np.ndarray | None,
    prediction_source_key: str | None,
    candidate_targets_by_split: dict[str, np.ndarray],
    max_prediction_distribution_distance: float,
) -> dict[str, Any]:
    if candidate_predictions is None:
        return {
            "prediction_drift": None,
            "available": False,
            "reason": "candidate_predictions_not_found",
        }

    labels = sorted(
        set(candidate_predictions.astype(int).tolist())
        | set(np.concatenate(list(candidate_targets_by_split.values())).astype(int).tolist())
    )

    reference_targets: np.ndarray | None = None
    reference_split = None

    for split_name, values in candidate_targets_by_split.items():
        if len(values) == len(candidate_predictions):
            reference_targets = values
            reference_split = split_name
            break

    if reference_targets is None:
        reference_targets = np.concatenate(list(candidate_targets_by_split.values()))
        reference_split = "all_splits"

    prediction_distribution = class_distribution(candidate_predictions, labels)
    target_distribution = class_distribution(reference_targets, labels)
    distance = total_variation_distance(prediction_distribution, target_distribution)
    drifted = distance > max_prediction_distribution_distance

    return {
        "prediction_drift": drifted,
        "available": True,
        "prediction_source_key": prediction_source_key,
        "reference_split": reference_split,
        "distance": distance,
        "max_prediction_distribution_distance": max_prediction_distribution_distance,
        "prediction_distribution": prediction_distribution,
        "target_distribution": target_distribution,
    }


def generate_evidently_report(
    reference_data: pd.DataFrame,
    candidate_data: pd.DataFrame,
    output_html: Path,
    output_raw_json: Path,
) -> dict[str, Any]:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception as exc:
        raise RuntimeError(
            "Evidently import failed. Check that evidently is installed in the Airflow image."
        ) from exc

    columns = numeric_feature_columns(reference_data) + ["target"]
    reference_for_report = reference_data[columns].copy()
    candidate_for_report = candidate_data[columns].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_for_report,
        current_data=candidate_for_report,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_html))

    try:
        raw_report = report.as_dict()
        write_json(raw_report, output_raw_json)
        raw_json_written = True
        raw_json_error = None
    except Exception as exc:
        raw_json_written = False
        raw_json_error = str(exc)

    return {
        "html_report_path": str(output_html),
        "raw_json_report_path": str(output_raw_json) if raw_json_written else None,
        "raw_json_written": raw_json_written,
        "raw_json_error": raw_json_error,
    }


def build_drift_summary(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    output_html = output_dir / args.output_html
    output_json = output_dir / args.output_json
    output_raw_json = output_dir / args.evidently_json

    candidate_result_path = resolve_path(args.candidate_result, project_root)
    champion_decision_path = resolve_path(args.champion_decision, project_root)

    candidate_result = read_json(candidate_result_path)
    champion_decision = read_json(champion_decision_path)

    candidate_npz = resolve_path(
        extract_candidate_npz(candidate_result),
        project_root,
    )

    reference_npz = resolve_path(
        args.reference_npz
        if args.reference_npz
        else extract_reference_npz(champion_decision, args.default_reference_npz),
        project_root,
    )

    predictions_npz_value = extract_predictions_npz(candidate_result)
    predictions_npz = (
        resolve_path(predictions_npz_value, project_root)
        if predictions_npz_value
        else None
    )

    reference_summary = summarize_npz(
        reference_npz,
        max_sequences_per_split=args.max_sequences_per_split,
        seed=args.random_seed,
    )
    candidate_summary = summarize_npz(
        candidate_npz,
        max_sequences_per_split=args.max_sequences_per_split,
        seed=args.random_seed,
    )

    dataset_summary = dataset_drift_summary(
        reference_data=reference_summary,
        candidate_data=candidate_summary,
        max_standardized_mean_diff=args.max_standardized_mean_diff,
        max_share_drifted_features=args.max_share_drifted_features,
    )
    target_summary = target_drift_summary(
        reference_data=reference_summary,
        candidate_data=candidate_summary,
        max_target_distribution_distance=args.max_target_distribution_distance,
    )

    candidate_targets_by_split = load_targets_from_npz(candidate_npz)

    prediction_classes = None
    prediction_source_key = None

    if predictions_npz is not None:
        prediction_classes, prediction_source_key = find_prediction_classes(predictions_npz)

    prediction_summary = prediction_drift_summary(
        candidate_predictions=prediction_classes,
        prediction_source_key=prediction_source_key,
        candidate_targets_by_split=candidate_targets_by_split,
        max_prediction_distribution_distance=args.max_prediction_distribution_distance,
    )

    evidently_info = generate_evidently_report(
        reference_data=reference_summary,
        candidate_data=candidate_summary,
        output_html=output_html,
        output_raw_json=output_raw_json,
    )

    dataset_drift = bool(dataset_summary["dataset_drift"])
    target_drift = bool(target_summary["target_drift"])
    prediction_drift_value = prediction_summary.get("prediction_drift")
    prediction_drift = bool(prediction_drift_value) if prediction_drift_value is not None else False

    critical_drift_detected = dataset_drift or target_drift or prediction_drift
    candidate_rejected_by_drift_check = critical_drift_detected

    reasons = []

    if dataset_drift:
        reasons.append("dataset_drift")

    if target_drift:
        reasons.append("target_drift")

    if prediction_drift:
        reasons.append("prediction_drift")

    if not reasons:
        reason = "No critical drift detected."
    else:
        reason = "Critical drift detected: " + ", ".join(reasons)

    payload = {
        "status": "success",
        "candidate_rejected_by_drift_check": candidate_rejected_by_drift_check,
        "critical_drift_detected": critical_drift_detected,
        "reason": reason,
        "dataset_drift": dataset_drift,
        "target_drift": target_drift,
        "prediction_drift": prediction_summary.get("prediction_drift"),
        "candidate_result_path": str(candidate_result_path),
        "champion_decision_path": str(champion_decision_path),
        "reference_npz": str(reference_npz),
        "candidate_npz": str(candidate_npz),
        "predictions_npz": str(predictions_npz) if predictions_npz is not None else None,
        "output_html": str(output_html),
        "output_json": str(output_json),
        "evidently": evidently_info,
        "sampling": {
            "max_sequences_per_split": args.max_sequences_per_split,
            "random_seed": args.random_seed,
            "reference_rows": int(len(reference_summary)),
            "candidate_rows": int(len(candidate_summary)),
        },
        "thresholds": {
            "max_standardized_mean_diff": args.max_standardized_mean_diff,
            "max_share_drifted_features": args.max_share_drifted_features,
            "max_target_distribution_distance": args.max_target_distribution_distance,
            "max_prediction_distribution_distance": args.max_prediction_distribution_distance,
        },
        "dataset_drift_details": dataset_summary,
        "target_drift_details": target_summary,
        "prediction_drift_details": prediction_summary,
        "generated_at_utc": utc_now_iso(),
    }

    write_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Evidently drift report for a retrained candidate dataset."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--candidate-result", default=DEFAULT_CANDIDATE_RESULT)
    parser.add_argument("--champion-decision", default=DEFAULT_CHAMPION_DECISION)
    parser.add_argument("--reference-npz", default=None)
    parser.add_argument("--default-reference-npz", default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-html", default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--evidently-json", default=DEFAULT_EVIDENTLY_JSON)
    parser.add_argument("--max-sequences-per-split", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-standardized-mean-diff", type=float, default=0.50)
    parser.add_argument("--max-share-drifted-features", type=float, default=0.30)
    parser.add_argument("--max-target-distribution-distance", type=float, default=0.20)
    parser.add_argument("--max-prediction-distribution-distance", type=float, default=0.30)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_drift_summary(args)

    if args.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
