from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.time_utils import format_utc_datetime, utc_now
from src.inference.predict_volcano_alert import validate_sequence_batch


DEFAULT_OUTPUT_NPZ_PATH = "data/preprocessing/processed/latest_batch.npz"
DEFAULT_OUTPUT_ARRAY_KEY = "X"
DEFAULT_METADATA_SUFFIX = "_metadata.json"

PREFERRED_SEQUENCE_KEYS = (
    "X",
    "x",
    "sequences",
    "inputs",
    "features",
)


@dataclass(frozen=True)
class PreparedLatestBatch:
    """
    Summary of a standardized latest inference batch preparation.
    """

    source_npz: str
    source_array_key: str
    output_npz: str
    output_array_key: str
    output_metadata_json: str | None
    created_at_utc: str
    source_batch_size: int
    batch_size: int
    selected_start_index: int
    selected_end_index: int
    seq_len: int
    n_features: int
    selected_times: list[str]
    selected_eruption_ids: list[str]
    feature_names_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_npz": self.source_npz,
            "source_array_key": self.source_array_key,
            "output_npz": self.output_npz,
            "output_array_key": self.output_array_key,
            "output_metadata_json": self.output_metadata_json,
            "created_at_utc": self.created_at_utc,
            "source_batch_size": self.source_batch_size,
            "batch_size": self.batch_size,
            "selected_start_index": self.selected_start_index,
            "selected_end_index": self.selected_end_index,
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "selected_times": self.selected_times,
            "selected_eruption_ids": self.selected_eruption_ids,
            "feature_names_count": self.feature_names_count,
        }


def validate_positive_integer(value: int, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be strictly positive.")

    return value


def default_metadata_path_for_npz(output_npz: str | Path) -> Path:
    output_path = Path(output_npz)
    return output_path.with_name(f"{output_path.stem}{DEFAULT_METADATA_SUFFIX}")


def resolve_source_array_key(
    npz_file: np.lib.npyio.NpzFile,
    requested_key: str | None,
) -> str:
    available_keys = list(npz_file.files)

    if requested_key is not None:
        if requested_key not in available_keys:
            raise KeyError(
                f"Requested source array key {requested_key!r} not found. "
                f"Available keys: {available_keys}"
            )

        return requested_key

    for key in PREFERRED_SEQUENCE_KEYS:
        if key in available_keys:
            array = npz_file[key]
            if array.ndim == 3:
                return key

    three_dimensional_keys = []

    for key in available_keys:
        array = npz_file[key]

        if isinstance(array, np.ndarray) and array.ndim == 3:
            three_dimensional_keys.append(key)

    if len(three_dimensional_keys) == 1:
        return three_dimensional_keys[0]

    if len(three_dimensional_keys) > 1:
        raise ValueError(
            "Multiple 3D arrays found in source NPZ. "
            "Pass --source-array-key explicitly. "
            f"3D candidates: {three_dimensional_keys}"
        )

    raise ValueError(
        "No 3D sequence array found in source NPZ. "
        f"Available keys: {available_keys}"
    )


def infer_metadata_candidate_keys(source_array_key: str) -> tuple[list[str], list[str]]:
    """
    Infer likely metadata keys for timestamps and eruption ids.

    Examples:
        X_test  -> test_times, test_eruption_ids
        X_val   -> val_times, val_eruption_ids
        X_train -> train_times, train_eruption_ids
        X       -> times, eruption_ids
    """
    if source_array_key.startswith("X_"):
        split_name = source_array_key.removeprefix("X_")

        return (
            [
                f"{split_name}_times",
                f"{split_name}_timestamps",
                f"{split_name}_datetime_utc",
            ],
            [
                f"{split_name}_eruption_ids",
                f"{split_name}_eruption_id",
            ],
        )

    return (
        [
            "times",
            "timestamps",
            "datetime_utc",
            "sample_times",
        ],
        [
            "eruption_ids",
            "eruption_id",
            "sample_eruption_ids",
        ],
    )


def read_optional_string_slice(
    npz_file: np.lib.npyio.NpzFile,
    candidate_keys: list[str],
    *,
    start_index: int,
    stop_index: int,
) -> tuple[str | None, list[str]]:
    for key in candidate_keys:
        if key not in npz_file.files:
            continue

        values = npz_file[key]

        if values.ndim != 1:
            raise ValueError(
                f"Metadata array {key!r} must be one-dimensional, "
                f"got shape {values.shape}."
            )

        if values.shape[0] < stop_index:
            raise ValueError(
                f"Metadata array {key!r} has length {values.shape[0]}, "
                f"but at least {stop_index} values are required."
            )

        return key, [str(value) for value in values[start_index:stop_index].tolist()]

    return None, []


def get_feature_names_count(npz_file: np.lib.npyio.NpzFile) -> int | None:
    if "feature_names" not in npz_file.files:
        return None

    feature_names = npz_file["feature_names"]

    if feature_names.ndim != 1:
        raise ValueError(
            "feature_names must be one-dimensional when present, "
            f"got shape {feature_names.shape}."
        )

    return int(feature_names.shape[0])


def select_latest_sequences(
    sequences: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, int, int]:
    validate_positive_integer(batch_size, "batch_size")

    source_batch_size = int(sequences.shape[0])

    if batch_size > source_batch_size:
        raise ValueError(
            f"batch_size={batch_size} exceeds source batch size "
            f"{source_batch_size}."
        )

    selected_start_index = source_batch_size - batch_size
    selected_end_index = source_batch_size - 1

    selected = sequences[selected_start_index:source_batch_size]

    return selected, selected_start_index, selected_end_index


def build_metadata_payload(
    *,
    prepared_batch: PreparedLatestBatch,
    available_npz_keys: list[str],
    selected_times_key: str | None,
    selected_eruption_ids_key: str | None,
) -> dict[str, Any]:
    return {
        "status": "success",
        "created_at_utc": prepared_batch.created_at_utc,
        "source": {
            "npz_path": prepared_batch.source_npz,
            "array_key": prepared_batch.source_array_key,
            "available_keys": available_npz_keys,
            "source_batch_size": prepared_batch.source_batch_size,
        },
        "output": {
            "npz_path": prepared_batch.output_npz,
            "array_key": prepared_batch.output_array_key,
            "metadata_json": prepared_batch.output_metadata_json,
        },
        "selection": {
            "mode": "last",
            "batch_size": prepared_batch.batch_size,
            "selected_start_index": prepared_batch.selected_start_index,
            "selected_end_index": prepared_batch.selected_end_index,
            "seq_len": prepared_batch.seq_len,
            "n_features": prepared_batch.n_features,
            "times_key": selected_times_key,
            "selected_times": prepared_batch.selected_times,
            "eruption_ids_key": selected_eruption_ids_key,
            "selected_eruption_ids": prepared_batch.selected_eruption_ids,
        },
        "features": {
            "feature_names_count": prepared_batch.feature_names_count,
        },
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def prepare_latest_batch(
    *,
    source_npz: str | Path,
    output_npz: str | Path = DEFAULT_OUTPUT_NPZ_PATH,
    source_array_key: str | None = None,
    output_array_key: str = DEFAULT_OUTPUT_ARRAY_KEY,
    batch_size: int = 1,
    expected_seq_len: int | None = None,
    expected_n_features: int | None = None,
    output_metadata_json: str | Path | None = None,
    write_metadata_json: bool = True,
    overwrite: bool = False,
) -> PreparedLatestBatch:
    source_path = Path(source_npz)

    if not source_path.exists():
        raise FileNotFoundError(f"Source NPZ not found: {source_path}")

    output_path = Path(output_npz)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output NPZ already exists: {output_path}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    if not output_array_key or not isinstance(output_array_key, str):
        raise ValueError("output_array_key must be a non-empty string.")

    validate_positive_integer(batch_size, "batch_size")

    with np.load(source_path, allow_pickle=False) as npz_file:
        resolved_array_key = resolve_source_array_key(
            npz_file=npz_file,
            requested_key=source_array_key,
        )

        sequences = np.asarray(npz_file[resolved_array_key])

        validate_sequence_batch(
            sequences,
            expected_seq_len=expected_seq_len,
            expected_n_features=expected_n_features,
        )

        selected_sequences, selected_start_index, selected_end_index = (
            select_latest_sequences(
                sequences,
                batch_size=batch_size,
            )
        )

        stop_index = selected_end_index + 1

        time_candidate_keys, eruption_candidate_keys = infer_metadata_candidate_keys(
            resolved_array_key
        )

        selected_times_key, selected_times = read_optional_string_slice(
            npz_file,
            time_candidate_keys,
            start_index=selected_start_index,
            stop_index=stop_index,
        )

        selected_eruption_ids_key, selected_eruption_ids = read_optional_string_slice(
            npz_file,
            eruption_candidate_keys,
            start_index=selected_start_index,
            stop_index=stop_index,
        )

        feature_names_count = get_feature_names_count(npz_file)
        available_npz_keys = list(npz_file.files)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **{
            output_array_key: selected_sequences.astype(np.float32, copy=False),
        },
    )

    metadata_path: Path | None

    if write_metadata_json:
        metadata_path = (
            Path(output_metadata_json)
            if output_metadata_json is not None
            else default_metadata_path_for_npz(output_path)
        )
    else:
        metadata_path = None

    prepared_batch = PreparedLatestBatch(
        source_npz=str(source_path),
        source_array_key=resolved_array_key,
        output_npz=str(output_path),
        output_array_key=output_array_key,
        output_metadata_json=str(metadata_path) if metadata_path is not None else None,
        created_at_utc=format_utc_datetime(utc_now()),
        source_batch_size=int(sequences.shape[0]),
        batch_size=int(batch_size),
        selected_start_index=int(selected_start_index),
        selected_end_index=int(selected_end_index),
        seq_len=int(selected_sequences.shape[1]),
        n_features=int(selected_sequences.shape[2]),
        selected_times=selected_times,
        selected_eruption_ids=selected_eruption_ids,
        feature_names_count=feature_names_count,
    )

    if metadata_path is not None:
        metadata_payload = build_metadata_payload(
            prepared_batch=prepared_batch,
            available_npz_keys=available_npz_keys,
            selected_times_key=selected_times_key,
            selected_eruption_ids_key=selected_eruption_ids_key,
        )

        write_json(metadata_path, metadata_payload)

    return prepared_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a standardized latest NPZ batch for Vulcadata inference."
    )

    parser.add_argument(
        "--source-npz",
        required=True,
        help="Source NPZ containing sequence arrays.",
    )
    parser.add_argument(
        "--source-array-key",
        default=None,
        help="Source array key, for example X_test or X.",
    )
    parser.add_argument(
        "--output-npz",
        default=DEFAULT_OUTPUT_NPZ_PATH,
        help="Output standardized NPZ path.",
    )
    parser.add_argument(
        "--output-array-key",
        default=DEFAULT_OUTPUT_ARRAY_KEY,
        help="Output array key. Default: X.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of latest sequences to keep.",
    )
    parser.add_argument(
        "--expected-seq-len",
        type=int,
        default=None,
        help="Optional expected sequence length validation.",
    )
    parser.add_argument(
        "--expected-n-features",
        type=int,
        default=None,
        help="Optional expected feature count validation.",
    )
    parser.add_argument(
        "--output-metadata-json",
        default=None,
        help=(
            "Optional metadata JSON output path. "
            "Default: same directory as output NPZ, with _metadata.json suffix."
        ),
    )
    parser.add_argument(
        "--no-metadata-json",
        action="store_true",
        help="Disable metadata JSON writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prepared_batch = prepare_latest_batch(
        source_npz=args.source_npz,
        output_npz=args.output_npz,
        source_array_key=args.source_array_key,
        output_array_key=args.output_array_key,
        batch_size=args.batch_size,
        expected_seq_len=args.expected_seq_len,
        expected_n_features=args.expected_n_features,
        output_metadata_json=args.output_metadata_json,
        write_metadata_json=not args.no_metadata_json,
        overwrite=args.overwrite,
    )

    print(json.dumps(prepared_batch.to_dict(), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())