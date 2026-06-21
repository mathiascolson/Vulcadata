from argparse import Namespace

import joblib
import numpy as np
import pandas as pd
import pytest

import src.preprocessing.preprocess_volcano_dataset as preprocess


def write_fake_aggregated_csv(path, n_rows=130):
    times = pd.date_range(
        "2020-01-01T00:00:00Z",
        periods=n_rows,
        freq="min",
    )

    x = np.linspace(0.0, 2.0 * np.pi, n_rows)

    df = pd.DataFrame(
        {
            "eruption_id": ["quiet_test"] * n_rows,
            "network": ["PF"] * n_rows,
            "station": ["CSS"] * n_rows,
            "channel": ["HHZ"] * n_rows,
            "time_min": times.astype(str),
            "amplitude_mean": np.sin(x),
            "amplitude_std": np.full(n_rows, 0.1),
            "amplitude_max": np.sin(x) + 0.2,
            "amplitude_min": np.sin(x) - 0.2,
            "amplitude_count": np.full(n_rows, 100),
            "energy_low_1_5_5": np.linspace(1.0, 2.0, n_rows),
            "energy_high_6_16": np.linspace(0.5, 1.5, n_rows),
            "frequency_index": np.linspace(0.1, 0.9, n_rows),
            "sampling_rate_source_hz": np.full(n_rows, 100.0),
            "filter_full_low_hz": np.full(n_rows, 1.0),
            "filter_full_high_hz": np.full(n_rows, 16.0),
            "filter_fi_low_min_hz": np.full(n_rows, 1.0),
            "filter_fi_low_max_hz": np.full(n_rows, 5.5),
            "filter_fi_high_min_hz": np.full(n_rows, 6.0),
            "filter_fi_high_max_hz": np.full(n_rows, 16.0),
        }
    )

    df.to_csv(path, index=False)


def write_reference_artifacts(path, n_features=992):
    path.mkdir(parents=True, exist_ok=True)

    feature_names = ["amplitude_mean__CSS__HHZ"] + [
        f"feature_{i:04d}" for i in range(1, n_features)
    ]

    (path / "feature_names.txt").write_text(
        "\n".join(feature_names),
        encoding="utf-8",
    )

    joblib.dump(
        {"statistics_": np.zeros(n_features, dtype=np.float32)},
        path / "imputer.joblib",
    )

    joblib.dump(
        {
            "mean_": np.zeros(n_features, dtype=np.float32),
            "scale_": np.ones(n_features, dtype=np.float32),
        },
        path / "scaler.joblib",
    )


def test_read_periods_accepts_semicolon_csv_with_quiet_and_inference(tmp_path):
    processed_dir = tmp_path / "processed_csv"
    processed_dir.mkdir()

    quiet_csv = processed_dir / f"quiet_test{preprocess.CSV_SUFFIX}"
    inference_csv = processed_dir / f"inference_test{preprocess.CSV_SUFFIX}"

    quiet_csv.write_text("dummy", encoding="utf-8")
    inference_csv.write_text("dummy", encoding="utf-8")

    periods_path = tmp_path / "extraction_periods.csv"
    periods_path.write_text(
        "\n".join(
            [
                "period_id;period_type;split;eruption_start_utc;csv_path",
                f"quiet_test;quiet;;;{quiet_csv.as_posix()}",
                f"inference_test;inference;;;{inference_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    meta = preprocess.read_periods(
        path=periods_path,
        processed_csv_dir=processed_dir,
        mode="inference",
        split_strategy="chronological",
    )

    assert meta["period_id"].tolist() == ["quiet_test", "inference_test"]
    assert meta["period_type"].tolist() == ["quiet", "inference"]


def test_read_periods_training_rejects_inference_period(tmp_path):
    processed_dir = tmp_path / "processed_csv"
    processed_dir.mkdir()

    inference_csv = processed_dir / f"inference_test{preprocess.CSV_SUFFIX}"
    inference_csv.write_text("dummy", encoding="utf-8")

    periods_path = tmp_path / "extraction_periods.csv"
    periods_path.write_text(
        "\n".join(
            [
                "period_id;period_type;split;eruption_start_utc;csv_path",
                f"inference_test;inference;;;{inference_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ne peut pas être utilisée en mode training"):
        preprocess.read_periods(
            path=periods_path,
            processed_csv_dir=processed_dir,
            mode="training",
            split_strategy="chronological",
        )


def test_inference_mode_writes_npz_x_with_reference_feature_count(tmp_path):
    processed_dir = tmp_path / "processed_csv"
    output_dir = tmp_path / "preprocessing_output"
    artifacts_dir = tmp_path / "reference_artifacts"

    processed_dir.mkdir()
    output_dir.mkdir()

    write_reference_artifacts(artifacts_dir, n_features=992)

    aggregated_csv = processed_dir / f"quiet_test{preprocess.CSV_SUFFIX}"
    write_fake_aggregated_csv(aggregated_csv, n_rows=130)

    periods_path = tmp_path / "extraction_periods.csv"
    periods_path.write_text(
        "\n".join(
            [
                "period_id;period_type;split;eruption_start_utc;csv_path",
                f"quiet_test;quiet;;;{aggregated_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    args = Namespace(
        mode="inference",
        periods=str(periods_path),
        processed_csv_dir=str(processed_dir),
        output_dir=str(output_dir),
        reference_artifacts_dir=str(artifacts_dir),
        training_output_name="volcano_multi.npz",
        inference_output_name="inference_source.npz",
        feature_window_minutes=10,
        seq_len=120,
        sequence_stride=5,
        max_horizon_hours=48.0,
        entropy_bins=20,
        include_post_eruption_as_zero=False,
        n_classes=6,
        split_strategy="chronological",
        train_ratio=0.70,
        val_ratio=0.15,
    )

    preprocess.main(args)

    output_npz = output_dir / "inference_source.npz"
    assert output_npz.exists()

    data = np.load(output_npz, allow_pickle=True)

    assert "X" in data.files
    assert "feature_names" in data.files
    assert "inference_times" in data.files
    assert "inference_period_ids" in data.files

    X = data["X"]

    assert X.shape == (3, 120, 992)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert len(data["feature_names"]) == 992
    assert data["inference_period_ids"].tolist() == ["quiet_test", "quiet_test", "quiet_test"]


def test_training_mode_uses_reference_feature_contract(tmp_path):
    processed_dir = tmp_path / "processed_csv"
    output_dir = tmp_path / "training_output"
    artifacts_dir = tmp_path / "reference_artifacts"

    processed_dir.mkdir()
    output_dir.mkdir()

    write_reference_artifacts(artifacts_dir, n_features=992)

    aggregated_csv = processed_dir / f"quiet_test{preprocess.CSV_SUFFIX}"
    write_fake_aggregated_csv(aggregated_csv, n_rows=130)

    periods_path = tmp_path / "extraction_periods.csv"
    periods_path.write_text(
        "\n".join(
            [
                "period_id;period_type;split;eruption_start_utc;csv_path",
                f"quiet_test;quiet;;;{aggregated_csv.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    args = Namespace(
        mode="training",
        periods=str(periods_path),
        processed_csv_dir=str(processed_dir),
        output_dir=str(output_dir),
        reference_artifacts_dir=str(artifacts_dir),
        training_output_name="volcano_multi.npz",
        inference_output_name="inference_source.npz",
        feature_window_minutes=10,
        seq_len=120,
        sequence_stride=5,
        max_horizon_hours=48.0,
        entropy_bins=20,
        include_post_eruption_as_zero=False,
        n_classes=6,
        split_strategy="chronological",
        train_ratio=0.34,
        val_ratio=0.33,
    )

    preprocess.main(args)

    output_npz = output_dir / "volcano_multi.npz"
    assert output_npz.exists()

    data = np.load(output_npz, allow_pickle=True)

    for split in ["train", "val", "test"]:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        assert X.ndim == 3
        assert X.shape[1:] == (120, 992)
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()

    assert len(data["feature_names"]) == 992
    assert (output_dir / "feature_names.txt").exists()
    assert (output_dir / "imputer.joblib").exists()
    assert (output_dir / "scaler.joblib").exists()
