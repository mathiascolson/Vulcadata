from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PERIODS_CSV = PROJECT_ROOT / "data" / "metadata" / "extraction_periods.csv"
DEFAULT_PROCESSED_CSV_DIR = PROJECT_ROOT / "data" / "extraction" / "processed_csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessing" / "processed"

CSV_SUFFIX = "_filtered_1_16Hz_aggregated_1min_with_fi.csv"

REQUIRED_COLUMNS = [
    "station",
    "time_min",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_max",
    "amplitude_min",
    "amplitude_count",
    "channel",
]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


def normalize_period_type(value: str) -> str:
    value = clean_text(value).lower()

    if value in {"eruption", "eruptive", "event"}:
        return "eruption"

    if value in {"quiet", "calm", "calme", "background", "non_eruptive"}:
        return "quiet"

    if value in {"inference", "predict", "prediction", "unknown"}:
        return "inference"

    raise ValueError(
        f"period_type invalide : {value}. "
        "Valeurs attendues : eruption, quiet ou inference."
    )


def read_periods(path: Path, processed_csv_dir: Path, mode: str, split_strategy: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier de périodes introuvable : {path}")

    periods = read_csv_auto(path)
    periods.columns = [str(c).strip() for c in periods.columns]

    required = {"period_id", "period_type"}
    missing = required - set(periods.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier de périodes : {sorted(missing)}")

    if "eruption_start_utc" not in periods.columns:
        periods["eruption_start_utc"] = ""

    if "eruption_end_utc" not in periods.columns:
        periods["eruption_end_utc"] = ""

    if "split" not in periods.columns:
        periods["split"] = ""

    if "csv_path" not in periods.columns:
        periods["csv_path"] = ""

    for col in periods.columns:
        periods[col] = periods[col].apply(clean_text)

    rows = []

    for _, row in periods.iterrows():
        period_id = row["period_id"]
        period_type = normalize_period_type(row["period_type"])

        if not period_id:
            raise ValueError("period_id vide dans le fichier de périodes.")

        if mode == "training":
            if period_type == "inference":
                raise ValueError(
                    f"{period_id} a period_type=inference. "
                    "Cette période ne peut pas être utilisée en mode training."
                )
            if period_type == "eruption" and not row["eruption_start_utc"]:
                raise ValueError(
                    f"eruption_start_utc manquant pour la période éruptive : {period_id}"
                )

        if split_strategy == "manual":
            allowed_splits = {"train", "val", "test"}
            if row["split"] not in allowed_splits:
                raise ValueError(
                    f"split invalide pour {period_id}: {row['split']}. "
                    "Valeurs attendues en split manuel : train, val ou test."
                )

        csv_path = row["csv_path"]
        if not csv_path:
            csv_path = str(processed_csv_dir / f"{period_id}{CSV_SUFFIX}")

        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"CSV agrégé introuvable pour {period_id} : {csv_path}"
            )

        rows.append(
            {
                "period_id": period_id,
                "period_type": period_type,
                "split": row["split"],
                "eruption_start_utc": row["eruption_start_utc"],
                "eruption_end_utc": row["eruption_end_utc"],
                "csv_path": csv_path,
            }
        )

    return pd.DataFrame(rows)


def shannon_entropy(values: np.ndarray, bins: int = 20) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return np.nan

    hist, _ = np.histogram(values, bins=bins, density=False)
    total = hist.sum()

    if total == 0:
        return np.nan

    p = hist / total
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def safe_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 4:
        return np.nan

    if np.nanstd(values) == 0:
        return 0.0

    return float(kurtosis(values, fisher=True, bias=False))


def slope_last_window(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)

    if mask.sum() < 3:
        return np.nan

    y = values[mask]
    x = np.arange(len(values))[mask]

    if len(np.unique(x)) < 2:
        return np.nan

    coef = np.polyfit(x, y, deg=1)[0]
    return float(coef)


def add_causal_rolling_features_one_group(
    g: pd.DataFrame,
    feature_window_minutes: int,
    entropy_bins: int,
) -> pd.DataFrame:
    g = g.sort_values("time_min").copy()

    g["amplitude_range"] = g["amplitude_max"] - g["amplitude_min"]
    g["amplitude_abs_mean"] = g["amplitude_mean"].abs()

    base_cols = [
        "amplitude_mean",
        "amplitude_std",
        "amplitude_max",
        "amplitude_min",
        "amplitude_range",
        "amplitude_abs_mean",
        "amplitude_count",
    ]

    roll = g[base_cols].rolling(
        window=feature_window_minutes,
        min_periods=max(3, feature_window_minutes // 2),
        center=False,
    )

    for col in base_cols:
        g[f"{col}_roll_mean"] = roll[col].mean()
        g[f"{col}_roll_std"] = roll[col].std()
        g[f"{col}_roll_min"] = roll[col].min()
        g[f"{col}_roll_max"] = roll[col].max()
        g[f"{col}_roll_median"] = roll[col].median()

    g["shannon_entropy"] = (
        g["amplitude_mean"]
        .rolling(
            window=feature_window_minutes,
            min_periods=max(3, feature_window_minutes // 2),
            center=False,
        )
        .apply(lambda x: shannon_entropy(x, bins=entropy_bins), raw=True)
    )

    g["kurtosis"] = (
        g["amplitude_mean"]
        .rolling(
            window=feature_window_minutes,
            min_periods=max(4, feature_window_minutes // 2),
            center=False,
        )
        .apply(safe_kurtosis, raw=True)
    )

    g["amplitude_mean_slope"] = (
        g["amplitude_mean"]
        .rolling(
            window=feature_window_minutes,
            min_periods=max(3, feature_window_minutes // 2),
            center=False,
        )
        .apply(slope_last_window, raw=True)
    )

    env_window = max(3, feature_window_minutes)

    g["shannon_entropy_env"] = (
        g["shannon_entropy"]
        .rolling(window=env_window, min_periods=3, center=False)
        .median()
    )

    g["kurtosis_env"] = (
        g["kurtosis"]
        .rolling(window=env_window, min_periods=3, center=False)
        .median()
    )

    g["amplitude_mean_env"] = (
        g["amplitude_mean_roll_mean"]
        .rolling(window=env_window, min_periods=3, center=False)
        .median()
    )

    return g


def build_features_for_period(
    df: pd.DataFrame,
    feature_window_minutes: int,
    entropy_bins: int,
) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {sorted(missing)}")

    df = df.copy()
    df["time_min"] = pd.to_datetime(df["time_min"], utc=True, errors="coerce")

    if df["time_min"].isna().any():
        n_bad = int(df["time_min"].isna().sum())
        raise ValueError(f"{n_bad} lignes ont un time_min invalide.")

    for col in [
        "amplitude_mean",
        "amplitude_std",
        "amplitude_max",
        "amplitude_min",
        "amplitude_count",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["station"] = df["station"].astype(str)
    df["channel"] = df["channel"].astype(str)
    df["sensor"] = df["station"] + "__" + df["channel"]

    df = df.sort_values(["sensor", "time_min"])

    enriched_parts = []
    for _, g in df.groupby("sensor", sort=False):
        enriched = add_causal_rolling_features_one_group(
            g,
            feature_window_minutes=feature_window_minutes,
            entropy_bins=entropy_bins,
        )
        enriched_parts.append(enriched)

    feat_long = pd.concat(enriched_parts, axis=0, ignore_index=True)

    exclude_cols = {
        "station",
        "channel",
        "sensor",
        "time_min",
    }

    numeric_feature_cols = [
        c
        for c in feat_long.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(feat_long[c])
    ]

    wide = feat_long.pivot_table(
        index="time_min",
        columns="sensor",
        values=numeric_feature_cols,
        aggfunc="mean",
    )

    wide.columns = [f"{feature}__{sensor}" for feature, sensor in wide.columns]
    wide = wide.sort_index()

    entropy_cols = [c for c in wide.columns if c.startswith("shannon_entropy__")]
    kurtosis_cols = [c for c in wide.columns if c.startswith("kurtosis__")]
    amp_cols = [c for c in wide.columns if c.startswith("amplitude_mean_roll_mean__")]

    if entropy_cols:
        wide["network_entropy_mean"] = wide[entropy_cols].mean(axis=1)
        wide["network_entropy_std"] = wide[entropy_cols].std(axis=1)

    if kurtosis_cols:
        wide["network_kurtosis_mean"] = wide[kurtosis_cols].mean(axis=1)
        wide["network_kurtosis_std"] = wide[kurtosis_cols].std(axis=1)

    if amp_cols:
        wide["network_amplitude_mean"] = wide[amp_cols].mean(axis=1)
        wide["network_amplitude_std"] = wide[amp_cols].std(axis=1)

    return wide


def delay_hours_to_multiclass_label(delay_hours: float) -> int:
    if delay_hours < 0:
        raise ValueError(f"delay_hours négatif inattendu : {delay_hours}")

    if delay_hours <= 6:
        return 5
    if delay_hours <= 12:
        return 4
    if delay_hours <= 24:
        return 3
    if delay_hours <= 36:
        return 2
    if delay_hours <= 48:
        return 1

    raise ValueError(
        f"delay_hours={delay_hours} dépasse l'horizon attendu. "
        "Vérifier --max-horizon-hours."
    )


def make_sequences_for_inference(
    feature_table: pd.DataFrame,
    seq_len: int,
    sequence_stride: int = 1,
):
    if sequence_stride < 1:
        raise ValueError("sequence_stride doit être >= 1.")

    feature_table = feature_table.sort_index().copy()

    X_list = []
    t_end_list = []

    values = feature_table.to_numpy(dtype=np.float32)

    for end_idx in range(seq_len - 1, len(feature_table), sequence_stride):
        t_end = pd.Timestamp(feature_table.index[end_idx]).tz_convert("UTC")
        start_idx = end_idx - seq_len + 1
        seq = values[start_idx : end_idx + 1]

        X_list.append(seq)
        t_end_list.append(t_end.isoformat())

    if not X_list:
        return np.empty((0, seq_len, feature_table.shape[1]), dtype=np.float32), []

    X = np.stack(X_list).astype(np.float32)
    return X, t_end_list


def make_sequences_for_quiet_period(
    feature_table: pd.DataFrame,
    seq_len: int,
    sequence_stride: int = 1,
    quiet_label: int = 0,
):
    X, t_end_list = make_sequences_for_inference(
        feature_table=feature_table,
        seq_len=seq_len,
        sequence_stride=sequence_stride,
    )
    y = np.full((X.shape[0],), quiet_label, dtype=np.int64)
    return X, y, t_end_list


def make_sequences_for_eruption(
    feature_table: pd.DataFrame,
    eruption_start_utc: pd.Timestamp,
    seq_len: int,
    max_horizon_hours: float,
    include_post_eruption_as_zero: bool = False,
    sequence_stride: int = 1,
):
    if sequence_stride < 1:
        raise ValueError("sequence_stride doit être >= 1.")

    feature_table = feature_table.sort_index().copy()

    X_list = []
    y_list = []
    t_end_list = []

    values = feature_table.to_numpy(dtype=np.float32)
    eruption_start_utc = pd.Timestamp(eruption_start_utc).tz_convert("UTC")

    for end_idx in range(seq_len - 1, len(feature_table), sequence_stride):
        t_end = pd.Timestamp(feature_table.index[end_idx]).tz_convert("UTC")
        delay_hours = (eruption_start_utc - t_end).total_seconds() / 3600.0

        if delay_hours < 0:
            if include_post_eruption_as_zero:
                delay_hours = 0.0
            else:
                continue

        if delay_hours > max_horizon_hours:
            continue

        start_idx = end_idx - seq_len + 1
        seq = values[start_idx : end_idx + 1]

        X_list.append(seq)
        y_list.append(delay_hours_to_multiclass_label(delay_hours))
        t_end_list.append(t_end.isoformat())

    if not X_list:
        return (
            np.empty((0, seq_len, feature_table.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    return X, y, t_end_list


def load_feature_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"feature_names.txt introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    if not names:
        raise ValueError(f"feature_names.txt vide : {path}")

    return names


def derive_feature_names(feature_tables: dict[str, pd.DataFrame]) -> list[str]:
    all_columns = set()
    for ft in feature_tables.values():
        all_columns = all_columns.union(set(ft.columns))

    if not all_columns:
        raise ValueError("Aucune feature disponible dans les tables de features.")

    return sorted(all_columns)


def resolve_training_feature_names(args, feature_tables: dict[str, pd.DataFrame]) -> tuple[list[str], str, str]:
    explicit_reference_dir = bool(clean_text(args.reference_artifacts_dir))

    if explicit_reference_dir:
        artifacts_dir = Path(args.reference_artifacts_dir)
    else:
        artifacts_dir = DEFAULT_OUTPUT_DIR

    feature_names_path = artifacts_dir / "feature_names.txt"

    if feature_names_path.exists():
        feature_names = load_feature_names(feature_names_path)
        return feature_names, "reference", str(feature_names_path)

    if explicit_reference_dir:
        raise FileNotFoundError(
            f"feature_names.txt introuvable dans le répertoire de référence demandé : {feature_names_path}"
        )

    feature_names = derive_feature_names(feature_tables)
    return feature_names, "derived", ""


def load_artifact_array(artifact, dict_key: str, attr_name: str) -> np.ndarray:
    if isinstance(artifact, dict) and dict_key in artifact:
        return np.asarray(artifact[dict_key], dtype=np.float32)

    if hasattr(artifact, attr_name):
        return np.asarray(getattr(artifact, attr_name), dtype=np.float32)

    raise ValueError(
        f"Artefact incompatible : clé {dict_key} ou attribut {attr_name} introuvable."
    )


def apply_saved_preprocessing(X: np.ndarray, artifacts_dir: Path) -> np.ndarray:
    imputer_path = artifacts_dir / "imputer.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"

    if not imputer_path.exists():
        raise FileNotFoundError(f"imputer.joblib introuvable : {imputer_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler.joblib introuvable : {scaler_path}")

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    medians = load_artifact_array(imputer, "statistics_", "statistics_")
    means = load_artifact_array(scaler, "mean_", "mean_")
    scales = load_artifact_array(scaler, "scale_", "scale_")

    n_features = X.shape[2]

    if len(medians) != n_features:
        raise ValueError(
            f"Dimension imputer incompatible : {len(medians)} valeurs pour {n_features} features."
        )

    if len(means) != n_features or len(scales) != n_features:
        raise ValueError(
            f"Dimension scaler incompatible : mean={len(means)}, scale={len(scales)}, features={n_features}."
        )

    X = X.astype(np.float32, copy=True)

    for j in range(n_features):
        col = X[:, :, j]
        bad_mask = ~np.isfinite(col)
        if bad_mask.any():
            col[bad_mask] = medians[j]

        scale = scales[j]
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        X[:, :, j] = (X[:, :, j] - means[j]) / scale

    return X.astype(np.float32, copy=False)


def fit_transform_train_only(X_train, X_val, X_test, output_dir: Path):
    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "X_train, X_val et X_test doivent avoir la forme (N, T, F)."
        )

    n_features = X_train.shape[2]

    if X_val.shape[2] != n_features or X_test.shape[2] != n_features:
        raise ValueError(
            "X_train, X_val et X_test doivent avoir le même nombre de features."
        )

    medians = np.zeros(n_features, dtype=np.float32)
    means = np.zeros(n_features, dtype=np.float32)
    scales = np.ones(n_features, dtype=np.float32)

    print("\n      Imputation/scaling mémoire optimisé, feature par feature")

    for j in range(n_features):
        train_col = X_train[:, :, j]
        finite_mask = np.isfinite(train_col)

        if finite_mask.any():
            median_j = np.median(train_col[finite_mask]).astype(np.float32)
        else:
            median_j = np.float32(0.0)

        medians[j] = median_j

        for X in (X_train, X_val, X_test):
            col = X[:, :, j]
            bad_mask = ~np.isfinite(col)
            if bad_mask.any():
                col[bad_mask] = median_j

        train_col = X_train[:, :, j]

        mean_j = np.mean(train_col, dtype=np.float64)
        std_j = np.std(train_col, dtype=np.float64)

        if not np.isfinite(mean_j):
            mean_j = 0.0

        if not np.isfinite(std_j) or std_j == 0.0:
            std_j = 1.0

        means[j] = np.float32(mean_j)
        scales[j] = np.float32(std_j)

        for X in (X_train, X_val, X_test):
            X[:, :, j] = (X[:, :, j] - means[j]) / scales[j]

        if (j + 1) % 100 == 0 or (j + 1) == n_features:
            print(f"      Features traitées : {j + 1}/{n_features}")

    imputer_artifact = {
        "type": "custom_featurewise_median_imputer",
        "strategy": "median",
        "statistics_": medians,
        "n_features": int(n_features),
    }

    scaler_artifact = {
        "type": "custom_featurewise_standard_scaler",
        "mean_": means,
        "scale_": scales,
        "n_features": int(n_features),
    }

    joblib.dump(imputer_artifact, output_dir / "imputer.joblib")
    joblib.dump(scaler_artifact, output_dir / "scaler.joblib")

    return (
        X_train.astype(np.float32, copy=False),
        X_val.astype(np.float32, copy=False),
        X_test.astype(np.float32, copy=False),
    )


def finite_check(name: str, arr: np.ndarray) -> list[str]:
    errors = []
    if np.isnan(arr).any():
        errors.append(f"{name} contient des NaN")
    if np.isinf(arr).any():
        errors.append(f"{name} contient des Inf")
    return errors


def validate_npz(path: Path, n_classes: int, mode: str) -> dict:
    data = np.load(path, allow_pickle=True)

    errors = []
    warnings = []

    if mode == "inference":
        required = ["X"]
    else:
        required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]

    for key in required:
        if key not in data:
            errors.append(f"Clé absente : {key}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    report = {
        "path": str(path),
        "mode": mode,
        "valid": True,
        "errors": errors,
        "warnings": warnings,
    }

    if mode == "inference":
        X = data["X"]
        report["X_shape"] = list(X.shape)
        report["X_dtype"] = str(X.dtype)

        if X.ndim != 3:
            errors.append(f"X doit être en 3D (N,T,F), reçu {X.shape}")

        errors.extend(finite_check("X", X))

        if "feature_names" in data and X.ndim == 3:
            feature_names = data["feature_names"]
            report["feature_names_count"] = int(len(feature_names))
            if len(feature_names) != X.shape[2]:
                errors.append(
                    f"feature_names contient {len(feature_names)} noms, mais X a {X.shape[2]} features"
                )
        else:
            warnings.append("feature_names absent du NPZ")

        report["valid"] = len(errors) == 0
        return report

    report["splits"] = {}
    n_features_ref = None
    seq_len_ref = None

    for split in ["train", "val", "test"]:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]

        report["splits"][split] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "X_dtype": str(X.dtype),
            "y_dtype": str(y.dtype),
        }

        if X.ndim != 3:
            errors.append(f"X_{split} doit être en 3D (N,T,F), reçu {X.shape}")
        else:
            if seq_len_ref is None:
                seq_len_ref = X.shape[1]
                n_features_ref = X.shape[2]
            elif X.shape[1] != seq_len_ref or X.shape[2] != n_features_ref:
                errors.append(
                    f"X_{split} a une forme incohérente : {X.shape}, attendu T={seq_len_ref}, F={n_features_ref}"
                )

        if y.ndim != 1:
            errors.append(f"y_{split} doit être en 1D, reçu {y.shape}")

        if X.shape[0] != y.shape[0]:
            errors.append(f"N incohérent pour {split}: X={X.shape[0]}, y={y.shape[0]}")

        errors.extend(finite_check(f"X_{split}", X))
        errors.extend(finite_check(f"y_{split}", y.astype(float)))

        unique, counts = np.unique(y, return_counts=True)
        class_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
        report["splits"][split]["class_counts"] = class_counts

        bad = unique[(unique < 0) | (unique >= n_classes)]
        if bad.size > 0:
            errors.append(f"Classes invalides dans y_{split}: {bad.tolist()}")

        missing = [cls for cls in range(n_classes) if cls not in unique]
        if missing:
            warnings.append(f"Classes absentes dans {split}: {missing}")

    if "feature_names" in data and n_features_ref is not None:
        feature_names = data["feature_names"]
        report["feature_names_count"] = int(len(feature_names))
        if len(feature_names) != n_features_ref:
            errors.append(
                f"feature_names contient {len(feature_names)} noms, mais X a {n_features_ref} features"
            )
    else:
        warnings.append("feature_names absent du NPZ")

    report["valid"] = len(errors) == 0
    return report


def split_indices_chronological(n: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError(
            f"Impossible de créer train/val/test avec seulement {n} séquences."
        )

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    return train_idx, val_idx, test_idx


def build_feature_tables(meta: pd.DataFrame, args) -> dict[str, pd.DataFrame]:
    feature_tables = {}

    for _, row in meta.iterrows():
        period_id = str(row["period_id"])
        csv_path = Path(row["csv_path"])

        print(f"\nLecture et features : {period_id}")
        print(f"CSV : {csv_path}")

        df = pd.read_csv(csv_path)

        feature_table = build_features_for_period(
            df,
            feature_window_minutes=args.feature_window_minutes,
            entropy_bins=args.entropy_bins,
        )

        feature_tables[period_id] = feature_table
        print(f"Table features : {feature_table.shape}")

    return feature_tables


def run_inference_mode(args, meta: pd.DataFrame) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.reference_artifacts_dir) if args.reference_artifacts_dir else output_dir
    feature_names = load_feature_names(artifacts_dir / "feature_names.txt")

    feature_tables = build_feature_tables(meta, args)

    X_parts = []
    times = []
    period_ids = []

    for _, row in meta.iterrows():
        period_id = str(row["period_id"])
        ft = feature_tables[period_id].reindex(columns=feature_names)

        X, t_end = make_sequences_for_inference(
            feature_table=ft,
            seq_len=args.seq_len,
            sequence_stride=args.sequence_stride,
        )

        print(f"Séquences inference {period_id} : X={X.shape}")

        if X.shape[0] == 0:
            continue

        X_parts.append(X)
        times.extend(t_end)
        period_ids.extend([period_id] * len(t_end))

    if not X_parts:
        raise ValueError("Aucune séquence produite pour l'inférence.")

    X = np.concatenate(X_parts, axis=0)
    X = apply_saved_preprocessing(X, artifacts_dir=artifacts_dir)

    output_npz = output_dir / args.inference_output_name

    np.savez_compressed(
        output_npz,
        X=X.astype(np.float32),
        feature_names=np.asarray(feature_names),
        inference_times=np.asarray(times),
        inference_period_ids=np.asarray(period_ids),
    )

    config = {
        "mode": "inference",
        "periods": str(args.periods),
        "processed_csv_dir": str(args.processed_csv_dir),
        "reference_artifacts_dir": str(artifacts_dir),
        "output_npz": str(output_npz),
        "array_key": "X",
        "feature_window_minutes": args.feature_window_minutes,
        "seq_len": args.seq_len,
        "sequence_stride": args.sequence_stride,
        "entropy_bins": args.entropy_bins,
        "n_features": len(feature_names),
        "X_shape": list(X.shape),
        "note": "Inference mode uses existing feature_names, imputer and scaler. It does not fit preprocessing artifacts on current data.",
    }

    with open(output_dir / "inference_preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    validation_report = validate_npz(output_npz, n_classes=args.n_classes, mode="inference")
    validation_path = output_dir / "inference_npz_validation_report.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    print(json.dumps(validation_report, indent=2, ensure_ascii=False))
    print(f"Fichier inference créé : {output_npz}")
    print(f"Rapport validation écrit : {validation_path}")

    if not validation_report["valid"]:
        raise SystemExit(1)


def run_training_mode(args, meta: pd.DataFrame) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_tables = build_feature_tables(meta, args)

    feature_names, feature_contract_mode, feature_contract_source = resolve_training_feature_names(
        args=args,
        feature_tables=feature_tables,
    )

    print()
    print(f"Contrat de features training : {feature_contract_mode}")
    if feature_contract_source:
        print(f"Source du contrat de features : {feature_contract_source}")
    print(f"Nombre de features attendues : {len(feature_names)}")

    if args.split_strategy == "manual":
        X_by_split = {"train": [], "val": [], "test": []}
        y_by_split = {"train": [], "val": [], "test": []}
        times_by_split = {"train": [], "val": [], "test": []}
        periods_by_split = {"train": [], "val": [], "test": []}
    else:
        sequence_rows = []

    for _, row in meta.iterrows():
        period_id = str(row["period_id"])
        period_type = str(row["period_type"])
        ft = feature_tables[period_id].reindex(columns=feature_names)

        if period_type == "quiet":
            X, y, t_end = make_sequences_for_quiet_period(
                feature_table=ft,
                seq_len=args.seq_len,
                sequence_stride=args.sequence_stride,
                quiet_label=0,
            )
        elif period_type == "eruption":
            eruption_start = pd.Timestamp(row["eruption_start_utc"], tz="UTC")
            X, y, t_end = make_sequences_for_eruption(
                feature_table=ft,
                eruption_start_utc=eruption_start,
                seq_len=args.seq_len,
                max_horizon_hours=args.max_horizon_hours,
                include_post_eruption_as_zero=args.include_post_eruption_as_zero,
                sequence_stride=args.sequence_stride,
            )
        else:
            raise ValueError(
                f"period_type={period_type} non autorisé en mode training pour {period_id}."
            )

        print(f"Séquences training {period_id} : X={X.shape} | y={y.shape}")

        if X.shape[0] == 0:
            continue

        if args.split_strategy == "manual":
            split = str(row["split"])
            X_by_split[split].append(X)
            y_by_split[split].append(y)
            times_by_split[split].extend(t_end)
            periods_by_split[split].extend([period_id] * len(t_end))
        else:
            for i, t in enumerate(t_end):
                sequence_rows.append(
                    {
                        "time": pd.Timestamp(t),
                        "period_id": period_id,
                        "X": X[i],
                        "y": y[i],
                    }
                )

    if args.split_strategy == "chronological":
        if not sequence_rows:
            raise ValueError("Aucune séquence produite pour le training.")

        sequence_rows = sorted(sequence_rows, key=lambda r: (r["time"], r["period_id"]))
        train_idx, val_idx, test_idx = split_indices_chronological(
            n=len(sequence_rows),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        def build_split(indices):
            X = np.stack([sequence_rows[i]["X"] for i in indices]).astype(np.float32)
            y = np.asarray([sequence_rows[i]["y"] for i in indices], dtype=np.int64)
            times = [sequence_rows[i]["time"].isoformat() for i in indices]
            period_ids = [sequence_rows[i]["period_id"] for i in indices]
            return X, y, times, period_ids

        X_train, y_train, train_times, train_period_ids = build_split(train_idx)
        X_val, y_val, val_times, val_period_ids = build_split(val_idx)
        X_test, y_test, test_times, test_period_ids = build_split(test_idx)

    else:
        def concat_split(split: str):
            if not X_by_split[split]:
                raise ValueError(f"Aucune donnée pour le split {split}.")
            X = np.concatenate(X_by_split[split], axis=0)
            y = np.concatenate(y_by_split[split], axis=0)
            return X, y

        X_train, y_train = concat_split("train")
        X_val, y_val = concat_split("val")
        X_test, y_test = concat_split("test")
        train_times = times_by_split["train"]
        val_times = times_by_split["val"]
        test_times = times_by_split["test"]
        train_period_ids = periods_by_split["train"]
        val_period_ids = periods_by_split["val"]
        test_period_ids = periods_by_split["test"]

    print("\nShapes avant imputation/scaling")
    print(f"Train : X={X_train.shape} | y={y_train.shape}")
    print(f"Val   : X={X_val.shape} | y={y_val.shape}")
    print(f"Test  : X={X_test.shape} | y={y_test.shape}")

    X_train, X_val, X_test = fit_transform_train_only(
        X_train,
        X_val,
        X_test,
        output_dir=output_dir,
    )

    output_npz = output_dir / args.training_output_name

    np.savez_compressed(
        output_npz,
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_val=X_val,
        y_val=y_val.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        feature_names=np.asarray(feature_names),
        train_times=np.asarray(train_times),
        val_times=np.asarray(val_times),
        test_times=np.asarray(test_times),
        train_eruption_ids=np.asarray(train_period_ids),
        val_eruption_ids=np.asarray(val_period_ids),
        test_eruption_ids=np.asarray(test_period_ids),
    )

    with open(output_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    config = {
        "mode": "training",
        "periods": str(args.periods),
        "processed_csv_dir": str(args.processed_csv_dir),
        "output_npz": str(output_npz),
        "feature_window_minutes": args.feature_window_minutes,
        "seq_len": args.seq_len,
        "sequence_stride": args.sequence_stride,
        "max_horizon_hours": args.max_horizon_hours,
        "entropy_bins": args.entropy_bins,
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "include_post_eruption_as_zero": args.include_post_eruption_as_zero,
        "n_features": len(feature_names),
        "feature_contract_mode": feature_contract_mode,
        "feature_contract_source": feature_contract_source,
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
        "target": "multiclass eruption horizon",
        "class_mapping": {
            "0": "quiet / non-eruptive",
            "1": "36-48h before eruption",
            "2": "24-36h before eruption",
            "3": "12-24h before eruption",
            "4": "6-12h before eruption",
            "5": "0-6h before eruption",
        },
        "leakage_note": "chronological split allows one extracted period to work, but validation is less robust than a split by independent eruption periods.",
    }

    with open(output_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    validation_report = validate_npz(output_npz, n_classes=args.n_classes, mode="training")
    validation_path = output_dir / "npz_validation_report.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    print(json.dumps(validation_report, indent=2, ensure_ascii=False))
    print(f"Fichier training créé : {output_npz}")
    print(f"Rapport validation écrit : {validation_path}")

    if not validation_report["valid"]:
        raise SystemExit(1)


def main(args) -> None:
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train_ratio et val_ratio doivent être > 0.")

    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio doit être < 1.")

    periods_path = Path(args.periods)
    processed_csv_dir = Path(args.processed_csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = read_periods(
        path=periods_path,
        processed_csv_dir=processed_csv_dir,
        mode=args.mode,
        split_strategy=args.split_strategy,
    )

    resolved_metadata_path = output_dir / "metadata_resolved_local_csv.csv"
    meta.to_csv(resolved_metadata_path, index=False)
    print(f"Metadata résolu écrit : {resolved_metadata_path}")
    print("\nRésumé périodes :")
    print(meta.groupby(["period_type"]).size())

    if args.mode == "inference":
        run_inference_mode(args, meta)
    else:
        run_training_mode(args, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Préprocessing Vulcadata : CSV agrégés vers NPZ training ou NPZ inference."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "inference"],
        default="inference",
        help="Mode de sortie : training pour volcano_multi.npz, inference pour inference_source.npz.",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default=str(DEFAULT_PERIODS_CSV),
        help="CSV des périodes utilisées.",
    )
    parser.add_argument(
        "--processed-csv-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_CSV_DIR),
        help="Répertoire contenant les CSV agrégés.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Répertoire de sortie du NPZ et des artefacts.",
    )
    parser.add_argument(
        "--reference-artifacts-dir",
        type=str,
        default="",
        help="Répertoire de référence. En mode inference : feature_names.txt, imputer.joblib et scaler.joblib sont utilisés. En mode training : feature_names.txt est utilisé comme contrat de features si présent. Par défaut training : data/preprocessing/processed.",
    )
    parser.add_argument(
        "--training-output-name",
        type=str,
        default="volcano_multi.npz",
        help="Nom du NPZ final en mode training.",
    )
    parser.add_argument(
        "--inference-output-name",
        type=str,
        default="inference_source.npz",
        help="Nom du NPZ source en mode inference. Contient la clé X.",
    )
    parser.add_argument(
        "--feature-window-minutes",
        type=int,
        default=10,
        help="Fenêtre causale de calcul des features, en minutes.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=120,
        help="Longueur de séquence en pas minute.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=5,
        help="Pas entre deux séquences, en minutes.",
    )
    parser.add_argument(
        "--max-horizon-hours",
        type=float,
        default=48.0,
        help="Horizon maximal avant éruption conservé dans le dataset training.",
    )
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=20,
        help="Nombre de bins pour Shannon Entropy.",
    )
    parser.add_argument(
        "--include-post-eruption-as-zero",
        action="store_true",
        help="Garder les lignes après début d'éruption avec y=0. Désactivé par défaut.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=6,
        help="Nombre de classes attendues.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["chronological", "manual"],
        default="chronological",
        help="Stratégie de split training. chronological découpe les séquences générées, manual utilise la colonne split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Part train en split chronological.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Part validation en split chronological. Le reste devient test.",
    )

    main(parser.parse_args())
