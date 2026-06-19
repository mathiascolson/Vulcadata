# prepare_volcano_sequences.py

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv

import joblib
import numpy as np
import pandas as pd

from scipy.stats import kurtosis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# CHARGEMENT .ENV
# ============================================================

# Le script est dans /scripts ; le .env est attendu à la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv(override=False)


# ============================================================
# CONFIG PAR DÉFAUT
# ============================================================

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


# ============================================================
# FEATURES
# ============================================================

def shannon_entropy(values: np.ndarray, bins: int = 20) -> float:
    """
    Shannon entropy sur une fenêtre de valeurs.
    Calcul causal : uniquement les valeurs déjà présentes dans la fenêtre passée.
    """
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
    """
    Kurtosis robuste.
    fisher=True donne 0 pour une distribution normale.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 4:
        return np.nan

    if np.nanstd(values) == 0:
        return 0.0

    return float(kurtosis(values, fisher=True, bias=False))


def slope_last_window(values: np.ndarray) -> float:
    """
    Pente linéaire simple sur la fenêtre.
    Utile pour capter une hausse ou baisse progressive du signal.
    """
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
    """
    Calcule des features causales pour un couple station/channel.
    Chaque ligne à t utilise uniquement les observations <= t.
    """
    g = g.sort_values("time_min").copy()

    # Features directes disponibles au pas minute
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
        center=False,  # CRITIQUE : pas de futur
    )

    for col in base_cols:
        g[f"{col}_roll_mean"] = roll[col].mean()
        g[f"{col}_roll_std"] = roll[col].std()
        g[f"{col}_roll_min"] = roll[col].min()
        g[f"{col}_roll_max"] = roll[col].max()
        g[f"{col}_roll_median"] = roll[col].median()

    # Indicateurs inspirés de l'étude
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

    # Enveloppes causales légères.
    # Attention : center=False. Pas de rolling centré.
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


def build_features_for_eruption(
    df: pd.DataFrame,
    feature_window_minutes: int,
    entropy_bins: int,
) -> pd.DataFrame:
    """
    Produit une table temporelle réseau :
    index = time_min
    colonnes = features par station/channel + features réseau globales.
    """
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

    # Pivot réseau : une ligne par minute, colonnes station/channel/feature.
    wide = feat_long.pivot_table(
        index="time_min",
        columns="sensor",
        values=numeric_feature_cols,
        aggfunc="mean",
    )

    # Flatten MultiIndex : feature__station__channel
    wide.columns = [f"{feature}__{sensor}" for feature, sensor in wide.columns]
    wide = wide.sort_index()

    # Features globales réseau, calculées au même instant t.
    # Elles n'utilisent pas le futur : seulement les colonnes disponibles à t.
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


# ============================================================
# SÉQUENCES ET LABELS
# ============================================================


def normalize_period_type(value: str) -> str:
    """
    Normalise le type de période.

    Valeurs acceptées :
    - eruption / eruptive / event
    - quiet / calm / calme / background / non_eruptive
    """
    value = str(value).strip().lower()

    if value in {"eruption", "eruptive", "event"}:
        return "eruption"

    if value in {"quiet", "calm", "calme", "background", "non_eruptive"}:
        return "quiet"

    raise ValueError(
        f"period_type invalide : {value}. "
        "Valeurs attendues : eruption ou quiet."
    )


def infer_period_type(row) -> str:
    """
    Déduit le type de période.

    Priorité :
    1. colonne period_type si elle existe ;
    2. eruption_id commençant par quiet_ ;
    3. sinon eruption.
    """
    if "period_type" in row.index and pd.notna(row["period_type"]):
        return normalize_period_type(row["period_type"])

    eruption_id = str(row.get("eruption_id", "")).strip().lower()

    if eruption_id.startswith("quiet_"):
        return "quiet"

    return "eruption"


def delay_hours_to_multiclass_label(delay_hours: float) -> int:
    """
    Convertit un délai avant éruption en classe.

    Convention :
        0 = calme / non-éruptif
        1 = 36-48h avant éruption
        2 = 24-36h avant éruption
        3 = 12-24h avant éruption
        4 = 6-12h avant éruption
        5 = 0-6h avant éruption
    """
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


def make_sequences_for_quiet_period(
    feature_table: pd.DataFrame,
    seq_len: int,
    sequence_stride: int = 1,
    quiet_label: int = 0,
):
    """
    Construit les séquences X pour une période calme.

    Toutes les séquences reçoivent le label quiet_label.
    Par convention : y = 0 pour calme / non-éruptif.
    """
    if sequence_stride < 1:
        raise ValueError("sequence_stride doit être >= 1.")

    feature_table = feature_table.sort_index().copy()

    X_list = []
    y_list = []
    t_end_list = []

    values = feature_table.to_numpy(dtype=np.float32)

    for end_idx in range(seq_len - 1, len(feature_table), sequence_stride):
        t_end = pd.Timestamp(feature_table.index[end_idx]).tz_convert("UTC")
        start_idx = end_idx - seq_len + 1
        seq = values[start_idx : end_idx + 1]

        X_list.append(seq)
        y_list.append(quiet_label)
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

def make_sequences_for_eruption(
    feature_table: pd.DataFrame,
    eruption_start_utc: pd.Timestamp,
    seq_len: int,
    max_horizon_hours: float,
    include_post_eruption_as_zero: bool = False,
    sequence_stride: int = 1,
):
    """
    Construit les séquences X et la cible y multi-classes.

    Pour chaque séquence :
        X_i = features de [t-seq_len+1, ..., t]
        y_i = classe d'horizon avant éruption à t

    Convention :
        0 = calme / non-éruptif
        1 = 36-48h avant éruption
        2 = 24-36h avant éruption
        3 = 12-24h avant éruption
        4 = 6-12h avant éruption
        5 = 0-6h avant éruption

    Aucun point futur n'est utilisé dans X.
    """
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

# ============================================================
# SPLIT SANS FUITE
# ============================================================

def flatten_for_sklearn(X: np.ndarray) -> np.ndarray:
    """
    Transforme (N, T, F) en (N*T, F) pour imputer/scaler feature par feature.
    """
    n, t, f = X.shape
    return X.reshape(n * t, f)


def restore_from_sklearn(X_flat: np.ndarray, original_shape) -> np.ndarray:
    n, t, f = original_shape
    return X_flat.reshape(n, t, f).astype(np.float32)


def fit_transform_train_only(X_train, X_val, X_test, output_dir: Path):
    """
    Imputation + standardisation.
    Fit uniquement sur train.

    Version mémoire optimisée :
    - pas de flatten massif en (N*T, F)
    - pas de SimpleImputer sklearn, qui crée un masque booléen énorme
    - calcul feature par feature
    - transformation in-place sur X_train, X_val, X_test
    """
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
            # Feature absente de tout le train.
            # On impute à 0 et on garde scale=1.
            median_j = np.float32(0.0)

        medians[j] = median_j

        # Imputation in-place sur train / val / test.
        for X in (X_train, X_val, X_test):
            col = X[:, :, j]
            bad_mask = ~np.isfinite(col)
            if bad_mask.any():
                col[bad_mask] = median_j

        # Moyenne et écart-type fit uniquement sur train après imputation.
        train_col = X_train[:, :, j]

        mean_j = np.mean(train_col, dtype=np.float64)
        std_j = np.std(train_col, dtype=np.float64)

        if not np.isfinite(mean_j):
            mean_j = 0.0

        if not np.isfinite(std_j) or std_j == 0.0:
            std_j = 1.0

        means[j] = np.float32(mean_j)
        scales[j] = np.float32(std_j)

        # Standardisation in-place sur train / val / test.
        for X in (X_train, X_val, X_test):
            X[:, :, j] = (X[:, :, j] - means[j]) / scales[j]

        if (j + 1) % 100 == 0 or (j + 1) == n_features:
            print(f"      Features traitées : {j + 1}/{n_features}")

    # On garde les noms de fichiers attendus par le reste du pipeline.
    # Ce ne sont pas des objets sklearn, mais des dictionnaires de paramètres.
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


# ============================================================
# S3 I/O
# ============================================================

def get_s3_bucket(args) -> str | None:
    """
    Résout le bucket S3 depuis les arguments ou les variables d'environnement.
    """
    return (
        args.s3_bucket
        or os.getenv("S3_BUCKET_NAME")
        or os.getenv("AWS_S3_BUCKET_NAME")
    )


def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse une URI S3 du type s3://bucket/key.
    """
    parsed = urlparse(s3_uri)

    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"URI S3 invalide : {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not key:
        raise ValueError(f"URI S3 sans key : {s3_uri}")

    return bucket, key


def build_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def safe_filename(value: str, suffix: str = ".csv") -> str:
    """
    Produit un nom de fichier local stable à partir d'un eruption_id.
    """
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    ).strip("_")

    if not cleaned:
        cleaned = "eruption"

    if not cleaned.endswith(suffix):
        cleaned += suffix

    return cleaned


def get_s3_client(args):
    """
    Crée un client S3 standard.
    Les credentials sont résolus par boto3 après chargement du .env :
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION / AWS_REGION
    """
    region = (
        args.aws_region
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_REGION")
    )

    client_kwargs = {}

    if region:
        client_kwargs["region_name"] = region

    return boto3.client("s3", **client_kwargs)


def download_s3_file(s3_client, bucket: str, key: str, local_path: Path) -> Path:
    """
    Télécharge un objet S3 vers un fichier local.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"      Téléchargement S3 : {build_s3_uri(bucket, key)}")
    print(f"      Vers local        : {local_path}")

    s3_client.download_file(bucket, key, str(local_path))

    if not local_path.exists():
        raise FileNotFoundError(f"Téléchargement S3 échoué : {local_path}")

    return local_path


def upload_s3_file(s3_client, local_path: Path, bucket: str, key: str) -> str:
    """
    Upload un fichier local vers S3.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Fichier local introuvable pour upload S3 : {local_path}")

    print(f"      Upload S3 : {local_path} → {build_s3_uri(bucket, key)}")
    s3_client.upload_file(str(local_path), bucket, key)

    return build_s3_uri(bucket, key)


def download_metadata_if_needed(
    metadata_arg: str,
    output_dir: Path,
    s3_client,
) -> Path:
    """
    Autorise un metadata local ou un metadata stocké sur S3.
    """
    if not is_s3_uri(metadata_arg):
        return Path(metadata_arg)

    bucket, key = parse_s3_uri(metadata_arg)
    local_path = output_dir / "metadata_from_s3" / Path(key).name

    print("\n[0/5] Téléchargement du metadata depuis S3")
    return download_s3_file(s3_client, bucket, key, local_path)


def resolve_row_s3_csv(row, args, default_bucket: str | None) -> tuple[str, str] | None:
    """
    Résout le bucket/key du CSV d'une éruption.

    Cas acceptés :
    - colonne s3_csv_path = s3://bucket/key.csv
    - colonne csv_path = s3://bucket/key.csv
    - colonne s3_csv_key = key.csv, avec bucket fourni par --s3-bucket ou env
    - colonne s3_bucket optionnelle dans le metadata pour surcharger le bucket ligne par ligne
    """
    row_bucket = None

    if "s3_bucket" in row.index and pd.notna(row["s3_bucket"]):
        row_bucket = str(row["s3_bucket"]).strip()

    if "s3_csv_path" in row.index and pd.notna(row["s3_csv_path"]):
        s3_csv_path = str(row["s3_csv_path"]).strip()

        if not s3_csv_path:
            raise ValueError(
                f"s3_csv_path vide pour eruption_id={row.get('eruption_id', 'UNKNOWN')}"
            )

        if is_s3_uri(s3_csv_path):
            return parse_s3_uri(s3_csv_path)

        bucket = row_bucket or default_bucket
        if not bucket:
            raise ValueError(
                "s3_csv_path est fourni sans URI s3:// complète, mais aucun bucket "
                "S3 n'est disponible. Renseigner --s3-bucket, S3_BUCKET_NAME, "
                "AWS_S3_BUCKET_NAME, ou une colonne s3_bucket dans le metadata."
            )

        return bucket, s3_csv_path

    csv_path = None
    if "csv_path" in row.index and pd.notna(row["csv_path"]):
        csv_path = str(row["csv_path"]).strip()

    if csv_path and is_s3_uri(csv_path):
        return parse_s3_uri(csv_path)

    if "s3_csv_key" in row.index and pd.notna(row["s3_csv_key"]):
        key = str(row["s3_csv_key"]).strip()

        if is_s3_uri(key):
            return parse_s3_uri(key)

        bucket = row_bucket or default_bucket
        if not bucket:
            raise ValueError(
                "Un s3_csv_key est fourni, mais aucun bucket S3 n'est disponible. "
                "Renseigner --s3-bucket, S3_BUCKET_NAME, AWS_S3_BUCKET_NAME, "
                "ou une colonne s3_bucket dans le metadata."
            )

        return bucket, key

    if csv_path and args.s3_csv_prefix:
        bucket = row_bucket or default_bucket
        if not bucket:
            raise ValueError(
                "--s3-csv-prefix est fourni, mais aucun bucket S3 n'est disponible. "
                "Renseigner --s3-bucket, S3_BUCKET_NAME ou AWS_S3_BUCKET_NAME."
            )

        key = f"{args.s3_csv_prefix.rstrip('/')}/{Path(csv_path).name}"
        return bucket, key

    return None


def materialize_csvs_from_s3(
    meta: pd.DataFrame,
    args,
    output_dir: Path,
    s3_client,
    default_bucket: str | None,
) -> pd.DataFrame:
    """
    Transforme les chemins S3 du metadata en chemins locaux utilisables par pd.read_csv.

    Le reste du pipeline conserve donc sa logique existante :
    il continue de lire une colonne csv_path locale.
    """
    meta = meta.copy()
    local_csv_dir = Path(args.local_csv_dir) if args.local_csv_dir else output_dir / "input_csv_from_s3"
    local_csv_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []

    print("\n[1/5] Résolution des CSV d'entrée")

    for _, row in meta.iterrows():
        eruption_id = str(row["eruption_id"])
        s3_location = resolve_row_s3_csv(row, args, default_bucket)

        csv_path = None
        if "csv_path" in row.index and pd.notna(row["csv_path"]):
            csv_path = str(row["csv_path"]).strip()

        if s3_location is None:
            if not csv_path:
                raise ValueError(
                    f"Aucun csv_path, s3_csv_path ou s3_csv_key fourni pour eruption_id={eruption_id}."
                )

            local_path = Path(csv_path)

            if not local_path.exists():
                raise FileNotFoundError(
                    f"CSV local introuvable pour eruption_id={eruption_id} : {local_path}"
                )

            print(f"      CSV local conservé : {eruption_id} → {local_path}")
            local_paths.append(str(local_path))
            continue

        bucket, key = s3_location
        local_path = local_csv_dir / safe_filename(eruption_id, suffix=".csv")

        download_s3_file(
            s3_client=s3_client,
            bucket=bucket,
            key=key,
            local_path=local_path,
        )

        local_paths.append(str(local_path))

    meta["csv_path"] = local_paths

    resolved_metadata_path = output_dir / "metadata_resolved_local_csv.csv"
    meta.to_csv(resolved_metadata_path, index=False)

    print(f"      Metadata résolu écrit : {resolved_metadata_path}")

    return meta


def upload_preprocessing_outputs_to_s3(
    output_dir: Path,
    output_name: str,
    args,
    s3_client,
    bucket: str,
) -> dict:
    """
    Upload les artefacts produits par le preprocessing vers S3.
    """
    prefix = args.s3_output_prefix.strip().strip("/")

    expected_outputs = [
        output_dir / output_name,
        output_dir / "imputer.joblib",
        output_dir / "scaler.joblib",
        output_dir / "preprocessing_config.json",
        output_dir / "feature_names.txt",
    ]

    uploaded = {}

    print("\n[5/5] Upload des sorties de preprocessing vers S3")

    for local_path in expected_outputs:
        key = f"{prefix}/{local_path.name}" if prefix else local_path.name
        uploaded[str(local_path)] = upload_s3_file(
            s3_client=s3_client,
            local_path=local_path,
            bucket=bucket,
            key=key,
        )

    manifest_path = output_dir / "s3_outputs_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(uploaded, f, indent=2, ensure_ascii=False)

    manifest_key = f"{prefix}/{manifest_path.name}" if prefix else manifest_path.name
    uploaded[str(manifest_path)] = upload_s3_file(
        s3_client=s3_client,
        local_path=manifest_path,
        bucket=bucket,
        key=manifest_key,
    )

    return uploaded


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3_client = get_s3_client(args)
    s3_bucket = get_s3_bucket(args)

    metadata_path = download_metadata_if_needed(
        metadata_arg=args.metadata,
        output_dir=output_dir,
        s3_client=s3_client,
    )

    meta = pd.read_csv(metadata_path)

    required_meta = {"eruption_id"}
    missing_meta = required_meta - set(meta.columns)
    if missing_meta:
        raise ValueError(f"Colonnes manquantes dans metadata : {sorted(missing_meta)}")

    if "period_type" not in meta.columns:
        print(
            "WARNING : colonne period_type absente du metadata. "
            "Les lignes dont eruption_id commence par quiet_ seront traitées comme calmes. "
            "Les autres seront traitées comme éruptives."
        )

    if "eruption_start_utc" not in meta.columns:
        meta["eruption_start_utc"] = pd.NA

    has_csv_path = "csv_path" in meta.columns
    has_s3_csv_key = "s3_csv_key" in meta.columns
    has_s3_csv_path = "s3_csv_path" in meta.columns

    if not has_csv_path and not has_s3_csv_key and not has_s3_csv_path:
        raise ValueError(
            "Le fichier metadata doit contenir soit une colonne csv_path, "
            "soit une colonne s3_csv_key, soit une colonne s3_csv_path."
        )

    if "split" not in meta.columns:
        raise ValueError(
            "Le fichier metadata doit contenir une colonne split "
            "avec les valeurs train, val ou test. "
            "Avec 4 ou 5 éruptions, le split doit être décidé par éruption."
        )

    allowed_splits = {"train", "val", "test"}
    bad_splits = set(meta["split"].unique()) - allowed_splits
    if bad_splits:
        raise ValueError(f"Valeurs split invalides : {bad_splits}")

    for _, row in meta.iterrows():
        period_type = infer_period_type(row)
        eruption_id = str(row["eruption_id"])

        if period_type == "eruption":
            if pd.isna(row["eruption_start_utc"]) or str(row["eruption_start_utc"]).strip() == "":
                raise ValueError(
                    f"eruption_start_utc manquant pour la période éruptive : {eruption_id}"
                )

    meta = materialize_csvs_from_s3(
        meta=meta,
        args=args,
        output_dir=output_dir,
        s3_client=s3_client,
        default_bucket=s3_bucket,
    )

    all_feature_tables = {}
    all_columns = None

    # Première passe : features par éruption.
    for _, row in meta.iterrows():
        eruption_id = str(row["eruption_id"])
        csv_path = Path(row["csv_path"])

        print(f"\n[2/5] Lecture et features : {eruption_id}")
        print(f"      CSV : {csv_path}")

        df = pd.read_csv(csv_path)

        feature_table = build_features_for_eruption(
            df,
            feature_window_minutes=args.feature_window_minutes,
            entropy_bins=args.entropy_bins,
        )

        all_feature_tables[eruption_id] = feature_table

        if all_columns is None:
            all_columns = set(feature_table.columns)
        else:
            all_columns = all_columns.union(set(feature_table.columns))

        print(f"      Table features : {feature_table.shape}")

    # Schéma de colonnes commun.
    # Important : on fixe les colonnes avant de construire X.
    feature_names = sorted(all_columns)

    X_by_split = {"train": [], "val": [], "test": []}
    y_by_split = {"train": [], "val": [], "test": []}
    times_by_split = {"train": [], "val": [], "test": []}
    eruptions_by_split = {"train": [], "val": [], "test": []}

    # Deuxième passe : alignement colonnes + séquences.
    for _, row in meta.iterrows():
        eruption_id = str(row["eruption_id"])
        split = str(row["split"])
        period_type = infer_period_type(row)

        print(f"\n[3/5] Séquences : {eruption_id} → {split}")
        print(f"      Type période : {period_type}")

        ft = all_feature_tables[eruption_id]
        ft = ft.reindex(columns=feature_names)

        if period_type == "quiet":
            X, y, t_end = make_sequences_for_quiet_period(
                feature_table=ft,
                seq_len=args.seq_len,
                sequence_stride=args.sequence_stride,
                quiet_label=0,
            )
        else:
            eruption_start = pd.Timestamp(row["eruption_start_utc"], tz="UTC")

            X, y, t_end = make_sequences_for_eruption(
                feature_table=ft,
                eruption_start_utc=eruption_start,
                seq_len=args.seq_len,
                max_horizon_hours=args.max_horizon_hours,
                include_post_eruption_as_zero=args.include_post_eruption_as_zero,
                sequence_stride=args.sequence_stride,
            )

        print(f"      X : {X.shape} | y : {y.shape}")

        if len(y) == 0:
            print(f"      WARNING : aucune séquence produite pour {eruption_id}.")
            continue

        X_by_split[split].append(X)
        y_by_split[split].append(y)
        times_by_split[split].extend(t_end)
        eruptions_by_split[split].extend([eruption_id] * len(y))

    def concat_split(split):
        if not X_by_split[split]:
            raise ValueError(f"Aucune donnée pour le split {split}.")
        X = np.concatenate(X_by_split[split], axis=0)
        y = np.concatenate(y_by_split[split], axis=0)
        return X, y

    X_train, y_train = concat_split("train")
    X_val, y_val = concat_split("val")
    X_test, y_test = concat_split("test")

    print("\n[4/5] Shapes avant imputation/scaling")
    print(f"      Train : X={X_train.shape} | y={y_train.shape}")
    print(f"      Val   : X={X_val.shape} | y={y_val.shape}")
    print(f"      Test  : X={X_test.shape} | y={y_test.shape}")

    # Fit imputer/scaler uniquement sur train.
    X_train, X_val, X_test = fit_transform_train_only(
        X_train,
        X_val,
        X_test,
        output_dir=output_dir,
    )

    print("\n[4/5] Sauvegarde volcano_multi.npz")

    output_npz = output_dir / args.output_name

    np.savez_compressed(
        output_npz,
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_val=X_val,
        y_val=y_val.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        feature_names=np.asarray(feature_names),
        train_times=np.asarray(times_by_split["train"]),
        val_times=np.asarray(times_by_split["val"]),
        test_times=np.asarray(times_by_split["test"]),
        train_eruption_ids=np.asarray(eruptions_by_split["train"]),
        val_eruption_ids=np.asarray(eruptions_by_split["val"]),
        test_eruption_ids=np.asarray(eruptions_by_split["test"]),
    )

    output_s3_uri = None
    if args.upload_to_s3:
        if not s3_bucket:
            raise ValueError(
                "Upload S3 demandé, mais aucun bucket S3 n'est disponible. "
                "Renseigner --s3-bucket, S3_BUCKET_NAME ou AWS_S3_BUCKET_NAME."
            )

        prefix = args.s3_output_prefix.strip().strip("/")
        output_s3_uri = build_s3_uri(
            s3_bucket,
            f"{prefix}/{args.output_name}" if prefix else args.output_name,
        )

    config = {
        "metadata": str(metadata_path),
        "resolved_metadata": str(output_dir / "metadata_resolved_local_csv.csv"),
        "output_npz": str(output_npz),
        "output_s3_uri": output_s3_uri,
        "s3_output_bucket": s3_bucket if args.upload_to_s3 else None,
        "s3_output_prefix": args.s3_output_prefix if args.upload_to_s3 else None,
        "feature_window_minutes": args.feature_window_minutes,
        "seq_len": args.seq_len,
        "sequence_stride": args.sequence_stride,
        "max_horizon_hours": args.max_horizon_hours,
        "entropy_bins": args.entropy_bins,
        "include_post_eruption_as_zero": args.include_post_eruption_as_zero,
        "n_features": len(feature_names),
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
        "leakage_controls": [
            "split by eruption_id",
            "no shuffle split",
            "sequence stride used to reduce overlap between consecutive sequences",
            "causal rolling only, center=False",
            "label computed at sequence end timestamp",
            "imputer fitted only on train",
            "scaler fitted only on train",
            "post-eruption rows excluded by default",
            "quiet periods have no eruption_start_utc and are labelled as class 0",
        ],
    }

    with open(output_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(output_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    uploaded_outputs = None
    if args.upload_to_s3:
        uploaded_outputs = upload_preprocessing_outputs_to_s3(
            output_dir=output_dir,
            output_name=args.output_name,
            args=args,
            s3_client=s3_client,
            bucket=s3_bucket,
        )

    print(f"\nFichier créé : {output_npz}")

    if uploaded_outputs:
        print("\nSorties uploadées sur S3 :")
        for _, s3_uri in uploaded_outputs.items():
            print(f"  - {s3_uri}")

    print(f"Nombre de features : {len(feature_names)}")
    print("Préprocessing terminé sans split aléatoire.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help=(
            "Chemin vers eruptions_metadata.csv. "
            "Peut être un chemin local ou une URI s3://bucket/key.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed",
        help="Répertoire de sortie",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="volcano_multi.npz",
        help="Nom du fichier npz final",
    )

    parser.add_argument(
        "--feature-window-minutes",
        type=int,
        default=10,
        help="Fenêtre causale de calcul des features, en minutes",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=120,
        help=(
            "Longueur de séquence en pas minute. "
            "120 = 2 heures si les CSV sont agrégés à la minute."
        ),
    )
    
    parser.add_argument(
    "--sequence-stride",
    type=int,
    default=5,
    help=(
        "Pas entre deux séquences, en minutes. "
        "1 = toutes les minutes ; 5 ou 10 réduit la redondance et l'overfit."
    ),
    )

    parser.add_argument(
        "--max-horizon-hours",
        type=float,
        default=48.0,
        help="Horizon maximal avant éruption conservé dans le dataset",
    )

    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=20,
        help="Nombre de bins pour Shannon Entropy",
    )

    parser.add_argument(
        "--include-post-eruption-as-zero",
        action="store_true",
        help=(
            "Si activé, les lignes après le début de l'éruption sont gardées "
            "avec y=0. Par défaut, elles sont exclues."
        ),
    )

    # ============================================================
    # ARGUMENTS S3
    # ============================================================

    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help=(
            "Bucket S3 utilisé pour télécharger les CSV d'entrée et uploader "
            "les sorties. Si absent, utilise S3_BUCKET_NAME ou AWS_S3_BUCKET_NAME."
        ),
    )

    parser.add_argument(
        "--s3-csv-prefix",
        type=str,
        default=None,
        help=(
            "Préfixe S3 optionnel où chercher les CSV si csv_path contient "
            "seulement des noms de fichiers locaux."
        ),
    )

    parser.add_argument(
        "--local-csv-dir",
        type=str,
        default=None,
        help=(
            "Répertoire local temporaire où télécharger les CSV depuis S3. "
            "Par défaut : <output-dir>/input_csv_from_s3."
        ),
    )

    parser.add_argument(
        "--s3-output-prefix",
        type=str,
        default="volcano/preprocessing/processed",
        help="Préfixe S3 où uploader les sorties du preprocessing.",
    )

    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Uploader les sorties du preprocessing vers S3 à la fin du traitement.",
    )

    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help=(
            "Région AWS optionnelle. Si absente, utilise AWS_DEFAULT_REGION "
            "ou AWS_REGION."
        ),
    )

    args = parser.parse_args()
    main(args)