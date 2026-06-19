# scripts/prepare_volcano_sequences_core.py

"""
Préprocessing Vulcadata - version core features.

Ce script réutilise le pipeline existant de prepare_volcano_sequences.py
mais remplace uniquement la construction des features.

Features conservées :
- frequency_index
- energy_low_1_5_5
- energy_high_6_16
- shannon_entropy
- shannon_entropy_env
- kurtosis
- kurtosis_env
- amplitude_mean_slope
- agrégats réseau associés : mean/std/min/max

Objectif :
- réduire fortement le nombre de features
- conserver les indicateurs sismiques les plus interprétables
- limiter le surapprentissage station/channel
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Le script est placé dans /scripts.
# Comme prepare_volcano_sequences.py est dans le même dossier,
# cet import fonctionne si la commande est lancée depuis la racine du projet.
import prepare_volcano_sequences as base


CORE_FEATURE_COLUMNS = [
    "frequency_index",
    "energy_low_1_5_5",
    "energy_high_6_16",
    "shannon_entropy",
    "shannon_entropy_env",
    "kurtosis",
    "kurtosis_env",
    "amplitude_mean_slope",
]


def build_features_for_eruption_core(
    df: pd.DataFrame,
    feature_window_minutes: int,
    entropy_bins: int,
) -> pd.DataFrame:
    """
    Produit une table temporelle réseau réduite.

    Contrairement au preprocessing full, on ne conserve pas toutes les colonnes
    numériques. On garde uniquement les indicateurs sismiques interprétables :

    - Frequency Index
    - énergies basse et haute fréquence
    - Shannon entropy
    - Kurtosis
    - pente d'amplitude
    - enveloppes causales entropy/kurtosis
    - agrégats réseau associés
    """
    missing = set(base.REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {sorted(missing)}")

    required_core_input = {
        "energy_low_1_5_5",
        "energy_high_6_16",
        "frequency_index",
    }

    missing_core = required_core_input - set(df.columns)
    if missing_core:
        raise ValueError(
            "Colonnes nécessaires au mode core absentes du CSV : "
            f"{sorted(missing_core)}"
        )

    df = df.copy()
    df["time_min"] = pd.to_datetime(df["time_min"], utc=True, errors="coerce")

    if df["time_min"].isna().any():
        n_bad = int(df["time_min"].isna().sum())
        raise ValueError(f"{n_bad} lignes ont un time_min invalide.")

    numeric_cols_to_convert = [
        "amplitude_mean",
        "amplitude_std",
        "amplitude_max",
        "amplitude_min",
        "amplitude_count",
        "energy_low_1_5_5",
        "energy_high_6_16",
        "frequency_index",
    ]

    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["station"] = df["station"].astype(str)
    df["channel"] = df["channel"].astype(str)
    df["sensor"] = df["station"] + "__" + df["channel"]

    df = df.sort_values(["sensor", "time_min"])

    enriched_parts = []

    for _, g in df.groupby("sensor", sort=False):
        enriched = base.add_causal_rolling_features_one_group(
            g,
            feature_window_minutes=feature_window_minutes,
            entropy_bins=entropy_bins,
        )
        enriched_parts.append(enriched)

    feat_long = pd.concat(enriched_parts, axis=0, ignore_index=True)

    available_core_features = [
        col
        for col in CORE_FEATURE_COLUMNS
        if col in feat_long.columns and pd.api.types.is_numeric_dtype(feat_long[col])
    ]

    missing_after_enrichment = set(CORE_FEATURE_COLUMNS) - set(available_core_features)

    if missing_after_enrichment:
        print(
            "      WARNING : certaines core features sont absentes ou non numériques : "
            f"{sorted(missing_after_enrichment)}"
        )

    if not available_core_features:
        raise ValueError("Aucune core feature disponible après enrichissement.")

    # Pivot réseau : une ligne par minute, colonnes feature__station__channel.
    wide = feat_long.pivot_table(
        index="time_min",
        columns="sensor",
        values=available_core_features,
        aggfunc="mean",
    )

    wide.columns = [f"{feature}__{sensor}" for feature, sensor in wide.columns]
    wide = wide.sort_index()

    # Agrégats réseau associés aux core features.
    # On agrège à chaque minute sur les stations/channels disponibles à t.
    for feature in available_core_features:
        sensor_cols = [c for c in wide.columns if c.startswith(f"{feature}__")]

        if not sensor_cols:
            continue

        network_prefix = f"network_{feature}"

        wide[f"{network_prefix}_mean"] = wide[sensor_cols].mean(axis=1)
        wide[f"{network_prefix}_std"] = wide[sensor_cols].std(axis=1)
        wide[f"{network_prefix}_min"] = wide[sensor_cols].min(axis=1)
        wide[f"{network_prefix}_max"] = wide[sensor_cols].max(axis=1)

    return wide


def main(args):
    print("\nMode preprocessing : CORE FEATURES")
    print("Features conservées :")
    for feature in CORE_FEATURE_COLUMNS:
        print(f"  - {feature}")
    print("Agrégats réseau ajoutés : mean/std/min/max par feature core\n")

    # Remplacement de la fonction de construction des features dans le pipeline existant.
    base.build_features_for_eruption = build_features_for_eruption_core

    # Réutilisation intégrale du pipeline existant :
    # - S3
    # - résolution metadata
    # - génération séquences
    # - sequence_stride
    # - split train/val/test
    # - imputation/scaling train only
    # - sauvegarde .npz
    base.main(args)


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
        default="data/preprocessing/processed_core",
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
        default=3,
        help=(
            "Pas entre deux séquences, en minutes. "
            "1 = toutes les minutes ; 3/5 réduit la redondance et l'overfit."
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
        default="volcano/preprocessing/processed_core",
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