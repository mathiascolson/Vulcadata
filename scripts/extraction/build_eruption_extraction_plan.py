# scripts/build_eruption_extraction_plan.py

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG PAR DÉFAUT
# ============================================================

DEFAULT_NETWORK = "PF"
DEFAULT_STATIONS = "CSS,DSO,ENO,FJS,HIM,SNE"
DEFAULT_CHANNELS = "HHZ,EHZ,HHE,HHN"
DEFAULT_S3_BUCKET = "vulcadata"

DEFAULT_PRE_HOURS = 48.0
DEFAULT_POST_HOURS = 6.0
DEFAULT_MAX_DURATION_DAYS = 5.0
DEFAULT_MIN_YEAR = 2015
DEFAULT_N_ERUPTIONS = 10

EXPECTED_COLUMNS = [
    "Type",
    "Date début",
    "Heure début (UTC)",
    "Date fin",
    "Heure fin (UTC)",
    "Durée (jours)",
    "Localisation",
    "Nom",
]


# ============================================================
# HELPERS TEXTE / DATE / HEURE
# ============================================================

def normalize_colname(name: str) -> str:
    """
    Normalise les noms de colonnes provenant d'Excel.
    """
    name = str(name).strip()
    name = unicodedata.normalize("NFKC", name)
    name = name.replace("\xa0", " ")
    name = re.sub(r"\s+", " ", name)
    return name


def clean_text(value) -> str:
    """
    Nettoyage robuste des cellules texte.
    """
    if pd.isna(value):
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()

    return value


def excel_date_to_timestamp(value):
    """
    Convertit une date Excel numérique ou textuelle en Timestamp sans timezone.

    Excel stocke souvent les dates sous forme de nombre de jours depuis 1899-12-30.
    Exemple :
      45109 -> 2023-07-02
    """
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, (int, float)) and not pd.isna(value):
        return pd.to_datetime(
            value,
            unit="D",
            origin="1899-12-30",
            utc=False,
        ).normalize()

    value_str = clean_text(value)

    if not value_str:
        return pd.NaT

    # Cas d'une date Excel numérique stockée en texte.
    try:
        numeric = float(value_str.replace(",", "."))
        if numeric > 20000:
            return pd.to_datetime(
                numeric,
                unit="D",
                origin="1899-12-30",
                utc=False,
            ).normalize()
    except ValueError:
        pass

    return pd.to_datetime(value_str, dayfirst=True, errors="coerce").normalize()


def excel_time_to_timedelta(value):
    """
    Convertit une heure Excel ou textuelle en Timedelta.

    Exemples :
      0.5       -> 12:00
      0.020833  -> 00:30
      "03:48"   -> 03:48
      "03h48"   -> 03:48
    """
    if pd.isna(value):
        return pd.Timedelta(0)

    if isinstance(value, pd.Timestamp):
        return pd.Timedelta(
            hours=value.hour,
            minutes=value.minute,
            seconds=value.second,
        )

    if isinstance(value, (int, float)) and not pd.isna(value):
        numeric = float(value)
        frac = numeric % 1
        return pd.to_timedelta(frac, unit="D")

    value_str = clean_text(value).lower()

    if not value_str:
        return pd.Timedelta(0)

    value_str = value_str.replace("h", ":")
    value_str = value_str.replace(".", ":")

    # Format HH:MM ou HH:MM:SS
    match = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", value_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3) or 0)
        return pd.Timedelta(hours=hour, minutes=minute, seconds=second)

    # Fraction de journée Excel stockée en texte.
    try:
        numeric = float(value_str.replace(",", "."))
        frac = numeric % 1
        return pd.to_timedelta(frac, unit="D")
    except ValueError:
        pass

    return pd.Timedelta(0)


def combine_date_time(date_value, time_value):
    """
    Combine Date + Heure UTC en timestamp timezone-aware UTC.
    """
    date_part = excel_date_to_timestamp(date_value)
    time_part = excel_time_to_timedelta(time_value)

    if pd.isna(date_part):
        return pd.NaT

    ts = date_part + time_part

    if ts.tzinfo is None:
        return pd.Timestamp(ts).tz_localize("UTC")

    return pd.Timestamp(ts).tz_convert("UTC")


def slugify(value: str) -> str:
    """
    Transforme un nom en identifiant stable.
    """
    value = clean_text(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def make_eruption_id(start_utc, name, location) -> str:
    """
    Crée un ID stable d'éruption.
    """
    date_part = pd.Timestamp(start_utc).strftime("%Y_%m_%d")

    name_slug = slugify(name)
    location_slug = slugify(location)

    if name_slug:
        return f"eruption_{date_part}_{name_slug}"

    if location_slug:
        return f"eruption_{date_part}_{location_slug}"

    return f"eruption_{date_part}"


def to_iso_z(ts) -> str:
    """
    Convertit un timestamp UTC en chaîne ISO propre.
    """
    if pd.isna(ts):
        return ""

    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# SPLIT
# ============================================================

def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split par éruption, sans fuite.

    Pour 10 éruptions :
      - 8 anciennes en train
      - avant-dernière en val
      - plus récente en test

    Avec moins d'éruptions :
      - garde au moins une éruption en test
      - garde une validation si possible
    """
    df = df.sort_values("eruption_start_utc").reset_index(drop=True)
    df["split"] = "train"

    n = len(df)

    if n >= 3:
        df.loc[df.index[-2], "split"] = "val"
        df.loc[df.index[-1], "split"] = "test"
    elif n == 2:
        df.loc[df.index[-1], "split"] = "test"
    elif n == 1:
        df.loc[df.index[0], "split"] = "test"

    return df


# ============================================================
# VALIDATION DU PLAN
# ============================================================

def validate_plan(df: pd.DataFrame, max_duration_days: float):
    errors = []
    warnings = []

    if df.empty:
        errors.append("Aucune éruption sélectionnée.")

    required = [
        "eruption_id",
        "eruption_start_utc",
        "eruption_end_utc",
        "duration_days",
        "extract_start_utc",
        "extract_end_utc",
        "extract_duration_days",
        "network",
        "stations",
        "channels",
        "s3_csv_path",
        "split",
    ]

    for col in required:
        if col not in df.columns:
            errors.append(f"Colonne manquante : {col}")

    if "eruption_id" in df.columns:
        duplicated = int(df["eruption_id"].duplicated().sum())
        if duplicated > 0:
            errors.append(f"eruption_id dupliqué : {duplicated}")

    if {"eruption_start_utc", "eruption_end_utc"}.issubset(df.columns):
        bad = int((df["eruption_end_utc"] <= df["eruption_start_utc"]).sum())
        if bad > 0:
            errors.append(
                f"eruption_end_utc <= eruption_start_utc pour {bad} ligne(s)."
            )

    if {"extract_start_utc", "extract_end_utc"}.issubset(df.columns):
        bad = int((df["extract_end_utc"] <= df["extract_start_utc"]).sum())
        if bad > 0:
            errors.append(
                f"extract_end_utc <= extract_start_utc pour {bad} ligne(s)."
            )

    if "duration_days" in df.columns:
        too_long = df[df["duration_days"] > max_duration_days]
        if not too_long.empty:
            errors.append(
                "Certaines éruptions dépassent la durée maximale autorisée : "
                + ", ".join(too_long["eruption_id"].tolist())
            )

    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()

        if "train" not in split_counts:
            warnings.append("Aucune éruption en train.")
        if "val" not in split_counts:
            warnings.append("Aucune éruption en validation.")
        if "test" not in split_counts:
            warnings.append("Aucune éruption en test.")

    return errors, warnings


# ============================================================
# MAIN
# ============================================================

def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lecture du fichier Excel : {input_path}")

    df_raw = pd.read_excel(input_path)
    df_raw.columns = [normalize_colname(c) for c in df_raw.columns]

    missing = set(EXPECTED_COLUMNS) - set(df_raw.columns)
    if missing:
        raise ValueError(
            "Colonnes attendues absentes du fichier Excel : "
            f"{sorted(missing)}\n"
            f"Colonnes détectées : {list(df_raw.columns)}"
        )

    df = df_raw.copy()

    for col in ["Type", "Localisation", "Nom"]:
        df[col] = df[col].apply(clean_text)

    # Dates/heures UTC normalisées.
    df["eruption_start_utc"] = [
        combine_date_time(d, t)
        for d, t in zip(df["Date début"], df["Heure début (UTC)"])
    ]

    df["eruption_end_utc"] = [
        combine_date_time(d, t)
        for d, t in zip(df["Date fin"], df["Heure fin (UTC)"])
    ]

    # Lignes valides uniquement.
    df = df.dropna(subset=["eruption_start_utc", "eruption_end_utc"]).copy()
    df = df[df["eruption_end_utc"] > df["eruption_start_utc"]].copy()

    # Filtrage année.
    df = df[df["eruption_start_utc"].dt.year >= args.min_year].copy()

    if args.max_year is not None:
        df = df[df["eruption_start_utc"].dt.year <= args.max_year].copy()

    # Durée réelle recalculée.
    df["duration_days"] = (
        df["eruption_end_utc"] - df["eruption_start_utc"]
    ).dt.total_seconds() / 86400.0

    # Filtrage durée max : demandé explicitement.
    df = df[df["duration_days"] <= args.max_duration_days].copy()

    # Sélection des N éruptions les plus récentes respectant la contrainte.
    df = df.sort_values("eruption_start_utc").tail(args.n_eruptions).copy()
    df = df.sort_values("eruption_start_utc").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Aucune éruption retenue après filtrage. "
            "Vérifier min_year, max_year et max_duration_days."
        )

    # Identifiants stables.
    df["eruption_id"] = [
        make_eruption_id(start, name, location)
        for start, name, location in zip(
            df["eruption_start_utc"],
            df["Nom"],
            df["Localisation"],
        )
    ]

    # Fenêtres d'extraction :
    # - 48h avant le début
    # - jusqu'à 6h après la fin
    df["pre_hours_requested"] = float(args.pre_hours)
    df["post_hours_requested"] = float(args.post_hours)

    df["extract_start_utc"] = df["eruption_start_utc"] - pd.to_timedelta(
        args.pre_hours,
        unit="h",
    )

    df["extract_end_utc"] = df["eruption_end_utc"] + pd.to_timedelta(
        args.post_hours,
        unit="h",
    )

    df["extract_duration_days"] = (
        df["extract_end_utc"] - df["extract_start_utc"]
    ).dt.total_seconds() / 86400.0

    # Paramètres fixes d'extraction.
    df["network"] = args.network
    df["stations"] = args.stations
    df["channels"] = args.channels

    # Chemins S3 individuels par éruption.
    df["s3_csv_path"] = df["eruption_id"].apply(
        lambda eid: (
            f"s3://{args.s3_bucket}/volcano/processed/aggregated_csv/"
            f"{eid}/{eid}_filtered_1_16Hz_aggregated_1min_with_fi.csv"
        )
    )

    df["s3_quality_prefix"] = df["eruption_id"].apply(
        lambda eid: (
            f"s3://{args.s3_bucket}/volcano/quality/extraction_reports/"
            f"{eid}/"
        )
    )

    # Split par éruption.
    df = assign_splits(df)

    # Format final.
    output_cols = [
        "eruption_id",
        "Type",
        "eruption_start_utc",
        "eruption_end_utc",
        "duration_days",
        "extract_start_utc",
        "extract_end_utc",
        "extract_duration_days",
        "pre_hours_requested",
        "post_hours_requested",
        "Localisation",
        "Nom",
        "network",
        "stations",
        "channels",
        "split",
        "s3_csv_path",
        "s3_quality_prefix",
    ]

    plan = df[output_cols].copy()

    plan = plan.rename(
        columns={
            "Type": "eruption_type",
            "Localisation": "location",
            "Nom": "name",
        }
    )

    # Convertit les timestamps en ISO Z pour éviter les ambiguïtés.
    datetime_cols = [
        "eruption_start_utc",
        "eruption_end_utc",
        "extract_start_utc",
        "extract_end_utc",
    ]

    for col in datetime_cols:
        plan[col] = plan[col].apply(to_iso_z)

    errors, warnings = validate_plan(
        plan.assign(
            eruption_start_utc=pd.to_datetime(plan["eruption_start_utc"], utc=True),
            eruption_end_utc=pd.to_datetime(plan["eruption_end_utc"], utc=True),
            extract_start_utc=pd.to_datetime(plan["extract_start_utc"], utc=True),
            extract_end_utc=pd.to_datetime(plan["extract_end_utc"], utc=True),
        ),
        max_duration_days=args.max_duration_days,
    )

    print("\nValidation du plan d'extraction")

    if warnings:
        print("WARNINGS :")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("ERRORS :")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError("Plan d'extraction invalide.")

    # Export 1 : plan complet pour extraction.
    extraction_plan_path = output_dir / "eruptions_extraction_plan.csv"
    plan.to_csv(extraction_plan_path, index=False)

    # Export 2 : metadata consommée par le preprocessing.
    metadata_cols = [
        "eruption_id",
        "s3_csv_path",
        "eruption_start_utc",
        "eruption_end_utc",
        "eruption_type",
        "location",
        "name",
        "split",
        "pre_hours_requested",
        "post_hours_requested",
    ]

    metadata_path = output_dir / "eruptions_metadata.csv"
    plan[metadata_cols].to_csv(metadata_path, index=False)

    # Export 3 : commandes d'extraction prêtes à lancer.
    commands_path = output_dir / "extraction_commands.ps1"

    with open(commands_path, "w", encoding="utf-8") as f:
        for _, row in plan.iterrows():
            cmd = (
                "python scripts/extract_filter_aggregate_pf.py "
                f"--eruption-id {row['eruption_id']} "
                f"--network {row['network']} "
                f"--stations {row['stations']} "
                f"--channels {row['channels']} "
                f"--starttime {row['extract_start_utc']} "
                f"--endtime {row['extract_end_utc']} "
                "--upload-s3 "
                f"--s3-bucket {args.s3_bucket}"
            )
            f.write(cmd + "\n")

    print("\nFichiers créés :")
    print(f"  - {extraction_plan_path}")
    print(f"  - {metadata_path}")
    print(f"  - {commands_path}")

    print("\nRésumé des éruptions sélectionnées :")
    summary_cols = [
        "eruption_id",
        "eruption_start_utc",
        "eruption_end_utc",
        "duration_days",
        "extract_start_utc",
        "extract_end_utc",
        "extract_duration_days",
        "split",
    ]

    print(plan[summary_cols].to_string(index=False))

    print("\nParamètres d'extraction retenus :")
    print(f"  network  = {args.network}")
    print(f"  stations = {args.stations}")
    print(f"  channels = {args.channels}")
    print(f"  pre_hours = {args.pre_hours}")
    print(f"  post_hours = {args.post_hours}")
    print(f"  max_duration_days = {args.max_duration_days}")
    print(f"  s3_bucket = {args.s3_bucket}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers Liste_Eruptions_Date_Heure.xlsx",
    )

    parser.add_argument(
        "--output-dir",
        default="data/metadata",
        help="Dossier de sortie des fichiers metadata.",
    )

    parser.add_argument(
        "--n-eruptions",
        type=int,
        default=DEFAULT_N_ERUPTIONS,
        help="Nombre d'éruptions récentes à retenir après filtrage.",
    )

    parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help="Année minimale retenue.",
    )

    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Année maximale retenue. Optionnel.",
    )

    parser.add_argument(
        "--max-duration-days",
        type=float,
        default=DEFAULT_MAX_DURATION_DAYS,
        help="Durée maximale d'éruption retenue, en jours.",
    )

    parser.add_argument(
        "--pre-hours",
        type=float,
        default=DEFAULT_PRE_HOURS,
        help="Nombre d'heures à extraire avant le début de l'éruption.",
    )

    parser.add_argument(
        "--post-hours",
        type=float,
        default=DEFAULT_POST_HOURS,
        help="Nombre d'heures à extraire après la fin de l'éruption.",
    )

    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK,
    )

    parser.add_argument(
        "--stations",
        default=DEFAULT_STATIONS,
    )

    parser.add_argument(
        "--channels",
        default=DEFAULT_CHANNELS,
    )

    parser.add_argument(
        "--s3-bucket",
        default=DEFAULT_S3_BUCKET,
    )

    args = parser.parse_args()
    main(args)