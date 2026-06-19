from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_CSV_DIR = PROJECT_ROOT / "data" / "extraction" / "processed_csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "volcano_metadata_with_quiet.csv"

CSV_SUFFIX = "_filtered_1_16Hz_aggregated_1min_with_fi.csv"


# ============================================================
# DATES DE DÉBUT D'ÉRUPTION
# ============================================================
# Ces dates sont utilisées pour fabriquer les classes d'horizon.
# Elles doivent donc correspondre au début réel du trémor / de l'éruption.
# Les périodes quiet_* n'ont pas besoin de date.
# ============================================================

ERUPTION_START_UTC = {
    "eruption_2015_10_29_flank_s": "2015-10-29T18:18:00Z",
    "eruption_2016_05_26_flank_se": "2016-05-26T04:05:00Z",
    "eruption_2017_05_17_flank_se": "2017-05-17T16:20:00Z",
    "eruption_2018_04_03_flank_n_ne": "2018-04-03T07:00:00Z",
    "eruption_2018_07_12_flank_n": "2018-07-13T00:30:00Z",
    "eruption_2018_07_13_summit": "2018-07-13T00:30:00Z",
    "eruption_2018_09_15_flank_s": "2018-09-15T00:25:00Z",
    "eruption_2019_02_18_flank_e": "2019-02-18T05:48:00Z",
    "eruption_2019_06_11_flank_e": "2019-06-11T02:35:00Z",
    "eruption_2019_07_29_flank_n": "2019-07-29T01:13:00Z",
    "eruption_2019_08_11_piton_louise_et_henri_cornu": "2019-08-11T12:20:00Z",
    "eruption_2019_08_15": "2019-08-15T04:30:00Z",
    "eruption_2019_10_25_piton_freri": "2019-10-25T10:40:00Z",
}


# ============================================================
# SPLITS FORCÉS
# ============================================================
# Le split est déterministe, sans tirage aléatoire.
# Les IDs absents de ce dictionnaire sont répartis par fallback stable.
# ============================================================

FORCED_SPLITS = {
    # Éruptions
    "eruption_2015_10_29_flank_s": "train",
    "eruption_2016_05_26_flank_se": "train",
    "eruption_2017_05_17_flank_se": "train",
    "eruption_2018_04_03_flank_n_ne": "train",
    "eruption_2018_07_12_flank_n": "val",
    "eruption_2018_07_13_summit": "val",
    "eruption_2018_09_15_flank_s": "test",
    "eruption_2019_02_18_flank_e": "train",
    "eruption_2019_06_11_flank_e": "train",
    "eruption_2019_07_29_flank_n": "val",
    "eruption_2019_08_11_piton_louise_et_henri_cornu": "test",
    "eruption_2019_08_15": "test",
    "eruption_2019_10_25_piton_freri": "test",

    # Périodes calmes préparées par extract_quiet_periods.ps1
    "quiet_2016_02_10": "train",
    "quiet_2016_11_23": "train",
    "quiet_2017_12_13": "val",
    "quiet_2018_12_24": "test",
}


def get_eruption_id_from_filename(path: Path) -> str:
    """
    Extrait l'ID depuis un fichier du type :
    eruption_2018_04_03_flank_n_ne_filtered_1_16Hz_aggregated_1min_with_fi.csv
    quiet_2016_02_10_filtered_1_16Hz_aggregated_1min_with_fi.csv
    """
    name = path.name

    if not name.endswith(CSV_SUFFIX):
        raise ValueError(f"Nom de fichier non conforme : {name}")

    return name[: -len(CSV_SUFFIX)]


def infer_period_type(eruption_id: str) -> str:
    if eruption_id.startswith("quiet_"):
        return "quiet"
    return "eruption"


def infer_split(eruption_id: str, index_by_type: int) -> str:
    """
    Déduit un split si l'ID n'est pas explicitement listé dans FORCED_SPLITS.

    Fallback stable :
    - 70 % train
    - 20 % val
    - 10 % test
    """
    if eruption_id in FORCED_SPLITS:
        return FORCED_SPLITS[eruption_id]

    mod = index_by_type % 10

    if mod <= 6:
        return "train"
    if mod <= 8:
        return "val"
    return "test"


def build_metadata() -> pd.DataFrame:
    if not PROCESSED_CSV_DIR.exists():
        raise FileNotFoundError(
            f"Dossier processed_csv introuvable : {PROCESSED_CSV_DIR}"
        )

    csv_files = sorted(PROCESSED_CSV_DIR.glob(f"*{CSV_SUFFIX}"))

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun CSV agrégé trouvé dans : {PROCESSED_CSV_DIR}\n"
            f"Pattern attendu : *{CSV_SUFFIX}"
        )

    rows = []
    missing_eruption_start_dates = []
    type_counters = {"eruption": 0, "quiet": 0}

    for csv_path in csv_files:
        eruption_id = get_eruption_id_from_filename(csv_path)
        period_type = infer_period_type(eruption_id)

        index_by_type = type_counters[period_type]
        type_counters[period_type] += 1

        split = infer_split(
            eruption_id=eruption_id,
            index_by_type=index_by_type,
        )

        if period_type == "eruption":
            eruption_start_utc = ERUPTION_START_UTC.get(eruption_id, "")
            if not eruption_start_utc:
                missing_eruption_start_dates.append(eruption_id)
        else:
            eruption_start_utc = ""

        rows.append(
            {
                "eruption_id": eruption_id,
                "period_type": period_type,
                "eruption_start_utc": eruption_start_utc,
                "split": split,
                "csv_path": str(csv_path),
            }
        )

    if missing_eruption_start_dates:
        print("\nDates eruption_start_utc manquantes pour les éruptions suivantes :\n")
        for eruption_id in missing_eruption_start_dates:
            print(f'    "{eruption_id}": "YYYY-MM-DDTHH:MM:SSZ",')

        raise ValueError(
            "\nCompléter ERUPTION_START_UTC avec les dates exactes ci-dessus.\n"
            "Ne pas utiliser une heure approximative : elle sert à construire les labels."
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["period_type", "split", "eruption_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return df


def validate_metadata(df: pd.DataFrame) -> None:
    required_columns = {
        "eruption_id",
        "period_type",
        "eruption_start_utc",
        "split",
        "csv_path",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    allowed_period_types = {"eruption", "quiet"}
    bad_period_types = set(df["period_type"].unique()) - allowed_period_types
    if bad_period_types:
        raise ValueError(f"period_type invalides : {bad_period_types}")

    allowed_splits = {"train", "val", "test"}
    bad_splits = set(df["split"].unique()) - allowed_splits
    if bad_splits:
        raise ValueError(f"split invalides : {bad_splits}")

    missing_files = []
    for path in df["csv_path"]:
        if not Path(path).exists():
            missing_files.append(path)

    if missing_files:
        print("\nFichiers CSV introuvables :")
        for path in missing_files:
            print(f"  - {path}")
        raise FileNotFoundError("Certains CSV référencés sont absents.")

    eruption_rows = df[df["period_type"] == "eruption"]
    missing_dates = eruption_rows[
        eruption_rows["eruption_start_utc"].isna()
        | (eruption_rows["eruption_start_utc"].astype(str).str.strip() == "")
    ]

    if not missing_dates.empty:
        print("\nÉruptions sans eruption_start_utc :")
        print(missing_dates[["eruption_id", "csv_path"]])
        raise ValueError(
            "Toutes les périodes éruptives doivent avoir eruption_start_utc."
        )

    for split in ["train", "val", "test"]:
        if split not in set(df["split"]):
            raise ValueError(f"Aucune ligne pour le split : {split}")

    if "quiet" not in set(df["period_type"]):
        raise ValueError(
            "Aucune période calme détectée. "
            "Vérifier que les fichiers quiet_* sont bien dans processed_csv."
        )


def print_summary(df: pd.DataFrame) -> None:
    print("\nRésumé metadata :")
    print(df.groupby(["split", "period_type"]).size().unstack(fill_value=0))

    print("\nFichiers intégrés :")
    for _, row in df.iterrows():
        print(
            f"  - {row['split']:5s} | "
            f"{row['period_type']:8s} | "
            f"{row['eruption_id']}"
        )


def main():
    df = build_metadata()
    validate_metadata(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nMetadata écrit : {OUTPUT_PATH}")
    print_summary(df)


if __name__ == "__main__":
    main()
