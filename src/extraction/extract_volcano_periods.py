from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from obspy import read


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PERIODS_CSV = PROJECT_ROOT / "data" / "metadata" / "extraction_periods.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "extraction"
DEFAULT_BASE_URL = "https://ws.ipgp.fr/fdsnws/dataselect/1/query"

CSV_SUFFIX = "_filtered_1_16Hz_aggregated_1min_with_fi.csv"

REQUIRED_PERIOD_COLUMNS = [
    "period_id",
    "period_type",
    "period_start_utc",
    "period_end_utc",
    "network",
    "stations",
    "channels",
]

REQUIRED_OUTPUT_COLUMNS = [
    "eruption_id",
    "network",
    "station",
    "channel",
    "time_min",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_max",
    "amplitude_min",
    "amplitude_count",
    "energy_low_1_5_5",
    "energy_high_6_16",
    "frequency_index",
    "sampling_rate_source_hz",
    "filter_full_low_hz",
    "filter_full_high_hz",
    "filter_fi_low_min_hz",
    "filter_fi_low_max_hz",
    "filter_fi_high_min_hz",
    "filter_fi_high_max_hz",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_utc_datetime(value: str, field_name: str) -> pd.Timestamp:
    value = clean_text(value)

    if not value:
        raise ValueError(f"{field_name} est obligatoire.")

    ts = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(ts):
        raise ValueError(
            f"{field_name} invalide : {value}. "
            "Format attendu : YYYY-MM-DDTHH:MM:SSZ"
        )

    return pd.Timestamp(ts).tz_convert("UTC")


def format_fdsn_utc(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_datetime_for_filename(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("-", "")
        .replace("T", "_")
        .replace("Z", "Z")
        .replace("+00:00", "Z")
    )


def validate_periods(periods: pd.DataFrame) -> pd.DataFrame:
    periods = periods.copy()
    periods.columns = [str(c).strip() for c in periods.columns]

    missing = set(REQUIRED_PERIOD_COLUMNS) - set(periods.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans extraction_periods.csv : {sorted(missing)}")

    optional_columns = ["split", "eruption_start_utc", "eruption_end_utc"]
    for col in optional_columns:
        if col not in periods.columns:
            periods[col] = ""

    for col in [
        "period_id",
        "period_type",
        "period_start_utc",
        "period_end_utc",
        "eruption_start_utc",
        "eruption_end_utc",
        "split",
        "network",
        "stations",
        "channels",
    ]:
        periods[col] = periods[col].apply(clean_text)

    duplicated = periods["period_id"].duplicated().sum()
    if duplicated > 0:
        raise ValueError(f"period_id dupliqué dans extraction_periods.csv : {duplicated}")

    for idx, row in periods.iterrows():
        period_id = row["period_id"]

        if not period_id:
            raise ValueError(f"period_id vide ligne {idx + 2}")

        period_type = normalize_period_type(row["period_type"])
        periods.loc[idx, "period_type"] = period_type

        start = parse_utc_datetime(row["period_start_utc"], f"{period_id}.period_start_utc")
        end = parse_utc_datetime(row["period_end_utc"], f"{period_id}.period_end_utc")

        if end <= start:
            raise ValueError(f"{period_id} : period_end_utc doit être > period_start_utc.")

        if row["eruption_start_utc"]:
            eruption_start = parse_utc_datetime(
                row["eruption_start_utc"],
                f"{period_id}.eruption_start_utc",
            )

            if row["eruption_end_utc"]:
                eruption_end = parse_utc_datetime(
                    row["eruption_end_utc"],
                    f"{period_id}.eruption_end_utc",
                )
                if eruption_end < eruption_start:
                    raise ValueError(
                        f"{period_id} : eruption_end_utc doit être >= eruption_start_utc."
                    )

    return periods


def build_fdsn_url(
    base_url: str,
    network: str,
    stations: str,
    channels: str,
    starttime: str,
    endtime: str,
) -> str:
    params = {
        "network": network,
        "station": stations,
        "starttime": starttime,
        "endtime": endtime,
        "nodata": "404",
        "channel": channels,
    }
    return f"{base_url}?{urlencode(params)}"


def format_bytes(size_bytes: float) -> str:
    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def format_seconds(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "inconnu"

    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def download_mseed(
    url: str,
    output_mseed: Path,
    timeout: int = 600,
    chunk_size: int = 1024 * 1024,
    force: bool = False,
) -> Path:
    if output_mseed.exists() and not force:
        print(f"MiniSEED déjà présent : {output_mseed} ({format_bytes(output_mseed.stat().st_size)})")
        return output_mseed

    print(f"Téléchargement MiniSEED : {url}")
    output_mseed.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            start_time = time.time()
            last_print_time = start_time

            if total_size > 0:
                print(f"Taille annoncée par le serveur : {format_bytes(total_size)}")
            else:
                print("Taille totale inconnue : le serveur ne fournit pas Content-Length.")

            with open(output_mseed, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()

                    if now - last_print_time >= 0.5:
                        elapsed = now - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0

                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            remaining = total_size - downloaded
                            eta = remaining / speed if speed > 0 else float("inf")

                            msg = (
                                f"\rTéléchargé : {format_bytes(downloaded)} / {format_bytes(total_size)} "
                                f"({percent:6.2f} %) | "
                                f"Débit moyen : {format_bytes(speed)}/s | "
                                f"ETA : {format_seconds(eta)}"
                            )
                        else:
                            msg = (
                                f"\rTéléchargé : {format_bytes(downloaded)} | "
                                f"Débit moyen : {format_bytes(speed)}/s | "
                                f"ETA : inconnue"
                            )

                        sys.stdout.write(msg)
                        sys.stdout.flush()
                        last_print_time = now

            print()

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Erreur téléchargement MiniSEED : {exc}") from exc

    if not output_mseed.exists() or output_mseed.stat().st_size == 0:
        raise RuntimeError("Fichier MiniSEED vide après téléchargement.")

    print(f"MiniSEED sauvegardé : {output_mseed} ({format_bytes(output_mseed.stat().st_size)})")
    return output_mseed


def validate_mseed_stream(stream, expected_network: str, expected_stations: set[str] | None = None) -> dict:
    report = {
        "validated_at": utc_now_iso(),
        "status": "success",
        "errors": [],
        "warnings": [],
        "n_traces": len(stream),
        "traces": [],
    }

    if len(stream) == 0:
        report["status"] = "failed"
        report["errors"].append("Stream ObsPy vide.")

    stations_seen = set()

    for tr in stream:
        stats = tr.stats
        stations_seen.add(str(stats.station))

        trace_report = {
            "network": str(stats.network),
            "station": str(stats.station),
            "channel": str(stats.channel),
            "starttime": str(stats.starttime),
            "endtime": str(stats.endtime),
            "sampling_rate": float(stats.sampling_rate),
            "npts": int(stats.npts),
        }

        if str(stats.network) != expected_network:
            report["warnings"].append(
                f"Network inattendu : {stats.network} pour {stats.station}.{stats.channel}"
            )

        if stats.sampling_rate <= 0:
            report["errors"].append(
                f"Sampling rate invalide pour {stats.station}.{stats.channel}"
            )

        if stats.npts <= 0:
            report["errors"].append(
                f"Trace vide pour {stats.station}.{stats.channel}"
            )

        data = np.asarray(tr.data)
        if data.size == 0:
            report["errors"].append(
                f"Données vides pour {stats.station}.{stats.channel}"
            )
        elif not np.isfinite(data.astype(float)).any():
            report["errors"].append(
                f"Aucune valeur finie pour {stats.station}.{stats.channel}"
            )

        report["traces"].append(trace_report)

    if expected_stations:
        missing = expected_stations - stations_seen
        if missing:
            report["warnings"].append(
                f"Stations attendues absentes du stream : {sorted(missing)}"
            )

    if report["errors"]:
        report["status"] = "failed"

    return report


def write_json_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Rapport qualité écrit : {path}")


def preprocess_stream(stream):
    st = stream.copy()
    st.merge(method=1, fill_value="interpolate")

    for tr in st:
        tr.detrend("demean")
        tr.detrend("linear")

    return st


def filter_trace(trace, freqmin: float, freqmax: float):
    tr = trace.copy()
    tr.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        corners=4,
        zerophase=True,
    )
    return tr


def aggregate_trace_to_minute(
    eruption_id: str,
    trace_full,
    trace_low,
    trace_high,
    eps: float = 1e-12,
) -> pd.DataFrame:
    stats = trace_full.stats

    sampling_rate = float(stats.sampling_rate)

    full_data = np.asarray(trace_full.data, dtype=float)
    low_data = np.asarray(trace_low.data, dtype=float)
    high_data = np.asarray(trace_high.data, dtype=float)

    n = min(len(full_data), len(low_data), len(high_data))

    if n == 0:
        raise ValueError(f"Trace vide après filtrage : {stats.station}.{stats.channel}")

    full_data = full_data[:n]
    low_data = low_data[:n]
    high_data = high_data[:n]

    times = trace_full.times("utcdatetime")[:n]

    df = pd.DataFrame(
        {
            "time": [pd.Timestamp(str(t), tz="UTC") for t in times],
            "amplitude": full_data,
            "low_squared": low_data**2,
            "high_squared": high_data**2,
        }
    )

    df["time_min"] = df["time"].dt.floor("min")

    agg = (
        df.groupby("time_min", as_index=False)
        .agg(
            amplitude_mean=("amplitude", "mean"),
            amplitude_std=("amplitude", "std"),
            amplitude_max=("amplitude", "max"),
            amplitude_min=("amplitude", "min"),
            amplitude_count=("amplitude", "count"),
            energy_low_1_5_5=("low_squared", "mean"),
            energy_high_6_16=("high_squared", "mean"),
        )
    )

    agg["frequency_index"] = np.log10(
        (agg["energy_high_6_16"] + eps) / (agg["energy_low_1_5_5"] + eps)
    )

    agg["eruption_id"] = eruption_id
    agg["network"] = str(stats.network)
    agg["station"] = str(stats.station)
    agg["channel"] = str(stats.channel)
    agg["sampling_rate_source_hz"] = sampling_rate

    agg["filter_full_low_hz"] = 1.0
    agg["filter_full_high_hz"] = 16.0
    agg["filter_fi_low_min_hz"] = 1.0
    agg["filter_fi_low_max_hz"] = 5.5
    agg["filter_fi_high_min_hz"] = 6.0
    agg["filter_fi_high_max_hz"] = 16.0

    agg = agg[REQUIRED_OUTPUT_COLUMNS]

    return agg


def validate_aggregated_csv(df: pd.DataFrame) -> dict:
    report = {
        "validated_at": utc_now_iso(),
        "status": "success",
        "errors": [],
        "warnings": [],
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
    }

    missing_cols = set(REQUIRED_OUTPUT_COLUMNS) - set(df.columns)
    if missing_cols:
        report["errors"].append(f"Colonnes manquantes : {sorted(missing_cols)}")

    if len(df) == 0:
        report["errors"].append("CSV agrégé vide.")

    if "time_min" in df.columns:
        parsed_time = pd.to_datetime(df["time_min"], utc=True, errors="coerce")
        n_bad_time = int(parsed_time.isna().sum())
        if n_bad_time > 0:
            report["errors"].append(f"time_min invalide pour {n_bad_time} lignes.")

    for col in [
        "amplitude_mean",
        "amplitude_std",
        "amplitude_max",
        "amplitude_min",
        "amplitude_count",
        "energy_low_1_5_5",
        "energy_high_6_16",
        "frequency_index",
    ]:
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                report["warnings"].append(f"{col} contient {n_null} valeurs nulles.")

    if "amplitude_count" in df.columns:
        n_bad = int((df["amplitude_count"] <= 0).sum())
        if n_bad > 0:
            report["errors"].append(f"amplitude_count <= 0 pour {n_bad} lignes.")

    if "amplitude_std" in df.columns:
        n_bad = int((df["amplitude_std"] < 0).sum())
        if n_bad > 0:
            report["errors"].append(f"amplitude_std < 0 pour {n_bad} lignes.")

    needed_amp = {"amplitude_min", "amplitude_mean", "amplitude_max"}
    if needed_amp.issubset(df.columns):
        n_bad_order = int(
            (
                (df["amplitude_min"] > df["amplitude_mean"])
                | (df["amplitude_mean"] > df["amplitude_max"])
            ).sum()
        )
        if n_bad_order > 0:
            report["errors"].append(
                f"Ordre amplitude_min <= amplitude_mean <= amplitude_max violé pour {n_bad_order} lignes."
            )

    for col in ["energy_low_1_5_5", "energy_high_6_16"]:
        if col in df.columns:
            n_bad = int((df[col] < 0).sum())
            if n_bad > 0:
                report["errors"].append(f"{col} < 0 pour {n_bad} lignes.")

    if {"station", "channel", "time_min"}.issubset(df.columns):
        n_dupes = int(df.duplicated(["station", "channel", "time_min"]).sum())
        if n_dupes > 0:
            report["errors"].append(
                f"Doublons station/channel/time_min : {n_dupes}"
            )

    if "frequency_index" in df.columns:
        n_inf = int(np.isinf(df["frequency_index"].to_numpy(dtype=float)).sum())
        if n_inf > 0:
            report["errors"].append(f"frequency_index infini pour {n_inf} lignes.")

    if report["errors"]:
        report["status"] = "failed"

    return report


def process_period(row: pd.Series, args) -> dict:
    period_id = row["period_id"]
    network = row["network"]
    stations = row["stations"]
    channels = row["channels"]

    start = parse_utc_datetime(row["period_start_utc"], f"{period_id}.period_start_utc")
    end = parse_utc_datetime(row["period_end_utc"], f"{period_id}.period_end_utc")

    start_str = format_fdsn_utc(start)
    end_str = format_fdsn_utc(end)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_mseed"
    processed_dir = output_dir / "processed_csv"
    quality_dir = output_dir / "quality_reports"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    safe_start = safe_datetime_for_filename(start_str)
    safe_end = safe_datetime_for_filename(end_str)

    mseed_path = raw_dir / f"{period_id}_{safe_start}_{safe_end}.mseed"
    csv_path = processed_dir / f"{period_id}{CSV_SUFFIX}"

    mseed_report_path = quality_dir / f"{period_id}_mseed_validation.json"
    csv_report_path = quality_dir / f"{period_id}_csv_validation.json"

    url = build_fdsn_url(
        base_url=args.base_url,
        network=network,
        stations=stations,
        channels=channels,
        starttime=start_str,
        endtime=end_str,
    )

    download_mseed(
        url=url,
        output_mseed=mseed_path,
        timeout=args.timeout,
        force=args.force_download,
    )

    try:
        stream = read(str(mseed_path))
    except Exception as exc:
        raise RuntimeError(f"Lecture ObsPy impossible : {exc}") from exc

    expected_stations = set(stations.split(",")) if stations else None

    mseed_report = validate_mseed_stream(
        stream=stream,
        expected_network=network,
        expected_stations=expected_stations,
    )
    write_json_report(mseed_report, mseed_report_path)

    if mseed_report["status"] == "failed":
        raise RuntimeError(f"Validation MiniSEED échouée. Voir rapport : {mseed_report_path}")

    stream_clean = preprocess_stream(stream)

    rows = []

    for tr in stream_clean:
        stats = tr.stats

        nyquist = float(stats.sampling_rate) / 2.0
        if nyquist <= 16.0:
            print(
                f"WARNING : trace ignorée, Nyquist <= 16 Hz "
                f"({stats.network}.{stats.station}.{stats.channel}, "
                f"sampling_rate={stats.sampling_rate})"
            )
            continue

        print(
            f"Traitement trace : {stats.network}.{stats.station}.{stats.channel} | "
            f"sampling_rate={float(stats.sampling_rate):.2f} Hz | npts={int(stats.npts)}"
        )

        try:
            tr_full = filter_trace(tr, 1.0, 16.0)
            tr_low = filter_trace(tr, 1.0, 5.5)
            tr_high = filter_trace(tr, 6.0, 16.0)

            agg = aggregate_trace_to_minute(
                eruption_id=period_id,
                trace_full=tr_full,
                trace_low=tr_low,
                trace_high=tr_high,
                eps=args.eps,
            )
            rows.append(agg)

        except Exception as exc:
            print(
                f"WARNING : échec traitement trace "
                f"{stats.network}.{stats.station}.{stats.channel} : {exc}"
            )

    if not rows:
        raise RuntimeError("Aucune trace exploitable après filtrage.")

    df_out = pd.concat(rows, ignore_index=True)
    df_out["time_min"] = pd.to_datetime(df_out["time_min"], utc=True)
    df_out = df_out.sort_values(["station", "channel", "time_min"]).reset_index(drop=True)

    csv_report = validate_aggregated_csv(df_out)
    write_json_report(csv_report, csv_report_path)

    if csv_report["status"] == "failed":
        raise RuntimeError(f"Validation CSV échouée. Voir rapport : {csv_report_path}")

    df_out.to_csv(csv_path, index=False)

    print(f"CSV agrégé écrit : {csv_path}")
    print(f"Shape CSV : {df_out.shape}")

    return {
        "period_id": period_id,
        "period_type": row["period_type"],
        "period_start_utc": start_str,
        "period_end_utc": end_str,
        "csv_path": str(csv_path),
        "mseed_path": str(mseed_path),
        "mseed_report_path": str(mseed_report_path),
        "csv_report_path": str(csv_report_path),
        "n_rows": int(len(df_out)),
        "n_columns": int(df_out.shape[1]),
        "status": "success",
    }


def main(args) -> None:
    periods_path = Path(args.periods)

    if not periods_path.exists():
        raise FileNotFoundError(f"Fichier de périodes introuvable : {periods_path}")

    periods = read_csv_auto(periods_path)
    periods = validate_periods(periods)

    summary = {
        "created_at": utc_now_iso(),
        "periods_file": str(periods_path),
        "output_dir": str(Path(args.output_dir)),
        "n_periods": int(len(periods)),
        "results": [],
        "failures": [],
    }

    print(f"Nombre de périodes à extraire : {len(periods)}")

    for idx, row in periods.iterrows():
        period_id = row["period_id"]

        print()
        print("=" * 100)
        print(f"[{idx + 1}/{len(periods)}] Extraction période : {period_id}")

        try:
            result = process_period(row, args)
            summary["results"].append(result)
        except Exception as exc:
            failure = {
                "period_id": period_id,
                "status": "failed",
                "error": repr(exc),
            }
            summary["failures"].append(failure)
            print(f"ERREUR sur {period_id} : {exc}")

    summary_path = Path(args.output_dir) / "quality_reports" / "extraction_summary.json"
    write_json_report(summary, summary_path)

    print()
    print("=" * 100)
    print("Extraction terminée.")
    print(f"Succès : {len(summary['results'])}")
    print(f"Échecs : {len(summary['failures'])}")

    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction MiniSEED IPGP/FDSN, filtrage 1-16 Hz et agrégation 1 minute pour Vulcadata."
    )

    parser.add_argument(
        "--periods",
        type=str,
        default=str(DEFAULT_PERIODS_CSV),
        help="CSV des périodes à extraire.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Répertoire de sortie extraction.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="URL FDSN dataselect.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout HTTP en secondes.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Epsilon pour le calcul du frequency_index.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Retélécharger les MiniSEED même s'ils existent déjà.",
    )

    main(parser.parse_args())
