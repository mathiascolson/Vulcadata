# scripts/extract_filter_aggregate_pf.py

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

# Charge le .env situé à la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

import boto3
import numpy as np
import pandas as pd
import requests
from obspy import read


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


def parse_s3_uri(s3_uri: str):
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"URI S3 invalide : {s3_uri}")
    path = s3_uri.replace("s3://", "", 1)
    bucket, key = path.split("/", 1)
    return bucket, key


def upload_file_to_s3(local_path: Path, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"Upload S3 OK : {local_path} → {s3_uri}")


def build_fdsn_url(base_url, network, stations, starttime, endtime, channels=None):
    url = (
        f"{base_url}"
        f"?network={network}"
        f"&station={stations}"
        f"&starttime={starttime}"
        f"&endtime={endtime}"
        f"&nodata=404"
    )

    if channels:
        url += f"&channel={channels}"

    return url


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


def download_mseed(url: str, output_mseed: Path, timeout: int = 600, chunk_size: int = 1024 * 1024):
    """
    Téléchargement MiniSEED avec suivi :
    - volume téléchargé
    - taille totale si disponible
    - pourcentage si Content-Length disponible
    - débit moyen
    - ETA si Content-Length disponible

    Remarque :
    L'ETA n'est disponible que si le serveur renvoie un header Content-Length.
    """
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

                    # Évite d'afficher trop souvent.
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

            print()  # retour ligne après la barre de progression

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Erreur téléchargement MiniSEED : {exc}") from exc

    if not output_mseed.exists() or output_mseed.stat().st_size == 0:
        raise RuntimeError("Fichier MiniSEED vide après téléchargement.")

    size_mb = output_mseed.stat().st_size / (1024 * 1024)
    print(f"MiniSEED sauvegardé : {output_mseed} ({size_mb:.2f} MB)")

    return output_mseed


def validate_mseed_stream(stream, expected_network: str, expected_stations: set | None = None):
    """
    Validation amont, avant filtrage.
    Ce n'est pas Great Expectations : ici on valide des objets ObsPy.
    """
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


def write_json_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Rapport qualité écrit : {path}")


def preprocess_stream(stream):
    """
    Nettoyage de base avant filtrage.
    """
    st = stream.copy()

    # Fusionne les segments d'une même trace.
    # Interpolation si petits trous.
    st.merge(method=1, fill_value="interpolate")

    for tr in st:
        tr.detrend("demean")
        tr.detrend("linear")

    return st


def filter_trace(trace, freqmin, freqmax):
    """
    Filtrage passe-bande.
    """
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
):
    """
    Produit un DataFrame agrégé par minute pour une trace :
    - stats amplitude sur le signal filtré 1–16 Hz
    - énergie basse 1–5.5 Hz
    - énergie haute 6–16 Hz
    - Frequency Index = log10((E_high + eps) / (E_low + eps))
    """
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


def validate_aggregated_csv(df: pd.DataFrame):
    """
    Validation type Great Expectations simplifiée pour l'étape extraction.
    Cette validation peut être reprise plus tard dans une vraie suite GE.
    """
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


def main(args):
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_mseed"
    processed_dir = output_dir / "processed_csv"
    quality_dir = output_dir / "quality_reports"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    expected_stations = set(args.stations.split(",")) if args.stations else None

    url = build_fdsn_url(
        base_url=args.base_url,
        network=args.network,
        stations=args.stations,
        starttime=args.starttime,
        endtime=args.endtime,
        channels=args.channels,
    )

    safe_start = args.starttime.replace(":", "").replace("-", "").replace("T", "_")
    safe_end = args.endtime.replace(":", "").replace("-", "").replace("T", "_")

    mseed_path = raw_dir / f"{args.eruption_id}_{safe_start}_{safe_end}.mseed"
    csv_path = processed_dir / f"{args.eruption_id}_filtered_1_16Hz_aggregated_1min_with_fi.csv"

    mseed_report_path = quality_dir / f"{args.eruption_id}_mseed_validation.json"
    csv_report_path = quality_dir / f"{args.eruption_id}_csv_validation.json"

    download_mseed(url=url, output_mseed=mseed_path, timeout=args.timeout)

    try:
        stream = read(str(mseed_path))
    except Exception as exc:
        raise RuntimeError(f"Lecture ObsPy impossible : {exc}")

    mseed_report = validate_mseed_stream(
        stream=stream,
        expected_network=args.network,
        expected_stations=expected_stations,
    )
    write_json_report(mseed_report, mseed_report_path)

    if mseed_report["status"] == "failed":
        raise RuntimeError(
            f"Validation MiniSEED échouée. Voir rapport : {mseed_report_path}"
        )

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
                eruption_id=args.eruption_id,
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
        raise RuntimeError(
            f"Validation CSV échouée. Voir rapport : {csv_report_path}"
        )

    df_out.to_csv(csv_path, index=False)
    print(f"CSV agrégé écrit : {csv_path}")
    print(df_out.head())
    print(df_out.shape)

    if args.upload_s3:
        if not args.s3_bucket:
            raise ValueError("--s3-bucket obligatoire si --upload-s3 est activé.")

        csv_s3_uri = (
            f"s3://{args.s3_bucket}/volcano/processed/aggregated_csv/"
            f"{args.eruption_id}/{csv_path.name}"
        )
        mseed_report_s3_uri = (
            f"s3://{args.s3_bucket}/volcano/quality/extraction_reports/"
            f"{args.eruption_id}/{mseed_report_path.name}"
        )
        csv_report_s3_uri = (
            f"s3://{args.s3_bucket}/volcano/quality/extraction_reports/"
            f"{args.eruption_id}/{csv_report_path.name}"
        )

        upload_file_to_s3(csv_path, csv_s3_uri)
        upload_file_to_s3(mseed_report_path, mseed_report_s3_uri)
        upload_file_to_s3(csv_report_path, csv_report_s3_uri)

        if args.upload_mseed:
            mseed_s3_uri = (
                f"s3://{args.s3_bucket}/volcano/raw/mseed/"
                f"{args.eruption_id}/{mseed_path.name}"
            )
            upload_file_to_s3(mseed_path, mseed_s3_uri)

    if not args.keep_mseed:
        try:
            os.remove(mseed_path)
            print(f"MiniSEED local supprimé : {mseed_path}")
        except OSError:
            pass

    print("Extraction / filtrage / agrégation terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eruption-id", required=True)
    parser.add_argument("--network", default="PF")
    parser.add_argument("--stations", required=True)
    parser.add_argument("--channels", default=None)

    parser.add_argument("--starttime", required=True)
    parser.add_argument("--endtime", required=True)

    parser.add_argument(
        "--base-url",
        default="https://ws.ipgp.fr/fdsnws/dataselect/1/query",
    )

    parser.add_argument("--output-dir", default="data/extraction")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--eps", type=float, default=1e-12)

    parser.add_argument("--upload-s3", action="store_true")
    parser.add_argument("--upload-mseed", action="store_true")
    parser.add_argument("--keep-mseed", action="store_true")
    parser.add_argument("--s3-bucket", default="vulcadata")

    args = parser.parse_args()
    main(args)