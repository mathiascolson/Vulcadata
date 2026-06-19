import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import boto3
import botocore
import pandas as pd
import streamlit as st

from dashboard_config import (
    CHANNELS,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_MIN_ALERT_CLASS,
    FDSN_STATION_URL,
    HISTORY_CACHE_SECONDS,
    HISTORY_PREFIX,
    LATEST_PREDICTION_KEY,
    N_CLASSES,
    NETWORK_CODE,
    PREDICTION_CACHE_SECONDS,
    S3_BUCKET,
    STATIC_STATIONS,
    STATIONS,
    STATIONS_CACHE_SECONDS,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_nested(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = data

    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def get_first_nested(data: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for path in paths:
        value = get_nested(data, path, default=None)
        if value is not None:
            return value

    return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False

    return default


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    text = str(value).replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def normalize_probabilities(value: Any) -> Optional[List[float]]:
    if isinstance(value, list) and len(value) == N_CLASSES:
        return [float(item) for item in value]

    if isinstance(value, dict):
        probabilities: List[float] = []

        for class_index in range(N_CLASSES):
            candidate_keys = [
                str(class_index),
                f"class_{class_index}",
                f"classe_{class_index}",
                f"probability_class_{class_index}",
                f"p_class_{class_index}",
            ]

            found = False
            for key in candidate_keys:
                if key in value:
                    probabilities.append(float(value[key]))
                    found = True
                    break

            if not found:
                return None

        return probabilities

    return None


def extract_probabilities(payload: Dict[str, Any]) -> Optional[List[float]]:
    candidates = [
        get_nested(payload, ["prediction", "probabilities"]),
        get_nested(payload, ["classification", "probabilities_by_class"]),
        get_nested(payload, ["classification", "probabilities"]),
        get_nested(payload, ["probabilities"]),
        get_nested(payload, ["class_probabilities"]),
        get_nested(payload, ["prediction_summary", "probabilities"]),
        get_nested(payload, ["prediction_summary", "class_probabilities"]),
    ]

    for candidate in candidates:
        probabilities = normalize_probabilities(candidate)
        if probabilities is not None:
            return probabilities

    return None


def extract_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    probabilities = extract_probabilities(payload)

    predicted_class = safe_int(
        get_first_nested(
            payload,
            [
                ["prediction", "predicted_class"],
                ["classification", "predicted_class"],
                ["prediction_summary", "predicted_class"],
                ["predicted_class"],
            ],
        )
    )

    predicted_probability = safe_float(
        get_first_nested(
            payload,
            [
                ["prediction", "predicted_probability"],
                ["classification", "predicted_probability"],
                ["prediction_summary", "predicted_probability"],
                ["predicted_probability"],
            ],
        )
    )

    p_alert_24h = safe_float(
        get_first_nested(
            payload,
            [
                ["prediction", "p_alert_24h"],
                ["alert", "p_alert_24h"],
                ["prediction_summary", "p_alert_24h"],
                ["p_alert_24h"],
            ],
        )
    )

    alert_24h = safe_bool(
        get_first_nested(
            payload,
            [
                ["prediction", "alert_24h"],
                ["alert", "active"],
                ["prediction_summary", "alert_24h"],
                ["alert_24h"],
            ],
        )
    )

    created_at_utc = get_first_nested(
        payload,
        [
            ["prediction", "created_at_utc"],
            ["prediction_summary", "created_at_utc"],
            ["generated_at_utc"],
            ["created_at_utc"],
            ["timestamp_utc"],
        ],
        default=utc_now_iso(),
    )

    threshold_24h = safe_float(
        get_first_nested(
            payload,
            [
                ["alert", "threshold_24h"],
                ["prediction", "alert_threshold_24h"],
                ["prediction_summary", "alert_threshold_24h"],
                ["metadata", "model_reference", "metadata", "classification_candidate", "alert_threshold_24h"],
            ],
        ),
        default=DEFAULT_ALERT_THRESHOLD,
    )

    min_class_alert = safe_int(
        get_first_nested(
            payload,
            [
                ["alert", "min_class_alert"],
                ["metadata", "model_reference", "metadata", "classification_candidate", "min_class_alert"],
            ],
        ),
        default=DEFAULT_MIN_ALERT_CLASS,
    )

    if probabilities is not None:
        if predicted_class is None:
            predicted_class = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))

        if predicted_probability is None and predicted_class is not None:
            predicted_probability = float(probabilities[predicted_class])

        if p_alert_24h is None and min_class_alert is not None:
            p_alert_24h = float(sum(probabilities[min_class_alert:]))

        if alert_24h is None and threshold_24h is not None and p_alert_24h is not None:
            alert_24h = bool(p_alert_24h >= threshold_24h)

    eruption_id = get_first_nested(
        payload,
        [
            ["eruption_id"],
            ["prediction", "eruption_id"],
            ["prediction_summary", "eruption_id"],
            ["metadata", "eruption_id"],
        ],
    )

    return {
        "created_at_utc": created_at_utc,
        "created_at_dt": parse_datetime(created_at_utc),
        "eruption_id": eruption_id,
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
        "p_alert_24h": p_alert_24h,
        "alert_24h": alert_24h,
        "threshold_24h": threshold_24h,
        "min_class_alert": min_class_alert,
        "probabilities": probabilities,
    }


@st.cache_resource
def get_s3_client():
    return boto3.client("s3")


def read_json_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    data = json.loads(body)

    if not isinstance(data, dict):
        raise ValueError("Le fichier S3 lu n'est pas un objet JSON.")

    return data


@st.cache_data(ttl=PREDICTION_CACHE_SECONDS)
def load_latest_payload(bucket: str = S3_BUCKET, key: str = LATEST_PREDICTION_KEY) -> Dict[str, Any]:
    return read_json_from_s3(bucket, key)


@st.cache_data(ttl=HISTORY_CACHE_SECONDS)
def load_history_dataframe(bucket: str = S3_BUCKET, prefix: str = HISTORY_PREFIX, limit: int = 200) -> pd.DataFrame:
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")

    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"MaxItems": 500}):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key.lower().endswith(".json"):
                objects.append(
                    {
                        "key": key,
                        "last_modified": item.get("LastModified"),
                        "size": item.get("Size", 0),
                    }
                )

    objects = sorted(
        objects,
        key=lambda item: item["last_modified"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[:limit]

    rows = []
    for item in objects:
        try:
            payload = read_json_from_s3(bucket, item["key"])
            prediction = extract_prediction(payload)
        except (botocore.exceptions.ClientError, ValueError, json.JSONDecodeError):
            continue

        rows.append(
            {
                "created_at_utc": prediction["created_at_utc"],
                "created_at_dt": prediction["created_at_dt"],
                "eruption_id": prediction["eruption_id"],
                "predicted_class": prediction["predicted_class"],
                "predicted_probability": prediction["predicted_probability"],
                "p_alert_24h": prediction["p_alert_24h"],
                "alert_24h": prediction["alert_24h"],
                "s3_key": item["key"],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.dropna(subset=["created_at_dt"]).sort_values("created_at_dt")
    return df


def parse_fdsn_text_table(text: str) -> pd.DataFrame:
    headers = None
    rows = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            candidate = line.lstrip("#").strip()
            if "|" in candidate:
                headers = [item.strip() for item in candidate.split("|")]
            continue

        if "|" not in line:
            continue

        values = [item.strip() for item in line.split("|")]

        if headers is not None and len(values) == len(headers):
            rows.append(dict(zip(headers, values)))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def fetch_fdsn_channel_metadata() -> Tuple[pd.DataFrame, Optional[str]]:
    query = {
        "network": NETWORK_CODE,
        "station": ",".join(STATIONS),
        "channel": ",".join(CHANNELS),
        "level": "channel",
        "format": "text",
    }

    url = f"{FDSN_STATION_URL}?{urlencode(query)}"

    try:
        with urlopen(url, timeout=12) as response:
            text = response.read().decode("utf-8")

        df = parse_fdsn_text_table(text)
        if df.empty:
            return pd.DataFrame(), "Aucune métadonnée FDSN exploitable."

        required_columns = {"Network", "Station", "Channel", "Latitude", "Longitude"}
        missing_columns = required_columns.difference(set(df.columns))
        if missing_columns:
            return pd.DataFrame(), f"Colonnes FDSN manquantes : {sorted(missing_columns)}"

        df = df.copy()
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["Elevation"] = pd.to_numeric(df.get("Elevation"), errors="coerce") if "Elevation" in df.columns else None
        df = df.dropna(subset=["Latitude", "Longitude"])

        grouped_rows = []
        for station, station_df in df.groupby("Station"):
            first = station_df.iloc[0]
            unique_channels = sorted(station_df["Channel"].dropna().astype(str).unique().tolist())
            grouped_rows.append(
                {
                    "Network": first.get("Network", NETWORK_CODE),
                    "Station": station,
                    "Latitude": float(first["Latitude"]),
                    "Longitude": float(first["Longitude"]),
                    "Elevation": first.get("Elevation"),
                    "channels_used": ", ".join(unique_channels),
                    "n_channels_used": len(unique_channels),
                }
            )

        return pd.DataFrame(grouped_rows), None

    except Exception as error:
        return pd.DataFrame(), f"Erreur FDSN : {type(error).__name__}: {error}"


@st.cache_data(ttl=STATIONS_CACHE_SECONDS)
def load_station_dataframe() -> Tuple[pd.DataFrame, str, Optional[str]]:
    station_df, error = fetch_fdsn_channel_metadata()

    if not station_df.empty:
        return station_df, "fdsn_channel", None

    fallback_df = pd.DataFrame(STATIC_STATIONS)
    return fallback_df, "static_fallback", error
