import io
import json
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_config import (
    APP_TITLE,
    CHANNELS,
    CLASS_COLORS,
    CLASS_DESCRIPTIONS,
    CLASS_LABELS,
    CLASS_ZONES,
    HISTORY_CACHE_SECONDS,
    HISTORY_PREFIX,
    LATEST_PREDICTION_KEY,
    NETWORK_CODE,
    PREDICTION_CACHE_SECONDS,
    S3_BUCKET,
    STATIONS,
    STATIONS_CACHE_SECONDS,
)


def percent(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value * 100:.{digits}f}%"


def class_label(class_id: Optional[int]) -> str:
    if class_id is None:
        return "Classe inconnue"
    return CLASS_LABELS.get(class_id, f"Classe {class_id}")


def alert_visuals(p_alert_24h: Optional[float], threshold: float) -> Dict[str, str]:
    value = 0.0 if p_alert_24h is None else float(p_alert_24h)

    if value < threshold:
        return {
            "label": "Pas d'alerte",
            "background": "rgba(22, 101, 52, 0.32)",
            "border": "#15803d",
            "text": "#dcfce7",
            "subtitle": "Niveau opérationnel normal.",
        }

    if value < 0.60:
        return {
            "label": "Vigilance",
            "background": "rgba(146, 64, 14, 0.35)",
            "border": "#f59e0b",
            "text": "#fffbeb",
            "subtitle": "Seuil d'alerte 24h franchi. Surveillance renforcée.",
        }

    if value < 0.80:
        return {
            "label": "Alerte élevée",
            "background": "rgba(154, 52, 18, 0.38)",
            "border": "#f97316",
            "text": "#fff7ed",
            "subtitle": "Signal élevé. Priorité opérationnelle forte.",
        }

    return {
        "label": "Alerte critique",
        "background": "rgba(127, 29, 29, 0.42)",
        "border": "#dc2626",
        "text": "#fef2f2",
        "subtitle": "Signal critique. Priorité opérationnelle maximale.",
    }


def inject_css() -> None:
    st.markdown(
        """
<style>
.alert-card {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border: 1px solid;
    margin: 1rem 0 1.2rem 0;
}
.alert-card-title {
    font-size: 0.95rem;
    opacity: 0.9;
    margin-bottom: 0.25rem;
}
.alert-card-status {
    font-size: 2.1rem;
    line-height: 1.15;
    font-weight: 750;
    margin-bottom: 0.35rem;
}
.alert-card-subtitle {
    font-size: 1.02rem;
    opacity: 0.96;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.set_page_config(page_title="Vulcadata Dashboard", layout="wide")
    inject_css()
    st.title(APP_TITLE)
    st.caption("Surveillance opérationnelle des prédictions d'alerte volcanique - Piton de la Fournaise")


def render_sidebar() -> None:
    st.sidebar.title("Source des prédictions")
    st.sidebar.write("Mode : `s3`")
    st.sidebar.write(f"Bucket : `{S3_BUCKET}`")
    st.sidebar.write(f"Dernière prédiction : `{LATEST_PREDICTION_KEY}`")
    st.sidebar.write(f"Historique : `{HISTORY_PREFIX}`")

    st.sidebar.title("Stations")
    st.sidebar.write(f"Réseau : `{NETWORK_CODE}`")
    st.sidebar.write(f"Stations : `{', '.join(STATIONS)}`")
    st.sidebar.write(f"Canaux : `{', '.join(CHANNELS)}`")

    st.sidebar.title("Lecture")
    st.sidebar.write(f"Cache prédiction : `{PREDICTION_CACHE_SECONDS} secondes`")
    st.sidebar.write(f"Cache historique : `{HISTORY_CACHE_SECONDS} secondes`")
    st.sidebar.write(f"Cache stations : `{STATIONS_CACHE_SECONDS // 3600} heure`")


def render_alert_card(prediction: Dict[str, Any]) -> None:
    visuals = alert_visuals(prediction["p_alert_24h"], prediction["threshold_24h"])

    st.markdown(
        f"""
<div class="alert-card" style="background:{visuals['background']}; border-color:{visuals['border']}; color:{visuals['text']};">
    <div class="alert-card-title">Statut opérationnel 24h</div>
    <div class="alert-card-status">{visuals['label']}</div>
    <div class="alert-card-subtitle">{visuals['subtitle']} — p_alert_24h = {percent(prediction['p_alert_24h'])} / seuil = {percent(prediction['threshold_24h'], digits=0)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(prediction: Dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)

    predicted_class = prediction["predicted_class"]
    label = class_label(predicted_class)

    col1.metric("Classe prédite", str(predicted_class) if predicted_class is not None else "n/d", label)
    col2.metric("Confiance classe prédite", percent(prediction["predicted_probability"]))
    col3.metric("Probabilité alerte 24h", percent(prediction["p_alert_24h"]))
    col4.metric("Seuil opérationnel", percent(prediction["threshold_24h"], digits=0))


def build_gauge(value: Optional[float], title: str, threshold: Optional[float] = None) -> go.Figure:
    value_percent = 0.0 if value is None else float(value) * 100.0

    gauge = {
        "axis": {"range": [0, 100], "ticksuffix": "%"},
        "bar": {"color": "#7cc5ff"},
        "steps": [
            {"range": [0, 35], "color": "rgba(34, 197, 94, 0.34)"},
            {"range": [35, 60], "color": "rgba(245, 158, 11, 0.34)"},
            {"range": [60, 80], "color": "rgba(249, 115, 22, 0.34)"},
            {"range": [80, 100], "color": "rgba(220, 38, 38, 0.34)"},
        ],
    }

    if threshold is not None:
        gauge["threshold"] = {
            "line": {"color": "#ffffff", "width": 4},
            "thickness": 0.75,
            "value": float(threshold) * 100.0,
        }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_percent,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": title},
            gauge=gauge,
        )
    )
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def render_gauges(prediction: Dict[str, Any]) -> None:
    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            build_gauge(
                prediction["p_alert_24h"],
                "Tachymètre - probabilité d'alerte 24h",
                prediction["threshold_24h"],
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            build_gauge(
                prediction["predicted_probability"],
                f"Tachymètre - confiance classe {prediction['predicted_class']}",
            ),
            use_container_width=True,
        )


def class_legend_dataframe() -> pd.DataFrame:
    rows = []
    for class_id in range(6):
        rows.append(
            {
                "Classe": class_id,
                "Libellé": CLASS_LABELS[class_id],
                "Interprétation": CLASS_DESCRIPTIONS[class_id],
                "Zone": CLASS_ZONES[class_id],
            }
        )
    return pd.DataFrame(rows)


def render_class_legend() -> None:
    st.subheader("Légende des classes de prédiction")
    st.dataframe(class_legend_dataframe(), use_container_width=True, hide_index=True)


def probability_dataframe(prediction: Dict[str, Any]) -> pd.DataFrame:
    probabilities = prediction.get("probabilities")

    if probabilities is None:
        return pd.DataFrame()

    rows = []
    for class_id, probability in enumerate(probabilities):
        rows.append(
            {
                "Classe": f"Classe {class_id}",
                "Libellé": CLASS_LABELS.get(class_id, f"Classe {class_id}"),
                "Probabilité": float(probability),
                "Probabilité (%)": float(probability) * 100.0,
                "Zone": CLASS_ZONES.get(class_id, "Inconnue"),
                "Couleur": CLASS_COLORS.get(class_id, "#7cc5ff"),
            }
        )

    return pd.DataFrame(rows)


def render_probability_chart(prediction: Dict[str, Any]) -> None:
    st.subheader("Distribution des probabilités par classe")

    df = probability_dataframe(prediction)
    if df.empty:
        st.info("Aucune probabilité détaillée disponible.")
        return

    fig = px.bar(
        df,
        x="Classe",
        y="Probabilité (%)",
        text="Probabilité (%)",
        color="Classe",
        color_discrete_sequence=df["Couleur"].tolist(),
        hover_data={"Libellé": True, "Zone": True, "Probabilité (%)": ":.2f"},
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", marker_line_width=0)
    fig.update_layout(
        height=430,
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Probabilité",
        yaxis_ticksuffix="%",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_yaxes(range=[0, max(5, df["Probabilité (%)"].max() * 1.15)])
    st.plotly_chart(fig, use_container_width=True)


def render_history(history_df: pd.DataFrame, prediction: Dict[str, Any]) -> None:
    st.subheader("Historique des prédictions")

    if history_df.empty:
        st.info("Historique indisponible ou vide.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["created_at_dt"],
            y=history_df["p_alert_24h"] * 100.0,
            mode="lines+markers",
            name="Probabilité d'alerte 24h",
            line={"color": "#7cc5ff", "width": 3},
            marker={"size": 7},
            hovertemplate="%{x}<br>p_alert_24h : %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=prediction["threshold_24h"] * 100.0,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Seuil {percent(prediction['threshold_24h'], digits=0)}",
        annotation_position="top left",
    )
    fig.update_layout(
        height=430,
        xaxis_title="Horodatage UTC",
        yaxis_title="Probabilité d'alerte 24h",
        yaxis_ticksuffix="%",
        margin=dict(l=20, r=20, t=20, b=20),
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_history_download(history_df: pd.DataFrame) -> None:
    st.subheader("Téléchargement de l'historique")

    if history_df.empty:
        st.write("Aucun historique exportable.")
        return

    export_df = history_df.copy()
    if "created_at_dt" in export_df.columns:
        export_df["created_at_dt"] = export_df["created_at_dt"].astype(str)

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Télécharger l'historique CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="vulcadata_prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(f"{len(history_df)} point(s) d'historique disponibles.")


def render_station_map(station_df: pd.DataFrame, source_mode: str, error: Optional[str]) -> None:
    st.subheader("Stations et capteurs exploités")

    if error:
        st.warning(error)

    if station_df.empty:
        st.info("Aucune coordonnée de station disponible.")
        return

    fig = px.scatter_mapbox(
        station_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Station",
        hover_data={
            "Network": True,
            "channels_used": True if "channels_used" in station_df.columns else False,
            "n_channels_used": True if "n_channels_used" in station_df.columns else False,
            "Latitude": ":.5f",
            "Longitude": ":.5f",
        },
        text="Station",
        zoom=10,
        height=430,
        center={
            "lat": float(station_df["Latitude"].mean()),
            "lon": float(station_df["Longitude"].mean()),
        },
    )
    fig.update_traces(marker={"size": 13, "color": "#ff6b6b"}, textposition="top center")
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Source stations : {source_mode}")

    display_columns = [
        column
        for column in ["Network", "Station", "Latitude", "Longitude", "Elevation", "channels_used", "n_channels_used"]
        if column in station_df.columns
    ]
    st.dataframe(station_df[display_columns].sort_values("Station"), use_container_width=True, hide_index=True)


def render_model_info(payload: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    st.subheader("Informations techniques")

    model = payload.get("model", {})
    metadata = payload.get("metadata", {})

    with st.expander("Modèle et traçabilité"):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Nom du modèle : `{model.get('model_name', 'n/d')}`")
            st.write(f"Version du modèle : `{model.get('model_version', 'n/d')}`")
            st.write(f"Run ID : `{model.get('run_id', 'n/d')}`")
            st.write(f"Eruption ID : `{prediction.get('eruption_id') or 'n/d'}`")

        with col2:
            st.write(f"Agrégation : `{metadata.get('aggregation', 'n/d')}`")
            st.write(f"Array key : `{metadata.get('array_key', 'n/d')}`")
            st.write(f"Batch size : `{metadata.get('batch_size', 'n/d')}`")
            st.write(f"Output shape : `{metadata.get('model_output_shape', metadata.get('output_shape', 'n/d'))}`")
            st.write(f"Clé S3 : `{LATEST_PREDICTION_KEY}`")

    with st.expander("Détail numérique des probabilités"):
        df = probability_dataframe(prediction)
        if df.empty:
            st.write("Aucune probabilité détaillée disponible.")
        else:
            detail_df = df[["Classe", "Libellé", "Zone", "Probabilité", "Probabilité (%)"]].copy()
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with st.expander("JSON brut de la dernière prédiction"):
        st.json(payload)
