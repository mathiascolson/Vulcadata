import botocore
import pandas as pd
import streamlit as st

from dashboard_components import (
    render_alert_card,
    render_class_legend,
    render_gauges,
    render_header,
    render_history,
    render_history_download,
    render_kpis,
    render_model_info,
    render_probability_chart,
    render_sidebar,
    render_station_map,
)
from dashboard_config import LATEST_PREDICTION_KEY, S3_BUCKET
from dashboard_data import (
    extract_prediction,
    load_history_dataframe,
    load_latest_payload,
    load_station_dataframe,
)


def main() -> None:
    render_header()
    render_sidebar()

    try:
        latest_payload = load_latest_payload()
        prediction = extract_prediction(latest_payload)
    except botocore.exceptions.NoCredentialsError:
        st.error("Credentials AWS absents. Vérifier les secrets Hugging Face du Space.")
        st.stop()
    except botocore.exceptions.ClientError as error:
        code = error.response.get("Error", {}).get("Code", "unknown")
        st.error(f"Lecture S3 impossible : {code}. Bucket={S3_BUCKET}, key={LATEST_PREDICTION_KEY}")
        st.stop()
    except Exception as error:
        st.error(f"Erreur pendant le chargement de la dernière prédiction : {type(error).__name__}: {error}")
        st.stop()

    try:
        history_df = load_history_dataframe()
    except Exception as error:
        st.warning(f"Historique indisponible : {type(error).__name__}: {error}")
        history_df = pd.DataFrame()

    try:
        station_df, station_source, station_error = load_station_dataframe()
    except Exception as error:
        station_df = pd.DataFrame()
        station_source = "unavailable"
        station_error = f"Stations indisponibles : {type(error).__name__}: {error}"

    render_alert_card(prediction)
    render_kpis(prediction)

    st.divider()

    render_gauges(prediction)

    st.divider()

    render_class_legend()

    st.divider()

    render_probability_chart(prediction)

    st.divider()

    history_left, history_right = st.columns([3, 1])
    with history_left:
        render_history(history_df, prediction)
    with history_right:
        render_history_download(history_df)
        st.subheader("Dernière mise à jour")
        st.write(prediction["created_at_utc"] or "n/d")

    st.divider()

    render_station_map(station_df, station_source, station_error)

    st.divider()

    render_model_info(latest_payload, prediction)


if __name__ == "__main__":
    main()
