import os

APP_TITLE = "Vulcadata - Dashboard opérationnel"

S3_BUCKET = os.getenv("VULCADATA_S3_BUCKET", "vulcadata")
LATEST_PREDICTION_KEY = os.getenv("VULCADATA_DASHBOARD_LATEST_KEY", "predictions/latest/prediction.json")
HISTORY_PREFIX = os.getenv("VULCADATA_DASHBOARD_HISTORY_PREFIX", "predictions/history/")

PREDICTION_CACHE_SECONDS = int(os.getenv("VULCADATA_DASHBOARD_PREDICTION_CACHE_SECONDS", "60"))
HISTORY_CACHE_SECONDS = int(os.getenv("VULCADATA_DASHBOARD_HISTORY_CACHE_SECONDS", "120"))
STATIONS_CACHE_SECONDS = int(os.getenv("VULCADATA_DASHBOARD_STATIONS_CACHE_SECONDS", "3600"))

FDSN_STATION_URL = os.getenv("VULCADATA_FDSN_STATION_URL", "https://ws.ipgp.fr/fdsnws/station/1/query")
NETWORK_CODE = os.getenv("VULCADATA_NETWORK_CODE", "PF")
STATIONS = [item.strip().upper() for item in os.getenv("VULCADATA_STATIONS", "CSS,DSO,ENO,FJS,HIM,SNE").split(",") if item.strip()]
CHANNELS = [item.strip().upper() for item in os.getenv("VULCADATA_CHANNELS", "HHZ,EHZ,HHE,HHN").split(",") if item.strip()]

N_CLASSES = 6
DEFAULT_ALERT_THRESHOLD = 0.35
DEFAULT_MIN_ALERT_CLASS = 3

CLASS_LABELS = {
    0: "Calme",
    1: "Activité faible",
    2: "Surveillance renforcée",
    3: "Pré-alerte 24h",
    4: "Alerte élevée",
    5: "Alerte critique",
}

CLASS_DESCRIPTIONS = {
    0: "Signal compatible avec une situation calme.",
    1: "Activité faible, sans signal d'alerte immédiat.",
    2: "Niveau intermédiaire justifiant une surveillance renforcée.",
    3: "Entrée dans les classes utilisées pour calculer l'alerte 24h.",
    4: "Signal élevé, proche d'un contexte d'alerte opérationnelle.",
    5: "Signal critique, niveau maximal du modèle.",
}

CLASS_ZONES = {
    0: "Hors alerte",
    1: "Hors alerte",
    2: "Hors alerte",
    3: "Alerte",
    4: "Alerte",
    5: "Alerte",
}

CLASS_COLORS = {
    0: "#6c757d",
    1: "#2a9d8f",
    2: "#e9c46a",
    3: "#f4a261",
    4: "#e76f51",
    5: "#d62828",
}

STATIC_STATIONS = [
    {"Station": "CSS", "Network": "PF", "Latitude": -21.2327, "Longitude": 55.7119, "channels_used": "HHZ, EHZ, HHE, HHN"},
    {"Station": "DSO", "Network": "PF", "Latitude": -21.2445, "Longitude": 55.6954, "channels_used": "HHZ, EHZ, HHE, HHN"},
    {"Station": "ENO", "Network": "PF", "Latitude": -21.2878, "Longitude": 55.7125, "channels_used": "HHZ, EHZ, HHE, HHN"},
    {"Station": "FJS", "Network": "PF", "Latitude": -21.2489, "Longitude": 55.7314, "channels_used": "HHZ, EHZ, HHE, HHN"},
    {"Station": "HIM", "Network": "PF", "Latitude": -21.2192, "Longitude": 55.7205, "channels_used": "HHZ, EHZ, HHE, HHN"},
    {"Station": "SNE", "Network": "PF", "Latitude": -21.2661, "Longitude": 55.7072, "channels_used": "HHZ, EHZ, HHE, HHN"},
]
