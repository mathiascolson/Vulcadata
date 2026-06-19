---
title: Vulcadata Dashboard
emoji: 🌋
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Vulcadata Dashboard

Dashboard Streamlit pour la visualisation des prédictions opérationnelles du projet Vulcadata.

L'application lit en priorité les sorties JSON produites par le pipeline d'inférence et stockées sur S3.

Variables d'environnement attendues dans les secrets Hugging Face Space :

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `VULCADATA_S3_BUCKET`
- `VULCADATA_DASHBOARD_LATEST_KEY`

Variables optionnelles :

- `VULCADATA_FDSN_STATION_URL`
- `VULCADATA_STATIONS`

Si les variables S3 sont absentes, l'application démarre avec une prédiction de démonstration afin de valider le déploiement.

La carte des stations interroge par défaut le service FDSN station de l'IPGP pour le réseau PF.
Configuration recommandée :

VULCADATA_DASHBOARD_LATEST_KEY=predictions/latest/prediction.json
