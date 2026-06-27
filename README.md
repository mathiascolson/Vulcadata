# VulcaData

Projet de prédiction d’alerte volcanique appliqué au Piton de la Fournaise.

VulcaData vise à exploiter des signaux sismiques collectés auprès de l’observatoire volcanologique du Piton de la Fournaise afin d’estimer un niveau d’alerte volcanique.

Le projet met en place une chaîne MLOps orchestrée à partir de données sismiques extraites localement : transformation des signaux, feature engineering, preprocessing, entraînement de modèles, inférence, validation qualité, monitoring, traçabilité des décisions modèle et visualisation des résultats.

## Objectif du projet

L’objectif est de transformer des signaux sismiques bruts en indicateurs exploitables, puis d’utiliser un modèle de classification pour estimer le niveau d’alerte associé à une séquence temporelle récente.

Le modèle opérationnel retourne une classe d’alerte parmi 6 niveaux :

* classe 0 : période calme ou non critique ;
* classes 1 à 2 : activité pré-éruptive éloignée ;
* classes 3 à 5 : activité pré-éruptive plus proche, utilisée pour le calcul de l’alerte à 24 heures.

La probabilité d’alerte à 24 heures est calculée en additionnant les probabilités des classes 3, 4 et 5.

## Référence scientifique

Le projet s’appuie fortement sur l’étude suivante :

Characterization of volcanic stages using seismic features: Case of Tajogaite (2021) and Colima (2013–2022), publiée en 2025.

Cette étude a servi de base méthodologique pour l’extraction de caractéristiques sismiques permettant de différencier plusieurs phases d’activité volcanique. VulcaData adapte cette approche au cas du Piton de la Fournaise.

Plusieurs features utilisées dans le projet sont directement inspirées de cette approche, notamment :

* la kurtosis, utilisée pour caractériser la forme statistique du signal ;
* l’entropie de Shannon, utilisée pour mesurer la complexité ou la dispersion de l’information dans le signal ;
* le Frequency Index, utilisé pour comparer l’énergie contenue dans différentes bandes de fréquences.

Ces indicateurs sont centraux dans l'étude de référence et permettent de résumer des signaux sismiques complexes sous forme de variables exploitables par les modèles de Machine Learning et de Deep Learning.


## Source des données

Les données sismiques sont collectées directement auprès de l’API de l’observatoire volcanologique du Piton de la Fournaise :

https://ws.ipgp.fr/fdsnws/dataselect/1

Les données brutes sont récupérées au format MiniSEED sur le réseau de stations sismiques du Piton de la Fournaise. Elles sont ensuite transformées en séries temporelles agrégées, puis en séquences exploitables par les modèles de Deep Learning.

Le périmètre des périodes utilisées par le projet est défini dans le fichier :

`data/metadata/extraction_periods.csv`

## Pipeline de données

Le pipeline de données se décompose en deux niveaux.

Dans la version actuelle, en amont d'Airflow, l’extraction des fichiers MiniSEED depuis l’API de l’observatoire est lancée manuellement via un script dédié.

Ce script lit le fichier :

`data/metadata/extraction_periods.csv`

Ce fichier fournit les informations nécessaires pour construire les requêtes vers l’API FDSN : périodes temporelles, type de période, réseau, stations et canaux sismiques.

L’extraction produit ensuite des CSV agrégés dans :

`data/extraction/processed_csv/`

Les DAGs Airflow prennent ensuite le relais à partir de ces CSV agrégés. Ils orchestrent le preprocessing, la construction des séquences temporelles, la validation qualité et les traitements MLOps associés à l’inférence ou au réentraînement.

Les principales étapes sont :

* définition des périodes à traiter dans `extraction_periods.csv` ;
* extraction manuelle des signaux MiniSEED via le script dédié ;
* filtrage, nettoyage et agrégation temporelle des signaux ;
* extraction de caractéristiques sismiques ;
* preprocessing orchestré par Airflow selon le cas d’usage ;
* construction de séquences temporelles au format (120, 992) ;
* validation qualité avec Great Expectations ;
* inférence ou entraînement d’un modèle candidat ;
* monitoring, comparaison modèle et traçabilité selon le pipeline exécuté.

Les principales familles de variables utilisées sont :

* amplitudes sismiques ;
* énergie dans différentes bandes de fréquences ;
* indice fréquentiel ;
* entropie ;
* kurtosis ;
* statistiques glissantes ;
* indicateurs agrégés par station et par canal.

## Modélisation

Plusieurs approches de modélisation ont été explorées pour apprendre les dynamiques temporelles des signaux sismiques.

Le modèle opérationnel retenu est un CNN-Transformer de classification. Il combine :

* des couches convolutionnelles pour extraire des motifs locaux dans les séquences ;
* un encodeur Transformer pour modéliser les dépendances temporelles ;
* une couche de classification pour produire les probabilités associées aux 6 classes d’alerte.

La règle d’alerte utilisée est la suivante :

p_alert_24h = P(classe 3) + P(classe 4) + P(classe 5)

Une alerte est déclenchée lorsque cette probabilité dépasse le seuil défini dans la configuration du projet.

## Architecture MLOps

Le projet intègre une architecture MLOps destinée à fiabiliser l’exécution des traitements et le suivi du modèle.

Les principales briques sont :

* Airflow pour l’orchestration des pipelines ;
* MLflow pour le suivi des expérimentations, des entraînements candidats, des métriques, des artefacts modèle et des décisions de comparaison champion/candidat ;
* Great Expectations pour la validation des données d’entrée ;
* Evidently pour la génération de rapports de monitoring et l’analyse de dérive ;
* S3 pour le stockage des sorties opérationnelles légères utilisées par le dashboard ;
* GitHub Actions pour l’exécution des tests en intégration continue ;
* Streamlit pour la visualisation des prédictions et de l’historique d’alerte.

L’architecture est organisée autour de deux DAGs Airflow opérationnels :

* `volcano_inference_pipeline` : pipeline d’inférence, depuis le preprocessing des CSV agrégés jusqu’à l’écriture des prédictions et rapports légers dans S3 ;
* `volcano_retraining_pipeline` : pipeline de réentraînement conditionnel, depuis le preprocessing training jusqu’à la décision de promotion ou de rejet d’un modèle candidat.

MLflow joue un rôle central dans la traçabilité du cycle modèle. Il permet de conserver l’historique des entraînements, les métriques de performance, les paramètres, les artefacts associés aux modèles et les décisions prises lors de la comparaison entre le modèle champion et un modèle candidat.

Dans la version actuelle, la décision opérationnelle de promotion ou de rejet reste orchestrée par Airflow et les scripts du projet. MLflow sert de référentiel de suivi et d’audit : il permet de justifier a posteriori pourquoi un candidat a été accepté, rejeté ou simplement archivé.

Great Expectations est utilisé comme contrôle qualité bloquant à deux niveaux :

* sur le dernier batch d’inférence, afin de vérifier la conformité du tenseur utilisé par le modèle opérationnel ;
* sur le dataset de réentraînement, afin de vérifier les clés NPZ attendues, les dimensions des splits, l’absence de valeurs non finies et la validité des labels avant fusion et entraînement.

Ainsi, un dataset invalide bloque le pipeline avant l’inférence ou avant le réentraînement.

Evidently complète cette supervision en produisant des rapports de monitoring permettant d’identifier d’éventuels écarts entre les données de référence et les données récentes. Ces rapports alimentent l’analyse de dérive et peuvent contribuer à la décision de maintenir, rejeter ou réentraîner un modèle.

## Dashboard

Un dashboard Streamlit permet de consulter les résultats d’inférence.

Il présente notamment :

* la dernière prédiction disponible ;
* la classe prédite ;
* la probabilité d’alerte à 24 heures ;
* le seuil d’alerte utilisé ;
* l’historique des prédictions ;
* l’évolution du niveau d’alerte dans le temps.

Dashboard :

https://vartkirl-vulcadata-dashboard.hf.space/

## Structure du projet

Le dépôt est organisé autour des principaux dossiers suivants :

* `configs/` : fichiers de configuration du projet ;
* `src/` : code source principal ;
* `infra/airflow/` : DAGs Airflow et configuration d’orchestration ;
* `infra/huggingface_spaces/streamlit` : application Streamlit ;
* `tests/` : tests unitaires et tests de contrat ;
* `reports/` : rapports générés localement ;
* `data/` : données locales ignorées par Git.

## Exécution des pipelines

Les pipelines s’appuient sur un fichier CSV décrivant les périodes sismiques à traiter :

`data/metadata/extraction_periods.csv`

Colonnes attendues :

`period_id;period_type;period_start_utc;period_end_utc;eruption_start_utc;eruption_end_utc;split;network;stations;channels`

Exemple :

`eruption_2019_08_15;eruption;2019-08-13T00:00:00Z;2019-08-16T00:00:00Z;2019-08-15T04:25:00Z;;;PF;CSS,DSO,ENO,FJS,HIM,SNE;HHZ,EHZ,HHE,HHN`

Le séparateur ; est recommandé sous Windows/Excel en configuration française. Les colonnes stations et channels peuvent contenir plusieurs valeurs séparées par des virgules.

Valeurs principales de `period_type` :

* `eruption` : période associée à une éruption connue ;
* `quiet` : période calme utilisée comme référence non éruptive ;
* `inference` : période destinée uniquement à l’inférence.

**Étape 1 — Extraction manuelle des données**

Depuis la racine du projet :

```
python -m src.extraction.extract_volcano_periods --periods data\metadata\extraction_periods.csv --output-dir data\extraction
```

Cette commande récupère les données MiniSEED, applique les traitements signal définis dans le script d’extraction et produit les CSV agrégés dans :

`data/extraction/processed_csv/`

**Étape 2 — Pipeline d’inférence**

À partir des CSV agrégés, Airflow prend le relais avec le DAG :

`volcano_inference_pipeline`

Ce DAG orchestre le preprocessing inference, la création du batch d’inférence, la validation Great Expectations, l’inférence, l’écriture des sorties opérationnelles, le monitoring Evidently et le logging MLflow.

Depuis le dossier Airflow :

`cd infra\airflow`

Lancer le DAG d’inférence :

```
docker compose exec airflow-scheduler airflow dags trigger volcano_inference_pipeline
```

**Étape 3 — Pipeline de réentraînement**

À partir des CSV agrégés, Airflow peut également lancer le DAG :

`volcano_retraining_pipeline`

Ce DAG orchestre le preprocessing training, la validation Great Expectations du dataset de réentraînement, la fusion avec le dataset de référence, l’entraînement candidat, le rapport Evidently, la comparaison au champion, la décision de promotion ou de rejet, l’archivage et le logging MLflow.

Depuis le dossier Airflow :

`cd infra\airflow`

Lancer le DAG de réentraînement :

```docker compose exec airflow-scheduler airflow dags trigger volcano_retraining_pipeline```

Une décision `reject_candidate` n’est pas une erreur pipeline. Elle signifie que le candidat a bien été entraîné et évalué, mais qu’il ne respecte pas les règles de promotion définies par le projet.

## Tests

Les tests principaux peuvent être exécutés avec :

`python -m pytest tests -v`

À l’état actuel du projet, la suite de tests principale valide notamment :

* les contrats de configuration ;
* le chargement du modèle opérationnel ;
* les fonctions d’inférence ;
* les écritures S3 simulées ;
* la validation Great Expectations du batch d’inférence ;
* la validation Great Expectations du dataset de réentraînement ;
* les contrats des rapports produits par les pipelines ;
* la topologie du DAG de réentraînement.

Le projet est également testé via GitHub Actions à chaque mise à jour du dépôt.

## Limites et évolutions possibles

Plusieurs évolutions peuvent renforcer le projet :

* automatiser la collecte des données sismiques récentes depuis l’observatoire ;
* automatiser ou semi-automatiser la labellisation des nouvelles périodes utilisables pour le réentraînement ;
* renforcer l’usage de MLflow Model Registry pour gérer formellement les versions champion/challenger, les promotions, les rejets et les rollbacks ;
* intégrer des sources complémentaires comme les données GPS, les gaz volcaniques ou l’imagerie satellite thermique ;
* élargir la période historique d’apprentissage ;
* tester la généralisation sur d’autres volcans actifs ;
* envisager une architecture data warehouse ou lakehouse pour historiser les données, structurer les features et faciliter les traitements à plus grande échelle ;
* envisager du calcul distribué pour les transformations massives et l’entraînement sur un historique élargi.


## Conclusion

VulcaData démontre la mise en place d’une chaîne MLOps de prédiction appliquée à un cas géophysique réel : préparation de données sismiques, transformation en features temporelles, entraînement de modèles de Deep Learning, inférence, orchestration, validation, monitoring et restitution des résultats dans un dashboard.

Le projet assume une limite de périmètre : l’extraction des données MiniSEED reste lancée manuellement. En revanche, les étapes de preprocessing, validation, inférence, réentraînement, monitoring et gouvernance modèle sont orchestrées dans Airflow.