# VulcaData

Projet de prédiction d’alerte volcanique appliqué au Piton de la Fournaise.

VulcaData vise à exploiter des signaux sismiques collectés auprès de l’observatoire volcanologique du Piton de la Fournaise afin d’estimer un niveau d’alerte volcanique à court terme. Le projet couvre la chaîne complète : collecte des données, transformation des signaux, feature engineering, entraînement de modèles de prédiction, inférence, validation, orchestration, monitoring et visualisation des résultats.

## Objectif du projet

L’objectif est de transformer des signaux sismiques bruts en indicateurs exploitables, puis d’utiliser un modèle de classification pour estimer le niveau d’alerte associé à une séquence temporelle récente.

Le modèle opérationnel retourne une classe d’alerte parmi 6 niveaux :

* classe 0 : période calme ou non critique ;
* classes 1 à 2 : activité pré-éruptive éloignée ;
* classes 3 à 5 : activité pré-éruptive plus proche, utilisée pour le calcul de l’alerte à 24 heures.

La probabilité d’alerte à 24 heures est calculée en additionnant les probabilités des classes 3, 4 et 5.

## Référence scientifique

Le projet s’appuie fortement sur l’étude suivante :

Characterization of volcanic stages using seismic features: Case of Tajogaite (2021) and Colima (2013–2022)
Pablo Rey-Devesa, Jesús M. Ibáñez, Ligdamis Gutiérrez, Janire Prudencio, Aarón Álvarez-Hernández, Mauricio Bretón, Raúl Arámbula, Félix Ortigosa, Imelda Plasencia, Alberto Ardid, Luca D’Auria, Nemesio Pérez, Manuel Titos, Carmen Benítez.

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

## Pipeline de données

Le pipeline de préparation des données comprend les étapes suivantes :

1. collecte des données sismiques MiniSEED ;
2. filtrage et agrégation temporelle des signaux ;
3. extraction de caractéristiques sismiques ;
4. construction de séquences temporelles ;
5. labellisation des périodes selon leur proximité avec les éruptions ;
6. séparation des données en jeux d’entraînement, validation et test ;
7. sauvegarde des datasets préparés au format NPZ.

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
* MLflow pour le suivi des expérimentations et des décisions modèle ;
* Great Expectations pour la validation des données d’entrée ;
* Evidently pour la génération de rapports de monitoring ;
* S3 pour le stockage des sorties opérationnelles légères ;
* GitHub Actions pour l’exécution des tests en intégration continue ;
* Streamlit pour la visualisation des prédictions et de l’historique d’alerte.

## Périmètre actuel

Dans la version actuelle, l’inférence est déclenchée manuellement depuis Airflow sur un batch local prépréparé.

Les fichiers MiniSEED, les CSV agrégés et les datasets NPZ restent stockés localement en raison de leur volume. S3 est utilisé uniquement pour les artefacts légers nécessaires au suivi opérationnel :

* prédictions récentes ;
* historique des prédictions ;
* rapports d’inférence ;
* rapports de monitoring ;
* décisions modèle ;
* artefacts de référence du modèle champion.

L’automatisation complète de la collecte périodique des données récentes depuis l’observatoire est identifiée comme une évolution du projet.

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
* `dashboard/streamlit/` : application Streamlit ;
* `tests/` : tests unitaires et tests de contrat ;
* `reports/` : rapports générés localement ;
* `data/` : données locales ignorées par Git.

## Exécution des pipelines

Les pipelines partent d’un fichier CSV décrivant les périodes sismiques à extraire.

Ce fichier doit être placé ici :

data/metadata/extraction_periods.csv

Colonnes attendues :

period_id;period_type;period_start_utc;period_end_utc;eruption_start_utc;eruption_end_utc;split;network;stations;channels

Exemple :

eruption_2019_08_15;eruption;2019-08-13T00:00:00Z;2019-08-16T00:00:00Z;2019-08-15T04:25:00Z;;;PF;CSS,DSO,ENO,FJS,HIM,SNE;HHZ,EHZ,HHE,HHN

Le séparateur ; est recommandé sous Windows/Excel en configuration française. Les colonnes stations et channels peuvent contenir plusieurs valeurs séparées par des virgules.

Valeurs principales de period_type :

eruption : période associée à une éruption connue ;
quiet : période calme utilisée comme référence non éruptive ;
inference : période destinée uniquement à l’inférence.
Pipeline extraction → inférence

L’extraction MiniSEED et le preprocessing sont exécutés en amont d’Airflow.

Depuis la racine du projet, lancer l’extraction :

python -m src.extraction.extract_volcano_periods --periods data\metadata\extraction_periods.csv --output-dir data\extraction

Cette commande produit les CSV agrégés dans :

data/extraction/processed_csv/

Préparer ensuite le fichier NPZ d’inférence :

python -m src.preprocessing.preprocess_volcano_dataset --mode inference --periods data\metadata\extraction_periods.csv --processed-csv-dir data\extraction\processed_csv --output-dir data\preprocessing\processed --inference-output-name inference_source.npz

Cette commande génère :

data/preprocessing/processed/inference_source.npz

À partir de ce fichier NPZ, Airflow prend le relais avec le DAG :

volcano_inference_pipeline

Ce DAG orchestre la préparation du dernier batch, la validation Great Expectations, l’inférence, l’écriture des prédictions dans S3, le monitoring Evidently et la vérification des sorties utilisées par le dashboard Streamlit.

Depuis le dossier Airflow :

cd infra\airflow

Lancer le DAG d’inférence en test local :

docker compose exec airflow-scheduler bash -lc "cd /opt/vulcadata && airflow dags test volcano_inference_pipeline 2026-06-20"

Pipeline extraction → réentraînement

Comme pour l’inférence, l’extraction MiniSEED et le preprocessing sont exécutés en amont d’Airflow.

Depuis la racine du projet, lancer l’extraction :

python -m src.extraction.extract_volcano_periods --periods data\metadata\extraction_periods.csv --output-dir data\extraction

Préparer ensuite un lot NPZ compatible avec le réentraînement :

python -m src.preprocessing.preprocess_volcano_dataset --mode training --periods data\metadata\extraction_periods.csv --processed-csv-dir data\extraction\processed_csv --output-dir data\retraining\ready --training-output-name volcano_multi_new_batch.npz --split-strategy chronological

Cette commande écrit le nouveau lot dans :

data/retraining/ready/

À partir de ce dépôt, Airflow prend le relais avec le DAG :

volcano_retraining_pipeline

Ce DAG détecte les nouveaux fichiers .npz, prépare le lot candidat, entraîne un modèle candidat, génère les rapports de suivi, compare le candidat au champion, applique la règle de promotion et archive les fichiers traités.

Depuis le dossier Airflow :

cd infra\airflow

Lancer le DAG de réentraînement en test local :

docker compose exec airflow-scheduler bash -lc "cd /opt/vulcadata && airflow dags test volcano_retraining_pipeline 2026-06-20"

L’extraction ne modifie pas directement les fichiers NPZ. Elle produit des CSV agrégés. Les fichiers NPZ sont générés uniquement par le script de preprocessing, selon les paramètres --output-dir, --inference-output-name et --training-output-name.

## Tests

Les tests principaux peuvent être exécutés avec :

`python -m pytest tests -v`

Les tests du dashboard peuvent être exécutés avec :

`python -m pytest dashboard/streamlit/tests -v`

Le projet est également testé via GitHub Actions à chaque mise à jour du dépôt.

## Limites et évolutions possibles

Plusieurs évolutions peuvent renforcer le projet :

* automatiser la collecte des données sismiques récentes depuis l’observatoire ;
* intégrer des sources complémentaires comme les données GPS, les gaz volcaniques ou l’imagerie satellite thermique ;
* élargir la période historique d’apprentissage ;
* tester la généralisation sur d’autres volcans actifs ;
* améliorer la chaîne de labellisation pour alimenter le réentraînement ;
* industrialiser davantage le suivi de dérive et la promotion automatique des modèles.

## Conclusion

VulcaData démontre la mise en place d’une chaîne complète de prédiction appliquée à un cas géophysique réel : extraction de données sismiques, transformation en features temporelles, entraînement de modèles de Deep Learning, inférence, orchestration, validation, monitoring et restitution des résultats dans un dashboard.
