# VulcaData

Prédiction d'alerte volcanique à partir des signaux sismiques du Piton de la Fournaise.

VulcaData transforme des données MiniSEED en séquences temporelles exploitables par un modèle de Deep Learning, puis orchestre l'inférence, la validation des données, le monitoring et le suivi des modèles dans une chaîne MLOps.

Ce dépôt documente l'état du projet présenté dans le cadre des certifications **CDSD** et **AIA**. Les évolutions envisagées après ces certifications sont regroupées dans une roadmap distincte et ne sont pas présentées comme déjà réalisées.

## Résultats clés

Le CNN-Transformer retenu produit six classes temporelles, mais la décision opérationnelle se concentre sur un objectif plus robuste : détecter une situation d'alerte à 24 heures.

| Métrique sur le jeu de test | Résultat |
|---|---:|
| F1 — alerte à 24 h | **0,720** |
| Recall — alerte à 24 h | **0,816** |
| Precision — alerte à 24 h | **0,644** |
| Seuil de décision | **0,35** |

La probabilité d'alerte à 24 heures est définie par :

`p_alert_24h = P(classe 3) + P(classe 4) + P(classe 5)`

Le recall élevé privilégie la détection des épisodes à risque, au prix d'un nombre plus important de faux positifs. Ces valeurs proviennent de [`configs/final_model_decision.json`](configs/final_model_decision.json).

### Limite importante du modèle

Les performances ne sont pas homogènes sur les six classes : le **macro-F1 multiclasse est d'environ 0,110** et la **balanced accuracy d'environ 0,192** sur le jeu de test. Certaines classes intermédiaires ne sont pas correctement distinguées.

Le modèle ne doit donc pas être interprété comme un classifieur multiclasse généraliste performant. Il a été retenu pour le sous-objectif métier d'alerte binaire à 24 heures, nettement plus exploitable. Une régression du délai avant éruption est conservée comme estimation exploratoire, pas comme prédiction opérationnelle fiable.

## Périmètre implémenté

Les éléments suivants sont présents dans le dépôt :

- extraction manuelle des signaux MiniSEED depuis l'API FDSN de l'Observatoire volcanologique du Piton de la Fournaise ;
- nettoyage, agrégation et feature engineering des signaux sismiques ;
- création de séquences temporelles de forme `(120, 992)` ;
- modèle CNN-Transformer de classification à six classes et règle d'alerte à 24 heures ;
- pipeline d'inférence orchestré par Airflow ;
- pipeline de réentraînement conditionnel avec comparaison champion/candidat ;
- validation des données avec Great Expectations ;
- rapports de monitoring avec Evidently ;
- traçabilité des expériences, métriques et décisions avec MLflow ;
- stockage S3 des sorties légères utilisées par le dashboard ;
- tests automatisés et intégration continue avec GitHub Actions.

La présence de ces composants et de leurs tests permet d'en examiner l'implémentation. Elle ne constitue pas, à elle seule, la preuve d'une exécution de bout en bout contre les services externes S3 et MLflow dans un nouvel environnement.

## Objectif et données

Les signaux sont collectés via l'API FDSN de l'observatoire :

<https://ws.ipgp.fr/fdsnws/dataselect/1>

Le périmètre temporel, les stations et les canaux sont définis dans [`data/metadata/extraction_periods.csv`](data/metadata/extraction_periods.csv). L'extraction, lancée manuellement en amont d'Airflow, produit des CSV agrégés dans `data/extraction/processed_csv/`.

Les principales familles de variables sont :

- amplitudes et énergie dans différentes bandes de fréquences ;
- indice fréquentiel ;
- entropie de Shannon ;
- kurtosis ;
- statistiques glissantes ;
- indicateurs agrégés par station et par canal.

La méthodologie de feature engineering s'inspire notamment de l'étude *Characterization of volcanic stages using seismic features: Case of Tajogaite (2021) and Colima (2013–2022)*, publiée en 2025, puis l'adapte au Piton de la Fournaise.

## Modélisation

Le modèle retenu combine :

- des couches convolutionnelles pour extraire des motifs locaux ;
- un encodeur Transformer pour apprendre les dépendances temporelles ;
- une tête de classification produisant les probabilités des six classes.

Les classes 0 à 2 représentent les périodes calmes ou pré-éruptives éloignées. Les probabilités des classes 3 à 5 sont agrégées pour calculer l'alerte à 24 heures.

## Architecture MLOps

Deux DAGs Airflow structurent la chaîne :

- [`volcano_inference_pipeline`](infra/airflow/dags/volcano_inference_pipeline.py) orchestre le preprocessing, la préparation du dernier batch, la validation Great Expectations, l'inférence, les vérifications de sortie, le rapport Evidently et le logging MLflow ;
- [`volcano_retraining_pipeline`](infra/airflow/dags/volcano_retraining_pipeline.py) orchestre la préparation et la validation du dataset, l'entraînement d'un candidat, sa comparaison au champion, sa promotion conditionnelle ou son rejet, l'archivage et la traçabilité MLflow.

MLflow sert de référentiel de suivi et d'audit. La décision de promotion ou de rejet reste pilotée par Airflow et les scripts du projet ; elle n'est pas déléguée à un Model Registry entièrement automatisé.

Great Expectations applique des contrôles bloquants avant l'inférence et avant le réentraînement. Evidently génère les rapports utilisés pour examiner les écarts entre données de référence et données récentes.

## Dashboard externe

Un dashboard Streamlit permet de consulter la dernière prédiction, la classe estimée, la probabilité d'alerte à 24 heures, le seuil utilisé et l'historique des prédictions :

<https://vartkirl-vulcadata-dashboard.hf.space/>

Le dashboard est déployé séparément sur Hugging Face Spaces. **Son code Streamlit n'est pas versionné dans ce dépôt** 

## Structure du dépôt

- `configs/` : configurations du modèle, de l'inférence et de l'entraînement ;
- `src/` : extraction, preprocessing, inférence, retraining et monitoring ;
- `infra/airflow/` : DAGs et environnement d'orchestration ;
- `scripts/` : entraînement, préparation des décisions modèle et utilitaires MLflow ;
- `tests/` : tests unitaires et tests de contrat ;
- `reports/` : rapports et artefacts de présentation versionnés ;
- `data/` : métadonnées versionnées et données locales ignorées par Git.

## Exécution

### 1. Extraire et préparer les signaux

Depuis la racine du projet :

```powershell
python -m src.extraction.extract_volcano_periods --periods data\metadata\extraction_periods.csv --output-dir data\extraction
```

Le fichier de périodes utilise les colonnes suivantes :

```text
period_id;period_type;period_start_utc;period_end_utc;eruption_start_utc;eruption_end_utc;split;network;stations;channels
```

Les principales valeurs de `period_type` sont `eruption`, `quiet` et `inference`.

### 2. Déclencher l'inférence

Depuis `infra/airflow/` :

```powershell
docker compose exec airflow-scheduler airflow dags trigger volcano_inference_pipeline
```

### 3. Déclencher le réentraînement

Depuis `infra/airflow/` :

```powershell
docker compose exec airflow-scheduler airflow dags trigger volcano_retraining_pipeline
```

Une décision `reject_candidate` signifie que le candidat a été entraîné et évalué, mais qu'il ne satisfait pas les règles de promotion. Ce résultat n'est pas une erreur du pipeline.

## Tests

```powershell
python -m pytest tests -v
```

La suite couvre notamment les contrats de configuration, le chargement du modèle, l'inférence, les écritures S3 simulées, les validations Great Expectations et la topologie du DAG de réentraînement. Certains tests de rapports d'exécution sont ignorés si les artefacts correspondants n'ont pas encore été générés ; une suite verte ne prouve donc pas à elle seule l'exécution complète des services externes.

## Roadmap — non implémentée

Les éléments ci-dessous sont des pistes d'évolution, pas des fonctionnalités livrées dans l'état actuel du dépôt :

- automatiser la collecte récente et la labellisation des nouvelles périodes ;
- renforcer la gestion formelle champion/challenger avec MLflow Model Registry ;
- intégrer des données GPS, de gaz volcaniques ou d'imagerie satellite thermique ;
- élargir l'historique d'apprentissage et tester la généralisation à d'autres volcans ;
- évaluer une architecture data warehouse ou lakehouse ;
- étudier le calcul distribué sur un historique plus large.

Les travaux envisagés autour de Kubernetes, Helm, Ray/KubeRay ou d'une migration NPZ vers Parquet ne sont pas présents dans cette version et ne sont pas revendiqués comme réalisés.

## Conclusion

VulcaData illustre une chaîne Data Science et MLOps complète sur un cas géophysique réel : préparation de signaux sismiques, modélisation temporelle, décision métier, orchestration, qualité des données, monitoring et traçabilité.

Sa principale force est la traduction d'un modèle multiclasse imparfait en un objectif d'alerte à 24 heures mesurable et plus pertinent pour le cas d'usage. Sa principale limite reste la performance faible sur la classification fine des six classes, ainsi que la collecte MiniSEED encore manuelle.
