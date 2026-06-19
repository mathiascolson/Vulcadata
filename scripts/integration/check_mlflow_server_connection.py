# scripts/test_mlflow_server.py

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI est manquant dans le .env")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("volcano_mlflow_connection_test")

    artifact_path = PROJECT_ROOT / "mlflow_test_artifact.txt"
    artifact_path.write_text("MLflow artifact test OK\n", encoding="utf-8")

    with mlflow.start_run(run_name="test_connection_without_model"):
        mlflow.set_tag("project", "vulcadata")
        mlflow.set_tag("phase", "mlflow_server_setup")
        mlflow.set_tag("model_required", "false")

        mlflow.log_param("test_type", "server_connection")
        mlflow.log_param("backend_store", "postgresql")
        mlflow.log_param("artifact_store", "s3")

        mlflow.log_metric("dummy_metric", 1.0)

        mlflow.log_artifact(str(artifact_path))

    artifact_path.unlink(missing_ok=True)

    print("Test MLflow terminé avec succès.")


if __name__ == "__main__":
    main()