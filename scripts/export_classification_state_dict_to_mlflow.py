from __future__ import annotations

import sys
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_project_dotenv(dotenv_path: str = ".env") -> None:
    """
    Load local environment variables from .env when python-dotenv is available.

    This is useful for local execution because PowerShell does not load .env
    files automatically.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(dotenv_path=dotenv_path, override=False)


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object.")

    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def import_model_class(import_path: str) -> type[torch.nn.Module]:
    """
    Import a PyTorch model class from a string path.

    Expected format:
        module.path:ClassName

    Example:
        scripts.training.train_cnn_transformer_classif:CNNTransformerClassifier
    """
    if ":" not in import_path:
        raise ValueError(
            "model_class must use the format 'module.path:ClassName'. "
            "Example: scripts.training.train_cnn_transformer_classif:"
            "CNNTransformerClassifier"
        )

    module_path, class_name = import_path.split(":", 1)

    if not module_path or not class_name:
        raise ValueError(
            "model_class must use the format 'module.path:ClassName'."
        )

    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    if not isinstance(model_class, type):
        raise TypeError(f"{import_path} does not point to a class.")

    if not issubclass(model_class, torch.nn.Module):
        raise TypeError(f"{import_path} is not a torch.nn.Module subclass.")

    return model_class


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Extract the PyTorch state_dict from a checkpoint dictionary.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint must be a dictionary containing model_state_dict."
        )

    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise ValueError(
        "Could not extract a PyTorch state_dict from checkpoint. "
        "Expected a key named 'model_state_dict' or 'state_dict'. "
        f"Available keys: {sorted(checkpoint.keys())}"
    )


def remove_module_prefix_if_needed(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove a DataParallel 'module.' prefix if all parameters have it.
    """
    if not state_dict:
        raise ValueError("state_dict is empty.")

    if not all(key.startswith("module.") for key in state_dict):
        return state_dict

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def build_constructor_values(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """
    Build a dictionary of possible constructor values from the checkpoint.

    The actual constructor parameters are filtered later with inspect.signature.
    This allows the script to support several common parameter names.
    """
    n_features = checkpoint.get("n_features")
    n_classes = checkpoint.get("n_classes")
    seq_len = checkpoint.get("seq_len")

    return {
        "feature_dim": n_features,
        "n_features": n_features,
        "input_dim": n_features,
        "input_size": n_features,
        "num_features": n_features,
        "seq_len": seq_len,
        "sequence_length": seq_len,
        "n_classes": n_classes,
        "num_classes": n_classes,
        "output_dim": n_classes,
        "d_model": checkpoint.get("d_model"),
        "nhead": checkpoint.get("nhead"),
        "n_heads": checkpoint.get("nhead"),
        "num_layers": checkpoint.get("num_layers"),
        "n_layers": checkpoint.get("num_layers"),
        "dim_feedforward": checkpoint.get("dim_feedforward"),
        "feedforward_dim": checkpoint.get("dim_feedforward"),
        "dropout": checkpoint.get("dropout"),
        "input_noise_std": checkpoint.get("input_noise_std"),
    }


def build_model_from_checkpoint(
    model_class: type[torch.nn.Module],
    checkpoint: dict[str, Any],
) -> torch.nn.Module:
    """
    Rebuild the model object using constructor parameters stored in checkpoint.
    """
    signature = inspect.signature(model_class.__init__)

    accepted_params = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    candidate_values = build_constructor_values(checkpoint)

    kwargs = {
        name: value
        for name, value in candidate_values.items()
        if name in accepted_params and value is not None
    }

    missing_required = []

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        if parameter.default is inspect.Parameter.empty and name not in kwargs:
            missing_required.append(name)

    if missing_required:
        raise ValueError(
            "Missing required constructor arguments for "
            f"{model_class.__name__}: {missing_required}. "
            f"Accepted params: {sorted(accepted_params)}. "
            f"Available checkpoint keys: {sorted(checkpoint.keys())}. "
            f"Detected kwargs: {kwargs}."
        )

    return model_class(**kwargs)


def ensure_input_example(
    input_npz: str | Path,
    array_key: str,
    input_example_size: int,
) -> np.ndarray:
    """
    Load a small input example from the NPZ used by the classifier.
    """
    if input_example_size <= 0:
        raise ValueError("input_example_size must be strictly positive.")

    input_path = Path(input_npz)

    if not input_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")

    npz = np.load(input_path, allow_pickle=False)

    if array_key not in npz.files:
        raise KeyError(
            f"Array key {array_key!r} not found in {input_path}. "
            f"Available keys: {npz.files}"
        )

    array = npz[array_key]

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D input array for {array_key!r}, got shape {array.shape}."
        )

    if array.shape[0] < input_example_size:
        raise ValueError(
            f"Input example size {input_example_size} exceeds available "
            f"rows in {array_key!r}: {array.shape[0]}."
        )

    return array[:input_example_size].astype(np.float32)


def update_final_model_decision(
    decision_path: str | Path,
    output_path: str | Path,
    *,
    export_run_id: str,
    source_run_id: str,
    artifact_path: str,
    checkpoint_artifact: str,
    model_class: str,
) -> None:
    """
    Create an exported decision JSON that points inference to the MLflow model
    export run while preserving the original training run id.
    """
    decision = read_json(decision_path)

    decision_block = decision.get("decision")

    if not isinstance(decision_block, dict):
        raise ValueError("final_model_decision.json has no valid 'decision' block.")

    selected_key = decision_block.get("main_operational_model")

    if not isinstance(selected_key, str) or not selected_key:
        raise ValueError("decision.main_operational_model is missing or invalid.")

    candidate = decision.get(selected_key)

    if not isinstance(candidate, dict):
        raise ValueError(f"Selected candidate block {selected_key!r} is missing.")

    candidate["training_run_id"] = source_run_id
    candidate["checkpoint_artifact"] = checkpoint_artifact
    candidate["model_class"] = model_class
    candidate["model_export_run_id"] = export_run_id
    candidate["run_id"] = export_run_id
    candidate["artifact_path"] = artifact_path
    candidate["flavor"] = "pyfunc"
    candidate["model_uri"] = f"runs:/{export_run_id}/{artifact_path}"

    decision["mlflow_model_export"] = {
        "source_training_run_id": source_run_id,
        "checkpoint_artifact": checkpoint_artifact,
        "model_class": model_class,
        "export_run_id": export_run_id,
        "artifact_path": artifact_path,
        "model_uri": f"runs:/{export_run_id}/{artifact_path}",
        "flavor": "pyfunc",
    }

    write_json(output_path, decision)


def log_checkpoint_params(checkpoint: dict[str, Any]) -> None:
    """
    Log scalar checkpoint metadata as MLflow params.
    """
    keys_to_log = (
        "model_type",
        "task",
        "n_classes",
        "n_features",
        "seq_len",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "input_noise_std",
        "best_val_score",
        "best_val_metric",
    )

    for key in keys_to_log:
        if key not in checkpoint:
            continue

        value = checkpoint[key]

        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)


def validate_environment() -> None:
    """
    Fail early when the required remote services are not configured.
    """
    tracking_uri = mlflow.get_tracking_uri()

    if not tracking_uri or tracking_uri.startswith("file:") or tracking_uri == "mlruns":
        raise RuntimeError(
            "MLflow tracking URI is local or missing. "
            "Set MLFLOW_TRACKING_URI in .env or pass --mlflow-tracking-uri."
        )

    required_env_vars = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    )

    missing = [
        name
        for name in required_env_vars
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing AWS environment variables required to download/upload "
            f"MLflow artifacts from S3: {missing}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Vulcadata classification state_dict checkpoint "
            "as a loadable MLflow PyTorch model."
        )
    )

    parser.add_argument(
        "--source-run-id",
        required=True,
        help="MLflow run id containing the raw PyTorch checkpoint artifact.",
    )
    parser.add_argument(
        "--checkpoint-artifact",
        default="best_cnn_transformer_classifier.pt",
        help="Checkpoint artifact path inside the source MLflow run.",
    )
    parser.add_argument(
        "--model-class",
        required=True,
        help=(
            "Import path of the model class. "
            "Example: scripts.training.train_cnn_transformer_classif:"
            "CNNTransformerClassifier"
        ),
    )
    parser.add_argument(
        "--decision-path",
        default="configs/final_model_decision.json",
        help="Input final model decision JSON.",
    )
    parser.add_argument(
        "--output-decision-path",
        default="configs/final_model_decision_exported.json",
        help="Output decision JSON pointing to the exported MLflow model.",
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path for the exported MLflow model.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Vulcadata",
        help="MLflow experiment name for the export run.",
    )
    parser.add_argument(
        "--input-npz",
        default="data/preprocessing/processed_full_stride5_with_quiet/volcano_multi.npz",
        help="NPZ file used to build a small MLflow input example.",
    )
    parser.add_argument(
        "--array-key",
        default="X_test",
        help="Input array key inside the NPZ file.",
    )
    parser.add_argument(
        "--input-example-size",
        type=int,
        default=2,
        help="Number of sequences used as MLflow input example.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI override.",
    )
    parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Path to the local .env file.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_project_dotenv(args.dotenv_path)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    validate_environment()

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    model_class = import_model_class(args.model_class)

    client = mlflow.tracking.MlflowClient()

    checkpoint_path = client.download_artifacts(
        run_id=args.source_run_id,
        path=args.checkpoint_artifact,
        dst_path="models/mlflow_downloads",
    )

    print(f"Downloaded checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "This script expects a checkpoint dictionary containing model_state_dict."
        )

    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix_if_needed(state_dict)

    model = build_model_from_checkpoint(
        model_class=model_class,
        checkpoint=checkpoint,
    )

    model.load_state_dict(state_dict)
    model.eval()

    input_example = ensure_input_example(
        input_npz=args.input_npz,
        array_key=args.array_key,
        input_example_size=args.input_example_size,
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(
        run_name="export_cnn_transformer_classifier_mlflow_model"
    ) as run:
        export_run_id = run.info.run_id

        mlflow.log_param("source_training_run_id", args.source_run_id)
        mlflow.log_param("checkpoint_artifact", args.checkpoint_artifact)
        mlflow.log_param("export_artifact_path", args.artifact_path)
        mlflow.log_param("model_class", args.model_class)
        mlflow.log_param("input_npz", str(args.input_npz))
        mlflow.log_param("array_key", args.array_key)
        mlflow.log_param("input_example_size", args.input_example_size)

        log_checkpoint_params(checkpoint)

        mlflow.set_tag("export_type", "classification_model_mlflow_export")
        mlflow.set_tag("source_run_id", args.source_run_id)
        mlflow.set_tag("model_task", "operational_alert_24h")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=args.artifact_path,
            input_example=input_example,
            pip_requirements=[
                "mlflow",
                "torch",
                "numpy",
            ],
        )

    update_final_model_decision(
        decision_path=args.decision_path,
        output_path=args.output_decision_path,
        export_run_id=export_run_id,
        source_run_id=args.source_run_id,
        artifact_path=args.artifact_path,
        checkpoint_artifact=args.checkpoint_artifact,
        model_class=args.model_class,
    )

    print(f"Export run id: {export_run_id}")
    print(f"Model URI: runs:/{export_run_id}/{args.artifact_path}")
    print(f"Updated decision file: {args.output_decision_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())