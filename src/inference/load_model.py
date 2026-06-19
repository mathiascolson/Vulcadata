from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ModelFlavor = Literal["pyfunc", "pytorch", "pytorch_checkpoint"]

SUPPORTED_MODEL_FLAVORS = {"pyfunc", "pytorch", "pytorch_checkpoint"}


@dataclass(frozen=True)
class ModelReference:
    """
    Reference to a model artifact or MLflow model URI.

    This object does not load the model by itself. It only stores the resolved
    location and minimal metadata needed by inference code.
    """

    uri: str
    flavor: ModelFlavor = "pyfunc"
    source_type: str = "unknown"
    model_name: str | None = None
    model_version: str | None = None
    model_stage: str | None = None
    model_alias: str | None = None
    run_id: str | None = None
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", validate_non_empty_string(self.uri, "uri"))
        object.__setattr__(self, "flavor", validate_model_flavor(self.flavor))
        object.__setattr__(
            self,
            "source_type",
            validate_non_empty_string(self.source_type, "source_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "flavor": self.flavor,
            "source_type": self.source_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_stage": self.model_stage,
            "model_alias": self.model_alias,
            "run_id": self.run_id,
            "artifact_path": self.artifact_path,
            "metadata": self.metadata,
        }


def validate_non_empty_string(value: Any, field_name: str) -> str:
    """
    Validate that a value is a non-empty string.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(f"{field_name} cannot be empty.")

    return cleaned_value


def validate_model_flavor(flavor: Any) -> ModelFlavor:
    """
    Validate supported model loading flavor.
    """
    cleaned_flavor = validate_non_empty_string(flavor, "flavor")

    if cleaned_flavor not in SUPPORTED_MODEL_FLAVORS:
        raise ValueError(
            f"Unsupported model flavor: {cleaned_flavor!r}. "
            f"Supported flavors are: {sorted(SUPPORTED_MODEL_FLAVORS)}."
        )

    return cleaned_flavor  # type: ignore[return-value]


def build_registered_model_uri(
    *,
    model_name: str,
    model_version: str | int | None = None,
    model_stage: str | None = None,
    model_alias: str | None = None,
) -> str:
    """
    Build an MLflow registered model URI.

    Supported forms:
    - models:/model_name/version
    - models:/model_name/stage
    - models:/model_name@alias
    """
    cleaned_model_name = validate_non_empty_string(model_name, "model_name")

    selectors = [
        model_version is not None,
        model_stage is not None,
        model_alias is not None,
    ]

    if sum(selectors) != 1:
        raise ValueError(
            "Exactly one of model_version, model_stage or model_alias must be provided."
        )

    if model_alias is not None:
        cleaned_alias = validate_non_empty_string(model_alias, "model_alias")
        return f"models:/{cleaned_model_name}@{cleaned_alias}"

    if model_version is not None:
        cleaned_version = validate_non_empty_string(str(model_version), "model_version")
        return f"models:/{cleaned_model_name}/{cleaned_version}"

    cleaned_stage = validate_non_empty_string(model_stage, "model_stage")
    return f"models:/{cleaned_model_name}/{cleaned_stage}"


def build_runs_model_uri(*, run_id: str, artifact_path: str) -> str:
    """
    Build an MLflow run artifact URI.

    Example:
    runs:/abc123/model
    """
    cleaned_run_id = validate_non_empty_string(run_id, "run_id")
    cleaned_artifact_path = validate_non_empty_string(
        artifact_path,
        "artifact_path",
    ).strip("/")

    if not cleaned_artifact_path:
        raise ValueError("artifact_path cannot be empty.")

    return f"runs:/{cleaned_run_id}/{cleaned_artifact_path}"


def read_model_decision(decision_path: str | Path) -> dict[str, Any]:
    """
    Read a final model decision JSON file.
    """
    path = Path(decision_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model decision file does not exist: {path}")

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON model decision file: {path}") from exc

    if not isinstance(content, dict):
        raise ValueError("Model decision file must contain a JSON object.")

    return content


def infer_model_reference(
    decision: Mapping[str, Any],
    *,
    default_flavor: ModelFlavor = "pyfunc",
    default_artifact_path: str = "model",
) -> ModelReference:
    """
    Infer a ModelReference from a model decision dictionary.

    Supported priority:
    1. Direct model URI at root level:
       model_uri / mlflow_model_uri / registered_model_uri / source_uri

    2. Declared operational candidate:
       decision.main_operational_model = "classification_candidate"
       then use the corresponding nested block.

    3. Direct model URI in selected scope:
       model_uri / mlflow_model_uri / registered_model_uri / source_uri

    4. Registered MLflow model:
       model_name + model_version
       model_name + model_stage
       model_name + model_alias

    5. MLflow run artifact:
       run_id + artifact_path

       If artifact_path is absent but run_id is present, default_artifact_path
       is used. The project convention is "model".

    6. Local or generic model path:
       model_path / local_model_path / path
    """
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping.")

    root_direct_uri = _optional_string(
        _find_first_value_top_level(
            decision,
            keys=("model_uri", "mlflow_model_uri", "registered_model_uri", "source_uri"),
            default=None,
        )
    )

    if root_direct_uri is not None:
        active_decision = decision
        selected_candidate_key = None
    else:
        active_decision, selected_candidate_key = resolve_operational_model_decision(decision)

    flavor = validate_model_flavor(
        _find_first_value(
            active_decision,
            keys=("flavor", "model_flavor", "mlflow_flavor"),
            default=default_flavor,
        )
    )

    model_name = _optional_string(
        _find_first_value(
            active_decision,
            keys=(
                "model_name",
                "registered_model_name",
                "model_family",
                "selection_name",
            ),
            default=None,
        )
    )
    model_version = _optional_string(
        _find_first_value(
            active_decision,
            keys=("model_version", "registered_model_version", "champion_version"),
            default=None,
        )
    )
    model_stage = _optional_string(
        _find_first_value(
            active_decision,
            keys=("model_stage", "stage"),
            default=None,
        )
    )
    model_alias = _optional_string(
        _find_first_value(
            active_decision,
            keys=("model_alias", "alias"),
            default=None,
        )
    )
    run_id = _optional_string(
        _find_first_value(
            active_decision,
            keys=("run_id", "mlflow_run_id", "best_run_id"),
            default=None,
        )
    )
    artifact_path = _optional_string(
        _find_first_value(
            active_decision,
            keys=("artifact_path", "model_artifact_path", "mlflow_artifact_path"),
            default=None,
        )
    )

    if run_id is not None and artifact_path is None:
        artifact_path = validate_non_empty_string(
            default_artifact_path,
            "default_artifact_path",
        )

    direct_uri = root_direct_uri or _optional_string(
        _find_first_value(
            active_decision,
            keys=("model_uri", "mlflow_model_uri", "registered_model_uri", "source_uri"),
            default=None,
        )
    )

    metadata = dict(decision)

    if selected_candidate_key is not None:
        metadata["selected_candidate_key"] = selected_candidate_key

    if direct_uri is not None:
        return ModelReference(
            uri=direct_uri,
            flavor=flavor,
            source_type=_infer_source_type_from_uri(direct_uri),
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage,
            model_alias=model_alias,
            run_id=run_id,
            artifact_path=artifact_path,
            metadata=metadata,
        )

    if model_name is not None and any(
        value is not None for value in (model_version, model_stage, model_alias)
    ):
        uri = build_registered_model_uri(
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage,
            model_alias=model_alias,
        )

        return ModelReference(
            uri=uri,
            flavor=flavor,
            source_type="mlflow_registered_model",
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage,
            model_alias=model_alias,
            run_id=run_id,
            artifact_path=artifact_path,
            metadata=metadata,
        )

    if run_id is not None and artifact_path is not None:
        uri = build_runs_model_uri(run_id=run_id, artifact_path=artifact_path)

        return ModelReference(
            uri=uri,
            flavor=flavor,
            source_type="mlflow_run_artifact",
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage,
            model_alias=model_alias,
            run_id=run_id,
            artifact_path=artifact_path,
            metadata=metadata,
        )

    model_path = _optional_string(
        _find_first_value(
            active_decision,
            keys=("model_path", "local_model_path", "path"),
            default=None,
        )
    )

    if model_path is not None:
        return ModelReference(
            uri=model_path,
            flavor=flavor,
            source_type="local_or_generic_path",
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage,
            model_alias=model_alias,
            run_id=run_id,
            artifact_path=artifact_path,
            metadata=metadata,
        )

    raise ValueError(
        "Could not infer model reference. Provide one of: "
        "model_uri, model_name + model_version/model_stage/model_alias, "
        "run_id + artifact_path, run_id with default_artifact_path, "
        "or model_path."
    )


def resolve_operational_model_decision(
    decision: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    """
    Resolve the operational model block from a final model decision structure.

    Supported project format:

    {
      "regression_candidate": {...},
      "classification_candidate": {...},
      "decision": {
        "main_operational_model": "classification_candidate"
      }
    }

    If no main operational model is declared, the original decision mapping is
    returned. This preserves compatibility with simpler decision files.
    """
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping.")

    selected_candidate_key: str | None = None

    decision_block = decision.get("decision")

    if isinstance(decision_block, Mapping):
        selected_candidate_key = _optional_string(
            decision_block.get("main_operational_model")
        )

    if selected_candidate_key is None:
        selected_candidate_key = _optional_string(
            decision.get("main_operational_model")
        )

    if selected_candidate_key is None:
        return decision, None

    selected_candidate = decision.get(selected_candidate_key)

    if not isinstance(selected_candidate, Mapping):
        raise ValueError(
            "decision.main_operational_model points to "
            f"{selected_candidate_key!r}, but this candidate block is missing "
            "or is not a mapping."
        )

    return selected_candidate, selected_candidate_key


def load_model(
    reference: ModelReference,
    *,
    flavor: ModelFlavor | None = None,
    loader: Callable[[str], Any] | None = None,
) -> Any:
    """
    Load a model from a ModelReference.

    In tests, pass an injected loader to avoid importing MLflow or connecting
    to a remote tracking server.

    In production:
    - flavor='pyfunc' uses mlflow.pyfunc.load_model
    - flavor='pytorch' uses mlflow.pytorch.load_model
    """
    if not isinstance(reference, ModelReference):
        raise TypeError("reference must be a ModelReference instance.")

    selected_flavor = validate_model_flavor(flavor or reference.flavor)

    if loader is not None:
        return loader(reference.uri)

    if selected_flavor == "pyfunc":
        try:
            import mlflow.pyfunc
        except ImportError as exc:
            raise RuntimeError(
                "mlflow is required to load a pyfunc model. "
                "Install mlflow or inject a loader."
            ) from exc

        return mlflow.pyfunc.load_model(reference.uri)

    if selected_flavor == "pytorch":
        try:
            import mlflow.pytorch
        except ImportError as exc:
            raise RuntimeError(
                "mlflow is required to load a PyTorch model. "
                "Install mlflow or inject a loader."
            ) from exc

        return mlflow.pytorch.load_model(reference.uri)

    if selected_flavor == "pytorch_checkpoint":
        return load_pytorch_checkpoint_model(reference)

    raise ValueError(f"Unsupported model flavor: {selected_flavor!r}.")


def load_model_from_decision_file(
    decision_path: str | Path,
    *,
    default_flavor: ModelFlavor = "pyfunc",
    flavor: ModelFlavor | None = None,
    loader: Callable[[str], Any] | None = None,
) -> Any:
    """
    Read a model decision JSON file, infer the model reference, then load it.
    """
    decision = read_model_decision(decision_path)

    reference = infer_model_reference(
        decision,
        default_flavor=default_flavor,
    )

    return load_model(
        reference,
        flavor=flavor,
        loader=loader,
    )


def _infer_source_type_from_uri(uri: str) -> str:
    cleaned_uri = validate_non_empty_string(uri, "uri")

    if cleaned_uri.startswith("models:/"):
        return "mlflow_registered_model"

    if cleaned_uri.startswith("runs:/"):
        return "mlflow_run_artifact"

    if cleaned_uri.startswith("s3://"):
        return "s3_model_artifact"

    return "local_or_generic_path"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    return validate_non_empty_string(str(value), "value")


def _find_first_value_top_level(
    mapping: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    """
    Find the first matching key at the current mapping level only.

    This avoids accidentally selecting a nested regression candidate before
    the declared operational candidate.
    """
    for key in keys:
        if key in mapping:
            return mapping[key]

    return default


def _find_first_value(
    mapping: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    """
    Find the first matching key in a nested mapping.

    Top-level keys have priority. Nested mappings are searched second.
    """
    for key in keys:
        if key in mapping:
            return mapping[key]

    for value in mapping.values():
        if isinstance(value, Mapping):
            nested_value = _find_first_value(
                value,
                keys=keys,
                default=None,
            )

            if nested_value is not None:
                return nested_value

    return default
def load_pytorch_checkpoint_model(reference: ModelReference) -> Any:
    """
    Load a direct CNN-Transformer PyTorch checkpoint from S3 or local storage.
    """
    import tempfile
    from pathlib import Path
    from urllib.parse import urlparse

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required to load a PyTorch checkpoint.") from exc

    active_metadata = _get_active_model_metadata(reference)

    feature_dim = int(active_metadata.get("feature_dim", 992))
    n_classes = int(active_metadata.get("n_classes", 6))
    d_model = int(active_metadata.get("d_model", 96))
    nhead = int(active_metadata.get("nhead", 4))
    num_layers = int(active_metadata.get("num_layers", 3))
    dim_feedforward = int(active_metadata.get("dim_feedforward", 256))
    dropout = float(active_metadata.get("dropout", 0.20))
    input_noise_std = float(active_metadata.get("input_noise_std", 0.01))

    class GaussianNoise(nn.Module):
        def __init__(self, std: float = 0.0) -> None:
            super().__init__()
            self.std = float(std)

        def forward(self, x):
            if self.training and self.std > 0.0:
                return x + torch.randn_like(x) * self.std
            return x

    class CNNTransformerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.input_noise = GaussianNoise(std=input_noise_std)

            self.conv = nn.Sequential(
                nn.Conv1d(feature_dim, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            x = self.input_noise(x)
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.head(x)

    class TorchCheckpointModelWrapper(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, model_input):
            tensor = _to_float_tensor(
                model_input,
                feature_dim=feature_dim,
                torch_module=torch,
            )
            return self.model(tensor)

        def predict(self, model_input):
            self.eval()
            with torch.no_grad():
                logits = self.forward(model_input)
            return logits.detach().cpu().numpy()

    with tempfile.TemporaryDirectory(prefix="vulcadata_checkpoint_") as temporary_directory:
        checkpoint_path = _resolve_checkpoint_path(
            uri=reference.uri,
            temporary_directory=Path(temporary_directory),
            urlparse_function=urlparse,
        )
        checkpoint = _torch_load_checkpoint(torch, checkpoint_path)

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)

    model = CNNTransformerClassifier()
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = TorchCheckpointModelWrapper(model)
    wrapper.eval()

    return wrapper


def _get_active_model_metadata(reference: ModelReference) -> Mapping[str, Any]:
    metadata = reference.metadata
    selected_candidate_key = metadata.get("selected_candidate_key")

    if isinstance(selected_candidate_key, str):
        selected_candidate = metadata.get(selected_candidate_key)

        if isinstance(selected_candidate, Mapping):
            return selected_candidate

    return metadata


def _resolve_checkpoint_path(
    *,
    uri: str,
    temporary_directory: Path,
    urlparse_function: Any,
) -> Path:
    cleaned_uri = validate_non_empty_string(uri, "uri")

    if cleaned_uri.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required to download a checkpoint from S3.") from exc

        parsed_uri = urlparse_function(cleaned_uri)
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")

        if not bucket or not key:
            raise ValueError(f"Invalid S3 checkpoint URI: {cleaned_uri}")

        checkpoint_path = temporary_directory / Path(key).name
        boto3.client("s3").download_file(bucket, key, str(checkpoint_path))
        return checkpoint_path

    if cleaned_uri.startswith("file://"):
        parsed_uri = urlparse_function(cleaned_uri)
        checkpoint_path = Path(parsed_uri.path)
    else:
        checkpoint_path = Path(cleaned_uri)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    return checkpoint_path


def _torch_load_checkpoint(torch_module: Any, checkpoint_path: Path) -> Any:
    try:
        return torch_module.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch_module.load(
            checkpoint_path,
            map_location="cpu",
        )


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)

            if isinstance(value, Mapping):
                return value

        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or a mapping "
        "containing model_state_dict/state_dict/model."
    )


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cleaned_state_dict = dict(state_dict)

    prefixes = ("module.", "model.")

    for prefix in prefixes:
        if cleaned_state_dict and all(key.startswith(prefix) for key in cleaned_state_dict):
            cleaned_state_dict = {
                key[len(prefix):]: value
                for key, value in cleaned_state_dict.items()
            }

    return cleaned_state_dict


def _to_float_tensor(model_input: Any, *, feature_dim: int, torch_module: Any) -> Any:
    import numpy as np

    if isinstance(model_input, torch_module.Tensor):
        tensor = model_input.detach().to(dtype=torch_module.float32, device="cpu")

        if tensor.ndim != 3:
            raise ValueError(
                f"Expected tensor input with 3 dimensions, got shape {tuple(tensor.shape)}."
            )

        if tensor.shape[-1] != feature_dim:
            raise ValueError(
                f"Expected last tensor dimension {feature_dim}, got shape {tuple(tensor.shape)}."
            )

        return tensor

    if hasattr(model_input, "to_numpy"):
        array = model_input.to_numpy(dtype=np.float32)
    else:
        array = np.asarray(model_input, dtype=np.float32)

    if array.ndim == 2:
        if array.shape[-1] == feature_dim:
            array = array.reshape(1, array.shape[0], array.shape[1])
        elif array.shape[1] % feature_dim == 0:
            array = array.reshape(array.shape[0], array.shape[1] // feature_dim, feature_dim)
        else:
            raise ValueError(f"Cannot reshape 2D input with shape {array.shape}.")

    if array.ndim != 3:
        raise ValueError(f"Expected input with 3 dimensions, got shape {array.shape}.")

    if array.shape[-1] != feature_dim:
        raise ValueError(f"Expected last input dimension {feature_dim}, got shape {array.shape}.")

    return torch_module.from_numpy(array.astype(np.float32, copy=False))

