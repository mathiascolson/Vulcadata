import argparse
import json
import os
import random
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv(override=False)


# ============================================================
# CONSTANTES CLASSIFICATION
# ============================================================

DEFAULT_CLASS_NAMES = [
    "quiet_non_eruptive",
    "36_48h_before_eruption",
    "24_36h_before_eruption",
    "12_24h_before_eruption",
    "6_12h_before_eruption",
    "0_6h_before_eruption",
]


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)

    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"URI S3 invalide : {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not key:
        raise ValueError(f"URI S3 sans key : {s3_uri}")

    return bucket, key


def get_s3_client():
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")

    if region:
        return boto3.client("s3", region_name=region)

    return boto3.client("s3")


def download_from_s3_if_needed(input_path: str, local_dir: Path) -> Path:
    if not is_s3_uri(input_path):
        local_path = Path(input_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {local_path}")

        return local_path

    bucket, key = parse_s3_uri(input_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(key).name

    print(f"Téléchargement S3 : {input_path}")
    print(f"Vers local        : {local_path}")

    s3_client = get_s3_client()
    s3_client.download_file(bucket, key, str(local_path))

    if not local_path.exists():
        raise FileNotFoundError(f"Téléchargement S3 échoué : {local_path}")

    return local_path


def upload_file_to_s3(local_path: Path, s3_uri: str) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Fichier local introuvable : {local_path}")

    bucket, key = parse_s3_uri(s3_uri)

    print(f"Upload S3 : {local_path} → {s3_uri}")

    s3_client = get_s3_client()
    s3_client.upload_file(str(local_path), bucket, key)


def load_npz_dataset(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    required_keys = [
        "X_train", "y_train",
        "X_val", "y_val",
        "X_test", "y_test",
    ]

    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        raise KeyError(
            f"Clés absentes du fichier NPZ : {missing_keys}. "
            f"Clés disponibles : {list(data.keys())}"
        )

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    feature_names = data["feature_names"] if "feature_names" in data else None

    class_names = (
        [str(x) for x in data["class_names"].tolist()]
        if "class_names" in data
        else DEFAULT_CLASS_NAMES
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, class_names


def validate_classification_targets(y_train, y_val, y_test, n_classes: int) -> None:
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if y.ndim != 1:
            raise ValueError(f"y_{split_name} doit être de forme (N,), reçu : {y.shape}")

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"y_{split_name} doit être entier pour CrossEntropyLoss.")

        unique = np.unique(y)
        bad = unique[(unique < 0) | (unique >= n_classes)]

        if bad.size > 0:
            raise ValueError(
                f"Classes invalides dans y_{split_name} : {bad.tolist()} ; "
                f"classes attendues : 0 à {n_classes - 1}."
            )


def compute_class_weights(y_train: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)

    if np.any(counts == 0):
        print(
            "WARNING : au moins une classe est absente du train. "
            "Poids fixé à 0 pour les classes absentes."
        )

    if mode == "balanced":
        weights = np.zeros(n_classes, dtype=np.float32)
        non_zero = counts > 0
        weights[non_zero] = counts.sum() / (n_classes * counts[non_zero])
        return torch.tensor(weights, dtype=torch.float32)

    if mode == "alert_priority":
        weights = np.zeros(n_classes, dtype=np.float32)
        non_zero = counts > 0
        weights[non_zero] = counts.sum() / (n_classes * counts[non_zero])

        # Accent métier : classes proches de l'éruption plus coûteuses.
        horizon_multiplier = np.array([1.0, 1.2, 1.4, 1.8, 2.2, 2.8], dtype=np.float32)
        weights = weights * horizon_multiplier
        return torch.tensor(weights, dtype=torch.float32)

    if mode == "early_warning_priority":
        weights = np.zeros(n_classes, dtype=np.float32)
        non_zero = counts > 0
        weights[non_zero] = counts.sum() / (n_classes * counts[non_zero])

        # Accent métier : préserver l'anticipation précoce.
        # Classe 0 = calme, classes 1 et 2 = horizons longs les plus difficiles.
        horizon_multiplier = np.array([1.5, 2.0, 1.8, 1.5, 1.3, 1.2], dtype=np.float32)
        weights = weights * horizon_multiplier
        return torch.tensor(weights, dtype=torch.float32)

    raise ValueError(f"class_weighting invalide : {mode}")


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)


def classification_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)

    y_pred = logits.argmax(axis=1)
    proba = softmax_numpy(logits)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Alerte large : classe 0 = calme ; classes 1..5 = pré-éruptif.
    y_true_alert = (y_true > 0).astype(np.int64)
    y_pred_alert = (y_pred > 0).astype(np.int64)

    metrics.update({
        "alert_precision": float(precision_score(y_true_alert, y_pred_alert, zero_division=0)),
        "alert_recall": float(recall_score(y_true_alert, y_pred_alert, zero_division=0)),
        "alert_f1": float(f1_score(y_true_alert, y_pred_alert, zero_division=0)),
    })

    # Alertes métier par horizon :
    # 24h = classes 3,4,5 ; 12h = classes 4,5 ; 6h = classe 5.
    for label, min_class in [
        ("alert_24h", 3),
        ("alert_12h", 4),
        ("alert_6h", 5),
    ]:
        y_true_h = (y_true >= min_class).astype(np.int64)
        y_pred_h = (y_pred >= min_class).astype(np.int64)

        metrics[f"{label}_precision"] = float(precision_score(y_true_h, y_pred_h, zero_division=0))
        metrics[f"{label}_recall"] = float(recall_score(y_true_h, y_pred_h, zero_division=0))
        metrics[f"{label}_f1"] = float(f1_score(y_true_h, y_pred_h, zero_division=0))

    # Métriques par classe.
    per_class_precision = precision_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )

    for cls in range(n_classes):
        metrics[f"class_{cls}_precision"] = float(per_class_precision[cls])
        metrics[f"class_{cls}_recall"] = float(per_class_recall[cls])
        metrics[f"class_{cls}_f1"] = float(per_class_f1[cls])
        metrics[f"class_{cls}_support"] = int((y_true == cls).sum())

    # Score métier classification à maximiser.
    # Il conserve les 6 classes, mais valorise explicitement :
    # - la classe calme pour éviter l'alerte permanente ;
    # - les classes 1 et 2 pour préserver l'anticipation précoce ;
    # - l'alerte 24h pour garder une lecture opérationnelle.
    metrics["business_score_classification"] = float(
        0.15 * metrics.get("macro_f1", 0.0)
        + 0.15 * metrics.get("balanced_accuracy", 0.0)
        + 0.15 * metrics.get("class_0_f1", 0.0)
        + 0.20 * metrics.get("class_1_f1", 0.0)
        + 0.20 * metrics.get("class_2_f1", 0.0)
        + 0.15 * metrics.get("alert_24h_f1", 0.0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return metrics, y_pred.astype(np.int64), proba, cm


def get_early_stopping_value(metrics: dict, metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(
            f"Métrique d'early stopping absente : {metric_name}. "
            f"Métriques disponibles : {sorted(metrics.keys())}"
        )
    return float(metrics[metric_name])


def metric_mode(metric_name: str) -> str:
    if metric_name in {"loss"} or metric_name.endswith("_loss"):
        return "min"
    return "max"


def is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return (best - current) > min_delta
    return (current - best) > min_delta


# ============================================================
# DATASET
# ============================================================

class VolcanoSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"X doit avoir la forme (N, T, F), reçu : {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y doit avoir la forme (N,), reçu : {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Nombre d'observations incohérent : X={X.shape[0]}, y={y.shape[0]}"
            )

        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ============================================================
# COMMON MODULES
# ============================================================

class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = float(std)

    def forward(self, x):
        if self.training and self.std > 0.0:
            return x + torch.randn_like(x) * self.std
        return x


# ============================================================
# MODEL
# ============================================================

class CNNTransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_classes: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.20,
        input_noise_std: float = 0.01,
    ):
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

        pooled = x.mean(dim=1)
        logits = self.head(pooled)
        return logits


MODEL_TYPE = "CNNTransformerClassifier"
DEFAULT_OUTPUT_DIR = "models/cnn_transformer_classifier"
DEFAULT_RUN_NAME = "cnn_transformer_volcano_multiclass"


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip: float,
    use_amp: bool,
):
    model.train()

    running_loss = 0.0
    n_samples = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, n_classes: int):
    model.eval()

    running_loss = 0.0
    n_samples = 0

    logits_all = []
    y_all = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(y_all)
    logits = np.concatenate(logits_all)

    metrics, y_pred, proba, cm = classification_metrics(
        y_true=y_true,
        logits=logits,
        n_classes=n_classes,
    )
    metrics["loss"] = float(running_loss / max(n_samples, 1))

    return metrics, y_true, y_pred, proba, logits, cm


def try_setup_mlflow(args):
    if not args.use_mlflow:
        return None

    try:
        import mlflow
    except ImportError:
        print("MLflow non installé. Entraînement sans tracking MLflow.")
        return None

    tracking_uri = (
        args.mlflow_tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
    )

    experiment_name = (
        args.mlflow_experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "Vulcadata"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    return mlflow


def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_cache_dir = output_dir / "input_cache"
    npz_path = download_from_s3_if_needed(args.input_npz, local_cache_dir)

    print(f"Chargement dataset : {npz_path}")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, class_names = load_npz_dataset(npz_path)

    n_classes = int(args.n_classes)

    validate_classification_targets(y_train, y_val, y_test, n_classes=n_classes)

    print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape} | y_val   : {y_val.shape}")
    print(f"X_test  : {X_test.shape} | y_test  : {y_test.shape}")
    print(f"Classes train : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Classes val   : {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"Classes test  : {dict(zip(*np.unique(y_test, return_counts=True)))}")

    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        raise ValueError("NaN détectés dans X après preprocessing.")

    if np.isinf(X_train).any() or np.isinf(X_val).any() or np.isinf(X_test).any():
        raise ValueError("Inf détectés dans X après preprocessing.")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    train_dataset = VolcanoSequenceDataset(X_train, y_train)
    val_dataset = VolcanoSequenceDataset(X_val, y_val)
    test_dataset = VolcanoSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = args.use_amp and device.type == "cuda"

    print(f"Device : {device}")
    print(f"AMP    : {use_amp}")

    model = build_model_from_args(
        args=args,
        n_features=n_features,
        seq_len=seq_len,
        n_classes=n_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Nombre de paramètres total        : {n_params:,}")
    print(f"Nombre de paramètres entraînables : {n_trainable_params:,}")

    class_weights = compute_class_weights(
        y_train=y_train,
        n_classes=n_classes,
        mode=args.class_weighting,
    )

    if class_weights is not None:
        print(f"Poids de classes : {class_weights.numpy().round(4).tolist()}")
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    early_metric = args.early_stopping_metric
    early_mode = metric_mode(early_metric)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=early_mode,
        factor=0.5,
        patience=max(1, args.lr_patience),
    )

    mlflow = try_setup_mlflow(args)

    best_val_score = float("inf") if early_mode == "min" else -float("inf")
    best_model_path = output_dir / args.best_model_name
    history = []
    patience_counter = 0

    run_context = mlflow.start_run(run_name=args.run_name) if mlflow else None

    try:
        if mlflow:
            mlflow.log_params({
                "model_type": MODEL_TYPE,
                "task": "multiclass_classification",
                "target": "multiclass_eruption_horizon",
                "n_classes": n_classes,
                "class_names": json.dumps(class_names, ensure_ascii=False),
                "input_npz": args.input_npz,
                "n_features": int(n_features),
                "seq_len": int(seq_len),
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "loss": "CrossEntropyLoss",
                "class_weighting": args.class_weighting,
                "label_smoothing": args.label_smoothing,
                "early_stopping_metric": early_metric,
                "early_stopping_mode": early_mode,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "lr_patience": args.lr_patience,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "n_params": int(n_params),
                "n_trainable_params": int(n_trainable_params),
                **model_params_for_mlflow(args),
            })

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
            )

            val_metrics, _, _, _, _, _ = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                n_classes=n_classes,
            )

            current_score = get_early_stopping_value(val_metrics, early_metric)
            scheduler.step(current_score)

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            history.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"val_alert_f1={val_metrics['alert_f1']:.4f} | "
                f"val_alert_24h_f1={val_metrics['alert_24h_f1']:.4f}"
            )

            if mlflow:
                mlflow.log_metrics(row, step=epoch)

            if is_improvement(
                current=current_score,
                best=best_val_score,
                mode=early_mode,
                min_delta=args.early_stopping_min_delta,
            ):
                best_val_score = current_score
                patience_counter = 0

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": MODEL_TYPE,
                    "task": "multiclass_classification",
                    "n_classes": int(n_classes),
                    "class_names": class_names,
                    "n_features": int(n_features),
                    "seq_len": int(seq_len),
                    "args": vars(args),
                    "best_val_score": float(best_val_score),
                    "best_val_metric": early_metric,
                    "feature_names": (
                        feature_names.tolist()
                        if hasattr(feature_names, "tolist")
                        else feature_names
                    ),
                    **checkpoint_extra(args),
                }

                torch.save(checkpoint, best_model_path)
                print(
                    f"  Nouveau meilleur modèle sauvegardé : {best_model_path} | "
                    f"{early_metric}={best_val_score:.6f}"
                )
            else:
                patience_counter += 1
                print(
                    f"  Pas d'amélioration ({patience_counter}/{args.early_stopping_patience})"
                )

            if patience_counter >= args.early_stopping_patience:
                print(
                    f"Early stopping déclenché après {args.early_stopping_patience} "
                    f"epochs sans amélioration."
                )
                break

        print(f"\nChargement du meilleur modèle : {best_model_path}")
        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, y_true_test, y_pred_test, y_proba_test, logits_test, cm_test = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            n_classes=n_classes,
        )

        print("\nMétriques TEST")
        for key, value in test_metrics.items():
            if isinstance(value, (int, np.integer)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {float(value):.6f}")

        print("\nMatrice de confusion TEST")
        print(cm_test)

        history_path = output_dir / args.history_name
        metrics_path = output_dir / args.metrics_name
        predictions_path = output_dir / args.predictions_name
        confusion_path = output_dir / args.confusion_name

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_metric": early_metric,
                    "best_val_score": float(best_val_score),
                    "test": test_metrics,
                    "n_params": int(n_params),
                    "n_trainable_params": int(n_trainable_params),
                    "class_names": class_names,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        np.savez_compressed(
            predictions_path,
            y_true=y_true_test.astype(np.int64),
            y_pred=y_pred_test.astype(np.int64),
            y_proba=y_proba_test.astype(np.float32),
            logits=logits_test.astype(np.float32),
        )

        np.save(confusion_path, cm_test.astype(np.int64))

        if mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric(f"best_val_{early_metric}", best_val_score)
            mlflow.log_metric("n_params", n_params)
            mlflow.log_metric("n_trainable_params", n_trainable_params)

            mlflow.log_artifact(str(best_model_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(confusion_path))

        if args.s3_output_prefix:
            prefix = args.s3_output_prefix.strip().strip("/")
            bucket = args.s3_bucket or os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET_NAME")

            if not bucket:
                raise ValueError(
                    "--s3-output-prefix est fourni, mais aucun bucket S3 n'est disponible."
                )

            for local_path in [
                best_model_path,
                history_path,
                metrics_path,
                predictions_path,
                confusion_path,
            ]:
                s3_uri = f"s3://{bucket}/{prefix}/{local_path.name}"
                upload_file_to_s3(local_path, s3_uri)

    finally:
        if run_context is not None:
            mlflow.end_run()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-npz",
        type=str,
        default="data/preprocessing/processed_with_quiet/volcano_multi.npz",
        help="Chemin local ou URI S3 vers volcano_multi.npz.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie local.",
    )

    parser.add_argument("--s3-bucket", type=str, default=None)

    parser.add_argument(
        "--s3-output-prefix",
        type=str,
        default="",
        help="Préfixe S3 pour uploader les artefacts modèle. Vide = pas d'upload.",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-metric", type=str, default="business_score_classification")
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--class-weighting",
        type=str,
        default="balanced",
        choices=["none", "balanced", "alert_priority", "early_warning_priority"],
        help="Pondération des classes pour CrossEntropyLoss.",
    )

    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--n-classes", type=int, default=6)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-amp", action="store_true")

    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)


def build_model_from_args(args, n_features: int, seq_len: int, n_classes: int):
    return CNNTransformerClassifier(
        feature_dim=n_features,
        n_classes=n_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        input_noise_std=args.input_noise_std,
    )


def model_params_for_mlflow(args) -> dict:
    return {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "input_noise_std": args.input_noise_std,
    }


def checkpoint_extra(args) -> dict:
    return {
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "num_layers": int(args.num_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
        "input_noise_std": float(args.input_noise_std),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.set_defaults(
        best_model_name="best_cnn_transformer_classifier.pt",
        history_name="history_cnn_transformer_classifier.json",
        metrics_name="metrics_cnn_transformer_classifier.json",
        predictions_name="predictions_cnn_transformer_classifier.npz",
        confusion_name="confusion_matrix_cnn_transformer_classifier.npy",
    )

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--input-noise-std", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
