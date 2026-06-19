# scripts/train_cnn_bilstm_regression.py

import argparse
import json
import math
import os
import random
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv(override=False)


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
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Clés absentes du NPZ : {missing}. Clés disponibles : {list(data.keys())}")

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    feature_names = data["feature_names"] if "feature_names" in data else None

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def validate_regression_targets(y_train, y_val, y_test, max_horizon_hours: float) -> None:
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if y.ndim != 1:
            raise ValueError(f"y_{split_name} doit être de forme (N,), reçu : {y.shape}")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError(f"NaN ou Inf détectés dans y_{split_name}.")
        if (y < 0).any():
            raise ValueError(f"Valeur négative détectée dans y_{split_name}. Régression attendue en heures avant éruption.")
        if max_horizon_hours > 0 and (y > max_horizon_hours + 1e-6).any():
            print(
                f"WARNING : y_{split_name} contient des valeurs > {max_horizon_hours}h. "
                "Vérifier le preprocessing ou l'argument --max-horizon-hours."
            )


# ============================================================
# MÉTRIQUES / PONDÉRATION MÉTIER
# ============================================================

HORIZON_BINS = [
    (0.0, 6.0, "0_6h"),
    (6.0, 12.0, "6_12h"),
    (12.0, 24.0, "12_24h"),
    (24.0, 36.0, "24_36h"),
    (36.0, 48.0, "36_48h"),
]


def regression_sample_weights(y: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pondération des erreurs selon l'horizon.

    none : pas de pondération.
    early_warning : priorité aux horizons éloignés, sans ignorer les horizons proches.
    balanced_horizon : poids uniforme par tranche pour réduire l'effet de fréquence.
    """
    if mode == "none":
        return torch.ones_like(y, dtype=torch.float32)

    weights = torch.ones_like(y, dtype=torch.float32)

    if mode == "early_warning":
        # Objectif métier : valoriser les prédictions anticipées.
        weights = torch.where((y >= 36.0) & (y <= 48.0), torch.tensor(2.0, device=y.device), weights)
        weights = torch.where((y >= 24.0) & (y < 36.0), torch.tensor(1.8, device=y.device), weights)
        weights = torch.where((y >= 12.0) & (y < 24.0), torch.tensor(1.5, device=y.device), weights)
        weights = torch.where((y >= 6.0) & (y < 12.0), torch.tensor(1.3, device=y.device), weights)
        weights = torch.where((y >= 0.0) & (y < 6.0), torch.tensor(1.2, device=y.device), weights)
        return weights

    if mode == "balanced_horizon":
        # Le poids est calculé au batch comme inverse de fréquence par tranche.
        # Ce n'est pas parfait, mais évite qu'un horizon très fréquent domine totalement la loss.
        for lo, hi, _ in HORIZON_BINS:
            mask = (y >= lo) & (y < hi if hi < 48.0 else y <= hi)
            n_bin = mask.sum().float()
            if n_bin > 0:
                weights[mask] = y.numel() / (len(HORIZON_BINS) * n_bin)
        return weights

    raise ValueError(f"loss_weighting invalide : {mode}")


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0, weighting: str = "none"):
        super().__init__()
        self.beta = float(beta)
        self.weighting = weighting
        self.base_loss = nn.SmoothL1Loss(beta=self.beta, reduction="none")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = self.base_loss(preds, target)
        weights = regression_sample_weights(target, self.weighting)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    # Métriques par horizon réel.
    horizon_maes = {}
    horizon_rmses = {}

    for lo, hi, name in HORIZON_BINS:
        if hi < 48.0:
            mask = (y_true >= lo) & (y_true < hi)
        else:
            mask = (y_true >= lo) & (y_true <= hi)

        metrics[f"n_{name}"] = int(mask.sum())

        if mask.sum() == 0:
            metrics[f"mae_{name}"] = float("nan")
            metrics[f"rmse_{name}"] = float("nan")
            continue

        mae_h = mean_absolute_error(y_true[mask], y_pred[mask])
        mse_h = mean_squared_error(y_true[mask], y_pred[mask])
        rmse_h = math.sqrt(mse_h)

        metrics[f"mae_{name}"] = float(mae_h)
        metrics[f"rmse_{name}"] = float(rmse_h)
        horizon_maes[name] = float(mae_h)
        horizon_rmses[name] = float(rmse_h)

    # Score métier à minimiser : plus de poids aux horizons anticipés.
    # Si une tranche est absente, elle est ignorée et les poids sont renormalisés.
    business_weights = {
        "36_48h": 0.25,
        "24_36h": 0.25,
        "12_24h": 0.20,
        "6_12h": 0.15,
        "0_6h": 0.15,
    }

    score_num = 0.0
    score_den = 0.0

    for name, weight in business_weights.items():
        value = metrics.get(f"mae_{name}", float("nan"))
        if np.isfinite(value):
            score_num += weight * value
            score_den += weight

    metrics["business_score"] = float(score_num / score_den) if score_den > 0 else float(mae)

    return metrics


def get_early_stopping_value(metrics: dict, metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(
            f"Métrique d'early stopping absente : {metric_name}. "
            f"Métriques disponibles : {sorted(metrics.keys())}"
        )
    return float(metrics[metric_name])


def is_improvement(current: float, best: float, min_delta: float) -> bool:
    # Toutes les métriques de régression utilisées ici sont à minimiser.
    return (best - current) > min_delta


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
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ============================================================
# MODEL
# ============================================================

class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = float(std)

    def forward(self, x):
        if self.training and self.std > 0.0:
            return x + torch.randn_like(x) * self.std
        return x


class CNNBiLSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.35,
        input_noise_std: float = 0.01,
    ):
        super().__init__()

        self.input_noise = GaussianNoise(std=input_noise_std)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=conv_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x):
        x = self.input_noise(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)

        out = self.head(pooled).squeeze(-1)
        return out


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
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

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
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    n_samples = 0

    preds_all = []
    y_all = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(preds_all)

    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = float(running_loss / max(n_samples, 1))

    return metrics, y_true, y_pred


def try_setup_mlflow(args):
    if not args.use_mlflow:
        return None

    try:
        import mlflow
    except ImportError:
        print("MLflow non installé. Entraînement sans tracking MLflow.")
        return None

    tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = args.mlflow_experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or "Vulcadata"

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

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_npz_dataset(npz_path)

    validate_regression_targets(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        max_horizon_hours=args.max_horizon_hours,
    )

    print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape} | y_val   : {y_val.shape}")
    print(f"X_test  : {X_test.shape} | y_test  : {y_test.shape}")

    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        raise ValueError("NaN détectés dans X après preprocessing.")

    if np.isinf(X_train).any() or np.isinf(X_val).any() or np.isinf(X_test).any():
        raise ValueError("Inf détectés dans X après preprocessing.")

    n_features = X_train.shape[2]
    seq_len = X_train.shape[1]

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
    print(f"Loss weighting : {args.loss_weighting}")
    print(f"Early stopping : {args.early_stopping_metric} (min)")

    model = CNNBiLSTMRegressor(
        n_features=n_features,
        conv_channels=args.conv_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        input_noise_std=args.input_noise_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Nombre de paramètres total        : {n_params:,}")
    print(f"Nombre de paramètres entraînables : {n_trainable_params:,}")

    criterion = WeightedSmoothL1Loss(
        beta=args.huber_beta,
        weighting=args.loss_weighting,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.lr_patience),
    )

    mlflow = try_setup_mlflow(args)

    best_val_score = float("inf")
    best_model_path = output_dir / "best_cnn_bilstm_regression.pt"
    history = []
    patience_counter = 0

    run_context = mlflow.start_run(run_name=args.run_name) if mlflow else None

    try:
        if mlflow:
            mlflow.log_params({
                "model_type": "CNN_BiLSTM",
                "task": "regression",
                "target": "hours_before_eruption",
                "input_npz": args.input_npz,
                "n_features": int(n_features),
                "seq_len": int(seq_len),
                "conv_channels": args.conv_channels,
                "lstm_hidden": args.lstm_hidden,
                "lstm_layers": args.lstm_layers,
                "input_noise_std": args.input_noise_std,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "loss": "WeightedSmoothL1Loss",
                "loss_weighting": args.loss_weighting,
                "huber_beta": args.huber_beta,
                "early_stopping_metric": args.early_stopping_metric,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "early_stopping_patience": args.early_stopping_patience,
                "lr_patience": args.lr_patience,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "n_params": int(n_params),
                "n_trainable_params": int(n_trainable_params),
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

            val_metrics, _, _ = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            current_score = get_early_stopping_value(val_metrics, args.early_stopping_metric)
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
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_business={val_metrics['business_score']:.4f} | "
                f"val_mae_36_48h={val_metrics['mae_36_48h']:.4f} | "
                f"val_mae_0_6h={val_metrics['mae_0_6h']:.4f}"
            )

            if mlflow:
                mlflow.log_metrics(row, step=epoch)

            if is_improvement(
                current=current_score,
                best=best_val_score,
                min_delta=args.early_stopping_min_delta,
            ):
                best_val_score = current_score
                patience_counter = 0

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": "CNN_BiLSTM",
                    "task": "regression",
                    "n_features": int(n_features),
                    "seq_len": int(seq_len),
                    "args": vars(args),
                    "best_val_score": float(best_val_score),
                    "best_val_metric": args.early_stopping_metric,
                    "feature_names": (
                        feature_names.tolist()
                        if hasattr(feature_names, "tolist")
                        else feature_names
                    ),
                    "conv_channels": int(args.conv_channels),
                    "lstm_hidden": int(args.lstm_hidden),
                    "lstm_layers": int(args.lstm_layers),
                    "dropout": float(args.dropout),
                    "input_noise_std": float(args.input_noise_std),
                }

                torch.save(checkpoint, best_model_path)
                print(
                    f"  Nouveau meilleur modèle sauvegardé : {best_model_path} | "
                    f"{args.early_stopping_metric}={best_val_score:.6f}"
                )
            else:
                patience_counter += 1
                print(f"  Pas d'amélioration ({patience_counter}/{args.early_stopping_patience})")

            if patience_counter >= args.early_stopping_patience:
                print(
                    f"Early stopping déclenché après {args.early_stopping_patience} "
                    "epochs sans amélioration."
                )
                break

        print(f"\nChargement du meilleur modèle : {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, y_true_test, y_pred_test = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print("\nMétriques TEST")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.6f}" if np.isfinite(value) else f"  {key}: NaN")

        history_path = output_dir / "history_cnn_bilstm_regression.json"
        metrics_path = output_dir / "metrics_cnn_bilstm_regression.json"
        predictions_path = output_dir / "predictions_cnn_bilstm_regression.npz"

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_metric": args.early_stopping_metric,
                    "best_val_score": float(best_val_score),
                    "test": test_metrics,
                    "n_params": int(n_params),
                    "n_trainable_params": int(n_trainable_params),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        np.savez_compressed(
            predictions_path,
            y_true=y_true_test.astype(np.float32),
            y_pred=y_pred_test.astype(np.float32),
        )

        if mlflow:
            try:
                mlflow.log_metrics({
                    f"test_{k}": v
                    for k, v in test_metrics.items()
                    if np.isfinite(v)
                })
                mlflow.log_metric(f"best_val_{args.early_stopping_metric}", best_val_score)
                mlflow.log_metric("n_params", n_params)
                mlflow.log_metric("n_trainable_params", n_trainable_params)

                mlflow.log_artifact(str(best_model_path))
                mlflow.log_artifact(str(history_path))
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(predictions_path))

            except Exception as exc:
                print("\nWARNING : échec du logging MLflow final.")
                print(f"Type erreur : {type(exc).__name__}")
                print(f"Détail      : {exc}")
                print("Les fichiers locaux ont été sauvegardés.")
                print("Le training est considéré comme terminé malgré l'échec MLflow.")

        if args.s3_output_prefix:
            prefix = args.s3_output_prefix.strip().strip("/")
            bucket = args.s3_bucket or os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET_NAME")

            if not bucket:
                raise ValueError("--s3-output-prefix est fourni, mais aucun bucket S3 n'est disponible.")

            for local_path in [best_model_path, history_path, metrics_path, predictions_path]:
                s3_uri = f"s3://{bucket}/{prefix}/{local_path.name}"
                upload_file_to_s3(local_path, s3_uri)

    finally:
        if run_context is not None:
            try:
                mlflow.end_run()
            except Exception as exc:
                print("\nWARNING : échec mlflow.end_run().")
                print(f"Type erreur : {type(exc).__name__}")
                print(f"Détail      : {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-npz", type=str, default="data/preprocessing/processed/volcano_multi.npz")
    parser.add_argument("--output-dir", type=str, default="models/cnn_bilstm_regression")
    parser.add_argument("--s3-bucket", type=str, default=None)
    parser.add_argument("--s3-output-prefix", type=str, default="")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="business_score",
        choices=[
            "loss",
            "mae",
            "rmse",
            "business_score",
            "mae_0_6h",
            "mae_6_12h",
            "mae_12_24h",
            "mae_24_36h",
            "mae_36_48h",
        ],
    )
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="none",
        choices=["none", "early_warning", "balanced_horizon"],
        help="Pondération de la loss de régression.",
    )
    parser.add_argument("--max-horizon-hours", type=float, default=48.0)

    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--input-noise-std", type=float, default=0.01)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-amp", action="store_true")

    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="cnn_bilstm_volcano_regression")

    args = parser.parse_args()
    main(args)


