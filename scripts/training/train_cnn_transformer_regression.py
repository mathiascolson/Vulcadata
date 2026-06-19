# scripts/train_cnn_transformer_regression.py

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
    """
    Compatible avec :
    - chemin local : data/preprocessing/.../volcano_multi.npz
    - chemin S3    : s3://bucket/path/volcano_multi.npz
    """
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
    """
    Structure attendue du fichier de preprocessing actuel :

    X_train : (N_train, seq_len, n_features)
    y_train : (N_train,)

    X_val   : (N_val, seq_len, n_features)
    y_val   : (N_val,)

    X_test  : (N_test, seq_len, n_features)
    y_test  : (N_test,)
    """
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
    y_train = data["y_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    feature_names = data["feature_names"] if "feature_names" in data else None

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


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
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ============================================================
# MODEL
# ============================================================

class CNNTransformerRegressor(nn.Module):
    """
    CNN-Transformer adapté à la régression horaire.

    Modifications par rapport à la version précédente :
    - ajout d'un dropout à 0.20 par défaut ;
    - dim_feedforward paramétrable, avec 256 par défaut ;
    - conservation de la taille principale du modèle :
        d_model=128
        nhead=4
        num_layers=4
    - sortie scalaire pour régression.
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x : (B, L, F)
        x = x.transpose(1, 2)      # (B, F, L)

        x = self.conv(x)           # (B, d_model, L)
        x = x.transpose(1, 2)      # (B, L, d_model)

        x = self.transformer(x)    # (B, L, d_model)

        # On garde le comportement initial : dernier token temporel.
        x = x[:, -1, :]            # (B, d_model)

        out = self.reg_head(x)     # (B, 1)
        return out.squeeze(-1)     # (B,)


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
):
    model.train()

    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

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


# ============================================================
# MLFLOW
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_cache_dir = output_dir / "input_cache"
    npz_path = download_from_s3_if_needed(args.input_npz, local_cache_dir)

    print(f"Chargement dataset : {npz_path}")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_npz_dataset(npz_path)

    print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape} | y_val   : {y_val.shape}")
    print(f"X_test  : {X_test.shape} | y_test  : {y_test.shape}")

    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        raise ValueError("NaN détectés dans X après preprocessing.")

    if np.isinf(X_train).any() or np.isinf(X_val).any() or np.isinf(X_test).any():
        raise ValueError("Inf détectés dans X après preprocessing.")

    if np.isnan(y_train).any() or np.isnan(y_val).any() or np.isnan(y_test).any():
        raise ValueError("NaN détectés dans y après preprocessing.")

    if np.isinf(y_train).any() or np.isinf(y_val).any() or np.isinf(y_test).any():
        raise ValueError("Inf détectés dans y après preprocessing.")

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

    print(f"Device : {device}")

    model = CNNTransformerRegressor(
        feature_dim=n_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Nombre de paramètres total      : {n_params:,}")
    print(f"Nombre de paramètres entraînables : {n_trainable_params:,}")

    # Loss adaptée à une cible horaire continue.
    # SmoothL1Loss est plus robuste que MSELoss si quelques cibles sont extrêmes.
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)

    # Optimizer conservé depuis l'ancien notebook : Adam, lr=1e-4 par défaut.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    mlflow = try_setup_mlflow(args)

    best_val_mae = float("inf")
    best_model_path = output_dir / "best_cnn_transformer_regression.pt"

    history = []
    patience_counter = 0

    run_context = mlflow.start_run(run_name=args.run_name) if mlflow else None

    try:
        if mlflow:
            mlflow.log_params({
                "model_type": "CNNTransformerRegressor",
                "input_npz": args.input_npz,
                "n_features": int(n_features),
                "seq_len": int(seq_len),

                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,

                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": "AdamW",
                "weight_decay": args.weight_decay,

                "epochs": args.epochs,
                "loss": "SmoothL1Loss",
                "huber_beta": args.huber_beta,

                "scheduler": "ReduceLROnPlateau",
                "scheduler_mode": "min",
                "scheduler_factor": 0.5,
                "scheduler_patience": 2,
                "scheduler_monitor": "val_mae",

                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
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
            )

            val_metrics, _, _ = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
            
            scheduler.step(val_metrics["mae"])

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_mae": float(val_metrics["mae"]),
                "val_mse": float(val_metrics["mse"]),
                "val_rmse": float(val_metrics["rmse"]),
                "val_r2": float(val_metrics["r2"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            history.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_r2={val_metrics['r2']:.4f}"
            )

            if mlflow:
                mlflow.log_metrics(row, step=epoch)

            improvement = best_val_mae - val_metrics["mae"]

            if improvement > args.early_stopping_min_delta:
                best_val_mae = val_metrics["mae"]
                patience_counter = 0

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": "CNNTransformerRegressor",
                    "n_features": int(n_features),
                    "seq_len": int(seq_len),
                    "d_model": int(args.d_model),
                    "nhead": int(args.nhead),
                    "num_layers": int(args.num_layers),
                    "best_val_mae": float(best_val_mae),
                    "args": vars(args),
                    "feature_names": (
                        feature_names.tolist()
                        if hasattr(feature_names, "tolist")
                        else feature_names
                    ),
                }

                torch.save(checkpoint, best_model_path)
                print(f"  Nouveau meilleur modèle sauvegardé : {best_model_path}")

            else:
                patience_counter += 1
                print(
                    f"  Pas d'amélioration suffisante "
                    f"({patience_counter}/{args.early_stopping_patience})"
                )

            if patience_counter >= args.early_stopping_patience:
                print(
                    f"Early stopping déclenché après "
                    f"{args.early_stopping_patience} epochs sans amélioration suffisante."
                )
                break

        print(f"\nChargement du meilleur modèle : {best_model_path}")

        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, y_true_test, y_pred_test = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print("\nMétriques TEST")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.6f}")

        history_path = output_dir / "history_cnn_transformer_regression.json"
        metrics_path = output_dir / "metrics_cnn_transformer_regression.json"
        predictions_path = output_dir / "predictions_cnn_transformer_regression.npz"

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_mae": float(best_val_mae),
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
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric("best_val_mae", best_val_mae)
            mlflow.log_metric("n_params", n_params)
            mlflow.log_metric("n_trainable_params", n_trainable_params)

            mlflow.log_artifact(str(best_model_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(predictions_path))

        if args.s3_output_prefix:
            prefix = args.s3_output_prefix.strip().strip("/")
            bucket = (
                args.s3_bucket
                or os.getenv("S3_BUCKET_NAME")
                or os.getenv("AWS_S3_BUCKET_NAME")
            )

            if not bucket:
                raise ValueError(
                    "--s3-output-prefix est fourni, mais aucun bucket S3 n'est disponible."
                )

            for local_path in [
                best_model_path,
                history_path,
                metrics_path,
                predictions_path,
            ]:
                s3_uri = f"s3://{bucket}/{prefix}/{local_path.name}"
                upload_file_to_s3(local_path, s3_uri)

    finally:
        if run_context is not None:
            mlflow.end_run()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-npz",
        type=str,
        default="data/preprocessing/processed_core_stride3/volcano_multi.npz",
        help="Chemin local ou URI S3 vers le fichier volcano_multi.npz.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/cnn_transformer_regression",
        help="Répertoire local de sortie.",
    )

    parser.add_argument("--s3-bucket", type=str, default=None)

    parser.add_argument(
        "--s3-output-prefix",
        type=str,
        default="",
        help="Préfixe S3 pour uploader les artefacts modèle. Vide = pas d'upload.",
    )


    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e54)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)


    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Early stopping léger : nombre d'epochs sans amélioration avant arrêt.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Amélioration minimale du MAE validation pour réinitialiser la patience.",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default="cnn_transformer_regression_hourly_target",
    )

    args = parser.parse_args()
    main(args)