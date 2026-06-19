# scripts/train_transformer.py

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

class GaussianNoise(nn.Module):
    """
    Bruit gaussien léger appliqué uniquement pendant l'entraînement.

    Utilité :
    - réduit la mémorisation de séquences quasi identiques
    - force le modèle à apprendre des motifs plus robustes
    - n'affecte pas validation/test car self.training=False
    """
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = float(std)

    def forward(self, x):
        if self.training and self.std > 0.0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal classique.
    """
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x : (B, T, D)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    """
    Entrée :
        x : (batch, seq_len, n_features)

    Architecture :
        - projection linéaire des features
        - positional encoding
        - transformer encoder
        - pooling temporel
        - régression
    """
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.30,
        input_noise_std: float = 0.01,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads.")

        self.input_noise = GaussianNoise(std=input_noise_std)

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max(seq_len, 1000),
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x : (B, T, F)
        x = self.input_noise(x)
        x = self.input_projection(x)       # (B, T, D)
        x = self.positional_encoding(x)    # (B, T, D)
        x = self.encoder(x)                # (B, T, D)
        x = self.norm(x)

        pooled = x.mean(dim=1)             # (B, D)

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

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_npz_dataset(npz_path)

    print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape} | y_val   : {y_val.shape}")
    print(f"X_test  : {X_test.shape} | y_test  : {y_test.shape}")

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

    model = TransformerRegressor(
        n_features=n_features,
        seq_len=seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        input_noise_std=args.input_noise_std,
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
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

    best_val_mae = float("inf")
    best_model_path = output_dir / "best_transformer.pt"
    history = []
    patience_counter = 0

    run_context = mlflow.start_run(run_name=args.run_name) if mlflow else None

    try:
        if mlflow:
            mlflow.log_params({
                "model_type": "TransformerEncoder",
                "input_npz": args.input_npz,
                "n_features": n_features,
                "seq_len": seq_len,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "input_noise_std": args.input_noise_std,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "loss": "SmoothL1Loss",
                "huber_beta": args.huber_beta,
                "seed": args.seed,
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

            scheduler.step(val_metrics["mae"])

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
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_r2={val_metrics['r2']:.4f}"
            )

            if mlflow:
                mlflow.log_metrics(row, step=epoch)

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                patience_counter = 0

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": "TransformerEncoder",
                    "n_features": int(n_features),
                    "seq_len": int(seq_len),
                    "args": vars(args),
                    "best_val_mae": float(best_val_mae),
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

        test_metrics, y_true_test, y_pred_test = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print("\nMétriques TEST")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.6f}")

        history_path = output_dir / "history_transformer.json"
        metrics_path = output_dir / "metrics_transformer.json"
        predictions_path = output_dir / "predictions_transformer.npz"

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"test": test_metrics, "best_val_mae": best_val_mae}, f, indent=2)

        np.savez_compressed(
            predictions_path,
            y_true=y_true_test.astype(np.float32),
            y_pred=y_pred_test.astype(np.float32),
        )

        if mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric("best_val_mae", best_val_mae)
            mlflow.log_artifact(str(best_model_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(predictions_path))

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
            ]:
                s3_uri = f"s3://{bucket}/{prefix}/{local_path.name}"
                upload_file_to_s3(local_path, s3_uri)

    finally:
        if run_context is not None:
            mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-npz",
        type=str,
        default="data/preprocessing/processed/volcano_multi.npz",
        help=(
            "Chemin local ou URI S3 vers volcano_multi.npz. "
            "Par défaut, le fichier est lu localement."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/transformer",
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=0.01,
        help=(
            "Écart-type du bruit gaussien ajouté aux entrées pendant l'entraînement. "
            "0 désactive cette régularisation."
        ),
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-amp", action="store_true")

    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="transformer_volcano_regression")

    args = parser.parse_args()
    main(args)