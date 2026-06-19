# scripts/sweep_transformer_models.py

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PROFILS DE MODÈLES
# ============================================================

TRANSFORMER_MODEL_PROFILES = [
    {
        "profile_name": "transformer_light",
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 128,
    },
    {
        "profile_name": "transformer_medium",
        "d_model": 96,
        "n_heads": 4,
        "n_layers": 3,
        "dim_feedforward": 256,
    },
    {
        "profile_name": "transformer_large",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "dim_feedforward": 512,
    },
]


CNN_TRANSFORMER_MODEL_PROFILES = [
    {
        "profile_name": "cnn_transformer_light",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
    },
    {
        "profile_name": "cnn_transformer_medium",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
    },
    {
        "profile_name": "cnn_transformer_heavier",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 512,
    },
]


# ============================================================
# PROFILS D’OPTIMISATION CLASSIFICATION
# ============================================================

OPTIMIZATION_PROFILES = [
    {
        "optim_name": "low_lr_balanced",
        "learning_rate": 3e-5,
        "weight_decay": 1e-4,
        "dropout": 0.20,
        "input_noise_std": 0.01,
        "class_weighting": "balanced",
        "label_smoothing": 0.0,
    },
    {
        "optim_name": "low_lr_alert_priority",
        "learning_rate": 3e-5,
        "weight_decay": 1e-4,
        "dropout": 0.20,
        "input_noise_std": 0.01,
        "class_weighting": "alert_priority",
        "label_smoothing": 0.0,
    },
    {
        "optim_name": "standard_lr_balanced",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.20,
        "input_noise_std": 0.01,
        "class_weighting": "balanced",
        "label_smoothing": 0.0,
    },
    {
        "optim_name": "standard_lr_light_regularization",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.10,
        "input_noise_std": 0.005,
        "class_weighting": "balanced",
        "label_smoothing": 0.0,
    },
    {
        "optim_name": "standard_lr_smoothed",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.20,
        "input_noise_std": 0.01,
        "class_weighting": "balanced",
        "label_smoothing": 0.05,
    },
]


BATCH_SIZES = [32, 64]


# ============================================================
# HELPERS
# ============================================================

def validate_transformer_profile(profile: dict) -> None:
    d_model = profile["d_model"]
    n_heads = profile["n_heads"]

    if d_model % n_heads != 0:
        raise ValueError(
            f"Profil invalide {profile['profile_name']} : "
            f"d_model={d_model} n'est pas divisible par n_heads={n_heads}."
        )


def validate_cnn_transformer_profile(profile: dict) -> None:
    d_model = profile["d_model"]
    nhead = profile["nhead"]

    if d_model % nhead != 0:
        raise ValueError(
            f"Profil invalide {profile['profile_name']} : "
            f"d_model={d_model} n'est pas divisible par nhead={nhead}."
        )


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "")
        .replace("=", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def build_run_name(model_family: str, model_profile: dict, optim_profile: dict, batch_size: int, dataset_tag: str) -> str:
    return (
        f"{model_family}_"
        f"{dataset_tag}_"
        f"{model_profile['profile_name']}_"
        f"{optim_profile['optim_name']}_"
        f"bs{batch_size}"
    )


def build_output_dir(base_output_dir: Path, run_name: str) -> Path:
    return base_output_dir / safe_name(run_name)


def run_command(cmd: list[str], dry_run: bool) -> int:
    print("\n" + "=" * 120)
    print("Commande :")
    print(" ".join(cmd))
    print("=" * 120)

    if dry_run:
        return 0

    completed = subprocess.run(cmd)
    return completed.returncode


def write_manifest_entry(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ============================================================
# COMMANDES TRANSFORMER SIMPLE
# ============================================================

def build_transformer_command(
    args,
    model_profile: dict,
    optim_profile: dict,
    batch_size: int,
    run_name: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        args.transformer_script,

        "--input-npz", args.input_npz,
        "--output-dir", str(output_dir),

        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),

        "--learning-rate", str(optim_profile["learning_rate"]),
        "--weight-decay", str(optim_profile["weight_decay"]),
        "--dropout", str(optim_profile["dropout"]),
        "--input-noise-std", str(optim_profile["input_noise_std"]),

        "--d-model", str(model_profile["d_model"]),
        "--n-heads", str(model_profile["n_heads"]),
        "--n-layers", str(model_profile["n_layers"]),
        "--dim-feedforward", str(model_profile["dim_feedforward"]),

        "--class-weighting", str(optim_profile["class_weighting"]),
        "--label-smoothing", str(optim_profile["label_smoothing"]),
        "--early-stopping-metric", args.early_stopping_metric,
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--lr-patience", str(args.lr_patience),
        "--grad-clip", str(args.grad_clip),
        "--n-classes", str(args.n_classes),

        "--run-name", run_name,
        "--use-mlflow",
    ]


# ============================================================
# COMMANDES CNN-TRANSFORMER
# ============================================================

def build_cnn_transformer_command(
    args,
    model_profile: dict,
    optim_profile: dict,
    batch_size: int,
    run_name: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        args.cnn_transformer_script,

        "--input-npz", args.input_npz,
        "--output-dir", str(output_dir),

        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),

        "--learning-rate", str(optim_profile["learning_rate"]),
        "--weight-decay", str(optim_profile["weight_decay"]),
        "--dropout", str(optim_profile["dropout"]),
        "--input-noise-std", str(optim_profile["input_noise_std"]),

        "--d-model", str(model_profile["d_model"]),
        "--nhead", str(model_profile["nhead"]),
        "--num-layers", str(model_profile["num_layers"]),
        "--dim-feedforward", str(model_profile["dim_feedforward"]),

        "--class-weighting", str(optim_profile["class_weighting"]),
        "--label-smoothing", str(optim_profile["label_smoothing"]),
        "--early-stopping-metric", args.early_stopping_metric,
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--lr-patience", str(args.lr_patience),
        "--grad-clip", str(args.grad_clip),
        "--n-classes", str(args.n_classes),

        "--run-name", run_name,
        "--use-mlflow",
    ]


# ============================================================
# MAIN
# ============================================================

def main(args):
    dataset_tag = args.dataset_tag or Path(args.input_npz).parent.name

    base_output_dir = Path(args.output_root)
    manifest_path = base_output_dir / f"sweep_manifest_{safe_name(dataset_tag)}.jsonl"

    base_output_dir.mkdir(parents=True, exist_ok=True)

    for profile in TRANSFORMER_MODEL_PROFILES:
        validate_transformer_profile(profile)

    for profile in CNN_TRANSFORMER_MODEL_PROFILES:
        validate_cnn_transformer_profile(profile)

    planned_runs = []

    if args.model_family in {"transformer", "both"}:
        for model_profile, optim_profile, batch_size in itertools.product(
            TRANSFORMER_MODEL_PROFILES,
            OPTIMIZATION_PROFILES,
            BATCH_SIZES,
        ):
            run_name = build_run_name(
                model_family="transformer",
                model_profile=model_profile,
                optim_profile=optim_profile,
                batch_size=batch_size,
                dataset_tag=dataset_tag,
            )

            output_dir = build_output_dir(base_output_dir, run_name)

            cmd = build_transformer_command(
                args=args,
                model_profile=model_profile,
                optim_profile=optim_profile,
                batch_size=batch_size,
                run_name=run_name,
                output_dir=output_dir,
            )

            planned_runs.append({
                "model_family": "transformer",
                "run_name": run_name,
                "output_dir": str(output_dir),
                "model_profile": model_profile,
                "optimization_profile": optim_profile,
                "batch_size": batch_size,
                "command": cmd,
            })

    if args.model_family in {"cnn_transformer", "both"}:
        for model_profile, optim_profile, batch_size in itertools.product(
            CNN_TRANSFORMER_MODEL_PROFILES,
            OPTIMIZATION_PROFILES,
            BATCH_SIZES,
        ):
            run_name = build_run_name(
                model_family="cnn_transformer",
                model_profile=model_profile,
                optim_profile=optim_profile,
                batch_size=batch_size,
                dataset_tag=dataset_tag,
            )

            output_dir = build_output_dir(base_output_dir, run_name)

            cmd = build_cnn_transformer_command(
                args=args,
                model_profile=model_profile,
                optim_profile=optim_profile,
                batch_size=batch_size,
                run_name=run_name,
                output_dir=output_dir,
            )

            planned_runs.append({
                "model_family": "cnn_transformer",
                "run_name": run_name,
                "output_dir": str(output_dir),
                "model_profile": model_profile,
                "optimization_profile": optim_profile,
                "batch_size": batch_size,
                "command": cmd,
            })

    if args.limit is not None:
        planned_runs = planned_runs[: args.limit]

    print("\nPlan de sweep classification")
    print(f"Dataset        : {args.input_npz}")
    print(f"Dataset tag    : {dataset_tag}")
    print(f"Famille modèle : {args.model_family}")
    print(f"Métrique stop  : {args.early_stopping_metric}")
    print(f"Nombre de runs : {len(planned_runs)}")
    print(f"Manifest       : {manifest_path}")
    print(f"Dry run        : {args.dry_run}")

    sweep_start = datetime.now().isoformat(timespec="seconds")

    for index, run_payload in enumerate(planned_runs, start=1):
        run_payload["sweep_index"] = index
        run_payload["sweep_total"] = len(planned_runs)
        run_payload["sweep_start"] = sweep_start

        print("\n" + "#" * 120)
        print(f"RUN {index}/{len(planned_runs)} — {run_payload['run_name']}")
        print("#" * 120)

        write_manifest_entry(
            manifest_path=manifest_path,
            payload={
                **run_payload,
                "status": "planned",
            },
        )

        return_code = run_command(
            cmd=run_payload["command"],
            dry_run=args.dry_run,
        )

        status = "success" if return_code == 0 else "failed"

        write_manifest_entry(
            manifest_path=manifest_path,
            payload={
                **run_payload,
                "status": status,
                "return_code": return_code,
            },
        )

        if return_code != 0 and args.stop_on_error:
            raise RuntimeError(
                f"Run échoué avec code {return_code} : {run_payload['run_name']}"
            )

    print("\nSweep terminé.")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Chemin vers le fichier volcano_multi.npz à tester.",
    )

    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Nom court du dataset pour les runs MLflow. Par défaut : nom du dossier parent du NPZ.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="models/sweeps_classification",
        help="Dossier racine des sorties locales de sweep.",
    )

    parser.add_argument(
        "--model-family",
        type=str,
        default="both",
        choices=["transformer", "cnn_transformer", "both"],
        help="Famille de modèle à tester.",
    )

    parser.add_argument(
        "--transformer-script",
        type=str,
        default="scripts/train_transformer_classif.py",
    )

    parser.add_argument(
        "--cnn-transformer-script",
        type=str,
        default="scripts/train_cnn_transformer_classif.py",
    )

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="macro_f1",
        help=(
            "Métrique utilisée pour choisir le meilleur modèle. "
            "Exemples : macro_f1, weighted_f1, balanced_accuracy, alert_f1, alert_24h_f1, loss."
        ),
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Patience commune.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
    )

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-classes", type=int, default=6)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre de runs pour un test rapide.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les commandes sans les exécuter.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Arrête le sweep au premier run en erreur.",
    )

    args = parser.parse_args()
    main(args)
