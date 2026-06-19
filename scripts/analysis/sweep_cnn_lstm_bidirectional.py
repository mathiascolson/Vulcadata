# scripts/sweep_cnn_bilstm.py

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# PROFILS COHÉRENTS CNN-BiLSTM
# ============================================================

CNN_BILSTM_MODEL_PROFILES = [
    {
        "profile_name": "cnn_bilstm_light",
        "conv_channels": 64,
        "lstm_hidden": 64,
        "lstm_layers": 1,
    },
    {
        "profile_name": "cnn_bilstm_medium",
        "conv_channels": 128,
        "lstm_hidden": 128,
        "lstm_layers": 1,
    },
    {
        "profile_name": "cnn_bilstm_deep",
        "conv_channels": 128,
        "lstm_hidden": 128,
        "lstm_layers": 2,
    },
    {
        "profile_name": "cnn_bilstm_heavier",
        "conv_channels": 192,
        "lstm_hidden": 192,
        "lstm_layers": 2,
    },
]


# ============================================================
# PROFILS D’OPTIMISATION
# ============================================================

OPTIMIZATION_PROFILES = [
    {
        "optim_name": "low_lr_regularized",
        "learning_rate": 3e-5,
        "weight_decay": 1e-4,
        "dropout": 0.20,
        "input_noise_std": 0.01,
    },
    {
        "optim_name": "low_lr_strong_regularization",
        "learning_rate": 3e-5,
        "weight_decay": 1e-3,
        "dropout": 0.30,
        "input_noise_std": 0.01,
    },
    {
        "optim_name": "standard_lr_regularized",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.20,
        "input_noise_std": 0.01,
    },
    {
        "optim_name": "standard_lr_light_regularization",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.10,
        "input_noise_std": 0.005,
    },
]


BATCH_SIZES = [32, 64]


# ============================================================
# HELPERS
# ============================================================

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


def build_run_name(model_profile: dict, optim_profile: dict, batch_size: int, dataset_tag: str) -> str:
    return (
        f"cnn_bilstm_"
        f"{dataset_tag}_"
        f"{model_profile['profile_name']}_"
        f"{optim_profile['optim_name']}_"
        f"bs{batch_size}"
    )


def build_output_dir(base_output_dir: Path, run_name: str) -> Path:
    return base_output_dir / safe_name(run_name)


def write_manifest_entry(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_command(cmd: list[str], dry_run: bool) -> int:
    print("\n" + "=" * 120)
    print("Commande :")
    print(" ".join(cmd))
    print("=" * 120)

    if dry_run:
        return 0

    completed = subprocess.run(cmd)
    return completed.returncode


# ============================================================
# COMMANDES CNN-BiLSTM
# ============================================================

def build_cnn_bilstm_command(
    args,
    model_profile: dict,
    optim_profile: dict,
    batch_size: int,
    run_name: str,
    output_dir: Path,
) -> list[str]:

    cmd = [
        sys.executable,
        args.train_script,

        "--input-npz", args.input_npz,
        "--output-dir", str(output_dir),

        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),

        "--learning-rate", str(optim_profile["learning_rate"]),
        "--weight-decay", str(optim_profile["weight_decay"]),
        "--dropout", str(optim_profile["dropout"]),
        "--input-noise-std", str(optim_profile["input_noise_std"]),

        "--conv-channels", str(model_profile["conv_channels"]),
        "--lstm-hidden", str(model_profile["lstm_hidden"]),
        "--lstm-layers", str(model_profile["lstm_layers"]),

        "--early-stopping-patience", str(args.early_stopping_patience),
        "--lr-patience", str(args.lr_patience),
        "--grad-clip", str(args.grad_clip),
        "--huber-beta", str(args.huber_beta),

        "--run-name", run_name,
        "--use-mlflow",
    ]

    if args.use_amp:
        cmd.append("--use-amp")

    return cmd


# ============================================================
# MAIN
# ============================================================

def main(args):
    dataset_tag = args.dataset_tag or Path(args.input_npz).parent.name

    base_output_dir = Path(args.output_root)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = base_output_dir / f"sweep_manifest_cnn_bilstm_{safe_name(dataset_tag)}.jsonl"

    planned_runs = []

    for model_profile, optim_profile, batch_size in itertools.product(
        CNN_BILSTM_MODEL_PROFILES,
        OPTIMIZATION_PROFILES,
        BATCH_SIZES,
    ):
        run_name = build_run_name(
            model_profile=model_profile,
            optim_profile=optim_profile,
            batch_size=batch_size,
            dataset_tag=dataset_tag,
        )

        output_dir = build_output_dir(base_output_dir, run_name)

        cmd = build_cnn_bilstm_command(
            args=args,
            model_profile=model_profile,
            optim_profile=optim_profile,
            batch_size=batch_size,
            run_name=run_name,
            output_dir=output_dir,
        )

        planned_runs.append({
            "model_family": "cnn_bilstm",
            "run_name": run_name,
            "output_dir": str(output_dir),
            "model_profile": model_profile,
            "optimization_profile": optim_profile,
            "batch_size": batch_size,
            "command": cmd,
        })

    if args.limit is not None:
        planned_runs = planned_runs[: args.limit]

    print("\nPlan de sweep CNN-BiLSTM")
    print(f"Dataset      : {args.input_npz}")
    print(f"Dataset tag  : {dataset_tag}")
    print(f"Nombre runs  : {len(planned_runs)}")
    print(f"Manifest     : {manifest_path}")
    print(f"Dry run      : {args.dry_run}")

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
        help="Chemin vers volcano_multi.npz.",
    )

    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Nom court du dataset. Exemple : full_stride5.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="models/sweeps_cnn_bilstm",
        help="Dossier racine des sorties locales.",
    )

    parser.add_argument(
        "--train-script",
        type=str,
        default="scripts/train_cnn_bilstm.py",
        help="Script d'entraînement CNN-BiLSTM.",
    )

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
    )

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)

    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Active AMP CUDA si supporté.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre de runs pour un test rapide.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les commandes sans les exécuter.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Arrêter au premier run échoué.",
    )

    args = parser.parse_args()
    main(args)