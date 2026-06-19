# scripts/sweep_targeted_regression.py

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BEST_CNN_BILSTM_PROFILE = {
    "profile_name": "cnn_bilstm_best_sweep",
    "conv_channels": 192,
    "lstm_hidden": 192,
    "lstm_layers": 2,
}

OPTIMIZATION_PROFILE = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "dropout": 0.10,
    "input_noise_std": 0.005,
    "batch_size": 32,
}

LOSS_WEIGHTINGS = ["none", "early_warning", "balanced_horizon"]


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


def run_command(cmd: list[str], dry_run: bool) -> int:
    print("\n" + "=" * 120)
    print("Commande :")
    print(" ".join(cmd))
    print("=" * 120)

    if dry_run:
        return 0

    completed = subprocess.run(cmd)
    return completed.returncode


def write_manifest_entry(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_runs(args):
    datasets = [("full", args.full_input_npz)]
    if args.core_input_npz:
        datasets.append(("core", args.core_input_npz))

    runs = []
    for dataset_name, input_npz in datasets:
        for loss_weighting in LOSS_WEIGHTINGS:
            run_name = (
                f"cnn_bilstm_regression_{args.dataset_tag}_{dataset_name}_"
                f"{BEST_CNN_BILSTM_PROFILE['profile_name']}_{loss_weighting}"
            )
            output_dir = Path(args.output_root) / safe_name(run_name)

            cmd = [
                sys.executable,
                args.train_script,
                "--input-npz", input_npz,
                "--output-dir", str(output_dir),
                "--epochs", str(args.epochs),
                "--batch-size", str(OPTIMIZATION_PROFILE["batch_size"]),
                "--learning-rate", str(OPTIMIZATION_PROFILE["learning_rate"]),
                "--weight-decay", str(OPTIMIZATION_PROFILE["weight_decay"]),
                "--dropout", str(OPTIMIZATION_PROFILE["dropout"]),
                "--input-noise-std", str(OPTIMIZATION_PROFILE["input_noise_std"]),
                "--conv-channels", str(BEST_CNN_BILSTM_PROFILE["conv_channels"]),
                "--lstm-hidden", str(BEST_CNN_BILSTM_PROFILE["lstm_hidden"]),
                "--lstm-layers", str(BEST_CNN_BILSTM_PROFILE["lstm_layers"]),
                "--loss-weighting", loss_weighting,
                "--early-stopping-metric", args.early_stopping_metric,
                "--early-stopping-patience", str(args.early_stopping_patience),
                "--early-stopping-min-delta", str(args.early_stopping_min_delta),
                "--lr-patience", str(args.lr_patience),
                "--grad-clip", str(args.grad_clip),
                "--huber-beta", str(args.huber_beta),
                "--max-horizon-hours", str(args.max_horizon_hours),
                "--run-name", run_name,
                "--use-mlflow",
            ]

            if args.use_amp:
                cmd.append("--use-amp")

            runs.append({
                "task": "regression",
                "model_family": "cnn_bilstm",
                "dataset_name": dataset_name,
                "input_npz": input_npz,
                "run_name": run_name,
                "output_dir": str(output_dir),
                "loss_weighting": loss_weighting,
                "command": cmd,
            })

    if args.limit is not None:
        runs = runs[: args.limit]

    return runs


def main(args):
    base_output = Path(args.output_root)
    base_output.mkdir(parents=True, exist_ok=True)
    manifest_path = base_output / f"sweep_manifest_targeted_regression_{safe_name(args.dataset_tag)}.jsonl"

    planned_runs = build_runs(args)
    sweep_start = datetime.now().isoformat(timespec="seconds")

    print("\nPlan de sweep ciblé régression")
    print(f"Full NPZ       : {args.full_input_npz}")
    print(f"Core NPZ       : {args.core_input_npz}")
    print(f"Dataset tag    : {args.dataset_tag}")
    print(f"Nombre de runs : {len(planned_runs)}")
    print(f"Manifest       : {manifest_path}")
    print(f"Dry run        : {args.dry_run}")

    for index, payload in enumerate(planned_runs, start=1):
        payload["sweep_index"] = index
        payload["sweep_total"] = len(planned_runs)
        payload["sweep_start"] = sweep_start

        print("\n" + "#" * 120)
        print(f"RUN {index}/{len(planned_runs)} — {payload['run_name']}")
        print("#" * 120)

        write_manifest_entry(manifest_path, {**payload, "status": "planned"})
        return_code = run_command(payload["command"], dry_run=args.dry_run)
        status = "success" if return_code == 0 else "failed"
        write_manifest_entry(manifest_path, {**payload, "status": status, "return_code": return_code})

        if return_code != 0 and args.stop_on_error:
            raise RuntimeError(f"Run échoué avec code {return_code} : {payload['run_name']}")

    print("\nSweep ciblé régression terminé.")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--full-input-npz", type=str, required=True)
    parser.add_argument("--core-input-npz", type=str, default=None)
    parser.add_argument("--dataset-tag", type=str, default="stride5")
    parser.add_argument("--output-root", type=str, default="models/sweeps_targeted_regression")
    parser.add_argument("--train-script", type=str, default="scripts/train_cnn_bilstm_regression_Vfinetuning.py")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-metric", type=str, default="business_score")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--max-horizon-hours", type=float, default=48.0)
    parser.add_argument("--use-amp", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()
    main(args)
