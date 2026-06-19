# scripts/run_leave_one_eruption_out.py

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def infer_period_type(row) -> str:
    if "period_type" in row.index and pd.notna(row["period_type"]):
        value = str(row["period_type"]).strip().lower()
        if value in {"quiet", "calm", "calme", "background", "non_eruptive"}:
            return "quiet"
        return "eruption"
    eruption_id = str(row.get("eruption_id", "")).lower()
    return "quiet" if eruption_id.startswith("quiet_") else "eruption"


def build_fold_metadata(meta: pd.DataFrame, test_event_id: str, val_event_id: str | None, task: str) -> pd.DataFrame:
    out = meta.copy()

    if task == "regression":
        out = out[out.apply(lambda row: infer_period_type(row) == "eruption", axis=1)].copy()

    out["split"] = "train"
    out.loc[out["eruption_id"].astype(str) == str(test_event_id), "split"] = "test"

    if val_event_id:
        out.loc[out["eruption_id"].astype(str) == str(val_event_id), "split"] = "val"
    else:
        candidates = [
            str(x) for x in out["eruption_id"].tolist()
            if str(x) != str(test_event_id)
        ]
        if not candidates:
            raise ValueError("Impossible de créer un split val : aucun événement disponible hors test.")
        out.loc[out["eruption_id"].astype(str) == candidates[0], "split"] = "val"

    return out


def select_folds(meta: pd.DataFrame, task: str, include_quiet_as_test: bool) -> list[str]:
    event_ids = []
    for _, row in meta.iterrows():
        period_type = infer_period_type(row)
        eruption_id = str(row["eruption_id"])
        if task == "regression" and period_type != "eruption":
            continue
        if task == "classification" and period_type == "quiet" and not include_quiet_as_test:
            continue
        event_ids.append(eruption_id)
    return event_ids


def build_preprocess_command(args, fold_meta_path: Path, fold_preprocess_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        args.preprocess_script,
        "--metadata", str(fold_meta_path),
        "--output-dir", str(fold_preprocess_dir),
        "--output-name", args.output_name,
        "--feature-window-minutes", str(args.feature_window_minutes),
        "--seq-len", str(args.seq_len),
        "--sequence-stride", str(args.sequence_stride),
        "--max-horizon-hours", str(args.max_horizon_hours),
        "--entropy-bins", str(args.entropy_bins),
    ]

    if args.upload_preprocessing_to_s3:
        cmd.append("--upload-to-s3")
    if args.s3_bucket:
        cmd.extend(["--s3-bucket", args.s3_bucket])
    if args.s3_csv_prefix:
        cmd.extend(["--s3-csv-prefix", args.s3_csv_prefix])
    if args.local_csv_dir:
        cmd.extend(["--local-csv-dir", args.local_csv_dir])

    return cmd


def build_train_command(args, input_npz: Path, fold_model_dir: Path, fold_name: str) -> list[str]:
    cmd = [
        sys.executable,
        args.train_script,
        "--input-npz", str(input_npz),
        "--output-dir", str(fold_model_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--early-stopping-metric", args.early_stopping_metric,
        "--lr-patience", str(args.lr_patience),
        "--grad-clip", str(args.grad_clip),
        "--run-name", f"{args.run_name_prefix}_{fold_name}",
    ]
    
    if args.use_mlflow:
        cmd.append("--use-mlflow")

    if args.task == "regression":
        cmd.extend([
            "--conv-channels", str(args.conv_channels),
            "--lstm-hidden", str(args.lstm_hidden),
            "--lstm-layers", str(args.lstm_layers),
            "--input-noise-std", str(args.input_noise_std),
            "--loss-weighting", args.loss_weighting,
            "--huber-beta", str(args.huber_beta),
            "--max-horizon-hours", str(args.max_horizon_hours),
        ])
    else:
        cmd.extend([
            "--d-model", str(args.d_model),
            "--nhead", str(args.nhead),
            "--num-layers", str(args.num_layers),
            "--dim-feedforward", str(args.dim_feedforward),
            "--input-noise-std", str(args.input_noise_std),
            "--class-weighting", args.class_weighting,
            "--label-smoothing", str(args.label_smoothing),
            "--n-classes", str(args.n_classes),
        ])

    if args.use_amp:
        cmd.append("--use-amp")
    if args.cpu:
        cmd.append("--cpu")

    return cmd


def main(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"loo_manifest_{safe_name(args.task)}_{safe_name(args.dataset_tag)}.jsonl"

    meta = pd.read_csv(args.metadata)
    if "eruption_id" not in meta.columns:
        raise ValueError("Le metadata doit contenir une colonne eruption_id.")

    folds = select_folds(
        meta=meta,
        task=args.task,
        include_quiet_as_test=args.include_quiet_as_test,
    )

    if args.limit is not None:
        folds = folds[: args.limit]

    print("\nValidation leave-one-eruption-out")
    print(f"Task           : {args.task}")
    print(f"Metadata       : {args.metadata}")
    print(f"Nombre folds   : {len(folds)}")
    print(f"Output root    : {output_root}")
    print(f"Manifest       : {manifest_path}")
    print(f"Dry run        : {args.dry_run}")

    start = datetime.now().isoformat(timespec="seconds")

    for idx, test_event_id in enumerate(folds, start=1):
        if idx < args.start_index:
            print(f"Fold {idx} ignoré car start_index={args.start_index}")
            continue
        
        fold_name = f"fold_{idx:02d}_test_{safe_name(test_event_id)}"
        fold_dir = output_root / fold_name
        fold_meta_path = fold_dir / "metadata_fold.csv"
        fold_preprocess_dir = fold_dir / "preprocessing"
        fold_model_dir = fold_dir / "model"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_event_id = None
        candidates = [x for x in folds if x != test_event_id]
        if candidates:
            val_event_id = candidates[(idx - 1) % len(candidates)]

        fold_meta = build_fold_metadata(
            meta=meta,
            test_event_id=test_event_id,
            val_event_id=val_event_id,
            task=args.task,
        )
        fold_meta.to_csv(fold_meta_path, index=False)

        preprocess_cmd = build_preprocess_command(args, fold_meta_path, fold_preprocess_dir)
        train_cmd = build_train_command(
            args=args,
            input_npz=fold_preprocess_dir / args.output_name,
            fold_model_dir=fold_model_dir,
            fold_name=fold_name,
        )

        payload = {
            "task": args.task,
            "dataset_tag": args.dataset_tag,
            "fold_index": idx,
            "fold_total": len(folds),
            "test_event_id": test_event_id,
            "val_event_id": val_event_id,
            "fold_name": fold_name,
            "fold_dir": str(fold_dir),
            "metadata_fold": str(fold_meta_path),
            "preprocess_cmd": preprocess_cmd,
            "train_cmd": train_cmd,
            "start": start,
        }

        print("\n" + "#" * 120)
        print(f"FOLD {idx}/{len(folds)} — test={test_event_id} | val={val_event_id}")
        print("#" * 120)

        write_jsonl(manifest_path, {**payload, "stage": "planned"})

        rc_preprocess = run_command(preprocess_cmd, dry_run=args.dry_run)
        write_jsonl(manifest_path, {**payload, "stage": "preprocess", "return_code": rc_preprocess})
        if rc_preprocess != 0:
            if args.stop_on_error:
                raise RuntimeError(f"Préprocessing échoué pour {fold_name}, code={rc_preprocess}")
            continue

        rc_train = run_command(train_cmd, dry_run=args.dry_run)
        write_jsonl(manifest_path, {**payload, "stage": "train", "return_code": rc_train})
        if rc_train != 0 and args.stop_on_error:
            raise RuntimeError(f"Training échoué pour {fold_name}, code={rc_train}")

    print("\nLeave-one-eruption-out terminé.")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=["regression", "classification"])
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--dataset-tag", type=str, default="loo")
    parser.add_argument("--output-root", type=str, default="reports/leave_one_eruption_out")
    parser.add_argument("--preprocess-script", type=str, required=True)
    parser.add_argument("--train-script", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="volcano_multi.npz")

    # Préprocessing
    parser.add_argument("--feature-window-minutes", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--max-horizon-hours", type=float, default=48.0)
    parser.add_argument("--entropy-bins", type=int, default=20)
    parser.add_argument("--upload-preprocessing-to-s3", action="store_true")
    parser.add_argument("--s3-bucket", type=str, default=None)
    parser.add_argument("--s3-csv-prefix", type=str, default=None)
    parser.add_argument("--local-csv-dir", type=str, default=None)

    # Training commun
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--input-noise-std", type=float, default=0.005)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-metric", type=str, default="business_score")
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--run-name-prefix", type=str, default="loo")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-mlflow",action="store_true",help="Transmet --use-mlflow au script d'entraînement pour logger chaque fold dans MLflow.",)

    # Régression CNN-BiLSTM
    parser.add_argument("--conv-channels", type=int, default=192)
    parser.add_argument("--lstm-hidden", type=int, default=192)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--loss-weighting", type=str, default="none")
    parser.add_argument("--huber-beta", type=float, default=1.0)

    # Classification CNN-Transformer
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--class-weighting", type=str, default="early_warning_priority")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--include-quiet-as-test", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index",type=int,default=1,help="Index du premier fold à exécuter. 1 = premier fold.",)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()
    main(args)
