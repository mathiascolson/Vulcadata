# scripts/validate_npz_dataset.py

import argparse
import json
from pathlib import Path

import numpy as np


def finite_check(name: str, arr: np.ndarray) -> list[str]:
    errors = []
    if np.isnan(arr).any():
        errors.append(f"{name} contient des NaN")
    if np.isinf(arr).any():
        errors.append(f"{name} contient des Inf")
    return errors


def validate_npz(path: Path, task: str, n_classes: int, max_horizon_hours: float) -> dict:
    data = np.load(path, allow_pickle=True)

    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    errors = []
    warnings = []

    for key in required:
        if key not in data:
            errors.append(f"Clé absente : {key}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    report = {
        "path": str(path),
        "task": task,
        "splits": {},
        "valid": True,
        "errors": errors,
        "warnings": warnings,
    }

    n_features_ref = None
    seq_len_ref = None

    for split in ["train", "val", "test"]:
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]

        report["splits"][split] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "X_dtype": str(X.dtype),
            "y_dtype": str(y.dtype),
        }

        if X.ndim != 3:
            errors.append(f"X_{split} doit être en 3D (N,T,F), reçu {X.shape}")
        else:
            if seq_len_ref is None:
                seq_len_ref = X.shape[1]
                n_features_ref = X.shape[2]
            elif X.shape[1] != seq_len_ref or X.shape[2] != n_features_ref:
                errors.append(
                    f"X_{split} a une forme incohérente : {X.shape}, attendu T={seq_len_ref}, F={n_features_ref}"
                )

        if y.ndim != 1:
            errors.append(f"y_{split} doit être en 1D, reçu {y.shape}")

        if X.shape[0] != y.shape[0]:
            errors.append(f"N incohérent pour {split}: X={X.shape[0]}, y={y.shape[0]}")

        errors.extend(finite_check(f"X_{split}", X))
        errors.extend(finite_check(f"y_{split}", y.astype(float)))

        if task == "classification":
            unique, counts = np.unique(y, return_counts=True)
            class_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
            report["splits"][split]["class_counts"] = class_counts

            bad = unique[(unique < 0) | (unique >= n_classes)]
            if bad.size > 0:
                errors.append(f"Classes invalides dans y_{split}: {bad.tolist()}")

            missing = [cls for cls in range(n_classes) if cls not in unique]
            if missing:
                warnings.append(f"Classes absentes dans {split}: {missing}")

        elif task == "regression":
            report["splits"][split]["y_min"] = float(np.nanmin(y))
            report["splits"][split]["y_max"] = float(np.nanmax(y))
            report["splits"][split]["y_mean"] = float(np.nanmean(y))

            if (y < 0).any():
                errors.append(f"y_{split} contient des valeurs négatives")
            if max_horizon_hours > 0 and (y > max_horizon_hours + 1e-6).any():
                warnings.append(f"y_{split} contient des valeurs > {max_horizon_hours}h")
        else:
            errors.append(f"task invalide : {task}")

    if "feature_names" in data and n_features_ref is not None:
        feature_names = data["feature_names"]
        report["feature_names_count"] = int(len(feature_names))
        if len(feature_names) != n_features_ref:
            errors.append(f"feature_names contient {len(feature_names)} noms, mais X a {n_features_ref} features")
    else:
        warnings.append("feature_names absent du NPZ")

    report["valid"] = len(errors) == 0
    return report


def main(args):
    path = Path(args.input_npz)
    output_path = Path(args.output_json) if args.output_json else path.parent / "npz_validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = validate_npz(
        path=path,
        task=args.task,
        n_classes=args.n_classes,
        max_horizon_hours=args.max_horizon_hours,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport écrit : {output_path}")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["regression", "classification"])
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--max-horizon-hours", type=float, default=48.0)
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()
    main(args)
