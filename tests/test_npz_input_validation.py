import numpy as np
import pytest


def validate_npz_arrays(X, expected_seq_len=120, expected_n_features=992):
    if X.ndim != 3:
        raise ValueError(f"X doit être 3D, reçu : {X.shape}")

    if X.shape[1] != expected_seq_len:
        raise ValueError(f"seq_len invalide : {X.shape[1]}")

    if X.shape[2] != expected_n_features:
        raise ValueError(f"n_features invalide : {X.shape[2]}")

    if np.isnan(X).any():
        raise ValueError("NaN détecté dans X")

    if np.isinf(X).any():
        raise ValueError("Inf détecté dans X")


def test_valid_npz_shape():
    X = np.zeros((4, 120, 992), dtype=np.float32)
    validate_npz_arrays(X)


def test_invalid_feature_count():
    X = np.zeros((4, 120, 991), dtype=np.float32)

    with pytest.raises(ValueError, match="n_features invalide"):
        validate_npz_arrays(X)


def test_nan_rejected():
    X = np.zeros((4, 120, 992), dtype=np.float32)
    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        validate_npz_arrays(X)