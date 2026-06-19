import numpy as np


def compute_alert_24h(y_proba, min_class_alert=3, threshold=0.35):
    p_alert = y_proba[:, min_class_alert:].sum(axis=1)
    alert = p_alert >= threshold
    return p_alert, alert


def test_alert_rule_24h():
    y_proba = np.array([
        [0.80, 0.10, 0.05, 0.02, 0.02, 0.01],  # p_alert = 0.05 -> no alert
        [0.20, 0.20, 0.20, 0.15, 0.15, 0.10],  # p_alert = 0.40 -> alert
        [0.30, 0.20, 0.15, 0.20, 0.10, 0.05],  # p_alert = 0.35 -> alert
    ], dtype=np.float32)

    p_alert, alert = compute_alert_24h(
        y_proba=y_proba,
        min_class_alert=3,
        threshold=0.35,
    )

    np.testing.assert_allclose(p_alert, np.array([0.05, 0.40, 0.35]), rtol=1e-6)
    np.testing.assert_array_equal(alert, np.array([False, True, True]))