import pytest
import numpy as np
from ml.anomaly_detector import AnomalyDetector

def test_train_and_predict(trained_detector):
    assert trained_detector.is_trained
    assert trained_detector.model is not None

    # Normal sample (accel ~ 1g, pressure ~ 500g)
    normal_vector = np.array([0.01, 0.01, 0.98, 0.98, 0.0, 500.0, 0.0, 45.0, 0.0])
    raw_score, risk_score, label = trained_detector.predict(normal_vector)

    assert label == 1 or risk_score < 50.0

    # Severe Anomaly sample (huge acceleration spike, 0 pressure)
    anom_vector = np.array([4.5, -3.2, 5.0, 7.5, 6.5, 0.0, 500.0, 150.0, 105.0])
    raw_score_anom, risk_score_anom, label_anom = trained_detector.predict(anom_vector)

    assert risk_score_anom > risk_score

def test_untrained_fallback():
    detector = AnomalyDetector(model_dir="/invalid_non_existent_dir_123")
    detector.is_trained = False
    detector.model = None

    vec = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 500.0, 0.0, 45.0, 0.0])
    raw_score, risk_score, label = detector.predict(vec)

    assert isinstance(risk_score, float)
    assert 0.0 <= risk_score <= 100.0
