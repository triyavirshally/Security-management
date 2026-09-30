import pytest
import numpy as np
from ml.preprocessor import SensorDataPreprocessor, TelemetryValidationError

def test_clean_telemetry_valid():
    raw = {'accel_x': 0.1, 'accel_y': -0.2, 'accel_z': 0.98, 'pressure': 500.0, 'proximity': 40.0}
    cleaned = SensorDataPreprocessor.clean_telemetry(raw)
    assert cleaned['accel_x'] == 0.1
    assert cleaned['accel_z'] == 0.98
    assert cleaned['pressure'] == 500.0

def test_clean_telemetry_missing_key():
    raw = {'accel_x': 0.1, 'accel_y': -0.2, 'accel_z': 0.98}
    with pytest.raises(TelemetryValidationError, match="Missing required telemetry key"):
        SensorDataPreprocessor.clean_telemetry(raw)

def test_clean_telemetry_out_of_bounds():
    raw = {'accel_x': 100.0, 'accel_y': 0.0, 'accel_z': 1.0, 'pressure': 500.0, 'proximity': 40.0}
    with pytest.raises(TelemetryValidationError, match="out of bounds"):
        SensorDataPreprocessor.clean_telemetry(raw)

def test_clean_telemetry_nan_rejection():
    raw = {'accel_x': float('nan'), 'accel_y': 0.0, 'accel_z': 1.0, 'pressure': 500.0, 'proximity': 40.0}
    with pytest.raises(TelemetryValidationError, match="NaN or Infinite"):
        SensorDataPreprocessor.clean_telemetry(raw)

def test_compute_magnitude():
    mag = SensorDataPreprocessor.compute_magnitude(3.0, 4.0, 0.0)
    assert abs(mag - 5.0) < 1e-5

def test_extract_features():
    cur = {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 1.0, 'pressure': 500.0, 'proximity': 40.0}
    prev = {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 1.0, 'pressure': 500.0, 'proximity': 40.0}
    feats = SensorDataPreprocessor.extract_features(cur, prev)
    assert len(feats) == 9
    assert feats[3] == 1.0  # magnitude
    assert feats[4] == 0.0  # delta_magnitude
