import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TelemetryValidationError(ValueError):
    """Custom exception raised when telemetry data violates safety or range constraints."""
    pass

class SensorDataPreprocessor:
    """
    Data Preprocessing & Feature Engineering Engine for IoT Sensor Telemetry.
    Validates sensor payloads, filters signal noise, and extracts mathematical features
    for the Isolation Forest Anomaly Detection Model.
    """

    # Physical Sensor Guardrails
    ACCEL_RANGE = (-50.0, 50.0)      # g-force limits
    PRESSURE_RANGE = (0.0, 5000.0)   # grams / force units
    PROXIMITY_RANGE = (0.0, 500.0)   # centimeters

    @staticmethod
    def clean_telemetry(raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate and sanitize incoming raw IoT telemetry fields.
        Applies input validation boundaries and rejects corrupted/NaN values.

        :param raw_data: Dictionary with accel_x, accel_y, accel_z, pressure, proximity keys.
        :return: Sanitized dictionary of floating-point sensor values.
        :raises TelemetryValidationError: If required fields are missing, invalid type, or out of bound.
        """
        required_keys = ['accel_x', 'accel_y', 'accel_z', 'pressure', 'proximity']
        for key in required_keys:
            if key not in raw_data:
                raise TelemetryValidationError(f"Missing required telemetry key: '{key}'")

        cleaned = {}
        for key in required_keys:
            val = raw_data[key]
            try:
                val_float = float(val)
            except (ValueError, TypeError):
                raise TelemetryValidationError(f"Field '{key}' has non-numeric value: {val}")

            if np.isnan(val_float) or np.isinf(val_float):
                raise TelemetryValidationError(f"Field '{key}' contains NaN or Infinite value.")

            cleaned[key] = val_float

        # Enforce physical guardrail ranges
        for axis in ['accel_x', 'accel_y', 'accel_z']:
            if not (SensorDataPreprocessor.ACCEL_RANGE[0] <= cleaned[axis] <= SensorDataPreprocessor.ACCEL_RANGE[1]):
                raise TelemetryValidationError(f"Axis '{axis}' value {cleaned[axis]} out of bounds {SensorDataPreprocessor.ACCEL_RANGE}")

        if not (SensorDataPreprocessor.PRESSURE_RANGE[0] <= cleaned['pressure'] <= SensorDataPreprocessor.PRESSURE_RANGE[1]):
            raise TelemetryValidationError(f"Pressure value {cleaned['pressure']} out of bounds {SensorDataPreprocessor.PRESSURE_RANGE}")

        if not (SensorDataPreprocessor.PROXIMITY_RANGE[0] <= cleaned['proximity'] <= SensorDataPreprocessor.PROXIMITY_RANGE[1]):
            raise TelemetryValidationError(f"Proximity value {cleaned['proximity']} out of bounds {SensorDataPreprocessor.PROXIMITY_RANGE}")

        return cleaned

    @staticmethod
    def compute_magnitude(accel_x: float, accel_y: float, accel_z: float) -> float:
        """
        Calculate total 3D acceleration vector magnitude: M = sqrt(x^2 + y^2 + z^2).
        """
        return float(np.sqrt(accel_x**2 + accel_y**2 + accel_z**2))

    @staticmethod
    def extract_features(current_read: Dict[str, float], previous_read: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Extract feature vector for model prediction:
        [accel_x, accel_y, accel_z, magnitude, delta_magnitude, pressure, pressure_delta, proximity, proximity_delta]

        :param current_read: Sanitized current sensor reading dictionary.
        :param previous_read: Optional previous sensor reading dictionary for delta calculations.
        :return: 1D NumPy float array with 9 engineered features.
        """
        magnitude = SensorDataPreprocessor.compute_magnitude(
            current_read['accel_x'],
            current_read['accel_y'],
            current_read['accel_z']
        )

        if previous_read:
            prev_mag = SensorDataPreprocessor.compute_magnitude(
                previous_read['accel_x'],
                previous_read['accel_y'],
                previous_read['accel_z']
            )
            delta_mag = abs(magnitude - prev_mag)
            pressure_delta = abs(current_read['pressure'] - previous_read['pressure'])
            proximity_delta = abs(current_read['proximity'] - previous_read['proximity'])
        else:
            delta_mag = 0.0
            pressure_delta = 0.0
            proximity_delta = 0.0

        features = [
            current_read['accel_x'],
            current_read['accel_y'],
            current_read['accel_z'],
            magnitude,
            delta_mag,
            current_read['pressure'],
            pressure_delta,
            current_read['proximity'],
            proximity_delta
        ]

        return np.array(features, dtype=np.float64)
