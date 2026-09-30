import os
import joblib
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from ml.preprocessor import SensorDataPreprocessor
from config import Config

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    AI-Based Anomaly Detection Engine using Isolation Forest.
    Learns normal sensor telemetry patterns of unattended belongings (laptops, bags, chargers)
    and identifies unusual interactions, movements, or displacement.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or Config.MODEL_DIR
        self.model_path = os.path.join(self.model_dir, 'isolation_forest.joblib')
        self.model: Optional[IsolationForest] = None
        self.is_trained = False
        self._ensure_model_directory()
        self.load_model()

    def _ensure_model_directory(self):
        """Create directory for saved models if it does not exist."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def generate_synthetic_training_data(self, n_normal: int = 1000, n_anomalies: int = 80) -> np.ndarray:
        """
        Generate synthetic training dataset mimicking hostel belonging environments.
        - Normal telemetry: Stationary on desk, subtle room vibration, constant pressure, static proximity.
        - Anomaly telemetry: Sudden lifting, rapid displacement, unplugging charger, pressure drop.

        :param n_normal: Number of normal state baseline samples.
        :param n_anomalies: Number of anomaly state samples.
        :return: 2D NumPy array of shape (N, 9) containing feature vectors.
        """
        np.random.seed(Config.RANDOM_SEED)

        # 1. Stationary Baseline Data (Normal operation)
        # Accel around (0, 0, 1.0g gravity) with micro-vibrations (+/- 0.03g)
        accel_x_norm = np.random.normal(0.0, 0.02, n_normal)
        accel_y_norm = np.random.normal(0.0, 0.02, n_normal)
        accel_z_norm = np.random.normal(0.98, 0.03, n_normal)
        pressure_norm = np.random.normal(500.0, 10.0, n_normal)  # ~500g belonging weight
        proximity_norm = np.random.normal(45.0, 2.0, n_normal)   # ~45cm distance to obstacle

        X_normal = []
        for i in range(n_normal):
            cur = {
                'accel_x': accel_x_norm[i],
                'accel_y': accel_y_norm[i],
                'accel_z': accel_z_norm[i],
                'pressure': max(0.0, pressure_norm[i]),
                'proximity': max(0.0, proximity_norm[i])
            }
            prev = {
                'accel_x': accel_x_norm[i-1] if i > 0 else accel_x_norm[i],
                'accel_y': accel_y_norm[i-1] if i > 0 else accel_y_norm[i],
                'accel_z': accel_z_norm[i-1] if i > 0 else accel_z_norm[i],
                'pressure': max(0.0, pressure_norm[i-1]) if i > 0 else max(0.0, pressure_norm[i]),
                'proximity': max(0.0, proximity_norm[i-1]) if i > 0 else max(0.0, proximity_norm[i])
            }
            X_normal.append(SensorDataPreprocessor.extract_features(cur, prev))

        # 2. Anomaly Data (Theft attempt / unexpected movement)
        accel_x_anom = np.random.uniform(-3.5, 3.5, n_anomalies)
        accel_y_anom = np.random.uniform(-3.5, 3.5, n_anomalies)
        accel_z_anom = np.random.uniform(-1.0, 4.0, n_anomalies)
        pressure_anom = np.random.uniform(0.0, 100.0, n_anomalies) # Weight removed
        proximity_anom = np.random.uniform(5.0, 150.0, n_anomalies) # Approach or sudden move

        X_anom = []
        for i in range(n_anomalies):
            cur = {
                'accel_x': accel_x_anom[i],
                'accel_y': accel_y_anom[i],
                'accel_z': accel_z_anom[i],
                'pressure': pressure_anom[i],
                'proximity': proximity_anom[i]
            }
            prev = {
                'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 1.0,
                'pressure': 500.0, 'proximity': 45.0
            }
            X_anom.append(SensorDataPreprocessor.extract_features(cur, prev))

        X_all = np.vstack([X_normal, X_anom])
        return X_all

    def train(self, X_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train or retrain the Isolation Forest model.

        :param X_data: Optional training data matrix. If None, generates synthetic dataset.
        :return: Metrics dictionary with training metadata.
        """
        if X_data is None or len(X_data) == 0:
            logger.info("Generating synthetic training data for Isolation Forest baseline...")
            X_data = self.generate_synthetic_training_data()

        logger.info(f"Training Isolation Forest with {len(X_data)} samples...")

        self.model = IsolationForest(
            n_estimators=Config.ISOLATION_FOREST_TREES,
            contamination=Config.ISOLATION_FOREST_CONTAMINATION,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1
        )
        self.model.fit(X_data)
        self.is_trained = True

        # Save trained model artifact
        joblib.dump(self.model, self.model_path)
        logger.info(f"Isolation Forest model persisted successfully to {self.model_path}")

        return {
            'sample_count': len(X_data),
            'n_estimators': Config.ISOLATION_FOREST_TREES,
            'contamination': Config.ISOLATION_FOREST_CONTAMINATION,
            'status': 'TRAINED'
        }

    def load_model(self) -> bool:
        """Load trained model from disk if available."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info("Loaded pre-trained Isolation Forest model from disk.")
                return True
            except Exception as e:
                logger.error(f"Failed to load model file {self.model_path}: {e}")
                self.is_trained = False
                return False
        else:
            logger.warning("No pre-trained model file found on disk.")
            self.is_trained = False
            return False

    def predict(self, feature_vector: np.ndarray) -> Tuple[float, float, int]:
        """
        Perform anomaly prediction on a single 9-feature sensor vector.

        :param feature_vector: 1D NumPy array (shape 9,) or 2D (shape 1, 9).
        :return: Tuple of (raw_decision_function_score, normalized_risk_score_0_100, binary_label [-1=anomaly, 1=normal]).
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Fallback error boundary if model is not yet trained
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained! Using heuristic fallback anomaly scoring.")
            return self._heuristic_fallback(feature_vector)

        try:
            raw_score = float(self.model.decision_function(feature_vector)[0])
            pred_label = int(self.model.predict(feature_vector)[0])

            # Isolation Forest decision_function returns negative values for anomalies, positive for normal.
            # Convert raw score to 0.0 - 100.0 risk score (Higher score = higher security threat)
            # Typically score ranges from ~ -0.3 to +0.3
            risk_score = float(np.clip((0.15 - raw_score) * 200.0, 0.0, 100.0))

            return raw_score, risk_score, pred_label

        except Exception as e:
            logger.error(f"ML Prediction Exception: {e}. Degrading gracefully to heuristic fallback.")
            return self._heuristic_fallback(feature_vector)

    def _heuristic_fallback(self, feature_vector: np.ndarray) -> Tuple[float, float, int]:
        """
        Graceful Error Boundary Fallback: Computes baseline threat score when ML model is unavailable.
        """
        # Feature order: [accel_x, accel_y, accel_z, magnitude, delta_mag, pressure, pressure_delta, proximity, proximity_delta]
        vec = feature_vector[0]
        delta_mag = vec[4]
        pressure_delta = vec[6]
        proximity_delta = vec[8]

        # Heuristic score calculation
        anomaly_index = (delta_mag * 40.0) + (pressure_delta * 0.1) + (proximity_delta * 1.5)
        risk_score = float(np.clip(anomaly_index, 0.0, 100.0))

        raw_score = 0.1 - (risk_score / 200.0)
        binary_label = -1 if risk_score > 50.0 else 1

        return raw_score, risk_score, binary_label
