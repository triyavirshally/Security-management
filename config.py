import os

class Config:
    """Application configuration parameters for Smart Belonging Security System."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smart-belonging-security-secret-key-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///smart_security.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Machine Learning / Anomaly Detection Parameters
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    ISOLATION_FOREST_CONTAMINATION = 0.08  # Expected anomaly rate (8%)
    ISOLATION_FOREST_TREES = 100
    RANDOM_SEED = 42

    # Sensor Baseline Thresholds & Multi-Sensor Guardrails
    ACCEL_NORMAL_MAX_MAGNITUDE = 1.25   # Baseline stationary g-force magnitude (~1.0g gravity + small vibration)
    ACCEL_ANOMALY_THRESHOLD = 0.35       # Delta magnitude shift threshold
    PRESSURE_DROP_CRITICAL_RATIO = 0.40  # 40% sudden weight reduction
    PROXIMITY_ALERT_DELTA = 15.0         # 15 cm change in distance

    # Security Alert Threat Levels
    THREAT_LEVEL_LOW = 'LOW'
    THREAT_LEVEL_MEDIUM = 'MEDIUM'
    THREAT_LEVEL_HIGH = 'HIGH'
    THREAT_LEVEL_CRITICAL = 'CRITICAL'
