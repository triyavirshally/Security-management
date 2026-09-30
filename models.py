from datetime import datetime, timezone
from database import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    """
    User Model: Represents hostel students / device owners.
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    devices = db.relationship('Device', backref='owner', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str):
        """Hash and set user password securely."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify user password against stored hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """Serialize User instance to JSON-friendly dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Device(db.Model):
    """
    Device Model: Represents an IoT security node attached to a student belonging.
    Belonging types: laptop, bag, charger, mobile, wallet, etc.
    """
    __tablename__ = 'devices'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    device_name = db.Column(db.String(100), nullable=False)
    belonging_type = db.Column(db.String(50), nullable=False, default='laptop')
    sensor_mac = db.Column(db.String(50), unique=True, nullable=False, index=True)
    is_armed = db.Column(db.Boolean, default=True, nullable=False)
    sensitivity_level = db.Column(db.String(20), default='MEDIUM', nullable=False)  # LOW, MEDIUM, HIGH
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    telemetry_records = db.relationship('SensorTelemetry', backref='device', lazy=True, cascade="all, delete-orphan")
    alerts = db.relationship('AnomalyAlert', backref='device', lazy=True, cascade="all, delete-orphan")
    feedbacks = db.relationship('UserFeedback', backref='device', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        """Serialize Device instance to JSON-friendly dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'device_name': self.device_name,
            'belonging_type': self.belonging_type,
            'sensor_mac': self.sensor_mac,
            'is_armed': self.is_armed,
            'sensitivity_level': self.sensitivity_level,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SensorTelemetry(db.Model):
    """
    Sensor Telemetry Model: Ingested IoT sensor data stream.
    Captures Accelerometer (3-axis), Pressure sensor, and Proximity sensor.
    """
    __tablename__ = 'sensor_telemetry'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('devices.id'), nullable=False, index=True)
    accel_x = db.Column(db.Float, nullable=False)
    accel_y = db.Column(db.Float, nullable=False)
    accel_z = db.Column(db.Float, nullable=False)
    pressure = db.Column(db.Float, nullable=False)     # Force / weight unit (grams/N)
    proximity = db.Column(db.Float, nullable=False)    # Distance unit (cm)
    magnitude = db.Column(db.Float, nullable=False)    # Composite vector magnitude sqrt(x^2+y^2+z^2)
    delta_magnitude = db.Column(db.Float, default=0.0) # Delta from previous sample
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    alerts = db.relationship('AnomalyAlert', backref='telemetry', lazy=True)

    def to_dict(self):
        """Serialize SensorTelemetry instance to JSON-friendly dictionary."""
        return {
            'id': self.id,
            'device_id': self.device_id,
            'accel_x': self.accel_x,
            'accel_y': self.accel_y,
            'accel_z': self.accel_z,
            'pressure': self.pressure,
            'proximity': self.proximity,
            'magnitude': self.magnitude,
            'delta_magnitude': self.delta_magnitude,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class AnomalyAlert(db.Model):
    """
    Anomaly Alert Model: Triggered security events flagged by Isolation Forest & Decision Logic.
    """
    __tablename__ = 'anomaly_alerts'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('devices.id'), nullable=False, index=True)
    telemetry_id = db.Column(db.Integer, db.ForeignKey('sensor_telemetry.id'), nullable=True)
    anomaly_score = db.Column(db.Float, nullable=False) # Raw Isolation Forest decision function score
    risk_score = db.Column(db.Float, nullable=False)    # Scaled threat score 0.0 - 100.0
    threat_level = db.Column(db.String(20), nullable=False) # LOW, MEDIUM, HIGH, CRITICAL
    status = db.Column(db.String(20), default='ACTIVE', nullable=False) # ACTIVE, RESOLVED, FALSE_POSITIVE
    alert_message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    feedbacks = db.relationship('UserFeedback', backref='alert', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        """Serialize AnomalyAlert instance to JSON-friendly dictionary."""
        return {
            'id': self.id,
            'device_id': self.device_id,
            'telemetry_id': self.telemetry_id,
            'anomaly_score': self.anomaly_score,
            'risk_score': self.risk_score,
            'threat_level': self.threat_level,
            'status': self.status,
            'alert_message': self.alert_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class UserFeedback(db.Model):
    """
    User Feedback Model: Owner response to security alerts.
    Used for adaptive tuning of model thresholds and false-positive reduction.
    """
    __tablename__ = 'user_feedback'

    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey('anomaly_alerts.id'), nullable=False, index=True)
    device_id = db.Column(db.Integer, db.ForeignKey('devices.id'), nullable=False, index=True)
    is_false_positive = db.Column(db.Boolean, nullable=False)
    notes = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Serialize UserFeedback instance to JSON-friendly dictionary."""
        return {
            'id': self.id,
            'alert_id': self.alert_id,
            'device_id': self.device_id,
            'is_false_positive': self.is_false_positive,
            'notes': self.notes,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class ModelMetadata(db.Model):
    """
    Model Metadata Model: Tracks ML model training runs, metrics, and parameters.
    """
    __tablename__ = 'model_metadata'

    id = db.Column(db.Integer, primary_key=True)
    model_version = db.Column(db.String(50), nullable=False)
    contamination = db.Column(db.Float, nullable=False)
    n_estimators = db.Column(db.Integer, nullable=False)
    sample_count = db.Column(db.Integer, nullable=False)
    false_positive_count = db.Column(db.Integer, default=0)
    last_trained_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    status_summary = db.Column(db.String(255), nullable=False)

    def to_dict(self):
        """Serialize ModelMetadata instance to JSON-friendly dictionary."""
        return {
            'id': self.id,
            'model_version': self.model_version,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'sample_count': self.sample_count,
            'false_positive_count': self.false_positive_count,
            'last_trained_at': self.last_trained_at.isoformat() if self.last_trained_at else None,
            'status_summary': self.status_summary
        }
