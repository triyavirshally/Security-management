import sys
import logging
from app import create_app
from database import db
from models import User, Device, SensorTelemetry, AnomalyAlert, UserFeedback, ModelMetadata
from ml.anomaly_detector import AnomalyDetector
from ml.preprocessor import SensorDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SeedData')

def seed():
    """Populate database with realistic hostel security dataset."""
    app = create_app()
    with app.app_context():
        logger.info("Dropping existing tables and recreating fresh database schema...")
        db.drop_all()
        db.create_all()

        # 1. Create Default Hostel Student User
        student = User(username='hostel_student1', email='student1@hostel.edu')
        student.set_password('security123')
        db.session.add(student)
        db.session.flush()

        logger.info(f"Created student user: {student.username} (ID: {student.id})")

        # 2. Register Belongings (Devices)
        devices = [
            Device(
                user_id=student.id,
                device_name="MacBook Pro M2 (Desk)",
                belonging_type="laptop",
                sensor_mac="AA:BB:CC:11:22:33",
                is_armed=True,
                sensitivity_level="HIGH"
            ),
            Device(
                user_id=student.id,
                device_name="Hostel Backpack",
                belonging_type="bag",
                sensor_mac="AA:BB:CC:44:55:66",
                is_armed=True,
                sensitivity_level="MEDIUM"
            ),
            Device(
                user_id=student.id,
                device_name="Fast Charger Adapter",
                belonging_type="charger",
                sensor_mac="AA:BB:CC:77:88:99",
                is_armed=True,
                sensitivity_level="MEDIUM"
            )
        ]

        db.session.add_all(devices)
        db.session.flush()
        logger.info(f"Registered {len(devices)} belongings for protection.")

        # 3. Generate Normal Baseline Telemetry
        detector = AnomalyDetector()
        if not detector.is_trained:
            detector.train()

        macbook = devices[0]

        # Normal readings
        normal_readings = [
            {'accel_x': 0.01, 'accel_y': 0.02, 'accel_z': 0.98, 'pressure': 500.0, 'proximity': 45.0},
            {'accel_x': 0.02, 'accel_y': 0.01, 'accel_z': 0.99, 'pressure': 502.0, 'proximity': 44.8},
            {'accel_x': -0.01, 'accel_y': 0.00, 'accel_z': 0.98, 'pressure': 498.0, 'proximity': 45.2},
        ]

        prev = None
        for read in normal_readings:
            cleaned = SensorDataPreprocessor.clean_telemetry(read)
            feats = SensorDataPreprocessor.extract_features(cleaned, prev)
            prev = cleaned

            t = SensorTelemetry(
                device_id=macbook.id,
                accel_x=cleaned['accel_x'],
                accel_y=cleaned['accel_y'],
                accel_z=cleaned['accel_z'],
                pressure=cleaned['pressure'],
                proximity=cleaned['proximity'],
                magnitude=float(feats[3]),
                delta_magnitude=float(feats[4])
            )
            db.session.add(t)

        db.session.flush()

        # 4. Create Simulated Anomaly Event & Alert
        theft_read = {'accel_x': 2.5, 'accel_y': -1.8, 'accel_z': 3.2, 'pressure': 15.0, 'proximity': 110.0}
        cleaned_theft = SensorDataPreprocessor.clean_telemetry(theft_read)
        feats_theft = SensorDataPreprocessor.extract_features(cleaned_theft, prev)

        theft_t = SensorTelemetry(
            device_id=macbook.id,
            accel_x=cleaned_theft['accel_x'],
            accel_y=cleaned_theft['accel_y'],
            accel_z=cleaned_theft['accel_z'],
            pressure=cleaned_theft['pressure'],
            proximity=cleaned_theft['proximity'],
            magnitude=float(feats_theft[3]),
            delta_magnitude=float(feats_theft[4])
        )
        db.session.add(theft_t)
        db.session.flush()

        alert = AnomalyAlert(
            device_id=macbook.id,
            telemetry_id=theft_t.id,
            anomaly_score=-0.285,
            risk_score=88.5,
            threat_level="CRITICAL",
            status="ACTIVE",
            alert_message="CRITICAL THEFT WARNING: Unattended belonging forcibly moved or lifted from surface!"
        )
        db.session.add(alert)
        db.session.flush()

        # 5. Model Metadata Entry
        meta = ModelMetadata(
            model_version="v1.0.0",
            contamination=0.08,
            n_estimators=100,
            sample_count=1080,
            false_positive_count=0,
            status_summary="Initial Isolation Forest model seeded successfully."
        )
        db.session.add(meta)

        db.session.commit()
        logger.info("Database seeding completed successfully!")

if __name__ == '__main__':
    seed()
