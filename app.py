import os
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from datetime import datetime, timezone
from config import Config
from database import db, init_db
from models import User, Device, SensorTelemetry, AnomalyAlert, UserFeedback, ModelMetadata
from ml.preprocessor import SensorDataPreprocessor, TelemetryValidationError
from ml.anomaly_detector import AnomalyDetector
from ml.decision_engine import DecisionEngine
from services.notification_service import NotificationService
from services.error_handler import (
    register_error_handlers,
    APIValidationError,
    ResourceNotFoundError,
    ModelInferenceError
)

# Configure logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('SecurityApp')

def create_app():
    """Application Factory for AI-Based Smart Belonging Security System."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize Database & Error Handlers
    init_db(app)
    register_error_handlers(app)

    # Initialize Anomaly Detection Engine
    detector = AnomalyDetector()

    # Pre-train model if no trained model exists
    with app.app_context():
        if not detector.is_trained:
            logger.info("Initializing baseline Isolation Forest training run...")
            detector.train()

    # =========================================================================
    # WEB DASHBOARD ROUTES
    # =========================================================================
    @app.route('/')
    def index():
        """Render main interactive security monitoring dashboard."""
        devices = Device.query.all()
        alerts = AnomalyAlert.query.order_by(AnomalyAlert.timestamp.desc()).limit(20).all()
        return render_template('index.html', devices=devices, alerts=alerts)

    @app.route('/login')
    def login_page():
        return render_template('login.html')

    @app.route('/register')
    def register_page():
        return render_template('register.html')


    # =========================================================================
    # REST API: SYSTEM & HEALTH
    # =========================================================================
    @app.route('/api/system/health', methods=['GET'])
    def system_health():
        """
        GET /api/system/health
        System health check endpoint for monitoring uptime, database connection, and ML status.
        """
        return jsonify({
            'status': 'HEALTHY',
            'system_name': 'AI-Based Smart Belonging Security System',
            'version': '1.0.0',
            'ml_engine_trained': detector.is_trained,
            'database': 'CONNECTED',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200


    # =========================================================================
    # REST API: USER AUTHENTICATION
    # =========================================================================
    @app.route('/api/auth/register', methods=['POST'])
    def register_user():
        """
        POST /api/auth/register
        Register a new student / belonging owner account.
        Body: { "username": "student1", "email": "student@hostel.edu", "password": "pass" }
        """
        data = request.get_json() or {}
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            raise APIValidationError("Username, email, and password are required fields.")

        if User.query.filter((User.username == username) | (User.email == email)).first():
            raise APIValidationError("User with given username or email already exists.")

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return jsonify({'message': 'User registered successfully', 'user': user.to_dict()}), 201

    @app.route('/api/auth/login', methods=['POST'])
    def login_user():
        """
        POST /api/auth/login
        Authenticate user credentials.
        Body: { "username": "student1", "password": "pass" }
        """
        data = request.get_json() or {}
        username = data.get('username')
        password = data.get('password')

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            raise APIValidationError("Invalid username or password credentials.")

        session['user_id'] = user.id
        return jsonify({'message': 'Login successful', 'user': user.to_dict()}), 200


    # =========================================================================
    # REST API: DEVICE MANAGEMENT
    # =========================================================================
    @app.route('/api/devices', methods=['GET'])
    def list_devices():
        """
        GET /api/devices
        List all registered belongings and security node statuses.
        """
        devices = Device.query.all()
        return jsonify({'devices': [d.to_dict() for d in devices]}), 200

    @app.route('/api/devices', methods=['POST'])
    def create_device():
        """
        POST /api/devices
        Register a new belonging (e.g., laptop, backpack, charger).
        Body: { "user_id": 1, "device_name": "Dell XPS 15", "belonging_type": "laptop", "sensor_mac": "AA:BB:CC:11:22:33", "sensitivity_level": "MEDIUM" }
        """
        data = request.get_json() or {}
        user_id = data.get('user_id', 1)
        device_name = data.get('device_name')
        belonging_type = data.get('belonging_type', 'laptop')
        sensor_mac = data.get('sensor_mac')
        sensitivity = data.get('sensitivity_level', 'MEDIUM')

        if not device_name or not sensor_mac:
            raise APIValidationError("device_name and sensor_mac are required fields.")

        if Device.query.filter_by(sensor_mac=sensor_mac).first():
            raise APIValidationError(f"Device with sensor_mac '{sensor_mac}' is already registered.")

        device = Device(
            user_id=user_id,
            device_name=device_name,
            belonging_type=belonging_type,
            sensor_mac=sensor_mac,
            sensitivity_level=sensitivity
        )
        db.session.add(device)
        db.session.commit()

        return jsonify({'message': 'Device registered successfully', 'device': device.to_dict()}), 201

    @app.route('/api/devices/<int:device_id>/toggle', methods=['POST'])
    def toggle_device_arm(device_id: int):
        """
        POST /api/devices/<device_id>/toggle
        Arm or Disarm the belonging security monitoring system.
        """
        device = Device.query.get(device_id)
        if not device:
            raise ResourceNotFoundError(f"Device with ID {device_id} not found.")

        device.is_armed = not device.is_armed
        db.session.commit()

        status_str = "ARMED" if device.is_armed else "DISARMED"
        logger.info(f"Device [{device.device_name}] is now {status_str}")

        return jsonify({'message': f"Device status changed to {status_str}", 'device': device.to_dict()}), 200


    # =========================================================================
    # REST API: TELEMETRY & ANOMALY DETECTION INGESTION
    # =========================================================================
    @app.route('/api/telemetry/ingest', methods=['POST'])
    def ingest_telemetry():
        """
        POST /api/telemetry/ingest
        Core IoT Ingestion Endpoint.
        Accepts real-time sensor streams (accelerometer, pressure, proximity), cleans & validates,
        runs Isolation Forest prediction, evaluates threat logic, and triggers alerts/push notifications.

        Body: {
            "sensor_mac": "AA:BB:CC:11:22:33",
            "accel_x": 0.05,
            "accel_y": 0.01,
            "accel_z": 0.98,
            "pressure": 510.0,
            "proximity": 45.0
        }
        """
        raw_payload = request.get_json() or {}
        sensor_mac = raw_payload.get('sensor_mac')

        if not sensor_mac:
            raise APIValidationError("Field 'sensor_mac' is required for telemetry ingestion.")

        device = Device.query.filter_by(sensor_mac=sensor_mac).first()
        if not device:
            raise ResourceNotFoundError(f"No registered device found for sensor MAC: '{sensor_mac}'")

        # 1. Clean & validate sensor data through preprocessor
        try:
            cleaned = SensorDataPreprocessor.clean_telemetry(raw_payload)
        except TelemetryValidationError as err:
            raise APIValidationError(f"Invalid IoT Telemetry Data: {str(err)}")

        # 2. Fetch last reading for delta calculations
        prev_read = SensorTelemetry.query.filter_by(device_id=device.id).order_by(SensorTelemetry.timestamp.desc()).first()
        prev_dict = {
            'accel_x': prev_read.accel_x,
            'accel_y': prev_read.accel_y,
            'accel_z': prev_read.accel_z,
            'pressure': prev_read.pressure,
            'proximity': prev_read.proximity
        } if prev_read else None

        # 3. Feature engineering
        features = SensorDataPreprocessor.extract_features(cleaned, prev_dict)
        magnitude = float(features[3])
        delta_mag = float(features[4])

        # Save Telemetry Record
        telemetry = SensorTelemetry(
            device_id=device.id,
            accel_x=cleaned['accel_x'],
            accel_y=cleaned['accel_y'],
            accel_z=cleaned['accel_z'],
            pressure=cleaned['pressure'],
            proximity=cleaned['proximity'],
            magnitude=magnitude,
            delta_magnitude=delta_mag
        )
        db.session.add(telemetry)
        db.session.flush()

        # 4. ML Model Anomaly Prediction
        raw_anomaly_score, risk_score, pred_label = detector.predict(features)

        # 5. False Positive count lookup for device
        fp_count = UserFeedback.query.filter_by(device_id=device.id, is_false_positive=True).count()

        # 6. Decision Engine Evaluation
        decision = DecisionEngine.evaluate_threat(
            anomaly_score=raw_anomaly_score,
            risk_score=risk_score,
            cleaned_telemetry=cleaned,
            features=features.tolist(),
            is_armed=device.is_armed,
            sensitivity=device.sensitivity_level,
            false_positive_count=fp_count
        )

        alert_dict = None
        push_report = None

        if decision['should_alert']:
            alert = AnomalyAlert(
                device_id=device.id,
                telemetry_id=telemetry.id,
                anomaly_score=decision['anomaly_score'],
                risk_score=decision['risk_score'],
                threat_level=decision['threat_level'],
                status='ACTIVE',
                alert_message=decision['alert_message']
            )
            db.session.add(alert)
            db.session.flush()
            alert_dict = alert.to_dict()

            # Trigger real-time mobile push notification
            push_report = NotificationService.send_push_notification(
                device_name=device.device_name,
                belonging_type=device.belonging_type,
                threat_level=decision['threat_level'],
                message=decision['alert_message'],
                risk_score=decision['risk_score']
            )

        db.session.commit()

        return jsonify({
            'status': 'PROCESSED',
            'telemetry_id': telemetry.id,
            'decision': decision,
            'alert': alert_dict,
            'push_notification': push_report
        }), 200

    @app.route('/api/telemetry/simulate', methods=['POST'])
    def simulate_telemetry_event():
        """
        POST /api/telemetry/simulate
        Simulation helper for demo & testing. Generates a normal or anomaly tick for a given device.
        Body: { "device_id": 1, "scenario": "NORMAL" | "THEFT_ATTEMPT" | "PROXIMITY_APPROACH" }
        """
        data = request.get_json() or {}
        device_id = data.get('device_id', 1)
        scenario = data.get('scenario', 'NORMAL')

        device = Device.query.get(device_id)
        if not device:
            raise ResourceNotFoundError(f"Device with ID {device_id} not found.")

        if scenario == 'THEFT_ATTEMPT':
            # Sudden pickup & displacement: large acceleration, weight drop
            payload = {
                'sensor_mac': device.sensor_mac,
                'accel_x': 2.8,
                'accel_y': -1.9,
                'accel_z': 3.1,
                'pressure': 10.0,
                'proximity': 120.0
            }
        elif scenario == 'PROXIMITY_APPROACH':
            # Hand or object close approach
            payload = {
                'sensor_mac': device.sensor_mac,
                'accel_x': 0.1,
                'accel_y': 0.1,
                'accel_z': 0.99,
                'pressure': 490.0,
                'proximity': 5.0
            }
        else:
            # Baseline normal stationary
            payload = {
                'sensor_mac': device.sensor_mac,
                'accel_x': 0.01,
                'accel_y': 0.02,
                'accel_z': 0.98,
                'pressure': 500.0,
                'proximity': 45.0
            }

        # Forward internally to ingest pipeline
        with app.test_request_context('/api/telemetry/ingest', method='POST', json=payload):
            return ingest_telemetry()


    # =========================================================================
    # REST API: ALERTS & USER FEEDBACK (FALSE-POSITIVE REDUCTION)
    # =========================================================================
    @app.route('/api/alerts', methods=['GET'])
    def list_alerts():
        """
        GET /api/alerts
        Retrieve active and historical security alerts.
        """
        alerts = AnomalyAlert.query.order_by(AnomalyAlert.timestamp.desc()).all()
        return jsonify({'alerts': [a.to_dict() for a in alerts]}), 200

    @app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
    def resolve_alert(alert_id: int):
        """
        POST /api/alerts/<alert_id>/resolve
        Mark an active alert as RESOLVED.
        """
        alert = AnomalyAlert.query.get(alert_id)
        if not alert:
            raise ResourceNotFoundError(f"Alert with ID {alert_id} not found.")

        alert.status = 'RESOLVED'
        db.session.commit()
        return jsonify({'message': 'Alert resolved successfully', 'alert': alert.to_dict()}), 200

    @app.route('/api/alerts/<int:alert_id>/feedback', methods=['POST'])
    def submit_user_feedback(alert_id: int):
        """
        POST /api/alerts/<alert_id>/feedback
        Submit owner feedback on an alert (e.g. flag False Positive).
        Feeds back into decision engine sensitivity tuning.
        Body: { "is_false_positive": true, "notes": "I moved my laptop myself" }
        """
        data = request.get_json() or {}
        alert = AnomalyAlert.query.get(alert_id)
        if not alert:
            raise ResourceNotFoundError(f"Alert with ID {alert_id} not found.")

        is_fp = bool(data.get('is_false_positive', False))
        notes = data.get('notes', '')

        if is_fp:
            alert.status = 'FALSE_POSITIVE'

        feedback = UserFeedback(
            alert_id=alert.id,
            device_id=alert.device_id,
            is_false_positive=is_fp,
            notes=notes
        )
        db.session.add(feedback)
        db.session.commit()

        logger.info(f"Feedback submitted for Alert #{alert_id}: FalsePositive={is_fp}")
        return jsonify({'message': 'User feedback recorded successfully', 'feedback': feedback.to_dict()}), 201


    # =========================================================================
    # REST API: ML MODEL METRICS & RE-TRAINING
    # =========================================================================
    @app.route('/api/model/status', methods=['GET'])
    def get_model_status():
        """
        GET /api/model/status
        Retrieve Isolation Forest model status, versioning, and training metrics.
        """
        meta = ModelMetadata.query.order_by(ModelMetadata.last_trained_at.desc()).first()
        meta_dict = meta.to_dict() if meta else None

        return jsonify({
            'model_type': 'IsolationForest',
            'is_trained': detector.is_trained,
            'contamination_factor': Config.ISOLATION_FOREST_CONTAMINATION,
            'n_trees': Config.ISOLATION_FOREST_TREES,
            'latest_metadata': meta_dict
        }), 200

    @app.route('/api/model/retrain', methods=['POST'])
    def retrain_model():
        """
        POST /api/model/retrain
        Trigger online retraining of Isolation Forest model using accumulated sensor telemetry.
        """
        all_telemetry = SensorTelemetry.query.order_by(SensorTelemetry.timestamp.desc()).limit(2000).all()

        if len(all_telemetry) >= 10:
            features_list = []
            for i, t in enumerate(all_telemetry):
                cur = {'accel_x': t.accel_x, 'accel_y': t.accel_y, 'accel_z': t.accel_z, 'pressure': t.pressure, 'proximity': t.proximity}
                prev = {'accel_x': all_telemetry[i-1].accel_x, 'accel_y': all_telemetry[i-1].accel_y, 'accel_z': all_telemetry[i-1].accel_z, 'pressure': all_telemetry[i-1].pressure, 'proximity': all_telemetry[i-1].proximity} if i > 0 else None
                features_list.append(SensorDataPreprocessor.extract_features(cur, prev))

            X_train = np.array(features_list)
        else:
            X_train = None

        res = detector.train(X_train)

        # Record retraining metadata
        meta = ModelMetadata(
            model_version=f"v1.{ModelMetadata.query.count() + 1}",
            contamination=Config.ISOLATION_FOREST_CONTAMINATION,
            n_estimators=Config.ISOLATION_FOREST_TREES,
            sample_count=res['sample_count'],
            status_summary="Retrained on telemetry baseline successfully"
        )
        db.session.add(meta)
        db.session.commit()

        return jsonify({'message': 'Isolation Forest model retrained successfully', 'metrics': res}), 200

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
