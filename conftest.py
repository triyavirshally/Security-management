import pytest
import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from database import db
from models import User, Device, ModelMetadata
from ml.anomaly_detector import AnomalyDetector

@pytest.fixture
def app():
    """Create test Flask app configured with in-memory SQLite database."""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SECRET_KEY': 'test-secret-key'
    })

    with app.app_context():
        db.create_all()
        # Seed test user and device
        user = User(username='teststudent', email='test@hostel.edu')
        user.set_password('pass123')
        db.session.add(user)
        db.session.flush()

        device = Device(
            user_id=user.id,
            device_name="Test Laptop",
            belonging_type="laptop",
            sensor_mac="TEST:MAC:00:11:22",
            is_armed=True,
            sensitivity_level="MEDIUM"
        )
        db.session.add(device)
        db.session.commit()

        yield app

        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    """Test client fixture."""
    return app.test_client()

@pytest.fixture
def trained_detector(tmp_path):
    """Trained IsolationForestDetector instance in a temporary folder."""
    detector = AnomalyDetector(model_dir=str(tmp_path))
    detector.train()
    return detector
