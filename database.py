from flask_sqlalchemy import SQLAlchemy

# Global SQLAlchemy instance
db = SQLAlchemy()

def init_db(app):
    """Initialize database connection with Flask application context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
