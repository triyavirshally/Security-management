import logging
from flask import jsonify, Flask
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)

class SecurityBaseException(Exception):
    """Base exception class for Smart Belonging Security System."""
    status_code = 500
    error_code = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str, status_code: int = None, details: dict = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_dict(self):
        return {
            'error': True,
            'code': self.error_code,
            'message': self.message,
            'details': self.details
        }

class APIValidationError(SecurityBaseException):
    status_code = 400
    error_code = "INVALID_PAYLOAD"

class ResourceNotFoundError(SecurityBaseException):
    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"

class ModelInferenceError(SecurityBaseException):
    status_code = 503
    error_code = "ML_MODEL_UNAVAILABLE"

class DatabaseConnectionError(SecurityBaseException):
    status_code = 500
    error_code = "DATABASE_ERROR"

def register_error_handlers(app: Flask):
    """
    Register application-wide error boundaries and custom exception handlers.
    Ensures that unexpected runtime crashes or invalid client payloads return clean,
    structured JSON error responses rather than unhandled HTML 500 tracebacks.
    """
    @app.errorhandler(SecurityBaseException)
    def handle_custom_exception(error: SecurityBaseException):
        logger.warning(f"Handled Custom Exception [{error.error_code}]: {error.message}")
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        logger.info(f"Handled HTTP Exception ({error.code}): {error.description}")
        return jsonify({
            'error': True,
            'code': f"HTTP_{error.code}",
            'message': error.description,
            'details': {}
        }), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error: Exception):
        logger.error(f"CRITICAL: Unhandled Global Exception Boundary Caught: {str(error)}", exc_info=True)
        return jsonify({
            'error': True,
            'code': 'UNHANDLED_SYSTEM_ERROR',
            'message': 'An unexpected internal error occurred. Graceful error boundary engaged.',
            'details': {'raw_error': str(error)}
        }), 500
