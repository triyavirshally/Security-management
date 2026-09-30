import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class NotificationService:
    """
    Real-Time Smartphone Notification Gateway.
    Simulates push notification delivery to owner's mobile device upon alert trigger.
    Maintains an in-memory dispatch history log.
    """

    _dispatch_log: List[Dict[str, Any]] = []

    @classmethod
    def send_push_notification(cls, device_name: str, belonging_type: str, threat_level: str, message: str, risk_score: float) -> Dict[str, Any]:
        """
        Simulate sending an instant mobile alert notification to the student's smartphone.

        :param device_name: Registered name of belonging (e.g. 'MacBook Pro M2')
        :param belonging_type: Belonging category (e.g. 'laptop')
        :param threat_level: LOW, MEDIUM, HIGH, CRITICAL
        :param message: Human-readable alert description
        :param risk_score: Scaled threat percentage (0-100%)
        :return: Dispatch report dictionary
        """
        payload = {
            'dispatch_id': f"PUSH-{len(cls._dispatch_log) + 1:05d}",
            'device_name': device_name,
            'belonging_type': belonging_type,
            'threat_level': threat_level,
            'risk_score': risk_score,
            'message': message,
            'status': 'DELIVERED',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        cls._dispatch_log.append(payload)
        logger.info(f"📱 PUSH NOTIFICATION SENT to Smartphone: [{threat_level}] {device_name} - {message}")

        return payload

    @classmethod
    def get_dispatch_history(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent push notification logs."""
        return cls._dispatch_log[-limit:]
