import logging
from typing import Dict, Any
from config import Config

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Multi-Sensor Decision & Security Alert Engine.
    Combines Isolation Forest threat scores with hard sensor guardrails, device arming status,
    device sensitivity settings, and user feedback history to prevent false alarms.
    """

    @staticmethod
    def evaluate_threat(
        anomaly_score: float,
        risk_score: float,
        cleaned_telemetry: Dict[str, float],
        features: list,
        is_armed: bool = True,
        sensitivity: str = 'MEDIUM',
        false_positive_count: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate sensor telemetry and model score to produce an actionable security decision.

        :param anomaly_score: Raw Isolation Forest decision score.
        :param risk_score: Scaled threat score (0.0 to 100.0).
        :param cleaned_telemetry: Dictionary of raw sensor readings.
        :param features: 9-element feature list.
        :param is_armed: Boolean indicating if belonging security is active.
        :param sensitivity: 'LOW', 'MEDIUM', or 'HIGH'.
        :param false_positive_count: Historical false positive count for adaptive threshold tuning.
        :return: Decision dictionary with should_alert, threat_level, risk_score, message.
        """
        if not is_armed:
            return {
                'should_alert': False,
                'threat_level': Config.THREAT_LEVEL_LOW,
                'risk_score': 0.0,
                'alert_message': 'System disarmed by owner.',
                'trigger_reason': 'DISARMED'
            }

        # Sensitivity multipliers
        sensitivity_multiplier = {
            'LOW': 0.8,
            'MEDIUM': 1.0,
            'HIGH': 1.25
        }.get(sensitivity.upper(), 1.0)

        # Adaptive false-positive correction factor
        # If owner flagged recent false alarms, slightly increase alert threshold to avoid nuisance notifications
        fp_damping_factor = max(0.75, 1.0 - (false_positive_count * 0.05))

        adjusted_risk_score = risk_score * sensitivity_multiplier * fp_damping_factor
        adjusted_risk_score = float(min(100.0, max(0.0, adjusted_risk_score)))

        # Feature order: [accel_x, accel_y, accel_z, mag, delta_mag, pressure, pressure_delta, proximity, proximity_delta]
        delta_mag = features[4] if len(features) > 4 else 0.0
        pressure_delta = features[6] if len(features) > 6 else 0.0
        proximity_delta = features[8] if len(features) > 8 else 0.0

        # Hard sensor trigger rules (Safety Guardrails)
        is_sudden_lift = delta_mag > 1.2
        is_weight_drop = pressure_delta > 150.0
        is_proximity_snatch = proximity_delta > 30.0

        # Decision Logic Hierarchy
        if adjusted_risk_score >= 80.0 or (is_sudden_lift and is_weight_drop):
            threat_level = Config.THREAT_LEVEL_CRITICAL
            should_alert = True
            msg = "CRITICAL THEFT WARNING: Unattended belonging forcibly moved or lifted from surface!"
            reason = "CRITICAL_DISPLACEMENT"

        elif adjusted_risk_score >= 60.0 or is_sudden_lift or is_weight_drop:
            threat_level = Config.THREAT_LEVEL_HIGH
            should_alert = True
            msg = "HIGH SECURITY ALERT: Unexpected interaction or physical displacement detected."
            reason = "PHYSICAL_INTERACTION"

        elif adjusted_risk_score >= 40.0 or is_proximity_snatch:
            threat_level = Config.THREAT_LEVEL_MEDIUM
            should_alert = True
            msg = "MEDIUM ALERT: Proximity or minor movement anomaly registered."
            reason = "PROXIMITY_ANOMALY"

        else:
            threat_level = Config.THREAT_LEVEL_LOW
            should_alert = False
            msg = "Belonging status normal. No unauthorized interaction."
            reason = "NORMAL_BASELINE"

        return {
            'should_alert': should_alert,
            'threat_level': threat_level,
            'risk_score': round(adjusted_risk_score, 2),
            'anomaly_score': round(anomaly_score, 4),
            'alert_message': msg,
            'trigger_reason': reason
        }
