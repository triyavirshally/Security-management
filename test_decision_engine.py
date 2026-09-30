from ml.decision_engine import DecisionEngine

def test_evaluate_threat_disarmed():
    res = DecisionEngine.evaluate_threat(
        anomaly_score=0.1,
        risk_score=90.0,
        cleaned_telemetry={},
        features=[0]*9,
        is_armed=False
    )
    assert res['should_alert'] is False
    assert res['threat_level'] == 'LOW'

def test_evaluate_threat_critical():
    res = DecisionEngine.evaluate_threat(
        anomaly_score=-0.35,
        risk_score=85.0,
        cleaned_telemetry={'accel_x': 3.0, 'pressure': 0.0, 'proximity': 100.0},
        features=[3.0, 0.0, 0.0, 3.0, 2.5, 0.0, 500.0, 100.0, 60.0],
        is_armed=True,
        sensitivity='HIGH'
    )
    assert res['should_alert'] is True
    assert res['threat_level'] in ['HIGH', 'CRITICAL']
    assert res['risk_score'] > 70.0

def test_false_positive_damping():
    res_normal = DecisionEngine.evaluate_threat(0.0, 50.0, {}, [0]*9, True, 'MEDIUM', false_positive_count=0)
    res_damped = DecisionEngine.evaluate_threat(0.0, 50.0, {}, [0]*9, True, 'MEDIUM', false_positive_count=4)

    assert res_damped['risk_score'] < res_normal['risk_score']
