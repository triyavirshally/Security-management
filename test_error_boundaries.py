import json

def test_telemetry_missing_mac(client):
    res = client.post('/api/telemetry/ingest', json={"accel_x": 0.0})
    assert res.status_code == 400
    data = res.get_json()
    assert data['error'] is True
    assert data['code'] == 'INVALID_PAYLOAD'

def test_telemetry_unregistered_mac(client):
    payload = {
        "sensor_mac": "UNKNOWN:MAC:99:99",
        "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1.0,
        "pressure": 500.0, "proximity": 40.0
    }
    res = client.post('/api/telemetry/ingest', json=payload)
    assert res.status_code == 404
    data = res.get_json()
    assert data['error'] is True
    assert data['code'] == 'RESOURCE_NOT_FOUND'

def test_telemetry_corrupt_non_numeric_fields(client):
    payload = {
        "sensor_mac": "TEST:MAC:00:11:22",
        "accel_x": "CORRUPTED_STRING_VALUE",
        "accel_y": 0.0, "accel_z": 1.0,
        "pressure": 500.0, "proximity": 40.0
    }
    res = client.post('/api/telemetry/ingest', json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert data['error'] is True
    assert 'non-numeric' in data['message']

def test_404_boundary(client):
    res = client.get('/api/invalid/route/does/not/exist')
    assert res.status_code == 404
    data = res.get_json()
    assert data['error'] is True
    assert data['code'] == 'HTTP_404'
