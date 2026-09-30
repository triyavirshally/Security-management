import json

def test_system_health(client):
    res = client.get('/api/system/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'HEALTHY'
    assert 'Smart Belonging Security' in data['system_name']
    assert data['ml_engine_trained'] is True

def test_list_devices(client):
    res = client.get('/api/devices')
    assert res.status_code == 200
    data = res.get_json()
    assert len(data['devices']) >= 1
    assert data['devices'][0]['device_name'] == "Test Laptop"

def test_create_device(client):
    payload = {
        "user_id": 1,
        "device_name": "Hostel Backpack",
        "belonging_type": "bag",
        "sensor_mac": "MAC:BAG:99:88:77",
        "sensitivity_level": "HIGH"
    }
    res = client.post('/api/devices', json=payload)
    assert res.status_code == 201
    data = res.get_json()
    assert data['device']['sensor_mac'] == "MAC:BAG:99:88:77"

def test_telemetry_ingest_and_alert(client):
    payload = {
        "sensor_mac": "TEST:MAC:00:11:22",
        "accel_x": 3.5,
        "accel_y": -2.1,
        "accel_z": 4.0,
        "pressure": 5.0,
        "proximity": 150.0
    }
    res = client.post('/api/telemetry/ingest', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'PROCESSED'
    assert data['decision']['should_alert'] is True

def test_user_registration_and_login(client):
    reg_payload = {
        "username": "newstudent",
        "email": "new@hostel.edu",
        "password": "secretpassword"
    }
    res = client.post('/api/auth/register', json=reg_payload)
    assert res.status_code == 201

    login_payload = {
        "username": "newstudent",
        "password": "secretpassword"
    }
    login_res = client.post('/api/auth/login', json=login_payload)
    assert login_res.status_code == 200
