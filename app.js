/**
 * Client-Side Application Logic for AI Smart Belonging Security & Theft Detection
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Smart Belonging Security App initialized.');

    // Handle Add Belonging Device Form
    const addDeviceForm = document.getElementById('addDeviceForm');
    if (addDeviceForm) {
        addDeviceForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const device_name = document.getElementById('newDeviceName').value;
            const belonging_type = document.getElementById('newBelongingType').value;
            const sensor_mac = document.getElementById('newSensorMac').value;
            const sensitivity_level = document.getElementById('newSensitivity').value;

            try {
                const res = await fetch('/api/devices', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: 1, device_name, belonging_type, sensor_mac, sensitivity_level })
                });

                const data = await res.json();
                if (res.ok) {
                    alert('Belonging device registered successfully!');
                    location.reload();
                } else {
                    alert('Error: ' + (data.message || 'Failed to register belonging device.'));
                }
            } catch (err) {
                alert('Network exception registering device.');
            }
        });
    }
});

// Toggle Arm/Disarm for Device
async function toggleArm(deviceId) {
    try {
        const res = await fetch(`/api/devices/${deviceId}/toggle`, { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            location.reload();
        } else {
            alert('Error: ' + data.message);
        }
    } catch (err) {
        alert('Failed to toggle device arm status.');
    }
}

// Run Live IoT Telemetry Simulation
async function runSimulation(scenario) {
    const deviceId = document.getElementById('simDeviceId').value;
    const resultBox = document.getElementById('simResultBox');
    const resultJson = document.getElementById('simResultJson');

    resultBox.classList.remove('d-none');
    resultJson.textContent = `Running ${scenario} telemetry simulation...`;

    try {
        const res = await fetch('/api/telemetry/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_id: parseInt(deviceId), scenario })
        });

        const data = await res.json();
        resultJson.textContent = JSON.stringify(data, null, 2);

        if (data.decision && data.decision.should_alert) {
            alert(`🚨 SECURITY ALERT TRIGGERED!\nThreat Level: ${data.decision.threat_level}\nMessage: ${data.decision.alert_message}`);
        }
    } catch (err) {
        resultJson.textContent = 'Error running simulation request.';
    }
}

// Resolve Active Alert
async function resolveAlert(alertId) {
    try {
        const res = await fetch(`/api/alerts/${alertId}/resolve`, { method: 'POST' });
        if (res.ok) {
            location.reload();
        }
    } catch (err) {
        alert('Failed to resolve alert.');
    }
}

// Submit Owner Feedback (Flag False Alarm)
async function submitFeedback(alertId) {
    const notes = prompt('Please enter optional feedback notes for this false alarm (e.g., "I picked up my bag myself"):');
    if (notes === null) return;

    try {
        const res = await fetch(`/api/alerts/${alertId}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_false_positive: true, notes: notes })
        });

        const data = await res.json();
        if (res.ok) {
            alert('False alarm feedback submitted. ML sensitivity tuned.');
            location.reload();
        } else {
            alert('Error submitting feedback.');
        }
    } catch (err) {
        alert('Network error submitting feedback.');
    }
}

// Retrain Model
async function retrainModel() {
    if (!confirm('Trigger Isolation Forest online model retraining on accumulated sensor data?')) return;
    try {
        const res = await fetch('/api/model/retrain', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            alert('Model retraining complete! Metrics updated.');
            location.reload();
        } else {
            alert('Retraining failed: ' + data.message);
        }
    } catch (err) {
        alert('Error retraining model.');
    }
}
