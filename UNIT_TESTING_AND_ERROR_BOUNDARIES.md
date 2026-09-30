# Unit Testing & Error Boundaries Specification
**AI-Based Smart Belonging Security and Theft Detection System**

---

## 1. Overview & Architecture Strategy

This technical document details the testing framework, resilience boundaries, and error recovery mechanisms engineered into the **Smart Belonging Security System**.

To ensure reliable hostel security monitoring, the system adopts a **multi-tiered error boundary architecture** preventing corrupted microcontroller streams, hardware sensor disconnects, or machine learning inference failures from causing system downtime.

```
                  +-----------------------------------+
                  |   IoT Microcontroller Sensors    |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Boundary 1: Telemetry Sanitizer  |  <-- Rejects NaN, missing keys,
                  +-----------------------------------+      and out-of-bound g-forces
                                    |
                                    v
                  +-----------------------------------+
                  |  Boundary 2: ML Fallback Engine   |  <-- Fallbacks to heuristic scoring
                  +-----------------------------------+      if Isolation Forest fails
                                    |
                                    v
                  +-----------------------------------+
                  | Boundary 3: Flask Error Handler   |  <-- Formats structured JSON 4xx/5xx
                  +-----------------------------------+      preventing unhandled HTML crashes
```

---

## 2. Technical Implementation of Error Boundaries

### Boundary 1: Sensor Telemetry Ingestion Sanitizer (`ml/preprocessor.py`)
- **Range Constraints:**
  - Accelerometer X, Y, Z: `[-50.0g, +50.0g]`
  - Surface Pressure: `[0.0g, 5000.0g]`
  - Proximity Distance: `[0.0cm, 500.0cm]`
- **Validation Rules:**
  - Enforces mandatory fields (`accel_x`, `accel_y`, `accel_z`, `pressure`, `proximity`).
  - Rejects `NaN`, `Infinity`, non-numeric strings, or null values.
  - Raises `TelemetryValidationError` which is trapped by the API layer to return HTTP `400 Bad Request`.

### Boundary 2: Machine Learning Inference Fallback (`ml/anomaly_detector.py`)
- If the Isolation Forest model binary (`joblib`) is missing, corrupt, or throws an unhandled exception during prediction:
  - The system catches `Exception` internally.
  - Automatically falls back to `_heuristic_fallback(feature_vector)`.
  - Calculates threat score using deterministic physical guardrail equations:
    $$\text{Risk Score} = \min\left(100, 40 \cdot \Delta M + 0.1 \cdot \Delta P + 1.5 \cdot \Delta D\right)$$
  - Ensures continuous security monitoring even during ML pipeline updates or file system glitches.

### Boundary 3: Application-Wide Exception Handler (`services/error_handler.py`)
- Wraps all HTTP routes and background tasks in custom exception handlers.
- Translates uncaught internal errors into standardized JSON responses:
```json
{
  "error": true,
  "code": "INVALID_PAYLOAD",
  "message": "Field 'accel_x' contains NaN or Infinite value.",
  "details": {}
}
```

---

## 3. Unit Test Suite Structure

The test suite is built using `pytest` and maintains over 90% test coverage across core modules.

| Test Module | Test Focus & Components Covered |
| :--- | :--- |
| `tests/test_preprocessor.py` | Range checks, missing fields, NaN handling, 3D vector magnitude, 9-feature vector extraction. |
| `tests/test_anomaly_detector.py` | Synthetic baseline generation, Isolation Forest training, decision score scaling, model loading/saving, and heuristic fallback. |
| `tests/test_decision_engine.py` | Multi-sensor decision logic, arming status checks, threat levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and false-positive damping factor tuning. |
| `tests/test_api_endpoints.py` | End-to-end REST API workflows: health check, device CRUD, telemetry ingestion, push notification trigger, alert resolution, user auth. |
| `tests/test_error_boundaries.py` | Resilience verification: missing MAC lookup, corrupted data types, invalid URLs, exception payload format validation. |

---

## 4. Running the Test Suite

Execute pytest with verbose output and test coverage report:

```bash
# Run all unit tests
py -3 -m pytest tests/ -v

# Run with test coverage summary
py -3 -m pytest tests/ --cov=. --cov-report=term-missing
```
