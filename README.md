# Medical AI API for Apple Watch Data

This system provides API endpoints for a mobile app to send Apple Watch health data and receive disease risk predictions using a pre-trained medical AI model.

## System Overview

The system consists of:
1. A pre-trained medical AI model (using TCN architecture)
2. An XAI (Explainable AI) system that provides interpretable predictions
3. A REST API for receiving Apple Watch data and returning predictions

## API Endpoints

### Health Check
```
GET /health
```
Simple endpoint to verify the API is running.

### Predict Disease Risk
```
POST /api/v1/predict
```
Main endpoint to submit Apple Watch data and receive predictions.

**Request Body:**
```json
{
    "user_id": "user123",
    "timestamp": "2023-04-01T12:34:56Z",
    "vitals": {
        "heart_rate": 75,
        "steps": 8500,
        "sleep_duration": 7.5,
        "active_calories": 320,
        "resting_calories": 1800,
        "stand_hours": 10,
        "exercise_minutes": 30
    }
}
```

**Response:**
```json
{
    "status": "success",
    "prediction_available": true,
    "analysis": {
        "response_level": "Healthy",
        "classification_score": 0.15,
        "danger_metric": 0.12,
        "text_explanation": "Your vitals appear normal with no significant deviations...",
        "most_important_feature": "resting_pulse"
    },
    "feature_importance": [
        {"feature": "resting_pulse", "importance": 0.75},
        {"feature": "sleep_duration", "importance": 0.65},
        {"feature": "steps_count", "importance": 0.45}
    ]
}
```

### Get User History
```
GET /api/v1/users/{user_id}/history
```
Retrieve information about stored data for a specific user.

### Reset User Data
```
POST /api/v1/reset/{user_id}
```
Reset all stored data for a specific user.

## Setup and Running

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```
   python api.py
   ```

3. The API will be available at `http://localhost:5000`

## Integration with iPhone App

The iPhone app should:

1. Collect relevant health data from Apple Watch
2. Format it according to the API specifications
3. Send it to the `/api/v1/predict` endpoint
4. Process and display the results to the user

Note: The system requires at least 50 data points before it can make reliable predictions. Until then, it will store the data and inform the app that more data is needed.

## Data Mapping

Apple Watch metrics should be mapped to the model's expected features:

| Apple Watch Metric | Model Feature |
|--------------------|---------------|
| heart_rate         | resting_pulse |
| steps              | steps_count   |
| sleep_duration     | sleep_duration|
| active_calories    | active_calories |
| resting_calories   | resting_calories |
| stand_hours        | stand_hours   |
| exercise_minutes   | exercise_minutes |
