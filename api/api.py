import os
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from XAI import MedicalXAISystem

app = Flask(__name__)

# Initialize the XAI system with the pre-trained classification model
model_path = os.path.join(os.path.dirname(__file__), 'tcn_final_model.h5')
xai_system = MedicalXAISystem(
    model_path=model_path,
    theta_healthy=0.2,    # Threshold for Healthy
    theta_slight=0.4,     # Threshold for Slight Change 
    theta_warning=0.6,    # Threshold for Warning
    theta_serious=0.8,     # Threshold for Serious Condition
    w_c=0.8, w_u=0.2        # Weights for classification and unusualness
)

# Define which features to use for forecasting (matching main.py)
forecasting_features = ['heart_rate', 'steps', 'sleep_quality']

# Global variable to store historical data for each user
user_data_store = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Medical AI API is running"
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive Apple Watch data and make predictions
    
    Expected JSON payload:
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
            "exercise_minutes": 30,
            ...other vital signs
        }
    }
    """
    try:
        data = request.get_json()
        
        # Extract data from request
        user_id = data.get('user_id')
        timestamp = data.get('timestamp')
        vitals = data.get('vitals', {})
        
        # Validate required fields
        if not user_id or not timestamp or not vitals:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: user_id, timestamp, or vitals"
            }), 400
        
        # Convert timestamp to datetime
        try:
            timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "Invalid timestamp format. Use ISO format (e.g., 2023-04-01T12:34:56Z)"
            }), 400
        
        # Process vitals data
        processed_data = process_watch_data(user_id, timestamp_dt, vitals)
        
        # If this is a new user or we don't have enough historical data,
        # add to history and return informative message
        if user_id not in user_data_store or len(user_data_store[user_id]) < 50:
            return jsonify({
                "status": "success",
                "message": f"Data received and stored. Need more data points before prediction (current: {len(user_data_store.get(user_id, []))})",
                "prediction_available": False,
                "data_points_collected": len(user_data_store.get(user_id, []))
            })
        
        # Prepare data for the XAI system (similar to analyze_patient in main.py)
        try:
            print(f"Preparing data for analysis for user {user_id}...")
            classification_data, historical_data = prepare_data_for_analysis(user_id)
            
            # Print feature information for debugging
            print(f"Classification data shape: {classification_data.shape}")
            print(f"Features available: {classification_data.columns.tolist()}")
            
            # Train the forecasting models for this user
            print("Training forecasting models...")
            xai_system.train(classification_data, historical_data, forecasting_features)
            
            # Get the most recent data for analysis (use last 24 hours of data)
            recent_data = classification_data.iloc[-24:].copy()
            
            # Add timestamp column required by analyze method
            if 'timestamp' not in recent_data.columns:
                recent_data['timestamp'] = historical_data['timestamp'].iloc[-24:].values
            
            # Analyze the data
            print("Analyzing data...")
            results = xai_system.analyze(recent_data, historical_data)
            print("Analysis complete!")
            
            # Format the response
            response = {
                "status": "success",
                "prediction_available": True,
                "analysis": {
                    "response_level": results['response_level'],
                    "classification_score": float(results['classification_score']),
                    "danger_metric": float(results['danger_metric']),
                    "text_explanation": results['text_explanation'],
                    "most_important_feature": results['most_important_feature']
                },
                "feature_importance": []
            }
            
            # Add feature importance
            for feature, importance in results['feature_importance'][:5]:
                response["feature_importance"].append({
                    "feature": feature,
                    "importance": float(abs(importance))
                })
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error in data analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Error during analysis: {str(e)}",
                "error_details": traceback.format_exc()
            }), 500
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

def process_watch_data(user_id, timestamp, vitals):
    """
    Process incoming Apple Watch data and store it in the user's history
    
    Args:
        user_id: Unique identifier for the user
        timestamp: Datetime of the measurement
        vitals: Dictionary of vital signs from Apple Watch
    
    Returns:
        Processed DataFrame
    """
    # Create a mapping from Apple Watch metrics to our model's expected features
    watch_to_model_mapping = {
        'heart_rate': 'resting_pulse',
        'steps': 'steps_count',
        'sleep_duration': 'sleep_duration',
        'active_calories': 'active_calories',
        'resting_calories': 'resting_calories',
        'stand_hours': 'stand_hours',
        'exercise_minutes': 'exercise_minutes'
    }
    
    # Create a row for the new data
    row_data = {
        'date': timestamp,
        'user_code': user_id
    }
    
    # Map Apple Watch vital signs to model features
    for watch_key, model_key in watch_to_model_mapping.items():
        if watch_key in vitals:
            row_data[model_key] = vitals[watch_key]
    
    # Add any additional vitals that might be present
    # Also accept unmapped features with their original names
    for key, value in vitals.items():
        if key not in watch_to_model_mapping.keys() and key not in row_data:
            # Use the key as is if it's not in the mapping
            row_data[key] = value
    
    # Initialize user data store if this is a new user
    if user_id not in user_data_store:
        user_data_store[user_id] = []
    
    # Add to user's history
    user_data_store[user_id].append(row_data)
    
    return pd.DataFrame([row_data])

def prepare_data_for_analysis(user_id):
    """
    Prepare user data for the XAI system
    
    Args:
        user_id: User ID to retrieve data for
    
    Returns:
        Tuple of (classification_data, historical_data)
    """
    # Convert user's data store to DataFrame
    user_df = pd.DataFrame(user_data_store[user_id])
    
    # Sort by date
    user_df = user_df.sort_values('date')
    
    # Prepare data for classification
    # Get all columns except these specific ones
    classification_columns = [col for col in user_df.columns 
                             if col not in ['date', 'user_code', 'target', 'days_to_symptoms']] # Removed 'Unnamed: 0' from the list ##################
    
    classification_data = user_df[classification_columns].copy()
    
    # Create historical data for forecasting
    historical_data = pd.DataFrame({
        'timestamp': user_df['date'],
        'heart_rate': user_df['resting_pulse'] if 'resting_pulse' in user_df.columns else None,
        'steps': user_df['steps_count'] if 'steps_count' in user_df.columns else None,
        'sleep_quality': user_df['sleep_duration'] if 'sleep_duration' in user_df.columns else None
    })
    
    # Fill any missing values
    historical_data = historical_data.fillna(method='ffill').fillna(0)
    
    # Check the number of features in the dataset
    # The model expects 29 features, so we need to make sure we have all of them
    print(f"Number of features in classification_data: {len(classification_data.columns)}")
    
    # If we're missing features, add dummy features with zeros
    # This is based on the fallback model expecting 10 timesteps * 28 features = 280 shape
    required_feature_count = 29  # From the error message
    
    if len(classification_data.columns) < required_feature_count:
        missing_features = required_feature_count - len(classification_data.columns)
        print(f"Adding {missing_features} dummy features to match model expectations")
        
        # Add dummy features with zeros
        for i in range(missing_features):
            dummy_feature_name = f"dummy_feature_{i+1}"
            classification_data[dummy_feature_name] = 0.0
    
    return classification_data, historical_data

@app.route('/api/v1/bulk-data', methods=['POST'])
def submit_bulk_data():
    """
    Endpoint to receive multiple data points at once
    
    Expected JSON payload:
    {
        "user_id": "user123",
        "data_points": [
            {
                "timestamp": "2023-04-01T12:34:56Z",
                "vitals": {
                    "heart_rate": 75,
                    "steps": 8500,
                    ...
                }
            },
            {
                "timestamp": "2023-04-01T13:34:56Z",
                "vitals": {
                    ...
                }
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Extract data from request
        user_id = data.get('user_id')
        data_points = data.get('data_points', [])
        
        # Validate required fields
        if not user_id or not data_points:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: user_id or data_points"
            }), 400
            
        # Process all data points
        for point in data_points:
            timestamp = point.get('timestamp')
            vitals = point.get('vitals', {})
            
            if not timestamp or not vitals:
                continue  # Skip invalid data points
            
            # Convert timestamp to datetime
            try:
                timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                continue  # Skip invalid timestamps
                
            # Process vitals data
            process_watch_data(user_id, timestamp_dt, vitals)
        
        # Return current data count
        data_count = len(user_data_store.get(user_id, []))
        
        # Check if we have enough data for prediction
        has_prediction = data_count >= 50
        
        return jsonify({
            "status": "success",
            "message": f"Processed {len(data_points)} data points",
            "user_id": user_id,
            "data_points_collected": data_count,
            "prediction_available": has_prediction
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/v1/users/<user_id>/history', methods=['GET'])
def get_user_history(user_id):
    """
    Get stored history for a specific user
    """
    if user_id not in user_data_store:
        return jsonify({
            "status": "error",
            "message": f"User {user_id} not found"
        }), 404
    
    # Return basic stats about user's data
    data_count = len(user_data_store[user_id])
    first_date = min(entry['date'] for entry in user_data_store[user_id])
    last_date = max(entry['date'] for entry in user_data_store[user_id])
    
    return jsonify({
        "status": "success",
        "user_id": user_id,
        "data_points": data_count,
        "first_date": first_date.isoformat(),
        "last_date": last_date.isoformat(),
        "has_sufficient_data": data_count >= 50
    })

@app.route('/api/v1/reset/<user_id>', methods=['POST'])
def reset_user_data(user_id):
    """
    Reset stored history for a specific user
    """
    if user_id in user_data_store:
        user_data_store[user_id] = []
    
    return jsonify({
        "status": "success",
        "message": f"Data for user {user_id} has been reset"
    })

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
