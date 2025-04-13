import os
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from XAI import MedicalXAISystem
import random

app = Flask(__name__)

# Initialize the XAI system with the pre-trained classification model
model_path = os.path.join(os.path.dirname(__file__), 'tcn_final_model.h5')

# Global variable to store historical data for each user
user_data_store = {}

# Global variable to store XAI models for each user
user_models = {}

# Global variable to store pre-calculated predictions
cached_predictions = {}

# Define which features to use for forecasting (matching main.py)
forecasting_features = ['heart_rate', 'steps', 'sleep_quality']

# Pre-defined demo users
DEMO_USERS = ["demo_user_1", "demo_user_2", "demo_user_3", "demo_user_4"]

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
    Endpoint to make predictions for all demo users at once.
    Returns pre-calculated predictions for fast demo access.
    """
    try:
        # Check if we have cached predictions for all demo users
        if not cached_predictions:
            return jsonify({
                "status": "error",
                "message": "Predictions not yet cached. Please wait for initialization to complete."
            }), 400
            
        # Return the cached predictions with current timestamp
        return jsonify({
            "status": "success",
            "predictions": cached_predictions,
            "timestamp": datetime.now().isoformat()
        })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

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
        
        # If we have enough data, train a model for this user if it doesn't exist
        if has_prediction and user_id not in user_models:
            try:
                # Prepare data for the XAI system
                classification_data, historical_data = prepare_data_for_analysis(user_id)
                
                # Create new model for user
                user_models[user_id] = MedicalXAISystem(
                    model_path=model_path,
                    theta_healthy=0.2,    # Threshold for Healthy
                    theta_slight=0.4,     # Threshold for Slight Change 
                    theta_warning=0.6,    # Threshold for Warning
                    theta_serious=0.8,     # Threshold for Serious Condition
                    w_c=0.8, w_u=0.2      # Weights for classification and unusualness
                )
                
                # Train the forecasting models for this user
                user_models[user_id].train(classification_data, historical_data, forecasting_features)
                print(f"Created and trained model for user {user_id}")
            except Exception as e:
                print(f"Error creating model for user {user_id}: {str(e)}")
                # Continue without model creation, will retry on predict endpoint
        
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
    Reset stored history and model for a specific user
    """
    if user_id in user_data_store:
        user_data_store[user_id] = []
    
    # Also remove the user's model if it exists
    if user_id in user_models:
        del user_models[user_id]
    
    return jsonify({
        "status": "success",
        "message": f"Data and model for user {user_id} has been reset"
    })

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
    # Initialize user data store if this is a new user
    if user_id not in user_data_store:
        user_data_store[user_id] = []
    
    # Create a data entry
    entry = {
        'date': timestamp,
        'vitals': vitals.copy()
    }
    
    # Add to user's data store
    user_data_store[user_id].append(entry)
    
    # Sort entries by date
    user_data_store[user_id] = sorted(user_data_store[user_id], key=lambda x: x['date'])
    
    return vitals  # Return the processed data

def prepare_data_for_analysis(user_id):
    """
    Prepare user data for the XAI system
    
    Args:
        user_id: User ID to retrieve data for
    
    Returns:
        Tuple of (classification_data, historical_data)
    """
    if user_id not in user_data_store or not user_data_store[user_id]:
        raise ValueError(f"No data found for user {user_id}")
    
    # Extract dates and vital signs
    dates = []
    all_vitals = []
    
    for entry in user_data_store[user_id]:
        dates.append(entry['date'])
        all_vitals.append(entry['vitals'])
    
    # Create a DataFrame from the vitals data
    vitals_df = pd.DataFrame(all_vitals)
    
    # Add date column
    vitals_df['date'] = dates
    
    # Check if we have required features for forecasting
    missing_features = [f for f in forecasting_features if f not in vitals_df.columns]
    if missing_features:
        print(f"Warning: Missing required features for forecasting: {missing_features}")
        # Add dummy values for missing features
        for feature in missing_features:
            vitals_df[feature] = 0.0
    
    # Prepare historical data (used for forecasting)
    historical_data = vitals_df.copy()
    
    # Prepare data for classification
    classification_data = prepare_classification_data(vitals_df)
    
    return classification_data, historical_data

def prepare_classification_data(user_df):
    """
    Transform user data to match the expected format for the classification model
    
    Args:
        user_df: DataFrame with user's vitals data
    
    Returns:
        DataFrame ready for classification
    """
    # Prepare data for classification
    # Get all columns except these specific ones
    classification_columns = [col for col in user_df.columns 
                             if col not in ['date', 'user_code', 'target', 'days_to_symptoms', 'Unnamed: 0']]
    
    classification_data = user_df[classification_columns].copy()
    
    # Get information about the model's expected input shape
    required_feature_count = 29  # The model expects 29 features, not 16 as previously set
    
    # If we have fewer features than required, add dummy features
    if len(classification_data.columns) < required_feature_count:
        missing_features = required_feature_count - len(classification_data.columns)
        print(f"Adding {missing_features} dummy features to match model expectations")
        
        # Add dummy features with zeros
        for i in range(missing_features):
            dummy_feature_name = f"dummy_feature_{i+1}"
            classification_data[dummy_feature_name] = 0.0
    
    # If we have too many features, remove excess
    elif len(classification_data.columns) > required_feature_count:
        excess_features = len(classification_data.columns) - required_feature_count
        print(f"Removing {excess_features} excess features to match model expectations")
        
        # Just keep the first required_feature_count columns
        classification_data = classification_data.iloc[:, :required_feature_count]
    
    return classification_data

def load_demo_data():
    """
    Load data for the demo users from the CSV file
    """
    try:
        # Load the dataset
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "deduplicated_data_.csv")
        dataset = pd.read_csv(data_path)
        dataset['date'] = pd.to_datetime(dataset['date'])
        print(f"Loaded dataset with {len(dataset)} samples for demo initialization.")
        
        # Find patients with different response levels
        user_counts = dataset.groupby('user_code')['target'].value_counts().unstack()
        valid_users = user_counts[(user_counts[0] >= 50) & (user_counts[1] >= 50)].index.tolist()
        
        if len(valid_users) < 4:
            print(f"Warning: Only {len(valid_users)} users have sufficient data for both classes. Using what's available.")
            selected_patients = valid_users
        else:
            # Shuffle the list to get random users
            random.shuffle(valid_users)
            selected_patients = valid_users[:4]
        
        print(f"Selected {len(selected_patients)} patients for demo: {selected_patients}")
        
        # Process data for each demo user
        for i, patient_id in enumerate(selected_patients):
            user_id = DEMO_USERS[i]
            
            # Get data for this patient
            patient_data = dataset[dataset['user_code'] == patient_id].copy()
            
            # Get a mix of healthy and disease data if possible
            healthy_data = patient_data[patient_data['target'] == 0]
            diseased_data = patient_data[patient_data['target'] == 1]
            
            print(f"\nProcessing user {user_id} (Patient ID: {patient_id}):")
            print(f"  Healthy data points: {len(healthy_data)}")
            print(f"  Diseased data points: {len(diseased_data)}")
            
            # Try to create a balanced dataset
            if len(healthy_data) >= 25 and len(diseased_data) >= 25:
                # Get 25 samples from each
                selected_healthy = healthy_data.sample(n=25) if len(healthy_data) > 25 else healthy_data
                selected_diseased = diseased_data.sample(n=25) if len(diseased_data) > 25 else diseased_data
                user_data = pd.concat([selected_healthy, selected_diseased])
            else:
                # Use all available data
                user_data = patient_data
            
            # Sort by date
            user_data = user_data.sort_values('date')
            
            print(f"Loaded {len(user_data)} data points for {user_id}")
            
            # Convert to API format and add to user_data_store
            for _, row in user_data.iterrows():
                timestamp = row['date']
                
                # Create vitals dictionary - use all columns except the non-vital ones
                vitals = {}
                for col in row.index:
                    if col not in ['date', 'user_code', 'target', 'days_to_symptoms', 'Unnamed: 0']:
                        # Map internal column names to expected watch vitals
                        watch_name = col
                        if col == 'resting_pulse':
                            watch_name = 'heart_rate'
                        elif col == 'steps_count':
                            watch_name = 'steps'
                        
                        vitals[watch_name] = float(row[col]) if pd.notna(row[col]) else 0
                
                # Process and add to user's data store
                process_watch_data(user_id, timestamp, vitals)
            
            # Create and train model for this user
            print(f"Training model for demo user {user_id}...")
            
            # Prepare data for the XAI system
            classification_data, historical_data = prepare_data_for_analysis(user_id)
            
            # Create new model for user
            user_models[user_id] = MedicalXAISystem(
                model_path=model_path,
                theta_healthy=0.2,    # Threshold for Healthy
                theta_slight=0.4,     # Threshold for Slight Change 
                theta_warning=0.6,    # Threshold for Warning
                theta_serious=0.8,     # Threshold for Serious Condition
                w_c=0.8, w_u=0.2      # Weights for classification and unusualness
            )
            
            # Train the forecasting models for this user
            user_models[user_id].train(classification_data, historical_data, forecasting_features)
            
            print(f"Successfully trained model for demo user {user_id}")
            
            # Pre-calculate prediction and cache it
            # This will make the predict endpoint instant for demos
            print(f"Pre-calculating prediction for user {user_id}...")
            
            try:
                # Get the most recent data for analysis
                recent_data = classification_data.iloc[-24:].copy()
                
                # Ensure timestamp column exists
                if 'timestamp' not in recent_data.columns:
                    # Create a safe copy of timestamps
                    timestamps = historical_data['date'].iloc[-24:].values
                    recent_data['timestamp'] = timestamps
                
                # Run the analysis
                analysis_results = user_models[user_id].analyze(recent_data, historical_data)
                
                # Extract important information
                response_level = analysis_results['response_level']
                text_explanation = analysis_results['text_explanation']
                danger_metric = float(analysis_results['danger_metric'])
                
                # Store in cache
                cached_predictions[user_id] = {
                    "response_level": response_level,
                    "danger_metric": danger_metric,
                    "explanation": text_explanation
                }
                
                print(f"Cached prediction for user {user_id}: {cached_predictions[user_id]['response_level']} (Danger: {danger_metric:.3f})")
                
            except Exception as e:
                print(f"Error pre-calculating prediction for user {user_id}: {str(e)}")
                cached_predictions[user_id] = {
                    "response_level": "Error",
                    "danger_metric": 0.0,
                    "explanation": f"Error during analysis: {str(e)}"
                }
                print(f"Using error response for user {user_id}")
        
        print("\nDemo initialization complete! All predictions cached and ready for fast access.")
        
    except Exception as e:
        import traceback
        print(f"Error initializing demo data: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    load_demo_data()
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
