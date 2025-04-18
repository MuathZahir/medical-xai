import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
import sys
import os
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# API endpoint URL
API_BASE_URL = "http://localhost:8080"

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def convert_dataframe_to_api_format(user_data, user_id):
    """
    Convert DataFrame rows to API request format
    
    Args:
        user_data: DataFrame containing patient data
        user_id: User ID for the API
        
    Returns:
        List of API request bodies
    """
    api_requests = []
    
    # Map DataFrame columns to Apple Watch vitals
    df_to_watch_mapping = {
        'resting_pulse': 'heart_rate',
        'steps_count': 'steps',
        'sleep_duration': 'sleep_duration',
        'active_calories': 'active_calories',
        'resting_calories': 'resting_calories',
        'stand_hours': 'stand_hours',
        'exercise_minutes': 'exercise_minutes'
    }
    
    # Extract all available columns that can be used as vitals
    all_vital_columns = [c for c in user_data.columns if c not in ['date', 'user_code', 'Unnamed: 0', 'target', 'days_to_symptoms']]
    
    # For each row in the DataFrame
    for _, row in user_data.iterrows():
        # Extract timestamp
        timestamp = row['date'].isoformat() if isinstance(row['date'], datetime) else row['date']
        
        # Create vitals dictionary
        vitals = {}
        for col in all_vital_columns:
            # Map column name to watch vitals name if mapping exists
            watch_name = df_to_watch_mapping.get(col, col)
            vitals[watch_name] = float(row[col]) if pd.notna(row[col]) else 0
        
        # Create request body
        request_body = {
            "user_id": user_id,
            "timestamp": timestamp,
            "vitals": vitals
        }
        
        api_requests.append(request_body)
    
    return api_requests

def find_patients_with_different_responses(dataset, num_patients=4):
    """
    Find patients with different response levels
    
    Args:
        dataset: The full dataset
        num_patients: Number of patients to find
        
    Returns:
        List of user codes for patients with different response levels
    """
    # Get user codes with sufficient data for both target classes
    user_counts = dataset.groupby('user_code')['target'].value_counts().unstack()
    valid_users = user_counts[(user_counts[0] >= 50) & (user_counts[1] >= 50)].index.tolist()
    
    if len(valid_users) < num_patients:
        print(f"Warning: Only {len(valid_users)} users have sufficient data for both classes.")
        return valid_users
    
    # Shuffle the list to get random users
    random.shuffle(valid_users)
    
    return valid_users[:num_patients]

def send_data_to_api(request_bodies, user_id):
    """
    Send data to the prediction API endpoint using bulk submission
    
    Args:
        request_bodies: List of request bodies to send
        user_id: User ID for the bulk submission
        
    Returns:
        Final API response for prediction (after all data is submitted)
    """
    # First use the bulk data endpoint to quickly submit all data points
    bulk_url = f"{API_BASE_URL}/api/v1/bulk-data"
    predict_url = f"{API_BASE_URL}/api/v1/predict"
    
    print(f"Submitting {len(request_bodies)} data points in bulk...")
    
    # Create the bulk request payload
    bulk_payload = {
        "user_id": user_id,
        "data_points": []
    }
    
    # Extract timestamp and vitals from each request body
    for body in request_bodies:
        bulk_payload["data_points"].append({
            "timestamp": body["timestamp"],
            "vitals": body["vitals"]
        })
    
    # Send bulk data to the API
    try:
        bulk_response = requests.post(bulk_url, json=bulk_payload)
        print(f"Bulk Data Response: Status {bulk_response.status_code}")
        print(bulk_response.json())
        
        # Check if we have enough data for prediction
        response_data = bulk_response.json()
        if response_data.get("prediction_available", False):
            print("\nSufficient data available, making prediction request...")
            
            # Make prediction request (only requires user_id now)
            prediction_payload = {
                "user_id": user_id
            }
            
            prediction_response = requests.post(predict_url, json=prediction_payload)
            print(f"Prediction Response: Status {prediction_response.status_code}")
            return prediction_response
        else:
            print(f"Not enough data for prediction yet. Current count: {response_data.get('data_points_collected', 0)}")
            return bulk_response
    
    except Exception as e:
        print(f"Error sending data to API: {e}")
        return None

def reset_user_data(user_id):
    """Reset stored data for a user"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/reset/{user_id}")
        print(f"Reset User Data Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_user_history(user_id):
    """Get history information for a user"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/users/{user_id}/history")
        print(f"User History Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def evaluate_predictions(user_data, api_prediction):
    """
    Compare API predictions against actual values in the dataset
    
    Args:
        user_data: DataFrame containing patient data with 'target' column
        api_prediction: Dictionary with API prediction results
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Check if target column exists in user data
    if 'target' not in user_data.columns:
        print("Warning: No 'target' column found in dataset. Cannot evaluate predictions.")
        return None
    
    # Extract actual COVID-19 status
    actual_values = user_data['target'].values
    
    # Get the predominant value (most common value)
    actual_status = np.round(np.mean(actual_values))
    
    # Map API response levels to binary values for comparison
    response_level = api_prediction.get('prediction', {}).get('response_level', '')
    predicted_status = 0
    
    if response_level in ['Warning', 'Serious Condition']:
        predicted_status = 1  # COVID-19 positive
    
    # Get danger metric
    danger_metric = api_prediction.get('prediction', {}).get('danger_metric', 0)
    
    # Create evaluation results
    evaluation = {
        "actual_status": int(actual_status),
        "predicted_status": predicted_status,
        "match": int(actual_status) == predicted_status,
        "danger_metric": danger_metric,
        "response_level": response_level,
    }
    
    return evaluation

def print_evaluation_summary(evaluations):
    """
    Print summary of evaluation metrics
    
    Args:
        evaluations: List of evaluation dictionaries
    """
    if not evaluations:
        print("No evaluations available.")
        return
    
    # Extract actual and predicted statuses
    actual = [e['actual_status'] for e in evaluations if e]
    predicted = [e['predicted_status'] for e in evaluations if e]
    
    if len(actual) < 2:
        print("Not enough data to calculate metrics.")
        return
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(actual, predicted, zero_division=0)
        recall = recall_score(actual, predicted, zero_division=0)
        f1 = f1_score(actual, predicted, zero_division=0)
        cm = confusion_matrix(actual, predicted)
        
        # Print metrics
        print("\n" + "="*50)
        print("PREDICTION EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Negative  Positive")
        print(f"Actual Negative    {cm[0][0]}        {cm[0][1]}")
        print(f"      Positive     {cm[1][0]}        {cm[1][1]}")
        
        # Print per-patient results
        print("\nPer-Patient Results:")
        print("-" * 90)
        print(f"{'Patient ID':<15} {'Actual':<10} {'Predicted':<12} {'Response Level':<20} {'Match':<8}")
        print("-" * 90)
        
        for i, e in enumerate(evaluations):
            if e:
                print(f"{i+1:<15} {'Positive' if e['actual_status'] == 1 else 'Negative':<10} "
                      f"{'Positive' if e['predicted_status'] == 1 else 'Negative':<12} "
                      f"{e['response_level']:<20} {'✓' if e['match'] else '✗':<8}")
                
    except Exception as e:
        print(f"Error calculating metrics: {e}")

def main():
    """Main function to test the API"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test the Medical AI API with different options')
    parser.add_argument('--no-upload', action='store_true', help='Skip uploading data (use existing data)')
    parser.add_argument('--no-reset', action='store_true', help='Do not reset user data before testing')
    parser.add_argument('--predict-only', action='store_true', help='Only make prediction requests without any data upload')
    parser.add_argument('--user-id', type=str, help='Specific user ID to test with')
    args = parser.parse_args()
    
    # Load test data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "deduplicated_data_.csv")
    try:
        dataset = pd.read_csv(data_path)
        # Convert 'date' column to datetime
        dataset['date'] = pd.to_datetime(dataset['date'])
        print(f"Loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Test health endpoint
    if not test_health_endpoint():
        print("Health endpoint check failed. Exiting.")
        return
    
    # If predict-only mode is activated, just make prediction requests
    if args.predict_only:
        user_id = args.user_id if args.user_id else "test_user_1"
        print(f"\nMaking prediction request for user {user_id} (predict-only mode)...")
        
        # Make prediction request
        prediction_payload = {
            "user_id": user_id
        }
        
        try:
            prediction_response = requests.post(f"{API_BASE_URL}/api/v1/predict", json=prediction_payload)
            print(f"Prediction Response: Status {prediction_response.status_code}")
            if prediction_response.status_code == 200:
                results = prediction_response.json()
                if "prediction" in results:
                    prediction = results.get("prediction", {})
                    print("\nAnalysis Results:")
                    print(f"Response Level: {prediction.get('response_level')}")
                    print(f"Danger Metric: {prediction.get('danger_metric'):.3f}")
                    print("\nText Explanation:")
                    print(prediction.get("explanation"))
                else:
                    print("No prediction in response. Verify that enough data was previously uploaded.")
            else:
                print(f"Error: {prediction_response.text}")
        except Exception as e:
            print(f"Error making prediction: {e}")
        return
    
    # Find a few patients for testing
    test_patients = find_patients_with_different_responses(dataset, num_patients=4)
    print(f"Selected {len(test_patients)} patients for testing.")
    
    all_results = []
    all_evaluations = []
    
    # Process each patient
    for i, patient_id in enumerate(test_patients):
        user_id = args.user_id if args.user_id else f"test_user_{i+1}"
        
        # Reset user data if not disabled
        if not args.no_reset:
            reset_user_data(user_id)
            print(f"Reset data for user {user_id}")
        else:
            print(f"Skipping data reset for user {user_id}")
        
        # Get data for this patient
        patient_data = dataset[dataset['user_code'] == patient_id].copy()
        
        # Split into 'healthy' (target=0) and 'diseased' (target=1)
        healthy_data = patient_data[patient_data['target'] == 0]
        diseased_data = patient_data[patient_data['target'] == 1]
        
        print(f"\nPatient {i+1} (ID: {patient_id}):")
        print(f"  Healthy data points: {len(healthy_data)}")
        print(f"  Diseased data points: {len(diseased_data)}")
        
        # Randomly select diseased or healthy data to send to API
        if (diseased_data.empty or len(diseased_data) < 10) and (healthy_data.empty or len(healthy_data) < 10):
            print("Not enough data for both diseased and healthy cases. Skipping this patient.")
            continue
        elif diseased_data.empty or len(diseased_data) < 10:
            user_data = healthy_data
        elif healthy_data.empty or len(healthy_data) < 10:
            user_data = diseased_data
        else:
            user_data = healthy_data if np.random.random() < 0.5 else diseased_data
        
        # Convert the dataframe data to API request format
        api_requests = convert_dataframe_to_api_format(user_data, user_id)
        
        # Send the data to the API if not skipped
        final_response = None
        if not args.no_upload:
            print(f"\nSending {len(api_requests)} data points to API for Patient {i+1}...")
            final_response = send_data_to_api(api_requests, user_id)
        else:
            print("\nSkipping data upload, making direct prediction request...")
            # Make prediction request
            prediction_payload = {
                "user_id": user_id
            }
            try:
                final_response = requests.post(f"{API_BASE_URL}/api/v1/predict", json=prediction_payload)
            except Exception as e:
                print(f"Error making prediction: {e}")
                continue
        
        # Check the user's history
        print(f"\nChecking history for User {user_id}...")
        get_user_history(user_id)
        
        # Process results
        if final_response and final_response.status_code == 200:
            results = final_response.json()
            
            if "prediction" in results:
                # Extract and store results
                prediction = results.get("prediction", {})
                
                result_summary = {
                    "patient_id": i+1,
                    "user_code": patient_id,
                    "api_user_id": user_id,
                    "response_level": prediction.get("response_level"),
                    "danger_metric": prediction.get("danger_metric"),
                    "text_explanation": prediction.get("explanation"),
                    "timestamp": results.get("timestamp")
                }
                
                all_results.append(result_summary)
                
                # Print results
                print("\nAnalysis Results:")
                print(f"Response Level: {prediction.get('response_level')}")
                print(f"Danger Metric: {prediction.get('danger_metric'):.3f}")
                print("\nText Explanation:")
                print(prediction.get("explanation"))
                
                # Compare API prediction with actual values in dataset
                print("\nComparing prediction with actual values...")
                evaluation = evaluate_predictions(user_data, results)
                all_evaluations.append(evaluation)
                
                # Print evaluation for this patient
                if evaluation:
                    print(f"Actual COVID-19 status: {'Positive' if evaluation['actual_status'] == 1 else 'Negative'}")
                    print(f"Predicted status: {'Positive' if evaluation['predicted_status'] == 1 else 'Negative'}")
                    print(f"Match: {'Yes' if evaluation['match'] else 'No'}")
            else:
                print("No prediction in response. Verify that enough data was submitted.")
                
        # Stop after first patient if using specified user ID
        if args.user_id:
            break
    
    # Print summary of all patients
    if all_results:
        print("\n" + "="*40)
        print("SUMMARY OF PATIENT ANALYSIS")
        print("="*40)
        
        for results in all_results:
            patient_id = results['patient_id']
            user_code = results['user_code']
            response_level = results['response_level']
            danger_metric = results['danger_metric']
            
            print(f"Patient {patient_id} (ID: {user_code}): {response_level} - COVID-19 Risk: {danger_metric:.3f}")
        
        # Print evaluation metrics summary
        print_evaluation_summary(all_evaluations)
    
    print("\nAPI Testing complete.")

if __name__ == "__main__":
    main()
