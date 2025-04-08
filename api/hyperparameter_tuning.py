import pandas as pd
import numpy as np
from XAI import MedicalXAISystem
from main import find_patients_with_different_responses, modify_patient_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import itertools
import json
from datetime import datetime
import os

def load_dataset():
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    dataset = pd.read_csv('Dataset/deduplicated_data_.csv')
    # Convert date strings to datetime objects
    dataset['date'] = pd.to_datetime(dataset['date'])
    return dataset

def evaluate_xai_system(dataset, xai_system, forecasting_features, num_patients=20, include_modified=True):
    """
    Evaluate the XAI system's performance with a specific parameter configuration
    
    Args:
        dataset: The full dataset
        xai_system: Initialized XAI system with specific parameters
        forecasting_features: Features to use for forecasting
        num_patients: Number of patients to evaluate
        include_modified: Whether to include artificially modified data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Find patients for evaluation
    patient_ids = find_patients_with_different_responses(dataset, num_patients=num_patients)
    
    actual_responses = []
    predicted_responses = []
    all_evaluations = []
    
    # Process each patient
    for i, patient_id in enumerate(patient_ids):
        if i % 5 == 0:
            print(f"Processing patient {i+1}/{len(patient_ids)}...")
        
        # Get data for the patient
        user_data = dataset[dataset['user_code'] == patient_id].sort_values('date')
        
        if len(user_data) < 50:  # Skip if not enough data
            continue
            
        # If including modified data, create different severity levels
        if include_modified:
            severity_levels = [0, 1, 2, 3]  # 0: Healthy, 1: Slight, 2: Warning, 3: Serious
        else:
            severity_levels = [0]  # Only use original data
            
        for severity_level in severity_levels:
            # Get actual COVID status from the data
            has_covid = np.mean(user_data['target'].values) > 0.5
            actual_status = 1 if has_covid else 0
            
            # For higher severity levels, the actual status should be positive
            if severity_level >= 2:
                actual_status = 1
            
            # Modify the data based on severity level
            if severity_level > 0:
                modified_data = modify_patient_data(user_data, severity_level)
            else:
                modified_data = user_data
            
            # Prepare data for classification
            classification_data = modified_data.drop(['Unnamed: 0', 'date', 'user_code', 'target', 'days_to_symptoms'], 
                                                   axis=1, errors='ignore')
            
            # Create historical data for forecasting
            historical_data = pd.DataFrame({
                'timestamp': modified_data['date'],
                'heart_rate': modified_data['resting_pulse'],
                'steps': modified_data['steps_count'],
                'sleep_quality': modified_data['sleep_duration']
            })

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
    
            
            # Train the forecasting models for this patient
            try:
                xai_system.train(classification_data, historical_data, forecasting_features)
                
                # Use the last 24 data points for analysis
                recent_data = classification_data.iloc[-24:].copy()
                recent_data['timestamp'] = historical_data['timestamp'].iloc[-24:].values
                
                # Get prediction
                results = xai_system.analyze(recent_data, historical_data)
                
                # Map response level to binary prediction
                response_level = results['response_level']
                predicted_status = 1 if response_level in ['Warning', 'Serious Condition'] else 0
                
                # Record results
                actual_responses.append(actual_status)
                predicted_responses.append(predicted_status)
                
                # Save detailed evaluation
                evaluation = {
                    'patient_id': patient_id,
                    'severity_level': severity_level,
                    'actual_status': actual_status,
                    'predicted_status': predicted_status,
                    'response_level': response_level,
                    'classification_score': results['classification_score'],
                    'danger_metric': results['danger_metric'],
                    'match': predicted_status == actual_status
                }
                all_evaluations.append(evaluation)
                
            except Exception as e:
                print(f"Error processing patient {patient_id} with severity {severity_level}: {e}")
    
    # Calculate metrics
    if len(actual_responses) == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'evaluations': []
        }
        
    accuracy = accuracy_score(actual_responses, predicted_responses)
    precision = precision_score(actual_responses, predicted_responses, zero_division=0)
    recall = recall_score(actual_responses, predicted_responses, zero_division=0)
    f1 = f1_score(actual_responses, predicted_responses, zero_division=0)
    cm = confusion_matrix(actual_responses, predicted_responses)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'evaluations': all_evaluations
    }

def grid_search(dataset, param_grid, forecasting_features, num_patients=20, include_modified=True):
    """
    Perform grid search over hyperparameter combinations
    
    Args:
        dataset: The dataset to use
        param_grid: Dictionary of parameter names and possible values
        forecasting_features: Features to use for forecasting
        num_patients: Number of patients to evaluate
        include_modified: Whether to include artificially modified data
        
    Returns:
        List of evaluation results for each parameter combination
    """
    # Generate all combinations of parameters
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    
    for i, combination in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_keys, combination))
        print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}")
        
        # Initialize XAI system with these parameters
        xai_system = MedicalXAISystem(
            model_path='tcn_final_model.h5',
            **params
        )
        
        # Evaluate the system
        metrics = evaluate_xai_system(
            dataset, 
            xai_system, 
            forecasting_features, 
            num_patients=num_patients,
            include_modified=include_modified
        )
        
        # Store results
        result = {
            'parameters': params,
            'metrics': metrics
        }
        results.append(result)
        
        # Print current results
        print(f"Results for {params}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
    return results

def optimize_thresholds(dataset, forecasting_features, num_patients=20, include_modified=True):
    """
    Find optimal threshold values for the XAI system
    
    Args:
        dataset: The dataset to use
        forecasting_features: Features to use for forecasting
        num_patients: Number of patients to evaluate
        include_modified: Whether to include artificially modified data
    """
    # Define parameter grid (possible values for each parameter)
    param_grid = {
        'theta_healthy': [0.1, 0.2, 0.3],
        'theta_slight': [0.3, 0.4, 0.5],
        'theta_warning': [0.5, 0.6, 0.7],
        'theta_serious': [0.7, 0.8, 0.9]
    }
    
    # Additional parameters that could be tuned
    param_grid_extended = {
        'theta_healthy': [0.1, 0.2, 0.3],
        'theta_slight': [0.3, 0.4, 0.5],
        'theta_warning': [0.5, 0.6, 0.7],
        'theta_serious': [0.7, 0.8, 0.9],
        'w_c': [0.4, 0.6, 0.8],  # Weight for classification score
        'w_u': [0.2, 0.4, 0.6]   # Weight for unusualness
    }
    
    # Perform grid search
    results = grid_search(
        dataset, 
        param_grid, 
        forecasting_features, 
        num_patients=num_patients,
        include_modified=include_modified
    )
    
    # Find best parameter combination based on F1 score
    best_result = max(results, key=lambda x: x['metrics']['f1'])
    
    print("\n" + "="*50)
    print("BEST HYPERPARAMETERS")
    print("="*50)
    print(f"Parameters: {best_result['parameters']}")
    print(f"F1 Score: {best_result['metrics']['f1']:.4f}")
    print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"Precision: {best_result['metrics']['precision']:.4f}")
    print(f"Recall: {best_result['metrics']['recall']:.4f}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_tuning_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'best_parameters': best_result['parameters'],
            'best_metrics': best_result['metrics']
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return best_result['parameters']

def main():
    print("=" * 50)
    print("HYPERPARAMETER TUNING FOR MEDICAL XAI SYSTEM")
    print("=" * 50)
    
    # Load dataset
    dataset = load_dataset()
    
    # Define forecasting features (same as in main.py)
    forecasting_features = ['heart_rate', 'steps', 'sleep_quality']
    
    # Set number of patients to use for tuning
    num_patients = 20  # Adjust based on available computation time
    
    # Perform hyperparameter optimization
    best_params = optimize_thresholds(
        dataset, 
        forecasting_features, 
        num_patients=num_patients,
        include_modified=True  # Set to False to use only original data
    )
    
    # Create a tuned XAI system with the optimal parameters
    tuned_xai_system = MedicalXAISystem(
        model_path='tcn_final_model.h5',
        **best_params
    )
    
    # Update API configuration file
    api_config = {
        'forecasting_features': forecasting_features,
        'xai_parameters': best_params,
        'tuning_timestamp': datetime.now().isoformat()
    }
    
    with open('api_config.json', 'w') as f:
        json.dump(api_config, f, indent=2)
    
    print("\nTuned parameters saved to api_config.json")
    print("\nYou can update your API by adding this code to api.py:")
    print("""
    # Load configuration
    with open('api_config.json', 'r') as f:
        config = json.load(f)
        
    # Initialize XAI system with tuned parameters
    xai_system = MedicalXAISystem(
        model_path='tcn_final_model.h5',
        **config['xai_parameters']
    )
    
    # Use tuned forecasting features
    forecasting_features = config['forecasting_features']
    """)

if __name__ == "__main__":
    main()
