import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from XAI import MedicalXAISystem
import matplotlib.pyplot as plt
import random

def plot_results(results, patient_id=None, save_fig=False):
    """Plot and display analysis results"""
    title_prefix = f"Patient {patient_id}: " if patient_id else ""
    
    # Show the most important vital sign graph
    if 'most_important_vital' in results and results['most_important_vital'] in results['feature_graphs']:
        vital = results['most_important_vital']
        fig = results['feature_graphs'][vital]
        plt.figure(fig.number)
        plt.title(f"{title_prefix}{vital.replace('_', ' ').title()} - {results['response_level']}")
        
        if save_fig and patient_id:
            plt.savefig(f"patient_{patient_id}_{vital}_graph.png")
        
        plt.show()
    elif 'feature_graphs' in results:
        # If most important vital not available, show the first graph
        for feature, graph in results['feature_graphs'].items():
            plt.figure(graph.number)
            plt.title(f"{title_prefix}{feature.replace('_', ' ').title()} - {results['response_level']}")
            
            if save_fig and patient_id:
                plt.savefig(f"patient_{patient_id}_{feature}_graph.png")
            
            plt.show()
            break  # Only show the first graph
    else:
        # Backward compatibility
        plt.figure(results['heart_rate_graph'].number)
        plt.title(f"{title_prefix}Heart Rate - {results['response_level']}")
        
        if save_fig and patient_id:
            plt.savefig(f"patient_{patient_id}_heart_rate_graph.png")
        
        plt.show()
    
    # Print analysis results
    print(f"\n{title_prefix}Analysis Results:")
    print(f"Response Level: {results['response_level']}")
    print(f"Classification Score: {results['classification_score']:.3f}")
    print(f"Danger Metric: {results['danger_metric']:.3f}")
    
    # Print text explanation
    if 'text_explanation' in results:
        print("\nExplanation:")
        print(results['text_explanation'])
    
    print("\nFeature Importance Analysis:")
    print(f"Most Important Feature: {results['most_important_feature']}")
    print("\nTop 5 Features Ranked by Importance:")
    for feature, importance in results['feature_importance'][:5]:
        print(f"  {feature}: {abs(importance):.3f}")
    
    # Print unusualness metrics if available
    if 'unusualness' in results:
        print("\nUnusualness Metrics:")
        for feature, (u_t, _) in results['unusualness'].items():
            print(f"  {feature}: Mean Deviation = {np.mean(u_t):.3f}")
    
    print("-" * 80)

def analyze_patient(dataset, user_code, xai_system, forecasting_features):
    """
    Analyze a specific patient from the dataset
    
    Args:
        dataset: The full dataset
        user_code: The user code of the patient to analyze
        xai_system: The initialized XAI system
        forecasting_features: List of features to use for forecasting
        
    Returns:
        Analysis results for the patient
    """
    # Get data for the specific user and sort by date
    user_data = dataset[dataset['user_code'] == user_code].sort_values('date')
    
    if len(user_data) < 50:  # Ensure we have enough data
        print(f"Not enough data for user {user_code}. Skipping.")
        return None
    
    # Prepare data for classification
    classification_data = user_data.drop(['Unnamed: 0', 'date', 'user_code', 'target', 'days_to_symptoms'], axis=1)
    
    # Create historical data for forecasting
    historical_data = pd.DataFrame({
        'timestamp': user_data['date'],
        'heart_rate': user_data['resting_pulse'],
        'steps': user_data['steps_count'],
        'sleep_quality': user_data['sleep_duration']
    })
    
    # Train the forecasting models for this patient
    xai_system.train(classification_data, historical_data, forecasting_features)
    
    # Use the most recent data points for analysis
    recent_data_size = 24  # Use last 24 data points for analysis
    simulation_data = user_data.iloc[-recent_data_size:].copy()
    
    # Format simulation data for analysis
    simulation_data = pd.DataFrame({
        'timestamp': simulation_data['date'],
        'heart_rate': simulation_data['resting_pulse'],
        'steps': simulation_data['steps_count'],
        'sleep_quality': simulation_data['sleep_duration']
    })
    
    # Add any other required columns from the dataset
    for col in classification_data.columns:
        if col not in simulation_data.columns and col not in ['timestamp', 'heart_rate', 'steps', 'sleep_quality']:
            simulation_data[col] = user_data.iloc[-recent_data_size:][col].values
    
    # Analyze the patient's data
    results = xai_system.analyze(simulation_data, historical_data)
    
    return results

def find_patients_with_different_responses(dataset, num_patients=4):
    """
    Find patients with different response levels
    
    Args:
        dataset: The full dataset
        num_patients: Number of patients to find
        
    Returns:
        List of user codes for patients with different response levels
    """
    # Get unique user codes with sufficient data
    user_counts = dataset['user_code'].value_counts()
    valid_users = user_counts[user_counts >= 50].index.tolist()
    
    if len(valid_users) < num_patients:
        print(f"Warning: Only {len(valid_users)} users have sufficient data.")
        return valid_users
    
    # Shuffle the list to get random users
    random.shuffle(valid_users)
    
    return valid_users[:num_patients]

def modify_patient_data(user_data, severity_level):
    """
    Modify patient data to simulate different severity levels
    
    Args:
        user_data: DataFrame containing patient data
        severity_level: Level of severity to simulate (0-3)
        
    Returns:
        Modified user data
    """
    modified_data = user_data.copy()
    
    # Modify vital signs based on severity level
    if severity_level == 1:  # Slight Change
        # Slightly increase resting pulse
        modified_data['resting_pulse'] = modified_data['resting_pulse'] * 1.1
        # Slightly decrease steps
        modified_data['steps_count'] = modified_data['steps_count'] * 0.9
    elif severity_level == 2:  # Warning
        # Moderately increase resting pulse
        modified_data['resting_pulse'] = modified_data['resting_pulse'] * 1.3
        # Moderately decrease steps
        modified_data['steps_count'] = modified_data['steps_count'] * 0.7
        # Slightly decrease sleep duration
        modified_data['sleep_duration'] = modified_data['sleep_duration'] * 0.9
    elif severity_level == 3:  # Serious Condition
        # Significantly increase resting pulse
        modified_data['resting_pulse'] = modified_data['resting_pulse'] * 1.5
        # Significantly decrease steps
        modified_data['steps_count'] = modified_data['steps_count'] * 0.4
        # Significantly decrease sleep duration
        modified_data['sleep_duration'] = modified_data['sleep_duration'] * 0.7
    
    return modified_data

if __name__ == "__main__":
    # Load the real dataset
    print("Loading dataset...")
    dataset = pd.read_csv('Dataset/deduplicated_data_.csv')
    
    # Convert date strings to datetime objects
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Define which features to use for forecasting
    forecasting_features = ['heart_rate', 'steps', 'sleep_quality']
    
    # Initialize the XAI system with the pre-trained classification model
    print("\nInitializing XAI system with pre-trained model...")
    xai_system = MedicalXAISystem(
        model_path='tcn_final_model.h5',
        theta_healthy=0.2,    # Threshold for Healthy
        theta_slight=0.4,     # Threshold for Slight Change
        theta_warning=0.6,    # Threshold for Warning
        theta_serious=0.8     # Threshold for Serious Condition
    )
    
    # Find patients for analysis
    print("\nFinding patients for analysis...")
    patient_ids = find_patients_with_different_responses(dataset)
    
    # Analyze each patient with different severity levels
    print("\nAnalyzing patients with different severity levels...")
    
    # Create a list to store patient results
    all_results = []
    
    # Process each patient
    for i, patient_id in enumerate(patient_ids):
        print(f"\nProcessing Patient {i+1} (ID: {patient_id})...")
        
        # Get data for the patient
        user_data = dataset[dataset['user_code'] == patient_id].sort_values('date')
        
        # # Create different severity levels
        # severity_level = i % 4  # 0: Healthy, 1: Slight Change, 2: Warning, 3: Serious Condition
        
        # # Modify the data based on severity level
        # if severity_level > 0:
        #     modified_data = modify_patient_data(user_data, severity_level)
        # else:
        #     modified_data = user_data
        
        # # Create a temporary dataset with this patient's modified data
        # temp_dataset = dataset.copy()
        # temp_dataset.loc[temp_dataset['user_code'] == patient_id] = modified_data
        
        # Analyze the patient
        results = analyze_patient(user_data, patient_id, xai_system, forecasting_features)
        
        if results:
            # Add patient ID to results
            results['patient_id'] = i+1
            results['user_code'] = patient_id
            
            # Store results
            all_results.append(results)
            
            # Plot and display results
            plot_results(results, patient_id=i+1, save_fig=True)
    
    # Print summary of all patients
    print("\n" + "="*40)
    print("SUMMARY OF PATIENT ANALYSIS")
    print("="*40)
    
    for results in all_results:
        patient_id = results['patient_id']
        user_code = results['user_code']
        response_level = results['response_level']
        danger_metric = results['danger_metric']
        
        print(f"Patient {patient_id} (ID: {user_code}): {response_level} - COVID-19 Risk: {danger_metric:.1%}")
    
    print("\nAnalysis complete.")