import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from XAI import MedicalXAISystem
import matplotlib.pyplot as plt

# def generate_synthetic_data(num_days=30, readings_per_day=24):
#     """Generate synthetic training data and heart rate historical data"""
#     # Create timestamps for heart rate data
#     timestamps = pd.date_range(
#         start='2024-01-01',
#         periods=num_days * readings_per_day,
#         freq='H'
#     )
    
#     # Generate heart rate data with daily patterns
#     hour_of_day = timestamps.hour
#     base_heart_rate = 70 + 10 * np.sin(2 * np.pi * hour_of_day / 24)
#     heart_rate = base_heart_rate + np.random.normal(0, 5, len(timestamps))
    
#     # Create heart rate historical data
#     historical_data = pd.DataFrame({
#         'timestamp': timestamps,
#         'heart_rate': heart_rate
#     })
    
#     # Generate classification training data
#     num_samples = 1000
#     classification_data = pd.DataFrame({
#         'heart_rate': np.random.normal(75, 10, num_samples),
#         'blood_oxygen': np.random.normal(97, 2, num_samples),
#         'sleep_quality': np.random.beta(5, 2, num_samples),
#         'steps': np.random.negative_binomial(10, 0.5, num_samples) * 100,
#     })
    
#     # Generate labels
#     conditions = [
#         (classification_data['heart_rate'] > 100) |
#         (classification_data['blood_oxygen'] < 95) |
#         (classification_data['sleep_quality'] < 0.3),
        
#         (classification_data['heart_rate'] > 120) |
#         (classification_data['blood_oxygen'] < 90) |
#         (classification_data['sleep_quality'] < 0.2)
#     ]
#     choices = [1, 2]  # Warning, Emergency
#     classification_data['disease_level'] = np.select(conditions, choices, default=0)
    
#     return historical_data, classification_data

# def generate_simulation_day(anomaly_factor=1.5):
#     """Generate a simulated day of vital signs"""
#     # Generate one day of data with timestamps
#     timestamps = pd.date_range(
#         start=datetime.now().replace(hour=0, minute=0, second=0),
#         periods=24,
#         freq='H'
#     )
    
#     # Generate baseline heart rate pattern
#     hour_of_day = timestamps.hour
#     base_heart_rate = 70 + 10 * np.sin(2 * np.pi * hour_of_day / 24)
    
#     # Add anomaly
#     heart_rate = base_heart_rate * anomaly_factor + np.random.normal(0, 2, 24)
    
#     # Generate other vitals
#     simulation_data = pd.DataFrame({
#         'timestamp': timestamps,
#         'heart_rate': heart_rate,
#         'blood_oxygen': np.random.normal(97, 1, 24),
#         'sleep_quality': np.random.beta(5, 2, 24),
#         'steps': np.random.poisson(500, 24)
#     })
    
#     return simulation_data

def plot_results(results):
    """Plot and display analysis results"""
    # Show feature graphs
    if 'feature_graphs' in results:
        for feature, graph in results['feature_graphs'].items():
            plt.figure(graph.number)
            plt.show()
    else:
        # Backward compatibility
        plt.figure(results['heart_rate_graph'].number)
        plt.show()
    
    # Print analysis results
    print("\nAnalysis Results:")
    print(f"Response Level: {results['response_level']}")
    print(f"Classification Score: {results['classification_score']:.3f}")
    print(f"Danger Metric: {results['danger_metric']:.3f}")
    
    print("\nFeature Importance Analysis:")
    print(f"Most Important Feature: {results['most_important_feature']}")
    print("\nAll Features Ranked by Importance:")
    for feature, importance in results['feature_importance']:
        print(f"  {feature}: {abs(importance):.3f}")
    
    # Print unusualness metrics if available
    if 'unusualness' in results:
        print("\nUnusualness Metrics:")
        for feature, (u_t, _) in results['unusualness'].items():
            print(f"  {feature}: Mean Deviation = {np.mean(u_t):.3f}")

if __name__ == "__main__":
    # Load the real dataset
    print("Loading dataset...")
    dataset = pd.read_csv('Dataset/deduplicated_data_.csv')
    
    # Convert date strings to datetime objects
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Select a user with sufficient data for analysis
    user_counts = dataset['user_code'].value_counts()
    selected_user = user_counts.index[0]  # User with most data points
    print(f"Selected user {selected_user} with {user_counts.iloc[0]} data points")
    
    # Get data for selected user and sort by date
    user_data = dataset[dataset['user_code'] == selected_user].sort_values('date')
    
    # Prepare data for classification
    classification_data = user_data.drop(['Unnamed: 0', 'date', 'user_code', 'target', 'days_to_symptoms'], axis=1)
    
    # Create historical data for forecasting
    # We'll include multiple time series features
    historical_data = pd.DataFrame({
        'timestamp': user_data['date'],
        'heart_rate': user_data['resting_pulse'],
        'steps': user_data['steps_count'],
        'sleep_quality': user_data['sleep_duration']
    })
    
    # Define which features to use for forecasting
    forecasting_features = ['heart_rate', 'steps', 'sleep_quality']
    
    # Initialize the XAI system with the pre-trained classification model
    print("\nInitializing XAI system with pre-trained model...")
    xai_system = MedicalXAISystem(model_path='tcn_final_model.h5')
    
    # Update the classifier's scaler using the real dataset and train the forecasting models
    print("\nUpdating scaler and training forecasting models...")
    xai_system.train(classification_data, historical_data, forecasting_features)
    
    # Generate simulation data (use the most recent data points)
    print("\nPreparing recent data for analysis...")
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
    
    print("\nAnalyzing data...")
    results = xai_system.analyze(simulation_data, historical_data)
    
    # Plot and display results
    print("\nPlotting results...")
    plot_results(results)

# comment