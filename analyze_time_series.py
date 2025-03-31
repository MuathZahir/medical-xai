import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset
df = pd.read_csv('Dataset/deduplicated_data_.csv')

# Convert date strings to datetime objects for proper time series analysis
df['date'] = pd.to_datetime(df['date'])

# Select a user with sufficient data for analysis
user_counts = df['user_code'].value_counts()
selected_user = user_counts.index[0]  # User with most data points
print(f"Selected user {selected_user} with {user_counts.iloc[0]} data points")

# Get data for selected user and sort by date
user_data = df[df['user_code'] == selected_user].sort_values('date')

# Potential time series features
time_series_features = [
    'hr_daily_mean', 'resting_pulse', 'steps_count', 'sleep_duration',
    'resting_pulse_mean_3d', 'basal_calories_burned_mean_3d', 'total_calories_mean_3d',
    'resting_pulse_std_3d', 'total_calories_std_3d'
]

# Analyze each feature
print("\nTime Series Feature Analysis:")
print("-" * 50)

forecasting_candidates = []

for feature in time_series_features:
    # Calculate basic statistics
    unique_values = user_data[feature].nunique()
    changes = user_data[feature].diff().abs().sum()
    std_dev = user_data[feature].std()
    
    # Determine if feature is suitable for forecasting
    is_candidate = (unique_values > 10) and (changes > 0)
    
    print(f"\nFeature: {feature}")
    print(f"  Unique values: {unique_values}")
    print(f"  Sum of changes: {changes:.2f}")
    print(f"  Standard deviation: {std_dev:.4f}")
    print(f"  Suitable for forecasting: {'Yes' if is_candidate else 'No'}")
    
    if is_candidate:
        forecasting_candidates.append(feature)

# Plot the time series for suitable features
print("\n\nRecommended features for forecasting:")
for feature in forecasting_candidates:
    print(f"- {feature}")

# Create plots for the top 4 forecasting candidates (or all if fewer)
plot_features = forecasting_candidates[:min(4, len(forecasting_candidates))]

if plot_features:
    fig, axes = plt.subplots(len(plot_features), 1, figsize=(12, 3*len(plot_features)))
    if len(plot_features) == 1:
        axes = [axes]  # Make it iterable when there's only one subplot
    
    for i, feature in enumerate(plot_features):
        axes[i].plot(user_data['date'], user_data[feature])
        axes[i].set_title(f"{feature} over time")
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(feature)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png')
    print("\nPlot saved as 'time_series_analysis.png'")
