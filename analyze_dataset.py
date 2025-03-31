import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Dataset/deduplicated_data_.csv')

# Basic dataset info
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample data types:")
print(df.dtypes.head(10))

# Check for missing values
print("\nMissing values:", df.isna().sum().sum())

# Target distribution
print("\nTarget distribution:")
print(df['target'].value_counts())

# User information
print("\nUnique users:", len(df['user_code'].unique()))

# Time range
print("\nTime range:")
print("First date:", df['date'].min())
print("Last date:", df['date'].max())

# Time series feature analysis
print("\nTime series feature analysis:")
time_series_candidates = [
    'hr_daily_mean', 'resting_pulse', 'steps_count', 'sleep_duration',
    'resting_pulse_mean_3d', 'basal_calories_burned_mean_3d', 'total_calories_mean_3d',
    'resting_pulse_std_3d', 'total_calories_std_3d'
]

for col in time_series_candidates:
    print(f"{col}: {df[col].nunique()} unique values")
    # Check if values change over time for a single user
    first_user = df['user_code'].iloc[0]
    user_data = df[df['user_code'] == first_user].sort_values('date')
    changes = user_data[col].diff().abs().sum()
    print(f"  Changes over time for first user: {changes:.2f}")

# Sample of time series data for one user
print("\nSample of time series data for first user:")
first_user = df['user_code'].iloc[0]
user_data = df[df['user_code'] == first_user].sort_values('date').head(5)
print(user_data[['date', 'hr_daily_mean', 'resting_pulse', 'steps_count', 'sleep_duration']])
