import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from classification import MedicalClassifier
from forecasting import VitalSignForecaster
from sklearn.preprocessing import StandardScaler

class MedicalXAISystem:
    def __init__(self, model_path, w_c=0.6, w_u=0.4, theta_w=0.5, theta_e=0.8):
        # Load the pre-trained classifier by providing its h5 file path
        self.classifier = MedicalClassifier(model_path)
        self.forecaster = VitalSignForecaster()
        self.w_c = w_c
        self.w_u = w_u
        self.theta_w = theta_w
        self.theta_e = theta_e
        self.forecasting_features = []  # Will be populated during training
        
    def train(self, classification_data, historical_data, forecasting_features=None):
        """
        Update the classifier's scaler and set feature names,
        and train the forecasting model.
        
        Args:
            classification_data: DataFrame containing features for classification
            historical_data: DataFrame containing time series data for forecasting
            forecasting_features: List of features to use for forecasting (default: use 'heart_rate')
        """
        # Determine feature columns for classification: use all columns except the label (if present)
        if 'disease_level' in classification_data.columns:
            features = [col for col in classification_data.columns if col != 'disease_level']
        else:
            features = list(classification_data.columns)
        
        # Fit the scaler on all vital sign features
        scaler = StandardScaler()
        scaler.fit(classification_data[features])
        self.classifier.load_scaler(scaler, features)
        
        # Determine which features to use for forecasting
        if forecasting_features is None:
            # Default to 'heart_rate' if available, otherwise use the first feature
            if 'heart_rate' in historical_data.columns:
                self.forecasting_features = ['heart_rate']
            else:
                self.forecasting_features = [historical_data.columns[1]]  # Skip timestamp column
        else:
            self.forecasting_features = forecasting_features
            
        print(f"Training forecasting models for: {', '.join(self.forecasting_features)}")
        
        # Prepare data for forecasting
        forecasting_data = {}
        for feature in self.forecasting_features:
            if feature in historical_data.columns:
                forecasting_data[feature] = historical_data[feature]
        
        # Train the forecasting models
        self.forecaster.train(forecasting_data, epochs=50, batch_size=32, verbose=0)
        
        # Validate the forecasting models
        self.validate_forecasting(forecasting_data)
    
    def validate_forecasting(self, data_dict, test_size=0.2):
        """
        Validate the forecasting models using a portion of the data
        
        Args:
            data_dict: Dictionary of time series data for each feature
            test_size: Fraction of data to use for testing
        """
        # Prepare test data
        test_data = {}
        for feature, data in data_dict.items():
            # Use the last portion of data for testing
            split_idx = int(len(data) * (1 - test_size))
            test_data[feature] = data.iloc[split_idx:]
        
        # Evaluate the models
        metrics = self.forecaster.evaluate(test_data)
        return metrics
    
    def calculate_danger_metric(self, c_t, unusualness_dict):
        """
        Calculate danger metric using classification score and unusualness
        
        Args:
            c_t: Classification score
            unusualness_dict: Dictionary of unusualness values for each feature
            
        Returns:
            Danger metric value
        """
        # Calculate average unusualness across all features
        if not unusualness_dict:
            return self.w_c * c_t  # If no unusualness data, use only classification
            
        avg_unusualness = 0
        count = 0
        
        for feature, (u_t, _) in unusualness_dict.items():
            avg_unusualness += np.mean(u_t)
            count += 1
            
        if count > 0:
            avg_unusualness /= count
            
        # Get maximum unusualness for normalization
        max_unusualness = 1
        for feature, (u_t, _) in unusualness_dict.items():
            feature_max = np.max(u_t)
            if feature_max > max_unusualness:
                max_unusualness = feature_max
                
        # Calculate danger metric
        return self.w_c * c_t + self.w_u * (avg_unusualness / max_unusualness)
    
    def determine_response(self, danger_metric):
        """Determine response level"""
        if danger_metric >= self.theta_e:
            return "Emergency"
        elif danger_metric >= self.theta_w:
            return "Warning"
        else:
            return "Normal"
    
    def generate_feature_graphs(self, current_values_dict, predicted_values_dict, timestamps):
        """
        Generate comparison graphs for multiple features
        
        Args:
            current_values_dict: Dictionary of current values for each feature
            predicted_values_dict: Dictionary of predicted values for each feature
            timestamps: Timestamps for the data points
            
        Returns:
            Dictionary mapping feature names to matplotlib figures
        """
        graphs = {}
        
        for feature in predicted_values_dict.keys():
            if feature not in current_values_dict:
                continue
                
            current_values = current_values_dict[feature]
            predicted_values = predicted_values_dict[feature]
            
            # Ensure arrays are the same length
            min_length = min(len(current_values), len(predicted_values), len(timestamps))
            current_values = current_values[:min_length]
            predicted_values = predicted_values[:min_length]
            time_points = timestamps[:min_length]
            
            # Create figure
            fig = plt.figure(figsize=(12, 6))
            
            plt.plot(time_points, predicted_values, 'r--', label='Expected Pattern')
            plt.plot(time_points, current_values, 'b-', label='Current Values')
            
            plt.title(f'{feature} Comparison')
            plt.xlabel('Time')
            plt.ylabel(f'{feature} (Normalized)')
            plt.legend()
            plt.grid(True)
            
            # Calculate and display unusualness
            u_t = np.abs(current_values - predicted_values)
            plt.text(0.02, 0.98, f'Mean Deviation: {np.mean(u_t):.2f}', 
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            graphs[feature] = fig
            
        return graphs

    def analyze(self, current_data, historical_data):
        """
        Analyze current data and generate explanation
        
        Args:
            current_data: DataFrame containing current vital sign measurements
            historical_data: DataFrame containing historical measurements
            
        Returns:
            Dictionary of analysis results
        """
        # Get classification prediction and explanation
        c_t = self.classifier.predict(current_data)
        explanation = self.classifier.explain_prediction(current_data)
        
        # Prepare data for forecasting
        recent_data_dict = {}
        current_values_dict = {}
        
        for feature in self.forecasting_features:
            if feature in historical_data.columns:
                # Get recent data for prediction
                recent_data_dict[feature] = historical_data[feature].iloc[-self.forecaster.sequence_length:]
                
                # Get current values for comparison
                if feature in current_data.columns:
                    current_values_dict[feature] = current_data[feature].values
        
        # Get predictions for each feature
        predicted_values_dict = self.forecaster.predict(recent_data_dict)
        
        # Calculate unusualness for each feature
        unusualness_dict = self.forecaster.calculate_unusualness(
            current_values_dict, predicted_values_dict)
        
        # Calculate danger metric
        danger = self.calculate_danger_metric(c_t, unusualness_dict)
        
        # Generate graphs for each feature
        graphs = self.generate_feature_graphs(
            current_values_dict,
            predicted_values_dict,
            current_data['timestamp'].iloc[:min(len(current_data), 24)]
        )
        
        # Generate results
        results = {
            'classification_score': c_t,
            'feature_importance': explanation['feature_importance'],
            'most_important_feature': explanation['most_important_feature'],
            'unusualness': unusualness_dict,
            'danger_metric': danger,
            'response_level': self.determine_response(danger),
            'feature_graphs': graphs,
            # For backward compatibility
            'heart_rate_graph': graphs.get('heart_rate', plt.figure()) if 'heart_rate' in graphs else plt.figure(),
            'heart_rate_unusualness': unusualness_dict.get('heart_rate', (np.array([]), np.array([])))[0] if 'heart_rate' in unusualness_dict else np.array([])
        }
        
        return results