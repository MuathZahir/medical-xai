import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from classification import MedicalClassifier
from forecasting import VitalSignForecaster
from sklearn.preprocessing import StandardScaler

class MedicalXAISystem:
    def __init__(self, model_path, w_c=0.6, w_u=0.4, theta_healthy=0.2, theta_slight=0.4, theta_warning=0.6, theta_serious=0.8):
        # Load the pre-trained classifier by providing its h5 file path
        self.classifier = MedicalClassifier(model_path)
        self.forecaster = VitalSignForecaster()
        self.w_c = w_c
        self.w_u = w_u
        # Thresholds for different response levels
        self.theta_healthy = theta_healthy
        self.theta_slight = theta_slight
        self.theta_warning = theta_warning
        self.theta_serious = theta_serious
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
        """
        Determine response level based on the danger metric
        
        Args:
            danger_metric: Calculated danger metric value
            
        Returns:
            Response level as a string
        """
        if danger_metric >= self.theta_serious:
            return "Serious Condition"
        elif danger_metric >= self.theta_warning and danger_metric < self.theta_serious:
            return "Warning"
        elif danger_metric >= self.theta_slight and danger_metric < self.theta_warning:
            return "Slight Change"
        else:
            return "Healthy"
    
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

    def generate_explanation(self, results):
        """
        Generate a text explanation based on the analysis results
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Text explanation of the results
        """
        response_level = results['response_level']
        classification_score = results['classification_score']
        most_important_feature = results['most_important_feature']
        
        # Base explanation on response level
        if response_level == "Serious Condition":
            explanation = (
                "SERIOUS CONDITION DETECTED: The patient's vital signs show significant abnormalities "
                "that strongly suggest a COVID-19 infection. Immediate medical attention is recommended. "
            )
        elif response_level == "Warning":
            explanation = (
                "WARNING: The patient's vital signs show concerning patterns that may indicate "
                "a developing COVID-19 infection. Close monitoring is advised. "
            )
        elif response_level == "Slight Change":
            explanation = (
                "SLIGHT CHANGES DETECTED: The patient's vital signs show minor deviations from their normal patterns. "
                "While this may not indicate COVID-19, continued monitoring is recommended. "
            )
        else:  # Healthy
            explanation = (
                "HEALTHY STATUS: The patient's vital signs appear normal and consistent with their typical patterns. "
                "No indications of COVID-19 are present at this time. "
            )
        
        # Add feature-specific information
        if most_important_feature:
            # Clean up feature name for display
            display_name = most_important_feature.replace('_', ' ').title()
            
            explanation += f"\nThe most significant indicator is the patient's {display_name}, "
            
            if response_level in ["Serious Condition", "Warning"]:
                explanation += "which shows abnormal patterns compared to the patient's baseline."
            else:
                explanation += "which remains within expected parameters."
        
        # Add unusualness information if available
        if 'unusualness' in results and results['unusualness']:
            unusual_features = []
            for feature, (u_t, _) in results['unusualness'].items():
                mean_deviation = np.mean(u_t)
                if mean_deviation > 0.3:  # Threshold for mentioning a feature
                    # Clean up feature name for display
                    display_name = feature.replace('_', ' ').title()
                    unusual_features.append(display_name)
            
            if unusual_features:
                explanation += f"\nUnusual patterns were detected in: {', '.join(unusual_features)}."
        
        # Add classification confidence
        explanation += f"\nCOVID-19 risk assessment: {classification_score:.1%} confidence level."
        
        return explanation
    
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
        
        # Determine response level
        response_level = self.determine_response(danger)
        
        # Generate graphs for each feature
        graphs = self.generate_feature_graphs(
            current_values_dict,
            predicted_values_dict,
            current_data['timestamp'].iloc[:min(len(current_data), 24)]
        )
        
        # Find the most important vital sign graph
        most_important_vital = None
        if explanation['most_important_feature'] in self.forecasting_features:
            most_important_vital = explanation['most_important_feature']
        elif self.forecasting_features:
            # If the most important feature isn't a vital sign, use the first vital sign
            most_important_vital = self.forecasting_features[0]
        
        # Generate text explanation
        results = {
            'classification_score': c_t,
            'feature_importance': explanation['feature_importance'],
            'most_important_feature': explanation['most_important_feature'],
            'unusualness': unusualness_dict,
            'danger_metric': danger,
            'response_level': response_level,
            'feature_graphs': graphs,
            'most_important_vital': most_important_vital,
            # For backward compatibility
            'heart_rate_graph': graphs.get('heart_rate', plt.figure()) if 'heart_rate' in graphs else plt.figure(),
            'heart_rate_unusualness': unusualness_dict.get('heart_rate', (np.array([]), np.array([])))[0] if 'heart_rate' in unusualness_dict else np.array([])
        }
        
        # Add text explanation
        results['text_explanation'] = self.generate_explanation(results)
        
        return results