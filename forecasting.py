import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class VitalSignForecaster:
    def __init__(self, sequence_length=24):
        """
        Initialize the forecasting model
        
        Args:
            sequence_length: Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.scalers = {}  # Dictionary to store scalers for each feature
        self.models = {}   # Dictionary to store models for each feature
        self.feature_names = []  # List of feature names being forecasted
        self.validation_metrics = {}  # Store validation metrics for each feature
        
    def create_sequences(self, data, feature_name):
        """
        Create sequences for LSTM training
        
        Args:
            data: pandas Series or numpy array of vital sign measurements
            feature_name: Name of the feature being processed
            
        Returns:
            X: Sequences for input
            y: Target values
        """
        # Convert to numpy array if it's a pandas Series
        if isinstance(data, pd.Series):
            values = data.to_numpy()
        else:
            values = np.array(data)
            
        # Ensure values are 2D for scaler
        values_2d = values.reshape(-1, 1)
        
        # Initialize scaler if not already done for this feature
        if feature_name not in self.scalers:
            self.scalers[feature_name] = MinMaxScaler()
            
        # Scale the data
        scaled_data = self.scalers[feature_name].fit_transform(values_2d).flatten()
        
        sequences = []
        targets = []
        
        # Create sequences
        for i in range(len(scaled_data) - self.sequence_length - 24):
            seq = scaled_data[i:i+self.sequence_length]
            target = scaled_data[i+self.sequence_length:i+self.sequence_length+24]
            sequences.append(seq)
            targets.append(target)
            
        if not sequences:
            raise ValueError(f"Not enough data to create sequences for {feature_name}. "
                           f"Need at least {self.sequence_length + 25} points.")
            
        # Reshape sequences for LSTM [samples, time steps, features]
        X = np.array(sequences).reshape(-1, self.sequence_length, 1)
        y = np.array(targets)
        
        return X, y
    
    def build_model(self, feature_name):
        """
        Build LSTM model for a specific vital sign
        
        Args:
            feature_name: Name of the feature for which to build the model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, 1), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(24)  # Predict next 24 hours
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.models[feature_name] = model
        
    def train(self, data_dict, epochs=50, batch_size=32, verbose=0, validation_split=0.2):
        """
        Train LSTM models for multiple vital signs
        
        Args:
            data_dict: Dictionary mapping feature names to pandas Series or numpy arrays
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level for training
            validation_split: Fraction of data to use for validation
        """
        self.feature_names = list(data_dict.keys())
        
        for feature_name, historical_data in data_dict.items():
            print(f"Training forecasting model for {feature_name}...")
            
            try:
                # Create sequences
                X, y = self.create_sequences(historical_data, feature_name)
                
                # Build model if not already built
                if feature_name not in self.models:
                    self.build_model(feature_name)
                
                # Train the model
                history = self.models[feature_name].fit(
                    X, y, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=verbose,
                    validation_split=validation_split
                )
                
                # Store validation metrics
                val_loss = history.history['val_loss'][-1]
                self.validation_metrics[feature_name] = {
                    'val_loss': val_loss,
                    'val_mse': val_loss  # MSE is the loss function
                }
                
                print(f"  Completed training for {feature_name}. Validation MSE: {val_loss:.4f}")
                
            except Exception as e:
                print(f"  Error training model for {feature_name}: {e}")
    
    def predict(self, recent_data_dict):
        """
        Predict usual values for the next 24 hours for multiple vital signs
        
        Args:
            recent_data_dict: Dictionary mapping feature names to recent measurements
            
        Returns:
            Dictionary mapping feature names to predicted values
        """
        predictions = {}
        
        for feature_name, recent_data in recent_data_dict.items():
            if feature_name not in self.models:
                print(f"Warning: No trained model for {feature_name}")
                continue
                
            try:
                # Convert to numpy array if needed
                if isinstance(recent_data, pd.Series):
                    values = recent_data.to_numpy()
                else:
                    values = np.array(recent_data)
                    
                # Ensure we have enough data
                if len(values) < self.sequence_length:
                    print(f"Warning: Not enough data for {feature_name}. Skipping.")
                    continue
                
                # Take the last sequence_length points
                values = values[-self.sequence_length:]
                
                # Reshape and scale
                values_2d = values.reshape(-1, 1)
                scaled_input = self.scalers[feature_name].transform(values_2d)
                
                # Reshape for LSTM [samples, time steps, features]
                scaled_input = scaled_input.reshape(1, self.sequence_length, 1)
                
                # Get prediction
                scaled_prediction = self.models[feature_name].predict(scaled_input, verbose=0)[0]
                
                # Reshape prediction for inverse transform
                scaled_prediction = scaled_prediction.reshape(-1, 1)
                
                # Inverse transform
                prediction = self.scalers[feature_name].inverse_transform(scaled_prediction)
                
                predictions[feature_name] = prediction.flatten()
                
            except Exception as e:
                print(f"Error predicting {feature_name}: {e}")
        
        return predictions
    
    def calculate_unusualness(self, current_values_dict, predicted_values_dict):
        """
        Calculate unusualness metrics for multiple vital signs
        
        Args:
            current_values_dict: Dictionary mapping feature names to current measurements
            predicted_values_dict: Dictionary mapping feature names to predicted values
            
        Returns:
            Dictionary mapping feature names to (unusualness values, normalized unusualness values)
        """
        unusualness_dict = {}
        
        for feature_name in predicted_values_dict.keys():
            if feature_name not in current_values_dict:
                continue
                
            current_values = current_values_dict[feature_name]
            predicted_values = predicted_values_dict[feature_name]
            
            # Convert to numpy arrays if needed
            if isinstance(current_values, pd.Series):
                current_values = current_values.to_numpy()
            if isinstance(predicted_values, pd.Series):
                predicted_values = predicted_values.to_numpy()
                
            # Ensure arrays are the same length
            min_length = min(len(current_values), len(predicted_values))
            current_values = current_values[:min_length]
            predicted_values = predicted_values[:min_length]
                
            # Calculate absolute difference
            u_t = np.abs(current_values - predicted_values)
            
            # Normalize
            u_max = np.max(u_t) if np.max(u_t) > 0 else 1
            normalized_u = u_t / u_max
            
            unusualness_dict[feature_name] = (u_t, normalized_u)
            
        return unusualness_dict
    
    def evaluate(self, test_data_dict):
        """
        Evaluate the forecasting models on test data
        
        Args:
            test_data_dict: Dictionary mapping feature names to test data
            
        Returns:
            Dictionary mapping feature names to evaluation metrics
        """
        evaluation_metrics = {}
        
        for feature_name, test_data in test_data_dict.items():
            if feature_name not in self.models:
                print(f"Warning: No trained model for {feature_name}")
                continue
                
            try:
                # Create sequences for testing
                X_test, y_test = self.create_sequences(test_data, feature_name)
                
                # Make predictions
                y_pred = self.models[feature_name].predict(X_test, verbose=0)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test.reshape(-1), y_pred.reshape(-1))
                
                evaluation_metrics[feature_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
                
                print(f"Evaluation metrics for {feature_name}:")
                print(f"  MSE: {mse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R²: {r2:.4f}")
                
            except Exception as e:
                print(f"Error evaluating model for {feature_name}: {e}")
        
        return evaluation_metrics