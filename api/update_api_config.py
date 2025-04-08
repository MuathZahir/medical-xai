import json
import os
from flask import Flask
from XAI import MedicalXAISystem

def update_api_config_file(config_file='api_config.json'):
    """
    Update the API configuration file with specified hyperparameters
    
    Args:
        config_file: Path to the configuration file
    """
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found. Creating with default values.")
        config = {
            'forecasting_features': ['heart_rate', 'steps', 'sleep_quality'],
            'xai_parameters': {
                'theta_healthy': 0.2,
                'theta_slight': 0.4,
                'theta_warning': 0.6,
                'theta_serious': 0.8,
                'w_c': 0.6,
                'w_u': 0.4
            }
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    return config_file

def create_xai_system_from_config(config_file='api_config.json'):
    """
    Create an XAI system using parameters from the configuration file
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Tuple of (XAI system, forecasting features)
    """
    # Ensure the config file exists
    update_api_config_file(config_file)
    
    # Load the configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    xai_params = config.get('xai_parameters', {})
    forecasting_features = config.get('forecasting_features', ['heart_rate', 'steps', 'sleep_quality'])
    
    # Create the XAI system
    model_path = os.path.join(os.path.dirname(__file__), 'tcn_final_model.h5')
    xai_system = MedicalXAISystem(
        model_path=model_path,
        **xai_params
    )
    
    return xai_system, forecasting_features

def apply_config_to_api(app, config_file='api_config.json'):
    """
    Apply configuration to an existing Flask app
    
    Args:
        app: Flask application instance
        config_file: Path to the configuration file
        
    Returns:
        Tuple of (XAI system, forecasting features)
    """
    xai_system, forecasting_features = create_xai_system_from_config(config_file)
    
    # Store in app context
    app.config['XAI_SYSTEM'] = xai_system
    app.config['FORECASTING_FEATURES'] = forecasting_features
    
    return xai_system, forecasting_features

if __name__ == "__main__":
    # Example usage
    print("Updating API configuration...")
    config_file = update_api_config_file()
    print(f"Configuration file: {config_file}")
    
    # Create a sample Flask app and apply configuration
    app = Flask(__name__)
    xai_system, forecasting_features = apply_config_to_api(app)
    
    print("\nAPI updated with the following configuration:")
    print(f"Forecasting Features: {forecasting_features}")
    
    # Print XAI parameters
    print("\nXAI System Parameters:")
    params = xai_system.__dict__
    for key, value in params.items():
        if key in ['theta_healthy', 'theta_slight', 'theta_warning', 'theta_serious', 'w_c', 'w_u']:
            print(f"  {key}: {value}")
    
    print("\nTo use this configuration in your API, update api.py with:")
    print("""
    # At the top of the file, add:
    from update_api_config import apply_config_to_api
    
    # Then replace the XAI system initialization with:
    xai_system, forecasting_features = apply_config_to_api(app)
    """)
