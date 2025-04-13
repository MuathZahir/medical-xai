"""
Simple demo client for the Medical XAI API

This script connects to the API and retrieves predictions for the 4 demo users.
No need to send data as the API already has pre-loaded data for each user.
"""
import requests
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Default API URL
# API_URL = "http://localhost:8080"
API_URL = "https://xai-api-osmy7.ondigitalocean.app/"

def test_health_check(base_url):
    """Test the API health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_predictions(base_url):
    """Get predictions for all demo users"""
    try:
        response = requests.post(f"{base_url}/api/v1/predict")
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def main():
    """Main function to run the demo client"""
    parser = argparse.ArgumentParser(description='Medical XAI API Demo Client')
    parser.add_argument('--url', type=str, default=API_URL, help='API base URL')
    args = parser.parse_args()
    
    console = Console()
    
    # Display header
    console.print(Panel.fit(
        "[bold cyan]Medical XAI Demo Client[/bold cyan]",
        subtitle="Connects to the API and retrieves predictions for demo users"
    ))
    
    # Check if API is running
    console.print("\n[bold]Checking API status...[/bold]")
    success, health_data = test_health_check(args.url)
    
    if not success:
        console.print(f"[bold red]Error:[/bold red] API is not responding. {health_data.get('error', '')}")
        return
    
    console.print(f"[bold green]API Status:[/bold green] {health_data.get('status', 'unknown')}")
    console.print(f"[bold green]Message:[/bold green] {health_data.get('message', '')}")
    
    # Get predictions
    console.print("\n[bold]Requesting predictions for demo users...[/bold]")
    status_code, predictions_data = get_predictions(args.url)
    
    if status_code != 200:
        console.print(f"[bold red]Error (Status {status_code}):[/bold red] {predictions_data.get('message', 'Unknown error')}")
        return
    
    # Display results in a table
    predictions = predictions_data.get("predictions", {})
    timestamp = predictions_data.get("timestamp", "Unknown")
    
    console.print(f"\n[bold]Prediction Results (as of {timestamp}):[/bold]\n")
    
    # Create a table for the summary
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("User ID")
    table.add_column("Response Level")
    table.add_column("Danger Metric")
    
    # Add rows to the table
    for user_id, prediction in predictions.items():
        response_level = prediction.get("response_level", "Unknown")
        danger_metric = prediction.get("danger_metric", 0)
        
        # Color code by response level
        level_color = "green"
        if response_level == "Slight Change":
            level_color = "yellow"
        elif response_level == "Warning":
            level_color = "orange"
        elif response_level == "Serious Condition":
            level_color = "red"
        
        table.add_row(
            user_id,
            f"[{level_color}]{response_level}[/{level_color}]",
            f"{danger_metric:.3f}"
        )
    
    console.print(table)
    
    # Display detailed explanations for each user
    console.print("\n[bold]Detailed Explanations:[/bold]\n")
    
    for user_id, prediction in predictions.items():
        explanation = prediction.get("explanation", "No explanation available")
        console.print(Panel(
            Text(explanation),
            title=f"[bold]User: {user_id} - {prediction.get('response_level', 'Unknown')}[/bold]",
            border_style="blue"
        ))
        console.print("\n")

if __name__ == "__main__":
    main()
