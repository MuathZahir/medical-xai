"""
Simple script to start the API server
"""
from api import app

if __name__ == "__main__":
    print("Starting Medical AI API server...")
    print(f"API will be available at http://{os.getenv('IP', '0.0.0.0')}:{os.getenv('PORT', 8080)}")
    print("Use Ctrl+C to stop the server")
    app.run(host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 8080)))