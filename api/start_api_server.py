"""
Simple script to start the API server
"""
from api import app

if __name__ == "__main__":
    print("Starting Medical AI API server...")
    print("API will be available at http://localhost:5000")
    print("Use Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=5000, debug=True)
