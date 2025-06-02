#!/usr/bin/env python3

# Simple test script to check basic functionality
print("Testing basic Python functionality...")

try:
    import sys
    print(f"Python version: {sys.version}")
    
    import os
    print("OS module imported successfully")
    
    import json
    print("JSON module imported successfully")
    
    # Try importing fastapi without pandas
    try:
        from fastapi import FastAPI
        print("FastAPI imported successfully")
        
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "Hello World", "status": "API is working"}
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy", "service": "prediction-api"}
        
        print("FastAPI app created successfully")
        print("You can run this with: uvicorn test_simple:app --host 0.0.0.0 --port 8000")
        
    except ImportError as e:
        print(f"FastAPI import failed: {e}")
    
    # Try importing pandas
    try:
        import pandas as pd
        print("Pandas imported successfully")
    except ImportError as e:
        print(f"Pandas import failed: {e}")
        print("This is the main issue preventing the prediction.py from running")
    
except Exception as e:
    print(f"Error: {e}")

print("Test completed.")