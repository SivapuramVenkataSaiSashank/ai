#!/bin/bash
echo "Starting AI Trace Finder..."
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your default browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""
streamlit run app.py

