#!/usr/bin/env python3
"""
Launcher script for AI Trace Finder Streamlit application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'torch',
        'torchvision',
        'scikit-learn',
        'opencv-python',
        'PIL',
        'numpy',
        'scikit-image',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'opencv-python':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if model files exist"""
    required_files = {
        'artifacts/metadata.joblib': 'Scanner fingerprints metadata',
        'models/resnet18_full.pth': 'Forgery detection model (full)',
        'models/resnet18_weights.pth': 'Forgery detection model (weights)'
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append((file_path, description))
    
    if missing_files:
        print("⚠️ Some model files are missing:")
        for file_path, description in missing_files:
            print(f"   - {file_path} ({description})")
        print("\nThe application will still run but some features may not work.")
    else:
        print("✅ All model files found")
    
    return len(missing_files) == 0

def run_streamlit():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting AI Trace Finder...")
        print("📱 The application will open in your default web browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("\n" + "="*50)
        
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")

def main():
    """Main launcher function"""
    print("AI Trace Finder - Application Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model files
    check_model_files()
    
    print("\n" + "="*40)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()


