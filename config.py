"""
Configuration file for AI Trace Finder application
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

# Model file paths
MODEL_PATHS = {
    'metadata': ARTIFACTS_DIR / "metadata.joblib",
    'svm': ARTIFACTS_DIR / "svm.joblib",
    'random_forest': ARTIFACTS_DIR / "random_forest.joblib",
    'xgboost': ARTIFACTS_DIR / "xgboost.joblib",
    'resnet18_full': MODELS_DIR / "resnet18_full.pth",
    'resnet18_weights': MODELS_DIR / "resnet18_weights.pth"
}

# Image processing settings
IMAGE_SETTINGS = {
    'max_size': (1024, 1024),
    'target_size': (256, 256),
    'supported_formats': ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],
    'fft_bins': 16
}

# Model settings
MODEL_SETTINGS = {
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    'batch_size': 1,
    'confidence_threshold': 0.5
}

# UI settings
UI_SETTINGS = {
    'page_title': "AI Trace Finder",
    'page_icon': "🔍",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Scanner prediction settings
SCANNER_SETTINGS = {
    'default_model': 'SVM',
    'available_models': ['SVM', 'Random Forest', 'XGBoost']
}

# Forgery detection settings
FORGERY_SETTINGS = {
    'class_names': ['Original', 'Forged'],
    'confidence_levels': {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.0
    }
}

def get_model_path(model_name: str) -> str:
    """Get the full path for a model file"""
    return str(MODEL_PATHS.get(model_name, ""))

def check_model_files() -> dict:
    """Check which model files exist"""
    status = {}
    for name, path in MODEL_PATHS.items():
        status[name] = path.exists()
    return status

def get_available_models() -> list:
    """Get list of available scanner prediction models"""
    available = []
    for model in SCANNER_SETTINGS['available_models']:
        model_key = model.lower().replace(' ', '_')
        if MODEL_PATHS.get(model_key, Path("")).exists():
            available.append(model)
    return available


