# AI Trace Finder - Setup Instructions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Option 1: Using the launcher script
python run_app.py

# Option 2: Direct Streamlit command
streamlit run app.py
```

### 3. Open in Browser
The application will automatically open in your default browser at `http://localhost:8501`

## 📁 File Structure

```
AI Trace Finder/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── demo.py               # Demo script
├── test_app.py           # Test script
├── run_app.py            # Launcher script
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── SETUP_INSTRUCTIONS.md # This file
├── artifacts/            # Scanner prediction models
│   ├── metadata.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
└── models/               # Forgery detection models
    ├── resnet18_full.pth
    └── resnet18_weights.pth
```

## 🔧 Model Requirements

### Scanner Prediction Models (Required)
- `artifacts/metadata.joblib` - Scanner fingerprints and metadata
- `artifacts/svm.joblib` - SVM model
- `artifacts/random_forest.joblib` - Random Forest model
- `artifacts/xgboost.joblib` - XGBoost model

### Forgery Detection Models (Required)
- `models/resnet18_full.pth` - Complete ResNet18 model (preferred)
- `models/resnet18_weights.pth` - ResNet18 weights only (fallback)

## 🧪 Testing

### Run Tests
```bash
python test_app.py
```

### Run Demo
```bash
python demo.py
```

## 🎯 Features

### Scanner Prediction
- **Models**: SVM, Random Forest, XGBoost
- **Scanners**: 11 different scanner types
- **Features**: Correlation analysis + FFT features
- **Output**: Predicted scanner + confidence scores

### Forgery Detection
- **Model**: ResNet18 deep learning
- **Classes**: Original vs Forged
- **Features**: Deep convolutional features
- **Output**: Classification + probability scores

## 🖼️ Supported Image Formats

- PNG, JPG, JPEG
- TIF, TIFF
- BMP

## 🔍 Usage

1. **Upload Image**: Use the file uploader in the web interface
2. **Select Model**: Choose scanner prediction model from sidebar
3. **View Results**: See both scanner prediction and forgery detection results
4. **Analyze**: Review confidence scores and probabilities

## 🛠️ Troubleshooting

### Common Issues

1. **Models not loading**
   - Check that all model files exist in correct directories
   - Verify file permissions

2. **CUDA errors**
   - Application automatically falls back to CPU
   - No action needed

3. **Memory issues**
   - Try smaller images
   - Restart the application

4. **PyTorch loading errors**
   - Models are loaded with `weights_only=False` for compatibility
   - This is safe for trusted model files

### Performance Tips

- Use images smaller than 1024x1024 for faster processing
- Close other applications to free up memory
- Use CPU mode if GPU memory is limited

## 📊 Model Performance

### Scanner Prediction
- **SVM**: Fast, good accuracy
- **Random Forest**: Robust, handles noise well
- **XGBoost**: High accuracy, best for complex patterns

### Forgery Detection
- **ResNet18**: High accuracy on document classification
- **Confidence**: Reliable probability estimates
- **Speed**: Fast inference on both CPU and GPU

## 🔒 Security Notes

- Model files are loaded with `weights_only=False` for compatibility
- Only use trusted model files
- The application processes images locally (no data sent to external servers)

## 📞 Support

If you encounter issues:

1. Check the test results: `python test_app.py`
2. Run the demo: `python demo.py`
3. Verify all model files are present
4. Check the console output for error messages

## 🎉 Success!

Once everything is working, you should see:
- ✅ All models loaded successfully
- ✅ Web interface accessible at localhost:8501
- ✅ Image upload and analysis working
- ✅ Results displayed with confidence scores


