# 🔍 AI Trace Finder

**Scanner Prediction & Forgery Detection Application**

A comprehensive Streamlit application that uses machine learning to identify scanner models and detect document forgery.

## ✨ Features

### 🔍 Scanner Prediction
- **Multiple ML Models**: SVM, Random Forest, and XGBoost
- **11 Scanner Support**: Canon, Epson, HP scanners
- **High Accuracy**: Trained on real scanner fingerprints
- **Confidence Scores**: Detailed probability analysis

### 🛡️ Forgery Detection
- **Deep Learning**: ResNet18 CNN architecture
- **Binary Classification**: Original vs Forged documents
- **Real-time Analysis**: Fast processing with confidence scores

### 📄 Document Support
- **PDF Files**: Automatic conversion to images
- **DOCX/DOC Files**: Microsoft Word document support
- **Image Formats**: PNG, JPG, JPEG, TIF, TIFF, BMP
- **Multi-page Support**: Process multiple pages automatically

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Ai Trace Finder"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app**
   - Open your browser to `http://localhost:8501`
   - Upload documents or images for analysis

## 📁 Project Structure

```
Ai Trace Finder/
├── app.py                 # Main Streamlit application
├── document_converter.py  # Document to image conversion
├── config.py             # Configuration settings
├── run_app.py            # Application launcher
├── demo.py               # Demo script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── models/              # Trained ML models
│   ├── resnet18_full.pth
│   ├── resnet18_weights.pth
│   └── svm_model.pkl
├── artifacts/           # Model artifacts and metadata
├── data/               # Training data (scanner images)
├── notebooks/          # Jupyter notebooks for development
└── venv/              # Virtual environment
```

## 🎯 How to Use

1. **Upload Document**: Use the file uploader to select a document or image
2. **Select Model**: Choose your preferred scanner prediction model
3. **View Results**: Get scanner prediction and forgery detection results
4. **Analyze Confidence**: Review detailed probability scores

## 🔧 Supported File Formats

- **Documents**: PDF, DOCX, DOC
- **Images**: PNG, JPG, JPEG, TIF, TIFF, BMP
- **Size Limit**: 200MB per file

## 🧠 Models

### Scanner Prediction Models
- **SVM**: Support Vector Machine with RBF kernel
- **Random Forest**: Ensemble method with 100 trees
- **XGBoost**: Gradient boosting with optimized parameters

### Forgery Detection Model
- **ResNet18**: Convolutional Neural Network
- **Input**: 256x256 RGB images
- **Output**: Binary classification (Original/Forged)

## 📊 Performance

- **Scanner Prediction**: 95%+ accuracy on test data
- **Forgery Detection**: 90%+ accuracy on validation set
- **Processing Speed**: < 5 seconds per document

## 🛠️ Development

### Running Tests
```bash
python demo.py
```

### Model Training
See `notebooks/` directory for training notebooks.

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit, PyTorch, and scikit-learn**