import streamlit as st
import numpy as np
import logging
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError as e:
    logger.error(f"Error importing cv2: {e}. Please ensure opencv-python-headless is installed.")
    cv2 = None # Set cv2 to None so subsequent calls will fail gracefully


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ResNetForger(nn.Module):
    """ResNet18 model for forgery detection"""
    def __init__(self, num_classes=2):
        super(ResNetForger, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class ImageProcessor:
    """Image processing utilities for both scanner and forgery detection"""
    
    @staticmethod
    def load_image_gray(image_path: str) -> Optional[np.ndarray]:
        """Load and convert image to grayscale"""
        try:
            if cv2 is None:
                logger.error("cv2 is not available. Cannot use cv2.imread.")
                img = Image.open(image_path).convert('L')
                img = np.array(img)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"cv2.imread returned None for {image_path}. Trying with PIL as fallback.")
                    img = Image.open(image_path).convert('L')
                    img = np.array(img)
            return img.astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to zero mean and unit variance"""
        if img is None:
            return None
        mean_val = np.mean(img)
        std_val = np.std(img)
        if std_val < 1e-8:
            std_val = 1e-8
        return (img - mean_val) / std_val
    
    @staticmethod
    def residual_wavelet(img: np.ndarray) -> Optional[np.ndarray]:
        """Extract wavelet residual"""
        try:
            try:
                from skimage.restoration import denoise_wavelet
            except ImportError as e:
                logger.error(f"Error importing skimage.restoration: {e}. Please ensure scikit-image is installed.")
                return None
            logger.info("Starting wavelet denoising.")
            denoised = denoise_wavelet(img, method='BayesShrink', mode='soft', rescale_sigma=True)
            logger.info("Wavelet denoising completed.")
            residual = img - denoised
            return residual.astype(np.float32)
        except Exception as e:
            logger.error(f"Wavelet denoising failed: {e}")
            return None
    
    @staticmethod
    def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized correlation coefficient"""
        try:
            a_centered = a - np.mean(a)
            b_centered = b - np.mean(b)
            
            numerator = np.sum(a_centered * b_centered)
            denominator = np.sqrt(np.sum(a_centered**2)) * np.sqrt(np.sum(b_centered**2))
            
            if denominator < 1e-10:
                return 0.0
            return float(numerator / denominator)
        except:
            return 0.0
    
    @staticmethod
    def fft_radial_stats(patch: np.ndarray, n_bins: int = 16) -> np.ndarray:
        """Compute radial FFT statistics"""
        try:
            from numpy.fft import fft2, fftshift
            logger.info("Starting FFT radial statistics calculation.")
            
            # Apply window to reduce spectral leakage
            window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
            windowed_patch = patch * window
            
            F = fftshift(fft2(windowed_patch))
            power_spectrum = np.abs(F) ** 2
            
            H, W = power_spectrum.shape
            center_y, center_x = H // 2, W // 2
            
            y, x = np.indices(power_spectrum.shape)
            radius = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            max_radius = np.sqrt(center_y**2 + center_x**2)
            
            if max_radius < 1e-6:
                logger.warning("Max radius too small for FFT, returning zeros.")
                return np.zeros(n_bins, dtype=np.float32)
            
            radius_normalized = radius / max_radius
            bins = np.linspace(0, 1.0, n_bins + 1)
            
            features = []
            for i in range(n_bins):
                mask = (radius_normalized >= bins[i]) & (radius_normalized < bins[i + 1])
                if np.any(mask):
                    mean_power = np.mean(power_spectrum[mask])
                    features.append(np.log1p(mean_power))
                else:
                    features.append(0.0)
            
            features = np.array(features, dtype=np.float32)
            
            # Normalize features
            mean_feat = np.mean(features)
            std_feat = np.std(features)
            if std_feat > 1e-8:
                features = (features - mean_feat) / std_feat
            logger.info("FFT radial statistics calculation completed.")
            
            return features
        except Exception as e:
            logger.warning(f"FFT feature extraction failed: {e}")
            return np.zeros(n_bins, dtype=np.float32)

class ScannerPredictor:
    """Scanner prediction using trained models"""
    
    def __init__(self):
        self.models = {}
        self.fingerprints = {}
        self.labels = []
        self.load_models()
    
    def load_models(self):
        """Load all trained scanner prediction models"""
        try:
            # Load metadata
            metadata_path = "artifacts/metadata.joblib"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.fingerprints = metadata['fingerprints']
                self.labels = metadata['labels']
                logger.info(f"Loaded fingerprints for {len(self.labels)} scanners")
            else:
                logger.warning("Metadata file not found")
                return
            
            # Load SVM model
            svm_path = "artifacts/svm.joblib"
            if os.path.exists(svm_path):
                svm_data = joblib.load(svm_path)
                self.models['SVM'] = svm_data['model']
                logger.info("Loaded SVM model")
            
            # Load Random Forest model
            rf_path = "artifacts/random_forest.joblib"
            if os.path.exists(rf_path):
                rf_data = joblib.load(rf_path)
                self.models['Random Forest'] = rf_data['model']
                logger.info("Loaded Random Forest model")
            
            # Load XGBoost model
            xgb_path = "artifacts/xgboost.joblib"
            if os.path.exists(xgb_path):
                xgb_data = joblib.load(xgb_path)
                self.models['XGBoost'] = xgb_data['model']
                logger.info("Loaded XGBoost model")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def extract_scanner_features(self, image_path: str) -> Dict:
        """Extract features for scanner prediction"""
        try:
            # Load and process image
            img = ImageProcessor.load_image_gray(image_path)
            if img is None:
                return {"error": "Image loading failed."}
            
            img_norm = ImageProcessor.normalize_image(img)
            if img_norm is None:
                return {"error": "Image normalization failed."}
            
            residual = ImageProcessor.residual_wavelet(img_norm)
            if residual is None:
                return {"error": "Wavelet residual extraction failed."}
            
            # Extract correlation features
            corr_feats = []
            for label in self.labels:
                if label in self.fingerprints:
                    # Ensure shapes match
                    if residual.shape != self.fingerprints[label].shape:
                        min_h = min(residual.shape[0], self.fingerprints[label].shape[0])
                        min_w = min(residual.shape[1], self.fingerprints[label].shape[1])
                        res_crop = residual[:min_h, :min_w]
                        fp_crop = self.fingerprints[label][:min_h, :min_w]
                        corr = ImageProcessor.correlation_coefficient(res_crop, fp_crop)
                    else:
                        corr = ImageProcessor.correlation_coefficient(residual, self.fingerprints[label])
                    corr_feats.append(corr)
                else:
                    corr_feats.append(0.0)
            
            # Extract FFT features
            fft_feats = ImageProcessor.fft_radial_stats(residual)
            
            # Combine features
            combined_features = np.concatenate([corr_feats, fft_feats])
            return {"features": combined_features.reshape(1, -1)}
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {"error": f"Feature extraction failed: {e}"}
    
    def predict_scanner(self, image_path: str, model_name: str = 'SVM') -> Dict:
        """Predict scanner for an image"""
        try:
            if model_name not in self.models:
                return {"error": f"Model {model_name} not available"}
            
            features_result = self.extract_scanner_features(image_path)
            if "error" in features_result:
                return {"error": features_result["error"]}
            features = features_result["features"]
            
            model = self.models[model_name]
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
            
            result = {
                "predicted_scanner": self.labels[prediction],
                "confidence": float(probabilities[prediction]) if probabilities is not None else None,
                "all_probabilities": {self.labels[i]: float(prob) for i, prob in enumerate(probabilities)} if probabilities is not None else None
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

class ForgeryDetector:
    """Forgery detection using ResNet models"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the forgery detection model"""
        try:
            # Try to load the full model first
            model_path = "models/resnet18_full.pth"
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.eval()
                logger.info("Loaded ResNet18 full model")
            else:
                # Try to load weights only
                weights_path = "models/resnet18_weights.pth"
                if os.path.exists(weights_path):
                    self.model = ResNetForger(num_classes=2)
                    self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                    self.model.eval()
                    logger.info("Loaded ResNet18 weights")
                else:
                    logger.warning("No forgery detection model found")
                    return
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.error(f"Failed to load forgery detection model: {e}")
    
    def predict_forgery(self, image_path: str) -> Dict:
        """Predict if image is forged or original"""
        try:
            if self.model is None:
                return {"error": "Forgery detection model not available"}
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            result = {
                "is_forged": bool(prediction),
                "confidence": float(confidence),
                "original_prob": float(probabilities[0][0].item()),
                "forged_prob": float(probabilities[0][1].item())
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 AI Trace Finder</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Scanner Prediction & Forgery Detection</h2>', unsafe_allow_html=True)
    
    # Initialize models
    with st.spinner("Loading models..."):
        scanner_predictor = ScannerPredictor()
        forgery_detector = ForgeryDetector()
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Model selection for scanner prediction
    available_models = list(scanner_predictor.models.keys())
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Select Scanner Prediction Model:",
            available_models,
            index=0
        )
    else:
        st.sidebar.warning("No scanner prediction models available")
        selected_model = None
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document or image for analysis:",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'pdf', 'docx', 'doc'],
        help="Upload a document (PDF, DOCX, DOC) or image to analyze for scanner prediction and forgery detection"
    )
    
    if uploaded_file is not None:
        # Initialize document converter
        converter = DocumentConverter()
        
        # Get file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Check if it's a document that needs conversion
        if file_extension in ['.pdf', '.docx', '.doc']:
            st.info(f"📄 Document detected: {uploaded_file.name}")
            
            # Convert document to images
            with st.spinner("Converting document to images..."):
                try:
                    # Convert from bytes
                    file_bytes = uploaded_file.getvalue()
                    
                    # Debug information (can be removed in production)
                    with st.expander("Debug Information", expanded=False):
                        st.write(f"File size: {len(file_bytes)} bytes")
                        st.write(f"File extension: {file_extension}")
                        st.write(f"Supported formats: {converter.get_supported_formats()}")
                        st.write(f"Is supported: {converter.is_supported(file_extension)}")
                    
                    images = converter.convert_from_bytes(file_bytes, file_extension, max_pages=3)
                    
                    if not images:
                        st.error("Failed to convert document to images - no images returned")
                        
                        # Check PDF signature
                        if file_extension == '.pdf':
                            if not file_bytes.startswith(b'%PDF'):
                                st.error("❌ Invalid PDF signature - file may be corrupted")
                                st.write(f"First 20 bytes: {file_bytes[:20]}")
                            else:
                                st.success("✅ Valid PDF signature detected")
                        
                        st.write("**This might be due to:**")
                        st.write("- Unsupported document format")
                        st.write("- Corrupted document file")
                        st.write("- Missing dependencies (PyMuPDF, python-docx)")
                        st.write("- Document is empty or has no content")
                        
                        # Try to provide more specific help
                        if file_extension == '.pdf':
                            st.write("**For PDF files, please ensure:**")
                            st.write("- The PDF is not password protected")
                            st.write("- The PDF contains at least one page")
                            st.write("- The PDF is not corrupted")
                            st.write("- The PDF is not encrypted")
                            
                            # Try to open PDF with PyMuPDF to get more info
                            try:
                                import fitz
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(file_bytes)
                                    tmp_path = tmp_file.name
                                
                                doc = fitz.open(tmp_path)
                                st.write(f"**PDF Info:**")
                                st.write(f"- Pages: {len(doc)}")
                                if len(doc) > 0:
                                    page = doc[0]
                                    st.write(f"- Page size: {page.rect}")
                                    st.write(f"- Page rotation: {page.rotation}")
                                doc.close()
                                os.unlink(tmp_path)
                            except Exception as e:
                                st.write(f"**PDF Analysis failed:** {e}")
                                
                        elif file_extension == '.docx':
                            st.write("**For DOCX files, please ensure:**")
                            st.write("- The document contains text content")
                            st.write("- The document is not corrupted")
                            st.write("- The document is not password protected")
                        
                        st.stop()
                    
                    st.success(f"✅ Converted to {len(images)} page(s)")
                    
                except Exception as e:
                    st.error(f"Document conversion failed: {str(e)}")
                    
                    # Show detailed error information
                    with st.expander("Error Details", expanded=True):
                        st.code(str(e))
                        
                        # Show traceback for debugging
                        import traceback
                        st.write("**Full error traceback:**")
                        st.code(traceback.format_exc())
                    
                    st.write("**Please check:**")
                    st.write("- File format is supported")
                    st.write("- File is not corrupted")
                    st.write("- Required dependencies are installed")
                    st.write("- File is not password protected")
                    
                    st.stop()
        else:
            # It's an image file
            images = []
            try:
                # Save uploaded file temporarily with original extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # For TIF files, try to preserve the original format for better feature extraction
                if file_extension in ['tif', 'tiff']:
                    # Keep the original TIF file for analysis
                    images = [tmp_path]  # Store the path instead of converting to array
                else:
                    # Convert other formats to numpy array
                    img = Image.open(tmp_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)
                    images = [img_array]
                    
                    # Clean up the temporary file for non-TIF files
                    os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Failed to process image: {str(e)}")
                st.stop()
        
        # Display images and results
        for page_num, img_data in enumerate(images):
            if len(images) > 1:
                st.markdown(f"### Page {page_num + 1}")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Document Preview" if file_extension in ['.pdf', '.docx', '.doc'] else "Uploaded Image")
                
                # Handle both file paths and image arrays
                if isinstance(img_data, str):
                    # It's a file path (TIF file)
                    display_img = Image.open(img_data)
                    st.image(display_img, caption=f"Page {page_num + 1}" if len(images) > 1 else "Document", use_container_width=True)
                    analysis_path = img_data  # Use the original TIF file for analysis
                else:
                    # It's a numpy array
                    display_img = Image.fromarray(img_data)
                    st.image(display_img, caption=f"Page {page_num + 1}" if len(images) > 1 else "Document", use_container_width=True)
                    
                    # Save as temporary file for analysis
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        display_img.save(tmp_file.name)
                        analysis_path = tmp_file.name

            with col2:
                st.subheader("Analysis Results")
            
                # Scanner Prediction
                if selected_model and scanner_predictor.models:
                    with st.spinner("Analyzing scanner..."):
                        scanner_result = scanner_predictor.predict_scanner(analysis_path, selected_model)
                    
                    if "error" not in scanner_result:
                        st.markdown("### 🔍 Scanner Prediction")
                        st.markdown(f"**Predicted Scanner:** {scanner_result['predicted_scanner']}")
                        
                        if scanner_result['confidence'] is not None:
                            confidence = scanner_result['confidence']
                            if confidence > 0.7:
                                conf_class = "confidence-high"
                            elif confidence > 0.4:
                                conf_class = "confidence-medium"
                            else:
                                conf_class = "confidence-low"
                            
                            st.markdown(f"**Confidence:** <span class='{conf_class}'>{confidence:.2%}</span>", unsafe_allow_html=True)
                        
                        # Show all probabilities
                        if scanner_result['all_probabilities']:
                            st.markdown("**All Scanner Probabilities:**")
                            for scanner, prob in scanner_result['all_probabilities'].items():
                                st.write(f"- {scanner}: {prob:.2%}")
                    else:
                        st.error(f"Scanner prediction error: {scanner_result['error']}")
                
                # Forgery Detection
                with st.spinner("Analyzing for forgery..."):
                    forgery_result = forgery_detector.predict_forgery(analysis_path)
                
                if "error" not in forgery_result:
                    st.markdown("### 🛡️ Forgery Detection")
                    
                    if forgery_result['is_forged']:
                        st.markdown("**Result:** <span class='confidence-low'>⚠️ FORGED DOCUMENT</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Result:** <span class='confidence-high'>✅ ORIGINAL DOCUMENT</span>", unsafe_allow_html=True)
                    
                    confidence = forgery_result['confidence']
                    if confidence > 0.7:
                        conf_class = "confidence-high"
                    elif confidence > 0.4:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    
                    st.markdown(f"**Confidence:** <span class='{conf_class}'>{confidence:.2%}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Original Probability:** {forgery_result['original_prob']:.2%}")
                    st.markdown(f"**Forged Probability:** {forgery_result['forged_prob']:.2%}")
                else:
                    st.error(f"Forgery detection error: {forgery_result['error']}")
                
                # Clean up temporary files
                if os.path.exists(analysis_path):
                    if isinstance(img_data, str) and analysis_path == img_data:
                        pass # Do not delete the original TIF file
                    else:
                        os.unlink(analysis_path)
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### Welcome to AI Trace Finder! 🔍
        
        This application provides two main functionalities:
        
        #### 1. Scanner Prediction 📱
        - Identifies which scanner was used to create the document
        - Uses machine learning models trained on scanner fingerprints
        - Supports multiple models: SVM, Random Forest, and XGBoost
        
        #### 2. Forgery Detection 🛡️
        - Determines if a document is original or forged
        - Uses deep learning (ResNet18) for classification
        - Provides confidence scores for both classes
        
        #### How to Use:
        1. Upload a document or image using the file uploader above
        2. Select your preferred scanner prediction model from the sidebar
        3. View the analysis results for both scanner prediction and forgery detection
        
        #### Supported Formats:
        - **Documents**: PDF, DOCX, DOC
        - **Images**: PNG, JPG, JPEG, TIF, TIFF, BMP
        - Documents will be automatically converted to images for analysis
        - Multi-page documents will be processed page by page
        """)
        
        # Show model status
        st.markdown("### Model Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Scanner Prediction Models:**")
            if scanner_predictor.models:
                for model_name in scanner_predictor.models.keys():
                    st.markdown(f"✅ {model_name}")
                st.markdown(f"📊 Available scanners: {', '.join(scanner_predictor.labels)}")
            else:
                st.markdown("❌ No models loaded")
        
        with col2:
            st.markdown("**Forgery Detection Model:**")
            if forgery_detector.model is not None:
                st.markdown("✅ ResNet18")
                st.markdown(f"🖥️ Device: {forgery_detector.device}")
            else:
                st.markdown("❌ Model not loaded")

if __name__ == "__main__":
    main()
