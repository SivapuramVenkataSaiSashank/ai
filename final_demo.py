#!/usr/bin/env python3
"""
Final demo script showing complete AI Trace Finder functionality
"""

import os
import tempfile
from PIL import Image
import numpy as np
from app import ScannerPredictor, ForgeryDetector
from document_converter import DocumentConverter

def demo_complete_functionality():
    """Demonstrate the complete AI Trace Finder functionality"""
    print("🔍 AI Trace Finder - Complete Functionality Demo")
    print("=" * 60)
    
    # Initialize all components
    print("\n1. Initializing components...")
    converter = DocumentConverter()
    scanner_predictor = ScannerPredictor()
    forgery_detector = ForgeryDetector()
    
    print(f"   ✅ Document converter: {len(converter.get_supported_formats())} formats supported")
    print(f"   ✅ Scanner models: {len(scanner_predictor.models)} models loaded")
    print(f"   ✅ Forgery detector: {'Available' if forgery_detector.model else 'Not available'}")
    
    # Test with different file types
    test_files = [
        ("test_document.pdf", "PDF Document"),
        ("test_document.docx", "DOCX Document"),
        ("test_image.png", "Image File")
    ]
    
    for file_path, file_type in test_files:
        if os.path.exists(file_path):
            print(f"\n2. Testing {file_type}: {file_path}")
            
            # Convert document to images
            images = converter.convert_document(file_path, max_pages=2)
            print(f"   📄 Converted to {len(images)} page(s)")
            
            # Analyze each page
            for page_num, img_array in enumerate(images):
                print(f"\n   --- Page {page_num + 1} Analysis ---")
                
                # Save as temporary file
                img = Image.fromarray(img_array)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    img.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                # Scanner Prediction
                if scanner_predictor.models:
                    for model_name in scanner_predictor.models.keys():
                        result = scanner_predictor.predict_scanner(tmp_path, model_name)
                        if 'error' not in result:
                            print(f"   🔍 {model_name}: {result['predicted_scanner']} ({result['confidence']:.2%})")
                
                # Forgery Detection
                if forgery_detector.model:
                    result = forgery_detector.predict_forgery(tmp_path)
                    if 'error' not in result:
                        status = 'FORGED' if result['is_forged'] else 'ORIGINAL'
                        print(f"   🛡️ Forgery: {status} ({result['confidence']:.2%})")
                
                # Clean up
                os.unlink(tmp_path)
        else:
            print(f"\n2. Skipping {file_type}: {file_path} not found")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\nThe AI Trace Finder now supports:")
    print("  📄 PDF documents with multi-page processing")
    print("  📝 DOCX documents with text extraction")
    print("  🖼️ Image files (PNG, JPG, TIF, BMP)")
    print("  🔍 Scanner prediction with 3 ML models")
    print("  🛡️ Forgery detection with deep learning")
    print("  🌐 Web interface with Streamlit")
    print("\nTo run the full application:")
    print("  python run_app.py")
    print("  or")
    print("  streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    demo_complete_functionality()
