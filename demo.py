#!/usr/bin/env python3
"""
Demo script for AI Trace Finder application
Shows how to use the models programmatically
"""

import os
import sys
from PIL import Image
import numpy as np

def create_demo_image(filename: str = "demo_image.png", size: tuple = (512, 512)):
    """Create a demo image with some text and patterns"""
    # Create a white background
    img_array = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Add some text-like patterns
    for i in range(50, size[0]-50, 30):
        img_array[i:i+3, 50:size[1]-50] = [0, 0, 0]  # Horizontal lines
    
    for j in range(50, size[1]-50, 40):
        img_array[50:size[0]-50, j:j+2] = [0, 0, 0]  # Vertical lines
    
    # Add some noise to simulate scanner artifacts
    noise = np.random.randint(0, 50, (*size, 3), dtype=np.uint8)
    img_array = np.clip(img_array.astype(np.int16) - noise, 0, 255).astype(np.uint8)
    
    # Save the image
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"Created demo image: {filename}")
    return filename

def demo_scanner_prediction():
    """Demonstrate scanner prediction functionality"""
    print("\n" + "="*50)
    print("🔍 SCANNER PREDICTION DEMO")
    print("="*50)
    
    try:
        from app import ScannerPredictor
        
        # Initialize predictor
        predictor = ScannerPredictor()
        
        if not predictor.models:
            print("❌ No scanner prediction models available")
            return
        
        # Create demo image
        demo_img = create_demo_image("scanner_demo.png")
        
        print(f"\nAnalyzing image: {demo_img}")
        print(f"Available models: {list(predictor.models.keys())}")
        print(f"Available scanners: {predictor.labels}")
        
        # Test each model
        for model_name in predictor.models.keys():
            print(f"\n--- {model_name} Results ---")
            result = predictor.predict_scanner(demo_img, model_name)
            
            if "error" not in result:
                print(f"Predicted Scanner: {result['predicted_scanner']}")
                if result['confidence']:
                    print(f"Confidence: {result['confidence']:.2%}")
                
                if result['all_probabilities']:
                    print("All Scanner Probabilities:")
                    for scanner, prob in sorted(result['all_probabilities'].items(), 
                                             key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  {scanner}: {prob:.2%}")
            else:
                print(f"Error: {result['error']}")
        
        # Clean up
        os.remove(demo_img)
        
    except Exception as e:
        print(f"❌ Error in scanner prediction demo: {e}")

def demo_forgery_detection():
    """Demonstrate forgery detection functionality"""
    print("\n" + "="*50)
    print("🛡️ FORGERY DETECTION DEMO")
    print("="*50)
    
    try:
        from app import ForgeryDetector
        
        # Initialize detector
        detector = ForgeryDetector()
        
        if detector.model is None:
            print("❌ No forgery detection model available")
            return
        
        # Create demo images
        original_img = create_demo_image("original_demo.png")
        forged_img = create_demo_image("forged_demo.png")
        
        print(f"Device: {detector.device}")
        
        # Test original image
        print(f"\n--- Analyzing Original Image ---")
        result = detector.predict_forgery(original_img)
        
        if "error" not in result:
            status = "FORGED" if result['is_forged'] else "ORIGINAL"
            print(f"Result: {status}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Original Probability: {result['original_prob']:.2%}")
            print(f"Forged Probability: {result['forged_prob']:.2%}")
        else:
            print(f"Error: {result['error']}")
        
        # Test forged image
        print(f"\n--- Analyzing Forged Image ---")
        result = detector.predict_forgery(forged_img)
        
        if "error" not in result:
            status = "FORGED" if result['is_forged'] else "ORIGINAL"
            print(f"Result: {status}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Original Probability: {result['original_prob']:.2%}")
            print(f"Forged Probability: {result['forged_prob']:.2%}")
        else:
            print(f"Error: {result['error']}")
        
        # Clean up
        os.remove(original_img)
        os.remove(forged_img)
        
    except Exception as e:
        print(f"❌ Error in forgery detection demo: {e}")

def demo_image_processing():
    """Demonstrate image processing capabilities"""
    print("\n" + "="*50)
    print("🖼️ IMAGE PROCESSING DEMO")
    print("="*50)
    
    try:
        from app import ImageProcessor
        
        # Create demo image
        demo_img = create_demo_image("processing_demo.png")
        
        print(f"Processing image: {demo_img}")
        
        # Load image
        img = ImageProcessor.load_image_gray(demo_img)
        if img is not None:
            print(f"✅ Image loaded successfully")
            print(f"   Shape: {img.shape}")
            print(f"   Data type: {img.dtype}")
            print(f"   Value range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Normalize image
        img_norm = ImageProcessor.normalize_image(img)
        if img_norm is not None:
            print(f"✅ Image normalized successfully")
            print(f"   Mean: {img_norm.mean():.6f}")
            print(f"   Std: {img_norm.std():.6f}")
        
        # Extract wavelet residual
        residual = ImageProcessor.residual_wavelet(img_norm)
        if residual is not None:
            print(f"✅ Wavelet residual extracted successfully")
            print(f"   Shape: {residual.shape}")
            print(f"   Mean: {residual.mean():.6f}")
            print(f"   Std: {residual.std():.6f}")
        
        # Extract FFT features
        fft_feats = ImageProcessor.fft_radial_stats(residual)
        if fft_feats is not None:
            print(f"✅ FFT features extracted successfully")
            print(f"   Number of features: {len(fft_feats)}")
            print(f"   Feature range: [{fft_feats.min():.3f}, {fft_feats.max():.3f}]")
        
        # Clean up
        os.remove(demo_img)
        
    except Exception as e:
        print(f"❌ Error in image processing demo: {e}")

def main():
    """Run all demos"""
    print("AI Trace Finder - Demo Script")
    print("=" * 40)
    print("This script demonstrates the core functionality")
    print("of the AI Trace Finder application.")
    
    # Check if models exist
    model_files = [
        "artifacts/metadata.joblib",
        "artifacts/svm.joblib",
        "models/resnet18_full.pth"
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    if missing_files:
        print(f"\n⚠️ Missing model files: {missing_files}")
        print("Some demos may not work correctly.")
    
    # Run demos
    demo_image_processing()
    demo_scanner_prediction()
    demo_forgery_detection()
    
    print("\n" + "="*50)
    print("🎉 Demo completed!")
    print("To run the full Streamlit application, use:")
    print("  python run_app.py")
    print("  or")
    print("  streamlit run app.py")
    print("="*50)

if __name__ == "__main__":
    main()


