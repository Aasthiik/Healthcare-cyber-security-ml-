#!/usr/bin/env python3
"""
Test script to verify all model files are working correctly
"""

import joblib
import numpy as np

def test_model_files():
    """Test all saved model components"""
    print("🔍 Testing model files...")
    
    try:
        # Test model
        model = joblib.load('model.sav')
        print(f"✅ Model loaded: expects {model.n_features_in_} features")
        
        # Test feature names
        features = joblib.load('feature_names.sav')
        print(f"✅ Feature names loaded: {len(features)} features")
        print(f"   First 5 features: {features[:5]}")
        
        # Test scaler
        scaler = joblib.load('scaler.sav')
        print("✅ Scaler loaded successfully")
        
        # Test label encoders
        encoders = joblib.load('label_encoders.sav')
        print(f"✅ Label encoders loaded: {len(encoders)} encoders")
        if encoders:
            print(f"   Encoder features: {list(encoders.keys())}")
        
        # Test prediction with sample data
        print("\n🔬 Testing prediction functionality...")
        test_data = np.zeros((1, len(features)))
        
        # Scale the data
        test_data_scaled = scaler.transform(test_data)
        
        # Make prediction
        prediction = model.predict(test_data_scaled)
        probabilities = model.predict_proba(test_data_scaled)
        
        print(f"✅ Test prediction successful!")
        print(f"   Prediction: {prediction[0]}")
        print(f"   Confidence: {max(probabilities[0]) * 100:.2f}%")
        
        print("\n🎉 All model files are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model files: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_files()