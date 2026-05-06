#!/usr/bin/env python
"""Debug script to inspect model predictions and probabilities"""

import joblib
import numpy as np
import os

print("\n" + "="*70)
print("MODEL ARTIFACT DEBUG")
print("="*70)

# Check if model files exist
print("\n[1] Checking model files...")
files = ['model.sav', 'scaler.sav', 'feature_names.sav']
for f in files:
    exists = os.path.exists(f)
    print(f"  {f:20} {'✓' if exists else '✗'}")

# Load model and test prediction
print("\n[2] Loading model and scaler...")
try:
    model = joblib.load('model.sav')
    scaler = joblib.load('scaler.sav')
    feature_names = joblib.load('feature_names.sav')
    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  ✓ Scaler loaded: {type(scaler).__name__}")
    print(f"  ✓ Feature names loaded: {len(feature_names)} features")
    print(f"    Features: {feature_names}")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    exit(1)

# Create test data
print("\n[3] Creating test sample...")
test_data = {
    "duration": 100,
    "protocol_type": 6,
    "service": 15,
    "flag": 0,
    "src_bytes": 1000,
    "dst_bytes": 500,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0
}

# Extract features in correct order
features = []
for feat in feature_names:
    val = test_data.get(feat, 0.0)
    features.append(float(val))

features_array = np.array([features])
print(f"  ✓ Input features: {features_array.shape}")
print(f"    Values: {features}")

# Scale features
print("\n[4] Scaling features...")
features_scaled = scaler.transform(features_array)
print(f"  ✓ Scaled features shape: {features_scaled.shape}")
print(f"    Scaled values: {features_scaled[0][:5]}... (showing first 5)")

# Make prediction
print("\n[5] Making prediction...")
try:
    prediction = model.predict(features_scaled)[0]
    print(f"  ✓ Prediction: {prediction} (type: {type(prediction).__name__})")
    
    # Get probabilities
    probabilities = model.predict_proba(features_scaled)[0]
    print(f"  ✓ Raw probabilities: {probabilities}")
    print(f"    Shape: {probabilities.shape}")
    print(f"    Min: {probabilities.min():.6f}, Max: {probabilities.max():.6f}")
    print(f"    Sum: {probabilities.sum():.6f}")
    
    # Check for issues
    print("\n[6] ISSUE ANALYSIS:")
    max_prob = probabilities.max()
    if max_prob > 1.0:
        print(f"  ✗ WARNING: Max probability {max_prob:.2f} > 1.0 (INVALID)")
        print(f"    This explains the 3000% confidence (max_prob * 100 = {max_prob * 100:.0f}%)")
    elif max_prob < 0.0:
        print(f"  ✗ WARNING: Max probability {max_prob:.2f} < 0.0 (INVALID)")
    else:
        print(f"  ✓ Max probability {max_prob:.2f} is valid (0-1 range)")
        print(f"    Confidence: {max_prob * 100:.2f}%")
    
    # Show all class probabilities
    print("\n[7] All class probabilities:")
    ATTACK_TYPES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
    for i, p in enumerate(probabilities):
        print(f"    {ATTACK_TYPES.get(i, f'Class{i}'):12} {p:.6f} ({p*100:.2f}%)")
        
except Exception as e:
    print(f"  ✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70 + "\n")
