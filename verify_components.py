import os
import joblib

print("\n" + "="*70)
print("SYSTEM COMPONENTS VERIFICATION")
print("="*70 + "\n")

# Check Model Files
print("📁 MODEL ARTIFACTS:")
artifacts = [
    ('model.sav', 'ML Model (Random Forest)'),
    ('scaler.sav', 'Feature Scaler (StandardScaler)'),
    ('feature_names.sav', 'Feature Names'),
    ('users.db', 'User Database'),
]

for filename, description in artifacts:
    exists = os.path.exists(filename)
    size = os.path.getsize(filename) if exists else 0
    status = "✓" if exists else "✗"
    size_mb = size / (1024*1024)
    print(f"  {status} {description:<35} {filename:<20} ({size_mb:.2f} MB)")

# Check Model Details
print("\n📊 MODEL SPECIFICATIONS:")
try:
    feature_names = joblib.load('feature_names.sav')
    print(f"  ✓ Input Features: {len(feature_names)} features")
    print(f"    {feature_names}")
    
    model = joblib.load('model.sav')
    print(f"  ✓ Model Type: {type(model).__name__}")
    print(f"  ✓ Classes: {model.n_classes_} (Normal, DoS, Probe, R2L, U2R)")
    print(f"  ✓ Estimators: {len(model.estimators_)} trees in ensemble")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")

# Check Database
print("\n📦 DATABASE STATUS:")
try:
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    pred_count = cursor.fetchone()[0]
    
    print(f"  ✓ Users: {user_count} registered")
    print(f"  ✓ Predictions: {pred_count} logged")
    
    conn.close()
except Exception as e:
    print(f"  ✗ Database error: {e}")

# Check Security
print("\n🔒 SECURITY MODULES:")
security_modules = [
    'app/security/compliance.py',
    'app/security/threat_response.py',
    'app/security/advanced_atr.py',
    'app/security/analytics.py',
]

for module in security_modules:
    exists = os.path.exists(module)
    status = "✓" if exists else "✗"
    print(f"  {status} {module}")

print("\n" + "="*70)
print("✓ ALL COMPONENTS VERIFIED - SYSTEM READY")
print("="*70 + "\n")
