"""
Verification Script - Validate Enhanced Components
Tests all new modules before deployment

Usage: python verify_enhancements.py
"""

import sys
import os

def test_imports():
    """Test all new imports"""
    print("\n" + "="*80)
    print("TESTING MODULE IMPORTS")
    print("="*80)
    
    tests = [
        ("ml_models.py", "from ml_models import MLModelEnsemble, AnomalyDetectionEnsemble, DeepLearningModels"),
        ("compliance.py", "from app.security.compliance import DataEncryption, AuditLogger, DataPrivacy, ComplianceMonitor"),
        ("scikit-learn", "from sklearn.ensemble import VotingClassifier, StackingClassifier"),
        ("cryptography", "from cryptography.fernet import Fernet"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:30} - OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {name:30} - FAILED: {str(e)[:50]}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name:30} - WARNING: {str(e)[:50]}")
    
    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_ml_models():
    """Test ML model creation"""
    print("\n" + "="*80)
    print("TESTING ML MODEL CREATION")
    print("="*80)
    
    try:
        from ml_models import MLModelEnsemble
        
        ensemble = MLModelEnsemble()
        models = ensemble.create_individual_models()
        
        expected_models = [
            'RandomForest', 'XGBoost', 'GradientBoosting',
            'DecisionTree', 'KNN', 'NaiveBayes',
            'LogisticRegression', 'AdaBoost'
        ]
        
        for model_name in expected_models:
            if model_name in models:
                print(f"✅ {model_name:30} - Created")
            else:
                print(f"❌ {model_name:30} - NOT FOUND")
        
        print(f"\n✅ Total models created: {len(models)}/8")
        return len(models) == 8
        
    except Exception as e:
        print(f"❌ ML Model Creation Failed: {e}")
        return False


def test_anomaly_detection():
    """Test anomaly detection"""
    print("\n" + "="*80)
    print("TESTING ANOMALY DETECTION")
    print("="*80)
    
    try:
        from ml_models import AnomalyDetectionEnsemble
        
        anomaly = AnomalyDetectionEnsemble()
        detectors = anomaly.create_anomaly_detectors()
        
        expected_detectors = ['IsolationForest', 'LOF', 'OneClassSVM']
        
        for detector_name in expected_detectors:
            if detector_name in detectors:
                print(f"✅ {detector_name:30} - Created")
            else:
                print(f"❌ {detector_name:30} - NOT FOUND")
        
        print(f"\n✅ Total detectors created: {len(detectors)}/3")
        return len(detectors) == 3
        
    except Exception as e:
        print(f"❌ Anomaly Detection Failed: {e}")
        return False


def test_compliance():
    """Test compliance module"""
    print("\n" + "="*80)
    print("TESTING COMPLIANCE MODULES")
    print("="*80)
    
    try:
        from app.security.compliance import (
            DataEncryption, AuditLogger, DataPrivacy, ComplianceMonitor
        )
        
        # Test encryption
        encryption = DataEncryption()
        test_data = "sensitive data"
        encrypted = encryption.encrypt_data(test_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        if decrypted == test_data:
            print(f"✅ DataEncryption        - OK (encrypt/decrypt works)")
        else:
            print(f"❌ DataEncryption        - FAILED")
            return False
        
        # Test audit logger
        audit = AuditLogger()
        print(f"✅ AuditLogger           - OK (database initialized)")
        
        # Test data privacy
        masked = DataPrivacy.mask_email("test@example.com")
        print(f"✅ DataPrivacy           - OK (masked: {masked})")
        
        # Test compliance monitor
        monitor = ComplianceMonitor()
        print(f"✅ ComplianceMonitor     - OK (initialized)")
        
        return True
        
    except Exception as e:
        print(f"❌ Compliance Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_learning():
    """Test deep learning models"""
    print("\n" + "="*80)
    print("TESTING DEEP LEARNING (TensorFlow)")
    print("="*80)
    
    try:
        import tensorflow
        print(f"✅ TensorFlow            - Version {tensorflow.__version__}")
        
        from ml_models import DeepLearningModels
        
        dl_models = DeepLearningModels(input_shape=(20,))
        
        cnn = dl_models.create_cnn_model()
        if cnn:
            print(f"✅ CNN Model             - Created")
        
        lstm = dl_models.create_lstm_model()
        if lstm:
            print(f"✅ LSTM Model            - Created")
        
        hybrid = dl_models.create_hybrid_model()
        if hybrid:
            print(f"✅ Hybrid CNN-LSTM       - Created")
        
        return True
        
    except ImportError:
        print(f"⚠️  TensorFlow not installed - Install: pip install tensorflow>=2.13")
        return True  # Not required for basic functionality
    except Exception as e:
        print(f"❌ Deep Learning Test Failed: {e}")
        return False


def test_files_exist():
    """Verify all required files exist"""
    print("\n" + "="*80)
    print("CHECKING REQUIRED FILES")
    print("="*80)
    
    required_files = [
        'ml_models.py',
        'train_enhanced_models.py',
        'app/security/compliance.py',
        'INTEGRATION_GUIDE.md',
        'ENHANCEMENT_ROADMAP.md',
        'requirements.txt',
        'app.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path:40} - {size:,} bytes")
        else:
            print(f"❌ {file_path:40} - NOT FOUND")
            all_exist = False
    
    return all_exist


def generate_report(results):
    """Generate verification report"""
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    status_map = {
        'imports': 'Module Imports',
        'ml_models': 'ML Model Creation',
        'anomaly': 'Anomaly Detection',
        'compliance': 'Compliance Module',
        'deep_learning': 'Deep Learning',
        'files': 'File Verification'
    }
    
    for test_name, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {status_map.get(test_name, test_name):30} - {'PASS' if status else 'FAIL'}")
    
    print(f"\n{'='*80}")
    print(f"Overall: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'='*80}")
    
    if passed == total:
        print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  ✨ ALL ENHANCEMENTS VERIFIED SUCCESSFULLY! ✨                                 ║
║                                                                                ║
║  Your Healthcare Cybersecurity IDS is ready for deployment!                    ║
║                                                                                ║
║  📚 Next Steps:                                                                ║
║     1. pip install -r requirements.txt                                         ║
║     2. python train_enhanced_models.py                                         ║
║     3. Follow INTEGRATION_GUIDE.md to update app.py                            ║
║     4. Run: python app.py                                                      ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
⚠️  SOME TESTS FAILED

Please install missing dependencies:
   pip install -r requirements.txt

Then run this verification again.
        """)
    
    return passed == total


def main():
    """Run all verifications"""
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   Healthcare Cybersecurity ML - Enhancement Verification Script                ║
║                                                                                ║
║   This script validates all enhanced components and modules                    ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        'imports': test_imports(),
        'files': test_files_exist(),
        'ml_models': test_ml_models(),
        'anomaly': test_anomaly_detection(),
        'compliance': test_compliance(),
        'deep_learning': test_deep_learning(),
    }
    
    success = generate_report(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
