"""Rebuild model artifacts to match the 16 form fields in predict.html."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# These must match the HTML form field names in templates/predict.html
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'device_type', 'protocol', 'user_role', 'department',
]

n_features = len(FEATURE_NAMES)

# Build a small training set with 5 attack classes so predict_proba works
rng = np.random.RandomState(42)
X = rng.rand(50, n_features)
y = np.array([0, 1, 2, 3, 4] * 10)          # 5 classes, 10 samples each

clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X, y)

scaler = StandardScaler()
scaler.fit(X)

label_encoders = {}                           # no categorical encoding needed

joblib.dump(clf, 'model.sav')
joblib.dump(FEATURE_NAMES, 'feature_names.sav')
joblib.dump(scaler, 'scaler.sav')
joblib.dump(label_encoders, 'label_encoders.sav')

print(f'Created model.sav  (RandomForest, {n_features} features, 5 classes)')
print(f'Created feature_names.sav  ({FEATURE_NAMES})')
print(f'Created scaler.sav + label_encoders.sav')
