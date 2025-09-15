# Fixed Model Creator - Matches Flask App Features
# This creates a model that expects exactly 41 features like your Flask app

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

print("🔧 Creating Fixed ML Model Files...")
print("="*50)

# Generate sample training data with ALL 41 features that Flask expects
np.random.seed(42)
n_samples = 5000

# All 41 features that your Flask app uses
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# Create synthetic data
data = []
attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
protocols = ['tcp', 'udp', 'icmp']
services = ['http', 'ftp', 'smtp', 'ssh', 'telnet', 'domain', 'private']
flags = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO']

for i in range(n_samples):
    attack_type = np.random.choice(attack_types, p=[0.6, 0.15, 0.15, 0.05, 0.05])
    
    # Generate all 41 features
    if attack_type == 'normal':
        duration = np.random.exponential(20)
        src_bytes = np.random.exponential(1000)
        dst_bytes = np.random.exponential(1000)
        failed_logins = 0 if np.random.random() > 0.1 else np.random.poisson(1)
        compromised = 0
        root_shell = 0
    elif attack_type == 'dos':
        duration = np.random.exponential(5)
        src_bytes = np.random.exponential(100)
        dst_bytes = np.random.exponential(50)
        failed_logins = 0
        compromised = 0
        root_shell = 0
    elif attack_type == 'probe':
        duration = np.random.exponential(2)
        src_bytes = np.random.exponential(50)
        dst_bytes = np.random.exponential(30)
        failed_logins = 0
        compromised = 0
        root_shell = 0
    else:  # r2l or u2r
        duration = np.random.exponential(15)
        src_bytes = np.random.exponential(500)
        dst_bytes = np.random.exponential(300)
        failed_logins = np.random.poisson(2) if attack_type == 'r2l' else 0
        compromised = np.random.poisson(1) if np.random.random() > 0.7 else 0
        root_shell = 1 if attack_type == 'u2r' and np.random.random() > 0.8 else 0
    
    # Create full feature vector with all 41 features
    row = [
        duration,                                    # duration
        np.random.choice(protocols),                 # protocol_type
        np.random.choice(services),                  # service
        np.random.choice(flags),                     # flag
        src_bytes,                                   # src_bytes
        dst_bytes,                                   # dst_bytes
        0,                                          # land
        0,                                          # wrong_fragment
        0,                                          # urgent
        np.random.poisson(0.5),                     # hot
        failed_logins,                              # num_failed_logins
        1,                                          # logged_in
        compromised,                                # num_compromised
        root_shell,                                 # root_shell
        0,                                          # su_attempted
        np.random.poisson(0.1),                     # num_root
        np.random.poisson(0.1),                     # num_file_creations
        0,                                          # num_shells
        0,                                          # num_access_files
        0,                                          # num_outbound_cmds
        0,                                          # is_host_login
        0,                                          # is_guest_login
        np.random.poisson(10),                      # count
        np.random.poisson(5),                       # srv_count
        np.random.random(),                         # serror_rate
        np.random.random(),                         # srv_serror_rate
        np.random.random(),                         # rerror_rate
        np.random.random(),                         # srv_rerror_rate
        np.random.random(),                         # same_srv_rate
        np.random.random(),                         # diff_srv_rate
        np.random.random(),                         # srv_diff_host_rate
        np.random.poisson(50),                      # dst_host_count
        np.random.poisson(20),                      # dst_host_srv_count
        np.random.random(),                         # dst_host_same_srv_rate
        np.random.random(),                         # dst_host_diff_srv_rate
        np.random.random(),                         # dst_host_same_src_port_rate
        np.random.random(),                         # dst_host_srv_diff_host_rate
        np.random.random(),                         # dst_host_serror_rate
        np.random.random(),                         # dst_host_srv_serror_rate
        np.random.random(),                         # dst_host_rerror_rate
        np.random.random()                          # dst_host_srv_rerror_rate
    ]
    data.append(row + [attack_type])  # Add attack_type at the end

# Create DataFrame
df = pd.DataFrame(data, columns=column_names + ['attack_type'])

print(f"✅ Generated {len(df)} samples with {len(column_names)} features")

# Encode categorical features
label_encoders = {}
categorical_columns = ['protocol_type', 'service', 'flag']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Map attack types
attack_mapping = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
df['attack_category'] = df['attack_type'].map(attack_mapping)

print("✅ Data preprocessing completed")

# Prepare features and target - USE ALL 41 FEATURES
X = df[column_names]  # All 41 features
y = df['attack_category']

print(f"✅ Using {X.shape[1]} features (matches Flask app)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model with ALL features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create scaler
scaler = StandardScaler()
scaler.fit(X_train)

print("✅ Model training completed")

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model accuracy: {accuracy:.4f}")

# Save all required files
joblib.dump(model, 'model.sav')
joblib.dump(scaler, 'scaler.sav') 
joblib.dump(label_encoders, 'label_encoders.sav')
joblib.dump(column_names, 'feature_names.sav')  # Save all 41 feature names

print("\n" + "="*50)
print("🎉 Fixed Model files created successfully!")
print("📁 Files created:")
print("   - model.sav (expects 41 features)")
print("   - scaler.sav")
print("   - label_encoders.sav")
print("   - feature_names.sav")
print(f"\n✅ Model now expects {len(column_names)} features (matches Flask app)")
print("🚀 Your Flask app should now work without errors!")
print("="*50)