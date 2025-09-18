from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import joblib
import sqlite3
import hashlib
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  prediction_result TEXT,
                  confidence REAL,
                  input_features TEXT,
                  prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Add indexes for better performance on analytics queries
    c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_user_id 
                 ON predictions(user_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_date 
                 ON predictions(prediction_date DESC)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_result 
                 ON predictions(prediction_result)''')
    
    conn.commit()
    conn.close()

# Load the trained model and preprocessing components
try:
    model = joblib.load('model.sav')
    print("Model loaded successfully!")
    
    # Load the correct feature names that match the trained model
    try:
        feature_names = joblib.load('feature_names.sav')
        print(f"Feature names loaded successfully! Model expects {len(feature_names)} features")
        print(f"Features: {feature_names[:5]}...")
    except FileNotFoundError:
        print("Warning: feature_names.sav not found. Using fallback feature names.")
        # Fallback feature names if file doesn't exist
        feature_names = [
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
            'dst_host_srv_rerror_rate',
            'device_type', 'protocol', 'user_role', 'department'
        ]
        
except FileNotFoundError:
    model = None
    feature_names = []
    print("Warning: Model file not found. Please train the model first.")

attack_types = {
    0: 'Normal',
    1: 'DoS Attack',
    2: 'Probe Attack',
    3: 'R2L Attack',
    4: 'U2R Attack'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                     (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    
    # Get user's prediction history
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""SELECT prediction_result, confidence, prediction_date 
                 FROM predictions WHERE user_id = ? 
                 ORDER BY prediction_date DESC LIMIT 10""", (session['user_id'],))
    recent_predictions = c.fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=recent_predictions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please login to make predictions.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Ensure we have the correct feature names and model
            if model is None:
                flash('Model not available. Please train the model first.', 'error')
                return render_template('predict.html', feature_names=feature_names)
            
            if not feature_names:
                flash('Feature names not available. Please check model files.', 'error')
                return render_template('predict.html', feature_names=[])
            
            # Collect input features using the correct feature names
            features = []
            feature_dict = {}
            
            print(f"Model expects {len(feature_names)} features")
            print(f"Feature names: {feature_names}")
            
            # Load preprocessing components if they exist
            try:
                scaler = joblib.load('scaler.sav')
                print("Scaler loaded successfully")
            except FileNotFoundError:
                scaler = None
                print("Warning: Scaler not found, using raw values")
            
            try:
                label_encoders = joblib.load('label_encoders.sav')
                print("Label encoders loaded successfully")
            except FileNotFoundError:
                label_encoders = {}
                print("Warning: Label encoders not found")
            
            # Process each feature according to the trained model
            for feature_name in feature_names:
                value = request.form.get(feature_name, 0)
                
                # Handle categorical features with label encoding if available
                if feature_name in label_encoders and isinstance(value, str):
                    try:
                        encoded_value = label_encoders[feature_name].transform([value])[0]
                        features.append(float(encoded_value))
                    except (ValueError, KeyError):
                        # If encoding fails, use 0 as default
                        features.append(0.0)
                else:
                    # Handle numeric features
                    try:
                        numeric_value = float(value)
                        features.append(numeric_value)
                    except ValueError:
                        features.append(0.0)
                
                feature_dict[feature_name] = features[-1]
            
            print(f"Processed {len(features)} features")
            print(f"First 5 features: {features[:5]}")
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Apply scaling if scaler is available
            if scaler is not None:
                features_array = scaler.transform(features_array)
                print("Features scaled successfully")
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            
            # Get confidence score
            try:
                probabilities = model.predict_proba(features_array)[0]
                confidence = max(probabilities) * 100
            except AttributeError:
                # If model doesn't support predict_proba, use default confidence
                confidence = 95.0
            
            result = attack_types.get(prediction, 'Unknown')
            
            print(f"Prediction: {result}, Confidence: {confidence:.2f}%")
            
            # Save prediction to database
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("""INSERT INTO predictions 
                         (user_id, prediction_result, confidence, input_features) 
                         VALUES (?, ?, ?, ?)""",
                     (session['user_id'], result, confidence, str(feature_dict)))
            conn.commit()
            conn.close()
            
            return render_template('result.html', 
                                 prediction=result, 
                                 confidence=round(confidence, 2),
                                 features=feature_dict)
        
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'Error making prediction: {str(e)}', 'error')
            return render_template('predict.html', feature_names=feature_names)
    
    return render_template('predict.html', feature_names=feature_names)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        flash('Please login to view analytics.', 'warning')
        return redirect(url_for('login'))
    
    # Optimized analytics data retrieval with connection management
    try:
        conn = sqlite3.connect('users.db', timeout=10)  # Add timeout
        conn.row_factory = sqlite3.Row  # Enable column access by name
        c = conn.cursor()
        
        user_id = session['user_id']
        
        # Optimized single query to get both prediction counts and recent activity
        # Get prediction counts by type with better performance
        c.execute("""SELECT prediction_result, COUNT(*) as count 
                     FROM predictions 
                     WHERE user_id = ? 
                     GROUP BY prediction_result 
                     ORDER BY count DESC""", (user_id,))
        prediction_counts = c.fetchall()
        
        # Get recent activity with limit and better indexing
        c.execute("""SELECT prediction_result, confidence, prediction_date 
                     FROM predictions 
                     WHERE user_id = ? 
                     ORDER BY prediction_date DESC 
                     LIMIT 15""", (user_id,))  # Reduced from 20 to 15 for better performance
        recent_activity = c.fetchall()
        
        # Convert to simple tuples for JSON serialization
        prediction_counts = [(row[0], row[1]) for row in prediction_counts]
        recent_activity = [(row[0], row[1], row[2]) for row in recent_activity]
        
        # Pre-calculate analytics data on server side to reduce client-side processing
        analytics_summary = {
            'total_predictions': sum(count for _, count in prediction_counts),
            'threat_count': sum(count for pred_type, count in prediction_counts if pred_type != 'Normal'),
            'normal_count': sum(count for pred_type, count in prediction_counts if pred_type == 'Normal'),
            'avg_confidence': round(sum(conf for _, conf, _ in recent_activity) / len(recent_activity), 1) if recent_activity else 0,
            'max_confidence': max((conf for _, conf, _ in recent_activity), default=0),
            'min_confidence': min((conf for _, conf, _ in recent_activity), default=0)
        }
        
        conn.close()
        
        return render_template('analytics.html', 
                             prediction_counts=prediction_counts,
                             recent_activity=recent_activity,
                             analytics_summary=analytics_summary)
                             
    except sqlite3.Error as e:
        print(f"Database error in analytics: {e}")
        if 'conn' in locals():
            conn.close()
        flash('Error loading analytics data. Please try again.', 'error')
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"General error in analytics: {e}")
        if 'conn' in locals():
            conn.close()
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)