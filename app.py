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
    conn.commit()
    conn.close()

# Load the trained model
try:
    model = joblib.load('model.sav')
    print("Model loaded successfully!")
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. Please train the model first.")

# Load feature names and preprocessing info
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
    'dst_host_srv_rerror_rate'
]

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
            # Collect input features - only use the exact features the model was trained on
            features = []
            feature_dict = {}
            
            # Get the actual number of features the model expects
            n_features_expected = len(feature_names)
            
            print(f"Model expects {n_features_expected} features")
            print(f"Available features: {feature_names[:10]}...")  # Show first 10
            
            for i, feature in enumerate(feature_names):
                if i < n_features_expected:  # Only collect features the model expects
                    value = request.form.get(feature, 0)
                    try:
                        numeric_value = float(value)
                        features.append(numeric_value)
                        feature_dict[feature] = numeric_value
                    except ValueError:
                        features.append(0.0)
                        feature_dict[feature] = 0.0
            
            # Ensure we have the exact number of features
            while len(features) < n_features_expected:
                features.append(0.0)
            
            # Truncate if we have too many features
            features = features[:n_features_expected]
            
            if model is None:
                flash('Model not available. Please train the model first.', 'error')
                return render_template('predict.html', feature_names=feature_names)
            
            print(f"Input features shape: {len(features)}")
            print(f"Features: {features[:5]}...")  # Show first 5 features
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
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
    
    # Get analytics data
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get prediction counts by type
    c.execute("""SELECT prediction_result, COUNT(*) as count 
                 FROM predictions WHERE user_id = ? 
                 GROUP BY prediction_result""", (session['user_id'],))
    prediction_counts = c.fetchall()
    
    # Get recent activity
    c.execute("""SELECT prediction_result, confidence, prediction_date 
                 FROM predictions WHERE user_id = ? 
                 ORDER BY prediction_date DESC LIMIT 20""", (session['user_id'],))
    recent_activity = c.fetchall()
    
    conn.close()
    
    return render_template('analytics.html', 
                         prediction_counts=prediction_counts,
                         recent_activity=recent_activity)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)