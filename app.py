from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import time
# Import live packet features lazily inside the background thread to avoid
# importing `pyshark` at module import time (which can block or fail).
from app.security.threat_response import threat_engine

# Import compliance modules for HIPAA/GDPR support
try:
    from app.security.compliance import (
        AuditLogger, DataEncryption, DataPrivacy, ComplianceMonitor
    )
    compliance_available = True
except Exception as e:
    print(f"Warning: Compliance module not available: {e}")
    compliance_available = False

# Lazy import/initialization for Enhanced ATR Engine to avoid import-time side effects
_enhanced_atr_engine = None
def get_enhanced_atr_engine():
    global _enhanced_atr_engine
    if _enhanced_atr_engine is None:
        from app.security.advanced_atr import EnhancedThreatResponseEngine
        _enhanced_atr_engine = EnhancedThreatResponseEngine()
    return _enhanced_atr_engine
import sys

# Attack types mapping - using threat engine constants
ATTACK_TYPES = {
    0: 'Normal',
    1: 'DoS Attack',
    2: 'Probe Attack',
    3: 'R2L Attack',
    4: 'U2R Attack'
}
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import time
from app.security.threat_response import threat_engine


app = Flask(__name__)
# Security: Use consistent secret key to prevent session issues
app.secret_key = 'healthcare-cybersecurity-ml-secure-key-2024'
app.permanent_session_lifetime = 3600  # 1 hour

# Initialize compliance modules for HIPAA/GDPR
if compliance_available:
    try:
        encryption = DataEncryption()
        audit_logger = AuditLogger()
        compliance_monitor = ComplianceMonitor()
        print("✅ Compliance modules initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize compliance modules: {e}")
        compliance_available = False
else:
    encryption = None
    audit_logger = None
    compliance_monitor = None

# Database setup
def init_db():
    # Use a reasonable timeout and enable WAL to reduce locking issues under concurrency
    conn = sqlite3.connect('users.db', timeout=30)
    try:
        # Enable WAL journal mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
    except Exception:
        pass
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
    
    # Create demo users for testing
    from werkzeug.security import generate_password_hash
    
    # Check if demo users already exist
    c.execute("SELECT COUNT(*) FROM users WHERE username IN ('admin', 'demo')")
    existing_demo_users = c.fetchone()[0]
    
    if existing_demo_users == 0:
        # Create demo admin user
        admin_password = generate_password_hash('admin123')
        c.execute("INSERT OR IGNORE INTO users (username, email, password) VALUES (?, ?, ?)",
                 ('admin', 'admin@demo.com', admin_password))
        
        # Create demo user
        demo_password = generate_password_hash('demo123')
        c.execute("INSERT OR IGNORE INTO users (username, email, password) VALUES (?, ?, ?)",
                 ('demo', 'demo@demo.com', demo_password))
        
        print("Demo users created: admin/admin123 and demo/demo123")
    
    conn.commit()
    conn.close()

# Load the trained model and preprocessing components
model = None
feature_names = None
scaler = None
label_encoders = None

# Lazy-load ML artifacts to avoid import-time pickle issues
def load_model_artifacts():
    global model, feature_names, scaler, label_encoders
    if model is not None:
        return True
    try:
        # Allow forcing subprocess-only prediction to avoid in-process unpickle
        if os.environ.get('FORCE_SUBPROCESS_PREDICT', '0').lower() in ('1', 'true'):
            print('FORCE_SUBPROCESS_PREDICT active; skipping in-process artifact load')
            return False
        import joblib
        import numpy as _np
        # attempt alias for private numpy modules used in old pickles
        try:
            sys.modules.setdefault('numpy._core', _np.core)
        except Exception:
            pass

        # Try loading enhanced models first
        try:
            model = joblib.load('model_ensemble.sav')
            print("✅ Loaded enhanced ensemble model")
        except:
            model = joblib.load('model.sav')
            print("ℹ️  Loaded original model (enhanced model not available yet)")
        
        try:
            feature_names = joblib.load('feature_names_ensemble.sav')
        except:
            try:
                feature_names = joblib.load('feature_names.sav')
            except Exception:
                feature_names = []
        try:
            scaler = joblib.load('scaler_ensemble.sav')
        except:
            try:
                scaler = joblib.load('scaler.sav')
            except Exception:
                scaler = None
        
        # Load anomaly detectors if available
        try:
            global anomaly_ensemble
            anomaly_ensemble = joblib.load('anomaly_detectors.sav')
            print("✅ Loaded anomaly detection ensemble")
        except:
            anomaly_ensemble = None
        
        try:
            label_encoders = joblib.load('label_encoders.sav')
        except Exception:
            label_encoders = None
        print('Model and artifacts loaded')
        return True
    except Exception as e:
        print(f'Error loading model artifacts: {e}')
        model = None
        return False

# Helper function for session validation
def requires_login():
    """Check if user is properly logged in"""
    return ('user_id' in session and 
            'username' in session and 
            session.get('logged_in', False) == True)

# Defer loading of ML artifacts until runtime to avoid import-time pickle errors
print('Model artifact loading deferred until runtime')

# Attack types mapping - using threat engine constants
ATTACK_TYPES = {
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
        
        # Try inserting with retries to avoid transient 'database is locked' errors
        try:
            conn = sqlite3.connect('users.db', timeout=30)
            c = conn.cursor()
            for attempt in range(5):
                try:
                    c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                             (username, email, hashed_password))
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    # Retry on database lock
                    if 'locked' in str(e).lower() and attempt < 4:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
        except sqlite3.OperationalError as e:
            print(f"Database operational error during register: {e}")
            flash('Database busy. Please try again shortly.', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Input validation
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('login.html')
        
        try:
            conn = sqlite3.connect('users.db', timeout=10)
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            
            if user and check_password_hash(user[3], password):
                # Clear any existing session data
                session.clear()
                # Set new session data
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['logged_in'] = True
                session.permanent = True
                
                # Log successful login
                if compliance_available and audit_logger:
                    try:
                        audit_logger.log_action(
                            user_id=user[0],
                            username=user[1],
                            action='LOGIN',
                            resource='authentication',
                            details={'method': 'password', 'result': 'success'},
                            ip_address=request.remote_addr,
                            status='SUCCESS'
                        )
                    except Exception as e:
                        print(f"Error logging login: {e}")
                
                flash('Login successful!', 'success')
                
                # Check if user was trying to access a specific page
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect(url_for('dashboard'))
            else:
                # Log failed login
                if compliance_available and audit_logger:
                    try:
                        audit_logger.log_action(
                            user_id=None,
                            username=username,
                            action='LOGIN',
                            resource='authentication',
                            details={'method': 'password', 'result': 'failed'},
                            ip_address=request.remote_addr,
                            status='FAILURE'
                        )
                    except Exception as e:
                        print(f"Error logging failed login: {e}")
                
                flash('Invalid username or password!', 'error')
        except sqlite3.Error as e:
            print(f"Database error in login: {e}")
            flash('An error occurred. Please try again.', 'error')
        finally:
            if 'conn' in locals():
                conn.close()
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not requires_login():
        flash('Please login to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    
    # Get user's prediction history with proper connection management
    try:
        conn = sqlite3.connect('users.db', timeout=10)
        c = conn.cursor()
        c.execute("""SELECT prediction_result, confidence, prediction_date 
                     FROM predictions WHERE user_id = ? 
                     ORDER BY prediction_date DESC LIMIT 10""", (session['user_id'],))
        recent_predictions = c.fetchall()
        
        return render_template('dashboard.html', 
                             predictions=recent_predictions,
                             username=session.get('username', 'User'))
    except sqlite3.Error as e:
        print(f"Database error in dashboard: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return render_template('dashboard.html', predictions=[], username=session.get('username', 'User'))
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not requires_login():
        flash('Please login to make predictions.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    if request.method == 'POST':
        try:
            # Log data access for compliance
            if compliance_available and audit_logger:
                try:
                    audit_logger.log_data_access(
                        user_id=session.get('user_id'),
                        data_type='network_traffic',
                        access_type='prediction_request',
                        record_count=1
                    )
                except Exception as e:
                    print(f"Error logging data access: {e}")
            
            # Ensure model artifacts are loaded (lazy)
            if model is None:
                load_model_artifacts()

            # We attempt to load artifacts but don't abort if loading fails —
            # prediction can be delegated to a subprocess to avoid C-extension unpickle issues.
            # Collect and process input features
            features = []
            feature_dict = {}

            # Determine which feature names to use. If feature_names aren't available
            # (e.g., model failed to load), fall back to the incoming form field order.
            if feature_names:
                expected_count = getattr(model, 'n_features_in_', len(feature_names))
                effective_feature_names = feature_names[:expected_count] if len(feature_names) > expected_count else feature_names
            else:
                # fall back to form keys order
                effective_feature_names = list(request.form.keys())
                expected_count = len(effective_feature_names)
            
            # Process each feature
            for feature_name in effective_feature_names:
                value = request.form.get(feature_name, 0)
                
                # Handle categorical features with label encoding
                if label_encoders and feature_name in label_encoders and isinstance(value, str):
                    try:
                        encoded_value = label_encoders[feature_name].transform([value])[0]
                        features.append(float(encoded_value))
                    except (ValueError, KeyError):
                        features.append(0.0)
                else:
                    # Handle numeric features with validation
                    try:
                        numeric_value = max(0.0, min(float(value), 1e10))  # Clamp values
                        features.append(numeric_value)
                    except ValueError:
                        features.append(0.0)
                
                feature_dict[feature_name] = features[-1]
            
            # Pad with zeros if needed
            if len(features) < expected_count:
                pad_len = expected_count - len(features)
                features.extend([0.0] * pad_len)
            
            # Convert to numpy array and validate dimensions
            features_array = np.array(features, dtype=float).reshape(1, -1)
            if features_array.shape[1] != expected_count:
                raise ValueError(f"Feature vector width {features_array.shape[1]} != expected {expected_count}")
            
            # Apply scaling if available
            if scaler is not None:
                features_array = scaler.transform(features_array)
            
            # Check for anomalies
            anomaly_flag = False
            if 'anomaly_ensemble' in globals() and anomaly_ensemble is not None:
                try:
                    anomalies, votes = anomaly_ensemble.detect_anomalies(features_array)
                    anomaly_flag = anomalies[0]
                    if anomaly_flag:
                        print(f"⚠️  Anomaly detected! Votes: {votes}")
                except Exception as e:
                    print(f"Error in anomaly detection: {e}")
            
            # Make prediction and get confidence. Use subprocess predictor if model
            # artifacts cannot be loaded in-process (prevents C-extension crashes).
            prediction = None
            confidence = None
            try:
                if model is None:
                    # call helper subprocess using the same Python that's running Flask
                    import subprocess, json
                    venv_python = sys.executable
                    proc_input = json.dumps({'features': features}).encode()
                    # allow longer timeout and make it configurable via env var
                    timeout_seconds = int(os.environ.get('SUBPROCESS_PREDICT_TIMEOUT', '20'))
                    proc = subprocess.run([venv_python, 'subprocess_predict.py'], input=proc_input, capture_output=True, timeout=timeout_seconds)
                    out = proc.stdout.decode('utf-8', errors='ignore')
                    err = proc.stderr.decode('utf-8', errors='ignore')
                    if proc.returncode != 0:
                        print(f"Subprocess predictor failed (rc={proc.returncode}): {err}")
                        raise RuntimeError(f"Subprocess predictor returned non-zero exit code: {proc.returncode}")
                    try:
                        result = json.loads(out)
                        if 'error' in result:
                            raise RuntimeError(result['error'])
                        prediction = int(result.get('prediction'))
                        confidence = float(result.get('confidence', 95.0))
                    except Exception as e:
                        print(f"Error parsing subprocess output. stdout={out} stderr={err}")
                        raise
                else:
                    prediction = model.predict(features_array)[0]
                    try:
                        probabilities = model.predict_proba(features_array)[0]
                        confidence = max(probabilities) * 100
                    except Exception:
                        confidence = 95.0
            except Exception as e:
                print(f"Prediction error (subprocess/in-process): {e}")
                flash('Error making prediction: model unavailable or processing error.', 'error')
                return render_template('predict.html', feature_names=feature_names or [])
            
            result = ATTACK_TYPES.get(prediction, 'Unknown')
            
            # Log prediction with compliance module
            if compliance_available and audit_logger:
                try:
                    audit_logger.log_action(
                        user_id=session.get('user_id'),
                        username=session.get('username'),
                        action='PREDICTION',
                        resource='threat_detection',
                        details={
                            'prediction': result,
                            'confidence': float(confidence),
                            'anomaly_detected': anomaly_flag
                        },
                        ip_address=request.remote_addr
                    )
                except Exception as e:
                    print(f"Error logging prediction: {e}")
            
            # Save to database with proper error handling
            try:
                conn = sqlite3.connect('users.db', timeout=10)
                c = conn.cursor()
                c.execute("""INSERT INTO predictions 
                             (user_id, prediction_result, confidence, input_features) 
                             VALUES (?, ?, ?, ?)""",
                         (session['user_id'], result, confidence, str(feature_dict)))
                conn.commit()
            except sqlite3.Error as e:
                print(f"Database error saving prediction: {e}")
            finally:
                if 'conn' in locals():
                    conn.close()
            
            # Simple threat response logging
            try:
                atr_record = threat_engine.respond(result, confidence, source_ip='unknown', features=feature_dict)
            except Exception:
                atr_record = {'action': 'logged', 'reason': 'Basic logging'}
            
            # Enhanced ATR for high confidence threats
            if confidence > 80.0:
                try:
                    # Use lazy ATR engine instance
                    atr = get_enhanced_atr_engine()
                    threat_context = atr.analyze_threat(
                        threat_type=result,
                        confidence=confidence,
                        source_ip=request.remote_addr or 'unknown',
                        features=feature_dict
                    )
                    atr.respond_to_threat(threat_context)
                except Exception as atr_error:
                    print(f"Enhanced ATR error: {atr_error}")
            
            return render_template('result.html', 
                                 prediction=result, 
                                 confidence=round(confidence, 2),
                                 features=feature_dict,
                                 anomaly_detected=anomaly_flag,
                                 atr=atr_record)
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            flash(f'Error making prediction: {str(e)}', 'error')
            return render_template('predict.html', feature_names=feature_names or [])
    
    return render_template('predict.html', feature_names=feature_names or [])

@app.route('/about')
def about():
    return render_template('about_enhanced.html')

# Error handlers for better user experience
@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('base.html'), 500

@app.route('/analytics')
def analytics():
    if not requires_login():
        flash('Please login to view analytics.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Optimized analytics with single connection
    conn = None
    try:
        conn = sqlite3.connect('users.db', timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        user_id = session['user_id']
        
        # Single query for prediction counts
        c.execute("""SELECT prediction_result, COUNT(*) as count 
                     FROM predictions 
                     WHERE user_id = ? 
                     GROUP BY prediction_result 
                     ORDER BY count DESC""", (user_id,))
        prediction_counts = [(row[0], row[1]) for row in c.fetchall()]
        
        # Single query for recent activity
        c.execute("""SELECT prediction_result, confidence, prediction_date 
                     FROM predictions 
                     WHERE user_id = ? 
                     ORDER BY prediction_date DESC 
                     LIMIT 15""", (user_id,))
        recent_activity = [(row[0], row[1], row[2]) for row in c.fetchall()]
        
        # Calculate analytics summary
        total_predictions = sum(count for _, count in prediction_counts)
        threat_count = sum(count for pred_type, count in prediction_counts if pred_type != 'Normal')
        normal_count = sum(count for pred_type, count in prediction_counts if pred_type == 'Normal')
        
        analytics_summary = {
            'total_predictions': total_predictions,
            'threat_count': threat_count,
            'normal_count': normal_count,
            'avg_confidence': round(sum(conf for _, conf, _ in recent_activity) / len(recent_activity), 1) if recent_activity else 0,
            'max_confidence': max((conf for _, conf, _ in recent_activity), default=0),
            'min_confidence': min((conf for _, conf, _ in recent_activity), default=0)
        }
        
        return render_template('analytics.html', 
                             prediction_counts=prediction_counts,
                             recent_activity=recent_activity,
                             analytics_summary=analytics_summary,
                             username=session.get('username', 'User'))
                             
    except sqlite3.Error as e:
        print(f"Database error in analytics: {e}")
        flash('Error loading analytics data. Please try again.', 'error')
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"General error in analytics: {e}")
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('dashboard'))
    finally:
        if conn:
            conn.close()

@app.route('/atr-dashboard')
def atr_dashboard():
    """Advanced Threat Response Dashboard"""
    if not requires_login():
        flash('Please login to access Auto Response dashboard.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    try:
        atr = get_enhanced_atr_engine()
        dashboard_data = atr.get_dashboard_data()
        return render_template('atr_dashboard.html', dashboard_data=dashboard_data)
    except Exception as e:
        print(f"ATR Dashboard error: {e}")
        flash('Error loading ATR dashboard.', 'error')
        return render_template('atr_dashboard.html', dashboard_data={})

@app.route('/api/atr/dashboard-data')
def get_atr_dashboard_data():
    """API endpoint for real-time dashboard data"""
    if not requires_login():
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        atr = get_enhanced_atr_engine()
        data = atr.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-threats')
def api_live_threats():
    """Real-time threat detection endpoint - returns latest threats"""
    if not requires_login():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        user_id = session.get('user_id')
        
        # Get threats from last 5 minutes
        c.execute('''SELECT prediction_result, confidence, 
                            prediction_date, id
                     FROM predictions 
                     WHERE user_id = ? 
                     AND prediction_result != 'Normal'
                     AND datetime(prediction_date) >= datetime('now', '-5 minutes')
                     ORDER BY prediction_date DESC 
                     LIMIT 5''', (user_id,))
        live_threats = c.fetchall()
        
        # Get latest prediction (for live stream)
        c.execute('''SELECT prediction_result, confidence, prediction_date
                     FROM predictions 
                     WHERE user_id = ? 
                     ORDER BY prediction_date DESC 
                     LIMIT 1''', (user_id,))
        latest = c.fetchone()
        
        conn.close()
        
        response_data = {
            'live_threats': [{
                'prediction': threat[0],
                'confidence': threat[1],
                'timestamp': threat[2],
                'id': threat[3]
            } for threat in live_threats],
            'latest_prediction': {
                'prediction': latest[0] if latest else 'No data',
                'confidence': latest[1] if latest else 0,
                'timestamp': latest[2] if latest else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            } if latest else None,
            'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Live threats API error: {str(e)}")
        return jsonify({
            'live_threats': [],
            'latest_prediction': None,
            'error': str(e)
        }), 200  # Return 200 to avoid breaking frontend

@app.route('/api/incident/<incident_id>')
def get_incident_details(incident_id):
    """API endpoint for incident details"""
    if not requires_login():
        return jsonify({'error': 'Authentication required'}), 401
        
    try:
        atr = get_enhanced_atr_engine()
        details = atr.get_incident_details(incident_id)
        if details:
            return jsonify(details)
        else:
            return jsonify({'error': 'Incident not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# COMPLIANCE & AUDIT ROUTES
# ============================================================================

@app.route('/compliance/report')
def compliance_report():
    """Generate HIPAA/GDPR compliance report"""
    if not requires_login():
        flash('Please login to view compliance report.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Check if user is admin (user_id 4 is admin, or username is 'admin')
    if session.get('user_id') != 4 and session.get('username') != 'admin':
        flash('Compliance reports are available to administrators only.', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        if not compliance_available or not audit_logger:
            flash('Compliance module not available.', 'error')
            return redirect(url_for('dashboard'))
        
        report = audit_logger.generate_compliance_report()
        return jsonify(report)
    except Exception as e:
        print(f"Compliance report error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/compliance/audit-trail')
def audit_trail():
    """Get audit trail for compliance"""
    if not requires_login():
        flash('Please login to view audit trail.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Check if user is admin
    if session.get('user_id') != 4 and session.get('username') != 'admin':
        flash('Audit trails are available to administrators only.', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        if not compliance_available or not audit_logger:
            flash('Compliance module not available.', 'error')
            return redirect(url_for('dashboard'))
        
        hours = request.args.get('hours', 24, type=int)
        trails = audit_logger.get_audit_trail(hours=hours)
        
        return render_template('audit_trail.html', 
                             audit_trails=trails,
                             hours=hours,
                             username=session.get('username'))
    except Exception as e:
        print(f"Audit trail error: {e}")
        flash(f'Error loading audit trail: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/compliance/audit-trail')
def api_audit_trail():
    """JSON API for audit trail"""
    if not requires_login():
        return jsonify({'error': 'Authentication required'}), 401
    
    if session.get('user_id') != 4 and session.get('username') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        if not compliance_available or not audit_logger:
            return jsonify({'error': 'Compliance module not available'}), 500
        
        hours = request.args.get('hours', 24, type=int)
        trails = audit_logger.get_audit_trail(hours=hours)
        
        return jsonify({
            'count': len(trails) if trails else 0,
            'audit_trails': [dict(trail) if hasattr(trail, 'keys') else trail for trail in (trails or [])]
        })
    except Exception as e:
        print(f"API audit trail error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/compliance/status')
def compliance_status():
    """Check compliance module status"""
    if not requires_login():
        return jsonify({'error': 'Authentication required'}), 401
    
    status = {
        'compliance_enabled': compliance_available,
        'encryption_enabled': encryption is not None,
        'audit_logging_enabled': audit_logger is not None,
        'compliance_monitor_enabled': compliance_monitor is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(status)

if __name__ == '__main__':
    init_db()
    # Start background thread for live packet capture and prediction
    def live_prediction_background():
        try:
            from live_packet_features import LivePacketFeatureExtractor, FEATURE_NAMES
        except Exception as e:
            print(f'Live packet feature extractor not available: {e}')
            return
        extractor = LivePacketFeatureExtractor(interface='Wi-Fi')
        while True:
            try:
                features_list = extractor.capture_and_extract(packet_limit=1)
                for features_dict in features_list:
                    try:
                        processed_features = []
                        for name in FEATURE_NAMES:
                            value = features_dict.get(name, 0)
                            # Encode categorical features if encoder exists
                            if label_encoders and name in label_encoders:
                                try:
                                    value = label_encoders[name].transform([value])[0]
                                except Exception:
                                    value = 0
                            processed_features.append(float(value))
                        features_array = np.array(processed_features).reshape(1, -1)
                        # Apply scaling if available
                        if scaler is not None:
                            features_array = scaler.transform(features_array)
                        # Make prediction and get confidence only if model loaded
                        if model is not None:
                            prediction = model.predict(features_array)[0]
                            try:
                                probabilities = model.predict_proba(features_array)[0]
                                confidence = max(probabilities) * 100
                            except Exception:
                                confidence = 95.0
                            result = ATTACK_TYPES.get(prediction, 'Unknown')
                            # Insert into DB as 'system' user (user_id=1 for demo/admin)
                            try:
                                conn = sqlite3.connect('users.db', timeout=10)
                                c = conn.cursor()
                                c.execute("INSERT INTO predictions (user_id, prediction_result, confidence, input_features) VALUES (?, ?, ?, ?)",
                                         (1, result, confidence, str(features_dict)))
                                conn.commit()
                            except Exception as e:
                                print(f"Live prediction DB error: {e}")
                            finally:
                                if 'conn' in locals():
                                    conn.close()
                    except RuntimeError as e:
                        # pyshark/asyncio may raise when running in a background thread
                        print(f"Live prediction processing error (runtime): {e}")
                        continue
                    except Exception as e:
                        print(f"Live prediction processing error: {e}")
                        continue
            except Exception as e:
                print(f"Live prediction error: {e}")
            time.sleep(5)  # Adjust interval as needed

    # Do not load artifacts at startup to avoid C-extension/unpickle crashes.
    # Artifacts will be loaded lazily on-demand by `load_model_artifacts()`.

    live_thread = threading.Thread(target=live_prediction_background, daemon=True)
    # Only start live capture if explicitly enabled to avoid pyshark/event-loop issues
    if os.environ.get('ENABLE_LIVE_CAPTURE', 'false').lower() == 'true':
        # ensure model artifacts are loaded before starting live capture
        if model is None:
            print('Live capture not started because model artifacts failed to load')
        else:
            live_thread.start()
    else:
        print('Live capture disabled. Set ENABLE_LIVE_CAPTURE=true to enable.')
    # Allow controlling debug/reloader via environment to avoid auto-restart
    debug_mode = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true')
    app.run(debug=debug_mode, use_reloader=debug_mode)