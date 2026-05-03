"""
Advanced ML Models for Healthcare Cybersecurity IDS
Includes multiple algorithms and ensemble methods as per project report

Author: Healthcare Cybersecurity Team
Version: 2.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier,
    AdaBoostClassifier, IsolationForest, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelEnsemble:
    """
    Comprehensive ML model ensemble for intrusion detection
    Combines multiple algorithms for improved accuracy
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_individual_models(self):
        """
        Create all individual ML models as per report
        Supports 8+ algorithms
        """
        logger.info("Creating individual ML models...")
        
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['XGBoost'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=self.random_state
        )
        
        self.models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=self.random_state
        )
        
        self.models['DecisionTree'] = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto'
        )
        
        self.models['NaiveBayes'] = GaussianNB()
        
        self.models['LogisticRegression'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['AdaBoost'] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=100,
            learning_rate=0.8,
            random_state=self.random_state
        )
        
        logger.info(f"✅ Created {len(self.models)} individual models")
        return self.models
    
    def train_models(self, X, y, test_size=0.2):
        """
        Train all models and evaluate performance
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Training all models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate each model
        model_scores = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            model_scores[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return model_scores, (X_train_scaled, X_test_scaled, y_train, y_test)
    
    def create_voting_ensemble(self):
        """
        Create voting classifier combining top models
        Soft voting for probability averaging
        """
        logger.info("Creating Voting Ensemble...")
        
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.models['RandomForest']),
                ('gb', self.models['GradientBoosting']),
                ('xgb', self.models['XGBoost']),
                ('ada', self.models['AdaBoost'])
            ],
            voting='soft'
        )
        
        return voting_clf
    
    def create_stacking_ensemble(self):
        """
        Create stacking classifier as per report
        Base learners: RF, GB, XGBoost, KNN
        Meta learner: Logistic Regression
        """
        logger.info("Creating Stacking Ensemble...")
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', self.models['RandomForest']),
                ('gb', self.models['GradientBoosting']),
                ('xgb', self.models['XGBoost']),
                ('knn', self.models['KNN'])
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=5
        )
        
        return stacking_clf
    
    def create_best_ensemble(self, X_train, X_test, y_train, y_test):
        """
        Create and train the best ensemble configuration
        """
        logger.info("Creating best performing ensemble...")
        
        # Create both ensemble types
        voting_clf = self.create_voting_ensemble()
        stacking_clf = self.create_stacking_ensemble()
        
        # Train voting
        logger.info("Training Voting Ensemble...")
        voting_clf.fit(X_train, y_train)
        voting_pred = voting_clf.predict(X_test)
        voting_score = f1_score(y_test, voting_pred, average='weighted')
        logger.info(f"Voting F1 Score: {voting_score:.4f}")
        
        # Train stacking
        logger.info("Training Stacking Ensemble...")
        stacking_clf.fit(X_train, y_train)
        stacking_pred = stacking_clf.predict(X_test)
        stacking_score = f1_score(y_test, stacking_pred, average='weighted')
        logger.info(f"Stacking F1 Score: {stacking_score:.4f}")
        
        # Return best ensemble
        if stacking_score >= voting_score:
            logger.info("✅ Stacking Ensemble selected as best model")
            self.ensemble_model = stacking_clf
        else:
            logger.info("✅ Voting Ensemble selected as best model")
            self.ensemble_model = voting_clf
        
        return self.ensemble_model
    
    def save_models(self, prefix='models/'):
        """Save all trained models"""
        import os
        os.makedirs(prefix, exist_ok=True)
        
        joblib.dump(self.ensemble_model, f'{prefix}best_ensemble.sav')
        joblib.dump(self.scaler, f'{prefix}scaler_ensemble.sav')
        joblib.dump(self.models, f'{prefix}individual_models.sav')
        logger.info(f"✅ All models saved to {prefix}")
    
    def predict(self, X):
        """Make prediction with ensemble"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained. Call train_models() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained.")
        
        X_scaled = self.scaler.transform(X)
        return self.ensemble_model.predict_proba(X_scaled)


class AnomalyDetectionEnsemble:
    """
    Anomaly detection algorithms for zero-day threat detection
    as mentioned in project report
    """
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.anomaly_detectors = {}
        self.scaler = StandardScaler()
        
    def create_anomaly_detectors(self):
        """Create multiple anomaly detection algorithms"""
        logger.info("Creating anomaly detection algorithms...")
        
        # Isolation Forest
        self.anomaly_detectors['IsolationForest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Local Outlier Factor
        self.anomaly_detectors['LOF'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination
        )
        
        # One-Class SVM
        self.anomaly_detectors['OneClassSVM'] = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=self.contamination
        )
        
        logger.info(f"✅ Created {len(self.anomaly_detectors)} anomaly detectors")
        return self.anomaly_detectors
    
    def train_anomaly_detectors(self, X_normal):
        """Train anomaly detectors on normal traffic"""
        logger.info("Training anomaly detectors on normal traffic...")
        
        X_scaled = self.scaler.fit_transform(X_normal)
        
        for name, detector in self.anomaly_detectors.items():
            logger.info(f"Training {name}...")
            detector.fit(X_scaled)
        
        logger.info("✅ Anomaly detectors trained")
    
    def detect_anomalies(self, X):
        """Detect anomalies using ensemble voting"""
        X_scaled = self.scaler.transform(X)
        
        anomaly_votes = np.zeros(len(X))
        
        for name, detector in self.anomaly_detectors.items():
            predictions = detector.predict(X_scaled)
            # Convert -1 (anomaly) to 1, 1 (normal) to 0
            anomaly_votes += (predictions == -1).astype(int)
        
        # Majority voting: if 2+ detectors flag as anomaly, it's an anomaly
        anomalies = anomaly_votes >= 2
        
        return anomalies, anomaly_votes


class DeepLearningModels:
    """
    Deep Learning models (CNN, LSTM) as per project report
    Requires TensorFlow/Keras
    """
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        
    def create_cnn_model(self):
        """
        Create CNN model for traffic classification
        As mentioned in project abstract
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            logger.info("Creating CNN model...")
            
            # CNN expects 3D input: (batch_size, timesteps, features)
            # Reshape input_shape to (sequence_length, features)
            if isinstance(self.input_shape, tuple) and len(self.input_shape) == 1:
                cnn_input_shape = (self.input_shape[0], 1)  # Treat as 1 timestep with n features
            else:
                cnn_input_shape = self.input_shape if len(self.input_shape) > 1 else (self.input_shape[0], 1)
            
            model = keras.Sequential([
                layers.Reshape((cnn_input_shape[0], 1), input_shape=self.input_shape),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(5, activation='softmax')  # 5 classes: Normal + 4 attack types
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.models['CNN'] = model
            logger.info("✅ CNN model created")
            return model
            
        except ImportError:
            logger.error("TensorFlow not installed. Skipping CNN model.")
            return None
    
    def create_lstm_model(self):
        """
        Create LSTM model for sequential threat detection
        As mentioned in project abstract
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            logger.info("Creating LSTM model...")
            
            # LSTM expects 3D input: (batch_size, timesteps, features)
            if isinstance(self.input_shape, tuple) and len(self.input_shape) == 1:
                lstm_input_shape = (self.input_shape[0], 1)  # Treat as 1 timestep with n features
            else:
                lstm_input_shape = self.input_shape if len(self.input_shape) > 1 else (self.input_shape[0], 1)
            
            model = keras.Sequential([
                layers.Reshape((lstm_input_shape[0], 1), input_shape=self.input_shape),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(5, activation='softmax')  # 5 classes
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.models['LSTM'] = model
            logger.info("✅ LSTM model created")
            return model
            
        except ImportError:
            logger.error("TensorFlow not installed. Skipping LSTM model.")
            return None
    
    def create_hybrid_model(self):
        """
        Create hybrid CNN-LSTM model for best performance
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            logger.info("Creating Hybrid CNN-LSTM model...")
            
            # Hybrid needs 3D input: (batch_size, timesteps, features)
            if isinstance(self.input_shape, tuple) and len(self.input_shape) == 1:
                model_input_shape = (self.input_shape[0], 1)
            else:
                model_input_shape = self.input_shape if len(self.input_shape) > 1 else (self.input_shape[0], 1)
            
            model = keras.Sequential([
                layers.Reshape((model_input_shape[0], 1), input_shape=self.input_shape),
                layers.Conv1D(32, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(32),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(5, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.models['CNN-LSTM'] = model
            logger.info("✅ Hybrid CNN-LSTM model created")
            return model
            
        except ImportError:
            logger.error("TensorFlow not installed. Skipping Hybrid model.")
            return None


# Example usage
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Healthcare Cybersecurity ML Models - Enhanced Version")
    logger.info("=" * 80)
    
    # Create ensemble
    ensemble = MLModelEnsemble()
    ensemble.create_individual_models()
    logger.info("✅ All ML models configured successfully")
    
    # Create anomaly detection
    anomaly = AnomalyDetectionEnsemble(contamination=0.1)
    anomaly.create_anomaly_detectors()
    logger.info("✅ All anomaly detection algorithms configured successfully")
    
    logger.info("=" * 80)
    logger.info("Ready for training with data")
    logger.info("=" * 80)
