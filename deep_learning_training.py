"""
Deep Learning Model Training and Optimization
Trains and validates CNN, LSTM, and Hybrid models for intrusion detection

Usage: python deep_learning_training.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepLearningTrainer:
    """Train and optimize deep learning models for IDS"""
    
    def __init__(self, input_dim=20, num_classes=5, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = {}
        self.histories = {}
        self.scaler = StandardScaler()
        
    def build_cnn_model(self):
        """Build CNN model for sequence learning"""
        logger.info("Building CNN Model...")
        
        model = models.Sequential([
            layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"✅ CNN Model built with {model.count_params()} parameters")
        return model
    
    def build_lstm_model(self):
        """Build LSTM model for sequential patterns"""
        logger.info("Building LSTM Model...")
        
        model = models.Sequential([
            layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            layers.LSTM(64, return_sequences=True, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"✅ LSTM Model built with {model.count_params()} parameters")
        return model
    
    def build_hybrid_model(self):
        """Build Hybrid CNN-LSTM model"""
        logger.info("Building Hybrid CNN-LSTM Model...")
        
        model = models.Sequential([
            layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            # CNN Block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            # LSTM Block
            layers.LSTM(32, return_sequences=True, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(16, return_sequences=False, activation='relu'),
            layers.Dropout(0.3),
            # Dense Block
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"✅ Hybrid Model built with {model.count_params()} parameters")
        return model
    
    def train_model(self, model_name, model, X_train, X_val, y_train, y_val):
        """Train a single model with early stopping and validation"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRAINING {model_name}")
        logger.info(f"{'=' * 80}")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.histories[model_name] = history
        self.models[model_name] = model
        
        logger.info(f"✅ {model_name} training completed")
        return history
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        logger.info(f"\n{'=' * 80}")
        logger.info("MODEL EVALUATION ON TEST SET")
        logger.info(f"{'=' * 80}")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            # Predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
        
        return results
    
    def ensemble_predictions(self, X_test, y_test):
        """Create ensemble predictions from all models"""
        logger.info(f"\n{'=' * 80}")
        logger.info("ENSEMBLE DEEP LEARNING MODEL")
        logger.info(f"{'=' * 80}")
        
        all_predictions = []
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict(X_test, verbose=0)
            all_predictions.append(y_pred_proba)
        
        # Average predictions from all models
        ensemble_proba = np.mean(all_predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        
        logger.info(f"\nEnsemble Performance:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        return {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_models(self, prefix='dl_'):
        """Save all trained models"""
        logger.info(f"\n{'=' * 80}")
        logger.info("SAVING DEEP LEARNING MODELS")
        logger.info(f"{'=' * 80}")
        
        for model_name, model in self.models.items():
            filename = f"{prefix}{model_name.lower().replace(' ', '_')}.h5"
            model.save(filename)
            logger.info(f"✅ Saved: {filename}")
        
        # Save histories
        joblib.dump(self.histories, f'{prefix}training_histories.pkl')
        joblib.dump(self.scaler, f'{prefix}scaler.sav')
        logger.info(f"✅ Saved training histories and scaler")


def load_and_prepare_data(filepath='processed.csv'):
    """Load and prepare data for deep learning"""
    logger.info("Loading data...")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded {len(df)} samples")
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from features
        target_cols = ['attack', 'attack_category', 'target', 'label']
        X_cols = [col for col in numeric_cols if col not in target_cols]
        
        X = df[X_cols].values
        
        # Get target
        if 'attack_category' in df.columns:
            y = df['attack_category'].values
        elif 'attack' in df.columns:
            y = df['attack'].values
        else:
            y = df.iloc[:, -1].values
        
        # Encode target if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None


def main():
    """Main training pipeline"""
    
    # Load data
    X, y = load_and_prepare_data()
    if X is None:
        logger.error("Failed to load data")
        return
    
    # Split data
    logger.info("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scale data
    trainer = DeepLearningTrainer(input_dim=X.shape[1], num_classes=len(np.unique(y)))
    X_train_scaled = trainer.scaler.fit_transform(X_train)
    X_val_scaled = trainer.scaler.transform(X_val)
    X_test_scaled = trainer.scaler.transform(X_test)
    
    logger.info(f"Train set: {X_train_scaled.shape}")
    logger.info(f"Val set:   {X_val_scaled.shape}")
    logger.info(f"Test set:  {X_test_scaled.shape}")
    
    # Build models
    cnn_model = trainer.build_cnn_model()
    lstm_model = trainer.build_lstm_model()
    hybrid_model = trainer.build_hybrid_model()
    
    # Train models
    trainer.train_model("CNN", cnn_model, X_train_scaled, X_val_scaled, y_train, y_val)
    trainer.train_model("LSTM", lstm_model, X_train_scaled, X_val_scaled, y_train, y_val)
    trainer.train_model("Hybrid", hybrid_model, X_train_scaled, X_val_scaled, y_train, y_val)
    
    # Evaluate models
    results = trainer.evaluate_models(X_test_scaled, y_test)
    
    # Ensemble
    ensemble_results = trainer.ensemble_predictions(X_test_scaled, y_test)
    
    # Save models
    trainer.save_models()
    
    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING COMPLETE ✅")
    logger.info(f"{'=' * 80}")
    logger.info(f"Best individual model: {max(results.items(), key=lambda x: x[1]['f1'])[0]}")
    logger.info(f"Ensemble F1-Score: {ensemble_results['f1']:.4f}")


if __name__ == '__main__':
    main()
