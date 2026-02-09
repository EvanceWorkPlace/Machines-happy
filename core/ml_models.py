"""
Advanced Machine Learning models for Aviator prediction.
Includes LSTM, Random Forest, and Ensemble methods.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from typing import Dict, Tuple, Any, List
from django.conf import settings

# TensorFlow imports (if available)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class BasePredictor:
    """Base class for all predictors"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
    
    def classify_volatility(self, predicted_value: float, history: List[float]) -> str:
        """Classify volatility level"""
        recent_std = np.std(history[-10:]) if len(history) >= 10 else 0
        
        if recent_std < 0.5:
            return "LOW"
        elif recent_std < 1.5:
            return "MEDIUM"
        else:
            return "HIGH"


class LSTMPredictor(BasePredictor):
    """LSTM-based prediction model for sequence learning"""
    
    def __init__(self, window_size: int = 20):
        super().__init__(window_size)
        self.model = None
        self.model_path = os.path.join(settings.BASE_DIR, 'models', 'lstm_model.h5')
    

    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Install with: pip install tensorflow")
            
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, values: List[float], epochs: int = 100, validation_split: float = 0.2):
        """Train the LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow required for LSTM model")
            
        # Prepare data
        data = np.array(values, dtype=np.float32)
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X, y = self.create_sequences(data_scaled)
        
        if len(X) < 10:
            print(f"Warning: Only {len(X)} sequences. Need more data for training.")
            return False
        
        # Build and train
        self.model = self.build_model((X.shape[1], 1))
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_trained = True
        self._save_model()
        return True
    
    def predict(self, last_values: List[float]) -> Dict[str, Any]:
        """Make prediction on next value"""
        if not self.is_trained or self.model is None:
            return {
                'predicted_value': None,
                'volatility': 'UNKNOWN',
                'confidence': 0.0,
                'error': 'Model not trained'
            }
        
        try:
            # Scale input
            input_scaled = self.scaler.transform(np.array(last_values).reshape(-1, 1)).flatten()
            input_seq = input_scaled[-self.window_size:].reshape(1, self.window_size, 1)
            
            # Predict
            pred_scaled = self.model.predict(input_seq, verbose=0)[0][0]
            predicted_value = float(self.scaler.inverse_transform([[pred_scaled]])[0][0])
            
            # Ensure positive value
            predicted_value = max(1.0, predicted_value)
            
            # Calculate confidence
            recent_mse = self._calculate_recent_error(last_values)
            confidence = 1.0 / (1.0 + recent_mse) if recent_mse >= 0 else 0.5
            confidence = np.clip(confidence, 0.0, 1.0)
            
            volatility = self.classify_volatility(predicted_value, last_values)
            
            return {
                'predicted_value': round(predicted_value, 2),
                'volatility': volatility,
                'confidence': round(confidence, 3),
                'error': None
            }
        except Exception as e:
            return {
                'predicted_value': None,
                'volatility': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_recent_error(self, values: List[float]) -> float:
        """Calculate MSE on recent values for confidence scoring"""
        if len(values) < 2:
            return 1.0
        recent = np.array(values[-20:])
        mean = np.mean(recent)
        mse = np.mean((recent - mean) ** 2)
        return mse
    
    def _save_model(self):
        """Save model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if self.model:
            self.model.save(self.model_path)
    
    def load_model(self):
        """Load saved model"""
        if os.path.exists(self.model_path) and TENSORFLOW_AVAILABLE:
            self.model = load_model(self.model_path)
            self.is_trained = True
            return True
        return False

class RandomForestPredictor(BasePredictor):
    """Random Forest-based predictor for pattern recognition"""
    
    def __init__(self, window_size: int = 20):
        super().__init__(window_size)
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model_path = os.path.join(settings.BASE_DIR, 'models', 'rf_model.pkl')
        self.feature_names = None
        
    def create_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create rich feature set from time series"""
        X, y = [], []
        
        for i in range(len(data) - self.window_size):
            window = data[i:i + self.window_size]
            
            features = [
                window.mean(),
                window.std(),
                np.min(window),
                np.max(window),
                window[-1],
                window[-1] - window[0],
                np.percentile(window, 25),
                np.percentile(window, 75),
                np.ptp(window),
                np.median(window),
                np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else 0,
                (np.std(window[-5:]) / np.std(window[:5])) if np.std(window[:5]) > 0 else 1,
            ]
            
            X.append(features)
            y.append(data[i + self.window_size])
        
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'last_value', 'trend',
            'q1', 'q3', 'range', 'median', 'autocorr', 'vol_ratio'
        ]
        
        return np.array(X), np.array(y)
    
    def train(self, values: List[float], test_size: float = 0.2):
        """Train the Random Forest model"""
        data = np.array(values, dtype=np.float32)
        X, y = self.create_features(data)
        
        if len(X) < 10:
            print(f"Warning: Only {len(X)} samples. Need more data for training.")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate R² score
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"RF - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        self.is_trained = True
        self._save_model()
        return True
    
    def predict(self, last_values: List[float]) -> Dict[str, Any]:
        """Make prediction"""
        if not self.is_trained:
            return {
                'predicted_value': None,
                'volatility': 'UNKNOWN',
                'confidence': 0.0,
                'error': 'Model not trained'
            }
        
        try:
            data = np.array(last_values[-self.window_size:], dtype=np.float32)
            features = self._extract_features(data)
            
            predicted_value = float(self.model.predict([features])[0])
            predicted_value = max(1.0, predicted_value)
            
            # Confidence from feature importance
            confidence = 0.5 + (0.5 * self.model.score([[f for f in features]], [predicted_value]))
            confidence = np.clip(confidence, 0.0, 1.0)
            
            volatility = self.classify_volatility(predicted_value, last_values)
            
            return {
                'predicted_value': round(predicted_value, 2),
                'volatility': volatility,
                'confidence': round(confidence, 3),
                'error': None
            }
        except Exception as e:
            return {
                'predicted_value': None,
                'volatility': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _extract_features(self, window: np.ndarray) -> List[float]:
        """Extract features from a single window"""
        return [
            window.mean(),
            window.std(),
            np.min(window),
            np.max(window),
            window[-1],
            window[-1] - window[0],
            np.percentile(window, 25),
            np.percentile(window, 75),
            np.ptp(window),
            np.median(window),
            np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else 0,
            (np.std(window[-5:]) / np.std(window[:5])) if np.std(window[:5]) > 0 else 1,
        ]
    
    def _save_model(self):
        """Save model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
    
    def load_model(self):
        """Load saved model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            return True
        return False


class EnsemblePredictor(BasePredictor):
    """Ensemble model combining multiple predictors"""
    
    def __init__(self, window_size: int = 20):
        super().__init__(window_size)
        self.lstm_pred = LSTMPredictor(window_size) if TENSORFLOW_AVAILABLE else None
        self.rf_pred = RandomForestPredictor(window_size)
        
    def train(self, values: List[float], **kwargs):
        """Train all ensemble members"""
        results = {}
        
        if self.lstm_pred and TENSORFLOW_AVAILABLE:
            lstm_result = self.lstm_pred.train(values, **kwargs)
            results['lstm'] = lstm_result
        
        rf_result = self.rf_pred.train(values, **kwargs)
        results['rf'] = rf_result
        
        self.is_trained = any(results.values())
        return results
    
    def predict(self, last_values: List[float]) -> Dict[str, Any]:
        """Predict using ensemble voting"""
        if not self.is_trained:
            return {
                'predicted_value': None,
                'volatility': 'UNKNOWN',
                'confidence': 0.0,
                'error': 'Ensemble not trained'
            }
        
        predictions = []
        confidences = []
        
        # LSTM prediction
        if self.lstm_pred and TENSORFLOW_AVAILABLE:
            lstm_pred = self.lstm_pred.predict(last_values)
            if lstm_pred['error'] is None:
                predictions.append(lstm_pred['predicted_value'])
                confidences.append(lstm_pred['confidence'])
        
        # Random Forest prediction
        rf_pred = self.rf_pred.predict(last_values)
        if rf_pred['error'] is None:
            predictions.append(rf_pred['predicted_value'])
            confidences.append(rf_pred['confidence'])
        
        if not predictions:
            return {
                'predicted_value': None,
                'volatility': 'UNKNOWN',
                'confidence': 0.0,
                'error': 'No valid predictions from ensemble members'
            }
        
        # Weighted average
        avg_confidence = np.mean(confidences)
        weights = np.array(confidences) / np.sum(confidences)
        ensemble_prediction = np.average(predictions, weights=weights)
        ensemble_prediction = max(1.0, ensemble_prediction)
        
        volatility = self.classify_volatility(ensemble_prediction, last_values)
        
        return {
            'predicted_value': round(ensemble_prediction, 2),
            'volatility': volatility,
            'confidence': round(float(avg_confidence), 3),
            'error': None,
            'individual_predictions': {
                'lstm': predictions[0] if len(predictions) > 0 and self.lstm_pred else None,
                'random_forest': predictions[1] if len(predictions) > 1 else predictions[0] if len(predictions) > 0 else None
            }
        }
    
    def load_models(self):
        """Load all saved models"""
        results = {}
        if self.lstm_pred:
            results['lstm'] = self.lstm_pred.load_model()
        results['rf'] = self.rf_pred.load_model()
        return results

# ==========================================
# SIMPLE SERVICE USED BY DASHBOARD
# ==========================================

class AviatorPredictionService:
    """
    Lightweight service for dashboard usage.
    Uses RandomForest only (fast & safe).
    """

    def __init__(self):
        self.predictor = RandomForestPredictor(window_size=20)
        self.predictor.load_model()

    def predict_next(self, values: list):
        if len(values) < 25:
            return {
                "prediction": None,
                "volatility": "UNKNOWN",
                "confidence": 0.0,
                "error": "Not enough data"
            }

        result = self.predictor.predict(values)

        return {
            "prediction": result["predicted_value"],
            "volatility": result["volatility"],
            "confidence": result["confidence"],
        }
