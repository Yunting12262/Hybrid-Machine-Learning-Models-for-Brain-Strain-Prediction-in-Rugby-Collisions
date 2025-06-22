import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainStrainFeatureExtractor:
    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract time domain features"""
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['kurtosis'] = signal.kurtosis(signal_data)
        features['skewness'] = signal.skew(signal_data)
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Hjorth parameters
        features['hjorth_activity'] = np.var(signal_data)
        features['hjorth_mobility'] = np.sqrt(np.var(np.gradient(signal_data)) / features['hjorth_activity'])
        features['hjorth_complexity'] = np.sqrt(np.var(np.gradient(np.gradient(signal_data))) / 
                                              np.var(np.gradient(signal_data)))
        
        # Entropy features
        features['shannon_entropy'] = -np.sum(np.histogram(signal_data, bins=50)[0] * 
                                            np.log2(np.histogram(signal_data, bins=50)[0] + 1e-10))
        
        return features
    
    def extract_frequency_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        features = {}
        
        # FFT analysis
        n = len(signal_data)
        fft_vals = np.abs(fft(signal_data))
        freqs = fftfreq(n, 1/self.sampling_rate)
        
        # Consider only positive frequencies
        positive_freq_mask = freqs > 0
        freqs = freqs[positive_freq_mask]
        fft_vals = fft_vals[positive_freq_mask]
        
        # Peak frequency
        peak_freq_idx = np.argmax(fft_vals)
        features['peak_frequency'] = freqs[peak_freq_idx]
        features['peak_magnitude'] = fft_vals[peak_freq_idx]
        
        # Frequency band energies
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        for band_name, (low, high) in freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            features[f'{band_name}_energy'] = np.sum(fft_vals[mask]**2)
        
        return features
    
    def extract_wavelet_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract wavelet features"""
        features = {}
        
        # Use db4 wavelet for decomposition
        coeffs = pywt.wavedec(signal_data, 'db4', level=4)
        
        # Calculate energy for each scale
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
            features[f'wavelet_entropy_level_{i}'] = -np.sum((coeff**2/np.sum(coeff**2)) * 
                                                           np.log2(coeff**2/np.sum(coeff**2) + 1e-10))
        
        return features
    
    def extract_all_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract all features"""
        features = {}
        features.update(self.extract_time_domain_features(signal_data))
        features.update(self.extract_frequency_domain_features(signal_data))
        features.update(self.extract_wavelet_features(signal_data))
        return features

class BrainStrainPredictor:
    def __init__(self):
        """Initialize predictor"""
        self.feature_extractor = BrainStrainFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature data"""
        features_list = []
        
        for _, row in data.iterrows():
            signal_data = row['signal_data']  # Assuming data contains 'signal_data' column
            features = self.feature_extractor.extract_all_features(signal_data)
            features_list.append(features)
            
        features_df = pd.DataFrame(features_list)
        return features_df
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        logger.info("Starting model training...")
        
        # Feature standardization
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logger.info(f"Model training completed. Validation MSE: {mse:.4f}, R2: {r2:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        importance = self.model.feature_importances_
        features = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': importance
        })
        return features.sort_values('importance', ascending=False)

if __name__ == "__main__":
    # Example usage
    logger.info("Initializing brain strain prediction model...")
    predictor = BrainStrainPredictor()
    
    # Load actual data here
    # data = pd.read_csv('your_data.csv')
    # X = predictor.prepare_features(data)
    # y = data['target']
    # predictor.train(X, y)
    
    logger.info("Model initialization completed. Please load data to start training.") 