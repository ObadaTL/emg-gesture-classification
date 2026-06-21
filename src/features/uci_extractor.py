import numpy as np
from typing import List, Dict, Any, Optional
from .base_extractor import BaseFeatureExtractor
import pandas as pd
import math
from scipy import stats
import os
import joblib
from pathlib import Path
import hashlib
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

class UCIFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for UCI EMG dataset
    
    Extracts time-domain features and autoregressive coefficients from EMG signals.
    Features include MAV, RMS, variance, log-RMS, kurtosis, skewness, and AR coefficients.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Set dimensions based on test mode
        if config.get('test_mode', False):
            self.segment_dim = config['test_settings']['segment_dim']
            self.increment_dim = config['test_settings']['increment_dim']
            self.feature_dim = config['test_settings']['feature_dim']
            self.channel_no = config['test_settings']['channel_no']
            self.channel_subset = config['test_settings'].get('channel_subset', list(range(1, self.channel_no + 1)))
        else:
            self.segment_dim = config['segment_dim']
            self.increment_dim = config['increment_dim']
            self.feature_dim = config['feature_dim']
            self.channel_no = config['channel_no']
            self.channel_subset = config.get('channel_subset', list(range(1, self.channel_no + 1)))
        
        # Calculate features per channel
        self.features_per_channel = self.feature_dim // self.channel_no
        print(f"\nFeature extraction configuration:")
        print(f"Total features requested: {self.feature_dim}")
        print(f"Number of channels: {self.channel_no}")
        print(f"Channel subset: {self.channel_subset}")
        print(f"Features per channel: {self.features_per_channel}")
        
        # Cache setup with fallback
        if 'interim_data' in config['paths']:
            cache_dir = Path(config['paths']['interim_data'])
        else:
            cache_dir = Path(config['paths']['processed_data']) / 'feature_cache'
        
        self.cache_dir = cache_dir / 'feature_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_ar_coeffs = 6
        # Add parallel processing configuration
        n_jobs = config.get('processing', {}).get('n_jobs', -1)
        self.n_jobs = os.cpu_count() if n_jobs <= 0 else min(n_jobs, os.cpu_count())
        
        # Suppress all warnings including TensorFlow ones
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
    
    def extract_features(self, data: pd.DataFrame, force_recompute: bool = False) -> np.ndarray:
        print("\nInitializing feature extraction...")
        
        if not self.features_per_channel:
            print("Calculating features per channel...")
            total_features = self.config['data']['feature_dim']
            self.features_per_channel = total_features // self.channel_no
            print(f"\nFeature extraction configuration:")
            print(f"Total features requested: {total_features}")
            print(f"Number of channels: {self.channel_no}")
            print(f"Features per channel: {self.features_per_channel}")
        
        # Try loading from cache first
        if not force_recompute:
            print("Checking cache for pre-computed features...")
            cached_features = self._load_from_cache(data)
            if cached_features is not None:
                print("Found cached features! Loading from cache...")
                return cached_features
            print("No cached features found. Computing new features...")
        
        # Extract features if not cached or force_recompute
        print("Starting feature extraction process...")
        return self._extract_and_cache_features(data)
    
    def _extract_and_cache_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features with parallel processing"""
        # Use the channel subset to determine which columns to use
        channel_cols = [f'channel{i}' for i in self.channel_subset]
        self._validate_input_data(data, channel_cols)
        
        # Pre-calculate segment indices
        segment_indices = range(0, len(data) - self.segment_dim, self.increment_dim)
        
        # Prepare data chunks for parallel processing
        chunks = []
        for time_ in segment_indices:
            chunk_data = {
                'time': time_,
                'data': {col: data[col].iloc[time_:time_ + self.segment_dim].values 
                        for col in channel_cols},
                'label': data['class'].iloc[time_:time_ + self.segment_dim].mode().iloc[0]
            }
            chunks.append(chunk_data)
        
        # Process chunks in parallel
        features_list = []
        labels_list = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self._process_chunk, chunk) 
                      for chunk in chunks]
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc="Extracting features"):
                try:
                    features, label = future.result()
                    features_list.append(features)
                    labels_list.append(label)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
        
        if not features_list:
            raise ValueError("No features were successfully extracted")
        
        final_matrix = self._combine_features_and_labels(features_list, labels_list)
        self._save_to_cache(data, final_matrix)
        return final_matrix
    
    def _process_chunk(self, chunk: Dict) -> tuple:
        """Process a single data chunk"""
        # Suppress TensorFlow warnings in worker processes
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        try:
            all_channel_features = []
            for channel, channel_data in chunk['data'].items():
                channel_features = self._extract_channel_features(channel_data)
                all_channel_features.extend(channel_features)
            
            expected_features = self.config['data']['feature_dim']
            if len(all_channel_features) != expected_features:
                if len(all_channel_features) < expected_features:
                    all_channel_features.extend([0.0] * (expected_features - len(all_channel_features)))
                else:
                    all_channel_features = all_channel_features[:expected_features]
            
            return all_channel_features, chunk['label']
        except Exception as e:
            print(f"Error processing chunk at time {chunk['time']}: {e}")
            raise
    
    def _extract_channel_features(self, channel_data: np.ndarray) -> List[float]:
        """Optimized feature extraction for a single channel"""
        try:
            channel_data = np.array(channel_data, dtype=float)
            self._validate_channel_data(channel_data)
            
            time_features = self._extract_time_domain_features(channel_data)
            ar_coeffs = self._extract_ar_coefficients(channel_data)
            
            # Simply combine the features
            total_features = ar_coeffs + time_features
            
            # Handle feature dimension matching
            if len(total_features) > self.features_per_channel:
                return total_features[:self.features_per_channel]
            elif len(total_features) < self.features_per_channel:
                return total_features + [0.0] * (self.features_per_channel - len(total_features))
            return total_features
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return [0.0] * self.features_per_channel
    
    def _extract_time_domain_features(self, data: np.ndarray) -> List[float]:
        """Extract time domain features from channel data"""
        iemg = np.sum(np.abs(data))
        mean = iemg / len(data)
        mav = np.abs(mean)
        var = np.var(data)
        rms = np.sqrt(np.mean(np.square(data)))
        log_rms = np.log(max(rms, 1e-10))
        kurt = stats.kurtosis(data)
        skew = stats.skew(data)
        
        return [iemg, log_rms, kurt, skew, rms, var, mav]
    
    def _extract_ar_coefficients(self, data: np.ndarray) -> List[float]:
        """Extract autoregressive coefficients"""
        try:
            model = AutoReg(data, lags=self.n_ar_coeffs).fit()
            coeffs = model.params[1:self.n_ar_coeffs + 1].tolist()
        except Exception as e:
            print(f"AR coefficient calculation failed: {e}")
            coeffs = [0.0] * self.n_ar_coeffs
        
        return coeffs[:self.n_ar_coeffs]
    
    def _validate_channel_data(self, data: np.ndarray) -> None:
        """Validate channel data"""
        if len(data) == 0:
            raise ValueError("Empty channel data")
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Channel data contains NaN or Inf values")
    
    def _validate_input_data(self, data: pd.DataFrame, channel_cols: List[str]) -> None:
        """Validate input DataFrame"""
        if not all(col in data.columns for col in channel_cols):
            raise ValueError(f"Missing channel columns. Expected: {channel_cols}")
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate unique cache key based on data and parameters"""
        channel_subset_str = '_'.join(map(str, self.channel_subset))
        param_string = f"{self.segment_dim}_{self.increment_dim}_{self.channel_no}_{channel_subset_str}"
        
        # Use only the selected channels for the cache key
        numeric_cols = [f'channel{i}' for i in self.channel_subset]
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols:
            # Fallback if none of the requested channels are available
            raise ValueError(f"None of the requested channels {numeric_cols} are available in the data")
            
        numeric_data = data[available_cols]
        
        data_info = (f"{data.shape}_"
                    f"{numeric_data.iloc[:100].sum().sum()}_"
                    f"{data['class'].iloc[:100].astype(str).str.cat()}")
        
        return hashlib.md5(f"{param_string}_{data_info}".encode()).hexdigest()
    
    def _load_from_cache(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Try to load features from cache"""
        print("Generating cache key...")
        cache_key = self._generate_cache_key(data)
        cache_path = self.cache_dir / f"features_{cache_key}.joblib"
        
        if cache_path.exists():
            print(f"Cache found at: {cache_path}")
            try:
                return joblib.load(cache_path)
            except Exception as e:
                print(f"Error loading cached features: {e}")
        print("No cache found.")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, features: np.ndarray) -> None:
        """Save features to cache"""
        cache_key = self._generate_cache_key(data)
        cache_path = self.cache_dir / f"features_{cache_key}.joblib"
        
        try:
            joblib.dump(features, cache_path)
        except Exception as e:
            print(f"Error caching features: {e}")
    
    def _combine_features_and_labels(self, features: List[List[float]], 
                                   labels: List[int]) -> np.ndarray:
        """Combine features and labels into final matrix"""
        features_array = np.array(features)
        labels_array = np.array(labels)
        return np.column_stack((features_array, labels_array))

    """Port of critical legacy utility functions"""
    def segment_mov(time_: int, segment_dim: int, y_int_train: np.ndarray) -> int:
        """Port of legacy segment_mov function"""
        mov_samplewindow = y_int_train[time_:time_ + segment_dim]
        counts = np.bincount(mov_samplewindow.astype(int))
        return np.argmax(counts)