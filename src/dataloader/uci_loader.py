from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import csv
from .base_loader import BaseDataLoader

class UCIDataLoader(BaseDataLoader):
    """Data loader for UCI EMG dataset with domain-specific handling"""
    
    GESTURE_LABELS = {
        0: "unmarked",
        1: "rest",
        2: "fist",
        3: "wrist_flexion",
        4: "wrist_extension",
        5: "radial_deviation",
        6: "ulnar_deviation",
        7: "extended_palm"
    }
    
    def __init__(self, config: dict):
        self.config = config  # Store config directly
        # Handle channel configuration based on test mode
        if config.get('test_mode', False):
            self.channel_no = config['test_settings']['channel_no']
            self.channel_subset = config['test_settings'].get('channel_subset', list(range(1, self.channel_no + 1)))
        else:
            self.channel_no = config['channel_no']
            self.channel_subset = config.get('channel_subset', list(range(1, self.channel_no + 1)))
            
        self.users = config['data']['users']
        self.chunk_size = config['data'].get('chunk_size', 10000)
        self.n_workers = config['data'].get('n_workers', 4)
        
        # Add cache path
        self.processed_data_path = Path(config['paths']['processed_data'])
        self.raw_data_path = Path(config['paths']['raw_data'])
        self.cache_file = self.processed_data_path / 'combined_emg_data.csv'

    def load_raw_data(self):
        """Load raw UCI EMG data with parallel processing or from cache if available"""
        # Check if cached data exists
        if self.cache_file.exists():
            print(f"Loading data from cache: {self.cache_file}")
            combined_data = pd.read_csv(self.cache_file)
            print("Unique class labels after loading:", combined_data['class'].unique())
            return combined_data

        file_paths = []
        
        # Collect all file paths first
        for i in range(1, self.users + 1):
            user_id = f"{i:02d}"
            user_dir = self.raw_data_path / user_id
            files = glob(f"{str(user_dir)}/*.txt")
            file_paths.extend([(f, user_id) for f in files])
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            data_list = list(tqdm(
                executor.map(lambda x: self._process_single_file(*x), file_paths),
                total=len(file_paths),
                desc="Processing files"
            ))

        combined_data = pd.concat(data_list, ignore_index=True, copy=False)
        
        # Save to cache
        print(f"Saving combined data to cache: {self.cache_file}")
        combined_data.to_csv(self.cache_file, index=False)
        
        print("Unique class labels after loading:", combined_data['class'].unique())
        return combined_data
    
    def _process_single_file(self, file_path: str, user_id: str) -> pd.DataFrame:
        """Process a single UCI data file with EMG-specific validation"""
        columns = ['time'] + [f'channel{i}' for i in range(1, 9)] + ['class']
        
        try:
            data = pd.read_csv(file_path, 
                             delimiter='\t',
                             names=columns,
                             skiprows=1)
            
            invalid_labels = set(data['class'].unique()) - set(self.GESTURE_LABELS.keys())
            if invalid_labels:
                print(f"Warning: Invalid gesture labels found in {file_path}: {invalid_labels}")
                data = data[data['class'].isin(self.GESTURE_LABELS.keys())]
            
            # Handle missing or infinite values
            if data['class'].isnull().any() or np.isinf(data['class']).any():
                data['class'].fillna(0, inplace=True)
                data.loc[np.isinf(data['class']), 'class'] = 0
            
            data['class'] = data['class'].astype(int)
            data['user_id'] = user_id
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame(columns=columns + ['user_id'])
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess UCI EMG data with signal-specific handling"""
        if 'time' in data.columns:
            data = data.drop('time', axis=1)
        
        # Get all available EMG columns
        all_emg_columns = [f'channel{i}' for i in range(1, 9)]
        
        # Select only the channels specified in the subset
        selected_columns = [f'channel{i}' for i in self.channel_subset]
        
        # Ensure that we only keep columns that actually exist in the data
        valid_columns = [col for col in selected_columns if col in data.columns]
        
        # Keep only the selected columns plus non-EMG columns
        non_emg_columns = [col for col in data.columns if col not in all_emg_columns]
        data = data[valid_columns + non_emg_columns]
        
        # Convert EMG data to float
        data[valid_columns] = data[valid_columns].astype(np.float32)
        
        for col in valid_columns:
            # Check for physiologically impossible values
            if (data[col].abs() > 2000).any():
                print(f"Warning: Extreme values detected in {col}")
            
            # Check for flat signals
            if (data[col].diff() == 0).sum() / len(data) > 0.1:
                print(f"Warning: Possible sensor issue in {col} - flat signal detected")
        
        data['gesture_name'] = data['class'].map(self.GESTURE_LABELS)
        
        # Handle missing values
        null_cols = data.columns[data.isnull().any()].tolist()
        if null_cols:
            print(f"Warning: Missing values found in columns: {null_cols}")
            data = data.interpolate(method='linear')
        
        return data
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        output_path = self.processed_data_path / filename
        data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}") 