import yaml
from pathlib import Path
import shutil
import os

# A class to clear the feature cache data

# Load config
with open('config/uci_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get cache directory
if 'interim_data' in config['paths']:
    cache_dir = Path(config['paths']['interim_data']) / 'feature_cache'
else:
    cache_dir = Path(config['paths']['processed_data']) / 'feature_cache'

if cache_dir.exists():
    print(f"Clearing feature cache: {cache_dir}")
    try:
        # Remove all files in the directory
        for file_path in cache_dir.glob('*'):
            if file_path.is_file():
                os.remove(file_path)
                print(f"Removed {file_path}")
        print("Cache cleared successfully!")
    except Exception as e:
        print(f"Error clearing cache: {e}")
else:
    print(f"Cache directory does not exist: {cache_dir}") 