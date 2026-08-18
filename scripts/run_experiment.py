import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple
import tensorflow as tf

# Allow running this script directly (`python scripts/run_experiment.py`) without
# requiring `pip install -e .` first, by putting src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emg_classifier.pipelines.uci_pipeline import UCIPipeline

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_experiment_configs(base_config: dict) -> List[Tuple[str, dict]]:
    """Generate experimental configurations"""
    all_configs = []
    
    # Calculate features per channel (7 time domain + 6 AR coefficients)
    FEATURES_PER_CHANNEL = 13
    
    if base_config.get('test_mode', False):
        # Test mode - single configuration
        test_settings = base_config['test_settings']
        channel_no = test_settings['channel_no']
        channel_subset = test_settings.get('channel_subset', list(range(1, channel_no + 1)))
        feature_dim = channel_no * FEATURES_PER_CHANNEL
        
        exp_config = base_config.copy()
        exp_config.update({
            'channel_no': channel_no,
            'channel_subset': channel_subset,
            'segment_dim': test_settings['segment_dim'],
            'increment_dim': test_settings['increment_dim'],
            'feature_dim': feature_dim,
            'data': {
                'feature_dim': feature_dim,
                **base_config.get('data', {})
            },
            'model': {
                'type': base_config['model']['test_type'],
                **base_config.get('model', {})
            }
        })
        
        # Format channel subset for experiment name
        ch_subset_str = '_'.join(map(str, channel_subset))
        exp_name = (f"TEST_ch{channel_no}_subset_{ch_subset_str}_"
                   f"seg{test_settings['segment_dim']}_"
                   f"inc{test_settings['increment_dim']}")
        all_configs.append((exp_name, exp_config))
        
    else:
        # Full experiment mode - all combinations
        for channel_config in base_config['data']['channel_configs']:
            # Get channel count and specific channels to use
            channel_no = channel_config['count']
            channel_subset = channel_config['channels']
            
            # Calculate feature_dim for this channel configuration
            feature_dim = channel_no * FEATURES_PER_CHANNEL
            
            for idx, segment_dim in enumerate(base_config['data']['segment_dims']):
                increment_dim = base_config['data']['increment_dims'][idx]
                for model_type in base_config['model']['types']:
                    exp_config = base_config.copy()
                    exp_config.update({
                        'channel_no': channel_no,
                        'channel_subset': channel_subset,
                        'segment_dim': segment_dim,
                        'increment_dim': increment_dim,
                        'feature_dim': feature_dim,
                        'data': {
                            'feature_dim': feature_dim,
                            **base_config.get('data', {})
                        },
                        'model': {
                            'type': model_type,
                            **base_config.get('model', {})
                        }
                    })
                    
                    # Format channel subset for experiment name
                    ch_subset_str = '_'.join(map(str, channel_subset))
                    exp_name = (f"ch{channel_no}_subset_{ch_subset_str}_"
                              f"seg{segment_dim}_"
                              f"inc{increment_dim}_"
                              f"{model_type}")
                    all_configs.append((exp_name, exp_config))
    
    return all_configs

def update_results_csv(results_path: Path, results: Dict[str, Any]):
    """Update results CSV file"""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not results_path.exists():
        headers = list(results.keys())
        with open(results_path, 'w') as f:
            f.write(','.join(headers) + '\n')
    
    # Append results
    with open(results_path, 'a') as f:
        values = [str(results[key]) for key in results.keys()]
        f.write(','.join(values) + '\n')

def clear_memory():
    """Clear memory between experiments"""
    import gc
    gc.collect()
    tf.keras.backend.clear_session()

def validate_config(config: dict) -> bool:
    """Validate experiment configuration"""
    required_keys = ['data', 'model', 'paths']
    if not all(key in config for key in required_keys):
        raise ValueError(f"Missing required config sections: {required_keys}")
    
    if len(config['data']['segment_dims']) != len(config['data']['increment_dims']):
        raise ValueError("Number of segment dimensions must match number of increment dimensions")
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run EMG classification experiments')
    parser.add_argument('--config', type=str, default='config/uci_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode with single configuration')
    parser.add_argument('--tune', action='store_true',
                      help='Run hyperparameter tuning before training')
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Set test mode if specified
    if args.test:
        base_config['test_mode'] = True
        mode = 'test'
    else:
        base_config['test_mode'] = False
        mode = 'full'
    
    # Validate configuration
    validate_config(base_config)
    
    # Generate experiment configurations
    experiment_configs = generate_experiment_configs(base_config)
    total_experiments = len(experiment_configs)
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_config['paths']['results'])
    results_path = results_dir / f'experiment_results_{mode}_{timestamp}.csv'
    log_file = results_dir / f'experiment_log_{mode}_{timestamp}.txt'
    
    print(f"\nExperiment Summary:")
    print(f"Total configurations to test: {total_experiments}")
    print(f"Channel configurations: {base_config['data']['channel_configs']}")
    print(f"Segment dimensions: {base_config['data']['segment_dims']}")
    print(f"Increment dimensions: {base_config['data']['increment_dims']}")
    print(f"Models to test: {base_config['model']['types']}")
    print(f"Log file: {log_file}\n")
    
    # Log configuration summary
    with open(log_file, 'w') as f:
        f.write(f"Experiment started at: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total configurations: {total_experiments}\n\n")
    
    successful_experiments = 0
    failed_experiments = 0
    
    for idx, (exp_name, exp_config) in enumerate(experiment_configs, 1):
        print(f"\n{'='*50}")
        print(f"Running experiment {idx}/{total_experiments}: {exp_name}")
        print(f"{'='*50}")
        
        try:
            # Clear memory before each experiment
            clear_memory()
            
            # Log experiment start
            with open(log_file, 'a') as f:
                f.write(f"\nStarting experiment {idx}: {exp_name}\n")
                f.write(f"Configuration:\n")
                f.write(f"- Channels: {exp_config['channel_no']}\n")
                f.write(f"- Segment dim: {exp_config['segment_dim']}\n")
                f.write(f"- Increment: {exp_config['increment_dim']}\n")
                f.write(f"- Features: {exp_config['feature_dim']}\n")
                f.write(f"- Model: {exp_config['model']['type']}\n")
            
            # Initialize pipeline with this configuration
            pipeline = UCIPipeline(exp_config)
            
            # Run experiment (with or without hyperparameter tuning)
            if args.tune and exp_config['model']['type'].lower() == 'tensorflow':
                metrics = pipeline.run_hyperparameter_tuning(exp_name)
            else:
                metrics = pipeline.run_experiment(exp_name)
            
            # Prepare results dictionary. Keep this key set identical to the failure
            # branch below so every row in the CSV has the same columns.
            results = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'experiment_name': exp_name,
                'channel_no': exp_config['channel_no'],
                'segment_dim': exp_config['segment_dim'],
                'increment_dim': exp_config['increment_dim'],
                'feature_dim': exp_config['feature_dim'],
                'model_type': exp_config['model']['type'],
                'split_strategy': exp_config.get('processing', {}).get('split_strategy', 'subject_independent'),
                'accuracy': metrics.get('accuracy', None),
                'f1_score': metrics.get('f1_score', None),
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None),
                'training_time': metrics.get('training_time', None),
                'inference_time': metrics.get('inference_time', None),
                'status': 'completed',
                'error': None,
            }
            
            # Update results CSV
            update_results_csv(results_path, results)
            
            # Log success
            with open(log_file, 'a') as f:
                f.write(f"Experiment completed successfully!\n")
                f.write(f"Accuracy: {metrics.get('accuracy', 'N/A')}\n")
            
            successful_experiments += 1
            print(f"Experiment {exp_name} completed successfully!")
            
        except Exception as e:
            failed_experiments += 1
            print(f"Error in experiment {exp_name}: {str(e)}")
            
            # Log failure
            with open(log_file, 'a') as f:
                f.write(f"Experiment failed: {str(e)}\n")
            
            # Record failed experiment (same key set as the success branch above)
            results = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'experiment_name': exp_name,
                'channel_no': exp_config.get('channel_no'),
                'segment_dim': exp_config.get('segment_dim'),
                'increment_dim': exp_config.get('increment_dim'),
                'feature_dim': exp_config.get('feature_dim'),
                'model_type': exp_config.get('model', {}).get('type'),
                'split_strategy': exp_config.get('processing', {}).get('split_strategy', 'subject_independent'),
                'accuracy': None,
                'f1_score': None,
                'precision': None,
                'recall': None,
                'training_time': None,
                'inference_time': None,
                'status': 'failed',
                'error': str(e),
            }
            update_results_csv(results_path, results)
    
    # Log final summary
    print(f"\n{'='*50}")
    print(f"Experiment Summary:")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Results saved to: {results_path}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*50}\n")
    
    with open(log_file, 'a') as f:
        f.write(f"\nExperiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Successful: {successful_experiments}\n")
        f.write(f"Failed: {failed_experiments}\n")

if __name__ == '__main__':
    main() 