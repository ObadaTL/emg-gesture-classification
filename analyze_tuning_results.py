import os
import json
import pandas as pd
from pathlib import Path
import argparse

def analyze_tuning_results(tuning_dir, top_n=10):
    """Analyze hyperparameter tuning results and rank them by validation accuracy."""
    
    print(f"Analyzing hyperparameter tuning results in {tuning_dir}...")
    
    # Find all trial directories
    trial_dirs = [d for d in os.listdir(tuning_dir) if d.startswith('trial_')]
    
    results = []
    
    # Parse each trial's results
    for trial_dir in trial_dirs:
        trial_path = os.path.join(tuning_dir, trial_dir, 'trial.json')
        
        if not os.path.exists(trial_path):
            continue
            
        try:
            with open(trial_path, 'r') as f:
                trial_data = json.load(f)
                
            # Get trial ID
            trial_id = trial_data.get('trial_id', 'unknown')
            
            # Get validation accuracy
            metrics = trial_data.get('metrics', {}).get('metrics', {})
            val_accuracy = None
            
            if 'val_accuracy' in metrics:
                observations = metrics['val_accuracy'].get('observations', [])
                if observations:
                    val_accuracy = observations[-1].get('value', [0])[0]
            
            # If trial didn't complete or has no validation accuracy, skip it
            if val_accuracy is None or trial_data.get('status') != 'COMPLETED':
                continue
                
            # Extract hyperparameters
            hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
            
            # Create a result entry
            result = {
                'trial_id': trial_id,
                'val_accuracy': val_accuracy,
                'n_layers': hyperparams.get('n_layers'),
                'learning_rate': hyperparams.get('learning_rate')
            }
            
            # Add layer-specific parameters
            for i in range(hyperparams.get('n_layers', 0)):
                result[f'units_{i}'] = hyperparams.get(f'units_{i}')
                result[f'dropout_{i}'] = hyperparams.get(f'dropout_{i}')
                
            results.append(result)
            
        except Exception as e:
            print(f"Error parsing trial {trial_dir}: {e}")
    
    # Convert to DataFrame for easier analysis
    if not results:
        print("No valid trials found!")
        return
        
    df = pd.DataFrame(results)
    
    # Sort by validation accuracy
    df_sorted = df.sort_values('val_accuracy', ascending=False)
    
    # Display top N results
    top_results = df_sorted.head(top_n)
    
    print(f"\nTop {top_n} Hyperparameter Configurations:")
    print("=" * 80)
    
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"Rank {i} (Trial {row['trial_id']}) - Validation Accuracy: {row['val_accuracy']:.6f}")
        print(f"  Layers: {row['n_layers']}")
        print(f"  Learning Rate: {row['learning_rate']:.8f}")
        
        # Print layer details
        for j in range(int(row['n_layers'])):
            if f'units_{j}' in row and f'dropout_{j}' in row:
                print(f"  Layer {j+1}: Units={row[f'units_{j}']}, Dropout={row[f'dropout_{j}']}")
                
        print("-" * 80)
    
    # Create a compact summary of the best model
    best_model = df_sorted.iloc[0]
    print("\nBest Model Configuration Summary:")
    print("=" * 80)
    print(f"Validation Accuracy: {best_model['val_accuracy']:.6f}")
    print(f"Number of Layers: {best_model['n_layers']}")
    print(f"Learning Rate: {best_model['learning_rate']:.8f}")
    
    # Format architecture for config file
    layers = int(best_model['n_layers'])
    units = [int(best_model[f'units_{i}']) for i in range(layers)]
    dropout_rates = [float(best_model[f'dropout_{i}']) for i in range(layers)]
    
    print("\nConfig File Format:")
    print("```yaml")
    print("model:")
    print("  architecture:")
    print(f"    layers: {layers}")
    print(f"    units: {units}")
    print(f"    dropout_rates: {dropout_rates}")
    print(f"  learning_rate: {best_model['learning_rate']}")
    print("```")
    
    # Save results to CSV
    output_file = 'tuning_results_ranked.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\nComplete ranked results saved to {output_file}")
    
    return df_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--dir', type=str, default='hyperparameter_tuning/emg_classifier',
                      help='Directory containing the tuning results')
    parser.add_argument('--top', type=int, default=10,
                      help='Number of top results to display')
    
    args = parser.parse_args()
    analyze_tuning_results(args.dir, args.top) 