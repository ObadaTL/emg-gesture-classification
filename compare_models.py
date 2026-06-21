import pandas as pd
import numpy as np
from tabulate import tabulate
from pathlib import Path

# Load the results
results_file = 'results/experiment_results_full_20250411_182045.csv'
results_df = pd.read_csv(results_file)

def preprocess_data(df):
    """Preprocess the data for analysis"""
    # Convert relevant columns to numeric
    numeric_cols = ['accuracy', 'f1_score', 'precision', 'recall']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract model parameters from experiment name
    df['model_name'] = df['experiment_name'].apply(lambda x: x.split('_')[-1])
    
    return df

def generate_model_comparison_table(df):
    """Generate a clean, formatted comparison table between TensorFlow and scikit-learn models"""
    # Prepare data for tabular display
    model_comparison = df.pivot_table(
        index=['channel_no', 'segment_dim'],
        columns='model_type',
        values=['accuracy', 'f1_score']
    ).reset_index()
    
    # Flatten the multi-level columns
    model_comparison.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in model_comparison.columns]
    
    # Add difference columns
    model_comparison['accuracy_diff'] = model_comparison['accuracy_tensorflow'] - model_comparison['accuracy_sklearn']
    model_comparison['f1_score_diff'] = model_comparison['f1_score_tensorflow'] - model_comparison['f1_score_sklearn']
    
    # Add a visual indicator column
    model_comparison['better_model'] = model_comparison.apply(
        lambda row: 'TensorFlow' if row['accuracy_diff'] > 0 
                    else 'scikit-learn' if row['accuracy_diff'] < 0 
                    else 'Equal', 
        axis=1
    )
    
    # Create a more readable table
    comparison_table = pd.DataFrame()
    comparison_table['Channels'] = model_comparison['channel_no'].astype(int)
    comparison_table['Window Size'] = model_comparison['segment_dim'].astype(int)
    comparison_table['TF Accuracy'] = model_comparison['accuracy_tensorflow'].round(4)
    comparison_table['SKL Accuracy'] = model_comparison['accuracy_sklearn'].round(4)
    comparison_table['Diff (TF-SKL)'] = model_comparison['accuracy_diff'].round(6)
    comparison_table['TF F1'] = model_comparison['f1_score_tensorflow'].round(4)
    comparison_table['SKL F1'] = model_comparison['f1_score_sklearn'].round(4)
    comparison_table['F1 Diff'] = model_comparison['f1_score_diff'].round(6)
    comparison_table['Better Model'] = model_comparison['better_model']
    
    # Sort by absolute difference to highlight the most significant differences
    comparison_table['abs_diff'] = comparison_table['Diff (TF-SKL)'].abs()
    comparison_table = comparison_table.sort_values(['Channels', 'abs_diff'], ascending=[True, False])
    comparison_table = comparison_table.drop('abs_diff', axis=1)
    
    return comparison_table

def print_summary_statistics(comparison_table):
    """Print summary statistics about the model comparison"""
    tf_wins = (comparison_table['Diff (TF-SKL)'] > 0).sum()
    skl_wins = (comparison_table['Diff (TF-SKL)'] < 0).sum()
    ties = (comparison_table['Diff (TF-SKL)'] == 0).sum()
    
    total_configs = len(comparison_table)
    
    print("\nSUMMARY STATISTICS:")
    print(f"Total configurations compared: {total_configs}")
    print(f"TensorFlow better: {tf_wins} ({tf_wins/total_configs:.1%})")
    print(f"scikit-learn better: {skl_wins} ({skl_wins/total_configs:.1%})")
    print(f"Equal performance: {ties} ({ties/total_configs:.1%})")
    
    print(f"\nAverage TensorFlow advantage: {comparison_table['Diff (TF-SKL)'].mean():.6f}")
    print(f"Max TensorFlow advantage: {comparison_table['Diff (TF-SKL)'].max():.6f}")
    print(f"Max scikit-learn advantage: {comparison_table['Diff (TF-SKL)'].min():.6f}")
    
    # Analyze by channel count
    print("\nPERFORMANCE BY CHANNEL COUNT:")
    channel_stats = comparison_table.groupby('Channels').apply(
        lambda x: pd.Series({
            'Configs': len(x),
            'TF Wins': (x['Diff (TF-SKL)'] > 0).sum(),
            'SKL Wins': (x['Diff (TF-SKL)'] < 0).sum(),
            'Avg Diff': x['Diff (TF-SKL)'].mean()
        })
    )
    
    print(tabulate(channel_stats, headers='keys', tablefmt='pretty'))

def main():
    # Process data
    print("Processing data...")
    df = preprocess_data(results_df)
    
    # Generate and display comparison table
    print("\nMODEL COMPARISON: TENSORFLOW vs SCIKIT-LEARN")
    print("="*80)
    comparison_table = generate_model_comparison_table(df)
    
    # Display table using tabulate for nice formatting
    print(tabulate(comparison_table, headers='keys', tablefmt='pretty', showindex=False))
    
    # Print summary statistics
    print_summary_statistics(comparison_table)
    
    # Save to CSV
    output_dir = Path('results/analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    comparison_table.to_csv(output_dir / 'model_comparison_table.csv', index=False)
    print(f"\nComparison table saved to {output_dir / 'model_comparison_table.csv'}")

if __name__ == "__main__":
    main() 