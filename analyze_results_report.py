import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Load the results
results_file = 'results/experiment_results_full_20250227_142920.csv'
results_df = pd.read_csv(results_file)

# Create output directory for visualizations and report
output_dir = Path('results/analysis')
output_dir.mkdir(exist_ok=True, parents=True)

def df_to_markdown(df, index=True):
    """Convert DataFrame to markdown table"""
    markdown = []
    
    # Get headers
    headers = df.columns.tolist()
    if index:
        headers = [''] + headers
    
    # Add header row
    markdown.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
    
    # Add separator row
    markdown.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    
    # Add data rows
    for i, row in df.iterrows():
        values = row.tolist()
        if index:
            values = [i] + values
        markdown.append('| ' + ' | '.join(str(v) for v in values) + ' |')
    
    return '\n'.join(markdown)

def preprocess_data(df):
    """Preprocess the data for analysis"""
    # Convert relevant columns to numeric
    numeric_cols = ['accuracy', 'f1_score', 'precision', 'recall']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract model parameters from experiment name
    df['model_name'] = df['experiment_name'].apply(lambda x: x.split('_')[-1])
    
    # Create a combined parameter column for easier grouping
    df['segment_config'] = df.apply(
        lambda row: f"seg{row['segment_dim']}_inc{row['increment_dim']}", 
        axis=1
    )
    
    return df

def generate_summary_stats(df):
    """Generate summary statistics for the report"""
    # Overall stats
    total_experiments = len(df)
    unique_configs = len(df.groupby(['channel_no', 'segment_dim', 'increment_dim', 'model_type']))
    
    # Best overall configuration
    best_idx = df['accuracy'].idxmax()
    best_config = df.loc[best_idx]
    
    # Worst overall configuration
    worst_idx = df['accuracy'].idxmin()
    worst_config = df.loc[worst_idx]
    
    # Average performance by model type
    model_performance = df.groupby('model_type')['accuracy'].mean().to_dict()
    
    # Average performance by channel count
    channel_performance = df.groupby('channel_no')['accuracy'].mean().to_dict()
    
    # Average performance by segment size
    segment_performance = df.groupby('segment_dim')['accuracy'].mean().to_dict()
    
    return {
        'total_experiments': total_experiments,
        'unique_configs': unique_configs,
        'best_config': best_config,
        'worst_config': worst_config,
        'model_performance': model_performance,
        'channel_performance': channel_performance,
        'segment_performance': segment_performance
    }

def generate_summary_table(df):
    """Generate a summary table of results"""
    # Group by channel number, segment dimension, and model type
    grouped = df.groupby(['channel_no', 'segment_dim', 'model_type']).agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).reset_index()
    
    # Format the table for better readability
    summary_table = pd.DataFrame()
    summary_table['Channels'] = grouped['channel_no']
    summary_table['Segment Size'] = grouped['segment_dim']
    summary_table['Model Type'] = grouped['model_type']
    summary_table['Accuracy'] = grouped['accuracy']['mean'].round(4)
    summary_table['F1 Score'] = grouped['f1_score']['mean'].round(4)
    
    # Save to CSV
    summary_table.to_csv(output_dir / 'summary_table.csv', index=False)
    
    return summary_table

def plot_accuracy_by_segment_size(df):
    """Plot accuracy by segment size for each channel configuration and model type"""
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    for ch in df['channel_no'].unique():
        for model in df['model_type'].unique():
            subset = df[(df['channel_no'] == ch) & (df['model_type'] == model)]
            plt.plot(
                subset['segment_dim'], 
                subset['accuracy'], 
                marker='o', 
                linewidth=2, 
                label=f"{ch} channels - {model}"
            )
    
    plt.xlabel('Segment Size')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Segment Size')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_segment_size.png', dpi=300)
    plt.close()
    
    return 'accuracy_by_segment_size.png'

def plot_model_comparison(df):
    """Plot comparison between sklearn and tensorflow models"""
    # Prepare data
    model_comparison = df.pivot_table(
        index=['channel_no', 'segment_dim'],
        columns='model_type',
        values='accuracy'
    ).reset_index()
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(
        model_comparison['sklearn'], 
        model_comparison['tensorflow'],
        s=100, 
        alpha=0.7,
        c=model_comparison['segment_dim'],  # Color by segment dimension
        cmap='viridis'
    )
    
    # Add diagonal line (y=x)
    min_val = min(df['accuracy'].min(), df['accuracy'].min())
    max_val = max(df['accuracy'].max(), df['accuracy'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add labels for each point
    for i, row in model_comparison.iterrows():
        plt.annotate(
            f"Ch{int(row['channel_no'])}-Seg{int(row['segment_dim'])}",
            (row['sklearn'], row['tensorflow']),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('sklearn Accuracy')
    plt.ylabel('tensorflow Accuracy')
    plt.title('sklearn vs tensorflow Model Performance')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar for segment dimension
    cbar = plt.colorbar()
    cbar.set_label('Segment Dimension')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300)
    plt.close()
    
    return 'model_comparison.png'

def plot_heatmap(df):
    """Create a heatmap of accuracy by channel number and segment size"""
    heatmap_files = []
    
    # Prepare data for heatmap
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        
        # Create pivot table
        heatmap_data = model_df.pivot_table(
            index='channel_no',
            columns='segment_dim',
            values='accuracy'
        )
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='.3f', 
            cmap='YlGnBu',
            vmin=0.7,  # Set minimum value for better color contrast
            vmax=1.0   # Set maximum value
        )
        
        plt.title(f'Accuracy Heatmap - {model_type.capitalize()} Model')
        plt.xlabel('Segment Size')
        plt.ylabel('Number of Channels')
        plt.tight_layout()
        
        filename = f'heatmap_{model_type}.png'
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()
        
        heatmap_files.append(filename)
    
    return heatmap_files

def plot_channel_impact(df):
    """Plot the impact of increasing channel numbers on accuracy"""
    plt.figure(figsize=(12, 8))
    
    # Group by segment size and channel number
    for seg_size in df['segment_dim'].unique():
        for model in df['model_type'].unique():
            subset = df[(df['segment_dim'] == seg_size) & (df['model_type'] == model)]
            plt.plot(
                subset['channel_no'], 
                subset['accuracy'], 
                marker='o', 
                linewidth=2, 
                label=f"Seg{seg_size} - {model}"
            )
    
    plt.xlabel('Number of Channels')
    plt.ylabel('Accuracy')
    plt.title('Impact of Channel Count on Classification Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_impact.png', dpi=300)
    plt.close()
    
    return 'channel_impact.png'

def generate_best_configurations(df):
    """Find the best configurations for each metric"""
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    best_configs = {}
    
    for metric in metrics:
        # Find the best configuration for each channel count
        for ch in df['channel_no'].unique():
            ch_df = df[df['channel_no'] == ch]
            best_idx = ch_df[metric].idxmax()
            best_row = ch_df.loc[best_idx]
            
            config_key = f"best_{metric}_ch{int(ch)}"
            best_configs[config_key] = {
                'experiment_name': best_row['experiment_name'],
                'segment_dim': best_row['segment_dim'],
                'increment_dim': best_row['increment_dim'],
                'model_type': best_row['model_type'],
                'value': best_row[metric]
            }
    
    # Convert to DataFrame for easier viewing
    best_df = pd.DataFrame.from_dict(best_configs, orient='index')
    best_df.to_csv(output_dir / 'best_configurations.csv')
    
    return best_df

def generate_markdown_report(stats, summary_table, best_configs, plot_files, df):
    """Generate a comprehensive markdown report"""
    report_path = output_dir / 'experiment_analysis_report.md'
    
    with open(report_path, 'w') as f:
        # Title
        f.write("# EMG Classification Experiment Analysis Report\n\n")
        
        # Summary section
        f.write("## 1. Executive Summary\n\n")
        f.write("This report analyzes the results of EMG classification experiments with various configurations.\n\n")
        
        f.write("### Key Findings:\n\n")
        f.write(f"- Total experiments conducted: {stats['total_experiments']}\n")
        f.write(f"- Best overall accuracy: **{stats['best_config']['accuracy']:.4f}** achieved with:\n")
        f.write(f"  - Configuration: {stats['best_config']['experiment_name']}\n")
        f.write(f"  - Channels: {stats['best_config']['channel_no']}\n")
        f.write(f"  - Segment size: {stats['best_config']['segment_dim']}\n")
        f.write(f"  - Model type: {stats['best_config']['model_type']}\n\n")
        
        f.write("### Performance by Model Type:\n\n")
        for model, acc in stats['model_performance'].items():
            f.write(f"- {model.capitalize()}: {acc:.4f}\n")
        f.write("\n")
        
        f.write("### Performance by Channel Count:\n\n")
        for ch, acc in stats['channel_performance'].items():
            f.write(f"- {int(ch)} channels: {acc:.4f}\n")
        f.write("\n")
        
        f.write("### Performance by Segment Size:\n\n")
        for seg, acc in sorted(stats['segment_performance'].items()):
            f.write(f"- Segment size {int(seg)}: {acc:.4f}\n")
        f.write("\n")
        
        # Detailed results section
        f.write("## 2. Detailed Results\n\n")
        
        f.write("### 2.1 Summary Table\n\n")
        f.write("The table below shows the average performance metrics for each configuration:\n\n")
        
        # Convert summary table to markdown
        f.write(df_to_markdown(summary_table, index=False))
        f.write("\n\n")
        
        # Visualizations section
        f.write("## 3. Visualizations\n\n")
        
        # Accuracy by segment size
        f.write("### 3.1 Accuracy by Segment Size\n\n")
        f.write("This plot shows how classification accuracy changes with different segment sizes for each channel configuration and model type.\n\n")
        f.write(f"![Accuracy by Segment Size](./analysis/{plot_files['accuracy_by_segment']})\n\n")
        
        # Model comparison
        f.write("### 3.2 Model Comparison (sklearn vs tensorflow)\n\n")
        f.write("This scatter plot compares the performance of sklearn and tensorflow models for each configuration. Points above the diagonal line indicate configurations where tensorflow outperforms sklearn, while points below the line indicate the opposite.\n\n")
        f.write(f"![Model Comparison](./analysis/{plot_files['model_comparison']})\n\n")
        
        # Heatmaps
        f.write("### 3.3 Accuracy Heatmaps\n\n")
        f.write("These heatmaps show the accuracy for different combinations of channel counts and segment sizes for each model type.\n\n")
        
        for i, heatmap_file in enumerate(plot_files['heatmaps']):
            model_type = heatmap_file.split('_')[1].split('.')[0]
            f.write(f"#### {model_type.capitalize()} Model\n\n")
            f.write(f"![{model_type.capitalize()} Heatmap](./analysis/{heatmap_file})\n\n")
        
        # Channel impact
        f.write("### 3.4 Impact of Channel Count\n\n")
        f.write("This plot shows how classification accuracy changes with different channel counts for each segment size and model type.\n\n")
        f.write(f"![Channel Impact](./analysis/{plot_files['channel_impact']})\n\n")
        
        # Best configurations section
        f.write("## 4. Best Configurations\n\n")
        f.write("The table below shows the best configurations for each metric and channel count:\n\n")
        
        # Convert best configurations to markdown
        f.write(df_to_markdown(best_configs))
        f.write("\n\n")
        
        # Conclusions section
        f.write("## 5. Conclusions and Recommendations\n\n")
        
        # Determine if sklearn or tensorflow is better overall
        better_model = "sklearn" if stats['model_performance']['sklearn'] > stats['model_performance']['tensorflow'] else "tensorflow"
        
        f.write("### Key Observations:\n\n")
        f.write(f"1. **Model Performance**: Overall, {better_model} models performed better than {'tensorflow' if better_model == 'sklearn' else 'sklearn'} models.\n\n")
        
        # Determine if more channels always lead to better performance
        channel_trend = "increasing" if list(stats['channel_performance'].values())[0] < list(stats['channel_performance'].values())[-1] else "not necessarily increasing"
        f.write(f"2. **Channel Impact**: Performance {'generally improves' if channel_trend == 'increasing' else 'does not always improve'} with more channels.\n\n")
        
        # Determine if larger segment sizes lead to better performance
        segment_trend = "increasing" if list(stats['segment_performance'].values())[0] < list(stats['segment_performance'].values())[-1] else "not necessarily increasing"
        f.write(f"3. **Segment Size Impact**: Larger segment sizes {'generally lead to better' if segment_trend == 'increasing' else 'do not always improve'} performance.\n\n")
        
        # Recommendations
        f.write("### Recommendations:\n\n")
        f.write(f"1. **Preferred Model**: Use {better_model} models for future implementations as they showed better overall performance.\n\n")
        
        # Best overall configuration
        f.write(f"2. **Optimal Configuration**: The best overall configuration is:\n")
        f.write(f"   - Channels: {stats['best_config']['channel_no']}\n")
        f.write(f"   - Segment size: {stats['best_config']['segment_dim']}\n")
        f.write(f"   - Increment: {stats['best_config']['increment_dim']}\n")
        f.write(f"   - Model type: {stats['best_config']['model_type']}\n\n")
        
        # Trade-off considerations
        f.write("3. **Trade-offs**: If computational resources are limited, consider the following:\n")
        
        # Find a good balance between performance and complexity
        balanced_configs = df[(df['accuracy'] > 0.85) & (df['segment_dim'] < 1024)]
        if not balanced_configs.empty:
            balanced_idx = balanced_configs['accuracy'].idxmax()
            balanced_config = df.loc[balanced_idx]
            f.write(f"   - A balanced configuration with good performance and lower complexity: {balanced_config['experiment_name']}\n")
            f.write(f"   - Accuracy: {balanced_config['accuracy']:.4f}\n")
            f.write(f"   - Channels: {balanced_config['channel_no']}\n")
            f.write(f"   - Segment size: {balanced_config['segment_dim']}\n\n")
        
        # Future work
        f.write("### Future Work:\n\n")
        f.write("1. Investigate the impact of different feature extraction methods.\n")
        f.write("2. Explore more advanced deep learning architectures for the tensorflow models.\n")
        f.write("3. Conduct real-time performance evaluations to assess practical usability.\n")
    
    return report_path

def main():
    """Main function to run all analyses and generate report"""
    print("Loading and preprocessing data...")
    df = preprocess_data(results_df)
    
    print("Generating summary statistics...")
    stats = generate_summary_stats(df)
    
    print("Generating summary table...")
    summary_table = generate_summary_table(df)
    
    print("Creating visualizations...")
    accuracy_by_segment_plot = plot_accuracy_by_segment_size(df)
    model_comparison_plot = plot_model_comparison(df)
    heatmap_plots = plot_heatmap(df)
    channel_impact_plot = plot_channel_impact(df)
    
    print("Finding best configurations...")
    best_configs = generate_best_configurations(df)
    
    # Collect all plot files
    plot_files = {
        'accuracy_by_segment': accuracy_by_segment_plot,
        'model_comparison': model_comparison_plot,
        'heatmaps': heatmap_plots,
        'channel_impact': channel_impact_plot
    }
    
    print("Generating markdown report...")
    report_path = generate_markdown_report(stats, summary_table, best_configs, plot_files, df)
    
    print(f"Analysis complete! Report saved to {report_path}")

if __name__ == "__main__":
    main() 