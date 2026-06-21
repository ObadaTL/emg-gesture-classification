import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)
sns.set_style("whitegrid")

# Load the results
results_file = 'results/experiment_results_full_20250411_182045.csv'
results_df = pd.read_csv(results_file)

# Create output directory for visualizations
output_dir = Path('results/analysis')
output_dir.mkdir(exist_ok=True, parents=True)

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
    
    # Define channel-based color map with better contrast
    channel_numbers = sorted(df['channel_no'].unique())
    
    # Define color scheme for channel numbers (red palette)
    color_base = {
        4: '#fcae91',  # Lightest red for 4 channels
        5: '#fb6a4a',  # Light red for 5 channels
        6: '#de2d26',  # Medium red for 6 channels
        7: '#a50f15',  # Dark red for 7 channels
        8: '#67000d',  # Darkest red for 8 channels
    }
    
    # Line styles for different models
    line_styles = {
        'sklearn': 'dashed',
        'tensorflow': 'solid'
    }
    
    # Line width and marker size
    line_width = 3.0
    marker_size = 9
    
    # Create a grouped bar chart
    for ch in channel_numbers:
        for model in df['model_type'].unique():
            subset = df[(df['channel_no'] == ch) & (df['model_type'] == model)]
            # Sort by segment dimension to ensure proper line connection
            subset = subset.sort_values('segment_dim')
            
            if not subset.empty:  # Only plot if data exists
                plt.plot(
                    subset['segment_dim'], 
                    subset['accuracy'], 
                    marker='o' if model == 'tensorflow' else 's',  # Different markers
                    markersize=marker_size,
                    linestyle=line_styles[model],
                    linewidth=line_width,
                    color=color_base[ch],
                    label=f"{ch} channels - {model}"
                )
    
    # Set x-axis ticks to show segment sizes
    segment_sizes = sorted(df['segment_dim'].unique())
    plt.xticks(segment_sizes)
    
    # Set y-axis to start at 0.89 to better show differences
    plt.ylim(0.89, 1.0)
    
    # Add labels and title with professional formatting
    plt.xlabel('Window Size', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=14)
    plt.title('Classification Accuracy by Window Size', fontweight='bold', fontsize=16)
    
    # Improve grid appearance - lighter grid for better contrast with dark lines
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    
    # Set white background for better contrast
    plt.gca().set_facecolor('white')
    
    # Add a border around the plot
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    
    # Create a custom legend with channel numbers and model types
    from matplotlib.lines import Line2D
    
    # Create legend elements for channel numbers (colors)
    channel_handles = []
    for ch in channel_numbers:
        channel_handles.append(
            Line2D([0], [0], color=color_base[ch], linewidth=line_width, 
                  label=f"{ch}")
        )
    
    # Create legend elements for model types (line styles)
    model_handles = []
    for model, style in line_styles.items():
        model_handles.append(
            Line2D([0], [0], color='black', linestyle=style, linewidth=line_width,
                  label=f"{model}")
        )
    
    # Add two legends - one for channel counts and one for model types
    first_legend = plt.legend(handles=channel_handles, loc='upper left', 
                             title='Channel Count', frameon=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=model_handles, loc='lower right', 
              title='Model Type', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_segment_size.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(df):
    """Plot comparison between sklearn and tensorflow models"""
    # Prepare data
    model_comparison = df.pivot_table(
        index=['channel_no', 'segment_dim'],
        columns='model_type',
        values='accuracy'
    ).reset_index()
    
    # Create a figure with sufficient size but without excessive padding
    plt.figure(figsize=(8, 8))
    
    # Create a categorical color scheme based on channel count
    channel_colors = {
        4: '#67001f',  # Dark red
        5: '#9e2230',  # Medium-dark red
        6: '#d6604d',  # Medium red
        7: '#f4a582',  # Light red
        8: '#fddbc7',  # Very light red
    }
    
    # Fixed marker size - simpler approach
    segment_sizes = {
        256: 50,
        512: 70,
        1024: 90,
        2048: 110
    }
    
    # Calculate axis limits before plotting
    min_acc = min(model_comparison['sklearn'].min(), model_comparison['tensorflow'].min())
    max_acc = max(model_comparison['sklearn'].max(), model_comparison['tensorflow'].max())
    
    # Add a bit of padding to the limits
    padding = 0.01
    min_val = min_acc - padding
    max_val = max_acc + padding
    
    # Create plot area with proper margins
    ax = plt.subplot(111)
    
    # Scatter plot with improved styling
    for idx, row in model_comparison.iterrows():
        ax.scatter(
            row['sklearn'], 
            row['tensorflow'],
            s=segment_sizes[int(row['segment_dim'])], 
            alpha=0.8,
            edgecolor='black',
            linewidth=0.8,
            c=channel_colors[row['channel_no']]
        )
    
    # Add diagonal line (y=x)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1.5)
    
    # Add labels for each point with a cleaner format - smaller font and fewer points
    # Only label points that are significantly different from diagonal
    for i, row in model_comparison.iterrows():
        # Calculate distance from diagonal
        diag_distance = abs(row['sklearn'] - row['tensorflow'])
        if diag_distance > 0.002:  # Only label points with significant difference
            ax.annotate(
                f"Ch{int(row['channel_no'])}-{int(row['segment_dim'])}",
                (row['sklearn'], row['tensorflow']),
                fontsize=7,
                alpha=0.9,
                fontweight='bold',
                xytext=(3, 3),
                textcoords='offset points'
            )
    
    # Set consistent axis limits and grid
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add explanatory text for the diagonal line - repositioned
    ax.text(
        0.95, 0.05, 
        "Equal performance", 
        ha='right', va='bottom', 
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
        rotation=45
    )
    
    # Add text labels for the regions - smaller and positioned to avoid overlap
    ax.text(
        0.25, 0.85, 
        "TensorFlow better",
        ha='center', va='center',
        fontsize=9,
        alpha=0.7,
        weight='bold',
        color='navy'
    )
    ax.text(
        0.85, 0.25, 
        "sklearn better",
        ha='center', va='center',
        fontsize=9,
        alpha=0.7,
        weight='bold',
        color='darkred'
    )
    
    # Add labels and title with professional formatting
    ax.set_xlabel('sklearn Accuracy', fontweight='bold', fontsize=10)
    ax.set_ylabel('tensorflow Accuracy', fontweight='bold', fontsize=10)
    ax.set_title('sklearn vs tensorflow Model Performance', fontweight='bold', fontsize=12)
    
    # Set white background for better contrast
    ax.set_facecolor('white')
    
    # Add a border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    # Create consolidated legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    
    # Create channel legend items (circles of different colors)
    channel_handles = []
    for ch in sorted(df['channel_no'].unique()):
        channel_handles.append(
            mpatches.Circle((0.5, 0.5), radius=0.25, 
                         facecolor=channel_colors[ch], 
                         label=f"Ch {ch}", 
                         edgecolor='black')
        )
    
    # Create segment size legend items (circles of different sizes)
    segment_handles = []
    for seg in sorted(df['segment_dim'].unique()):
        segment_handles.append(
            Line2D([0], [0], marker='o', color='gray',
                  markersize=np.sqrt(segment_sizes[seg])/3,
                  label=f"{seg}")
        )
    
    # Position legends to the right of the figure
    # First adjust the main plot area to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Create a single legend with both channel and segment information
    legend_elements = []
    
    # Add channel legend header
    legend_elements.append(Line2D([0], [0], color='w', label='Channel Count:'))
    # Add channel elements
    for ch in sorted(df['channel_no'].unique()):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=channel_colors[ch],
                  markersize=6, label=f"{ch}")
        )
    
    # Add window size header with some spacing
    legend_elements.append(Line2D([0], [0], color='w', label=' '))
    legend_elements.append(Line2D([0], [0], color='w', label='Window Size:'))
    # Add window size elements
    for seg in sorted(df['segment_dim'].unique()):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='gray', 
                  markersize=np.sqrt(segment_sizes[seg])/3,
                  label=f"{seg}")
        )
    
    # Create the legend with smaller font
    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1.01, 0.5),
             fontsize=8,
             frameon=True,
             title='Legend')
    
    # Save with reduced size and appropriate DPI
    plt.savefig(output_dir / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_heatmap(df):
    """Create a heatmap of accuracy by channel number and segment size"""
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
        plt.savefig(output_dir / f'heatmap_{model_type}.png', dpi=300)
        plt.close()

def plot_channel_impact(df):
    """Plot the impact of increasing channel numbers on accuracy"""
    plt.figure(figsize=(12, 8))
    
    # Define segment size-based grayscale color map with better contrast
    segment_sizes = sorted(df['segment_dim'].unique())
    
    # Define specific grayscale colors for each segment size - from light to dark
    colors = [
        '#D9D9D9',  # Light gray for 256
        '#A0A0A0',  # Medium gray for 512
        '#606060',  # Dark gray for 1024
        '#000000',  # Black for 2048
    ]
    
    # Line styles for different models
    line_styles = {
        'sklearn': 'dashed',
        'tensorflow': 'solid'
    }
    
    # Line width and marker size
    line_width = 3.0
    marker_size = 9
    
    # Group by segment size and channel number
    seg_model_combinations = []
    for i, seg_size in enumerate(segment_sizes):
        for model in df['model_type'].unique():
            subset = df[(df['segment_dim'] == seg_size) & (df['model_type'] == model)]
            if not subset.empty:  # Only plot if data exists
                # Sort by channel number to ensure line connects points in order
                subset = subset.sort_values('channel_no')
                
                # Plot the line
                plt.plot(
                    subset['channel_no'], 
                    subset['accuracy'], 
                    marker='o' if model == 'tensorflow' else 's',  # Different markers
                    markersize=marker_size,
                    linestyle=line_styles[model],
                    linewidth=line_width,
                    color=colors[i],
                    label=f"Seg{seg_size} - {model}"
                )
                seg_model_combinations.append((seg_size, model))
    
    # Set x-axis ticks to only show integer values (no .5 values)
    channel_numbers = sorted(df['channel_no'].unique())
    plt.xticks(channel_numbers)
    
    # Set y-axis to start at 0.89 to better show differences
    plt.ylim(0.89, 1.0)
    
    # Add labels and title with professional formatting
    plt.xlabel('Number of Channels', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=14)
    plt.title('Impact of Channel Count on Classification Accuracy', fontweight='bold', fontsize=16)
    
    # Improve grid appearance - lighter grid for better contrast with dark lines
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    
    # Set white background for better contrast
    plt.gca().set_facecolor('white')
    
    # Create a custom legend with segment sizes and model types
    from matplotlib.lines import Line2D
    
    # Create legend elements for segment sizes (colors)
    segment_handles = []
    for i, seg_size in enumerate(segment_sizes):
        segment_handles.append(
            Line2D([0], [0], color=colors[i], linewidth=line_width, 
                  label=f"{seg_size}")
        )
    
    # Create legend elements for model types (line styles)
    model_handles = []
    for model, style in line_styles.items():
        model_handles.append(
            Line2D([0], [0], color='black', linestyle=style, linewidth=line_width,
                  label=f"{model}")
        )
    
    # Add two legends - one for segment sizes and one for model types
    first_legend = plt.legend(handles=segment_handles, loc='upper left', 
                             title='Window Size', frameon=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=model_handles, loc='lower right', 
              title='Model Type', frameon=True)
    
    # Add a border around the plot
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(df):
    """Plot comparison of different metrics for each configuration"""
    # Melt the dataframe to have metrics as a variable
    metrics_df = df.melt(
        id_vars=['experiment_name', 'channel_no', 'segment_dim', 'model_type'],
        value_vars=['accuracy', 'f1_score', 'precision', 'recall'],
        var_name='metric',
        value_name='value'
    )
    
    # Create a nicer color palette for metrics
    metrics_palette = {
        'accuracy': '#4e79a7',    # Blue
        'f1_score': '#f28e2c',    # Orange
        'precision': '#59a14f',   # Green
        'recall': '#e15759'       # Red
    }
    
    # Plot with seaborn's Figure-level interface
    g = sns.catplot(
        data=metrics_df,
        x='segment_dim',
        y='value',
        hue='metric',
        col='channel_no',
        row='model_type',
        kind='bar',
        height=3.5,
        aspect=1.2,
        palette=metrics_palette,
        legend=False  # Don't create legend here, we'll do it manually
    )
    
    # Customize the plot
    g.set_axis_labels('Window Size', 'Score')
    g.set_titles('Channels: {col_name} - Model: {row_name}')
    
    # Create a custom legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=metrics_palette['accuracy'], label='Accuracy'),
        Patch(facecolor=metrics_palette['f1_score'], label='F1 Score'),
        Patch(facecolor=metrics_palette['precision'], label='Precision'),
        Patch(facecolor=metrics_palette['recall'], label='Recall')
    ]
    
    # Add the legend to the figure, not to any specific axis
    g.fig.legend(handles=legend_elements, 
                loc='lower center', 
                ncol=4, 
                bbox_to_anchor=(0.5, 0),
                frameon=True)
    
    # Adjust the figure layout to make room for the legend
    g.fig.subplots_adjust(bottom=0.15)
    
    # Save the figure with tight layout and appropriate DPI
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

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

def print_model_comparison_table(df):
    """Print a formatted table of model comparison results to the terminal"""
    print("\n" + "="*80)
    print(" MODEL COMPARISON RESULTS ".center(80, "="))
    print("="*80)
    
    # Prepare data for tabular display
    model_comparison = df.pivot_table(
        index=['channel_no', 'segment_dim'],
        columns='model_type',
        values='accuracy'
    ).reset_index()
    
    # Add a difference column
    model_comparison['diff'] = model_comparison['tensorflow'] - model_comparison['sklearn']
    
    # Sort by absolute difference to highlight the most significant differences
    model_comparison['abs_diff'] = model_comparison['diff'].abs()
    model_comparison = model_comparison.sort_values('abs_diff', ascending=False)
    
    # Format the table headers
    print(f"{'Channels':<10}{'Window Size':<15}{'sklearn':<15}{'tensorflow':<15}{'Difference':<15}")
    print("-"*80)
    
    # Print each row with formatting
    for _, row in model_comparison.iterrows():
        ch_num = int(row['channel_no'])
        seg_dim = int(row['segment_dim'])
        sklearn_acc = row['sklearn']
        tf_acc = row['tensorflow']
        diff = row['diff']
        
        # Format with colors for positive/negative differences
        diff_str = f"{diff:.6f}"
        if diff > 0:
            # TensorFlow better (would be colored green in a terminal that supports it)
            diff_display = f"+{diff_str}"
        else:
            # sklearn better (would be colored red in a terminal that supports it)
            diff_display = f"{diff_str}"
        
        print(f"{ch_num:<10}{seg_dim:<15}{sklearn_acc:.6f}{' ':<9}{tf_acc:.6f}{' ':<9}{diff_display}")
    
    print("="*80)
    
    # Additional summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Average tensorflow advantage: {model_comparison['diff'].mean():.6f}")
    print(f"Max tensorflow advantage: {model_comparison['diff'].max():.6f}")
    print(f"Max sklearn advantage: {model_comparison['diff'].min():.6f}")
    print(f"Number of configurations where tensorflow is better: {(model_comparison['diff'] > 0).sum()}")
    print(f"Number of configurations where sklearn is better: {(model_comparison['diff'] < 0).sum()}")
    if (model_comparison['diff'] == 0).sum() > 0:
        print(f"Number of configurations with equal performance: {(model_comparison['diff'] == 0).sum()}")
    print("="*80 + "\n")

def main():
    """Main function to run all analyses"""
    print("Loading and preprocessing data...")
    df = preprocess_data(results_df)
    
    print("Generating summary table...")
    summary = generate_summary_table(df)
    print(summary.head())
    
    # Print model comparison table
    print_model_comparison_table(df)
    
    print("Creating visualizations...")
    plot_accuracy_by_segment_size(df)
    plot_model_comparison(df)
    plot_heatmap(df)
    plot_channel_impact(df)
    plot_metrics_comparison(df)
    
    print("Finding best configurations...")
    best_configs = generate_best_configurations(df)
    print(best_configs.head())
    
    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 