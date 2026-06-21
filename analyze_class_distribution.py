import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import seaborn as sns

# Set up the plotting style for publication quality
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'arial'

# Define gesture class names for proper labeling
GESTURE_NAMES = {
    0: "Unmarked",
    1: "Hand at Rest",
    2: "Fist",
    3: "Wrist Flexion",
    4: "Wrist Extension",
    5: "Radial Deviation",
    6: "Ulnar Deviation",
    7: "Extended Palm"
}

# Load configuration
with open('config/uci_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load processed data if available
cache_file = Path(config['paths']['processed_data']) / 'combined_emg_data.csv'

if cache_file.exists():
    print(f"Loading data from: {cache_file}")
    data = pd.read_csv(cache_file)
    print(f"Data shape: {data.shape}")
    
    # Analyze class distribution
    class_counts = data['class'].value_counts().sort_index()
    print("\nClass distribution:")
    print(class_counts)
    
    # Calculate percentages
    class_percentages = class_counts / len(data) * 100
    print("\nClass percentages:")
    for cls, pct in class_percentages.items():
        print(f"Class {cls} ({GESTURE_NAMES.get(cls, 'Unknown')}): {pct:.2f}%")
    
    # Create DataFrame for plotting with proper labels
    plot_df = pd.DataFrame({
        'Percentage': class_percentages,
        'Count': class_counts,
        'Gesture': [GESTURE_NAMES.get(cls, f"Class {cls}") for cls in class_percentages.index]
    })
    
    # Create a single large figure
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Create the bar plot with a single color (black) for dissertation formality
    ax = sns.barplot(x=plot_df.index, y='Percentage', data=plot_df, color='black', edgecolor='black')
    
    # Add value labels on bars showing both percentage and count
    for i, bar in enumerate(ax.patches):
        percentage = plot_df['Percentage'].iloc[i]
        count = plot_df['Count'].iloc[i]
        
        # Format large numbers with commas
        count_str = f"{count:,}"
        
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.5,
            f'{percentage:.1f}%\n({count_str})',
            ha='center', va='bottom', fontsize=9
        )
    
    # Customize plot
    ax.set_title("UCI EMG Dataset Class Distribution", fontsize=14, pad=20)
    ax.set_xlabel("Gesture Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Percentage (%)", fontsize=12, labelpad=10)
    ax.set_ylim(0, max(plot_df['Percentage']) * 1.15)  # Add headroom for labels
    
    # Set better x-tick labels with gesture names
    ax.set_xticklabels([GESTURE_NAMES.get(i, f"Class {i}") for i in plot_df.index], 
                      rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    
    # Calculate the theoretical balanced distribution (if all classes had equal representation)
    num_classes = len(class_percentages.index)
    expected_balanced = 100 / num_classes  # Each class would have this percentage in a balanced dataset
    
    # Add a horizontal dashed line at the expected balanced percentage
    plt.axhline(y=expected_balanced, color='darkgray', linestyle='--', alpha=0.8)
    
    # Add annotation for the balanced line
    plt.text(len(class_percentages)*0.7, expected_balanced+0.7, 
             f'Ideal balanced distribution: {expected_balanced:.1f}%', 
             color='dimgray', fontsize=9, style='italic')
    
    # Add footnote with explanation
    plt.figtext(0.5, 0.01, 
                f"Total samples: {len(data):,} | 'Ideal balanced distribution' represents equal representation across all {num_classes} classes", 
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save the figure in high resolution
    output_file = "class_distribution_formal.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Formal dissertation-style plot saved to {output_file}")

else:
    print(f"Data file not found: {cache_file}") 