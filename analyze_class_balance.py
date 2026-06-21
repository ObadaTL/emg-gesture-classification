import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def visualize_simple_class_balance(original_counts, final_counts):
    """
    Create a simple before-and-after visualization of class balancing focused on percentages.
    
    Args:
        original_counts: The original class distribution
        final_counts: Final distribution after balancing
    """
    # Calculate percentages
    original_percentages = (original_counts / np.sum(original_counts)) * 100
    final_percentages = (final_counts / np.sum(final_counts)) * 100
    
    # Create a DataFrame for plotting
    data = []
    for i, (orig_pct, final_pct) in enumerate(zip(original_percentages, final_percentages)):
        if i < len(GESTURE_NAMES):
            data.append({
                'Class': i,
                'Gesture': GESTURE_NAMES[i],
                'Original': orig_pct,
                'Balanced': final_pct
            })
    
    df = pd.DataFrame(data)
    
    # Melt the DataFrame for seaborn
    df_melted = pd.melt(df, 
                      id_vars=['Class', 'Gesture'], 
                      value_vars=['Original', 'Balanced'],
                      var_name='Distribution', 
                      value_name='Percentage')
    
    # Create the plot
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Create the grouped bar chart with black for Original and light grey for Balanced
    ax = sns.barplot(x='Gesture', y='Percentage', hue='Distribution', 
                    data=df_melted, palette=['black', '#CCCCCC'])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=9)
    
    # Customize plot
    ax.set_title("UCI EMG Dataset Class Distribution Before and After Balancing", fontsize=14, pad=20)
    ax.set_xlabel("Gesture Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Percentage (%)", fontsize=12, labelpad=10)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    
    # Calculate the theoretical balanced distribution
    num_classes = len(GESTURE_NAMES)
    expected_balanced = 100 / num_classes
    
    # Add a horizontal dashed line at the expected balanced percentage
    plt.axhline(y=expected_balanced, color='red', linestyle='--', alpha=0.5)
    
    # Add annotation for the balanced line
    plt.text(1, expected_balanced + 1, 
            f'Ideal balanced distribution: {expected_balanced:.1f}%', 
            color='red', fontsize=9, style='italic')
    
    # Add improvement statistics as annotation
    original_ratio = original_counts[0] / min(original_counts)
    final_ratio = max(final_counts) / min(final_counts)
    improvement = original_ratio / final_ratio
    
    annotation_text = (
        # f"Class imbalance improvements:\n"
        # f"• Original imbalance ratio: {original_ratio:.1f}:1\n"
        # f"• Final imbalance ratio: {final_ratio:.1f}:1\n"
        # f"• Balance improvement factor: {improvement:.1f}×"
    )
    
    # plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='figure fraction', 
    #             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
    #             fontsize=9)
    
    plt.tight_layout()
    
    # Save the figure in high resolution
    output_file = "class_balance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Class balance comparison saved to {output_file}")
    plt.close()

def main():
    # Original counts from the class distribution analysis
    original_counts = np.array([5311, 492, 473, 489, 489, 497, 495, 28])
    
    # Final balanced counts from empirical results
    final_counts = np.array([472, 839, 822, 804, 839, 823, 825, 157])
    
    # Create the simple visualization
    visualize_simple_class_balance(original_counts, final_counts)
    
    # Print statistics for verification
    print("\nClass Balancing Statistics:")
    print(f"Original distribution: {original_counts}")
    print(f"Final balanced distribution: {final_counts}")
    
    # Calculate improvement in balance
    original_ratio = original_counts[0] / min(original_counts)
    final_ratio = max(final_counts) / min(final_counts)
    improvement = original_ratio / final_ratio
    
    print(f"\nImbalance ratio reduced from {original_ratio:.1f}:1 to {final_ratio:.1f}:1")
    print(f"Balance improvement factor: {improvement:.1f}x")
    
    # Calculate percentages for comparison
    original_pct = (original_counts[0] / sum(original_counts)) * 100
    final_pct = (final_counts[0] / sum(final_counts)) * 100
    
    print(f"\nMajority class (Unmarked) reduced from {original_pct:.1f}% to {final_pct:.1f}% of the dataset")
    print(f"Minority class (Extended Palm) increased from {(original_counts[7]/sum(original_counts))*100:.2f}% to {(final_counts[7]/sum(final_counts))*100:.2f}% of the dataset")

if __name__ == "__main__":
    main()