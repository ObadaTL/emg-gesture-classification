"""Analyze and visualize `run_experiment.py --results-file` sweep output.

Consolidates what used to be three near-duplicate scripts (analyze_results.py,
analyze_results_report.py, compare_models.py), each hardcoding a specific
timestamped CSV filename, into one parameterized entry point.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-file results/experiment_results_full_20250101_000000.csv
    python scripts/analyze_results.py --report
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)
sns.set_style("whitegrid")

RESULTS_DIR = Path('results')


def latest_results_file() -> Path:
    """Most recently modified full-sweep results CSV in results/."""
    candidates = sorted(RESULTS_DIR.glob('experiment_results_full_*.csv'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No 'experiment_results_full_*.csv' files found in {RESULTS_DIR}/. "
            "Run scripts/run_experiment.py first, or pass --results-file explicitly."
        )
    return candidates[0]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce metric columns to numeric and derive convenience columns."""
    numeric_cols = ['accuracy', 'f1_score', 'precision', 'recall']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['model_name'] = df['experiment_name'].apply(lambda x: x.split('_')[-1])
    df['segment_config'] = df.apply(
        lambda row: f"seg{row['segment_dim']}_inc{row['increment_dim']}", axis=1
    )
    return df


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Mean/std of each metric per (channels, segment size, model type)."""
    grouped = df.groupby(['channel_no', 'segment_dim', 'model_type']).agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).reset_index()

    summary_table = pd.DataFrame()
    summary_table['Channels'] = grouped['channel_no']
    summary_table['Segment Size'] = grouped['segment_dim']
    summary_table['Model Type'] = grouped['model_type']
    summary_table['Accuracy'] = grouped['accuracy']['mean'].round(4)
    summary_table['F1 Score'] = grouped['f1_score']['mean'].round(4)

    summary_table.to_csv(output_dir / 'summary_table.csv', index=False)
    return summary_table


def model_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(channels, segment size) TensorFlow vs scikit-learn comparison table."""
    pivot = df.pivot_table(
        index=['channel_no', 'segment_dim'],
        columns='model_type',
        values=['accuracy', 'f1_score']
    ).reset_index()
    pivot.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in pivot.columns]

    pivot['accuracy_diff'] = pivot['accuracy_tensorflow'] - pivot['accuracy_sklearn']
    pivot['f1_score_diff'] = pivot['f1_score_tensorflow'] - pivot['f1_score_sklearn']
    pivot['better_model'] = pivot['accuracy_diff'].apply(
        lambda d: 'TensorFlow' if d > 0 else ('scikit-learn' if d < 0 else 'Equal')
    )

    table = pd.DataFrame()
    table['Channels'] = pivot['channel_no'].astype(int)
    table['Window Size'] = pivot['segment_dim'].astype(int)
    table['TF Accuracy'] = pivot['accuracy_tensorflow'].round(4)
    table['SKL Accuracy'] = pivot['accuracy_sklearn'].round(4)
    table['Diff (TF-SKL)'] = pivot['accuracy_diff'].round(6)
    table['TF F1'] = pivot['f1_score_tensorflow'].round(4)
    table['SKL F1'] = pivot['f1_score_sklearn'].round(4)
    table['F1 Diff'] = pivot['f1_score_diff'].round(6)
    table['Better Model'] = pivot['better_model']

    table['abs_diff'] = table['Diff (TF-SKL)'].abs()
    table = table.sort_values(['Channels', 'abs_diff'], ascending=[True, False]).drop('abs_diff', axis=1)
    return table


def print_model_comparison(table: pd.DataFrame) -> None:
    print("\nMODEL COMPARISON: TENSORFLOW vs SCIKIT-LEARN")
    print("=" * 80)
    print(tabulate(table, headers='keys', tablefmt='pretty', showindex=False))

    tf_wins = (table['Diff (TF-SKL)'] > 0).sum()
    skl_wins = (table['Diff (TF-SKL)'] < 0).sum()
    ties = (table['Diff (TF-SKL)'] == 0).sum()
    total = len(table)

    print("\nSUMMARY STATISTICS:")
    print(f"Total configurations compared: {total}")
    print(f"TensorFlow better: {tf_wins} ({tf_wins / total:.1%})")
    print(f"scikit-learn better: {skl_wins} ({skl_wins / total:.1%})")
    print(f"Equal performance: {ties} ({ties / total:.1%})")
    print(f"Average TensorFlow advantage: {table['Diff (TF-SKL)'].mean():.6f}")


def plot_accuracy_by_segment_size(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(12, 8))
    channel_numbers = sorted(df['channel_no'].unique())
    color_base = {4: '#fcae91', 5: '#fb6a4a', 6: '#de2d26', 7: '#a50f15', 8: '#67000d'}
    line_styles = {'sklearn': 'dashed', 'tensorflow': 'solid'}

    for ch in channel_numbers:
        for model in df['model_type'].unique():
            subset = df[(df['channel_no'] == ch) & (df['model_type'] == model)].sort_values('segment_dim')
            if subset.empty:
                continue
            plt.plot(
                subset['segment_dim'], subset['accuracy'],
                marker='o' if model == 'tensorflow' else 's', markersize=9,
                linestyle=line_styles[model], linewidth=3.0,
                color=color_base.get(ch, '#333333'), label=f"{ch} channels - {model}"
            )

    plt.xticks(sorted(df['segment_dim'].unique()))
    plt.xlabel('Window Size', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=14)
    plt.title('Classification Accuracy by Window Size', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_segment_size.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_channel_impact(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(12, 8))
    segment_sizes = sorted(df['segment_dim'].unique())
    colors = ['#D9D9D9', '#A0A0A0', '#606060', '#000000']
    line_styles = {'sklearn': 'dashed', 'tensorflow': 'solid'}

    for i, seg_size in enumerate(segment_sizes):
        for model in df['model_type'].unique():
            subset = df[(df['segment_dim'] == seg_size) & (df['model_type'] == model)].sort_values('channel_no')
            if subset.empty:
                continue
            plt.plot(
                subset['channel_no'], subset['accuracy'],
                marker='o' if model == 'tensorflow' else 's', markersize=9,
                linestyle=line_styles[model], linewidth=3.0,
                color=colors[i % len(colors)], label=f"Seg{seg_size} - {model}"
            )

    plt.xticks(sorted(df['channel_no'].unique()))
    plt.xlabel('Number of Channels', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=14)
    plt.title('Impact of Channel Count on Classification Accuracy', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_impact.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot_table(index=['channel_no', 'segment_dim'], columns='model_type', values='accuracy').reset_index()
    if 'sklearn' not in pivot.columns or 'tensorflow' not in pivot.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    min_val = min(pivot['sklearn'].min(), pivot['tensorflow'].min()) - 0.01
    max_val = max(pivot['sklearn'].max(), pivot['tensorflow'].max()) + 0.01

    scatter = ax.scatter(pivot['sklearn'], pivot['tensorflow'], c=pivot['segment_dim'],
                          cmap='viridis', s=90, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('sklearn Accuracy', fontweight='bold')
    ax.set_ylabel('tensorflow Accuracy', fontweight='bold')
    ax.set_title('sklearn vs tensorflow Model Performance', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.colorbar(scatter, label='Segment Dimension')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    for model_type in df['model_type'].unique():
        heatmap_data = df[df['model_type'] == model_type].pivot_table(
            index='channel_no', columns='segment_dim', values='accuracy'
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0.5, vmax=1.0)
        plt.title(f'Accuracy Heatmap - {model_type.capitalize()} Model')
        plt.xlabel('Segment Size')
        plt.ylabel('Number of Channels')
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{model_type}.png', dpi=300)
        plt.close()


def generate_best_configurations(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    best_configs = {}
    for metric in metrics:
        for ch in df['channel_no'].unique():
            ch_df = df[df['channel_no'] == ch]
            best_row = ch_df.loc[ch_df[metric].idxmax()]
            best_configs[f"best_{metric}_ch{int(ch)}"] = {
                'experiment_name': best_row['experiment_name'],
                'segment_dim': best_row['segment_dim'],
                'increment_dim': best_row['increment_dim'],
                'model_type': best_row['model_type'],
                'value': best_row[metric]
            }
    best_df = pd.DataFrame.from_dict(best_configs, orient='index')
    best_df.to_csv(output_dir / 'best_configurations.csv')
    return best_df


def df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    headers = ([''] if index else []) + df.columns.tolist()
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for i, row in df.iterrows():
        values = ([i] if index else []) + row.tolist()
        lines.append('| ' + ' | '.join(str(v) for v in values) + ' |')
    return '\n'.join(lines)


def generate_markdown_report(df: pd.DataFrame, summary_table: pd.DataFrame,
                              best_configs: pd.DataFrame, output_dir: Path) -> Path:
    best = df.loc[df['accuracy'].idxmax()]
    model_perf = df.groupby('model_type')['accuracy'].mean().to_dict()
    channel_perf = df.groupby('channel_no')['accuracy'].mean().to_dict()
    segment_perf = df.groupby('segment_dim')['accuracy'].mean().to_dict()

    report_path = output_dir / 'experiment_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write("# EMG Gesture Classification - Experiment Analysis Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total experiments: {len(df)}\n")
        f.write(f"- Best accuracy: **{best['accuracy']:.4f}** "
                f"({best['experiment_name']}, {best['channel_no']} channels, "
                f"segment {best['segment_dim']}, {best['model_type']})\n\n")

        f.write("### Performance by model type\n\n")
        for model, acc in model_perf.items():
            f.write(f"- {model}: {acc:.4f}\n")

        f.write("\n### Performance by channel count\n\n")
        for ch, acc in sorted(channel_perf.items()):
            f.write(f"- {int(ch)} channels: {acc:.4f}\n")

        f.write("\n### Performance by segment size\n\n")
        for seg, acc in sorted(segment_perf.items()):
            f.write(f"- segment {int(seg)}: {acc:.4f}\n")

        f.write("\n## Summary table\n\n")
        f.write(df_to_markdown(summary_table, index=False))

        f.write("\n\n## Best configurations\n\n")
        f.write(df_to_markdown(best_configs))

        f.write("\n\n## Plots\n\n")
        for fname in ['accuracy_by_segment_size.png', 'channel_impact.png', 'model_comparison.png']:
            f.write(f"![{fname}](./{fname})\n\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze run_experiment.py sweep results")
    parser.add_argument('--results-file', type=Path, default=None,
                         help="Path to a results CSV (default: most recent results/experiment_results_full_*.csv)")
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR / 'analysis',
                         help="Where to write tables/plots (default: results/analysis)")
    parser.add_argument('--report', action='store_true',
                         help="Also generate a markdown summary report")
    args = parser.parse_args()

    results_file = args.results_file or latest_results_file()
    print(f"Loading results from {results_file}")
    df = preprocess_data(pd.read_csv(results_file))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating summary table...")
    summary_table = generate_summary_table(df, args.output_dir)
    print(summary_table.head())

    print_model_comparison(model_comparison_table(df))
    model_comparison_table(df).to_csv(args.output_dir / 'model_comparison_table.csv', index=False)

    print("Creating visualizations...")
    plot_accuracy_by_segment_size(df, args.output_dir)
    plot_channel_impact(df, args.output_dir)
    plot_model_comparison_scatter(df, args.output_dir)
    plot_heatmaps(df, args.output_dir)

    print("Finding best configurations...")
    best_configs = generate_best_configurations(df, args.output_dir)

    if args.report:
        report_path = generate_markdown_report(df, summary_table, best_configs, args.output_dir)
        print(f"Markdown report written to {report_path}")

    print(f"\nAnalysis complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
