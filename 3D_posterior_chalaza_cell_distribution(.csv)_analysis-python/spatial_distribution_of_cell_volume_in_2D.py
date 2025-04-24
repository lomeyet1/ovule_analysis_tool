#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: Ziqiang Luo date: 2025-01-22
# Description:
# Process x (Anterior-Posterior axis),y (Proximal-Distal axis),volume csv files under each sample folder from MGX.

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import subprocess
# List of required packages with their import names and pip names
REQUIRED_PACKAGES = [
    {"import_name": "numpy", "pip_name": "numpy"},
    {"import_name": "pandas", "pip_name": "pandas"},
    {"import_name": "scipy", "pip_name": "scipy"},
    {"import_name": "matplotlib", "pip_name": "matplotlib"}
]

def check_and_install_packages():
    """
    Check if required packages are installed and install missing ones.

    This function verifies the presence of all required packages listed in REQUIRED_PACKAGES.
    If any package is missing, it attempts to install it using pip.
    """
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package["import_name"])
        except ImportError:
            print(f"Package '{package['import_name']}' is not installed. Installing now...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package["pip_name"]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"Successfully installed {package['pip_name']}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package['pip_name']}. Please install manually.")
                sys.exit(1)

check_and_install_packages()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process x (Anterior-Posterior axis),y (Proximal-Distal axis),volume csv files under each sample folder from MGX.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing sample folders.")
    parser.add_argument("--output_folder", default=None,
                        help="Path to the output folder. Default is results_<input_folder>. Do not create a new folder in the input folder.")
    parser.add_argument("--csv_labels", default=None,
                        help="Custom labels to find csv files of x (AP, Anterior-Posterior axis),y (PD, Proximal-Distal axis) and volume. Default is using --csv_labels x,y,volume")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins for both x_norm and y_norm. Default is 5.")
    parser.add_argument("--sample_plots", type=bool, default=None, help="put yes to generate sample plots")
    return parser.parse_args()


def get_file_mapping(input_folder, custom_labels=None):
    default_labels = ["x", "y", "volume"]
    labels = custom_labels.split(',') if custom_labels else default_labels
    mapping = {label: [] for label in labels}

    for file in glob.glob(os.path.join(input_folder, "*.csv")):
        filename = os.path.basename(file)
        for label in labels:
            if label in filename:
                mapping[label].append(file)
                break

    return mapping


def process_files(sample_name, files, labels):
    data = {}
    for label, file_list in files.items():
        if not file_list:
            continue
        df = pd.read_csv(file_list[0], usecols=[0, 1], names=["cell_id", label], header=0)
        df[label] = pd.to_numeric(df[label], errors='coerce')
        if df[label].isnull().any():
            continue
        data[label] = df.set_index("cell_id")

    merged = pd.concat(data.values(), axis=1, join='inner').reset_index()
    merged.insert(0, "sample_name", sample_name)

    # Normalize values

    user_labels = list(files.keys())
    merged[f"{labels[0]}_norm"] = merged[user_labels[0]] / merged[user_labels[0]].max()
    merged[f"{labels[1]}_norm"] = merged[user_labels[1]] / merged[user_labels[1]].max()
    merged[f"{labels[2]}_rescale"] = merged[user_labels[2]] / merged[user_labels[2]].mean()  # rescale volume to this sample mean

    return merged



# def plot_combined(sample_name, df, bins, output_folder, labels, input_name, use_pool_rescale=False):
#     # Create figure and gridspec
#     fig = plt.figure(figsize=(8, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=(7, 5), height_ratios=(7, 5),
#                           wspace=0.3, hspace=0.2)
#
#     # Scatter plot
#     ax_scatter = fig.add_subplot(gs[0, 0])
#     x_bin_edges = np.linspace(0, 1, bins + 1)
#     y_bin_edges = np.linspace(0, 1, bins + 1)
#
#     volume_col = f"{labels[2]}_rescale" if use_pool_rescale == False else f"{labels[2]}_pool_rescale"
#
#     scatter = ax_scatter.scatter(df[f"{labels[0]}_norm"], df[f"{labels[1]}_norm"], c=df[volume_col],
#                                  cmap='coolwarm', s=50)
#
#     ax_scatter.set_xticks(x_bin_edges)
#     ax_scatter.set_yticks(y_bin_edges)
#
#     ax_scatter.set_xticklabels([f'{x:.2f}' for x in x_bin_edges])
#     ax_scatter.set_yticklabels([f'{y:.2f}' for y in y_bin_edges])
#
#     ax_scatter.invert_xaxis()
#     ax_scatter.yaxis.tick_right()
#     ax_scatter.yaxis.set_label_position("right")
#
#
#
#     # x_norm scatter plot with error bars and significance (inverted)
#     ax_scatterx = fig.add_subplot(gs[1, 0], sharex=ax_scatter)
#
#     # Create bins for x_norm
#     x_bin_edges = np.linspace(0, 1, bins + 1)
#     df['x_bin'] = pd.cut(df[f"{labels[0]}_norm"], bins=x_bin_edges, include_lowest=True)
#     df['x_bin_center'] = df['x_bin'].apply(lambda x: x.mid)
#
#     # Calculate sum of volume_col for each sample in each x_bin
#     scatter_data = df.groupby(['sample_name', 'x_bin', 'x_bin_center'], observed = True)[volume_col].sum().reset_index()
#
#     # Calculate mean and CI for each x_bin
#     bin_stats = scatter_data.groupby('x_bin_center', observed = True)[volume_col].agg(['mean', 'std', 'count']).reset_index()
#     bin_stats['ci'] = 1.96 * bin_stats['std'] / np.sqrt(bin_stats['count'])
#
#     # Plot scatter points (all gray)
#     ax_scatterx.scatter(scatter_data['x_bin_center'], scatter_data[volume_col],
#                         color='gray', alpha=0.7, s=50)
#
#     # Add error bars
#     ax_scatterx.errorbar(bin_stats['x_bin_center'], bin_stats['mean'],
#                          yerr=bin_stats['ci'], fmt='o', color='black',
#                          capsize=5, markersize=8, linewidth=2)
#
#     # Perform pairwise t-tests between x_bins
#     ttest_results = []
#     x_bins = sorted(scatter_data['x_bin_center'].unique())
#
#     for i in range(len(x_bins)):
#         for j in range(i + 1, len(x_bins)):
#             bin1_data = scatter_data[scatter_data['x_bin_center'] == x_bins[i]][volume_col]
#             bin2_data = scatter_data[scatter_data['x_bin_center'] == x_bins[j]][volume_col]
#
#             if len(bin1_data) > 1 and len(bin2_data) > 1:  # Need at least 2 samples for t-test
#                 t_stat, p_value = stats.ttest_ind(bin1_data, bin2_data)
#                 ttest_results.append({
#                     'bin1': x_bins[i],
#                     'bin2': x_bins[j],
#                     'p_value': p_value,
#                     'bin1_mean': bin1_data.mean(),
#                     'bin2_mean': bin2_data.mean()
#                 })
#
#     # Add significance markers
#     for test in ttest_results:
#         if test['p_value'] < 0.05:  # Significant at p<0.05
#             # Draw line between bins
#             ax_scatterx.plot([test['bin1'], test['bin2']],
#                              [max(test['bin1_mean'], test['bin2_mean']) * 1.1] * 2,
#                              color='black', linewidth=1)
#
#             # Add star at midpoint
#             mid_x = (test['bin1'] + test['bin2']) / 2
#             ax_scatterx.text(mid_x, max(test['bin1_mean'], test['bin2_mean']) * 1.15,
#                              '*', ha='center', va='center',
#                              fontsize=12, color='red', weight='bold')
#
#     ax_scatterx.yaxis.tick_right()
#     ax_scatterx.yaxis.set_label_position('right')
#     ax_scatterx.xaxis.set_visible(False)
#     ax_scatterx.invert_yaxis()
#     ax_scatterx.set_ylabel(f'Sum of rescaled {labels[2]} by sample \n in normalized AD bin')
#
#
#     # y_norm histogram (mirrored)
#     ax_histy = fig.add_subplot(gs[0, 1], sharey=ax_scatter)
#     ax_histy.hist(df[f"{labels[1]}_norm"], bins=bins, weights=df[volume_col], alpha=0.7, orientation='horizontal')
#     ax_histy.yaxis.set_visible(False)
#     ax_histy.set_xlabel(f'Sum of rescaled {labels[2]} of \n cells in normalized PD bin')
#
#     # Add the colorbar to the inset axis
#     cax = fig.add_axes([0.65, 0.3, 0.25, 0.03])
#     cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
#     cbar.set_label(f'Rescaled cell {labels[2]}')
#
#     # Add text
#     fig.text(0.2, 0.41, 'Anterior-Posterior axis', fontsize=12)  # X-axis
#     fig.text(0.59, 0.6, 'Proximal-Distal axis', fontsize=12, rotation=90)   # Y-axis
#     # fig.text(0.2, 0.9, 'title', fontsize=12)  # title
#
#     fig.text(0.65, 0.12, f'Sample: {sample_name} \n Rescaled cell {labels[2]}  \n distribution  \n in normalized AD-PD space',
#              fontsize=12)   # note
#
#     # Save the figure
#     plt.savefig(os.path.join(output_folder, f"{sample_name}_bin={bins}_combined_plot_{input_name}.png"))
#     plt.close()


def create_scatter_plot_with_stats(ax, df, bin_var, volume_col, bins, orientation='vertical'):
    """
    Create scatter plot with error bars and significance markers

    Parameters:
        ax: matplotlib axis object
        df: DataFrame containing the data
        bin_var: variable to bin by (e.g., "x_norm" or "y_norm")
        volume_col: column containing volume data
        bins: number of bins
        orientation: 'vertical' or 'horizontal'
    """
    # Create bins
    bin_edges = np.linspace(0, 1, bins + 1)
    df['bin'] = pd.cut(df[bin_var], bins=bin_edges, include_lowest=True)
    df['bin_center'] = df['bin'].apply(lambda x: x.mid)

    # Calculate sum of volume_col for each sample in each bin
    scatter_data = df.groupby(['sample_name', 'bin', 'bin_center'], observed=True)[volume_col].sum().reset_index()

    # Calculate mean and CI for each bin
    bin_stats = scatter_data.groupby('bin_center', observed=True)[volume_col].agg(
        ['mean', 'std', 'count']).reset_index()
    bin_stats['ci'] = 1.96 * bin_stats['std'] / np.sqrt(bin_stats['count'])

    # Plot scatter points (all gray)
    if orientation == 'vertical':
        ax.scatter(scatter_data['bin_center'], scatter_data[volume_col],
                   color='gray', alpha=0.7, s=50)
    else:
        ax.scatter(scatter_data[volume_col], scatter_data['bin_center'],
                   color='gray', alpha=0.7, s=50)

    # Add error bars
    if orientation == 'vertical':
        ax.errorbar(bin_stats['bin_center'], bin_stats['mean'],
                    yerr=bin_stats['ci'], fmt='o', color='black',
                    capsize=5, markersize=8, linewidth=2)
    else:
        ax.errorbar(bin_stats['mean'], bin_stats['bin_center'],
                    xerr=bin_stats['ci'], fmt='o', color='black',
                    capsize=5, markersize=8, linewidth=2)

    # Perform pairwise t-tests between bins
    ttest_results = []
    bin_centers = sorted(scatter_data['bin_center'].unique())

    for i in range(len(bin_centers)):
        for j in range(i + 1, len(bin_centers)):
            bin1_data = scatter_data[scatter_data['bin_center'] == bin_centers[i]][volume_col]
            bin2_data = scatter_data[scatter_data['bin_center'] == bin_centers[j]][volume_col]

            if len(bin1_data) > 1 and len(bin2_data) > 1:  # Need at least 2 samples for t-test
                t_stat, p_value = stats.ttest_ind(bin1_data, bin2_data)
                ttest_results.append({
                    'bin1': bin_centers[i],
                    'bin2': bin_centers[j],
                    'p_value': p_value,
                    'bin1_mean': bin1_data.mean(),
                    'bin2_mean': bin2_data.mean()
                })

    # Add significance markers
    for test in ttest_results:
        if test['p_value'] < 0.05:  # Significant at p<0.05
            # Determine which bin has higher mean (A > B)
            if test['bin1_mean'] > test['bin2_mean']:
                higher_bin, lower_bin = test['bin1'], test['bin2']
                higher_mean, lower_mean = test['bin1_mean'], test['bin2_mean']
            else:
                higher_bin, lower_bin = test['bin2'], test['bin1']
                higher_mean, lower_mean = test['bin2_mean'], test['bin1_mean']

            # Position line slightly above higher mean
            line_pos = higher_mean * 1.05

            if orientation == 'vertical':
                # Draw horizontal line between bins
                ax.plot([higher_bin, lower_bin], [line_pos, line_pos],
                        color='black', linewidth=1)
                # Add star at lower bin's x position but higher bin's y position
                ax.text(lower_bin, line_pos, '*', ha='center', va='bottom',
                        fontsize=12, color='red', weight='bold')
            else:
                # Draw vertical line between bins
                ax.plot([line_pos, line_pos], [higher_bin, lower_bin],
                        color='black', linewidth=1)
                # Add star at lower bin's y position but higher bin's x position
                ax.text(line_pos, lower_bin, '*', ha='left', va='center',
                        fontsize=12, color='red', weight='bold')

    return scatter_data, bin_stats, ttest_results


def plot_combined(sample_name, df, bins, output_folder, labels, input_name, use_pool_rescale=False):
    # Create figure and gridspec
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 5), height_ratios=(7, 5),
                          wspace=0.3, hspace=0.2)

    # Scatter plot
    ax_scatter = fig.add_subplot(gs[0, 0])
    x_bin_edges = np.linspace(0, 1, bins + 1)
    y_bin_edges = np.linspace(0, 1, bins + 1)

    volume_col = f"{labels[2]}_rescale" if not use_pool_rescale else f"{labels[2]}_pool_rescale"

    scatter = ax_scatter.scatter(df[f"{labels[0]}_norm"], df[f"{labels[1]}_norm"], c=df[volume_col],
                                 cmap='coolwarm', s=50)

    ax_scatter.set_xticks(x_bin_edges)
    ax_scatter.set_yticks(y_bin_edges)

    ax_scatter.set_xticklabels([f'{x:.2f}' for x in x_bin_edges])
    ax_scatter.set_yticklabels([f'{y:.2f}' for y in y_bin_edges])

    ax_scatter.invert_xaxis()
    ax_scatter.yaxis.tick_right()
    ax_scatter.yaxis.set_label_position("right")

    # x_norm scatter plot (inverted)
    ax_scatterx = fig.add_subplot(gs[1, 0], sharex=ax_scatter)
    x_scatter_data, x_bin_stats, x_ttest_results = create_scatter_plot_with_stats(
        ax_scatterx, df, f"{labels[0]}_norm", volume_col, bins, orientation='vertical'
    )
    ax_scatterx.yaxis.tick_right()
    ax_scatterx.yaxis.set_label_position('right')
    ax_scatterx.xaxis.set_visible(False)
    ax_scatterx.invert_yaxis()
    ax_scatterx.set_ylabel(f'Sum of rescaled {labels[2]} by sample \n in normalized AD bin')

    # y_norm scatter plot (mirrored)
    ax_scattery = fig.add_subplot(gs[0, 1], sharey=ax_scatter)
    y_scatter_data, y_bin_stats, y_ttest_results = create_scatter_plot_with_stats(
        ax_scattery, df, f"{labels[1]}_norm", volume_col, bins, orientation='horizontal'
    )
    ax_scattery.yaxis.set_visible(False)
    ax_scattery.set_xlabel(f'Sum of rescaled {labels[2]} by sample \n in normalized PD bin')

    # Add the colorbar to the inset axis
    cax = fig.add_axes([0.65, 0.3, 0.25, 0.03])
    cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
    cbar.set_label(f'Rescaled cell {labels[2]}')

    # Add text
    fig.text(0.2, 0.41, 'Anterior-Posterior axis', fontsize=12)  # X-axis
    fig.text(0.59, 0.6, 'Proximal-Distal axis', fontsize=12, rotation=90)  # Y-axis

    fig.text(0.65, 0.12,
             f'Sample: {sample_name} \n Rescaled cell {labels[2]} distribution \n in normalized AD-PD space',
             fontsize=12)  # note

    # Save the figure
    plt.savefig(os.path.join(output_folder, f"{sample_name}_bin={bins}_combined_plot_{input_name}.png"))
    plt.close()

    # Save statistics data
    if use_pool_rescale:
        prefix = "pooled_"
    else:
        prefix = ""

    x_scatter_data.to_csv(os.path.join(output_folder, f"{prefix}x_bin_scatter_data.csv"), index=False)
    x_bin_stats.to_csv(os.path.join(output_folder, f"{prefix}x_bin_stats.csv"), index=False)
    pd.DataFrame(x_ttest_results).to_csv(os.path.join(output_folder, f"{prefix}x_bin_ttest_results.csv"), index=False)

    y_scatter_data.to_csv(os.path.join(output_folder, f"{prefix}y_bin_scatter_data.csv"), index=False)
    y_bin_stats.to_csv(os.path.join(output_folder, f"{prefix}y_bin_stats.csv"), index=False)
    pd.DataFrame(y_ttest_results).to_csv(os.path.join(output_folder, f"{prefix}y_bin_ttest_results.csv"), index=False)


def main():
    args = parse_arguments()

    input_folder = args.input_folder
    custom_labels = args.csv_labels
    labels = custom_labels.split(',') if custom_labels else ["x", "y", "volume"]
    bins = args.bins
    sample_plots = args.sample_plots

    output_folder = args.output_folder or os.path.join(os.path.dirname(input_folder), f"results_{os.path.basename(input_folder)}")
    os.makedirs(output_folder, exist_ok=True)

    all_samples = []
    sample_numbers = 0
    for subfolder in os.listdir(input_folder):
        sample_path = os.path.join(input_folder, subfolder)
        if not os.path.isdir(sample_path):
            continue

        files_mapping = get_file_mapping(sample_path, custom_labels)

        sample_data = process_files(subfolder, files_mapping, labels)

        all_samples.append(sample_data)
        sample_numbers += 1
        # Generate sample_plots plots if the parameter is set
        if sample_plots:
            plot_combined(subfolder, sample_data, bins, output_folder, labels, os.path.basename(input_folder))

            # plot_bubble(subfolder, sample_data, bins, output_folder)

    # Create combined plots for all samples
    combined_df = pd.concat(all_samples)
    combined_df[f"{labels[2]}_pool_rescale"] = combined_df[labels[2]] / combined_df[labels[2]].mean()

    plot_combined(f"all {sample_numbers} samples", combined_df, bins, output_folder, labels, os.path.basename(input_folder), use_pool_rescale=True)

    combined_df.to_csv(os.path.join(output_folder, f"all_samples_from_{os.path.basename(input_folder)}.csv"), index=False)

    print("Processing complete. Results saved in " + output_folder)


if __name__ == "__main__":
    main()
