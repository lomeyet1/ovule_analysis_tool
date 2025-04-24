#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author: Ziqiang Luo date: 2025-01-13
# usage: python sum_plot.py <folder_path> [separate]
# Description:
# This script processes CSV files in a specified directory, extracting data from subfolders that contain files ending in
# post.csv and ant.csv. It computes the sum of values from the second column (assumed to be named 'Value') for each
# file type, calculates the ratio of the sums (i.e. post / ant), and stores the results in CSV files. Additionally,
# it generates scatter plots showing the ratio for each subfolder and computes the number of cells for each sample.

import os
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

import subprocess
# List of required packages with their import names and pip names
REQUIRED_PACKAGES = [
    {"import_name": "numpy", "pip_name": "numpy"},
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



def process_sample_folder(sample_folder_path):
    sum_post = 0
    sum_ant = 0
    post_count = 0
    ant_count = 0
    has_valid_files = False

    # Process 'post.csv' files
    for file_name in os.listdir(sample_folder_path):
        if file_name.endswith('post.csv'):
            post_csv_path = os.path.join(sample_folder_path, file_name)
            if os.path.isfile(post_csv_path):
                has_valid_files = True
                with open(post_csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
                    post_count += len(rows)
                    sum_post += sum(float(row['Value']) for row in rows)

    # Process 'ant.csv' files
    for file_name in os.listdir(sample_folder_path):
        if file_name.endswith('ant.csv'):
            ant_csv_path = os.path.join(sample_folder_path, file_name)
            if os.path.isfile(ant_csv_path):
                has_valid_files = True
                with open(ant_csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
                    ant_count += len(rows)
                    sum_ant += sum(float(row['Value']) for row in rows)



    return sum_post, sum_ant, post_count, ant_count, has_valid_files

def process_folder(base_path):
    result = []
    stage_sample_counts = {}

    # Iterate through subfolders in the specified directory
    for subfolder in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_results = []
            sample_count = 0

            # Iterate through sample folders
            for sample_folder in os.listdir(subfolder_path):
                sample_folder_path = os.path.join(subfolder_path, sample_folder)
                if os.path.isdir(sample_folder_path):
                    sum_post, sum_ant, post_count, ant_count, has_valid_files = process_sample_folder(sample_folder_path)

                    if has_valid_files:
                        ratio_vol = sum_post / sum_ant if sum_ant != 0 else 0
                        ratio_no = post_count / ant_count if ant_count != 0 else 0
                        ratio_vol_no = ratio_vol / ratio_no if ratio_no != 0 else 0
                        subfolder_results.append([sample_folder, sum_post, sum_ant, post_count, ant_count, ratio_vol, ratio_no, ratio_vol_no])
                        sample_count += 1
                    else:
                        print(f"Sample '{sample_folder}' in stage '{subfolder}' contains neither post.csv nor ant.csv.")

            if sample_count:
                result.append((subfolder, subfolder_results))
                stage_sample_counts[subfolder] = sample_count

    for stage, count in stage_sample_counts.items():
        print(f"Stage '{stage}' has {count} valid samples.")

    return result, stage_sample_counts

def plot_results(output_folder, result, stage_sample_counts):
    stage_names = []
    all_ratios = []
    avg_ratios = {}
    errors = {}

    confidence_level = 1.96  # For a 95% confidence interval

    # Collect ratios for plotting
    for subfolder, subfolder_results in result:
        stage_ratios = [row[5] for row in subfolder_results]  # Use 'Ratio-vol' for plotting
        if stage_ratios:
            avg_ratio = np.mean(stage_ratios)
            std_dev = np.std(stage_ratios, ddof=1)  # Sample standard deviation
            sample_size = len(stage_ratios)
            error_margin = confidence_level * (std_dev / np.sqrt(sample_size))
        else:
            avg_ratio = 0
            error_margin = 0

        stage_names.extend([subfolder] * len(stage_ratios))
        all_ratios.extend(stage_ratios)
        avg_ratios[subfolder] = avg_ratio
        errors[subfolder] = error_margin

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    x_positions = []
    unique_stages = list(avg_ratios.keys())
    for i, stage in enumerate(unique_stages):
        num_samples = stage_sample_counts[stage]

        # Create x positions with slight offsets to prevent overlap
        x_positions.extend([i + 1 + (j - num_samples / 2) * 0.01 for j in range(num_samples)])

    plt.scatter(x_positions, all_ratios, color='blue', marker='o', label='Sample Ratios')

    # Plot average ratio lines with error bars for each stage
    x_labels = []
    for i, stage in enumerate(unique_stages):
        avg_ratio = avg_ratios[stage]
        error_margin = errors[stage]
        sample_size = stage_sample_counts[stage]
        x_label = f"{stage} (n={sample_size})"
        x_labels.append(x_label)

        # Plot average line with limited length
        plt.hlines(avg_ratio, xmin=i + 0.9, xmax=i + 1.1, colors='red', linestyles='--')
        plt.errorbar(i + 1, avg_ratio, yerr=error_margin, fmt='o', color='red', capsize=5)

    plt.title('Differential growth in nucellus')
    plt.xlabel('Stage')
    plt.ylabel('Ratio (posteriors/anterior)')
    plt.xticks(ticks=np.arange(1, len(unique_stages) + 1), labels=x_labels, rotation=45, ha='right')

    # Adjust y-axis ticks
    plt.yticks(np.arange(0.5, 2.5, 0.5))

    plt.tight_layout()

    # Save plot as an image
    plot_filename = os.path.join(output_folder, 'Differential growth in nucellus.png')
    plt.savefig(plot_filename)
    plt.close()

def write_result(output_folder, result, separate):
    if separate:
        for subfolder, subfolder_results in result:
            output_file = os.path.join(output_folder, f"{subfolder}.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['sample', 'sum-post-vol', 'sum-ant-vol', 'post-cell-No.', 'ant-cell-No.', 'Ratio-vol', 'Ratio-No.', 'Ratio-vol/Ratio-No.'])
                writer.writerows(subfolder_results)
    else:
        output_file = os.path.join(output_folder, 'all_stages.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            for subfolder, subfolder_results in result:
                writer.writerow([subfolder])
                writer.writerow(['sample', 'sum-post-vol', 'sum-ant-vol', 'post-cell-No.', 'ant-cell-No.', 'Ratio-vol', 'Ratio-No.', 'Ratio-vol/Ratio-No.'])
                writer.writerows(subfolder_results)
                writer.writerow([])  # Blank line between subfolder sections

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <folder_path> [separate]")
        #
        sys.exit(1)

    folder_path = sys.argv[1]
    separate = len(sys.argv) == 3 and sys.argv[2].lower() == 'separate'

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    result, stage_sample_counts = process_folder(folder_path)

    output_folder = os.path.join(folder_path, '3D_nucellus_differential_growth')
    os.makedirs(output_folder, exist_ok=True)

    write_result(output_folder, result, separate)
    plot_results(output_folder, result, stage_sample_counts)  # Generate scatter plot
    print("\t Processing complete. Results saved and plot generated.")
