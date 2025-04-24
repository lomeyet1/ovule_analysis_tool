#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes cell measurement data from MorphoGraphX area, volume heatmap and parent labels,
the parent_label_to_name.txt gives those numbers means what structure of ovule (can copy and change),
aggregating cell statistics (area, volume, count), and generating summary reports.

author: Ziqiang Luo
date: 2025-04-13

Key features:
- Validates required Python packages and installs missing ones
- Relates parent numbers to what structure of ovule
- Filters cells by minimum volume threshold
-Filters out specified samples when counting some incomplete tissues
- Generates comprehensive statistics and exports to CSV
"""

from __future__ import print_function, division
import os
import argparse
import csv
import sys
import subprocess
import pandas as pd


# Check and install required packages
REQUIRED_PACKAGES = ['pandas', 'matplotlib']


def install_packages():
    """Install missing required packages using pip."""
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_packages()


def parent_labels(mapping_file):
    """
    Check for missing parent labels in parent.csv files and validate data integrity.

    Args:
        mapping_file (str): Path to parent label to name mapping file
    """

    # Validate mapping file existence
    mapping_file_path = mapping_file
    if not os.path.exists(mapping_file_path):
        print(f"Error: Mapping file '{mapping_file}' not found in the input folder.")
        return

    # Read the label mapping table (label, name, abbreviation)
    label_to_name = {}
    label_to_abbreviation = {}
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 3:
                    label, name, abbreviation = row[:3]
                    label_to_name[str(label)] = name.strip()
                    label_to_abbreviation[str(label)] = abbreviation.strip()
    except UnicodeDecodeError as e:
        print(f"Error reading mapping file: {e}")
        return
    except Exception as e:
        print(f"Error processing mapping file: {e}")
        return

    return label_to_name, label_to_abbreviation



def aggregate_data(input_folder, label_to_name, label_to_abbreviation, minimum_Volume,filter_abbreviation_IDs):
    """
    Aggregate cell measurement data (Area, Volume, Count) by parent labels.

    Args:
        input_folder (str): Input data directory path
        label_to_name (dict): Mapping of labels to tissue names
        label_to_abbreviation (dict): Mapping of labels to abbreviations
        minimum_Volume (float): Minimum cell volume threshold
        filter_abbreviation_IDs(dict): tissue abbreviation is key, sample ID is in value_list
    """
    output_folder = os.path.join(input_folder, "python_export_data")
    all_data = []  # For aggregated tissue-level data
    cell_data = []  # For per-cell data
    current_folder = os.path.basename(input_folder)
    parent_folder = os.path.basename(os.path.dirname(input_folder))

    source = f"{parent_folder}_{current_folder}"

    # Create output directory for aggregated results
    save_folder = os.path.join(input_folder, "python_export_data_aggregated")
    os.makedirs(save_folder, exist_ok=True)

    # Process each sample's data files
    for filename in os.listdir(output_folder):
        if filename.endswith("_parent.csv"):
            sample_name = os.path.splitext(filename)[0].replace("_parent", "")
            parent_csv_path = os.path.join(output_folder, filename)
            area_csv_path = os.path.join(output_folder, f"{sample_name}_Area.csv")
            volume_csv_path = os.path.join(output_folder, f"{sample_name}_Volume.csv")
            sample_name = sample_name.split('_Mesh')[0].strip().upper()  # Clean sample name

            copy_label_to_abbreviation = label_to_abbreviation.copy()
            copy_label_to_name = label_to_name.copy()

            if filter_abbreviation_IDs:
                matching_abbreviation = [key for key, value_list in filter_abbreviation_IDs.items() if sample_name in value_list]

                if matching_abbreviation:
                    for matching_abb in matching_abbreviation:
                        matching_key = [key for key, abbreviation in label_to_abbreviation.items() if matching_abb == abbreviation]
                        del copy_label_to_abbreviation[matching_key[0]]
                        del copy_label_to_name[matching_key[0]]

            # Read and merge data from all three files (parent, area, volume)
            parent_data = pd.read_csv(parent_csv_path, header=0, usecols=[0, 1], names=["Label", "parent_label"])
            area_data = pd.read_csv(area_csv_path, header=0, usecols=[0, 2], names=["Label", "Area"])
            volume_data = pd.read_csv(volume_csv_path, header=0, usecols=[0, 2], names=["Label", "Volume"])

            # Ensure consistent data types for merging
            for df in [parent_data, area_data, volume_data]:
                df["Label"] = df["Label"].astype(str)

            # Merge all data sources
            merged_data = parent_data.merge(area_data, on="Label", how="left")
            merged_data = merged_data.merge(volume_data, on="Label", how="left")

            # Apply volume threshold filter
            filtered_data = merged_data[merged_data["Volume"] >= minimum_Volume]

            # Prepare per-cell data for output
            sample_cell_data = pd.DataFrame({
                "sample_name": sample_name,
                "Abbreviation": filtered_data["parent_label"].astype(str).map(copy_label_to_abbreviation),
                "MGX Cell ID": filtered_data["Label"],
                "Volume": filtered_data["Volume"]
            })
            cell_data.append(sample_cell_data)

            # Aggregate statistics by parent label
            grouped_data = filtered_data.groupby("parent_label").agg({
                "Area": "sum",
                "Volume": "sum",
                "Label": "count"  # Cell count
            }).reset_index()
            grouped_data.columns = ["parent_label", "Area", "Volume", "cell number"]
            grouped_data["sample"] = sample_name

            # Map labels to names and abbreviations
            grouped_data["parent_label"] = grouped_data["parent_label"].astype(str)
            grouped_data["tissue name"] = grouped_data["parent_label"].map(copy_label_to_name)
            grouped_data["abbreviation"] = grouped_data["parent_label"].map(copy_label_to_abbreviation)

            all_data.append(grouped_data)

    # Save per-cell data files for each tissue type
    cell_data = pd.concat(cell_data, ignore_index=True)
    for abbreviation in label_to_abbreviation.values():
        cell_type_data = cell_data[cell_data["Abbreviation"] == abbreviation].pivot(
            columns='sample_name', values='Volume')
        cell_type_data.to_csv(
            os.path.join(save_folder, f'cell_of_{abbreviation}-minium_cell_volume={minimum_Volume}.csv'),
            index=False)

    # Combine all sample data
    all_data = pd.concat(all_data, ignore_index=True)
    column_order = ["sample", "tissue name", "abbreviation", "parent_label", "Area", "Volume", "cell number"]
    all_data = all_data[column_order]

    # Filter invalid or empty abbreviations
    valid_data = all_data.dropna(subset=["abbreviation"])
    valid_data = valid_data[valid_data["abbreviation"] != ""]

    # Calculate whole-ovule (total)
    total_data = valid_data.groupby('sample')[['Area', 'Volume', 'cell number']].sum().reset_index()
    total_data['tissue name'] = 'Ovule'
    total_data['abbreviation'] = 'Ovule'
    total_data['parent_label'] = ""

    for key, value_list in filter_abbreviation_IDs.items():
        for value in value_list:
            to_drop = total_data[(total_data["sample"] == value)].index
            total_data.drop(to_drop, inplace=True)

    all_data = pd.concat([all_data, total_data])

    # exclude fu
    valid_data = valid_data[valid_data["abbreviation"] != "fu"]

    # Calculate whole-ovule (total)
    total_data = valid_data.groupby('sample')[['Area', 'Volume', 'cell number']].sum().reset_index()
    total_data['tissue name'] = 'Ovule exclude funiculus'
    total_data['abbreviation'] = 'Ovule-fu'
    total_data['parent_label'] = ""
    all_data = pd.concat([all_data, total_data])


    # Add composite tissue data (sums of related tissues)
    # Outer integument (oi1 + oi2)
    oi_data = all_data[all_data['abbreviation'].isin(['oi1', 'oi2'])]
    sum_data = oi_data.groupby('sample')[['Area', 'Volume', 'cell number']].sum().reset_index()
    sum_data['tissue name'] = 'outer integument'
    sum_data['abbreviation'] = 'oi'
    sum_data['parent_label'] = ""
    all_data = pd.concat([all_data, sum_data])

    # Inner integument (ii1 + ii1` + ii2)
    ii_data = all_data[all_data['abbreviation'].isin(['ii1', 'ii2', "ii1`"])]
    sum_data = ii_data.groupby('sample')[['Area', 'Volume', 'cell number']].sum().reset_index()
    sum_data['tissue name'] = 'inner integument'
    sum_data['abbreviation'] = 'ii'
    sum_data['parent_label'] = ""
    all_data = pd.concat([all_data, sum_data])

    # Chalaza (pc + ac)
    ch_data = all_data[all_data['abbreviation'].isin(['pc', 'ac'])]
    sum_data = ch_data.groupby('sample')[['Area', 'Volume', 'cell number']].sum().reset_index()
    sum_data['tissue name'] = 'chalaza'
    sum_data['abbreviation'] = 'ch'
    sum_data['parent_label'] = ""
    all_data = pd.concat([all_data, sum_data])

    # Add metadata and calculated metrics
    all_data['Source'] = source
    all_data["Area_mean_per_cell (μm²)"] = all_data["Area"] / all_data["cell number"]
    all_data["Volume_mean_per_cell (μm³)"] = all_data["Volume"] / all_data["cell number"]

    # Final column ordering
    output_columns = [
        'Source', "sample", "tissue name", "abbreviation", "parent_label",
        "Area", "Volume", "cell number",
        "Area_mean_per_cell (μm²)", "Volume_mean_per_cell (μm³)"
    ]
    all_data = all_data[output_columns]
    all_data.rename(columns={'Volume': 'Volume (μm³)', 'Area': 'Area (μm²)'}, inplace=True)

    # Save aggregated data with volume threshold note
    output_csv = os.path.join(save_folder, f"aggregated_data-minium_cell_volume={minimum_Volume}.csv")
    with open(output_csv, 'w', encoding='utf-8') as file:
        file.write(
            f"# only cells with Volume >= {minimum_Volume} μm³ and with parent label recorded in the mapping_file (default is parent_label_to_name.txt) counted in \n")
    all_data.to_csv(output_csv, mode='a', index=False)

    print(f"Aggregated data saved in {parent_folder}/{current_folder}/python_export_data_aggregated")

    # Generate and save statistics
    statistics(all_data, save_folder, minimum_Volume)


def statistics(all_data, save_folder, minimum_Volume):
    """
    Calculate descriptive statistics and save formatted results.

    Args:
        all_data (DataFrame): Aggregated measurement data
        save_folder (str): Output directory path
        minimum_Volume (float): Minimum cell volume threshold used
    """
    # Calculate mean and standard deviation for all metrics
    stats_data = all_data.groupby("abbreviation").agg({
        "cell number": ["mean", "std"],
        "Area (μm²)": ["mean", "std"],
        "Area_mean_per_cell (μm²)": ["mean", "std"],
        "Volume (μm³)": ["mean", "std"],
        "Volume_mean_per_cell (μm³)": ["mean", "std"]
    }).reset_index()

    # Flatten multi-index columns
    stats_data.columns = [
        "abbreviation", "cell number_mean", "cell number_std",
        "Area_mean", "Area_std", "Area_mean_per_cell_mean", "Area_mean_per_cell_std",
        "Volume_mean", "Volume_std", "Volume_mean_per_cell_mean", "Volume_mean_per_cell_std"
    ]

    # add sample numer
    sample_counts = all_data.groupby('abbreviation')['sample'].nunique().reset_index()
    sample_counts.columns = ['abbreviation', 'N_samples']
    stats_data = pd.merge(stats_data, sample_counts, on='abbreviation', how='left')
    cols = stats_data.columns.tolist()
    cols = cols[:1] + ['N_samples'] + cols[1:-1]
    stats_data = stats_data[cols]

    # Create formatted output strings for key metrics
    stats_data["N cells"] = (
            stats_data["cell number_mean"].round(1).astype(str) + " ± " +
            stats_data["cell number_std"].round(1).astype(str)
    )

    stats_data["Cell volume (μm³)"] = (
            stats_data["Volume_mean_per_cell_mean"].round(1).astype(str) + " ± " +
            stats_data["Volume_mean_per_cell_std"].round(1).astype(str)
    )

    stats_data["Tissue volume (x10^4 μm³)"] = (
            (stats_data["Volume_mean"] / 1e4).round(2).astype(str) + " ± " +
            (stats_data["Volume_std"] / 1e4).round(1).astype(str)
    )

    stats_data["Cell Area (μm²)"] = (
            stats_data["Area_mean_per_cell_mean"].round(1).astype(str) + " ± " +
            stats_data["Area_mean_per_cell_std"].round(1).astype(str)
    )

    stats_data["Tissue Area (x10^4 μm²)"] = (
            (stats_data["Area_mean"] / 1e4).round(2).astype(str) + " ± " +
            (stats_data["Area_std"] / 1e4).round(1).astype(str)
    )

    # Save statistics to CSV
    stats_csv = os.path.join(save_folder, f"aggregated_statistics-minium_cell_volume={minimum_Volume}.csv")
    stats_data.to_csv(stats_csv, index=False)


if __name__ == "__main__":
    # Configure command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Processes cell measurement data from MorphoGraphX area, volume heatmap and parent labels "
                    "(generated by get_volume_data.py, saved in python_export_data folder with naming format).",
        epilog="Example: python process_cell_data.py /path/to/data --minium_Volume 25 --filter 'fu:1-1503S2,1-1503S4,27S1B,27S1A '"
    )
    parser.add_argument(
        "--input_folder",
        help="Path to folder containing the python_export_data folder and csv files with _Area, _parent , _Volume in file name "
    )
    parser.add_argument(
        "--minium_Volume",
        type=int,
        default=25,
        help="Minimum cell volume threshold (μm³) for inclusion (default: 25)"
    )
    parser.add_argument(
        "--mapping_file",
        default="aggregate_and_compare_data_from_MGX_heatmap\parent_label_to_name.txt",
        help="Path to parent label to name mapping file (default: parent_label_to_name.txt)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="filter out samples when counting some incomplete tissues, input as 'abbreviation : sample ID1, sample ID2; ac: 1-1503s1A, 1-1503s1B'; need use ' if this string has blanks"
    )
    args = parser.parse_args()

    filter = args.filter

    filter_abbreviation_IDs = filter
    if filter:
        filter_abbreviation_IDs = {
            k.strip(): [x.strip().upper() for x in v.split(',')]
            for k, v in (pair.split(':', 1) for pair in filter.split(';'))
            }

    # print(filter_abbreviation_IDs)

    # Process all valid data folders recursively
    for root, dirs, files in os.walk(args.input_folder):
        if 'python_export_data' in dirs:
            label_to_name, label_to_abbreviation = parent_labels(args.mapping_file)

            # Proceed with data aggregation if all checks pass
            aggregate_data(root, label_to_name, label_to_abbreviation, args.minium_Volume,filter_abbreviation_IDs)

    print("\t finished !")