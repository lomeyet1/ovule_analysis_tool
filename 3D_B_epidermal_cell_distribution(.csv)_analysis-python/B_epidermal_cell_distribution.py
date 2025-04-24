#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process CSV files about B_epidermal_cell_distribution from MorphoGraphX and generate summary data and scatter plots.

author: Ziqiang Luo
date: 2025-04-13
usage: python B_epidermal_cell_distribution.py --input_folder <folder_path> [--markers MARKERS_lidt] [--bin NUM_BINS]
"""

import argparse
import subprocess
import sys
import os
import itertools
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



# List of required packages with their import names and pip names
REQUIRED_PACKAGES = [
    {"import_name": "pandas", "pip_name": "pandas"},
    {"import_name": "numpy", "pip_name": "numpy"},
    {"import_name": "matplotlib", "pip_name": "matplotlib"},
    {"import_name": "seaborn", "pip_name": "seaborn"},
    {"import_name": "scipy", "pip_name": "scipy"},
]

# Default markers to search for in filenames
DEFAULT_MARKERS = ["B_MAXMID", "B_MAXMIN", "B_COORDX", "B_AREA", "B_VOL"]

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



def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following attributes:
            - input_folder: Path to input folder containing stage and sample folders
            - markers: List of markers to search for in CSV filenames
            - bin: Number of bins for histograms (default=4)
    """

    parser = argparse.ArgumentParser(description="Process CSV files about B_epidermal_cell_distribution from MorphoGraphX and generate summary data and scatter plots.")
    parser.add_argument("--input_folder", required=True, help="Input folder containing stage folders which contain sample folders.")
    parser.add_argument("--markers", type=str, default=DEFAULT_MARKERS, help="Markers to search for in csv filenames under each sample folder.")
    parser.add_argument("--bin", type=int, default=4, help="Number of bins for Normalized Y Coordinate on B_epidermal (default=4).")
    return parser.parse_args()


def find_csv_files(input_folder, markers):
    """
    Recursively search for CSV files containing specified markers in the folder structure.

    Args:
        input_folder (str): Root directory containing stage/sample folders
        markers (list): List of marker strings to search for in filenames

    Returns:
        dict: Nested dictionary structure: {stage: {sample: {marker: filepath}}}
    """
    data_files = {}
    for stage in os.listdir(input_folder):
        stage_path = os.path.join(input_folder, stage)
        if not os.path.isdir(stage_path):
            continue

        data_files[stage] = {}
        for sample in os.listdir(stage_path):
            sample_path = os.path.join(stage_path, sample)
            if not os.path.isdir(sample_path):
                continue

            data_files[stage][sample] = {}
            for root, _, files in os.walk(sample_path):
                for file in files:
                    if file.endswith(".csv") and any(marker in file.strip().upper() for marker in markers):
                        marker = next(marker for marker in markers if marker in file.strip().upper())
                        data_files[stage][sample][marker] = os.path.join(root, file)

                        df = pd.read_csv(os.path.join(root, file), sep=",")
                        if len(df.columns) < 2:
                            data_files[stage].pop(sample)
                            print(f"Skipping sample {sample} in {stage} due to missing column in {marker} file.")
                            break
                if sample not in data_files[stage]:
                    break

            if sample in data_files[stage]:
                if len(data_files[stage][sample]) != len(markers):
                    keys_list = list(data_files[stage][sample].keys())
                    data_files[stage].pop(sample)
                    missing_markers = [marker for marker in markers if marker not in keys_list]
                    print(f"Skipping sample {sample} in {stage} due to missing {missing_markers} files in the sample folder.")

    return data_files


def process_csv_files(data_files, markers):
    """
    Process found CSV files and merge data into a structured DataFrame.

    Args:
        data_files (dict): Nested dictionary from find_csv_files()
        markers (list): List of marker strings

    Returns:
        pd.DataFrame: Combined data with columns: Stage, Sample, Cell_Label, and marker values
    """
    all_data = []
    for stage, samples in data_files.items():
        for sample, files in samples.items():
            # Load CSV data
            extracted_data = {}
            for marker, filepath in files.items():
                df = pd.read_csv(filepath, sep=",")
                extracted_data[marker] = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # {cell_label: value}

            # Merge data based on cell labels
            cell_labels = set().union(*[data.keys() for data in extracted_data.values()])
            for cell in cell_labels:
                row = {
                    "Stage": stage,
                    "Sample": sample,
                    "Cell_Label": cell,
                    **{marker: extracted_data.get(marker, {}).get(cell, np.nan) for marker in markers}
                }
                all_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    return df


def normalize_and_bin(df, num_bins, markers,output_dir):
    """
    Normalize coordinates and calculate bin-based statistics.

    Args:
        df (pd.DataFrame): Input data from process_csv_files()
        num_bins (int): Number of bins to create
        markers (list): List of marker strings
        output_dir (str): Directory to save output files

    Returns:
        pd.DataFrame: Binned and aggregated data
    """

    # Normalize B_COORDX per sample and compute pooled rescaled B_AREA and B_VOL
    for stage, group in df.groupby("Stage"):
        # Normalize B_COORDX per sample
        df.loc[group.index, f"Norm_{markers[2]}"] = group.groupby("Sample")[markers[2]].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        # Compute pooled rescaled B_AREA and B_VOL per stage
        avg_area = group[markers[3]].mean()
        avg_vol = group[markers[4]].mean()
        df.loc[group.index, f"{markers[3]}_pool_rescale"] = group[markers[3]] / avg_area
        df.loc[group.index, f"{markers[4]}_pool_rescale"] = group[markers[4]] / avg_vol

    # Bin data and compute sums
    bins = np.linspace(0, 1, num_bins + 1)
    paired_bins = [f"[{bins[i]:.2f}, {bins[i + 1]:.2f}]" for i in range(len(bins) - 1)]

    df["Binn_range"] = pd.cut(df[f"Norm_{markers[2]}"], bins, labels=paired_bins, include_lowest=True)

    # oder the columns
    cols = ["Stage","Sample",'Cell_Label',"Binn_range",markers[0],markers[1],f"{markers[4]}_pool_rescale",f"{markers[3]}_pool_rescale",markers[4],markers[3]]
    df = df[cols]

    # Group by Stage, Sample and Binn_range and aggregate values
    agg_dict = {
        markers[0]: 'sum',
        markers[1]: 'sum',
        f"{markers[4]}_pool_rescale": 'sum',
        f"{markers[3]}_pool_rescale": 'sum',
        markers[4]: 'sum',
        markers[3]: 'sum'
    }

    grouped = df.groupby(["Stage", "Sample", "Binn_range"], observed=True).agg(agg_dict).reset_index()

    # Add cell number column (count of rows that were merged)
    cell_counts = df.groupby(["Stage", "Sample", "Binn_range"], observed=True).size().reset_index(name="Cell_number")
    grouped = grouped.merge(cell_counts, on=["Stage", "Sample", "Binn_range"])

    # Reorder columns to insert Cell_number as the 4th column
    new_cols = grouped.columns.tolist()
    new_cols.insert(3, new_cols.pop(new_cols.index("Cell_number")))
    grouped = grouped[new_cols]
    grouped.rename(columns={markers[4]: 'Volume (μm³)', markers[3]: 'Area (μm²)'}, inplace=True)
    grouped.rename(columns={f"{markers[4]}_pool_rescale": 'Rescaled Volume (μm³)', f"{markers[3]}_pool_rescale": 'Rescaled Area (μm²)'}, inplace=True)

    # Calculate per cell values from the 6th column onwards
    for col in grouped.columns[6:]:
        grouped[f"{col} per cell"] = grouped[col] / grouped['Cell_number']

    # Save the processed data
    grouped.to_csv(os.path.join(output_dir, f"Samples_collection_and_binned.csv"), sep=",", index=False)

    return grouped



def plot_binned_data_with_stats(binned_df, output_dir):
    """
    Generate swarm plots with statistics for binned data.

    Args:
        binned_df (pd.DataFrame): Binned data from normalize_and_bin()
        output_dir (str): Directory to save plots and statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    plot_columns = binned_df.columns[3:]
    stats_list = []
    all_ttest_results = []

    for col in plot_columns:
        plt.figure(figsize=(10, 8))
        ax = sns.swarmplot(
            y="Binn_range",
            x=col,
            hue="Stage",
            data=binned_df,
            dodge=True,
            palette="viridis",
            size=5
        )

        # Calculate statistics
        stats_df = binned_df.groupby(["Stage", "Binn_range"], observed=True)[col].agg(
            ["mean", "std", "count"]
        ).reset_index()

        # Calculate 95% confidence intervals
        def calc_ci(x):
            if len(x) > 1:
                ci = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
                return pd.Series({"CI_lower": ci[0], "CI_upper": ci[1]})
            return pd.Series({"CI_lower": np.nan, "CI_upper": np.nan})

        ci_df = binned_df.groupby(["Stage", "Binn_range"], observed=True)[col].apply(calc_ci).reset_index()
        stats_df = pd.merge(stats_df, ci_df, on=["Stage", "Binn_range"])

        # Rename columns and store
        stats_df.columns = [
            "Stage", "Binn_range", f"{col}_mean", f"{col}_std",
            f"{col}_count", f"{col}_CI_lower", f"{col}_CI_upper"
        ]
        stats_list.append(stats_df)

        # Generate all possible group comparisons
        groups = binned_df.groupby(["Stage", "Binn_range"], observed=True)
        group_keys = list(groups.groups.keys())

        # Compare same stage, different bins
        for stage in binned_df["Stage"].unique():
            stage_groups = [key for key in group_keys if key[0] == stage]
            comparisons = itertools.combinations(stage_groups, 2)

            for (s1, b1), (s2, b2) in comparisons:
                g1 = groups.get_group((s1, b1))[col]
                g2 = groups.get_group((s2, b2))[col]

                t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)

                all_ttest_results.append({
                    "data_type": col,
                    "stage1": s1, "Binn_range1": b1,
                    "stage2": s2, "Binn_range2": b2,
                    "t_statistic": t_stat, "p_value": p_value,
                    "group1_size": len(g1), "group2_size": len(g2),
                    "group1_mean": np.mean(g1), "group2_mean": np.mean(g2)
                })

        # Add mean points with error bars
        sns.pointplot(
            y="Binn_range",
            x=col,
            hue="Stage",
            data=binned_df,
            dodge=0.5,
            join=False,
            palette=["gray"] * len(binned_df["Stage"].unique()),
            markers="d",
            scale=0.75,
            errorbar=("ci", 95),
            errwidth=1.5,
            capsize=0.1,
            estimator="mean",
            ax=ax
        )

        # Format plot
        plt.title(f"{col} distribution on normalized Y Coordinate range")
        plt.xlabel(col)
        plt.ylabel("Binned Normalized Y Coordinate")
        plt.yticks(rotation=0)

        # Adjust legend
        handles, labels = ax.get_legend_handles_labels()
        unique_stages = binned_df["Stage"].unique()
        ax.legend(
            handles[:len(unique_stages)],
            labels[:len(unique_stages)],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Sample Stage"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_plot.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Save statistics
    if stats_list:
        stats_result = pd.concat(stats_list, axis=1)
        stats_result = stats_result.loc[:, ~stats_result.columns.duplicated()]
        stats_result.to_csv(os.path.join(output_dir, "binned_data_statistics.csv"), index=False)

    # Save t-test results
    if all_ttest_results:
        ttest_df = pd.DataFrame(all_ttest_results)
        ttest_df.to_csv(os.path.join(output_dir, "t_test_results.csv"), index=False)

    # Save sample counts
    sample_counts = binned_df.groupby("Stage", observed=True)["Sample"].nunique().reset_index()
    sample_counts.columns = ["Stage", "sample_count"]
    sample_counts.to_csv(os.path.join(output_dir, "sample_counts.csv"), index=False)


def main():
    check_and_install_packages()
    args = parse_arguments()
    num_bins = int(args.bin)

    markers = args.markers
    if markers != DEFAULT_MARKERS:
        markers = [marker.strip().upper() for marker in args.markers.split(',')]

    output_dir = os.path.join(args.input_folder, f"B_epidermal_cell_distribution_bin={num_bins}")
    os.makedirs(output_dir, exist_ok=True)
    data_files = find_csv_files(args.input_folder, markers)
    df= process_csv_files(data_files, markers)


    binned_df = normalize_and_bin(df, num_bins, markers,output_dir)
    plot_binned_data_with_stats(binned_df, output_dir)
    print(f" \t saved in {os.path.basename(args.input_folder)}/B_epidermal_cell_distribution_bin={num_bins}")




if __name__ == "__main__":
    main()