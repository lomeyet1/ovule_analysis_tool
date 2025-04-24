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

# Default markers to search for in filenames
DEFAULT_MARKERS = ["B_MAXMID", "B_MAXMIN", "B_COORDX", "B_AREA", "B_VOL"]

# List of required packages with their import names and pip names
REQUIRED_PACKAGES = [
    {"import_name": "pandas", "pip_name": "pandas"},
    {"import_name": "numpy", "pip_name": "numpy"},
    {"import_name": "matplotlib", "pip_name": "matplotlib"},
    {"import_name": "seaborn", "pip_name": "seaborn"},
    {"import_name": "scipy", "pip_name": "scipy"},
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


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following attributes:
            - input_folder: Path to input folder containing stage and sample folders
            - markers: List of markers to search for in CSV filenames
            - bin: Number of bins for histograms (default=4)
    """
    parser = argparse.ArgumentParser(description="Process CSV files and generate summary data.")
    parser.add_argument("--input_folder", required=True, help="Input folder containing stage folders which contain sample folders.")
    parser.add_argument("--markers", nargs="+", default=DEFAULT_MARKERS, help="Markers to search for in csv filenames under each sample folder.")
    parser.add_argument("--bin", type=int, default=4, help="Number of bins for histograms (default=4).")
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
                    if file.endswith(".csv") and any(marker in file for marker in markers):
                        marker = next(marker for marker in markers if marker in file)
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
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # 获取需要绘制的列（从第4列开始）
    plot_columns = binned_df.columns[3:]

    # 准备存储统计结果的DataFrame
    stats_list = []

    # 存储所有t检验结果的列表
    all_ttest_results = []

    # 为每一列创建图表和统计
    for col in plot_columns:
        plt.figure(figsize=(10, 8))

        # 创建散点图
        ax = sns.swarmplot(
            y="Binn_range",
            x=col,
            hue="Stage",
            data=binned_df,
            dodge=True,
            palette="viridis",
            size=5
        )

        # 计算基本统计量
        stats_df = binned_df.groupby(["Stage", "Binn_range"], observed=True)[col].agg(
            ['mean', 'std', 'count']
        ).reset_index()

        # 计算95%置信区间
        def calc_ci(x):
            if len(x) > 1:
                ci = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
                return pd.Series({'CI_lower': ci[0], 'CI_upper': ci[1]})
            return pd.Series({'CI_lower': np.nan, 'CI_upper': np.nan})

        ci_df = binned_df.groupby(["Stage", "Binn_range"], observed=True)[col].apply(calc_ci).reset_index()
        stats_df = pd.merge(stats_df, ci_df, on=["Stage", "Binn_range"])

        # 重命名列
        stats_df.columns = ["Stage", "Binn_range", f"{col}_mean", f"{col}_std",
                            f"{col}_count", f"{col}_CI_lower", f"{col}_CI_upper"]

        # 保存统计结果
        stats_list.append(stats_df)

        # 准备t检验比较
        groups = binned_df.groupby(["Stage", "Binn_range"], observed=True)
        group_keys = list(groups.groups.keys())

        # 生成所有可能的比较组合
        comparisons = []

        # 比较同Stage不同Binn_range的数据
        for stage in binned_df["Stage"].unique():
            stage_groups = [key for key in group_keys if key[0] == stage]
            if len(stage_groups) > 1:
                comparisons.extend(itertools.combinations(stage_groups, 2))

        # 比较同Binn_range不同Stage的数据
        for binn in binned_df["Binn_range"].unique():
            binn_groups = [key for key in group_keys if key[1] == binn]
            if len(binn_groups) > 1:
                comparisons.extend(itertools.combinations(binn_groups, 2))

        # 执行t检验并记录结果
        for (stage1, binn1), (stage2, binn2) in comparisons:
            group1 = groups.get_group((stage1, binn1))[col]
            group2 = groups.get_group((stage2, binn2))[col]

            # 执行Welch's t-test (不假设方差相等)
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

            # 记录结果
            result = {
                'data_type': col,
                'stage1': stage1,
                'Binn_range1': binn1,
                'stage2': stage2,
                'Binn_range2': binn2,
                't_statistic': t_stat,
                'p_value': p_value,
                'group1_size': len(group1),
                'group2_size': len(group2),
                'group1_mean': np.mean(group1),
                'group2_mean': np.mean(group2)
            }
            all_ttest_results.append(result)

        # 添加平均值和误差线
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
            errorbar=('ci', 95),
            errwidth=1.5,
            capsize=0.1,
            estimator='mean',
            ax=ax
        )

        # 图表美化
        plt.title(f"{col} distribution on normalized Y Coordinate range of ovule B epidermal")
        plt.xlabel(col)
        plt.ylabel("Binned Normalized Y Coordinate")
        plt.yticks(rotation=0)

        # 处理图例
        handles, labels = ax.get_legend_handles_labels()
        unique_stages = binned_df["Stage"].unique()
        handles = handles[:len(unique_stages)]
        labels = labels[:len(unique_stages)]
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Sample Stage")

        plt.tight_layout()

        # 保存图表
        plt.savefig(os.path.join(output_dir, f"{col}_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 合并所有统计结果
    stats_result = stats_list[0]
    for df in stats_list[1:]:
        stats_result = pd.merge(stats_result, df, on=["Stage", "Binn_range"], how="outer")

    # 创建t检验结果DataFrame
    ttest_df = pd.DataFrame(all_ttest_results)

    # 计算每个Stage的不重复Sample数量
    sample_number = binned_df.groupby(["Stage"], observed=True)["Sample"].nunique().reset_index()
    sample_number.columns = ["Stage", "sample number"]

    # 保存所有结果
    stats_result.to_csv(os.path.join(output_dir, "binned_data_statistics.csv"), index=False)
    sample_number.to_csv(os.path.join(output_dir, "sample_number.csv"), index=False)

    if not ttest_df.empty:
        ttest_df.to_csv(os.path.join(output_dir, "t_test_results.csv"), index=False)




def main():
    check_and_install_packages()
    args = parse_arguments()
    num_bins = int(args.bin)

    output_dir = os.path.join(args.input_folder, f"B_epidermal_cell_distribution_bin={num_bins}")
    os.makedirs(output_dir, exist_ok=True)
    data_files = find_csv_files(args.input_folder, args.markers)
    df= process_csv_files(data_files, args.markers)


    binned_df = normalize_and_bin(df, num_bins, args.markers,output_dir)
    plot_binned_data_with_stats(binned_df, output_dir)
    print(f" \t saved in {os.path.basename(args.input_folder)}/B_epidermal_cell_distribution_bin={num_bins}")




if __name__ == "__main__":
    main()