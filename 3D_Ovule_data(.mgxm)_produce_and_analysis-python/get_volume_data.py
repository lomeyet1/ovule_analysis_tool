#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Produce python scripts to run MorphoGraphX to produce area, volume heatmap and parent labels for each mesh.

author: Ziqiang Luo
date: 2025-04-13
usage: python get_volume_data.py <folder1_path> <folder2_path> [output_folder]
"""

import os
import subprocess
import argparse

def process_mgxm_files(input_folder, generate_area=True, generate_volume=True, generate_parent=True):
    """
    Process .mgxm files in the specified input folder.

    :param input_folder: The folder containing .mgxm files.
    :param generate_area: Whether to generate Area-related lines.
    :param generate_volume: Whether to generate Volume-related lines.
    :param generate_parent: Whether to generate parent-related lines.
    """
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print("Error: The specified input folder does not exist.")
        return

    input_folder = os.path.abspath(input_folder)

    # Define the output folder
    output_folder = os.path.join(input_folder, "python_export_data")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Created 'python_export_data' folder.")

    # Initialize the script content
    script_content = []
    sample_count = 0

    # Iterate through all .mgxm files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mgxm"):
            sample_name = os.path.splitext(filename)[0]
            sample_path = os.path.join(input_folder, filename)
            sample_count += 1

            # Generate script content for each .mgxm file
            script_content.append("Process.Mesh__System__Load('{}', 'no', 'no', '0')".format(sample_path))
            script_content.append("Process.Mesh__Selection__Select_All()")

            # Conditionally generate Area, Volume, and parent lines
            if generate_area:
                script_content.append("Process.Mesh__Heat_Map__Heat_Map_Classic('Area', 'Geometry', 'python_export_data/{}_Area.csv', 'Geometry', 'No', '0', '65535', 'Yes', 'Yes', 'None', 'No', 'Increasing', 'Ratio', '.001', '1.0')".format(sample_name))
            if generate_volume:
                script_content.append("Process.Mesh__Heat_Map__Heat_Map_Classic('Volume', 'Geometry', 'python_export_data/{}_Volume.csv', 'Geometry', 'No', '0', '65535', 'Yes', 'Yes', 'None', 'No', 'Increasing', 'Ratio', '.001', '1.0')".format(sample_name))
            if generate_parent:
                script_content.append("Process.Mesh__Lineage_Tracking__Save_Parents('python_export_data/{}_parent', 'Yes')".format(sample_name))

            script_content.append("Process.Mesh__System__Reset('0')")

    # Write the script content to python_export_data.py
    script_file_path = os.path.join(input_folder, "python_export_data.py")
    with open(script_file_path, "w") as script_file:
        script_file.write("\n".join(script_content))
    print("Generated 'python_export_data.py' with {} samples.".format(sample_count))

    # Run the generated script using mgx
    try:
        subprocess.call(["mgx", "--run", script_file_path])
        print("Successfully ran the generated script.")
    except Exception as e:
        print("Error running the script: {}".format(str(e)))

    # Print the number of samples and a warning message
    print("Detected {} samples.".format(sample_count))
    print("Warning: If any sample has no Parent labels and you did not use --no-parent, the MGX GUI will stop there when export the parent csv. You need to manually move it out.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process .mgxm files and generate a script for mgx.")
    parser.add_argument("--input_folder", help="Path to the folder containing .mgxm files.")
    parser.add_argument("--no-area", action="store_true", help="Do not generate Area-related lines.")
    parser.add_argument("--no-volume", action="store_true", help="Do not generate Volume-related lines.")
    parser.add_argument("--no-parent", action="store_true", help="Do not generate parent-related lines.")

    args = parser.parse_args()

    # Set generation flags based on arguments
    generate_area = not args.no_area
    generate_volume = not args.no_volume
    generate_parent = not args.no_parent

    # Process the files
    process_mgxm_files(args.input_folder, generate_area, generate_volume, generate_parent)