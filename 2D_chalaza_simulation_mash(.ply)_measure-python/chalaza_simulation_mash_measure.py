#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process 2D chalaza simulation files (.ply) to calculate kink angle.

author: Ziqiang Luo
date: 2025-04-13
usage: python chalaza_simulation_mash_measure.py <folder_path>
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import open3d as o3d

# Check and install required packages
REQUIRED_PACKAGES = ['scipy', 'matplotlib','numpy','open3d']


def install_packages():
    """Install missing required packages using pip."""
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_packages()


def is_binary_file(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(1024)
    return b'\0' in raw_data


def convert_ply_binary_to_text(directory, filename, savename):
    file_path = os.path.join(directory, filename)
    save_path = os.path.join(directory, savename)
    if is_binary_file(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
        return savename
    return filename


def read_ply_files(directory,filename):
    outline_x, outline_y = [], []
    with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        data_start = lines.index("end_header\n") + 1
        x, y = [], []
        for line in lines[data_start:]:
            values = line.split()
            if float(values[2]) != 0:    # outline always at z=0 and above the points have z != 0 in ply files
                break
            x.append(float(values[0]))
            y.append(float(values[1]))
        outline_x.extend(x)
        outline_y.extend(y)
    return outline_x, outline_y


def normalize_coordinates(x, y):
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    x_normalized = [(xi - x_min) / (x_max - x_min) for xi in x]
    y_normalized = [(yi - y_min) / (y_max - y_min) for yi in y]

    x_scale = x_max - x_min
    y_scale = y_max - y_min

    return x_normalized, y_normalized, x_scale, y_scale


def find_y_distribution_extremes(x, y):
    from collections import Counter
    y_counts = Counter(y)
    max_y = max(y_counts, key=y_counts.get)

    # find the most common y value below 0.2
    y_below = [val for val in y if val < 0.2]
    if y_below:
        most_common_y_below_minus_40 = Counter(y_below).most_common(1)[0][0]
    else:
        most_common_y_below_minus_40 = max_y

    y_indices = [i for i, val in enumerate(y) if val == most_common_y_below_minus_40]
    ymin_xmin_idx = min(y_indices, key=lambda i: x[i])
    ymin_xmax_idx = max(y_indices, key=lambda i: x[i])

    return ymin_xmin_idx, ymin_xmax_idx


def determine_order(x, y, ymin_xmin_idx):
    if ymin_xmin_idx + 1 < len(x):
        next_x = x[ymin_xmin_idx + 1]
    else:
        next_x = x[0]

    if next_x > x[ymin_xmin_idx]:
        return "Counterclockwise"
    else:
        return "Clockwise"


def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude == 0:
        return 180
    angle = np.arccos(dot_product / magnitude) * 180 / np.pi
    return angle


def find_sharpest_turn(x, y, order, right_point_idx, left_point_idx):
    n = len(x)

    if order == "Counterclockwise":
        start = (right_point_idx - 1) % n  # Use % n to ensure that the indexes circulate within the range of the array.
        end = (left_point_idx - 2) % n    # If ymax_xmin_idx -1 is less than 0,% n will adjust it to the last index of the array.
    elif order == "Clockwise":
        start = (left_point_idx + 2) % n
        end = (right_point_idx + 1) % n

    idx_range = list(range(start, end))
    if start > end:
        idx_range = list(range(start, n)) + list(range(0, end))

    min_angle = 180
    min_angle_idx = None
    for i in idx_range:
        p1 = (x[(i - 1) % n], y[(i - 1) % n])
        p2 = (x[i], y[i])
        p3 = (x[(i + 1) % n], y[(i + 1) % n])
        angle = calculate_angle(p1, p2, p3)
        if angle < min_angle:
            min_angle = angle
            min_angle_idx = i

    return min_angle, min_angle_idx


def calculate_path_length(x, y, start_idx, end_idx, order):
    n = len(x)
    length = 0

    if start_idx > end_idx:
        idx_range = list(range(start_idx, n)) + list(range(0, end_idx + 1))
    else:
        idx_range = list(range(start_idx, end_idx + 1))

    for i in range(len(idx_range) - 1):
        p1 = np.array([x[idx_range[i]], y[idx_range[i]]])
        p2 = np.array([x[idx_range[i + 1]], y[idx_range[i + 1]]])
        length += np.linalg.norm(p1 - p2)

    return length, idx_range


def calculate_polygon_area(x_coords, y_coords):
    n = len(x_coords)
    # Shoelace formula  https://en.wikipedia.org/wiki/Shoelace_formula
    area = 0.5 * abs(sum(x_coords[i] * y_coords[(i + 1) % n] - y_coords[i] * x_coords[(i + 1) % n] for i in range(n)))
    return area


def calculate_slope_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:  # avoid division by zero
        return 90.0  # 90 degrees is vertical
    slope = (y2 - y1) / (x2 - x1)
    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def plot_smoothed_curve(x, y, extreme_points, sharpest_turn_idx, order, pos_path_indices, ant_path_indices):
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    t = np.linspace(0, len(x) - 1, 300)
    spl_x = make_interp_spline(np.arange(len(x)), x, k=1)
    spl_y = make_interp_spline(np.arange(len(y)), y, k=1)
    smooth_x, smooth_y = spl_x(t), spl_y(t)

    max_abs = max(abs(min(x)), abs(max(x)), abs(min(y)), abs(max(y)))
    plt.xlim(-5, max_abs+40)
    plt.ylim(-1, max_abs+45)

    plt.plot(smooth_x, smooth_y, 'k-', alpha=0.5, label=f"Outline")

    path_x = [x[i] for i in pos_path_indices]
    path_y = [y[i] for i in pos_path_indices]
    plt.plot(path_x, path_y, 'b-', linewidth=1, label='posterior Path')

    path_x = [x[i] for i in ant_path_indices]
    path_y = [y[i] for i in ant_path_indices]
    plt.plot(path_x, path_y, 'r-', linewidth=1, label='anterior Path')

    for idx in extreme_points:
        plt.scatter(x[idx], y[idx], c='k', s=5)
    plt.scatter(x[sharpest_turn_idx], y[sharpest_turn_idx], c='r', s=5, label='Sharpest Turn (posterior side)')

    # Draw the connection of Sharpest_turn and xmin_ymax
    xmin_ymax_idx = extreme_points[3]
    plt.plot([x[sharpest_turn_idx], x[xmin_ymax_idx]], [y[sharpest_turn_idx], y[xmin_ymax_idx]], 'g--', linewidth=1, label='Nucellar_neck_line')

    # Draw the dotted line that is parallel to the X -axis from the XMIN_YMAX point
    end_x = x[sharpest_turn_idx]
    plt.plot([x[xmin_ymax_idx], end_x], [y[xmin_ymax_idx], y[xmin_ymax_idx]], 'g--', linewidth=1, label='Parallel line to x-axis')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    # plt.show()


def main():
    if len(sys.argv) < 1 or len(sys.argv) > 3:
        print("python chalaza_simulation_mash_measure.py <folder_path> <ino_mutant>")
        #
        sys.exit(1)

    directory = sys.argv[1] # input folder
    output_folder = os.path.join(directory, "simulation_mash_kink_angle_measure")
    os.makedirs(output_folder, exist_ok=True)

    ino_mutant = sys.argv[2]

    ant_path_lengths = []
    pos_path_lengths = []
    Ratio_P_As = []
    ply_names = []
    kink_angle_degrees = []
    areas = []
    # deformations = []

    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            if 'text' in filename:       # skip the old text files
                continue
            ply_names.append(filename)

            savename = f"{filename.split('.')[0]}_text.ply"  # Save the file as text format if it is binary and read it later
            filename = convert_ply_binary_to_text(directory, filename, savename)
            # print(f"Processing {filename}")

            # maybe because of using the same original points simulation template, the order and coordinates of the points crosse all samples are similar.
            # so later I use some values to filter the points I need.
            x, y = read_ply_files(directory, filename)
            x = [xi - min(x) for xi in x]
            y = [yi - min(y) for yi in y]
            area = calculate_polygon_area(x, y)
            areas.append(area)

            # normalize for finding points based on cretin values
            x_normalized, y_normalized, x_scale, y_scale = normalize_coordinates(x,y)
            # Find the most common y value below -40 as baseline and use it to find the leftmost and rightmost points of it.
            # so that won't be the straight line above -40.
            ymin_xmin_idx, ymin_xmax_idx = find_y_distribution_extremes(x_normalized, y_normalized)

            # xmin_ymax_idx is up tip of anterior line; it should be always the leftmost (and then highest) point because of chalaza can grow freely on anterior side.
            xmin_ymax_idx = min([i for i, val in enumerate(x) if val == min(x)], key=lambda i: y[i])
            # for ino mutant, this end point of chalaza is on ymax_xmin
            if ino_mutant.strip().upper() == 'YES':
                xmin_ymax_idx = min([i for i, val in enumerate(y) if val == max(y)], key=lambda i: x[i])

            # Find point on the mid of posterior line; which should has y value between 0.2 and 0.4 and find the rightmost point in that range.
            # the sharpest turn should be between xmin_ymax and xmax_in_range
            valid_indices = [i for i, val in enumerate(y_normalized) if 0.2 <= val <= 0.4]
            xmax_in_range_idx = max(valid_indices, key=lambda i: x[i])
            # valid_indices = [i for i, val in enumerate(x_normalized) if 0.2 <= val <= 0.4]
            # ymax_in_range_idx = max(valid_indices, key=lambda i: y[i])


            # determine the order of connection, it seems to be always counterclockwise and start on a point on the middle of Anterior line.
            order = determine_order(x, y, ymin_xmin_idx)
            # find the sharpest turn point between xmin_ymax and xmax_in_range
            min_angle, sharpest_turn_idx = find_sharpest_turn(x, y, order, xmax_in_range_idx, xmin_ymax_idx)

            # Calculate the path length of the anterior and posterior sides use those landmarks
            if order == "Clockwise":
                pos_path_length, pos_path_indices = calculate_path_length(x, y, sharpest_turn_idx, ymin_xmax_idx, order)
                ant_path_length, ant_path_indices = calculate_path_length(x, y, ymin_xmin_idx, xmin_ymax_idx, order)
            else:
                pos_path_length, pos_path_indices = calculate_path_length(x, y, ymin_xmax_idx, sharpest_turn_idx, order)
                ant_path_length, ant_path_indices = calculate_path_length(x, y, xmin_ymax_idx, ymin_xmin_idx, order)
            ant_path_lengths.append(ant_path_length)
            pos_path_lengths.append(pos_path_length)
            Ratio_P_As.append(pos_path_length / ant_path_length)

            ymax_xmin_idx = min([i for i, val in enumerate(y) if val == max(y)], key=lambda i: x[i])
            # # the deformation is defined as the
            # # ymax_xmin_idx is the highest point region
            # if ymax_xmin_idx == xmin_ymax_idx:
            #     valid_indices = [i for i, val in enumerate(y) if val == max(y) and x[i] > x[xmin_ymax_idx]]
            #     ymax_xmin_idx = min(valid_indices, key=lambda i: x[i])
            #
            # if order == "Counterclockwise":
            #     if sharpest_turn_idx >= ymin_xmin_idx or ymin_xmin_idx - sharpest_turn_idx > len(x) / 2: # ymin_xmin_idx may be the last point and sharpest_turn_idx may be the first point
            #         deformations.append("No")
            #     else:
            #         deformations.append("Yes")
            # if order == "Clockwise":
            #     if sharpest_turn_idx <= ymin_xmin_idx or sharpest_turn_idx - ymin_xmin_idx > -len(x) / 2: # sharpest_turn_idx may be the last point and ymin_xmin_idx may be the first point
            #         deformations.append("No")
            #     else:
            #         deformations.append("Yes")


            # Plot the smoothed curve and mark those landmarks and kink angle (sharpest_turn to xmin_ymax and parallel line to x-axis)
            plot_smoothed_curve(x, y, [ymin_xmin_idx, ymin_xmax_idx, ymax_xmin_idx, xmin_ymax_idx], sharpest_turn_idx, order,
                                pos_path_indices,ant_path_indices)

            kink_angle_degree = calculate_slope_angle((x[sharpest_turn_idx], y[sharpest_turn_idx]),
                                                      (x[xmin_ymax_idx], y[xmin_ymax_idx]))

            kink_angle_degrees.append(kink_angle_degree)

            # Save the figure
            plt.savefig(os.path.join(output_folder, f"{filename}.png"))
            # plt.show()
            plt.close()
    # plt.show()  # Show all the figures overlaid on each other

    with open(os.path.join(output_folder, "chalaza_simulation_mash_measure.csv"), "w") as file:
        file.write(
            f"Sample_name \t Area \t Kink angle \t Ratio (P/A) \t Posterior Length \t Anterior Length\n")
        for i in range(len(ant_path_lengths)):
            file.write(f"{ply_names[i]} \t {areas[i]} \t {kink_angle_degrees[i]} \t {Ratio_P_As[i]} \t {pos_path_lengths[i]} \t {ant_path_lengths[i]}\n")

    print("data and images saved in your input folder/simulation_mash_kink_angle_measure")
if __name__ == "__main__":
    main()
