# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

def read_data(file_path, skip_lines=1):
    indices = []
    X = []
    Y = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i < skip_lines:
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                index = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                indices.append(index)
                X.append(x)
                Y.append(y)
            except ValueError:
                continue
    return indices, np.array(X), np.array(Y)

def plot_original_data(X, Y, output_filename):
    plt.figure(figsize=(12, 10))
    plt.scatter(X, Y, color='black', alpha=0.6, s=10, edgecolors='none')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Original Data Scatter Plot')
    plt.grid(True)
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def filter_points_by_resolution(points, step_x, step_y):
    """
    Filter point set by grid resolution.
    Keep only one point in each grid defined by step_x and step_y.
    """
    unique_points = []
    seen_grids = set()
    for p in points:
        grid_x = round(float(p[1]) / step_x)
        grid_y = round(float(p[2]) / step_y)
        grid_key = (grid_x, grid_y)
        if grid_key not in seen_grids:
            seen_grids.add(grid_key)
            unique_points.append(p)

    return np.array(unique_points)

def export_points_to_txt(points, output_file):
    with open(output_file, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {float(point[1]):.6f} {float(point[2]):.6f}\n")
    print(f"Points exported to {output_file}")

def copy_selected_POSCARs(points_file, poscars_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    df = pd.read_csv(points_file, delim_whitespace=True, header=None)
    indices = df[0].astype(str).tolist()

    print(f"Total points to copy: {len(indices)}")

    success_count = 0
    failure_count = 0
    for index in indices:
        filename = f"POSCAR_{index}"
        src_path = os.path.join(poscars_folder, filename)
        dest_path = os.path.join(destination_folder, filename)

        if os.path.isfile(src_path):
            try:
                shutil.copy(src_path, dest_path)
                success_count += 1
            except Exception as e:
                print(f"Failed to copy {filename}: {e}")
                failure_count += 1
        else:
            print(f"Source file does not exist: {src_path}")
            failure_count += 1

    print(f"Copied {success_count} files successfully. Failed to copy {failure_count} files.")

def plot_points_from_txt(txt_file):
    if not os.path.isfile(txt_file):
        print(f"{txt_file} does not exist, skipping plot.")
        return

    df = pd.read_csv(txt_file, delim_whitespace=True, header=None)
    if df.shape[1] < 3:
        print(f"{txt_file} format error, not enough columns.")
        return

    x = df[1].values
    y = df[2].values

    plt.figure(figsize=(10,8))
    plt.scatter(x, y, color='black', s=10, alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Scatter plot from {txt_file}")
    plt.grid(True)

    png_file = os.path.splitext(txt_file)[0] + '.png'
    plt.savefig(png_file, format='png', dpi=300)
    plt.close()
    print(f"Plot saved as {png_file}")

def main():
    # Original data file path and number of lines to skip (modify according to actual situation)
    file_path = 'results_r_8.txt'
    skip_lines = 0
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    indices, X, Y = read_data(file_path, skip_lines=skip_lines)
    if len(X) == 0 or len(Y) == 0:
        print("No valid data read. Please check the file format.")
        return

    print(f"Read {len(X)} data points after skipping {skip_lines} lines.")
    original_points = np.column_stack((indices, X, Y))

    # Plot original data
    original_plot_file = 'original_data_scatter.png'
    plot_original_data(X, Y, original_plot_file)

    # User-defined multiple rounds of resolution parameters (filtering by resolution only)
    resolutions = [
        (0.2, 0.006),
	(0.1, 0.005),
        (0.05, 0.0025),
	(0.04, 0.0024),
	(0.02, 0.0022),
	(0.025, 0.0002),
	(0.001, 0.0001)
        # Add more rounds as needed
    ]

    # Initialize remaining point set as original points
    remaining_points = original_points.copy()
    chosen_indices_all_rounds = set()
    round_txt_files = []
    poscars_folder = 'POSCARs'

    # Multiple rounds of resolution filtering
    for i, (step_x, step_y) in enumerate(resolutions, start=1):
        print(f"=== Round {i}, step_x={step_x}, step_y={step_y} ===")

        # Filter remaining point set based on this round's resolution
        filtered_points = filter_points_by_resolution(remaining_points, step_x, step_y)

        # Remove duplicate points already selected in previous rounds (if any)
        filtered_points = [p for p in filtered_points if p[0] not in chosen_indices_all_rounds]
        filtered_points = np.array(filtered_points)

        # Update global records
        newly_chosen_indices = set(filtered_points[:,0]) if len(filtered_points) > 0 else set()
        chosen_indices_all_rounds.update(newly_chosen_indices)

        # Remove points selected in this round from remaining points
        remaining_points = np.array([p for p in remaining_points if p[0] not in newly_chosen_indices])

        # Export this round's results
        round_txt_file = f"new_points_with_internal_round_{i}.txt"
        export_points_to_txt(filtered_points, round_txt_file)
        # Draw scatter plot for this round's txt file
        plot_points_from_txt(round_txt_file)
        round_txt_files.append(round_txt_file)

        # Copy POSCAR files for this round's points (if POSCARs folder exists)
        if os.path.isdir(poscars_folder):
            dest_folder = f'selected_POSCARs_round_{i}'
            copy_selected_POSCARs(round_txt_file, poscars_folder, dest_folder)
        else:
            print(f"POSCARs folder does not exist: {poscars_folder}, skipping copying for round {i}")

    # Export remaining points after all rounds
    leftover_file = "remaining_points.txt"
    export_points_to_txt(remaining_points, leftover_file)
    # Draw scatter plot for remaining points
    plot_points_from_txt(leftover_file)

    # Summary information
    print(f"Original data points: {len(X)}")
    total_chosen = len(chosen_indices_all_rounds)
    print(f"Total chosen points after all rounds: {total_chosen}")
    print(f"Remaining points: {len(remaining_points)}")

if __name__ == "__main__":
    main()
