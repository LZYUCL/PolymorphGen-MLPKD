# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import dpdata
import os

# Main directory list (adjust according to actual paths)
main_dirs = ['cluster_name_1', 'cluster_name_2', 'cluster_name_3', 'cluster_name_4', 'cluster_name_5']

# Initialize MultiSystems object
ms = dpdata.MultiSystems()

# Traverse main directories and load OUTCAR files
for main_dir in main_dirs:
    # Traverse subdirectories starting with vasp_POSCAR_
    sub_dirs = [d for d in os.listdir(main_dir) if d.startswith('vasp_POSCAR_')]
    for sub_dir in sub_dirs:
        outcar_path = os.path.join(main_dir, sub_dir, 'OUTCAR')
        if os.path.isfile(outcar_path):
            try:
                # Load OUTCAR file
                ls = dpdata.LabeledSystem(outcar_path, fmt='vasp/outcar')
                if len(ls) > 0:  # Ensure loaded dataset is not empty
                    ms.append(ls)
            except Exception as e:
                print(f"Unable to process file {outcar_path}: {e}")

# Convert merged dataset to DeePMD raw and npy formats
ms.to_deepmd_raw('deepmd_combined')
ms.to_deepmd_npy('deepmd_combined')

print("Dataset conversion completed, saved in deepmd_combined folder")
