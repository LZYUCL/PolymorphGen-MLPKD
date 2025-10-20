# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import os
import shutil
import numpy as np
import dpdata
from deepmd.utils.data import DeepmdData
from deepmd.infer.deep_eval import DeepEval

def generate_zero_files(target_folder):
    """Generate zero-padded energy/force/virial files"""
    coord_path = os.path.join(target_folder, 'coord.raw')
    
    # Verify coord.raw exists
    if not os.path.exists(coord_path):
        raise FileNotFoundError(f"Cannot find {coord_path}")
        
    # Read coordinate file
    with open(coord_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    # Get number of frames and columns
    num_frames = len(lines)
    if num_frames == 0:
        raise ValueError("coord.raw file is empty")
    
    num_cols = len(lines[0].split())
    
    # Verify all rows have consistent column count
    for idx, ln in enumerate(lines):
        if len(ln.split()) != num_cols:
            raise ValueError(f"Row {idx+1} has inconsistent column count")

    # Generate energy.raw (single column)
    energy_path = os.path.join(target_folder, 'energy.raw')
    with open(energy_path, 'w') as f:
        f.write('\n'.join(['0'] * num_frames))
    
    # Generate virial.raw (9 columns)
    virial_path = os.path.join(target_folder, 'virial.raw')
    with open(virial_path, 'w') as f:
        f.write('\n'.join(['0 0 0 0 0 0 0 0 0'] * num_frames))
    
    # Generate force.raw (same column count as coord.raw)
    force_path = os.path.join(target_folder, 'force.raw')
    with open(force_path, 'w') as f:
        zero_line = ' '.join(['0'] * num_cols) + '\n'
        f.write(zero_line * num_frames)

def process_poscar(poscar_path, out_dir):
    """
    Processing pipeline for a single POSCAR file:
    1. Convert POSCAR to DeepMD raw format (stored in out_dir/deepmd)
    2. Generate zero-padded files (in out_dir/deepmd)
    3. Create LabeledSystem and convert to generate deepmd_combined data (stored in out_dir/deepmd_combined)
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Define intermediate output directory deepmd
    deepmd_dir = os.path.join(out_dir, 'deepmd')
    
    # Step 1: Convert POSCAR to DeepMD format
    system = dpdata.System(poscar_path, fmt='vasp/poscar')
    system.to_deepmd_raw(deepmd_dir)
    
    # Step 2: Generate zero-padded files (energy, force, virial)
    generate_zero_files(deepmd_dir)
    
    # Step 3: Create LabeledSystem and convert to generate deepmd_combined data
    combined_dir = os.path.join(out_dir, 'deepmd_combined')
    ls = dpdata.LabeledSystem(deepmd_dir, fmt='deepmd/raw')
    ls.to_deepmd_raw(combined_dir)
    ls.to_deepmd_npy(combined_dir)
    
    return combined_dir

def get_subfolder_name(folder_path):
    """Get the unique subfolder name under the folder"""
    items = os.listdir(folder_path)
    subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    
    if len(subfolders) == 0:
        raise ValueError(f"No subfolders found in {folder_path}")
    elif len(subfolders) > 1:
        print(f"Warning: Multiple subfolders found in {folder_path}: {subfolders}, will use the first one: {subfolders[0]}")
    
    return subfolders[0]

def delete_unwanted_files(folder_path):
    """Delete unwanted energy/force/virial files"""
    files_to_delete = ['energy.raw', 'force.raw', 'virial.raw']
    set_folder = os.path.join(folder_path, 'set.000')
    
    # Delete files in root directory
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    
    # Delete files in set.000 folder
    if os.path.exists(set_folder):
        npy_files_to_delete = ['energy.npy', 'force.npy', 'virial.npy']
        for file_name in npy_files_to_delete:
            file_path = os.path.join(set_folder, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def copy_necessary_files(source_folder, target_folder):
    """Copy necessary files from source_folder to target_folder"""
    files_to_copy = ['box.raw', 'coord.raw', 'type_map.raw', 'type.raw']
    
    for file_name in files_to_copy:
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Copied: {source_path} -> {target_path}")
        else:
            print(f"Warning: Source file not found {source_path}")

def predict_with_deepmd_model():
    """Use frozen model to predict energy, force, and virial"""
    # Set paths
    deepmd_all_dir = 'deepmd_all'
    subfolder_name = get_subfolder_name(deepmd_all_dir)
    input_folder = os.path.join(deepmd_all_dir, subfolder_name)
    model_file = "frozen_model.pth"

    # Output directories
    output_npy_dir = os.path.join("dpdata", "deepmd_npy")
    output_raw_dir = os.path.join("dpdata", "deepmd_raw")
    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(output_raw_dir, exist_ok=True)

    # Load data
    data = DeepmdData(input_folder, set_prefix="set", shuffle_test=False, sort_atoms=False)
    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    data.add("force", 3, atomic=True, must=False, high_prec=False)
    data.add("virial", 9, atomic=False, must=False, high_prec=False)

    test_data = data.get_test()
    nframes = test_data["box"].shape[0]

    # Data preprocessing
    coords = test_data["coord"][:nframes].reshape(nframes, -1)
    box = test_data["box"][:nframes]
    if not data.pbc:
        box = None
    
    if data.mixed_type:
        atype = test_data["type"][:nframes].reshape(nframes, -1)
    else:
        atype = test_data["type"][0]

    # Load model and predict
    dp = DeepEval(model_file)
    ret = dp.eval(coords, box, atype, atomic=False, mixed_type=data.mixed_type)
    energy = ret[0].reshape(nframes, 1)
    force = ret[1].reshape(nframes, -1)
    virial = ret[2].reshape(nframes, 9)

    # Save as deepmd_npy format
    np.save(os.path.join(output_npy_dir, "coord.npy"), test_data["coord"])
    np.save(os.path.join(output_npy_dir, "box.npy"), test_data["box"])
    np.save(os.path.join(output_npy_dir, "type.npy"), test_data["type"])
    np.save(os.path.join(output_npy_dir, "energy.npy"), energy)
    np.save(os.path.join(output_npy_dir, "force.npy"), force)
    np.save(os.path.join(output_npy_dir, "virial.npy"), virial)

    # Save as deepmd_raw format
    np.savetxt(os.path.join(output_raw_dir, "coord.raw"),
               test_data["coord"].reshape(nframes, -1), fmt="%.8e")
    np.savetxt(os.path.join(output_raw_dir, "box.raw"),
               test_data["box"].reshape(nframes, -1), fmt="%.8e")
    np.savetxt(os.path.join(output_raw_dir, "type.raw"),
               test_data["type"].reshape(nframes, -1), fmt="%d")
    np.savetxt(os.path.join(output_raw_dir, "energy.raw"), energy, fmt="%.8e")
    np.savetxt(os.path.join(output_raw_dir, "force.raw"), force, fmt="%.8e")
    np.savetxt(os.path.join(output_raw_dir, "virial.raw"), virial, fmt="%.8e")

    print("DeepMD model prediction completed!")
    return input_folder, output_raw_dir

def main():
    # Step 1: Process all POSCAR files
    print("=== Step 1: Processing POSCAR files ===")
    poscars_folder = 'POSCARs'
    if not os.path.isdir(poscars_folder):
        print(f"Error: Directory {poscars_folder} not found")
        return

    processed_frames_dir = 'processed_frames'
    os.makedirs(processed_frames_dir, exist_ok=True)
    
    poscar_files = [f for f in os.listdir(poscars_folder) 
                    if f.startswith('POSCAR_') and os.path.isfile(os.path.join(poscars_folder, f))]
    
    if not poscar_files:
        print("No structure files starting with 'POSCAR_' found")
        return
    
    processed_dirs = []
    
    for poscar_file in poscar_files:
        poscar_path = os.path.join(poscars_folder, poscar_file)
        base_name = os.path.splitext(poscar_file)[0]
        out_dir = os.path.join(processed_frames_dir, base_name)
        try:
            combined_dir = process_poscar(poscar_path, out_dir)
            print(f"Processed {poscar_file} successfully, files generated at {combined_dir}")
            processed_dirs.append(combined_dir)
        except Exception as e:
            print(f"Failed to process {poscar_file}: {e}")
    
    if not processed_dirs:
        print("No files processed successfully, dataset merging failed.")
        return
    
    # Merge all frame data
    ms = dpdata.MultiSystems()
    for d in processed_dirs:
        try:
            ls = dpdata.LabeledSystem(d, fmt='deepmd/raw')
            if len(ls) > 0:
                ms.append(ls)
            else:
                print(f"Dataset in directory {d} is empty, skipping.")
        except Exception as e:
            print(f"Unable to load data from directory {d}: {e}")
    
    merged_dir = 'deepmd_all'
    ms.to_deepmd_raw(merged_dir)
    ms.to_deepmd_npy(merged_dir)
    print("All datasets merged, results saved in deepmd_all folder")
    
    # Step 2: Delete unwanted files
    print("\n=== Step 2: Deleting unwanted files ===")
    subfolder_name = get_subfolder_name(merged_dir)
    deepmd_subfolder = os.path.join(merged_dir, subfolder_name)
    delete_unwanted_files(deepmd_subfolder)
    
    # Step 3: Use DeepMD model for prediction
    print("\n=== Step 3: Using DeepMD model for prediction ===")
    source_folder, target_folder = predict_with_deepmd_model()
    
    # Step 4: Copy necessary files
    print("\n=== Step 4: Copying necessary files ===")
    copy_necessary_files(source_folder, target_folder)
    
    # Step 5: Convert to final format
    print("\n=== Step 5: Converting to final format ===")
    ls = dpdata.LabeledSystem(target_folder, fmt='deepmd/raw')
    ls.to_deepmd_raw('deepmd_mace_make')
    ls.to_deepmd_npy('deepmd_mace_make')
    print("Final data saved to deepmd_mace_make folder")
    
    print("\n=== All steps completed! ===")

if __name__ == "__main__":
    main()
