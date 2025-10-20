# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import simps
from ase.io import read
from ase.neighborlist import NeighborList
from scipy.special import sph_harm
import multiprocessing

def read_poscar(file_path):
    """
    Read POSCAR file and return ASE structure object.
    """
    structure = read(file_path, format='vasp')
    return structure

def calculate_g_r(structure, r_max=8.0, bins=100, sigma=0.0125):
    """
    Calculate smoothed radial distribution function g(r).
    r_max: maximum distance
    bins: number of equally spaced r intervals
    sigma: Gaussian smoothing parameter
    """
    positions = structure.get_positions()
    cell = structure.get_cell()
    r_edges = np.linspace(0, r_max, bins + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r[1] - r[0]

    distances = []
    num_atoms = len(positions)
    # Collect all atom pair distances (considering PBC)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            delta = positions[j] - positions[i]
            # PBC: minimum image vector
            delta -= np.round(delta @ np.linalg.inv(cell)) @ cell
            dist = np.linalg.norm(delta)
            if dist < r_max:
                distances.append(dist)
    distances = np.array(distances)

    # Apply Gaussian smoothing to distance distribution
    g_r = np.zeros_like(r)
    for d in distances:
        g_r += np.exp(-0.5 * ((r - d) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # Normalization
    g_r /= (4 * np.pi * r**2 * dr * num_atoms)

    return r, g_r

def calculate_entropy(r, g_r, density):
    """
    Calculate two-body entropy sS.
    sS = -2 * pi * rho * ∫ [g(r) ln g(r) - g(r) + 1] r^2 dr
    """
    integrand = g_r * np.log(g_r + 1e-10) - g_r + 1
    sS = -2 * np.pi * density * simps(integrand * r**2, r)
    return sS

def calculate_Q6(structure, cutoff):
    """
    Calculate average Q6 for structure at given cutoff.
    """
    num_atoms = len(structure)
    positions = structure.get_positions()
    cell = structure.get_cell()

    # ASE NeighborList requires cutoff for each atom, simply set to cutoff/2.0
    cutoffs = [cutoff / 2.0] * num_atoms
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(structure)

    Q6_values = []
    l = 6  # Order of spherical harmonics

    for i in range(num_atoms):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) == 0:
            Q6_values.append(0.0)
            continue

        # Calculate neighbor coordinates (considering PBC)
        neighbors = positions[indices] + np.dot(offsets, cell)
        vecs = neighbors - positions[i]
        r_vec = np.linalg.norm(vecs, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arccos(vecs[:, 2] / r_vec)
            phi = np.arctan2(vecs[:, 1], vecs[:, 0])
        theta = np.nan_to_num(theta)
        phi = np.nan_to_num(phi)

        qlm = np.array([sph_harm(m, l, phi, theta) for m in range(-l, l+1)])
        qlm_avg = np.sum(qlm, axis=1) / len(neighbors)
        Q6_i = np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm_avg)**2))
        Q6_values.append(Q6_i)

    return np.mean(Q6_values)

def compute_sS_for_structure(structure):
    """
    Calculate sS once for given structure.
    """
    volume = structure.get_volume()
    num_atoms = len(structure)
    density = num_atoms / volume
    r, g_r = calculate_g_r(structure, r_max=8.0, bins=100, sigma=0.0125)
    sS = calculate_entropy(r, g_r, density)
    return sS

def try_int(x):
    """
    Try to convert string x to int; return None if fails.
    """
    try:
        return int(x)
    except ValueError:
        return None

def compute_sS_item(args):
    """
    Parallel computation of sS for each structure. Input parameters: (structure_id, filename, file_path)
    """
    structure_id, filename, file_path = args
    try:
        structure = read_poscar(file_path)
        sS_val = compute_sS_for_structure(structure)
        return {"id": structure_id, "filename": filename, "structure": structure, "sS": sS_val}
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

def compute_Q6_item(args):
    """
    Parallel computation of Q6 for single structure at given cutoff.
    Input parameters: (structure, cutoff)
    """
    structure, cutoff = args
    try:
        q6 = calculate_Q6(structure, cutoff)
        return q6
    except Exception as e:
        print(f"Error calculating Q6: {e}")
        return 0.0

def main(poscar_dir="./all_generation_structures",
         output_prefix="results",
         cutoff_start=2.0,
         cutoff_end=3.0,
         cutoff_step=0.1):
    """
    Main program:
    1. Traverse all "POSCAR_number" files in poscar_dir, extract numeric IDs and sort.
    2. Read all structures in parallel and calculate sS once.
    3. For each cutoff in [cutoff_start, cutoff_end] range, calculate Q6 for all structures at that cutoff,
       plot (sS, Q6) scatter plot with colors representing structure ID gradient, and save images.
    """
    # 1. Find all files matching POSCAR_* pattern, extract IDs
    poscar_files = []
    for f in os.listdir(poscar_dir):
        if f.startswith("POSCAR_"):
            id_str = f.replace("POSCAR_", "")
            id_int = try_int(id_str)
            if id_int is not None:
                poscar_files.append((id_int, f, os.path.join(poscar_dir, f)))
            else:
                print(f"Warning: File {f} ID is not pure number, ignored.")

    if not poscar_files:
        print("No files matching 'POSCAR_number' pattern found, program terminated.")
        return

    poscar_files.sort(key=lambda x: x[0])

    # 2. Read all structures in parallel and calculate sS (40 cores)
    with multiprocessing.Pool(processes=40) as pool:
        results = pool.map(compute_sS_item, poscar_files)
    data_list = [res for res in results if res is not None]

    if not data_list:
        print("No structures successfully read, program terminated.")
        return

    # 3. Prepare cutoff range
    cutoffs = np.arange(cutoff_start, cutoff_end + 1e-8, cutoff_step)
    all_ids = [item["id"] for item in data_list]
    id_min, id_max = min(all_ids), max(all_ids)

    # For each cutoff, calculate Q6 for all structures in parallel using 40 cores, then plot
    for cutoff in cutoffs:
        with multiprocessing.Pool(processes=40) as pool:
            q6_results = pool.map(compute_Q6_item, [(item["structure"], cutoff) for item in data_list])
        sS_vals = [item["sS"] for item in data_list]
        ID_vals = [item["id"] for item in data_list]

        sS_array = np.array(sS_vals, dtype=float)
        Q6_array = np.array(q6_results, dtype=float)
        ID_array = np.array(ID_vals, dtype=float)

        # Plot scatter plot
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        sc = ax.scatter(
            sS_array, 
            Q6_array, 
            c=ID_array,
            cmap='viridis',
            vmin=id_min,
            vmax=id_max,
            s=50,
            alpha=0.8
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("POSCAR ID", fontsize=10)
        ax.set_xlabel("sS", fontsize=12)
        ax.set_ylabel("Q6", fontsize=12)
        ax.set_title(f"Cutoff = {cutoff:.2f}", fontsize=14)
        plt.tight_layout()
        png_name = f"{output_prefix}_cutoff_{cutoff:.2f}.png"
        plt.savefig(png_name, dpi=300)
        plt.close()
        print(f"Scatter plot saved: {png_name}")

    print("All cutoff range calculations and plotting completed.")

if __name__ == "__main__":
    poscar_dir = "./POSCARs/"
    main(
        poscar_dir=poscar_dir,
        output_prefix="results",
        cutoff_start=3,
        cutoff_end=4.7,
        cutoff_step=0.1
    )
