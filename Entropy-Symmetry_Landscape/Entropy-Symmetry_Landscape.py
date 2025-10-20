# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.integrate import simps
from ase.io import read
from ase.neighborlist import NeighborList
from scipy.special import sph_harm

def read_poscar(file_path):
    """
    Read POSCAR file and return structure object.
    """
    structure = read(file_path, format='vasp')
    return structure

def calculate_g_r(structure, r_max, bins, sigma=0.0125):
    """
    Calculate smoothed radial distribution function g(r).
    """
    positions = structure.get_positions()
    cell = structure.get_cell()
    r_edges = np.linspace(0, r_max, bins + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r[1] - r[0]

    distances = []
    num_atoms = len(positions)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            delta = positions[j] - positions[i]
            # Consider periodic boundary conditions
            delta = delta - np.round(delta @ np.linalg.inv(cell)) @ cell
            dist = np.linalg.norm(delta)
            if dist < r_max:
                distances.append(dist)

    distances = np.array(distances)

    # Use Gaussian smoothing for distance distribution
    g_r = np.zeros_like(r)
    for d in distances:
        g_r += np.exp(-0.5 * ((r - d) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    g_r /= (4 * np.pi * r**2 * dr * num_atoms)  # Normalization

    return r, g_r

def calculate_entropy(r, g_r, density):
    """
    Calculate sS entropy value.
    """
    integrand = g_r * np.log(g_r + 1e-10) - g_r + 1
    sS = -2 * np.pi * density * simps(integrand * r**2, r)
    return sS

def calculate_Q6(structure):
    """
    Calculate average Q6 parameter.
    """
    num_atoms = len(structure)
    positions = structure.get_positions()
    cell = structure.get_cell()
    # Define cutoff radius, adjust according to system (usually nearest neighbor distance)
    cutoff = 3.5  # Unit: Angstrom
    # Create neighbor list
    cutoffs = [cutoff / 2.0] * num_atoms
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(structure)  # Pass structure object, not positions and cell

    Q6_values = []
    l = 6  # Order of spherical harmonics

    for i in range(num_atoms):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) == 0:
            # Skip if no neighbors to avoid division by zero
            Q6_values.append(0)
            continue

        neighbors = positions[indices] + np.dot(offsets, cell)

        # Calculate vectors from atom i to its neighbors
        vecs = neighbors - positions[i]
        # Convert to spherical coordinates
        r_vec = np.linalg.norm(vecs, axis=1)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.arccos(vecs[:, 2] / r_vec)
            phi = np.arctan2(vecs[:, 1], vecs[:, 0])

        # Handle possible NaN values
        theta = np.nan_to_num(theta)
        phi = np.nan_to_num(phi)

        # Calculate spherical harmonics
        qlm = np.array([sph_harm(m, l, phi, theta) for m in range(-l, l+1)])
        # Average over neighbors
        qlm_avg = np.sum(qlm, axis=1) / len(neighbors)
        # Calculate Q6 value
        Q6_i = np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm_avg)**2))
        Q6_values.append(Q6_i)

    # Return Q6 average for the entire structure
    Q6_avg = np.mean(Q6_values)
    return Q6_avg

def process_poscar(file_path):
    """
    Process a single POSCAR file, calculate sS and Q6.
    """
    try:
        # Read POSCAR
        structure = read_poscar(file_path)
        volume = structure.get_volume()
        num_atoms = len(structure)
        density = num_atoms / volume

        # Calculate radial distribution function and sS entropy
        r, g_r = calculate_g_r(structure, r_max=8.0, bins=100, sigma=0.0125)
        sS = calculate_entropy(r, g_r, density)

        # Calculate Q6 parameter
        Q6_avg = calculate_Q6(structure)

        # Extract number or identifier
        file_name = os.path.basename(file_path)
        poscar_identifier = file_name.split('_', 1)[-1]  # Extract part after POSCAR_

        return poscar_identifier, sS, Q6_avg
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main(poscar_dir, output_file="results_r_8_modify.txt", num_workers=20):
    """
    Main program: Process POSCAR files in parallel, calculate sS and Q6.
    """
    # Find all POSCAR files
    poscar_files = [os.path.join(poscar_dir, f) for f in os.listdir(poscar_dir) if f.startswith("POSCAR_")]

    # Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_poscar, poscar_files):
            if result is not None:
                results.append(result)

    # Sort by number or identifier
    results.sort(key=lambda x: x[0])

    # Save to file
    with open(output_file, 'w') as f:
        f.write("Identifier sS Q6\n")
        for identifier, sS, Q6_avg in results:
            f.write(f"{identifier} {sS:.5f} {Q6_avg:.5f}\n")

    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    poscar_dir = "./all_generation_structures"  # POSCAR directory path
    main(poscar_dir, num_workers=40)
