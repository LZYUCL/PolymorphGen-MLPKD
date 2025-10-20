# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import os
import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import simps
from ase.io import read, write
from ase.neighborlist import NeighborList
from scipy.special import sph_harm
import logging
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------
# Utility functions: Generate target points
# -------------------------

def distribute_points_on_sides(Xmin, Ymin, Xmax, Ymax, points_per_side):
    """
    Evenly distribute points on the four sides of a rectangle, ensuring the same number of points on each side.
    """
    points = []

    # Side 1: Bottom edge, from (Xmin, Ymin) to (Xmax, Ymin)
    for i in range(points_per_side):
        if points_per_side == 1:
            x = Xmin
        else:
            x = Xmin + i * (Xmax - Xmin) / (points_per_side -1)
        y = Ymin
        points.append( (round(x, 5), round(y, 5)) )

    # Side 2: Right edge, from (Xmax, Ymin) to (Xmax, Ymax)
    for i in range(1, points_per_side -1):
        y = Ymin + i * (Ymax - Ymin) / (points_per_side -1)
        x = Xmax
        points.append( (round(x, 5), round(y, 5)) )

    # Side 3: Top edge, from (Xmax, Ymax) to (Xmin, Ymax)
    for i in range(points_per_side):
        if points_per_side ==1:
            x = Xmax
        else:
            x = Xmax - i * (Xmax - Xmin) / (points_per_side -1)
        y = Ymax
        points.append( (round(x, 5), round(y, 5)) )

    # Side 4: Left edge, from (Xmin, Ymax) to (Xmin, Ymin)
    for i in range(1, points_per_side -1):
        y = Ymax - i * (Ymax - Ymin) / (points_per_side -1)
        x = Xmin
        points.append( (round(x, 5), round(y, 5)) )

    return points

def save_points_to_txt(points, filename, sS_min, sS_max, q6_min, q6_max):
    """
    Save distributed point coordinates to txt file in specified format.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for point in points:
            f.write("{\n")
            f.write(f"    'target_point': {point},\n")
            f.write("    'target_areas': [\n")
            f.write(f"        ('q6', {q6_min}, {q6_max}),\n")
            f.write(f"        ('sS', {sS_min}, {sS_max}),\n")
            f.write("    ]\n")
            f.write("},\n")

def plot_rectangle_and_points(Xmin, Ymin, Xmax, Ymax, points, output_image):
    """
    Plot rectangle boundaries and distributed points, and save as PNG image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot rectangle boundaries
    rectangle_x = [Xmin, Xmax, Xmax, Xmin, Xmin]
    rectangle_y = [Ymin, Ymin, Ymax, Ymax, Ymin]
    ax.plot(rectangle_x, rectangle_y, 'b-', label='Rectangle Boundary')

    # Separate X and Y coordinates of points
    if points:
        points_x, points_y = zip(*points)
        ax.scatter(points_x, points_y, color='red', label='Points')
        for idx, (x, y) in enumerate(points):
            ax.annotate(f"{idx+1}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Set title and labels
    ax.set_title('Rectangle Boundary and Distributed Points')
    ax.set_xlabel('sS')
    ax.set_ylabel('Q6')
    ax.legend()

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Set axis range slightly larger than rectangle
    padding_x = (Xmax - Xmin) * 0.05
    padding_y = (Ymax - Ymin) * 0.05
    ax.set_xlim(Xmin - padding_x, Xmax + padding_x)
    ax.set_ylim(Ymin - padding_y, Ymax + padding_y)

    # Save as PNG image
    plt.savefig(output_image, dpi=300)
    plt.close()
    logging.info(f"Saved rectangle and points plot to '{output_image}'")

# -------------------------
# Step 1: Define sS and Q6 calculation functions
# -------------------------

def read_poscar(file_path: str):
    """
    Read POSCAR file and return structure object.
    """
    try:
        structure = read(file_path, format='vasp')
        return structure
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return None

def calculate_g_r(structure, r_max: float, bins: int, sigma: float = 0.0125):
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
    # Normalization
    g_r /= (4 * np.pi * r**2 * dr * num_atoms)

    return r, g_r

def calculate_entropy(r: np.ndarray, g_r: np.ndarray, density: float):
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
    cutoff = 4.2  # Unit: Angstrom
    # Create neighbor list
    cutoffs = [cutoff / 2.0] * num_atoms
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(structure)

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

    Q6_avg = np.mean(Q6_values)
    return Q6_avg

def process_poscar(file_path: str):
    """
    Process a single POSCAR file, calculate sS and Q6.
    """
    try:
        structure = read_poscar(file_path)
        if structure is None:
            return None
        volume = structure.get_volume()
        num_atoms = len(structure)
        density = num_atoms / volume

        r, g_r = calculate_g_r(structure, r_max=8.0, bins=100, sigma=0.0125)
        sS = calculate_entropy(r, g_r, density)
        Q6_avg = calculate_Q6(structure)

        file_name = os.path.basename(file_path)
        poscar_identifier = file_name
        logging.info(f"Processed {poscar_identifier}: sS = {sS:.5f}, Q6 = {Q6_avg:.5f}")

        return poscar_identifier, sS, Q6_avg, structure
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

def write_sS_q6_to_txt(identifiers, sS_values, q6_values, filename='poscar_sS_q6.txt'):
    """
    Write calculated sS and Q6 values to txt file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Identifier sS Q6\n")
        for ident, sS, q6 in zip(identifiers, sS_values, q6_values):
            f.write(f"{ident} {sS} {q6}\n")
    logging.info(f"sS/Q6 values written to {filename}.")

def read_sS_q6_from_txt(filename='poscar_sS_q6.txt'):
    """
    Read sS and Q6 values from txt file. Return None if file doesn't exist or format is incorrect.
    """
    if not os.path.exists(filename):
        return None
    identifiers, sS_values, q6_values = [], [], []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Skip first header line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            ident, sS, q6 = parts[0], float(parts[1]), float(parts[2])
            identifiers.append(ident)
            sS_values.append(sS)
            q6_values.append(q6)
        logging.info(f"Read {len(identifiers)} sS/Q6 records from {filename}.")
        return identifiers, sS_values, q6_values
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return None

def compute_existing_sS_q6(poscar_dir: str, num_workers: int = 40):
    """
    Calculate sS and Q6 values for existing POSCAR files. If txt file exists, read directly from it, otherwise calculate and store to txt.
    """
    txt_file = 'poscar_sS_q6.txt'
    read_result = read_sS_q6_from_txt(txt_file)
    if read_result is not None:
        identifiers, sS_values, q6_values = read_result
        # No need to calculate structure objects at this point, load via POSCAR files
        structures = []
        for ident in identifiers:
            file_path = os.path.join(poscar_dir, ident)
            st = read_poscar(file_path)
            if st is not None:
                structures.append(st)
            else:
                # Skip if file doesn't exist or read fails
                pass
        # Maintain consistent return format
        return identifiers, sS_values, q6_values, structures

    # If no txt file or read fails, recalculate
    poscar_files = [os.path.join(poscar_dir, f) for f in os.listdir(poscar_dir) if f.startswith("POSCAR")]
    logging.info(f"Found {len(poscar_files)} POSCAR files to process.")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_poscar, poscar_files):
            if result is not None:
                results.append(result)

    identifiers = []
    sS_values = []
    q6_values = []
    structures = []

    for identifier, sS, Q6_avg, structure in results:
        identifiers.append(identifier)
        sS_values.append(sS)
        q6_values.append(Q6_avg)
        structures.append(structure)

    write_sS_q6_to_txt(identifiers, sS_values, q6_values, txt_file)
    logging.info("Existing structures' sS and Q6 calculation completed.")
    return identifiers, sS_values, q6_values, structures

# -------------------------
# Step 2: Genetic algorithm implementation
# -------------------------

def initialize_population(structures: List, population_size: int, is_mutate=False):
    initial_population = []
    num_structures = len(structures)
    for i in range(population_size):
        parent = copy.deepcopy(random.choice(structures))
        if is_mutate:
            offspring = mutate_structure(parent, mutation_rate=0.1)  # Increase mutation rate
            if is_physical(offspring):
                initial_population.append(offspring)
        else:
            initial_population.append(parent)
    logging.info(f"Initial population of size {population_size} generated.")
    return initial_population

def fitness_function(offspring_sS, offspring_q6, target_point: Tuple[float, float]):
    target_sS, target_q6 = target_point
    distance = np.sqrt((target_sS - offspring_sS) ** 2 + (target_q6 - offspring_q6) ** 2)
    fitness = 1 / (distance + 1e-6)
    return fitness

def select_parents(population: List, fitnesses: List[float]):
    if len(fitnesses) != len(population):
        raise ValueError(f"Population size ({len(population)}) doesn't match fitness list size ({len(fitnesses)})")
    min_fitness = min(fitnesses)
    if min_fitness <= 0:
        fitnesses = [max(f, 1e-6) for f in fitnesses]
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    selection_probs = [f / total_fitness for f in fitnesses]
    parent = random.choices(population, weights=selection_probs, k=1)[0]
    return parent

def mutate_structure(structure, mutation_rate: float):
    mutated_structure = copy.deepcopy(structure)

    # Lattice perturbation
    scale_min = 0.90
    scale_max = 1.10
    scale_factor = random.uniform(scale_min, scale_max)
    mutated_structure.set_cell(mutated_structure.get_cell() * scale_factor, scale_atoms=True)
    shear_angle_deg = random.uniform(-5.0, 5.0)
    shear_angle_rad = math.radians(shear_angle_deg)
    shear = math.tan(shear_angle_rad)
    cell = mutated_structure.get_cell().array.copy()
    cell[1][0] += shear * cell[0][0]
    cell[1][1] += shear * cell[0][1]
    cell[1][2] += shear * cell[0][2]
    mutated_structure.set_cell(cell, scale_atoms=True)

    # Atomic position perturbation
    positions = mutated_structure.get_positions()
    for i in range(len(positions)):
        if random.random() < mutation_rate:
            displacement = np.random.normal(0, 0.1, size=3)
            positions[i] += displacement
    mutated_structure.set_positions(positions)

    if not is_physical(mutated_structure):
        return copy.deepcopy(structure)

    return mutated_structure

def is_physical(structure):
    min_distance_threshold = 2.0
    distances = structure.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    min_distance = np.min(distances)
    if min_distance < min_distance_threshold:
        return False
    else:
        return True

def filter_population_by_target(population: List, sS_q6_values: List[Tuple[float, float]],
                                target_areas: List[Tuple[str, float, float]]):
    filtered_population = []
    filtered_sS_q6_values = []

    for individual, (sS, q6) in zip(population, sS_q6_values):
        keep = True
        for condition, min_value, max_value in target_areas:
            if condition == 'q6' and not (min_value <= q6 <= max_value):
                keep = False
                break
            elif condition == 'sS' and not (min_value <= sS <= max_value):
                keep = False
                break
        if keep:
            filtered_population.append(individual)
            filtered_sS_q6_values.append((sS, q6))

    return filtered_population, filtered_sS_q6_values

# -------------------------
# Step 3: Run genetic algorithm
# -------------------------

def genetic_algorithm(existing_structures: List, existing_sS_q6: np.ndarray, target_point: Tuple[float, float],
                     target_areas: List[Tuple[str, float, float]], generations: int = 10,
                     population_size: int = 50, target_idx: int = 1):
    output_dir = f"./png_out/target_{target_idx}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    population = initialize_population(existing_structures, population_size, is_mutate=False)
    all_sS_q6 = existing_sS_q6.tolist()
    all_structures = existing_structures.copy()

    for gen in range(generations):
        logging.info(f"\nTarget {target_idx} - Generation {gen + 1} started.")
        fitnesses = []
        sS_q6_values = []
        new_structures = []
        new_structures_sS_q6_values = []

        for idx_pop, individual in enumerate(population):
            try:
                volume = individual.get_volume()
                num_atoms = len(individual)
                density = num_atoms / volume
                r, g_r = calculate_g_r(individual, r_max=8.0, bins=100, sigma=0.0125)
                sS = calculate_entropy(r, g_r, density)
                Q6_avg = calculate_Q6(individual)
                fitness = fitness_function(sS, Q6_avg, target_point)
                fitnesses.append(fitness)
                sS_q6_values.append((sS, Q6_avg))
                logging.info(f"Target {target_idx} - Gen {gen + 1} - Ind {idx_pop}: Fit={fitness:.5f}, sS={sS:.5f}, Q6={Q6_avg:.5f}")
            except Exception as e:
                logging.error(f"Error in Target {target_idx} - Generation {gen + 1} - Individual {idx_pop}: {e}")
                fitnesses.append(0)
                sS_q6_values.append((0, 0))

        # Increase diversity
        diversity_population_size = int(0.3 * population_size)
        random_new_individuals = initialize_population(existing_structures, diversity_population_size, is_mutate=True)
        new_structures.extend(random_new_individuals)
        population.extend(random_new_individuals)

        for individual in random_new_individuals:
            try:
                volume = individual.get_volume()
                num_atoms = len(individual)
                density = num_atoms / volume
                r, g_r = calculate_g_r(individual, r_max=8.0, bins=100, sigma=0.0125)
                sS = calculate_entropy(r, g_r, density)
                Q6_avg = calculate_Q6(individual)
                fitness = fitness_function(sS, Q6_avg, target_point)
                fitnesses.append(fitness)
                new_structures_sS_q6_values.append((sS, Q6_avg))
                sS_q6_values.append((sS, Q6_avg))
            except Exception as e:
                logging.error(f"Error in Target {target_idx} - Generation {gen + 1} - New Individual: {e}")
                fitnesses.append(0)
                new_structures_sS_q6_values.append((0, 0))
                sS_q6_values.append((0, 0))

        if fitnesses:
            max_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            logging.info(f"Target {target_idx} - Gen {gen + 1} - Max Fitness: {max_fitness:.5f}, Avg Fitness: {avg_fitness:.5f}")
        else:
            logging.info(f"Target {target_idx} - Gen {gen + 1} - No fitnesses calculated.")

        new_population = []
        attempts = 0
        max_attempts = population_size * 10
        while len(new_population) < population_size and attempts < max_attempts:
            attempts += 1
            parent = select_parents(population, fitnesses)
            child = mutate_structure(parent, mutation_rate=0.3)
            if is_physical(child):
                new_structures.append(child)
                new_population.append(child)

        for individual in new_population:
            try:
                volume = individual.get_volume()
                num_atoms = len(individual)
                density = num_atoms / volume
                r, g_r = calculate_g_r(individual, r_max=8.0, bins=100, sigma=0.0125)
                sS = calculate_entropy(r, g_r, density)
                Q6_avg = calculate_Q6(individual)
                fitness = fitness_function(sS, Q6_avg, target_point)
                fitnesses.append(fitness)
                new_structures_sS_q6_values.append((sS, Q6_avg))
                sS_q6_values.append((sS, Q6_avg))
            except Exception as e:
                logging.error(f"Error in Target {target_idx} - Generation {gen + 1} - New Pop Individual: {e}")
                fitnesses.append(0)
                new_structures_sS_q6_values.append((0, 0))
                sS_q6_values.append((0, 0))

        new_population, new_sS_q6_values = filter_population_by_target(new_population, sS_q6_values, target_areas)
        new_structures, _ = filter_population_by_target(new_structures, new_structures_sS_q6_values, target_areas)
        population = new_population
        all_structures.extend(population)

        visualize_population_diff(all_sS_q6, new_sS_q6_values, gen + 1, output_dir, target_areas, target_point)
        all_sS_q6.extend(new_sS_q6_values)
        save_new_structures(new_structures, gen + 1, target_idx)

        logging.info(f"Target {target_idx} - Generation {gen + 1} completed.")

    return all_structures, all_sS_q6

# -------------------------
# Step 4: Visualization and result saving
# -------------------------

def visualize_population_diff(last_sS_q6_values: List[Tuple[float, float]], 
                              new_sS_q6_values: List[Tuple[float, float]], 
                              generation: int, 
                              output_dir: str, 
                              target_areas: List[Tuple[str, float, float]], 
                              target_point: Tuple[float, float]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(last_sS_q6_values) == 0:
        last_sS_values, last_q6_values = [], []
    else:
        last_sS_values, last_q6_values = zip(*last_sS_q6_values)
    if len(new_sS_q6_values) == 0:
        new_sS_values, new_q6_values = [], []
    else:
        new_sS_values, new_q6_values = zip(*new_sS_q6_values)

    plt.figure(figsize=(8, 6))
    if last_sS_values and last_q6_values:
        plt.scatter(last_sS_values, last_q6_values, c='blue', label='Last Generation', alpha=0.6)
    if new_sS_values and new_q6_values:
        plt.scatter(new_sS_values, new_q6_values, c='red', label='Current Generation', alpha=0.6)

    xmin, xmax, ymin, ymax = None, None, None, None
    for area in target_areas:
        if area[0] == 'q6':
            ymin, ymax = area[1], area[2]
        elif area[0] == 'sS':
            xmin, xmax = area[1], area[2]

    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        plt.plot([xmin, xmax], [ymin, ymin], color='green', linestyle='--')
        plt.plot([xmin, xmax], [ymax, ymax], color='green', linestyle='--')
        plt.plot([xmin, xmin], [ymin, ymax], color='green', linestyle='--')
        plt.plot([xmax, xmax], [ymin, ymax], color='green', linestyle='--')

    target_x, target_y = target_point
    plt.scatter(target_x, target_y, c='purple', label='Target Point', s=100, edgecolor='black', zorder=5)

    plt.title(f'Generation {generation}')
    plt.xlabel(r'$\mathit{s_S}$')
    plt.ylabel(r'$\mathit{Q_6}$')
    plt.legend()

    output_path = os.path.join(output_dir, f'generation_{generation}.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"sS-Q6 scatter plot for Generation {generation} saved at {output_path}")

def save_new_structures(population: List, generation: int, target_idx: int):
    output_dir = f'./structures/target_{target_idx}_generation_{generation}_structures'
    output_all = f'./all_generation_structures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_all):
        os.makedirs(output_all)
    for idx, individual in enumerate(population):
        filename1 = os.path.join(output_dir, f'POSCAR_target_{target_idx}_gen{generation}_{idx}.vasp')
        filename2 = os.path.join(output_all, f'POSCAR_target_{target_idx}_gen{generation}_{idx}.vasp')
        try:
            write(filename1, individual, format='vasp')
            write(filename2, individual, format='vasp')
        except Exception as e:
            logging.error(f"Failed to save structure {idx} in target {target_idx}, generation {generation}: {e}")
    logging.info(f"Target {target_idx} - Generation {generation} structures saved in directory '{output_dir}'.")

# -------------------------
# Step 5: Main program entry
# -------------------------

import os
import logging
import numpy as np

# Assume the following functions are already defined or imported
# compute_existing_sS_q6(), save_points_to_txt(), plot_rectangle_and_points(), genetic_algorithm()

def main():
    poscar_dir = "./POSCARs"
    num_workers = 40
    generations = 20
    population_size = 50

    # Check if POSCAR directory exists
    if not os.path.exists(poscar_dir):
        logging.error(f"POSCAR directory '{poscar_dir}' does not exist.")
        return

    # Calculate or read sS and Q6 values for existing structures
    logging.info("Starting to compute or read sS and Q6 for existing structures...")
    identifiers, sS_values, q6_values, structures = compute_existing_sS_q6(poscar_dir, num_workers=num_workers)
    if len(structures) == 0:
        logging.error("No valid structures found. Exiting.")
        return

    # Determine target area based on existing structures' sS and Q6
    sS_array = np.array(sS_values)
    q6_array = np.array(q6_values)
    sS_min, sS_max = np.min(sS_array), np.max(sS_array)
    q6_min, q6_max = np.min(q6_array), np.max(q6_array)

    # Optionally expand the range (set to 0 times here, keep original range), can be adjusted as needed
    sS_range = sS_max - sS_min
    q6_range = q6_max - q6_min
    sS_min_expanded = sS_min - 0.1 * sS_range
    sS_max_expanded = sS_max + 0.1 * sS_range
    q6_min_expanded = q6_min - 0.1 * q6_range
    q6_max_expanded = q6_max + 0.1 * q6_range

    # --- New logic: Change to manually defined target points ---
    # Below shows three example points (sS, Q6), you can modify according to your needs
    points = [
        (-18,0.18)
    ]

    # Save these target points to TXT file
    save_points_to_txt(points, 'points.txt',
                       sS_min_expanded, sS_max_expanded,
                       q6_min_expanded, q6_max_expanded)

    # Plot rectangle area and target point distribution
    plot_rectangle_and_points(sS_min_expanded, q6_min_expanded,
                              sS_max_expanded, q6_max_expanded,
                              points, 'rectangle_points.png')

    # Prepare target area information (example remains unchanged, can be adjusted as needed)
    target_areas = [
        ('q6', q6_min_expanded, q6_max_expanded),
        ('sS', sS_min_expanded, sS_max_expanded),
    ]

    # Assemble final targets
    targets = []
    for pt in points:
        targets.append({
            'target_point': pt,
            'target_areas': target_areas
        })

    # Run genetic algorithm for each target sequentially
    all_structures = structures.copy()
    all_sS_q6 = list(zip(sS_values, q6_values))

    for idx, target in enumerate(targets):
        logging.info(f"\nStarting Genetic Algorithm for Target {idx + 1}: {target['target_point']}")
        all_structures, all_sS_q6 = genetic_algorithm(
            existing_structures=all_structures,
            existing_sS_q6=np.array(all_sS_q6),
            target_point=target['target_point'],
            target_areas=target['target_areas'],
            generations=generations,
            population_size=population_size,
            target_idx=idx + 1
        )

    logging.info("All GA runs completed.")

if __name__ == "__main__":
    main()
