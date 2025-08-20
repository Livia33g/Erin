# init.py - REFACTORED FOR "Generic Attraction, Specific Repulsion" MODEL

import signac
import numpy as np
import random
import math
import string

# Initialize the signac project in the current directory
project = signac.init_project()

# ===================================================================
# ### Central Configuration Section ###
# ===================================================================

# Define the number of monomer species. This is the main control.
num_species = 10  # For example, for A, B, ..., J
species_names = list(string.ascii_uppercase[:num_species])
print(f"Initializing for {num_species} species: {species_names}")

# === MODIFICATION 1: Define parameters for the new physical model ===
# ATTRACTION: Now generic, between P1 and P2 patches.
D_attractive = 3.0

# REPULSION: Now species-specific, with two strength levels.
# The ratio of repulsion strengths is matched to the old attraction ratio.
OLD_ATTRACTION_RATIO = 10.0 / 1.09

# Strong repulsion for non-consecutive species (e.g., AM-CM)
rep_A_strong = 500.0
# Weak repulsion for consecutive species (e.g., AM-BM), derived from the ratio.
rep_A_weak = rep_A_strong / OLD_ATTRACTION_RATIO
# ===================================================================

# System parameters
replicas = [1]
alpha = 5.0  # This alpha is for the Morse potential

# ... (File reading logic is unchanged) ...
parameter_file = "all_params.txt"
simulation_parameters = []
with open(parameter_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        values = line.split(",")
        try:
            if len(values) < (1 + num_species):
                print(
                    f"Warning: Skipping line with insufficient columns for {num_species} species: {line}"
                )
                continue
            kT_index = -(num_species + 1)
            kT = float(values[kT_index])
            concentration_values = [float(v) for v in values[-num_species:]]
            monomer_concentrations = dict(zip(species_names, concentration_values))
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping line due to error ({e}): {line}")
            continue
        simulation_parameters.append((monomer_concentrations, kT))

initial_total_volume = 10000000
max_iterations = 100
tolerance = 0.00005
for monomer_concentrations, kT in simulation_parameters:
    for replica in replicas:
        target_total_concentration = sum(monomer_concentrations.values())
        if target_total_concentration == 0:
            continue
        monomer_counts = {
            k: round(v * initial_total_volume)
            for k, v in monomer_concentrations.items()
        }
        monomer_counts = {
            k: max(1, v) if monomer_concentrations[k] > 0 else 0
            for k, v in monomer_counts.items()
        }
        current_volume = initial_total_volume
        for i in range(max_iterations):
            total_monomers = sum(monomer_counts.values())
            if total_monomers == 0:
                break
            computed_concentration = total_monomers / current_volume
            if abs(computed_concentration - target_total_concentration) <= tolerance:
                break
            if computed_concentration > 0:
                current_volume *= target_total_concentration / computed_concentration
        final_volume = current_volume
        final_side_length = round(final_volume ** (1 / 3), 2)

        # === MODIFICATION 2: Update the statepoint dictionary with new parameters ===
        sp = {
            # System Information
            "box_L": final_side_length,
            "seed": random.randint(1, 65535),
            "replica": replica,
            "concentration": target_total_concentration,
            "monomer_counts": monomer_counts,
            # Rule-based potential parameters for the new model
            "D_attractive": D_attractive,
            "rep_A_strong": rep_A_strong,
            "rep_A_weak": rep_A_weak,
            # Simulation Setup
            "kT": kT,
            "alpha": alpha,
            "equil_step": int(2e6),
            "run_step": int(2e8),
        }

        # Merge with the second dictionary, also ensuring integer types
        # Note: "rep_A" is removed as it's replaced by strong/weak versions
        sp_full = {
            **sp,
            **{
                "scale": 0.99,
                "a": 1.0,
                "b": 0.1,
                "r": 1.1,
                "separation": 2.0,
                "r0": 0.0,
                "r_cut": 8.0 / alpha,
                "rep_alpha": 2.5,
                "rep_r_min": 0.0,
                "rep_r_max": 2.0,
                "rep_r_cut": 6,
                "dt": 0.001,
                "tau": 1.0,
                "dump_period": int(1e2),
                "log_period": int(1e3),
            },
        }

        job = project.open_job(sp_full)
        job.init()
        print(f"Initialized job {job.id} for kT={kT}, replica={replica}")
