# pair_potentials.py - REFACTORED FOR "Generic Attraction, Specific Repulsion"

# pair_potentials.py
import numpy as np


def set_pair_potentials_params(job, morse, table, tab_strong, tab_weak):
    species = sorted(job.sp.monomer_counts.keys())
    middle = [f"{s}M" for s in species]

    # (Optional) clear lingering params if these objects are reused
    for obj in (morse, table):
        try:
            obj.params.clear()
            obj.r_cut.clear()
        except AttributeError:
            pass

    # --- Generic attraction: P1–P2 (both orders) ---
    D0, alpha, r0, rc = job.sp.D_attractive, job.sp.alpha, job.sp.r0, job.sp.r_cut
    morse.params[("P1", "P2")] = dict(D0=D0, alpha=alpha, r0=r0)
    morse.r_cut[("P1", "P2")] = rc
    morse.params[("P2", "P1")] = dict(D0=D0, alpha=alpha, r0=r0)
    morse.r_cut[("P2", "P1")] = rc

    # --- Specific repulsion among middle beads only ---
    U_s, F_s = np.asarray(tab_strong[0]), np.asarray(tab_strong[1])  # STRONG
    U_w, F_w = np.asarray(tab_weak[0]), np.asarray(tab_weak[1])  # WEAK
    rc_rep = job.sp.rep_r_cut

    n = len(species)
    for i, si in enumerate(species):
        mi = f"{si}M"
        for j, sj in enumerate(species):
            mj = f"{sj}M"
            if i == j:
                # same-species => STRONG
                U, F = U_s, F_s
            elif abs(i - j) == 1:
                # consecutive => WEAK
                U, F = U_w, F_w
            else:
                # non-consecutive => STRONG
                U, F = U_s, F_s

            table.params[(mi, mj)] = dict(r_min=0, U=U, F=F)
            table.r_cut[(mi, mj)] = rc_rep

    # Nothing else is defined:
    # - cores A,B,C,... are inert to everyone
    # - P1–P1, P2–P2, patch–middle, patch–core remain undefined (no force)

    # Nothing else is set:
    # - cores (A, B, …) are inert to all
    # - P1–P1, P2–P2, patch–middle, patch–core all remain undefined (no force)


'''
def set_pair_potentials_params(job, morse, table, table_strong, table_weak):
    """
    Configures potentials for a model with generic attraction and specific repulsion.

    Rules:
    - Morse potential is generic between P1 and P2 patches.
    - Table potential is specific for middle particles (AM, BM, etc.):
        - Weak repulsion for CONSECUTIVE species (e.g., AM-BM).
        - Strong repulsion for NON-CONSECUTIVE species (e.g., AM-CM).
    """
    species_names = sorted(job.sp.monomer_counts.keys())

    # === MODIFICATION 1: Define types for the new memory-efficient scheme ===
    core_types = list(species_names)
    # Patches are generic, middle particles are species-specific
    middle_types = [f'{s}M' for s in species_names]
    all_types = core_types + ['P1', 'P2'] + middle_types
    # =======================================================================

    # Create a map from species name to its index (0, 1, 2...)
    # This is crucial for checking if species are "consecutive".
    species_to_idx = {name: i for i, name in enumerate(species_names)}

    # Get all unique pairs of types to loop over
    all_type_pairs = list(itertools.combinations_with_replacement(all_types, 2))

    for type_i, type_j in all_type_pairs:
        pair = (type_i, type_j)

        # Default all interactions to OFF
        morse.params[pair] = dict(D0=0, alpha=1, r0=0)
        morse.r_cut[pair] = 0
        table.params[pair] = dict(r_min=0, U=[0, 0], F=[0, 0])
        table.r_cut[pair] = 0

        # --- RULE 1: Generic Attraction for P1-P2 pairs ---
        if (type_i == 'P1' and type_j == 'P2') or \
           (type_i == 'P2' and type_j == 'P1'):
            morse.params[pair] = dict(D0=job.sp.D_attractive, alpha=job.sp.alpha, r0=job.sp.r0)
            morse.r_cut[pair] = job.sp.r_cut

        # --- RULE 2: Specific Repulsion for Middle-Middle pairs ---
        if type_i.endswith('M') and type_j.endswith('M'):
            species_i = type_i[:-1]
            species_j = type_j[:-1]
            
            # Skip self-interaction (e.g., AM-AM)
            if species_i == species_j:
                continue

            idx_i = species_to_idx[species_i]
            idx_j = species_to_idx[species_j]
            
            # Check if the species are consecutive in the alphabet
            if abs(idx_i - idx_j) == 1:
                # Apply WEAK repulsion for consecutive species
                table.params[pair] = dict(r_min=0, U=table_weak[0], F=table_weak[1])
                table.r_cut[pair] = job.sp.rep_r_cut
            else:
                # Apply STRONG repulsion for non-consecutive species
                table.params[pair] = dict(r_min=0, U=table_strong[0], F=table_strong[1])
                table.r_cut[pair] = job.sp.rep_r_cut
'''
