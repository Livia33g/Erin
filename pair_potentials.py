import numpy as np


def set_pair_potentials_params(job, morse, lj_repulsion):
    """
    Sets pair potentials for Morse (attraction) and LJ (repulsion).
    """
    species = sorted(job.sp.monomer_counts.keys())

    # ----- DEFAULTS (all interactions off) -----
    morse.params.default = dict(D0=0.0, alpha=1.0, r0=float(job.sp.r0))
    morse.r_cut.default = 0.0

    # For LJ, epsilon=0 turns off the interaction. sigma is just a placeholder.
    lj_repulsion.params.default = dict(epsilon=0.0, sigma=1.0)
    lj_repulsion.r_cut.default = 0.0

    # ----- Morse: P1–P2 attraction -----
    D0, alpha, r0, rc_morse = (
        float(job.sp.D_attractive),
        float(job.sp.alpha),
        float(job.sp.r0),
        float(job.sp.r_cut),
    )
    morse.params[("P1", "P2")] = dict(D0=D0, alpha=alpha, r0=r0)
    morse.r_cut[("P1", "P2")] = rc_morse

    # ----- LJ Repulsion: Middle-Middle interactions -----
    # We map your old parameters to the LJ parameters.
    # sigma is the characteristic size of the interaction.
    # epsilon is the energy (strength) of the interaction.
    sigma = float(job.sp.rep_r_max)
    epsilon_strong = float(job.sp.rep_A_strong)
    epsilon_weak = float(job.sp.rep_A_weak)

    # This cutoff makes the LJ potential purely repulsive (WCA potential)
    rc_wca = sigma * (2.0 ** (1.0 / 6.0))

    n = len(species)
    for i, si in enumerate(species):
        mi = f"{si}M"
        for j, sj in enumerate(species):
            mj = f"{sj}M"

            # Determine the strength of the repulsion
            if i == j or abs(i - j) > 1:  # Self-interaction or non-consecutive
                epsilon = epsilon_strong
            else:  # Consecutive neighbors
                epsilon = epsilon_weak

            # Set the LJ parameters for this pair
            lj_repulsion.params[(mi, mj)] = dict(epsilon=epsilon, sigma=sigma)
            lj_repulsion.r_cut[(mi, mj)] = rc_wca
