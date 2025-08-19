'''
import numpy as np
import jax.numpy as jnp
from jax import vmap, grad
import signac
from flow import FlowProject
from flow import directives
import flow.environments
import json
import os
import gsd.hoomd
import hoomd
from hoomd import md
from scipy.spatial.transform import Rotation
from pair_potentials import set_pair_potentials_params


class Project(FlowProject):
    pass


@Project.label
def initialized(job):
    """Checks if the initial, un-simulated state file has been created."""
    return os.path.isfile(job.fn("init.gsd"))


@Project.post(initialized)
@Project.operation(directives={"walltime": 5, "nranks": 1})
def initialize(job):
    import numpy as np
    from hoomd import md

    species_names = sorted(job.sp.monomer_counts.keys())
    first, last = species_names[0], species_names[-1]

    # all types
    core_types = list(species_names)
    middle_types = [f"{s}M" for s in species_names]
    all_types = core_types + ["P1", "P2"] + middle_types

    # geometry
    a, b = job.sp.a, job.sp.b
    all_pos = np.array(
        [
            [-a, 0.0, b],
            [-a, b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
            [-a, -b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
            [0.0, 0.0, a],
            [0.0, a * np.cos(np.pi / 6), -a * np.sin(np.pi / 6)],
            [0.0, -a * np.cos(np.pi / 6), -a * np.sin(np.pi / 6)],
            [a, 0.0, b],
            [a, b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
            [a, -b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
        ],
        dtype=np.float64,
    )
    all_orients = np.array([[1, 0, 0, 0]] * 9, dtype=np.float64)

    # scatter centers
    box = job.sp.box_L
    centers, ids = [], []

    def valid(r):
        buf = max(a, b)
        lb, ub = -box / 2 + buf, box / 2 - buf
        if np.any(r < lb) or np.any(r > ub):
            return False
        return all(np.linalg.norm(r - c) > 1.5 for c in centers)

    idx = {s: i for i, s in enumerate(species_names)}
    for s, count in job.sp.monomer_counts.items():
        for _ in range(count):
            while True:
                r = np.random.uniform(-box / 2, box / 2, 3)
                if valid(r):
                    centers.append(r)
                    ids.append(idx[s])
                    break

    centers = np.array(centers)
    N = len(centers)

    # build snapshot of centers
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    snap = hoomd.Snapshot()
    snap.configuration.box = [box, box, box, 0, 0, 0]
    snap.particles.types = all_types
    snap.particles.N = N
    snap.particles.typeid[:] = ids
    snap.particles.position[:] = centers
    snap.particles.orientation[:] = np.array([[1, 0, 0, 0]] * N, dtype=np.float64)

    # inertia (for centers)
    MOI = np.zeros(3)
    for i, p in enumerate(all_pos):
        m = 1.0 if i in (3, 4, 5) else 0.2
        MOI[0] += m * (p[1] ** 2 + p[2] ** 2)
        MOI[1] += m * (p[0] ** 2 + p[2] ** 2)
        MOI[2] += m * (p[0] ** 2 + p[1] ** 2)
    snap.particles.moment_inertia[:] = [MOI.tolist()] * N

    # rigid recipes with ends swapped
    rigid = md.constrain.Rigid()
    for s in species_names:
        if s == first:
            # first: middle + P2
            recipe = [f"{s}M"] * 3 + ["P2"] * 3
            pos = all_pos[3:]
            orients = all_orients[3:]
        elif s == last:
            # last: P1 + middle
            recipe = ["P1"] * 3 + [f"{s}M"] * 3
            pos = all_pos[:6]
            orients = all_orients[:6]
        else:
            # full P1 + middle + P2
            recipe = ["P1"] * 3 + [f"{s}M"] * 3 + ["P2"] * 3
            pos = all_pos
            orients = all_orients

        rigid.body[s] = {
            "constituent_types": recipe,
            "positions": pos,
            "orientations": orients,
        }

    # write only centers → then locally build ghosts
    sim.create_state_from_snapshot(snap)
    hoomd.write.GSD.write(state=sim.state, mode="wb", filename=job.fn("init.gsd"))
    rigid.create_bodies(sim.state)
    print(f"init.gsd: wrote {N} centers in box {box}")


@Project.label
def dumped(job):
    return os.path.isfile(job.fn("dump.gsd"))


# In project.py

# In project.py


@Project.pre.after(initialize)
@Project.post(dumped)
@Project.operation(
    directives={"walltime": 48, "nranks": 1, "gres": "gpu:a100:1", "memory": "32G"}
)
def equilibrate(job):
    """
    Loads centers from init.gsd, rebuilds rigid bodies, assigns interactions, then
    runs compression, annealing, and production. HOOMD v4 / GPU-safe.
    """
    import numpy as np
    import hoomd
    from hoomd import md
    from pair_potentials import set_pair_potentials_params

    # ---------- species & geometry ----------
    species_names = sorted(job.sp.monomer_counts.keys())
    first, last = species_names[0], species_names[-1]

    a, b = float(job.sp.a), float(job.sp.b)
    monomer_positions = np.array(
        [
            [-a, 0.0, b],
            [-a, b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
            [-a, -b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
            [0.0, 0.0, a],
            [0.0, a * np.cos(np.pi / 6.0), -a * np.sin(np.pi / 6.0)],
            [0.0, -a * np.cos(np.pi / 6.0), -a * np.sin(np.pi / 6.0)],
            [a, 0.0, b],
            [a, b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
            [a, -b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
        ],
        dtype=np.float64,
    )
    identity_orients = np.array([[1.0, 0.0, 0.0, 0.0]] * 9, dtype=np.float64)

    # ---------- simulation state ----------
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=int(job.sp.seed))
    sim.create_state_from_gsd(filename=job.fn("init.gsd"))

    # ---------- rigid body rebuild (swap caps at ends) ----------
    rigid = md.constrain.Rigid()
    for s in species_names:
        if s == first:
            recipe, pos, orients = (
                [f"{s}M"] * 3 + ["P2"] * 3,
                monomer_positions[3:],
                identity_orients[3:],
            )
        elif s == last:
            recipe, pos, orients = (
                ["P1"] * 3 + [f"{s}M"] * 3,
                monomer_positions[:6],
                identity_orients[:6],
            )
        else:
            recipe, pos, orients = (
                ["P1"] * 3 + [f"{s}M"] * 3 + ["P2"] * 3,
                monomer_positions,
                identity_orients,
            )
        rigid.body[s] = {
            "constituent_types": recipe,
            "positions": pos,
            "orientations": orients,
        }
    rigid.create_bodies(sim.state)

    # ---------- integrator & nlist ----------
    integrator = md.Integrator(dt=float(job.sp.dt), integrate_rotational_dof=True)
    integrator.rigid = rigid

    nl = md.nlist.Cell(buffer=0.3, exclusions=["body"])
    morse = md.pair.Morse(nlist=nl, default_r_cut=0.0)
    table = md.pair.Table(nlist=nl, default_r_cut=0.0)

    # ---------- tabulated repulsion (analytic; no JAX) ----------
    width = 1001
    table.width = int(width)  # must set before params
    assert table.width == width

    rc_rep = float(job.sp.rep_r_cut)
    rmax = float(job.sp.rep_r_max)
    alpha = float(job.sp.rep_alpha)

    r = np.linspace(0.0, rc_rep, width, dtype=np.float32)
    base = np.maximum(rmax - r, 0.0)  # clamp

    def make_UF(A):
        # The rest of the function will now operate with float32 arrays
        A = float(A)
        U = (A / (alpha * rmax)) * (base**alpha)
        F = (A / rmax) * (base ** (alpha - 1.0))
        # zero beyond rmax (numerical hygiene)
        mask = base > 0.0
        U = np.where(mask, U, 0.0)
        F = np.where(mask, F, 0.0)
        return U, F


    U_strong, F_strong = make_UF(job.sp.rep_A_strong)
    U_weak, F_weak = make_UF(job.sp.rep_A_weak)

    # ---------- assign pair potentials ----------
    set_pair_potentials_params(
        job, morse, table, (U_strong, F_strong), (U_weak, F_weak)
    )
    assert table.width == len(U_strong) == len(U_weak)

    morse.mode = "shift"
    integrator.forces.extend([morse, table])

    # ---------- guardrail: cores must be inert (no active cutoffs) ----------
    core = set(species_names)

    def _rcut_tbl(p):
        try:
            return float(table.r_cut[p])
        except KeyError:
            return 0.0

    def _rcut_morse(p):
        try:
            return float(morse.r_cut[p])
        except KeyError:
            return 0.0

    bad = []
    for i in sim.state.particle_types:
        for j in sim.state.particle_types:
            if i in core or j in core:
                if _rcut_tbl((i, j)) > 0.0 or _rcut_morse((i, j)) > 0.0:
                    bad.append((i, j))
    if bad:
        raise RuntimeError(
            f"Core should be inert, but these pairs are active: {bad[:8]}{' ...' if len(bad) > 8 else ''}"
        )

    # Debug sanity (comment out if noisy)
    print(
        "Active Morse pairs:",
        [
            (i, j)
            for i in sim.state.particle_types
            for j in sim.state.particle_types
            if _rcut_morse((i, j)) > 0.0
        ],
    )
    print(
        "Active Table pairs (count):",
        sum(
            _rcut_tbl((i, j)) > 0.0
            for i in sim.state.particle_types
            for j in sim.state.particle_types
        ),
    )

    # ---------- thermostat & compression ----------
    rb_filter = hoomd.filter.Rigid(("center", "free"))
    thermostat = md.methods.thermostats.MTTK(
        kT=float(2.0 + job.sp.kT), tau=float(job.sp.tau)
    )
    cv_method = md.methods.ConstantVolume(filter=rb_filter, thermostat=thermostat)
    integrator.methods.append(cv_method)

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(filter=rb_filter, kT=float(2.0 + job.sp.kT))

    N_monomers = int(sum(job.sp.monomer_counts.values()))
    final_L = float(N_monomers / float(job.sp.concentration)) ** (1.0 / 3.0)

    box_resize = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(10),
        box=hoomd.variant.box.InverseVolumeRamp(
            initial_box=sim.state.box,
            final_volume=final_L**3,
            t_start=sim.timestep,
            t_ramp=int(job.sp.equil_step),
        ),
    )
    sim.operations.updaters.append(box_resize)
    print("Starting compression run...")
    sim.run(int(job.sp.equil_step))
    sim.operations.updaters.remove(box_resize)
    print("Compression finished.")

    # ---------- logging & writers ----------
    thermo = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(categories=["scalar", "sequence"])
    logger.add(sim, quantities=["timestep"])
    logger.add(
        thermo,
        quantities=[
            "pressure",
            "potential_energy",
            "translational_kinetic_energy",
            "rotational_kinetic_energy",
        ],
    )

    h5 = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(int(job.sp.log_period)),
        filename=job.fn("dump_log.h5"),
        mode="w",
        logger=logger,
    )
    gsd = hoomd.write.GSD(
        filename=job.fn("dump.gsd"),
        trigger=hoomd.trigger.Periodic(int(job.sp.dump_period)),
        mode="wb",
    )
    sim.operations.writers.extend([h5, gsd])

    # ---------- annealing & production ----------
    print("Starting annealing...")
    for T in np.arange(2.0 + float(job.sp.kT), float(job.sp.kT), -0.1):
        cv_method.thermostat.kT = float(T)
        sim.run(int(5e5))

    print("Starting production run...")
    cv_method.thermostat.kT = float(job.sp.kT)
    sim.run(int(job.sp.run_step))
    print("Simulation finished.")


if __name__ == "__main__":
    Project().main()

'''

# project.py

import numpy as np
import signac
from flow import FlowProject
from flow import directives
import os
import hoomd
from hoomd import md
from pair_potentials import set_pair_potentials_params


class Project(FlowProject):
    pass


@Project.label
def initialized(job):
    """Checks if the initial, un-simulated state file has been created."""
    return os.path.isfile(job.fn("init.gsd"))


@Project.post(initialized)
@Project.operation(directives={"walltime": 5, "nranks": 1})
def initialize(job):
    import numpy as np
    from hoomd import md

    species_names = sorted(job.sp.monomer_counts.keys())
    first, last = species_names[0], species_names[-1]

    # <<< FIX: THIS IS THE CORRECTED TYPE DEFINITION
    # We must declare ALL types that will ever exist in the simulation state.
    # This includes the temporary central types ('A', 'B', ...) which exist in
    # the initial snapshot, and all the final constituent types ('AM', 'P1', etc.)
    # which are created by `rigid.create_bodies`.
    center_types = species_names  # ['A', 'B', ...]
    constituent_types = [f"{s}M" for s in species_names] + ["P1", "P2"]
    all_types = center_types + constituent_types

    # geometry
    a, b = job.sp.a, job.sp.b
    all_pos = np.array(
        [
            [-a, 0.0, b],
            [-a, b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
            [-a, -b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
            [0.0, 0.0, a],
            [0.0, a * np.cos(np.pi / 6), -a * np.sin(np.pi / 6)],
            [0.0, -a * np.cos(np.pi / 6), -a * np.sin(np.pi / 6)],
            [a, 0.0, b],
            [a, b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
            [a, -b * np.cos(np.pi / 6), -b * np.sin(np.pi / 6)],
        ],
        dtype=np.float64,
    )
    all_orients = np.array([[1, 0, 0, 0]] * 9, dtype=np.float64)

    # scatter centers
    box = job.sp.box_L
    centers, ids = [], []

    def valid(r):
        buf = max(a, b)
        lb, ub = -box / 2 + buf, box / 2 - buf
        if np.any(r < lb) or np.any(r > ub):
            return False
        return all(np.linalg.norm(r - c) > 1.5 for c in centers)

    # NOTE: The type IDs here map to the 'center_types' ('A', 'B', etc.)
    idx = {s: i for i, s in enumerate(species_names)}
    for s, count in job.sp.monomer_counts.items():
        for _ in range(count):
            while True:
                r = np.random.uniform(-box / 2, box / 2, 3)
                if valid(r):
                    centers.append(r)
                    ids.append(idx[s])
                    break

    centers = np.array(centers)
    N = len(centers)

    # build snapshot of centers
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    snap = hoomd.Snapshot()
    snap.configuration.box = [box, box, box, 0, 0, 0]
    snap.particles.types = all_types  # Use the comprehensive list
    snap.particles.N = N
    snap.particles.typeid[:] = ids
    snap.particles.position[:] = centers
    snap.particles.orientation[:] = np.array([[1, 0, 0, 0]] * N, dtype=np.float64)

    # inertia (for centers)
    MOI = np.zeros(3)
    for i, p in enumerate(all_pos):
        m = 1.0 if i in (3, 4, 5) else 0.2
        MOI[0] += m * (p[1] ** 2 + p[2] ** 2)
        MOI[1] += m * (p[0] ** 2 + p[2] ** 2)
        MOI[2] += m * (p[0] ** 2 + p[1] ** 2)
    snap.particles.moment_inertia[:] = [MOI.tolist()] * N

    # rigid recipes with ends swapped
    rigid = md.constrain.Rigid()
    for s in species_names:
        if s == first:
            recipe = [f"{s}M"] * 3 + ["P2"] * 3
            pos = all_pos[3:]
            orients = all_orients[3:]
        elif s == last:
            recipe = ["P1"] * 3 + [f"{s}M"] * 3
            pos = all_pos[:6]
            orients = all_orients[:6]
        else:
            recipe = ["P1"] * 3 + [f"{s}M"] * 3 + ["P2"] * 3
            pos = all_pos
            orients = all_orients

        # The key of the recipe dictionary is the temporary central type ('A')
        # The values are the final constituent types ('AM', 'P1', etc.)
        rigid.body[s] = {
            "constituent_types": recipe,
            "positions": pos,
            "orientations": orients,
        }

    sim.create_state_from_snapshot(snap)

    # The order of operations here remains correct: create bodies first, THEN write.
    rigid.create_bodies(sim.state)
    hoomd.write.GSD.write(state=sim.state, mode="wb", filename=job.fn("init.gsd"))

    print(f"init.gsd: wrote {N} rigid bodies in box {box}")


@Project.label
def dumped(job):
    return os.path.isfile(job.fn("dump.gsd"))


@Project.pre.after(initialize)
@Project.post(dumped)
@Project.operation(
    directives={"walltime": 48, "nranks": 1, "gres": "gpu:a100:1", "memory": "32G"}
)
def equilibrate(job):
    """
    Loads centers from init.gsd, assigns interactions, then
    runs compression, annealing, and production. HOOMD v4 / GPU-safe.
    """
    import numpy as np
    import hoomd
    from hoomd import md
    from pair_potentials import set_pair_potentials_params

    # ---------- simulation state ----------
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=int(job.sp.seed))
    # This now loads the GSD with FULLY FORMED rigid bodies.
    sim.create_state_from_gsd(filename=job.fn("init.gsd"))

    # ---------- rigid body constraint ----------
    # We still need to tell the integrator about the rigid constraint,
    # but we DO NOT need to redefine or recreate the bodies themselves.
    rigid = md.constrain.Rigid()

    # <<< FIX 3: REMOVE REDUNDANT CODE. Since init.gsd now contains the
    # correct, fully-formed bodies, we don't need to rebuild them here.
    # The simulation state loaded from GSD already knows which particles
    # belong to which body.
    #
    # The entire block below has been removed:
    #
    # species_names = sorted(job.sp.monomer_counts.keys())
    # first, last = species_names[0], species_names[-1]
    # a, b = float(job.sp.a), float(job.sp.b)
    # ... (monomer_positions and identity_orients definitions) ...
    # for s in species_names:
    #   ... (if/elif/else block to define recipe, pos, orients) ...
    #   rigid.body[s] = { ... }
    # rigid.create_bodies(sim.state)
    # -------------------------------------------------------------------

    # ---------- integrator & nlist ----------
    integrator = md.Integrator(dt=float(job.sp.dt), integrate_rotational_dof=True)
    integrator.rigid = rigid  # This is still necessary!

    nl = md.nlist.Cell(buffer=0.3, exclusions=["body"])
    morse = md.pair.Morse(nlist=nl, default_r_cut=0.0)
    # NOTE: You were mixing md.pair.Table and md.pair.LJ. I'm assuming you want LJ.
    lj_repulsion = md.pair.LJ(nlist=nl, default_r_cut=0.0)

    # ---------- assign pair potentials ----------
    set_pair_potentials_params(job, morse, lj_repulsion)

    morse.mode = "shift"
    integrator.forces.extend([morse, lj_repulsion])

    # ---------- guardrail: check that constituent types don't interact ----------
    # Note: I've updated your guardrail. The old one checked for 'A', 'B' types
    # which no longer exist after the bodies are created. This one checks that
    # P1-P1 and P2-P2 interactions are off, which is a good sanity check.
    def _rcut(potential, pair):
        try:
            return float(potential.r_cut[pair])
        except KeyError:
            return 0.0

    bad_pairs = []
    # Check that patches only interact with the other type
    if _rcut(morse, ("P1", "P1")) > 0.0 or _rcut(lj_repulsion, ("P1", "P1")) > 0.0:
        bad_pairs.append(("P1", "P1"))
    if _rcut(morse, ("P2", "P2")) > 0.0 or _rcut(lj_repulsion, ("P2", "P2")) > 0.0:
        bad_pairs.append(("P2", "P2"))

    # Check that patches and middles don't interact
    species_names = sorted(job.sp.monomer_counts.keys())
    for s in species_names:
        m_type = f"{s}M"
        if (
            _rcut(morse, (m_type, "P1")) > 0.0
            or _rcut(lj_repulsion, (m_type, "P1")) > 0.0
        ):
            bad_pairs.append((m_type, "P1"))
        if (
            _rcut(morse, (m_type, "P2")) > 0.0
            or _rcut(lj_repulsion, (m_type, "P2")) > 0.0
        ):
            bad_pairs.append((m_type, "P2"))

    if bad_pairs:
        raise RuntimeError(f"Inactive pairs have interactions set: {bad_pairs}")

    # ---------- thermostat & compression ----------
    # Correct filter for rigid bodies in HOOMD v4+
    rb_filter = hoomd.filter.Rigid(("center", "free"))
    thermostat = md.methods.thermostats.MTTK(
        kT=float(2.0 + job.sp.kT), tau=float(job.sp.tau)
    )
    cv_method = md.methods.ConstantVolume(filter=rb_filter, thermostat=thermostat)
    integrator.methods.append(cv_method)

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(filter=rb_filter, kT=float(2.0 + job.sp.kT))

    # Compression logic looks okay.
    N_monomers = int(sum(job.sp.monomer_counts.values()))
    final_L = float(N_monomers / float(job.sp.concentration)) ** (1.0 / 3.0)

    box_resize = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(10),
        box=hoomd.variant.box.InverseVolumeRamp(
            initial_box=sim.state.box,
            final_volume=final_L**3,
            t_start=sim.timestep,
            t_ramp=int(job.sp.equil_step),
        ),
    )
    sim.operations.updaters.append(box_resize)
    print("Starting compression run...")
    sim.run(int(job.sp.equil_step))
    sim.operations.updaters.remove(box_resize)
    print("Compression finished.")

    # ---------- logging & writers ----------
    thermo = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(categories=["scalar", "sequence"])
    logger.add(sim, quantities=["timestep"])
    logger.add(
        thermo,
        quantities=[
            "pressure",
            "potential_energy",
            "translational_kinetic_energy",
            "rotational_kinetic_energy",
        ],
    )

    h5 = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(int(job.sp.log_period)),
        filename=job.fn("dump_log.h5"),
        mode="w",
        logger=logger,
    )
    gsd = hoomd.write.GSD(
        filename=job.fn("dump.gsd"),
        trigger=hoomd.trigger.Periodic(int(job.sp.dump_period)),
        mode="wb",
    )
    sim.operations.writers.extend([h5, gsd])

    # ---------- annealing & production ----------
    print("Starting annealing...")
    for T in np.arange(2.0 + float(job.sp.kT), float(job.sp.kT), -0.1):
        cv_method.thermostat.kT = float(T)
        sim.run(int(5e5))

    print("Starting production run...")
    cv_method.thermostat.kT = float(job.sp.kT)
    sim.run(int(job.sp.run_step))
    print("Simulation finished.")


if __name__ == "__main__":
    Project().main()
