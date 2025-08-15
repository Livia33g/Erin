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
    Loads the saved centers from init.gsd, re-creates the rigid bodies
    (with swapped end caps), then runs compression, annealing, and production.
    """
    import numpy as np
    import jax.numpy as jnp
    from jax import vmap, grad
    import hoomd
    from hoomd import md
    from pair_potentials import set_pair_potentials_params

    # identify first/last species
    species_names = sorted(job.sp.monomer_counts.keys())
    first, last = species_names[0], species_names[-1]

    # re-define monomer geometry
    a, b = job.sp.a, job.sp.b
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
    identity_orients = np.array([[1, 0, 0, 0]] * 9, dtype=np.float64)

    # load only centers
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    sim.create_state_from_gsd(filename=job.fn("init.gsd"))

    # rebuild rigid-body constraint with swapped end caps
    rigid = md.constrain.Rigid()
    for s in species_names:
        if s == first:
            # first monomer: middle + P2
            recipe, pos, orients = (
                [f"{s}M"] * 3 + ["P2"] * 3,
                monomer_positions[3:],
                identity_orients[3:],
            )
        elif s == last:
            # last monomer: P1 + middle
            recipe, pos, orients = (
                ["P1"] * 3 + [f"{s}M"] * 3,
                monomer_positions[:6],
                identity_orients[:6],
            )
        else:
            # interior monomer: P1 + middle + P2
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

    # integrator + forces (unchanged)
    integrator = md.Integrator(dt=job.sp.dt, integrate_rotational_dof=True)
    integrator.rigid = rigid

    nl = md.nlist.Cell(buffer=0, exclusions=["body"])
    morse = md.pair.Morse(default_r_cut=job.sp.r_cut, nlist=nl)
    table = md.pair.Table(nlist=nl, default_r_cut=job.sp.rep_r_cut)

    def repulsive_potential(rmin, rmax, A, alpha):
        def _V(r):
            eps = 1e-6
            base = jnp.maximum(rmax - r, eps)
            return jnp.where(r < rmax, (A / (alpha * rmax)) * base**alpha, 0.0)

        return _V

    r_range = np.linspace(0, job.sp.rep_r_cut, 1001)
    strong_fn = repulsive_potential(
        job.sp.rep_r_min, job.sp.rep_r_max, job.sp.rep_A_strong, job.sp.rep_alpha
    )
    weak_fn = repulsive_potential(
        job.sp.rep_r_min, job.sp.rep_r_max, job.sp.rep_A_weak, job.sp.rep_alpha
    )
    tab_strong = (
        np.array(strong_fn(r_range)),
        np.array(-1 * vmap(grad(strong_fn))(r_range)),
    )
    tab_weak = (np.array(weak_fn(r_range)), np.array(-1 * vmap(grad(weak_fn))(r_range)))

    set_pair_potentials_params(job, morse, table, tab_strong, tab_weak)
    morse.mode = "shift"
    integrator.forces.extend([morse, table])

    # After set_pair_potentials_params(...), before sim.run(...)
    core = set(species_names)  # {'A','B','C',...}

    def _rcut(tbl, pair):
        try:
            return tbl.r_cut[pair]
        except KeyError:
            return 0.0

    for i in sim.state.particle_types:
        for j in sim.state.particle_types:
            if i in core or j in core:
                if _rcut(table, (i, j)) > 0.0 or (i, j) in morse.params:
                    raise RuntimeError(f"Core should be inert, but {(i,j)} has params")

    # thermostat + compression
    rb_filter = hoomd.filter.Rigid(("center", "free"))
    thermostat = md.methods.thermostats.MTTK(kT=2.0 + job.sp.kT, tau=job.sp.tau)
    cv_method = md.methods.ConstantVolume(filter=rb_filter, thermostat=thermostat)
    integrator.methods.append(cv_method)

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(filter=rb_filter, kT=2.0 + job.sp.kT)

    concentration = job.sp.concentration
    # total_constituent_N = sim.state.N_particles
    # final_L = (total_constituent_N / concentration) ** (1 / 3)
    N_monomers = sum(job.sp.monomer_counts.values())  # count centers only
    final_L = (N_monomers / job.sp.concentration) ** (1 / 3)

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

    # logging + writers (unchanged)
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

    # annealing + production (unchanged)
    print("Starting annealing...")
    for T in np.arange(2.0 + job.sp.kT, job.sp.kT, -0.1):
        cv_method.thermostat.kT = T
        sim.run(int(5e5))

    print("Starting production run...")
    cv_method.thermostat.kT = job.sp.kT
    sim.run(int(job.sp.run_step))
    print("Simulation finished.")


if __name__ == "__main__":
    Project().main()
