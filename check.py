#!/usr/bin/env python3
import sys
import numpy as np
import gsd.hoomd
import signac

def minimum_image_disp(vec, box):
    """Apply minimum image convention to displacement vector."""
    return vec - np.round(vec / box) * box

def quat_rotate(q, v):
    """Rotate vector v by quaternion q (scalar-first)."""
    # normalize quaternion
    q = np.array(q) / np.linalg.norm(q)
    w, x, y, z = q
    uv  = np.cross(q[1:], v)
    uuv = np.cross(q[1:], uv)
    return v + 2*(w*uv + uuv)

def check_particle_counts(job_id):
    project = signac.get_project()
    job = project.open_job(id=job_id)
    expected = None
    with gsd.hoomd.open(name=job.fn("dump.gsd"), mode="r") as traj:
        for snap in traj:
            N = snap.particles.N
            if expected is None:
                expected = N
            elif N != expected:
                raise RuntimeError(f"[ERROR] Frame {snap.configuration.step}: "
                                   f"particle count {N} != {expected}")
    print(f"[OK] Particle count consistent: {expected} per frame.")

def check_rigid_constraint(job_id, tol=1e-4):
    project = signac.get_project()
    job = project.open_job(id=job_id)

    # Build recipes: list of (relative_position, type_name)
    species = sorted(job.sp.monomer_counts.keys())
    first, last = species[0], species[-1]
    a, b = job.sp.a, job.sp.b
    all_rel = [
        (-a,  0.0,  b, "P1"),  (-a,  b*np.cos(np.pi/6), -b*np.sin(np.pi/6), "P1"),
        (-a, -b*np.cos(np.pi/6), -b*np.sin(np.pi/6), "P1"),
        ( 0.0, 0.0,  a, f"{species[0]}M"), # we'll overwrite type
        ( 0.0,  a*np.cos(np.pi/6), -a*np.sin(np.pi/6), f"{species[0]}M"),
        ( 0.0, -a*np.cos(np.pi/6), -a*np.sin(np.pi/6), f"{species[0]}M"),
        ( a,  0.0,  b, "P2"),  ( a,  b*np.cos(np.pi/6), -b*np.sin(np.pi/6), "P2"),
        ( a, -b*np.cos(np.pi/6), -b*np.sin(np.pi/6), "P2"),
    ]
    # Actually build per-species recipes
    recipes = {}
    for s in species:
        rels = []
        for x,y,z,_ in all_rel:
            if s == first:
                # first: only middle+P2 => skip P1 positions (first three)
                if _ == "P1":
                    continue
                # for middle positions override type to f"{s}M"
                t = "P2" if _ == "P2" else f"{s}M"
            elif s == last:
                # last: only P1+middle => skip P2 positions (last three)
                if _ == "P2":
                    continue
                t = "P1" if _ == "P1" else f"{s}M"
            else:
                # interior: full tri-sphere
                t = _ if _ in ("P1","P2") else f"{s}M"
            rels.append((np.array((x,y,z),dtype=np.float64), t))
        recipes[s] = rels

    # Now scan trajectory
    with gsd.hoomd.open(name=job.fn("dump.gsd"), mode="r") as traj:
        for snap in traj:
            box = np.array(snap.configuration.box[:3], dtype=np.float64)
            types = snap.particles.types
            ids   = snap.particles.typeid
            pos   = snap.particles.position
            ori   = snap.particles.orientation

            # find center indices
            centers = [i for i,tidx in enumerate(ids) if types[tidx] in species]
            for ci in centers:
                stype = types[ids[ci]]
                cpos  = pos[ci]
                crot  = ori[ci]
                for rel_vec, typ in recipes[stype]:
                    # expected world position
                    world = cpos + quat_rotate(crot, rel_vec)
                    # find candidate ghost indices of that type
                    candidates = [i for i,tidx in enumerate(ids) if types[tidx] == typ]
                    # compute min-image distance to each candidate
                    dists = []
                    for gi in candidates:
                        disp = minimum_image_disp(pos[gi] - world, box)
                        dists.append(np.linalg.norm(disp))
                    if not dists:
                        raise RuntimeError(f"[ERROR] No ghost of type {typ} found for center #{ci}")
                    dist = min(dists)
                    if dist > tol:
                        raise RuntimeError(f"[ERROR] Frame {snap.configuration.step}: "
                                           f"center #{ci} ghost {typ} off by {dist:.2e} > tol {tol:.1e}")
    print(f"[OK] All ghost sites within {tol:.1e} of their centers.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check.py <job_id>")
        sys.exit(1)
    jid = sys.argv[1]
    check_particle_counts(jid)
    check_rigid_constraint(jid)
