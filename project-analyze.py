import numpy as np
import signac
from flow import FlowProject, directives
import json
import os
import gsd.hoomd
import freud
import fresnel
import matplotlib.pyplot as plt
import h5py

# ... (Cluster class and other helpers are the same) ...
class Cluster:
    def __init__(self, sequence, counts, is_cyclic):
        self.sequence = "".join(sorted(sequence))
        self.counts = counts
        self.is_cyclic = is_cyclic
    def compare(self, other_cluster):
        return self.sequence == other_cluster.sequence

class Project(FlowProject):
    pass

def is_file_ready(job, filename):
    path = job.fn(filename)
    return os.path.isfile(path) and os.path.getsize(path) > 0


# =============================================================================
# Render Operation with BATCHING
# =============================================================================
@Project.label
def rendered(job):
    return job.isfile("render.png")

@Project.pre(lambda job: is_file_ready(job, "dump.gsd"))
@Project.post(rendered)
@Project.operation(directives={"gres": "gpu:1"})
def render(job):
    """Renders monomer centers by batching them to conserve memory."""
    dump_filename = job.fn("dump.gsd")
    print(f"Rendering monomer centers for job {job.id} using batching...")

    with gsd.hoomd.open(name=dump_filename, mode="r") as gsd_file:
        snap = gsd_file[-1]
    
    all_types = snap.particles.types
    species_names = sorted(job.sp.monomer_counts.keys())
    core_type_ids = [i for i, t in enumerate(all_types) if t in species_names]
    if not core_type_ids:
        print("Error: No core types found.")
        return

    is_core_particle = np.isin(snap.particles.typeid, core_type_ids)
    core_positions = snap.particles.position[is_core_particle]
    core_typeids = snap.particles.typeid[is_core_particle]
    num_monomers = len(core_positions)
    print(f"Found {num_monomers} monomer centers to render.")

    scene = fresnel.Scene()
    core_colors = [fresnel.color.linear(plt.cm.tab10(i)) for i in range(len(core_type_ids))]
    color_map = {tid: color for tid, color in zip(core_type_ids, core_colors)}

    # =========================================================================
    # BATCHING LOGIC
    # =========================================================================
    batch_size = 100000  # Process 100,000 particles at a time. Adjust if needed.
    
    for i in range(0, num_monomers, batch_size):
        start = i
        end = min(i + batch_size, num_monomers)
        print(f"  - Rendering batch {i//batch_size + 1}: particles {start} to {end}")
        
        # Create a small geometry object for just this batch
        geom = fresnel.geometry.Sphere(scene, N=(end - start))
        geom.position[:] = core_positions[start:end]
        geom.radius[:] = 1.0
        geom.color[:] = [color_map[tid] for tid in core_typeids[start:end]]
        geom.material = fresnel.material.Material(roughness=0.2, specular=0.8)
    # =========================================================================

    fresnel.geometry.Box(scene, snap.configuration.box, box_radius=0.07)
    scene.camera = fresnel.camera.Orthographic.fit(scene)
    scene.lights = fresnel.light.lightbox()
    
    from PIL import Image
    print("All batches loaded. Path tracing final image...")
    img_array = fresnel.pathtrace(scene, w=1600, h=900, samples=64)
    Image.fromarray(img_array, mode='RGBA').save(job.fn("render.png"))
    print(f"Batch render for job {job.id} complete.")


# =============================================================================
# Polymer Analysis Operation (already memory-optimized)
# =============================================================================
@Project.label
def polymers_identified(job):
    return job.isfile(job.fn("polymer_dist.json"))

@Project.pre(lambda job: is_file_ready(job, "dump.gsd"))
@Project.post(polymers_identified)
@Project.operation(directives={"nranks": 1})
def identify_polymers(job):
    """Identifies polymers by clustering MONOMER CENTERS, which is highly efficient."""
    dump_filename = job.fn("dump.gsd")
    output_dir = job.fn("polymers")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analyzing monomer clusters for job {job.id}...")

    all_cluster_sizes = []
    species_names = sorted(job.sp.monomer_counts.keys())
    BONDING_DISTANCE = 2.5 

    with gsd.hoomd.open(dump_filename, 'r') as gsd_file:
        num_frames_total = len(gsd_file)
        if num_frames_total == 0: return
        start_frame_index = max(0, num_frames_total - 10)
        
        for i in range(start_frame_index, num_frames_total):
            frame = gsd_file[i]
            
            all_types = frame.particles.types
            core_type_ids = [idx for idx, t in enumerate(all_types) if t in species_names]
            if not core_type_ids: continue
            
            is_core_particle = np.isin(frame.particles.typeid, core_type_ids)
            core_pos = frame.particles.position[is_core_particle]
            if len(core_pos) == 0: continue

            box = freud.box.Box.from_box(frame.configuration.box)
            cl = freud.cluster.Cluster()
            cl.compute(system=(box, core_pos), neighbors={'r_max': BONDING_DISTANCE}, num_threads=1)
            
            cluster_lengths = [len(key) for key in cl.cluster_keys]
            all_cluster_sizes.extend(cluster_lengths)
            
    from collections import Counter
    if not all_cluster_sizes:
        print(f"No monomer clusters found for job {job.id}")
        with open(job.fn('polymer_dist.json'), 'w') as f: json.dump({}, f)
        return

    num_frames_analyzed = (num_frames_total - start_frame_index)
    counts = Counter(all_cluster_sizes)
    dist = {f"{l} Monomers": counts[l] / num_frames_analyzed for l in sorted(counts.keys())}
    
    with open(job.fn('polymer_dist.json'), 'w') as f:
        json.dump(dist, f, indent=4)
        
    labels, avg_vals = list(dist.keys()), list(dist.values())
        
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4)))
    ax.barh(labels, avg_vals)
    ax.set_title('Polymer Size Distribution (Avg per frame)')
    ax.set_xlabel('Average Count')
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'polymer_size_dist.png'))
    plt.close(fig)
    print(f"Polymer analysis for job {job.id} complete.")



# =============================================================================
# Plotting Operation (This one is already working, no changes)
# =============================================================================
@Project.label
def plotted(job):
    plot_dir = job.fn("plots")
    if not os.path.isdir(plot_dir): return False
    return job.isfile(os.path.join(plot_dir, "potential_energy.png"))

@Project.pre(lambda job: is_file_ready(job, "dump_log.h5"))
@Project.post(plotted)
@Project.operation(directives={"nranks": 1})
def plot_quantities(job):
    log_filename = job.fn('dump_log.h5')
    output_dir = job.fn("plots")
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(log_filename, 'r') as data:
        fields = {
            'pressure': 'md/compute/ThermodynamicQuantities/pressure',
            'potential_energy': 'md/compute/ThermodynamicQuantities/potential_energy',
            'translational_kinetic_energy': 'md/compute/ThermodynamicQuantities/translational_kinetic_energy',
            'rotational_kinetic_energy': 'md/compute/ThermodynamicQuantities/rotational_kinetic_energy'
        }
        fig, ax = plt.subplots(figsize=(10, 6))
        t_path = 'hoomd-data/Simulation/timestep'
        if t_path not in data: return
        for name, path in fields.items():
            full_path = f'hoomd-data/{path}'
            if full_path in data:
                y_data = data[full_path][:]
                x_data = data[t_path][:len(y_data)]
                ax.clear()
                ax.plot(x_data, y_data)
                ax.set(xlabel='Timestep', ylabel=name.replace('_',' ').title())
                ax.grid(True, linestyle='--')
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.close(fig)

if __name__ == '__main__':
    Project().main()