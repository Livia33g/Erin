import sys
import json
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from statistics import mean, stdev

import signac


def save_polymer_distribution():
    """
    Aggregate per-job polymer distributions and yields across kT and concentration.
    Dynamically handles any number of monomer species.
    Outputs polymer_dist_by_kT.json.
    """
    project = signac.get_project()
    polymer_size_by_kT = {}
    # Gather unique concentrations
    concentrations = {job.sp.concentration for job in project}

    for concentration in sorted(concentrations):
        conc_key = f"concentration={concentration}"
        polymer_size_by_kT[conc_key] = {}

        # Find one job to determine number of species N and target sequence
        jobs_for_conc = list(project.find_jobs({"concentration": concentration}))
        if not jobs_for_conc:
            continue
        rep_job = jobs_for_conc[0]
        species = sorted(rep_job.sp.monomer_counts.keys())
        N = len(species)
        target_label = "".join(species)
        # Store metadata
        polymer_size_by_kT[conc_key]["_N_species"] = N
        polymer_size_by_kT[conc_key]["_target_label"] = target_label

        # Iterate all replicas for this concentration
        for job in jobs_for_conc:
            if not job.isfile("polymer_dist.json") or not job.isfile(
                "target_cluster_yield.json"
            ):
                continue
            kT_key = f"kT={job.sp.kT}"
            rep_key = f"replica {job.sp.replica}"
            # Initialize nested dicts
            polymer_size_by_kT[conc_key].setdefault(kT_key, {})

            # Load distribution and yield
            with job:
                with open("polymer_dist.json") as f:
                    dist_data = json.load(f)
                with open("target_cluster_yield.json") as f:
                    target_list = json.load(f)
            avg_counts = dist_data  # keys like '1 Monomers', '2 Monomers', etc.
            target_fraction = float(target_list[0]) if target_list else 0.0

            # Total average count of all polymers
            total_count = sum(avg_counts.values())
            # Expected absolute count of target N-mers
            target_count = target_fraction * total_count

            # Build dynamic categories
            categories = {
                "monomers": 0.0,
                "dimers": 0.0,
                f"Target {target_label}": 0.0,
                f"Off-target {N}-mers": 0.0,
                "Other large polymers": 0.0,
            }

            # Distribute counts
            for size_key, cnt in avg_counts.items():
                size_int = int(size_key.split()[0])
                if size_int == 1:
                    categories["monomers"] += cnt
                elif size_int == 2:
                    categories["dimers"] += cnt
                elif size_int == N:
                    # Separate target vs off-target N-mers
                    off = max(0.0, cnt - target_count)
                    categories[f"Off-target {N}-mers"] += off
                    categories[f"Target {target_label}"] = target_count
                else:
                    categories["Other large polymers"] += cnt
            # If no N-mers present, still record target_count (possibly zero)
            if f"{N} Monomers" not in avg_counts:
                categories[f"Target {target_label}"] = target_count

            polymer_size_by_kT[conc_key][kT_key][rep_key] = categories

        # Compute averages and std dev per kT
        for kT_key, reps in list(polymer_size_by_kT[conc_key].items()):
            if kT_key.startswith("_"):
                continue
            # Gather all category names
            all_cats = set()
            for rep_data in reps.values():
                all_cats.update(rep_data.keys())
            # Initialize storage
            data_per_cat = {cat: [] for cat in all_cats}
            for rep_data in reps.values():
                for cat in all_cats:
                    data_per_cat[cat].append(rep_data.get(cat, 0.0))
            # Compute stats
            average = {cat: mean(vals) for cat, vals in data_per_cat.items()}
            stdevs = {
                cat: stdev(vals) if len(vals) > 1 else 0.0
                for cat, vals in data_per_cat.items()
            }
            # Store back
            polymer_size_by_kT[conc_key][kT_key]["Average"] = average
            polymer_size_by_kT[conc_key][kT_key]["Std_Dev"] = stdevs

    # Save aggregated JSON
    with open("polymer_dist_by_kT.json", "w") as out:
        json.dump(polymer_size_by_kT, out, indent=4)
    print("Saved aggregated data to polymer_dist_by_kT.json")


def plot_polymer_distribution():
    """
    Reads polymer_dist_by_kT.json and plots yields vs 1/kT for each concentration.
    Categories dynamically set based on N_species and target_label.
    """
    # Visual style
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "lines.linewidth": 2.5,
            "figure.figsize": (10, 6),
        }
    )

    with open("polymer_dist_by_kT.json") as f:
        data = json.load(f)

    for conc_key, conc_data in data.items():
        # Skip metadata-only
        kT_keys = [k for k in conc_data if not k.startswith("_")]
        if not kT_keys:
            continue
        N = conc_data.get("_N_species")
        target_label = conc_data.get("_target_label")

        # Define plot categories and styles
        plot_cats = [
            ("Monomers", ["monomers"], "-", "o"),
            ("Dimers", ["dimers"], "--", "s"),
            (f"Target {target_label}", [f"Target {target_label}"], "-.", "D"),
            (f"Off-target {N}-mers", [f"Off-target {N}-mers"], ":", "^"),
            ("Other large polymers", ["Other large polymers"], "-", "v"),
        ]

        invkT = []
        cat_vals = {label: [] for label, _, _, _ in plot_cats}
        cat_errs = {label: [] for label, _, _, _ in plot_cats}

        # Extract data
        for kT_key in sorted(kT_keys, key=lambda x: float(x.split("=")[1])):
            kT_val = float(kT_key.split("=")[1])
            invkT.append(1.0 / kT_val)
            stats = conc_data[kT_key]
            avg = stats["Average"]
            dev = stats["Std_Dev"]
            total = sum(avg.values())
            for label, keys, style, marker in plot_cats:
                val = sum(avg.get(k, 0.0) for k in keys) / (total if total > 0 else 1)
                err = math.sqrt(sum(dev.get(k, 0.0) ** 2 for k in keys)) / (
                    total if total > 0 else 1
                )
                cat_vals[label].append(val)
                cat_errs[label].append(err)

        # Plot
        fig, ax = plt.subplots()
        for label, keys, style, marker in plot_cats:
            ax.errorbar(
                invkT,
                cat_vals[label],
                yerr=cat_errs[label],
                label=label,
                linestyle=style,
                marker=marker,
            )
        ax.set(xlabel="1/kT", ylabel="Fraction", title=f"Cluster yields ({conc_key})")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="best")
        plt.savefig(f"cluster_yields_{conc_key}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved cluster_yields_{conc_key}.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        func = globals().get(sys.argv[1])
        if callable(func):
            func()
        else:
            print("Choose save_polymer_distribution or plot_polymer_distribution")
    else:
        print("No function specified.")
