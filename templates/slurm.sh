{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --job-name="{{ id }}"
#SBATCH -t 02:00:00

# Target the high-memory GPU partitions
#SBATCH -p gpu,seas_gpu

# Request a single GPU (even for the CPU job, to land on a high-mem node)
#SBATCH --gres=gpu:1

# Request a large, specific amount of memory. 500G is a valid request on your cluster.
#SBATCH --mem=500G

{% if job_output %}
#SBATCH --output={{ job_output }}
#SBATCH --error={{ job_output }}
{% endif %}
{% block tasks %}{% endblock %}
{% endblock %}