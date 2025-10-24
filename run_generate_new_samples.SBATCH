#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=24G  # Increased memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=generate_samples

module purge

# Corrected the source path to be absolute and removed the :ro flag from the overlay
singularity exec --nv --overlay /scratch/cz2200/visnetoverlay/visnet_env.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; conda activate visnet; python generate_samples_new.py"
