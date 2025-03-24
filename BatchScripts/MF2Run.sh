#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1  # Each job gets 1 GPU
#SBATCH --mem=120GB
#SBATCH -A p_btw_challenge2025
#SBATCH --job-name=MF2
#SBATCH --output=mf2_layer_%A_%a.out
#SBATCH --array=0-49  # Launch 50 jobs per scaling method

echo "Starting training script for task ID $SLURM_ARRAY_TASK_ID..."

# Load environment
module purge
module load release/23.10 GCCcore/11.3.0 Python
source /path/to/venv

echo "Environment loaded successfully."

# Move to project directory
cd path/to/script
echo "Current directory: $(pwd)"

# Define scaling method
SCALING_METHOD=$1

if [ -z "$SCALING_METHOD" ]; then
  echo "No scaling method provided. Usage: sbatch MF2Run.sh <scaling_method>"
  exit 1
fi

# Pass SLURM task ID to Python script
python3 layersRun.py "$SCALING_METHOD" "$SLURM_ARRAY_TASK_ID"

# Deactivate environment and exit
echo "Training process completed for task ID $SLURM_ARRAY_TASK_ID."
deactivate
exit 0

