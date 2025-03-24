#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1  # Each job gets 1 GPU
#SBATCH --mem=120GB
#SBATCH -A p_btw_challenge2025
#SBATCH --job-name=EvalModels
#SBATCH --output=eval_layers_%A_%a.out
#SBATCH --array=0-49  # Launch 50 jobs per scaling method

echo "Starting evaluation for task ID $SLURM_ARRAY_TASK_ID..."

# Load environment
module purge
module load release/23.10 GCCcore/11.3.0 Python
source /path/to/venv

echo "Environment loaded successfully."

# Move to project directory
cd path/to/script
echo "Current directory: $(pwd)"

# Define scaling parameter
SCALING_PARAM=$1

if [ -z "$SCALING_PARAM" ]; then
  echo "No scaling parameter provided. Usage: sbatch evaluate_models.sh <scaling_parameter>"
  exit 1
fi

# Pass SLURM task ID to Python script
python3 layersEvaluation.py "$SCALING_PARAM" "$SLURM_ARRAY_TASK_ID"

# Deactivate environment and exit
echo "Evaluation completed for task ID $SLURM_ARRAY_TASK_ID."
deactivate
exit 0

