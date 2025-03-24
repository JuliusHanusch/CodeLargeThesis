#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1  # Each job gets 1 GPU
#SBATCH --mem=120GB
#SBATCH -A p_automl
#SBATCH --job-name=SpeedupAll
#SBATCH --output=speedupAll_%A_%a.out
#SBATCH --array=0-61  # Array of 54 jobs (task IDs 0-45)

echo "Starting training for job ID: $SLURM_ARRAY_TASK_ID"

# Load environment
module purge
module load release/23.10 GCCcore/11.3.0 Python
source /data/horse/ws/juha972b-Tlm/Tlm/bin/activate

echo "Environment loaded successfully."

# Move to project directory
cd /data/horse/ws/juha972b-Tlm/Tlm
echo "Current directory: $(pwd)"

# Fetch the config_id for the current task ID (job array)
CONFIG_ID=$((SLURM_ARRAY_TASK_ID + 1))  # Adjust for 1-based indexing

# Execute the training script for the specific config
python3 speedupRunsAll.py "$CONFIG_ID"

# Deactivate environment and exit
echo "Training process completed for job ID: $SLURM_ARRAY_TASK_ID."
deactivate
exit 0

