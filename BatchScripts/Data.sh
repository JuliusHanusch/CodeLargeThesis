#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH -A p_automl
#SBATCH --job-name=Data
#SBATCH --output=data%j.out
#SBATCH --array=0


module purge
module load release/23.10 GCCcore/11.3.0 Python 
source /path/to/venv

echo "Start"

cd path/to/script

python3 getTrainingData.py

echo "Finish"
exit
deactivate
