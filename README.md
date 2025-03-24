# Code for My Large Thesis

This repository contains the code used in my large thesis. Since I used the [Chronos Forecasting Repository](https://github.com/amazon-science/chronos-forecasting/tree/main) as a base, this repository includes only modified or additional code.

## Repository Structure

- **BatchScripts/**: Contains all batch scripts used to run the pretraining and evaluation of all model configurations. Additionally, it includes the batch script used to retrieve the training data.
- **ModifiedScripts/**: Contains a modified `train.py` to accommodate additional hyperparameters and a modified `evaluate_new.py` to include extra evaluation metrics.
- **Experiment Directories**: Each directory corresponds to an experiment (e.g., speedup, halved training time, or detailed hyperparameter searches). Each experiment directory contains scripts to:
  - Create the database
  - Insert configuration files
  - Run the pretraining process
  - Evaluate the models
- **Results Directories**: Located within each experiment directory, these contain result files with four evaluation metrics used to assess the models.
- **Training Steps Evaluation**: No dedicated directory is needed, as the model versions from the halved training time experiment were used to evaluate this scaling parameter.

## Reproducing an Experiment

To replicate an experiment, follow these steps:

1. **Install Chronos**
   - Follow the installation instructions provided in the [Chronos Forecasting Repository](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts).

2. **Set Up the Directory Structure**
   - For minimal path adaptation, store scripts in the parent directory of `chronos-forecasting`.
   - Replace `train.py` with the modified `train.py`.
   - Insert `evaluate_new.py` in the same directory as `evaluate.py`.

3. **Download and Convert Training Data**
   - Use `getTrainingData.py` to download and convert the data to the Arrow format.
   - Run this script once for augmented data and once for kernel-synthesized data.

4. **Create the Database**
   - Run the database creation script.

5. **Generate Configuration Files**
   - Run the configuration script to generate the model configs.
   - Modify paths in the script to point to the downloaded and augmented data as needed.

6. **Run Pretraining and Evaluation**
   - Execute the pretraining and evaluation scripts.
   - Modify the base path as necessary.
   - If scripts are not stored in the parent directory of `chronos-forecasting`, adjust data paths accordingly.

7. **Access Results**
   - The results will be available in the database.
