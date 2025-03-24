import os
import sys
import sqlite3
import csv

#define the scaling params and the range of the corresponding configuration ids
SCALING_PARAMS = {
    "num_heads_2": range(1, 51),
    "num_heads_4": range(51, 101),
    "num_heads_6": range(101, 151),
    "num_heads_8": range(151, 200)

}

BASE_PATH = "/data/horse/ws/juha972b-Tlm/Tlm/output"
EVAL_CONFIGS = {
    "in-domain": "huggingface/scripts/evaluation/configs/in-domain.yaml",
    "zero-shot": "huggingface/scripts/evaluation/configs/zero-shot.yaml"
}
DB_PATH = "Heads.db"

def evaluation_exists(model_version_id, eval_type):
    """Check if an evaluation result already exists in the database."""
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM EvaluationResults WHERE model_version_id = ? AND evaluation_type = ?",
        (model_version_id, eval_type)
    )
    exists = cursor.fetchone() is not None
    connection.close()
    return exists

def insert_evaluation_result(model_version_id, eval_type, mase, wql, rmse, mae):
    """Insert evaluation results into the EvaluationResults table if not already present."""
    if evaluation_exists(model_version_id, eval_type):
        print(f"Skipping {eval_type} evaluation for ModelVersion {model_version_id}, already exists.")
        return
    
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO EvaluationResults (model_version_id, evaluation_type, mase, wql, rmse, mae)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (model_version_id, eval_type, mase, wql, rmse, mae)
    )
    connection.commit()
    connection.close()

def get_model_version_id(config_id, training_step):
    """Retrieve the existing ModelVersion ID using config_id and training_step."""
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute("SELECT run_id FROM TrainingRuns WHERE config_id = ?", (config_id,))
    run = cursor.fetchone()
    
    if not run:
        print(f"Error: No TrainingRun found for config_id {config_id}.")
        connection.close()
        sys.exit(1)
    
    run_id = run[0]
    cursor.execute(
        "SELECT model_version_id FROM ModelVersions WHERE run_id = ? AND training_step = ?",
        (run_id, training_step)
    )
    existing_version = cursor.fetchone()
    connection.close()
    
    if existing_version:
        return existing_version[0]
    else:
        print(f"Error: No ModelVersion found for run_id {run_id} and training_step {training_step}.")
        sys.exit(1)

def parse_results(results_path):
    """Parse evaluation results and compute mean MASE, WQL, RMSE, and MAE."""
    try:
        mase_values, wql_values, rmse_values, mae_values = [], [], [], []
        with open(results_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Check and clean column names if needed
                mase_values.append(float(row["MASE"]))
                wql_values.append(float(row["WQL"]))
                rmse_values.append(float(row["RMSE[mean]"]))  # Adjusted column name
                mae_values.append(float(row["MAE"]))
        
        if not mase_values or not wql_values or not rmse_values or not mae_values:
            print(f"Error: No valid metric values found in {results_path}")
            return None, None, None, None

        return (
            sum(mase_values) / len(mase_values),
            sum(wql_values) / len(wql_values),
            sum(rmse_values) / len(rmse_values),
            sum(mae_values) / len(mae_values)
        )
    except Exception as e:
        print(f"Error parsing results from {results_path}: {e}")
        return None, None, None, None


def evaluate_model(scaling_param, task_id):
    '''evaluated a model based on the assigned scaling param and the slurm array id'''
    if scaling_param not in SCALING_PARAMS:
        print(f"Error: Invalid scaling parameter '{scaling_param}'.")
        sys.exit(1)
    
    config_ids = list(SCALING_PARAMS[scaling_param])
    if task_id < 0 or task_id >= len(config_ids):
        print(f"Error: Task ID {task_id} out of range for scaling parameter '{scaling_param}'.")
        sys.exit(1)
    
    #create model paths
    config_id = config_ids[task_id]
    training_step = 100000 if scaling_param == "max_steps" else 200000
    model_path = f"{BASE_PATH}/default/{config_id}/run-0/checkpoint-100000" if scaling_param == "max_steps" else f"{BASE_PATH}/{scaling_param}/{config_id}/run-0/checkpoint-final"
    results_dir = f"{model_path}/results"
    os.makedirs(results_dir, exist_ok=True)
    model_version_id = get_model_version_id(config_id, training_step)
    
    #runs evaluation for in-domain and zero-shot
    for eval_type, config_file in EVAL_CONFIGS.items():
        if evaluation_exists(model_version_id, eval_type):
            print(f"Skipping {eval_type} evaluation for ModelVersion {model_version_id}, already exists.")
            continue
        
        results_path = f"{results_dir}/{eval_type}.csv"
        print(f"Evaluating {eval_type} model at {model_path}")
        os.system(f"python3 huggingface/scripts/evaluation/evaluate_new.py {config_file} {results_path} --chronos-model-id {model_path} --batch-size=32 --device=cuda:0 --num-samples 20")
        mase, wql, rmse, mae = parse_results(results_path)
        if mase is not None and wql is not None and rmse is not None and mae is not None:
            insert_evaluation_result(model_version_id, eval_type, mase, wql, rmse, mae)
            print(f"Inserted {eval_type} evaluation results for ModelVersion {model_version_id}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_models.py <scaling_parameter> <task_id>")
        sys.exit(1)
    evaluate_model(sys.argv[1], int(sys.argv[2]))
