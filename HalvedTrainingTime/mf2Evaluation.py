import os
import sys
import sqlite3
import csv
import time

#defines scaling params and config ranges. Since we safed model checkpoints after each 10000 steps we use the chekpoints of the default configurations 
#to evaluate the modified number of traing steps 
SCALING_PARAMS = {
    "default": range(1, 51),
    "context_length": range(51, 101),
    "num_heads": range(101, 151),
    "num_layers": range(151, 201),
    "max_steps": range(1, 51),  # Use the default model IDs 1-50
    "steps_10000": range(1, 51),
    "steps_30000": range(1, 51),
    "steps_60000": range(1, 51),
}

BASE_PATH = "/data/horse/ws/juha972b-Tlm/Tlm/output"
EVAL_CONFIGS = {
    "in-domain": "huggingface/scripts/evaluation/configs/in-domain.yaml",
    "zero-shot": "huggingface/scripts/evaluation/configs/zero-shot.yaml"
}
DB_PATH = "MF2.db"

#map scaling params to model checkpoints
CHECKPOINT_MAP = {
    "max_steps": "checkpoint-100000",
    "steps_10000": "checkpoint-10000",
    "steps_30000": "checkpoint-30000",
    "steps_60000": "checkpoint-60000",
    "default": "checkpoint-final",
}

MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

def execute_with_retries(query, params=(), fetchone=False, commit=False):
    """Executes a SQLite query with retries in case of database locking."""
    for attempt in range(MAX_RETRIES):
        try:
            connection = sqlite3.connect(DB_PATH)
            cursor = connection.cursor()
            cursor.execute(query, params)
            
            if commit:
                connection.commit()
                connection.close()
                return True
            else:
                result = cursor.fetchone() if fetchone else cursor.fetchall()
                connection.close()
                return result

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database is locked. Retrying {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Database error: {e}")
                sys.exit(1)

    print("Error: Maximum retries reached. Exiting.")
    sys.exit(1)

def evaluation_exists(model_version_id, eval_type):
    '''Checks if an evlauation result already exists'''
    query = "SELECT 1 FROM EvaluationResults WHERE model_version_id = ? AND evaluation_type = ?"
    return execute_with_retries(query, (model_version_id, eval_type), fetchone=True) is not None

def insert_evaluation_result(model_version_id, eval_type, mase, wql, rmse, mae):
    '''Inserts result, skips if already exists'''
    if evaluation_exists(model_version_id, eval_type):
        print(f"Skipping {eval_type} evaluation for ModelVersion {model_version_id}, already exists.")
        return
    
    query = """
        INSERT INTO EvaluationResults (model_version_id, evaluation_type, mase, wql, rmse, mae)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    execute_with_retries(query, (model_version_id, eval_type, mase, wql, rmse, mae), commit=True)

def get_model_version_id(config_id, training_step):
    '''retrieves the model_version_id based on config_id and trainigs steps(checkpoint)'''
    run = execute_with_retries("SELECT run_id FROM TrainingRuns WHERE config_id = ?", (config_id,), fetchone=True)
    
    if not run:
        print(f"Error: No TrainingRun found for config_id {config_id}.")
        sys.exit(1)
    
    run_id = run[0]
    version = execute_with_retries(
        "SELECT model_version_id FROM ModelVersions WHERE run_id = ? AND training_step = ?",
        (run_id, training_step),
        fetchone=True
    )

    if version:
        return version[0]
    else:
        print(f"Error: No ModelVersion found for run_id {run_id} and training_step {training_step}.")
        sys.exit(1)

def parse_results(results_path):
    '''Retrieves results from CSV files created by evaluate_new script'''
    try:
        mase_values, wql_values, rmse_values, mae_values = [], [], [], []
        with open(results_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
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
    '''evaluated a single model based on scaling param and slurm task id'''
    if scaling_param not in SCALING_PARAMS:
        print(f"Error: Invalid scaling parameter '{scaling_param}'.")
        sys.exit(1)
    
    config_ids = list(SCALING_PARAMS[scaling_param])
    if task_id < 0 or task_id >= len(config_ids):
        print(f"Error: Task ID {task_id} out of range for scaling parameter '{scaling_param}'.")
        sys.exit(1)
    
    config_id = config_ids[task_id]
    training_step = int(scaling_param.split('_')[-1]) if "steps" in scaling_param else 200000
    checkpoint = CHECKPOINT_MAP.get(scaling_param, "checkpoint-final")
    
    #creates model paths
    model_path = f"{BASE_PATH}/default/{config_id}/run-0/{checkpoint}" if "steps" in scaling_param else f"{BASE_PATH}/{scaling_param}/{config_id}/run-0/{checkpoint}"
    results_dir = f"{model_path}/results"
    os.makedirs(results_dir, exist_ok=True)
    model_version_id = get_model_version_id(config_id, training_step)
    
    #runs in-domain and zero-shot evaluation
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
