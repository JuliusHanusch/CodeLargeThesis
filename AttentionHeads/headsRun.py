import sqlite3
import yaml
import os
import subprocess
import sys
import tempfile
import datetime
import time

#functions which trys to connect to DB and retrys if if fails
def connect_with_retry(db_path, retries=5, delay=1):
    """Attempt to connect to the database, retrying if locked."""
    for attempt in range(retries):
        try:
            connection = sqlite3.connect(db_path)
            return connection
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database is locked, retrying {attempt + 1}/{retries}...")
                time.sleep(delay)
            else:
                raise
    raise sqlite3.OperationalError("Database is locked after multiple retries")

def train_model(scaling_method, config_id, config_json):
    """Runs training for a single configuration based on scaling method and config_ig and logs start/end time in DB."""
    
    DB_PATH = "Heads.db"
    TRAIN_SCRIPT = "huggingface/scripts/training/train.py"

    # Connect to database with retry
    connection = connect_with_retry(DB_PATH)
    cursor = connection.cursor()

    # Record the start time
    start_time = datetime.datetime.now()
    print(f"Training started for config ID: {config_id} at {start_time}")

    # Insert new run entry in TrainingRuns table
    cursor.execute(
        "INSERT INTO TrainingRuns (config_id, start_time) VALUES (?, ?)",
        (config_id, start_time),
    )
    run_id = cursor.lastrowid  # Get the newly created run ID
    connection.commit()

    # Convert config JSON to dictionary
    config = eval(config_json)  

    # Create a temporary YAML config file
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = os.path.join(temp_dir, f"config_run_{config_id}.yaml")
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(config, yaml_file)

        # Run training
        result = subprocess.run(
            ["python3", TRAIN_SCRIPT, "--config", yaml_path],
            capture_output=True,
            text=True
        )

        # Check for errors
        if result.returncode != 0:
            print(f"Training failed for config ID: {config_id}. Error: {result.stderr}")
            connection.close()
            return

        # Define model checkpoint paths
        trained_model_paths = {
            step: f"./output/{scaling_method}/{config_id}/run-0/checkpoint-{step}" if step != 200000 else f"./output/{scaling_method}/{config_id}/run-0/checkpoint-final"
            for step in range(10000, 210000, 10000)
        }

        # Insert model versions into DB
        for step, model_checkpoint_path in trained_model_paths.items():
            if os.path.exists(model_checkpoint_path):
                cursor.execute(
                    """
                    INSERT INTO ModelVersions (run_id, training_step, model_path)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, step, model_checkpoint_path),
                )
            else:
                print(f"Warning: Model checkpoint not found for step {step} at {model_checkpoint_path}")

    # Record the end time
    end_time = datetime.datetime.now()
    print(f"Training completed for config ID: {config_id} at {end_time}")

    # Update the TrainingRuns table with end time
    cursor.execute(
        "UPDATE TrainingRuns SET end_time = ? WHERE run_id = ?",
        (end_time, run_id),
    )
    
    # Commit DB changes
    connection.commit()
    connection.close()

def main(scaling_method, task_id):
    """Fetches configurations for the given scaling method and assigns one per SLURM array job."""
    
    DB_PATH = "Heads.db"

    # Connect to DB with retry
    connection = connect_with_retry(DB_PATH)
    cursor = connection.cursor()

    # Get scaling method ID
    cursor.execute("SELECT scaling_method_id FROM ScalingMethods WHERE scaling_method_name = ?", (scaling_method,))
    scaling_method_result = cursor.fetchone()

    if not scaling_method_result:
        print(f"No scaling method found for: {scaling_method}")
        connection.close()
        sys.exit(1)

    scaling_method_id = scaling_method_result[0]

    # Fetch ONLY configurations for this scaling method
    cursor.execute("SELECT config_id, config_json FROM Configs WHERE scaling_method_id = ?", (scaling_method_id,))
    configs = cursor.fetchall()
    connection.close()

    if not configs:
        print(f"No configurations found for scaling method: {scaling_method}")
        sys.exit(1)

    # Ensure task ID is within range
    if task_id >= len(configs):
        print(f"Task ID {task_id} is out of range. Only {len(configs)} configurations available.")
        sys.exit(1)

    # Select the correct config using task_id as an index into the filtered list
    config_id, config_json = configs[task_id]
    train_model(scaling_method, config_id, config_json)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scaledRun.py <scaling_method> <task_id>")
        sys.exit(1)

    scaling_method = sys.argv[1]
    task_id = int(sys.argv[2])
    main(scaling_method, task_id)
