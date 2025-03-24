import sqlite3
import yaml
import os
import subprocess
import sys
import tempfile
import datetime
import json
import time


def train_model(config_id, config_json):
    """Runs training for a single configuration and logs start/end time in DB."""
    
    DB_PATH = "Speedup.db"  # Updated to use Speedup.db
    TRAIN_SCRIPT = "chronos-forecasting/scripts/training/train.py"

    # Connect to database with a busy timeout to avoid locking errors
    connection = sqlite3.connect(DB_PATH, timeout=30)  # 30 seconds timeout
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
    config = json.loads(config_json)  

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


def main():
    """Fetches a specific configuration from Speedup.db based on the provided config_id and trains it."""
    
    # Ensure a config_id is provided
    if len(sys.argv) != 2:
        print("Usage: python speedupRunsAll.py <config_id>")
        sys.exit(1)

    # Get the config_id passed from the SLURM job array
    config_id = int(sys.argv[1])

    DB_PATH = "Speedup.db"

    # Connect to DB with a retry mechanism and timeout
    connection = sqlite3.connect(DB_PATH, timeout=30)  # 30 seconds timeout
    cursor = connection.cursor()

    # Fetch the configuration for the given config_id
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            cursor.execute("SELECT config_json FROM Configs WHERE config_id = ?", (config_id,))
            config = cursor.fetchone()

            if not config:
                print(f"No configuration found for config_id: {config_id}")
                sys.exit(1)

            config_json = config[0]
            break
        except sqlite3.OperationalError as e:
            if attempt < retry_attempts - 1:
                print(f"Database is locked, retrying ({attempt + 1}/{retry_attempts})...")
                time.sleep(5)  # wait 5 seconds before retrying
            else:
                print(f"Failed to fetch configuration after {retry_attempts} attempts. Error: {e}")
                sys.exit(1)

    # Close the DB connection
    connection.close()

    print(f"Starting training for config ID: {config_id}")

    # Call the function to train the model with the fetched configuration
    train_model(config_id, config_json)


if __name__ == "__main__":
    main()

