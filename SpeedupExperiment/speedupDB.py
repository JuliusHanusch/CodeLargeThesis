import sqlite3

# Connect to SQLite (or create the database if it doesn't exist)
connection = sqlite3.connect("Speedup.db")  # New database file
cursor = connection.cursor()

# Create the ScalingMethods table
cursor.execute("""
CREATE TABLE IF NOT EXISTS ScalingMethods (
    scaling_method_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scaling_method_name TEXT NOT NULL
);
""")

# Create the Configs table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Configs (
    config_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scaling_method_id INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    FOREIGN KEY (scaling_method_id) REFERENCES ScalingMethods(scaling_method_id)
);
""")

# Create the TrainingRuns table
cursor.execute("""
CREATE TABLE IF NOT EXISTS TrainingRuns (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES Configs(config_id)
);
""")

# Commit the changes and close the connection
connection.commit()
connection.close()

print("Database 'Speedup.db' created successfully.")
