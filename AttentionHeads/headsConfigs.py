import sqlite3
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import os
from itertools import product

# Fixed parameters (note that "num_heads" will be adapted below)
fixed_config = {
    "training_data_paths": [
        "/data/horse/ws/juha972b-Tlm/Tlm/training_mix.arrow",
        "/data/horse/ws/juha972b-Tlm/Tlm/huggingface/scripts/kernelsynth.arrow"
    ],
    "probability": [0.9, 0.1],
    "context_length": 512,
    "prediction_length": 64,
    "min_past": 60,
    "max_steps": 200000,
    "save_steps": 200000,
    "log_steps": 500,
    "optim": "adamw_torch_fused",
    "num_samples": 20,
    "shuffle_buffer_length": 100000,
    "gradient_accumulation_steps": 1,
    "tokenizer_class": "MeanScaleUniformBins",
    "model_id": "google/t5-efficient-tiny",
    "model_type": "seq2seq",
    "random_init": True,
    "tf32": True,
    "torch_compile": True,
    "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
    "dataloader_num_workers": 1,
    "max_missing_prop": 0.9,
    "lr_scheduler_type": "linear",
    "use_eos_token": True,
    "d_model": 512,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 2048
}

# Scaling methods with multiple values for "num_heads"
scaling_values = {
    "num_heads_2": [2],
    "num_heads_4": [4],
    "num_heads_6": [6],
    "num_heads_8": [8]

}

scaling_methods = list(scaling_values.keys())

# Define the search space for tunable parameters
cs = ConfigurationSpace()
cs.add_hyperparameter(CategoricalHyperparameter("n_tokens", [2048, 4096, 8192]))
cs.add_hyperparameter(CategoricalHyperparameter("per_device_train_batch_size", [8, 16, 32]))
cs.add_hyperparameter(CategoricalHyperparameter("learning_rate", [0.01, 0.001, 0.0001]))
cs.add_hyperparameter(CategoricalHyperparameter("warmup_ratio", [0.0, 0.05, 0.1]))
cs.add_hyperparameter(CategoricalHyperparameter("dropout_rate", [0.0, 0.1, 0.2]))
cs.add_hyperparameter(CategoricalHyperparameter("feed_forward_proj", ["relu", "gated-relu"]))
cs.add_hyperparameter(CategoricalHyperparameter("layer_norm_epsilon", [1e-05, 1e-06, 1e-07]))
cs.add_hyperparameter(CategoricalHyperparameter("tie_embeddings", [True, False]))

# Sample configurations
configs = cs.sample_configuration(50)

# Connect to the database   
connection = sqlite3.connect("Heads.db")
cursor = connection.cursor()

# Insert full scaling method combinations into the database
scaling_method_ids = {}
for method in scaling_methods:
    cursor.execute("INSERT INTO ScalingMethods (scaling_method_name) VALUES (?)", (method,))
    scaling_method_ids[method] = cursor.lastrowid

#Adapt scaling value in each config and safe it to DB.
for method, scaling_values_list in scaling_values.items():
    scaling_method_id = scaling_method_ids[method]
    
    for config in configs:
        # Start with the sampled config and update it with the fixed parameters
        base_config = dict(config)
        base_config.update(fixed_config)
        
        # Apply the current scaling value (e.g., num_heads=2, num_heads=4, or num_heads=6)
        config_variant = base_config.copy()  # Create a copy for this variant
        config_variant["num_heads"] = scaling_values_list[0]  # Override "num_heads" with the scaling value
        
        # Convert config to JSON string (using double quotes)
        config_json = str(config_variant).replace("'", '"')
        cursor.execute("""
        INSERT INTO Configs (scaling_method_id, config_json)
        VALUES (?, ?)
        """, (scaling_method_id, config_json))
        config_id = cursor.lastrowid  # Get the unique config_id
        
        # Update output_dir in the configuration to include the config_id and scaling method
        config_variant["output_dir"] = f"./output/{method}/{config_id}/"
        
        # Update the Configs table with the updated output_dir
        config_json_with_config_id = str(config_variant).replace("'", '"')
        cursor.execute("""
        UPDATE Configs
        SET config_json = ?
        WHERE config_id = ?
        """, (config_json_with_config_id, config_id))

# Commit changes and close the connection
connection.commit()
connection.close()

print("Configurations and training runs successfully added to the database.")
