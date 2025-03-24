import sqlite3
import json
import os

# Fixed parameters
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
    "per_device_train_batch_size": 32,
    "learning_rate": 0.001,
    "optim": "adamw_torch_fused",
    "num_samples": 20,
    "shuffle_buffer_length": 100000,
    "gradient_accumulation_steps": 1,
    "model_id": "google/t5-efficient-tiny",
    "model_type": "seq2seq",
    "random_init": True,
    "tie_embeddings": True,
    "output_dir": "./output/speedup/{scaling_method}/{config_id}/",
    "tf32": True,
    "torch_compile": True,
    "tokenizer_class": "MeanScaleUniformBins",
    "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
    "n_tokens": 4096,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.0,
    "dataloader_num_workers": 1,
    "max_missing_prop": 0.9,
    "use_eos_token": True,
    "d_model": 512,
    "dropout_rate": 0.1,
    "feed_forward_proj": "relu",
    "layer_norm_epsilon": 1e-06,
    "is_encoder_decoder": True,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 2048
}

scaling_methods = ["d_ff", "num_heads", "num_layers", "context_length", "n_tokens", "max_steps", "d_model", "d_ff_d_model"]

# Scaling values
scaling_values = {
    "d_ff": [16, 32, 64, 128, 265, 512, 1024, 2048],                        #8
    "num_heads": [1, 2, 3, 4, 5, 6, 7, 8],                                  #16
    "num_layers": [1, 2, 3, 4, 5, 6],                                       #22
    "context_length": [4, 8, 16, 32, 64, 128, 256, 512],                    #30
    "n_tokens": [32, 64, 128, 256, 512, 1024, 2048, 4096],                  #38
    "max_steps": [300, 1000, 3000, 10000, 30000, 60000, 100000, 200000],    #46
    "d_model": [4, 8, 16, 32, 64, 128, 256, 512],                           #54
}

# Define d_ff_d_model scaling by pairing corresponding values
scaling_values["d_ff_d_model"] = list(zip(scaling_values["d_ff"], scaling_values["d_model"]))   #62

# Connect to the database
connection = sqlite3.connect("Speedup.db")
cursor = connection.cursor()

# Insert scaling methods into the database
scaling_method_ids = {}
for method in scaling_methods:
    cursor.execute("INSERT INTO ScalingMethods (scaling_method_name) VALUES (?)", (method,))
    scaling_method_ids[method] = cursor.lastrowid

# Generate and insert configurations
for scaling_method, values in scaling_values.items():
    scaling_method_id = scaling_method_ids[scaling_method]  # Get the method ID
    
    for value in values:
        config_dict = fixed_config.copy()

        if scaling_method == "d_ff_d_model":
            config_dict["d_ff"], config_dict["d_model"] = value  # Assign paired values
        else:
            config_dict[scaling_method] = value  # Modify only one parameter
        
        # Insert into Configs table
        config_json = json.dumps(config_dict)
        cursor.execute("INSERT INTO Configs (scaling_method_id, config_json) VALUES (?, ?)", (scaling_method_id, config_json))
        config_id = cursor.lastrowid
        
        # Update output_dir in the config to include the config_id
        config_dict["output_dir"] = fixed_config["output_dir"].format(scaling_method=scaling_method, config_id=config_id)
        
        # Update the Configs table with the updated output_dir
        config_json_with_config_id = json.dumps(config_dict)
        cursor.execute("UPDATE Configs SET config_json = ? WHERE config_id = ?", (config_json_with_config_id, config_id))

# Commit changes and close the connection
connection.commit()
connection.close()

print("Configurations and training runs successfully added to the database")