import os
import json
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker Hyperparameters
    parser.add_argument("--gnn_hidden_channels", type=int, default=32)
    parser.add_argument("--gnn_n_hops", type=int, default=2)
    parser.add_argument("--gnn_layer", type=str, default="SAGEConv")
    parser.add_argument("--gnn_dropout_prob", type=float, default=0.1)
    parser.add_argument("--gnn_batch_size", type=int, default=4096)
    parser.add_argument("--gnn_fan_out", type=int, default=10)
    parser.add_argument("--gnn_num_epochs", type=int, default=8)
    
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.2)
    parser.add_argument("--xgb_num_parallel_tree", type=int, default=3)
    parser.add_argument("--xgb_num_boost_round", type=int, default=512)
    parser.add_argument("--xgb_gamma", type=float, default=0.0)
    
    # Container environment paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_GNN", "/opt/ml/input/data/gnn"))
    
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    
    print("Received arguments:", args)
    
    # Define output directory for the model artifacts
    # The GNN training script writes to 'output_dir'
    # SageMaker expects artifacts in SM_MODEL_DIR to be tarred and uploaded
    # The Triton server expects a specific structure (python_backend_model_repository)
    output_dir = os.path.join(args.model_dir, "python_backend_model_repository")
    os.makedirs(output_dir, exist_ok=True)

    # Construct config.json
    config = {
        "paths": {
            "data_dir": args.train_dir,
            "output_dir": output_dir
        },
        "models": [
            {
                "kind": "GNN_XGBoost",
                "gpu": "single",
                "hyperparameters": {
                    "gnn": {
                        "hidden_channels": args.gnn_hidden_channels,
                        "n_hops": args.gnn_n_hops,
                        "layer": args.gnn_layer,
                        "dropout_prob": args.gnn_dropout_prob,
                        "batch_size": args.gnn_batch_size,
                        "fan_out": args.gnn_fan_out,
                        "num_epochs": args.gnn_num_epochs,
                    },
                    "xgb": {
                        "max_depth": args.xgb_max_depth,
                        "learning_rate": args.xgb_learning_rate,
                        "num_parallel_tree": args.xgb_num_parallel_tree,
                        "num_boost_round": args.xgb_num_boost_round,
                        "gamma": args.xgb_gamma,
                    },
                },
            }
        ],
    }
    
    config_path = "/app/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated {config_path}:")
    print(json.dumps(config, indent=2))
    
    # Execute the original training script
    # The source code is at /app/main.py in the NGC container
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "/app/main.py",
        "--config",
        config_path
    ]
    
    print("Executing command:", " ".join(cmd))

    # Ensure CUDA forward compatibility libs are used
    env = os.environ.copy()
    cuda_compat_path = "/usr/local/cuda/compat"
    if os.path.exists(cuda_compat_path):
        env["LD_LIBRARY_PATH"] = f"{cuda_compat_path}:{env.get('LD_LIBRARY_PATH', '')}"
        print(f"Set LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")

    # Run and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",  # Run from /app to ensure imports work if needed
        env=env
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        sys.exit(process.returncode)
        
    print("Training complete.")
    
    # Verify output
    if os.path.exists(output_dir):
        print(f"Contents of {output_dir}:")
        os.system(f"ls -R {output_dir}")
    else:
        print(f"Warning: Output directory {output_dir} does not exist.")

if __name__ == "__main__":
    main()
