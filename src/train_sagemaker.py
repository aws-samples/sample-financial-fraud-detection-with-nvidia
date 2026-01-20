import argparse
import json
import os
import subprocess
import sys


def setup_cuda_compat():
    """Find and configure CUDA forward compatibility libraries."""
    # Search for cuda-compat libraries
    compat_paths = [
        "/usr/local/cuda-13.0/compat",
        "/usr/local/cuda/compat",
        "/usr/lib/x86_64-linux-gnu",
    ]

    # Also search for any cuda-compat installation
    for cuda_dir in ["/usr/local/cuda-13.0", "/usr/local/cuda-13", "/usr/local/cuda"]:
        compat_dir = os.path.join(cuda_dir, "compat")
        if os.path.isdir(compat_dir):
            compat_paths.insert(0, compat_dir)

    # Find where libcuda.so actually lives in compat
    found_compat = None
    for path in compat_paths:
        if os.path.isdir(path):
            files = os.listdir(path)
            if any("libcuda" in f for f in files):
                found_compat = path
                print(f"Found CUDA compat libraries in: {path}")
                print(f"Contents: {files[:10]}...")
                break

    if found_compat:
        # Prepend to LD_LIBRARY_PATH
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld_path = (
            f"{found_compat}:{current_ld_path}" if current_ld_path else found_compat
        )
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        print(f"Set LD_LIBRARY_PATH={new_ld_path[:200]}...")
    else:
        print("WARNING: Could not find CUDA compat libraries!")
        # Debug: show what's installed
        os.system("find /usr/local -name 'libcuda*' 2>/dev/null | head -20")
        os.system(
            "ls -la /usr/local/cuda*/compat/ 2>/dev/null || echo 'No compat dirs found'"
        )


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
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_GNN", "/opt/ml/input/data/gnn"),
    )

    return parser.parse_known_args()


def main():
    # Setup CUDA compat first, before any CUDA initialization
    print("=" * 60)
    print("CUDA Forward Compatibility Setup")
    print("=" * 60)
    setup_cuda_compat()

    # Debug current environment
    print(f"\nLD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:300]}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    os.system("nvidia-smi 2>/dev/null || echo 'nvidia-smi not available'")
    print("=" * 60)

    args, unknown = parse_args()

    print("Received arguments:", args)

    # Define output directory for the model artifacts
    output_dir = os.path.join(args.model_dir, "python_backend_model_repository")
    os.makedirs(output_dir, exist_ok=True)

    # Construct config.json
    config = {
        "paths": {"data_dir": args.train_dir, "output_dir": output_dir},
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
    cmd = [
        "CUDA_VISIBLE_DEVICES=0",
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "/app/main.py",
        "--config",
        config_path,
    ]

    print("Executing command:", " ".join(cmd))

    # Run with explicit environment including LD_LIBRARY_PATH
    env = os.environ.copy()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
        env=env,
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
