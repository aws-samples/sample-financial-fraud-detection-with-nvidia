import argparse
import json
import os
import subprocess
import sys


def setup_cuda_compat():
    """Find and configure CUDA forward compatibility libraries."""
    import glob
    # Known CUDA compat paths (in order of preference)
    cuda_compat_paths = [
        "/usr/local/cuda-13.0/compat",
        "/usr/local/cuda-13/compat",
        "/usr/local/cuda/compat",
    ]

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    # Find and add compat paths to LD_LIBRARY_PATH
    # Find valid compat paths and prepend them
    valid_paths = []
    for path in cuda_compat_paths:
        if os.path.isdir(path) and path not in current_ld_path:
            valid_paths.append(path)
            print(f"Adding CUDA compat path: {path}")
            if os.path.isdir(path):
                print(f"  Contents: {os.listdir(path)[:5]}")

    if valid_paths:
        new_ld_path = ":".join(valid_paths) + ":" + current_ld_path if current_ld_path else ":".join(valid_paths)
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        print(f"LD_LIBRARY_PATH now starts with: {new_ld_path[:150]}...")
    else:
        # Check if compat is already first
        if any(current_ld_path.startswith(p) for p in cuda_compat_paths):
            print(f"CUDA compat already configured in LD_LIBRARY_PATH")
        else:
            print("WARNING: No CUDA compat paths found!")
            os.system("ls -la /usr/local/cuda*/compat/ 2>/dev/null || echo 'No compat dirs'")

    # Also try LD_PRELOAD for the specific libcuda.so to ensure it loads first
    for path in cuda_compat_paths:
        libcuda_files = glob.glob(os.path.join(path, "libcuda.so.*.*"))
        if libcuda_files:
            # Get the actual library file (not symlink targets)
            libcuda = sorted(libcuda_files)[-1]  # Get newest version
            current_preload = os.environ.get("LD_PRELOAD", "")
            if libcuda not in current_preload:
                new_preload = f"{libcuda}:{current_preload}" if current_preload else libcuda
                os.environ["LD_PRELOAD"] = new_preload
                print(f"Set LD_PRELOAD={new_preload}")
            break

    print(f"Final LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')[:200]}")


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
