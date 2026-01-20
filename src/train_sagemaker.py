import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(
    cmd: list[str] | str, check: bool = False, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run a command using subprocess."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def setup_cuda_compat():
    """Find and configure CUDA forward compatibility libraries."""
    cuda_compat_paths = [
        "/usr/local/cuda-13.0/compat",
        "/usr/local/cuda-13/compat",
        "/usr/local/cuda/compat",
        "/usr/local/cuda/compat/lib.real",
    ]

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    valid_paths = [
        p for p in cuda_compat_paths if Path(p).is_dir() and p not in current_ld_path
    ]

    if valid_paths:
        new_ld_path = ":".join(valid_paths) + (
            ":" + current_ld_path if current_ld_path else ""
        )
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        print(f"Added CUDA compat paths: {valid_paths}")
    elif any(current_ld_path.startswith(p) for p in cuda_compat_paths):
        print("CUDA compat already configured in LD_LIBRARY_PATH")
    else:
        print("WARNING: No CUDA compat paths found")

    # Try LD_PRELOAD for libcuda.so to ensure it loads first
    for path in cuda_compat_paths:
        libcuda_files = glob.glob(os.path.join(path, "libcuda.so.*.*"))
        if libcuda_files:
            libcuda = sorted(libcuda_files)[-1]
            current_preload = os.environ.get("LD_PRELOAD", "")
            if libcuda not in current_preload:
                os.environ["LD_PRELOAD"] = (
                    f"{libcuda}:{current_preload}" if current_preload else libcuda
                )
                print(f"Set LD_PRELOAD to use: {libcuda}")
            break


def parse_args():
    parser = argparse.ArgumentParser()

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
    setup_cuda_compat()

    result = run_cmd(["nvidia-smi"], capture=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("nvidia-smi not available")

    args, _ = parse_args()
    print(f"Arguments: {args}")

    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "paths": {"data_dir": args.train_dir, "output_dir": str(output_dir)},
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

    config_path = Path("/app/config.json")
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Generated {config_path}:\n{json.dumps(config, indent=2)}")

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "/app/main.py",
        "--config",
        str(config_path),
    ]

    print(f"Executing: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
        env=os.environ.copy(),
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        sys.exit(process.returncode)

    print("Training complete.")

    if output_dir.exists():
        print(f"Contents of {output_dir}:")
        for f in output_dir.rglob("*"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
