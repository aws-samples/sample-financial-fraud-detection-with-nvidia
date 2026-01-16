import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess_TabFormer_lp import preprocess_data
except ImportError:
    try:
        from src.preprocess_TabFormer_lp import preprocess_data
    except ImportError:
        print(
            "Could not import preprocess_TabFormer_lp. Ensure it is in the python path."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    # Use /tmp as working directory since SageMaker output dirs are pre-created
    # and the parent is read-only. preprocess_data() expects:
    #   base_path/raw/card_transaction.v1.csv (input)
    #   base_path/xgb/ (output)
    #   base_path/gnn/ (output)
    base_path = "/tmp/tabformer"
    raw_dir = os.path.join(base_path, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Find the input CSV file
    input_path = Path(args.input_dir)
    source_file = None
    if input_path.is_file():
        source_file = input_path
    else:
        files = list(input_path.glob("*.csv"))
        if files:
            source_file = files[0]
        else:
            source_file = input_path / "card_transaction.v1.csv"

    if not source_file.exists():
        print(f"Error: Input file not found at {source_file}")
        print(f"Contents of {args.input_dir}:")
        os.system(f"ls -R {args.input_dir}")
        sys.exit(1)

    # Copy input file to base_path/raw/
    target_file = os.path.join(raw_dir, "card_transaction.v1.csv")
    print(f"Copying {source_file} to {target_file}")
    shutil.copy2(source_file, target_file)

    print("Starting preprocessing...")
    preprocess_data(base_path)
    print("Preprocessing complete.")

    # Copy outputs to SageMaker output directories
    xgb_src = os.path.join(base_path, "xgb")
    gnn_src = os.path.join(base_path, "gnn")
    xgb_dst = os.path.join(args.output_dir, "xgb")
    gnn_dst = os.path.join(args.output_dir, "gnn")

    print(f"Copying {xgb_src} to {xgb_dst}")
    for f in os.listdir(xgb_src):
        shutil.copy2(os.path.join(xgb_src, f), os.path.join(xgb_dst, f))

    print(f"Copying {gnn_src} to {gnn_dst}")
    shutil.copytree(gnn_src, gnn_dst, dirs_exist_ok=True)

    print(f"Output contents:")
    os.system(f"ls -R {args.output_dir}")


if __name__ == "__main__":
    main()
