
import os
import sys
import shutil
import argparse
from pathlib import Path

# Add current directory to path to allow importing the sibling module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess_TabFormer_lp import preprocess_data
except ImportError:
    # If running in a different context, try relative import or assume it's in path
    try:
        from src.preprocess_TabFormer_lp import preprocess_data
    except ImportError:
        print("Could not import preprocess_TabFormer_lp. Ensure it is in the python path.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    # SageMaker mounts inputs to a specific directory (e.g. /opt/ml/processing/input)
    # The existing preprocess_data function expects:
    #   base_path/raw/card_transaction.v1.csv
    #   base_path/xgb/ (output)
    #   base_path/gnn/ (output)
    
    # We will use a temporary workspace or the output dir as the base path
    # and symlink the input data to where the function expects it.
    
    base_path = "/opt/ml/processing/workspace"
    raw_dir = os.path.join(base_path, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check if input is a file or directory containing the file
    input_path = Path(args.input_dir)
    source_file = None
    if input_path.is_file():
        source_file = input_path
    else:
        # Search for the expected file
        files = list(input_path.glob("*.csv"))
        if files:
            source_file = files[0]
        else:
            # Fallback for explicit name
            source_file = input_path / "card_transaction.v1.csv"
            
    if not source_file.exists():
        print(f"Error: Input file not found at {source_file}")
        # List dir to help debugging
        print(f"Contents of {args.input_dir}:")
        os.system(f"ls -R {args.input_dir}")
        sys.exit(1)
        
    # Symlink input file to base_path/raw/card_transaction.v1.csv
    target_file = os.path.join(raw_dir, "card_transaction.v1.csv")
    if os.path.exists(target_file):
        os.remove(target_file)
    os.symlink(source_file, target_file)
    
    print(f"Prepared input at {target_file}")
    
    # Run the preprocessing
    print("Starting preprocessing...")
    preprocess_data(base_path)
    print("Preprocessing complete.")
    
    # Move outputs from workspace to SageMaker output directories
    # SageMaker expects outputs in specific locations to upload them to S3
    # The function writes to base_path/xgb and base_path/gnn
    
    source_xgb = os.path.join(base_path, "xgb")
    source_gnn = os.path.join(base_path, "gnn")
    
    # We can map these to the output args or standard SageMaker locations
    # Let's assume we mapped /opt/ml/processing/output/xgb and /opt/ml/processing/output/gnn 
    # in the ProcessingJob definition.
    
    target_xgb = os.path.join(args.output_dir, "xgb")
    target_gnn = os.path.join(args.output_dir, "gnn")
    
    # Move/Copy logic
    if os.path.exists(source_xgb):
        print(f"Moving {source_xgb} to {target_xgb}")
        if os.path.exists(target_xgb):
            shutil.rmtree(target_xgb)
        shutil.copytree(source_xgb, target_xgb)
        
    if os.path.exists(source_gnn):
        print(f"Moving {source_gnn} to {target_gnn}")
        if os.path.exists(target_gnn):
            shutil.rmtree(target_gnn)
        shutil.copytree(source_gnn, target_gnn)

    print("Outputs moved to SageMaker output directories.")
    os.system(f"ls -R {args.output_dir}")

if __name__ == "__main__":
    main()
