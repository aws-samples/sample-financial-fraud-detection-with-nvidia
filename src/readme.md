
#### Container Scripts

Scripts that run inside SageMaker containers:

- **`preprocess_TabFormer_sagemaker.py`** — SageMaker Processing Job entrypoint (calls `preprocess_TabFormer_lp.py`)
- **`preprocess_TabFormer_lp.py`** — Core RAPIDS/cuDF preprocessing logic (graph construction from TabFormer data)
- **`train_sagemaker.py`** — SageMaker Training Job entrypoint (launches GNN + XGBoost training via torchrun)
- **`preprocess_TabFormer.py`** — Standalone preprocessing example (not used by the pipeline)
