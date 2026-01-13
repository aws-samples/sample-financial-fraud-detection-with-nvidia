# SageMaker Pipeline Workflows

This directory contains the SageMaker Pipelines implementation for the financial fraud detection ML workflow.

Based on the original NVIDIA blueprint: https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection/tree/main/src

## Overview

The pipeline orchestrates three main steps:
1. **Preprocessing** - RAPIDS-based GPU preprocessing of transaction data
2. **Training** - Combined GNN (GraphSAGE) and XGBoost model training
3. **Model Registration** - Package and register the trained model for deployment

## Pipeline Definition

The main pipeline is defined in `src/workflows/sagemaker_fraud_detection_pipeline.py`.

## Running the Pipeline

```bash
# Install dependencies
uv sync

# Deploy the pipeline
python -m src.workflows.sagemaker_fraud_detection_pipeline \
  --role-arn <sagemaker-execution-role-arn> \
  --default-bucket <s3-bucket-name> \
  --region <aws-region> \
  --profile <aws-profile>  # optional
```

The pipeline can also be triggered through:
- SageMaker Studio UI
- AWS CLI
- SageMaker Python SDK
- CloudFormation/CDK

## Architecture

The pipeline uses SageMaker's native capabilities:
- **Processing Jobs** for GPU-accelerated preprocessing
- **Training Jobs** for model training
- **Model Registry** for versioning and deployment
- **Step Caching** to avoid re-running unchanged steps

All intermediate data is stored in S3, with automatic artifact management by SageMaker.
