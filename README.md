# GNN based financial fraud detection on AWS with Nvidia

Financial fraud costs institutions billions annually, and traditional rule-based systems struggle to keep pace with increasingly sophisticated attack patterns. This blueprint demonstrates a different approach: using Graph Neural Networks to analyze the relationships between transactions, merchants, and cardholders as a connected graph rather than isolated events.

The insight is simple. Fraudsters don't operate in isolation. They create patterns across multiple accounts, merchants, and time windows that become visible when you model transactions as a graph. A single suspicious transaction might look normal. That same transaction connected to a web of related activity tells a different story.

![Architecture](docs/arch-diagram.png)

This implementation runs entirely on AWS using SageMaker Pipelines for orchestration, RAPIDS for GPU-accelerated preprocessing, and NVIDIA Triton for inference. The infrastructure deploys via CDK with a fully managed, serverless approach that eliminates cluster management overhead.

## The Pipeline

When you trigger a pipeline run, three stages execute in sequence.

**Preprocessing** transforms raw transactions into graph structure using RAPIDS. The TabFormer dataset (IBM's synthetic credit card dataset with 24 million transactions across 5,000 cardholders and 1,000 merchants) gets converted into a graph where cardholders and merchants become nodes, and transactions become edges. This runs on GPU-accelerated SageMaker Processing Jobs. What would take hours with pandas completes in minutes with cuDF.

**Training** combines a Graph Neural Network (GraphSAGE) with XGBoost for fraud prediction. The GNN learns embeddings from the transaction graph structure while XGBoost handles the tabular features. Together they outperform either alone. SageMaker Training Jobs orchestrate this on GPU instances, automatically handling checkpointing, metrics logging, and artifact storage.

**Model Registration** packages the trained model and registers it in SageMaker Model Registry with approval workflow. The model includes GNN weights, XGBoost model, and configuration for NVIDIA Triton inference serving. Models awaiting approval can be reviewed in SageMaker Studio before deployment.

## Getting Started

You'll need an AWS account with permissions for SageMaker, ECR, S3, IAM, CloudFormation, CodeBuild, and Secrets Manager. Locally you'll need Docker, Node.js 20+, Python 3.12+, AWS CLI v2, and `uv` for Python package management. The full deployment takes about 30 minutes.

First, store your NVIDIA NGC API key. The infrastructure pulls base images from NVIDIA GPU Cloud:

```bash
aws secretsmanager create-secret \
  --name nvidia-ngc-api-key \
  --secret-string "<your-ngc-api-key>" \
  --profile <your-aws-profile>
```

Install dependencies and deploy:

```bash
make install

# Bootstrap CDK (first time only)
cd infra && npx cdk bootstrap aws://<ACCOUNT>/<REGION> --profile <your-aws-profile>

# Deploy everything
make cdk-deploy-all
```

This creates S3 buckets for data and models, ECR repositories for custom containers, SageMaker execution roles, a SageMaker Domain for Studio access, and CodeBuild projects that automatically build container images.

Wait for the container images to finish building. This takes 15-20 minutes:

```bash
# Check build status
aws codebuild list-builds-for-project \
  --project-name triton-inference-image-build \
  --profile <your-aws-profile>
```

Upload the training data:

```bash
aws s3 cp card_transaction.v1.csv \
  s3://fraud-detection-<account>-sm/data/TabFormer/raw/ \
  --profile <your-aws-profile>
```

## Running Your First Pipeline

Register the pipeline with SageMaker:

```bash
make pipeline
```

This creates the `FraudDetectionPipeline` in SageMaker. You can execute it from Studio or the CLI.

To run from Studio, go to AWS Console → SageMaker → Domains, launch Studio, click Pipelines in the sidebar, find FraudDetectionPipeline, and click Create execution. The UI shows real-time progress with logs and metrics for each step.

To run from CLI:

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name FraudDetectionPipeline \
  --profile <your-aws-profile>
```

The pipeline runs through PreprocessData (~15 minutes), TrainModel (~5 minutes), and RegisterModel (~2 minutes). SageMaker automatically provisions GPU instances, runs containers, stores artifacts, and cleans up when complete.

## Deploying to an Endpoint

Once training completes, the model registers in Model Registry with status `PendingManualApproval`. Approve it in the SageMaker console, then deploy:

```bash
make register   # Register new model package with latest Triton image
make deploy     # Deploy endpoint with defaults
```

To customize the deployment:

```bash
make deploy ENDPOINT_NAME=fraud-prod INSTANCE_TYPE=ml.g5.xlarge
```

The endpoint runs NVIDIA Triton and accepts merchant features, cardholder features, and graph structure as inputs. It returns fraud probability plus Shapley values for explainability.

## Quick Reference

All commands run from the project root. The Makefile handles CloudFormation lookups automatically.

```bash
# Setup
make install              # Install CDK + Python dependencies
make info                 # Show current configuration

# Infrastructure
make cdk-list             # List all stacks
make cdk-deploy-all       # Deploy everything
make cdk-diff             # Preview changes

# Pipeline
make pipeline             # Create/update pipeline definition

# Model & Deployment
make register             # Register new model version
make deploy               # Deploy endpoint
make deploy ENDPOINT_NAME=my-endpoint INSTANCE_TYPE=ml.g4dn.xlarge

# Image Building
make build-triton         # Rebuild Triton image
make build-training       # Rebuild training image
make build-preprocessing  # Rebuild preprocessing image
make build-all            # Rebuild all images

# Utilities
make logs                 # Fetch latest endpoint logs
make clean-endpoints      # Delete all endpoint configs
```

Override the AWS profile or region with environment variables:

```bash
AWS_PROFILE=my-profile AWS_REGION=us-west-2 make deploy
```

## Understanding the Workflow

The `workflows/` directory contains the SageMaker pipeline definition:

```
workflows/
├── src/workflows/
│   └── sagemaker_fraud_detection_pipeline.py
├── pyproject.toml
└── uv.lock
```

SageMaker manages artifact passing between steps via S3:

```
┌──────────────────────┐
│  Raw Data (S3)       │
│  TabFormer Dataset   │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Preprocessing       │
│  (Processing Job)    │
│  RAPIDS/cuDF on GPU  │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Processed Data (S3) │
│  GNN graph + XGB     │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Training            │
│  (Training Job)      │
│  GNN + XGBoost       │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Model Registry      │
│  Versioning +        │
│  Approval workflow   │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Triton Endpoint     │
│  Real-time inference │
│  + Explainability    │
└──────────────────────┘
```

The model is called `prediction_and_shapley`. It takes merchant features, user features, and graph edge information as inputs. It returns fraud probability plus Shapley values that explain which features contributed most to the prediction. This explainability matters for regulatory compliance and fraud analyst workflows.

## What's Next

The default hyperparameters work reasonably well, but you can tune them through pipeline parameters:

- `GnnNumEpochs`: More epochs generally improve accuracy but increase training time (default: 8)
- `XgbNumBoostRound`: XGBoost boosting rounds, higher values risk overfitting (default: 512)
- `GnnHiddenChannels`: Width of the GNN layers, larger captures more patterns (default: 32)
- `GnnNHops`: Number of graph neighborhood hops (default: 2)

Adjust these when starting a pipeline execution:

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name FraudDetectionPipeline \
  --pipeline-parameters \
    Name=GnnNumEpochs,Value=12 \
    Name=XgbNumBoostRound,Value=768 \
  --profile <your-aws-profile>
```

To use your own data, replace the `InputDataUrl` parameter with your S3 path. Ensure your CSV matches TabFormer's schema or modify `src/preprocess_TabFormer_sagemaker.py` for different data formats.

## Cleanup

```bash
make cdk-destroy STACK=SageMakerTritonEndpointStack
make cdk-destroy STACK=SageMakerDomainStack
# Continue for other stacks, or:
cd infra && npx cdk destroy --all --profile <your-aws-profile>
```

S3 buckets are configured with auto-delete. You may need to manually delete SageMaker endpoints and model packages before destroying the infrastructure stacks.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for our contribution guide.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for reporting security issues.

## License

MIT-0 License. See [LICENSE](LICENSE).

## Authors

- Shardul Vaidya, AWS Partner Solutions Architect
- Zachary Jacobson, AWS Partner Solutions Architect
- Ragib Ahsan, AWS AI Acceleration Architect
- Artiom Troyanovsky, Senior Machine Learning Engineer at Thoughtworks
