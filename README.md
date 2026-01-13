# GNN based financial fraud detection on AWS with Nvidia

Financial fraud costs institutions billions annually, and traditional rule-based systems struggle to keep pace with increasingly sophisticated attack patterns. This blueprint demonstrates a different approach: using Graph Neural Networks to analyze the relationships between transactions, merchants, and cardholders as a connected graph rather than isolated events.

The insight is simple. Fraudsters don't operate in isolation. They create patterns across multiple accounts, merchants, and time windows that become visible when you model transactions as a graph. A single suspicious transaction might look normal. That same transaction connected to a web of related activity tells a different story.

![Architecture](docs/arch-diagram.png)

This implementation runs entirely on AWS using SageMaker Pipelines for orchestration, RAPIDS for GPU-accelerated preprocessing, and NVIDIA Triton for inference. The infrastructure deploys via CDK with a serverless, managed approach that eliminates cluster management overhead.

## The Pipeline

When you trigger a pipeline run, three stages execute:

**Preprocessing** transforms raw transactions into graph structure using RAPIDS. The TabFormer dataset (IBM's synthetic credit card dataset with 24 million transactions across 5,000 cardholders and 1,000 merchants) gets converted into a graph where cardholders and merchants become nodes, and transactions become edges. This runs on GPU-accelerated SageMaker Processing Jobs using custom RAPIDS containers. What would take hours with pandas completes in minutes with cuDF.

**Training** combines a Graph Neural Network (GraphSAGE) with XGBoost for fraud prediction. The GNN learns embeddings from the transaction graph structure while XGBoost handles the tabular features. SageMaker Training Jobs orchestrate this on GPU instances, automatically handling checkpointing, metrics logging, and artifact storage to S3.

**Model Registration** packages the trained model and registers it in SageMaker Model Registry with approval workflow. The model includes GNN weights, XGBoost model, and integration with NVIDIA Triton for inference serving. Models awaiting approval can be reviewed in SageMaker Studio before deployment.

## Getting Started

You'll need an AWS account with permissions for SageMaker, ECR, S3, and IAM. Locally you'll need Docker, Node.js 20+, and AWS CLI. The deployment takes about 15 minutes.

```bash
cd infra
npm install

# First time only
npx cdk bootstrap aws://<ACCOUNT>/<REGION>

# Set your environment
export CDK_DEFAULT_ACCOUNT=<your-account>
export CDK_DEFAULT_REGION=<your-region>

# Deploy the infrastructure
npx cdk deploy --all
```

This creates:
- S3 buckets for data and models
- ECR repositories for custom containers (RAPIDS preprocessing, training, Triton)
- SageMaker execution roles with appropriate permissions
- SageMaker Domain for Studio access
- CodeBuild projects that automatically build container images

The CDK deployment outputs the SageMaker execution role ARN and S3 bucket names. Save these for pipeline deployment.

Once infrastructure deploys, upload your data to S3:

```bash
# Example: Upload TabFormer dataset
aws s3 cp card_transaction.v1.csv s3://fraud-detection-<account>-sm/data/TabFormer/raw/
```

Access SageMaker Studio through the AWS Console at SageMaker → Domains → fraud-detection-domain → Studio. This gives you access to pipelines, experiments, model registry, and endpoints.

## Running Your First Pipeline

Deploy the pipeline definition to SageMaker:

```bash
cd workflows

# Install dependencies
uv sync

# Deploy the pipeline (creates/updates the pipeline definition)
python -m src.workflows.sagemaker_fraud_detection_pipeline \
  --role-arn <sagemaker-execution-role-arn-from-cdk-output> \
  --default-bucket fraud-detection-<account>-sm \
  --region <your-region>
```

This registers the pipeline with SageMaker. You can now trigger runs through:

**SageMaker Studio UI**: Navigate to Pipelines → FraudDetectionPipeline → Create execution. Adjust hyperparameters if needed, then start the run. The Studio interface shows real-time progress, logs, and metrics.

**Python SDK**: From a notebook or script:

```python
from sagemaker.mlops.workflow.pipeline import Pipeline

pipeline = Pipeline(name="FraudDetectionPipeline")
execution = pipeline.start()
execution.wait()
```

**AWS CLI**:

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name FraudDetectionPipeline \
  --pipeline-parameters Name=ProcessingInstanceType,Value=ml.g4dn.2xlarge
```

Each execution runs through preprocessing, training, and model registration. SageMaker automatically provisions GPU instances, runs the containers, stores artifacts in S3, and cleans up resources when complete. Step caching ensures that unchanged steps don't re-run on subsequent executions.

## Understanding the Workflow

The `workflows/` directory contains the SageMaker pipeline definition:

```
workflows/
├── src/workflows/
│   └── sagemaker_fraud_detection_pipeline.py  # Pipeline definition
├── pyproject.toml                              # Dependencies
└── uv.lock                                     # Lock file
```

The pipeline uses S3 for artifact passing between steps. SageMaker automatically manages the data flow:

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
│  - RAPIDS/cuDF       │
│  - ml.g4dn.2xlarge   │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Processed Data (S3) │
│  - GNN graph         │
│  - XGB features      │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Training            │
│  (Training Job)      │
│  - GNN + XGBoost     │
│  - ml.g4dn.2xlarge   │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Model Artifacts     │
│  (S3)                │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Model Registry      │
│  (SageMaker)         │
│  - Versioning        │
│  - Approval workflow │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Triton Endpoint     │
│  (SageMaker)         │
│  - Real-time infer   │
│  - Auto-scaling      │
└──────────────────────┘
```

The trained model can be deployed to a SageMaker endpoint running NVIDIA Triton. The model accepts merchant features, cardholder features, and graph structure as inputs. It returns fraud probability plus Shapley values for explainability, which is critical for regulatory compliance and fraud analyst workflows.

## What's Next

The default hyperparameters work reasonably well, but you can tune them through pipeline parameters:

- `gnn_num_epochs`: More epochs generally improve accuracy but increase training time
- `xgb_num_boost_round`: XGBoost boosting rounds, higher values risk overfitting
- `gnn_hidden_channels`: Width of the GNN layers, larger captures more patterns

Adjust these when starting a pipeline execution:

```python
execution = pipeline.start(
    parameters={
        "GnnNumEpochs": 12,
        "XgbNumBoostRound": 768,
        "ProcessingInstanceType": "ml.g5.2xlarge"
    }
)
```

To use your own data, replace the `InputDataUrl` parameter with your S3 path. Ensure your CSV matches TabFormer's schema, or modify the preprocessing container to handle different data formats.

For production deployments, consider:
- Setting up SageMaker Model Monitor for drift detection
- Configuring SageMaker Pipelines schedules for automated retraining
- Using SageMaker Projects for MLOps templates with CI/CD
- Enabling VPC mode for SageMaker jobs if network isolation is required

## Cleanup

```bash
cd infra
npx cdk destroy --all
```

This removes the SageMaker Domain, IAM roles, ECR repositories, and S3 buckets. Note that S3 buckets are configured with auto-delete, but you may need to manually delete SageMaker endpoints, model packages, and pipeline executions before destroying the stack.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for our contribution guide.

## Security

See [CONTRIBUTING](CONTRIBUTING.md##security-issue-notifications) for reporting security issues.

## License

MIT-0 License. See [LICENSE](LICENSE).

## Authors

- Shardul Vaidya, AWS Partner Solutions Architect
- Zachary Jacobson, AWS Partner Solutions Architect
- Ragib Ahsan, AWS AI Acceleration Architect
- Artiom Troyanovsky, Senior Machine Learning Engineer at Thoughtworks
