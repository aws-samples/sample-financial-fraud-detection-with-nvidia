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

### Prerequisites

**AWS Account Requirements:**
- Permissions for SageMaker, ECR, S3, IAM, CloudFormation, CodeBuild, Secrets Manager
- AWS CLI configured with appropriate profile

**Local Requirements:**
- Docker (for building containers)
- Node.js 20+ and npm
- Python 3.12+
- AWS CLI v2
- `uv` (Python package manager): `pip install uv`

**NVIDIA NGC API Key:**
The infrastructure pulls base images from NVIDIA GPU Cloud. You need an NGC API key:

1. Create account at https://ngc.nvidia.com/
2. Generate API key: NGC → Setup → Generate API Key
3. Store in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name nvidia-ngc-api-key \
  --secret-string "<your-ngc-api-key>" \
  --profile <your-aws-profile>
```

### Step 1: Deploy Infrastructure

```bash
cd infra
npm install

# Bootstrap CDK (first time only)
npx cdk bootstrap aws://<ACCOUNT>/<REGION> --profile <your-aws-profile>

# Set your environment
export CDK_DEFAULT_ACCOUNT=<your-account>
export CDK_DEFAULT_REGION=<your-region>

# Deploy all stacks
npx cdk deploy SageMakerTrainingImageRepoStack SageMakerPreprocessingImageRepoStack TritonImageRepoStack NvidiaFraudDetectionBlueprint SageMakerInfraStack SageMakerDomainStack --profile <your-aws-profile> --require-approval never
```

This creates:
- S3 buckets for data and models
- ECR repositories for custom containers (RAPIDS preprocessing, training, Triton)
- SageMaker execution roles with appropriate permissions
- SageMaker Domain for Studio access
- CodeBuild projects that automatically build container images

**Important:** Save the outputs from the deployment:
- `SageMakerExecutionRoleArn`
- `DataBucketName`
- `DomainId`

### Step 2: Wait for Container Images to Build

CodeBuild automatically starts building Docker images after deployment. Monitor progress:

```bash
# Check CodeBuild status
aws codebuild list-builds-for-project \
  --project-name sagemaker-training-image-copy \
  --profile <your-aws-profile>

# Verify images in ECR (wait until all 3 repos have images)
aws ecr describe-images \
  --repository-name rapids-preprocessing \
  --profile <your-aws-profile>

aws ecr describe-images \
  --repository-name nvidia-training-repo-sagemaker \
  --profile <your-aws-profile>

aws ecr describe-images \
  --repository-name triton-inference-server \
  --profile <your-aws-profile>
```

Image builds take 10-20 minutes. **Do not proceed until all images are built.**

### Step 3: Upload Training Data

```bash
# Download TabFormer dataset (or use your own data)
# Upload to S3
aws s3 cp card_transaction.v1.csv \
  s3://fraud-detection-<account>-sm/data/TabFormer/raw/ \
  --profile <your-aws-profile>
```

### Step 4: Access SageMaker Studio

Open SageMaker Studio to view and execute pipelines:

1. Go to AWS Console → SageMaker → Domains
2. Click on `fraud-detection-domain`
3. Click "Launch" → "Studio"
4. Wait for Studio to load (first launch takes 2-3 minutes)

## Running Your First Pipeline

### Step 5: Register the Pipeline

Deploy the pipeline definition to SageMaker:

```bash
cd workflows

# Install dependencies
uv sync

# Register the pipeline (creates/updates the pipeline definition)
uv run python src/workflows/sagemaker_fraud_detection_pipeline.py \
  --role-arn <sagemaker-execution-role-arn-from-cdk-output> \
  --default-bucket fraud-detection-<account>-sm \
  --region <your-region> \
  --profile <your-aws-profile>
```

This registers the pipeline with SageMaker. You'll see: `Pipeline FraudDetectionPipeline created/updated.`

### Step 6: Execute the Pipeline

**Option 1: SageMaker Studio UI (Recommended)**

1. In SageMaker Studio, click **Pipelines** in the left sidebar
2. Find and click **FraudDetectionPipeline**
3. Click **Create execution**
4. Review parameters (or use defaults)
5. Click **Start**
6. Monitor execution progress in real-time with logs and metrics

**Option 2: AWS CLI**

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name FraudDetectionPipeline \
  --profile <your-aws-profile>
```

**Option 3: Python SDK (SageMaker 3.x)**

```python
from sagemaker.mlops.workflow.pipeline import Pipeline

pipeline = Pipeline(name="FraudDetectionPipeline")
execution = pipeline.start()
execution.wait()
```

### Monitoring Execution

Each execution runs through three steps:
1. **PreprocessData** - RAPIDS preprocessing (~10-15 minutes)
2. **TrainModel** - GNN + XGBoost training (~5 minutes)
3. **RegisterModel** - Model registration to Model Registry (~2 minutes)

SageMaker automatically provisions GPU instances, runs containers, stores artifacts in S3, and cleans up resources when complete. Step caching ensures unchanged steps don't re-run on subsequent executions.

View execution details in Studio:
- **Graph view**: Visual pipeline DAG with step status
- **Execution list**: All pipeline runs with timestamps and status
- **Step details**: Logs, metrics, input/output artifacts for each step

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
│  - ml.g4dn.4xlarge   │
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

## Troubleshooting


### CodeBuild Fails to Pull NGC Images

**Error:** `Error response from daemon: Get https://nvcr.io/v2/: unauthorized`

**Solution:** Verify NGC API key is correctly stored in Secrets Manager:

```bash
aws secretsmanager get-secret-value \
  --secret-id nvidia-ngc-api-key \
  --profile <your-aws-profile>
```

### Pipeline Step Shows "Image Not Found"

**Error:** `CannotPullContainerError: Error response from daemon: manifest for <image> not found`

**Solution:** Wait for CodeBuild to finish building images. Check build status:

```bash
aws codebuild list-builds-for-project \
  --project-name sagemaker-training-image-copy \
  --profile <your-aws-profile>
```

### SageMaker Studio Won't Load

**Solution:** First launch takes 2-3 minutes. If it times out, refresh the page. Check Domain status:

```bash
aws sagemaker describe-domain \
  --domain-id <domain-id-from-cdk-output> \
  --profile <your-aws-profile>
```

### Input Data Not Found

**Error:** `ClientError: Could not find s3 object`

**Solution:** Verify data is uploaded to the correct S3 path:

```bash
aws s3 ls s3://fraud-detection-<account>-sm/data/TabFormer/raw/ \
  --profile <your-aws-profile>
```

## What's Next

### Hyperparameter Tuning

The default hyperparameters work reasonably well, but you can tune them through pipeline parameters:

- `GnnNumEpochs`: More epochs generally improve accuracy but increase training time (default: 8)
- `XgbNumBoostRound`: XGBoost boosting rounds, higher values risk overfitting (default: 512)
- `GnnHiddenChannels`: Width of the GNN layers, larger captures more patterns (default: 32)
- `GnnNHops`: Number of graph neighborhood hops (default: 2)

Adjust these when starting a pipeline execution in Studio or via CLI:

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name FraudDetectionPipeline \
  --pipeline-parameters \
    Name=GnnNumEpochs,Value=12 \
    Name=XgbNumBoostRound,Value=768 \
  --profile <your-aws-profile>
```

### Using Your Own Data

Replace the `InputDataUrl` parameter with your S3 path. Ensure your CSV matches TabFormer's schema:
- Columns: `User`, `Card`, `Timestamp`, `Amount`, `Use Chip`, `Merchant Name`, `Merchant City`, `Merchant State`, `Zip`, `MCC`, `Errors?`, `Is Fraud?`

Or modify `src/preprocess_TabFormer_sagemaker.py` to handle different data formats.

### Production Considerations

For production deployments, consider:
- **Model Monitoring**: Set up SageMaker Model Monitor for drift detection
- **Automated Retraining**: Configure SageMaker Pipelines schedules for periodic retraining
- **CI/CD**: Use SageMaker Projects for MLOps templates with automated testing
- **Network Isolation**: Enable VPC mode for SageMaker jobs if required by security policies
- **Cost Optimization**: Use Spot instances for training jobs to reduce costs by up to 70%

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
