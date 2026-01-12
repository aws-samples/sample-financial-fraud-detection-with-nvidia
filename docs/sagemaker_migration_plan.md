# SageMaker-Only Migration Plan
## v2_sagemaker Branch: Kubeflow → SageMaker Native

---

## Executive Summary

This plan migrates the v2 financial fraud detection pipeline from Kubeflow Pipelines on EKS to native AWS SageMaker services. The migration maintains complete feature parity with v2 while simplifying infrastructure and reducing operational complexity.

**User Requirements:**
- **Inference**: SageMaker Endpoints with custom Triton container (not EKS)
- **Orchestration**: SageMaker Pipelines SDK v2 (not Kubeflow)
- **Preprocessing**: Keep RAPIDS cuDF GPU acceleration (SageMaker Processing Jobs)
- **Infrastructure**: Simplified CDK (remove EKS, Kubeflow, Karpenter, ArgoCD)
- **Container Build**: CodeBuild automation copies NGC image to ECR (no modifications)

**Key Innovation**: CodeBuild automatically copies NGC training image to ECR for faster pulls, following the same pattern as the existing Triton image in v2. NGC container used as-is with no modifications. NGC credentials stored in AWS Secrets Manager.

**Timeline**: 15 days (3 weeks)
**Estimated Cost**: $567/month (comparable to v2 with optimizations)

---

## Architecture Transformation

### Current (v2): Kubeflow on EKS
```
┌─────────────────────────────────────────────────────────────┐
│                      EKS Cluster (Kubernetes 1.33)          │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Kubeflow Pipelines (5 components)                  │     │
│  │  ├─ download_raw_data_to_pvc (S3 → PVC)          │     │
│  │  ├─ run_cudf_preprocessing (RAPIDS/GPU)          │     │
│  │  ├─ prepare_training_config (JSON writer)        │     │
│  │  ├─ run_nvidia_training (GNN+XGB/GPU)            │     │
│  │  └─ upload_model_to_s3 (PVC → S3)                │     │
│  │                                                     │     │
│  │  Data Flow: S3 → PVC(100Gi) → PVC(10Gi) → S3     │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Triton Inference (ArgoCD GitOps)                   │     │
│  │  - Namespace: triton                               │     │
│  │  - GPU: g4dn nodes via Karpenter                  │     │
│  │  - Model: S3 auto-reload                          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  Infrastructure: EKS + Karpenter + GPU Operator + ArgoCD   │
└─────────────────────────────────────────────────────────────┘
```

### Target (v2_sagemaker): SageMaker Native
```
┌─────────────────────────────────────────────────────────────┐
│                   SageMaker Pipelines                        │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Pipeline Steps (S3-based artifact passing)         │     │
│  │  ├─ Step 1: PreprocessData (Processing Job, GPU)  │     │
│  │  ├─ Step 2: TrainModel (Training Job, GPU)        │     │
│  │  └─ Step 3: RegisterModel (Model Registry)        │     │
│  │                                                     │     │
│  │  Data Flow: S3 → S3 → S3 → Model Registry         │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │ SageMaker Endpoint (Triton)                        │     │
│  │  - Custom Triton container (PyTorch 2.7 + Captum) │     │
│  │  - GPU: ml.g4dn.2xlarge                           │     │
│  │  - Auto-scaling: 1-3 instances                    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  Infrastructure: VPC + S3 + IAM (no EKS)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Mapping

| Kubeflow Component | SageMaker Equivalent | Notes |
|-------------------|---------------------|-------|
| download_raw_data_to_pvc | **Skip** (direct S3 read) | SageMaker reads from S3 natively |
| run_cudf_preprocessing | **Processing Job** (GPU) | RAPIDS container, ml.g4dn.2xlarge |
| prepare_training_config | **Merged into Training** | Hyperparameters via SageMaker API |
| run_nvidia_training | **Training Job** (GPU) | Existing container adapted |
| upload_model_to_s3 | **Built-in** | SageMaker auto-uploads model.tar.gz |
| PVCs (100Gi + 10Gi) | **S3 artifact passing** | No persistent volumes needed |
| Triton on EKS | **SageMaker Endpoint** | Custom Triton container |

---

## Critical Files to Create/Modify

### Phase 1: Infrastructure (CDK)

#### 1. `/infra/lib/sagemaker-training-image-repo.ts` (NEW)
**Purpose**: CodeBuild project to copy NGC training image to ECR

**Pattern**: Mirrors existing `triton-image-repo.ts` approach

**Key Components**:
- ECR Repository: `nvidia-training-repo-sagemaker`
- CodeBuild Project:
  - Pulls NGC base image: `nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0`
  - Tags and pushes to ECR with `:latest` and `:${COMMIT_HASH}` tags
  - **No modifications** - uses NGC container as-is
- NGC Credentials: Stored in AWS Secrets Manager (`nvidia-ngc-api-key`)
- Lambda Trigger: Auto-starts build on CDK deploy
- Lifecycle Rules: Keep 5 most recent images

**Buildspec**:
```yaml
phases:
  pre_build:
    commands:
      # Login to NGC using Secrets Manager
      - export NGC_API_KEY=$(aws secretsmanager get-secret-value --secret-id nvidia-ngc-api-key --query SecretString --output text)
      - echo $NGC_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin
      # Login to ECR
      - aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO_URI
      # Get commit hash for tagging
      - git clone --depth 1 --branch $GIT_BRANCH $GIT_REPO_URL repo
      - export COMMIT_HASH=$(cd repo && git rev-parse --short HEAD)
  build:
    commands:
      # Pull NGC image
      - docker pull nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
      # Tag for ECR
      - docker tag nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0 $ECR_REPO_URI:latest
      - docker tag nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0 $ECR_REPO_URI:$COMMIT_HASH
  post_build:
    commands:
      - docker push $ECR_REPO_URI:latest
      - docker push $ECR_REPO_URI:$COMMIT_HASH
```

**No Dockerfile needed** - NGC container used as-is

**Training Invocation** (in SageMaker Pipeline):
```python
# SageMaker Estimator will use bash commands like Kubeflow:
estimator = Estimator(
    image_uri=f"{account}.dkr.ecr.{region}.amazonaws.com/nvidia-training-repo-sagemaker:latest",
    hyperparameters={...},  # GNN and XGBoost params
    ...
)

# Training script generates config.json from hyperparameters and calls:
# torchrun --standalone --nnodes=1 --nproc-per-node=1 /app/main.py --config /app/config.json
```

**Outputs**: ECR image URI for SageMaker Training Jobs

---

#### 2. `/infra/lib/sagemaker-infrastructure-stack.ts` (NEW)
**Purpose**: Foundation - Creates IAM roles and permissions for SageMaker

**Key Components**:
- `PipelineExecutionRole`: Orchestrates pipeline execution
- `SageMakerExecutionRole`: Runs Processing/Training Jobs and Endpoints
- S3 bucket permissions (data, model, model-registry)
- ECR pull permissions for custom containers
- Secrets Manager access for NGC credentials (used by CodeBuild)

**Outputs**: Role ARNs for pipeline and job execution

---

#### 3. `/infra/lib/sagemaker-triton-endpoint-stack.ts` (NEW)
**Purpose**: Deploys Triton on SageMaker Endpoint

**Key Components**:
- `CfnModel`: Points to custom Triton container + model.tar.gz in S3
- `CfnEndpointConfig`: ml.g4dn.2xlarge instance configuration
- `CfnEndpoint`: Real-time inference endpoint
- Auto-scaling: Target tracking on InvocationsPerInstance (1-3 instances)

**Outputs**: Endpoint name for inference testing

---

#### 4. `/infra/lib/nvidia-fraud-detection-blueprint.ts` (MODIFY)
**Changes**:
- **Remove**: EKS cluster, Karpenter, GPU Operator, ArgoCD, Kubeflow
- **Keep**: VPC, S3 buckets, ECR repositories
- **Add**: Reference to SageMaker training image repo, infrastructure, and endpoint stacks

### Phase 2: Preprocessing

#### 5. `/src/preprocess_TabFormer_sagemaker.py` (NEW)
**Purpose**: Adapts existing preprocessing for SageMaker Processing Job contract

**Changes from original**:
```python
# Input: /opt/ml/processing/input/data/card_transaction.v1.csv
# Output: /opt/ml/processing/output/xgb/{training,validation,test}.csv
#         /opt/ml/processing/output/gnn/nodes/...
#         /opt/ml/processing/output/gnn/edges/...

# Reuses existing logic from src/preprocess_TabFormer.py (966 lines)
# - cuDF GPU acceleration
# - Same feature engineering
# - Same graph structure generation
```

**Critical**: Must maintain exact output schema for training job compatibility

#### 6. `/infra/docker/rapids-preprocessing/Dockerfile` (NEW)
**Purpose**: SageMaker-compatible RAPIDS container for Processing Jobs

```dockerfile
FROM rapidsai/base:25.12-cuda13-py3.12

# Install SageMaker toolkit
RUN pip install sagemaker-training category_encoders scikit-learn

# Copy preprocessing script
COPY src/preprocess_TabFormer_sagemaker.py /opt/ml/code/preprocess.py

ENV SAGEMAKER_PROGRAM preprocess.py
```

### Phase 3: Training

**No training-specific files needed** - NGC container used as-is

**Training Approach**:
- SageMaker Training Job invokes the NGC container directly (same as Kubeflow)
- Config generation happens in the SageMaker Pipeline Python code
- Command: `torchrun --standalone --nnodes=1 --nproc-per-node=1 /app/main.py --config /app/config.json`
- Paths mapping handled via hyperparameters in config.json:
  - `data_dir`: `/opt/ml/input/data/gnn`
  - `output_dir`: `/opt/ml/model/python_backend_model_repository`

**Critical**: Ensure output structure matches Triton expectations

### Phase 4: Orchestration

#### 8. `/workflows/src/workflows/sagemaker_fraud_detection_pipeline.py` (NEW)
**Purpose**: Main SageMaker Pipeline definition - replaces Kubeflow pipeline

**Pipeline Structure**:
```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep

# Step 1: Preprocessing (RAPIDS on GPU)
preprocessing_step = ProcessingStep(
    name="PreprocessData",
    processor=rapids_processor,  # ml.g4dn.2xlarge
    inputs=[ProcessingInput(s3_raw_data)],
    outputs=[
        ProcessingOutput("xgb", s3_output_xgb),
        ProcessingOutput("gnn", s3_output_gnn)
    ]
)

# Step 2: Training (GNN+XGBoost on GPU)
training_step = TrainingStep(
    name="TrainModel",
    estimator=nvidia_estimator,  # ml.g4dn.2xlarge
    inputs={
        "gnn": preprocessing_step.properties...S3Uri,
        "xgb": preprocessing_step.properties...S3Uri
    }
)

# Step 3: Register Model
register_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        model_package_group_name="fraud-detection-models",
        approval_status="PendingManualApproval"
    )
)

pipeline = Pipeline(
    name="FraudDetectionPipeline",
    steps=[preprocessing_step, training_step, register_step]
)
```

**Pipeline Parameters**:
- Data bucket, model bucket, S3 paths
- Instance types (processing, training)
- Hyperparameters (GNN, XGBoost)

### Phase 5: Inference

#### 9. `/triton/serve` (NEW)
**Purpose**: SageMaker entrypoint for Triton inference

```bash
#!/bin/bash
# SageMaker extracts model.tar.gz to /opt/ml/model/
MODEL_REPO=/opt/ml/model/python_backend_model_repository

tritonserver \
  --model-repository=$MODEL_REPO \
  --http-port=8080 \
  --grpc-port=8081 \
  --log-verbose=1
```

#### 10. `/triton/Dockerfile` (MODIFY)
**Changes**:
```dockerfile
# Add to existing Triton image (already has PyTorch 2.7, XGBoost 3.0, Captum)
COPY triton/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

ENV SAGEMAKER_PROGRAM serve
```

---

## Data Flow

### Kubeflow (PVC-based)
```
S3 → PVC (100Gi) → [preprocess] → PVC → [train] → PVC (10Gi) → S3
     └─ Shared volume                └─ Shared volume
```

### SageMaker (S3-based)
```
S3 (raw CSV) → [Processing Job] → S3 (processed) → [Training Job] → S3 (model.tar.gz) → Endpoint
   4GB              ml.g4dn.2xlarge      ~10GB        ml.g4dn.2xlarge        200MB
   15 min                                              45 min
```

**Key Differences**:
- No shared volumes - each job reads/writes to S3
- Automatic archiving - SageMaker creates model.tar.gz
- Pipeline tracks artifact URIs automatically
- Caching available for processing outputs

---

## Implementation Timeline (15 Days)

### Week 1: Foundation + Preprocessing

**Day 1-2: Infrastructure**
- **Setup NGC Credentials**:
  ```bash
  # Store NGC API key in Secrets Manager
  aws secretsmanager create-secret \
    --name nvidia-ngc-api-key \
    --secret-string "your-ngc-api-key-here"
  ```
- Create `sagemaker-training-image-repo.ts` (CodeBuild for training image)
- Create `sagemaker-infrastructure-stack.ts` (IAM roles)
- Create `sagemaker-triton-endpoint-stack.ts` (endpoint infrastructure)
- Deploy CDK stacks:
  ```bash
  npx cdk deploy SageMakerTrainingImageRepoStack
  npx cdk deploy SageMakerInfraStack
  ```
- Validate:
  - CodeBuild automatically builds training image (check CloudWatch logs)
  - Training image pushed to ECR with `:latest` tag
  - IAM roles created successfully

**Day 3-4: Preprocessing Container**
- Create `preprocess_TabFormer_sagemaker.py`
- Build RAPIDS Dockerfile
- Test locally:
  ```bash
  docker run -v $(pwd)/data:/opt/ml/processing/input/data \
             -v $(pwd)/output:/opt/ml/processing/output \
             rapids-preprocessing:latest
  ```
- Push to ECR

**Day 5: Processing Job Test**
- Run standalone SageMaker Processing Job
- Compare outputs to Kubeflow (byte-by-byte validation)
- Benchmark: processing time, cost

### Week 2: Training + Pipeline

**Day 6-7: Training Job Testing**
- Verify training image is ready in ECR (built by CodeBuild on Day 1)
- Create test Training Job script
- Test with SageMaker local mode (optional)
- Run Training Job with sample preprocessed data
- Validate model.tar.gz structure
- Verify Triton-compatible model repository structure

**Day 8-9: SageMaker Pipeline**
- Define `sagemaker_fraud_detection_pipeline.py`
- Connect preprocessing → training → register model
- Execute pipeline: `pipeline.start()`
- Monitor in SageMaker Studio
- Validate S3 artifacts

### Week 3: Inference + Cleanup

**Day 10: Triton Endpoint**
- Modify Triton Dockerfile with serve script
- Create `sagemaker-triton-endpoint-stack.ts`
- Deploy endpoint via CDK
- Test inference with sample transaction
- Validate Shapley values

**Day 11-13: Integration Testing**
- End-to-end pipeline run
- Performance benchmarking (vs Kubeflow)
- Cost analysis
- Load testing endpoint (auto-scaling validation)

**Day 14: Documentation**
- Update README with SageMaker instructions
- Create migration guide
- Update architecture diagrams
- Create example notebook

**Day 15: EKS Cleanup**
- Remove Kubeflow/EKS-related files
- Update CDK main stack
- `npx cdk destroy NvidiaFraudDetectionBlueprint`
- Final validation

---

## Testing Strategy

### Unit Tests
```python
# tests/test_preprocessing_sagemaker.py
def test_preprocessing_paths():
    # Validate input/output directory structure

def test_preprocessing_output_schema():
    # Compare columns/dtypes to Kubeflow baseline

# tests/test_training_sagemaker.py
def test_hyperparameter_parsing():
    # Mock SM_HPS, validate config.json generation

def test_model_tar_structure():
    # Validate python_backend_model_repository/ structure
```

### Integration Tests
```python
# tests/integration/test_processing_job.py
def test_processing_job_e2e():
    processor.run(inputs=[...], outputs=[...])
    # Compare to golden dataset

# tests/integration/test_training_job.py
def test_training_job_e2e():
    estimator.fit(inputs={...})
    # Validate model metrics

# tests/integration/test_pipeline.py
def test_pipeline_execution():
    execution = pipeline.start()
    # Validate all steps succeeded
```

### Validation Tests (Kubeflow Parity)
```python
# tests/validation/test_parity.py
def test_preprocessing_output_parity():
    # Compare SageMaker output to Kubeflow output (schema + values)

def test_model_performance_parity():
    # Compare AUC, F1 score (tolerance: ±1%)

def test_inference_latency():
    # Compare p50, p99 latency
```

---

## Cost Analysis

### Current (v2 Kubeflow): $582/month
- EKS control plane: $73
- NAT Gateway: $32
- MNG (m5.xlarge): $439
- Karpenter GPU runs: $0.77/run × 10 = $7.70
- EBS volumes: $30

### Target (v2_sagemaker): $567/month (optimized)
- NAT Gateway: $32
- S3 storage: $1.50
- Processing + Training (spot): $0.36/run × 10 = $3.60
- SageMaker Endpoint (auto-scaling to min 1): $530

**Cost Optimizations**:
1. Use SageMaker Managed Spot Training (70% discount)
2. Endpoint auto-scaling (scale down during off-hours)
3. Remove always-on notebook instance (use Studio on-demand)
4. Pipeline caching (reuse preprocessing if data unchanged)

---

## Success Criteria

### Functional Requirements
- ✅ Preprocessing produces identical output to Kubeflow (schema, row counts)
- ✅ Training completes with equivalent metrics (AUC, F1 within 1%)
- ✅ Inference returns fraud probability + Shapley values
- ✅ Pipeline runs end-to-end without manual intervention
- ✅ Endpoint auto-scaling works under load

### Performance Requirements
- ✅ Preprocessing time: ≤ 20 minutes (baseline: 15min)
- ✅ Training time: ≤ 60 minutes (baseline: 45min)
- ✅ Inference latency: p99 ≤ 200ms (baseline: 150ms)
- ✅ Pipeline end-to-end: ≤ 90 minutes

### Operational Requirements
- ✅ CDK deployment < 15 minutes
- ✅ Pipeline triggered via notebook, CLI, or EventBridge
- ✅ CloudWatch dashboards for metrics
- ✅ Documentation updated
- ✅ Example notebook runs successfully

---

## Potential Challenges & Mitigations

### 1. RAPIDS Container Compatibility
**Issue**: `rapidsai/base` may not support SageMaker toolkit
**Mitigation**: Build custom image with `pip install sagemaker-training`, test with local mode first

### 2. cuDF GPU Memory
**Issue**: 24M rows may exceed 16GB RAM on ml.g4dn.2xlarge
**Mitigation**: Use ml.g4dn.4xlarge (64GB) if needed, optimize cuDF chunk size

### 3. Training Path Remapping
**Issue**: Existing `main.py` expects `/data/` paths
**Mitigation**: Thin wrapper `train.py` remaps paths, or use symlinks

### 4. Model Packaging for Triton
**Issue**: Output must match Triton's expected structure
**Mitigation**: Validate locally with Docker Triton before deploying endpoint

### 5. Shapley Computation
**Issue**: Python backend may have import issues with Captum
**Mitigation**: Test Python backend locally, ensure all deps in container

### 6. Cold Start Latency
**Issue**: First endpoint request may be slow
**Mitigation**: Keep min_instance_count=1, use endpoint data capture to warm cache

### 7. S3 I/O Performance
**Issue**: S3 slower than PVC
**Mitigation**: Use S3 VPC endpoint, enable Transfer Acceleration if needed

### 8. Cost of GPU Instances
**Issue**: ml.g4dn.2xlarge costs ~$1.50/hr
**Mitigation**: Use SageMaker Managed Spot (up to 90% discount)

---

## Files to Delete (EKS Cleanup)

### CDK Infrastructure (TypeScript)
```
infra/lib/kf-addon.ts                         # Kubeflow addon for EKS Blueprints
infra/lib/deploykf-app.ts                     # deployKF ArgoCD application
infra/lib/kf-policy.ts                        # Kubeflow IAM policies
infra/lib/nodepool-addon.ts                   # Karpenter node pool addon
infra/lib/argocd-values.ts                    # ArgoCD configuration values
```

### Kubernetes Manifests (YAML/Helm)
```
infra/manifests/argocd/                       # ArgoCD bootstrap manifests (entire directory)
  ├── Chart.yaml
  ├── templates/
  │   ├── triton-server.yaml
  │   └── notebook.yaml
  └── values.yaml

infra/manifests/helm/triton/                  # Triton Helm chart for EKS (entire directory)
  ├── Chart.yaml
  ├── templates/
  │   ├── deployment.yaml
  │   ├── ingress.yaml
  │   └── service.yaml
  └── values.yaml

infra/manifests/helm/notebook/                # Kubeflow Notebook Helm chart (entire directory)
  ├── Chart.yaml
  ├── templates/
  │   ├── kubeflow-notebook-server.yaml
  │   ├── external-secret.yaml
  │   └── secret-store.yaml
  └── values.yaml
```

### Kubeflow Pipelines (Python)
```
workflows/src/workflows/cudf_e2e_pipeline.py  # Kubeflow pipeline definition (REPLACE with SageMaker version)
workflows/src/workflows/components/           # Kubeflow components directory (entire directory)
  ├── __init__.py
  └── preprocess_tabformer.py                 # Component-specific preprocessing
```

### Notebooks
```
notebooks/kubeflow-fraud-detection.ipynb      # Kubeflow notebook (REPLACE with SageMaker version)
```

### Documentation (Optional - Update Instead)
```
docs/ngc-setup.md                             # NGC setup for EKS (UPDATE for SageMaker or DELETE)
docs/roadmap/                                 # Kubeflow migration docs (ARCHIVE or DELETE)
  ├── 01-project-overview.md
  ├── 02-kubeflow-installation.md
  ├── 03-pipeline-components.md
  └── 04-migration-phases.md
```

**Total**: ~35 files removed (5 CDK stacks + 3 manifest directories + 2 workflow files + 1 notebook + optional docs)

**Note**: The following files will be **MODIFIED**, not deleted:
- `infra/lib/nvidia-fraud-detection-blueprint.ts` - Remove EKS, add SageMaker stacks
- `infra/bin/nvidia-fraud-detection-blueprint.ts` - Update stack instantiation

---

## Verification Steps

After implementation, validate each component:

### 1. Preprocessing Validation
```bash
# Run processing job
python scripts/run_processing_job.py

# Download outputs
aws s3 sync s3://.../preprocessing/output/ ./output/

# Compare to Kubeflow baseline
python tests/validation/compare_preprocessing.py --kubeflow ./baseline/ --sagemaker ./output/
```

### 2. Training Validation
```bash
# Run training job
python scripts/run_training_job.py

# Download model
aws s3 cp s3://.../model.tar.gz ./
tar -xzf model.tar.gz

# Validate structure
python tests/validation/validate_model_structure.py --model ./python_backend_model_repository/
```

### 3. Pipeline Validation
```bash
# Submit pipeline
python workflows/src/workflows/sagemaker_fraud_detection_pipeline.py

# Monitor execution
python scripts/monitor_pipeline.py --pipeline-name FraudDetectionPipeline
```

### 4. Endpoint Validation
```bash
# Test inference
python scripts/test_endpoint.py --endpoint fraud-detection-triton

# Load test
locust -f tests/load/locustfile.py --host https://{endpoint}
```

---

## Rollback Plan

If migration fails or performance inadequate:

1. **Revert to v2** (1 day):
   - Checkout v2 branch: `git checkout v2`
   - Redeploy CDK: `npx cdk deploy --all`
   - Verify Kubeflow UI accessible

2. **Partial Rollback** (hybrid):
   - Keep SageMaker Endpoint for inference
   - Revert to Kubeflow for orchestration
   - Models in S3 compatible with both

3. **Data Preservation**:
   - S3 buckets unchanged
   - Model artifacts compatible with both environments

---

## Next Steps (Day 1 Actions)

### Prerequisites
1. **Store NGC API Key in Secrets Manager**:
   ```bash
   aws secretsmanager create-secret \
     --name nvidia-ngc-api-key \
     --secret-string "your-ngc-api-key-here" \
     --region us-west-2
   ```

### Infrastructure Setup
2. **Create CDK Stacks**:
   - `infra/lib/sagemaker-training-image-repo.ts` (CodeBuild + ECR - copies NGC image)
   - `infra/lib/sagemaker-infrastructure-stack.ts` (IAM roles)
   - `infra/lib/sagemaker-triton-endpoint-stack.ts` (endpoint config)

   **Note**: No Dockerfile or training wrapper needed - NGC container used as-is

3. **Update Main Stack**:
   - `infra/bin/nvidia-fraud-detection-blueprint.ts` to instantiate all SageMaker stacks

4. **Deploy CDK**:
   ```bash
   cd infra
   npx cdk deploy SageMakerTrainingImageRepoStack --require-approval never
   # Wait for CodeBuild to complete (~15 minutes)
   npx cdk deploy SageMakerInfraStack --require-approval never
   ```

5. **Validate Deployment**:
   - Check CodeBuild logs: Training image build succeeded
   - Check ECR: `nvidia-training-repo-sagemaker:latest` exists
   - Check IAM: SageMaker execution roles created
   - Test Secrets Manager access from CodeBuild role

### Next: Preprocessing (Day 3)
6. Create `src/preprocess_TabFormer_sagemaker.py`
7. Build RAPIDS preprocessing container
8. Test SageMaker Processing Job

Ready to begin implementation!
