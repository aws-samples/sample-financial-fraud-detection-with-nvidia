# Changelog

All notable changes to the NVIDIA Financial Fraud Detection Blueprint are documented here.

## [3.0.0] - January 2025 (Latest)

Simplified architecture migrating back to SageMaker Pipelines for fully managed, serverless ML orchestration.

### Major Changes

- **SageMaker Pipelines**: Native SageMaker pipeline orchestration replaces Kubeflow/EKS for simpler operations
- **Serverless Architecture**: Eliminated EKS cluster management, Karpenter, and Kubernetes overhead
- **Managed Infrastructure**: SageMaker handles job scheduling, resource provisioning, and cleanup automatically
- **GPU Processing & Training**: Custom RAPIDS and training containers run on SageMaker Processing/Training Jobs
- **Model Registry Integration**: Built-in model versioning and approval workflows via SageMaker Model Registry
- **Simplified Deployment**: ~15 minute infrastructure setup vs 30+ minutes for EKS-based approach

### Infrastructure Changes

- Removed EKS cluster and all Kubernetes components (Karpenter, ArgoCD, Kubeflow, deployKF)
- Removed VPC infrastructure (SageMaker jobs run in AWS-managed VPCs by default)
- Added `SageMakerInfraStack` for IAM roles and permissions
- Added `SageMakerDomainStack` for Studio access
- Added `SageMakerTritonEndpointStack` for inference deployment
- Simplified to 3 ECR repositories: preprocessing, training, and Triton inference
- Bucket naming updated with `-sm` suffix to distinguish from Kubeflow deployment

### Pipeline Changes

- Consolidated to 3 steps: Preprocessing → Training → Model Registration
- Direct S3 artifact passing managed by SageMaker
- Step caching prevents re-running unchanged pipeline stages
- Pipeline parameters for hyperparameter tuning and instance type selection
- Python SDK deployment via `sagemaker_fraud_detection_pipeline.py`

### Removed

- EKS cluster, Karpenter, GPU Operator, ALB Controller
- Kubeflow Pipelines, deployKF, ArgoCD
- Kubernetes-specific configurations (PVCs, IRSA, node selectors)
- Kubeflow notebook servers (replaced by SageMaker Studio)
- Custom download and upload components (SageMaker handles data movement)

### Benefits

- **Lower operational overhead**: No cluster management or Kubernetes expertise required
- **Cost optimization**: Pay only for job execution time, no idle cluster resources
- **Faster deployment**: Infrastructure deploys in ~15 minutes vs 30+ minutes
- **Native AWS integration**: Seamless integration with IAM, CloudWatch, and AWS console
- **Enterprise ready**: Built-in security, compliance, and governance features

## [2.0.0] - January 2025

Complete migration from SageMaker to Kubeflow Pipelines on EKS.

### New Features

- **Kubeflow Pipelines Integration**: Full ML pipeline orchestration via KFP v2 SDK with GPU-accelerated preprocessing and training stages
- **Interactive Notebook**: `notebooks/kubeflow-fraud-detection.ipynb` defines and submits pipelines directly from Kubeflow notebook servers
- **cuDF Preprocessing**: RAPIDS-based GPU preprocessing replaces pandas, reducing preprocessing time from hours to minutes on 24M transactions
- **Custom Triton Image**: ECR-hosted Triton image with PyTorch, torch_geometric, XGBoost, and Captum for Shapley explainability
- **CodeBuild Auto-Build**: Triton image automatically rebuilds on CDK deploy via Lambda-triggered CodeBuild project
- **PVC Artifact Passing**: Pipeline stages share data through persistent volumes instead of S3 round-trips
- **deployKF**: Kubeflow installation via deployKF Helm charts managed by ArgoCD

### Infrastructure Changes

- Added `triton-image-repo.ts` CDK stack for ECR repository and CodeBuild
- Added Kubeflow IAM roles with IRSA for S3 and ECR access
- Added `team-1` namespace with notebook server and pipeline runner service accounts
- Added nginx proxy for stable local port-forwarding during development
- Updated Karpenter node pools for GPU workloads (g4dn, g5, g6e support)
- Migrated ArgoCD to manage both Kubeflow and Triton deployments

### Removed

- SageMaker Training Jobs - replaced by KFP components on EKS
- SageMaker Notebook Instances - replaced by Kubeflow Notebook Servers
- `sagemaker-training-role.ts` and `sagemaker-notebook-role.ts` CDK stacks
- `sagemaker_config.json` configuration file

### Documentation

- Added `docs/roadmap/` with migration planning documents
- Added `docs/kubeflow-pipeline-journey.md` detailing the implementation journey
- Rewrote README as guided walkthrough with architecture explanations
- Added dashboard screenshots in `docs/img/`

## [1.0.0] - December 2024

Initial release with SageMaker-based training workflow.

### Features

- EKS cluster via CDK with EKS Blueprints
- Karpenter v1 for GPU node autoscaling (g4dn instances)
- NVIDIA GPU Operator v25.3.2
- Triton Inference Server deployment via ArgoCD and Helm
- SageMaker Training Jobs for GNN+XGBoost model training
- SageMaker Notebook Instance for development
- S3-based model registry with Triton auto-reload
- ALB Controller for load balancer provisioning

### Infrastructure

- VPC with 3 AZs and NAT gateway
- EKS 1.32 cluster with managed node groups
- ArgoCD for GitOps deployments
- Secrets Store CSI Driver integration
- IAM roles for SageMaker training and notebook access
