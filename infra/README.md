# NVIDIA Fraud Detection — CDK Infrastructure

This directory contains the AWS CDK infrastructure code for deploying the fraud detection pipeline on SageMaker.

## CDK Stacks

Stacks are defined in `bin/nvidia-fraud-detection-blueprint.ts` and deploy in dependency order:

| Stack | Purpose |
|---|---|
| `SageMakerTrainingImageRepoStack` | ECR repo + CodeBuild project that copies the NGC training image |
| `SageMakerPreprocessingImageRepoStack` | ECR repo + CodeBuild project that builds the RAPIDS preprocessing image |
| `TritonImageRepoStack` | ECR repo + CodeBuild project that builds the Triton inference image |
| `NvidiaFraudDetectionBlueprint` | S3 buckets for data, models, and model registry |
| `SageMakerInfraStack` | SageMaker execution role with S3, ECR, and Secrets Manager permissions |
| `SageMakerDomainStack` | SageMaker Domain for Studio access to pipelines |
| `SageMakerTritonEndpointStack` | SageMaker Model, Endpoint Config, and Endpoint (requires trained model) |

## Stack source files

```
lib/
├── nvidia-fraud-detection-blueprint.ts   # S3 buckets
├── sagemaker-infrastructure-stack.ts     # IAM roles
├── sagemaker-domain-stack.ts             # SageMaker Studio domain
├── sagemaker-training-image-repo.ts      # Training image ECR + CodeBuild
├── sagemaker-preprocessing-image-repo.ts # Preprocessing image ECR + CodeBuild
├── triton-image-repo.ts                  # Triton image ECR + CodeBuild
└── sagemaker-triton-endpoint-stack.ts    # SageMaker endpoint
```

## Prerequisites

- Node.js 20+
- AWS CDK CLI (`npm install -g aws-cdk`)
- AWS CLI v2 configured with appropriate credentials
- NVIDIA NGC API key stored in Secrets Manager as `nvidia-ngc-api-key`

## Setup

```bash
npm install
npx cdk bootstrap aws://<ACCOUNT>/<REGION> --profile <your-aws-profile>
```

## Deploy

```bash
# Deploy everything except endpoint (first run — no model yet)
npx cdk deploy --all \
  --exclude SageMakerTritonEndpointStack \
  --profile <your-aws-profile> \
  --require-approval never

# After pipeline has trained a model:
npx cdk deploy SageMakerTritonEndpointStack --profile <your-aws-profile>
```

## Cleanup

```bash
npx cdk destroy --all --profile <your-aws-profile>
```

S3 buckets are configured with `autoDeleteObjects: true`. You may need to manually delete SageMaker model packages from the registry before destroying stacks.
