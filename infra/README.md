# NVIDIA Fruad Detection CDK Blueprint Infrastructure Overview

This directory contains the AWS CDK infrastructure code for deploying the Nvidia Fraud Detection Blueprint. The infrastructure uses AWS CDK with TypeScript and the EKS Blueprints framework to create a production-ready Kubernetes cluster optimized for GPU workloads.

## Architecture Components

### EKS Stack

- **Amazon EKS Cluster** - Managed Kubernetes cluster using EKS Blueprints
- **VPC and Networking** - Custom VPC with public/private subnets across 3 AZs
- **GPU Node Pool** - Dedicated node group for GPU workloads using g4dn instances
- **AWS Load Balancer Controller** - For ingress management
- **NVIDIA GPU Operator** - For GPU support in Kubernetes
- **AWS Secrets Store** - For managing sensitive information
- **ArgoCD** - For GitOps-based deployment management

### Model Extraction Stack

- **S3 Bucket** - Model registry for trained fraud detection models
- **Lambda Function** - Automated model extraction and deployment pipeline

## Key Features

- Automated GPU node pool management with Karpenter
- GPU-optimized instance selection (g4dn.xlarge, g4dn.2xlarge)
- Automatic node tainting for GPU workload isolation
- Integration with AWS Secrets Store
- GitOps-based deployment workflow using ArgoCD
- Built-in cost optimization through node expiration policies

## Prerequisites

- Node.js 20.x or later
- AWS CDK CLI installed (`npm install -g aws-cdk`)
- AWS CLI configured with appropriate credentials
- AWS Account with permissions to create EKS, EC2, S3, Lambda, and IAM resources

## Setup Instructions

1. Install dependencies:

```bash
npm install
```

2. Bootstrap CDK (First time only):

```bash
npx cdk bootstrap aws://<ACCOUNT>/<REGION>
```

3. Configure environment variables:

```bash
# AWS Configuration
export CDK_DEFAULT_ACCOUNT=<your-account>
export CDK_DEFAULT_REGION=<your-region>

# Optional: Training output bucket configuration (defaults to "ml-on-containers")
export MODEL_BUCKET_NAME=your-custom-bucket-name
```

4. Deploy the stack:

```bash
npx cdk deploy --all

# You can also pass in your bucket name as a command line argument during deployment
npx cdk deploy --all --context modelBucketName=your-custom-bucket-name
```

## Verify Deployment

Once the stack deploys:

1. **Check EKS Cluster**

```bash
aws eks update-kubeconfig --region <aws-region> --name ClusterBlueprint
kubectl get nodes
```

2. Verify GPU Nodes

```bash
kubectl get nodes -l node-type=gpu
kubectl describe node <gpu-node-name>
```

3. Check Triton Server Deployment

```bash
kubectl get deployment -n triton
```

## Cleanup

To destroy the infrastructure:

```bash
# Delete all stacks
cdk destroy --all
```

## Customization

The infrastructure can be customized by modifying:

- Node pool specifications in `nvidia-fraud-detection-blueprint.ts`
- VPC configuration
- EKS version and add-ons
- ArgoCD configuration and Git repository settings

## Security Considerations

- The stack uses AWS Secrets Store for sensitive information
- GPU nodes are isolated using Kubernetes taints
- IAM roles follow least privilege principle
- VPC is configured with private subnets for workload isolation

## Monitoring and Maintenance

- GPU utilization can be monitored through CloudWatch metrics
- Node pool auto-scaling is handled by Karpenter
- Infrastructure updates can be deployed using standard CDK workflows
- ArgoCD provides GitOps-based application deployment monitoring

## Troubleshooting

1. Check CloudWatch logs for EKS cluster issues
2. Verify GPU operator status in the EKS cluster
3. Ensure ArgoCD has correct repository credentials
4. Monitor Karpenter logs for node provisioning issues

For more detailed documentation, refer to the `docs` directory in the root of this repository.
