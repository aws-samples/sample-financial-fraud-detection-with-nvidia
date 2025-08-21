# Infrastructure Overview

This directory contains the AWS CDK infrastructure code for deploying the Nvidia Fraud Detection Blueprint. The infrastructure uses AWS CDK with TypeScript and the EKS Blueprints framework to create a production-ready Kubernetes cluster optimized for GPU workloads.

## Architecture Components

- **Amazon EKS Cluster** - Managed Kubernetes cluster using EKS Blueprints
- **GPU Node Pool** - Dedicated node group for GPU workloads using g4dn instances
- **AWS Load Balancer Controller** - For ingress management
- **NVIDIA GPU Operator** - For GPU support in Kubernetes
- **AWS Secrets Store** - For managing sensitive information
- **ArgoCD** - For GitOps-based deployment management

## Key Features

- Automated GPU node pool management with Karpenter
- GPU-optimized instance selection (g4dn.xlarge, g4dn.2xlarge)
- Automatic node tainting for GPU workload isolation
- Integration with AWS Secrets Store
- GitOps-based deployment workflow using ArgoCD
- Built-in cost optimization through node expiration policies

## Prerequisites

- Node.js 14.x or later
- AWS CDK CLI installed (`npm install -g aws-cdk`)
- AWS CLI configured with appropriate credentials
- Docker (for CDK asset bundling)
- TypeScript knowledge for infrastructure modifications

## Setup Instructions

1. Install dependencies:
```bash
npm install
```

2. Bootstrap CDK (if not already done):
```bash
cdk bootstrap aws://<ACCOUNT>/<REGION> --qualifier nvidia
```

3. Configure environment variables:
```bash
export CDK_DEFAULT_ACCOUNT=<your-account>
export CDK_DEFAULT_REGION=<your-region>
```

4. Deploy the stack:
```bash
npm run build
cdk deploy
```

## Available CDK Stacks

- `NvidiaFraudDetectionBlueprint` - Main infrastructure stack that creates:
  - VPC and networking components
  - EKS cluster with GPU support
  - Required IAM roles and policies
  - ArgoCD setup for GitOps
  - GPU node pool configuration

## Useful Commands

* `npm run build`   compile TypeScript to JavaScript
* `npm run watch`   watch for changes and compile
* `npm run test`    perform the jest unit tests
* `npx cdk deploy`  deploy this stack to your default AWS account/region
* `npx cdk diff`    compare deployed stack with current state
* `npx cdk synth`   emits the synthesized CloudFormation template

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