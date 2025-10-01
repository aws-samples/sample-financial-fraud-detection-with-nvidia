# Getting Started - AWS Edition

## Overview

This NVIDIA Financial Fraud Detection AI Blueprint along with the AWS Deployment guide provides a reference example to deploy an end-to-end blueprint using Graph Neural Networks (GNNs) for higher accuracy and reduced false positives. We leverage Amazon SageMaker for training the fraud detection model and Amazon EKS (Elastic Kubernetes Service) with NVIDIA Triton Inference Server for hosting and serving the trained model.

This is the general architecture diagram for how we host the blueprint on AWS.

![Architecture diagram showing the end-to-end AWS deployment workflow: A
Sagemaker development environment connects to S3 for data storage and
preprocessing. The workflow shows data moving from Sagemaker to S3, then to a
Sagemaker Training job. The trained model is stored in S3 and loaded into an
EKS cluster running Nvidia Triton for model inference. The diagram illustrates
the complete pipeline from development to production deployment.](./docs/arch-diagram.png)

1. We will host our development environment inside SageMaker as it permits us to
   offload long lived compute to the cloud without having to keep our system up.
2. The notebook then does the data preprocessing for our model, splits it into
   training and testing sets, and uploads the data sets into S3.
3. The notebook kicks off a Sagemaker AI training job with pointers to our data
   and configuration, this then fully replicates our data, and executes the
   training job.
4. The trained model is output into S3, and that kicks off the model reload
   process in our inference instance.
5. The model is taken from S3 and loaded into the EKS cluster so that it's
   ready to go into production.
6. For inference, we deploy Nvidia Triton on an EKS cluster with GPU-enabled nodes
   (g4dn instances) that reads the model from S3 and serves it through a load-balanced
   endpoint.

## Prerequisites

To successfully deploy this blueprint, you will need:

1. **AWS Account** - with appropriate permissions to create SageMaker, EKS, EC2,
and ECR resources

2. **Nvidia NGC API Key** - required to access Nvidia's container registry and models

- Please refer to the [Nvidia NGC
    documentation](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key)
    for instructions on obtaining your API key

3. **Local machine** - with Docker installed and approximately 30GB of available storage space

4. **Git** - for cloning the repository

## Setup Instructions

### Set up SageMaker Studio

1. Navigate to the AWS SageMaker console
2. Create a new SageMaker domain if you don't have one already
3. Create a new user profile or use an existing one
4. Launch SageMaker Studio
5. Create a new notebook instance with at least `ml.g4dn.4xlarge` instance type
   and 50GB of storage

### Deploy Infrastructure with AWS CDK

From your local machine with Node.js and AWS CDK installed:

1. Clone this repository

```sh
git clone https://github.com/aws-samples/financial-fraud-detection-with-nvidia
cd financial-fraud-detection-with-nvidia/infra
```

2. Configure your AWS credentials and install dependencies:

```bash
aws configure
npm install
```

3. Bootstrap CDK (First time only):

```bash
npx cdk bootstrap aws://<ACCOUNT>/<REGION>
```

4. Configure environment variables:

```bash
# AWS Configuration
export CDK_DEFAULT_ACCOUNT=<your-account>
export CDK_DEFAULT_REGION=<your-region>

# Optional: Training output bucket configuration (defaults to "ml-on-containers")
export MODEL_BUCKET_NAME=your-custom-bucket-name
```

5. Deploy the stack:

```bash
npx cdk deploy --all

# You can also pass in your bucket name as a command line argument during deployment
npx cdk deploy --all --context modelBucketName=your-custom-bucket-name
```

This will:

- Create a new VPC with public and private subnets
- Deploy an EKS cluster with GPU-enabled node groups
- Install the NVIDIA GPU Operator
- Configure ArgoCD for GitOps-based deployments
- Set up AWS Load Balancer Controller
- Configure IAM roles and security groups

You can monitor the deployment progress in the AWS Console under CloudFormation.

### Upload the NVIDIA Training Image to ECR

Login to the NVIDIA Docker Registry and pull the image:

```bash
export NGC_API_KEY=<your-api-key>
docker login nvcr.io --username '$oauthtoken' --password $NGC_API_KEY

docker pull nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
```

Get the ECR URI from the cloudformation stack:

```bash
export NVIDIA_ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name NvidiaFraudDetectionTrainingImageRepo \
  --query "Stacks[0].Outputs[?OutputKey=='TrainingImageRepoUri'].OutputValue" \
  --output text)
```

Login to ECR:

```bash

export ECR_REPO=$(echo "${NVIDIA_ECR_URI%%/*}")
aws ecr get-login-password --region region | docker login --username AWS --password-stdin $ECR_REPO
```

Tag the NVIDIA Image and push it to ECR:

```bash

docker tag nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0 $NVIDIA_ECR_URI:latest
docker push $NVIDIA_ECR_URI:latest
```

### Set up Development Environment in SageMaker Studio

From within your SageMaker Studio environment:

1. Clone the repository

```sh
git clone https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia.git
```

2. Set up the required conda environment from a terminal within the JupyterLab environment

```sh
conda env create -f ./sample-financial-fraud-detection-with-nvidia/conda/notebook_env.yaml
conda install -y ipykernel -n fraud_blueprint_env
conda init
conda activate fraud_blueprint_env
python -m ipykernel install --user --name fraud_blueprint_env --display-name "user_env:(fraud_blueprint_env)"
```

3. Open the notebook and follow the instructions
   - Navigate to the `notebooks` directory and open the main notebook
   - The notebook will guide you through the process of data preparation, model
     training, and deployment

### Running the Solution

Follow the step-by-step instructions in the notebook to:

1. Preprocess the financial transaction data
2. Train the fraud detection model using the Nvidia container on SageMaker
3. Deploy the trained model to Triton Inference Server on EKS
4. Configure the ArgoCD deployment for automated model updates

For detailed troubleshooting and additional configuration options, refer to the
documentation in the `docs` directory.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

## Authors

- Shardul Vaidya, AWS NGDE Architect
- Ragib Ahsan, AWS AI Acceleration Architect
- Zachary Jacobson, AWS Partner Solutions Architect
