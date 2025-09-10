# Getting Started - AWS Edition

## Overview
This Nvidia AI Blueprint along with the AWS Deployment guide
provides a reference example to deploy an End to End financial fraud detection
blueprint using Graph Neural Networks. We will leverage Nvidia's Triton inference
server to host our models while we use Sagemaker for training the model and ECS to host it.

This is the general architecture diagram for how we host the blueprint on AWS.

![Architecture diagram showing the end-to-end AWS deployment workflow: A
Sagemaker development environment connects to S3 for data storage and
preprocessing. The workflow shows data moving from Sagemaker to S3, then to a
Sagemaker Training job. The trained model is stored in S3 and loaded into EFS,
which connects to an EC2/ECS instance running Nvidia Triton for model inference.
The diagram illustrates the complete pipeline from development to production
deployment.](./docs/arch-diagram.png)

1. We will host our development environment inside SageMaker as it permits us to
   offload long lived compute to the cloud without having to keep our system up.
2. The notebook then does the data preprocessing for our model, splits it into
   training and testing sets, and uploads the data sets into S3.
3. The notebook kicks off a Sagemaker AI training job with pointers to our data
   and configuration, this then fully replicates our data, and executes the
   training job.
4. The trained model is output into S3, and that kicks off the model reload
   process in our inference instance.
5. The model is taken from S3 and loaded into EFS asynchronously so that it's
   ready to go into production.
6. For inference, we host an ECS service with a GPU attached to it that
   reads the model from an attached EFS drive and hosts it on Nvidia
   Dynamo-Triton.

## Prerequisites

To successfully deploy this blueprint, you will need:

1. **AWS Account** - with appropriate permissions to create SageMaker, EC2, ECS,
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

### Upload Nvidia Container to ECR

From your local machine with Docker installed:

1. Clone this repository
```sh
git clone https://github.com/aws-samples/fraud-detection-blueprint-with-triton
cd fraud-detection-blueprint-with-triton
```

2. Configure your AWS credentials
```sh
aws configure
```

3. Log in to Nvidia NGC
```sh
docker login nvcr.io
```
When prompted, use your NGC API key as the password and `$oauthtoken` as the username

If you don't have an NGC API key, follow these steps:
1. Go to the [NVIDIA NGC website](https://ngc.nvidia.com/)
2. Create an account or sign in
3. Navigate to your account settings
4. Generate an API key in the "Setup" section

4. Execute the provided Makefile to build and push the container to ECR
```sh
make setup-ecr AWS_REGION=your-aws-region
```
This will:
- Create an ECR repository if it doesn't exist
- Pull the necessary Nvidia container from NGC
- Tag it appropriately for your ECR repository
- Push the container to your ECR

You can view all available Makefile commands with:
```sh
make help
```

### Set up Development Environment in SageMaker Studio

From within your SageMaker Studio environment:

1. Clone the repository
```sh
git clone https://github.com/aws-samples/fraud-detection-blueprint-with-triton
```

2. Set up the required conda environment
```sh
conda env create -f ./fraud-detection-blueprint-with-triton/conda/notebook_env.yaml
conda activate notebook_env
```

3. Open the notebook and follow the instructions
   - Navigate to the `notebooks` directory and open the main notebook
   - The notebook will guide you through the process of data preparation, model
     training, and deployment

### Running the Solution

Follow the step-by-step instructions in the notebook to:
1. Preprocess the financial transaction data
2. Train the fraud detection model using the Nvidia container on SageMaker
3. Deploy the trained model to Triton Inference Server
4. Set up the inference pipeline on EC2/ECS

For detailed troubleshooting and additional configuration options, refer to the
documentation in the `docs` directory.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
