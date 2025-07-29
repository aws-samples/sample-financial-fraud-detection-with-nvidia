# Getting Started - AWS Edition

## Overview
This Nvidia AI Blueprint along with the AWS Deployment guide provides a reference example to deploy an End to End financial fraud detection worfklow using Machine Learning. We will leverage Nvidia's Triton inference server to host our models while we use Sagemaker for training the model and EC2/ECS to host it.

This is the general architecture diagram for how we host the blueprint on AWS.

![]()

1. We will host our development environment inside sagemaker as it permits us to offload long lived compute to the cloud without having to keep our system up.
2. The notebook then does the data preprocessing for our model, splits it into training and testing sets, and uploads the data sets into S3.
3. The notebook kicks off a Sagemaker AI training job with pointers to our data and configuration, this then fully replicates our data, and executes the training job.
4. The trained model is output into S3, and that kicks off the model reload process in our inference instance.
5. The model is taken from S3 and loaded into EFS asynchronously so that it's ready to go into production.
6. For inference, we host an EC2/ECS service with a GPU attached to it that reads the model from an attached EFS drive and hosts it on Nvidia Dynamo-Triton.

### Prerequisites
Literally just an AWS Account
It would also be preferred if you had some kind of local machine with docker and like 30Gb of space.
And a Sagemaker studio jupyter labs environment thingy.

### Developer Environment setup

To minimize client side setup, we can leverage Sagemaker Studio's Jupyter notebook environment. We recommend at least `ml.g4dn.4xlarge` with 50Gb of storage.

From inside the jupyter environment, you can then clone this repo.
```sh
git clone https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection
```

Then setup the required conda environment -- this command assumes you're in the home directory of the workspace, and have cloned the repo as is.

```sh
conda env create -f ./financial-fraud-detection/conda/notebook_env.yaml
conda activate notebook_env
```
