<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Financial Fraud Detection
</h2>

### Overview
Transaction fraud detection is a [$43B problem annually](https://nilsonreport.com/articles/card-fraud-losses-worldwide-2/) and poses a big challenge for financial institutions to detect and prevent sophisticated fraudulent activities. Traditional fraud detection methods, which rely on rules-based systems, or statistical methods, are reactive and increasingly ineffective in identifying sophisticated fraudulent activities. As data volumes grow and fraud tactics evolve, financial institutions need more proactive, intelligent approaches to detect and prevent fraudulent transactions. 

This NVIDIA AI blueprint provides a reference example to detect and prevent sophisticated fraudulent activities for financial services with high accuracy and reduced false positives. It shows developers how to build a Financial Fraud Detection workflow using NVIDIA NIM microservices for fraud detection.  The model building NIM augments fraud detection using graph neural networks (GNNs), a deep learning technique, for improved accuracy. Inference is done using the NVIDIA Triton NIM and produces fraud scores along with Shapely values for explainability.

<img width="1000" alt="Architecture Diagram" src="docs/financial-fraud-bp.png"/>

This NVIDIA AI blueprint is broken down into three steps, which map to processes within a typical payment processing environment, those steps being: (1) Data Preparation, (2) Model Building, and (3) Data Inference. For this example, the data is just a collection of files with synthetic data. Within a production system, the event data would most likely be saved within a database or a data lake. The data os prepared and then feed into the model buulding NIM, called financial-fraud-training.  The output of the NIM folder with all the artifacts needed to be passed to Triton for inference. 


### Software Components
- financial-fraud-training  (LINK to NGC when ready)
- Triton Inference Server


### Target Audience
This Blueprint targets users that:
* understand the financial fraud space
* understand how to deploy container-based microservices
* understands how to run a Jupyter notebook
* understand a confusion matrix


### Prerequisites
- NVIDIA AI Enterprise (NVAIE) developer license required
- API catalog keys:
    NVIDIA [API catalog](https://build.nvidia.com/) or [NGC](https://org.ngc.nvidia.com/setup/personal-keys)

### Hardware Requirements
* GPU: 1x A6000, A100, or H100, minimum of 32GB of memory 
* CPU: x86_64 architecture
* Storage: 10GB
* System Memory: 16GB



# Quickstart Guide

1. [**Clone the repo and setup environment to run the notebook**](./setup.md)
<br>

2. **Obtain NVIDIA Microservices (NIM) API key**

There are two possible methods to generate an API key for NIM:
   - Sign in to the [NVIDIA Build](https://build.nvidia.com/explore/discover?signin=true) portal with your email.
      - Click on any [model](https://build.nvidia.com/meta/llama-3_1-70b-instruct), then click "Get API Key", and finally click "Generate Key".
   - Sign in to the [NVIDIA NGC](https://ngc.nvidia.com/) portal with your email.
      - Select your organization from the dropdown menu after logging in. You must select an organization which has NVIDIA AI Enterprise (NVAIE) enabled.
      - Click on your account in the top right, select "Setup" from the dropdown.
      - Click the "Generate Personal Key" option and then the "+ Generate Personal Key" button to create your API key.
         - This will be used in the NVIDIA_API_KEY environment variable.
      - Click the "Generate API Key" option and then the "+ Generate API Key" button to create the API key.

IMPORTANT: This will be used in the NVIDIA_API_KEY environment variable below.


3. **Set environment variables**

   ```bash
   #Create env file with required variables in /home/<username>/.local/bin/env  
   echo "NVIDIA_API_KEY=your_key" >> .env
   ```
