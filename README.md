<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Financial Fraud Detection
</h2>

## Table of Contents
- [Overview](#overview)
    - [Software Components](#software-components)
    - [Target Audience](#target-audience)
    - [Prerequisites](#prerequisites)
    - [Hardware Requirements](#hardware-requirements)
- [Getting Started](#getting-started)
    - [Install System Requirements](#install-system-requirements)
    - [Obtain API key](#obtain-api-key)
    - [Clone The Repository](#clone-the-repository)
    - [Set up the environment file](#set-up-the-environment-file)
    - [Authenticate Docker with NGC](#authenticate-docker-with-ngc)
- [Running the workflow](#running-the-workflow)
- [Customizing the Workflow](#customizing-the-workflow)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Terms of Use](#terms-of-use)

----

<br>


# Overview
Transaction fraud detection is a [$43B problem annually](https://nilsonreport.com/articles/card-fraud-losses-worldwide-2/) and poses a big challenge for financial institutions to detect and prevent sophisticated fraudulent activities. Traditional fraud detection methods, which rely on rules-based systems, or statistical methods, are reactive and increasingly ineffective in identifying sophisticated fraudulent activities. As data volumes grow and fraud tactics evolve, financial institutions need more proactive, intelligent approaches to detect and prevent fraudulent transactions. 

This NVIDIA AI blueprint provides a reference example to detect and prevent sophisticated fraudulent activities for financial services with high accuracy and reduced false positives. It shows developers how to build a Financial Fraud Detection workflow using NVIDIA NIM microservices for fraud detection.  The model building NIM augments fraud detection using graph neural networks (GNNs), a deep learning technique, for improved accuracy. Inference is done using the NVIDIA Triton NIM and produces fraud scores along with Shapely values for explainability.

<img width="1000" alt="Architecture Diagram" src="docs/financial-fraud-bp.png"/>

This NVIDIA AI blueprint is broken down into three steps, which map to processes within a typical payment processing environment, those steps being: (1) Data Preparation, (2) Model Building, and (3) Data Inference. For this example, the data is just a collection of files with synthetic data. Within a production system, the event data would most likely be saved within a database or a data lake. The data os prepared and then feed into the model buulding NIM, called financial-fraud-training.  The output of the NIM folder with all the artifacts needed to be passed to Triton for inference. 

This blueprint does not use any NVIDIA hosted services and runs fully in a self-hosted docker environment.

<br>

### Software Components
The following software components are used:
- financial-fraud-training  (LINK to NGC when ready)
- [Triton Inference Server](https://developer.nvidia.com/triton-inference-server)

Eerything is run via a Jupyer Notebook.

<br>

### Target Audience
This Blueprint targets users that:
* understand the financial fraud space
* understand how to deploy container-based microservices
* understands how to run a Jupyter notebook

This notebook is a simple example of how to orchestrate a financial fraud detection workflow that leverage the financial-fraud-training microservice. The notebook using a synthetic dataset and output the accuracy and a confusion matrix. In a production environment, real data would be used, but the workflow is the same.  

<br>

### Prerequisites
  - [Obtain NVIDIA NVAIE key](#obtain-nvidia-nvaie-key)
  - [CUDA 12.5+ drivers](https://developer.nvidia.com/cuda-downloads) installed

<br>

### Hardware Requirements
* GPU: 1x A6000, A100, or H100, minimum of 32GB of memory 
* CPU: x86_64 architecture
* Storage: 10GB
* System Memory: 16GB

<br>

# Getting Started
### Install System Requirements
  - [git](https://git-scm.com/)
  - [Jupyter Notebook or Jupyter Lab](https://jupyter.org/install)

The Notebook contains a cell that install all needed Python packages.

<br>

### Obtain API key
There are two possible methods to generate an API key for NIM:
   - Sign in to the [NVIDIA Build](https://build.nvidia.com/explore/discover?signin=true) portal with your email.
   - Sign in to the [NVIDIA NGC](https://ngc.nvidia.com/) portal with your email.
      - Select your organization from the dropdown menu after logging in. You must select an organization which has NVIDIA AI Enterprise (NVAIE) enabled.
      - Click on your account in the top right, select "Setup" from the dropdown.
      - Click the "Generate Personal Key" option and then the "+ Generate Personal Key" button to create your API key.
         - This will be used in the NVIDIA_API_KEY environment variable.
      - Click the "Generate API Key" option and then the "+ Generate API Key" button to create the API key.

IMPORTANT: This will be used in the NVIDIA_API_KEY environment variable below.

- API catalog keys:
    NVIDIA [API catalog](https://build.nvidia.com/) or [NGC](https://org.ngc.nvidia.com/setup/personal-keys)

<br>

### Clone The Repository
   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection
   ```

<br>

### Set up the environment file
   ```bash
   # Create the API environm,ent   
   export NVIDIA_API_KEY=your_key
   ```

Note: The environment variable could also be added to the `.bashrc` file to persist it beyond a single instance
```bash
echo "export NVIDIA_API_KEY=your_key" >> ~./bashrc
```

<br>

### Authenticate Docker with NGC
In order to pull images required by the Blueprint from NGC, you must first authenticate Docker with NGC. You can use same the NVIDIA API Key obtained in the [Obtain API key](#obtain-api-key) section (saved in the `NVIDIA_API_KEY` environment variable).

```bash
echo "${NVIDIA_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

<br>

# Running the workflow


# Customizing the Workflow

# Troubleshooting

# License
See [license file](./LICENSE)

Copyright (c) 2025, NVIDIA CORPORATION.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


# Terms of Use
