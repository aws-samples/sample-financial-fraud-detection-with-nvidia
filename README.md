<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Financial Fraud Detection
</h2>

### Overview
Transaction fraud detection is a [$43B problem annually](https://nilsonreport.com/articles/card-fraud-losses-worldwide-2/) and poses a big challenge for financial institutions to detect and prevent sophisticated fraudulent activities. Traditional fraud detection methods, which rely on rules-based systems, or statistical methods, are reactive and increasingly ineffective in identifying sophisticated fraudulent activities. As data volumes grow and fraud tactics evolve, financial institutions need more proactive, intelligent approaches to detect and prevent fraudulent transactions. 

This NVIDIA AI blueprint provides a reference example to detect and prevent sophisticated fraudulent activities for financial services with high accuracy and reduced false positives. It shows developers how to build a Financial Fraud Detection workflow using NVIDIA NIM microservices for fraud detection.  The model building NIM augments fraud detection using graph neural networks (GNNs), a deep learning technique, for improved accuracy. Inference is done using the NVIDIA Triton NIM and produces fraud scores along with Shapely values for explainability.


### Software Components
- financial-fraud-training  (LINK to NGC when ready)
- Triton Inference Server

This NVIDIA AI blueprint is broken down into three steps, which map to processes within a typical payment processing environment, those steps being: (1) Data Preparation, (2) Model Building, and (3) Data Inference. Within a production system, the event data would most likely be saved within a database or a data lake.  For this example, the data is just a collection of files with synthetic data. 

<img width="1000" alt="Architecture Diagram" src="docs/financial-fraud-bp.png"/>


### Target Audience
This Blueprint targets users that:
* understand the financial fraud space
* understand how to deploy container-based microservices
* understands how to run a Jupyter notebook
* understand a confusion matrix


### Prerequisites
- NVAIE developer license
- API Key

### Hardware Requirements
* GPU: 1x A6000, A100, or H100, minimum of 32GB of memory 
* CPU: x86_64 architecture
* Storage: 10GB
* System Memory: 16GB

### Quickstart Guide
- TBD


