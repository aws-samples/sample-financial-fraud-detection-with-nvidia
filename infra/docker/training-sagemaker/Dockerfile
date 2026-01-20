FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0

# Install CUDA forward compatibility package for running CUDA 13 on host driver 570 (CUDA 12.8)
# See: https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
RUN apt-get update && apt-get install -y cuda-compat-13-0 && rm -rf /var/lib/apt/lists/*

# Set LD_LIBRARY_PATH to use the forward compatibility libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}

ENV CUDA_VISIBLE_DEVICES=0

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# Copy the SageMaker training wrapper
COPY src/train_sagemaker.py train_sagemaker.py

# Set the entrypoint to the wrapper script
ENV SAGEMAKER_PROGRAM train_sagemaker.py
ENTRYPOINT [ "python", "train_sagemaker.py" ]
