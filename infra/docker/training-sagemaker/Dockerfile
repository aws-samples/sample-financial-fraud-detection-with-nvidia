FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0

# Install CUDA Forward Compatibility package for older drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-compat-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Enable CUDA forward compatibility
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# Copy the SageMaker training wrapper
COPY src/train_sagemaker.py train_sagemaker.py

# Set the entrypoint to the wrapper script
ENV SAGEMAKER_PROGRAM train_sagemaker.py
ENTRYPOINT [ "python", "train_sagemaker.py" ]
