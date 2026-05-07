# Stage 1: install deps into a scratch area
FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0 AS builder

RUN apt-get update && apt-get install -y cuda-compat-13-0 && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir sagemaker-training

# Stage 2: flatten onto the base image in a single layer
FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0

# Copy everything the builder installed (CUDA compat + pip packages) in one layer
COPY --from=builder /usr/local/cuda-13.0/compat /usr/local/cuda-13.0/compat
COPY --from=builder /opt/conda/lib/python3.12/site-packages /opt/conda/lib/python3.12/site-packages

ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}
ENV CUDA_VISIBLE_DEVICES=0
ENV SAGEMAKER_PROGRAM=train_sagemaker.py

# Copy the SageMaker training wrapper
COPY src/neptune_client.py neptune_client.py
COPY src/train_sagemaker.py train_sagemaker.py

ENTRYPOINT [ "python", "train_sagemaker.py" ]
