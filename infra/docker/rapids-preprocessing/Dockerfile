FROM nvcr.io/nvidia/rapidsai/base:25.12-cuda13-py3.12

USER root

# Fix apt (RAPIDS image strips it) then install gcc for pip compilation
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

USER rapids

RUN pip install --no-cache-dir \
    sagemaker-training \
    category_encoders \
    scikit-learn \
    matplotlib \
    networkx \
    scipy

# Copy preprocessing scripts
# Assumes build context is the repository root
COPY src/preprocess_TabFormer_sagemaker.py /opt/ml/code/preprocess.py
COPY src/preprocess_TabFormer_lp.py /opt/ml/code/preprocess_TabFormer_lp.py

# Set the entrypoint to the wrapper script
ENV SAGEMAKER_PROGRAM preprocess.py
