FROM nvcr.io/nvidia/rapidsai/base:25.12-cuda13-py3.12

# Install gcc/g++ via conda (since this is a conda-based image) and pip packages
RUN conda install -y -c conda-forge gcc gxx make && \
    pip install --no-cache-dir \
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
