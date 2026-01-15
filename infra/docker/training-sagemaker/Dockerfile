FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0

# Install SageMaker training toolkit
RUN pip install sagemaker-training

# Copy the SageMaker training wrapper
COPY src/train_sagemaker.py train_sagemaker.py

# Set the entrypoint to the wrapper script
ENV SAGEMAKER_PROGRAM train_sagemaker.py
ENTRYPOINT [ "python", "train_sagemaker.py" ]
