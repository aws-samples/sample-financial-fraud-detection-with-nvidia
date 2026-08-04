FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0

RUN apt-get update && apt-get install -y cuda-compat-13-0 && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir sagemaker-training

COPY src/neptune_client.py src/train_sagemaker.py /app/

ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH} \
    CUDA_VISIBLE_DEVICES=0 \
    SAGEMAKER_PROGRAM=train_sagemaker.py

ENTRYPOINT ["python", "/app/train_sagemaker.py"]
