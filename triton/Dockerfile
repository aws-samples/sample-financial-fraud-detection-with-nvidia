FROM nvcr.io/nvidia/tritonserver:25.04-py3

WORKDIR /opt

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/miniconda

# Install base dependencies and clean cache
RUN apt-get update && apt-get install -y \
    wget curl git build-essential cmake sudo \
    # libnccl2 libnccl-dev \
    libopenblas-dev libssl-dev libtinfo-dev \
    && rm -rf /var/lib/apt/lists/*


# Ensure bash features and a stable PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV CONDA_DIR=/opt/miniconda
ENV PATH=${CONDA_DIR}/bin:${PATH}


RUN set -eux; \
    ARCH="$(uname -m)"; \
    if [ "$ARCH" = "aarch64" ]; then \
      MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
      CONDA_SUBDIR=linux-aarch64; \
    else \
      MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
      CONDA_SUBDIR=linux-64; \
    fi; \
    wget -q "$MINICONDA_URL" -O miniconda.sh; \
    bash miniconda.sh -b -p "$CONDA_DIR"; \
    rm miniconda.sh; \
    conda config --set subdir "$CONDA_SUBDIR"; \
    conda config --set always_yes true; \
    # Accept TOS
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main; \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    conda update -n base -c defaults conda


# Set Conda Python as default
ENV PATH="$CONDA_DIR/bin:$PATH"
ENV PYTHONPATH="$CONDA_DIR/lib/python3.12/site-packages:$PYTHONPATH"
ENV LD_LIBRARY_PATH="$CONDA_DIR/lib:$LD_LIBRARY_PATH"
ENV CONDA_OVERRIDE_CUDA=12.8
ENV CONDA_ALWAYS_YES=true

RUN conda install python=3.12

# Install GPU-enabled PyTorch and XGBoost

# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda install --override-channels -c nvidia -c conda-forge \
    pytorch=2.7.0=*cuda126* \
    py-xgboost=3.0.2=*cuda128* \
    ncurses

RUN conda install  --override-channels -c conda-forge cupy=13.4.1
# Install PyTorch Geometric and Captum for CUDA 12.6 + torch 2.7.0
RUN $CONDA_DIR/bin/pip install \
    torch-geometric==2.6.1 \
    captum==0.7.0 \
    --extra-index-url https://data.pyg.org/whl/torch-2.7.0+cu126.html

# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda install -y --override-channels -c rapidsai -c nvidia -c conda-forge cudf=25.04 python=3.12

# Validate installation with explicit Python path
RUN python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
RUN python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
RUN python -c "import captum; print('Captum:', captum.__version__)"
RUN python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"

# Entrypoint remains unchanged from Triton base image
