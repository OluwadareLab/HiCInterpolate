# =========================
# Base CUDA (PyTorch 2.1.1)
# =========================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =========================
# System deps
# =========================
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python 3.9
# =========================
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python

# =========================
# R 4.1.0
# =========================
RUN wget https://cran.r-project.org/src/base/R-4/R-4.1.0.tar.gz && \
    tar -xzf R-4.1.0.tar.gz && \
    cd R-4.1.0 && \
    ./configure --enable-R-shlib --with-blas --with-lapack && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf R-4.1.0*

ENV R_HOME=/usr/local/lib/R
ENV PATH="${R_HOME}/bin:${PATH}"

# =========================
# Install recommended R packages
# =========================
RUN Rscript -e "install.packages(c('KernSmooth','MASS','Matrix','boot','class','cluster','codetools','foreign','lattice','mgcv','nlme','nnet','rpart','spatial','survival'), repos='https://cloud.r-project.org/')"

# =========================
# Python packages
# =========================
RUN pip install --upgrade pip setuptools wheel

# PyTorch first
RUN pip install \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# NVIDIA RAPIDS (CUDA 12)
RUN pip install \
    cupy-cuda12x==13.3.0 \
    cugraph-cu12 \
    --extra-index-url https://pypi.nvidia.com

# Remaining Python packages
RUN pip install \
    torch-geometric==2.5.3 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    torch-spline-conv==1.2.2 \
    tensorflow==2.15.0 \
    numpy==1.26.0 \
    scipy==1.15.1 \
    pandas==2.2.3 \
    matplotlib==3.9.4 \
    scikit-learn==1.6.1 \
    seaborn==0.13.2 \
    networkx==3.2.1 \
    tqdm==4.65.0 \
    torchmetrics==1.7.1 \
    omegaconf==2.3.0 \
    lpips \
    wandb \
    cooler

# =========================
# Environment variables
# =========================
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace
CMD ["/bin/bash"]