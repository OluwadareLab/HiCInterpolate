# =========================
# Base CUDA (PyTorch 2.1.1)
# =========================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =========================
# System dependencies
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
    libreadline-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libbz2-dev \
    liblzma-dev \
    libpcre2-dev \
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

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# =========================
# R 4.1.0 (build from source)
# =========================
RUN wget https://cran.r-project.org/src/base/R-4/R-4.1.0.tar.gz && \
    tar -xzf R-4.1.0.tar.gz && \
    cd R-4.1.0 && \
    ./configure \
        --enable-R-shlib \
        --with-blas \
        --with-lapack \
        --with-readline \
        --with-x=no && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf R-4.1.0*

ENV R_HOME=/usr/local/lib/R
ENV PATH=/usr/local/lib/R/bin:$PATH

# =========================
# Recommended R packages
# =========================
RUN Rscript -e "install.packages( \
    c('KernSmooth','MASS','Matrix','boot','class','cluster', \
      'codetools','foreign','lattice','mgcv','nlme','nnet', \
      'rpart','spatial','survival'), \
    repos='https://cloud.r-project.org/')"

# =========================
# Python packages
# =========================
RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 11.8)
RUN pip install \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# RAPIDS / CuPy (CUDA 11)
RUN pip install \
    cupy-cuda11x==13.3.0 \
    cugraph-cu11 \
    --extra-index-url https://pypi.nvidia.com

# PyTorch Geometric (precompiled wheels)
RUN pip install \
    torch-geometric==2.5.3 \
    --find-links https://data.pyg.org/whl/torch-2.1.1+cu118.html

# Remaining Python packages (compatible with Python 3.9)
RUN pip install \
    tensorflow==2.13.0 \
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
    cooler \
    scipy==1.10.1

# =========================
# Environment variables
# =========================
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN pip install gensim
# fastdtw
RUN pip install fastdtw

# PyTorch Geometric dependencies
RUN pip install torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

RUN pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

RUN git clone https://github.com/OluwadareLab/HiCInterpolate.git
WORKDIR /HiCInterpolate
# Default shell
CMD ["/bin/bash"]