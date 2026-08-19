# =========================
# Base CUDA (PyTorch 2.1.1)
# =========================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    gnupg \
    dirmngr \
    pkg-config \
    cmake \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libgit2-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libreadline-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libbz2-dev \
    liblzma-dev \
    libpcre2-dev \
    zlib1g-dev \
    libhdf5-dev \
    hdf5-tools \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python 3.9
# =========================
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# =========================
# R (CRAN binary for Ubuntu 20.04)
# =========================
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | tee /usr/share/keyrings/cran-archive-keyring.asc >/dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/cran-archive-keyring.asc] https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/" \
        > /etc/apt/sources.list.d/cran.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
    && rm -rf /var/lib/apt/lists/*

ENV R_HOME=/usr/lib/R
ENV PATH=/usr/bin:$PATH

# =========================
# R packages (HiCGNN KR + FLAMINGOr)
# =========================
RUN Rscript -e "install.packages( \
    c('KernSmooth','MASS','Matrix','boot','class','cluster', \
      'codetools','foreign','lattice','mgcv','nlme','nnet', \
      'rpart','spatial','survival','data.table','remotes'), \
    repos='https://cloud.r-project.org/', Ncpus=parallel::detectCores())"

RUN Rscript -e "remotes::install_github('wangjr03/FLAMINGO', subdir='FLAMINGOr', upgrade='never', dependencies=TRUE)"

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

RUN pip install torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

RUN pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

# Remaining Python packages (compatible with Python 3.9 / TF 2.13)
RUN pip install \
    numpy==1.24.4 \
    tensorflow==2.13.0 \
    pandas==2.2.3 \
    matplotlib==3.9.4 \
    scikit-learn==1.6.1 \
    scikit-image==0.22.0 \
    seaborn==0.13.2 \
    networkx==3.2.1 \
    tqdm==4.65.0 \
    torchmetrics==1.7.1 \
    omegaconf==2.3.0 \
    lpips \
    wandb \
    scipy==1.10.1 \
    cython \
    h5py \
    cooler \
    cooltools \
    bioframe \
    mustache-hic \
    gensim \
    fastdtw

# =========================
# Environment variables
# =========================
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/HiCInterpolate:/HiCInterpolate/src

# =========================
# Project source
# =========================
COPY . /HiCInterpolate
WORKDIR /HiCInterpolate

CMD ["/bin/bash"]
