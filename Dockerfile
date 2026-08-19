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
    libgomp1 \
    libicu-dev \
    libglpk-dev \
    libgmp-dev \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Python 3.9 (deadsnakes; HTTPS key to avoid GPG keyserver timeouts)
# =========================
RUN mkdir -p /etc/apt/keyrings && \
    curl --retry 8 --retry-delay 3 -fsSL \
        "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /etc/apt/keyrings/deadsnakes.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu focal main" \
        > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    curl --retry 8 --retry-delay 3 -fsSL \
        -o /tmp/pip-25.1.1-py3-none-any.whl \
        https://files.pythonhosted.org/packages/29/a2/d40fb2460e883eca5199c62cfc2463fd261f760556ae6290f88488c362c0/pip-25.1.1-py3-none-any.whl && \
    python /tmp/pip-25.1.1-py3-none-any.whl/pip install --no-cache-dir \
        /tmp/pip-25.1.1-py3-none-any.whl setuptools wheel && \
    rm -f /tmp/pip-25.1.1-py3-none-any.whl && \
    python -m pip --version

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
ENV PATH=/usr/local/bin:/usr/bin:$PATH

# =========================
# R packages (HiCGNN KR + FLAMINGOr)
# =========================
RUN Rscript -e "options(timeout=600); \
    pkgs <- c('KernSmooth','MASS','Matrix','boot','class','cluster', \
      'codetools','foreign','lattice','mgcv','nlme','nnet', \
      'rpart','spatial','survival','data.table','remotes', \
      'igraph','Rcpp','RcppArmadillo'); \
    install.packages(pkgs, repos='https://cloud.r-project.org/', Ncpus=parallel::detectCores()); \
    missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; \
    if (length(missing)) stop(paste('missing R packages:', paste(missing, collapse=', ')))"

COPY docker/fetch_flamingor.py /tmp/fetch_flamingor.py
RUN python /tmp/fetch_flamingor.py && \
    test -d /tmp/FLAMINGO/FLAMINGOr && \
    Rscript -e "options(timeout=600); remotes::install_local('/tmp/FLAMINGO/FLAMINGOr', upgrade='never', dependencies=TRUE)" && \
    rm -rf /tmp/FLAMINGO /tmp/fetch_flamingor.py

# =========================
# Python packages
# =========================
# PyTorch (CUDA 11.8)
RUN python -m pip install \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# RAPIDS / CuPy (CUDA 11; pin for Python 3.9)
RUN python -m pip install \
    cupy-cuda11x==13.3.0 \
    cugraph-cu11==24.6.1 \
    --extra-index-url https://pypi.nvidia.com

# PyTorch Geometric (precompiled wheels)
RUN python -m pip install \
    torch-geometric==2.5.3 \
    --find-links https://data.pyg.org/whl/torch-2.1.1+cu118.html

RUN python -m pip install torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

RUN python -m pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

# Remaining Python packages (compatible with Python 3.9 / TF 2.13)
RUN python -m pip install \
    numpy==1.24.3 \
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
    fastdtw \
    numba \
    statsmodels

# =========================
# Environment variables
# =========================
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org pypi.nvidia.com data.pyg.org"
ENV PYTHONPATH=/HiCInterpolate:/HiCInterpolate/src
ENV MPLBACKEND=Agg

# =========================
# Project source
# =========================
COPY . /HiCInterpolate
WORKDIR /HiCInterpolate

# Verify entry-point modules import (data/weights are mounted at runtime)
RUN python -c "import ast; ast.parse(open('hicinterpolate.py').read()); ast.parse(open('inference.py').read()); ast.parse(open('dsa.py').read()); ast.parse(open('test_hicinterpolate.py').read())" && \
    python hicinterpolate.py -h >/dev/null && \
    python inference.py -h >/dev/null && \
    python dsa.py -h >/dev/null && \
    python -c "import torch, numpy, omegaconf, cooler, cooltools, sklearn, h5py, torch_geometric; import flow_based_interpolation; import _4DMax.model; from downstream_analysis import run_compartment, run_embedtad, run_mustache, run_flamingo"

CMD ["/bin/bash"]
