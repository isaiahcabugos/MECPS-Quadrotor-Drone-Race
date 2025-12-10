# Use DEVEL image (includes all CUDA dev tools)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# CUDA paths
ENV PATH=/usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# XLA configuration (multiple fallbacks)
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8"
ENV TF_CUDA_PATHS="/usr/local/cuda-11.8,/usr/lib/x86_64-linux-gnu,/usr/include"
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices=true"

# Install Python 3.9
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Install TensorFlow and dependencies
RUN pip install --no-cache-dir tensorflow==2.13.0 numpy==1.24.3
RUN pip install --no-cache-dir tf-agents==0.17.0
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    matplotlib==3.7.2 \
    tensorboard==2.13.0 \
    tqdm==4.66.1 \
    pandas==2.0.3 \
    scipy==1.10.1 \
    psutil \
    gputil

WORKDIR /workspace

# CRITICAL: Create libdevice symlinks in multiple locations
RUN ln -sf /usr/local/cuda-11.8/nvvm/libdevice/libdevice.10.bc \
           /usr/local/cuda-11.8/libdevice.10.bc && \
    ln -sf /usr/local/cuda-11.8/nvvm/libdevice/libdevice.10.bc \
           /workspace/libdevice.10.bc && \
    ln -sf /usr/local/cuda-11.8/nvvm/libdevice/libdevice.10.bc \
           /libdevice.10.bc

# Copy source code
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/
COPY train.py /workspace/

# Create output directories
RUN mkdir -p /workspace/checkpoints /workspace/logs

EXPOSE 6006

CMD ["python", "train.py"]