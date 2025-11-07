# Multi-stage Dockerfile for NeuralLayers
# Supports both CPU and GPU deployments

ARG PYTORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

# Base image with PyTorch
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime AS base

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    streamlit \
    tensorboard

# Copy project files
COPY . .

# Install NeuralLayers in development mode
RUN pip install -e .

# Expose ports
EXPOSE 8888  # Jupyter
EXPOSE 8501  # Streamlit
EXPOSE 6006  # TensorBoard

# Create directories for data and outputs
RUN mkdir -p /workspace/data \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/exports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.torch
ENV HF_HOME=/workspace/.huggingface

# Default command (can be overridden)
CMD ["python", "demo_app.py", "--mode", "streamlit"]

# -------------------------------------------------------------------
# CPU-only variant
# -------------------------------------------------------------------
FROM python:3.11-slim AS cpu

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8888 8501 6006

CMD ["python", "demo_app.py", "--mode", "streamlit"]

# -------------------------------------------------------------------
# Development variant with additional tools
# -------------------------------------------------------------------
FROM base AS dev

RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    black \
    flake8 \
    isort \
    mypy \
    bandit \
    ipdb \
    pre-commit

CMD ["/bin/bash"]
