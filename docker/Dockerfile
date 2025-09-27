# Build arg determines CPU vs CUDA 
# Options:
#   pytorch/pytorch:2.3.1-cpu
#   pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ARG PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cpu
FROM ${PYTORCH_IMAGE}

# Keep model cache inside project (easier bind-mount)
ENV TORCH_HOME=/app/.cache/torch
WORKDIR /app

# system dependencies (lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (torch/torchvision already in base)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# source files
COPY src /app/src
COPY data /app/data

# Useful default env
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default entrypoint runs CLI, override args at runtime
ENTRYPOINT ["python", "-m", "src.main"]
