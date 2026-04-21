#FROM python:3.12
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set working directory (matches devcontainer workspace)
WORKDIR /workspace

ENV PYTHONPATH=/workspace:/workspace/predict

# Optional basic tools
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies only (not your source code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

