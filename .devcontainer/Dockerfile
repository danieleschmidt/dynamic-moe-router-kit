# Development container for dynamic-moe-router-kit
FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    htop \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install development tools
RUN pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /workspaces/dynamic-moe-router-kit

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install project dependencies
RUN pip install -e .[dev,torch,jax,tf]

# Install additional development tools
RUN pip install \
    jupyterlab \
    ipywidgets \
    tensorboard \
    wandb \
    matplotlib \
    seaborn

# Set up pre-commit
RUN git config --global init.defaultBranch main

# Expose ports for development servers
EXPOSE 8000 8888 6006

# Set default command
CMD ["/bin/bash"]