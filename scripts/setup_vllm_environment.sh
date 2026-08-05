#!/bin/bash
# =============================================================================
# Setup Script for VLLM + Qwen on Winston HPC
# =============================================================================
# Run this ONCE before submitting jobs
#
# Usage:
#   chmod +x setup_vllm_environment.sh
#   ./setup_vllm_environment.sh
# =============================================================================

echo "=============================================="
echo "Setting up VLLM Environment for Silicon Sampling"
echo "=============================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda found"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    else
        echo "Installing Miniconda locally..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        bash ~/miniconda.sh -b -p ~/miniconda3
        rm ~/miniconda.sh
        export PATH=~/miniconda3/bin:$PATH
        conda init bash
        echo "✓ Miniconda installed"
        echo ""
        echo "IMPORTANT: Please log out and log back in, then run this script again."
        exit 0
    fi
fi

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh

# Environment name
ENV_NAME="silicon_vllm"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment."
        echo "To activate: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

echo ""
echo "Creating conda environment: ${ENV_NAME} (Python 3.11)..."
conda create -n ${ENV_NAME} python=3.11 -y

echo ""
echo "Activating environment..."
conda activate ${ENV_NAME}

echo ""
echo "Installing packages (this may take 10-15 minutes)..."
pip install --upgrade pip

# Core data science packages
echo "Installing core packages..."
pip install pandas numpy scipy matplotlib seaborn statsmodels tqdm

# For reading Stata files
echo "Installing pyreadstat..."
pip install pyreadstat

# VLLM - this is the key package (requires CUDA)
# Note: VLLM installation may take a while as it compiles CUDA kernels
echo ""
echo "Installing VLLM (this may take several minutes)..."
pip install vllm

# Transformers for tokenizers
echo "Installing transformers..."
pip install transformers accelerate

# OpenAI client (optional, for server mode)
pip install openai

echo ""
echo "=============================================="
echo "✓ Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Available Qwen models (choose based on GPU memory):"
echo "  - Qwen/Qwen2.5-1.5B-Instruct  (~3GB VRAM, fastest)"
echo "  - Qwen/Qwen2.5-7B-Instruct    (~14GB VRAM, good balance)"
echo "  - Qwen/Qwen2.5-14B-Instruct   (~28GB VRAM, better quality)"
echo "  - Qwen/Qwen2.5-32B-Instruct   (~64GB VRAM, high quality)"
echo ""
echo "Check Winston GPU availability:"
echo "  sinfo -p winston-gpu"
echo ""
echo "Request an interactive GPU session for testing:"
echo "  srun --partition=winston-gpu --gres=gpu:1 --time=1:00:00 --mem=32G --pty bash"
echo ""
