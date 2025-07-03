#!/bin/bash
# Conda Environment Management Guide

echo "=== Conda Environment Management ==="

echo "1. Activate your conda environment:"
echo "conda activate prostate38"

echo -e "\n2. Check current environment:"
echo "conda info --envs"

echo -e "\n3. Check installed packages:"
echo "conda list"

echo -e "\n4. Install TensorBoard in conda environment:"
echo "conda activate prostate38"
echo "pip install tensorboard"

echo -e "\n5. Install additional monitoring tools:"
echo "pip install gpustat"
echo "pip install wandb"  # Alternative to TensorBoard

echo -e "\n6. Deactivate environment:"
echo "conda deactivate"

echo -e "\n7. Create new environment (if needed):"
echo "conda create -n prostate_new python=3.8"

echo -e "\n8. Remove environment (if needed):"
echo "conda remove -n prostate38 --all"

echo -e "\n=== Quick Activation Commands ==="
echo "# Add to your ~/.bashrc for quick activation:"
echo "alias act_prostate='conda activate prostate38'"
echo "alias deact='conda deactivate'"
