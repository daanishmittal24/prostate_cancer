# ViTDet + Mask R-CNN Setup and Training Script for Windows
# This script automates the setup and training process for prostate cancer detection

param(
    [string]$DataDir = "",
    [int]$NumGPUs = 4,
    [switch]$InstallOnly,
    [switch]$SkipInstall,
    [string]$Config = "configs/vitdet_config.yaml"
)

Write-Host "ViTDet + Mask R-CNN Training Setup for Windows" -ForegroundColor Green
Write-Host "=" * 60

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Function to run command with error handling
function Invoke-SafeCommand {
    param([string]$Command, [string]$Description)
    
    Write-Host "Running: $Description" -ForegroundColor Yellow
    Write-Host "Command: $Command" -ForegroundColor Gray
    
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }
        Write-Host "✓ $Description completed successfully" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "✗ $Description failed: $_" -ForegroundColor Red
        return $false
    }
}

# Check Python installation
Write-Host "`nChecking Python installation..." -ForegroundColor Cyan
if (-not (Test-Command "python")) {
    Write-Host "✗ Python not found. Please install Python 3.8+ and add to PATH" -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Host "✓ Found Python: $pythonVersion" -ForegroundColor Green

# Check pip
if (-not (Test-Command "pip")) {
    Write-Host "✗ pip not found. Please ensure pip is installed" -ForegroundColor Red
    exit 1
}

# Check GPU and CUDA
Write-Host "`nChecking GPU and CUDA..." -ForegroundColor Cyan
if (Test-Command "nvidia-smi") {
    Write-Host "✓ NVIDIA GPU detected" -ForegroundColor Green
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
} else {
    Write-Host "✗ No NVIDIA GPU found or nvidia-smi not available" -ForegroundColor Red
    $continue = Read-Host "Continue without GPU? (y/N)"
    if ($continue.ToLower() -ne "y") {
        exit 1
    }
}

if (-not $SkipInstall) {
    Write-Host "`nInstalling dependencies..." -ForegroundColor Cyan
    
    # Install PyTorch with CUDA
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
    $pytorchCommand = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    if (-not (Invoke-SafeCommand $pytorchCommand "PyTorch installation")) {
        Write-Host "Failed to install PyTorch" -ForegroundColor Red
        exit 1
    }
    
    # Verify PyTorch CUDA
    Write-Host "Verifying PyTorch CUDA installation..." -ForegroundColor Yellow
    $cudaCheck = python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
    Write-Host $cudaCheck -ForegroundColor Green
    
    # Install Detectron2
    Write-Host "Installing Detectron2..." -ForegroundColor Yellow
    $detectron2Command = "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    if (-not (Invoke-SafeCommand $detectron2Command "Detectron2 installation")) {
        Write-Host "Trying alternative Detectron2 installation..." -ForegroundColor Yellow
        # Try pre-built wheel
        $detectron2Alt = "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"
        if (-not (Invoke-SafeCommand $detectron2Alt "Detectron2 alternative installation")) {
            Write-Host "Failed to install Detectron2. Please install manually." -ForegroundColor Red
            exit 1
        }
    }
    
    # Install other dependencies
    Write-Host "Installing other dependencies..." -ForegroundColor Yellow
    if (Test-Path "requirements_detectron.txt") {
        $depsCommand = "pip install -r requirements_detectron.txt"
        Invoke-SafeCommand $depsCommand "Dependencies installation"
    } else {
        # Install individual packages
        $packages = @(
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scikit-learn>=1.0.0",
            "albumentations>=1.2.0",
            "opencv-python>=4.5.0",
            "Pillow>=8.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
            "tqdm>=4.62.0",
            "PyYAML>=6.0",
            "pycocotools>=2.0.4"
        )
        
        foreach ($package in $packages) {
            $cmd = "pip install $package"
            Invoke-SafeCommand $cmd "Installing $package"
        }
    }
    
    # Verify installation
    Write-Host "`nVerifying installation..." -ForegroundColor Cyan
    $verifyScript = @"
import torch
import detectron2
import numpy as np
import pandas as pd
import cv2
import yaml
import albumentations

print("✓ All packages imported successfully")
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPUs: {torch.cuda.device_count()}")
print("✓ Detectron2 installed")
"@
    
    $verifyScript | python
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Installation verification failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✓ All dependencies installed successfully" -ForegroundColor Green
}

if ($InstallOnly) {
    Write-Host "`n✓ Installation completed. Exiting as requested." -ForegroundColor Green
    exit 0
}

# Check data directory
if (-not $DataDir) {
    $DataDir = Read-Host "Please enter the path to your data directory"
}

if (-not (Test-Path $DataDir)) {
    Write-Host "✗ Data directory not found: $DataDir" -ForegroundColor Red
    exit 1
}

Write-Host "`nChecking data directory structure..." -ForegroundColor Cyan
$requiredItems = @("train.csv", "train_images", "train_label_masks")
$missingItems = @()

foreach ($item in $requiredItems) {
    $path = Join-Path $DataDir $item
    if (Test-Path $path) {
        if ($item.EndsWith(".csv")) {
            Write-Host "✓ $item found" -ForegroundColor Green
        } else {
            $fileCount = (Get-ChildItem $path).Count
            Write-Host "✓ $item found ($fileCount files)" -ForegroundColor Green
        }
    } else {
        Write-Host "✗ $item not found" -ForegroundColor Red
        $missingItems += $item
    }
}

if ($missingItems.Count -gt 0) {
    Write-Host "Missing items: $($missingItems -join ', ')" -ForegroundColor Red
    Write-Host "Please ensure your data directory contains:" -ForegroundColor Yellow
    Write-Host "  - train.csv (metadata file)" -ForegroundColor Yellow
    Write-Host "  - train_images/ (WSI image files)" -ForegroundColor Yellow  
    Write-Host "  - train_label_masks/ (mask files)" -ForegroundColor Yellow
    exit 1
}

# Update config with data directory
if (Test-Path $Config) {
    Write-Host "`nUpdating config with data directory..." -ForegroundColor Cyan
    
    # Simple config update using Python
    $updateScript = @"
import yaml
import sys

config_path = '$Config'
data_dir = '$DataDir'

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['data_dir'] = data_dir
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f'✓ Updated config with data directory: {data_dir}')
except Exception as e:
    print(f'✗ Failed to update config: {e}')
    sys.exit(1)
"@
    
    $updateScript | python
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to update config" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✗ Config file not found: $Config" -ForegroundColor Red
    exit 1
}

# Launch training
Write-Host "`nLaunching training..." -ForegroundColor Cyan
Write-Host "Configuration: $Config" -ForegroundColor Gray
Write-Host "Data directory: $DataDir" -ForegroundColor Gray
Write-Host "Number of GPUs: $NumGPUs" -ForegroundColor Gray

$trainingCommand = "python train_detectron.py --config $Config --num-gpus $NumGPUs"
Write-Host "Training command: $trainingCommand" -ForegroundColor Gray

Write-Host "`nStarting training..." -ForegroundColor Yellow
try {
    Invoke-Expression $trainingCommand
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Training completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "`n✗ Training failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nTraining completed. Check the logs directory for results." -ForegroundColor Green
Write-Host "Monitor training with: tensorboard --logdir logs_vitdet" -ForegroundColor Cyan
