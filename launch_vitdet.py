#!/usr/bin/env python3
"""
Launch script for ViTDet + Mask R-CNN training on prostate cancer detection.
This script handles environment setup, dependency installation, and training launch.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def check_gpu():
    """Check GPU availability and CUDA version."""
    print("=== GPU and CUDA Information ===")
    
    try:
        # Check nvidia-smi
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
        else:
            print("✗ No NVIDIA GPU found")
            return False
    except:
        print("✗ nvidia-smi not available")
        return False
    
    try:
        # Check CUDA version
        result = run_command("nvcc --version", check=False)
        if result.returncode == 0:
            print("✓ CUDA toolkit available")
        else:
            print("✓ CUDA runtime available (toolkit not required for inference)")
    except:
        print("? CUDA toolkit not in PATH (this is usually fine)")
    
    return True

def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\n=== Installing PyTorch with CUDA Support ===")
    
    # Check if PyTorch is already installed with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda} already installed")
            print(f"✓ {torch.cuda.device_count()} GPU(s) available")
            return True
    except ImportError:
        pass
    
    # Install PyTorch with CUDA 11.8 (compatible with most recent systems)
    pytorch_cmd = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    try:
        run_command(pytorch_cmd)
        
        # Verify installation
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch {torch.__version__} with CUDA installed successfully")
            print(f"✓ {torch.cuda.device_count()} GPU(s) available")
            return True
        else:
            print("✗ PyTorch installed but CUDA not available")
            return False
            
    except Exception as e:
        print(f"✗ Failed to install PyTorch: {e}")
        return False

def install_detectron2():
    """Install Detectron2."""
    print("\n=== Installing Detectron2 ===")
    
    try:
        import detectron2
        print(f"✓ Detectron2 already installed")
        return True
    except ImportError:
        pass
    
    # Install Detectron2 from source
    detectron2_cmd = (
        "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    )
    
    try:
        run_command(detectron2_cmd)
        
        # Verify installation
        import detectron2
        print(f"✓ Detectron2 installed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to install Detectron2: {e}")
        print("Trying alternative installation method...")
        
        # Try pre-built wheel for common configurations
        try:
            import torch
            torch_version = torch.__version__
            cuda_version = torch.version.cuda
            
            if cuda_version:
                detectron2_wheel = (
                    f"pip install detectron2 -f "
                    f"https://dl.fbaipublicfiles.com/detectron2/wheels/cu{cuda_version.replace('.', '')}/torch{torch_version}/index.html"
                )
                run_command(detectron2_wheel)
                
                import detectron2
                print(f"✓ Detectron2 installed from pre-built wheel")
                return True
        except:
            pass
        
        print("✗ Failed to install Detectron2. Please install manually.")
        return False

def install_dependencies():
    """Install all required dependencies."""
    print("\n=== Installing Dependencies ===")
    
    # Install from requirements file
    requirements_file = "requirements_detectron.txt"
    if os.path.exists(requirements_file):
        print(f"Installing from {requirements_file}...")
        run_command(f"pip install -r {requirements_file}")
    else:
        print("Installing individual packages...")
        packages = [
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
            "pycocotools>=2.0.4",
            "openslide-python>=1.1.2"
        ]
        
        for package in packages:
            try:
                run_command(f"pip install {package}")
            except:
                print(f"Failed to install {package}, continuing...")

def verify_installation():
    """Verify that all components are correctly installed."""
    print("\n=== Verifying Installation ===")
    
    success = True
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda} with {torch.cuda.device_count()} GPU(s)")
        else:
            print("✗ CUDA not available")
            success = False
    except ImportError:
        print("✗ PyTorch not found")
        success = False
    
    # Check Detectron2
    try:
        import detectron2
        print(f"✓ Detectron2")
    except ImportError:
        print("✗ Detectron2 not found")
        success = False
    
    # Check other key dependencies
    dependencies = [
        "numpy", "pandas", "cv2", "PIL", "albumentations", 
        "sklearn", "yaml", "wandb", "pycocotools"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} not found")
            success = False
    
    # Check openslide
    try:
        import openslide
        print("✓ openslide")
    except ImportError:
        print("? openslide not found (install openslide-python if using WSI files)")
    
    return success

def check_data_directory(data_dir):
    """Check if data directory structure is correct."""
    print(f"\n=== Checking Data Directory: {data_dir} ===")
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    # Check required files and directories
    required_items = [
        "train.csv",
        "train_images",
        "train_label_masks"
    ]
    
    missing_items = []
    for item in required_items:
        path = os.path.join(data_dir, item)
        if os.path.exists(path):
            if item.endswith('.csv'):
                print(f"✓ {item} found")
            else:
                file_count = len(os.listdir(path)) if os.path.isdir(path) else 0
                print(f"✓ {item} found ({file_count} files)")
        else:
            print(f"✗ {item} not found")
            missing_items.append(item)
    
    if missing_items:
        print(f"\nMissing items: {missing_items}")
        print("Please ensure your data directory contains:")
        print("  - train.csv (metadata file)")
        print("  - train_images/ (WSI image files)")
        print("  - train_label_masks/ (mask files)")
        return False
    
    return True

def launch_training(args):
    """Launch the training script."""
    print("\n=== Launching Training ===")
    
    # Build command
    cmd_parts = [
        "python", "train_detectron.py",
        "--config", args.config,
        "--num-gpus", str(args.num_gpus)
    ]
    
    if args.resume:
        cmd_parts.append("--resume")
    
    if args.eval_only:
        cmd_parts.append("--eval-only")
    
    if args.opts:
        cmd_parts.extend(["--opts"] + args.opts)
    
    cmd = " ".join(cmd_parts)
    
    print(f"Running: {cmd}")
    
    # Run training
    try:
        run_command(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Launch ViTDet + Mask R-CNN training setup and execution')
    parser.add_argument('--config', type=str, default='configs/vitdet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip dependency installation')
    parser.add_argument('--install-only', action='store_true',
                        help='Only install dependencies, do not train')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='Override config options')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ViTDet + Mask R-CNN Training Setup")
    print("=" * 60)
    
    # Check GPU availability
    if not check_gpu():
        print("\nWARNING: No GPU detected. This pipeline requires CUDA-capable GPUs.")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            sys.exit(1)
    
    if not args.skip_install:
        # Install PyTorch with CUDA
        if not install_pytorch_cuda():
            print("Failed to install PyTorch with CUDA")
            sys.exit(1)
        
        # Install Detectron2
        if not install_detectron2():
            print("Failed to install Detectron2")
            sys.exit(1)
        
        # Install other dependencies
        install_dependencies()
        
        # Verify installation
        if not verify_installation():
            print("Installation verification failed")
            sys.exit(1)
    
    if args.install_only:
        print("\n✓ Installation completed successfully!")
        return
    
    # Check data directory
    if not check_data_directory(args.data_dir):
        print("Data directory check failed")
        sys.exit(1)
    
    # Update config with data directory
    import yaml
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['data_dir'] = args.data_dir
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✓ Updated config with data directory: {args.data_dir}")
    
    # Launch training
    if launch_training(args):
        print("\n✓ Training completed successfully!")
    else:
        print("\n✗ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
