#!/usr/bin/env python3
"""
Example script demonstrating how to use the ViTDet + Mask R-CNN pipeline
for prostate cancer detection.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

def example_training():
    """Example of how to start training."""
    
    print("=== ViTDet + Mask R-CNN Training Example ===")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. This pipeline requires GPU.")
        return False
    
    print(f"✓ Found {torch.cuda.device_count()} GPU(s)")
    
    # Example data directory structure
    data_dir = "/path/to/your/prostate_data"
    
    print(f"\nExpected data structure:")
    print(f"{data_dir}/")
    print(f"├── train.csv")
    print(f"├── train_images/")
    print(f"│   ├── image_001.tiff")
    print(f"│   └── ...")
    print(f"└── train_label_masks/")
    print(f"    ├── image_001_mask.tiff")
    print(f"    └── ...")
    
    # Example training commands
    print(f"\nTraining commands:")
    
    print(f"\n1. Automated setup and training:")
    print(f"   python launch_vitdet.py --data-dir {data_dir} --num-gpus 4")
    
    print(f"\n2. Manual training (after setup):")
    print(f"   python train_detectron.py --config configs/vitdet_config.yaml --num-gpus 4")
    
    print(f"\n3. Single GPU training:")
    print(f"   python train_detectron.py --config configs/vitdet_config.yaml --num-gpus 1")
    
    print(f"\n4. Resume training:")
    print(f"   python train_detectron.py --config configs/vitdet_config.yaml --num-gpus 4 --resume")
    
    return True

def example_config_customization():
    """Example of how to customize the configuration."""
    
    print("\n=== Configuration Customization Example ===")
    
    config_example = """
# Example custom configuration changes

# For smaller GPU memory (e.g., RTX 3080 with 10GB)
data:
  batch_size: 2        # Reduce from 4
  patch_size: 512      # Reduce from 1024

# For faster prototyping
training:
  epochs: 10           # Reduce from 50
  eval_period: 500     # More frequent evaluation

# For different learning rate
training:
  base_lr: 0.0002      # Increase learning rate

# For different model size
model:
  backbone: "ViT-L/16" # Use larger model (requires more memory)
"""
    
    print(config_example)
    
    print("To use custom settings:")
    print("1. Copy configs/vitdet_config.yaml to configs/my_config.yaml")
    print("2. Edit the values as needed")
    print("3. Use: python train_detectron.py --config configs/my_config.yaml")

def example_data_preparation():
    """Example of how to prepare your data."""
    
    print("\n=== Data Preparation Example ===")
    
    # Example CSV structure
    csv_example = """
# train.csv should contain at least these columns:
image_id,isup_grade,gleason_score
image_001,0,0+0
image_002,1,3+3
image_003,2,3+4
image_004,3,4+3
image_005,4,4+4
image_006,5,5+5
"""
    
    print("Required CSV format:")
    print(csv_example)
    
    print("Data requirements:")
    print("- WSI images in .tiff format")
    print("- Mask images in .tiff format (optional, but recommended)")
    print("- Image IDs should match between CSV and file names")
    print("- ISUP grades should be integers 0-5")

def example_evaluation():
    """Example of how to evaluate the model."""
    
    print("\n=== Model Evaluation Example ===")
    
    print("Evaluation commands:")
    
    print("\n1. Evaluate during training (automatic):")
    print("   Set eval_period in config (default: every 2000 iterations)")
    
    print("\n2. Evaluation-only mode:")
    print("   python train_detectron.py --config configs/vitdet_config.yaml --eval-only")
    
    print("\n3. Inference on new WSI:")
    print("   python inference.py --checkpoint logs/best_model.pth --wsi_path /path/to/slide.tiff")
    
    print("\nEvaluation metrics:")
    print("- mAP (mean Average Precision) at IoU 0.5 and 0.75")
    print("- Per-class AP for each ISUP grade")
    print("- Cohen's Kappa for classification agreement")

def example_troubleshooting():
    """Example troubleshooting steps."""
    
    print("\n=== Troubleshooting Example ===")
    
    print("Common issues and solutions:")
    
    print("\n1. CUDA Out of Memory:")
    print("   - Reduce batch_size to 1 or 2")
    print("   - Reduce patch_size to 512")
    print("   - Use gradient checkpointing")
    
    print("\n2. Slow training:")
    print("   - Use SSD storage for data")
    print("   - Increase num_workers in config")
    print("   - Use mixed precision (enabled by default)")
    
    print("\n3. Dependencies issues:")
    print("   - Use the automated installer: python launch_vitdet.py --install-only")
    print("   - Check CUDA version compatibility")
    print("   - Install OpenSlide system dependencies")
    
    print("\n4. Data loading errors:")
    print("   - Check file paths in config")
    print("   - Verify WSI file formats (.tiff)")
    print("   - Check CSV column names")

def example_monitoring():
    """Example of how to monitor training."""
    
    print("\n=== Training Monitoring Example ===")
    
    print("Monitoring options:")
    
    print("\n1. TensorBoard:")
    print("   tensorboard --logdir logs_vitdet")
    print("   Open: http://localhost:6006")
    
    print("\n2. Weights & Biases:")
    print("   - Automatic if use_wandb: true in config")
    print("   - Dashboard at: https://wandb.ai")
    
    print("\n3. Console output:")
    print("   - Loss values every few iterations")
    print("   - Evaluation metrics every eval_period")
    print("   - Training progress with ETA")
    
    print("\n4. Log files:")
    print("   - Check logs_vitdet/train.log for detailed logs")
    print("   - Model checkpoints saved automatically")

def main():
    """Main example function."""
    
    print("ViTDet + Mask R-CNN for Prostate Cancer Detection")
    print("=" * 60)
    
    # Run all examples
    if not example_training():
        return
    
    example_config_customization()
    example_data_preparation()
    example_evaluation()
    example_monitoring()
    example_troubleshooting()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Prepare your data in the required format")
    print("2. Run: python launch_vitdet.py --data-dir /path/to/data --num-gpus 4")
    print("3. Monitor training with TensorBoard or W&B")
    print("4. Evaluate results and iterate on configuration")
    print("\nFor more details, see README.md")

if __name__ == "__main__":
    main()
