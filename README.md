# Prostate Cancer Detection using ViTDet + Mask R-CNN

A state-of-the-art deep learning system for automated detection and grading of prostate cancer from whole-slide histopathology images using Vision Transformer with Detection (ViTDet) and Mask R-CNN.

## 🚀 Features

- **ViTDet + Mask R-CNN**: Modern object detection and segmentation pipeline using Vision Transformers
- **Multi-GPU Training**: Distributed training support for 4x V100 GPUs
- **Large-Scale WSI Processing**: Efficient patch extraction and processing from whole-slide images  
- **COCO Format**: Standard object detection evaluation metrics
- **Advanced Augmentations**: Histology-specific data augmentations using Albumentations
- **Multi-Class Detection**: Simultaneous detection of different ISUP grades (0-5)
- **Mixed Precision**: Fast training with automatic mixed precision
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration

## 📁 Project Structure

```
prostate_cancer/
├── configs/
│   ├── base_config.yaml          # Original ViT config (deprecated)
│   └── vitdet_config.yaml        # ViTDet + Mask R-CNN config
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Original dataset (deprecated)
│   ├── detection_dataset.py     # New Detectron2 dataset
│   └── transforms.py            # Original transforms (deprecated)
├── models/
│   ├── __init__.py
│   └── vit.py                   # Original ViT model (deprecated)
├── requirements.txt             # Original requirements
├── requirements_detectron.txt   # Detectron2 requirements
├── train.py                     # Original training script (deprecated)
├── train_detectron.py           # ViTDet + Mask R-CNN training
├── launch_vitdet.py             # Automated setup and launch script
├── inference.py                 # Inference script
└── README.md                    # This file
```

## 🔧 Quick Start

### Option 1: Automated Setup (Recommended)

The easiest way to get started is using the automated launch script:

```bash
# Install and launch training in one command
python launch_vitdet.py --data-dir /path/to/your/data --num-gpus 4

# Or install dependencies only
python launch_vitdet.py --data-dir /path/to/your/data --install-only
```

### Option 2: Manual Setup

1. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Detectron2:**
   ```bash
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

3. **Install other dependencies:**
   ```bash
   pip install -r requirements_detectron.txt
   ```

4. **Verify CUDA and GPU availability:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
   ```

## 📊 Data Preparation

Your data directory should have the following structure:

```
data/
├── train.csv                 # Metadata with image_id, isup_grade columns
├── train_images/            # WSI files (.tiff format)
│   ├── image_001.tiff
│   ├── image_002.tiff
│   └── ...
└── train_label_masks/       # Mask files (.tiff format)
    ├── image_001_mask.tiff
    ├── image_002_mask.tiff
    └── ...
```

## 🚀 Training

### ViTDet + Mask R-CNN Training (Recommended)

**Single GPU:**
```bash
python train_detectron.py --config configs/vitdet_config.yaml --num-gpus 1
```

**Multi-GPU (4x V100):**
```bash
python train_detectron.py --config configs/vitdet_config.yaml --num-gpus 4
```

**With custom options:**
```bash
python train_detectron.py \
    --config configs/vitdet_config.yaml \
    --num-gpus 4 \
    --opts SOLVER.BASE_LR 0.0002 SOLVER.MAX_ITER 50000
```

**Resume training:**
```bash
python train_detectron.py --config configs/vitdet_config.yaml --num-gpus 4 --resume
```

### Original ViT Training (Deprecated)

The original PyTorch ViT training is still available but deprecated:

```bash
python train.py --config configs/base_config.yaml --gpus 4
```

## 📈 Configuration

The main configuration file is `configs/vitdet_config.yaml`. Key settings:

### Model Settings
```yaml
model:
  name: "vitdet_maskrcnn"
  backbone: "ViT-B/16"
  num_classes: 6  # ISUP grades 0-5
  img_size: 1024  # Larger patches for better detection
```

### Training Settings
```yaml
training:
  epochs: 50
  base_lr: 0.0001
  mixed_precision: true
  eval_period: 2000
```

### Data Settings
```yaml
data:
  batch_size: 4  # Smaller for large images
  patch_size: 1024  # Large patches
  overlap: 128
  min_tumor_area: 1000
```

## 🔍 Evaluation and Inference

**Evaluation only:**
```bash
python train_detectron.py --config configs/vitdet_config.yaml --eval-only
```

**WSI Inference (original script):**
```bash
python inference.py \
    --checkpoint logs/best_model.pth \
    --wsi_path /path/to/slide.tiff \
    --output_dir ./results
```

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs_vitdet
```

### Weights & Biases
Training automatically logs to W&B if enabled in config:
```yaml
logging:
  use_wandb: true
  wandb_project: "prostate-cancer-vitdet"
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size in config: `data.batch_size: 2`
   - Reduce patch size: `data.patch_size: 512`

2. **Detectron2 Installation:**
   ```bash
   # Try pre-built wheel
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
   ```

3. **OpenSlide Issues:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install openslide-tools
   
   # Windows: Install from https://openslide.org/download/
   ```

4. **Multi-GPU Issues:**
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check CUDA_VISIBLE_DEVICES: `export CUDA_VISIBLE_DEVICES=0,1,2,3`

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Increase num_workers** for data loading if CPU allows
3. **Use mixed precision** (`mixed_precision: true`)
4. **Monitor GPU utilization** with `nvidia-smi -l 1`

## 📋 System Requirements

- **GPU**: 4x V100 (32GB VRAM recommended) or similar
- **CUDA**: 11.4+ (compatible with PyTorch)
- **RAM**: 64GB+ recommended for large WSI processing
- **Storage**: SSD recommended, 500GB+ free space
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11

## 🔄 Migration from Original ViT

If migrating from the original ViT pipeline:

1. **Install new dependencies:**
   ```bash
   pip install -r requirements_detectron.txt
   ```

2. **Use new config:**
   ```bash
   cp configs/vitdet_config.yaml configs/my_config.yaml
   # Edit my_config.yaml as needed
   ```

3. **Update training command:**
   ```bash
   # Old
   python train.py --config configs/base_config.yaml
   
   # New  
   python train_detectron.py --config configs/vitdet_config.yaml
   ```

## 📈 Performance Comparison

| Model | mAP@0.5 | mAP@0.75 | Training Time | GPU Memory |
|-------|---------|----------|---------------|------------|
| Original ViT | - | - | ~12h (4x V100) | 16GB |
| ViTDet + Mask R-CNN | TBD | TBD | ~24h (4x V100) | 28GB |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Detectron2**: Facebook AI Research
- **ViTDet**: Vision Transformer for Object Detection
- **PANDA Dataset**: Prostate cANcer graDe Assessment Challenge
- **OpenSlide**: Library for reading whole-slide images
