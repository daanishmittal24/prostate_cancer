# TransUNet for Prostate Cancer Detection

A clean, easy-to-use implementation of TransUNet for prostate cancer classification and segmentation. This implementation combines Vision Transformer (ViT) with U-Net architecture to perform both ISUP grading classification and tissue segmentation.

## 🚀 Features

- **Dual Task Learning**: Simultaneous classification (ISUP grades 0-5) and segmentation
- **Clean Architecture**: Well-documented, modular TransUNet implementation
- **Multi-GPU Support**: Efficient training on multiple GPUs with DataParallel
- **Mixed Precision**: Faster training with automatic mixed precision (AMP)
- **Easy Inference**: Simple inference script with visualization
- **Comprehensive Logging**: TensorBoard integration for monitoring
- **Flexible Configuration**: YAML-based configuration system

## 📋 Requirements

```bash
pip install torch torchvision torchaudio  # PyTorch with CUDA support
pip install -r requirements.txt
```

## 🏗️ Architecture

The TransUNet model consists of:

1. **Patch Embedding**: Converts input images to patch tokens
2. **Vision Transformer Encoder**: Self-attention based feature extraction
3. **Classification Head**: Predicts ISUP grades (0-5) from [CLS] token
4. **Segmentation Decoder**: U-Net style decoder for pixel-level segmentation

### Model Specifications

- **Input Size**: 224×224 RGB images
- **Patch Size**: 16×16 patches (14×14 = 196 patches)
- **Embedding Dimension**: 768
- **Transformer Layers**: 12 layers with 12 attention heads
- **Parameters**: ~86M trainable parameters

## 📁 Project Structure

```
prostate_cancer/
├── models/
│   ├── __init__.py
│   ├── vit.py                    # Original ViT model (legacy)
│   └── transunet.py             # New TransUNet implementation ⭐
├── data/
│   ├── __init__.py
│   ├── dataset.py              # PANDA dataset with lazy loading
│   └── transforms.py           # Data augmentation
├── configs/
│   ├── base_config.yaml        # ViT config (legacy)
│   └── transunet_config.yaml   # TransUNet config ⭐
├── train_transunet_clean.py     # Clean training script ⭐
├── inference_transunet_clean.py # Clean inference script ⭐
├── example_usage.py             # Usage examples ⭐
└── requirements.txt
```

## 🚀 Quick Start

### 1. Training

Train TransUNet on your dataset:

```bash
python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 4
```

**Key Training Features:**
- Automatic train/validation split (80/20)
- Mixed precision training for efficiency
- Gradient clipping for stable training
- Cosine annealing learning rate schedule
- Multi-loss optimization (classification + segmentation)
- Comprehensive metrics (accuracy, kappa score)

### 2. Resume Training

Continue training from a checkpoint:

```bash
python train_transunet_clean.py --config configs/transunet_config.yaml --resume checkpoints/transunet_best.pth
```

### 3. Inference

Run inference on new images:

```bash
python inference_transunet_clean.py \
    --config configs/transunet_config.yaml \
    --checkpoint checkpoints/transunet_best.pth \
    --input /path/to/image.jpg \
    --output ./results \
    --visualize
```

## ⚙️ Configuration

Edit `configs/transunet_config.yaml` to customize training:

```yaml
# Model Configuration
model:
  img_size: 224          # Input image size
  patch_size: 16         # Patch size for ViT
  num_classes: 6         # ISUP grades 0-5
  seg_classes: 1         # Binary segmentation
  embed_dim: 768         # Transformer embedding dimension
  depth: 12              # Number of transformer layers
  num_heads: 12          # Number of attention heads
  dropout: 0.1           # Dropout rate

# Training Configuration
training:
  epochs: 25             # Number of training epochs
  lr: 1e-4              # Learning rate
  weight_decay: 0.01     # Weight decay
  mixed_precision: true  # Use mixed precision training
  grad_clip: 1.0         # Gradient clipping threshold

# Data Configuration
data:
  batch_size: 16         # Batch size per GPU
  num_workers: 8         # Number of data loading workers
  patches_per_image: 12  # Patches extracted per image
```

## 🔧 Model Variants

The implementation supports different model sizes:

```python
# Small model (faster, less memory)
model = create_transunet_small(img_size=224, num_classes=6)

# Base model (default)
model = create_transunet_base(img_size=224, num_classes=6)

# Large model (best performance, more memory)
model = create_transunet_large(img_size=224, num_classes=6)
```

## 📊 Loss Function

The model optimizes a combined loss:

```python
total_loss = classification_loss + 0.5 * segmentation_loss
```

- **Classification Loss**: CrossEntropyLoss for ISUP grading
- **Segmentation Loss**: BCEWithLogitsLoss for binary segmentation
- **Weighting**: Segmentation loss weighted by 0.5

## 📈 Monitoring

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir ./logs
```

**Tracked Metrics:**
- Training/Validation Loss (total, classification, segmentation)
- Accuracy and Quadratic Weighted Kappa
- Learning Rate
- Confusion Matrix

## 🖼️ Inference Output

The inference script produces:

1. **Classification Results**: ISUP grade prediction with confidence scores
2. **Segmentation Mask**: Binary tissue segmentation
3. **Visualization**: Combined visualization with overlay
4. **Text Results**: Detailed prediction summary

## 💡 Usage Examples

### Basic Model Usage

```python
from models import create_transunet
import torch

# Create model
model = create_transunet(img_size=224, num_classes=6, seg_classes=1)

# Forward pass
image = torch.randn(1, 3, 224, 224)
outputs = model(image)

# Get predictions
cls_logits = outputs['classification']  # Shape: [1, 6]
seg_logits = outputs['segmentation']    # Shape: [1, 1, 224, 224]
```

### Custom Training Loop

```python
# Initialize model and losses
model = create_transunet(img_size=224, num_classes=6, seg_classes=1)
cls_criterion = torch.nn.CrossEntropyLoss()
seg_criterion = torch.nn.BCEWithLogitsLoss()

# Forward pass
outputs = model(images)
cls_loss = cls_criterion(outputs['classification'], labels)
seg_loss = seg_criterion(outputs['segmentation'], masks)
total_loss = cls_loss + 0.5 * seg_loss

# Backward pass
total_loss.backward()
optimizer.step()
```

## 🎯 Key Improvements Over Original ViT

1. **Dual Task Learning**: Combines classification and segmentation
2. **Better Feature Utilization**: Uses patch features for segmentation
3. **Cleaner Code**: Modular, well-documented implementation
4. **Enhanced Training**: Better loss functions and training procedures
5. **Easy Inference**: Simple, ready-to-use inference pipeline

## 🔧 Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (e.g., RTX 3070)
- **RAM**: 16GB system memory
- **Storage**: 50GB for dataset + checkpoints

### Recommended Setup
- **GPU**: 4× 16GB GPUs (e.g., V100, A100)
- **RAM**: 64GB system memory
- **Storage**: 500GB SSD for fast data loading

## 📊 Performance Expectations

With proper hyperparameter tuning:

- **Classification Accuracy**: 85-90% (depends on dataset quality)
- **Quadratic Weighted Kappa**: 0.80-0.85
- **Training Time**: ~2-4 hours on 4× V100 (25 epochs)
- **Memory Usage**: ~12GB per GPU with batch_size=16

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use `mixed_precision: true`
   - Try gradient accumulation

2. **Slow Training**
   - Increase `num_workers` for data loading
   - Use SSD storage for dataset
   - Enable `persistent_workers: true`

3. **Poor Convergence**
   - Check learning rate (try 1e-4 to 1e-5)
   - Verify data augmentation isn't too aggressive
   - Ensure proper train/val split

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to train and use TransUNet for prostate cancer detection! 🎯**

For questions or issues, please open a GitHub issue or contact the maintainers.
