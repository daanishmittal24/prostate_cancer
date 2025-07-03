# Prostate Cancer Detection using Vision Transformers (ViT)

A state-of-the-art deep learning system for automated detection and grading of prostate cancer from whole-slide histopathology images.

## Features

- **Multi-task Learning**: Simultaneous classification (ISUP grade) and segmentation (tumor regions)
- **Vision Transformer**: Leveraging google/vit-base-patch16-224 architecture
- **Scalable Training**: Multi-GPU support with PyTorch Distributed
- **Reproducible**: Complete environment and configuration management

## Project Structure

```
prostate_cancer/
├── configs/               # Configuration files
├── data/                  # Data loading and preprocessing
├── models/                # Model architectures
├── training/              # Training loops and utilities
├── utils/                 # Helper functions
├── inference.py           # Inference script
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd prostate_cancer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the PANDA dataset and update the configuration file.

## Training

```bash
python train.py --config configs/base_config.yaml --gpus 4
```

## Inference

```bash
python inference.py --model_checkpoint checkpoints/best_model.pth --wsi_path /path/to/wsi
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
