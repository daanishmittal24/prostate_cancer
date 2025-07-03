#!/usr/bin/env python3
"""
Example usage script for TransUNet - Prostate Cancer Detection
Shows how to use the clean TransUNet implementation
"""

import os
import torch
import numpy as np
from PIL import Image

# Assuming you're in the project directory
from models import create_transunet

def example_model_usage():
    """Example of how to create and use the TransUNet model."""
    
    print("=== TransUNet Model Example ===")
    
    # Model parameters
    img_size = 224
    num_classes = 6  # ISUP grades 0-5
    seg_classes = 1  # Binary segmentation
    
    # Create model
    print("Creating TransUNet model...")
    model = create_transunet(
        img_size=img_size,
        num_classes=num_classes,
        seg_classes=seg_classes
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input
    print(f"Creating dummy input of size: {img_size}x{img_size}")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Check outputs
    cls_output = outputs['classification']
    seg_output = outputs['segmentation']
    
    print(f"Classification output shape: {cls_output.shape}")  # Should be [batch_size, num_classes]
    print(f"Segmentation output shape: {seg_output.shape}")    # Should be [batch_size, seg_classes, img_size, img_size]
    
    # Get predictions
    cls_preds = torch.argmax(cls_output, dim=1)
    seg_preds = torch.sigmoid(seg_output) > 0.5
    
    print(f"Classification predictions: {cls_preds.numpy()}")
    print(f"Segmentation positive pixels: {seg_preds.sum(dim=[1,2,3]).numpy()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def example_training_setup():
    """Example of how to set up training."""
    
    print("\n=== Training Setup Example ===")
    
    # Create model
    model = create_transunet(img_size=224, num_classes=6, seg_classes=1)
    
    # Set up for training
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create loss functions
    cls_criterion = torch.nn.CrossEntropyLoss()
    seg_criterion = torch.nn.BCEWithLogitsLoss()
    
    # Dummy batch
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 6, (4,))  # ISUP grades 0-5
    masks = torch.rand(4, 1, 224, 224)  # Random segmentation masks
    
    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
    
    # Forward pass
    outputs = model(images)
    
    # Calculate losses
    cls_loss = cls_criterion(outputs['classification'], labels)
    seg_loss = seg_criterion(outputs['segmentation'], masks)
    total_loss = cls_loss + 0.5 * seg_loss  # Weighted combination
    
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"Segmentation loss: {seg_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("Training step completed successfully!")

def example_inference():
    """Example of how to run inference."""
    
    print("\n=== Inference Example ===")
    
    # Create model and set to eval mode
    model = create_transunet(img_size=224, num_classes=6, seg_classes=1)
    model.eval()
    
    # Simulate loading a trained model (normally you'd load from checkpoint)
    print("Model ready for inference (normally you'd load trained weights)")
    
    # Create sample input (normally you'd load and preprocess a real image)
    image = torch.randn(1, 3, 224, 224)
    print(f"Input image shape: {image.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(image)
    
    # Get predictions
    cls_logits = outputs['classification']
    seg_logits = outputs['segmentation']
    
    # Convert to predictions
    cls_probs = torch.softmax(cls_logits, dim=1)
    cls_pred = torch.argmax(cls_logits, dim=1)
    
    seg_probs = torch.sigmoid(seg_logits)
    seg_pred = seg_probs > 0.5
    
    # ISUP grade labels
    isup_labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    
    print(f"Predicted ISUP Grade: {isup_labels[cls_pred.item()]}")
    print(f"Confidence: {cls_probs[0, cls_pred].item():.3f}")
    print(f"All class probabilities: {cls_probs[0].numpy()}")
    print(f"Segmentation positive pixels: {seg_pred.sum().item()}")
    
def show_usage_instructions():
    """Show how to use the training and inference scripts."""
    
    print("\n=== Usage Instructions ===")
    
    print("1. Training:")
    print("   python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 4")
    
    print("\n2. Resume training:")
    print("   python train_transunet_clean.py --config configs/transunet_config.yaml --resume checkpoints/transunet_best.pth")
    
    print("\n3. Inference:")
    print("   python inference_transunet_clean.py --config configs/transunet_config.yaml --checkpoint checkpoints/transunet_best.pth --input /path/to/image.jpg --output ./results --visualize")
    
    print("\n4. Required directory structure:")
    print("   prostate_cancer/")
    print("   ├── models/")
    print("   │   ├── __init__.py")
    print("   │   ├── vit.py")
    print("   │   └── transunet.py")
    print("   ├── data/")
    print("   │   ├── __init__.py")
    print("   │   ├── dataset.py")
    print("   │   └── transforms.py")
    print("   ├── configs/")
    print("   │   └── transunet_config.yaml")
    print("   ├── train_transunet_clean.py")
    print("   ├── inference_transunet_clean.py")
    print("   └── requirements.txt")
    
    print("\n5. Key features of this TransUNet implementation:")
    print("   - Clean, readable code with proper documentation")
    print("   - Supports both classification (ISUP grading) and segmentation")
    print("   - Multi-GPU training with DataParallel")
    print("   - Mixed precision training for efficiency")
    print("   - Comprehensive logging with TensorBoard")
    print("   - Easy-to-use inference with visualization")
    print("   - Modular design for easy customization")

def main():
    """Run all examples."""
    print("TransUNet for Prostate Cancer Detection - Example Usage")
    print("=" * 60)
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
        print()
        
        # Run examples
        example_model_usage()
        example_training_setup()
        example_inference()
        
    except ImportError:
        print("PyTorch not available. Showing usage instructions only.")
    
    show_usage_instructions()
    
    print("\n" + "=" * 60)
    print("Example completed! You're ready to train and use TransUNet.")

if __name__ == "__main__":
    main()
