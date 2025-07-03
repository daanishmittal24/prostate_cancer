#!/usr/bin/env python3
"""
Simple Inference script for TransUNet - Prostate Cancer Detection
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path

from models import create_transunet
from data import get_valid_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='TransUNet Inference for Prostate Cancer')
    parser.add_argument('--config', type=str, default='configs/transunet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, checkpoint_path, device):
    """Load trained TransUNet model."""
    # Create model
    model_cfg = config['model']
    model = create_transunet(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        num_classes=model_cfg['num_classes'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        mlp_ratio=model_cfg.get('mlp_ratio', 4),
        dropout=model_cfg.get('dropout', 0.1),
        seg_classes=model_cfg.get('seg_classes', 1)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel wrapper
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix from keys
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint accuracy: {checkpoint.get('accuracy', 'N/A')}")
    print(f"Checkpoint kappa: {checkpoint.get('kappa', 'N/A')}")
    
    return model

def preprocess_image(image_path, img_size, transform=None):
    """Load and preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms if provided
    if transform:
        image = transform(image)
    else:
        # Basic preprocessing
        image = image.resize((img_size, img_size))
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    return image

def predict(model, image, device):
    """Run inference on image."""
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        
        # Get classification prediction
        cls_logits = outputs['classification']
        cls_probs = F.softmax(cls_logits, dim=1)
        cls_pred = torch.argmax(cls_logits, dim=1)
        
        # Get segmentation prediction
        seg_logits = outputs['segmentation']
        seg_probs = torch.sigmoid(seg_logits)
        seg_pred = (seg_probs > 0.5).float()
        
        return {
            'classification': {
                'prediction': cls_pred.cpu().numpy()[0],
                'probabilities': cls_probs.cpu().numpy()[0],
                'logits': cls_logits.cpu().numpy()[0]
            },
            'segmentation': {
                'prediction': seg_pred.cpu().numpy()[0],
                'probabilities': seg_probs.cpu().numpy()[0],
                'logits': seg_logits.cpu().numpy()[0]
            }
        }

def visualize_results(image_path, predictions, output_dir, img_size):
    """Create visualization of results."""
    # Load original image
    orig_image = Image.open(image_path).convert('RGB')
    orig_image = orig_image.resize((img_size, img_size))
    orig_array = np.array(orig_image)
    
    # Get predictions
    cls_pred = predictions['classification']['prediction']
    cls_probs = predictions['classification']['probabilities']
    seg_pred = predictions['segmentation']['prediction'][0]  # Remove channel dimension
    seg_probs = predictions['segmentation']['probabilities'][0]
    
    # ISUP grade labels
    isup_labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(orig_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Classification prediction
    axes[0, 1].bar(range(len(cls_probs)), cls_probs)
    axes[0, 1].set_title(f'Classification: {isup_labels[cls_pred]} (Confidence: {cls_probs[cls_pred]:.3f})')
    axes[0, 1].set_xlabel('ISUP Grade')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_xticks(range(len(isup_labels)))
    axes[0, 1].set_xticklabels([f'G{i}' for i in range(len(isup_labels))])
    
    # Segmentation prediction (binary mask)
    axes[1, 0].imshow(seg_pred, cmap='gray')
    axes[1, 0].set_title('Segmentation Prediction')
    axes[1, 0].axis('off')
    
    # Overlay segmentation on original image
    overlay = orig_array.copy()
    mask = seg_pred > 0.5
    overlay[mask] = [255, 0, 0]  # Red overlay for positive regions
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Segmentation Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'prediction_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also save individual results
    seg_output_path = os.path.join(output_dir, 'segmentation_mask.png')
    plt.figure(figsize=(6, 6))
    plt.imshow(seg_pred, cmap='gray')
    plt.axis('off')
    plt.savefig(seg_output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"Segmentation mask saved to {seg_output_path}")
    
    plt.close('all')

def save_results(predictions, output_dir, image_path):
    """Save prediction results to text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ISUP grade labels
    isup_labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    
    cls_pred = predictions['classification']['prediction']
    cls_probs = predictions['classification']['probabilities']
    
    results_text = f"""TransUNet Inference Results
===========================

Input Image: {os.path.basename(image_path)}

Classification Results:
----------------------
Predicted ISUP Grade: {isup_labels[cls_pred]} (Class {cls_pred})
Confidence: {cls_probs[cls_pred]:.4f}

Class Probabilities:
"""
    
    for i, prob in enumerate(cls_probs):
        results_text += f"  {isup_labels[i]}: {prob:.4f}\n"
    
    results_text += f"""
Segmentation Results:
--------------------
Segmentation map generated with shape: {predictions['segmentation']['prediction'].shape}
Positive pixels: {np.sum(predictions['segmentation']['prediction'] > 0.5)}
"""
    
    # Save to file
    results_path = os.path.join(output_dir, 'prediction_results.txt')
    with open(results_path, 'w') as f:
        f.write(results_text)
    
    print(f"Results saved to {results_path}")

def main():
    """Main inference function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Load and preprocess image
    transform = get_valid_transforms(img_size=config['model']['img_size'])
    image = preprocess_image(args.input, config['model']['img_size'], transform)
    
    print(f"Processing image: {args.input}")
    print(f"Image shape: {image.shape}")
    
    # Run inference
    predictions = predict(model, image, device)
    
    # Print results
    isup_labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    cls_pred = predictions['classification']['prediction']
    cls_conf = predictions['classification']['probabilities'][cls_pred]
    
    print(f"\nPrediction Results:")
    print(f"ISUP Grade: {isup_labels[cls_pred]} (confidence: {cls_conf:.3f})")
    print(f"Segmentation: {np.sum(predictions['segmentation']['prediction'] > 0.5)} positive pixels")
    
    # Save results
    save_results(predictions, args.output, args.input)
    
    # Generate visualization if requested
    if args.visualize:
        visualize_results(args.input, predictions, args.output, config['model']['img_size'])
    
    print(f"\nInference completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()
