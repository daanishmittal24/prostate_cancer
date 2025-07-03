#!/usr/bin/env python3
"""
Inference script for Prostate Cancer Detection using Vision Transformers.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import openslide

from models import ViTForProstateCancer
from data.transforms import get_valid_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Vision Transformer for Prostate Cancer Detection')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--wsi_path', type=str, required=True,
                        help='Path to input WSI file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output')
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Size of patches to extract from WSI')
    parser.add_argument('--stride', type=int, default=112,
                        help='Stride for patch extraction')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config, device):
    """Load model from checkpoint."""
    model = ViTForProstateCancer(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel or DDP wrapping
    if all(k.startswith('module.') for k in state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def process_wsi(
    wsi_path: str,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 224,
    stride: int = 112,
    transform=None
) -> dict:
    """
    Process a whole slide image and make predictions.
    
    Args:
        wsi_path: Path to the WSI file
        model: Trained model
        device: Device to run inference on
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        transform: Image transforms to apply
        
    Returns:
        Dictionary containing predictions and attention maps
    """
    # Open the WSI
    wsi = openslide.OpenSlide(wsi_path)
    wsi_w, wsi_h = wsi.dimensions
    
    # Calculate number of patches in each dimension
    n_w = (wsi_w - patch_size) // stride + 1
    n_h = (wsi_h - patch_size) // stride + 1
    
    # Initialize output arrays
    pred_map = np.zeros((n_h, n_w), dtype=np.float32)
    prob_maps = np.zeros((n_h, n_w, 6), dtype=np.float32)  # For each ISUP grade
    attention_map = np.zeros((n_h, n_w), dtype=np.float32)
    
    # Process each patch
    with torch.no_grad():
        for i in tqdm(range(n_h), desc='Processing rows'):
            for j in range(n_w):
                # Calculate patch coordinates
                x = j * stride
                y = i * stride
                
                # Extract patch
                patch = wsi.read_region((x, y), 0, (patch_size, patch_size))
                patch = patch.convert('RGB')
                
                # Apply transforms
                if transform:
                    transformed = transform(image=np.array(patch))
                    patch_tensor = transformed['image']
                else:
                    patch_tensor = torch.from_numpy(np.array(patch)).permute(2, 0, 1).float() / 255.0
                
                # Move to device and add batch dimension
                patch_tensor = patch_tensor.unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(pixel_values=patch_tensor)
                
                # Get predictions
                probs = torch.softmax(outputs['logits'], dim=1)
                pred = torch.argmax(probs, dim=1)
                
                # Store results
                pred_map[i, j] = pred.item()
                prob_maps[i, j] = probs.cpu().numpy()[0]
                
                # Store attention (average of attention maps from last layer)
                if hasattr(model, 'vit'):
                    # Get attention weights from the last layer
                    attentions = model.vit(patch_tensor, output_attentions=True).attentions
                    # Average attention across all heads and layers
                    attention = torch.mean(torch.stack([a.mean(dim=1) for a in attentions]), dim=0)
                    # Store the average attention value for this patch
                    attention_map[i, j] = attention.mean().item()
    
    # Close the WSI
    wsi.close()
    
    return {
        'pred_map': pred_map,
        'prob_maps': prob_maps,
        'attention_map': attention_map,
        'wsi_size': (wsi_w, wsi_h),
        'patch_size': patch_size,
        'stride': stride
    }

def visualize_results(wsi_path: str, results: dict, output_dir: str):
    """Visualize and save the results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    
    # 1. Plot prediction map
    plt.figure(figsize=(12, 10))
    plt.imshow(results['pred_map'], cmap='viridis')
    plt.colorbar(label='ISUP Grade (0-5)')
    plt.title(f'Predicted ISUP Grades - {wsi_name}')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{wsi_name}_predictions.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Plot attention map
    plt.figure(figsize=(12, 10))
    plt.imshow(results['attention_map'], cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Attention')
    plt.title(f'Attention Map - {wsi_name}')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{wsi_name}_attention.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Plot probability maps for each class
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i in range(6):
        axes[i].imshow(results['prob_maps'][:, :, i], cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f'ISUP {i} Probability')
        axes[i].axis('off')
    
    plt.suptitle(f'Class Probability Maps - {wsi_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{wsi_name}_probabilities.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Save results as numpy arrays
    np.savez_compressed(
        os.path.join(output_dir, f'{wsi_name}_results.npz'),
        pred_map=results['pred_map'],
        prob_maps=results['prob_maps'],
        attention_map=results['attention_map'],
        wsi_size=results['wsi_size'],
        patch_size=results['patch_size'],
        stride=results['stride']
    )

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = load_model(args.checkpoint, config, device)
    
    # Get transforms
    transform = get_valid_transforms(img_size=args.patch_size)
    
    # Process WSI
    print(f'Processing WSI: {args.wsi_path}')
    results = process_wsi(
        wsi_path=args.wsi_path,
        model=model,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        transform=transform
    )
    
    # Visualize and save results
    print('Saving results...')
    visualize_results(args.wsi_path, results, args.output_dir)
    
    # Print summary
    avg_grade = np.mean(results['pred_map'])
    print(f'\nAnalysis complete!')
    print(f'Average predicted ISUP grade: {avg_grade:.2f}')
    print(f'Results saved to: {os.path.abspath(args.output_dir)}')

if __name__ == '__main__':
    main()
