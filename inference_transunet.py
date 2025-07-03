#!/usr/bin/env python3
"""
Simple Inference script for TransUNet - Prostate Cancer Detection
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

from models import create_transunet

def parse_args():
    parser = argparse.ArgumentParser(description='TransUNet Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    """Load trained TransUNet model"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_transunet(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model'].get('mlp_ratio', 4),
        dropout=config['model'].get('dropout', 0.1),
        seg_classes=config['model'].get('seg_classes', 1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def preprocess_image(image_path, img_size=224):
    """Preprocess input image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def postprocess_segmentation(seg_probs, original_size):
    """Postprocess segmentation results"""
    # Convert to numpy
    seg_mask = seg_probs.squeeze().cpu().numpy()
    
    # Resize to original size
    seg_mask = cv2.resize(seg_mask, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Threshold to binary mask
    binary_mask = (seg_mask > 0.5).astype(np.uint8) * 255
    
    return seg_mask, binary_mask

def visualize_results(original_image, seg_mask, binary_mask, cls_pred, cls_prob, output_path):
    """Create visualization of results"""
    import matplotlib.pyplot as plt
    
    # Convert PIL to numpy
    original_np = np.array(original_image)
    
    # Create overlay
    overlay = original_np.copy()
    overlay[binary_mask > 0] = [255, 0, 0]  # Red for predicted tumor regions
    
    # Blend original and overlay
    blended = cv2.addWeighted(original_np, 0.7, overlay, 0.3, 0)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'TransUNet Results\\nPredicted ISUP Grade: {cls_pred} (Confidence: {cls_prob:.3f})', fontsize=14)
    
    # Original image
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmentation probability map
    axes[0, 1].imshow(seg_mask, cmap='hot')
    axes[0, 1].set_title('Segmentation Probability')
    axes[0, 1].axis('off')
    
    # Binary mask
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Binary Segmentation Mask')
    axes[1, 0].axis('off')
    
    # Overlay
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('Overlay (Red = Predicted Tumor)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def main():
    """Main inference function"""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess image
    img_size = config['model']['img_size']
    image_tensor, original_image = preprocess_image(args.image, img_size)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Classification results
        cls_logits = outputs['logits']
        cls_probs = F.softmax(cls_logits, dim=1)
        cls_pred = torch.argmax(cls_probs, dim=1).item()
        cls_confidence = cls_probs[0, cls_pred].item()
        
        # Segmentation results
        seg_probs = outputs['seg_probs']
    
    # Print results
    print(f"\\nResults:")
    print(f"Predicted ISUP Grade: {cls_pred}")
    print(f"Confidence: {cls_confidence:.3f}")
    
    # Postprocess segmentation
    original_size = original_image.size
    seg_mask, binary_mask = postprocess_segmentation(seg_probs, original_size)
    
    # Calculate tumor area percentage
    tumor_area = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1]) * 100
    print(f"Predicted tumor area: {tumor_area:.2f}%")
    
    # Save results
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Save visualization
    viz_path = os.path.join(args.output_dir, f"{base_name}_results.png")
    visualize_results(original_image, seg_mask, binary_mask, cls_pred, cls_confidence, viz_path)
    
    # Save individual results
    seg_path = os.path.join(args.output_dir, f"{base_name}_segmentation.png")
    cv2.imwrite(seg_path, (seg_mask * 255).astype(np.uint8))
    
    mask_path = os.path.join(args.output_dir, f"{base_name}_binary_mask.png")
    cv2.imwrite(mask_path, binary_mask)
    
    print(f"\\nResults saved to: {args.output_dir}")
    print(f"- Visualization: {viz_path}")
    print(f"- Segmentation: {seg_path}")
    print(f"- Binary mask: {mask_path}")

if __name__ == '__main__':
    main()
