#!/usr/bin/env python3
"""
Simple Training script for TransUNet - Prostate Cancer Detection
Clean implementation that works with the TransUNet model
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
import pandas as pd

from models import create_transunet
from data import PANDA_Dataset, get_train_transforms, get_valid_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train TransUNet for Prostate Cancer')
    parser.add_argument('--config', type=str, default='configs/transunet_config.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=4,
                        help='Number of GPUs to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_datasets(config):
    """Create training and validation datasets."""
    data_cfg = config['data']
    
    # Load CSV files
    full_df = pd.read_csv(os.path.join(data_cfg['data_dir'], data_cfg['train_csv']))
    
    # Split into train/val (80/20)
    val_size = int(0.2 * len(full_df))
    train_df = full_df.iloc[val_size:].reset_index(drop=True)
    val_df = full_df.iloc[:val_size].reset_index(drop=True)
    
    print(f"Dataset split: {len(train_df)} train samples, {len(val_df)} validation samples")
    print(f"Train patches: {len(train_df) * config['data'].get('patches_per_image', 16)}")
    print(f"Val patches: {len(val_df) * config['data'].get('patches_per_image', 16)}")
    
    # Create datasets
    train_transforms = get_train_transforms(
        img_size=config['model']['img_size'],
        scale=config['augmentation'].get('scale_limit', 0.1)
    )
    
    val_transforms = get_valid_transforms(
        img_size=config['model']['img_size']
    )
    
    train_dataset = PANDA_Dataset(
        data_dir=data_cfg['data_dir'],
        df=train_df,
        transform=train_transforms,
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        patches_per_image=config['data'].get('patches_per_image', 16)
    )
    
    val_dataset = PANDA_Dataset(
        data_dir=data_cfg['data_dir'],
        df=val_df,
        transform=val_transforms,
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        patches_per_image=config['data'].get('patches_per_image', 16)
    )
    
    return train_dataset, val_dataset

def create_model(config):
    """Create TransUNet model."""
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
    return model

def compute_loss(outputs, labels, masks, device):
    """Compute combined classification and segmentation loss."""
    # Classification loss
    cls_criterion = nn.CrossEntropyLoss()
    cls_loss = cls_criterion(outputs['classification'], labels)
    
    # Segmentation loss (only if masks are available)
    seg_loss = torch.tensor(0.0, device=device)
    if masks is not None and masks.sum() > 0:
        # Binary segmentation with BCE loss
        seg_criterion = nn.BCEWithLogitsLoss()
        seg_loss = seg_criterion(outputs['segmentation'], masks)
    
    # Combined loss (weighted)
    total_loss = cls_loss + 0.5 * seg_loss
    
    return {
        'total_loss': total_loss,
        'cls_loss': cls_loss,
        'seg_loss': seg_loss
    }

def train_epoch(model, loader, optimizer, scaler, device, epoch, writer, config):
    """Train for one epoch."""
    model.train()
    
    train_loss = 0.0
    train_cls_loss = 0.0
    train_seg_loss = 0.0
    cls_correct = 0
    cls_total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1} [Train]')
    
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].float().to(device)
        
        # Ensure masks have the right shape [batch_size, 1, H, W]
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        
        # Check if masks are all zeros (dummy masks)
        has_real_masks = masks.sum(dim=[1, 2, 3]) > 0
        if not has_real_masks.any():
            masks = None
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=config['training']['mixed_precision']):
            outputs = model(images)
            losses = compute_loss(outputs, labels, masks, device)
            loss = losses['total_loss']
        
        # Backward pass
        if config['training']['mixed_precision']:
            scaler.scale(loss).backward()
            if config['training'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['grad_clip']
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['grad_clip']
                )
            optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        train_cls_loss += losses['cls_loss'].item()
        train_seg_loss += losses['seg_loss'].item()
        
        # Classification metrics
        preds = torch.argmax(outputs['classification'], dim=1)
        cls_correct += (preds == labels).sum().item()
        cls_total += labels.size(0)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        # Update progress bar
        accuracy = cls_correct / cls_total
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': train_loss / (step + 1),
            'acc': f'{accuracy:.3f}'
        })
    
    # Calculate final metrics
    train_loss /= len(loader)
    train_cls_loss /= len(loader)
    train_seg_loss /= len(loader)
    train_acc = cls_correct / cls_total
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Log metrics
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    writer.add_scalar('Loss/train_cls', train_cls_loss, epoch)
    writer.add_scalar('Loss/train_seg', train_seg_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Kappa/train', kappa, epoch)
    
    return train_loss, train_acc, kappa

def validate(model, loader, device, epoch, writer, config):
    """Validate the model."""
    model.eval()
    
    val_loss = 0.0
    val_cls_loss = 0.0
    val_seg_loss = 0.0
    cls_correct = 0
    cls_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1} [Val]')
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].float().to(device)
            
            # Ensure masks have the right shape
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Check if masks are all zeros
            has_real_masks = masks.sum(dim=[1, 2, 3]) > 0
            if not has_real_masks.any():
                masks = None
            
            # Forward pass
            with autocast(enabled=config['training']['mixed_precision']):
                outputs = model(images)
                losses = compute_loss(outputs, labels, masks, device)
                loss = losses['total_loss']
            
            # Update metrics
            val_loss += loss.item()
            val_cls_loss += losses['cls_loss'].item()
            val_seg_loss += losses['seg_loss'].item()
            
            # Classification metrics
            preds = torch.argmax(outputs['classification'], dim=1)
            cls_correct += (preds == labels).sum().item()
            cls_total += labels.size(0)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            accuracy = cls_correct / cls_total
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': val_loss / (step + 1),
                'acc': f'{accuracy:.3f}'
            })
    
    # Calculate final metrics
    val_loss /= len(loader)
    val_cls_loss /= len(loader)
    val_seg_loss /= len(loader)
    val_acc = cls_correct / cls_total
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Log metrics
    writer.add_scalar('Loss/val_total', val_loss, epoch)
    writer.add_scalar('Loss/val_cls', val_cls_loss, epoch)
    writer.add_scalar('Loss/val_seg', val_seg_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Kappa/val', kappa, epoch)
    
    # Log confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(config['model']['num_classes']))
    
    return val_loss, val_acc, kappa

def save_checkpoint(model, optimizer, scaler, epoch, loss, acc, kappa, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'accuracy': acc,
        'kappa': kappa
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up device and multi-GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = torch.device('cpu')
        config['training']['mixed_precision'] = False
    else:
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create output directory
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(
        config['logging']['log_dir'],
        config['logging']['experiment_name']
    ))
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers']
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1 and args.gpus > 1:
        print(f"Using {min(args.gpus, torch.cuda.device_count())} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(min(args.gpus, torch.cuda.device_count()))))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_kappa = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_kappa = checkpoint.get('kappa', 0.0)
        print(f"Resumed from epoch {start_epoch}, best kappa: {best_kappa:.4f}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc, train_kappa = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, config
        )
        
        # Validate
        val_loss, val_acc, val_kappa = validate(
            model, val_loader, device, epoch, writer, config
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch results
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")
        
        # Save checkpoint
        is_best = val_kappa > best_kappa
        if is_best:
            best_kappa = val_kappa
            
        checkpoint_path = os.path.join(
            config['logging']['checkpoint_dir'],
            f"transunet_epoch_{epoch + 1}.pth"
        )
        save_checkpoint(model, optimizer, scaler, epoch, val_loss, val_acc, val_kappa, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                config['logging']['checkpoint_dir'],
                "transunet_best.pth"
            )
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, val_acc, val_kappa, best_path)
            print(f"New best model saved! Kappa: {best_kappa:.4f}")
    
    writer.close()
    print(f"\nTraining completed! Best validation kappa: {best_kappa:.4f}")

if __name__ == "__main__":
    main()
