#!/usr/bin/env python3
"""
Simple Training script for TransUNet - Prostate Cancer Detection
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

def train_epoch(model, loader, optimizer, scaler, device, epoch, writer, config):
    """Train for one epoch."""
    model.train()
    
    train_loss = 0.0
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
            outputs = model(images, labels=labels, masks=masks)
            loss = outputs['loss']
        
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
        
        # Classification metrics
        preds = torch.argmax(outputs['logits'], dim=1)
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
    train_acc = cls_correct / cls_total
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Log metrics
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Kappa/train', kappa, epoch)
    
    return train_loss, train_acc, kappa

def validate(model, loader, device, epoch, writer, config):
    """Validate the model."""
    model.eval()
    
    val_loss = 0.0
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
                outputs = model(images, labels=labels, masks=masks)
                loss = outputs['loss']
            
            # Update metrics
            val_loss += loss.item()
            
            # Classification metrics
            preds = torch.argmax(outputs['logits'], dim=1)
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
    val_acc = cls_correct / cls_total
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Log metrics
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Kappa/val', kappa, epoch)
    
    # Log confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(config['model']['num_classes']))
    
    return val_loss, val_acc, kappa

def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up device
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
    
    # Multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for DataParallel training!')
        model = nn.DataParallel(model)
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_kappa = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_kappa = checkpoint.get('best_kappa', 0.0)
    
    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_loss, train_acc, train_kappa = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, config
        )
        
        # Validate
        val_loss, val_acc, val_kappa = validate(
            model, val_loader, device, epoch, writer, config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_kappa > best_kappa
        if is_best:
            best_kappa = val_kappa
        
        # Save model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_kappa': best_kappa,
            'val_kappa': val_kappa,
            'val_loss': val_loss,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(config['logging']['log_dir'], 'latest_checkpoint.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(config['logging']['log_dir'], 'best_model.pth'))
        
        # Print epoch summary
        print(f'\\nEpoch {epoch + 1}/{config["training"]["epochs"]}:')
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, Kappa={train_kappa:.4f}')
        print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, Kappa={val_kappa:.4f}')
        print(f'  Best Kappa: {best_kappa:.4f}')
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LR', current_lr, epoch)
    
    print("\\nTraining completed!")
    print(f"Best validation kappa: {best_kappa:.4f}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
