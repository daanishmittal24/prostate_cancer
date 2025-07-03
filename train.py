#!/usr/bin/env python3
"""
Training script for Prostate Cancer Detection using Vision Transformers.
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
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd

from models import ViTForProstateCancer
from data import PANDA_Dataset, get_train_transforms, get_valid_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer for Prostate Cancer Detection')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1,
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
    train_df = pd.read_csv(os.path.join(data_cfg['data_dir'], data_cfg['train_csv']))
    
    # Split into train/val (80/20)
    val_size = int(0.2 * len(train_df))
    train_df = train_df.iloc[val_size:].reset_index(drop=True)
    val_df = train_df.iloc[:val_size].reset_index(drop=True)
    
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
        patch_size=config['model']['patch_size']
    )
    
    val_dataset = PANDA_Dataset(
        data_dir=data_cfg['data_dir'],
        df=val_df,
        transform=val_transforms,
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size']
    )
    
    return train_dataset, val_dataset

def create_model(config):
    """Create and initialize the model."""
    model_cfg = config['model']
    model = ViTForProstateCancer(
        model_name=model_cfg['name'],
        num_classes=model_cfg['num_classes'],
        pretrained=model_cfg['pretrained'],
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size']
    )
    return model

def train_epoch(model, loader, optimizer, scaler, device, epoch, writer, config):
    """Train for one epoch."""
    model.train()
    
    train_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1} [Train]')
    
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch.get('mask', None)
        if masks is not None:
            masks = masks.unsqueeze(1).float().to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        with autocast(enabled=config['training']['mixed_precision']):
            outputs = model(
                pixel_values=images,
                labels=labels,
                masks=masks
            )
            loss = outputs['loss']
        
        # Backward pass and optimize
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
        
        # Get predictions
        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': train_loss / (step + 1)
        })
    
    # Calculate metrics
    train_loss /= len(loader)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Log metrics
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Kappa/train', kappa, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    
    return train_loss, kappa, accuracy

def validate(model, loader, device, epoch, writer, config):
    """Validate the model."""
    model.eval()
    
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1} [Val]')
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.unsqueeze(1).float().to(device)
            
            # Forward pass
            with autocast(enabled=config['training']['mixed_precision']):
                outputs = model(
                    pixel_values=images,
                    labels=labels,
                    masks=masks
                )
                loss = outputs['loss']
            
            # Update metrics
            val_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': val_loss / (step + 1)
            })
    
    # Calculate metrics
    val_loss /= len(loader)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Log metrics
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Kappa/val', kappa, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    
    # Log confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(config['model']['num_classes']))
    writer.add_figure(
        'Confusion Matrix',
        plot_confusion_matrix(cm, class_names=[str(i) for i in range(config['model']['num_classes'])], title='Confusion Matrix'),
        epoch
    )
    
    return val_loss, kappa, accuracy

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Create a confusion matrix plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.close(fig)
    return fig

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Mixed precision training
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {start_epoch}')
    
    # Multi-GPU training
    if torch.cuda.device_count() > 1 and args.gpus > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)
    
    # Training loop
    best_kappa = 0.0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train for one epoch
        train_loss, train_kappa, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, config
        )
        
        # Validate
        val_loss, val_kappa, val_acc = validate(
            model, val_loader, device, epoch, writer, config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_kappa > best_kappa
        if is_best:
            best_kappa = val_kappa
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_kappa': val_kappa,
                'val_loss': val_loss,
            }, os.path.join(config['logging']['log_dir'], 'best_model.pth'))
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_kappa': val_kappa,
            'val_loss': val_loss,
        }, os.path.join(config['logging']['log_dir'], 'latest_checkpoint.pth'))
        
        # Print epoch summary
        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Kappa: {train_kappa:.4f}, Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Kappa: {val_kappa:.4f}, Acc: {val_acc:.4f}')
        print(f'  Best Kappa: {best_kappa:.4f}')
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
