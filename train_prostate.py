        #!/usr/bin/env python3
"""
Training script for Prostate Cancer Detection using Vision Transformers (ViT).
Supports both classification (ISUP grade) and segmentation (tumor region) for Radboud and Karolinska datasets.
"""

import os
import argparse
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import dataset and model
from data.panda_dataset import ProstateCancerDataset
from models.vit import ViTForProstateCancer

import yaml
import time

CONFIG_FILE = 'local_config.yaml'

# Defaults
DEFAULTS = {
    'data_dir': './data',
    'csv_file': 'train.csv',
    'image_dir': 'train_images',
    'mask_dir': 'train_label_masks',
    'output_dir': './output',
    'batch_size': 16,
    'num_epochs': 20,
    'lr': 5e-5,
    'img_size': 224,
    'patch_size': 16,
    'num_workers': 4,
    'data_provider': 'all',
}

def get_local_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    return None

def save_local_config(config):
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)

def get_or_prompt_base_path():
    config = get_local_config()
    if config and 'data_dir' in config:
        return config['data_dir']
    base_path = input('Enter base data directory (where train.csv, train_images, train_label_masks are located): ').strip()
    config = {'data_dir': base_path}
    save_local_config(config)
    return base_path

def get_transforms(img_size):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

def create_data_loaders(config):
    csv_path = os.path.join(config['data_dir'], config['csv_file'])
    df = pd.read_csv(csv_path)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_transform, val_transform = get_transforms(config['img_size'])
    train_dataset = ProstateCancerDataset(
        data_dir=config['data_dir'], df=train_df, transform=train_transform,
        img_size=config['img_size'], patch_size=config['patch_size'], data_provider=config['data_provider']
    )
    val_dataset = ProstateCancerDataset(
        data_dir=config['data_dir'], df=val_df, transform=val_transform,
        img_size=config['img_size'], patch_size=config['patch_size'], data_provider=config['data_provider']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, device, scaler, epoch, writer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device)
        masks = batch.get('mask')
        if masks is not None:
            masks = masks.to(device).float()
        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values=images, labels=labels, masks=masks)
            loss = outputs['loss']
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds = torch.argmax(outputs['logits'], dim=1)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(loader.dataset)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    if writer:
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Kappa/train', kappa, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
    return epoch_loss, kappa, acc

def validate(model, loader, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Val]')
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device)
            masks = batch.get('mask')
            if masks is not None:
                masks = masks.to(device).float()
            with autocast():
                outputs = model(pixel_values=images, labels=labels, masks=masks)
                loss = outputs['loss']
            preds = torch.argmax(outputs['logits'], dim=1)
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(loader.dataset)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    if writer:
        writer.add_scalar('Loss/val', epoch_loss, epoch)
        writer.add_scalar('Kappa/val', kappa, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
        cm = confusion_matrix(all_labels, all_preds)
        fig = plot_confusion_matrix(cm, classes=[str(i) for i in range(6)])
        writer.add_figure('Confusion Matrix', fig, epoch)
    return epoch_loss, kappa, acc

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.tight_layout()
    return fig

def main():
    # Load config and remember base path
    config = DEFAULTS.copy()
    base_path = get_or_prompt_base_path()
    config['data_dir'] = base_path
    # Output dir (auto-timestamped)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config['output_dir'], f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'logs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f"Training data: {os.path.join(config['data_dir'], config['csv_file'])}")
    print(f"Images: {os.path.join(config['data_dir'], config['image_dir'])}")
    print(f"Masks: {os.path.join(config['data_dir'], config['mask_dir'])}")
    print(f"Output: {run_dir}\n")
    train_loader, val_loader = create_data_loaders(config)
    model = ViTForProstateCancer(
        num_classes=6,
        img_size=config['img_size'],
        patch_size=config['patch_size']
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scaler = GradScaler()
    start_epoch = 0
    best_kappa = 0.0
    print('Starting training...')
    total_epochs = config['num_epochs']
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        train_loss, train_kappa, train_acc = train_epoch(
            model, train_loader, optimizer, device, scaler, epoch, writer
        )
        val_loss, val_kappa, val_acc = validate(
            model, val_loader, device, epoch, writer
        )
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (total_epochs - epoch - 1)
        is_best = val_kappa > best_kappa
        if is_best:
            best_kappa = val_kappa
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_loss': val_loss
            }, os.path.join(run_dir, 'checkpoints', 'best_model.pth'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_kappa': val_kappa,
            'val_loss': val_loss
        }, os.path.join(run_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))
        print(f"Epoch {epoch+1}/{total_epochs} | Time: {epoch_time:.1f}s | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | Train Loss: {train_loss:.4f} | Train Kappa: {train_kappa:.4f} | Val Loss: {val_loss:.4f} | Val Kappa: {val_kappa:.4f} | Best Kappa: {best_kappa:.4f}")
    writer.close()
    print('Training complete.')

if __name__ == '__main__':
    main()
